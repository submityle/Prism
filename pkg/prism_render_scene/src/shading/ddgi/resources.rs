//! Per-view GPU resources backing the DDGI irradiance-volume passes.
//!
//! The subsystem runs across three passes over a shared lattice of octahedral
//! probes:
//!
//! * `probe_update_main` integrates incoming radiance into each probe's
//!   octahedral irradiance tile and depth-moment tile (temporal blend), writing
//!   both atlases (storage) and the per-probe relocation / activity metadata.
//! * `sample_main` reconstructs each pixel's world position + normal, locates
//!   the enclosing probe cell and blends the eight corner probes, reading the
//!   two atlases (texture) + the metadata buffer, writing the GI export.
//! * `composite` folds `gi_out` back over `scene_color` under confidence.
//!
//! This module owns all of them: the two octahedral atlas [`CachedTexture`]s
//! (irradiance + depth / visibility, tiled 2D with a one-texel gutter border per
//! probe, `rgba16float`, `STORAGE_BINDING | TEXTURE_BINDING`), the per-probe
//! [`ProbeMeta`] storage [`Buffer`] (relocation offset + activity flag), the
//! full-resolution `gi_out` GI export [`CachedTexture`] (`rgba16float`, a GI
//! *export* a downstream resolve samples — never copied over `scene_color`), and
//! the `gi_base` scratch copy of `scene_color` the composite's copy pass lifts
//! the shaded colour into before the fold pass reads it (matching SSGI /
//! `world_space_gi`).
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates them to match the
//! configured lattice + viewport, gated on the DDGI enable *and* on the presence
//! of the SSR prepass textures (reverse-Z depth + packed `normal_roughness` the
//! sample pass reconstructs from) and the visibility buffer (the pre-exposed
//! scene colour the probe update gathers), on single-sample views. The atlas
//! dimensions follow the same tiling the WESL (`ddgi_sample.wesl`
//! `probe_tile_origin`) reconstructs with, so the device layout and the shader
//! addressing agree exactly.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, TextureDescriptor,
        TextureDimension, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::ssr::ViewSsrTextures;
use super::settings::PrismDdgiSettings;

/// Per-probe [`ProbeMeta`] storage stride in bytes: one `vec3<f32>` relocation
/// offset + a trailing `f32` activity flag = one 16-byte std430 row, matching
/// the WESL `ProbeMeta` struct in `ddgi_sample.wesl`.
pub(crate) const PROBE_META_STRIDE: u64 = 16;

/// The octahedral atlas texture format: wide HDR, written as a storage image by
/// the probe update and `textureLoad`ed by the sample pass. `rgba16float` is the
/// cheapest WebGPU storage-capable float format that carries the three
/// irradiance channels (`rgb`) and the two depth moments (`rg`) the sample
/// kernel reads; the surplus channels are left unused.
pub(crate) const DDGI_ATLAS_FORMAT: bevy_render::render_resource::TextureFormat =
    bevy_render::render_resource::TextureFormat::Rgba16Float;

/// Tiled-2D octahedral atlas dimensions in texels for `probe_count` probes whose
/// interior octahedral resolution is `interior` (so each padded tile is
/// `interior + 2` texels square, including the one-texel gutter border).
///
/// Lays the tiles out in a near-square grid, `per_row = ceil(sqrt(probe_count))`
/// tiles across. The width is an exact multiple of the padded tile size so the
/// shader's integer `atlas_width / padded` recovers `per_row` exactly (see
/// `ddgi_sample.wesl` `probe_tile_origin`).
pub(crate) fn atlas_dims(probe_count: u32, interior: u32) -> UVec2 {
    let padded = interior.max(1) + 2;
    let probes = probe_count.max(1);
    // Integer ceil-sqrt (no float transcendental): smallest `per_row` with
    // `per_row * per_row >= probes`.
    let mut per_row = 1u32;
    while per_row * per_row < probes {
        per_row += 1;
    }
    let rows = probes.div_ceil(per_row);
    UVec2::new(per_row * padded, rows * padded)
}

/// The per-view DDGI resources, present only while the passes are enabled and
/// their backing prepass + visibility buffers are resident.
#[derive(Component)]
pub(crate) struct ViewDdgi {
    /// Octahedral *irradiance* atlas (`rgba16float`): one padded tile per probe,
    /// `rgb` = irradiance. Written by `probe_update_main` (storage) and read by
    /// `sample_main` (texture).
    irradiance_atlas: CachedTexture,
    /// Octahedral *depth / visibility* atlas (`rgba16float`): one padded tile per
    /// probe, `rg` = `[mean, mean_sq]` depth moments. Written by
    /// `probe_update_main` (storage) and read by `sample_main` (texture).
    depth_atlas: CachedTexture,
    /// Per-probe [`ProbeMeta`] storage buffer: `probe_count` x
    /// [`PROBE_META_STRIDE`] bytes (relocation offset + activity flag), written
    /// by the relocation / classification stage and read by `sample_main`.
    probe_meta: Buffer,
    /// Uniform-buffer twin of the lattice + field metadata ([`GpuDdgiVolume`]),
    /// bound at group(0) binding 2 of `sample_main`. Re-uploaded whenever the
    /// configured lattice changes (static for the view's lifetime otherwise).
    volume_uniform: Buffer,
    /// Full-resolution diffuse GI irradiance export (`rgba16float`): `rgb` =
    /// irradiance, `a` = blend confidence. Written by `sample_main` and sampled
    /// by the composite; never copied over `scene_color`.
    gi_out: CachedTexture,
    /// Full-resolution scratch copy of `scene_color` (`rgba16float`) the
    /// composite's copy pass lifts the shaded colour into before the fold pass
    /// reads it, side-stepping the read/write aliasing hazard on the
    /// single-sample scene-colour target.
    gi_base: CachedTexture,
    /// Viewport extent the full-resolution textures were sized for; drives
    /// reallocation on resize.
    size: UVec2,
    /// Probe count the atlases + metadata buffer were sized for; drives
    /// reallocation when the configured lattice changes.
    probe_count: u32,
}

impl ViewDdgi {
    /// Storage/texture view of the octahedral irradiance atlas.
    pub(crate) fn irradiance_atlas_view(&self) -> &TextureView {
        &self.irradiance_atlas.default_view
    }

    /// Storage/texture view of the octahedral depth / visibility atlas.
    pub(crate) fn depth_atlas_view(&self) -> &TextureView {
        &self.depth_atlas.default_view
    }

    /// Per-probe [`ProbeMeta`] storage buffer.
    pub(crate) fn probe_meta(&self) -> &Buffer {
        &self.probe_meta
    }

    /// Uniform buffer carrying the [`GpuDdgiVolume`] lattice + field metadata.
    pub(crate) fn volume_uniform(&self) -> &Buffer {
        &self.volume_uniform
    }

    /// Storage/texture view of the GI irradiance export written by `sample_main`.
    pub(crate) fn gi_out_view(&self) -> &TextureView {
        &self.gi_out.default_view
    }

    /// View of the scratch `scene_color` copy the composite's copy pass writes
    /// and its fold pass reads.
    pub(crate) fn gi_base_view(&self) -> &TextureView {
        &self.gi_base.default_view
    }

    /// Probe count the atlases + metadata buffer were sized for.
    pub(crate) fn probe_count(&self) -> u32 {
        self.probe_count
    }

    /// Full-resolution framebuffer extent the GI export was sized for.
    pub(crate) fn size(&self) -> UVec2 {
        self.size
    }
}

/// (Re)allocates [`ViewDdgi`] for every view whose SSR prepass and visibility
/// buffers are resident while the DDGI is enabled, and removes it otherwise.
///
/// Gated on `settings.enabled`, the presence of [`ViewSsrTextures`] (depth +
/// `normal_roughness`) and [`ViewVisibilityBuffer`] (scene colour), and on
/// single-sample views (the scene colour buffer is itself single-sample). The
/// probe-meta buffer + atlases are re-created whenever the configured probe
/// count changes; the full-resolution textures whenever the viewport resizes.
pub(crate) fn prepare_ddgi_textures(
    mut commands: Commands,
    settings: Res<PrismDdgiSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewSsrTextures>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewDdgi>,
    )>,
) {
    for (entity, camera, msaa, ssr, visibility, existing) in &views {
        let enabled = settings.enabled
            && ssr.is_some()
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewDdgi>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };

        // Probe count follows the golden lattice the settings are configured
        // with, so the CPU reference and the device allocation agree exactly.
        let (_grid, volume) = settings.configured_volume();
        let probe_count = volume.probe_count().max(1);

        if existing.is_some_and(|r| r.size == size && r.probe_count == probe_count) {
            continue;
        }

        let irradiance_dims = atlas_dims(probe_count, settings.irradiance_interior);
        let depth_dims = atlas_dims(probe_count, settings.depth_interior);

        // Octahedral irradiance atlas: tiled 2D, written by the probe update
        // (storage) and sampled by `sample_main` (texture).
        let irradiance_atlas = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism DDGI irradiance atlas"),
                size: irradiance_dims.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DDGI_ATLAS_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Octahedral depth / visibility atlas: tiled 2D, `rg` = depth moments.
        let depth_atlas = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism DDGI depth atlas"),
                size: depth_dims.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DDGI_ATLAS_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Per-probe relocation offset + activity flag. STORAGE: written by the
        // relocation / classification stage, read-only by `sample_main`.
        // COPY_DST so a zero-init clear can be scheduled when the lattice is
        // (re)allocated.
        let probe_meta = device.create_buffer(&BufferDescriptor {
            label: Some("prism DDGI probe meta"),
            size: probe_count as u64 * PROBE_META_STRIDE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Lattice + field metadata uniform, uploaded once per (re)allocation;
        // `sample_main` reads it at group(0) binding 2.
        let volume_uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism DDGI volume"),
            contents: bytemuck::bytes_of(&volume),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // GI export: full-resolution wide HDR, written by `sample_main`
        // (storage) and sampled by the composite (texture).
        let gi_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism DDGI output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Scratch copy of `scene_color` the composite's copy pass lifts the
        // shaded colour into (the fold pass then `textureLoad`s it), matching
        // SSGI / `world_space_gi`'s `gi_base`.
        let gi_base = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism DDGI base"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewDdgi {
            irradiance_atlas,
            depth_atlas,
            probe_meta,
            volume_uniform,
            gi_out,
            gi_base,
            size,
            probe_count,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_meta_stride_is_one_vec4_row() {
        // vec3<f32> offset + f32 flag = one 16-byte std430 row.
        assert_eq!(PROBE_META_STRIDE, 16);
        assert_eq!(PROBE_META_STRIDE % 16, 0);
    }

    #[test]
    fn atlas_format_is_storage_capable_wide_hdr() {
        assert_eq!(
            DDGI_ATLAS_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn gi_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn atlas_dims_tile_width_recovers_per_row() {
        // 6x6 irradiance interior -> 8-texel padded tiles. The shader recovers
        // `per_row` as `atlas_width / padded`; the allocation must make that
        // integer division exact for every probe count.
        for &probes in &[1u32, 2, 3, 4, 5, 16, 17, 2048] {
            for &interior in &[6u32, 16] {
                let padded = interior + 2;
                let dims = atlas_dims(probes, interior);
                // Width is an exact multiple of the padded tile size.
                assert_eq!(dims.x % padded, 0, "width not tile-aligned");
                assert_eq!(dims.y % padded, 0, "height not tile-aligned");
                let per_row = dims.x / padded;
                let rows = dims.y / padded;
                // The near-square grid holds every probe.
                assert!(per_row >= 1 && rows >= 1);
                assert!(
                    per_row * rows >= probes,
                    "atlas cannot hold all probes: {per_row}x{rows} < {probes}"
                );
                // Near-square: one row of slack at most beyond the ceil-sqrt.
                assert!(per_row * per_row >= probes.max(1));
            }
        }
    }

    #[test]
    fn atlas_dims_handles_the_degenerate_zero_count() {
        // A zero probe count is clamped to a single tile rather than a
        // zero-area texture.
        let dims = atlas_dims(0, 6);
        assert_eq!(dims, UVec2::new(8, 8));
    }
}
