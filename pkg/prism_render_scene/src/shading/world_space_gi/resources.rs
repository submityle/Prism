//! Per-view GPU resources backing the world-space GI passes.
//!
//! The subsystem runs as two same-frame compute passes over a shared probe
//! storage buffer:
//!
//! * `probe_update_main` writes one L1 SH radiance probe per screen tile into
//!   the probe storage buffer.
//! * `resolve_main` reads that buffer back, interpolates the four probes
//!   around each pixel and writes the diffuse GI irradiance into `gi_out`.
//!
//! This module owns both: the probe storage `Buffer` (sized from the golden
//! [`prism_render_shading::probe_count`], recreated on viewport resize) and the
//! `gi_out` [`CachedTexture`] (`rgba16float`, `STORAGE_BINDING | TEXTURE_BINDING`
//! — a GI *export* buffer a downstream resolve samples, never copied back over
//! `scene_color`, matching SSGI's `ssgi_out`).
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates them to match
//! the viewport, gated on the GI enable *and* on the presence of the SSR
//! prepass textures (the reverse-Z depth + packed `normal_roughness` both
//! passes reconstruct from) and the visibility buffer (the pre-exposed scene
//! colour the probe capture gathers), on single-sample views.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureUsages,
        TextureView,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};
use prism_render_shading::{probe_count, probe_grid_dims};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::ssr::ViewSsrTextures;
use super::settings::PrismWorldSpaceGiSettings;

/// Per-probe storage stride in bytes: six `vec4<f32>` (four SH coefficients,
/// the probe meta and the valid flag) = 96 bytes, matching the WGSL `Probe`
/// struct's std430 layout and its 16-byte array stride.
pub(crate) const PROBE_STRIDE: u64 = 96;

/// The per-view world-space GI resources, present only while the passes are
/// enabled and their backing prepass + visibility buffers are resident.
#[derive(Component)]
pub(crate) struct ViewWorldSpaceGi {
    /// Screen-probe storage buffer: `probe_count` x [`PROBE_STRIDE`] bytes,
    /// written by `probe_update_main` (read-write) and read by `resolve_main`
    /// (read-only).
    probe_buffer: Buffer,
    /// Full-resolution diffuse GI irradiance export (`rgba16float`): `rgb` =
    /// irradiance, `a` = blend confidence. Written by `resolve_main` and
    /// sampled by downstream consumers; never copied over `scene_color`.
    gi_out: CachedTexture,
    /// Full-resolution scratch copy of `scene_color` (`rgba16float`) the
    /// composite's copy pass lifts the shaded colour into before the fold
    /// pass reads it, side-stepping the read/write aliasing hazard on the
    /// single `rgba16float` `scene_color` storage image (mirrors SSGI's
    /// `gi_base`).
    gi_base: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
    /// Probe-grid dimensions (`div_ceil(size, tile)` per axis), the dispatch
    /// extent for `probe_update_main`.
    pub(crate) probe_grid: UVec2,
}

impl ViewWorldSpaceGi {
    /// The screen-probe storage buffer.
    pub(crate) fn probe_buffer(&self) -> &Buffer {
        &self.probe_buffer
    }

    /// Storage view of the GI irradiance export written by `resolve_main`.
    pub(crate) fn gi_out_view(&self) -> &TextureView {
        &self.gi_out.default_view
    }

    /// View of the scratch `scene_color` copy the composite's copy pass
    /// writes and its fold pass reads (write in copy, `textureLoad` in fold).
    pub(crate) fn gi_base_view(&self) -> &TextureView {
        &self.gi_base.default_view
    }
}

/// (Re)allocates [`ViewWorldSpaceGi`] for every view whose SSR prepass and
/// visibility buffers are resident while the GI is enabled, and removes it
/// otherwise.
///
/// Gated on `settings.enabled`, the presence of [`ViewSsrTextures`] (depth +
/// `normal_roughness`) and [`ViewVisibilityBuffer`] (scene colour), and on
/// single-sample views (the scene colour buffer is itself single-sample). The
/// probe buffer and the GI texture are re-created whenever the viewport size
/// changes.
pub(crate) fn prepare_world_space_gi_textures(
    mut commands: Commands,
    settings: Res<PrismWorldSpaceGiSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewSsrTextures>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewWorldSpaceGi>,
    )>,
) {
    for (entity, camera, msaa, ssr, visibility, existing) in &views {
        let enabled = settings.enabled
            && ssr.is_some()
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewWorldSpaceGi>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|resources| resources.size == size) {
            continue;
        }

        // Probe grid + count follow the golden `probe_grid_dims` / `probe_count`
        // so the CPU reference and the device allocation agree exactly.
        let probe_grid = probe_grid_dims(size, settings.tile);
        let count = probe_count(size, settings.tile).max(1);

        let probe_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism world-space GI probes"),
            size: count as u64 * PROBE_STRIDE,
            // STORAGE: bound read-write by probe_update, read-only by resolve.
            // COPY_DST so a zero-init clear can be scheduled if ever needed.
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GI export: full-resolution wide HDR, written by `resolve_main`
        // (storage) and sampled by downstream consumers (texture).
        let gi_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism world-space GI output"),
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
        // SSGI's `gi_base`. Same wide-HDR format + storage/texture usages as
        // `gi_out`; the fold reads it via a non-filterable sampled binding and
        // the copy pass writes it via a storage binding.
        let gi_base = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism world-space GI base"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewWorldSpaceGi {
            probe_buffer,
            gi_out,
            gi_base,
            size,
            probe_grid,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_stride_is_the_six_vec4_layout() {
        // Six `vec4<f32>` (four SH coefficients + meta + flags) = 96 bytes,
        // a multiple of the 16-byte std430 array stride.
        assert_eq!(PROBE_STRIDE, 96);
        assert_eq!(PROBE_STRIDE % 16, 0);
    }

    #[test]
    fn gi_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
