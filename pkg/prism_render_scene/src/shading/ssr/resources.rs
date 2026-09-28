//! Per-view GPU textures backing the SSR geometry prepass.
//!
//! SSR runs as a chain of compute steps between the visibility raster and the
//! shading resolve. This slice lands the *geometry prepass*, which decodes the
//! visibility buffer into the two inputs the trace consumes (there is no
//! G-buffer to sample):
//!
//! 1. `scene_depth` — reverse-Z device depth (`R32Float`), `0` = background, and
//! 2. `view_normal` — the unit view-space surface normal (`Rgba16Float`).
//!
//! plus the reverse-Z "nearest depth" Hi-Z pyramid the trace marches. This
//! module owns the cached textures (and the pyramid's per-mip views) and the
//! [`ExtractedCamera`]-driven prepare system that (re)allocates them to match
//! the viewport, mirroring [`super::super::ao::prepare_gtao_textures`] and
//! [`super::super::resources::prepare_visibility_buffers`]. The roughness repack
//! and the trace target land in following slices.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;

/// Reverse-Z device depth written by the SSR geometry prepass and later climbed
/// by the Hi-Z march. `R32Float` keeps the post-divide depth at full precision
/// so the screen-space trace reconstructs view positions exactly like the
/// golden `reconstruct_view_position`.
pub(crate) const SSR_DEPTH_FORMAT: TextureFormat = TextureFormat::R32Float;
/// View-space unit normal written by the prepass. `Rgba16Float` holds a signed
/// unit vector with headroom to spare; the roughness repack folds it into the
/// trace's `normal_roughness` input in a following slice.
pub(crate) const SSR_NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// View-space normal (rgb, biased to `[0, 1]`) plus perceptual roughness (a),
/// the single `normal_roughness` texture the trace samples. The repack folds the
/// prepass's signed `view_normal` and the material roughness into this format so
/// `ssr.wesl` can decode `nr.xyz * 2 - 1` for the normal and read `nr.w` for the
/// roughness in one `textureLoad`.
pub(crate) const SSR_NORMAL_ROUGHNESS_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Reverse-Z "nearest depth" Hi-Z pyramid climbed by the SSR march. Shares the
/// full-precision `R32Float` device-depth encoding of [`SSR_DEPTH_FORMAT`] so a
/// coarse cell's stored maximum compares directly against the per-pixel depth
/// the trace reconstructs. Allocated with a full mip chain; level 0 is a copy
/// of `scene_depth` and each coarser level is a 2x2 max-reduction of the finer
/// one, mirroring the CPU golden pyramid the shader twin marches.
pub(crate) const SSR_HZB_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Current-frame scene-colour mip pyramid sampled by the trace. Matches the
/// resolve's [`super::super::resources::SCENE_COLOR_FORMAT`] byte for byte so
/// level 0 is a loss-free copy of the shaded HDR radiance and the coarser
/// box-filtered levels carry the pre-blurred reflection rougher surfaces read.
pub(crate) const SSR_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Reflection buffer the trace writes (`rgb` = reflected radiance, `a` = blend
/// confidence) and the composite blends over the shaded scene colour. Wide HDR
/// so the reflected radiance keeps its range up to the composite.
pub(crate) const SSR_OUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The two per-view SSR prepass textures, present only while SSR is enabled and
/// the viewport size is known.
#[derive(Component)]
pub(crate) struct ViewSsrTextures {
    scene_depth: CachedTexture,
    view_normal: CachedTexture,
    /// Trace input packed by the repack: `rgb = view_normal * 0.5 + 0.5`,
    /// `a = perceptual roughness`. Written by `ssr_repack.wesl`, sampled by the
    /// trace.
    normal_roughness: CachedTexture,
    /// Full mip-chain Hi-Z pyramid built from `scene_depth`. Level 0 is a copy
    /// of `scene_depth`; each coarser level is a 2x2 max-reduction of the one
    /// below (reverse-Z nearest), matching the CPU golden. Retained to keep the
    /// backing texture out of the [`TextureCache`] free list while its per-mip
    /// views are live; the multi-mip [`Self::hzb_view`] the trace climbs is
    /// wired in the trace slice.
    hzb: CachedTexture,
    /// One single-mip view per pyramid level, used both as the storage write
    /// target when building that level and as the sampled source when reducing
    /// into the next one. `hzb_mip_views[0]` is the mip-0 copy target.
    hzb_mip_views: Vec<TextureView>,
    /// Number of pyramid levels (including level 0), i.e. `hzb_mip_views.len()`.
    hzb_mip_count: u32,
    /// Current-frame scene-colour mip pyramid the trace samples for reflected
    /// radiance. Level 0 is a 1:1 copy of the resolve's `scene_color`; each
    /// coarser level is its 2x2 box average, built every frame *after* the
    /// resolve by [`super::color_mips`]. Purely intra-frame (a [`TextureCache`]
    /// transient, recycled next frame) — the trace reads the current frame's
    /// own colour, never a reprojected previous frame, so no cross-frame
    /// history, motion vectors, or reprojection matrices are involved.
    color: CachedTexture,
    /// One single-mip view per colour-pyramid level (the copy/reduce write
    /// target for its level and the reduce source for the next).
    color_mip_views: Vec<TextureView>,
    /// Number of colour-pyramid levels (including level 0).
    color_mip_count: u32,
    /// Reflection output written by the trace (`rgb` radiance, `a` confidence)
    /// and consumed by the composite. Single mip, full resolution.
    ssr_out: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewSsrTextures {
    /// Storage/sampling view of the reverse-Z device-depth prepass target.
    pub(crate) fn scene_depth_view(&self) -> &TextureView {
        &self.scene_depth.default_view
    }

    /// Storage/sampling view of the view-space normal prepass target.
    pub(crate) fn view_normal_view(&self) -> &TextureView {
        &self.view_normal.default_view
    }

    /// Storage/sampling view of the packed `normal_roughness` trace input. The
    /// repack writes it (storage) and the trace reads it (sampled).
    pub(crate) fn normal_roughness_view(&self) -> &TextureView {
        &self.normal_roughness.default_view
    }

    /// The full-precision reverse-Z device depth as a *sampled* view; the Hi-Z
    /// build copies it into pyramid level 0 and the trace reconstructs view
    /// positions from it.
    pub(crate) fn scene_depth_sampled(&self) -> &TextureView {
        &self.scene_depth.default_view
    }

    /// Multi-mip sampling view spanning the whole Hi-Z pyramid, bound by the
    /// trace so `textureLoad(hzb, cell, level)` climbs every level.
    pub(crate) fn hzb_view(&self) -> &TextureView {
        &self.hzb.default_view
    }

    /// Number of pyramid levels, including level 0.
    pub(crate) fn hzb_mip_count(&self) -> u32 {
        self.hzb_mip_count
    }

    /// Single-mip view of pyramid level `mip`, or `None` when out of range.
    pub(crate) fn hzb_mip_view(&self, mip: u32) -> Option<&TextureView> {
        self.hzb_mip_views.get(mip as usize)
    }

    /// Texel extent of pyramid level `mip` (`max(size >> mip, 1)` per axis),
    /// the standard `wgpu` mip dimension the allocated texture actually holds.
    pub(crate) fn hzb_mip_size(&self, mip: u32) -> bevy_math::UVec2 {
        mip_size(self.size, mip)
    }

    /// All-mip sampling view of the current-frame colour pyramid, bound by the
    /// trace so `textureSampleLevel(color, sampler, uv, mip)` climbs every
    /// level for the roughness-selected reflection blur.
    pub(crate) fn color_sampled_view(&self) -> &TextureView {
        &self.color.default_view
    }

    /// Single-mip view of colour-pyramid level `mip`, or `None` when out of
    /// range. `color_mip_view(0)` is the copy target; coarser levels are reduce
    /// targets (and, one finer, the reduce source).
    pub(crate) fn color_mip_view(&self, mip: u32) -> Option<&TextureView> {
        self.color_mip_views.get(mip as usize)
    }

    /// Number of colour-pyramid levels, including level 0.
    pub(crate) fn color_mip_count(&self) -> u32 {
        self.color_mip_count
    }

    /// Texel extent of colour-pyramid level `mip` (`max(size >> mip, 1)` per
    /// axis), matching the allocated texture's mip dimension.
    pub(crate) fn color_mip_size(&self, mip: u32) -> bevy_math::UVec2 {
        mip_size(self.size, mip)
    }

    /// Storage/sampling view of the trace's reflection output. The trace writes
    /// it (storage) and the composite reads it (sampled).
    pub(crate) fn ssr_out_view(&self) -> &TextureView {
        &self.ssr_out.default_view
    }
}

/// Number of mip levels a full pyramid over `size` needs, including level 0.
///
/// This is the standard `wgpu` mip count, `1 + floor(log2(max_dim))`, so the
/// coarsest level is exactly `1x1`. (The CPU golden's `div_ceil` chain can be
/// one level longer for non-power-of-two extents; the trace samples whatever
/// levels the GPU texture actually holds via `textureNumLevels`, and the march
/// logic — not the pyramid's coarse-level rounding — is what mirrors the golden
/// bit-for-bit.)
pub(crate) fn hzb_mip_count(size: bevy_math::UVec2) -> u32 {
    let max_dim = size.x.max(size.y).max(1);
    32 - max_dim.leading_zeros()
}

/// Texel extent of pyramid level `mip`: each axis is `max(size >> mip, 1)`,
/// the floor-halving `wgpu` uses for texture mip dimensions.
fn mip_size(size: bevy_math::UVec2, mip: u32) -> bevy_math::UVec2 {
    bevy_math::UVec2::new(
        (size.x >> mip).max(1),
        (size.y >> mip).max(1),
    )
}

/// (Re)allocates [`ViewSsrTextures`] for every view that has a resident
/// visibility buffer while SSR is enabled, and removes them otherwise.
///
/// Gated on both `enable_ssr` and `enable_visibility_buffer`: SSR's geometry
/// prepass decodes the visibility buffer, so it is meaningless without it, and
/// on single-sample views because the visibility buffer it decodes is itself
/// single-sample. The textures are re-created whenever the viewport size
/// changes, exactly like the visibility buffer they shadow.
pub(crate) fn prepare_ssr_textures(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSsrTextures>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enable_ssr
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewSsrTextures>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        let scene_depth = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR scene depth"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_DEPTH_FORMAT,
                // Written by the prepass, sampled by the HZB build + trace.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let view_normal = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR view normal"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_NORMAL_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let normal_roughness = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR normal roughness"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_NORMAL_ROUGHNESS_FORMAT,
                // Written by the repack, sampled by the trace.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Hi-Z pyramid: a full `1 + floor(log2(max_dim))` mip chain in the same
        // `R32Float` device-depth encoding. `STORAGE_BINDING` lets each build
        // pass write one level; `TEXTURE_BINDING` lets the reduce read the finer
        // level and the trace climb every level.
        let mip_count = hzb_mip_count(size);
        let hzb = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR HZB"),
                size: size.to_extents(),
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_HZB_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        // One single-mip view per level. Each is the storage write target when
        // building its level and the sampled source when reducing into the next.
        let hzb_mip_views = (0..mip_count)
            .map(|mip| {
                hzb.texture.create_view(&TextureViewDescriptor {
                    label: Some("prism SSR HZB mip view"),
                    dimension: Some(TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Current-frame scene-colour pyramid: a full `R16g16b16a16Float` mip
        // chain matching the resolve's `scene_color`. `STORAGE_BINDING` lets the
        // colour-mip build write each level; `TEXTURE_BINDING` lets the reduce
        // read the finer level and the trace climb every level. It is a
        // frame-transient (`TextureCache`) target — the trace samples the
        // current frame's own colour, never a reprojected previous frame.
        let color = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR colour pyramid"),
                size: size.to_extents(),
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let color_mip_views = (0..mip_count)
            .map(|mip| {
                color.texture.create_view(&TextureViewDescriptor {
                    label: Some("prism SSR colour mip view"),
                    dimension: Some(TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Reflection output: single-mip, full-resolution wide HDR. The trace
        // writes it (storage) and the composite reads it (sampled).
        let ssr_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR reflection output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSR_OUT_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewSsrTextures {
            scene_depth,
            view_normal,
            normal_roughness,
            hzb,
            hzb_mip_views,
            hzb_mip_count: mip_count,
            color,
            color_mip_views,
            color_mip_count: mip_count,
            ssr_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::UVec2;

    #[test]
    fn ssr_texture_formats_match_the_shader_bindings() {
        // Device depth is scalar float storage; the view normal needs three
        // signed channels, so a wide RGBA16F carries it.
        assert_eq!(SSR_DEPTH_FORMAT, TextureFormat::R32Float);
        assert_eq!(SSR_NORMAL_FORMAT, TextureFormat::Rgba16Float);
        // The trace's single packed input shares the wide RGBA16F layout so it
        // can carry the biased normal in rgb and roughness in a.
        assert_eq!(SSR_NORMAL_ROUGHNESS_FORMAT, TextureFormat::Rgba16Float);
        // The pyramid shares the device-depth encoding so a coarse cell's
        // stored maximum compares directly against reconstructed depth.
        assert_eq!(SSR_HZB_FORMAT, TextureFormat::R32Float);
    }

    #[test]
    fn hzb_mip_count_is_the_standard_wgpu_chain_length() {
        // 1 + floor(log2(max_dim)); the coarsest level is exactly 1x1.
        assert_eq!(hzb_mip_count(UVec2::new(1, 1)), 1);
        assert_eq!(hzb_mip_count(UVec2::new(2, 1)), 2);
        assert_eq!(hzb_mip_count(UVec2::new(1024, 1024)), 11);
        assert_eq!(hzb_mip_count(UVec2::new(1920, 1080)), 11);
        // Degenerate zero extents never underflow: a single level survives.
        assert_eq!(hzb_mip_count(UVec2::ZERO), 1);
    }

    #[test]
    fn mip_size_floor_halves_each_level_to_one() {
        let size = UVec2::new(1920, 1080);
        assert_eq!(mip_size(size, 0), size);
        assert_eq!(mip_size(size, 1), UVec2::new(960, 540));
        assert_eq!(mip_size(size, 2), UVec2::new(480, 270));
        // Floor halving: 1080 >> 3 = 135, and the last level clamps to 1x1.
        assert_eq!(mip_size(size, 3), UVec2::new(240, 135));
        let last = hzb_mip_count(size) - 1;
        assert_eq!(mip_size(size, last), UVec2::new(1, 1));
    }
}
