//! Per-view and persistent `GPU` resources backing the surface-cache passes.
//!
//! Unlike the stateless world-space `GI` subsystem, the surfel radiance cache
//! is *persistent*: its accumulated radiance (`EMA` state) must survive the
//! per-frame render-world entity rebuild. The surfel storage buffer is
//! therefore owned by a [`SurfaceCacheBuffers`] resource keyed by the stable
//! [`RetainedViewEntity`] (modelled on `virtual_shadow`'s
//! `VsmReceiverBufferCache`), not by a per-view component that would be dropped
//! and reallocated every frame.
//!
//! Each frame four same-frame compute passes run over the surfel buffers:
//!
//! * `surface_cache_alloc_main` seeds a fresh surfel per screen tile into the
//!   `scratch_current` buffer.
//! * `surface_cache_update_main` blends `scratch_current` into the persistent
//!   `surfels` buffer via the golden confidence-weighted `EMA`.
//! * `surface_cache_spatial_filter_main` bilaterally filters `surfels` into the
//!   `scratch_filtered` buffer.
//! * `surface_cache_coverage_main` gathers `scratch_filtered` per pixel into
//!   the `gi_out` export texture.
//!
//! The three surfel storage buffers persist across frames (keyed by the view);
//! the two scratch textures (`gi_out`, `gi_base`) are transient and come from
//! the per-frame [`TextureCache`]. The prepare system is gated on the enable
//! *and* on the presence of the `SSR` prepass textures (reverse-Z depth +
//! packed `normal_roughness`) and the visibility buffer (pre-exposed scene
//! colour), on single-sample views.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureUsages,
        TextureView,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, RetainedViewEntity},
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::ssr::ViewSsrTextures;
use super::abi::GpuSurfel;
use super::settings::PrismSurfaceCacheSettings;

/// Per-surfel storage stride in bytes: the all-scalar [`GpuSurfel`] std430
/// element (48 bytes).
pub(crate) const SURFEL_STRIDE: u64 = size_of::<GpuSurfel>() as u64;

/// One view's cached surfel storage: the persistent accumulation buffer plus
/// the two per-frame scratch buffers, and the extent / grid they were sized
/// for so a resize can detect the mismatch and reallocate all three together.
#[derive(Clone)]
struct CachedSurfaceCache {
    /// Persistent per-surfel `EMA` accumulation buffer (survives frames).
    surfels: Buffer,
    /// Scratch buffer of the frame's freshly sampled surfels (`alloc` out,
    /// `update` in).
    scratch_current: Buffer,
    /// Scratch buffer of the spatially filtered surfels (`spatial_filter` out,
    /// `coverage` in).
    scratch_filtered: Buffer,
    /// Full-resolution framebuffer extent this allocation was sized for.
    size: UVec2,
    /// Surfel-grid dimensions this allocation was sized for.
    surfel_grid: UVec2,
}

/// Render-world cache of each view's persistent surfel storage, keyed by its
/// stable [`RetainedViewEntity`]. A view that persists across frames with an
/// unchanged viewport reuses the same surfel allocation (so radiance
/// accumulates); a resize rebuilds it and a vanished view is dropped so buffers
/// never leak.
#[derive(Resource, Default)]
pub(crate) struct SurfaceCacheBuffers {
    buffers: HashMap<RetainedViewEntity, CachedSurfaceCache>,
}

impl SurfaceCacheBuffers {
    /// Drops every cached allocation (used when the feature is disabled, so
    /// nothing lingers resident).
    fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Returns the cached surfel storage for `retained`, (re)allocating all
    /// three buffers when absent or sized for a different viewport / grid.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        size: UVec2,
        surfel_grid: UVec2,
    ) -> CachedSurfaceCache {
        let needs_new = self
            .buffers
            .get(&retained)
            .is_none_or(|cached| cached.size != size || cached.surfel_grid != surfel_grid);
        if needs_new {
            let count = u64::from(surfel_grid.x) * u64::from(surfel_grid.y);
            let byte_size = (count.max(1)) * SURFEL_STRIDE;
            let surfels = device.create_buffer(&BufferDescriptor {
                label: Some("prism surface cache surfels"),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let scratch_current = device.create_buffer(&BufferDescriptor {
                label: Some("prism surface cache scratch current"),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let scratch_filtered = device.create_buffer(&BufferDescriptor {
                label: Some("prism surface cache scratch filtered"),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cached = CachedSurfaceCache {
                surfels,
                scratch_current,
                scratch_filtered,
                size,
                surfel_grid,
            };
            self.buffers.insert(retained, cached.clone());
            cached
        } else {
            self.buffers
                .get(&retained)
                .expect("surfel buffers present after the is_none_or check")
                .clone()
        }
    }

    /// Drops any cached allocation whose view was not seen this frame.
    fn retain_seen(&mut self, seen: &HashSet<RetainedViewEntity>) {
        self.buffers.retain(|retained, _| seen.contains(retained));
    }
}

/// The per-view surface-cache resources, present only while the passes are
/// enabled and their backing prepass + visibility buffers are resident. Holds
/// cloned handles to the persistent surfel buffers plus the per-frame `GI`
/// export / base scratch textures.
#[derive(Component)]
pub(crate) struct ViewSurfaceCache {
    /// Persistent per-surfel `EMA` accumulation buffer (read-write by
    /// `update`, read-only by `spatial_filter`).
    surfels: Buffer,
    /// Scratch buffer of the frame's freshly sampled surfels.
    scratch_current: Buffer,
    /// Scratch buffer of the spatially filtered surfels.
    scratch_filtered: Buffer,
    /// Full-resolution diffuse `GI` irradiance export (`rgba16float`): `rgb` =
    /// irradiance, `a` = coverage confidence. Written by `coverage` and read by
    /// the composite fold.
    gi_out: CachedTexture,
    /// Scratch copy of `scene_color` the composite's copy pass lifts the shaded
    /// colour into before the fold pass reads it, side-stepping the read/write
    /// aliasing hazard on the single `rgba16float` `scene_color` storage image.
    gi_base: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
    /// Surfel-grid dimensions (`div_ceil(size, tile)` per axis).
    pub(crate) surfel_grid: UVec2,
}

impl ViewSurfaceCache {
    /// The persistent surfel accumulation buffer.
    pub(crate) fn surfels(&self) -> &Buffer {
        &self.surfels
    }

    /// The freshly sampled scratch buffer.
    pub(crate) fn scratch_current(&self) -> &Buffer {
        &self.scratch_current
    }

    /// The spatially filtered scratch buffer.
    pub(crate) fn scratch_filtered(&self) -> &Buffer {
        &self.scratch_filtered
    }

    /// Storage view of the `GI` irradiance export written by `coverage`.
    pub(crate) fn gi_out_view(&self) -> &TextureView {
        &self.gi_out.default_view
    }

    /// View of the scratch `scene_color` copy the composite copy pass writes
    /// and its fold pass reads.
    pub(crate) fn gi_base_view(&self) -> &TextureView {
        &self.gi_base.default_view
    }
}

/// (Re)allocates the persistent surfel buffers and the per-frame `GI` textures
/// for every view whose `SSR` prepass and visibility buffers are resident while
/// the surface cache is enabled, and removes the per-view component / clears the
/// persistent cache otherwise.
///
/// Gated on `settings.enabled`, the presence of [`ViewSsrTextures`] (depth +
/// `normal_roughness`) and [`ViewVisibilityBuffer`] (scene colour), and on
/// single-sample views. The surfel buffers persist across frames via
/// [`SurfaceCacheBuffers`]; the textures are per-frame. Read and written in the
/// same frame's passes, so there is no `+1`-frame latency.
pub(crate) fn prepare_surface_cache_resources(
    mut commands: Commands,
    settings: Res<PrismSurfaceCacheSettings>,
    mut texture_cache: ResMut<TextureCache>,
    mut cache: ResMut<SurfaceCacheBuffers>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewSsrTextures>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewSurfaceCache>,
    )>,
) {
    if !settings.enabled {
        cache.clear();
        return;
    }

    let tile = settings.tile.max(1);
    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, camera, msaa, ssr, visibility, existing) in &views {
        let enabled =
            ssr.is_some() && visibility.is_some() && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewSurfaceCache>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if size.x == 0 || size.y == 0 {
            continue;
        }

        let surfel_grid = UVec2::new(size.x.div_ceil(tile), size.y.div_ceil(tile));
        let retained = view.retained_view_entity;
        let cached = cache.get_or_create(&device, retained, size, surfel_grid);
        seen.insert(retained);

        // GI export: full-resolution wide HDR, written by `coverage` (storage)
        // and read by the composite fold (texture).
        let gi_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism surface cache output"),
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
        // shaded colour into (the fold pass then `textureLoad`s it).
        let gi_base = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism surface cache base"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewSurfaceCache {
            surfels: cached.surfels,
            scratch_current: cached.scratch_current,
            scratch_filtered: cached.scratch_filtered,
            gi_out,
            gi_base,
            size,
            surfel_grid,
        });
    }
    cache.retain_seen(&seen);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surfel_stride_is_the_scalar_layout() {
        // The all-scalar GpuSurfel is 48 bytes, a multiple of the 16-byte
        // std430 base alignment and never zero.
        assert_eq!(SURFEL_STRIDE, 48);
        assert_eq!(SURFEL_STRIDE % 16, 0);
    }

    #[test]
    fn gi_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
