//! Per-view froxel storage volumes for the volumetric-fog passes and the
//! per-frame immediate blocks the passes upload.
//!
//! The fog runs two chained compute passes over a view-frustum-fitted 3D grid:
//! **scatter** writes each froxel's source radiance + slice thickness and its
//! extinction, then **integrate** reads those back and marches each column
//! front-to-back into the integrated in-scattering + transmittance volumes. That
//! needs four `rgba16float` 3D storage textures per view:
//!
//! * `froxel_scattering` — rgb = source radiance (`in_scatter + emissive`),
//!   a = slice thickness (scatter writes, integrate reads);
//! * `froxel_extinction` — rgb = `sigma_t` (scatter writes, integrate reads);
//! * `integrated_scattering` — rgb = camera-to-slice accumulated in-scattering
//!   (integrate writes); and
//! * `integrated_transmittance` — rgb = camera-to-slice transmittance
//!   (integrate writes).
//!
//! All four carry `STORAGE_BINDING | TEXTURE_BINDING` (the scatter volumes are
//! written as storage by scatter and read as sampled textures by integrate) and
//! are cached across frames keyed by [`RetainedViewEntity`], reallocated only
//! when the froxel grid dimensions change, so a steady-state camera never churns
//! GPU allocations — mirroring [`super::super::virtual_shadow`]'s per-view
//! receiver-buffer cache.

use bevy_ecs::prelude::*;
use bevy_math::UVec3;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_resource::{
        Extent3d, Texture, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::RenderDevice,
    view::{ExtractedView, RetainedViewEntity},
};

use super::abi::{GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams};
use super::pipeline::VOLUMETRICS_FROXEL_FORMAT;
use super::settings::PrismVolumetricsSettings;

/// Per-view froxel state resolved each frame: the four `rgba16float` 3D volumes
/// the scatter/integrate passes bind, the grid dimensions the dispatch derives
/// its workgroup counts from, and the two immediate blocks the passes upload.
///
/// Present only on views the prepare step ran this frame (fog enabled and a
/// non-degenerate view).
#[derive(Component)]
pub(crate) struct ViewVolumetrics {
    /// Scatter output / integrate input: rgb = source radiance, a = thickness.
    froxel_scattering: TextureView,
    /// Scatter output / integrate input: rgb = `sigma_t` extinction.
    froxel_extinction: TextureView,
    /// Integrate output: rgb = camera-to-slice accumulated in-scattering.
    integrated_scattering: TextureView,
    /// Integrate output: rgb = camera-to-slice transmittance.
    integrated_transmittance: TextureView,
    /// Froxel grid dimensions; the dispatch derives its workgroup counts here.
    pub(crate) grid: UVec3,
    /// Scatter-pass immediate block, rebuilt from the settings + this view's
    /// reconstructed frustum half-tangents each frame.
    pub(crate) scatter_params: GpuVolumetricsScatterParams,
    /// Integrate-pass immediate block (the grid dimensions).
    pub(crate) integrate_params: GpuVolumetricsIntegrateParams,
}

impl ViewVolumetrics {
    /// Scatter output / integrate input volume (source radiance + thickness).
    pub(crate) fn froxel_scattering(&self) -> &TextureView {
        &self.froxel_scattering
    }

    /// Scatter output / integrate input volume (extinction).
    pub(crate) fn froxel_extinction(&self) -> &TextureView {
        &self.froxel_extinction
    }

    /// Integrate output volume (accumulated in-scattering).
    pub(crate) fn integrated_scattering(&self) -> &TextureView {
        &self.integrated_scattering
    }

    /// Integrate output volume (accumulated transmittance).
    pub(crate) fn integrated_transmittance(&self) -> &TextureView {
        &self.integrated_transmittance
    }
}

/// The four cached froxel volumes plus the grid extent they were sized for, so a
/// grid resize can detect the mismatch and reallocate.
struct CachedVolumetrics {
    froxel_scattering: TextureView,
    froxel_extinction: TextureView,
    integrated_scattering: TextureView,
    integrated_transmittance: TextureView,
    grid: UVec3,
}

/// Render-world cache of each view's four persistent froxel volumes, keyed by
/// its stable [`RetainedViewEntity`]. A view that persists across frames with an
/// unchanged grid reuses the same allocations; a grid change rebuilds them and a
/// vanished (or fog-disabled) view is dropped so textures never leak.
#[derive(Resource, Default)]
pub(crate) struct VolumetricsTextureCache {
    views: HashMap<RetainedViewEntity, CachedVolumetrics>,
}

impl VolumetricsTextureCache {
    /// Drops every cached volume (used when the feature is disabled, so nothing
    /// lingers resident).
    fn clear(&mut self) {
        self.views.clear();
    }

    /// Returns the four cached volumes for `retained`, (re)allocating them when
    /// absent or sized for a different froxel grid.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        grid: UVec3,
    ) -> &CachedVolumetrics {
        let needs_new = self
            .views
            .get(&retained)
            .is_none_or(|cached| cached.grid != grid);
        if needs_new {
            self.views.insert(
                retained,
                CachedVolumetrics {
                    froxel_scattering: create_volume(device, grid, "prism volumetrics froxel scattering"),
                    froxel_extinction: create_volume(device, grid, "prism volumetrics froxel extinction"),
                    integrated_scattering: create_volume(
                        device,
                        grid,
                        "prism volumetrics integrated scattering",
                    ),
                    integrated_transmittance: create_volume(
                        device,
                        grid,
                        "prism volumetrics integrated transmittance",
                    ),
                    grid,
                },
            );
        }
        self.views
            .get(&retained)
            .expect("froxel volumes present after the is_none_or check")
    }

    /// Drops any cached volume whose view was not seen this frame.
    fn retain_seen(&mut self, seen: &HashSet<RetainedViewEntity>) {
        self.views.retain(|retained, _| seen.contains(retained));
    }
}

/// Allocates one `rgba16float` 3D froxel volume at `grid`, with both storage and
/// texture binding so it can be written as storage by one pass and read as a
/// sampled `texture_3d` by the next.
fn create_volume(device: &RenderDevice, grid: UVec3, label: &'static str) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: grid.x,
            height: grid.y,
            depth_or_array_layers: grid.z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: VOLUMETRICS_FROXEL_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    })
}

/// Reconstructs the camera frustum half-tangents `[tan(fov_x/2), tan(fov_y/2)]`
/// from the perspective `clip_from_view` matrix. For a standard perspective
/// projection the diagonal terms are the reciprocals of the half-tangents
/// (`clip_from_view.x_axis.x = 1 / tan(fov_x/2)`), so the fog reconstructs each
/// froxel's view ray on device from these. Degenerate (zero) terms fall back to
/// a 90-degree square frustum so the reconstruction never divides by zero.
fn frustum_half_tangents(view: &ExtractedView) -> [f32; 2] {
    let clip_from_view = view.clip_from_view;
    let sx = clip_from_view.x_axis.x.abs();
    let sy = clip_from_view.y_axis.y.abs();
    let tan_x = if sx > 1.0e-6 { 1.0 / sx } else { 1.0 };
    let tan_y = if sy > 1.0e-6 { 1.0 / sy } else { 1.0 };
    [tan_x, tan_y]
}

/// `PrepareResources` system: for every view, ensure the four correctly-sized
/// froxel volumes are resident, build both immediate blocks (folding in the
/// view's reconstructed frustum half-tangents) and attach them as a
/// [`ViewVolumetrics`] component.
///
/// Gated on [`PrismVolumetricsSettings::enabled`]; when disabled the cache is
/// cleared and no component is inserted, so the dispatch is a no-op that frame.
pub(crate) fn prepare_volumetrics_resources(
    mut commands: Commands,
    settings: Res<PrismVolumetricsSettings>,
    device: Res<RenderDevice>,
    mut cache: ResMut<VolumetricsTextureCache>,
    views: Query<(Entity, &ExtractedView)>,
) {
    if !settings.enabled {
        cache.clear();
        return;
    }

    let grid = settings.grid();
    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view) in &views {
        let retained = view.retained_view_entity;
        seen.insert(retained);

        let tan_half_fov = frustum_half_tangents(view);
        let scatter_params = settings.scatter_params(tan_half_fov);
        let integrate_params = settings.integrate_params();

        let cached = cache.get_or_create(&device, retained, grid);
        commands.entity(entity).insert(ViewVolumetrics {
            froxel_scattering: cached.froxel_scattering.clone(),
            froxel_extinction: cached.froxel_extinction.clone(),
            integrated_scattering: cached.integrated_scattering.clone(),
            integrated_transmittance: cached.integrated_transmittance.clone(),
            grid,
            scatter_params,
            integrate_params,
        });
    }
    cache.retain_seen(&seen);
}
