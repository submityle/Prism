//! Per-view froxel storage volumes for the volumetric-fog passes and the
//! per-frame immediate blocks the passes upload.
//!
//! The fog runs three chained compute passes over a view-frustum-fitted 3D grid:
//! **scatter** writes each froxel's source radiance + slice thickness and its
//! extinction, **integrate** reads those back and marches each column
//! front-to-back into the integrated in-scattering + transmittance volumes, then
//! **apply** resolves the fog per screen pixel and composites it over the lit
//! scene colour into a full-res target the dispatch blits back over
//! `scene_color`. That needs four `rgba16float` 3D storage textures plus one
//! `rgba16float` full-res 2D composite target per view:
//!
//! * `froxel_scattering` — rgb = source radiance (`in_scatter + emissive`),
//!   a = slice thickness (scatter writes, integrate reads);
//! * `froxel_extinction` — rgb = `sigma_t` (scatter writes, integrate reads);
//! * `integrated_scattering` — rgb = camera-to-slice accumulated in-scattering
//!   (integrate writes, apply samples);
//! * `integrated_transmittance` — rgb = camera-to-slice transmittance
//!   (integrate writes, apply samples); and
//! * `fog_applied` — rgb = the lit scene colour with the fog composited over it
//!   (apply writes, the dispatch copies back over `scene_color`).
//!
//! The four 3D volumes carry `STORAGE_BINDING | TEXTURE_BINDING` (the scatter
//! volumes are written as storage by scatter and read as textures by integrate;
//! the integrated volumes are written by integrate and sampled by apply). The
//! 2D `fog_applied` target carries `STORAGE_BINDING | COPY_SRC` (apply writes it
//! as storage, then the dispatch copies it back over `scene_color`). Everything
//! is cached across frames keyed by [`RetainedViewEntity`], reallocated only
//! when the froxel grid dimensions or the framebuffer size change, so a
//! steady-state camera never churns GPU allocations — mirroring
//! [`super::super::virtual_shadow`]'s per-view receiver-buffer cache.

use bevy_ecs::prelude::*;
use bevy_math::{UVec2, UVec3};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Extent3d, Texture, TextureDescriptor, TextureDimension, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::RenderDevice,
    view::{ExtractedView, RetainedViewEntity},
};

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::{
    GpuVolumetricsApplyParams, GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams,
};
use super::pipeline::VOLUMETRICS_FROXEL_FORMAT;
use super::settings::PrismVolumetricsSettings;

/// Per-view froxel state resolved each frame: the four `rgba16float` 3D volumes
/// the scatter/integrate passes bind, the full-res `fog_applied` composite the
/// apply pass writes and the dispatch copies back over `scene_color`, the grid
/// dimensions the dispatch derives its workgroup counts from, the framebuffer
/// extent the apply pass dispatches over, and the three immediate blocks the
/// passes upload.
///
/// Present only on views the prepare step ran this frame (fog enabled, a
/// non-degenerate view and a sized camera viewport).
#[derive(Component)]
pub(crate) struct ViewVolumetrics {
    /// Scatter output / integrate input: rgb = source radiance, a = thickness.
    froxel_scattering: TextureView,
    /// Scatter output / integrate input: rgb = `sigma_t` extinction.
    froxel_extinction: TextureView,
    /// Integrate output / apply input: rgb = camera-to-slice in-scattering.
    integrated_scattering: TextureView,
    /// Integrate output / apply input: rgb = camera-to-slice transmittance.
    integrated_transmittance: TextureView,
    /// Apply output: the lit scene colour with the fog composited over it,
    /// copied back over `scene_color` by the dispatch.
    fog_applied_view: TextureView,
    /// The `fog_applied` GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    fog_applied: Texture,
    /// Froxel grid dimensions; the dispatch derives its scatter/integrate
    /// workgroup counts here.
    pub(crate) grid: UVec3,
    /// Full-resolution framebuffer extent; the apply dispatch derives its pixel
    /// workgroup counts here and the copy-back uses it as the blit extent.
    pub(crate) size: UVec2,
    /// Scatter-pass immediate block, rebuilt from the settings + this view's
    /// reconstructed frustum half-tangents each frame.
    pub(crate) scatter_params: GpuVolumetricsScatterParams,
    /// Integrate-pass immediate block (the grid dimensions).
    pub(crate) integrate_params: GpuVolumetricsIntegrateParams,
    /// Apply-pass immediate block (inverse projection + framebuffer extent +
    /// grid + the grid's view-space depth range and slice power).
    pub(crate) apply_params: GpuVolumetricsApplyParams,
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

    /// Integrate output / apply input volume (accumulated in-scattering).
    pub(crate) fn integrated_scattering(&self) -> &TextureView {
        &self.integrated_scattering
    }

    /// Integrate output / apply input volume (accumulated transmittance).
    pub(crate) fn integrated_transmittance(&self) -> &TextureView {
        &self.integrated_transmittance
    }

    /// Apply output view (the fog-composited scene colour), the storage target
    /// the apply pass writes.
    pub(crate) fn fog_applied_view(&self) -> &TextureView {
        &self.fog_applied_view
    }

    /// The `fog_applied` GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn fog_applied_texture(&self) -> &Texture {
        &self.fog_applied
    }
}

/// The four cached froxel volumes and the full-res composite target plus the
/// grid extent and framebuffer size they were sized for, so a grid resize or
/// window resize can detect the mismatch and reallocate.
struct CachedVolumetrics {
    froxel_scattering: TextureView,
    froxel_extinction: TextureView,
    integrated_scattering: TextureView,
    integrated_transmittance: TextureView,
    fog_applied_view: TextureView,
    fog_applied: Texture,
    grid: UVec3,
    size: UVec2,
}

/// Render-world cache of each view's persistent froxel volumes + composite
/// target, keyed by its stable [`RetainedViewEntity`]. A view that persists
/// across frames with an unchanged grid and framebuffer size reuses the same
/// allocations; a grid or size change rebuilds them and a vanished (or
/// fog-disabled) view is dropped so textures never leak.
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

    /// Returns the cached volumes + composite target for `retained`,
    /// (re)allocating them when absent or sized for a different froxel grid or
    /// framebuffer extent.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        retained: RetainedViewEntity,
        grid: UVec3,
        size: UVec2,
    ) -> &CachedVolumetrics {
        let needs_new = self
            .views
            .get(&retained)
            .is_none_or(|cached| cached.grid != grid || cached.size != size);
        if needs_new {
            let (fog_applied, fog_applied_view) = create_fog_target(device, size);
            self.views.insert(
                retained,
                CachedVolumetrics {
                    froxel_scattering: create_volume(
                        device,
                        grid,
                        "prism volumetrics froxel scattering",
                    ),
                    froxel_extinction: create_volume(
                        device,
                        grid,
                        "prism volumetrics froxel extinction",
                    ),
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
                    fog_applied_view,
                    fog_applied,
                    grid,
                    size,
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

/// Allocates the full-resolution `fog_applied` composite target: a
/// `SCENE_COLOR_FORMAT` 2D texture the apply pass writes as storage and the
/// dispatch copies back over `scene_color` (`STORAGE_BINDING | COPY_SRC`, framebuffer sized).
fn create_fog_target(device: &RenderDevice, size: UVec2) -> (Texture, TextureView) {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some("prism volumetrics fog applied"),
        size: Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SCENE_COLOR_FORMAT,
        // STORAGE_BINDING: written by `volumetrics_apply`. COPY_SRC: copied back
        // over `scene_color` after the pass so the downstream composite reads
        // the fog-composited image.
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("prism volumetrics fog applied"),
        ..Default::default()
    });
    (texture, view)
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

/// `PrepareResources` system: for every sized view, ensure the four
/// correctly-sized froxel volumes and the full-res composite target are
/// resident, build all three immediate blocks (folding in the view's
/// reconstructed frustum half-tangents and inverse projection) and attach them
/// as a [`ViewVolumetrics`] component.
///
/// Gated on [`PrismVolumetricsSettings::enabled`]; when disabled the cache is
/// cleared and no component is inserted, so the dispatch is a no-op that frame.
/// A view without a sized camera viewport is skipped (fog needs the framebuffer
/// extent for the composite target and the apply dispatch).
pub(crate) fn prepare_volumetrics_resources(
    mut commands: Commands,
    settings: Res<PrismVolumetricsSettings>,
    device: Res<RenderDevice>,
    mut cache: ResMut<VolumetricsTextureCache>,
    views: Query<(Entity, &ExtractedView, &ExtractedCamera)>,
) {
    if !settings.enabled {
        cache.clear();
        return;
    }

    let grid = settings.grid();
    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, camera) in &views {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if size.x == 0 || size.y == 0 {
            continue;
        }
        let retained = view.retained_view_entity;
        seen.insert(retained);

        let tan_half_fov = frustum_half_tangents(view);
        let scatter_params = settings.scatter_params(tan_half_fov);
        let integrate_params = settings.integrate_params();
        let apply_params = settings.apply_params(view.clip_from_view.inverse(), size);

        let cached = cache.get_or_create(&device, retained, grid, size);
        commands.entity(entity).insert(ViewVolumetrics {
            froxel_scattering: cached.froxel_scattering.clone(),
            froxel_extinction: cached.froxel_extinction.clone(),
            integrated_scattering: cached.integrated_scattering.clone(),
            integrated_transmittance: cached.integrated_transmittance.clone(),
            fog_applied_view: cached.fog_applied_view.clone(),
            fog_applied: cached.fog_applied.clone(),
            grid,
            size,
            scatter_params,
            integrate_params,
            apply_params,
        });
    }
    cache.retain_seen(&seen);
}
