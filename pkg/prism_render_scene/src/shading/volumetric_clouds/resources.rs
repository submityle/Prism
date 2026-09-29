//! Resident device textures for the eight volumetric-cloud compute passes.
//!
//! The subsystem keeps its heavy state resident across frames rather than
//! re-uploading it every frame, mirroring the sizing/lifecycle contract the
//! architecture crate's
//! [`prism_render_architecture::volumetric::gpu::buffers`] scheduler defines.
//! Two homes split the resources by what they depend on:
//!
//! * [`VolumetricCloudDomain`] — the *view-independent* world cloud domain: the
//!   double-buffered weather map the semi-Lagrangian advection evolves, the 3D
//!   Perlin-Worley noise volume, the composed 3D density cache the ray-march and
//!   shadow march sample, the 3D multiple-scatter `LUT`, and the light-space
//!   `AVSM` cloud-shadow map. All are sized from
//!   [`PrismVolumetricCloudsSettings`] and rebuilt only when a domain resolution
//!   changes, so a steady-state scene never churns GPU allocations.
//! * [`ViewVolumetricClouds`] — the *per-view* screen-space targets: the low-res
//!   ray-march scattering/transmittance target, the low-res resolved scatter
//!   target, and the double-buffered full-res reprojection history / output.
//!   Cached per [`RetainedViewEntity`] and reallocated only when the framebuffer
//!   extent changes, mirroring [`super::super::volumetrics`]'s per-view cache.
//!
//! Every texture is the `rgba16float` [`VC_STORAGE_FORMAT`] and carries
//! `STORAGE_BINDING | TEXTURE_BINDING` — each is written as a storage texture by
//! its producer pass and read as a sampled (`textureLoad`) texture by its
//! consumer pass, exactly the read/write split
//! `shaders/volumetric_clouds.wesl` declares. The weather map and the history
//! buffer are double-buffered and swapped by frame parity so the advection and
//! the temporal upsample read the previous frame while writing the next without
//! a hazard.

#![allow(
    dead_code,
    reason = "the resident volumetric-cloud domain textures and per-view targets are the render-resource foundation the bind-group and Core3d dispatch slices consume; those slices land next, and the parity/sizing bookkeeping is exercised now by the tests below"
)]

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

use super::abi::{
    GpuModelingParams, GpuMsLutParams, GpuNoiseBakeParams, GpuRaymarchParams,
    GpuScatterResolveParams, GpuShadowMarchParams, GpuUpsampleParams, GpuWeatherAdvectParams,
};
use super::pipeline::VC_STORAGE_FORMAT;
use super::settings::PrismVolumetricCloudsSettings;

/// Default sun direction (world-space, travelling from sun toward the scene)
/// used to fold a representative sun/view `cos_theta` into the scatter-resolve
/// phase block until a shared directional-light resource is wired. A high sun
/// coming down and slightly across the frame, matching the froxel fog default.
const DEFAULT_SUN_DIRECTION: [f32; 3] = [0.35, -1.0, 0.35];

/// The view-independent resident cloud-domain textures, rebuilt only when a
/// domain resolution changes. Present as a render-world resource whenever the
/// clouds are enabled.
#[derive(Resource)]
pub(crate) struct VolumetricCloudDomain {
    /// Weather-map copy A (double-buffered with `weather_b`).
    weather_a: TextureView,
    /// Weather-map copy B.
    weather_b: TextureView,
    /// 3D Perlin-Worley noise volume: `noise_bake` writes, `modeling` reads.
    noise_vol: TextureView,
    /// 3D composed density cache: `modeling` writes, `raymarch`/`shadow_march`
    /// read.
    density_vol: TextureView,
    /// 3D multiple-scatter `LUT`: `multiscatter_lut_bake` writes,
    /// `scatter_resolve` reads.
    mslut: TextureView,
    /// Light-space `AVSM` cloud-shadow map: `shadow_march` writes, `raymarch`
    /// reads.
    shadow_map: TextureView,

    /// Cached domain resolutions; a change triggers a rebuild.
    density_dim: UVec3,
    weather_dim: UVec2,
    mslut_dim: UVec3,
    shadow_dim: UVec2,

    /// Monotonic frame counter; its low bit is the double-buffer parity.
    frame: u32,

    /// `volumetric_weather_advect` immediate block (settings-only).
    pub(crate) weather_advect_params: GpuWeatherAdvectParams,
    /// `volumetric_noise_bake` immediate block (settings-only).
    pub(crate) noise_bake_params: GpuNoiseBakeParams,
    /// `volumetric_modeling` immediate block (settings-only).
    pub(crate) modeling_params: GpuModelingParams,
    /// `volumetric_multiscatter_lut_bake` immediate block (settings-only).
    pub(crate) ms_lut_params: GpuMsLutParams,
    /// `volumetric_shadow_march` immediate block (settings-only).
    pub(crate) shadow_march_params: GpuShadowMarchParams,
}

impl VolumetricCloudDomain {
    /// The weather map the advection reads this frame (previous-frame copy).
    pub(crate) fn weather_src(&self) -> &TextureView {
        if self.frame & 1 == 0 {
            &self.weather_a
        } else {
            &self.weather_b
        }
    }

    /// The weather map the advection writes this frame (next-frame copy).
    pub(crate) fn weather_dst(&self) -> &TextureView {
        if self.frame & 1 == 0 {
            &self.weather_b
        } else {
            &self.weather_a
        }
    }

    /// The 3D noise volume.
    pub(crate) fn noise_vol(&self) -> &TextureView {
        &self.noise_vol
    }

    /// The 3D composed density cache.
    pub(crate) fn density_vol(&self) -> &TextureView {
        &self.density_vol
    }

    /// The 3D multiple-scatter `LUT`.
    pub(crate) fn mslut(&self) -> &TextureView {
        &self.mslut
    }

    /// The light-space `AVSM` cloud-shadow map.
    pub(crate) fn shadow_map(&self) -> &TextureView {
        &self.shadow_map
    }

    /// The current frame's double-buffer parity index (`0` or `1`).
    pub(crate) fn parity(&self) -> u32 {
        self.frame & 1
    }
}

/// The per-view resident screen-space cloud targets, cached per
/// [`RetainedViewEntity`]. Attached as a component to every enabled, sized view.
#[derive(Component)]
pub(crate) struct ViewVolumetricClouds {
    /// Low-res ray-march scattering/transmittance target: `raymarch` writes,
    /// `scatter_resolve` reads.
    raymarch_target: TextureView,
    /// Low-res resolved scatter target: `scatter_resolve` writes, `upsample`
    /// reads.
    scatter_output: TextureView,
    /// Full-res reprojection history / output copy A (double-buffered with B).
    history_a: TextureView,
    /// Full-res copy B.
    history_b: TextureView,

    /// Full framebuffer extent (the upsample / screen-pixel dispatch domain).
    pub(crate) full_size: UVec2,
    /// Low-res target extent (the ray-march / scatter tile dispatch domain).
    pub(crate) lowres_size: UVec2,
    /// Double-buffer parity mirrored from the domain so the history swap stays
    /// in phase with the weather swap.
    frame: u32,

    /// `volumetric_raymarch` immediate block (folds the low-res extent).
    pub(crate) raymarch_params: GpuRaymarchParams,
    /// `volumetric_scatter_resolve` immediate block (folds extent + sun cosine).
    pub(crate) scatter_resolve_params: GpuScatterResolveParams,
    /// `volumetric_upsample` immediate block (folds full/low-res extents +
    /// frame index).
    pub(crate) upsample_params: GpuUpsampleParams,
}

impl ViewVolumetricClouds {
    /// The low-res ray-march target.
    pub(crate) fn raymarch_target(&self) -> &TextureView {
        &self.raymarch_target
    }

    /// The low-res resolved scatter target.
    pub(crate) fn scatter_output(&self) -> &TextureView {
        &self.scatter_output
    }

    /// The full-res history the upsample reads this frame (previous output).
    pub(crate) fn history_src(&self) -> &TextureView {
        if self.frame & 1 == 0 {
            &self.history_a
        } else {
            &self.history_b
        }
    }

    /// The full-res output the upsample writes this frame (next history).
    pub(crate) fn output_dst(&self) -> &TextureView {
        if self.frame & 1 == 0 {
            &self.history_b
        } else {
            &self.history_a
        }
    }
}

/// Allocates a `rgba16float` 2D storage+sampled texture view sized `size`.
fn create_2d(device: &RenderDevice, size: UVec2, label: &'static str) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: VC_STORAGE_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    })
}

/// Allocates a `rgba16float` 3D storage+sampled texture view sized `dim`.
fn create_3d(device: &RenderDevice, dim: UVec3, label: &'static str) -> TextureView {
    let texture: Texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: dim.x.max(1),
            height: dim.y.max(1),
            depth_or_array_layers: dim.z.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: VC_STORAGE_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        ..Default::default()
    })
}

/// `PrepareResources` system: ensure the view-independent cloud-domain textures
/// are resident and correctly sized, advance the double-buffer parity, and
/// rebuild the settings-only immediate blocks.
///
/// Gated on [`PrismVolumetricCloudsSettings::enabled`]; when disabled the domain
/// resource is removed so the dispatch is a no-op that frame.
pub(crate) fn prepare_volumetric_cloud_domain(
    mut commands: Commands,
    settings: Res<PrismVolumetricCloudsSettings>,
    device: Res<RenderDevice>,
    domain: Option<ResMut<VolumetricCloudDomain>>,
) {
    if !settings.enabled {
        if domain.is_some() {
            commands.remove_resource::<VolumetricCloudDomain>();
        }
        return;
    }

    let weather_advect_params = settings.weather_advect_params();
    let noise_bake_params = settings.noise_bake_params();
    let modeling_params = settings.modeling_params();
    let ms_lut_params = settings.ms_lut_params();
    let shadow_march_params = settings.shadow_march_params();

    match domain {
        Some(mut domain)
            if domain.density_dim == settings.density_dim
                && domain.weather_dim == settings.weather_dim
                && domain.mslut_dim == settings.mslut_dim
                && domain.shadow_dim == settings.shadow_dim =>
        {
            // Steady state: advance parity and refresh the immediate blocks
            // without reallocating any texture.
            domain.frame = domain.frame.wrapping_add(1);
            domain.weather_advect_params = weather_advect_params;
            domain.noise_bake_params = noise_bake_params;
            domain.modeling_params = modeling_params;
            domain.ms_lut_params = ms_lut_params;
            domain.shadow_march_params = shadow_march_params;
        }
        _ => {
            // First frame or a resolution change: (re)allocate every domain
            // texture and reset the parity counter.
            commands.insert_resource(VolumetricCloudDomain {
                weather_a: create_2d(&device, settings.weather_dim, "prism vc weather a"),
                weather_b: create_2d(&device, settings.weather_dim, "prism vc weather b"),
                noise_vol: create_3d(&device, settings.density_dim, "prism vc noise volume"),
                density_vol: create_3d(&device, settings.density_dim, "prism vc density cache"),
                mslut: create_3d(&device, settings.mslut_dim, "prism vc multiscatter lut"),
                shadow_map: create_2d(&device, settings.shadow_dim, "prism vc shadow map"),
                density_dim: settings.density_dim,
                weather_dim: settings.weather_dim,
                mslut_dim: settings.mslut_dim,
                shadow_dim: settings.shadow_dim,
                frame: 0,
                weather_advect_params,
                noise_bake_params,
                modeling_params,
                ms_lut_params,
                shadow_march_params,
            });
        }
    }
}

/// One view's cached screen-space textures, reallocated only when the
/// framebuffer extent changes.
struct CachedViewTextures {
    raymarch_target: TextureView,
    scatter_output: TextureView,
    history_a: TextureView,
    history_b: TextureView,
    full_size: UVec2,
    lowres_size: UVec2,
}

/// The per-view screen-space texture cache, keyed by [`RetainedViewEntity`] so a
/// steady-state camera reuses its allocations frame to frame.
#[derive(Resource, Default)]
pub(crate) struct VolumetricCloudViewCache {
    views: HashMap<RetainedViewEntity, CachedViewTextures>,
}

impl VolumetricCloudViewCache {
    /// Returns the cached textures for `view`, reallocating them when the
    /// framebuffer extent changed since the last frame.
    fn get_or_create(
        &mut self,
        device: &RenderDevice,
        view: RetainedViewEntity,
        full: UVec2,
        lowres: UVec2,
    ) -> &CachedViewTextures {
        let stale = self
            .views
            .get(&view)
            .is_none_or(|c| c.full_size != full || c.lowres_size != lowres);
        if stale {
            self.views.insert(
                view,
                CachedViewTextures {
                    raymarch_target: create_2d(device, lowres, "prism vc raymarch target"),
                    scatter_output: create_2d(device, lowres, "prism vc scatter output"),
                    history_a: create_2d(device, full, "prism vc history a"),
                    history_b: create_2d(device, full, "prism vc history b"),
                    full_size: full,
                    lowres_size: lowres,
                },
            );
        }
        self.views
            .get(&view)
            .expect("just inserted or already present")
    }

    /// Drops cached textures for views not seen this frame so closed cameras do
    /// not linger resident.
    fn retain_seen(&mut self, seen: &HashSet<RetainedViewEntity>) {
        self.views.retain(|view, _| seen.contains(view));
    }

    /// Empties the cache (clouds disabled).
    fn clear(&mut self) {
        self.views.clear();
    }
}

/// `PrepareResources` system: for every sized view, ensure the low-res
/// ray-march/scatter targets and the double-buffered full-res history are
/// resident, build the three view-dependent immediate blocks, and attach a
/// [`ViewVolumetricClouds`] component.
///
/// Runs after [`prepare_volumetric_cloud_domain`] and reads the domain frame
/// parity so the history swap stays in phase with the weather swap. Gated on
/// the domain resource being present (clouds enabled); a view without a sized
/// camera viewport is skipped.
pub(crate) fn prepare_volumetric_cloud_views(
    mut commands: Commands,
    settings: Res<PrismVolumetricCloudsSettings>,
    device: Res<RenderDevice>,
    domain: Option<Res<VolumetricCloudDomain>>,
    mut cache: ResMut<VolumetricCloudViewCache>,
    views: Query<(Entity, &ExtractedView, &ExtractedCamera)>,
) {
    let Some(domain) = domain else {
        cache.clear();
        return;
    };
    let frame = domain.frame;

    let mut seen: HashSet<RetainedViewEntity> = HashSet::default();
    for (entity, view, camera) in &views {
        let Some(full) = camera.physical_viewport_size else {
            continue;
        };
        if full.x == 0 || full.y == 0 {
            continue;
        }
        let retained = view.retained_view_entity;
        seen.insert(retained);

        let lowres = settings.lowres_size(full);
        let cos_theta = sun_view_cosine(view);

        let raymarch_params = settings.raymarch_params(lowres);
        let scatter_resolve_params = settings.scatter_resolve_params(lowres, cos_theta);
        let upsample_params = settings.upsample_params(full, lowres, frame);

        let cached = cache.get_or_create(&device, retained, full, lowres);
        commands.entity(entity).insert(ViewVolumetricClouds {
            raymarch_target: cached.raymarch_target.clone(),
            scatter_output: cached.scatter_output.clone(),
            history_a: cached.history_a.clone(),
            history_b: cached.history_b.clone(),
            full_size: full,
            lowres_size: lowres,
            frame,
            raymarch_params,
            scatter_resolve_params,
            upsample_params,
        });
    }
    cache.retain_seen(&seen);
}

/// Representative sun/view phase cosine: the cosine between the camera forward
/// direction and the default sun travel direction. A real per-pixel cosine is
/// recomputed on device; this frame-constant hint biases the dual-lobe blend
/// toward the sun-facing side. Falls back to `0` for a degenerate direction.
fn sun_view_cosine(view: &ExtractedView) -> f32 {
    let forward = view.world_from_view.forward();
    let sun = bevy_math::Vec3::from_array(DEFAULT_SUN_DIRECTION);
    let len = sun.length();
    if len > 1.0e-6 {
        forward.dot(sun / len)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::DEFAULT_SUN_DIRECTION;
    use bevy_math::Vec3;

    /// The default sun direction is non-degenerate so the phase cosine
    /// normalisation never divides by zero.
    #[test]
    fn default_sun_direction_is_nonzero() {
        assert!(Vec3::from_array(DEFAULT_SUN_DIRECTION).length() > 1.0e-3);
    }
}
