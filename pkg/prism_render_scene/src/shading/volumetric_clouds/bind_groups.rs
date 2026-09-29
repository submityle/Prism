//! `PrepareBindGroups` assembly of the eight volumetric-cloud passes' group-0
//! bind groups.
//!
//! `shaders/volumetric_clouds.wesl` assigns every resource a *unique*
//! `@group(0)` binding across the whole file (bindings `0`–`17`) and each kernel
//! binds only its own subset, so every bind group is built with
//! [`BindGroupEntries::with_indices`] to pin the exact global binding numbers
//! the matching [`super::pipeline`] layout declares (the same non-zero-based
//! idiom [`super::super::dof::bind_groups`] uses), not the sequential `0..N` a
//! per-pass group would imply.
//!
//! The bind groups split by what they reference, mirroring the resource split
//! in [`super::resources`]:
//!
//! * [`VolumetricCloudDomainBindGroups`] — the five *view-independent* passes
//!   whose every texture lives on the [`VolumetricCloudDomain`] world resource:
//!   `weather_advect` (`0` weather src → `1` weather dst), `noise_bake` (`2`
//!   noise write), `modeling` (`3` noise read + `4` weather read → `5` density
//!   write), `multiscatter_lut_bake` (`6` `LUT` write) and `shadow_march` (`13`
//!   density read → `14` shadow write). Assembled once per frame as a render
//!   resource so the domain passes are not needlessly rebuilt per view.
//! * [`ViewVolumetricCloudBindGroups`] — the three *per-view* passes that mix
//!   the per-view screen targets with the shared domain textures: `raymarch`
//!   (`7` density + `8` shadow read → `9` low-res target write), `scatter_resolve`
//!   (`10` low-res read + `11` `LUT` read → `12` resolved write) and `upsample`
//!   (`15` resolved read + `16` history read → `17` full-res output write).
//!
//! The weather double-buffer read/write split (`0`/`1`) and the history
//! double-buffer split (`16`/`17`) follow the frame parity the prepared
//! resources already resolved, so the advection and temporal upsample read the
//! previous frame while writing the next without a hazard.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};
use prism_render_architecture::volumetric::gpu::kernels::VolumetricKernel;

use super::pipeline::VolumetricCloudPipelines;
use super::resources::{ViewVolumetricClouds, VolumetricCloudDomain};

/// The five view-independent cloud-domain passes' group-0 bind groups, present
/// as a render resource whenever the [`VolumetricCloudDomain`] is resident
/// (clouds enabled).
#[derive(Resource)]
pub(crate) struct VolumetricCloudDomainBindGroups {
    /// group 0 for `volumetric_weather_advect`: weather src (`0`) → dst (`1`).
    weather_advect: BindGroup,
    /// group 0 for `volumetric_noise_bake`: noise volume write (`2`).
    noise_bake: BindGroup,
    /// group 0 for `volumetric_modeling`: noise read (`3`) + weather read (`4`)
    /// → density write (`5`).
    modeling: BindGroup,
    /// group 0 for `volumetric_multiscatter_lut_bake`: `LUT` write (`6`).
    ms_lut: BindGroup,
    /// group 0 for `volumetric_shadow_march`: density read (`13`) → shadow
    /// write (`14`).
    shadow_march: BindGroup,
}

impl VolumetricCloudDomainBindGroups {
    /// group-0 bind group for the requested domain kernel, or `None` when the
    /// kernel is view-dependent (ray-march / scatter-resolve / upsample).
    pub(crate) fn group(&self, kernel: VolumetricKernel) -> Option<&BindGroup> {
        match kernel {
            VolumetricKernel::WeatherAdvect => Some(&self.weather_advect),
            VolumetricKernel::NoiseBake => Some(&self.noise_bake),
            VolumetricKernel::Modeling => Some(&self.modeling),
            VolumetricKernel::MultiscatterLutBake => Some(&self.ms_lut),
            VolumetricKernel::ShadowMarch => Some(&self.shadow_march),
            VolumetricKernel::Raymarch
            | VolumetricKernel::ScatterResolve
            | VolumetricKernel::Upsample => None,
        }
    }
}

/// The three per-view cloud passes' group-0 bind groups, attached to every
/// enabled, sized view alongside its [`ViewVolumetricClouds`] targets.
#[derive(Component)]
pub(crate) struct ViewVolumetricCloudBindGroups {
    /// group 0 for `volumetric_raymarch`: density (`7`) + shadow (`8`) read →
    /// low-res target write (`9`).
    raymarch: BindGroup,
    /// group 0 for `volumetric_scatter_resolve`: low-res read (`10`) + `LUT`
    /// read (`11`) → resolved write (`12`).
    scatter_resolve: BindGroup,
    /// group 0 for `volumetric_upsample`: resolved read (`15`) + history read
    /// (`16`) → full-res output write (`17`).
    upsample: BindGroup,
}

impl ViewVolumetricCloudBindGroups {
    /// group-0 bind group for the requested per-view kernel, or `None` when the
    /// kernel is view-independent (a domain pass).
    pub(crate) fn group(&self, kernel: VolumetricKernel) -> Option<&BindGroup> {
        match kernel {
            VolumetricKernel::Raymarch => Some(&self.raymarch),
            VolumetricKernel::ScatterResolve => Some(&self.scatter_resolve),
            VolumetricKernel::Upsample => Some(&self.upsample),
            VolumetricKernel::WeatherAdvect
            | VolumetricKernel::NoiseBake
            | VolumetricKernel::Modeling
            | VolumetricKernel::MultiscatterLutBake
            | VolumetricKernel::ShadowMarch => None,
        }
    }
}

/// `PrepareBindGroups` system building [`VolumetricCloudDomainBindGroups`] from
/// the resident [`VolumetricCloudDomain`]. When the domain is absent (clouds
/// disabled) the resource is removed so the domain dispatches are a no-op.
pub(crate) fn prepare_volumetric_cloud_domain_bind_groups(
    mut commands: Commands,
    pipelines: Res<VolumetricCloudPipelines>,
    device: Res<RenderDevice>,
    domain: Option<Res<VolumetricCloudDomain>>,
) {
    let Some(domain) = domain else {
        commands.remove_resource::<VolumetricCloudDomainBindGroups>();
        return;
    };

    // weather_advect: weather src (0) → weather dst (1).
    let weather_advect = device.create_bind_group(
        "prism vc weather-advect",
        pipelines.layout(VolumetricKernel::WeatherAdvect),
        &BindGroupEntries::with_indices(((0, domain.weather_src()), (1, domain.weather_dst()))),
    );

    // noise_bake: noise volume write (2).
    let noise_bake = device.create_bind_group(
        "prism vc noise-bake",
        pipelines.layout(VolumetricKernel::NoiseBake),
        &BindGroupEntries::with_indices(((2, domain.noise_vol()),)),
    );

    // modeling: noise read (3) + weather read (4) → density write (5). The
    // weather map read is the freshly advected copy (`weather_dst`).
    let modeling = device.create_bind_group(
        "prism vc modeling",
        pipelines.layout(VolumetricKernel::Modeling),
        &BindGroupEntries::with_indices((
            (3, domain.noise_vol()),
            (4, domain.weather_dst()),
            (5, domain.density_vol()),
        )),
    );

    // multiscatter_lut_bake: LUT write (6).
    let ms_lut = device.create_bind_group(
        "prism vc multiscatter-lut",
        pipelines.layout(VolumetricKernel::MultiscatterLutBake),
        &BindGroupEntries::with_indices(((6, domain.mslut()),)),
    );

    // shadow_march: density read (13) → shadow write (14).
    let shadow_march = device.create_bind_group(
        "prism vc shadow-march",
        pipelines.layout(VolumetricKernel::ShadowMarch),
        &BindGroupEntries::with_indices(((13, domain.density_vol()), (14, domain.shadow_map()))),
    );

    commands.insert_resource(VolumetricCloudDomainBindGroups {
        weather_advect,
        noise_bake,
        modeling,
        ms_lut,
        shadow_march,
    });
}

/// `PrepareBindGroups` system building [`ViewVolumetricCloudBindGroups`] for
/// every view carrying prepared [`ViewVolumetricClouds`] targets. Gated on the
/// resident [`VolumetricCloudDomain`] because the ray-march and scatter-resolve
/// passes read the shared density / shadow / `LUT` domain textures; a frame
/// without the domain simply attaches no per-view bind groups.
pub(crate) fn prepare_volumetric_cloud_view_bind_groups(
    mut commands: Commands,
    pipelines: Res<VolumetricCloudPipelines>,
    device: Res<RenderDevice>,
    domain: Option<Res<VolumetricCloudDomain>>,
    views: Query<(Entity, &ViewVolumetricClouds)>,
) {
    let Some(domain) = domain else {
        return;
    };

    for (entity, view) in &views {
        // raymarch: density (7) + shadow (8) read → low-res target write (9).
        let raymarch = device.create_bind_group(
            "prism vc raymarch",
            pipelines.layout(VolumetricKernel::Raymarch),
            &BindGroupEntries::with_indices((
                (7, domain.density_vol()),
                (8, domain.shadow_map()),
                (9, view.raymarch_target()),
            )),
        );

        // scatter_resolve: low-res read (10) + LUT read (11) → resolved write
        // (12).
        let scatter_resolve = device.create_bind_group(
            "prism vc scatter-resolve",
            pipelines.layout(VolumetricKernel::ScatterResolve),
            &BindGroupEntries::with_indices((
                (10, view.raymarch_target()),
                (11, domain.mslut()),
                (12, view.scatter_output()),
            )),
        );

        // upsample: resolved read (15) + history read (16) → full-res output
        // write (17).
        let upsample = device.create_bind_group(
            "prism vc upsample",
            pipelines.layout(VolumetricKernel::Upsample),
            &BindGroupEntries::with_indices((
                (15, view.scatter_output()),
                (16, view.history_src()),
                (17, view.output_dst()),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewVolumetricCloudBindGroups {
                raymarch,
                scatter_resolve,
                upsample,
            });
    }
}
