//! `PrepareBindGroups` assembly of the froxel-fog passes' group-0 bind groups.
//!
//! Mirrors [`super::super::taa::bind_groups`]: for every view carrying prepared
//! [`super::resources::ViewVolumetrics`] froxel volumes, it assembles the
//! group-0 bind groups in the byte-identical order the pipeline layouts and
//! `volumetrics.wesl` declare:
//!
//! * **scatter** group — the two write-only froxel volumes:
//!   binding 0 = `froxel_scattering` (source + thickness),
//!   binding 1 = `froxel_extinction` (`sigma_t`).
//! * **integrate** group — the two scatter volumes read + the two integrated
//!   volumes written:
//!   binding 0 = `froxel_scattering` (read),
//!   binding 1 = `froxel_extinction` (read),
//!   binding 2 = `integrated_scattering` (write),
//!   binding 3 = `integrated_transmittance` (write).
//! * **apply** group — the scene depth + the two integrated volumes + the
//!   linear sampler + the scene colour + the composite output:
//!   binding 0 = `scene_depth` (SSR reverse-Z device depth, sampled),
//!   binding 1 = `integrated_scattering` (sampled),
//!   binding 2 = `integrated_transmittance` (sampled),
//!   binding 3 = the linear-clamp sampler,
//!   binding 4 = `scene_color` (lit HDR, sampled),
//!   binding 5 = `fog_applied` (write-only composite output).
//!   Built only when the view has both a resident [`ViewSsrTextures`] (the depth
//!   source) and [`ViewVisibilityBuffer`] (the scene-colour source), so a view
//!   without the geometry prepass simply skips the composite that frame.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::VolumetricsPipeline;
use super::resources::ViewVolumetrics;

/// The froxel-fog passes' group-0 bind groups for a single view. The scatter +
/// integrate groups are always present when the prepared [`ViewVolumetrics`]
/// froxel volumes are resident; the apply group is present only when the view
/// also carries the SSR depth + visibility scene-colour buffers the composite
/// reads.
#[derive(Component)]
pub(crate) struct ViewVolumetricsBindGroups {
    /// group 0 for `volumetrics_scatter`: the two write-only froxel volumes.
    scatter: BindGroup,
    /// group 0 for `volumetrics_integrate`: the two scatter volumes read + the
    /// two integrated volumes written.
    integrate: BindGroup,
    /// group 0 for `volumetrics_apply`: scene depth, the two integrated volumes,
    /// the linear sampler, scene colour, and the composite output. `None` when
    /// the SSR depth or visibility scene-colour buffers are not yet resident.
    apply: Option<BindGroup>,
}

impl ViewVolumetricsBindGroups {
    /// The scatter pass's group-0 bind group.
    pub(crate) fn scatter(&self) -> &BindGroup {
        &self.scatter
    }

    /// The integrate pass's group-0 bind group.
    pub(crate) fn integrate(&self) -> &BindGroup {
        &self.integrate
    }

    /// The apply pass's group-0 bind group, present only when the composite's
    /// depth + scene-colour inputs are resident.
    pub(crate) fn apply(&self) -> Option<&BindGroup> {
        self.apply.as_ref()
    }
}

/// `PrepareBindGroups` system building [`ViewVolumetricsBindGroups`] for every
/// view with prepared [`ViewVolumetrics`] froxel volumes. The apply group is
/// only assembled when the view also carries a resident [`ViewSsrTextures`]
/// (depth) and [`ViewVisibilityBuffer`] (scene colour).
pub(crate) fn prepare_volumetrics_bind_groups(
    mut commands: Commands,
    pipeline: Res<VolumetricsPipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVolumetrics,
        Option<&ViewSsrTextures>,
        Option<&ViewVisibilityBuffer>,
    )>,
) {
    for (entity, volumetrics, ssr, visibility) in &views {
        // Order mirrors the scatter layout / `volumetrics_scatter`:
        // froxel_scattering(0), froxel_extinction(1).
        let scatter = device.create_bind_group(
            "prism volumetrics scatter",
            pipeline.scatter_layout(),
            &BindGroupEntries::sequential((
                volumetrics.froxel_scattering(),
                volumetrics.froxel_extinction(),
            )),
        );

        // Order mirrors the integrate layout / `volumetrics_integrate`:
        // froxel_scattering(0), froxel_extinction(1), integrated_scattering(2),
        // integrated_transmittance(3).
        let integrate = device.create_bind_group(
            "prism volumetrics integrate",
            pipeline.integrate_layout(),
            &BindGroupEntries::sequential((
                volumetrics.froxel_scattering(),
                volumetrics.froxel_extinction(),
                volumetrics.integrated_scattering(),
                volumetrics.integrated_transmittance(),
            )),
        );

        // Order mirrors the apply layout / `volumetrics_apply`:
        // scene_depth(0), integrated_scattering(1), integrated_transmittance(2),
        // linear_sampler(3), scene_color(4), fog_applied(5). Only assembled when
        // both the SSR depth and the visibility scene colour are resident.
        let apply = match (ssr, visibility) {
            (Some(ssr), Some(visibility)) => Some(device.create_bind_group(
                "prism volumetrics apply",
                pipeline.apply_layout(),
                &BindGroupEntries::sequential((
                    ssr.scene_depth_sampled(),
                    volumetrics.integrated_scattering(),
                    volumetrics.integrated_transmittance(),
                    pipeline.linear_sampler(),
                    visibility.scene_color_view(),
                    volumetrics.fog_applied_view(),
                )),
            )),
            _ => None,
        };

        commands.entity(entity).insert(ViewVolumetricsBindGroups {
            scatter,
            integrate,
            apply,
        });
    }
}
