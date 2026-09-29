//! `PrepareBindGroups` assembly of the froxel-fog passes' group-0 bind groups.
//!
//! Mirrors [`super::super::taa::bind_groups`]: for every view carrying prepared
//! [`super::resources::ViewVolumetrics`] froxel volumes, it assembles the two
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

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::pipeline::VolumetricsPipeline;
use super::resources::ViewVolumetrics;

/// The froxel-fog passes' group-0 bind groups for a single view. Present only
/// when the prepared [`ViewVolumetrics`] froxel volumes are resident.
#[derive(Component)]
pub(crate) struct ViewVolumetricsBindGroups {
    /// group 0 for `volumetrics_scatter`: the two write-only froxel volumes.
    scatter: BindGroup,
    /// group 0 for `volumetrics_integrate`: the two scatter volumes read + the
    /// two integrated volumes written.
    integrate: BindGroup,
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
}

/// `PrepareBindGroups` system building [`ViewVolumetricsBindGroups`] for every
/// view with prepared [`ViewVolumetrics`] froxel volumes.
pub(crate) fn prepare_volumetrics_bind_groups(
    mut commands: Commands,
    pipeline: Res<VolumetricsPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVolumetrics)>,
) {
    for (entity, volumetrics) in &views {
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

        commands
            .entity(entity)
            .insert(ViewVolumetricsBindGroups { scatter, integrate });
    }
}
