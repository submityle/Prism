//! `Core3d` scheduling-system pass recording both froxel-fog compute dispatches
//! for every view.
//!
//! Mirrors [`super::super::taa::dispatch`] / [`super::super::virtual_shadow`]'s
//! dispatch node, but records *two* chained compute passes in one command
//! encoder: the scatter pass fills every froxel's medium + source radiance +
//! extinction, then the integrate pass marches each column front-to-back into
//! the integrated in-scattering + transmittance volumes. The two passes are
//! recorded as separate `begin_compute_pass` blocks so the implicit
//! storage-write / texture-read barrier between them orders the integrate reads
//! after the scatter writes.
//!
//! Both entries bounds-check every invocation, so a partially-filled edge
//! workgroup is safe. Gated on [`PrismVolumetricsSettings::enabled`]; the
//! per-view components only exist on views the prepare/bind-group steps ran, so
//! a disabled frame records nothing.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::{VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE, VOLUMETRICS_SCATTER_WORKGROUP_SIZE};
use super::bind_groups::ViewVolumetricsBindGroups;
use super::pipeline::VolumetricsPipeline;
use super::resources::ViewVolumetrics;
use super::settings::PrismVolumetricsSettings;

/// `Core3d` scheduling-system pass recording the scatter then integrate froxel
/// dispatches for every view.
pub(crate) fn volumetrics_pass(
    settings: Res<PrismVolumetricsSettings>,
    view: ViewQuery<(&ViewVolumetrics, &ViewVolumetricsBindGroups)>,
    pipeline: Res<VolumetricsPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (volumetrics, groups) = view.into_inner();

    // Both pipelines must be resident before either pass records; the integrate
    // pass reads what the scatter pass writes, so it is pointless to run one
    // without the other.
    let (Some(scatter), Some(integrate)) = (
        cache.get_compute_pipeline(pipeline.scatter()),
        cache.get_compute_pipeline(pipeline.integrate()),
    ) else {
        return;
    };

    let grid = volumetrics.grid;
    if grid.x == 0 || grid.y == 0 || grid.z == 0 {
        return;
    }

    let encoder = ctx.command_encoder();

    // Scatter: one invocation per froxel over the full 3D grid.
    {
        let scatter_x = grid.x.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);
        let scatter_y = grid.y.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);
        let scatter_z = grid.z.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism volumetrics scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(scatter);
        pass.set_bind_group(0, groups.scatter(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&volumetrics.scatter_params));
        pass.dispatch_workgroups(scatter_x, scatter_y, scatter_z);
    }

    // Integrate: one invocation per froxel *column* (all Z slices), so the grid
    // is dispatched over X/Y only and each invocation marches Z internally.
    {
        let integrate_x = grid.x.div_ceil(VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE);
        let integrate_y = grid.y.div_ceil(VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism volumetrics integrate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(integrate);
        pass.set_bind_group(0, groups.integrate(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&volumetrics.integrate_params));
        pass.dispatch_workgroups(integrate_x, integrate_y, 1);
    }
}
