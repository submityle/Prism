//! `Core3d` scheduling system recording the world-space `ReSTIR` composite for
//! every view.
//!
//! Runs after `world_restir_resolve_pass` (its `gi_out` input) and, being a
//! `scene_color` writer, after the other GI composites in the
//! `scene_color`-writer chain, and before the main pass that presents
//! `scene_color`. Records the copy then the fold in a single encoder; wgpu
//! inserts the storage barrier between them. Dispatches one workgroup per
//! [`COMPOSITE_WORKGROUP_SIZE`]-pixel tile on each axis; both shader entry
//! points bounds-check every invocation.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::super::settings::PrismWorldRestirSettings;
use super::abi::{GpuWorldRestirCompositeParams, COMPOSITE_WORKGROUP_SIZE};
use super::bind_groups::ViewWorldRestirCompositeBindGroups;
use super::pipeline::WorldRestirCompositePipeline;
use super::resources::ViewWorldRestirComposite;

/// `Core3d` scheduling system recording the world-space `ReSTIR` composite for
/// every view whose composite scratch and bind groups are live.
pub(crate) fn world_restir_composite_pass(
    settings: Res<PrismWorldRestirSettings>,
    view: ViewQuery<(
        &ViewWorldRestirComposite,
        &ViewWorldRestirCompositeBindGroups,
    )>,
    pipeline: Res<WorldRestirCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (composite, groups) = view.into_inner();

    let (Some(copy), Some(fold)) = (
        cache.get_compute_pipeline(pipeline.copy()),
        cache.get_compute_pipeline(pipeline.fold()),
    ) else {
        return;
    };

    let size = composite.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuWorldRestirCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(COMPOSITE_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(COMPOSITE_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space ReSTIR composite copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(copy);
        pass.set_bind_group(0, groups.copy(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space ReSTIR composite fold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fold);
        pass.set_bind_group(0, groups.fold(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
