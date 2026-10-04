//! `Core3d` scheduling system recording the world-space `ReSTIR` resolve
//! dispatch for every view.
//!
//! Gate on the enable, resolve the view, recover the inverse projection + the
//! view->world matrix + the camera world position from the [`ExtractedView`],
//! build the [`GpuWorldRestirResolveParams`] immediate block from the resident
//! settings, then dispatch one 2-D workgroup block per
//! [`RESOLVE_WORKGROUP_SIZE`] pixels on each axis. `resolve_main` bounds-checks
//! every invocation against the live `(screen_w, screen_h)`, so the rounded
//! extent never reads or writes a pixel past the framebuffer.
//!
//! The resolve consumes the finalised reservoir table, so it runs after the
//! fill pass and before the main pass each frame — the last world-space
//! `ReSTIR` pass before the direct-illumination export is folded into
//! `scene_color`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::super::settings::PrismWorldRestirSettings;
use super::abi::{GpuWorldRestirResolveParams, RESOLVE_WORKGROUP_SIZE};
use super::bind_groups::ViewWorldRestirResolveBindGroup;
use super::pipeline::WorldRestirResolvePipeline;
use super::resources::ViewWorldRestirResolve;

/// Number of 1-D workgroups the resolve dispatch records along one axis of
/// `extent` pixels: one workgroup per [`RESOLVE_WORKGROUP_SIZE`] pixels,
/// rounded up, with a floor of one so a degenerate `extent == 0` axis still
/// issues a single (fully bounds-checked, no-op) workgroup.
fn resolve_workgroups(extent: u32) -> u32 {
    extent.max(1).div_ceil(RESOLVE_WORKGROUP_SIZE)
}

/// `Core3d` scheduling system recording the `resolve_main` dispatch for every
/// view whose resolve export and bind group are live.
pub(crate) fn world_restir_resolve_pass(
    settings: Res<PrismWorldRestirSettings>,
    view: ViewQuery<(
        &ViewWorldRestirResolve,
        &ViewWorldRestirResolveBindGroup,
        &ExtractedView,
    )>,
    pipeline: Res<WorldRestirResolvePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (resolve, bind_group, extracted) = view.into_inner();

    // The resolve pipeline must be resident before the pass runs.
    let Some(resolve_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    // Inverse reverse-Z projection reconstructs the view-space shading point
    // from a device depth; the view->world matrix then lifts the point and its
    // normal into world space for the `SHARC` hash, and the camera world
    // position anchors the identical hash-grid level selection the inject /
    // fill passes used.
    let view_from_clip = extracted.clip_from_view.inverse();
    let world_from_view = extracted.world_from_view.to_matrix();
    let camera_position = extracted.world_from_view.translation();
    let screen = resolve.size;
    let params = GpuWorldRestirResolveParams::new(
        view_from_clip,
        world_from_view,
        camera_position,
        screen,
        &settings,
    );

    let groups_x = resolve_workgroups(screen.x);
    let groups_y = resolve_workgroups(screen.y);

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism world-space ReSTIR resolve"),
        timestamp_writes: None,
    });
    pass.set_pipeline(resolve_pipeline);
    pass.set_bind_group(0, bind_group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, groups_y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_workgroups_rounds_pixels_up_to_the_workgroup_size() {
        assert_eq!(resolve_workgroups(RESOLVE_WORKGROUP_SIZE), 1);
        assert_eq!(resolve_workgroups(RESOLVE_WORKGROUP_SIZE + 1), 2);
        assert_eq!(resolve_workgroups(2 * RESOLVE_WORKGROUP_SIZE), 2);
    }

    #[test]
    fn resolve_workgroups_floors_degenerate_axis_at_one() {
        assert_eq!(resolve_workgroups(0), 1);
        assert_eq!(resolve_workgroups(1), 1);
    }
}
