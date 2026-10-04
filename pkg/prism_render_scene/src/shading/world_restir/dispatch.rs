//! `Core3d` scheduling system recording the world-space `ReSTIR` fill dispatch
//! for every view.
//!
//! Mirrors [`super::super::world_space_gi::dispatch`], scaled to the single
//! fill pass: gate on the enable, resolve the view, recover the camera world
//! position from the [`ExtractedView`], build the [`GpuWorldRestirFillParams`]
//! immediate block from the live settings, then dispatch one 1-D workgroup per
//! [`WORLD_RESTIR_WORKGROUP_SIZE`] block of reservoir slots. `fill_main`
//! bounds-checks every invocation against the live `capacity`, so the rounded
//! extent never processes a slot past the table.
//!
//! The fill pass is pure spatial reuse over the resident table: it streams no
//! fresh light candidates (that is the seed pass's job in a follow-up slice),
//! so `light_count` is `0` and the grid-phase `jitter` is zero here; both are
//! still forwarded through the immediate block so the shader's cell
//! quantisation and `RNG` seed stay well-defined.

use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{
    GpuWorldRestirFillParams, GpuWorldRestirInjectParams, WORLD_RESTIR_INJECT_WORKGROUP_SIZE,
    WORLD_RESTIR_WORKGROUP_SIZE,
};
use super::bind_groups::ViewWorldRestirBindGroups;
use super::pipeline::WorldRestirPipeline;
use super::resources::ViewWorldRestir;
use super::settings::PrismWorldRestirSettings;

/// Number of 1-D workgroups the fill dispatch records for a resident table of
/// `capacity` slots: one workgroup per [`WORLD_RESTIR_WORKGROUP_SIZE`] slots,
/// rounded up, with a floor of one so a degenerate `capacity == 0` still issues
/// a single (fully bounds-checked, no-op) workgroup rather than an empty
/// dispatch.
fn fill_workgroups(capacity: u32) -> u32 {
    capacity.max(1).div_ceil(WORLD_RESTIR_WORKGROUP_SIZE)
}

/// `Core3d` scheduling system recording the `fill_main` dispatch for every view
/// whose resident reservoir table and bind group are live.
pub(crate) fn world_restir_fill_pass(
    settings: Res<PrismWorldRestirSettings>,
    view: ViewQuery<(&ViewWorldRestir, &ViewWorldRestirBindGroups, &ExtractedView)>,
    pipeline: Res<WorldRestirPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (restir, groups, extracted) = view.into_inner();

    // The fill pipeline must be resident before the pass runs.
    let Some(fill_pipeline) = cache.get_compute_pipeline(pipeline.fill()) else {
        return;
    };

    // Camera world position anchors the hash-grid level selection (cells grow
    // with distance from the viewer, golden `HashGridParams::level_scale`).
    let camera_position = extracted.world_from_view.translation();
    // Pure spatial reuse: no fresh light stream and no grid-phase jitter this
    // pass (both are the seed pass's responsibility in a later slice).
    let params = GpuWorldRestirFillParams::from_settings(
        camera_position,
        Vec3::ZERO,
        0,
        restir.frame(),
        &settings,
    );

    let groups_x = fill_workgroups(restir.capacity);

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism world-space ReSTIR fill"),
        timestamp_writes: None,
    });
    pass.set_pipeline(fill_pipeline);
    pass.set_bind_group(0, groups.fill_group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, 1, 1);
}

/// `Core3d` scheduling system recording the `inject_main` dispatch for every
/// view whose resident reservoir table, inject bind group and visible-point
/// list are live.
///
/// The producer fills the per-frame visible-point list, the inject pass then
/// open-address claims one `SHARC` cell per visible point and writes the slot
/// geometry, and the fill pass streams spatial reuse over the claimed cells.
/// Inject therefore runs after the producer and before fill. Each frame starts
/// from an empty, spatial-only table, so the per-slot claim guard and the
/// reservoir table inject writes geometry into are cleared on the encoder
/// before the compute pass (no temporal accumulation yet).
pub(crate) fn world_restir_inject_pass(
    settings: Res<PrismWorldRestirSettings>,
    view: ViewQuery<(
        &ViewWorldRestir,
        &ViewWorldRestirBindGroups,
        &super::visible_points::ViewWorldRestirVisiblePoints,
        &ExtractedView,
    )>,
    pipeline: Res<WorldRestirPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (restir, groups, points, extracted) = view.into_inner();

    // The inject bind group is present only when the producer's visible-point
    // list is resident; absence (disabled producer / no SSR prepass) skips.
    let Some(inject_group) = groups.inject_group() else {
        return;
    };
    // The inject pipeline must be resident before the pass runs.
    let Some(inject_pipeline) = cache.get_compute_pipeline(pipeline.inject()) else {
        return;
    };

    // Camera world position anchors the same hash-grid level selection the fill
    // pass uses; grid-phase jitter stays zero (spatial-only, no temporal phase).
    let camera_position = extracted.world_from_view.translation();
    let params = GpuWorldRestirInjectParams::from_settings(
        camera_position,
        Vec3::ZERO,
        points.point_count(),
        restir.frame(),
        &settings,
    );

    let groups_x = points
        .point_count()
        .max(1)
        .div_ceil(WORLD_RESTIR_INJECT_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    // Per-frame spatial-only reset: clear the claim guard and the reservoir
    // table inject writes slot geometry into before any cell is claimed.
    encoder.clear_buffer(restir.slot_state_buffer(), 0, None);
    encoder.clear_buffer(restir.src_buffer(), 0, None);
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism world-space ReSTIR inject"),
        timestamp_writes: None,
    });
    pass.set_pipeline(inject_pipeline);
    pass.set_bind_group(0, inject_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_workgroups_rounds_capacity_up_to_the_workgroup_size() {
        // Exact multiples map one-to-one; partial tails round up so the last
        // slots are still covered.
        assert_eq!(fill_workgroups(WORLD_RESTIR_WORKGROUP_SIZE), 1);
        assert_eq!(fill_workgroups(WORLD_RESTIR_WORKGROUP_SIZE + 1), 2);
        assert_eq!(fill_workgroups(2 * WORLD_RESTIR_WORKGROUP_SIZE), 2);
        assert_eq!(
            fill_workgroups(131_072),
            131_072 / WORLD_RESTIR_WORKGROUP_SIZE
        );
    }

    #[test]
    fn fill_workgroups_floors_degenerate_capacity_at_one() {
        // `capacity == 0` still issues one (bounds-checked, no-op) workgroup
        // rather than an empty dispatch.
        assert_eq!(fill_workgroups(0), 1);
        assert_eq!(fill_workgroups(1), 1);
    }
}
