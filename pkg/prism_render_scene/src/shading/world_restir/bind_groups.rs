//! `PrepareBindGroups` system building the world-space `ReSTIR` fill group-0
//! bind group per view.
//!
//! Mirrors [`super::super::world_space_gi::bind_groups`], scaled to the single
//! fill pass. Unlike the GI probe passes, the fill pass reads no prepass
//! textures: its only inputs are the subsystem's own resident reservoir
//! tables, so the bind group is built straight from [`ViewWorldRestir`] and is
//! present exactly when that table is resident (i.e. while the subsystem is
//! enabled). The ping-pong `src`/`dst` selection lives in [`ViewWorldRestir`],
//! so the group is rebuilt every frame to follow the flip.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::lights::WorldRestirLights;
use super::pipeline::WorldRestirPipeline;
use super::resources::ViewWorldRestir;
use super::visible_points::ViewWorldRestirVisiblePoints;

/// The group-0 bind group a single view's fill pass records against. Present
/// only while the view's resident reservoir table ([`ViewWorldRestir`]) is.
#[derive(Component)]
pub(crate) struct ViewWorldRestirBindGroups {
    /// group 0 for `fill_main`: previous reservoir table read-only (0) and the
    /// next reservoir table read-write (1).
    fill: BindGroup,
    /// group 0 for `inject_main`: the per-frame visible-point list read-only
    /// (0), this frame's reservoir table read-write (1) and the per-slot
    /// claim-guard array read-write (2). Present only when the view carries a
    /// resident visible-point list (i.e. the producer pass ran this frame).
    inject: Option<BindGroup>,
    /// group 0 for `seed_main`: the injected reservoir table read-only (0, the
    /// per-frame `src` the inject pass wrote cell geometry into), the seeded
    /// reservoir table read-write (1, this frame's `dst`) and the per-frame
    /// candidate light list read-only (2). Present only when the candidate
    /// light buffer is resident (i.e. `prepare_world_restir_lights` uploaded at
    /// least once); an unlit scene with no buffer skips the seed dispatch.
    seed: Option<BindGroup>,
}

impl ViewWorldRestirBindGroups {
    /// group-0 bind group for the `fill_main` dispatch.
    pub(crate) fn fill_group(&self) -> &BindGroup {
        &self.fill
    }

    /// group-0 bind group for the `inject_main` dispatch, present only when the
    /// view's visible-point list is resident this frame.
    pub(crate) fn inject_group(&self) -> Option<&BindGroup> {
        self.inject.as_ref()
    }

    /// group-0 bind group for the `seed_main` dispatch, present only when the
    /// view's candidate light buffer is resident this frame.
    pub(crate) fn seed_group(&self) -> Option<&BindGroup> {
        self.seed.as_ref()
    }
}

/// `PrepareBindGroups` system building [`ViewWorldRestirBindGroups`] for every
/// view whose resident reservoir table is live.
///
/// The ping-pong flip advances in `prepare_world_restir_reservoirs`, so this
/// must run after it and rebuilds the group every frame to pick up the swapped
/// `src`/`dst` buffers.
pub(crate) fn prepare_world_restir_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldRestirPipeline>,
    device: Res<RenderDevice>,
    lights: Res<WorldRestirLights>,
    views: Query<(
        Entity,
        &ViewWorldRestir,
        Option<&ViewWorldRestirVisiblePoints>,
    )>,
) {
    for (entity, restir, visible_points) in &views {
        // fill group: the seeded reservoir table read-only (0, this frame's
        // `dst` the seed pass finalised) and the pooled-output table read-write
        // (1, this frame's `src`). The GRIS spatial pool reads center+neighbour
        // reservoirs from @0 and writes the re-finalised pool to @1; `src` is
        // safe to overwrite because the seed pass already consumed it.
        let fill = device.create_bind_group(
            "prism world-space ReSTIR fill",
            pipeline.fill_layout(),
            &BindGroupEntries::sequential((
                restir.dst_buffer().as_entire_binding(),
                restir.src_buffer().as_entire_binding(),
            )),
        );

        // inject group: the per-frame visible-point list read-only (0), this
        // frame's reservoir table read-write (1) and the per-slot claim-guard
        // array read-write (2). Built only when the view carries a resident
        // visible-point list, so the inject pass auto-skips a view whose
        // producer pass did not run (disabled or missing SSR prepass).
        let inject = visible_points.map(|vp| {
            device.create_bind_group(
                "prism world-space ReSTIR inject",
                pipeline.inject_layout(),
                &BindGroupEntries::sequential((
                    vp.buffer().as_entire_binding(),
                    restir.src_buffer().as_entire_binding(),
                    restir.slot_state_buffer().as_entire_binding(),
                )),
            )
        });

        // seed group: the injected reservoir table read-only (0, this frame's
        // `src` the inject pass wrote cell geometry into), the seeded reservoir
        // table read-write (1, this frame's `dst`) and the per-frame candidate
        // light list read-only (2). Built only when the candidate light buffer
        // is resident, so the seed pass auto-skips an unlit view.
        let seed = lights.buffer().map(|light_buffer| {
            device.create_bind_group(
                "prism world-space ReSTIR seed",
                pipeline.seed_layout(),
                &BindGroupEntries::sequential((
                    restir.src_buffer().as_entire_binding(),
                    restir.dst_buffer().as_entire_binding(),
                    light_buffer.as_entire_binding(),
                )),
            )
        });

        commands
            .entity(entity)
            .insert(ViewWorldRestirBindGroups { fill, inject, seed });
    }
}
