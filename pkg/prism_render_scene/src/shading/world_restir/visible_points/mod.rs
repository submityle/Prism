//! World-space `ReSTIR` visible-point producer: the screen-space seed of the
//! resident world-space reservoir table.
//!
//! One compute invocation per screen tile reconstructs the tile-centre shading
//! point from the SSR prepass reverse-Z depth + packed view-space normal, lifts
//! it into world space, and appends it to a per-frame visible-point list in the
//! frozen [`super::abi::GpuWorldRestirInjectPoint`] layout. The downstream
//! inject pass hashes each record into its `SHARC` world cell and claims a
//! reservoir slot, so this pass turns the frame's visible geometry into the
//! sparse set of world cells the `ReSTIR` seed/fill passes then light.
//!
//! Mirrors the structure of a scaled-down [`super::super::ssr`] prepass
//! consumer: [`resources`] owns the resident per-view list sized to the SSR
//! framebuffer's tile grid, [`pipeline`] owns the producer compute pipeline and
//! its group-0 layout, [`bind_groups`] builds the group from the SSR prepass
//! views + the list, and [`dispatch`] records the `Core3d` pass. The producer
//! shares the SSR prepass inputs, so it exists exactly when both the subsystem
//! is enabled and the view carries a resident [`super::super::ssr`] prepass.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_world_restir_visible_points_bind_groups;
pub(crate) use dispatch::world_restir_visible_points_pass;
pub(crate) use pipeline::init_world_restir_visible_points_pipeline;
pub(crate) use resources::{prepare_world_restir_visible_points, ViewWorldRestirVisiblePoints};
