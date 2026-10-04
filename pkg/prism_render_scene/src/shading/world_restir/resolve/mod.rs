//! World-space `ReSTIR` resolve: the screen-space consumer that turns the
//! resident reservoir table back into a direct-illumination export.
//!
//! One compute invocation per screen pixel reconstructs the pixel's world
//! shading point from the SSR prepass reverse-Z depth + packed view-space
//! normal (byte-identical to the [`super::visible_points`] producer), re-hashes
//! it into the same `SHARC` cell the inject pass claimed, linear-probes the
//! finalised reservoir table ([`super::resources::ViewWorldRestir::src_buffer`],
//! the fill pass output) for that cell's reservoir and — on a hit —
//! re-evaluates the stored light sample under a reconnection shift (golden
//! `reconnection_target` semantics), writing the pre-BRDF direct irradiance
//! (`rgb`) + hit confidence (`a`) into the `gi_out` export. The export is a
//! GI-style buffer a downstream composite folds into `scene_color`; this pass
//! never touches the framebuffer, exactly mirroring
//! [`super::super::world_space_gi`]'s resolve.
//!
//! Mirrors the structure of the [`super::visible_points`] producer and
//! [`super::super::world_space_gi`]'s resolve: [`resources`] owns the per-view
//! export sized to the camera viewport, [`pipeline`] owns the resolve compute
//! pipeline and its group-0 layout, [`bind_groups`] builds the group from the
//! SSR prepass views + the reservoir table + the export, and [`dispatch`]
//! records the `Core3d` pass. The resolve consumes both the SSR prepass and the
//! resident reservoir table, so it exists exactly when the subsystem is enabled
//! and the view carries both a [`super::super::ssr`] prepass and a reservoir
//! table.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_world_restir_resolve_bind_groups;
pub(crate) use dispatch::world_restir_resolve_pass;
pub(crate) use pipeline::init_world_restir_resolve_pipeline;
pub(crate) use resources::prepare_world_restir_resolve;
