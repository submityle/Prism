//! Weighted-blended order-independent transparency (WBOIT) subsystem.
//!
//! Transparent geometry cannot live in the visibility buffer (one opaque
//! surface per pixel), so it is drawn in a separate forward pass that
//! accumulates into two MRT targets with McGuire & Bavoil blending (JCGT 2013),
//! then composited over the resolved opaque scene. The math layer is the golden
//! `prism_render_shading::oit` with its GPU twin `shaders/oit.wesl`.
//!
//! Like the other shading stages it is split into cohesive files:
//!
//! * [`targets`] - per-view [`ViewOitTargets`](targets::ViewOitTargets)
//!   (accumulation + revealage) and their `PrepareResources` allocator, gated
//!   exactly like the visibility buffer.
//! * [`phase`] - the [`TransparentOit3d`](phase::TransparentOit3d) binned phase
//!   item the forward draw pass renders.
//! * [`pipeline`] - the [`OitForwardPipeline`](pipeline::OitForwardPipeline)
//!   `SpecializedMeshPipeline` writing the two WBOIT MRT targets with the
//!   additive/multiplicative blend states.
//! * [`queue`] - selects the transparent slice of the unified visibility work
//!   list and specializes the forward pipeline per mesh.
//! * [`draw`] - the [`DrawTransparentOit`](draw::DrawTransparentOit) render
//!   command tuple (view/scene/material bind groups + GPU-Scene mesh draw).
//! * [`forward_node`] - the `Core3d` node that clears the targets to the blend
//!   identity and renders the transparent phase into them.
//! * [`composite_pipeline`] - the [`OitCompositePipeline`] render resource, its
//!   owned group-0 layout, the `RenderStartup` initializer and the per-view
//!   specialization system that stashes a `ViewOitCompositePipelineId`.
//! * [`composite_bind_groups`] - per-view preparation of the group-0 bind group
//!   holding the two WBOIT targets.
//! * [`composite_node`] - the `Core3d` node blending resolved transparency over
//!   the view target, wired after the opaque composite and before tonemapping.

mod composite_bind_groups;
mod composite_node;
mod composite_pipeline;
mod draw;
mod forward_node;
mod phase;
mod pipeline;
mod queue;
mod targets;

#[cfg(test)]
mod shader_tests;

pub(crate) use composite_bind_groups::prepare_oit_composite_bind_groups;
pub(crate) use composite_node::oit_composite;
pub(crate) use composite_pipeline::{
    init_oit_composite_pipeline, prepare_oit_composite_pipelines, OitCompositePipeline,
};
pub(crate) use draw::DrawTransparentOit;
pub(crate) use forward_node::transparent_forward_pass;
pub(crate) use phase::TransparentOit3d;
pub(crate) use pipeline::{init_oit_forward_pipeline, OitForwardPipeline};
pub(crate) use queue::queue_transparent_oit;
pub(crate) use targets::prepare_oit_targets;
