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
//! * [`targets`] - per-view [`ViewOitTargets`] (accumulation + revealage) and
//!   their `PrepareResources` allocator, gated exactly like the visibility
//!   buffer.
//! * [`composite_pipeline`] - the [`OitCompositePipeline`] render resource, its
//!   owned group-0 layout, the `RenderStartup` initializer and the per-view
//!   specialization system that stashes a [`ViewOitCompositePipelineId`].
//! * [`composite_bind_groups`] - per-view preparation of the group-0 bind group
//!   holding the two WBOIT targets.
//! * [`clear_node`] - the interim `Core3d` node clearing both targets to the
//!   blend identity each frame (replaced by the forward draw pass, next slice).
//! * [`composite_node`] - the `Core3d` node blending resolved transparency over
//!   the view target, wired after the opaque composite and before tonemapping.

mod clear_node;
mod composite_bind_groups;
mod composite_node;
mod composite_pipeline;
mod targets;

pub(crate) use clear_node::clear_oit_targets;
pub(crate) use composite_bind_groups::prepare_oit_composite_bind_groups;
pub(crate) use composite_node::oit_composite;
pub(crate) use composite_pipeline::{
    init_oit_composite_pipeline, prepare_oit_composite_pipelines, OitCompositePipeline,
};
pub(crate) use targets::prepare_oit_targets;
