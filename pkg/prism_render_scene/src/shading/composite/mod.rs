//! HDR scene-color -> view-target composite subsystem.
//!
//! This is the terminal stage of the GPU shading chain.  After
//! [`super::resolve`] has written linear HDR radiance into each view's
//! `scene_color` storage texture for every covered pixel, this stage copies
//! that radiance onto the core-3d view target with a fullscreen pass, so the
//! downstream tonemapping node maps the *Prism-shaded* image to the display
//! instead of any placeholder the main pass drew.
//!
//! Like the resolve stage, it is split into cohesive files rather than one
//! large module:
//!
//! * [`pipeline`] — the [`ShadingCompositePipeline`] render resource, its owned
//!   group-0 layout, the `RenderStartup` initializer and the per-view
//!   specialization system that stashes a [`ViewCompositePipelineId`].
//! * [`bind_groups`] — per-view preparation of the group-0 bind group holding
//!   the three resolve outputs.
//! * [`node`] — the `Core3d` graph node that records the fullscreen
//!   copy-or-discard draw, wired *after* the main pass and *before*
//!   tonemapping.
//! * `shader_tests` — WESL compilation coverage for `composite.wesl`.

mod bind_groups;
mod node;
mod pipeline;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_shading_composite_bind_groups;
pub(crate) use node::composite_shading;
pub(crate) use pipeline::{
    init_shading_composite_pipeline, prepare_shading_composite_pipelines, ShadingCompositePipeline,
};
