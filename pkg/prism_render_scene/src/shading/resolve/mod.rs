//! GPU shading *resolve* subsystem.
//!
//! This is the compute stage that turns the material-classified visibility
//! buffer into a shaded HDR image.  It runs after
//! [`super::classification_gpu`] has bucketed every covered pixel into a
//! per-`MaterialShadingClass` worklist and computed the indirect dispatch
//! arguments, and before the main pass composites the result to the screen.
//!
//! The stage is deliberately split into cohesive files instead of one large
//! module:
//!
//! * [`abi`] - `#[repr(C)]` records shared with `shaders/shading_resolve.wesl`
//!   (immediate push-constant block, workgroup constants) plus their
//!   `size_of` contract tests.
//! * [`pipeline`] - the [`ShadingResolvePipeline`] render resource, its
//!   bind-group layouts and the `RenderStartup` initializer.
//! * [`bind_groups`] - per-view preparation of the shading-geometry, HDR
//!   storage-texture and lighting bind groups.
//! * [`dispatch`] - the `Core3d` graph node that records the indirect compute
//!   dispatches, one per shading class.
//! * [`motion`] - the per-view current/previous view-projection uniform plus
//!   its render-world history, feeding the motion-vector G-buffer output.
//! * `shader_tests` - WESL compilation / import-resolution coverage for
//!   `brdf.wesl`, `lighting.wesl` and `shading_resolve.wesl`.

mod abi;
mod bind_groups;
mod dispatch;
mod motion;
mod pipeline;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_shading_resolve_bind_groups;

pub(crate) use bind_groups::ViewResolveVsmPageTable;
pub(crate) use dispatch::dispatch_shading_resolve;
pub(crate) use motion::{prepare_resolve_motion, ResolveMotionHistory};
pub(crate) use abi::GpuVsmResolveParams;
pub(crate) use pipeline::init_shading_resolve_pipeline;
