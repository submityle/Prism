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
//! * `shader_tests` - WESL compilation / import-resolution coverage for
//!   `brdf.wesl`, `lighting.wesl` and `shading_resolve.wesl`.

mod abi;

#[cfg(test)]
mod shader_tests;
