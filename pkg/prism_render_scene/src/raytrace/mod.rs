//! GPU ray-traversal compute subsystem.
//!
//! The packed acceleration-structure `ABI` and its `CPU`-golden traversal
//! (`closest_hit` / `any_hit` over the flattened `nodes` / `triangles` buffers,
//! the `TLAS` instance transform and the shared `BLAS` pool) are owned by the
//! dependency-free, float-audited `prism_render_architecture::ray_scene`
//! module: `GpuBvhBuffers` / `GpuTlasBuffers` / `GpuBlasPool` pin the exact word
//! layout (`NODE_WORDS = 12`, `TRIANGLE_WORDS = 12`, `INSTANCE_WORDS = 16`,
//! `BLAS_OFFSET_WORDS = 4`) and their walks are the authoritative reference. This
//! module is the `bevy`-render half that turns that `ABI` into a real `wgpu`
//! compute dispatch against the sibling `WESL` shader
//! `shaders/ray_traverse.wesl`.
//!
//! Mirroring the water and cloth compute subsystems, the module is split into
//! cohesive files rather than one large module:
//!
//! * [`abi`] - the packed record strides and field offsets shared with the
//!   traversal shader, re-exported from the golden
//!   `prism_render_architecture::ray_scene` layout and pinned to it by the
//!   contract tests so a stride drift fails the build.
//!
//! The `wgpu` pipeline, bind groups and dispatch node (and their real-device
//! `GPU` parity tests against `GpuBvhBuffers::closest_hit`) land in the
//! following slices.

mod abi;

#[cfg(test)]
mod shader_tests;
