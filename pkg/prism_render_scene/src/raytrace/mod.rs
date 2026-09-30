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
//! * [`gpu_tests`] - real-device `GPU` parity coverage (compiled only under
//!   `#[cfg(test)]`): it builds a real `Bvh`, packs it with
//!   `GpuBvhBuffers::from_bvh`, binds the `ray_traverse` compute pipeline on a
//!   live `wgpu` device and asserts the read-back hits ray-for-ray against the
//!   `CPU` golden `GpuBvhBuffers::closest_hit` / `any_hit` walks, skipping
//!   gracefully when no adapter is present.
//!
//! The production `wgpu` pipeline / bind-group helpers and the render-graph
//! dispatch node land once a real ray-tracing consumer (screen-space or
//! world-space reflections, ray-traced shadows) fixes their exact binding
//! interface; this slice proves the kernel is numerically correct on device
//! first.

mod abi;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod shader_tests;
