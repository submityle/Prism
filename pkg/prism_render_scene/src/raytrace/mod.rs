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
//! compute dispatch against the sibling `WESL` shaders
//! `shaders/ray_traverse.wesl` (single-`BLAS` walk),
//! `shaders/tlas_traverse.wesl` (two-level `TLAS`-over-pool walk) and
//! `shaders/ray_footprint.wesl` (ray-cone footprint / texture-`LOD` mip
//! selection).
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
//! * [`tlas_gpu_tests`] - the two-level analogue (also `#[cfg(test)]`):
//!   it builds a real `Tlas` over several `BLAS` soups with affine
//!   instances, packs it with `GpuTlasBuffers::from_tlas` /
//!   `GpuBlasPool::from_blases`, binds the `tlas_traverse` compute
//!   pipeline on a live `wgpu` device and asserts the read-back hits
//!   ray-for-ray against the golden `GpuTlasBuffers::closest_hit` /
//!   `any_hit` walks, skipping gracefully when no adapter is present.
//!
//! * [`footprint_gpu_tests`] - the ray-cone footprint / texture-`LOD`
//!   analogue (also `#[cfg(test)]`): it packs a batch of ray-cone
//!   footprints (each with a per-surface texel size), binds the
//!   `ray_footprint` compute pipeline on a live `wgpu` device and
//!   asserts the read-back `projected_width` / `texel_span` /
//!   `mip_level` / `mip_floor` against the golden
//!   `prism_render_architecture::ray_scene::footprint::RayFootprint`
//!   math ray-for-ray, skipping gracefully when no adapter is present.
//!
//! * [`scene`] - the render-world bridge that turns the stable per-geometry
//!   surface table and the per-instance object→world transforms into a packed
//!   `GpuBlasPool` / `GpuTlasBuffers` pair, the one input the upload path
//!   needs that the golden layout could not synthesize on its own. It is a
//!   pure, device-free `CPU` transform verified against the golden
//!   `GpuTlasBuffers::closest_hit` walk.
//!
//! * [`extract`] - the Bevy half of that bridge: it gathers [`scene`]'s inputs
//!   each frame from the live `RenderGpuScene` mirror and the resident
//!   `RenderShadingGeometryRegistry`, caches the packed hierarchy in the
//!   `RenderWorldAcceleration` resource and rebuilds only when a folded
//!   scene-epoch / geometry-revision signature changes. The gather itself is
//!   the same pure `CPU` transform, unit-tested against the golden walk.
//!
//! * [`resources`] / [`pipeline`] / [`bind_groups`] / [`dispatch`] - the
//!   production service that promotes the proven parity harness into a
//!   reusable, consumer-callable object. [`GpuRayTraversal`] compiles the three
//!   kernels once and exposes synchronous nearest-hit / occlusion / footprint
//!   walks over a plain `RenderDevice` / `RenderQueue`, each the on-device twin
//!   of its `CPU` golden. It is deliberately a raw-`wgpu` service rather than a
//!   render-graph node because no single graph consumer exists yet (screen- and
//!   world-space reflections and ray-traced shadows each want a different
//!   schedule); a future consumer can wrap it without re-porting the kernels.

#![allow(
    dead_code,
    reason = "the ray-traversal subsystem is a verified, consumer-callable production service (GpuRayTraversal + its resources / pipeline / bind-group / dispatch halves) whose first render-graph consumer (screen- or world-space reflections, ray-traced shadows) is not yet wired; every path is exercised on device by the parity tests, so the methods and record types stay as the finished public surface rather than being deleted and re-ported when a consumer lands"
)]

mod abi;
mod bind_groups;
mod dispatch;
mod extract;
mod pipeline;
mod resources;
mod scene;

#[cfg(test)]
mod footprint_gpu_tests;
#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod service_tests;
#[cfg(test)]
mod shader_tests;
#[cfg(test)]
mod tlas_gpu_tests;
