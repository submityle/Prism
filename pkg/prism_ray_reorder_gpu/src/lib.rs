//! Optional `wgpu` compute backend for Prism's Shader-Execution-Reordering
//! (SER) ray-coherence sort.
//!
//! Hardware `SER` (NVIDIA Ada / `OptiX`) and the classic ray-sorting technique
//! both reorder the ray/hit stream so threads shaded together share a material,
//! a direction octant, and a spatial neighbourhood — turning divergent path
//! tracing back into coherent, cache-friendly shading. The deterministic
//! contract for that reorder — the `CoherenceKey` packing and the stable
//! permutation it induces — lives device-free in
//! [`prism_render_architecture`](prism_render_architecture::ray_scene::reorder).
//!
//! This crate is the real-device half: a stable 64-bit `LSD` radix sort of
//! `CoherenceKey`/ray-index pairs whose output is validated element-for-element
//! against that `CPU` golden. Device acquisition is best-effort via
//! [`GpuContext::try_headless`], so the parity tests exercise a real Apple
//! `M`-series (or other native) `GPU` when present and skip cleanly otherwise.
//!
//! Provenance: classical, openly published `GPU` radix-sort technique (Blelloch
//! 1990; Satish, Harris, Garland 2009). No Unreal Engine source or derived
//! code.

pub mod buffer;
pub mod context;
pub mod radix;

pub use context::GpuContext;
pub use radix::GpuRayReorder;
