//! `GPU` stable 64-bit least-significant-digit (`LSD`) radix sort of SER
//! ray-coherence keys.
//!
//! Path tracing diverges: adjacent threads in a wavefront bounce toward
//! unrelated materials and directions, serialising shading and scattering
//! memory accesses. Shader Execution Reordering fixes this by sorting the
//! ray/hit stream by a coherence key so threads shaded together are coherent.
//! [`prism_render_architecture`](prism_render_architecture::ray_scene::reorder)
//! defines the device-free contract — the `CoherenceKey` packing and the
//! stable `radix_order` permutation; this module is the real-device `wgpu`
//! twin that reproduces that permutation bit-for-bit.
//!
//! The key is 64 bits, so the sort runs [`PASSES`] `= 8` stable passes over
//! [`RADIX_BITS`]-bit digits (versus 4 for a 32-bit key). Each pass tiles the
//! keys into [`TILE`]-wide blocks and runs a device `count` histogram, a host
//! exclusive scan of that small digit-major histogram, and a device stable
//! `scatter` — exactly the per-block histogram decomposition the engine's
//! 32-bit physics radix already uses, extended to two `u32` key words.
//!
//! # Provenance
//!
//! The `LSD` radix sort with a per-block, digit-major histogram scanned to
//! output offsets is a classical, openly published `GPU` technique (Blelloch
//! 1990; Satish, Harris, Garland, "Designing Efficient Sorting Algorithms for
//! Manycore GPUs", 2009). This module contains no Unreal Engine source or
//! derived code.

pub mod config;
pub mod gpu;
pub mod layout;

pub use config::{LOW_WORD_PASSES, PASSES, RADIX, RADIX_BITS, TILE};
pub use gpu::GpuRayReorder;
