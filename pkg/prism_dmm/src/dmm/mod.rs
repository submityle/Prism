//! Device-free Displaced Micro-Map (`DMM`) baking primitives.
//!
//! This module groups the pure-`CPU` stages of the baker. Each stage lives in
//! its own file so the `GPU` twin can mirror the layout one pass at a time:
//!
//! - [`subdivision`] — the canonical micro-vertex lattice and ordering.
//! - [`heightmap`] — the [`DisplacementMap`] trait and a bilinear texture impl.
//! - [`quantize`] — `11-bit` unorm quantization and the per-triangle
//!   [`DisplacementScaleBias`].
//! - [`encode`] — the raw (uncompressed) `11-bit` bitstream packing.
//! - [`bake`] — the triangle baker, world-space displacement, and the
//!   deduplicating [`DmmBuilder`].

pub mod bake;
pub mod encode;
pub mod heightmap;
pub mod quantize;
pub mod subdivision;

pub use bake::{
    bake_triangle, displaced_position, BakedDmm, DmmBakeInput, DmmBuilder, DmmBuilderOutput,
    ScaleBiasMode,
};
pub use encode::{pack_unorm11, packed_len, unpack_unorm11, BITS_PER_CODE};
pub use heightmap::{DisplacementMap, TextureDisplacementMap, WrapMode};
pub use quantize::{dequantize_unorm11, quantize_unorm11, DisplacementScaleBias, UNORM11_MAX};
pub use subdivision::{
    barycentric_f32, micro_vertex_at, micro_vertices, DmmSubdivisionLevel, MAX_SUBDIVISION_LEVEL,
};
