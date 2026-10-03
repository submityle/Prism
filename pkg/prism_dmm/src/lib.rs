//! Prism device-free Displaced Micro-Map (`DMM`) baker.
//!
//! A **Displaced Micro-Map** tessellates a base triangle into a regular
//! micro-mesh and stores a per-micro-**vertex** scalar displacement. At trace
//! or raster time each micro-vertex is pushed along an interpolated
//! displacement direction by its decoded height, turning a cheap flat triangle
//! into dense geometric detail without authoring a high-poly mesh. The feature
//! ships as `DXR` / `VK_EXT_displacement_micromap`, the displacement sibling of
//! the opacity micro-map.
//!
//! This crate is the device-free half of that pipeline: the `CPU` golden that
//! bakes a displacement height field into the exact `11-bit` unorm layout the
//! hardware consumes. It performs no device work and pulls in no graphics
//! `API`; a future `wgpu` twin is expected to reproduce this golden
//! element-for-element, matching the Prism "`CPU` golden to `GPU` kernel to
//! real-device parity" contract.
//!
//! # What it computes
//!
//! Given a triangle's three texture coordinates, a subdivision level, and a
//! [`dmm::DisplacementMap`], the baker:
//!
//! 1. enumerates the `(n + 1) * (n + 2) / 2` micro-vertices (`n == 2^level`) in
//!    a documented, bijective canonical order ([`dmm::subdivision`]);
//! 2. samples the displacement height at each micro-vertex `UV`
//!    ([`dmm::heightmap`]);
//! 3. maps the triangle height range onto `[0, 1]` and quantizes to `11-bit`
//!    unorm ([`dmm::quantize`]);
//! 4. packs the codes into the raw little-endian `11-bit` bitstream
//!    ([`dmm::encode`]); and
//! 5. optionally deduplicates byte-identical micro-maps across many triangles
//!    ([`dmm::DmmBuilder`]).
//!
//! # Numerical policy
//!
//! Everything here is classical, deterministic floating-point and integer
//! arithmetic (barycentric interpolation, bilinear filtering, round-to-nearest
//! quantization, bit twiddling). There is no `AI`/`ML`, no neural path, and no
//! Unreal Engine source or derived code. The packing is the **uncompressed /
//! raw** `11-bit` layout, not the vendor anchor-plus-correction block
//! compression; it is a correct golden that a future compressor can consume
//! without upstream changes.
//!
//! # Example
//!
//! ```
//! use prism_dmm::prelude::*;
//!
//! // A 2x2 height texture ramping along the U axis.
//! let map = TextureDisplacementMap::new(2, 2, vec![0.0, 1.0, 0.0, 1.0], WrapMode::Clamp);
//! let input = DmmBakeInput {
//!     uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
//!     level: DmmSubdivisionLevel::new(2).unwrap(),
//!     scale_bias_mode: ScaleBiasMode::PerTriangle,
//! };
//! let baked = bake_triangle(&input, &map);
//! assert_eq!(baked.micro_vertex_count(), 15);
//! assert_eq!(baked.unpack_codes().as_deref(), Some(baked.codes()));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod dmm;

pub use dmm::{
    bake_triangle, barycentric_f32, dequantize_unorm11, displaced_position, micro_vertex_at,
    micro_vertices, pack_unorm11, packed_len, quantize_unorm11, unpack_unorm11, BakedDmm,
    DisplacementMap, DisplacementScaleBias, DmmBakeInput, DmmBuilder, DmmBuilderOutput,
    DmmSubdivisionLevel, ScaleBiasMode, TextureDisplacementMap, WrapMode,
};

/// Convenience re-exports for the common `DMM` baking entry points.
pub mod prelude {
    pub use crate::dmm::{
        bake_triangle, displaced_position, BakedDmm, DisplacementMap, DisplacementScaleBias,
        DmmBakeInput, DmmBuilder, DmmBuilderOutput, DmmSubdivisionLevel, ScaleBiasMode,
        TextureDisplacementMap, WrapMode,
    };
}

#[cfg(test)]
mod tests;
