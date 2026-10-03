//! Prism device-free Opacity Micromap (`OMM`) baker.
//!
//! Hardware ray tracing pays a heavy price for alpha-tested geometry: every
//! ray that pierces the bounding triangle of a leaf, a chain-link fence, or a
//! hair card must invoke an any-hit shader to sample the opacity texture and
//! decide whether the hit is real. An **Opacity Micromap** (`OMM`, the
//! `DXR` 1.2 / `VK_EXT_opacity_micromap` feature) amortises that cost: the
//! triangle is subdivided into a regular grid of micro-triangles, each is
//! pre-classified once as fully `Transparent`, fully `Opaque`, or `Unknown`,
//! and at trace time the fixed-function traversal resolves the two certain
//! states without ever entering the any-hit shader. Only the `Unknown`
//! micro-triangles — the thin sliver straddling the alpha edge — still pay for
//! a texture tap.
//!
//! This crate is the device-free half of that pipeline: the `CPU` golden that
//! bakes an alpha-coverage mask into the exact bit layout the hardware
//! consumes. It performs no device work and pulls in no graphics `API`; the
//! optional `wgpu` twin (a sibling `prism_micromap_gpu` crate) is expected to
//! reproduce this golden element-for-element, matching the established Prism
//! "`CPU` golden to `GPU` kernel to real-device parity" contract.
//!
//! # What it computes
//!
//! Given a triangle's three texture coordinates, a subdivision level, and an
//! [`omm::AlphaMask`], the baker:
//!
//! 1. enumerates the `4^level` micro-triangles in a documented, bijective
//!    recursive barycentric order ([`omm::subdivision`]);
//! 2. conservatively classifies each micro-triangle against the mask
//!    ([`omm::classify`]) into an [`omm::OpacityState`];
//! 3. packs the states into the `DXR`-compatible `1-bit` (2-state) or
//!    `2-bit` (4-state) little-endian layout ([`omm::pack`]); and
//! 4. optionally deduplicates identical micromaps across many triangles,
//!    emitting a compact data array plus a per-triangle index
//!    ([`omm::OmmBuilder`]).
//!
//! # Numerical policy
//!
//! Everything here is classical, deterministic floating-point and integer
//! arithmetic (barycentric interpolation, point-in-triangle tests, bit
//! twiddling). There is no `AI`/`ML`, no neural path, and no Unreal Engine
//! source or derived code. The micro-triangle ordering is Prism's own
//! canonical recursive order, fully specified in [`omm::subdivision`], so the
//! `GPU` twin can match it bit-for-bit.
//!
//! # Example
//!
//! ```
//! use prism_micromap::omm::{
//!     bake_triangle, OmmBakeInput, OmmFormat, SampleStrategy, SubdivisionLevel,
//!     TextureAlphaMask,
//! };
//!
//! // A 2x2 alpha mask: left column transparent, right column opaque.
//! let mask = TextureAlphaMask::new(2, 2, [0.0, 1.0, 0.0, 1.0].to_vec(), 0.5);
//! let input = OmmBakeInput {
//!     uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
//!     level: SubdivisionLevel::new(2).unwrap(),
//!     format: OmmFormat::FourState,
//!     strategy: SampleStrategy::Uniform { samples_per_edge: 4 },
//! };
//! let baked = bake_triangle(&input, &mask);
//! assert_eq!(baked.micro_triangle_count(), 16);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod omm;

pub use omm::{
    bake_triangle, BakedOmm, OmmBakeInput, OmmBuilder, OmmBuilderOutput, OmmFormat, OmmStats,
    OpacityState, SampleStrategy, SubdivisionLevel, TextureAlphaMask, WrapMode,
};

/// Convenience re-exports for the common `OMM` baking entry points.
pub mod prelude {
    pub use crate::omm::{
        bake_triangle, AlphaMask, BakedOmm, OmmBakeInput, OmmBuilder, OmmBuilderOutput, OmmFormat,
        OmmStats, OpacityState, SampleStrategy, SubdivisionLevel, TextureAlphaMask, WrapMode,
    };
}

#[cfg(test)]
mod tests;
