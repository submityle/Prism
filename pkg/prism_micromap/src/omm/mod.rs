//! Device-free Opacity Micromap (`OMM`) baking primitives.
//!
//! This module groups the pure-`CPU` stages of the baker. Each stage lives in
//! its own file so the `GPU` twin can mirror the layout one compute pass at a
//! time:
//!
//! - [`state`] — the tri-state [`OpacityState`] alphabet and the [`OmmFormat`]
//!   bit widths it packs into.
//! - [`subdivision`] — the canonical recursive micro-triangle ordering.
//! - [`mask`] — the [`AlphaMask`] sampling trait and a texture-backed impl.
//! - [`classify`] — conservative tri-state classification of one micro-triangle.
//! - [`pack`] — `DXR`-compatible little-endian bit packing.
//! - [`bake`] — the triangle baker and the deduplicating [`OmmBuilder`].

pub mod bake;
pub mod classify;
pub mod mask;
pub mod pack;
pub mod state;
pub mod subdivision;

pub use bake::{bake_triangle, BakedOmm, OmmBakeInput, OmmBuilder, OmmBuilderOutput, OmmStats};
pub use classify::{classify_micro_triangle, SampleStrategy};
pub use mask::{AlphaMask, TextureAlphaMask, WrapMode};
pub use pack::{pack, packed_len, unpack};
pub use state::{OmmFormat, OpacityState};
pub use subdivision::{
    micro_triangle_at, micro_triangles, MicroTriangle, SubdivisionLevel, MAX_SUBDIVISION_LEVEL,
};
