//! Host-side mip-chain generation for decoded `RGBA8` textures.
//!
//! This module complements [`texture_codec`](crate::texture_codec): after a
//! block-compressed level is decoded to `RGBA8`, AAA pipelines often need to
//! build the remaining mips on the host (gamma-correctly for colour). The
//! reduction is an exact, deterministic box filter -- a CPU golden for a GPU
//! compute down-sampler, with no AI/ML anywhere.
//!
//! * [`srgb`] -- the sRGB <-> scene-linear transfer used by the colour path.
//! * [`box_filter`] -- the [`Rgba8Image`] level type plus [`box_downsample`]
//!   and [`generate_mip_chain`] under a [`ColorSpace`] policy.

mod box_filter;
mod srgb;

pub use box_filter::{box_downsample, generate_mip_chain, ColorSpace, Rgba8Image};
pub use srgb::{linear_to_srgb, srgb_to_linear};
