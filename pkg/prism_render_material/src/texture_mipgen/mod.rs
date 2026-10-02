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
//! * [`windowed`] -- higher-quality separable Lanczos-2/3 reduction
//!   ([`windowed_downsample`] / [`generate_mip_chain_windowed`]) that suppresses
//!   mip shimmering better than the box filter.
//! * [`kaiser`] -- a Kaiser-windowed sinc reduction ([`kaiser_downsample`] /
//!   [`generate_mip_chain_kaiser`]) with a tunable `beta` shape parameter for
//!   the sharpness/ringing trade-off, the texture-tool high-quality default.
//! * [`gaussian`] -- a strictly non-negative Gaussian reduction
//!   ([`gaussian_downsample`] / [`generate_mip_chain_gaussian`]) that never
//!   over- or under-shoots (no ringing), the soft prefilter for roughness,
//!   height, and coverage maps where sinc side-lobes are unacceptable.
//! * [`alpha_coverage`] -- preserve alpha-test coverage across a mip chain
//!   ([`preserve_alpha_coverage`]) so alpha-tested foliage/decals do not thin
//!   out in the distance; a filter-agnostic post-pass.
//! * [`premultiplied`] -- alpha-weighted (premultiplied) box reduction
//!   ([`premultiplied_box_downsample`] / [`generate_mip_chain_premultiplied`])
//!   so transparent texels do not bleed colour into alpha-blended edges.

mod alpha_coverage;
mod box_filter;
mod gaussian;
mod kaiser;
mod premultiplied;
mod resample_core;
mod srgb;
mod windowed;

pub use alpha_coverage::{
    alpha_test_coverage, apply_alpha_scale, preserve_alpha_coverage, solve_alpha_scale,
};
pub use box_filter::{box_downsample, generate_mip_chain, ColorSpace, Rgba8Image};
pub use gaussian::{gaussian_downsample, generate_mip_chain_gaussian, GaussianFilter};
pub use kaiser::{generate_mip_chain_kaiser, kaiser_downsample, KaiserFilter};
pub use premultiplied::{generate_mip_chain_premultiplied, premultiplied_box_downsample};
pub use srgb::{linear_to_srgb, srgb_to_linear};
pub use windowed::{generate_mip_chain_windowed, windowed_downsample, WindowedKernel};
