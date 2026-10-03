//! Color spaces and conversions (M5).
//!
//! The type family makes the linear/non-linear distinction explicit so the
//! classic "added colors in sRGB and everything got dark" bug is unrepresentable:
//!
//! - [`Srgba`] — gamma-encoded sRGB, for **storage/display**.
//! - [`LinearRgba`] — linear-light sRGB primaries, the **hub** for blending,
//!   lighting, and interpolation.
//! - [`Xyza`] — CIE 1931 XYZ (D65), the device-independent bridge.
//! - [`Oklaba`] / [`Oklcha`] — perceptually-uniform mixing/gradient space.
//! - [`Hsla`] / [`Hsva`] — cylindrical pickers defined over sRGB.
//!
//! All conversions route through [`LinearRgba`] (or [`Srgba`] for HSL/HSV),
//! keeping the graph small and the round-trips tight. Alpha is always a
//! straight, linear value and never passes through a gamma curve.
//!
//! HDR tone mapping and gamut compression are deliberately **out of scope**
//! here (they belong in `prism_render`); this module only defines the numeric
//! color-space transforms.

mod hsl;
mod linear;
mod oklab;
mod srgb;
pub mod transfer;
mod temperature;
mod xyz;

pub use hsl::{Hsla, Hsva};
pub use linear::LinearRgba;
pub use oklab::{Oklaba, Oklcha};
pub use srgb::Srgba;
pub use temperature::{MAX_KELVIN, MIN_KELVIN, planckian_locus_xy};
pub use xyz::Xyza;
