//! sRGB electro-optical transfer functions (gamma encode/decode).
//!
//! These convert between **non-linear sRGB** component values (as stored in
//! textures and framebuffers) and **linear-light** values suitable for
//! blending, filtering, and lighting. Alpha is always linear and never passes
//! through these curves.
//!
//! The piecewise curve is the exact IEC 61966-2-1 definition; the approximate
//! `fast_*` variants trade a little accuracy for a single `powf` and are
//! appropriate under the `fast-math` philosophy described in the design doc.

use crate::float::f32 as mf;

/// Decode one non-linear sRGB component to linear light (exact piecewise).
///
/// Input and output are nominally in `[0, 1]`, but values outside that range
/// are handled monotonically so HDR-ish inputs stay well-defined.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_448_237 {
        c / 12.92
    } else {
        mf::powf((c + 0.055) / 1.055, 2.4)
    }
}

/// Encode one linear-light component to non-linear sRGB (exact piecewise).
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * mf::powf(c, 1.0 / 2.4) - 0.055
    }
}

/// Cheap single-`powf` approximation of [`srgb_to_linear`] (gamma 2.2).
#[inline]
pub fn fast_srgb_to_linear(c: f32) -> f32 {
    mf::powf(c, 2.2)
}

/// Cheap single-`powf` approximation of [`linear_to_srgb`] (gamma 2.2).
#[inline]
pub fn fast_linear_to_srgb(c: f32) -> f32 {
    mf::powf(c, 1.0 / 2.2)
}
