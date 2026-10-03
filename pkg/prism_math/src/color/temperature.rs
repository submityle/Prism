//! Correlated color temperature → color, via the Planckian locus.
//!
//! Converts a blackbody temperature in Kelvin to a chromaticity on the
//! Planckian locus using the Kim et al. (2002) cubic-spline approximation of
//! CIE 1931 `(x, y)`, then to XYZ (luminance normalized to `Y = 1`) and finally
//! to linear sRGB. Valid for roughly `1667 K ..= 25000 K`; inputs are clamped
//! to that range.

use crate::color::linear::LinearRgba;
use crate::color::xyz::Xyza;

/// Smallest supported temperature (Kelvin).
pub const MIN_KELVIN: f32 = 1667.0;
/// Largest supported temperature (Kelvin).
pub const MAX_KELVIN: f32 = 25000.0;

/// CIE 1931 `(x, y)` chromaticity on the Planckian locus for temperature
/// `kelvin` (clamped to `[MIN_KELVIN, MAX_KELVIN]`).
#[inline]
pub fn planckian_locus_xy(kelvin: f32) -> (f32, f32) {
    let t = kelvin.clamp(MIN_KELVIN, MAX_KELVIN);
    let inv = 1.0 / t;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;

    let x = if t <= 4000.0 {
        -0.266_123_9e9 * inv3 - 0.234_358_9e6 * inv2 + 0.877_695_6e3 * inv + 0.179_910
    } else {
        -3.025_846_9e9 * inv3 + 2.107_038e6 * inv2 + 0.222_634_7e3 * inv + 0.240_390
    };

    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_2 * x2 + 2.185_558_3 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_9 * x2 + 2.091_37 * x - 0.167_488_67
    } else {
        3.081_758 * x3 - 5.873_387 * x2 + 3.751_129_9 * x - 0.370_014_83
    };

    (x, y)
}

impl LinearRgba {
    /// Build a (unit-luminance) linear-light color for a blackbody at
    /// `kelvin`. Negative components (temperatures whose chromaticity falls
    /// outside the sRGB gamut) are clamped to `0`.
    #[inline]
    pub fn from_temperature(kelvin: f32) -> Self {
        let (x, y) = planckian_locus_xy(kelvin);
        // xyY (Y = 1) -> XYZ.
        let xyz = Xyza::new(x / y, 1.0, (1.0 - x - y) / y, 1.0);
        let c = Self::from_xyz(xyz);
        Self::new(c.red.max(0.0), c.green.max(0.0), c.blue.max(0.0), 1.0)
    }
}
