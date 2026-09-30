//! White-balance / colour-temperature *linear-space `RGB` gain* contract
//! (design §5, §30).
//!
//! Production grading stacks expose a *temperature* + *tint* control that shifts
//! the neutral point of an image the way changing the light source would: a warm
//! tungsten bulb pushes the image red, an overcast sky pushes it blue, and the
//! green-magenta *tint* axis corrects the residual cast fluorescent lighting
//! leaves behind. This module owns the `CPU`-verifiable *maths* of that control:
//! it turns a [`WhiteBalance`] setting into a per-channel linear-space `RGB`
//! *gain* a `GPU` kernel multiplies onto each pixel.
//!
//! # Distinction from the other colour modules
//!
//! This module is deliberately *not* a look-up table and *not* a curve:
//!
//! * [`super::color_grade_lut`] is a full 3D `RGB`->`RGB` colour cube (an
//!   arbitrary creative transform).
//! * [`super::tonemap`] is a per-channel *tone curve* mapping `HDR` to display.
//! * [`super::color_gradient`] is a 1D over-life `RGBA` ramp.
//!
//! White balance is the simplest of the family: three scalar multipliers, one
//! per channel, applied in linear space before any of the above. The green
//! channel is normalised to `~1.0` so temperature only trades red against blue,
//! and the tint axis then trades green against magenta.
//!
//! # Planckian locus, rational-polynomial approximation
//!
//! The temperature->`RGB` mapping follows the *planckian locus* (the chromaticity
//! a black body radiates as it heats from a dull red at `1000 K` to a cold blue
//! past `15000 K`). The classic closed forms (see Tanner Helland's widely-cited
//! fit) evaluate `log`/`pow` of the temperature. This module avoids every
//! transcendental: each channel is a **piecewise rational-polynomial
//! approximation** (a ratio of two ordinary polynomials evaluated by Horner's
//! method) fitted to the same locus. No `sin`/`cos`/`exp`/`ln`/`pow`/`atan` is
//! ever called, so the `CPU` reference stays bit-reproducible against a future
//! `GPU` evaluator, matching the determinism contract of the sibling
//! [`super::simulation`] module.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Lowest colour temperature the locus fit is valid for; cooler requests clamp
/// here (a deep tungsten/candle red).
pub const KELVIN_MIN: f32 = 1000.0;

/// Highest colour temperature the locus fit is valid for; hotter requests clamp
/// here (a cold overcast blue).
pub const KELVIN_MAX: f32 = 15000.0;

/// The neutral daylight reference: a white-balance set to this temperature with
/// zero tint yields a gain of `~[1, 1, 1]` (an approximate identity).
pub const NEUTRAL_KELVIN: f32 = 6500.0;

/// Below this temperature the blue channel of the locus has fallen to zero (a
/// pure red-orange black body emits no blue), so the blue gain is held at `0`.
const BLUE_ZERO_KELVIN: f32 = 1900.0;

/// The temperature at which the piecewise blue fit switches from its warm
/// branch to its cool branch (the daylight knee of the locus).
const WARM_COOL_SPLIT_KELVIN: f32 = 6600.0;

/// Minimum tint (full magenta correction along the green-magenta axis).
pub const TINT_MIN: f32 = -1.0;

/// Maximum tint (full green correction along the green-magenta axis).
pub const TINT_MAX: f32 = 1.0;

/// How strongly a positive tint lifts the green channel.
const TINT_GREEN_GAIN: f32 = 0.4;

/// How strongly a positive tint suppresses the magenta (red+blue) channels.
const TINT_MAGENTA_GAIN: f32 = 0.2;

/// Red-channel gain fit over the whole `[KELVIN_MIN, KELVIN_MAX]` range, in
/// `kilokelvin` (`temperature / 1000`). Numerator over [`R_DEN`], both in
/// ascending powers, evaluated by [`eval_rational`].
const R_NUM: [f32; 4] = [-3.09790, -1.02807, 0.398053, -0.0351264];
/// Denominator of the red-channel rational fit (ascending powers).
const R_DEN: [f32; 4] = [1.0, -2.55882, 0.601709, -0.0449934];

/// Blue-channel gain numerator for the *warm* branch
/// (`BLUE_ZERO_KELVIN..=WARM_COOL_SPLIT_KELVIN`, ascending powers).
const BW_NUM: [f32; 4] = [2.54166, -3.12041, 1.14996, -0.111423];
/// Blue-channel gain denominator for the warm branch (ascending powers).
const BW_DEN: [f32; 5] = [1.0, -1.81911, 0.870344, -0.0941453, 8.75e-05];

/// Blue-channel gain numerator for the *cool* branch
/// (`WARM_COOL_SPLIT_KELVIN..=KELVIN_MAX`, ascending powers).
const BC_NUM: [f32; 2] = [0.911769, -0.158482];
/// Blue-channel gain denominator for the cool branch (ascending powers).
const BC_DEN: [f32; 5] = [1.0, -0.222001, 0.0105773, -0.000512, 9.9e-06];

/// Evaluates a polynomial given ascending-power coefficients by Horner's method
/// (no allocation, no transcendental, iterator-driven so no manual indexing).
#[must_use]
fn eval_poly(coeffs: &[f32], x: f32) -> f32 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Evaluates a rational polynomial `num(x) / den(x)`; both are ascending-power
/// coefficient slices. The fitted denominators never vanish on their intended
/// domains, so this never divides by zero for an in-range temperature.
#[must_use]
fn eval_rational(num: &[f32], den: &[f32], x: f32) -> f32 {
    eval_poly(num, x) / eval_poly(den, x)
}

/// A white-balance setting: a colour temperature plus a green-magenta tint.
///
/// Both fields are authored values; the derived linear-space gain is produced by
/// [`WhiteBalance::rgb_gain`]. The type is a small, `Copy` parameter block so a
/// grading pass can hold one per layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteBalance {
    /// Target correlated colour temperature in `kelvin`. Values outside
    /// `[KELVIN_MIN, KELVIN_MAX]` clamp into range before evaluation.
    pub temp_kelvin: f32,
    /// Green-magenta tint in `[TINT_MIN, TINT_MAX]`; positive lifts green and
    /// suppresses magenta, negative does the reverse, and `0` leaves the green
    /// axis untouched. Values outside the range clamp before evaluation.
    pub tint: f32,
}

impl WhiteBalance {
    /// Number of meaningful scalar fields packed into the `std430` block (the
    /// three resolved gain channels).
    const FIELD_COUNT: usize = 3;

    /// Byte size of the `std430` packing: three gain scalars padded up to a
    /// single `vec4` slot so the block honours the 16-byte `std430` base
    /// alignment the `GPU` kernel expects.
    pub const STD430_SIZE: usize = Self::FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

    /// A neutral daylight white-balance: [`NEUTRAL_KELVIN`] with zero tint,
    /// whose gain is an approximate identity.
    pub const NEUTRAL: Self = Self {
        temp_kelvin: NEUTRAL_KELVIN,
        tint: 0.0,
    };

    /// Builds a white-balance from an explicit temperature and tint.
    #[must_use]
    pub const fn new(temp_kelvin: f32, tint: f32) -> Self {
        Self { temp_kelvin, tint }
    }

    /// Builds a white-balance from a colour temperature alone, with zero tint.
    #[must_use]
    pub const fn from_kelvin(temp_kelvin: f32) -> Self {
        Self {
            temp_kelvin,
            tint: 0.0,
        }
    }

    /// The temperature clamped into the fit's valid `[KELVIN_MIN, KELVIN_MAX]`
    /// range.
    #[must_use]
    fn clamped_kelvin(&self) -> f32 {
        self.temp_kelvin.clamp(KELVIN_MIN, KELVIN_MAX)
    }

    /// The tint clamped into `[TINT_MIN, TINT_MAX]`.
    #[must_use]
    fn clamped_tint(&self) -> f32 {
        self.tint.clamp(TINT_MIN, TINT_MAX)
    }

    /// The green-normalised black-body gain `[red, 1, blue]` for the clamped
    /// temperature, before any tint is applied. Both the red and blue channels
    /// are held non-negative.
    #[must_use]
    fn blackbody_gain(&self) -> [f32; 3] {
        let kelvin = self.clamped_kelvin();
        let x = kelvin / 1000.0;
        let red = eval_rational(&R_NUM, &R_DEN, x).max(0.0);
        let blue = if kelvin < BLUE_ZERO_KELVIN {
            0.0
        } else if kelvin <= WARM_COOL_SPLIT_KELVIN {
            eval_rational(&BW_NUM, &BW_DEN, x).max(0.0)
        } else {
            eval_rational(&BC_NUM, &BC_DEN, x).max(0.0)
        };
        [red, 1.0, blue]
    }

    /// The resolved linear-space `RGB` gain for this white-balance.
    ///
    /// The temperature sets a green-normalised red/blue split along the
    /// planckian locus; the tint then trades green against magenta. At
    /// [`NEUTRAL_KELVIN`] with zero tint the gain is `~[1, 1, 1]`; below
    /// `~5000 K` red dominates (a warm cast) and above `~8000 K` blue dominates
    /// (a cool cast). Every channel is non-negative.
    #[must_use]
    pub fn rgb_gain(&self) -> [f32; 3] {
        let base = self.blackbody_gain();
        let tint = self.clamped_tint();
        // Green-magenta axis: green rises with tint, magenta (red+blue) falls.
        let green_mul = 1.0 + tint * TINT_GREEN_GAIN;
        let magenta_mul = (1.0 - tint * TINT_MAGENTA_GAIN).max(0.0);
        [
            base[0] * magenta_mul,
            base[1] * green_mul,
            base[2] * magenta_mul,
        ]
    }

    /// Applies this white-balance to a linear-space `RGB` triple by multiplying
    /// each channel by the matching [`WhiteBalance::rgb_gain`] factor.
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let gain = self.rgb_gain();
        [rgb[0] * gain[0], rgb[1] * gain[1], rgb[2] * gain[2]]
    }

    /// Packs the resolved [`WhiteBalance::rgb_gain`] into its `std430` block.
    ///
    /// Laid out little-endian as the red, green and blue gain scalars (`f32`)
    /// followed by one `f32` of zero padding, so the block spans a single
    /// `vec4` slot ([`WhiteBalance::STD430_SIZE`] bytes) the `GPU` grading
    /// kernel binds and multiplies onto each pixel.
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let gain = self.rgb_gain();
        let mut bytes = [0u8; Self::STD430_SIZE];
        bytes[0..4].copy_from_slice(&gain[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&gain[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&gain[2].to_le_bytes());
        bytes
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// white-balance gain blocks, clamped up to a single element per the shared
    /// [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

impl Default for WhiteBalance {
    /// The neutral daylight white-balance ([`WhiteBalance::NEUTRAL`]).
    fn default() -> Self {
        Self::NEUTRAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` comparisons used by the tests; direct
    /// `==` on floating point is intentionally avoided.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// Reads a little-endian `f32` back out of packed `std430` bytes.
    fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(buf)
    }

    #[test]
    fn neutral_daylight_gain_is_close_to_one() {
        let gain = WhiteBalance::from_kelvin(NEUTRAL_KELVIN).rgb_gain();
        assert!((gain[0] - 1.0).abs() < 0.05);
        assert!(approx(gain[1], 1.0));
        assert!((gain[2] - 1.0).abs() < 0.05);
    }

    #[test]
    fn green_channel_is_normalised_to_one_without_tint() {
        for &k in &[1000.0_f32, 3000.0, 6500.0, 9000.0, 15000.0] {
            let gain = WhiteBalance::from_kelvin(k).rgb_gain();
            assert!(approx(gain[1], 1.0));
        }
    }

    #[test]
    fn warm_temperature_is_red_biased() {
        let gain = WhiteBalance::from_kelvin(3000.0).rgb_gain();
        assert!(gain[0] > 1.0);
        assert!(gain[2] < 1.0);
        assert!(gain[0] > gain[2]);
    }

    #[test]
    fn very_warm_temperature_has_no_blue() {
        let gain = WhiteBalance::from_kelvin(1500.0).rgb_gain();
        assert!(approx(gain[2], 0.0));
        assert!(gain[0] > 1.0);
    }

    #[test]
    fn cool_temperature_is_blue_biased() {
        let gain = WhiteBalance::from_kelvin(10000.0).rgb_gain();
        assert!(gain[2] > 1.0);
        assert!(gain[0] < 1.0);
        assert!(gain[2] > gain[0]);
    }

    #[test]
    fn red_gain_decreases_monotonically_with_temperature() {
        let mut prev = f32::INFINITY;
        let mut k = KELVIN_MIN;
        while k <= KELVIN_MAX {
            let red = WhiteBalance::from_kelvin(k).rgb_gain()[0];
            assert!(red <= prev + CMP_EPS, "red rose at {k}");
            prev = red;
            k += 250.0;
        }
    }

    #[test]
    fn blue_gain_increases_monotonically_with_temperature() {
        let mut prev = f32::NEG_INFINITY;
        let mut k = KELVIN_MIN;
        while k <= KELVIN_MAX {
            let blue = WhiteBalance::from_kelvin(k).rgb_gain()[2];
            assert!(blue >= prev - CMP_EPS, "blue fell at {k}");
            prev = blue;
            k += 250.0;
        }
    }

    #[test]
    fn low_temperature_clamps_to_kelvin_min() {
        let below = WhiteBalance::from_kelvin(200.0).rgb_gain();
        let at = WhiteBalance::from_kelvin(KELVIN_MIN).rgb_gain();
        for (b, a) in below.iter().zip(at.iter()) {
            assert!(approx(*b, *a));
        }
    }

    #[test]
    fn high_temperature_clamps_to_kelvin_max() {
        let above = WhiteBalance::from_kelvin(40000.0).rgb_gain();
        let at = WhiteBalance::from_kelvin(KELVIN_MAX).rgb_gain();
        for (hi, a) in above.iter().zip(at.iter()) {
            assert!(approx(*hi, *a));
        }
    }

    #[test]
    fn tint_zero_leaves_green_axis_unchanged() {
        let base = WhiteBalance::from_kelvin(6500.0).rgb_gain();
        let tinted = WhiteBalance::new(6500.0, 0.0).rgb_gain();
        assert!(approx(base[1], 1.0));
        for (b, t) in base.iter().zip(tinted.iter()) {
            assert!(approx(*b, *t));
        }
    }

    #[test]
    fn positive_tint_raises_green_and_lowers_magenta() {
        let neutral = WhiteBalance::new(6500.0, 0.0).rgb_gain();
        let green_tint = WhiteBalance::new(6500.0, 0.5).rgb_gain();
        assert!(green_tint[1] > neutral[1]);
        assert!(green_tint[0] < neutral[0]);
        assert!(green_tint[2] < neutral[2]);
    }

    #[test]
    fn negative_tint_lowers_green_and_raises_magenta() {
        let neutral = WhiteBalance::new(6500.0, 0.0).rgb_gain();
        let magenta_tint = WhiteBalance::new(6500.0, -0.5).rgb_gain();
        assert!(magenta_tint[1] < neutral[1]);
        assert!(magenta_tint[0] > neutral[0]);
        assert!(magenta_tint[2] > neutral[2]);
    }

    #[test]
    fn green_gain_is_monotonic_in_tint() {
        let mut prev = f32::NEG_INFINITY;
        let mut t = TINT_MIN;
        while t <= TINT_MAX {
            let green = WhiteBalance::new(6500.0, t).rgb_gain()[1];
            assert!(green >= prev - CMP_EPS);
            prev = green;
            t += 0.1;
        }
    }

    #[test]
    fn magenta_gain_is_monotonic_in_tint() {
        let mut prev = f32::INFINITY;
        let mut t = TINT_MIN;
        while t <= TINT_MAX {
            let red = WhiteBalance::new(6500.0, t).rgb_gain()[0];
            assert!(red <= prev + CMP_EPS);
            prev = red;
            t += 0.1;
        }
    }

    #[test]
    fn tint_clamps_beyond_range() {
        let over = WhiteBalance::new(6500.0, 5.0).rgb_gain();
        let at = WhiteBalance::new(6500.0, TINT_MAX).rgb_gain();
        for (o, a) in over.iter().zip(at.iter()) {
            assert!(approx(*o, *a));
        }
    }

    #[test]
    fn apply_multiplies_each_channel_by_gain() {
        let wb = WhiteBalance::new(4000.0, 0.2);
        let gain = wb.rgb_gain();
        let out = wb.apply([0.5, 0.25, 0.75]);
        assert!(approx(out[0], 0.5 * gain[0]));
        assert!(approx(out[1], 0.25 * gain[1]));
        assert!(approx(out[2], 0.75 * gain[2]));
    }

    #[test]
    fn apply_at_neutral_is_approximate_identity() {
        let out = WhiteBalance::NEUTRAL.apply([0.4, 0.6, 0.8]);
        assert!((out[0] - 0.4).abs() < 0.05);
        assert!(approx(out[1], 0.6));
        assert!((out[2] - 0.8).abs() < 0.05);
    }

    #[test]
    fn from_kelvin_sets_zero_tint() {
        let wb = WhiteBalance::from_kelvin(5200.0);
        assert!(approx(wb.tint, 0.0));
        assert!(approx(wb.temp_kelvin, 5200.0));
    }

    #[test]
    fn default_is_neutral() {
        assert_eq!(WhiteBalance::default(), WhiteBalance::NEUTRAL);
    }

    #[test]
    fn std430_roundtrips_the_gain() {
        let wb = WhiteBalance::new(3500.0, 0.3);
        let gain = wb.rgb_gain();
        let bytes = wb.to_std430();
        assert_eq!(bytes.len(), VEC4_STRIDE);
        assert!(approx(read_le_f32(&bytes, 0), gain[0]));
        assert!(approx(read_le_f32(&bytes, 4), gain[1]));
        assert!(approx(read_le_f32(&bytes, 8), gain[2]));
        // The trailing padding slot is zero.
        assert!(approx(read_le_f32(&bytes, 12), 0.0));
    }

    #[test]
    fn std430_size_is_one_vec4() {
        let bytes = WhiteBalance::NEUTRAL.to_std430();
        assert_eq!(bytes.len(), VEC4_STRIDE);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(WhiteBalance::gpu_storage_bytes(0), VEC4_STRIDE);
        assert_eq!(WhiteBalance::gpu_storage_bytes(1), VEC4_STRIDE);
        assert_eq!(WhiteBalance::gpu_storage_bytes(8), 8 * VEC4_STRIDE);
    }

    #[test]
    fn gains_are_finite_and_non_negative_across_the_sweep() {
        let mut k = KELVIN_MIN;
        while k <= KELVIN_MAX {
            for &t in &[TINT_MIN, 0.0_f32, TINT_MAX] {
                let gain = WhiteBalance::new(k, t).rgb_gain();
                for &g in &gain {
                    assert!(g.is_finite());
                    assert!(g >= 0.0);
                }
            }
            k += 100.0;
        }
    }
}
