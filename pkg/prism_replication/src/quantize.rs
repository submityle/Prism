//! Uniform scalar quantization (design §15 "定点量化").
//!
//! A replicated transform or animation phase travels as an absolute `f32` (32
//! bits) by default, carrying far more precision than presentation needs. A
//! [`Quantizer`] maps an `f32` in a fixed `[min, max]` range onto a `bits`-bit
//! unsigned integer and back, so a delta ships 10–16 bits instead of 32 and the
//! receiver knows the worst-case error up front. Quantization is *lossy on
//! purpose*; the caller chooses the range and width to bound the error
//! ([`Quantizer::max_error`]).

/// A uniform scalar quantizer over `[min, max]` with `bits` of precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantizer {
    min: f32,
    max: f32,
    /// Codes per unit range (`levels / (max - min)`); its reciprocal is the step.
    inv_step: f32,
    levels: u64,
    bits: u8,
}

impl Quantizer {
    /// A quantizer mapping `[min, max]` onto `0..=2^bits - 1`.
    ///
    /// # Panics
    /// Panics if `min`/`max` are not finite, if `min >= max`, or if `bits` is 0
    /// or greater than 24. 24 bits is ~0.06 ppm of the range — already past any
    /// presentation use — and capping keeps `2^bits` and the reciprocal step
    /// exactly representable in `f32`, so the round-trip endpoints are exact.
    #[must_use]
    pub fn new(min: f32, max: f32, bits: u8) -> Self {
        assert!(
            min.is_finite() && max.is_finite() && min < max,
            "quantizer range must be finite and increasing"
        );
        assert!((1..=24).contains(&bits), "quantizer bits must be 1..=24");
        let levels = (1u64 << bits) - 1;
        Self {
            min,
            max,
            inv_step: levels as f32 / (max - min),
            levels,
            bits,
        }
    }

    /// The range's lower bound.
    #[must_use]
    pub const fn min(self) -> f32 {
        self.min
    }

    /// The range's upper bound.
    #[must_use]
    pub const fn max(self) -> f32 {
        self.max
    }

    /// The code width in bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// The highest code (`2^bits - 1`).
    #[must_use]
    pub const fn levels(self) -> u64 {
        self.levels
    }

    /// Quantize `value`, clamping it into `[min, max]` first. The result is in
    /// `0..=2^bits - 1`.
    #[must_use]
    pub fn quantize(self, value: f32) -> u64 {
        // `NaN` maps to code 0 (the saturating cast). Round half up without
        // `f32::round` (std-only): add 0.5 and let the truncating cast do it.
        let clamped = value.clamp(self.min, self.max);
        // Pin the endpoints: `inv_step` is a rounded `f32`, so for an extreme
        // range the computed top code can land one short of `levels`; forcing it
        // keeps both ends exact and the interior error within `max_error`.
        if clamped >= self.max {
            return self.levels;
        }
        if clamped <= self.min {
            return 0;
        }
        let code = ((clamped - self.min) * self.inv_step + 0.5) as u64;
        code.min(self.levels)
    }

    /// Dequantize a code back to an `f32` in `[min, max]` (codes above the max
    /// are clamped).
    #[must_use]
    pub fn dequantize(self, code: u64) -> f32 {
        let code = code.min(self.levels);
        self.min + (code as f32) / self.inv_step
    }

    /// The worst-case absolute error of one quantization step (`step / 2`).
    #[must_use]
    pub fn max_error(self) -> f32 {
        0.5 / self.inv_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn range_endpoints_round_trip_exactly() {
        let q = Quantizer::new(-1.0, 1.0, 16);
        assert_eq!(q.quantize(-1.0), 0);
        assert_eq!(q.quantize(1.0), q.levels());
        assert!((q.dequantize(0) - -1.0).abs() < EPSILON);
        assert!((q.dequantize(q.levels()) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn values_within_range_are_within_the_error_bound() {
        let q = Quantizer::new(0.0, 100.0, 12);
        let bound = q.max_error();
        for i in 0..=1000 {
            let value = i as f32 * 0.1;
            let round_tripped = q.dequantize(q.quantize(value));
            assert!(
                (round_tripped - value).abs() <= bound + EPSILON,
                "value {value} round-tripped to {round_tripped} (bound {bound})"
            );
        }
    }

    #[test]
    fn out_of_range_values_clamp_to_the_endpoints() {
        let q = Quantizer::new(10.0, 20.0, 8);
        assert_eq!(q.quantize(0.0), 0);
        assert_eq!(q.quantize(1_000.0), q.levels());
        // An over-wide code dequantizes back into range.
        assert!(q.dequantize(u64::MAX) <= q.max());
    }

    #[test]
    fn codes_are_monotone_in_the_value() {
        let q = Quantizer::new(-5.0, 5.0, 10);
        let mut previous = 0;
        for i in 0..=1000 {
            let code = q.quantize(-5.0 + i as f32 * 0.01);
            assert!(code >= previous, "codes must not decrease");
            previous = code;
        }
    }

    #[test]
    fn one_bit_is_two_levels() {
        let q = Quantizer::new(0.0, 1.0, 1);
        assert_eq!(q.quantize(0.1), 0);
        assert_eq!(q.quantize(0.9), 1);
        assert!((q.max_error() - 0.5).abs() < EPSILON);
    }

    /// Both endpoints pin exactly to the top/bottom code across benign and
    /// extreme ranges (regression: the top code could land one short).
    #[test]
    fn endpoints_are_exact_across_ranges() {
        for (min, max, bits) in [
            (-1.0_f32, 1.0_f32, 16),
            (0.0, 100.0, 12),
            (-1.0e6, 3.5, 24),
            (0.0, 0.001, 8),
        ] {
            let q = Quantizer::new(min, max, bits);
            assert_eq!(q.quantize(min), 0, "min of {min}..{max}");
            assert_eq!(q.quantize(max), q.levels(), "max of {min}..{max}");
        }
    }

}
