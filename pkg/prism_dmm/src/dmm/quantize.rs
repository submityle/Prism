//! 11-bit unorm quantization and the per-triangle displacement scale/bias.
//!
//! A `DMM` stores each micro-vertex displacement as an `11-bit` unsigned
//! normalized (`unorm`) code in `0..=2047`. Authored heights rarely live in
//! `[0, 1]`, so a per-triangle [`DisplacementScaleBias`] first maps the
//! triangle's own `[min, max]` height range onto `[0, 1]`; the normalized
//! value is then quantized to `11-bit` by round-to-nearest.

/// The largest representable `11-bit` unorm code.
pub const UNORM11_MAX: u16 = 2047;

/// Quantizes a value in `[0, 1]` to an `11-bit` unorm code in `0..=2047`.
///
/// The input is clamped to `[0, 1]` and rounded to the nearest code, so
/// `0.0` maps to `0` and `1.0` maps to `2047`.
#[must_use]
pub fn quantize_unorm11(value: f32) -> u16 {
    let clamped = value.clamp(0.0, 1.0);
    let scaled = libm::roundf(clamped * UNORM11_MAX as f32);
    scaled.clamp(0.0, UNORM11_MAX as f32) as u16
}

/// Dequantizes an `11-bit` unorm code back into `[0, 1]`.
///
/// Codes above [`UNORM11_MAX`] are clamped so the result never exceeds `1.0`.
#[must_use]
pub fn dequantize_unorm11(code: u16) -> f32 {
    let clamped = code.min(UNORM11_MAX);
    f32::from(clamped) / UNORM11_MAX as f32
}

/// Maps an arbitrary per-triangle height range `[min, max]` onto `[0, 1]`.
///
/// Storing the scale/bias per triangle lets each triangle use the full
/// `11-bit` precision over just its own displacement extent. A degenerate
/// range (`max == min`, i.e. a flat triangle) is guarded: [`normalize`] then
/// returns `0.0` for every height so the whole triangle bakes to code `0`.
///
/// [`normalize`]: DisplacementScaleBias::normalize
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplacementScaleBias {
    /// Lowest height in the triangle, mapped to normalized `0.0`.
    pub min: f32,
    /// Highest height in the triangle, mapped to normalized `1.0`.
    pub max: f32,
}

impl DisplacementScaleBias {
    /// Creates a scale/bias from an explicit `[min, max]` height range.
    #[must_use]
    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Returns `true` when the range is degenerate (`max == min`).
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.max == self.min
    }

    /// Maps a height into normalized `[0, 1]`, clamped to that range.
    ///
    /// Returns `0.0` for a degenerate range to avoid a division by zero.
    #[must_use]
    pub fn normalize(&self, height: f32) -> f32 {
        if self.is_degenerate() {
            return 0.0;
        }
        ((height - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Maps a normalized value in `[0, 1]` back into the height range.
    #[must_use]
    pub fn denormalize(&self, t: f32) -> f32 {
        self.min + t * (self.max - self.min)
    }

    /// Normalizes then quantizes a height into an `11-bit` unorm code.
    #[must_use]
    pub fn quantize(&self, height: f32) -> u16 {
        quantize_unorm11(self.normalize(height))
    }

    /// Dequantizes an `11-bit` unorm code back into a height.
    #[must_use]
    pub fn dequantize(&self, code: u16) -> f32 {
        self.denormalize(dequantize_unorm11(code))
    }
}
