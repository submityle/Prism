//! A scrambled Halton low-discrepancy sampler for primary-ray dimensions.
//!
//! The film driver antialiases each pixel by averaging many jittered primary
//! rays. Drawing that sub-pixel jitter from an independent pseudorandom source
//! converges at the slow Monte-Carlo rate because the points clump and leave
//! gaps. A low-discrepancy (quasi-Monte-Carlo, `QMC`) sequence instead spreads
//! the points as evenly as possible, so primary-visibility edges resolve with
//! far fewer samples.
//!
//! This module provides the classic Halton construction: the two sub-pixel
//! dimensions use the radical inverse in the coprime bases 2 and 3. To keep
//! neighbouring pixels from sharing an identical pattern (which would reveal
//! structured aliasing), each pixel applies a Cranley–Patterson (`CP`) rotation
//! drawn from a dedicated `PCG` stream keyed by the render seed and the pixel
//! index. The rotation preserves the per-pixel low-discrepancy stratification
//! while decorrelating the image globally, and the whole construction is
//! deterministic: identical arguments reproduce a bit-identical sample set.
//!
//! Only integer digit arithmetic, division, and `floor` are used, honouring the
//! crate's no-transcendentals determinism policy (no `sin`/`cos`/`exp`/`ln`).

use super::sampler::{Rng, Sample2};

/// The radical-inverse base for the pixel `x` jitter (the first prime).
const BASE_X: u32 = 2;
/// The radical-inverse base for the pixel `y` jitter (the second prime).
const BASE_Y: u32 = 3;
/// A golden-ratio-derived salt mixed into the render seed so the sampler's
/// decorrelation streams never coincide with the integrator's per-pixel path
/// streams (which are keyed by the bare seed).
const STREAM_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The radical inverse of `index` in `base`: reflect the base-`b` digits of
/// `index` about the radix point to land in `[0, 1)`.
///
/// The reflected fraction is accumulated in `f64` so it stays accurate for
/// large sample indices, then narrowed to `f32`. Uses only integer digit
/// extraction and division, so it introduces no transcendental functions.
#[must_use]
fn radical_inverse(base: u32, mut index: u64) -> f32 {
    let base64 = u64::from(base);
    let inv_base = 1.0 / f64::from(base);
    let mut inv_bn = inv_base;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = (index % base64) as f64;
        result += digit * inv_bn;
        inv_bn *= inv_base;
        index /= base64;
    }
    result as f32
}

/// The fractional part of `x`, wrapped into `[0, 1)`. Used to apply a
/// Cranley–Patterson rotation without a modulo on floats.
#[must_use]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// A deterministic, per-pixel scrambled Halton sampler for the two sub-pixel
/// jitter dimensions.
///
/// Construct one sampler per pixel; it owns that pixel's Cranley–Patterson
/// rotation and reproduces the same stratified `[0, 1)^2` points for a given
/// `(seed, pixel_index)` pair.
#[derive(Clone, Copy, Debug)]
pub struct HaltonPixelSampler {
    /// Rotation offset added to the base-2 (`x`) radical inverse before wrap.
    offset_x: f32,
    /// Rotation offset added to the base-3 (`y`) radical inverse before wrap.
    offset_y: f32,
}

impl HaltonPixelSampler {
    /// Builds the sampler for the pixel at flat `pixel_index`, deriving its
    /// decorrelating rotation from a `PCG` stream keyed by `seed` (salted so it
    /// cannot collide with the integrator's per-pixel path streams) and the
    /// pixel index.
    #[must_use]
    pub fn new(seed: u64, pixel_index: u64) -> Self {
        let mut rng = Rng::with_stream(seed ^ STREAM_SALT, pixel_index.wrapping_add(1));
        let offset_x = rng.next_f32();
        let offset_y = rng.next_f32();
        Self { offset_x, offset_y }
    }

    /// The sub-pixel jitter for the `sample_index`-th sample, in `[0, 1)^2`.
    ///
    /// Sample indices start at zero; the radical inverse is evaluated at
    /// `sample_index + 1` so the degenerate zeroth Halton point `(0, 0)` is
    /// skipped and the rotation is the only thing anchoring the first sample.
    #[must_use]
    pub fn sample(&self, sample_index: u64) -> Sample2 {
        let index = sample_index.wrapping_add(1);
        Sample2 {
            x: fract(radical_inverse(BASE_X, index) + self.offset_x),
            y: fract(radical_inverse(BASE_Y, index) + self.offset_y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_base_2_reflects_bits() {
        // Base-2 digit reflection: 1 -> .1, 2 -> .01, 3 -> .11, 4 -> .001.
        assert!((radical_inverse(2, 1) - 0.5).abs() < 1e-7);
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-7);
        assert!((radical_inverse(2, 3) - 0.75).abs() < 1e-7);
        assert!((radical_inverse(2, 4) - 0.125).abs() < 1e-7);
        assert!(radical_inverse(2, 0).abs() < 1e-7);
    }

    #[test]
    fn radical_inverse_base_3_reflects_trits() {
        // Base-3 digit reflection: 1 -> .1 (1/3), 3 -> .01 (1/9), 4 -> .11.
        assert!((radical_inverse(3, 1) - 1.0 / 3.0).abs() < 1e-6);
        assert!((radical_inverse(3, 3) - 1.0 / 9.0).abs() < 1e-6);
        assert!((radical_inverse(3, 4) - (1.0 / 3.0 + 1.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn halton_2d_fills_every_stratum() {
        // The (2, 3) Halton points over indices 0..6 form a stratified net:
        // each of the 2x3 = 6 cells (x split in halves, y split in thirds) is
        // visited exactly once, which pseudorandom jitter cannot guarantee.
        let mut seen = [false; 6];
        for i in 0..6u64 {
            let x = radical_inverse(2, i);
            let y = radical_inverse(3, i);
            let cx = (x * 2.0) as usize; // 0 or 1
            let cy = (y * 3.0) as usize; // 0, 1, or 2
            let cell = cy * 2 + cx;
            assert!(!seen[cell], "cell {cell} hit twice");
            seen[cell] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "every stratum must be covered");
    }

    #[test]
    fn samples_stay_in_unit_square() {
        let sampler = HaltonPixelSampler::new(7, 123);
        for s in 0..4096u64 {
            let p = sampler.sample(s);
            assert!(p.x >= 0.0 && p.x < 1.0, "x out of range: {}", p.x);
            assert!(p.y >= 0.0 && p.y < 1.0, "y out of range: {}", p.y);
        }
    }

    #[test]
    fn sampler_is_deterministic() {
        let a = HaltonPixelSampler::new(42, 999);
        let b = HaltonPixelSampler::new(42, 999);
        for s in 0..256u64 {
            let pa = a.sample(s);
            let pb = b.sample(s);
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
        }
    }

    #[test]
    fn neighbouring_pixels_are_decorrelated() {
        // Distinct pixels receive distinct rotations, so their first samples
        // differ even though they share the same base sequence.
        let p0 = HaltonPixelSampler::new(1, 0).sample(0);
        let p1 = HaltonPixelSampler::new(1, 1).sample(0);
        let delta = (p0.x - p1.x).abs() + (p0.y - p1.y).abs();
        assert!(
            delta > 1e-4,
            "neighbouring pixels should not share a pattern"
        );
    }

    #[test]
    fn rotated_block_is_better_stratified_than_random() {
        // Averaged over pixels, a Halton block covers the unit square more
        // evenly than independent jitter: measure the worst-case deviation from
        // the expected count across a 4x4 grid for a modest sample budget.
        let grid = 4usize;
        let samples = 64u64;
        let expected = (samples as f32) / ((grid * grid) as f32);

        let worst = |use_halton: bool| -> f32 {
            let mut total_worst = 0.0_f32;
            let pixels = 16u64;
            for pixel in 0..pixels {
                let mut counts = [0u32; 16];
                let halton = HaltonPixelSampler::new(5, pixel);
                let mut rng = Rng::with_stream(5, pixel + 1);
                for s in 0..samples {
                    let p = if use_halton {
                        halton.sample(s)
                    } else {
                        Sample2 {
                            x: rng.next_f32(),
                            y: rng.next_f32(),
                        }
                    };
                    let cx = ((p.x * grid as f32) as usize).min(grid - 1);
                    let cy = ((p.y * grid as f32) as usize).min(grid - 1);
                    counts[cy * grid + cx] += 1;
                }
                let mut pixel_worst = 0.0_f32;
                for c in counts {
                    pixel_worst = pixel_worst.max((c as f32 - expected).abs());
                }
                total_worst += pixel_worst;
            }
            total_worst / (pixels as f32)
        };

        let halton_worst = worst(true);
        let random_worst = worst(false);
        assert!(
            halton_worst < random_worst,
            "halton deviation {halton_worst} should beat random {random_worst}"
        );
    }
}
