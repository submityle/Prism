//! An Owen-scrambled Sobol low-discrepancy sampler for primary-ray dimensions.
//!
//! The scrambled Halton sampler in [`super::halton`] spreads sub-pixel jitter
//! more evenly than independent jitter, but at higher sample counts the Sobol
//! (0, 2)-sequence is the stronger quasi-Monte-Carlo (`QMC`) construction: its
//! first two dimensions form a base-2 (0, 2)-net, so every power-of-two prefix
//! of the sequence places exactly one point in each dyadic elementary interval.
//! Owen scrambling then randomises that net per pixel without destroying the
//! stratification, which both decorrelates neighbouring pixels (removing
//! structured aliasing) and provably lowers the integration error below the
//! plain sequence. This is the sampler modern production path tracers reach for.
//!
//! The net is generated from the two classic direction-number recurrences:
//! dimension zero is the base-2 radical inverse (bit reversal of the index) and
//! dimension one is the Antonov-Saleev / Sobol second dimension (the `v ^= v >>
//! 1` Gray-code recurrence). Owen scrambling uses the hash-based nested-uniform
//! scramble (Burley's Laine-Karras permutation), seeded per pixel and per
//! dimension from a dedicated `PCG` stream so it never collides with the Halton
//! sampler's or the integrator's streams.
//!
//! Only integer bit arithmetic (`reverse_bits`, shifts, xor, wrapping add and
//! multiply) and a final fixed-point-to-float divide are used, honouring the
//! crate's no-transcendentals determinism policy (no `sin`/`cos`/`exp`/`ln`):
//! identical arguments reproduce a bit-identical sample set.

use super::sampler::{Rng, Sample2};

/// A golden-ratio-derived salt mixed into the render seed so the Sobol
/// sampler's scramble streams never coincide with the Halton sampler's rotation
/// streams or the integrator's per-pixel path streams (both keyed differently).
const SOBOL_STREAM_SALT: u64 = 0xA076_1D64_78BD_642F;

/// Reciprocal of `2^24`, scaling the top 24 fixed-point bits into `[0, 1)` with
/// an exactly representable `f32` mantissa.
const INV_2_POW_24: f32 = 1.0 / 16_777_216.0;

/// Dimension-zero Sobol direction: the base-2 radical inverse of `index`,
/// returned as a 32-bit fixed-point fraction whose most significant bit is the
/// first fractional bit. Bit reversal is the closed form of that radical
/// inverse for base two.
#[must_use]
fn sobol_dimension_zero(index: u32) -> u32 {
    index.reverse_bits()
}

/// Dimension-one Sobol direction as a 32-bit fixed-point fraction.
///
/// Accumulates the Sobol direction vectors for the set bits of `index` using
/// the Antonov-Saleev Gray-code recurrence `v ^= v >> 1`, which generates the
/// Pascal-matrix (binomial-coefficient-modulo-two) direction numbers of the
/// classic Sobol second dimension. Together with [`sobol_dimension_zero`] this
/// yields a base-2 (0, 2)-net.
#[must_use]
fn sobol_dimension_one(mut index: u32) -> u32 {
    let mut result: u32 = 0;
    let mut direction: u32 = 1u32 << 31;
    while index != 0 {
        if index & 1 == 1 {
            result ^= direction;
        }
        index >>= 1;
        direction ^= direction >> 1;
    }
    result
}

/// A hash-based nested-uniform Owen scramble of a 32-bit fixed-point fraction.
///
/// Reverses the fraction so its least significant tree level is in the low
/// bits, applies Burley's Laine-Karras permutation (a bijection that flips each
/// binary tree node independently as a function of `seed` and the bits above
/// it), then reverses back. This realises an Owen scramble that preserves the
/// `(t, m, s)`-net stratification while randomising the point set.
#[must_use]
fn owen_scramble(value: u32, seed: u32) -> u32 {
    let mut bits = value.reverse_bits();
    bits = bits.wrapping_add(seed);
    bits ^= bits.wrapping_mul(0x6c50_b47c);
    bits ^= bits.wrapping_mul(0xb82f_1e52);
    bits ^= bits.wrapping_mul(0xc7af_e638);
    bits ^= bits.wrapping_mul(0x8d22_f6e6);
    bits.reverse_bits()
}

/// Maps a 32-bit fixed-point fraction into `[0, 1)` by taking its top 24 bits
/// as an `f32` mantissa (so the largest input maps strictly below one).
#[must_use]
fn to_unit_interval(bits: u32) -> f32 {
    ((bits >> 8) as f32) * INV_2_POW_24
}

/// A deterministic, per-pixel Owen-scrambled Sobol sampler for the two
/// sub-pixel jitter dimensions.
///
/// Construct one sampler per pixel; it owns that pixel's two Owen scramble seeds
/// and reproduces the same stratified `[0, 1)^2` net for a given `(seed,
/// pixel_index)` pair.
#[derive(Clone, Copy, Debug)]
pub struct OwenScrambledSobolSampler {
    /// Owen scramble seed for the `x` (dimension-zero) coordinate.
    scramble_x: u32,
    /// Owen scramble seed for the `y` (dimension-one) coordinate.
    scramble_y: u32,
}

impl OwenScrambledSobolSampler {
    /// Builds the sampler for the pixel at flat `pixel_index`, deriving its two
    /// Owen scramble seeds from a `PCG` stream keyed by `seed` (salted so it
    /// cannot collide with the Halton or integrator streams) and the index.
    #[must_use]
    pub fn new(seed: u64, pixel_index: u64) -> Self {
        let mut rng = Rng::with_stream(seed ^ SOBOL_STREAM_SALT, pixel_index.wrapping_add(1));
        let scramble_x = rng.next_u32();
        let scramble_y = rng.next_u32();
        Self {
            scramble_x,
            scramble_y,
        }
    }

    /// The sub-pixel jitter for the `sample_index`-th sample, in `[0, 1)^2`.
    ///
    /// The 64-bit sample index is reduced to the low 32 bits that drive the
    /// Sobol direction numbers, which is ample for any realistic per-pixel
    /// sample budget.
    #[must_use]
    pub fn sample(&self, sample_index: u64) -> Sample2 {
        let index = (sample_index & 0xFFFF_FFFF) as u32;
        let x = owen_scramble(sobol_dimension_zero(index), self.scramble_x);
        let y = owen_scramble(sobol_dimension_one(index), self.scramble_y);
        Sample2 {
            x: to_unit_interval(x),
            y: to_unit_interval(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radical_inverse_dimension_zero_reflects_bits() {
        // Fixed-point bit reversal: 1 -> .1 (top bit), 2 -> .01, 3 -> .11.
        assert_eq!(sobol_dimension_zero(1), 1u32 << 31);
        assert_eq!(sobol_dimension_zero(2), 1u32 << 30);
        assert_eq!(sobol_dimension_zero(3), (1u32 << 31) | (1u32 << 30));
        assert_eq!(sobol_dimension_zero(0), 0);
    }

    #[test]
    fn unscrambled_net_fills_every_dyadic_cell() {
        // The first 16 points of the (0, 2)-sequence form a (0, 4)-net: each
        // dyadic elementary interval of volume 1/16 holds exactly one point.
        // Check the three balanced splits a 16-point net must satisfy.
        for (nx, ny) in [(4usize, 4usize), (2, 8), (8, 2)] {
            let mut seen = alloc::vec![false; nx * ny];
            for i in 0..16u32 {
                let x = to_unit_interval(sobol_dimension_zero(i));
                let y = to_unit_interval(sobol_dimension_one(i));
                let cx = ((x * nx as f32) as usize).min(nx - 1);
                let cy = ((y * ny as f32) as usize).min(ny - 1);
                let cell = cy * nx + cx;
                assert!(!seen[cell], "net cell {cell} hit twice for split {nx}x{ny}");
                seen[cell] = true;
            }
            assert!(
                seen.iter().all(|hit| *hit),
                "every {nx}x{ny} dyadic cell must hold exactly one net point"
            );
        }
    }

    #[test]
    fn samples_stay_in_unit_square() {
        let sampler = OwenScrambledSobolSampler::new(7, 123);
        for s in 0..4096u64 {
            let p = sampler.sample(s);
            assert!(p.x >= 0.0 && p.x < 1.0, "x out of range: {}", p.x);
            assert!(p.y >= 0.0 && p.y < 1.0, "y out of range: {}", p.y);
        }
    }

    #[test]
    fn sampler_is_deterministic() {
        let a = OwenScrambledSobolSampler::new(42, 999);
        let b = OwenScrambledSobolSampler::new(42, 999);
        for s in 0..256u64 {
            let pa = a.sample(s);
            let pb = b.sample(s);
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
        }
    }

    #[test]
    fn neighbouring_pixels_are_decorrelated() {
        // Distinct pixels receive distinct scramble seeds, so their sample sets
        // differ even though they share the same underlying net.
        let p0 = OwenScrambledSobolSampler::new(1, 0).sample(1);
        let p1 = OwenScrambledSobolSampler::new(1, 1).sample(1);
        let delta = (p0.x - p1.x).abs() + (p0.y - p1.y).abs();
        assert!(
            delta > 1e-4,
            "neighbouring pixels should not share a scrambled pattern"
        );
    }

    #[test]
    fn owen_scramble_is_a_bijection_on_a_full_level() {
        // A valid Owen scramble permutes the 2^m dyadic intervals: scrambling
        // every 4-bit prefix must reproduce all 16 distinct top-nibble buckets.
        let mut seen = [false; 16];
        for i in 0..16u32 {
            let scrambled = owen_scramble(i << 28, 0x1234_5678);
            let bucket = (scrambled >> 28) as usize;
            assert!(!seen[bucket], "bucket {bucket} produced twice");
            seen[bucket] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "scramble must be a bijection");
    }

    #[test]
    fn scrambled_block_is_better_stratified_than_random() {
        // Averaged over pixels, an Owen-scrambled Sobol block covers the unit
        // square more evenly than independent jitter: compare the worst-case
        // deviation from the expected per-cell count across a 4x4 grid.
        let grid = 4usize;
        let samples = 64u64;
        let expected = (samples as f32) / ((grid * grid) as f32);

        let worst = |use_sobol: bool| -> f32 {
            let pixels = 16u64;
            let mut total_worst = 0.0_f32;
            for pixel in 0..pixels {
                let mut counts = [0u32; 16];
                let sobol = OwenScrambledSobolSampler::new(5, pixel);
                let mut rng = Rng::with_stream(5, pixel + 1);
                for s in 0..samples {
                    let p = if use_sobol {
                        sobol.sample(s)
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

        let sobol_worst = worst(true);
        let random_worst = worst(false);
        assert!(
            sobol_worst < random_worst,
            "sobol deviation {sobol_worst} should beat random {random_worst}"
        );
    }
}
