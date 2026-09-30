//! Stochastic *hashed alpha testing* for particle shading (design §17).
//!
//! Fixed-threshold alpha testing (`alpha < 0.5 -> discard`) produces a hard,
//! aliased silhouette that crawls under motion and cannot cooperate with
//! `MSAA` coverage or a `TAA` history buffer. This module owns the `CPU`
//! reference for the alternative popularized by `Wyman` & `McGuire`'s 2017 "Hashed
//! Alpha Testing": instead of one shared cutoff, every fragment draws a
//! *per-location* threshold from a stable spatial hash and keeps the fragment
//! when `alpha >= threshold`. Because the threshold is uniformly distributed in
//! `[0, 1)`, the fraction of surviving fragments equals the fragment `alpha`,
//! so the surface reads as stochastically transparent. The randomness is
//! anchored to object/world coordinates (not screen pixels), so it stays put on
//! the surface and resolves cleanly through `MSAA` sample masks and `TAA`
//! temporal accumulation rather than shimmering.
//!
//! The pipeline is: (1) [`hash3`] turns quantized integer lattice coordinates
//! into a deterministic threshold in `[0, 1)` with a self-contained integer
//! avalanche mixer; (2) [`alpha_hash_threshold`] anchors that hash to a
//! world/screen scale, quantizing the anchor at two adjacent power-of-two
//! `LOD` levels and blending the two hashes with a linear factor so the noise
//! does not pop as the surface changes size; and (3) [`hashed_alpha_test`]
//! performs the `alpha >= threshold` keep/discard decision. [`coverage_from_alpha`]
//! reports the expected surviving coverage for an alpha, matching what an
//! `MSAA`-aware resolve should converge to.
//!
//! **Anisotropic `LOD` stabilization.** As a sprite shrinks, one hashed lattice
//! cell eventually covers many pixels, which would make the threshold flicker
//! frame to frame. [`alpha_hash_threshold`] defends against that by discretizing
//! the anisotropic derivative scale into integer `LOD` levels (pure `floor` +
//! integer arithmetic) and linearly interpolating the hash sampled at the two
//! bracketing levels. Sub-cell camera jitter therefore leaves the threshold
//! unchanged, and growth across a level boundary is continuous rather than a
//! jump. The `log2` used to pick the level is read straight from the `f32`
//! exponent bits, and `2^level` is rebuilt by writing the exponent field, so no
//! transcendental (`powf`/`exp`/`sin`) ever runs.
//!
//! **Deliberately out of scope.** This module *only* produces a hash threshold
//! and the keep/discard test plus its `LOD` stabilization. It is intentionally
//! not an order-independent-transparency path: weighted-blended accumulation
//! and its weight curve live in [`super::oit`], and this file never touches
//! them. It is also not a *dissolve*: life-driven noise erosion with an
//! emissive rim is [`super::alpha_erosion`]'s job. And it does not fade
//! particles against scene depth — that soft-particle contribution belongs to
//! its own sibling. Hashed alpha testing is a hard, stochastic keep/discard;
//! keeping those concerns separate is what lets each stay a small, verifiable
//! contract. The only sibling dependency is the shared `std430` layout in
//! [`super::gpu_layout`]; the hash itself is entirely self-contained integer
//! math.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Comparison epsilon guarding every `f32` denominator and equality-style
/// check in this module, so no `==`/`!=` on floats is ever needed and no guard
/// can divide by zero into a `NaN`.
const CMP_EPS: f32 = 1e-6;

/// Reciprocal of `2^24`, normalizing a 24-bit hash mantissa into `[0, 1)`.
///
/// A 24-bit numerator times this constant is strictly below `1.0`, so a hashed
/// threshold can never reach the `alpha == 1.0` opaque case and an opaque
/// fragment always survives (see [`hashed_alpha_test`]).
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// Byte stride of one packed [`AlphaHashParams`] record in a `std430` storage
/// buffer: four scalars in a single `vec4` slot.
pub const ALPHA_HASH_PARAMS_STRIDE: usize = VEC4_STRIDE;

/// Byte stride of one precomputed per-particle threshold entry (a scalar `f32`
/// `LUT` slot) in a `std430` storage buffer.
pub const THRESHOLD_STRIDE: usize = U32_STRIDE;

/// Clamps a scalar into `0..=1` without branching on float equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Fractional part of `x` in `[0, 1)`, defined by `x - floor(x)` so it stays
/// well-behaved for negative inputs (e.g. `frac(-1.25) == 0.75`).
#[must_use]
fn frac(x: f32) -> f32 {
    x - x.floor()
}

/// Linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// One xor-shift / odd-multiply avalanche round mixing `value` into the hash
/// accumulator `h`.
#[must_use]
fn mix(h: u32, value: u32) -> u32 {
    let mut x = h ^ value.wrapping_mul(0x9e37_79b9);
    x ^= x >> 15;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

/// Deterministic 3D spatial hash of integer lattice coordinates, in `[0, 1)`.
///
/// Folds the three signed lattice indices (via a two's-complement cast, so the
/// whole signed lattice is addressable) and the `seed` through the [`mix`]
/// avalanche, then normalizes the top 24 bits with [`INV_2POW24`]. It is a pure
/// function with no transcendental call, so the `CPU` reference and a future
/// `GPU` kernel agree bit for bit, and equal inputs always yield equal output.
#[must_use]
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = seed ^ 0x811c_9dc5;
    h = mix(h, x as u32);
    h = mix(h, y as u32);
    h = mix(h, z as u32);
    let mantissa = h >> 8;
    (mantissa as f32) * INV_2POW24
}

/// Rebuilds `2^level` as an `f32` by writing the biased exponent field.
///
/// This replaces `exp2`/`powf`: for an integer `level` the result is an exact
/// power of two, produced purely by integer bit assembly. `level` is clamped to
/// the normal-`f32` exponent range so the shift can never construct a subnormal
/// or an infinity.
#[must_use]
fn exp2_pow(level: i32) -> f32 {
    let clamped = level.clamp(-126, 127);
    let biased = (clamped + 127) as u32;
    f32::from_bits(biased << 23)
}

/// Piecewise-linear `log2(x)` read from the `f32` exponent and mantissa bits.
///
/// For `x > 0` the exponent field gives the integer octave and the mantissa in
/// `[1, 2)` supplies a linear fraction inside it. The approximation is exact at
/// every power of two and strictly increasing everywhere, which is all the
/// `LOD` level selection in [`alpha_hash_threshold`] needs — and it uses no
/// transcendental. Non-positive inputs are floored to [`CMP_EPS`] by the caller.
#[must_use]
fn approx_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let mantissa = f32::from_bits(mantissa_bits);
    (exponent as f32) + (mantissa - 1.0)
}

/// Quantizes a world/screen anchor component to an integer lattice index at a
/// given `scale` via `floor(coord * scale)`.
#[must_use]
fn quantize_coord(coord: f32, scale: f32) -> i32 {
    (coord * scale).floor() as i32
}

/// The two-level hash blend: linear interpolation of the coarse-level hash `h0`
/// and the fine-level hash `h1` by the octave fraction `f`.
///
/// Kept as a named helper so the stabilization is monotonic in `f` by
/// construction (it is a plain `lerp`) and can be tested in isolation.
#[must_use]
fn two_level_lerp(h0: f32, h1: f32, f: f32) -> f32 {
    lerp(h0, h1, f)
}

/// Parameters controlling a renderer's hashed alpha test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaHashParams {
    /// Spatial granularity of the hash noise. Larger values shrink the hashed
    /// lattice cells (finer stochastic dither); it scales the anchor before
    /// quantization in [`alpha_hash_threshold`].
    pub hash_scale: f32,
    /// Lower clamp applied to the produced threshold, keeping it strictly above
    /// `0.0` so a fully transparent fragment (`alpha == 0`) is always discarded.
    pub min_threshold: f32,
    /// Seed decorrelating one renderer's dither pattern from another's.
    pub seed: u32,
}

impl AlphaHashParams {
    /// Creates hashed-alpha parameters from all fields.
    #[must_use]
    pub const fn new(hash_scale: f32, min_threshold: f32, seed: u32) -> Self {
        Self {
            hash_scale,
            min_threshold,
            seed,
        }
    }

    /// The effective hash scale, floored to [`CMP_EPS`] so a zero or negative
    /// authored value can never collapse the lattice or divide by zero.
    #[must_use]
    fn effective_scale(&self) -> f32 {
        self.hash_scale.max(CMP_EPS)
    }

    /// The threshold floor, clamped into `0..=1`.
    #[must_use]
    fn threshold_floor(&self) -> f32 {
        clamp01(self.min_threshold)
    }

    /// Packs the parameters into their `std430` `vec4`-aligned scalar layout.
    ///
    /// Layout: `[hash_scale_bits, min_threshold_bits, seed, pad]` — one `vec4`
    /// slot matching [`ALPHA_HASH_PARAMS_STRIDE`]. Float fields are emitted as
    /// their `f32::to_bits` patterns and `seed` as its raw `u32`, so the block
    /// round-trips exactly with no lossy cast. The trailing `pad` is zero.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        [
            self.hash_scale.to_bits(),
            self.min_threshold.to_bits(),
            self.seed,
            0,
        ]
    }
}

/// Anisotropic `LOD` scale from a pair of screen-space anchor derivatives.
///
/// Returns `max(len(ddx), len(ddy))`, the larger footprint of the anchor across
/// the two pixel axes. Feeding this into [`alpha_hash_threshold`] as `lod_scale`
/// makes the level selection track the anisotropic footprint, matching the
/// derivative-driven scale in the reference algorithm. Uses only `f32::sqrt`.
#[must_use]
pub fn anisotropic_lod_scale(ddx: [f32; 3], ddy: [f32; 3]) -> f32 {
    let len_x = (ddx[0] * ddx[0] + ddx[1] * ddx[1] + ddx[2] * ddx[2]).sqrt();
    let len_y = (ddy[0] * ddy[0] + ddy[1] * ddy[1] + ddy[2] * ddy[2]).sqrt();
    len_x.max(len_y)
}

/// The `LOD`-stabilized hashed alpha threshold for a surface `anchor`.
///
/// `anchor` is a stable world- or object-space coordinate for the fragment (it
/// must not depend on screen position, or the dither would crawl). `lod_scale`
/// is the anisotropic footprint of that anchor (see [`anisotropic_lod_scale`]);
/// a larger footprint selects a coarser hashed lattice so one cell keeps
/// covering roughly one pixel.
///
/// The steps are: derive a pixel-scale `1 / (hash_scale * lod_scale)`; take its
/// `log2` (from exponent bits) to get a continuous octave; `floor` it to the
/// coarse integer level and keep the fraction; rebuild the two bracketing
/// power-of-two scales with [`exp2_pow`]; hash the anchor quantized at each
/// level with [`hash3`]; and linearly blend the two hashes by the fraction via
/// [`two_level_lerp`]. The result is clamped into `[min_threshold, 1]`.
///
/// Because the level and the quantized coordinates come from `floor`, a
/// sub-cell change in `lod_scale` or `anchor` between two frames leaves the
/// threshold identical, which is what keeps the stochastic dither from
/// flickering.
#[must_use]
pub fn alpha_hash_threshold(anchor: [f32; 3], lod_scale: f32, params: &AlphaHashParams) -> f32 {
    let scale = params.effective_scale();
    let footprint = lod_scale.max(CMP_EPS);

    // Pixel scale: how many hashed cells span the anchor footprint. Guarded so
    // the following log2 always sees a strictly positive argument.
    let pix_scale = (1.0 / (scale * footprint)).max(CMP_EPS);

    let level = approx_log2(pix_scale);
    let coarse_level = level.floor();
    let frac_level = frac(level);
    let coarse = coarse_level as i32;

    let scale_lo = exp2_pow(coarse) * scale;
    let scale_hi = exp2_pow(coarse + 1) * scale;

    let h0 = hash3(
        quantize_coord(anchor[0], scale_lo),
        quantize_coord(anchor[1], scale_lo),
        quantize_coord(anchor[2], scale_lo),
        params.seed,
    );
    let h1 = hash3(
        quantize_coord(anchor[0], scale_hi),
        quantize_coord(anchor[1], scale_hi),
        quantize_coord(anchor[2], scale_hi),
        params.seed,
    );

    let blended = two_level_lerp(h0, h1, frac_level);
    blended.clamp(params.threshold_floor(), 1.0)
}

/// The hashed alpha test: keep the fragment when `alpha >= threshold`.
///
/// `threshold` comes from [`alpha_hash_threshold`]. Since that value is strictly
/// below `1.0` (bounded by [`INV_2POW24`]) yet at least `min_threshold > 0`, an
/// opaque fragment (`alpha == 1`) always survives and a fully transparent one
/// (`alpha == 0`) is always discarded, while intermediate alphas survive in
/// proportion to their value.
#[must_use]
pub fn hashed_alpha_test(alpha: f32, threshold: f32) -> bool {
    alpha >= threshold
}

/// Expected surviving coverage for a fragment `alpha` under the hashed test.
///
/// Because the threshold is uniformly distributed in `[0, 1)`, the probability
/// that `alpha >= threshold` is simply `alpha`, so the expected coverage equals
/// the (clamped) alpha. This is the value an `MSAA`/`TAA` resolve should
/// converge the stochastic samples toward.
#[must_use]
pub fn coverage_from_alpha(alpha: f32) -> f32 {
    clamp01(alpha)
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`AlphaHashParams`] records, using the shared clamp-to-one-element rule.
#[must_use]
pub fn alpha_hash_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(ALPHA_HASH_PARAMS_STRIDE, count)
}

/// Total byte size of a `std430` storage buffer holding `count` precomputed
/// per-particle threshold entries (a scalar `f32` `LUT`).
#[must_use]
pub fn threshold_buffer_bytes(count: usize) -> usize {
    storage_bytes(THRESHOLD_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared parameters for the behavioural tests: fine noise, a small
    /// strictly-positive floor, arbitrary seed.
    fn params() -> AlphaHashParams {
        AlphaHashParams::new(8.0, 1.0e-3, 0x1234_5678)
    }

    #[test]
    fn hash3_is_deterministic() {
        let a = hash3(3, -7, 11, 42);
        let b = hash3(3, -7, 11, 42);
        assert!((a - b).abs() < CMP_EPS);
    }

    #[test]
    fn hash3_stays_in_unit_interval() {
        for x in -20..20 {
            for y in -20..20 {
                let v = hash3(x, y, x ^ y, 7);
                assert!(v >= 0.0);
                assert!(v < 1.0);
            }
        }
    }

    #[test]
    fn hash3_distribution_covers_range() {
        let mut lo = 1.0_f32;
        let mut hi = 0.0_f32;
        let mut sum = 0.0_f32;
        let mut count = 0.0_f32;
        for i in 0..64 {
            for j in 0..64 {
                let v = hash3(i, j, 0, 99);
                lo = lo.min(v);
                hi = hi.max(v);
                sum += v;
                count += 1.0;
            }
        }
        // A well-mixed hash reaches near both ends and averages around 0.5.
        assert!(lo < 0.05);
        assert!(hi > 0.95);
        let mean = sum / count;
        assert!((mean - 0.5).abs() < 0.05);
    }

    #[test]
    fn hash3_seed_changes_output() {
        let a = hash3(1, 2, 3, 0);
        let b = hash3(1, 2, 3, 1);
        assert!((a - b).abs() > CMP_EPS);
    }

    #[test]
    fn hash3_distinct_cells_decorrelate() {
        // Neighboring cells should overwhelmingly differ; count matches.
        let mut collisions = 0;
        let base = hash3(0, 0, 0, 5);
        for i in 1..200 {
            if (hash3(i, 0, 0, 5) - base).abs() < CMP_EPS {
                collisions += 1;
            }
        }
        assert!(collisions <= 1);
    }

    #[test]
    fn alpha_one_always_keeps() {
        let p = params();
        for i in 0..50 {
            let anchor = [i as f32 * 0.13, 1.0 - i as f32 * 0.07, i as f32 * 0.31];
            let t = alpha_hash_threshold(anchor, 1.0, &p);
            assert!(hashed_alpha_test(1.0, t));
        }
    }

    #[test]
    fn alpha_zero_always_discards() {
        let p = params();
        for i in 0..50 {
            let anchor = [i as f32 * 0.23, i as f32 * -0.17, 2.0 + i as f32 * 0.05];
            let t = alpha_hash_threshold(anchor, 1.0, &p);
            assert!(!hashed_alpha_test(0.0, t));
        }
    }

    #[test]
    fn threshold_stays_in_unit_interval() {
        let p = params();
        for i in 0..40 {
            let anchor = [i as f32 * 0.37, i as f32 * 0.91, i as f32 * -0.53];
            let t = alpha_hash_threshold(anchor, 0.5, &p);
            assert!(t >= p.threshold_floor() - CMP_EPS);
            assert!(t <= 1.0);
        }
    }

    #[test]
    fn threshold_varies_with_anchor() {
        let p = params();
        let first = alpha_hash_threshold([0.0, 0.0, 0.0], 1.0, &p);
        let mut differ = 0;
        for i in 1..40 {
            let t = alpha_hash_threshold([i as f32, 0.0, 0.0], 1.0, &p);
            if (t - first).abs() > CMP_EPS {
                differ += 1;
            }
        }
        // The threshold is a function of the anchor, so it must move around.
        assert!(differ > 20);
    }

    #[test]
    fn threshold_respects_min_floor() {
        // A large floor forces every threshold up to at least that value.
        let p = AlphaHashParams::new(8.0, 0.75, 3);
        for i in 0..40 {
            let anchor = [i as f32 * 0.29, i as f32 * 0.61, 0.0];
            let t = alpha_hash_threshold(anchor, 1.0, &p);
            assert!(t >= 0.75 - CMP_EPS);
        }
    }

    #[test]
    fn lod_micro_jitter_is_stable() {
        let p = params();
        let anchor = [4.25, -2.5, 7.75];
        // Away from an octave boundary a sub-cell footprint wobble leaves the
        // integer level and quantized cell untouched, so the threshold barely
        // moves — the linear blend only nudges by the frac drift, well under a
        // pixel-visible amount.
        let base = alpha_hash_threshold(anchor, 1.3, &p);
        for step in 0..8 {
            let jitter = 1.3 + (step as f32) * 1.0e-5;
            let t = alpha_hash_threshold(anchor, jitter, &p);
            assert!((t - base).abs() < 1.0e-3);
        }
    }

    #[test]
    fn lod_boundary_crossing_is_continuous() {
        // `lod_scale == 1.0` lands exactly on an octave boundary (pix_scale is
        // an exact power of two). The two-level blend must keep the threshold
        // *continuous* across it — a tiny wobble may not cause the hard,
        // O(1) hash flip a naive per-frame re-hash would. The change stays
        // small even though the integer level ticks over.
        let p = params();
        let anchor = [4.25, -2.5, 7.75];
        let base = alpha_hash_threshold(anchor, 1.0, &p);
        for step in -6..=6 {
            let jitter = 1.0 + (step as f32) * 1.0e-4;
            let t = alpha_hash_threshold(anchor, jitter, &p);
            assert!((t - base).abs() < 0.05);
        }
    }

    #[test]
    fn lod_coarser_scale_can_change_threshold() {
        let p = params();
        let anchor = [3.3, 5.1, -1.2];
        let fine = alpha_hash_threshold(anchor, 1.0, &p);
        let mut changed = false;
        // Sweep the footprint across several octaves; the level-driven hash
        // should differ from the fine sample at least once.
        for k in 1..12 {
            let coarse = alpha_hash_threshold(anchor, (1 << k) as f32, &p);
            if (coarse - fine).abs() > CMP_EPS {
                changed = true;
            }
        }
        assert!(changed);
    }

    #[test]
    fn two_level_lerp_is_monotonic() {
        let h0 = 0.2_f32;
        let h1 = 0.8_f32;
        let mut prev = two_level_lerp(h0, h1, 0.0);
        for step in 1..=20 {
            let f = step as f32 / 20.0;
            let cur = two_level_lerp(h0, h1, f);
            assert!(cur >= prev - CMP_EPS);
            prev = cur;
        }
    }

    #[test]
    fn two_level_lerp_endpoints() {
        let h0 = 0.3_f32;
        let h1 = 0.9_f32;
        assert!((two_level_lerp(h0, h1, 0.0) - h0).abs() < CMP_EPS);
        assert!((two_level_lerp(h0, h1, 1.0) - h1).abs() < CMP_EPS);
    }

    #[test]
    fn exp2_pow_matches_known_powers() {
        assert!((exp2_pow(0) - 1.0).abs() < CMP_EPS);
        assert!((exp2_pow(1) - 2.0).abs() < CMP_EPS);
        assert!((exp2_pow(3) - 8.0).abs() < CMP_EPS);
        assert!((exp2_pow(-1) - 0.5).abs() < CMP_EPS);
        assert!((exp2_pow(-2) - 0.25).abs() < CMP_EPS);
    }

    #[test]
    fn approx_log2_is_exact_at_powers_and_monotonic() {
        assert!((approx_log2(1.0) - 0.0).abs() < CMP_EPS);
        assert!((approx_log2(2.0) - 1.0).abs() < CMP_EPS);
        assert!((approx_log2(8.0) - 3.0).abs() < CMP_EPS);
        let mut prev = approx_log2(0.1);
        let mut x = 0.2_f32;
        while x < 64.0 {
            let cur = approx_log2(x);
            assert!(cur >= prev - CMP_EPS);
            prev = cur;
            x *= 1.3;
        }
    }

    #[test]
    fn coverage_from_alpha_maps_and_clamps() {
        assert!((coverage_from_alpha(0.4) - 0.4).abs() < CMP_EPS);
        assert!((coverage_from_alpha(-1.0) - 0.0).abs() < CMP_EPS);
        assert!((coverage_from_alpha(2.0) - 1.0).abs() < CMP_EPS);
    }

    #[test]
    fn hashed_alpha_test_boundary_keeps_equal() {
        // alpha exactly equal to the threshold is kept (>=).
        assert!(hashed_alpha_test(0.5, 0.5));
        assert!(!hashed_alpha_test(0.49, 0.5));
    }

    #[test]
    fn std430_layout_roundtrips() {
        let p = AlphaHashParams::new(8.0, 0.125, 0xdead_beef);
        let packed = p.to_std430();
        assert_eq!(packed.len() * U32_STRIDE, ALPHA_HASH_PARAMS_STRIDE);
        assert!((f32::from_bits(packed[0]) - 8.0).abs() < CMP_EPS);
        assert!((f32::from_bits(packed[1]) - 0.125).abs() < CMP_EPS);
        assert_eq!(packed[2], 0xdead_beef);
        assert_eq!(packed[3], 0);
    }

    #[test]
    fn params_buffer_bytes_scale_and_clamp() {
        assert_eq!(alpha_hash_params_buffer_bytes(0), ALPHA_HASH_PARAMS_STRIDE);
        assert_eq!(
            alpha_hash_params_buffer_bytes(4),
            4 * ALPHA_HASH_PARAMS_STRIDE
        );
    }

    #[test]
    fn threshold_buffer_bytes_scale_and_clamp() {
        assert_eq!(threshold_buffer_bytes(0), THRESHOLD_STRIDE);
        assert_eq!(threshold_buffer_bytes(16), 16 * THRESHOLD_STRIDE);
    }

    #[test]
    fn anisotropic_lod_scale_takes_larger_axis() {
        let s = anisotropic_lod_scale([3.0, 4.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((s - 5.0).abs() < CMP_EPS);
    }
}
