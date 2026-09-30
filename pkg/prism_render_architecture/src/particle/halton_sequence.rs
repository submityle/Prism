//! Low-discrepancy quasi-random sequences (`Halton` / `Hammersley`) for the
//! particle engine's sampling passes (design §16-§21, §25).
//!
//! Many particle passes need a *point set that fills a domain more evenly than
//! random points do*: `TAA` sub-pixel jitter wants successive frames to land on
//! well-separated offsets inside one pixel, soft-shadow and ambient-occlusion
//! (`AO`) kernels want their taps spread over a disk or `hemisphere` without
//! clumping, and Monte-Carlo estimators converge faster when their strata are
//! covered. Uniform pseudo-random draws clump and leave gaps; a *low-discrepancy
//! sequence* trades independence for coverage, driving the estimator error down
//! toward `O(log(n)^d / n)` instead of `O(1/sqrt(n))`. This module is the
//! device-free, `CPU`-verifiable golden reference for those sequences so a
//! future `GPU` kernel can match it bit for bit.
//!
//! The building blocks are:
//!
//! * **Radical inverse.** [`radical_inverse_base2`] reflects an index's binary
//!   digits about the radix point with a single `u32` bit reversal, while
//!   [`radical_inverse`] handles an arbitrary base with a pure integer digit
//!   loop and a rational accumulation (`digit / base^k`). Neither uses a
//!   transcendental function.
//! * **`Halton` points.** [`halton_point_2d`] pairs base 2 and base 3 (the
//!   canonical two-dimensional `Halton` set), and [`halton_point`] takes an
//!   explicit, ideally coprime, base pair.
//! * **`Hammersley` points.** [`hammersley_point`] fixes the first axis to the
//!   equidistant `i / n` and the second to the base-2 radical inverse, giving
//!   the lowest-discrepancy set when the point count is known ahead of time.
//! * **Pixel jitter.** [`pixel_jitter`] and [`taa_jitter`] remap a unit-square
//!   sample into the `[-0.5, 0.5)` sub-pixel offset a `TAA` pass adds to its
//!   projection matrix.
//! * **`Cranley-Patterson` rotation.** [`cranley_patterson`] and its 2-D form
//!   decorrelate the same base sequence across pixels or frames by adding a
//!   per-instance offset and taking the fractional part, without disturbing the
//!   sequence's stratification.
//! * **Sequence buffers.** [`halton_sequence`] / [`hammersley_sequence`]
//!   materialize point buffers, and [`pack_sequence_std430`] /
//!   [`sequence_storage_bytes`] describe their `std430` `vec2<f32>` byte layout
//!   for a storage binding.
//!
//! Deliberately out of scope, and never imported here: [`super::determinism`]
//! owns the *stateless hash* `RNG` (a `RngKey` / `StreamId` pseudo-random
//! source) — this module is a *deterministic low-discrepancy* generator, not a
//! hash `RNG`. Ordered `Bayer` dithering lives in [`super::temporal_dither`],
//! and value / gradient noise lives in [`super::noise`], [`super::worley`], and
//! [`super::curl_noise`]. This file re-derives none of them.
//!
//! Only `f32::sqrt`-free algebra with `f32::floor`, integer arithmetic, and a
//! single `u32` bit reversal appear here — no transcendental functions — so the
//! output is deterministic and platform independent.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// `2^32` as an `f32`; the exact span used to normalize a reflected `u32` word
/// into the unit interval (`2^32` is representable in an `f32` mantissa).
const U32_SPAN: f32 = 4_294_967_296.0;

/// `2^-32`, the reciprocal of [`U32_SPAN`]; multiplying a bit-reversed `u32` by
/// this yields the base-2 radical inverse. Exact, since it is a power of two.
const INV_2_POW_32: f32 = 1.0 / U32_SPAN;

/// Byte stride of one packed sequence point (`vec2<f32>`) in a `std430` storage
/// buffer; the sequence buffers share the shared `vec2` stride.
pub const SEQUENCE_ELEMENT_STRIDE: usize = VEC2_STRIDE;

/// Widens a `u32` to `f32`.
///
/// Callers pass either a small digit / counter or a bit-reversed index whose
/// low bits are intentionally rounded away by the base-2 radical inverse, so
/// the documented precision loss is exactly the intended behavior.
#[expect(
    clippy::cast_precision_loss,
    reason = "index/digit widening; low-bit rounding is the intended radical-inverse behavior"
)]
fn u32_to_f32(v: u32) -> f32 {
    v as f32
}

/// Returns the fractional part of `v` in `[0, 1)` by subtracting its floor.
fn wrap01(v: f32) -> f32 {
    v - v.floor()
}

/// Base-2 radical inverse (van der Corput sequence) of `i`.
///
/// Reflects the binary digits of `i` about the radix point: bit `k` of `i`
/// becomes the weight `2^-(k+1)`. Implemented as one `u32` bit reversal plus a
/// multiply by `2^-32`, so the result lies in `[0, 1)` and never calls a
/// transcendental function. This is the second `Hammersley` axis.
#[must_use]
pub fn radical_inverse_base2(i: u32) -> f32 {
    u32_to_f32(i.reverse_bits()) * INV_2_POW_32
}

/// Radical inverse of `i` in an arbitrary integer `base` (`base >= 2`).
///
/// Walks the base-`base` digits of `i` from least to most significant with a
/// pure integer loop, accumulating each digit at the rational weight
/// `base^-k`. No transcendental function is used: only integer division /
/// remainder and `f32` multiply / add. A degenerate `base < 2` yields `0.0`
/// rather than looping forever. The result lies in `[0, 1)`.
#[must_use]
pub fn radical_inverse(base: u32, mut i: u32) -> f32 {
    if base < 2 {
        return 0.0;
    }
    let inv_base = 1.0 / u32_to_f32(base);
    let mut inv_base_k = 1.0f32;
    let mut result = 0.0f32;
    while i > 0 {
        let digit = i % base;
        i /= base;
        inv_base_k *= inv_base;
        result += u32_to_f32(digit) * inv_base_k;
    }
    result
}

/// Two-dimensional `Halton` point for index `i` using the canonical base pair
/// (2, 3).
///
/// The first axis uses the fast base-2 bit-reversal inverse and the second uses
/// the base-3 digit loop, giving a well-stratified unit-square point.
#[must_use]
pub fn halton_point_2d(i: u32) -> [f32; 2] {
    [radical_inverse_base2(i), radical_inverse(3, i)]
}

/// Two-dimensional `Halton` point for index `i` using an explicit base pair.
///
/// For good coverage `base_x` and `base_y` should be small, distinct primes
/// (for example 2 and 3, or 5 and 7); correlated or equal bases degrade the
/// stratification. Each component lies in `[0, 1)`.
#[must_use]
pub fn halton_point(base_x: u32, base_y: u32, i: u32) -> [f32; 2] {
    [radical_inverse(base_x, i), radical_inverse(base_y, i)]
}

/// `Hammersley` point `i` of an `n`-point set.
///
/// The first axis is the equidistant `i / n` and the second is the base-2
/// radical inverse of `i`. This is the lowest-discrepancy two-dimensional set
/// when the total count `n` is fixed in advance (a degenerate `n = 0` is
/// treated as `n = 1`). Both components lie in `[0, 1)`.
#[must_use]
pub fn hammersley_point(i: u32, n: u32) -> [f32; 2] {
    let x = u32_to_f32(i) / u32_to_f32(n.max(1));
    [x, radical_inverse_base2(i)]
}

/// Remaps a unit-square sample into a `[-0.5, 0.5)` sub-pixel offset by
/// subtracting the pixel center.
///
/// This is the offset a `TAA` pass adds to the projection matrix so each frame
/// samples a different point inside one pixel.
#[must_use]
pub fn pixel_jitter(point: [f32; 2]) -> [f32; 2] {
    [point[0] - 0.5, point[1] - 0.5]
}

/// Sub-pixel `TAA` jitter for frame `frame`.
///
/// Advances the `Halton` (2, 3) sequence by one so frame 0 is not the pixel
/// center, then remaps the point into `[-0.5, 0.5)` via [`pixel_jitter`].
#[must_use]
pub fn taa_jitter(frame: u32) -> [f32; 2] {
    pixel_jitter(halton_point_2d(frame.wrapping_add(1)))
}

/// `Cranley-Patterson` rotation of a scalar sample by `offset`.
///
/// Adds the offset and keeps the fractional part, toroidally shifting the
/// sequence so different pixels or frames draw decorrelated points from the
/// same base sequence without losing its stratification. The result stays in
/// `[0, 1)`.
#[must_use]
pub fn cranley_patterson(value: f32, offset: f32) -> f32 {
    wrap01(value + offset)
}

/// Two-dimensional [`cranley_patterson`] rotation.
#[must_use]
pub fn cranley_patterson_2d(point: [f32; 2], offset: [f32; 2]) -> [f32; 2] {
    [
        cranley_patterson(point[0], offset[0]),
        cranley_patterson(point[1], offset[1]),
    ]
}

/// Materializes `count` `Halton` points starting at index `start_index` using
/// the base pair (`base_x`, `base_y`).
///
/// Callers typically pass `start_index = 1` so the buffer skips the origin
/// point that index 0 produces. The index saturates at `u32::MAX` rather than
/// wrapping.
#[must_use]
pub fn halton_sequence(base_x: u32, base_y: u32, start_index: u32, count: u32) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for offset in 0..count {
        let idx = start_index.saturating_add(offset);
        out.push(halton_point(base_x, base_y, idx));
    }
    out
}

/// Materializes the full `count`-point `Hammersley` set (indices `0..count`).
#[must_use]
pub fn hammersley_sequence(count: u32) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for i in 0..count {
        out.push(hammersley_point(i, count));
    }
    out
}

/// Total `std430` byte size of a storage buffer holding `count` sequence
/// points, clamped up to a single element for an empty set.
#[must_use]
pub fn sequence_storage_bytes(count: u32) -> usize {
    storage_bytes(SEQUENCE_ELEMENT_STRIDE, usize::try_from(count).unwrap_or(0))
}

/// Packs sequence points into a `std430` `vec2<f32>` byte buffer, little-endian.
///
/// The layout is a tight array of `vec2<f32>` elements (two little-endian `f32`
/// words each), matching [`SEQUENCE_ELEMENT_STRIDE`] per element.
#[must_use]
pub fn pack_sequence_std430(points: &[[f32; 2]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(points.len() * SEQUENCE_ELEMENT_STRIDE);
    for point in points {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing `f32` reference values.
    const CMP_EPS: f32 = 1e-6;

    /// Nine grid thresholds (`k` for `k` in `0..=8`) used to bucket a value
    /// scaled by 8 into one of eight strata without any integer-to-float cast.
    const STRATA8: [f32; 9] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn in_unit_interval(v: f32) -> bool {
        (0.0..1.0).contains(&v)
    }

    /// A simple grid star-discrepancy estimate: the worst deviation of the
    /// fraction of points inside a corner box `[0, bx) x [0, by)` from `bx*by`.
    fn star_discrepancy(points: &[[f32; 2]]) -> f32 {
        let grid = 8u32;
        let n = u32_to_f32(u32::try_from(points.len()).unwrap_or(0)).max(1.0);
        let mut worst = 0.0f32;
        for jx in 1..=grid {
            for jy in 1..=grid {
                let bx = u32_to_f32(jx) / u32_to_f32(grid);
                let by = u32_to_f32(jy) / u32_to_f32(grid);
                let mut inside = 0u32;
                for p in points {
                    if p[0] < bx && p[1] < by {
                        inside += 1;
                    }
                }
                let dev = (u32_to_f32(inside) / n - bx * by).abs();
                if dev > worst {
                    worst = dev;
                }
            }
        }
        worst
    }

    #[test]
    fn base2_known_values() {
        assert!(approx(radical_inverse_base2(1), 0.5));
        assert!(approx(radical_inverse_base2(2), 0.25));
        assert!(approx(radical_inverse_base2(3), 0.75));
        assert!(approx(radical_inverse_base2(4), 0.125));
    }

    #[test]
    fn base3_known_values() {
        assert!(approx(radical_inverse(3, 1), 1.0 / 3.0));
        assert!(approx(radical_inverse(3, 2), 2.0 / 3.0));
        assert!(approx(radical_inverse(3, 3), 1.0 / 9.0));
        assert!(approx(radical_inverse(3, 0), 0.0));
    }

    #[test]
    fn arbitrary_base_known_values() {
        assert!(approx(radical_inverse(5, 1), 0.2));
        assert!(approx(radical_inverse(5, 5), 0.04));
        assert!(approx(radical_inverse(10, 123), 0.321));
    }

    #[test]
    fn base2_range_unit_interval() {
        for i in 0..4096u32 {
            assert!(in_unit_interval(radical_inverse_base2(i)));
        }
    }

    #[test]
    fn arbitrary_base_range_unit_interval() {
        for base in [2u32, 3, 5, 7, 11, 13] {
            for i in 0..1024u32 {
                assert!(in_unit_interval(radical_inverse(base, i)));
            }
        }
    }

    #[test]
    fn halton_point_2d_known() {
        let p1 = halton_point_2d(1);
        assert!(approx(p1[0], 0.5));
        assert!(approx(p1[1], 1.0 / 3.0));
        let p2 = halton_point_2d(2);
        assert!(approx(p2[0], 0.25));
        assert!(approx(p2[1], 2.0 / 3.0));
        let p3 = halton_point_2d(3);
        assert!(approx(p3[0], 0.75));
        assert!(approx(p3[1], 1.0 / 9.0));
    }

    #[test]
    fn halton_point_pair_matches_components() {
        for i in [1u32, 7, 42, 1000] {
            let p = halton_point(5, 7, i);
            assert!(approx(p[0], radical_inverse(5, i)));
            assert!(approx(p[1], radical_inverse(7, i)));
        }
    }

    #[test]
    fn hammersley_origin_start() {
        let p = hammersley_point(0, 16);
        assert!(approx(p[0], 0.0));
        assert!(approx(p[1], 0.0));
    }

    #[test]
    fn hammersley_x_is_i_over_n() {
        assert!(approx(hammersley_point(2, 4)[0], 0.5));
        assert!(approx(hammersley_point(3, 4)[0], 0.75));
        assert!(approx(hammersley_point(1, 8)[0], 0.125));
    }

    #[test]
    fn base2_perfect_stratification() {
        // Over the first 8 indices the base-2 radical inverse hits each of the
        // eight equal sub-intervals exactly once (a permutation), which the
        // dimensional coverage of a low-discrepancy sequence guarantees.
        let mut seen = [false; 8];
        for i in 0..8u32 {
            let scaled = radical_inverse_base2(i) * 8.0;
            for (k, pair) in STRATA8.windows(2).enumerate() {
                if (pair[0]..pair[1]).contains(&scaled) {
                    seen[k] = true;
                }
            }
        }
        assert!(seen.iter().all(|&hit| hit));
    }

    #[test]
    fn halton_lower_discrepancy_than_clustered() {
        let halton = halton_sequence(2, 3, 1, 64);
        // A deliberately clustered baseline: every index maps to the same two
        // values, leaving large empty regions.
        let mut clustered = Vec::with_capacity(64);
        for i in 0..64u32 {
            let v = wrap01(u32_to_f32(i) * 0.5);
            clustered.push([v, v]);
        }
        assert!(star_discrepancy(&halton) < star_discrepancy(&clustered));
    }

    #[test]
    fn sequence_no_near_duplicates() {
        let seq = halton_sequence(2, 3, 1, 24);
        for a in 0..seq.len() {
            for b in (a + 1)..seq.len() {
                let dx = seq[a][0] - seq[b][0];
                let dy = seq[a][1] - seq[b][1];
                assert!(dx * dx + dy * dy > CMP_EPS);
            }
        }
    }

    #[test]
    fn cranley_patterson_stays_unit_interval() {
        for base_i in 0..256u32 {
            let v = radical_inverse_base2(base_i);
            for off in [0.1f32, 0.37, 0.5, 0.99, 1.5, 2.75] {
                assert!(in_unit_interval(cranley_patterson(v, off)));
            }
        }
    }

    #[test]
    fn cranley_patterson_known_value() {
        assert!(approx(cranley_patterson(0.7, 0.6), 0.3));
        let p = cranley_patterson_2d([0.2, 0.9], [0.9, 0.9]);
        assert!(approx(p[0], 0.1));
        assert!(approx(p[1], 0.8));
    }

    #[test]
    fn jitter_range_centered() {
        for i in 0..1024u32 {
            let j = pixel_jitter(halton_point_2d(i));
            assert!((-0.5..0.5).contains(&j[0]));
            assert!((-0.5..0.5).contains(&j[1]));
        }
    }

    #[test]
    fn jitter_known_mapping() {
        let corner = pixel_jitter([0.0, 0.0]);
        assert!(approx(corner[0], -0.5));
        assert!(approx(corner[1], -0.5));
        let center = pixel_jitter([0.5, 0.5]);
        assert!(approx(center[0], 0.0));
        assert!(approx(center[1], 0.0));
    }

    #[test]
    fn taa_jitter_within_pixel() {
        for frame in 0..512u32 {
            let j = taa_jitter(frame);
            assert!((-0.5..0.5).contains(&j[0]));
            assert!((-0.5..0.5).contains(&j[1]));
        }
    }

    #[test]
    fn halton_sequence_length_and_range() {
        let seq = halton_sequence(2, 3, 1, 50);
        assert_eq!(seq.len(), 50);
        for p in &seq {
            assert!(in_unit_interval(p[0]));
            assert!(in_unit_interval(p[1]));
        }
    }

    #[test]
    fn hammersley_sequence_length_and_origin() {
        let seq = hammersley_sequence(32);
        assert_eq!(seq.len(), 32);
        assert!(approx(seq[0][0], 0.0));
        assert!(approx(seq[0][1], 0.0));
        for p in &seq {
            assert!(in_unit_interval(p[0]));
            assert!(in_unit_interval(p[1]));
        }
    }

    #[test]
    fn storage_bytes_match_std430() {
        assert_eq!(sequence_storage_bytes(3), SEQUENCE_ELEMENT_STRIDE * 3);
        assert_eq!(sequence_storage_bytes(256), SEQUENCE_ELEMENT_STRIDE * 256);
        // An empty set still reserves one element for a valid storage binding.
        assert_eq!(sequence_storage_bytes(0), SEQUENCE_ELEMENT_STRIDE);
    }

    #[test]
    fn pack_roundtrip_first_value() {
        let seq = halton_sequence(2, 3, 1, 4);
        let packed = pack_sequence_std430(&seq);
        assert_eq!(packed.len(), seq.len() * SEQUENCE_ELEMENT_STRIDE);
        let x = f32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
        let y = f32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
        assert!(approx(x, seq[0][0]));
        assert!(approx(y, seq[0][1]));
    }

    #[test]
    fn radical_inverse_guards_small_base() {
        assert!(approx(radical_inverse(0, 123), 0.0));
        assert!(approx(radical_inverse(1, 123), 0.0));
    }
}
