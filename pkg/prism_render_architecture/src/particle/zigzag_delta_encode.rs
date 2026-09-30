//! Delta + `ZigZag` integer transcoding: the lossless integer preconditioner
//! that turns slowly varying or signed attribute streams into small unsigned
//! magnitudes before a byte-oriented compressor (or GPU upload) sees them
//! (design §27).
//!
//! Two classic, closed-form integer transforms compose here:
//!
//! * **`ZigZag`** maps a signed integer to an unsigned one so that small
//!   magnitudes — regardless of sign — become small unsigned values:
//!   `0 -> 0`, `-1 -> 1`, `1 -> 2`, `-2 -> 3`, `2 -> 4`, ... This is the
//!   mapping used by Protocol Buffers / varint encoders, and it is exact and
//!   bijective over the whole `i32`/`i64` range.
//! * **Delta** replaces each element after the first with the difference from
//!   its predecessor. Monotonic or slowly varying streams (particle ids,
//!   lifetime buckets, sorted keys, timestamps) collapse to a run of tiny
//!   deltas; a constant stream collapses to all-zero deltas.
//!
//! The combined entry points [`delta_zigzag_encode`] / [`delta_zigzag_decode`]
//! first take signed adjacent differences and then fold them through `ZigZag`,
//! producing an unsigned stream whose values are small exactly when the input
//! is smooth. Every transform here is exact: `decode(encode(x)) == x` holds
//! for *all* inputs, including the `i32::MIN`/`i32::MAX` boundaries, because the
//! differencing uses two's-complement wrapping arithmetic that the prefix-sum
//! reconstruction unwinds precisely.
//!
//! All arithmetic is integer only — shifts, xor, and `wrapping_*` add/sub — with
//! no `f32` and no transcendental calls, so this reference is bit-reproducible
//! against a future GPU kernel.
//!
//! ## Boundaries
//!
//! This module owns *only* the reversible integer preconditioning transforms
//! (`ZigZag` remap and adjacent differencing). It is deliberately distinct from
//! its neighbours:
//!
//! * [`super::compression`] performs *lossy* numeric quantization (`fp16`,
//!   `snorm`/`unorm`, octahedral). Nothing here is lossy.
//! * [`super::rgbe_encode`] performs *lossy* shared-exponent `HDR` packing.
//! * [`super::run_length_encode`] collapses maximal runs of equal values into
//!   `(value, count)` pairs; it never differences neighbours or remaps signs.
//!
//! `ZigZag` + delta and `RLE` are complementary — delta turns a ramp into a
//! constant that `RLE` can then collapse — but the two codecs share no types
//! and no functions.

use alloc::vec::Vec;

/// Map a signed [`i32`] to an unsigned [`u32`] so small magnitudes of either
/// sign become small unsigned values (`0 -> 0`, `-1 -> 1`, `1 -> 2`, ...).
///
/// The transform is `(n << 1) ^ (n >> 31)`, where the arithmetic right shift
/// broadcasts the sign bit to either `0` or `all-ones`. It is exact and
/// bijective over the entire `i32` range; see [`zigzag_decode_i32`] for the
/// inverse.
#[must_use]
#[inline]
pub const fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Inverse of [`zigzag_encode_i32`]: recover the signed [`i32`] from its
/// `ZigZag`-encoded [`u32`].
///
/// Computed as `(z >> 1) ^ -(z & 1)`, where `-(z & 1)` is `0` for even `z`
/// (originally non-negative) and `all-ones` for odd `z` (originally negative).
#[must_use]
#[inline]
pub const fn zigzag_decode_i32(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

/// Map a signed [`i64`] to an unsigned [`u64`]; the 64-bit analogue of
/// [`zigzag_encode_i32`], using `(n << 1) ^ (n >> 63)`.
#[must_use]
#[inline]
pub const fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Inverse of [`zigzag_encode_i64`]: recover the signed [`i64`] from its
/// `ZigZag`-encoded [`u64`].
#[must_use]
#[inline]
pub const fn zigzag_decode_i64(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

/// Adjacent-difference an [`i32`] slice: the first element is copied verbatim,
/// each later element becomes `input[i].wrapping_sub(input[i - 1])`.
///
/// Wrapping subtraction keeps the transform exact and total: even a jump from
/// `i32::MIN` to `i32::MAX` produces a well-defined delta that
/// [`delta_decode_i32`] reverses. An empty input yields an empty output.
#[must_use]
pub fn delta_encode_i32(input: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(input.len());
    let mut prev: i32 = 0;
    for (i, &value) in input.iter().enumerate() {
        if i == 0 {
            out.push(value);
        } else {
            out.push(value.wrapping_sub(prev));
        }
        prev = value;
    }
    out
}

/// Inverse of [`delta_encode_i32`]: prefix-sum the deltas back into the
/// original [`i32`] stream using wrapping addition.
#[must_use]
pub fn delta_decode_i32(deltas: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(deltas.len());
    let mut acc: i32 = 0;
    for (i, &delta) in deltas.iter().enumerate() {
        if i == 0 {
            acc = delta;
        } else {
            acc = acc.wrapping_add(delta);
        }
        out.push(acc);
    }
    out
}

/// Adjacent-difference a [`u32`] slice: the first element is copied verbatim,
/// each later element becomes `input[i].wrapping_sub(input[i - 1])`.
///
/// Uses `u32` wrapping subtraction, so the deltas are reduced modulo `2^32`
/// and [`delta_decode_u32`] reverses them exactly for any input.
#[must_use]
pub fn delta_encode_u32(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut prev: u32 = 0;
    for (i, &value) in input.iter().enumerate() {
        if i == 0 {
            out.push(value);
        } else {
            out.push(value.wrapping_sub(prev));
        }
        prev = value;
    }
    out
}

/// Inverse of [`delta_encode_u32`]: prefix-sum the deltas back into the
/// original [`u32`] stream using wrapping addition.
#[must_use]
pub fn delta_decode_u32(deltas: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(deltas.len());
    let mut acc: u32 = 0;
    for (i, &delta) in deltas.iter().enumerate() {
        if i == 0 {
            acc = delta;
        } else {
            acc = acc.wrapping_add(delta);
        }
        out.push(acc);
    }
    out
}

/// Delta then `ZigZag`: adjacent-difference the signed [`i32`] stream (see
/// [`delta_encode_i32`]) and fold each resulting delta through
/// [`zigzag_encode_i32`], yielding an unsigned [`u32`] stream whose values are
/// small exactly when the input is smooth.
///
/// This is the recommended integer preconditioner to run ahead of a
/// byte-oriented compressor. It is fully lossless; see [`delta_zigzag_decode`].
#[must_use]
pub fn delta_zigzag_encode(input: &[i32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut prev: i32 = 0;
    for (i, &value) in input.iter().enumerate() {
        let delta = if i == 0 {
            value
        } else {
            value.wrapping_sub(prev)
        };
        out.push(zigzag_encode_i32(delta));
        prev = value;
    }
    out
}

/// Inverse of [`delta_zigzag_encode`]: `ZigZag`-decode each element back to a
/// signed delta (see [`zigzag_decode_i32`]) and prefix-sum the deltas with
/// wrapping addition to recover the original [`i32`] stream.
#[must_use]
pub fn delta_zigzag_decode(encoded: &[u32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(encoded.len());
    let mut acc: i32 = 0;
    for (i, &z) in encoded.iter().enumerate() {
        let delta = zigzag_decode_i32(z);
        acc = if i == 0 {
            delta
        } else {
            acc.wrapping_add(delta)
        };
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // --- ZigZag i32: exact known vectors -----------------------------------

    #[test]
    fn zigzag_encode_known_small_vectors() {
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(-2), 3);
        assert_eq!(zigzag_encode_i32(2), 4);
        assert_eq!(zigzag_encode_i32(-3), 5);
        assert_eq!(zigzag_encode_i32(3), 6);
    }

    #[test]
    fn zigzag_decode_known_small_vectors() {
        assert_eq!(zigzag_decode_i32(0), 0);
        assert_eq!(zigzag_decode_i32(1), -1);
        assert_eq!(zigzag_decode_i32(2), 1);
        assert_eq!(zigzag_decode_i32(3), -2);
        assert_eq!(zigzag_decode_i32(4), 2);
        assert_eq!(zigzag_decode_i32(5), -3);
        assert_eq!(zigzag_decode_i32(6), 3);
    }

    #[test]
    fn zigzag_i32_boundary_exact_values() {
        // i32::MAX = 2147483647 -> 4294967294; i32::MIN -> 4294967295.
        assert_eq!(zigzag_encode_i32(i32::MAX), 4_294_967_294);
        assert_eq!(zigzag_encode_i32(i32::MIN), 4_294_967_295);
        assert_eq!(zigzag_decode_i32(4_294_967_294), i32::MAX);
        assert_eq!(zigzag_decode_i32(4_294_967_295), i32::MIN);
    }

    #[test]
    fn zigzag_i32_roundtrip_boundaries() {
        for &n in &[
            0_i32,
            -1,
            1,
            2,
            -2,
            i32::MIN,
            i32::MAX,
            i32::MIN + 1,
            i32::MAX - 1,
        ] {
            assert_eq!(zigzag_decode_i32(zigzag_encode_i32(n)), n);
        }
    }

    #[test]
    fn zigzag_i32_roundtrip_swept_range() {
        // Sweep a wide signed range with a stride to keep the test cheap.
        let mut n: i64 = i32::MIN as i64;
        while n <= i32::MAX as i64 {
            let v = n as i32;
            assert_eq!(zigzag_decode_i32(zigzag_encode_i32(v)), v);
            n += 97_003;
        }
    }

    #[test]
    fn zigzag_i32_is_injective_on_small_window() {
        // Distinct signed inputs must map to distinct unsigned codes.
        let mut seen = vec![false; 4001];
        for v in -2000_i32..=2000 {
            let code = zigzag_encode_i32(v) as usize;
            // Small magnitudes stay within a small unsigned window.
            assert!(code <= 4000, "code {code} out of expected small window");
            assert!(!seen[code], "collision at code {code}");
            seen[code] = true;
        }
    }

    #[test]
    fn zigzag_small_magnitude_maps_to_small_code() {
        // The whole point of ZigZag: a magnitude-1 negative beats a large positive.
        assert!(zigzag_encode_i32(-1) < zigzag_encode_i32(1000));
        assert!(zigzag_encode_i32(3) < zigzag_encode_i32(-1000));
    }

    // --- ZigZag i64 --------------------------------------------------------

    #[test]
    fn zigzag_encode_i64_known_vectors() {
        assert_eq!(zigzag_encode_i64(0), 0);
        assert_eq!(zigzag_encode_i64(-1), 1);
        assert_eq!(zigzag_encode_i64(1), 2);
        assert_eq!(zigzag_encode_i64(-2), 3);
    }

    #[test]
    fn zigzag_i64_boundary_exact_values() {
        assert_eq!(zigzag_encode_i64(i64::MAX), u64::MAX - 1);
        assert_eq!(zigzag_encode_i64(i64::MIN), u64::MAX);
        assert_eq!(zigzag_decode_i64(u64::MAX - 1), i64::MAX);
        assert_eq!(zigzag_decode_i64(u64::MAX), i64::MIN);
    }

    #[test]
    fn zigzag_i64_roundtrip_boundaries() {
        for &n in &[
            0_i64,
            -1,
            1,
            -2,
            2,
            i64::MIN,
            i64::MAX,
            i64::MIN + 1,
            i64::MAX - 1,
        ] {
            assert_eq!(zigzag_decode_i64(zigzag_encode_i64(n)), n);
        }
    }

    // --- Delta i32 ---------------------------------------------------------

    #[test]
    fn delta_encode_empty_is_empty() {
        assert!(delta_encode_i32(&[]).is_empty());
        assert!(delta_decode_i32(&[]).is_empty());
    }

    #[test]
    fn delta_encode_single_element_is_verbatim() {
        assert_eq!(delta_encode_i32(&[42]), vec![42]);
        assert_eq!(delta_decode_i32(&[42]), vec![42]);
    }

    #[test]
    fn delta_encode_constant_sequence_is_all_zero_after_first() {
        let input = [7_i32, 7, 7, 7, 7];
        let deltas = delta_encode_i32(&input);
        assert_eq!(deltas, vec![7, 0, 0, 0, 0]);
        assert_eq!(delta_decode_i32(&deltas), input);
    }

    #[test]
    fn delta_encode_monotonic_ramp_yields_small_deltas() {
        let input = [100_i32, 101, 103, 106, 110];
        let deltas = delta_encode_i32(&input);
        assert_eq!(deltas, vec![100, 1, 2, 3, 4]);
        assert_eq!(delta_decode_i32(&deltas), input);
    }

    #[test]
    fn delta_encode_alternating_signs_roundtrip() {
        let input = [5_i32, -5, 5, -5, 5, -5];
        let deltas = delta_encode_i32(&input);
        assert_eq!(deltas, vec![5, -10, 10, -10, 10, -10]);
        assert_eq!(delta_decode_i32(&deltas), input);
    }

    #[test]
    fn delta_encode_i32_wrapping_boundary_no_panic() {
        // A jump between the extremes must be representable via wrapping.
        let input = [i32::MIN, i32::MAX, i32::MIN, i32::MAX];
        let deltas = delta_encode_i32(&input);
        assert_eq!(delta_decode_i32(&deltas), input);
    }

    #[test]
    fn delta_encode_decode_i32_roundtrip_various() {
        let cases: [&[i32]; 4] = [
            &[0],
            &[i32::MIN, 0, i32::MAX],
            &[-3, -2, -1, 0, 1, 2, 3],
            &[1_000_000, -1_000_000, 0, 999],
        ];
        for case in cases {
            assert_eq!(delta_decode_i32(&delta_encode_i32(case)), case.to_vec());
        }
    }

    // --- Delta u32 ---------------------------------------------------------

    #[test]
    fn delta_encode_u32_ramp() {
        let input = [10_u32, 12, 15, 19];
        let deltas = delta_encode_u32(&input);
        assert_eq!(deltas, vec![10, 2, 3, 4]);
        assert_eq!(delta_decode_u32(&deltas), input);
    }

    #[test]
    fn delta_encode_u32_wrapping_boundary_no_panic() {
        let input = [0_u32, u32::MAX, 0, u32::MAX];
        let deltas = delta_encode_u32(&input);
        assert_eq!(delta_decode_u32(&deltas), input);
    }

    #[test]
    fn delta_u32_empty_and_single() {
        assert!(delta_encode_u32(&[]).is_empty());
        assert_eq!(delta_encode_u32(&[9]), vec![9]);
        assert_eq!(delta_decode_u32(&[9]), vec![9]);
    }

    // --- Combined delta + ZigZag ------------------------------------------

    #[test]
    fn delta_zigzag_constant_sequence_is_small() {
        let input = [500_i32; 6];
        let encoded = delta_zigzag_encode(&input);
        // First element ZigZag(500)=1000, remaining deltas are 0 -> 0.
        assert_eq!(encoded, vec![1000, 0, 0, 0, 0, 0]);
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }

    #[test]
    fn delta_zigzag_descending_ramp_is_small() {
        // A descending ramp produces negative deltas that ZigZag keeps small.
        let input = [10_i32, 9, 8, 7, 6];
        let encoded = delta_zigzag_encode(&input);
        // ZigZag(10)=20, then delta -1 -> 1 four times.
        assert_eq!(encoded, vec![20, 1, 1, 1, 1]);
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }

    #[test]
    fn delta_zigzag_empty_and_single() {
        assert!(delta_zigzag_encode(&[]).is_empty());
        assert!(delta_zigzag_decode(&[]).is_empty());
        assert_eq!(delta_zigzag_encode(&[-1]), vec![1]);
        assert_eq!(delta_zigzag_decode(&[1]), vec![-1]);
    }

    #[test]
    fn delta_zigzag_boundary_roundtrip_no_panic() {
        let input = [i32::MIN, i32::MAX, 0, i32::MIN, i32::MAX];
        let encoded = delta_zigzag_encode(&input);
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }

    #[test]
    fn delta_zigzag_alternating_roundtrip() {
        let input = [-100_i32, 100, -100, 100, -100];
        let encoded = delta_zigzag_encode(&input);
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }

    #[test]
    fn delta_zigzag_matches_manual_composition() {
        let input = [3_i32, -7, 42, 42, -1000, i32::MAX];
        let via_combined = delta_zigzag_encode(&input);
        let via_manual: Vec<u32> = delta_encode_i32(&input)
            .iter()
            .map(|&d| zigzag_encode_i32(d))
            .collect();
        assert_eq!(via_combined, via_manual);
    }

    // --- Randomized round-trips (deterministic LCG) ------------------------

    #[test]
    fn zigzag_i32_random_roundtrip() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..5000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let v = (state >> 32) as i32;
            assert_eq!(zigzag_decode_i32(zigzag_encode_i32(v)), v);
        }
    }

    #[test]
    fn zigzag_i64_random_roundtrip() {
        let mut state: u64 = 0xdead_beef_cafe_babe;
        for _ in 0..5000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let v = state as i64;
            assert_eq!(zigzag_decode_i64(zigzag_encode_i64(v)), v);
        }
    }

    #[test]
    fn delta_zigzag_random_sequence_roundtrip() {
        let mut state: u64 = 0x0f0f_0f0f_1234_5678;
        let mut input = Vec::with_capacity(256);
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            input.push((state >> 33) as i32);
        }
        let encoded = delta_zigzag_encode(&input);
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }

    #[test]
    fn delta_i32_random_sequence_roundtrip() {
        let mut state: u64 = 0x00c0_ffee_00c0_ffee;
        let mut input = Vec::with_capacity(512);
        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            input.push(state as i32);
        }
        assert_eq!(delta_decode_i32(&delta_encode_i32(&input)), input);
    }

    #[test]
    fn delta_zigzag_smooth_stream_stays_bounded() {
        // A gently increasing signal encodes to small unsigned codes after
        // the first sample, demonstrating the intended bandwidth win.
        let input: Vec<i32> = (0..100).map(|i| 10_000 + i * 2).collect();
        let encoded = delta_zigzag_encode(&input);
        for &code in &encoded[1..] {
            assert!(code <= 4, "delta code {code} unexpectedly large");
        }
        assert_eq!(delta_zigzag_decode(&encoded), input);
    }
}
