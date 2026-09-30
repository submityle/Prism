//! Run-length encoding (`RLE`) for `u32` streams: collapse each maximal run of
//! identical values into a `(value, count)` pair, and expand those pairs back
//! into the original stream. This is the classic lossless codec used ahead of
//! GPU upload for attribute streams that contain long stretches of a repeated
//! value (sleeping-particle flags, tile/material ids, lifetime buckets), where
//! a `(value, count)` pair is far cheaper than storing the value once per
//! element.
//!
//! The codec is exact and closed: [`decode`] applied to the output of
//! [`encode`] reproduces the input bit-for-bit (`decode(encode(x)) == x`). All
//! arithmetic is integer only — no `f32`, no transcendentals — so the reference
//! here is bit-reproducible against a future GPU kernel.
//!
//! ## Run splitting
//!
//! A single value can repeat more times than a `count` field can hold. Rather
//! than widen `count`, a run longer than [`MAX_RUN`] (or a caller-supplied cap
//! via [`encode_capped`]) is split into several consecutive pairs that share
//! the same `value`. Concatenating their counts recovers the full run length,
//! and [`decode`] stitches them back together transparently. Emitted counts are
//! therefore always in `1..=cap`; a zero count is never produced.
//!
//! ## Boundaries
//!
//! This module owns *only* run-length semantics. It is deliberately distinct
//! from its neighbours:
//!
//! * [`super::compression`] performs lossy numeric quantization (`fp16`,
//!   `snorm`/`unorm`, octahedral) on individual values; it never groups equal
//!   values into runs.
//! * [`super::gpu_stream_append`] and [`super::gpu_compact`] are GPU
//!   stream-compaction primitives: they drop dead slots and pack survivors, but
//!   preserve each survivor verbatim and assign no `(value, count)` structure.
//! * Delta / zig-zag coding (authored separately this batch) rewrites values as
//!   signed differences; it is orthogonal to and composable with `RLE`, but is
//!   not implemented here.

use alloc::vec::Vec;

/// Largest run length a single [`Run`] can hold. Runs longer than this are
/// split across multiple consecutive pairs sharing the same `value`.
pub const MAX_RUN: u32 = u32::MAX;

/// One run-length pair: a `value` repeated `count` times.
///
/// A `count` of `0` is never emitted by [`encode`]/[`encode_capped`]; if such a
/// pair is handed to [`decode`] it is treated as an empty (skipped) run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Run {
    /// The repeated value.
    pub value: u32,
    /// How many times `value` repeats in this run.
    pub count: u32,
}

impl Run {
    /// Builds a run pairing `value` with `count`.
    #[must_use]
    pub const fn new(value: u32, count: u32) -> Self {
        Self { value, count }
    }
}

/// Encodes `values` into run-length pairs, splitting any run longer than
/// [`MAX_RUN`] into consecutive pairs that share the same `value`.
///
/// The result is *canonical*: no emitted count is `0`, and no two adjacent
/// pairs share the same `value` unless the earlier pair is saturated at the
/// cap (see [`is_canonical`]).
#[must_use]
pub fn encode(values: &[u32]) -> Vec<Run> {
    encode_capped(values, MAX_RUN)
}

/// Like [`encode`], but caps each emitted run at `max_run` elements. A
/// `max_run` of `0` is meaningless and is treated as `1` so encoding always
/// makes progress and never emits a zero-count run.
#[must_use]
pub fn encode_capped(values: &[u32], max_run: u32) -> Vec<Run> {
    let cap = max_run.max(1);
    let mut runs: Vec<Run> = Vec::new();

    let mut iter = values.iter().copied();
    let Some(mut current) = iter.next() else {
        return runs;
    };
    let mut count: u32 = 1;

    for value in iter {
        if value == current && count < cap {
            count += 1;
        } else {
            runs.push(Run::new(current, count));
            current = value;
            count = 1;
        }
    }
    runs.push(Run::new(current, count));
    runs
}

/// Decodes run-length pairs back into the flat `u32` stream. Pairs with a
/// `count` of `0` contribute nothing, so malformed input degrades gracefully
/// rather than panicking.
#[must_use]
pub fn decode(runs: &[Run]) -> Vec<u32> {
    let total = decoded_len(runs);
    let mut out: Vec<u32> = Vec::with_capacity(usize::try_from(total).unwrap_or(usize::MAX));
    for run in runs {
        out.extend(core::iter::repeat_n(run.value, run.count as usize));
    }
    out
}

/// Total number of elements [`decode`] would produce for `runs`, computed in
/// `u64` so summing many saturated counts cannot overflow.
#[must_use]
pub fn decoded_len(runs: &[Run]) -> u64 {
    runs.iter().map(|run| u64::from(run.count)).sum()
}

/// Reports whether `runs` is a canonical encoding: every count lies in
/// `1..=cap`, and no two adjacent pairs share the same `value` unless the
/// earlier one is saturated at `cap` (a legitimate split of an over-long run).
#[must_use]
pub fn is_canonical(runs: &[Run], cap: u32) -> bool {
    let cap = cap.max(1);
    let mut prev: Option<Run> = None;
    for &run in runs {
        if run.count == 0 || run.count > cap {
            return false;
        }
        if let Some(previous) = prev
            && previous.value == run.value
            && previous.count < cap
        {
            // Adjacent equal values are only allowed when the earlier pair was
            // forced to split because it hit the cap.
            return false;
        }
        prev = Some(run);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ---- encode: shape / edge cases -------------------------------------

    #[test]
    fn encode_empty_yields_no_runs() {
        assert!(encode(&[]).is_empty());
    }

    #[test]
    fn encode_single_element() {
        assert_eq!(encode(&[7]), vec![Run::new(7, 1)]);
    }

    #[test]
    fn encode_two_equal_elements() {
        assert_eq!(encode(&[9, 9]), vec![Run::new(9, 2)]);
    }

    #[test]
    fn encode_two_distinct_elements() {
        assert_eq!(encode(&[3, 4]), vec![Run::new(3, 1), Run::new(4, 1)]);
    }

    #[test]
    fn encode_all_identical_collapses_to_one_run() {
        assert_eq!(encode(&[5, 5, 5, 5, 5]), vec![Run::new(5, 5)]);
    }

    #[test]
    fn encode_no_repeats_yields_unit_runs() {
        let values = [1u32, 2, 3, 4];
        let runs = encode(&values);
        assert_eq!(
            runs,
            vec![
                Run::new(1, 1),
                Run::new(2, 1),
                Run::new(3, 1),
                Run::new(4, 1),
            ]
        );
    }

    #[test]
    fn encode_alternating_never_groups() {
        let values = [0u32, 1, 0, 1, 0, 1];
        let runs = encode(&values);
        assert_eq!(runs.len(), 6);
        assert!(runs.iter().all(|run| run.count == 1));
    }

    #[test]
    fn encode_mixed_run_lengths() {
        let values = [2u32, 2, 2, 8, 8, 5];
        assert_eq!(
            encode(&values),
            vec![Run::new(2, 3), Run::new(8, 2), Run::new(5, 1)]
        );
    }

    #[test]
    fn encode_never_emits_zero_count() {
        let values = [0u32, 0, 1, 2, 2, 2, 3];
        assert!(encode(&values).iter().all(|run| run.count > 0));
    }

    #[test]
    fn encode_output_is_canonical() {
        let values = [4u32, 4, 4, 1, 1, 9, 9, 9, 9, 0];
        let runs = encode(&values);
        assert!(is_canonical(&runs, MAX_RUN));
    }

    // ---- run splitting via a small cap ----------------------------------

    #[test]
    fn encode_capped_splits_long_run() {
        let values = [7u32; 10];
        let runs = encode_capped(&values, 4);
        assert_eq!(runs, vec![Run::new(7, 4), Run::new(7, 4), Run::new(7, 2)]);
    }

    #[test]
    fn encode_capped_exact_multiple_of_cap() {
        let values = [1u32; 6];
        let runs = encode_capped(&values, 3);
        assert_eq!(runs, vec![Run::new(1, 3), Run::new(1, 3)]);
    }

    #[test]
    fn encode_capped_cap_plus_one() {
        let values = [2u32; 5];
        let runs = encode_capped(&values, 4);
        assert_eq!(runs, vec![Run::new(2, 4), Run::new(2, 1)]);
    }

    #[test]
    fn encode_capped_cap_one_makes_all_unit_runs() {
        let values = [5u32, 5, 5];
        let runs = encode_capped(&values, 1);
        assert_eq!(runs.len(), 3);
        assert!(runs.iter().all(|run| run.count == 1));
    }

    #[test]
    fn encode_capped_zero_cap_is_treated_as_one() {
        let values = [8u32, 8, 8, 8];
        let runs = encode_capped(&values, 0);
        assert_eq!(runs.len(), 4);
        assert!(runs.iter().all(|run| run.value == 8 && run.count == 1));
    }

    #[test]
    fn encode_capped_split_run_is_canonical() {
        let values = [3u32; 9];
        let runs = encode_capped(&values, 4);
        assert!(is_canonical(&runs, 4));
    }

    // ---- decode ----------------------------------------------------------

    #[test]
    fn decode_empty_yields_no_values() {
        assert!(decode(&[]).is_empty());
    }

    #[test]
    fn decode_single_run() {
        assert_eq!(decode(&[Run::new(6, 3)]), vec![6, 6, 6]);
    }

    #[test]
    fn decode_multiple_runs() {
        let runs = [Run::new(1, 2), Run::new(9, 1), Run::new(4, 3)];
        assert_eq!(decode(&runs), vec![1, 1, 9, 4, 4, 4]);
    }

    #[test]
    fn decode_skips_zero_count_runs() {
        let runs = [Run::new(1, 2), Run::new(7, 0), Run::new(3, 1)];
        assert_eq!(decode(&runs), vec![1, 1, 3]);
    }

    #[test]
    fn decode_all_zero_counts_is_empty() {
        let runs = [Run::new(1, 0), Run::new(2, 0)];
        assert!(decode(&runs).is_empty());
    }

    // ---- decoded_len -----------------------------------------------------

    #[test]
    fn decoded_len_empty_is_zero() {
        assert_eq!(decoded_len(&[]), 0);
    }

    #[test]
    fn decoded_len_sums_counts() {
        let runs = [
            Run::new(0, 5),
            Run::new(1, 2),
            Run::new(2, 0),
            Run::new(3, 8),
        ];
        assert_eq!(decoded_len(&runs), 15);
    }

    #[test]
    fn decoded_len_matches_decode_length() {
        let values = [4u32, 4, 4, 4, 9, 9, 1];
        let runs = encode(&values);
        assert_eq!(decoded_len(&runs), decode(&runs).len() as u64);
    }

    #[test]
    fn decoded_len_does_not_overflow_on_saturated_runs() {
        let runs = [Run::new(0, MAX_RUN), Run::new(1, MAX_RUN)];
        assert_eq!(decoded_len(&runs), 2 * u64::from(MAX_RUN));
    }

    // ---- is_canonical ----------------------------------------------------

    #[test]
    fn is_canonical_rejects_zero_count() {
        assert!(!is_canonical(&[Run::new(1, 0)], MAX_RUN));
    }

    #[test]
    fn is_canonical_rejects_count_above_cap() {
        assert!(!is_canonical(&[Run::new(1, 5)], 4));
    }

    #[test]
    fn is_canonical_rejects_unsaturated_adjacent_equal_values() {
        let runs = [Run::new(2, 1), Run::new(2, 1)];
        assert!(!is_canonical(&runs, 4));
    }

    #[test]
    fn is_canonical_accepts_saturated_adjacent_equal_values() {
        let runs = [Run::new(2, 4), Run::new(2, 1)];
        assert!(is_canonical(&runs, 4));
    }

    // ---- round trips -----------------------------------------------------

    #[test]
    fn round_trip_all_identical() {
        let values = [5u32; 32];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn round_trip_distinct() {
        let values: Vec<u32> = (0..20).collect();
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn round_trip_alternating() {
        let values = [0u32, 1, 0, 1, 0, 1, 0];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn round_trip_survives_cap_splitting() {
        let values = [42u32; 37];
        let runs = encode_capped(&values, 5);
        assert!(runs.len() > 1, "expected the long run to split");
        assert_eq!(decode(&runs), values);
    }

    #[test]
    fn round_trip_with_extreme_values() {
        let values = [u32::MAX, u32::MAX, 0, 0, 0, u32::MAX];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn round_trip_deterministic_pseudorandom_small_alphabet() {
        // A small LCG folded into a tiny alphabet so runs of length > 1 occur.
        let mut state: u32 = 0x1234_5678;
        let mut values: Vec<u32> = Vec::with_capacity(500);
        for _ in 0..500 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            values.push(state >> 29); // 0..=7
        }
        let runs = encode(&values);
        assert!(is_canonical(&runs, MAX_RUN));
        assert_eq!(decode(&runs), values);
    }

    #[test]
    fn round_trip_capped_matches_uncapped_stream() {
        let mut state: u32 = 0x9E37_79B9;
        let mut values: Vec<u32> = Vec::with_capacity(300);
        for _ in 0..300 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            values.push(state >> 30); // 0..=3, lots of repeats
        }
        let capped = encode_capped(&values, 6);
        assert!(is_canonical(&capped, 6));
        assert_eq!(decode(&capped), values);
        // Capping only splits runs; it must never change the decoded stream.
        assert_eq!(decode(&capped), decode(&encode(&values)));
    }
}
