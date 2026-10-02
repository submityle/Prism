//! Real-device parity for the `quickselect` order-statistic twin:
//! [`GpuQuickselectU32`](prism_volumetric_gpu::quickselect_u32::GpuQuickselectU32)
//! must reproduce the `CPU` golden
//! [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
//! and
//! [`median`](prism_render_architecture::particle::quickselect_u32::median)
//! across an empty batch, a single element, an out-of-range rank, the minimum
//! and maximum ranks, odd- and even-length medians, an all-equal array, an
//! array of `u32::MAX` keys, duplicate keys selected at several ranks in one
//! batch, every rank of one source array, and a large pseudo-random batch of
//! varied lengths and ranks compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A selected key, a rank, a median and a validity flag are all integers, so
//! parity is asserted with **exact** `==` on every field of every lane: there
//! is no floating point on the path and therefore no `ULP` boundary, no
//! rounding and no tie band, and any mismatch is a genuine port bug (a wrong
//! pivot, a dropped swap, a miscounted rank). `WGSL` has no `u64`, and none
//! appears here or in the twin; the host-side `LCG` that drives the random
//! fixtures keeps its `u64` state in this harness only and is not part of the
//! twinned kernel.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::quickselect_u32`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quickselect_u32::{
    cpu_reference, GpuQuickselect, GpuQuickselectU32, QuickselectQuery, MAX_N,
};
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms. Only integer arithmetic
/// appears; the `u64` state never leaves this harness and is not part of the
/// twinned kernel.
fn lcg_next(state: &mut u64) -> u64 {
    // Numerical Recipes constants; full period over `u64`.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws the next pseudo-random `u32` key from `state`.
fn next_u32(state: &mut u64) -> u32 {
    (lcg_next(state) >> 32) as u32
}

/// Builds one query from a key slice and a rank.
fn query(values: &[u32], k: u32) -> QuickselectQuery {
    QuickselectQuery {
        values: values.to_vec(),
        k,
    }
}

/// Runs the `GPU` dispatch over `queries` and asserts strict lane-for-lane
/// parity against the `CPU` golden [`cpu_reference`]: every field of every
/// record matches exactly, with no tolerance. Returns the `GPU` verdicts for
/// extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuQuickselectU32,
    queries: &[QuickselectQuery],
) -> Vec<GpuQuickselect> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "one result record per input query"
    );
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        assert_eq!(*g, want, "lane {lane}: gpu {g:?} vs cpu {want:?}");
    }
    got
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn single_element_selects_itself() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Rank 0 of a one-element array is that element; the lower median is too.
    let got = check(&ctx, &gpu, &[query(&[42], 0)]);
    assert_eq!(got[0].count, 1, "one live key");
    assert_eq!(got[0].selected_valid, 1, "rank 0 is in range");
    assert_eq!(got[0].selected_value, 42, "the only key is selected");
    assert_eq!(got[0].median_valid, 1, "a single element has a median");
    assert_eq!(got[0].median_value, 42, "the median is the only key");
}

#[test]
fn empty_array_is_a_double_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // An empty key slot: both the selected rank and the median report `None`,
    // carried back as a zero validity flag, matching the reference short
    // circuits. The batch itself is non-empty so a dispatch still happens.
    let got = check(&ctx, &gpu, &[query(&[], 0)]);
    assert_eq!(got[0].count, 0, "no live keys");
    assert_eq!(got[0].selected_valid, 0, "an empty array has no rank");
    assert_eq!(got[0].median_valid, 0, "an empty array has no median");
}

#[test]
fn out_of_range_rank_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // `k == len` and `k > len` both fall out of range, so the selected rank is a
    // miss while the median of the non-empty array is still valid.
    let data = [7u32, 3, 9, 1];
    let got = check(&ctx, &gpu, &[query(&data, 4), query(&data, 99)]);
    for lane in &got {
        assert_eq!(lane.selected_valid, 0, "an out-of-range rank is a miss");
        assert_eq!(lane.median_valid, 1, "the array still has a median");
    }
}

#[test]
fn minimum_rank_is_the_smallest_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Rank 0 is the minimum key regardless of input order.
    let got = check(&ctx, &gpu, &[query(&[50, 10, 30, 20, 40], 0)]);
    assert_eq!(got[0].selected_valid, 1, "rank 0 is in range");
    assert_eq!(got[0].selected_value, 10, "rank 0 is the minimum");
}

#[test]
fn maximum_rank_is_the_largest_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // The last rank (`len - 1`) is the maximum key.
    let data = [50u32, 10, 30, 20, 40];
    let got = check(&ctx, &gpu, &[query(&data, data.len() as u32 - 1)]);
    assert_eq!(got[0].selected_valid, 1, "the last rank is in range");
    assert_eq!(got[0].selected_value, 50, "the last rank is the maximum");
}

#[test]
fn median_odd_length_is_the_middle_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Sorted 1 3 5 7 9; the unique middle key is 5.
    let got = check(&ctx, &gpu, &[query(&[5, 1, 9, 3, 7], 0)]);
    assert_eq!(got[0].median_valid, 1, "an odd array has a median");
    assert_eq!(got[0].median_value, 5, "the middle key is the median");
}

#[test]
fn median_even_length_is_the_lower_middle_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Sorted 1 2 3 4; the lower median is the sorted index `len / 2 - 1` -> 2.
    let got = check(&ctx, &gpu, &[query(&[4, 1, 3, 2], 0)]);
    assert_eq!(got[0].median_valid, 1, "an even array has a median");
    assert_eq!(
        got[0].median_value, 2,
        "the lower of the two middles is taken"
    );
}

#[test]
fn all_equal_keys_select_the_same_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Every key is identical, so every rank and the median resolve to that key;
    // the Lomuto cursor still partitions correctly when all keys tie the pivot.
    let data = [7u32; 9];
    let queries: Vec<QuickselectQuery> = (0..data.len() as u32).map(|k| query(&data, k)).collect();
    let got = check(&ctx, &gpu, &queries);
    for lane in &got {
        assert_eq!(lane.selected_valid, 1, "every rank is in range");
        assert_eq!(lane.selected_value, 7, "every rank is the tied value");
        assert_eq!(lane.median_value, 7, "the median is the tied value");
    }
}

#[test]
fn extreme_u32_values_are_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Keys at the ends of the `u32` domain: no float path, so the exact
    // comparison must survive `0` and `u32::MAX` without clamping or rounding.
    let data = [u32::MAX, 0, u32::MAX - 1, 1, u32::MAX / 2];
    let queries: Vec<QuickselectQuery> = (0..data.len() as u32).map(|k| query(&data, k)).collect();
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].selected_value, 0, "rank 0 is the minimum");
    assert_eq!(
        got[data.len() - 1].selected_value,
        u32::MAX,
        "the last rank is u32::MAX"
    );
}

#[test]
fn duplicate_keys_at_many_ranks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Heavy duplication with a couple of singletons; selecting every rank in one
    // batch stresses the pivot tree across repeated keys. The `check` helper
    // asserts exact parity against the reference for each lane.
    let data = [3u32, 1, 3, 2, 3, 1, 4, 2, 3, 0];
    let queries: Vec<QuickselectQuery> = (0..=data.len() as u32).map(|k| query(&data, k)).collect();
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].selected_value, 0, "rank 0 is the minimum");
    assert_eq!(
        got[data.len()].selected_valid,
        0,
        "the one-past-the-end rank is a miss"
    );
}

#[test]
fn every_rank_of_one_source_is_sorted_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Selecting every rank of one array must reproduce fully sorted order, since
    // each private copy is reordered independently of the others.
    let base = [30u32, 10, 20, 50, 40, 60, 0, 70];
    let mut sorted = base;
    sorted.sort_unstable();
    let queries: Vec<QuickselectQuery> = (0..base.len() as u32).map(|k| query(&base, k)).collect();
    let got = check(&ctx, &gpu, &queries);
    for (k, lane) in got.iter().enumerate() {
        assert_eq!(lane.selected_valid, 1, "rank {k} is in range");
        assert_eq!(lane.selected_value, sorted[k], "rank {k} is sorted order");
    }
}

#[test]
fn full_slot_at_max_n_is_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // Fill the whole fixed slot (`MAX_N` keys) and select a few ranks, including
    // the first, the lower-median index and the last, to exercise the slot edge.
    let mut state = 0x0f0f_0f0f_a5a5_a5a5_u64;
    let data: Vec<u32> = (0..MAX_N as u32).map(|_| next_u32(&mut state)).collect();
    let ranks = [
        0u32,
        (MAX_N as u32) / 2 - 1,
        (MAX_N as u32) / 2,
        MAX_N as u32 - 1,
    ];
    let queries: Vec<QuickselectQuery> = ranks.iter().map(|&k| query(&data, k)).collect();
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().all(|lane| lane.selected_valid == 1),
        "every in-range rank of a full slot is valid"
    );
}

#[test]
fn large_random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // A broad pseudo-random batch: lengths sweep 1..=MAX_N, values span the full
    // domain or a crushed domain so duplicates abound, and ranks include some
    // deliberately out of range. Parity is asserted lane for lane; the batch is
    // checked to exercise both a valid and an invalid selected flag.
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let mut queries: Vec<QuickselectQuery> = Vec::new();
    for len in 1..=MAX_N {
        // Alternate a wide domain with a crushed one so ties are common.
        let crush = len.is_multiple_of(3);
        let values: Vec<u32> = (0..len)
            .map(|_| {
                let v = next_u32(&mut state);
                if crush {
                    v % 5
                } else {
                    v
                }
            })
            .collect();
        // A rank inside the array and, every few lanes, one past the end.
        let in_range = (next_u32(&mut state) as usize % len) as u32;
        queries.push(query(&values, in_range));
        if len.is_multiple_of(4) {
            queries.push(query(&values, len as u32));
        }
    }
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got.iter().any(|lane| lane.selected_valid == 1),
        "the batch must contain at least one valid selection"
    );
    assert!(
        got.iter().any(|lane| lane.selected_valid == 0),
        "the batch must contain at least one out-of-range selection"
    );
    assert!(
        got.iter().all(|lane| lane.median_valid == 1),
        "every array in the batch is non-empty and has a median"
    );
}

#[test]
fn oversized_array_is_clamped_to_max_n() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuickselectU32::new(&ctx);
    // More keys than the slot holds: both host and device clamp to `MAX_N`, so
    // only the first `MAX_N` keys participate. The `check` helper pins the twin
    // to the reference, which clamps identically.
    let mut state = 0xdead_beef_0bad_c0de_u64;
    let data: Vec<u32> = (0..MAX_N + 32).map(|_| next_u32(&mut state)).collect();
    let got = check(
        &ctx,
        &gpu,
        &[query(&data, 0), query(&data, MAX_N as u32 / 2)],
    );
    for lane in &got {
        assert_eq!(lane.count, MAX_N as u32, "the live count clamps to MAX_N");
    }
}
