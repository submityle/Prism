//! Real-device parity for the `u16` counting-sort `histogram` twin:
//! [`GpuCountingSortU16`](prism_volumetric_gpu::counting_sort_u16::GpuCountingSortU16)
//! must reproduce the `CPU` golden
//! [`histogram`](prism_render_architecture::particle::counting_sort_u16::histogram)
//! across an empty batch (the all-zero histogram), a single key, a batch that
//! collapses entirely into one bucket (all keys identical), a full-range sparse
//! batch spread one key per bucket across the whole `u16` domain, and a large
//! pseudo-random batch (both broad-spread and heavily-clustered variants).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A histogram bucket count is an integer, so parity is asserted with **exact
//! per-bucket equality**, not a float tolerance: there is no floating point on
//! the path and therefore no `ULP` boundary or degenerate region, and any
//! mismatch is a genuine port bug (a dropped increment, a wrong mask, a
//! miscounted key). `WGSL` has no `u64`, and none appears here or in the twin;
//! the golden's `u64` lives only in its internal test `RNG` and is not ported.
//! This harness reuses that same `LCG` structure on `u16` keys so the fixtures
//! are reproducible across runs and platforms.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::counting_sort_u16`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::counting_sort_u16::{
    exclusive_prefix_sum, histogram, BUCKET_COUNT,
};
use prism_volumetric_gpu::counting_sort_u16::{CountingSortU16Query, GpuCountingSortU16};
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms. Only integer arithmetic
/// appears; the `u64` state never leaves this harness and is not part of the
/// twinned kernel.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws the next pseudo-random `u16` key from `state`.
fn next_u16(state: &mut u64) -> u16 {
    (lcg_next(state) >> 48) as u16
}

/// Runs the `GPU` histogram and asserts exact per-bucket parity against the
/// `CPU` golden [`histogram`], returning the `GPU` counts for extra assertions.
fn check(ctx: &GpuContext, gpu: &GpuCountingSortU16, keys: &[u16]) -> Vec<u32> {
    let query = CountingSortU16Query {
        keys: keys.to_vec(),
    };
    let got = gpu.eval(ctx, &query);
    let want = histogram(keys);

    assert_eq!(
        got.len(),
        want.len(),
        "histogram length must equal BUCKET_COUNT"
    );
    assert_eq!(got.len(), BUCKET_COUNT, "histogram length is BUCKET_COUNT");
    for (bucket, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g as usize, w,
            "bucket {bucket}: gpu count {g} vs cpu count {w}"
        );
    }
    got
}

#[test]
fn empty_input_is_the_zero_histogram() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // No key issues no dispatch; the result is the correctly sized zero
    // histogram and matches the reference exactly.
    let got = check(&ctx, &gpu, &[]);
    assert_eq!(got.len(), BUCKET_COUNT, "BUCKET_COUNT length is preserved");
    assert!(got.iter().all(|&c| c == 0), "every bucket is empty");
}

#[test]
fn single_key_lands_in_one_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    let got = check(&ctx, &gpu, &[40_000u16]);
    assert_eq!(got[40_000], 1, "the single key is counted once");
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        1,
        "no other bucket is touched"
    );
}

#[test]
fn all_identical_keys_collapse_to_one_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // Every key is the same, so one bucket holds the whole batch and the rest
    // stay zero. Includes the boundary values 0 and u16::MAX in sibling checks.
    for &key in &[0u16, 1, 12_345, u16::MAX] {
        let keys = vec![key; 5_000];
        let got = check(&ctx, &gpu, &keys);
        assert_eq!(got[key as usize], 5_000, "bucket {key} holds the batch");
        assert_eq!(
            got.iter().map(|&c| u64::from(c)).sum::<u64>(),
            5_000,
            "no key escapes bucket {key}"
        );
    }
}

#[test]
fn full_range_sparse_one_per_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // One key per bucket across the entire u16 domain: every bucket must end up
    // with exactly one, exercising the full 65536-bucket storage buffer.
    let keys: Vec<u16> = (0..BUCKET_COUNT).map(|k| k as u16).collect();
    let got = check(&ctx, &gpu, &keys);
    assert!(
        got.iter().all(|&c| c == 1),
        "each of the 65536 buckets receives exactly one key"
    );
}

#[test]
fn full_range_sparse_including_both_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // A hand-picked sparse set touching both ends of the domain and a few
    // interior buckets; the gaps between them stay empty.
    let keys = [0u16, u16::MAX, 1, 65_534, 32_768, 255, 256];
    let got = check(&ctx, &gpu, &keys);
    for key in keys {
        assert_eq!(got[key as usize], 1, "sparse key {key} is counted once");
    }
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        keys.len() as u64,
        "exactly the sparse keys are counted"
    );
}

#[test]
fn large_random_batch_conserves_totals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // A broad pseudo-random spread over the whole domain; parity is checked
    // bucket for bucket and the grand total must equal the key count.
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let keys: Vec<u16> = (0..40_000).map(|_| next_u16(&mut state)).collect();
    let got = check(&ctx, &gpu, &keys);
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        keys.len() as u64,
        "every key is counted exactly once"
    );
    assert!(
        got.iter().any(|&c| c > 1),
        "a 40k batch over 65536 buckets must collide somewhere"
    );
}

#[test]
fn clustered_random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCountingSortU16::new(&ctx);
    // Keys crushed into a handful of buckets so counts pile up high, then a
    // high-byte cluster: both stress a few heavily-incremented buckets.
    let mut state = 0x0bad_c0de_0f0f_0f0f_u64;
    let mut keys: Vec<u16> = (0..8_000).map(|_| next_u16(&mut state) % 11).collect();
    keys.extend((0..8_000).map(|_| next_u16(&mut state) & 0xff00));
    let got = check(&ctx, &gpu, &keys);
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        keys.len() as u64,
        "clustered keys are all counted"
    );
}

#[test]
fn exclusive_prefix_sum_matches_histogram_offsets() {
    // CPU-side cross-check of the golden's companion `exclusive_prefix_sum`:
    // offsets[k] is the running total of counts[..k], so the final running
    // total equals the key count. No `GPU` dispatch and no `u64` are involved.
    let mut state = 0x2468_1357_dead_beef_u64;
    let keys: Vec<u16> = (0..5_000).map(|_| next_u16(&mut state) % 97).collect();
    let counts = histogram(&keys);
    let offsets = exclusive_prefix_sum(&counts);
    assert_eq!(offsets.len(), counts.len(), "offsets match counts length");
    assert_eq!(offsets[0], 0, "the first offset is always zero");
    let mut running = 0usize;
    for (offset, &count) in offsets.iter().zip(counts.iter()) {
        assert_eq!(*offset, running, "offset is the exclusive running total");
        running += count;
    }
    assert_eq!(running, keys.len(), "offsets account for every key");
}
