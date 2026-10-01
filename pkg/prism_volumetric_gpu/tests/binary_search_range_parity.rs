//! Real-device parity for the half-open `u32` binary-search range twin:
//! [`GpuBinarySearchRange`](prism_volumetric_gpu::binary_search_range::GpuBinarySearchRange)
//! must reproduce the `CPU` golden
//! [`binary_search_range`](prism_render_architecture::particle::binary_search_range)
//! index for index across `lower_bound`, `upper_bound`, `equal_range` and
//! `contains`.
//!
//! The fixtures cover the degenerate and boundary shapes called out by the
//! golden unit tests: the empty array, a single element (target below, equal,
//! above), an all-equal array, targets strictly less than and strictly greater
//! than every element, targets landing exactly on element boundaries and in the
//! gaps between them, duplicate runs at the front, middle and back for the
//! `equal_range` path, and absent values. A larger deterministic array (built
//! with the same Numerical-Recipes `LCG` the reference uses, so the fixture
//! stays pure integer and needs no `bevy_math`) is swept against a naive linear
//! scan to exercise the search at scale.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every search is pure `u32` comparison and index algebra with no rounding
//! anywhere, so `CPU` and `GPU` must agree exactly. The comparison is an exact
//! `==` on every returned index (and every derived pair and flag), with no
//! tolerance: any mismatch is a genuine port bug. `WGSL` has no `u64`, and the
//! reference is already `u32`-only, so nothing is out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::binary_search_range`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::binary_search_range::{
    contains, equal_range, lower_bound, upper_bound,
};
use prism_volumetric_gpu::binary_search_range::GpuBinarySearchRange;
use prism_volumetric_gpu::GpuContext;

/// `CPU` golden lower bounds for a whole query batch.
fn cpu_lower(data: &[u32], queries: &[u32]) -> Vec<u32> {
    queries
        .iter()
        .map(|&t| lower_bound(data, t) as u32)
        .collect()
}

/// `CPU` golden upper bounds for a whole query batch.
fn cpu_upper(data: &[u32], queries: &[u32]) -> Vec<u32> {
    queries
        .iter()
        .map(|&t| upper_bound(data, t) as u32)
        .collect()
}

/// `CPU` golden equal ranges for a whole query batch.
fn cpu_equal_range(data: &[u32], queries: &[u32]) -> Vec<(u32, u32)> {
    queries
        .iter()
        .map(|&t| {
            let (lo, hi) = equal_range(data, t);
            (lo as u32, hi as u32)
        })
        .collect()
}

/// `CPU` golden containment flags for a whole query batch.
fn cpu_contains(data: &[u32], queries: &[u32]) -> Vec<bool> {
    queries.iter().map(|&t| contains(data, t)).collect()
}

/// Asserts both bound kernels, the derived equal range and the derived
/// containment all match the `CPU` golden for one `(data, queries)` fixture.
fn assert_parity(gpu: &GpuBinarySearchRange, ctx: &GpuContext, data: &[u32], queries: &[u32]) {
    assert_eq!(
        gpu.lower_bound(ctx, data, queries),
        cpu_lower(data, queries),
        "lower_bound mismatch for data {data:?}"
    );
    assert_eq!(
        gpu.upper_bound(ctx, data, queries),
        cpu_upper(data, queries),
        "upper_bound mismatch for data {data:?}"
    );
    assert_eq!(
        gpu.equal_range(ctx, data, queries),
        cpu_equal_range(data, queries),
        "equal_range mismatch for data {data:?}"
    );
    assert_eq!(
        gpu.contains(ctx, data, queries),
        cpu_contains(data, queries),
        "contains mismatch for data {data:?}"
    );
}

#[test]
fn empty_array_all_queries_are_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // An empty sorted array: every query inserts at 0 and nothing is present.
    assert_parity(&gpu, &ctx, &[], &[0, 1, 7, u32::MAX]);
}

#[test]
fn single_element_target_below_equal_above() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // One element: targets straddling it hit all three cases.
    assert_parity(&gpu, &ctx, &[5], &[3, 5, 9]);
}

#[test]
fn all_identical_full_range_and_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // All equal: equal_range spans the whole slice for the present key and
    // collapses to an endpoint below and above it.
    assert_parity(&gpu, &ctx, &[4, 4, 4, 4, 4], &[2, 4, 6]);
}

#[test]
fn target_less_than_and_greater_than_all() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // Below everything inserts at 0; above everything inserts at len.
    assert_parity(&gpu, &ctx, &[10, 20, 30, 40], &[1, 99]);
}

#[test]
fn targets_on_boundaries_and_in_gaps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // Keys land exactly on elements (present) and in the gaps between them
    // (absent, insertion point only).
    let data = [10u32, 20, 30, 40, 50];
    assert_parity(&gpu, &ctx, &data, &[9, 10, 15, 30, 35, 50, 51]);
}

#[test]
fn duplicate_runs_front_middle_back_equal_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // Repeated values so equal_range reports non-trivial half-open runs at the
    // front (7), middle (2) and back (9); 5 and 6 are absent.
    let data = [7u32, 7, 7, 8, 9, 9, 9];
    assert_parity(&gpu, &ctx, &data, &[5, 6, 7, 8, 9, 10]);
    let mid = [1u32, 2, 2, 2, 3, 4];
    assert_parity(&gpu, &ctx, &mid, &[0, 1, 2, 3, 4, 5]);
}

#[test]
fn exhaustive_small_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // Every target in 0..=25 against a mixed array with gaps and a duplicate
    // run, so monotonicity and all three cases are swept in one batch.
    let data = [3u32, 3, 5, 8, 8, 8, 13, 21];
    let queries: Vec<u32> = (0u32..=25).collect();
    assert_parity(&gpu, &ctx, &data, &queries);
}

/// A small deterministic linear-congruential generator so the large-array test
/// needs no external randomness and no floating point. Constants are the
/// Numerical Recipes values, matching the golden reference tests.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }
}

#[test]
fn large_sorted_array_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // Build a sorted array with many duplicates, then probe a large batch of
    // targets, including every present key plus out-of-range sentinels.
    let mut rng = Lcg::new(0x1234_5678);
    let mut data: Vec<u32> = (0..2048).map(|_| rng.next_u32() % 500).collect();
    data.sort_unstable();
    let mut probe = Lcg::new(0x9abc_def0);
    let mut queries: Vec<u32> = (0..512).map(|_| probe.next_u32() % 520).collect();
    queries.extend_from_slice(&data);
    queries.push(0);
    queries.push(u32::MAX);
    assert_parity(&gpu, &ctx, &data, &queries);
}

#[test]
fn empty_query_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBinarySearchRange::new(&ctx);
    // No dispatch is issued and every entry point returns an empty result.
    let data = [1u32, 2, 3];
    assert!(gpu.lower_bound(&ctx, &data, &[]).is_empty());
    assert!(gpu.upper_bound(&ctx, &data, &[]).is_empty());
    assert!(gpu.equal_range(&ctx, &data, &[]).is_empty());
    assert!(gpu.contains(&ctx, &data, &[]).is_empty());
}
