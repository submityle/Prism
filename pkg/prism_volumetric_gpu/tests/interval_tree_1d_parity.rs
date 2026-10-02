//! Real-device parity for the augmented `1D` interval-tree count twin:
//! [`GpuIntervalTree1d`](prism_volumetric_gpu::interval_tree_1d::GpuIntervalTree1d)
//! must reproduce the `CPU` golden
//! [`interval_tree_1d`](prism_render_architecture::particle::interval_tree_1d)
//! stabbing and overlap *counts* exactly:
//! [`count_point`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::count_point)
//! and
//! [`query_overlap`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::query_overlap)
//! length, for every query in a batch.
//!
//! The fixtures mirror the degenerate and boundary shapes the golden unit tests
//! call out: the empty tree, a single interval (point inside, below, above),
//! shared touching endpoints, fully overlapping identical spans, disjoint sets
//! and their gaps, duplicate intervals, negative coordinates, a sequential
//! unit-interval window, a staircase of nested spans, and a larger
//! deterministic set built with the same Numerical-Recipes `LCG` the reference
//! uses so the fixture stays pure integer and needs no external math library.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion and scope
//!
//! The golden coordinates are `i64`; `WGSL` has no `i64`, so this twin covers
//! the `i32` subset with `|lo|, |hi| <= COORD_LIMIT` (one million). Within that
//! range every stab/overlap count is pure integer logic with no rounding, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on every
//! returned count, with no tolerance. Only the counts are twinned; the
//! index-set variants `query_point` / `query_overlap` return order-unspecified,
//! variable-length `Vec<usize>` lists and stay host-side on the golden.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::interval_tree_1d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::interval_tree_1d::{Interval, IntervalTree};
use prism_volumetric_gpu::interval_tree_1d::{
    GpuIntervalTree1d, IntervalTree1dQuery, IntervalTree1dResult,
};
use prism_volumetric_gpu::GpuContext;

/// Builds the golden tree from an `i32` interval set, lifting endpoints to the
/// reference `i64` coordinate space.
fn golden(data: &[(i32, i32)]) -> IntervalTree {
    let ivs: Vec<Interval> = data
        .iter()
        .map(|&(a, b)| Interval::new(i64::from(a), i64::from(b)))
        .collect();
    IntervalTree::build(&ivs)
}

/// `CPU` golden stabbing counts for a whole point batch.
fn cpu_count_point(data: &[(i32, i32)], points: &[i32]) -> Vec<u32> {
    let t = golden(data);
    points
        .iter()
        .map(|&p| t.count_point(i64::from(p)) as u32)
        .collect()
}

/// `CPU` golden overlap counts for a whole interval-query batch.
fn cpu_count_overlap(data: &[(i32, i32)], queries: &[(i32, i32)]) -> Vec<u32> {
    let t = golden(data);
    queries
        .iter()
        .map(|&(lo, hi)| {
            t.query_overlap(Interval::new(i64::from(lo), i64::from(hi)))
                .len() as u32
        })
        .collect()
}

/// Flattens result structs to their raw counts for an exact `==` comparison.
fn counts(results: &[IntervalTree1dResult]) -> Vec<u32> {
    results.iter().map(|r| r.count).collect()
}

/// Asserts the stabbing-count kernel matches the golden for a point batch.
fn assert_point_parity(
    gpu: &GpuIntervalTree1d,
    ctx: &GpuContext,
    data: &[(i32, i32)],
    points: &[i32],
) {
    let queries: Vec<IntervalTree1dQuery> = points
        .iter()
        .map(|&p| IntervalTree1dQuery::point(p))
        .collect();
    let got = counts(&gpu.count_point(ctx, data, &queries));
    assert_eq!(
        got,
        cpu_count_point(data, points),
        "count_point mismatch for data {data:?}"
    );
}

/// Asserts the overlap-count kernel matches the golden for an interval batch.
fn assert_overlap_parity(
    gpu: &GpuIntervalTree1d,
    ctx: &GpuContext,
    data: &[(i32, i32)],
    queries: &[(i32, i32)],
) {
    let packed: Vec<IntervalTree1dQuery> = queries
        .iter()
        .map(|&(lo, hi)| IntervalTree1dQuery::interval(lo, hi))
        .collect();
    let got = counts(&gpu.count_overlap(ctx, data, &packed));
    assert_eq!(
        got,
        cpu_count_overlap(data, queries),
        "count_overlap mismatch for data {data:?}"
    );
}

/// Shared reference interval set used by the hand-computed golden tests.
///
/// Index map:
/// `0:[15,20] 1:[10,30] 2:[17,19] 3:[5,20] 4:[12,15] 5:[30,40] 6:[25,30] 7:[0,3]`.
fn reference_set() -> [(i32, i32); 8] {
    [
        (15, 20),
        (10, 30),
        (17, 19),
        (5, 20),
        (12, 15),
        (30, 40),
        (25, 30),
        (0, 3),
    ]
}

#[test]
fn empty_tree_counts_are_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    assert_point_parity(&gpu, &ctx, &[], &[-5, 0, 42, 1000]);
    assert_overlap_parity(&gpu, &ctx, &[], &[(-5, 5), (0, 0), (100, 200)]);
}

#[test]
fn single_interval_inside_below_above() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(2, 7)];
    assert_point_parity(&gpu, &ctx, &data, &[1, 2, 5, 7, 8, 9]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(0, 1), (2, 2), (5, 9), (8, 10)]);
}

#[test]
fn shared_touching_endpoint_both_stabbed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    // [0,5] and [5,10] touch exactly at 5; a closed stab hits both.
    let data = [(0, 5), (5, 10)];
    assert_point_parity(&gpu, &ctx, &data, &[4, 5, 6]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(5, 5), (3, 7), (11, 12)]);
}

#[test]
fn fully_overlapping_identical_spans() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(1, 9), (1, 9), (1, 9)];
    assert_point_parity(&gpu, &ctx, &data, &[0, 1, 5, 9, 10]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(4, 5), (0, 0), (9, 20)]);
}

#[test]
fn disjoint_set_points_and_gaps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(0, 2), (5, 7), (10, 12)];
    assert_point_parity(&gpu, &ctx, &data, &[1, 3, 6, 8, 11, 13]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(3, 4), (2, 5), (0, 12)]);
}

#[test]
fn duplicate_intervals_counted_each() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(2, 8), (2, 8), (2, 8), (2, 8)];
    assert_point_parity(&gpu, &ctx, &data, &[1, 2, 5, 8, 9]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(4, 5), (8, 8), (9, 10)]);
}

#[test]
fn negative_coordinates_stab_and_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(-50, -40), (-45, -5), (-10, 10)];
    assert_point_parity(&gpu, &ctx, &data, &[-45, -8, -41, 0, 10, 11]);
    assert_overlap_parity(&gpu, &ctx, &data, &[(-42, -42), (-100, -60), (-10, -10)]);
}

#[test]
fn reference_set_stab_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = reference_set();
    // Sweep every integer across and beyond the set's span.
    let points: Vec<i32> = (-5..=45).collect();
    assert_point_parity(&gpu, &ctx, &data, &points);
}

#[test]
fn reference_set_overlap_windows() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = reference_set();
    let queries = [
        (16, 18),
        (3, 5),
        (25, 25),
        (0, 40),
        (41, 50),
        (-10, -1),
        (30, 30),
    ];
    assert_overlap_parity(&gpu, &ctx, &data, &queries);
}

#[test]
fn sequential_unit_intervals_window() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    // 64 unit-length intervals [i,i]; a window [10,13] hits exactly 4.
    let data: Vec<(i32, i32)> = (0..64).map(|i| (i, i)).collect();
    assert_overlap_parity(&gpu, &ctx, &data, &[(10, 13), (0, 0), (63, 63), (70, 80)]);
    let points: Vec<i32> = (-2..=66).collect();
    assert_point_parity(&gpu, &ctx, &data, &points);
}

#[test]
fn nested_staircase_stab_counts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    // Interval i covers [0, i], so point p is covered by indices p..=n-1.
    let data: Vec<(i32, i32)> = (0..40).map(|i| (0, i)).collect();
    let points: Vec<i32> = (-1..=41).collect();
    assert_point_parity(&gpu, &ctx, &data, &points);
    assert_overlap_parity(&gpu, &ctx, &data, &[(0, 0), (10, 10), (39, 39), (40, 40)]);
}

/// A small deterministic linear-congruential generator so the large-set test
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

    /// Draws an `i32` in `-span..=span`.
    fn next_in(&mut self, span: i32) -> i32 {
        let m = (span as u32) * 2 + 1;
        (self.next_u32() % m) as i32 - span
    }
}

#[test]
fn large_deterministic_set_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    // Build a few hundred intervals with mixed signs and widths, then probe a
    // large batch of points and overlap windows, including out-of-span values.
    let mut rng = Lcg::new(0x1234_5678);
    let data: Vec<(i32, i32)> = (0..512)
        .map(|_| {
            let a = rng.next_in(2000);
            let w = (rng.next_u32() % 400) as i32;
            (a, a + w)
        })
        .collect();

    let mut probe = Lcg::new(0x9abc_def0);
    let points: Vec<i32> = (0..600).map(|_| probe.next_in(2200)).collect();
    assert_point_parity(&gpu, &ctx, &data, &points);

    let queries: Vec<(i32, i32)> = (0..600)
        .map(|_| {
            let a = probe.next_in(2200);
            let w = (probe.next_u32() % 300) as i32;
            (a, a + w)
        })
        .collect();
    assert_overlap_parity(&gpu, &ctx, &data, &queries);
}

#[test]
fn empty_query_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalTree1d::new(&ctx);
    let data = [(1, 3), (2, 5)];
    assert!(gpu.count_point(&ctx, &data, &[]).is_empty());
    assert!(gpu.count_overlap(&ctx, &data, &[]).is_empty());
}
