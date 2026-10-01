//! Real-device parity for the one-dimensional interval-algebra twin:
//! [`GpuIntervalOverlap1d`](prism_volumetric_gpu::interval_overlap_1d::GpuIntervalOverlap1d)
//! must reproduce the `CPU` golden
//! [`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d)
//! across the full degeneracy map: the empty sentinel, a degenerate point
//! interval, overlapping / disjoint / containing / exactly-touching pairs, both
//! signs of `gap`, an `expand` that shrinks a non-empty interval down to a
//! collapsed point, the all-covering `[-inf, +inf]` interval, and a randomized
//! batch of clearly-conditioned finite pairs compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The four boolean predicates and every emptiness verdict are driven purely by
//! `f32` orderings, so they must match *exactly* and are asserted with `==`. The
//! finite continuous fields are a fixed, non-reorderable sequence of orderings,
//! `min` / `max` / `clamp` selections and at most one add, subtract and halving;
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate, so they
//! are asserted with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The `±inf` and
//! `NaN` sentinels (the empty / all-covering bounds and the `NaN` midpoint of an
//! unbounded interval) are compared by exact classification, not tolerance.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of a boundary tie and of the degeneracy cracks:
//! predicate pairs are either clearly overlapping (shared sub-interval with a
//! margin) or clearly disjoint (a gap margin of at least one unit), the touching
//! pair uses integer coordinates whose shared endpoint is bit-identical on both
//! devices, the collapsing `expand` uses integer bounds and an amount whose
//! over-shrink is unambiguous, and the randomized sweep draws finite half-widths
//! of at least one unit and a non-negative grow amount so no `expand` lands on
//! the collapse boundary. This keeps `CPU` and `GPU` on the same side of every
//! branch regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓
//! [`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d)；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::interval_overlap_1d::Interval;
use prism_volumetric_gpu::interval_overlap_1d::{
    GpuIntervalOverlap1d, IntervalOverlapQuery, IntervalOverlapResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree. Non-finite values (the `±inf` sentinels
/// and the `NaN` midpoint) are compared by exact classification: both `NaN`, or
/// an identical infinite bit pattern. Finite values use the absolute-or-relative
/// tolerance.
fn close(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a.to_bits() == b.to_bits();
    }
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two intervals agree bound-for-bound within the parity rule.
fn close_interval(label: &str, idx: usize, got: Interval, want: Interval) {
    assert!(
        close(got.min, want.min) && close(got.max, want.max),
        "query {idx} {label}: gpu [{}, {}] vs cpu [{}, {}]",
        got.min,
        got.max,
        want.min,
        want.max
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Builds a clearly-conditioned query. Even draws produce a concentric pair
/// (both intervals share a centre, so they clearly overlap and intersect with a
/// margin of at least one unit); odd draws produce a clearly-disjoint pair with
/// a gap margin of at least one unit. The grow amount is non-negative so
/// `expand` never approaches the collapse boundary, and `value` is `a`'s centre
/// so `contains` is unambiguous.
fn sweep_query(state: &mut u64, disjoint: bool) -> IntervalOverlapQuery {
    let centre = signed(state, 50.0);
    // Half-widths at least one unit so every interval has clear interior.
    let ha = 1.0 + lcg(state) * 4.0;
    let delta = signed(state, 20.0);
    let amount = lcg(state) * 3.0;
    if disjoint {
        let hb = 1.0 + lcg(state) * 4.0;
        // Separate b from a by a gap margin of at least one unit.
        let gap = 1.0 + lcg(state) * 4.0;
        let a = Interval::new(centre - ha, centre + ha);
        let b_lo = centre + ha + gap;
        let b = Interval::new(b_lo, b_lo + 2.0 * hb);
        IntervalOverlapQuery::new(a, b, centre, amount, delta)
    } else {
        // b shares a's centre with a smaller half-width, so b is contained in a
        // (clear overlap and a clear, non-empty intersection).
        let hb = 0.25 + lcg(state) * (ha - 0.5);
        let a = Interval::new(centre - ha, centre + ha);
        let b = Interval::new(centre - hb, centre + hb);
        IntervalOverlapQuery::new(a, b, centre, amount, delta)
    }
}

/// Computes the reference answer for one query by calling the `CPU` golden
/// `Interval` methods directly, so the comparison runs the exact twinned path.
fn golden(query: &IntervalOverlapQuery) -> IntervalOverlapResult {
    IntervalOverlapResult {
        is_empty: query.a.is_empty(),
        contains: query.a.contains(query.value),
        contains_interval: query.a.contains_interval(query.b),
        overlaps: query.a.overlaps(query.b),
        length: query.a.length(),
        center: query.a.center(),
        clamp_value: query.a.clamp_value(query.value),
        gap: query.a.gap(query.b),
        intersect: query.a.intersect(query.b),
        hull: query.a.hull(query.b),
        expand: query.a.expand(query.amount),
        translate: query.a.translate(query.delta),
    }
}

/// Pins one `GPU` result against the `CPU` golden: exact on the predicates, the
/// parity rule on every continuous field and interval bound.
fn pin(idx: usize, query: &IntervalOverlapQuery, got: &IntervalOverlapResult) {
    let want = golden(query);

    assert_eq!(
        got.is_empty, want.is_empty,
        "query {idx} is_empty: gpu {} vs cpu {}",
        got.is_empty, want.is_empty
    );
    assert_eq!(
        got.contains, want.contains,
        "query {idx} contains: gpu {} vs cpu {}",
        got.contains, want.contains
    );
    assert_eq!(
        got.contains_interval, want.contains_interval,
        "query {idx} contains_interval: gpu {} vs cpu {}",
        got.contains_interval, want.contains_interval
    );
    assert_eq!(
        got.overlaps, want.overlaps,
        "query {idx} overlaps: gpu {} vs cpu {}",
        got.overlaps, want.overlaps
    );

    assert!(
        close(got.length, want.length),
        "query {idx} length: gpu {} vs cpu {}",
        got.length,
        want.length
    );
    assert!(
        close(got.center, want.center),
        "query {idx} center: gpu {} vs cpu {}",
        got.center,
        want.center
    );
    assert!(
        close(got.clamp_value, want.clamp_value),
        "query {idx} clamp_value: gpu {} vs cpu {}",
        got.clamp_value,
        want.clamp_value
    );
    assert!(
        close(got.gap, want.gap),
        "query {idx} gap: gpu {} vs cpu {}",
        got.gap,
        want.gap
    );

    close_interval("intersect", idx, got.intersect, want.intersect);
    close_interval("hull", idx, got.hull, want.hull);
    close_interval("expand", idx, got.expand, want.expand);
    close_interval("translate", idx, got.translate, want.translate);
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuIntervalOverlap1d, queries: &[IntervalOverlapQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn empty_interval_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // The empty sentinel [+inf, -inf]: is_empty true, length 0, a NaN centre, a
    // +inf clamp_value, and intersect / expand / translate all return empty.
    let query = IntervalOverlapQuery::new(
        Interval::empty(),
        Interval::new(1.0, 2.0),
        0.5,
        1.0,
        3.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn point_interval_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // A degenerate point interval: zero length, centre at the point, and a
    // containing b so contains_interval and overlaps both fire.
    let query = IntervalOverlapQuery::new(
        Interval::point(3.0),
        Interval::new(0.0, 6.0),
        3.0,
        2.0,
        -1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn overlapping_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // Clearly overlapping, neither contained: overlaps true, intersect finite,
    // gap zero, contains_interval false.
    let query = IntervalOverlapQuery::new(
        Interval::new(0.0, 5.0),
        Interval::new(3.0, 8.0),
        2.5,
        1.0,
        4.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disjoint_pair_gap_forward_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // a entirely below b: a positive forward gap of three, an empty intersect,
    // overlaps false.
    let query = IntervalOverlapQuery::new(
        Interval::new(0.0, 1.0),
        Interval::new(4.0, 5.0),
        0.5,
        1.0,
        2.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn disjoint_pair_gap_reverse_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // b entirely below a: exercises the reverse branch of gap (a.min - b.max).
    let query = IntervalOverlapQuery::new(
        Interval::new(4.0, 5.0),
        Interval::new(0.0, 1.0),
        4.5,
        1.0,
        -2.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn containing_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // b wholly inside a: contains_interval true, intersect equals b, hull equals
    // a.
    let query = IntervalOverlapQuery::new(
        Interval::new(-2.0, 10.0),
        Interval::new(1.0, 4.0),
        0.0,
        0.5,
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn touching_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // Exactly tangent at the integer coordinate 2.0, bit-identical on both
    // devices: overlaps true (touching counts), gap zero, intersect a point.
    let query = IntervalOverlapQuery::new(
        Interval::new(0.0, 2.0),
        Interval::new(2.0, 4.0),
        1.0,
        1.0,
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn expand_collapse_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // expand(-2) over-shrinks [0, 2] (half-length one) well past collapse, so the
    // bounds invert and new() pins the result to the midpoint. Integer bounds and
    // an unambiguous over-shrink keep the branch deterministic.
    let query = IntervalOverlapQuery::new(
        Interval::new(0.0, 2.0),
        Interval::new(0.5, 1.5),
        1.0,
        -2.0,
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn everything_interval_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    // The all-covering [-inf, +inf]: not empty, +inf length, a NaN centre, a
    // finite clamp_value (clamp of a finite value into unbounded bounds), and a
    // finite b whose hull with everything is everything.
    let query = IntervalOverlapQuery::new(
        Interval::everything(),
        Interval::new(-3.0, 7.0),
        2.0,
        1.0,
        5.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic degeneracy fixtures with many
    // clearly-conditioned random pairs, dispatched together so the per-thread
    // indexing and the contiguous storage layout are both exercised, then pinned
    // element-for-element.
    let mut queries = vec![
        IntervalOverlapQuery::new(Interval::empty(), Interval::new(1.0, 2.0), 0.5, 1.0, 3.0),
        IntervalOverlapQuery::new(Interval::point(3.0), Interval::new(0.0, 6.0), 3.0, 2.0, -1.0),
        IntervalOverlapQuery::new(Interval::new(0.0, 2.0), Interval::new(2.0, 4.0), 1.0, 1.0, 0.5),
        IntervalOverlapQuery::new(
            Interval::everything(),
            Interval::new(-3.0, 7.0),
            2.0,
            1.0,
            5.0,
        ),
    ];
    for lane in 0..48 {
        queries.push(sweep_query(&mut state, lane % 2 == 0));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuIntervalOverlap1d::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) alternating clearly-overlapping
    // and clearly-disjoint pairs pins every reported field across many random
    // geometries.
    let queries: Vec<IntervalOverlapQuery> = (0..200)
        .map(|lane| sweep_query(&mut state, lane % 2 == 0))
        .collect();
    check(&ctx, &gpu, &queries);
}
