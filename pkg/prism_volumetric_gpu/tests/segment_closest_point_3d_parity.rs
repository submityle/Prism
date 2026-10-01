//! Real-device parity for the 3D segment-closest-point twin:
//! [`GpuSegmentClosestPoint3d`](prism_volumetric_gpu::segment_closest_point_3d::GpuSegmentClosestPoint3d)
//! must reproduce the `CPU` golden
//! [`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d)
//! across interior skew pairs (both parameters strictly inside `[0, 1]`),
//! endpoint-clamped pairs (the line-line optimum falls past an endpoint so `s`
//! or `t` pins to a bound), exactly-parallel pairs (an integer-valued
//! determinant that is zero on both devices, exercising the parallel pin), a
//! point-vs-segment pair (one segment collapsed to a single point), and a
//! randomized batch of clearly-conditioned skew pairs compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: skew pairs have a determinant far above the compare
//! epsilon with both parameters clearly interior, endpoint-clamped pairs land
//! clearly past a bound, the parallel pair uses integer coordinates whose
//! determinant is exactly zero on both devices, and the collapsed segment has
//! coincident endpoints (squared length exactly zero). This keeps `CPU` and
//! `GPU` on the same side of every branch regardless of a few units in the last
//! place of slack.
//!
//! Provenance: twinned from this repository's
//! [`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::segment_closest_point_3d::{
    closest_point_on_segment, closest_points_between_segments, point_segment_distance_squared,
    Segment, Vec3,
};
use prism_volumetric_gpu::segment_closest_point_3d::{
    GpuSegmentClosestPoint3d, SegmentClosestQuery, SegmentClosestResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
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

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Builds a clearly-conditioned skew pair by rejection sampling: two segments
/// are drawn from well-spread endpoints and accepted only when the `CPU`
/// reference places both parameters comfortably interior (`s` and `t` in
/// `[0.08, 0.92]`), the two directions are far from parallel (their normalized
/// squared dot is below `0.64` so the determinant sits well above the compare
/// epsilon) and both segments clear a safe non-degenerate length. This keeps
/// every random fixture inside the general-case branch, far from the parallel
/// pin, the endpoint clamps and the degeneracy cracks, so `CPU` and `GPU` share
/// every branch. The query point sits near the first segment's interior midspan.
fn skew_query(state: &mut u64) -> SegmentClosestQuery {
    loop {
        let first = Segment::new(rand_vec(state, 5.0), rand_vec(state, 5.0));
        let second = Segment::new(rand_vec(state, 5.0), rand_vec(state, 5.0));

        let d1 = first.direction();
        let d2 = second.direction();
        let len1_sq = d1.length_squared();
        let len2_sq = d2.length_squared();
        // Both segments must be clearly non-degenerate.
        if len1_sq < 1.0 || len2_sq < 1.0 {
            continue;
        }
        // Directions must be far from parallel so the determinant is large.
        let dot12 = d1.dot(d2);
        let cos_sq = (dot12 * dot12) / (len1_sq * len2_sq);
        if cos_sq > 0.64 {
            continue;
        }

        let cp = closest_points_between_segments(&first, &second);
        // Both parameters must be comfortably interior, clear of every clamp.
        if !(0.08..=0.92).contains(&cp.s) || !(0.08..=0.92).contains(&cp.t) {
            continue;
        }

        let mid = first.a.plus(d1.scale(0.5)).plus(rand_vec(state, 1.0));
        return SegmentClosestQuery::new(first, second, mid);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the clamped
/// point-to-segment projection, the point-to-segment squared distance, and the
/// full segment-segment closest-point record must all agree within bound.
fn pin(idx: usize, query: &SegmentClosestQuery, got: &SegmentClosestResult) {
    let (want_seg_t, want_seg_point) = closest_point_on_segment(&query.first, query.point);
    let want_seg_dist = point_segment_distance_squared(&query.first, query.point);
    let want = closest_points_between_segments(&query.first, &query.second);

    assert!(
        close(got.seg_param, want_seg_t),
        "query {idx} seg_param: gpu {} vs cpu {}",
        got.seg_param,
        want_seg_t
    );
    close_vec("seg_point", idx, got.seg_point, want_seg_point);
    assert!(
        close(got.seg_distance_squared, want_seg_dist),
        "query {idx} seg_distance_squared: gpu {} vs cpu {}",
        got.seg_distance_squared,
        want_seg_dist
    );

    assert!(
        close(got.s, want.s),
        "query {idx} s: gpu {} vs cpu {}",
        got.s,
        want.s
    );
    assert!(
        close(got.t, want.t),
        "query {idx} t: gpu {} vs cpu {}",
        got.t,
        want.t
    );
    close_vec("point_on_first", idx, got.point_on_first, want.point_on_first);
    close_vec(
        "point_on_second",
        idx,
        got.point_on_second,
        want.point_on_second,
    );
    assert!(
        close(got.distance_squared, want.distance_squared),
        "query {idx} distance_squared: gpu {} vs cpu {}",
        got.distance_squared,
        want.distance_squared
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuSegmentClosestPoint3d, queries: &[SegmentClosestQuery]) {
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
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_skew_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    // Two clearly skew segments whose closest approach lands strictly inside
    // both segments; the query point projects to the first segment's interior.
    let query = SegmentClosestQuery::new(
        Segment::new(Vec3::new(-3.0, -1.0, 0.0), Vec3::new(3.0, 1.0, 0.0)),
        Segment::new(Vec3::new(0.0, -2.0, 2.0), Vec3::new(0.0, 2.0, 2.0)),
        Vec3::new(0.5, 0.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn endpoint_clamped_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    // The second segment sits off one end of the first, so the line-line optimum
    // falls clearly past an endpoint and the solver pins to a bound.
    let query = SegmentClosestQuery::new(
        Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)),
        Segment::new(Vec3::new(10.0, 3.0, 1.0), Vec3::new(12.0, 5.0, 2.0)),
        Vec3::new(9.0, 1.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn parallel_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    // Two parallel segments with integer coordinates: the determinant
    // a*e - b*b = 36*121 - 66*66 = 0 is exactly zero on both devices, so the
    // parallel pin fires identically. The query point projects onto the first.
    let query = SegmentClosestQuery::new(
        Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0)),
        Segment::new(Vec3::new(-2.0, 3.0, 0.0), Vec3::new(9.0, 3.0, 0.0)),
        Vec3::new(2.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn collapsed_second_segment_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    // The second segment has coincident endpoints (squared length exactly zero),
    // so it is a single point projected onto the first segment: the degenerate
    // branch fires deterministically on both devices.
    let query = SegmentClosestQuery::new(
        Segment::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
        Segment::new(Vec3::new(0.5, 3.0, 0.0), Vec3::new(0.5, 3.0, 0.0)),
        Vec3::new(1.0, 1.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random skew pairs,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        SegmentClosestQuery::new(
            Segment::new(Vec3::new(-3.0, -1.0, 0.0), Vec3::new(3.0, 1.0, 0.0)),
            Segment::new(Vec3::new(0.0, -2.0, 2.0), Vec3::new(0.0, 2.0, 2.0)),
            Vec3::new(0.5, 0.0, 1.0),
        ),
        SegmentClosestQuery::new(
            Segment::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)),
            Segment::new(Vec3::new(10.0, 3.0, 1.0), Vec3::new(12.0, 5.0, 2.0)),
            Vec3::new(9.0, 1.0, 0.0),
        ),
    ];
    for _ in 0..48 {
        queries.push(skew_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_skew_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentClosestPoint3d::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned skew pairs (several workgroups'
    // worth) pins every reported field across many random segment geometries.
    let queries: Vec<SegmentClosestQuery> = (0..200).map(|_| skew_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
