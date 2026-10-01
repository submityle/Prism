//! Real-device parity for the 2D segment-intersection twin:
//! [`GpuSegmentIntersect2d`](prism_volumetric_gpu::segment_intersect_2d::GpuSegmentIntersect2d)
//! must reproduce the `CPU` golden
//! [`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d)
//! across proper interior crossings, boundary touches (shared endpoints and
//! `T`-junctions), `collinear` configurations (overlap, containment, endpoint
//! contact and disjoint), parallel-offset and crossing-outside pairs, zero-length
//! degenerate segments, and a randomized integer-coordinate batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The classification code (the `SegIntersect` discriminant) and the "do they
//! meet" boolean are pure sign/epsilon decisions, so the comparison asserts an
//! exact `==` on them. The crossing coordinate and the `(t, u)` parameters are a
//! fixed, non-reorderable sequence of multiplies, adds and divides; a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, so the comparison allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on those `f32` values.
//!
//! # Conditioning
//!
//! Every fixture uses integer (or exactly representable) coordinates, so each
//! cross product is an integer far from the compare epsilon and the orientation
//! signs are identical on both devices. This keeps `CPU` and `GPU` on the same
//! side of every classification branch regardless of a few units in the last
//! place of slack in the parametric coordinates. No fixture uses a
//! transcendental function to synthesize an angle.
//!
//! Provenance: twinned from this repository's
//! [`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::segment_intersect_2d::{
    intersect, intersect_params, line_intersection, SegIntersect,
};
use prism_volumetric_gpu::segment_intersect_2d::{
    GpuSegmentIntersect2d, SegmentIntersectQuery, SegmentIntersectResult, CODE_COLLINEAR,
    CODE_NONE, CODE_POINT,
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

/// Asserts two 2D points agree channel-for-channel within the parity bound.
fn close_point(label: &str, idx: usize, got: [f32; 2], want: [f32; 2]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]),
        "query {idx} {label}: gpu ({}, {}) vs cpu ({}, {})",
        got[0],
        got[1],
        want[0],
        want[1]
    );
}

/// The reference classification code for one [`SegIntersect`] verdict.
fn code_of(verdict: &SegIntersect) -> u32 {
    match verdict {
        SegIntersect::None => CODE_NONE,
        SegIntersect::Point(_) => CODE_POINT,
        SegIntersect::Collinear => CODE_COLLINEAR,
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw state's high bits.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 40) as u32
}

/// A pseudo-random integer coordinate in `[-span, span]`, kept exactly
/// representable in `f32` so every cross product stays integer-valued.
fn coord(state: &mut u64, span: i32) -> f32 {
    let modulus = (2 * span + 1) as u32;
    (lcg(state) % modulus) as i32 as f32 - span as f32
}

/// A pseudo-random integer-coordinate point.
fn rand_point(state: &mut u64, span: i32) -> [f32; 2] {
    [coord(state, span), coord(state, span)]
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the
/// classification code and the meet boolean exactly, the crossing coordinate and
/// the `(t, u)` parameters and the infinite-line crossing to within bound.
fn pin(idx: usize, query: &SegmentIntersectQuery, got: &SegmentIntersectResult) {
    let want = intersect(query.p1, query.p2, query.p3, query.p4);
    let want_code = code_of(&want);
    assert_eq!(
        got.code, want_code,
        "query {idx} code: gpu {} vs cpu {}",
        got.code, want_code
    );
    assert_eq!(
        got.meets,
        want_code != CODE_NONE,
        "query {idx} meets: gpu {} vs cpu {}",
        got.meets,
        want_code != CODE_NONE
    );
    if let SegIntersect::Point(want_point) = want {
        close_point("point", idx, got.point, want_point);
    }

    let want_params = intersect_params(query.p1, query.p2, query.p3, query.p4);
    assert_eq!(
        got.params.is_some(),
        want_params.is_some(),
        "query {idx} params presence: gpu {:?} vs cpu {:?}",
        got.params,
        want_params
    );
    if let (Some((gt, gu)), Some((wt, wu))) = (got.params, want_params) {
        assert!(
            close(gt, wt) && close(gu, wu),
            "query {idx} params: gpu ({gt}, {gu}) vs cpu ({wt}, {wu})"
        );
    }

    let want_line = line_intersection(query.p1, query.p2, query.p3, query.p4);
    assert_eq!(
        got.line_point.is_some(),
        want_line.is_some(),
        "query {idx} line presence: gpu {:?} vs cpu {:?}",
        got.line_point,
        want_line
    );
    if let (Some(gl), Some(wl)) = (got.line_point, want_line) {
        close_point("line_point", idx, gl, wl);
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuSegmentIntersect2d, queries: &[SegmentIntersectQuery]) {
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
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn proper_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // An X: the two segments straddle each other and cross at the origin.
    let query = SegmentIntersectQuery::new([-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn shared_endpoint_touch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    let query = SegmentIntersectQuery::new([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn t_junction_touch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // The endpoint p3 lies on the interior of segment p1 p2.
    let query = SegmentIntersectQuery::new([0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 3.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn parallel_offset_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // Parallel non-collinear: never meet; intersect_params is None.
    let query = SegmentIntersectQuery::new([0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn crossing_outside_segments_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // Their infinite lines meet, but neither segment reaches the crossing.
    let query = SegmentIntersectQuery::new([0.0, 0.0], [1.0, 0.0], [2.0, -1.0], [2.0, 1.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn collinear_overlap_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    let queries = [
        // Partial overlap.
        SegmentIntersectQuery::new([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]),
        // Fully contained.
        SegmentIntersectQuery::new([0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [2.0, 0.0]),
        // Endpoint contact (single collinear point).
        SegmentIntersectQuery::new([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]),
        // Disjoint collinear.
        SegmentIntersectQuery::new([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]),
        // Diagonal collinear overlap off the axes.
        SegmentIntersectQuery::new([0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_zero_length_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    let queries = [
        // Zero-length segment sitting on the other segment: a point contact.
        SegmentIntersectQuery::new([2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [4.0, 0.0]),
        // Zero-length segment off the other segment: no contact.
        SegmentIntersectQuery::new([2.0, 5.0], [2.0, 5.0], [0.0, 0.0], [4.0, 0.0]),
        // Both segments degenerate and coincident.
        SegmentIntersectQuery::new([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
        // Both segments degenerate and apart.
        SegmentIntersectQuery::new([1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    // One batch mixing the deterministic fixtures with constructed crossings,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        SegmentIntersectQuery::new([-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]),
        SegmentIntersectQuery::new([0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 3.0]),
        SegmentIntersectQuery::new([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]),
    ];
    // A fan of guaranteed proper crossings: a vertical bar crossed by lines
    // through the origin at integer slopes.
    for k in 1..=16 {
        let y = k as f32;
        queries.push(SegmentIntersectQuery::new(
            [-5.0, 0.0],
            [5.0, 0.0],
            [0.0, -y],
            [0.0, y],
        ));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_integer_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentIntersect2d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Several workgroups' worth of random integer-coordinate pairs. Integer
    // cross products are far from the compare epsilon, so every orientation sign
    // is identical on both devices and the two sides share each branch.
    let queries: Vec<SegmentIntersectQuery> = (0..200)
        .map(|_| {
            SegmentIntersectQuery::new(
                rand_point(&mut state, 10),
                rand_point(&mut state, 10),
                rand_point(&mut state, 10),
                rand_point(&mut state, 10),
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}
