//! Real-device parity for the segment-triangle intersection twin:
//! [`GpuSegmentTriangleIntersect`](prism_volumetric_gpu::segment_triangle_intersect::GpuSegmentTriangleIntersect)
//! must reproduce the `CPU` golden
//! [`intersect`](prism_render_architecture::particle::segment_triangle_intersect::intersect)
//! across an interior crossing (a segment piercing the triangle well inside its
//! edges), a vertex-aligned crossing (full weight on one vertex), an edge-grazing
//! crossing (a `barycentric` weight at zero), several clean misses (outside the
//! face, too short to reach the plane, past the end, before the start, and
//! parallel/coplanar), a degenerate zero-area triangle, and a randomized batch of
//! clearly-conditioned interior crossings compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded reciprocal, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on the `f32` fields while pinning the hit-presence
//! classification exactly.
//!
//! # Conditioning
//!
//! Every randomized fixture is deliberately well away from the degenerate
//! regions and the inclusion boundaries: the triangle has a comfortable area,
//! the segment crosses the plane at an interior `barycentric` point with both
//! `u` and `v` bounded away from `0` and `1`, and the crossing lands near the
//! segment midpoint (`t` clearly inside `[0, 1]`). This keeps `CPU` and `GPU` on
//! the same side of every branch regardless of a few units in the last place of
//! slack.
//!
//! Provenance: twinned from this repository's
//! [`segment_triangle_intersect`](prism_render_architecture::particle::segment_triangle_intersect);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::segment_triangle_intersect::{
    intersect, Hit, Segment, Vec3,
};
use prism_volumetric_gpu::segment_triangle_intersect::{
    GpuSegmentTriangleIntersect, SegmentTriangleQuery,
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

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        range(state, -span, span),
        range(state, -span, span),
        range(state, -span, span),
    )
}

/// Builds a well-conditioned interior-crossing fixture: a comfortably-sized
/// triangle and a segment that pierces it at an interior `barycentric` point
/// near the segment midpoint, far from every edge, vertex and endpoint.
fn interior_query(state: &mut u64) -> SegmentTriangleQuery {
    // A triangle with a healthy area, randomly placed and sized.
    let v0 = rand_vec(state, 3.0);
    let v1 = v0.plus(rand_vec(state, 2.0).plus(Vec3::new(3.0, 0.0, 0.0)));
    let v2 = v0.plus(rand_vec(state, 2.0).plus(Vec3::new(0.0, 3.0, 0.0)));

    // Interior barycentric weights bounded well inside the simplex.
    let u = range(state, 0.2, 0.4);
    let v = range(state, 0.2, 0.4);
    let w = 1.0 - u - v;
    let target = v0.scale(w).plus(v1.scale(u)).plus(v2.scale(v));

    // A face normal to offset the endpoints off-plane on opposite sides.
    let edge1 = v1.minus(v0);
    let edge2 = v2.minus(v0);
    let normal = edge1.cross(edge2);
    let scale = range(state, 1.0, 2.0);
    let start = target.plus(normal.scale(scale));
    let end = target.minus(normal.scale(scale));

    SegmentTriangleQuery::new(Segment::new(start, end), v0, v1, v2)
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the hit presence
/// must match exactly, and on a hit the point, the three `barycentric` weights
/// and the segment parameter must all agree within the parity bound.
fn pin(idx: usize, query: &SegmentTriangleQuery, got: &Option<Hit>) {
    let want = intersect(&query.segment, query.v0, query.v1, query.v2);
    match (got, want) {
        (Some(g), Some(w)) => {
            close_vec("hit.point", idx, g.point, w.point);
            assert!(
                close(g.u, w.u),
                "query {idx} hit.u: gpu {} vs cpu {}",
                g.u,
                w.u
            );
            assert!(
                close(g.v, w.v),
                "query {idx} hit.v: gpu {} vs cpu {}",
                g.v,
                w.v
            );
            assert!(
                close(g.w, w.w),
                "query {idx} hit.w: gpu {} vs cpu {}",
                g.w,
                w.w
            );
            assert!(
                close(g.t, w.t),
                "query {idx} hit.t: gpu {} vs cpu {}",
                g.t,
                w.t
            );
        }
        (None, None) => {}
        (g, w) => panic!("query {idx}: hit presence mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuSegmentTriangleIntersect, queries: &[SegmentTriangleQuery]) {
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

/// The canonical unit right-triangle in the `z = 0` plane.
fn unit_triangle() -> (Vec3, Vec3, Vec3) {
    (
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Pierces the interior at (0.3, 0.3, 0) halfway along the segment.
    let seg = Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, -1.0));
    check(&ctx, &gpu, &[SegmentTriangleQuery::new(seg, v0, v1, v2)]);
}

#[test]
fn vertex_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Crosses exactly at v1, so the full weight lands on u.
    let seg = Segment::new(Vec3::new(1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, -1.0));
    check(&ctx, &gpu, &[SegmentTriangleQuery::new(seg, v0, v1, v2)]);
}

#[test]
fn edge_grazing_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Point (0.5, 0, 0) sits exactly on edge v0v1 (v == 0): a grazing hit.
    let seg = Segment::new(Vec3::new(0.5, 0.0, 1.0), Vec3::new(0.5, 0.0, -1.0));
    check(&ctx, &gpu, &[SegmentTriangleQuery::new(seg, v0, v1, v2)]);
}

#[test]
fn long_direction_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // A long segment whose direction is far from unit length still resolves t as
    // a fraction of that direction.
    let seg = Segment::new(Vec3::new(0.2, 0.2, 50.0), Vec3::new(0.2, 0.2, -50.0));
    check(&ctx, &gpu, &[SegmentTriangleQuery::new(seg, v0, v1, v2)]);
}

#[test]
fn clean_misses_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    let queries = vec![
        // Crosses the plane far outside the face.
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(2.0, 2.0, 1.0), Vec3::new(2.0, 2.0, -1.0)),
            v0,
            v1,
            v2,
        ),
        // Stops before reaching the plane (t = 2 > 1).
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, 0.5)),
            v0,
            v1,
            v2,
        ),
        // Both endpoints above the plane (crossing at t > 1).
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.3, 0.3, 2.0), Vec3::new(0.3, 0.3, 1.0)),
            v0,
            v1,
            v2,
        ),
        // Both endpoints below the plane (crossing at t < 0).
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.3, 0.3, -0.5), Vec3::new(0.3, 0.3, -1.5)),
            v0,
            v1,
            v2,
        ),
        // Direction parallel to the triangle plane.
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.1, 0.1, 0.5), Vec3::new(0.6, 0.1, 0.5)),
            v0,
            v1,
            v2,
        ),
        // Segment coplanar with the triangle (degenerate determinant).
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.1, 0.1, 0.0), Vec3::new(0.5, 0.1, 0.0)),
            v0,
            v1,
            v2,
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_triangle_never_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    // A zero-area triangle (all three vertices collinear) is degenerate: it has
    // no plane to pierce, so the solve reports a miss, matching the reference's
    // area short-circuit.
    let v0 = Vec3::new(0.0, 0.0, 0.0);
    let v1 = Vec3::new(1.0, 0.0, 0.0);
    let v2 = Vec3::new(2.0, 0.0, 0.0);
    let seg = Segment::new(Vec3::new(0.5, 0.0, 1.0), Vec3::new(0.5, 0.0, -1.0));
    check(&ctx, &gpu, &[SegmentTriangleQuery::new(seg, v0, v1, v2)]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random interior
    // crossings, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element by
    // element.
    let mut queries = vec![
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(0.3, 0.3, 1.0), Vec3::new(0.3, 0.3, -1.0)),
            v0,
            v1,
            v2,
        ),
        SegmentTriangleQuery::new(
            Segment::new(Vec3::new(2.0, 2.0, 1.0), Vec3::new(2.0, 2.0, -1.0)),
            v0,
            v1,
            v2,
        ),
    ];
    for _ in 0..48 {
        queries.push(interior_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_interior_crossings_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentTriangleIntersect::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned interior crossings (several
    // workgroups' worth) pins the hit point, the barycentric weights and the
    // segment parameter across many random triangle geometries.
    let queries: Vec<SegmentTriangleQuery> = (0..200).map(|_| interior_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
