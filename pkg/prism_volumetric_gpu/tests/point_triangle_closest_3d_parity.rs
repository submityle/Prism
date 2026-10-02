//! Real-device parity for the 3D point-to-triangle nearest-point twin:
//! [`GpuPointTriangleClosest3d`](prism_volumetric_gpu::point_triangle_closest_3d::GpuPointTriangleClosest3d)
//! must reproduce the `CPU` golden
//! [`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d)
//! across an interior-face query (the point projects strictly inside the
//! triangle), a vertex-region query (the point sits in the Voronoi region of a
//! corner), an edge-region query (the point projects onto an edge interior), a
//! degenerate collinear triangle (zero area, routed to the edge-minimum
//! fallback), and a randomized batch of clearly-interior face projections
//! compared element-for-element.
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
//! Every fixture is deliberately well away from a region tie and from the
//! degeneracy crack: the interior query projects to a clearly-interior face
//! point, the vertex and edge queries land clearly inside one Voronoi region,
//! the degenerate triangle has an exactly zero normal, and each random fixture
//! is built by offsetting a clearly-interior face point along the triangle
//! normal, so its nearest point is that face point by construction and both
//! devices take the interior branch regardless of a few units in the last place
//! of slack.
//!
//! Provenance: twinned from this repository's
//! [`point_triangle_closest_3d`](prism_render_architecture::particle::point_triangle_closest_3d);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::point_triangle_closest_3d::{
    closest_point_on_triangle, ClosestPointOnTriangle, Vec3,
};
use prism_volumetric_gpu::point_triangle_closest_3d::{
    GpuPointTriangleClosest3d, PointTriangleQuery,
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

/// Builds a clearly-conditioned interior-face query by rejection sampling: a
/// well-spread non-degenerate triangle is drawn, a face point is formed from
/// barycentric weights held comfortably interior (each in `[0.2, 0.6]`), and the
/// query point is that face point offset along the triangle normal. The nearest
/// point on the triangle is therefore exactly the face point by construction,
/// so `CPU` and `GPU` both take the interior branch, far from every edge and
/// vertex region and from the degeneracy crack.
fn interior_query(state: &mut u64) -> PointTriangleQuery {
    loop {
        let a = rand_vec(state, 4.0);
        let b = rand_vec(state, 4.0);
        let c = rand_vec(state, 4.0);

        let ab = b.minus(a);
        let ac = c.minus(a);
        let normal = ab.cross(ac);
        // The triangle must be clearly non-degenerate (large squared area).
        if normal.length_squared() < 1.0 {
            continue;
        }

        let u = 0.2 + lcg(state) * 0.4;
        let v = 0.2 + lcg(state) * 0.4;
        let w = 1.0 - u - v;
        // The third weight must also stay comfortably interior.
        if !(0.2..=0.6).contains(&w) {
            continue;
        }

        let face = a.scale(u).plus(b.scale(v)).plus(c.scale(w));
        // Offset along the normal: the plane projection stays at the face point.
        let height = signed(state, 3.0);
        let point = face.plus(normal.scale(height));
        return PointTriangleQuery::new(point, a, b, c);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the nearest
/// point, the three barycentric weights and the squared distance must all agree
/// within bound.
fn pin(idx: usize, query: &PointTriangleQuery, got: &ClosestPointOnTriangle) {
    let want = closest_point_on_triangle(query.point, query.a, query.b, query.c);

    close_vec("point", idx, got.point, want.point);
    for (lane, (g, c)) in got.bary.iter().zip(want.bary.iter()).enumerate() {
        assert!(
            close(*g, *c),
            "query {idx} bary[{lane}]: gpu {g} vs cpu {c}"
        );
    }
    assert!(
        close(got.distance_squared, want.distance_squared),
        "query {idx} distance_squared: gpu {} vs cpu {}",
        got.distance_squared,
        want.distance_squared
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPointTriangleClosest3d, queries: &[PointTriangleQuery]) {
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
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_face_query_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    // The query point sits directly above the triangle interior, so its nearest
    // point is the in-plane projection and the interior branch fires.
    let query = PointTriangleQuery::new(
        Vec3::new(0.25, 0.25, 1.5),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn vertex_region_query_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    // The query point lies clearly in the Voronoi region of corner A (behind the
    // two edges leaving A), so the nearest point is A itself.
    let query = PointTriangleQuery::new(
        Vec3::new(-3.0, -3.0, 0.5),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn edge_region_query_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    // The query point projects onto the interior of edge AB (below the edge, well
    // clear of either endpoint region), so the AB edge branch fires.
    let query = PointTriangleQuery::new(
        Vec3::new(1.0, -2.0, 0.5),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_collinear_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    // Three collinear corners: the normal is exactly zero on both devices, so the
    // degenerate edge-minimum fallback fires identically.
    let query = PointTriangleQuery::new(
        Vec3::new(0.5, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic interior fixture with many random
    // interior-face queries, dispatched together so the per-thread indexing and
    // the contiguous storage layout are both exercised, then pinned
    // element-for-element.
    let mut queries = vec![PointTriangleQuery::new(
        Vec3::new(0.25, 0.25, 1.5),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    )];
    for _ in 0..48 {
        queries.push(interior_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_interior_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointTriangleClosest3d::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned interior-face queries (several
    // workgroups' worth) pins every reported field across many random triangle
    // geometries.
    let queries: Vec<PointTriangleQuery> = (0..200).map(|_| interior_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
