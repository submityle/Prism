//! Real-device parity for the polyline / polygon-`SDF` twin:
//! [`GpuPolylineSdf2d`](prism_volumetric_gpu::polyline_sdf_2d::GpuPolylineSdf2d)
//! must reproduce the `CPU` golden
//! [`polyline_sdf_2d`](prism_render_architecture::particle::polyline_sdf_2d)
//! across a convex square, a concave `L`-shape, a twelve-vertex convex ring (to
//! exercise a larger fixed-capacity fill), a degenerate two-vertex segment and
//! randomized batches of query points compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` per edge, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. They are not bit-exact for the continuous distances: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the three distances while
//! demanding bit-exact agreement on the integer `winding` and the `inside`
//! classification code.
//!
//! # Conditioning
//!
//! Every random query is kept well away from the branch cracks the winding and
//! sign decisions turn on: a point is rejected unless its `y` is comfortably
//! clear of every vertex `y` (so no horizontal-crossing tie flips a count) and
//! its boundary distance is comfortably positive (so neither the boundary snap
//! nor the inside/outside sign is on a tie). All fixture polygons use exact
//! integer lattice coordinates so the reference and the device agree on every
//! branch regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::polyline_sdf_2d::Vec2;
use prism_volumetric_gpu::polyline_sdf_2d::{
    golden, GpuPolylineSdf, GpuPolylineSdf2d, PolylineSdf2dQuery,
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

/// Vertical clearance a random query keeps from every vertex `y`, so no
/// horizontal-crossing comparison in the winding sum sits on a tie.
const Y_MARGIN: f32 = 0.1;

/// Boundary clearance a random query keeps from the ring, so neither the
/// boundary snap nor the inside/outside sign sits on a tie.
const DIST_MARGIN: f32 = 0.1;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// The convex unit square centred on the origin.
fn unit_square() -> Vec<Vec2> {
    vec![
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ]
}

/// A `CCW` concave `L`-shaped polygon occupying `x in [0,2] y in [0,1]` plus
/// `x in [0,1] y in [1,2]`.
fn l_shape() -> Vec<Vec2> {
    vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(2.0, 0.0),
        Vec2::new(2.0, 1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(1.0, 2.0),
        Vec2::new(0.0, 2.0),
    ]
}

/// A twelve-vertex convex ring whose vertices are exact radius-`5` integer
/// lattice points (every vertex satisfies `x^2 + y^2 = 25`), so it exercises a
/// larger fixed-capacity fill without introducing any irrational coordinate.
fn dodecagon() -> Vec<Vec2> {
    vec![
        Vec2::new(5.0, 0.0),
        Vec2::new(4.0, 3.0),
        Vec2::new(3.0, 4.0),
        Vec2::new(0.0, 5.0),
        Vec2::new(-3.0, 4.0),
        Vec2::new(-4.0, 3.0),
        Vec2::new(-5.0, 0.0),
        Vec2::new(-4.0, -3.0),
        Vec2::new(-3.0, -4.0),
        Vec2::new(0.0, -5.0),
        Vec2::new(3.0, -4.0),
        Vec2::new(4.0, -3.0),
    ]
}

/// Returns whether `p` is well away from every branch crack of `poly`: clear of
/// every vertex `y` and comfortably off the boundary. Random points failing
/// this are rejected so the winding and sign stay off any tie.
fn well_conditioned(p: Vec2, poly: &[Vec2]) -> bool {
    for v in poly {
        if (p.y - v.y).abs() < Y_MARGIN {
            return false;
        }
    }
    let want = golden(&PolylineSdf2dQuery::new(p, poly.to_vec()));
    want.boundary_dist >= DIST_MARGIN
}

/// Draws a well-conditioned random query point over `poly` by rejection
/// sampling within a box that comfortably brackets the ring.
fn rand_query(state: &mut u64, poly: &[Vec2]) -> PolylineSdf2dQuery {
    loop {
        let p = Vec2::new(signed(state, 7.0), signed(state, 7.0));
        if well_conditioned(p, poly) {
            return PolylineSdf2dQuery::new(p, poly.to_vec());
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the three
/// distances agree within bound and the integer `winding` and `inside` code
/// match exactly.
fn pin(idx: usize, query: &PolylineSdf2dQuery, got: &GpuPolylineSdf) {
    let want = golden(query);
    assert!(
        close(got.polyline_dist, want.polyline_dist),
        "query {idx} polyline_dist: gpu {} vs cpu {}",
        got.polyline_dist,
        want.polyline_dist
    );
    assert!(
        close(got.boundary_dist, want.boundary_dist),
        "query {idx} boundary_dist: gpu {} vs cpu {}",
        got.boundary_dist,
        want.boundary_dist
    );
    assert!(
        close(got.signed_dist, want.signed_dist),
        "query {idx} signed_dist: gpu {} vs cpu {}",
        got.signed_dist,
        want.signed_dist
    );
    assert_eq!(
        got.winding, want.winding,
        "query {idx} winding: gpu {} vs cpu {}",
        got.winding, want.winding
    );
    assert_eq!(
        got.inside, want.inside,
        "query {idx} inside: gpu {} vs cpu {}",
        got.inside, want.inside
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPolylineSdf2d, queries: &[PolylineSdf2dQuery]) {
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
    let gpu = GpuPolylineSdf2d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn square_interior_and_exterior_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolylineSdf2d::new(&ctx);
    let sq = unit_square();
    // Centre: inside, winding 1, boundary distance 1, signed -1. Far corner:
    // outside, winding 0. Both use exact integer geometry.
    let queries = vec![
        PolylineSdf2dQuery::new(Vec2::new(0.0, 0.0), sq.clone()),
        PolylineSdf2dQuery::new(Vec2::new(0.5, 0.25), sq.clone()),
        PolylineSdf2dQuery::new(Vec2::new(3.0, 0.25), sq),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn l_shape_concave_cases_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolylineSdf2d::new(&ctx);
    let poly = l_shape();
    // Interior arm, interior upper arm, the notch (outside) and a far exterior
    // point exercise the concave winding and signed-distance branches.
    let queries = vec![
        PolylineSdf2dQuery::new(Vec2::new(0.5, 0.5), poly.clone()),
        PolylineSdf2dQuery::new(Vec2::new(0.5, 1.5), poly.clone()),
        PolylineSdf2dQuery::new(Vec2::new(1.5, 1.5), poly.clone()),
        PolylineSdf2dQuery::new(Vec2::new(-1.0, 0.5), poly),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_segment_has_no_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolylineSdf2d::new(&ctx);
    // Two vertices cannot enclose area; winding is 0 and the signed distance is
    // the non-negative boundary distance.
    let seg = vec![Vec2::new(0.0, 0.0), Vec2::new(4.0, 0.0)];
    let query = PolylineSdf2dQuery::new(Vec2::new(2.0, 3.0), seg);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolylineSdf2d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the three ring shapes with many random, well-conditioned
    // query points, dispatched together so the per-thread indexing and the
    // contiguous fixed-capacity storage layout are both exercised.
    let mut queries = vec![
        PolylineSdf2dQuery::new(Vec2::new(0.0, 0.25), unit_square()),
        PolylineSdf2dQuery::new(Vec2::new(0.5, 0.5), l_shape()),
        PolylineSdf2dQuery::new(Vec2::new(0.0, 0.25), dodecagon()),
    ];
    for _ in 0..16 {
        queries.push(rand_query(&mut state, &unit_square()));
        queries.push(rand_query(&mut state, &l_shape()));
        queries.push(rand_query(&mut state, &dodecagon()));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolylineSdf2d::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) over the twelve-vertex convex
    // ring pins every field across many random query points.
    let poly = dodecagon();
    let queries: Vec<PolylineSdf2dQuery> =
        (0..200).map(|_| rand_query(&mut state, &poly)).collect();
    check(&ctx, &gpu, &queries);
}
