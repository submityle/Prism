//! Real-device parity for the analytic ray-cylinder twin:
//! [`GpuRayCylinder`](prism_volumetric_gpu::ray_cylinder::GpuRayCylinder) must
//! reproduce the `CPU` golden
//! [`ray_cylinder`](prism_render_architecture::particle::ray_cylinder) across
//! side-wall hits (a ray crossing the curved surface mid-height), end-cap hits
//! (a ray entering through the top disk), axial penetrations (a ray running
//! along the axis into the base cap), grazing misses (a ray parallel to the wall
//! that clears the radius) and degenerate cylinders (a near-zero radius that
//! encloses nothing), plus the point-containment predicate on an interior and an
//! exterior point, and a randomized batch of well-conditioned wall and cap hits
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
//! `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` fields while pinning the boolean
//! classification (`contains`, `intersects`, hit presence) exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! side/cap classification boundary: no ray strikes exactly at a rim corner
//! (where a wall root and a cap root would tie), interior points sit well inside
//! both the radius and the axial span, exterior points clear the radius by more
//! than its length, and wall-hit impact parameters stay below half the radius so
//! the quadratic has two well-separated roots and the radial normal is far from
//! the zero vector. This keeps `CPU` and `GPU` on the same side of every branch
//! regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`ray_cylinder`](prism_render_architecture::particle::ray_cylinder); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::ray_cylinder::{Cylinder, Ray, Vec3};
use prism_volumetric_gpu::ray_cylinder::{GpuRayCylinder, RayCylinderQuery, RayCylinderResult};
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

/// A unit vector drawn from `state`, retried until it is comfortably non-zero so
/// the normalization is well conditioned.
fn rand_unit(state: &mut u64) -> Vec3 {
    loop {
        let v = rand_vec(state, 1.0);
        if v.length_squared() > 0.2 {
            return v.normalize_or_zero();
        }
    }
}

/// A unit vector perpendicular to `axis`, built by crossing `axis` with whichever
/// cardinal axis is least parallel to it.
fn perp_unit(axis: Vec3) -> Vec3 {
    let helper = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    axis.cross(helper).normalize_or_zero()
}

/// Builds a clearly side-wall-hitting query: a cylinder with a random axis and a
/// ray fired perpendicular to that axis at mid-height, aimed within half the
/// radius of the axis so the quadratic has two well-separated roots and the
/// caps (parallel to the ray) contribute nothing. The test point is the axis
/// midpoint, strictly inside the solid.
fn side_hit_query(state: &mut u64) -> RayCylinderQuery {
    let base = rand_vec(state, 6.0);
    let ca = rand_unit(state);
    let radius = 1.0 + lcg(state) * 2.0;
    let height = 4.0 + lcg(state) * 4.0;
    let away = perp_unit(ca);
    let side = ca.cross(away).normalize_or_zero();
    let mid = base.plus(ca.scale(height * 0.5));
    let dist = radius + 4.0 + lcg(state) * 4.0;
    let origin = mid.plus(away.scale(dist));
    let offset = signed(state, radius * 0.4);
    let target = mid.plus(side.scale(offset));
    let dir = target.minus(origin);
    RayCylinderQuery::new(
        Ray::new_normalized(origin, dir),
        Cylinder::new(base, ca, radius, height),
        mid,
    )
}

/// Builds a clearly end-cap-hitting query: a ray running parallel to the axis
/// (so the side-wall quadratic collapses) enters through the top disk within
/// half the radius of the center. The test point is the axis midpoint, inside.
fn cap_hit_query(state: &mut u64) -> RayCylinderQuery {
    let base = rand_vec(state, 6.0);
    let ca = rand_unit(state);
    let radius = 1.0 + lcg(state) * 2.0;
    let height = 4.0 + lcg(state) * 4.0;
    let away = perp_unit(ca);
    let top = base.plus(ca.scale(height));
    let dist = 3.0 + lcg(state) * 4.0;
    let offset = signed(state, radius * 0.4);
    let origin = top.plus(ca.scale(dist)).plus(away.scale(offset));
    // Straight down the axis into the top cap.
    let dir = ca.scale(-1.0);
    let mid = base.plus(ca.scale(height * 0.5));
    RayCylinderQuery::new(
        Ray::new_normalized(origin, dir),
        Cylinder::new(base, ca, radius, height),
        mid,
    )
}

/// Builds a clear miss: a ray parallel to the wall at mid-height whose impact
/// parameter clears the radius by more than its length, so the quadratic has a
/// negative discriminant and the caps (parallel to the ray) see nothing. The
/// test point sits outside the radius, so containment is false.
fn miss_query(state: &mut u64) -> RayCylinderQuery {
    let base = rand_vec(state, 6.0);
    let ca = rand_unit(state);
    let radius = 1.0 + lcg(state) * 2.0;
    let height = 4.0 + lcg(state) * 4.0;
    let away = perp_unit(ca);
    let side = ca.cross(away).normalize_or_zero();
    let mid = base.plus(ca.scale(height * 0.5));
    let dist = radius + 4.0 + lcg(state) * 4.0;
    let origin = mid.plus(away.scale(dist));
    let offset = radius * (2.5 + lcg(state) * 2.0);
    let target = mid.plus(side.scale(offset));
    let dir = target.minus(origin);
    let outside = mid.plus(side.scale(radius * 3.0));
    RayCylinderQuery::new(
        Ray::new_normalized(origin, dir),
        Cylinder::new(base, ca, radius, height),
        outside,
    )
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the containment
/// and intersect predicates and the nearest forward hit must all agree, booleans
/// exactly and `f32` fields within bound.
fn pin(idx: usize, query: &RayCylinderQuery, got: &RayCylinderResult) {
    let want_contains = query.cylinder.contains(query.point);
    let want_intersects = query.cylinder.intersects(query.ray);
    let want_hit = query.cylinder.first_hit(query.ray);

    assert_eq!(
        got.contains, want_contains,
        "query {idx}: contains must match the reference"
    );
    assert_eq!(
        got.intersects, want_intersects,
        "query {idx}: intersects must match the reference"
    );

    match (got.hit, want_hit) {
        (Some(g), Some(w)) => {
            assert!(
                close(g.t, w.t),
                "query {idx} hit.t: gpu {} vs cpu {}",
                g.t,
                w.t
            );
            close_vec("hit.point", idx, g.point, w.point);
            close_vec("hit.normal", idx, g.normal, w.normal);
        }
        (None, None) => {}
        (g, w) => panic!("query {idx}: hit presence mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuRayCylinder, queries: &[RayCylinderQuery]) {
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

/// The canonical unit test cylinder: axis up `+y`, base at the origin, radius `1`
/// and height `4`.
fn unit_cyl() -> Cylinder {
    Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0, 4.0)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn side_wall_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // A ray down -x through the wall of a unit cylinder: the near wall is at
    // x = 1 with a radial +x outward normal; the interior test point is inside.
    let query = RayCylinderQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        unit_cyl(),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn top_cap_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // A ray straight down the axis from above enters the top disk at y = 4 with a
    // +y outward normal; the side-wall quadratic collapses (ray parallel to axis).
    let query = RayCylinderQuery::new(
        Ray::new_normalized(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
        unit_cyl(),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn axial_penetration_hits_base_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // A ray running up the axis from below the base enters the base disk at
    // y = 0 with a -y outward normal, the base cap being the nearer forward hit.
    let query = RayCylinderQuery::new(
        Ray::new_normalized(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        unit_cyl(),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn grazing_miss_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // A ray parallel to the wall but offset in z beyond the radius: the quadratic
    // discriminant is negative and the caps are parallel, so it is a clear miss.
    // The exterior test point also lies outside the radius.
    let query = RayCylinderQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 2.0), Vec3::new(-1.0, 0.0, 0.0)),
        unit_cyl(),
        Vec3::new(2.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_cylinder_never_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    // A zero-radius cylinder encloses nothing: it never intersects and contains
    // no point, exactly as the reference short-circuits on `is_degenerate`.
    let query = RayCylinderQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.0, 4.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing all three well-conditioned hit/miss categories, dispatched
    // together so the per-thread indexing and the contiguous storage layout are
    // both exercised, then pinned element-for-element.
    let mut queries = Vec::new();
    for _ in 0..32 {
        queries.push(side_hit_query(&mut state));
        queries.push(cap_hit_query(&mut state));
        queries.push(miss_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_side_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCylinder::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clear side-wall hits (several workgroups' worth) pins the
    // root, hit point and radial outward normal across many random geometries.
    let queries: Vec<RayCylinderQuery> = (0..200).map(|_| side_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
