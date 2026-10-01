//! Real-device parity for the analytic ray-capsule twin:
//! [`GpuRayCapsule`](prism_volumetric_gpu::ray_capsule::GpuRayCapsule) must
//! reproduce the `CPU` golden
//! [`ray_capsule`](prism_render_architecture::particle::ray_capsule) across side-wall
//! hits (a ray striking the cylindrical mid-section), end-cap hits (a ray
//! striking a hemisphere apex), an axial pass-through (a ray fired straight down
//! the axis that enters through the near cap), grazing misses (a line that
//! clears the side wall by more than the radius) and a degenerate capsule (a
//! zero-radius primitive that never intersects and contains nothing), plus a
//! randomized batch of clearly-conditioned side-wall hits compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a few
//! `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` fields while pinning the boolean
//! classification (`intersects`, `contains`, hit presence) exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! wall/cap classification boundary: side-wall hits land near the mid-segment
//! (axial coordinate `m` clearly inside `[0, height]`), cap hits land clearly
//! past an endpoint (`m` well outside the segment), the grazing miss clears the
//! wall by more than the radius so the discriminant is clearly negative, and no
//! fixture is tuned to a wall/cap tie. This keeps `CPU` and `GPU` on the same
//! side of every branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`ray_capsule`](prism_render_architecture::particle::ray_capsule); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::ray_capsule::{Capsule, Ray, Vec3};
use prism_volumetric_gpu::ray_capsule::{GpuRayCapsule, RayCapsuleQuery, RayCapsuleResult};
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

/// Returns a unit vector perpendicular to `axis`, built by crossing `axis` with
/// whichever cardinal axis is least parallel to it.
fn perp_unit(axis: Vec3) -> Vec3 {
    let helper = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    axis.cross(helper).normalize_or_zero()
}

/// Builds a clearly-conditioned side-wall hit: the ray comes in radially toward
/// the mid-segment with a perpendicular impact parameter below half the radius,
/// so the hit lands on the cylindrical wall well inside `[0, height]`, far from
/// the wall/cap boundary and with two well-separated roots. The query point sits
/// on the axis mid-segment, so it is clearly contained.
fn wall_hit_query(state: &mut u64) -> RayCapsuleQuery {
    let axis = rand_unit(state);
    let height = 3.0 + lcg(state) * 3.0;
    let a = rand_vec(state, 4.0);
    let b = a.plus(axis.scale(height));
    let radius = 1.0 + lcg(state) * 2.0;
    let mid = a.plus(axis.scale(height * 0.5));

    // Two independent directions perpendicular to the axis: `away` carries the
    // ray in from outside, `side` adds a small in-plane impact offset.
    let away = perp_unit(axis);
    let side = axis.cross(away).normalize_or_zero();

    let dist = radius + 4.0 + lcg(state) * 4.0;
    let origin = mid.plus(away.scale(dist));
    let offset = signed(state, radius * 0.4);
    let target = mid.plus(side.scale(offset));
    let dir = target.minus(origin);

    RayCapsuleQuery::new(
        Ray::new_normalized(origin, dir),
        Capsule::new(a, b, radius),
        mid,
    )
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the intersect and
/// contains predicates, the closest axis-segment point and the nearest forward
/// hit must all agree, booleans exactly and `f32` fields within bound.
fn pin(idx: usize, query: &RayCapsuleQuery, got: &RayCapsuleResult) {
    let want_hit = query.capsule.first_hit(query.ray);
    let want_intersects = query.capsule.intersects(query.ray);
    let want_contains = query.capsule.contains(query.point);
    let want_closest = query.capsule.closest_on_segment(query.point);

    assert_eq!(
        got.intersects, want_intersects,
        "query {idx}: intersects must match the reference"
    );
    assert_eq!(
        got.contains, want_contains,
        "query {idx}: contains must match the reference"
    );
    close_vec("closest", idx, got.closest, want_closest);

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
fn check(ctx: &GpuContext, gpu: &GpuRayCapsule, queries: &[RayCapsuleQuery]) {
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

/// A canonical unit-radius capsule along +Y with cap centers at `y = 0` and
/// `y = 4` (so the solid spans `y in [-1, 5]` once the round caps are added).
fn unit_cap() -> Capsule {
    Capsule::new(Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0), 1.0)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
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
    let gpu = GpuRayCapsule::new(&ctx);
    // A ray down -x striking the cylindrical wall at x = 1, mid-height (m = 2,
    // clearly inside [0, 4]); outward normal +x. The query point is on the axis
    // mid-segment, so it is strictly contained.
    let query = RayCapsuleQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        unit_cap(),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn far_cap_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    // A ray down -y striking the far hemisphere apex at y = 5 (m = 5, clearly
    // past height = 4); outward normal +y. The query point is just outside the
    // side wall, so it is not contained.
    let query = RayCapsuleQuery::new(
        Ray::new_normalized(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
        unit_cap(),
        Vec3::new(1.5, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn axial_passthrough_enters_near_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    // A ray fired straight up the axis from below: it enters through the near
    // hemisphere apex at y = -1 (m = -1, clearly on the near outer half); outward
    // normal -y. The query point sits within the near round cap, so it is
    // contained even though it is below y = 0.
    let query = RayCapsuleQuery::new(
        Ray::new_normalized(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        unit_cap(),
        Vec3::new(0.0, -0.5, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn grazing_miss_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    // A ray down -x offset to z = 2, more than the radius clear of the axis, so
    // the side-wall discriminant is clearly negative and the caps are missed
    // too: no intersect, no hit. The query point beyond the far apex is also not
    // contained.
    let query = RayCapsuleQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 2.0), Vec3::new(-1.0, 0.0, 0.0)),
        unit_cap(),
        Vec3::new(0.0, 6.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_capsule_never_intersects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    // A zero-radius capsule is degenerate: it encloses no volume, so it never
    // reports a hit and contains nothing, matching the reference's is_degenerate
    // short-circuit. The ray would otherwise strike the axis dead-on.
    let query = RayCapsuleQuery::new(
        Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        Capsule::new(Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0), 0.0),
        Vec3::new(0.0, 2.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random side-wall
    // hits, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        RayCapsuleQuery::new(
            Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
            unit_cap(),
            Vec3::new(0.0, 2.0, 0.0),
        ),
        RayCapsuleQuery::new(
            Ray::new_normalized(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            unit_cap(),
            Vec3::new(1.5, 2.0, 0.0),
        ),
        RayCapsuleQuery::new(
            Ray::new_normalized(Vec3::new(5.0, 2.0, 2.0), Vec3::new(-1.0, 0.0, 0.0)),
            unit_cap(),
            Vec3::new(0.0, 6.0, 0.0),
        ),
    ];
    for _ in 0..48 {
        queries.push(wall_hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_wall_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCapsule::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned side-wall hits (several workgroups'
    // worth) pins the hit parameter, point and outward normal across many random
    // capsule geometries.
    let queries: Vec<RayCapsuleQuery> = (0..200).map(|_| wall_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
