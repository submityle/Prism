//! Real-device parity for the analytic ray-cone twin:
//! [`GpuRayCone`](prism_volumetric_gpu::ray_cone::GpuRayCone) must reproduce the
//! `CPU` golden
//! [`ray_cone`](prism_render_architecture::particle::ray_cone) across a forward
//! nappe hit (a ray striking the opening side of an infinite cone), a backward
//! nappe cull (a ray whose only algebraic roots land on the reflected nappe, so
//! the result is a miss), a grazing axis hit (a ray fired along the axis that
//! touches the apex through the degenerate single-root branch), the finite
//! lateral-wall and base-cap hits (an infinite hit clipped to the band versus
//! one closed off by the flat cap), a pointing-away miss, plus a randomized
//! batch of clearly-conditioned forward-nappe hits compared element for element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 4e-3 + 1e-3 * (|a| + |b|)`
//! on the `f32` fields (the reference's grazing double-root `sqrt`-scale slack)
//! while pinning the hit flag exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! branch boundaries: random hits land near the mid-nappe (axial coordinate
//! clearly positive, two well-separated roots with a large discriminant), the
//! deterministic misses clear every root, and no fixture is tuned to a
//! wall/cap or forward/backward-nappe tie. This keeps `CPU` and `GPU` on the
//! same side of every branch regardless of a few units in the last place of
//! slack.
//!
//! Provenance: twinned from this repository's
//! [`ray_cone`](prism_render_architecture::particle::ray_cone); no third-party
//! engine source or derived code.

use prism_render_architecture::particle::ray_cone::{
    ray_finite_cone, ray_infinite_cone, RayConeHit,
};
use prism_volumetric_gpu::ray_cone::{GpuRayCone, RayConeQuery, RayConeResult};
use prism_volumetric_gpu::GpuContext;

/// Cosine of a `45`-degree half-angle, the convenient cone where radius equals
/// axial distance. Kept as the same literal the reference tests use so the
/// fixtures stay trig-free.
const COS_45: f32 = 0.707_106_77;

/// Returns whether `a` and `b` agree within the reference's parity bound: an
/// absolute floor of `4e-3` absorbing the grazing double-root `sqrt`-scale
/// error plus a relative term keeping large-magnitude hits meaningful.
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= 4.0e-3 + 1.0e-3 * (a.abs() + b.abs())
}

/// Asserts two points agree channel-for-channel within the parity bound.
fn approx_pt(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        approx(got[0], want[0]) && approx(got[1], want[1]) && approx(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
}

/// Euclidean dot product `a . b`.
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b`.
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Cross product `a x b`.
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit vector in the direction of `v` (only `sqrt` is used, no transcendental).
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = v_dot(v, v).sqrt();
    v_scale(v, 1.0 / len)
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
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// A unit vector drawn from `state`, retried until it is comfortably non-zero so
/// the normalization is well conditioned.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = rand_vec(state, 1.0);
        if v_dot(v, v) > 0.2 {
            return normalize(v);
        }
    }
}

/// Returns a unit vector perpendicular to `axis`, built by crossing `axis` with
/// whichever cardinal axis is least parallel to it.
fn perp_unit(axis: [f32; 3]) -> [f32; 3] {
    let helper = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(v_cross(axis, helper))
}

/// Builds a clearly-conditioned forward-nappe infinite-cone hit: a `45`-degree
/// cone whose forward surface is struck by a radial ray that crosses the nappe
/// transversally at the mid-section (axial coordinate clearly positive, two
/// well-separated roots with a large discriminant, hit parameter well away from
/// zero). The query is rejection-sampled so `CPU` and `GPU` land on the same
/// branch regardless of a few units in the last place.
fn cone_hit_query(state: &mut u64) -> RayConeQuery {
    loop {
        let axis = rand_unit(state);
        let apex = rand_vec(state, 3.0);
        let l = 2.0 + lcg(state) * 3.0;
        let perp = perp_unit(axis);
        // A point on the forward nappe: axial distance `l`, radius `l` (cos45).
        let target = v_add(v_add(apex, v_scale(axis, l)), v_scale(perp, l));
        // Fire a mostly-radial ray in from outside, with a small tangential jog.
        let perp2 = normalize(v_cross(axis, perp));
        let dist = 4.0 + lcg(state) * 4.0;
        let origin = v_add(
            target,
            v_add(v_scale(perp, dist), v_scale(perp2, signed(state, 0.5))),
        );
        let dir = normalize(v_sub(target, origin));

        // Replicate the reference coefficients to reject ill-conditioned draws.
        let co = v_sub(origin, apex);
        let dd = v_dot(dir, axis);
        let cd = v_dot(co, axis);
        let k2 = COS_45 * COS_45;
        let a = dd * dd - k2 * v_dot(dir, dir);
        let b = 2.0 * (cd * dd - k2 * v_dot(co, dir));
        let c = cd * cd - k2 * v_dot(co, co);
        let disc = b * b - 4.0 * a * c;
        if a.abs() < 1.0e-2 || disc < 4.0e-2 {
            continue;
        }

        let query = RayConeQuery::infinite(origin, dir, apex, axis, COS_45);
        if let Some(hit) = ray_infinite_cone(origin, dir, apex, axis, COS_45) {
            // Keep only hits well away from t = 0 and the nappe-cull boundary.
            if hit.t > 0.2 && cd + hit.t * dd > 0.3 {
                return query;
            }
        }
    }
}

/// Evaluates the `CPU` golden for one query, selecting the solver by `kind`.
fn golden_hit(query: &RayConeQuery) -> Option<RayConeHit> {
    if query.kind == prism_volumetric_gpu::ray_cone::KIND_FINITE {
        ray_finite_cone(
            query.origin,
            query.dir,
            query.apex,
            query.axis,
            query.cos_half_angle,
            query.height,
        )
    } else {
        ray_infinite_cone(
            query.origin,
            query.dir,
            query.apex,
            query.axis,
            query.cos_half_angle,
        )
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the hit flag must
/// match exactly, and when present the ray parameter and point must agree within
/// the parity bound.
fn pin(idx: usize, query: &RayConeQuery, got: &RayConeResult) {
    let want = golden_hit(query);
    let want_hit = u32::from(want.is_some());
    assert_eq!(
        got.hit, want_hit,
        "query {idx}: hit flag must match the reference (gpu {}, cpu {want_hit})",
        got.hit
    );
    if let Some(hit) = want {
        assert!(
            approx(got.t, hit.t),
            "query {idx} hit.t: gpu {} vs cpu {}",
            got.t,
            hit.t
        );
        approx_pt("hit.point", idx, got.point, hit.point);
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuRayCone, queries: &[RayConeQuery]) {
    let got = gpu.intersect(ctx, queries);
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
    let gpu = GpuRayCone::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.intersect(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn infinite_perpendicular_hits_nearest() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // A horizontal ray crossing a +z cone: nearest crossing at x = -2, t = 8.
    let query = RayConeQuery::infinite(
        [-10.0, 0.0, 2.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn infinite_axis_grazes_apex() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // A ray fired up the axis from below grazes the apex via the single-root
    // (grazing double-root) branch.
    let query = RayConeQuery::infinite(
        [0.0, 0.0, -5.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn infinite_backward_nappe_is_culled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // A horizontal ray below the apex crosses only the reflected backward nappe,
    // so the forward-nappe result is a miss even though the quadratic has roots.
    let query = RayConeQuery::infinite(
        [-10.0, 0.0, -2.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn infinite_pointing_away_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // The cone lies entirely behind the origin along this direction: a miss.
    let query = RayConeQuery::infinite(
        [20.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn finite_lateral_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // A ray from inside a tall finite cone exits through the lateral wall at
    // x = 3, axial 3 clearly inside [0, 10].
    let query = RayConeQuery::finite(
        [0.0, 0.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
        10.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn finite_base_cap_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    // A ray fired straight down -z through the wide end strikes the flat base
    // cap at z = 4 (radius 4 at cos45), nearer than the lateral crossing.
    let query = RayConeQuery::finite(
        [1.0, 0.0, 10.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        COS_45,
        4.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random forward-nappe
    // hits, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        RayConeQuery::infinite(
            [-10.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        ),
        RayConeQuery::infinite(
            [-10.0, 0.0, -2.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
        ),
        RayConeQuery::finite(
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            10.0,
        ),
        RayConeQuery::finite(
            [1.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            COS_45,
            4.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(cone_hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_forward_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayCone::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned forward-nappe hits (several
    // workgroups' worth) pins the hit parameter and point across many random
    // cone geometries.
    let queries: Vec<RayConeQuery> = (0..200).map(|_| cone_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
