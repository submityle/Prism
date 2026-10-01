//! Real-device parity for the `ray`-triangle `Moller-Trumbore` twin:
//! [`GpuRayTriangle`](prism_volumetric_gpu::ray_triangle::GpuRayTriangle) must
//! reproduce the `CPU` golden
//! [`ray_triangle`](prism_render_architecture::particle::ray_triangle) across a
//! front-facing interior hit, a back-facing interior hit (the same geometry
//! struck from the other side), a parallel `ray` (determinant below the guard),
//! a behind-origin triangle (`t` negative so the forward test rejects it), a
//! clearly-outside miss, a collinear degenerate triangle (zero determinant), and
//! a randomized batch of clearly-interior hits compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of cross products, dot
//! products and one guarded division, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the continuous `t`, `u`,
//! `v` fields while requiring an *exact* match on the discrete hit and
//! front-face flags.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie: hits land clearly
//! interior with both `barycentric` weights far from `0` and their sum far from
//! `1`, misses fail their test by a wide margin, the parallel `ray` has a
//! determinant that is exactly zero on both devices, and the degenerate
//! triangle is collinear with integer coordinates so its determinant is exactly
//! zero. This keeps `CPU` and `GPU` on the same side of every branch regardless
//! of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_triangle`；
//! 非第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ray_triangle::{intersect_moller_trumbore, Ray, Vec3};
use prism_volumetric_gpu::ray_triangle::{GpuRayTriangle, RayTriangleHit, RayTriangleQuery};
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

/// Builds a clearly-conditioned interior hit by rejection sampling: three
/// well-spread vertices and a direction are drawn with only four-op arithmetic
/// (no transcendental), an interior point is placed with `barycentric` weights
/// comfortably off every edge (`u`, `v` in `[0.15, 0.7]` with `u + v <= 0.8`),
/// and the pair is accepted only when the determinant sits well above the guard
/// so `CPU` and `GPU` share the general-case branch. The `ray` is launched from
/// a point `k` units back along the direction so the hit is a strictly forward
/// crossing at `t = k`.
fn interior_hit_query(state: &mut u64) -> RayTriangleQuery {
    loop {
        let v0 = rand_vec(state, 4.0);
        let v1 = rand_vec(state, 4.0);
        let v2 = rand_vec(state, 4.0);

        // Interior barycentric weights clear of every edge: u, v in [0.15, 0.7]
        // with u + v <= 0.8 so w = 1 - u - v stays >= 0.2.
        let u = 0.15 + lcg(state) * 0.55;
        let v = 0.15 + lcg(state) * 0.55;
        if u + v > 0.8 {
            continue;
        }
        let w = 1.0 - u - v;
        let point = v0.scale(w).plus(v1.scale(u)).plus(v2.scale(v));

        // A direction that must cross the triangle plane: reject near-parallel
        // casts by checking the Moller-Trumbore determinant magnitude directly.
        let dir = rand_vec(state, 3.0);
        let edge1 = v1.minus(v0);
        let edge2 = v2.minus(v0);
        let det = edge1.dot(dir.cross(edge2));
        if det.abs() < 1.0 {
            continue;
        }

        // Launch k units back along the direction so the hit is forward at t=k.
        let k = 1.0 + lcg(state) * 3.0;
        let origin = point.minus(dir.scale(k));
        return RayTriangleQuery::new(Ray::new(origin, dir), v0, v1, v2);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the discrete hit
/// and front-face verdicts must match exactly, and the `barycentric` `(t, u, v)`
/// must agree within bound whenever the reference reports a hit.
fn pin(idx: usize, query: &RayTriangleQuery, got: &RayTriangleHit) {
    // No-cull reference decides the hit and reports (t, u, v); cull-backface
    // reference decides the front-face verdict (it keeps only positive-det hits).
    let no_cull = intersect_moller_trumbore(&query.ray, query.v0, query.v1, query.v2, false);
    let culled = intersect_moller_trumbore(&query.ray, query.v0, query.v1, query.v2, true);

    let want_hit = no_cull.is_some();
    let want_front = culled.is_some();

    assert_eq!(
        got.hit, want_hit,
        "query {idx} hit: gpu {} vs cpu {}",
        got.hit, want_hit
    );
    assert_eq!(
        got.front_face, want_front,
        "query {idx} front_face: gpu {} vs cpu {}",
        got.front_face, want_front
    );

    if let Some(hit) = no_cull {
        assert!(
            close(got.t, hit.t),
            "query {idx} t: gpu {} vs cpu {}",
            got.t,
            hit.t
        );
        assert!(
            close(got.u, hit.u),
            "query {idx} u: gpu {} vs cpu {}",
            got.u,
            hit.u
        );
        assert!(
            close(got.v, hit.v),
            "query {idx} v: gpu {} vs cpu {}",
            got.v,
            hit.v
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuRayTriangle, queries: &[RayTriangleQuery]) {
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

/// A counter-clockwise triangle in the `z = 0` plane whose geometric normal
/// points toward `+z` (so its front face is the `+z` side).
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
    let gpu = GpuRayTriangle::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn front_facing_interior_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Front face is the +z side; strike it travelling toward -z, clearly inside.
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, -1.0)),
        v0,
        v1,
        v2,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_facing_interior_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Same interior point struck from the -z side: a back-facing hit (negative
    // determinant). The no-cull reference still hits; the cull reference misses.
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::new(0.0, 0.0, 1.0)),
        v0,
        v1,
        v2,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn parallel_ray_misses_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // Direction lies in the triangle plane: the determinant is exactly zero on
    // both devices, so the guard rejects it as a miss.
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::new(1.0, 0.0, 0.0)),
        v0,
        v1,
        v2,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn behind_origin_misses_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // The triangle sits behind the origin along the travel direction, so the
    // forward test (t > EPS) rejects the crossing.
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, 1.0)),
        v0,
        v1,
        v2,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn outside_triangle_misses_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    // The crossing lands clearly outside the triangle (u + v well above 1).
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.9, 0.9, -1.0), Vec3::new(0.0, 0.0, 1.0)),
        v0,
        v1,
        v2,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_triangle_misses_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    // Three collinear vertices: the determinant is exactly zero (integer
    // coordinates), so the guard rejects every ray as a miss.
    let query = RayTriangleQuery::new(
        Ray::new(Vec3::new(0.5, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0)),
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
    let gpu = GpuRayTriangle::new(&ctx);
    let (v0, v1, v2) = unit_triangle();
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    // One batch mixing the deterministic fixtures with many random interior
    // hits, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        RayTriangleQuery::new(
            Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, -1.0)),
            v0,
            v1,
            v2,
        ),
        RayTriangleQuery::new(
            Ray::new(Vec3::new(0.9, 0.9, -1.0), Vec3::new(0.0, 0.0, 1.0)),
            v0,
            v1,
            v2,
        ),
    ];
    for _ in 0..48 {
        queries.push(interior_hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_interior_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayTriangle::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned interior hits (several workgroups'
    // worth) pins every reported field across many random triangle geometries.
    let queries: Vec<RayTriangleQuery> = (0..200).map(|_| interior_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
