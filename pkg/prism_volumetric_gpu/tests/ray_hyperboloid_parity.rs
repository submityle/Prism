//! Real-device parity for the analytic ray-hyperboloid-of-one-sheet twin:
//! [`GpuRayHyperboloid`](prism_volumetric_gpu::ray_hyperboloid::GpuRayHyperboloid)
//! must reproduce the closed-form reference intersection across a throat (waist)
//! hit, an oblique lateral hit, an out-of-band miss (the only algebraic roots
//! land where the axial coordinate leaves the finite band), a back-face hit (a
//! ray leaving the throat from inside), a pointing-away miss, a `flare = 0`
//! cylinder degeneracy, plus a rejection-sampled random batch compared element
//! for element.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent reimplementation of the same closed form documented on the twin
//! (substitute the ray into the implicit quadric
//! `F = |p - center|^2 - (1 + flare^2) * dot(p - center, n_hat)^2 - waist^2`,
//! solve the scalar quadratic, clip both roots to the finite axial band and the
//! ray interval, pick the nearest valid root, normalize the analytic gradient
//! and orient it against the incident ray). A passing parity run is therefore
//! evidence the `WGSL` kernel and an independent `CPU` evaluation of the same
//! geometry agree, not merely that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! `sqrt`, so the two evaluations compute the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! host leaves separate, perturbing the low mantissa bits by a few units in the
//! last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the `f32` fields while pinning the
//! hit flag and the front-face flag exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! branch boundaries: random hits land with the axial coordinate comfortably
//! inside the band, two well-separated roots and a large discriminant, and the
//! hit parameter well away from the interval ends; the deterministic misses
//! clear every branch; no fixture is tuned to a grazing double root, a band
//! edge or a vanishing leading coefficient. This keeps `CPU` and `GPU` on the
//! same side of every branch regardless of a few units in the last place.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::hyperboloid`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_hyperboloid::{
    GpuRayHyperboloid, RayHyperboloidQuery, RayHyperboloidResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on the `f32` fields.
const DIST_ABS: f32 = 1.0e-4;
/// Relative parity slope on the `f32` fields.
const DIST_REL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// A finite sentinel standing in for the "no candidate yet" running best ray
/// parameter, mirroring the kernel. Every fixture parameter is far smaller.
const T_SENTINEL: f32 = 1.0e30;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= DIST_ABS || diff <= DIST_REL * a.abs().max(b.abs()).max(REL_FLOOR)
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

/// The host result shape: hit flag, ray parameter, oriented unit normal and the
/// front-face flag, mirroring [`RayHyperboloidResult`].
struct Oracle {
    hit: u32,
    t: f32,
    normal: [f32; 3],
    front_face: u32,
}

/// Independent reimplementation of the reference `Hyperboloid::intersect` closed
/// form, used as the parity oracle. Ordered comparisons replace every bare
/// `f32` equality the reference writes.
fn oracle(q: &RayHyperboloidQuery) -> Oracle {
    let miss = Oracle {
        hit: 0,
        t: 0.0,
        normal: [0.0, 0.0, 0.0],
        front_face: 0,
    };

    let w = v_sub(q.top, q.center);
    let h2 = v_dot(w, w);
    let dd = v_dot(q.dir, q.dir);
    if h2 <= 0.0 || dd <= 0.0 {
        return miss;
    }

    let h = h2.sqrt();
    let inv_h = 1.0 / h;
    let n = v_scale(w, inv_h);
    let g = 1.0 + q.flare * q.flare;

    let a = v_sub(q.origin, q.center);
    let za = v_dot(a, n);
    let zd = v_dot(q.dir, n);
    let ad = v_dot(a, q.dir);
    let aa = v_dot(a, a);

    let coeff_a = dd - g * zd * zd;
    let coeff_b = 2.0 * (ad - g * za * zd);
    let coeff_c = aa - g * za * za - q.waist * q.waist;

    // Candidate roots: quadratic, or linear when the leading coefficient is
    // zero. `abs(x) > 0.0` reproduces the reference `x != 0.0` with an ordered
    // comparison only.
    let (r0, r1, has_roots) = if coeff_a.abs() > 0.0 {
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if disc < 0.0 {
            return miss;
        }
        let sqrt_disc = disc.sqrt();
        let inv_2a = 1.0 / (2.0 * coeff_a);
        let ra = (-coeff_b - sqrt_disc) * inv_2a;
        let rb = (-coeff_b + sqrt_disc) * inv_2a;
        (ra.min(rb), ra.max(rb), true)
    } else if coeff_b.abs() > 0.0 {
        let r = -coeff_c / coeff_b;
        (r, r, true)
    } else {
        (0.0, 0.0, false)
    };

    if !has_roots {
        return miss;
    }

    let mut best_t = T_SENTINEL;
    let mut best_outward = [0.0f32; 3];
    let mut found = false;
    for t in [r0, r1] {
        if t < q.t_min || t > q.t_max || t >= best_t {
            continue;
        }
        let z = za + t * zd;
        if z < -h || z > h {
            continue;
        }
        let p = v_add(a, v_scale(q.dir, t));
        let gz = g * z;
        let grad = v_scale(v_sub(p, v_scale(n, gz)), 2.0);
        let nn = v_dot(grad, grad);
        if nn <= 0.0 {
            continue;
        }
        best_t = t;
        best_outward = v_scale(grad, 1.0 / nn.sqrt());
        found = true;
    }

    if !found {
        return miss;
    }

    let front_face = v_dot(q.dir, best_outward) < 0.0;
    let normal = if front_face {
        best_outward
    } else {
        [-best_outward[0], -best_outward[1], -best_outward[2]]
    };
    Oracle {
        hit: 1,
        t: best_t,
        normal,
        front_face: u32::from(front_face),
    }
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

/// Builds a clearly-conditioned lateral-wall hit: a random hyperboloid struck by
/// a mostly-radial ray that crosses the wall at an axial coordinate comfortably
/// inside the finite band, with two well-separated roots and a large
/// discriminant, so `CPU` and `GPU` land on the same branch regardless of a few
/// units in the last place. The query is rejection-sampled against the same
/// coefficients the kernel evaluates.
fn hit_query(state: &mut u64) -> RayHyperboloidQuery {
    loop {
        let axis = rand_unit(state);
        let center = rand_vec(state, 3.0);
        let half = 2.0 + lcg(state) * 2.0;
        let top = v_add(center, v_scale(axis, half));
        let waist = 0.8 + lcg(state) * 1.2;
        let flare = 0.3 + lcg(state) * 0.7;

        // A surface point well inside the band: axial `z0` in roughly the middle
        // half of `[-h, h]`, radius from the profile `rho(z)^2 = waist^2 +
        // flare^2 * z0^2`.
        let z0 = signed(state, half * 0.5);
        let radius = (waist * waist + flare * flare * z0 * z0).sqrt();
        let perp = perp_unit(axis);
        let perp2 = normalize(v_cross(axis, perp));
        let cu = signed(state, 1.0);
        let su = signed(state, 1.0);
        if cu * cu + su * su < 0.2 {
            continue;
        }
        let radial = normalize(v_add(v_scale(perp, cu), v_scale(perp2, su)));
        let surface = v_add(v_add(center, v_scale(axis, z0)), v_scale(radial, radius));

        // Fire a mostly-radial ray in from outside the wall.
        let dist = 4.0 + lcg(state) * 4.0;
        let origin = v_add(surface, v_scale(radial, dist));
        let dir = normalize(v_sub(surface, origin));
        let t_min = 0.0;
        let t_max = 50.0;

        let query = RayHyperboloidQuery::new(origin, dir, center, top, waist, flare, t_min, t_max);

        // Replicate the reference coefficients to reject ill-conditioned draws.
        let a = v_sub(origin, center);
        let n = axis;
        let g = 1.0 + flare * flare;
        let za = v_dot(a, n);
        let zd = v_dot(dir, n);
        let ad = v_dot(a, dir);
        let aa = v_dot(a, a);
        let dd = v_dot(dir, dir);
        let coeff_a = dd - g * zd * zd;
        let coeff_b = 2.0 * (ad - g * za * zd);
        let coeff_c = aa - g * za * za - waist * waist;
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if coeff_a.abs() < 1.0e-2 || disc < 4.0e-2 {
            continue;
        }

        // Keep only clear hits whose nearest root is well inside the band and
        // well away from the interval ends.
        let got = oracle(&query);
        if got.hit == 1 {
            let z = za + got.t * zd;
            if got.t > 0.3 && got.t < t_max - 0.3 && z.abs() < half - 0.3 {
                return query;
            }
        }
    }
}

/// Pins one `GPU` result against the host oracle for `query`: the hit flag and
/// the front-face flag must match exactly, and when present the ray parameter
/// and the unit normal must agree within the parity bound.
fn pin(idx: usize, query: &RayHyperboloidQuery, got: &RayHyperboloidResult) {
    let want = oracle(query);
    assert_eq!(
        got.hit, want.hit,
        "query {idx}: hit flag must match the oracle (gpu {}, cpu {})",
        got.hit, want.hit
    );
    if want.hit == 1 {
        assert!(
            close(got.t, want.t),
            "query {idx} hit.t: gpu {} vs cpu {}",
            got.t,
            want.t
        );
        assert_eq!(
            got.front_face, want.front_face,
            "query {idx}: front-face flag must match the oracle (gpu {}, cpu {})",
            got.front_face, want.front_face
        );
        assert!(
            close(got.normal[0], want.normal[0])
                && close(got.normal[1], want.normal[1])
                && close(got.normal[2], want.normal[2]),
            "query {idx} normal: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
            got.normal[0],
            got.normal[1],
            got.normal[2],
            want.normal[0],
            want.normal[1],
            want.normal[2]
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRayHyperboloid, queries: &[RayHyperboloidQuery]) {
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
    let gpu = GpuRayHyperboloid::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.intersect(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn throat_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // A horizontal ray through the waist plane strikes the near wall at x = -1
    // (t = 9) on a unit-throat hyperboloid.
    let query = RayHyperboloidQuery::new(
        [-10.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
        0.5,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn oblique_lateral_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // A slanted ray crossing the wall at an off-waist axial coordinate inside
    // the band.
    let query = RayHyperboloidQuery::new(
        [-8.0, 2.0, -1.0],
        normalize([1.0, -0.2, 0.15]),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        1.2,
        0.4,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn out_of_band_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // The only algebraic crossings sit at axial z = 5, far outside the band
    // [-2, 2], so every root is culled and the result is a miss.
    let query = RayHyperboloidQuery::new(
        [-10.0, 0.0, 5.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
        0.5,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_face_hit_from_inside_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // A ray leaving the throat from the center strikes the wall from inside at
    // x = 1 (t = 1), so the front-face flag is zero.
    let query = RayHyperboloidQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
        0.5,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn pointing_away_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // The hyperboloid lies entirely behind the origin along this direction: both
    // roots are negative and the clamped interval culls them.
    let query = RayHyperboloidQuery::new(
        [-10.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
        0.5,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn flare_zero_cylinder_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    // `flare = 0` degenerates to a cylinder of radius `waist`; a horizontal ray
    // strikes the near wall at x = -1.5 inside the band.
    let query = RayHyperboloidQuery::new(
        [-10.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        1.5,
        0.0,
        0.0,
        100.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random lateral hits,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element for element.
    let mut queries = vec![
        RayHyperboloidQuery::new(
            [-10.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
            0.5,
            0.0,
            100.0,
        ),
        RayHyperboloidQuery::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
            0.5,
            0.0,
            100.0,
        ),
        RayHyperboloidQuery::new(
            [-10.0, 0.0, 5.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
            0.5,
            0.0,
            100.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayHyperboloid::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned lateral hits (several workgroups'
    // worth) pins the hit parameter, oriented normal and front-face flag across
    // many random hyperboloid geometries.
    let queries: Vec<RayHyperboloidQuery> = (0..512).map(|_| hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
