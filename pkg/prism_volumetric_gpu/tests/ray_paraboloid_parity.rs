//! Real-device parity for the analytic *paraboloid* ray-intersection twin:
//! `GpuRayParaboloid` must reproduce the `CPU` closed form of the oracle
//! `prism_render_architecture::ray_scene::paraboloid` — `Paraboloid::intersect`
//! — across frontal hits, side-wall hits, out-of-range misses, back-face hits
//! and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form in scalar `f32`,
//! operation-for-operation: substitute the ray into the implicit quadric
//! `k * rho^2 = z`, solve the quadratic (or linear) in `t`, clip the roots to
//! the finite `z` in `[0, h]` band, take the nearest valid root, build the
//! analytic gradient normal and orient it against the ray. Because the
//! reference and this oracle are both scalar `f32`, a `GPU == oracle` pass is
//! direct evidence the ported kernel computes the same intersection the
//! reference does.
//!
//! # Parity criterion
//!
//! The discrete `hit` and `front_face` flags are asserted exactly. The
//! continuous `t` and `normal` thread through products, sums, a discriminant
//! `sqrt` and a normalize divide, so a `GPU` result may land a few units in the
//! last place from the scalar oracle; each is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6`
//! so a near-zero expected component does not inflate the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep reject-sample away from every branch cliff
//! so a last-place wobble never flips which side of a comparison the `CPU` and
//! `GPU` land on: the leading coefficient near zero (a ray near-parallel to the
//! axis), the discriminant near zero (a tangent ray), a root near a `t_min` /
//! `t_max` end, the axial coordinate near the `0` / `h` band edges, the squared
//! gradient near zero, and the incidence dot near zero (a grazing hit that is
//! on the front/back-face cliff). The random queries stay off the linear
//! (axis-parallel) branch entirely.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::paraboloid`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_paraboloid::{
    GpuRayParaboloid, RayParaboloidQuery, RayParaboloidResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on `t` and each normal component. A `GPU` multiply-add may
/// land a few units in the last place from the scalar oracle; `1e-4` admits
/// that legal slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on `t` and each normal component, applied for larger
/// magnitudes where a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Guard separating a "nonzero" leading coefficient from the parallel-to-axis
/// degenerate branch, matching the kernel's `A_EPS`. The oracle uses the same
/// threshold so the two never disagree on which branch is taken; the sweep then
/// keeps real queries far above it.
const A_EPS: f32 = 1.0e-7;

/// A resolved host-oracle intersection, mirroring the kernel's reported fields.
struct Hit {
    /// Ray parameter at the nearest valid hit.
    t: f32,
    /// Unit surface normal oriented against the ray.
    normal: [f32; 3],
    /// `true` when the ray struck the outward convex side.
    front_face: bool,
}

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise difference `a - b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a 3-vector by a scalar.
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Length of a 3-vector via a host-side `sqrt` (not transcendental).
fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

/// Independent reimplementation of the reference `Paraboloid::intersect`:
/// substitute the ray into the implicit quadric `k * rho^2 = z`, solve the
/// quadratic (or linear) in `t`, clip to the `z` in `[0, h]` band, pick the
/// nearest valid root, and orient the analytic gradient normal against the ray.
///
/// Returns `None` on a miss (degenerate paraboloid/ray, no real root, no root
/// inside the interval and band, or a degenerate gradient).
fn intersect(
    origin: [f32; 3],
    dir: [f32; 3],
    t_min: f32,
    t_max: f32,
    apex: [f32; 3],
    top: [f32; 3],
    radius: f32,
) -> Option<Hit> {
    let w = sub3(top, apex);
    let h2 = dot3(w, w);
    if h2 <= 0.0 {
        return None;
    }
    let r2 = radius * radius;
    if r2 <= 0.0 {
        return None;
    }
    let dd = dot3(dir, dir);
    if dd <= 0.0 {
        return None;
    }
    let h = h2.sqrt();
    let inv_h = 1.0 / h;
    let n = scale3(w, inv_h);
    let k = h / r2;

    let a = sub3(origin, apex);
    let za = dot3(a, n);
    let zd = dot3(dir, n);
    let ad = dot3(a, dir);
    let aa = dot3(a, a);

    let coeff_a = k * (dd - zd * zd);
    let coeff_b = 2.0 * k * (ad - za * zd) - zd;
    let coeff_c = k * (aa - za * za) - za;

    let (root0, root1, root_count) = if coeff_a.abs() > A_EPS {
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let inv_2a = 1.0 / (2.0 * coeff_a);
        let ra = (-coeff_b - sq) * inv_2a;
        let rb = (-coeff_b + sq) * inv_2a;
        (ra.min(rb), ra.max(rb), 2u32)
    } else if coeff_b.abs() > A_EPS {
        let r = -coeff_c / coeff_b;
        (r, r, 1u32)
    } else {
        return None;
    };

    let mut best_t = f32::INFINITY;
    let mut best_out = [0.0_f32; 3];
    for i in 0..root_count {
        let t = if i == 0 { root0 } else { root1 };
        if !(t >= t_min && t <= t_max) || t >= best_t {
            continue;
        }
        let z = za + t * zd;
        if z < 0.0 || z > h {
            continue;
        }
        let p = add3(a, scale3(dir, t));
        let perp = sub3(p, scale3(n, z));
        let grad = sub3(scale3(perp, 2.0 * k), n);
        let nn = dot3(grad, grad);
        if nn <= 0.0 {
            continue;
        }
        best_t = t;
        best_out = scale3(grad, 1.0 / nn.sqrt());
    }

    if !best_t.is_finite() {
        return None;
    }
    let incidence = dot3(dir, best_out);
    let front_face = incidence < 0.0;
    let normal = if front_face {
        best_out
    } else {
        scale3(best_out, -1.0)
    };
    Some(Hit {
        t: best_t,
        normal,
        front_face,
    })
}

/// Pins one `GPU` result against the host oracle: the discrete `hit` and
/// `front_face` flags exactly, and `t`/`normal` under the shared tolerance. On
/// a hit the normal is additionally checked to be unit length.
fn check_one(idx: usize, got: &RayParaboloidResult, want: &Option<Hit>) {
    match want {
        None => {
            assert_eq!(got.hit, 0, "query {idx}: oracle miss but gpu reports a hit");
        }
        Some(h) => {
            assert_eq!(got.hit, 1, "query {idx}: oracle hit but gpu reports a miss");
            assert_eq!(
                got.front_face,
                u32::from(h.front_face),
                "query {idx}: front_face flag disagrees (gpu {} vs cpu {})",
                got.front_face,
                u32::from(h.front_face)
            );
            assert!(
                close(got.t, h.t, SD_ABS, SD_REL),
                "query {idx} t: gpu {} vs cpu {}",
                got.t,
                h.t
            );
            for (axis, (g, w)) in got.normal.iter().zip(h.normal.iter()).enumerate() {
                assert!(
                    close(*g, *w, SD_ABS, SD_REL),
                    "query {idx} normal[{axis}]: gpu {g} vs cpu {w}"
                );
            }
            let len = length3(got.normal);
            assert!(
                close(len, 1.0, SD_ABS, SD_REL),
                "query {idx}: gpu normal must be unit length, got {len}"
            );
        }
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRayParaboloid, queries: &[RayParaboloidQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = intersect(
            q.origin,
            q.direction,
            q.t_min,
            q.t_max,
            q.apex,
            q.top,
            q.radius,
        );
        check_one(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Returns a unit copy of `v`; callers reject near-zero draws first.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = length3(v);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Whether a random query sits clear of every branch cliff, so the `CPU` and
/// `GPU` cannot pick different sides of a comparison (see `# Conditioning`).
fn well_conditioned(q: &RayParaboloidQuery) -> bool {
    let w = sub3(q.top, q.apex);
    let h2 = dot3(w, w);
    if h2 <= 0.0 {
        return false;
    }
    let r2 = q.radius * q.radius;
    if r2 <= 0.0 {
        return false;
    }
    let dd = dot3(q.direction, q.direction);
    // Keep the direction well above the zero-length degenerate guard.
    if dd <= 0.2 {
        return false;
    }
    let h = h2.sqrt();
    let n = scale3(w, 1.0 / h);
    let k = h / r2;
    let a = sub3(q.origin, q.apex);
    let za = dot3(a, n);
    let zd = dot3(q.direction, n);
    let ad = dot3(a, q.direction);
    let aa = dot3(a, a);
    let coeff_a = k * (dd - zd * zd);
    let coeff_b = 2.0 * k * (ad - za * zd) - zd;
    let coeff_c = k * (aa - za * za) - za;

    // Stay well off the parallel-to-axis (linear) branch cliff.
    if coeff_a.abs() <= 1.0e-2 {
        return false;
    }
    let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
    let disc_margin = 1.0e-3 * (coeff_b * coeff_b + (4.0 * coeff_a * coeff_c).abs() + 1.0e-6);
    if disc.abs() <= disc_margin {
        return false;
    }
    if disc < 0.0 {
        // A clean miss (discriminant safely negative) is well conditioned.
        return true;
    }
    let sq = disc.sqrt();
    let inv_2a = 1.0 / (2.0 * coeff_a);
    let ra = (-coeff_b - sq) * inv_2a;
    let rb = (-coeff_b + sq) * inv_2a;
    let roots = [ra.min(rb), ra.max(rb)];

    let mut best_t = f32::INFINITY;
    let mut best_out = [0.0_f32; 3];
    for &t in &roots {
        // Reject whenever any root sits near a parameter-interval end, where
        // the in-range test would be last-place sensitive.
        if (t - q.t_min).abs() <= 2.0e-3 || (t - q.t_max).abs() <= 2.0e-3 {
            return false;
        }
        if !(t >= q.t_min && t <= q.t_max) || t >= best_t {
            continue;
        }
        let z = za + t * zd;
        // Reject near the band edges, where the z in [0, h] clip is sensitive.
        if z.abs() <= 2.0e-3 || (z - h).abs() <= 2.0e-3 {
            return false;
        }
        if z < 0.0 || z > h {
            continue;
        }
        let p = add3(a, scale3(q.direction, t));
        let perp = sub3(p, scale3(n, z));
        let grad = sub3(scale3(perp, 2.0 * k), n);
        let nn = dot3(grad, grad);
        if nn <= 1.0e-4 {
            return false;
        }
        best_t = t;
        best_out = scale3(grad, 1.0 / nn.sqrt());
    }
    if best_t.is_finite() {
        // Keep off the front/back-face cliff where the incidence dot flips sign.
        let incidence = dot3(q.direction, best_out);
        if incidence.abs() <= 5.0e-3 {
            return false;
        }
    }
    true
}

/// Builds one well-conditioned random query: a bounded apex, a rim offset along
/// a random unit axis of length `[0.8, 3.0]`, a rim radius in `[0.5, 2.5]`, a
/// bounded origin and direction, over the fixed `[0, 50]` parameter interval.
fn random_query(state: &mut u64) -> RayParaboloidQuery {
    loop {
        let apex = [
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
        ];
        let axis_raw = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        if length3(axis_raw) < 0.3 {
            continue;
        }
        let axis = normalize(axis_raw);
        let h_len = uniform(state, 0.8, 3.0);
        let top = add3(apex, scale3(axis, h_len));
        let radius = uniform(state, 0.5, 2.5);
        let origin = [
            uniform(state, -6.0, 6.0),
            uniform(state, -6.0, 6.0),
            uniform(state, -6.0, 6.0),
        ];
        let direction = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let q = RayParaboloidQuery::new(origin, direction, 0.0, 50.0, apex, top, radius);
        if well_conditioned(&q) {
            return q;
        }
    }
}

/// A fixed battery of named queries: a frontal/side hit, a second side-wall hit
/// from another angle, an out-of-range miss, and a concave back-face hit.
fn fixture_queries() -> Vec<RayParaboloidQuery> {
    vec![
        // Side-wall hit on the convex side: dish apex at the origin pointing +z,
        // ray along +y strikes the wall at y = -1/sqrt(2); front_face.
        RayParaboloidQuery::new(
            [0.0, -3.0, 1.0],
            [0.0, 1.0, 0.0],
            0.0,
            50.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
        ),
        // Another side-wall hit, a wider dish entered along +x.
        RayParaboloidQuery::new(
            [-4.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            0.0,
            50.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            1.5,
        ),
        // Same ray as the first fixture but the interval ends before the hit at
        // t ~= 2.29, so it is an out-of-range miss.
        RayParaboloidQuery::new(
            [0.0, -3.0, 1.0],
            [0.0, 1.0, 0.0],
            0.0,
            0.5,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
        ),
        // Back-face hit: a ray launched from inside the bowl strikes the
        // concave side, so the oriented normal reports front_face = false.
        RayParaboloidQuery::new(
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            0.0,
            50.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            1.0,
        ),
    ]
}

#[test]
fn frontal_side_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = intersect(
        q.origin,
        q.direction,
        q.t_min,
        q.t_max,
        q.apex,
        q.top,
        q.radius,
    );
    check_one(0, &got[0], &want);
    // A convex-side hit reports a hit on the outward (front) face.
    assert_eq!(got[0].hit, 1, "the frontal ray must hit");
    assert_eq!(got[0].front_face, 1, "the convex side is the front face");
}

#[test]
fn out_of_range_ray_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    let q = fixture_queries()[2];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].hit, 0,
        "a hit beyond t_max must be clipped to a miss"
    );
    let want = intersect(
        q.origin,
        q.direction,
        q.t_min,
        q.t_max,
        q.apex,
        q.top,
        q.radius,
    );
    check_one(0, &got[0], &want);
}

#[test]
fn back_face_hit_reports_back_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    let q = fixture_queries()[3];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = intersect(
        q.origin,
        q.direction,
        q.t_min,
        q.t_max,
        q.apex,
        q.top,
        q.radius,
    );
    check_one(0, &got[0], &want);
    assert_eq!(got[0].hit, 1, "the inside ray must hit the concave wall");
    assert_eq!(
        got[0].front_face, 0,
        "a concave-side hit reports the back face"
    );
}

#[test]
fn degenerate_paraboloids_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    // Zero-length axis, zero radius, and zero-length direction each short the
    // kernel to a miss, matching the reference guards.
    let zero_axis = RayParaboloidQuery::new(
        [0.0, -3.0, 1.0],
        [0.0, 1.0, 0.0],
        0.0,
        50.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        1.0,
    );
    let zero_radius = RayParaboloidQuery::new(
        [0.0, -3.0, 1.0],
        [0.0, 1.0, 0.0],
        0.0,
        50.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        0.0,
    );
    let zero_dir = RayParaboloidQuery::new(
        [0.0, -3.0, 1.0],
        [0.0, 0.0, 0.0],
        0.0,
        50.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
    );
    let queries = [zero_axis, zero_radius, zero_dir];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 3);
    for (idx, result) in got.iter().enumerate() {
        assert_eq!(result.hit, 0, "degenerate query {idx} must miss");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    let mut state = 0x0bad_c0de_dead_beef_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported hit across a wide span of rays and paraboloids.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ray_paraboloid parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRayParaboloid::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}
