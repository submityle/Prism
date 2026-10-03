//! Real-device parity for the analytic ray vs *bilinear patch* intersector
//! twin:
//! [`GpuRayBilinearPatch`](prism_volumetric_gpu::ray_bilinear_patch::GpuRayBilinearPatch)
//! must reproduce the `CPU` closed form of
//! `prism_render_architecture::ray_scene::bilinear_patch` —
//! `BilinearPatch::intersect` (Reshetov's *"Cool Patches"*, 2019) — across the
//! planar and non-planar branches, interior and corner hits, clear misses and a
//! randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the
//! translation-invariant edge vectors, the quadratic `a*u^2 + b*u + c = 0` with
//! its stable root pair, the per-root `solve_v` back-substitution, and the
//! oriented analytic normal `dP/du x dP/dv`. Because the reference and this
//! oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence the
//! ported kernel computes the same hits the reference does.
//!
//! # Parity criterion
//!
//! Every continuous output threads through products, quotients and one `sqrt`,
//! so a `GPU` result may land a few units in the last place from the scalar
//! oracle; `t`, `u`, `v` and each normal component are asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a relative floor of `1e-6` so
//! a near-zero expected value does not inflate the relative error. The discrete
//! `hit` and `front_face` flags are compared for exact equality.
//!
//! # Conditioning
//!
//! The branch-switch loci are the quadratic discriminant (`det ~ 0`, grazing),
//! the planar test (`|c| ~ 0`, where the quadratic collapses to the linear
//! branch), the parameter edges (`u` or `v` at `0`/`1`), the ray-interval ends
//! (`t` at `t_min`/`t_max`) and the degenerate tangent frames (`det2 ~ 0` or
//! `len2 ~ 0`). The named fixtures sit well inside a single branch and the
//! randomized sweep rejects any sample within a safety margin of each locus, so
//! both sides fold the identical verdict and no cliff can appear.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bilinear_patch`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_bilinear_patch::{
    GpuRayBilinearPatch, RayBilinearPatchQuery, RayBilinearPatchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any continuous output. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any continuous output, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Planar test floor shared with the kernel: `|c|` at or below this routes to
/// the linear branch, mirroring the reference `c == 0` test without a forbidden
/// bare `f32` equality.
const EPS: f32 = 1.0e-5;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Adds `a` and `b` componentwise.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Linear interpolation `a + t*(b - a)`, componentwise.
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Independent reimplementation of
/// `ray_scene::bilinear_patch::BilinearPatch::intersect`: reduce the ray to a
/// quadratic in `u`, back-substitute each root in `[0, 1]` to recover `(t, v)`,
/// and keep the nearest valid hit with its oriented unit normal. The planar
/// test uses the shared [`EPS`] floor so this oracle and the kernel pick the
/// same branch.
fn intersect_host(q: &RayBilinearPatchQuery) -> RayBilinearPatchResult {
    let miss = RayBilinearPatchResult {
        hit: 0,
        t: 0.0,
        normal: [0.0, 0.0, 0.0],
        front_face: 0,
        u: 0.0,
        v: 0.0,
    };

    let ro = q.origin;
    let rd = q.direction;
    // Ray-interval clamp mirroring the kernel and `Ray::new` for finite inputs.
    let t_lo = q.t_min.max(0.0);
    let t_hi_init = q.t_max.max(t_lo);

    let e10 = sub(q.p10, q.p00);
    let e11 = sub(q.p11, q.p10);
    let e00 = sub(q.p01, q.p00);
    let f = sub(q.p11, q.p01);
    let qn = cross(e10, sub(q.p01, q.p11));
    let q00 = sub(q.p00, ro);
    let q10 = sub(q.p10, ro);

    let a = dot(cross(q00, rd), e00);
    let c = dot(qn, rd);
    let b = dot(cross(q10, rd), e11) - a - c;

    let det = b * b - 4.0 * a * c;
    if det < 0.0 {
        return miss;
    }
    let sq = det.sqrt();

    let (u1, u2);
    if c.abs() <= EPS {
        if b.abs() <= EPS {
            return miss;
        }
        u1 = -a / b;
        u2 = -1.0;
    } else {
        let signed_sq = if b >= 0.0 { sq } else { -sq };
        let big = (-b - signed_sq) * 0.5;
        u1 = big / c;
        u2 = a / big;
    }

    let mut out = miss;
    let mut t_hi = t_hi_init;
    for u in [u1, u2] {
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let pa = mix(q00, q10, u);
        let pb = mix(e00, e11, u);
        let n0 = cross(rd, pb);
        let det2 = dot(n0, n0);
        if det2 <= 0.0 {
            continue;
        }
        let m = cross(n0, pa);
        let t = dot(m, pb) / det2;
        let v = dot(m, rd) / det2;
        if !(0.0..=1.0).contains(&v) {
            continue;
        }
        if t < t_lo || t > t_hi {
            continue;
        }
        let dpdu = add(scale(e10, 1.0 - v), scale(f, v));
        let dpdv = add(scale(e00, 1.0 - u), scale(e11, u));
        let g = cross(dpdu, dpdv);
        let len2 = dot(g, g);
        if len2 <= 0.0 {
            continue;
        }
        let inv_len = 1.0 / len2.sqrt();
        let outward = scale(g, inv_len);
        let facing = dot(rd, outward) < 0.0;
        let normal = if facing {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };
        t_hi = t;
        out = RayBilinearPatchResult {
            hit: 1,
            t,
            normal,
            front_face: u32::from(facing),
            u,
            v,
        };
    }
    out
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Safety margin for the randomized sweep's rejection bands.
const MARGIN: f32 = 0.02;

/// Returns `true` when a sweep sample sits too close to a branch-switch locus
/// and must be rejected, so the kept comparison stays far from any cliff. It
/// recomputes the quadratic and inspects both candidate roots: the planar
/// floor, the grazing discriminant, the near-singular stable root, the
/// parameter edges, the ray-interval ends and the degenerate tangent frames.
fn reject_sample(q: &RayBilinearPatchQuery) -> bool {
    let ro = q.origin;
    let rd = q.direction;
    let t_lo = q.t_min.max(0.0);
    let t_hi = q.t_max.max(t_lo);

    let e10 = sub(q.p10, q.p00);
    let e11 = sub(q.p11, q.p10);
    let e00 = sub(q.p01, q.p00);
    let f = sub(q.p11, q.p01);
    let qn = cross(e10, sub(q.p01, q.p11));
    let q00 = sub(q.p00, ro);
    let q10 = sub(q.p10, ro);

    let a = dot(cross(q00, rd), e00);
    let c = dot(qn, rd);
    let b = dot(cross(q10, rd), e11) - a - c;

    // Keep the sweep in the quadratic branch, well clear of the planar floor.
    if c.abs() < 0.05 {
        return true;
    }
    let det = b * b - 4.0 * a * c;
    if det < 0.0 {
        // Clean miss: nothing to disagree about, keep it.
        return false;
    }
    if det < 0.02 {
        // Grazing: the sqrt is ill-conditioned here, reject.
        return true;
    }
    let sq = det.sqrt();
    let signed_sq = if b >= 0.0 { sq } else { -sq };
    let big = (-b - signed_sq) * 0.5;
    if big.abs() < 1.0e-3 {
        return true;
    }
    let u1 = big / c;
    let u2 = a / big;

    for u in [u1, u2] {
        // Only roots near the valid interval can flip the verdict.
        if !(-0.5..=1.5).contains(&u) {
            continue;
        }
        if u.abs() < MARGIN || (u - 1.0).abs() < MARGIN {
            return true;
        }
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let pa = mix(q00, q10, u);
        let pb = mix(e00, e11, u);
        let n0 = cross(rd, pb);
        let det2 = dot(n0, n0);
        if det2 < 0.01 {
            return true;
        }
        let m = cross(n0, pa);
        let t = dot(m, pb) / det2;
        let v = dot(m, rd) / det2;
        if v.abs() < MARGIN || (v - 1.0).abs() < MARGIN {
            return true;
        }
        if (t - t_lo).abs() < MARGIN || (t - t_hi).abs() < MARGIN {
            return true;
        }
        let dpdu = add(scale(e10, 1.0 - v), scale(f, v));
        let dpdv = add(scale(e00, 1.0 - u), scale(e11, u));
        let g = cross(dpdu, dpdv);
        if dot(g, g) < 0.01 {
            return true;
        }
    }
    false
}

/// Dispatches `queries` and asserts every output matches the host oracle: the
/// `hit` flag exactly, and when hit the `front_face` flag exactly and `t`,
/// `u`, `v` and each normal component within tolerance.
fn check_batch(ctx: &GpuContext, gpu: &GpuRayBilinearPatch, queries: &[RayBilinearPatchQuery]) {
    let got: Vec<RayBilinearPatchResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden = intersect_host(q);
        assert_eq!(
            r.hit, golden.hit,
            "hit flag mismatch: gpu={} golden={} (query={q:?})",
            r.hit, golden.hit
        );
        if golden.hit == 1 {
            assert_eq!(
                r.front_face, golden.front_face,
                "front_face mismatch: gpu={} golden={} (query={q:?})",
                r.front_face, golden.front_face
            );
            assert!(
                close(r.t, golden.t),
                "t mismatch: gpu={} golden={} (query={q:?})",
                r.t,
                golden.t
            );
            assert!(
                close(r.u, golden.u),
                "u mismatch: gpu={} golden={} (query={q:?})",
                r.u,
                golden.u
            );
            assert!(
                close(r.v, golden.v),
                "v mismatch: gpu={} golden={} (query={q:?})",
                r.v,
                golden.v
            );
            for axis in 0..3 {
                assert!(
                    close(r.normal[axis], golden.normal[axis]),
                    "normal[{axis}] mismatch: gpu={} golden={} (query={q:?})",
                    r.normal[axis],
                    golden.normal[axis]
                );
            }
        }
    }
}

/// A planar unit quad in the `z = 0` plane, `u` along `+x`, `v` along `+y`.
const PLANAR: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) = (
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
);

/// A non-planar hyperbolic-paraboloid saddle `z = (1 - 2u)(1 - 2v)`.
const SADDLE: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) = (
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
);

/// Builds one query from a patch tuple and a ray.
fn make_query(
    patch: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    origin: [f32; 3],
    direction: [f32; 3],
    t_min: f32,
    t_max: f32,
) -> RayBilinearPatchQuery {
    RayBilinearPatchQuery {
        origin,
        direction,
        t_min,
        t_max,
        p00: patch.0,
        p10: patch.1,
        p11: patch.2,
        p01: patch.3,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ray_bilinear_patch parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn planar_center_hit_linear_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    let q = make_query(PLANAR, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0);
    // The planar quad gives c == 0, so the oracle takes the linear branch and
    // the straight-down ray hits dead centre on the front face.
    let golden = intersect_host(&q);
    assert_eq!(golden.hit, 1, "planar centre should hit");
    assert_eq!(golden.front_face, 1, "ray strikes the outward side");
    assert!(close(golden.u, 0.5) && close(golden.v, 0.5), "centre hit");
    assert!(close(golden.t, 2.0), "hit at t = 2");
    assert!(
        close(golden.normal[0], 0.0)
            && close(golden.normal[1], 0.0)
            && close(golden.normal[2], 1.0),
        "normal points back along +z"
    );
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn saddle_oblique_hit_quadratic_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    // An oblique ray makes c far from zero, exercising the stable quadratic
    // root pair on the non-planar saddle.
    let q = make_query(SADDLE, [0.0, 0.0, 3.0], [0.3, 0.2, -1.0], 0.0, 20.0);
    let golden = intersect_host(&q);
    assert_eq!(golden.hit, 1, "oblique saddle ray should hit");
    assert!(q.direction[0].abs() > 0.0, "ray is genuinely oblique");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn miss_far_off_the_patch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    // Parallel to the quad and far outside its extent: a clean miss.
    let q = make_query(PLANAR, [5.0, 5.0, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0);
    assert_eq!(intersect_host(&q).hit, 0, "far ray misses");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn miss_u_out_of_bounds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    // Projects to u well beyond 1, so the root is rejected.
    let q = make_query(PLANAR, [2.5, 0.0, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0);
    assert_eq!(intersect_host(&q).hit, 0, "out-of-bounds u misses");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn miss_behind_interval() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    // The patch sits at t = 2 but the interval is capped below it.
    let q = make_query(PLANAR, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 0.0, 1.0);
    assert_eq!(intersect_host(&q).hit, 0, "hit lies beyond t_max");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    let queries = vec![
        make_query(PLANAR, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0),
        make_query(SADDLE, [0.0, 0.0, 3.0], [0.3, 0.2, -1.0], 0.0, 20.0),
        make_query(PLANAR, [5.0, 5.0, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0),
        make_query(SADDLE, [-0.2, 0.1, 3.0], [-0.25, 0.15, -1.0], 0.0, 20.0),
        make_query(PLANAR, [0.5, -0.3, 2.0], [0.0, 0.0, -1.0], 0.0, 10.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayBilinearPatch::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_6f01;
    let mut queries: Vec<RayBilinearPatchQuery> = Vec::new();
    let mut guard = 0u32;
    while queries.len() < 512 {
        guard += 1;
        assert!(guard < 200_000, "rejection sampling failed to converge");

        let p00 = [
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
        ];
        let p10 = [
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
        ];
        let p11 = [
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
        ];
        let p01 = [
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
            draw(&mut state, -1.5, 1.5),
        ];
        let patch = (p00, p10, p11, p01);
        let centroid = scale(add(add(p00, p10), add(p11, p01)), 0.25);

        let origin = [
            draw(&mut state, -2.0, 2.0),
            draw(&mut state, -2.0, 2.0),
            draw(&mut state, 2.5, 4.0),
        ];
        // Aim roughly through the patch so a meaningful fraction hit, then jitter.
        let aim = sub(centroid, origin);
        let direction = [
            aim[0] + draw(&mut state, -0.4, 0.4),
            aim[1] + draw(&mut state, -0.4, 0.4),
            aim[2],
        ];

        let q = make_query(patch, origin, direction, 0.0, 20.0);
        if reject_sample(&q) {
            continue;
        }
        queries.push(q);
    }
    check_batch(&ctx, &gpu, &queries);
}
