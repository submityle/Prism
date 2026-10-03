//! Real-device parity for the ray vs oriented rectangle (parallelogram) twin:
//! [`GpuRayRectangle`](prism_volumetric_gpu::ray_rectangle::GpuRayRectangle)
//! must reproduce the analytic golden
//! `prism_render_architecture::ray_scene::rectangle::Rectangle::intersect`
//! across an empty batch, a frontal center hit, an oblique interior hit, an
//! edge-outside miss, a back-face hit with a flipped normal, a mixed fixture
//! batch and a large pseudo-random sweep compared lane for lane.
//!
//! The host oracle in this file re-derives the closed form independently (it
//! does **not** import the golden crate), so a passing run is direct evidence
//! the ported kernel folds the same ray/plane solve, the same reciprocal-basis
//! containment test and the same incident-oriented normal — not merely that
//! its shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of one guarded plane
//! division, a reciprocal-basis `2×2` solve and one `sqrt`, so `CPU` and `GPU`
//! evaluate the same closed form in the same associativity. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! the continuous `t` and `normal` values yet asserts an *exact* match on the
//! discrete `hit` and `front_face` flags. Every fixture and every random query
//! is placed clear of the plane-parallel, edge (`|a|`/`|b|` near `1`) and
//! `[t_min, t_max]` boundaries, the only places a legal `ULP` perturbation can
//! flip a verdict, so the exact-flag assertion is unconditional.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::rectangle`；
//! 无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_rectangle::{GpuRayRectangle, RayRectangleQuery, RayRectangleResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the plane parameter and normal components. A `GPU`
/// may fuse a multiply-add the scalar reference leaves separate, perturbing the
/// low mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Magnitude floor guarding the ray/plane division, matching the kernel's
/// `DENOM_EPS`: a direction whose component along the plane normal has
/// magnitude at or below this floor runs parallel to the plane and never
/// crosses it. The oracle uses the identical guard so the discrete `hit`
/// verdict matches the device exactly on well-conditioned queries.
const DENOM_EPS: f32 = 1.0e-12;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Componentwise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Componentwise `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by the scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A miss verdict: the all-zero lane the kernel writes for any rejected query.
fn miss() -> RayRectangleResult {
    RayRectangleResult {
        hit: false,
        t: 0.0,
        normal: [0.0, 0.0, 0.0],
        front_face: false,
    }
}

/// Independent host re-derivation of the golden ray/rectangle intersection,
/// guard for guard. It does not call the reference crate; it reimplements the
/// closed form so the parity assertion compares two independent solutions.
fn oracle(q: &RayRectangleQuery) -> RayRectangleResult {
    let n = cross(q.axis_u, q.axis_v);
    let nn = dot(n, n);
    // A zero-area (parallel or zero-length edges) rectangle has no plane.
    if nn <= 0.0 {
        return miss();
    }
    // `|denom|` at or below the floor means the ray runs parallel to the plane.
    let denom = dot(q.dir, n);
    if denom.abs() <= DENOM_EPS {
        return miss();
    }
    let oc = sub(q.center, q.origin);
    let t = dot(oc, n) / denom;
    if t < q.t_min || t > q.t_max {
        return miss();
    }
    // Reciprocal-basis solve of `p = a·u + b·v`; the determinant equals `nn`.
    let p = sub(add(q.origin, scale(q.dir, t)), q.center);
    let uu = dot(q.axis_u, q.axis_u);
    let vv = dot(q.axis_v, q.axis_v);
    let uv = dot(q.axis_u, q.axis_v);
    let pu = dot(p, q.axis_u);
    let pv = dot(p, q.axis_v);
    let a = (vv * pu - uv * pv) / nn;
    let b = (uu * pv - uv * pu) / nn;
    if !(-1.0..=1.0).contains(&a) || !(-1.0..=1.0).contains(&b) {
        return miss();
    }
    let inv = 1.0 / nn.sqrt();
    let unit = scale(n, inv);
    let front = denom < 0.0;
    let normal = if front { unit } else { scale(unit, -1.0) };
    RayRectangleResult {
        hit: true,
        t,
        normal,
        front_face: front,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// independent host oracle: the `hit` and `front_face` flags match exactly, and
/// the plane parameter and each normal component match within tolerance when
/// the query hits. Returns the `GPU` verdicts for extra per-test assertions.
/// Use only for queries placed clear of every boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRayRectangle,
    queries: &[RayRectangleQuery],
) -> Vec<RayRectangleResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let o = oracle(q);
        assert_eq!(
            g.hit, o.hit,
            "lane {lane}: hit gpu {} vs cpu {}",
            g.hit, o.hit
        );
        assert_eq!(
            g.front_face, o.front_face,
            "lane {lane}: front_face gpu {} vs cpu {}",
            g.front_face, o.front_face
        );
        if o.hit {
            assert!(close(g.t, o.t), "lane {lane}: t gpu {} vs cpu {}", g.t, o.t);
            for axis in 0..3 {
                assert!(
                    close(g.normal[axis], o.normal[axis]),
                    "lane {lane}: normal[{axis}] gpu {} vs cpu {}",
                    g.normal[axis],
                    o.normal[axis]
                );
            }
        }
    }
    got
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

/// Maps a `[0, 1)` sample into `[lo, hi)`.
fn uniform(x: f32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * x
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn frontal_center_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // Unit rectangle in the xy plane, normal +z. The ray drops straight down
    // -z onto the center: a frontal hit (dir·n < 0) with the +z normal.
    let q = RayRectangleQuery::new(
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -1.0],
        0.0,
        100.0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].hit && got[0].front_face,
        "frontal ray should hit front"
    );
    assert!(close(got[0].t, 5.0), "t {}", got[0].t);
    assert!(
        close(got[0].normal[2], 1.0),
        "normal.z {}",
        got[0].normal[2]
    );
}

#[test]
fn oblique_interior_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // Oblique descent landing at (0.4, 0.3) in the plane — well inside the unit
    // parallelogram, clear of the edges.
    let q = RayRectangleQuery::new(
        [0.0, 0.0, 5.0],
        [0.08, 0.06, -1.0],
        0.0,
        100.0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].hit && got[0].front_face,
        "oblique ray should hit front"
    );
    assert!(close(got[0].t, 5.0), "t {}", got[0].t);
}

#[test]
fn edge_outside_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // Lands at a = 2.0, well past the |a| <= 1 half-edge extent: a clear miss.
    let q = RayRectangleQuery::new(
        [2.0, 0.0, 5.0],
        [0.0, 0.0, -1.0],
        0.0,
        100.0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(!got[0].hit, "a ray past the half-edge extent must miss");
}

#[test]
fn back_face_hit_flips_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // Ray arrives from behind (-z side) heading +z: dir·n > 0, so the hit is a
    // back face with front_face == false and the normal flipped to -z.
    let q = RayRectangleQuery::new(
        [0.0, 0.0, -5.0],
        [0.0, 0.0, 1.0],
        0.0,
        100.0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        got[0].hit && !got[0].front_face,
        "back arrival hits back face"
    );
    assert!(close(got[0].t, 5.0), "t {}", got[0].t);
    assert!(
        close(got[0].normal[2], -1.0),
        "normal.z {}",
        got[0].normal[2]
    );
}

#[test]
fn parallel_ray_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // Ray travelling in +x is parallel to the xy-plane rectangle (dir·n == 0),
    // so it never crosses the plane: a clear miss on both devices.
    let q = RayRectangleQuery::new(
        [-5.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        0.0,
        100.0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(!got[0].hit, "a ray parallel to the plane must miss");
}

#[test]
fn fixture_batch_covers_both_faces() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);
    // A non-orthogonal (sheared) parallelogram away from the origin, probed by
    // a front hit, a back hit and an edge miss in one dispatch.
    let center = [1.0, -0.5, 2.0];
    let au = [1.0, 0.2, 0.0];
    let av = [0.3, 1.0, 0.0];
    let front = RayRectangleQuery::new(
        [1.0, -0.5, 7.0],
        [0.0, 0.0, -1.0],
        0.0,
        100.0,
        center,
        au,
        av,
    );
    let back = RayRectangleQuery::new(
        [1.0, -0.5, -7.0],
        [0.0, 0.0, 1.0],
        0.0,
        100.0,
        center,
        au,
        av,
    );
    let outside = RayRectangleQuery::new(
        [6.0, -0.5, 7.0],
        [0.0, 0.0, -1.0],
        0.0,
        100.0,
        center,
        au,
        av,
    );
    let got = check(&ctx, &gpu, &[front, back, outside]);
    assert!(got[0].hit && got[0].front_face, "front lane hits front");
    assert!(got[1].hit && !got[1].front_face, "back lane hits back");
    assert!(!got[2].hit, "outside lane misses");
}

#[test]
fn random_sweep_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRectangle::new(&ctx);

    let mut state: u64 = 0x0f0e_0d0c_0b0a_0908;
    let mut queries: Vec<RayRectangleQuery> = Vec::new();
    let mut saw_hit = false;
    let mut saw_miss = false;
    let mut saw_front = false;
    let mut saw_back = false;

    for _ in 0..512 {
        // A well-separated, non-collinear parallelogram: reject near-collinear
        // or near-degenerate edge pairs so nn stays comfortably positive and
        // the containment solve is well conditioned.
        let center = [
            uniform(lcg(&mut state), -2.0, 2.0),
            uniform(lcg(&mut state), -2.0, 2.0),
            uniform(lcg(&mut state), -2.0, 2.0),
        ];
        let au = [
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
        ];
        let av = [
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
            uniform(lcg(&mut state), -1.5, 1.5),
        ];
        let n = cross(au, av);
        let nn = dot(n, n);
        let uu = dot(au, au);
        let vv = dot(av, av);
        // sin² of the edge angle; reject near-collinear (and tiny) edges.
        if uu < 0.25 || vv < 0.25 || nn <= 0.25 * uu * vv {
            continue;
        }
        let inv = 1.0 / nn.sqrt();
        let unit_n = scale(n, inv);

        // Choose in-plane coordinates: a clear interior hit (|a|,|b| <= 0.8) or
        // a clear exterior miss (|a| or |b| in [1.3, 2.5]), both away from the
        // |·| == 1 edge where a ULP flip could occur.
        let want_hit = lcg(&mut state) < 0.5;
        let (a, b) = if want_hit {
            (
                uniform(lcg(&mut state), -0.8, 0.8),
                uniform(lcg(&mut state), -0.8, 0.8),
            )
        } else {
            // Push one coordinate clearly outside while keeping the other sane.
            let sign = if lcg(&mut state) < 0.5 { -1.0 } else { 1.0 };
            let outside = sign * uniform(lcg(&mut state), 1.3, 2.5);
            if lcg(&mut state) < 0.5 {
                (outside, uniform(lcg(&mut state), -0.8, 0.8))
            } else {
                (uniform(lcg(&mut state), -0.8, 0.8), outside)
            }
        };
        let target = add(center, add(scale(au, a), scale(av, b)));

        // Place the origin off the plane with a strong normal offset (so the
        // ray is far from parallel) plus a small in-plane jitter for obliquity.
        // The sign of the normal offset selects a front or back arrival. With
        // dir = target - origin the plane crossing lands exactly at t = 1.
        let dist = uniform(lcg(&mut state), 1.0, 3.0);
        let front_side = lcg(&mut state) < 0.5;
        let s = if front_side { 1.0 } else { -1.0 };
        let jit_u = uniform(lcg(&mut state), -0.25, 0.25);
        let jit_v = uniform(lcg(&mut state), -0.25, 0.25);
        let offset = add(
            scale(unit_n, s * dist),
            add(scale(au, jit_u), scale(av, jit_v)),
        );
        let origin = add(target, offset);
        let dir = sub(target, origin);

        queries.push(RayRectangleQuery::new(
            origin, dir, 0.0, 1.0e4, center, au, av,
        ));
    }

    assert!(
        !queries.is_empty(),
        "sweep should retain conditioned queries"
    );
    let got = check(&ctx, &gpu, &queries);
    for (g, q) in got.iter().zip(queries.iter()) {
        let o = oracle(q);
        saw_hit |= o.hit;
        saw_miss |= !o.hit;
        if o.hit {
            saw_front |= o.front_face;
            saw_back |= !o.front_face;
        }
        let _ = g;
    }

    assert!(
        saw_hit && saw_miss,
        "sweep should produce both hits and misses"
    );
    assert!(
        saw_front && saw_back,
        "sweep should produce both front and back hits"
    );
}
