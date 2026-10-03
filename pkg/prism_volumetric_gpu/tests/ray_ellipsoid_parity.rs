//! Real-device parity for the ray/ellipsoid intersection twin:
//! [`GpuRayEllipsoid`](prism_volumetric_gpu::ray_ellipsoid::GpuRayEllipsoid)
//! must reproduce the closed-form hit of the host-side independent
//! reimplementation
//! [`intersect_ellipsoid`](prism_volumetric_gpu::ray_ellipsoid::intersect_ellipsoid)
//! across an exterior front-face hit, an interior-origin back-face exit, a
//! grazing exterior miss, an anisotropic-radii hit, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected hit comes from the module's own host-side independent
//! reimplementation; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! solves the same quadratic and classifies the same degenerate cases.
//!
//! # Parity criterion
//!
//! The ray parameter `t` and the normal are *continuous* quantities threaded
//! through `sqrt`, divisions and products, so every continuous assertion
//! compares with an absolute-or-relative tolerance (`abs <= 1e-4 ||
//! rel <= 1e-3`, relative floor `1e-6`). The discrete `hit` and `front_face`
//! flags are pinned exactly. The genuine degeneracies — a near-zero
//! discriminant graze, a root pressed against `t_lo` or `t_hi`, a ray origin on
//! the surface, a near-zero radius, and a near-grazing front/back flip — are
//! kept off their thresholds by the fixtures and the rejection-sampled sweep so
//! the two sides agree on the discrete classification.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::ellipsoid`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_ellipsoid::{
    intersect_ellipsoid, GpuRayEllipsoid, RayEllipsoidQuery, RayEllipsoidResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous `t`/normal comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous `t`/normal comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Smallest accepted axis radius, keeping the degenerate zero-radius branch off
/// its threshold in the randomized sweep.
const R_MARGIN: f32 = 0.2;
/// Smallest accepted scaled-direction squared length, keeping the zero-length
/// (`a <= 0`) rejection branch off its threshold.
const A_MARGIN: f32 = 0.01;
/// Minimum discriminant magnitude, so hit versus miss stays decisive and a
/// multiply-add fusion cannot flip the sign between host and device.
const DISC_MARGIN: f32 = 0.03;
/// Minimum root distance from either interval boundary, so the in-range
/// classification cannot flip between host and device.
const T_MARGIN: f32 = 0.05;
/// Minimum distance of the origin from the surface (`|dot(so, so) - 1|`), so an
/// origin-on-surface degeneracy never sits on its threshold.
const SURF_MARGIN: f32 = 0.05;
/// Minimum `|dot(direction, outward)|` at the hit, keeping the front/back-face
/// flag decisive instead of grazing.
const FACE_MARGIN: f32 = 0.05;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Dot product of two 3-vectors, used only by the conditioning check.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes the expected hit from the host-side reimplementation.
fn oracle(q: &RayEllipsoidQuery) -> Option<(f32, [f32; 3], bool)> {
    intersect_ellipsoid(
        [q.ox, q.oy, q.oz],
        [q.dx, q.dy, q.dz],
        q.t_lo,
        q.t_hi,
        [q.cx, q.cy, q.cz],
        [q.rx, q.ry, q.rz],
    )
}

/// Dispatches every query and pins each `GPU` hit against the oracle: the `hit`
/// and `front_face` flags match exactly, and when hit the `t` and normal match
/// within tolerance.
fn check(ctx: &GpuContext, gpu: &GpuRayEllipsoid, queries: &[RayEllipsoidQuery]) {
    let got: Vec<RayEllipsoidResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        match oracle(q) {
            Some((t, normal, front_face)) => {
                assert!(result.hit, "query {idx}: cpu hit but gpu missed");
                assert_eq!(
                    result.front_face, front_face,
                    "query {idx}: front-face flag mismatch"
                );
                assert!(
                    close(result.t, t),
                    "query {idx}: t gpu {} vs cpu {}",
                    result.t,
                    t
                );
                assert!(
                    close(result.normal[0], normal[0])
                        && close(result.normal[1], normal[1])
                        && close(result.normal[2], normal[2]),
                    "query {idx}: normal gpu {:?} vs cpu {:?}",
                    result.normal,
                    normal
                );
            }
            None => {
                assert!(!result.hit, "query {idx}: cpu missed but gpu hit");
            }
        }
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// Rejects queries that sit on any genuine degeneracy of the solve, so the
/// discrete `hit`/`front_face` classification cannot flip between host and
/// device: a near-zero radius, a near-zero scaled direction, an origin on the
/// surface, a near-grazing discriminant, a root pressed against the interval
/// boundary, and a near-grazing front/back flip. It recomputes the full solve
/// so the margins pin the exact thresholds the kernel branches on.
fn well_conditioned(q: &RayEllipsoidQuery) -> bool {
    let r = [q.rx.abs(), q.ry.abs(), q.rz.abs()];
    if r[0] < R_MARGIN || r[1] < R_MARGIN || r[2] < R_MARGIN {
        return false;
    }
    let so = [
        (q.ox - q.cx) / r[0],
        (q.oy - q.cy) / r[1],
        (q.oz - q.cz) / r[2],
    ];
    let sd = [q.dx / r[0], q.dy / r[1], q.dz / r[2]];
    let a = dot3(sd, sd);
    if a < A_MARGIN {
        return false;
    }
    let half_b = dot3(so, sd);
    let c_term = dot3(so, so) - 1.0;
    if c_term.abs() < SURF_MARGIN {
        return false;
    }
    let disc = half_b * half_b - a * c_term;
    if disc.abs() < DISC_MARGIN {
        return false;
    }
    if disc < 0.0 {
        // A decisive miss: no root to pin against the interval.
        return true;
    }
    let sqrt_disc = disc.sqrt();
    let signed = if half_b < 0.0 { -sqrt_disc } else { sqrt_disc };
    let k = -(half_b + signed);
    let (t_near, t_far) = if k.abs() > 0.0 {
        let r0 = k / a;
        let r1 = c_term / k;
        if r0 <= r1 {
            (r0, r1)
        } else {
            (r1, r0)
        }
    } else {
        let rr = -half_b / a;
        (rr, rr)
    };
    for root in [t_near, t_far] {
        if (root - q.t_lo).abs() < T_MARGIN || (root - q.t_hi).abs() < T_MARGIN {
            return false;
        }
    }
    let hit_t = if t_near >= q.t_lo && t_near <= q.t_hi {
        Some(t_near)
    } else if t_far >= q.t_lo && t_far <= q.t_hi {
        Some(t_far)
    } else {
        None
    };
    if let Some(t) = hit_t {
        let point = [q.ox + t * q.dx, q.oy + t * q.dy, q.oz + t * q.dz];
        let grad = [
            (point[0] - q.cx) / (r[0] * r[0]),
            (point[1] - q.cy) / (r[1] * r[1]),
            (point[2] - q.cz) / (r[2] * r[2]),
        ];
        let inv_len = 1.0 / dot3(grad, grad).sqrt();
        let outward = [grad[0] * inv_len, grad[1] * inv_len, grad[2] * inv_len];
        let facing = dot3([q.dx, q.dy, q.dz], outward);
        if facing.abs() < FACE_MARGIN {
            return false;
        }
    }
    true
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "a skipped device test should report why on hosts without a GPU"
        )]
        {
            eprintln!("skipping ray_ellipsoid parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn exterior_hit_is_front_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    // A ray from outside a unit sphere aimed at its center: the near root lands
    // in range and the ray enters, so the hit is a front face.
    let q = RayEllipsoidQuery::new(
        -5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    );
    assert!(
        well_conditioned(&q),
        "fixture must stay off every threshold"
    );
    let (_, _, front_face) = oracle(&q).expect("exterior ray hits the sphere");
    assert!(front_face, "an entering exterior ray hits a front face");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn interior_origin_exits_back_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    // A ray starting at the center of a unit sphere exits through the surface:
    // the only in-range root is the far one, the outward gradient agrees with
    // the ray, so the hit is a back face with the normal flipped inward.
    let q = RayEllipsoidQuery::new(
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    );
    assert!(
        well_conditioned(&q),
        "fixture must stay off every threshold"
    );
    let (_, normal, front_face) = oracle(&q).expect("interior ray exits the sphere");
    assert!(!front_face, "an exiting interior ray hits a back face");
    assert!(
        normal[0] < 0.0,
        "the back-face normal is flipped against the ray"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn grazing_ray_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    // A ray parallel to the x-axis but offset two units in y never reaches the
    // unit sphere: the discriminant is decisively negative.
    let q = RayEllipsoidQuery::new(
        -5.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    );
    assert!(
        well_conditioned(&q),
        "fixture must stay off every threshold"
    );
    assert!(oracle(&q).is_none(), "an offset parallel ray misses");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn anisotropic_radii_hit_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    // An elongated ellipsoid with unequal radii, hit along its longest axis, so
    // the scaled-frame quadratic has a non-unit `t^2` coefficient.
    let q = RayEllipsoidQuery::new(
        -5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.5,
    );
    assert!(
        well_conditioned(&q),
        "fixture must stay off every threshold"
    );
    let (t, _, front_face) = oracle(&q).expect("the ray hits the elongated ellipsoid");
    assert!(front_face, "an entering exterior ray hits a front face");
    assert!(
        close(t, 3.0),
        "the near hit is at the ellipsoid's leading surface"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayEllipsoid::new(&ctx);
    let mut state: u64 = 0x2468_ACE0_1357_9BDF;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let cx = uniform(&mut state, -1.0, 1.0);
        let cy = uniform(&mut state, -1.0, 1.0);
        let cz = uniform(&mut state, -1.0, 1.0);
        let rx = uniform(&mut state, 0.4, 1.2);
        let ry = uniform(&mut state, 0.4, 1.2);
        let rz = uniform(&mut state, 0.4, 1.2);
        let ox = uniform(&mut state, -4.0, 4.0);
        let oy = uniform(&mut state, -4.0, 4.0);
        let oz = uniform(&mut state, -4.0, 4.0);
        // Aim the direction loosely toward the center with jitter, so the sweep
        // mixes decisive hits and decisive misses.
        let dx = (cx - ox) + uniform(&mut state, -0.8, 0.8);
        let dy = (cy - oy) + uniform(&mut state, -0.8, 0.8);
        let dz = (cz - oz) + uniform(&mut state, -0.8, 0.8);
        let q = RayEllipsoidQuery::new(ox, oy, oz, dx, dy, dz, 0.0, 100.0, cx, cy, cz, rx, ry, rz);
        // Reject anything sitting on a genuine degeneracy so the discrete
        // classification agrees between host and device.
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }
    check(&ctx, &gpu, &queries);
}
