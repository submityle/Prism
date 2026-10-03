//! Real-device parity for the normalized quaternion interpolation twin:
//! [`GpuQuaternionNlerp`](prism_volumetric_gpu::quaternion_nlerp::GpuQuaternionNlerp)
//! must reproduce the `nlerp` of the host-side independent reimplementation
//! [`nlerp_components`](prism_volumetric_gpu::quaternion_nlerp::nlerp_components)
//! across the endpoints, the sign-flip (shorter-arc) case, aligned and
//! near-opposite pairs, and a randomized sweep over unit quaternions.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected blend comes from the module's own host-side independent
//! reimplementation; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! interpolates identically.
//!
//! # Parity criterion
//!
//! Each blended component is a *continuous* quantity, so every assertion
//! compares with an absolute-or-relative tolerance (`abs <= 1e-5 ||
//! rel <= 1e-4`, relative floor `1e-6`). The randomized sweep rejects
//! near-antipodal pairs so the normalized direction stays well conditioned.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate::Quat::nlerp`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quaternion_nlerp::{
    nlerp_components, GpuQuaternionNlerp, QuaternionNlerpQuery, QuaternionNlerpResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous component comparison.
const EPS: f32 = 1e-5;
/// Relative tolerance for the continuous component comparison.
const REL: f32 = 1e-4;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Computes the expected blend from the host-side independent reimplementation.
fn oracle(q: &QuaternionNlerpQuery) -> [f32; 4] {
    nlerp_components(q.ax, q.ay, q.az, q.aw, q.bx, q.by, q.bz, q.bw, q.t)
}

/// Dispatches every query and pins each `GPU` component against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuQuaternionNlerp, queries: &[QuaternionNlerpQuery]) {
    let got: Vec<QuaternionNlerpResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.x, want[0])
                && close(result.y, want[1])
                && close(result.z, want[2])
                && close(result.w, want[3]),
            "query {idx} nlerp: gpu ({}, {}, {}, {}) vs cpu ({}, {}, {}, {})",
            result.x,
            result.y,
            result.z,
            result.w,
            want[0],
            want[1],
            want[2],
            want[3],
        );
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

/// Normalizes four components to a unit quaternion (one `sqrt`; `sqrt` is a
/// core arithmetic primitive, not a transcendental). A near-zero vector falls
/// back to the identity.
fn unit(x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
    let len = (x * x + y * y + z * z + w * w).sqrt();
    if len < 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len;
    [x * inv, y * inv, z * inv, w * inv]
}

/// The deterministic fixtures: both endpoints, the sign-flip (shorter-arc)
/// case, aligned endpoints, a near-opposite pair, and mixed axes. Every pair is
/// kept clear of the exactly-antipodal collapse.
fn edge_fixtures() -> Vec<QuaternionNlerpQuery> {
    let a = unit(0.0, 0.0, 0.0, 1.0);
    let b = unit(0.0, 0.0, 1.0, 1.0);
    let c = unit(0.3, 0.4, 0.1, 0.86);
    vec![
        // t = 0 returns the (normalized) first endpoint.
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 0.0),
        // t = 1 returns the (normalized) second endpoint.
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 1.0),
        // Midpoint between distinct orientations.
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 0.5),
        // Quarter and three-quarter blends.
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 0.25),
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 0.75),
        // Sign-flip case: `b` is the negation of `a`'s neighbor, so a . b < 0
        // and the twin must flip the sign to take the shorter arc.
        QuaternionNlerpQuery::new(c[0], c[1], c[2], c[3], -c[0], -c[1], -c[2], 0.2, 0.5),
        // Aligned endpoints: the blend is `a` itself.
        QuaternionNlerpQuery::new(c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3], 0.5),
        // Negative t extrapolates past `a` (still normalized).
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], -0.3),
        // t beyond 1 extrapolates past `b`.
        QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], c[0], c[1], c[2], c[3], 1.3),
        // Mixed-axis pair at the midpoint.
        {
            let p = unit(0.2, -0.5, 0.3, 0.78);
            let r = unit(-0.1, 0.6, -0.2, 0.76);
            QuaternionNlerpQuery::new(p[0], p[1], p[2], p[3], r[0], r[1], r[2], r[3], 0.5)
        },
        // Near-opposite but not antipodal, well clear of the collapse.
        {
            let p = unit(0.0, 0.0, 0.0, 1.0);
            let r = unit(0.1, 0.05, 0.02, -0.99);
            QuaternionNlerpQuery::new(p[0], p[1], p[2], p[3], r[0], r[1], r[2], r[3], 0.5)
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping quaternion_nlerp parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn endpoint_t_zero_returns_first() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let a = unit(0.0, 0.0, 0.0, 1.0);
    let b = unit(0.0, 0.0, 1.0, 1.0);
    let q = QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 0.0);
    let want = oracle(&q);
    assert!(close(want[3], 1.0), "t = 0 recovers the first endpoint");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn endpoint_t_one_returns_second() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let a = unit(0.0, 0.0, 0.0, 1.0);
    let b = unit(0.0, 0.0, 1.0, 1.0);
    let q = QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 1.0);
    let want = oracle(&q);
    assert!(close(want[2], b[2]), "t = 1 recovers the second endpoint");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn sign_flip_takes_shorter_arc() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let c = unit(0.3, 0.4, 0.1, 0.86);
    // a . b < 0, so the twin must flip b's sign before blending.
    let q = QuaternionNlerpQuery::new(c[0], c[1], c[2], c[3], -c[0], -c[1], -c[2], 0.2, 0.5);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn aligned_endpoints_are_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let c = unit(0.3, 0.4, 0.1, 0.86);
    let q = QuaternionNlerpQuery::new(c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3], 0.5);
    let want = oracle(&q);
    assert!(
        close(want[0], c[0]),
        "aligned endpoints blend to themselves"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn result_is_unit_length() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let a = unit(0.2, -0.5, 0.3, 0.78);
    let b = unit(-0.1, 0.6, -0.2, 0.76);
    let q = QuaternionNlerpQuery::new(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], 0.5);
    let got: Vec<QuaternionNlerpResult> = gpu.evaluate(&ctx, &[q]);
    let r = got[0];
    let len = (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w).sqrt();
    assert!(close(len, 1.0), "nlerp output must be a unit quaternion");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionNlerp::new(&ctx);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let a = unit(
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
        );
        let b = unit(
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
        );
        let t = uniform(&mut state, -0.25, 1.25);
        // Reject near-antipodal pairs: after the shorter-arc sign flip the
        // aligned dot must stay well clear of -1 so the chord does not collapse
        // toward zero length and the normalized direction stays conditioned.
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        if dot.abs() > 0.98 {
            // Skip both the near-identical and near-opposite extremes to keep
            // the blended length comfortably above the identity clamp.
            if dot < 0.0 {
                continue;
            }
        }
        // Guard the collapse directly: the post-flip blend length must stay
        // above a safe margin for every sampled t.
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let bx = a[0] + t * (sign * b[0] - a[0]);
        let by = a[1] + t * (sign * b[1] - a[1]);
        let bz = a[2] + t * (sign * b[2] - a[2]);
        let bw = a[3] + t * (sign * b[3] - a[3]);
        let len = (bx * bx + by * by + bz * bz + bw * bw).sqrt();
        if len < 0.1 {
            continue;
        }
        queries.push(QuaternionNlerpQuery::new(
            a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], t,
        ));
    }
    check(&ctx, &gpu, &queries);
}
