//! Real-device parity for the per-sample disocclusion twin:
//! [`GpuMotionDisocclusion`](prism_volumetric_gpu::motion_disocclusion::GpuMotionDisocclusion)
//! must reproduce the numeric core of the `CPU` golden
//! [`classify`](prism_render_architecture::motion::disocclusion::classify) — the
//! graded confidence, the accept flag, and the rejection reason bit mask —
//! across a coherent accept sample, each single-signal failure (surface id,
//! depth, normal), a `min`-combination sample, a threshold-reject-without-reason
//! sample, a surface-check-disabled sample, and a randomized batch compared
//! sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden helpers
//! [`classify`](prism_render_architecture::motion::disocclusion::classify),
//! [`depth_consistency`](prism_render_architecture::motion::disocclusion::depth_consistency),
//! and
//! [`normal_consistency`](prism_render_architecture::motion::disocclusion::normal_consistency)
//! are `pub`, so each `GPU` verdict is pinned directly against `classify` run on
//! the same `(current, history, params)` sample. Parameters are built through
//! [`DisocclusionParams::new`](prism_render_architecture::motion::disocclusion::DisocclusionParams::new)
//! so the `GPU` and the oracle share identically sanitized thresholds.
//!
//! # Parity criterion
//!
//! The `accepted` flag and the `reasons` bit mask are integers / booleans built
//! from integer equality and sign comparisons, so they agree exactly for the
//! chosen fixtures and are asserted with `==`. The `confidence` threads through
//! a subtract and a divide only, so it is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture is deliberately away from a decision boundary. A sample whose
//! depth or normal sub-confidence is near zero is rejected (its reason bit could
//! flip once a `GPU` divide and a `CPU` divide disagree by a unit in the last
//! place), and a sample whose combined confidence is near `accept_threshold` is
//! rejected (its `accepted` flag could flip). The surviving samples keep every
//! sub-confidence and the accept margin far beyond the `f32` slack, so `CPU` and
//! `GPU` stay on the same side of every discrete decision.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::disocclusion`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::disocclusion::{
    classify, DisocclusionParams, RejectionReasons, SurfacePoint,
};
use prism_volumetric_gpu::motion_disocclusion::{
    GpuMotionDisocclusion, MotionDisocclusionParams, MotionDisocclusionQuery, MotionSurfacePoint,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a confidence.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Mirrors the golden `DisocclusionParams` into the twin's parameter struct.
fn gpu_params(p: &DisocclusionParams) -> MotionDisocclusionParams {
    MotionDisocclusionParams::new(
        p.depth_relative_tolerance,
        p.normal_cos_threshold,
        p.require_surface_match,
        p.accept_threshold,
    )
}

/// Mirrors a golden `SurfacePoint` into the twin's surface sample.
fn gpu_point(s: &SurfacePoint) -> MotionSurfacePoint {
    MotionSurfacePoint::new(s.depth, s.normal, s.surface_id)
}

/// Builds a twin query from the golden inputs.
fn query(
    current: SurfacePoint,
    history: SurfacePoint,
    p: DisocclusionParams,
) -> MotionDisocclusionQuery {
    MotionDisocclusionQuery::new(gpu_point(&current), gpu_point(&history), gpu_params(&p))
}

/// Pins one `GPU` verdict against the golden `classify` oracle: confidence to
/// tolerance, accept flag and reason mask exactly.
fn check_one(
    idx: usize,
    current: SurfacePoint,
    history: SurfacePoint,
    p: DisocclusionParams,
    got: &prism_volumetric_gpu::motion_disocclusion::MotionDisocclusionResult,
) {
    let want = classify(current, history, p);
    assert!(
        close(got.confidence, want.confidence),
        "sample {idx} confidence: gpu {} vs cpu {}",
        got.confidence,
        want.confidence
    );
    assert_eq!(
        got.accepted, want.accepted,
        "sample {idx} accepted: gpu {} vs cpu {}",
        got.accepted, want.accepted
    );
    assert_eq!(
        got.reasons,
        want.reasons.bits(),
        "sample {idx} reasons: gpu {} vs cpu {}",
        got.reasons,
        want.reasons.bits()
    );
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

/// Normalizes a 3-vector with the single allowed `sqrt`; returns `None` when the
/// vector is too short to normalize stably.
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < 0.05 {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

const UP: [f32; 3] = [0.0, 1.0, 0.0];
const RIGHT: [f32; 3] = [1.0, 0.0, 0.0];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_disocclusion parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn accepts_coherent_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = SurfacePoint::new(1.0, UP, 42);
    let history = SurfacePoint::new(1.0, UP, 42);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(got[0].accepted, "coherent surface must be accepted");
    assert_eq!(got[0].reasons, 0, "coherent surface sets no reason bits");
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn rejects_surface_mismatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    let current = SurfacePoint::new(1.0, UP, 7);
    let history = SurfacePoint::new(1.0, UP, 8);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(!got[0].accepted, "mismatched ids must reject");
    assert_eq!(
        got[0].reasons & RejectionReasons::SURFACE_MISMATCH.bits(),
        RejectionReasons::SURFACE_MISMATCH.bits(),
        "surface mismatch reason bit must be set"
    );
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn ignores_surface_when_disabled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let p = DisocclusionParams::new(0.05, 0.9, false, 0.5);
    let current = SurfacePoint::new(1.0, UP, 7);
    let history = SurfacePoint::new(1.0, UP, 8);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(got[0].accepted, "surface check disabled must accept");
    assert_eq!(
        got[0].reasons & RejectionReasons::SURFACE_MISMATCH.bits(),
        0,
        "surface mismatch reason bit must stay clear when disabled"
    );
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn rejects_depth_discontinuity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    // A 5x depth gap is far beyond the 5% tolerance, so the depth sub-confidence
    // clamps hard to zero.
    let current = SurfacePoint::new(1.0, UP, 1);
    let history = SurfacePoint::new(5.0, UP, 1);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(!got[0].accepted, "depth discontinuity must reject");
    assert_eq!(
        got[0].reasons & RejectionReasons::DEPTH_DISCONTINUITY.bits(),
        RejectionReasons::DEPTH_DISCONTINUITY.bits(),
        "depth discontinuity reason bit must be set"
    );
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn rejects_normal_discontinuity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let p = DisocclusionParams::default();
    // Orthogonal normals (cos 0) are far below the 0.9 cosine threshold.
    let current = SurfacePoint::new(1.0, UP, 1);
    let history = SurfacePoint::new(1.0, RIGHT, 1);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(!got[0].accepted, "normal discontinuity must reject");
    assert_eq!(
        got[0].reasons & RejectionReasons::NORMAL_DISCONTINUITY.bits(),
        RejectionReasons::NORMAL_DISCONTINUITY.bits(),
        "normal discontinuity reason bit must be set"
    );
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn min_combination_accepts_above_threshold() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    // Wide tolerances so both signals land on a comfortable partial confidence
    // whose minimum stays clear of the accept threshold.
    let p = DisocclusionParams::new(0.5, 0.5, true, 0.5);
    // Depth: relative gap 0.15 against tolerance 0.5 => depth_conf 0.7.
    let current = SurfacePoint::new(1.0, [1.0, 0.0, 0.0], 3);
    // Normal: dot 0.85 against threshold 0.5 => normal_conf 0.7.
    let hist_normal = normalize([0.85, 0.526_782_3, 0.0]).expect("fixture normal is normalizable");
    let history = SurfacePoint::new(0.85, hist_normal, 3);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].accepted,
        "min confidence above threshold must accept"
    );
    assert_eq!(got[0].reasons, 0, "no hard reason fires for a partial pass");
    check_one(0, current, history, p, &got[0]);
}

#[test]
fn threshold_reject_without_reason() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    // A high accept threshold rejects a partial-but-positive confidence without
    // raising any hard reason bit.
    let p = DisocclusionParams::new(0.5, 0.5, true, 0.9);
    let current = SurfacePoint::new(1.0, [1.0, 0.0, 0.0], 5);
    // Depth relative gap 0.15 / tol 0.5 => depth_conf 0.7, below the 0.9 accept.
    let history = SurfacePoint::new(0.85, [1.0, 0.0, 0.0], 5);
    let got = gpu.evaluate(&ctx, &[query(current, history, p)]);
    assert_eq!(got.len(), 1);
    assert!(
        !got[0].accepted,
        "confidence below accept threshold must reject"
    );
    assert_eq!(
        got[0].reasons, 0,
        "a threshold-only rejection raises no reason bits"
    );
    check_one(0, current, history, p, &got[0]);
}

/// Raw (pre-clamp) depth grade, mirroring the golden formula, used only to
/// reject fixtures near the reason boundary.
fn depth_raw(c: f32, h: f32, tol: f32) -> f32 {
    let tol = tol.max(1.0e-6);
    let denom = c.abs().max(h.abs()).max(1.0e-6);
    let relative = (c - h).abs() / denom;
    1.0 - relative / tol
}

/// Raw (pre-clamp) normal grade, mirroring the golden formula, used only to
/// reject fixtures near the reason boundary.
fn normal_raw(dot: f32, thr: f32) -> f32 {
    let threshold = thr.clamp(-1.0, 1.0 - 1.0e-6);
    (dot - threshold) / (1.0 - threshold)
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionDisocclusion::new(&ctx);
    let mut state = 0x51ed_2740_9a3b_c6d1_u64;

    let mut currents: Vec<SurfacePoint> = Vec::new();
    let mut histories: Vec<SurfacePoint> = Vec::new();
    let mut params: Vec<DisocclusionParams> = Vec::new();
    let mut queries: Vec<MotionDisocclusionQuery> = Vec::new();

    // Collect a few hundred well-conditioned samples via rejection sampling.
    let mut guard = 0u32;
    while queries.len() < 256 && guard < 100_000 {
        guard += 1;

        // Sanitized parameters within sane ranges, built through the golden ctor.
        let p = DisocclusionParams::new(
            ranged(&mut state, 0.02, 0.6),
            ranged(&mut state, 0.2, 0.95),
            lcg(&mut state) & 1 == 0,
            ranged(&mut state, 0.2, 0.8),
        );

        // Depth: current away from zero; history a random relative gap.
        let cur_depth = ranged(&mut state, 0.5, 10.0);
        let hist_depth = cur_depth * ranged(&mut state, 0.3, 1.8);

        // Two random unit normals.
        let Some(cur_n) = normalize([
            ranged(&mut state, -1.0, 1.0),
            ranged(&mut state, -1.0, 1.0),
            ranged(&mut state, -1.0, 1.0),
        ]) else {
            continue;
        };
        let Some(his_n) = normalize([
            ranged(&mut state, -1.0, 1.0),
            ranged(&mut state, -1.0, 1.0),
            ranged(&mut state, -1.0, 1.0),
        ]) else {
            continue;
        };

        // Surface ids: sometimes equal, sometimes different, across both halves.
        let base_id = (u64::from(lcg(&mut state)) << 32) | u64::from(lcg(&mut state));
        let same_id = lcg(&mut state) & 1 == 0;
        let hist_id = if same_id {
            base_id
        } else {
            base_id ^ (1u64 << (lcg(&mut state) % 64))
        };

        // Reject fixtures near a reason boundary.
        let d_raw = depth_raw(cur_depth, hist_depth, p.depth_relative_tolerance);
        if d_raw.abs() < 0.04 {
            continue;
        }
        let dot = cur_n[0] * his_n[0] + cur_n[1] * his_n[1] + cur_n[2] * his_n[2];
        let n_raw = normal_raw(dot, p.normal_cos_threshold);
        if n_raw.abs() < 0.04 {
            continue;
        }

        let current = SurfacePoint::new(cur_depth, cur_n, base_id);
        let history = SurfacePoint::new(hist_depth, his_n, hist_id);

        // Reject fixtures whose combined confidence is near the accept threshold
        // (only meaningful when no hard reason zeroes the confidence).
        let want = classify(current, history, p);
        if want.reasons.is_empty() && (want.confidence - p.accept_threshold).abs() < 0.06 {
            continue;
        }

        currents.push(current);
        histories.push(history);
        params.push(p);
        queries.push(query(current, history, p));
    }

    assert!(
        queries.len() >= 256,
        "expected a full random batch, got {}",
        queries.len()
    );

    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match input count"
    );
    for (idx, result) in got.iter().enumerate() {
        check_one(idx, currents[idx], histories[idx], params[idx], result);
    }
}
