//! Real-device parity for the adaptive-quality decision twin:
//! [`GpuQualityDecision`](prism_volumetric_gpu::quality_decision::GpuQualityDecision)
//! must reproduce the numeric core of the `CPU` golden
//! [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at) —
//! the `clamp_unit` + `lerp` mapping of a quality level `q` through four knob
//! ranges — across the low / high endpoints, out-of-range clamping, the
//! midpoint and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)
//! is public, so each `GPU` field is pinned directly against a host call: a
//! [`QualityBounds`](prism_render_architecture::quality::controller::QualityBounds)
//! is rebuilt from the query's endpoint pairs and mapped at the query's `q`.
//!
//! # Parity criterion
//!
//! `render_scale`, `geometry_error_pixels` and `gi_ray_scale` are continuous
//! `f32` interpolations and are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. `shadow_page_budget` is an integer and is asserted
//! exactly with `==`; the golden rounds the `u32`-endpoint interpolation
//! round-half-away-from-zero, which the kernel reproduces with
//! `floor(value + 0.5)` for the non-negative values here.
//!
//! # Conditioning
//!
//! The randomized sweep rejects any `(endpoints, q)` whose budget interpolation
//! lands within a margin of a half-integer, so a last-place difference in the
//! shared `f32` interpolation can never flip the integer rounding and the
//! `== ` assertion stays faithful. Every fixed fixture lands the budget on an
//! exact integer.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::quality::controller`；无第三方引擎源码或衍生代码。

use prism_render_architecture::quality::controller::QualityBounds;
use prism_render_architecture::quality::QualityDecision;
use prism_volumetric_gpu::quality_decision::{
    GpuQualityDecision, QualityDecisionQuery, QualityDecisionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous knob. A `GPU` reciprocal/`lerp` may
/// land a few units in the last place from the scalar reference; `1e-4` admits
/// that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Rebuilds the golden [`QualityBounds`] from a query's endpoint pairs.
fn bounds_of(q: &QualityDecisionQuery) -> QualityBounds {
    QualityBounds {
        render_scale: q.render_scale,
        geometry_error_pixels: q.geometry_error_pixels,
        shadow_page_budget: q.shadow_page_budget,
        gi_ray_scale: q.gi_ray_scale,
    }
}

/// Computes the expected decision in-host via the golden `decision_at`: the
/// faithful oracle the `GPU` is pinned against.
fn oracle(q: &QualityDecisionQuery) -> QualityDecision {
    bounds_of(q).decision_at(q.q)
}

/// Pins one `GPU` decision against the in-host oracle: the three continuous
/// knobs within tolerance, the integer budget exactly.
fn check_sample(idx: usize, got: &QualityDecisionResult, want: &QualityDecision) {
    assert!(
        close(got.render_scale, want.render_scale),
        "sample {idx} render_scale: gpu {} vs cpu {}",
        got.render_scale,
        want.render_scale
    );
    assert!(
        close(got.geometry_error_pixels, want.geometry_error_pixels),
        "sample {idx} geometry_error_pixels: gpu {} vs cpu {}",
        got.geometry_error_pixels,
        want.geometry_error_pixels
    );
    assert!(
        close(got.gi_ray_scale, want.gi_ray_scale),
        "sample {idx} gi_ray_scale: gpu {} vs cpu {}",
        got.gi_ray_scale,
        want.gi_ray_scale
    );
    assert_eq!(
        got.shadow_page_budget, want.shadow_page_budget,
        "sample {idx} shadow_page_budget: gpu {} vs cpu {}",
        got.shadow_page_budget, want.shadow_page_budget
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuQualityDecision, queries: &[QualityDecisionQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_sample(idx, result, &want);
    }
}

/// The reference default per-knob ranges: render scale `0.5 -> 1.0`, geometry
/// error `4.0 -> 0.25` (inverted), shadow budget `1024 -> 8192`, `GI` ray scale
/// `0.25 -> 1.0`.
const DEFAULT_RENDER_SCALE: (f32, f32) = (0.5, 1.0);
const DEFAULT_GEOMETRY_ERROR: (f32, f32) = (4.0, 0.25);
const DEFAULT_SHADOW_BUDGET: (u32, u32) = (1024, 8192);
const DEFAULT_GI_RAY_SCALE: (f32, f32) = (0.25, 1.0);

/// Builds a query over the reference default ranges at quality level `q`.
fn default_query(q: f32) -> QualityDecisionQuery {
    QualityDecisionQuery::new(
        DEFAULT_RENDER_SCALE,
        DEFAULT_GEOMETRY_ERROR,
        DEFAULT_SHADOW_BUDGET,
        DEFAULT_GI_RAY_SCALE,
        q,
    )
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a quality level in `[0.0, 1.0]` at milli resolution from `state`.
fn quality(state: &mut u64) -> f32 {
    (lcg(state) % 1001) as f32 / 1000.0
}

/// Draws a positive `f32` endpoint in `[0.0, 4.0)` at milli resolution.
fn scalar_endpoint(state: &mut u64) -> f32 {
    (lcg(state) % 4000) as f32 / 1000.0
}

/// Draws a `u32` budget endpoint in `[0, 65_535]` from `state`.
fn budget_endpoint(state: &mut u64) -> u32 {
    lcg(state) % 65_536
}

/// Whether the budget interpolation is clear of a half-integer rounding tie, so
/// the exact integer assertion cannot be flipped by a last-place difference in
/// the shared `f32` interpolation. Uses only `floor` and ordered comparisons.
fn budget_well_conditioned(a: u32, b: u32, q: f32) -> bool {
    let qc = if q.is_nan() || q < 0.0 {
        0.0
    } else if q > 1.0 {
        1.0
    } else {
        q
    };
    let value = a as f32 + (b as f32 - a as f32) * qc;
    let frac = value - value.floor();
    !(0.4..=0.6).contains(&frac)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping quality_decision parity: no wgpu adapter");
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn lowest_quality_hits_low_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // q = 0 settles every knob on its lowest-quality endpoint.
    check(&ctx, &gpu, &[default_query(0.0)]);
}

#[test]
fn highest_quality_hits_high_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // q = 1 settles every knob on its highest-quality endpoint.
    check(&ctx, &gpu, &[default_query(1.0)]);
}

#[test]
fn midpoint_interpolates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // q = 0.5 lands the shadow budget on an exact integer (4608).
    check(&ctx, &gpu, &[default_query(0.5), default_query(0.25)]);
}

#[test]
fn out_of_range_quality_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // A negative q and a NaN q both clamp to 0 (low endpoints); a q above 1
    // clamps to 1 (high endpoints). The oracle applies the same clamp_unit, so
    // the GPU must agree.
    check(
        &ctx,
        &gpu,
        &[
            default_query(-0.5),
            default_query(1.5),
            default_query(f32::NAN),
        ],
    );
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    // Several distinct bounds dispatched together so per-thread indexing and the
    // contiguous output slots are both exercised. Every budget lands exactly.
    let queries = [
        default_query(0.0),
        default_query(0.25),
        default_query(0.5),
        default_query(1.0),
        QualityDecisionQuery::new((0.6, 0.95), (2.0, 0.5), (256, 4096), (0.3, 0.9), 0.5),
        QualityDecisionQuery::new((1.0, 0.5), (0.25, 4.0), (8192, 1024), (1.0, 0.25), 0.25),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQualityDecision::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = vec![default_query(0.0), default_query(1.0)];
    // Several workgroups' worth of random (bounds, q) samples, each with a
    // budget interpolation kept clear of a half-integer tie.
    let mut drawn = 0u32;
    while drawn < 256 {
        let rs = (scalar_endpoint(&mut state), scalar_endpoint(&mut state));
        let ge = (scalar_endpoint(&mut state), scalar_endpoint(&mut state));
        let sb = (budget_endpoint(&mut state), budget_endpoint(&mut state));
        let gi = (scalar_endpoint(&mut state), scalar_endpoint(&mut state));
        let q = quality(&mut state);
        if !budget_well_conditioned(sb.0, sb.1, q) {
            continue;
        }
        queries.push(QualityDecisionQuery::new(rs, ge, sb, gi, q));
        drawn += 1;
    }
    check(&ctx, &gpu, &queries);
}
