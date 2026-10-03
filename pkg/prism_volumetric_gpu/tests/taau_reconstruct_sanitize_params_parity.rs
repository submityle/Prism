//! Real-device parity for the resolve-parameter sanitizer twin:
//! [`GpuTaauReconstructSanitizeParams`](prism_volumetric_gpu::taau_reconstruct_sanitize_params::GpuTaauReconstructSanitizeParams)
//! must reproduce the `CPU` golden
//! [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized)
//! across pass-through values, both negative/sub-`1` branches, the `>=` boundary
//! values, the `NaN`-folds-to-default degenerate, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden
//! [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized)
//! directly on a `ResolveParams` built from the same raw inputs.
//!
//! # Parity criterion
//!
//! The kernel performs only comparisons and copies, so every finite fixture is
//! reproduced with zero difference; the `NaN` fixtures fold to the shared finite
//! default on both sides. Every continuous output is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reconstruct`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::reconstruct::ResolveParams;
use prism_volumetric_gpu::taau_reconstruct_sanitize_params::{
    GpuTaauReconstructSanitizeParams, TaauReconstructSanitizeParamsQuery,
    TaauReconstructSanitizeParamsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound.
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

/// Reconstructs the golden result in-host by sanitizing a `ResolveParams` built
/// from the same raw inputs.
fn oracle(q: &TaauReconstructSanitizeParamsQuery) -> TaauReconstructSanitizeParamsResult {
    let sanitized = ResolveParams {
        variance_gamma: q.variance_gamma,
        max_confidence: q.max_confidence,
    }
    .sanitized();
    TaauReconstructSanitizeParamsResult {
        variance_gamma: sanitized.variance_gamma,
        max_confidence: sanitized.max_confidence,
    }
}

/// Pins one `GPU` result against the in-host oracle on both outputs.
fn check_query(
    idx: usize,
    got: &TaauReconstructSanitizeParamsResult,
    want: &TaauReconstructSanitizeParamsResult,
) {
    assert!(
        close(got.variance_gamma, want.variance_gamma),
        "query {idx} variance_gamma: gpu {} vs cpu {}",
        got.variance_gamma,
        want.variance_gamma
    );
    assert!(
        close(got.max_confidence, want.max_confidence),
        "query {idx} max_confidence: gpu {} vs cpu {}",
        got.max_confidence,
        want.max_confidence
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTaauReconstructSanitizeParams,
    queries: &[TaauReconstructSanitizeParamsQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a `u32` draw to a finite `f32` in `[lo, hi]`.
fn draw_range(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn pass_through_values_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    // Values already inside the safe range pass straight through unchanged.
    let queries = [
        TaauReconstructSanitizeParamsQuery::new(2.0, 8.0),
        TaauReconstructSanitizeParamsQuery::new(0.5, 32.0),
        TaauReconstructSanitizeParamsQuery::new(3.25, 2.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn out_of_range_values_fold_to_default() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    // A negative gamma and a sub-1 confidence each fall back to their default.
    let queries = [
        TaauReconstructSanitizeParamsQuery::new(-1.0, 0.0),
        TaauReconstructSanitizeParamsQuery::new(-0.001, 0.5),
        TaauReconstructSanitizeParamsQuery::new(-100.0, 0.999),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_values_are_kept() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    // The thresholds are inclusive: gamma 0 (clips to the mean) and confidence 1
    // (a single frame) sit on the bound and are kept rather than replaced.
    let queries = [
        TaauReconstructSanitizeParamsQuery::new(0.0, 1.0),
        TaauReconstructSanitizeParamsQuery::new(0.0, 16.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn nan_inputs_fold_to_default() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    // A NaN compares false against every bound, so each axis resolves to its
    // finite default; the folded defaults are what parity compares.
    let queries = [
        TaauReconstructSanitizeParamsQuery::new(f32::NAN, 8.0),
        TaauReconstructSanitizeParamsQuery::new(2.0, f32::NAN),
        TaauReconstructSanitizeParamsQuery::new(f32::NAN, f32::NAN),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauReconstructSanitizeParams::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries: Vec<TaauReconstructSanitizeParamsQuery> = Vec::new();
    // Several workgroups' worth of random finite inputs spanning the negative,
    // sub-1 and in-range regions pin both branches across a wide span.
    for _ in 0..512 {
        let variance_gamma = draw_range(lcg(&mut state), -5.0, 20.0);
        let max_confidence = draw_range(lcg(&mut state), -5.0, 64.0);
        queries.push(TaauReconstructSanitizeParamsQuery::new(
            variance_gamma,
            max_confidence,
        ));
    }
    check(&ctx, &gpu, &queries);
}
