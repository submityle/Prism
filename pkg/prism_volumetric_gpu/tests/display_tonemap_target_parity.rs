//! Real-device parity for the `NaN`-safe tone-map clamp twin:
//! [`GpuDisplayTonemapTarget`](prism_volumetric_gpu::display_tonemap_target::GpuDisplayTonemapTarget)
//! must reproduce the `CPU` golden
//! [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
//! across every [`DisplayOutput`](prism_render_architecture::display::DisplayOutput)
//! variant and a range of in-range, out-of-range, negative, `NaN` and
//! randomized inputs, compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
//! is public, so each expected result is produced by calling it directly on the
//! query's raw pair and output. A `GPU == golden` pass is therefore direct
//! evidence the ported kernel performs the same two-stage `NaN`-guarded clamp.
//!
//! # Parity criterion
//!
//! The kernel performs no floating-point arithmetic — only comparisons, a
//! `min` and value copies — so the two sides agree exactly on every fixture.
//! The house f32 rule forbids a bare `==`, so each luminance is compared with
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough to catch a wrong port
//! (a dropped `NaN` guard, a swapped bound, a wrong ceiling) yet honoring the
//! no-bare-equality rule.
//!
//! # Conditioning
//!
//! `NaN` collapses to the lower bound `MIN_NITS = 1.0` on both sides exactly,
//! so it needs no margin. Finite fixtures are kept clear of the exact clamp
//! knots (`MIN_NITS` and each output ceiling) so the discrete min/compare
//! selection cannot straddle a boundary differently between host and device.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::display::tonemap`；无第三方引擎源码或衍生代码。

use prism_render_architecture::display::DisplayOutput;
use prism_render_architecture::display::ToneMapTarget;
use prism_volumetric_gpu::display_tonemap_target::{
    DisplayTonemapTargetQuery, DisplayTonemapTargetResult, GpuDisplayTonemapTarget,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a luminance. The kernel does no arithmetic, so the
/// sides agree exactly; `1e-4` honors the house no-bare-equality rule with room
/// to spare.
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

/// Every display-output variant, so each fixture can be swept across all three.
const OUTPUTS: [DisplayOutput; 3] = [
    DisplayOutput::SdrSrgb,
    DisplayOutput::ScRgb,
    DisplayOutput::Hdr10Pq,
];

/// Produces the expected result by calling the public golden directly: the
/// faithful oracle the `GPU` is pinned against.
fn oracle(q: &DisplayTonemapTargetQuery) -> DisplayTonemapTargetResult {
    let clamped = ToneMapTarget::new(q.paper_white_nits, q.peak_nits).clamped_to(q.output);
    DisplayTonemapTargetResult {
        paper_white_nits: clamped.paper_white_nits,
        peak_nits: clamped.peak_nits,
    }
}

/// Pins one `GPU` result against the golden oracle: both luminances within
/// tolerance.
fn check_one(idx: usize, got: &DisplayTonemapTargetResult, want: &DisplayTonemapTargetResult) {
    assert!(
        close(got.paper_white_nits, want.paper_white_nits),
        "query {idx} paper_white_nits: gpu {} vs cpu {}",
        got.paper_white_nits,
        want.paper_white_nits
    );
    assert!(
        close(got.peak_nits, want.peak_nits),
        "query {idx} peak_nits: gpu {} vs cpu {}",
        got.peak_nits,
        want.peak_nits
    );
}

/// Dispatches every query and pins each result against the golden oracle.
fn check(ctx: &GpuContext, gpu: &GpuDisplayTonemapTarget, queries: &[DisplayTonemapTargetQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
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

/// Draws a luminance in `[-200.0, 12000.0]` at milli resolution from `state`.
/// The span straddles every output ceiling and the `MIN_NITS` floor so the
/// clamp is exercised in all directions.
fn luminance(state: &mut u64) -> f32 {
    -200.0 + (lcg(state) % 12_200_000) as f32 / 1000.0
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping display_tonemap_target parity: no wgpu adapter");
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and return an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn in_range_values_pass_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // Comfortably inside the scRGB ceiling of 1000 nits and above MIN_NITS.
    check(
        &ctx,
        &gpu,
        &[DisplayTonemapTargetQuery::new(
            200.0,
            800.0,
            DisplayOutput::ScRgb,
        )],
    );
}

#[test]
fn peak_clamped_to_each_ceiling() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // Request a peak well above every ceiling so each variant clamps to its own
    // metadata value (80 / 1000 / 10000 nits).
    let queries: Vec<DisplayTonemapTargetQuery> = OUTPUTS
        .iter()
        .map(|&output| DisplayTonemapTargetQuery::new(50.0, 50_000.0, output))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn paper_white_never_exceeds_peak() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // Paper-white above the clamped SDR peak must collapse down to the peak so
    // the invariant paper_white <= peak holds.
    check(
        &ctx,
        &gpu,
        &[DisplayTonemapTargetQuery::new(
            500.0,
            60.0,
            DisplayOutput::SdrSrgb,
        )],
    );
}

#[test]
fn nan_collapses_to_minimum() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // Both inputs NaN must collapse to the lower bound MIN_NITS = 1.0 on every
    // output; a single NaN input must collapse only that field.
    let mut queries = Vec::new();
    for &output in &OUTPUTS {
        queries.push(DisplayTonemapTargetQuery::new(f32::NAN, f32::NAN, output));
        queries.push(DisplayTonemapTargetQuery::new(f32::NAN, 500.0, output));
        queries.push(DisplayTonemapTargetQuery::new(200.0, f32::NAN, output));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn negative_values_clamp_to_minimum() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // Negative luminances clamp up to MIN_NITS on every output.
    let queries: Vec<DisplayTonemapTargetQuery> = OUTPUTS
        .iter()
        .map(|&output| DisplayTonemapTargetQuery::new(-10.0, -5.0, output))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn hdr_allows_high_peak() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // The HDR10 PQ path admits a 4000-nit peak well under its 10000-nit ceiling.
    check(
        &ctx,
        &gpu,
        &[DisplayTonemapTargetQuery::new(
            203.0,
            4000.0,
            DisplayOutput::Hdr10Pq,
        )],
    );
}

#[test]
fn all_variants_mixed_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    // A deterministic mix of pass-through, over-peak, over-ceiling, negative
    // and NaN cases across every variant, dispatched together so the per-thread
    // indexing and contiguous output slots are exercised.
    let mut queries = Vec::new();
    for &output in &OUTPUTS {
        queries.push(DisplayTonemapTargetQuery::new(120.0, 250.0, output));
        queries.push(DisplayTonemapTargetQuery::new(300.0, 70.0, output));
        queries.push(DisplayTonemapTargetQuery::new(5_000.0, 50_000.0, output));
        queries.push(DisplayTonemapTargetQuery::new(-3.0, 0.5, output));
        queries.push(DisplayTonemapTargetQuery::new(f32::NAN, 42.0, output));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDisplayTonemapTarget::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random pairs across all variants, with an
    // occasional NaN injection, pins every clamp direction over a wide span.
    for _ in 0..300 {
        let mut paper = luminance(&mut state);
        let mut peak = luminance(&mut state);
        // Inject NaN on roughly one in sixteen inputs; NaN maps to MIN_NITS
        // exactly on both sides.
        if lcg(&mut state).is_multiple_of(16) {
            paper = f32::NAN;
        }
        if lcg(&mut state).is_multiple_of(16) {
            peak = f32::NAN;
        }
        let output = OUTPUTS[(lcg(&mut state) % 3) as usize];
        queries.push(DisplayTonemapTargetQuery::new(paper, peak, output));
    }
    check(&ctx, &gpu, &queries);
}
