//! Real-device parity for the volumetric cloud coverage-remap twin:
//! [`GpuCloudCoverageRemap`](prism_volumetric_gpu::cloud_coverage_remap::GpuCloudCoverageRemap)
//! must reproduce the two scalar maps of the `CPU` golden — the shared
//! [`remap`](prism_render_architecture::volumetric::math) and the cloud
//! [`coverage_remap`](prism_render_architecture::volumetric::modeling) — across
//! identity, extrapolation, collapsed-span and coverage-edge fixtures, a mixed
//! batch, and a randomized sweep compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids calling the golden crate: the host oracle re-implements
//! the two closed forms independently from the published reference math
//! (`EPS = 1e-6`, `saturate(x) = clamp(x, 0, 1)`,
//! `remap` guarding a collapsed span to a zero interpolant, and
//! `coverage_remap(base, cov) = saturate(remap(saturate(base),
//! 1 - saturate(cov), 1, 0, 1))`). The kernel and the oracle are therefore two
//! independent transcriptions of the same math, and a `GPU == oracle` pass is
//! direct evidence both agree.
//!
//! # Parity criterion
//!
//! Both outputs thread through a subtract, a divide and a `clamp`, so a `GPU`
//! divide or fused multiply-add may land a few units in the last place from the
//! scalar reference; each is asserted within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-4`.
//!
//! # Conditioning
//!
//! The `|span| < EPS` guard is a magnitude comparison, so a fixture whose input
//! span sits near the `EPS` threshold could take different branches on the two
//! sides. Every fixture therefore keeps its input span either exactly collapsed
//! (`in_lo == in_hi`, where both sides deterministically take the zero
//! interpolant) or a full unit clear of `EPS`; likewise the random sweep keeps
//! `|in_hi - in_lo| >= 1` and keeps the saturated coverage either exactly zero
//! or at least `0.1`, so both sides stay on the same side of every branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::volumetric`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloud_coverage_remap::{
    CloudCoverageRemapQuery, CloudCoverageRemapResult, GpuCloudCoverageRemap,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a mapped value. A `GPU` divide may land a few units
/// in the last place from the scalar reference; `1e-5` admits that legal slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Reference span-collapse epsilon, matching the golden `EPS`.
const GOLD_EPS: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Independent host transcription of the reference `saturate`.
fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Independent host transcription of the reference `remap`: a collapsed input
/// span (`|span| < EPS`) takes a zero interpolant so no divide-by-(near)-zero
/// occurs. Never uses an `f32` `==`.
fn remap(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    let t = if span.abs() < GOLD_EPS {
        0.0
    } else {
        (x - in_lo) / span
    };
    out_lo + (out_hi - out_lo) * t
}

/// Independent host transcription of the reference cloud `coverage_remap`.
fn coverage_remap(base_shape: f32, coverage: f32) -> f32 {
    let base = saturate(base_shape);
    let cov = saturate(coverage);
    saturate(remap(base, 1.0 - cov, 1.0, 0.0, 1.0))
}

/// Reconstructs the two golden outputs for one query in-host: the faithful
/// oracle the `GPU` is pinned against.
fn oracle(q: &CloudCoverageRemapQuery) -> CloudCoverageRemapResult {
    CloudCoverageRemapResult {
        remap_value: remap(q.x, q.in_lo, q.in_hi, q.out_lo, q.out_hi),
        coverage_remap_value: coverage_remap(q.base_shape, q.coverage),
    }
}

/// Pins one `GPU` result against the in-host oracle: both outputs within
/// tolerance.
fn check_one(idx: usize, got: &CloudCoverageRemapResult, want: &CloudCoverageRemapResult) {
    assert!(
        close(got.remap_value, want.remap_value),
        "query {idx} remap_value: gpu {} vs cpu {}",
        got.remap_value,
        want.remap_value
    );
    assert!(
        close(got.coverage_remap_value, want.coverage_remap_value),
        "query {idx} coverage_remap_value: gpu {} vs cpu {}",
        got.coverage_remap_value,
        want.coverage_remap_value
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuCloudCoverageRemap, queries: &[CloudCoverageRemapQuery]) {
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let unit = (lcg(state) % 1_000_001) as f32 / 1_000_000.0;
    lo + (hi - lo) * unit
}

/// Builds a well-conditioned random query: the input span is kept at least one
/// unit wide (clear of the `EPS` collapse threshold) and the saturated coverage
/// is kept at least `0.1` wide, so both sides take identical branches.
fn random_query(state: &mut u64) -> CloudCoverageRemapQuery {
    let in_lo = uniform(state, -5.0, 5.0);
    // Magnitude at least 1.0, sign chosen from a bit, so the span never nears EPS
    // and reversed ranges are exercised too.
    let magnitude = uniform(state, 1.0, 6.0);
    let delta = if lcg(state) & 1 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let in_hi = in_lo + delta;
    let out_lo = uniform(state, -5.0, 5.0);
    let out_hi = uniform(state, -5.0, 5.0);
    let x = uniform(state, -6.0, 6.0);
    // base_shape sweeps past both ends of [0, 1] to exercise saturate.
    let base_shape = uniform(state, -0.3, 1.3);
    // coverage kept at least 0.1 so the inner span (= saturated coverage) never
    // nears the EPS collapse threshold.
    let coverage = uniform(state, 0.1, 1.0);
    CloudCoverageRemapQuery::new(x, in_lo, in_hi, out_lo, out_hi, base_shape, coverage)
}

/// The deterministic named fixtures covering identity, extrapolation, a
/// collapsed span, reversed ranges, coverage edges and out-of-range saturation.
fn fixture_queries() -> Vec<CloudCoverageRemapQuery> {
    vec![
        // Identity-ish: x in the middle of a well-separated range.
        CloudCoverageRemapQuery::new(0.5, 0.0, 1.0, 0.0, 1.0, 0.6, 0.4),
        // Below the input range: negative interpolant extrapolation.
        CloudCoverageRemapQuery::new(-2.0, 0.0, 4.0, 10.0, 20.0, 0.3, 0.7),
        // Above the input range: interpolant past 1 extrapolation.
        CloudCoverageRemapQuery::new(9.0, 1.0, 3.0, -1.0, 1.0, 0.8, 0.2),
        // Collapsed span (in_lo == in_hi): both sides take the zero interpolant,
        // so remap_value == out_lo exactly.
        CloudCoverageRemapQuery::new(3.3, 2.0, 2.0, -4.0, 7.0, 0.5, 0.5),
        // Reversed input range (in_lo > in_hi): a negative span still rescales.
        CloudCoverageRemapQuery::new(1.5, 4.0, 1.0, 0.0, 1.0, 0.45, 0.55),
        // Coverage zero clears the sky: coverage_remap collapses to 0.
        CloudCoverageRemapQuery::new(0.5, 0.0, 1.0, 0.0, 1.0, 0.9, 0.0),
        // Coverage below zero saturates to zero: same clear-sky collapse.
        CloudCoverageRemapQuery::new(0.5, 0.0, 1.0, 0.0, 1.0, 0.9, -0.4),
        // Coverage one: remap(base, 0, 1, 0, 1) == saturate(base) passthrough.
        CloudCoverageRemapQuery::new(0.2, -3.0, 3.0, 0.0, 1.0, 0.73, 1.0),
        // Coverage above one saturates to one: same passthrough.
        CloudCoverageRemapQuery::new(0.2, -3.0, 3.0, 0.0, 1.0, 0.73, 1.4),
        // base_shape above one saturates before the inner remap.
        CloudCoverageRemapQuery::new(2.0, 0.0, 5.0, 0.0, 10.0, 1.6, 0.5),
        // base_shape below zero saturates to zero before the inner remap.
        CloudCoverageRemapQuery::new(2.0, 0.0, 5.0, 0.0, 10.0, -0.4, 0.5),
        // Negative output range mid interpolation.
        CloudCoverageRemapQuery::new(2.5, 0.0, 5.0, -8.0, -2.0, 0.35, 0.65),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloud_coverage_remap parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_range_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[CloudCoverageRemapQuery::new(
            0.5, 0.0, 1.0, 0.0, 1.0, 0.6, 0.4,
        )],
    );
}

#[test]
fn extrapolation_below_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[CloudCoverageRemapQuery::new(
            -2.0, 0.0, 4.0, 10.0, 20.0, 0.3, 0.7,
        )],
    );
}

#[test]
fn extrapolation_above_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[CloudCoverageRemapQuery::new(
            9.0, 1.0, 3.0, -1.0, 1.0, 0.8, 0.2,
        )],
    );
}

#[test]
fn collapsed_span_takes_output_low() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    // in_lo == in_hi collapses the span; both sides take the zero interpolant so
    // remap_value is exactly out_lo.
    let q = CloudCoverageRemapQuery::new(3.3, 2.0, 2.0, -4.0, 7.0, 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.remap_value, -4.0),
        "collapsed span oracle must equal out_lo"
    );
    check_one(0, &got[0], &want);
}

#[test]
fn reversed_range_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[CloudCoverageRemapQuery::new(
            1.5, 4.0, 1.0, 0.0, 1.0, 0.45, 0.55,
        )],
    );
}

#[test]
fn coverage_zero_clears_sky() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    // coverage 0 collapses the inner span, so coverage_remap_value is 0.
    let q = CloudCoverageRemapQuery::new(0.5, 0.0, 1.0, 0.0, 1.0, 0.9, 0.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.coverage_remap_value, 0.0),
        "coverage 0 must clear the sky"
    );
    check_one(0, &got[0], &want);
}

#[test]
fn coverage_one_passes_shape_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    // coverage 1 maps remap(base, 0, 1, 0, 1) == saturate(base).
    let q = CloudCoverageRemapQuery::new(0.2, -3.0, 3.0, 0.0, 1.0, 0.73, 1.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.coverage_remap_value, 0.73),
        "coverage 1 must pass the saturated shape through"
    );
    check_one(0, &got[0], &want);
}

#[test]
fn out_of_range_inputs_saturate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            CloudCoverageRemapQuery::new(0.5, 0.0, 1.0, 0.0, 1.0, 0.9, 1.4),
            CloudCoverageRemapQuery::new(2.0, 0.0, 5.0, 0.0, 10.0, 1.6, 0.5),
            CloudCoverageRemapQuery::new(2.0, 0.0, 5.0, 0.0, 10.0, -0.4, 0.5),
        ],
    );
}

#[test]
fn negative_output_range_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[CloudCoverageRemapQuery::new(
            2.5, 0.0, 5.0, -8.0, -2.0, 0.35, 0.65,
        )],
    );
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    // Every named fixture dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCloudCoverageRemap::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of well-conditioned random queries pin every
    // output across a wide span of ranges, shapes and coverages.
    for _ in 0..384 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
