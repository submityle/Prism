//! Real-device parity for the temporal-upscale color-transform twin:
//! [`GpuTaauColorTransform`](prism_volumetric_gpu::taau_color_transform::GpuTaauColorTransform)
//! must reproduce the numeric core of the `CPU` golden
//! [`color`](prism_render_architecture::temporal_upscale::color) — the Rec. 709
//! luminance, the reversible `YCoCg` transform, the Karis tone-map pair and the
//! firefly weight — across black, gray, saturated, `HDR` and randomized pixels.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden's primitives
//! ([`luminance`](prism_render_architecture::temporal_upscale::color::luminance),
//! [`rgb_to_ycocg`](prism_render_architecture::temporal_upscale::color::rgb_to_ycocg),
//! [`ycocg_to_rgb`](prism_render_architecture::temporal_upscale::color::ycocg_to_rgb),
//! [`tonemap`](prism_render_architecture::temporal_upscale::color::tonemap),
//! [`untonemap`](prism_render_architecture::temporal_upscale::color::untonemap)
//! and
//! [`tonemap_weight`](prism_render_architecture::temporal_upscale::color::tonemap_weight))
//! are all public, so each `GPU` field is pinned directly against a host call to
//! the matching golden function. The two round-trip fields additionally compose
//! a transform with its inverse, so a passing suite proves both the forward and
//! the inverse port are faithful.
//!
//! # Parity criterion
//!
//! Every output is a continuous `f32` threaded through a shared `+ - * /` /
//! `max` sequence, so a `GPU` reciprocal may land a few units in the last place
//! from the scalar reference; all fields are asserted within `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3`. The firefly weight's non-finite / non-positive
//! fallback is exact (`1.0`), and is checked with the same bound.
//!
//! # Conditioning
//!
//! The weighting luma is carried independently of the color so the zero, large
//! and degenerate (`NaN` / negative) branches of
//! [`tonemap_weight`](prism_render_architecture::temporal_upscale::color::tonemap_weight)
//! are probed directly; a degenerate pixel keeps a benign finite color so only
//! its weight exercises the fallback. The randomized sweep keeps the luma
//! strictly positive, away from the zero tie, so host and device stay on the
//! same side of the fallback branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::color`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::color::{
    luminance, rgb_to_ycocg, tonemap, tonemap_weight, untonemap, ycocg_to_rgb,
};
use prism_volumetric_gpu::taau_color_transform::{
    GpuTaauColorTransform, TaauColorTransformQuery, TaauColorTransformResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A `GPU` reciprocal may land a
/// few units in the last place from the scalar reference; `1e-4` admits that
/// legal slack while still failing a wrong port.
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

/// Returns whether two colors agree channel-wise within [`close`].
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Reconstructs the expected bundle in-host by calling each golden primitive
/// directly: the faithful oracle the `GPU` is pinned against.
fn oracle(q: &TaauColorTransformQuery) -> TaauColorTransformResult {
    let rgb = q.rgb;
    TaauColorTransformResult {
        luminance: luminance(rgb),
        ycocg: rgb_to_ycocg(rgb),
        tonemap: tonemap(rgb),
        tonemap_weight: tonemap_weight(q.weight_luma),
        rgb_round_trip: ycocg_to_rgb(rgb_to_ycocg(rgb)),
        hdr_round_trip: untonemap(tonemap(rgb)),
    }
}

/// Pins one `GPU` pixel result against the in-host oracle: every continuous
/// field within tolerance.
fn check_pixel(idx: usize, got: &TaauColorTransformResult, want: &TaauColorTransformResult) {
    assert!(
        close(got.luminance, want.luminance),
        "pixel {idx} luminance: gpu {} vs cpu {}",
        got.luminance,
        want.luminance
    );
    assert!(
        close3(got.ycocg, want.ycocg),
        "pixel {idx} ycocg: gpu {:?} vs cpu {:?}",
        got.ycocg,
        want.ycocg
    );
    assert!(
        close3(got.tonemap, want.tonemap),
        "pixel {idx} tonemap: gpu {:?} vs cpu {:?}",
        got.tonemap,
        want.tonemap
    );
    assert!(
        close(got.tonemap_weight, want.tonemap_weight),
        "pixel {idx} tonemap_weight: gpu {} vs cpu {}",
        got.tonemap_weight,
        want.tonemap_weight
    );
    assert!(
        close3(got.rgb_round_trip, want.rgb_round_trip),
        "pixel {idx} rgb_round_trip: gpu {:?} vs cpu {:?}",
        got.rgb_round_trip,
        want.rgb_round_trip
    );
    assert!(
        close3(got.hdr_round_trip, want.hdr_round_trip),
        "pixel {idx} hdr_round_trip: gpu {:?} vs cpu {:?}",
        got.hdr_round_trip,
        want.hdr_round_trip
    );
}

/// Dispatches every pixel and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTaauColorTransform, queries: &[TaauColorTransformQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_pixel(idx, result, &want);
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

/// Draws a channel in `[0.0, 8.0)` at milli resolution from `state`.
fn channel(state: &mut u64) -> f32 {
    (lcg(state) % 8_000) as f32 / 1000.0
}

/// Draws a strictly positive weighting luma in `[0.01, 20.0]` at milli
/// resolution from `state`, kept clear of the zero fallback tie.
fn weight_luma(state: &mut u64) -> f32 {
    0.01 + (lcg(state) % 19_990) as f32 / 1000.0
}

/// The deterministic fixed-color fixtures spanning black, grays, saturated hues
/// and `HDR` magnitudes, each paired with a positive weighting luma.
fn fixed_fixtures() -> Vec<TaauColorTransformQuery> {
    vec![
        TaauColorTransformQuery::new([0.0, 0.0, 0.0], 0.0),
        TaauColorTransformQuery::new([0.25, 0.25, 0.25], 0.25),
        TaauColorTransformQuery::new([0.5, 0.5, 0.5], 0.5),
        TaauColorTransformQuery::new([0.75, 0.75, 0.75], 0.75),
        TaauColorTransformQuery::new([0.8, 0.1, 0.3], 0.4),
        TaauColorTransformQuery::new([0.2, 0.9, 0.4], 0.6),
        TaauColorTransformQuery::new([0.05, 0.6, 0.95], 0.33),
        TaauColorTransformQuery::new([10.0, 2.0, 0.5], 3.0),
        TaauColorTransformQuery::new([100.0, 50.0, 25.0], 55.0),
        TaauColorTransformQuery::new([1000.0, 1.0, 0.0], 400.0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_color_transform parity: no wgpu adapter");
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn black_pixel_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[TaauColorTransformQuery::new([0.0, 0.0, 0.0], 0.0)],
    );
}

#[test]
fn gray_ramp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            TaauColorTransformQuery::new([0.25, 0.25, 0.25], 0.25),
            TaauColorTransformQuery::new([0.5, 0.5, 0.5], 0.5),
            TaauColorTransformQuery::new([0.75, 0.75, 0.75], 0.75),
        ],
    );
}

#[test]
fn saturated_colors_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            TaauColorTransformQuery::new([0.8, 0.1, 0.3], 0.4),
            TaauColorTransformQuery::new([0.2, 0.9, 0.4], 0.6),
            TaauColorTransformQuery::new([0.05, 0.6, 0.95], 0.33),
        ],
    );
}

#[test]
fn hdr_values_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            TaauColorTransformQuery::new([10.0, 2.0, 0.5], 3.0),
            TaauColorTransformQuery::new([100.0, 50.0, 25.0], 55.0),
            TaauColorTransformQuery::new([1000.0, 1.0, 0.0], 400.0),
        ],
    );
}

#[test]
fn weight_edge_cases_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    // The weighting luma is independent of the color: a benign gray carries the
    // zero, large, `NaN` and negative luma branches so only the weight exercises
    // the fallback. The golden maps `NaN` and non-positive luma to full weight.
    let rgb = [0.4, 0.5, 0.6];
    let queries = [
        TaauColorTransformQuery::new(rgb, 0.0),
        TaauColorTransformQuery::new(rgb, 1000.0),
        TaauColorTransformQuery::new(rgb, f32::NAN),
        TaauColorTransformQuery::new(rgb, -2.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    // luma = 0 -> full weight 1.
    assert!(
        close(got[0].tonemap_weight, 1.0),
        "zero luma must give full weight, got {}",
        got[0].tonemap_weight
    );
    // Large luma -> 1 / (1 + 1000).
    assert!(
        close(got[1].tonemap_weight, 1.0 / 1001.0),
        "large luma weight: gpu {} vs cpu {}",
        got[1].tonemap_weight,
        1.0 / 1001.0
    );
    // NaN luma -> full weight 1 (degenerate fallback).
    assert!(
        close(got[2].tonemap_weight, 1.0),
        "NaN luma must give full weight, got {}",
        got[2].tonemap_weight
    );
    // Negative luma -> full weight 1 (non-positive fallback).
    assert!(
        close(got[3].tonemap_weight, 1.0),
        "negative luma must give full weight, got {}",
        got[3].tonemap_weight
    );
    // Each pixel's color fields still match the oracle regardless of the luma.
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.luminance, want.luminance),
            "pixel {idx} luminance mismatch"
        );
        assert!(
            close3(result.ycocg, want.ycocg),
            "pixel {idx} ycocg mismatch"
        );
        assert!(
            close3(result.tonemap, want.tonemap),
            "pixel {idx} tonemap mismatch"
        );
    }
}

#[test]
fn round_trips_recover_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    let queries = fixed_fixtures();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close3(result.rgb_round_trip, q.rgb),
            "pixel {idx} YCoCg round trip must recover {:?}, got {:?}",
            q.rgb,
            result.rgb_round_trip
        );
        assert!(
            close3(result.hdr_round_trip, q.rgb),
            "pixel {idx} tone-map round trip must recover {:?}, got {:?}",
            q.rgb,
            result.hdr_round_trip
        );
    }
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    // Every fixed fixture dispatched together so per-thread indexing and the
    // contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixed_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauColorTransform::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixed_fixtures();
    // Several workgroups' worth of random pixels pin every reported field across
    // a wide span of colors and weighting lumas.
    for _ in 0..256 {
        let rgb = [
            channel(&mut state),
            channel(&mut state),
            channel(&mut state),
        ];
        queries.push(TaauColorTransformQuery::new(rgb, weight_luma(&mut state)));
    }
    check(&ctx, &gpu, &queries);
}
