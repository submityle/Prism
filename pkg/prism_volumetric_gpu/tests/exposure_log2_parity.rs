//! Real-device parity for the cheap binary-`log2` and auto-exposure twin:
//! [`GpuExposureLog2`](prism_volumetric_gpu::exposure_log2::GpuExposureLog2)
//! must reproduce two `CPU` reference formulas — the cheap `log2_linear`
//! approximation from the ray-scene footprint stage and the guarded
//! `exposure_from_average` division from the luminance-histogram stage —
//! across the degenerate branches (empty batch, powers of two, non-power
//! ratios, a black-frame `avg_lum` driven to the `EPS` guard), a mixed batch
//! resolved in one dispatch, and a randomized sweep compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave reimplements both reference formulas independently on the host
//! rather than calling the golden crate; [`log2_linear_host`] and
//! [`exposure_host`] below are the faithful ports, and the `GPU` is pinned
//! against them per query.
//!
//! # Parity criterion
//!
//! Both formulas thread through only `+ - * /`, comparisons, and bounded
//! `while` loops with no transcendental and no reorderable reduction, so the
//! `CPU` and `GPU` evaluate the same expression in the same iteration order. A
//! `GPU` may still fuse a multiply-add the scalar reference leaves separate, so
//! both scalars are asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`.
//! No `f32` `==` is used anywhere.
//!
//! # Conditioning
//!
//! The `while` loops multiply by the exactly representable `0.5` and `2.0`, so
//! the loop trip count matches on both paths for the finite, positive ratios
//! the host supplies, and no rejection sampling is needed; the sweep simply
//! keeps `ratio` clearly positive. The exposure `avg_lum < EPS` guard is the
//! only sharp branch and is covered explicitly by the black-frame fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::footprint` 与 `prism_render_architecture::particle::luminance_hist`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::exposure_log2::{ExposureLog2Query, ExposureLog2Result, GpuExposureLog2};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous quantity. A `GPU` arithmetic pipeline
/// may land a few units in the last place from the scalar reference; `1e-5`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Host port of the golden `ray_scene::footprint::log2_linear`: a deliberately
/// cheap `log2` approximation that is exact on powers of two and linear in the
/// mantissa between them.
fn log2_linear_host(ratio: f32) -> f32 {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 0.0;
    }
    let mut mantissa = ratio;
    let mut exponent = 0.0;
    while mantissa >= 2.0 {
        mantissa *= 0.5;
        exponent += 1.0;
    }
    while mantissa < 1.0 {
        mantissa *= 2.0;
        exponent -= 1.0;
    }
    exponent + (mantissa - 1.0)
}

/// Host port of the golden `particle::luminance_hist::exposure_from_average`:
/// the auto-exposure scale `key / avg_lum` with `avg_lum` guarded to at least
/// `EPS = 1e-6` so a black frame cannot divide by zero.
fn exposure_host(avg_lum: f32, key: f32) -> f32 {
    const EPS_G: f32 = 1.0e-6;
    let denom = if avg_lum < EPS_G { EPS_G } else { avg_lum };
    key / denom
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

/// Builds one combined query.
fn query(ratio: f32, avg_lum: f32, key: f32) -> ExposureLog2Query {
    ExposureLog2Query {
        ratio,
        avg_lum,
        key,
    }
}

/// Dispatches `queries` and asserts every scalar matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuExposureLog2, queries: &[ExposureLog2Query]) {
    let got: Vec<ExposureLog2Result> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_log2 = log2_linear_host(q.ratio);
        let golden_exposure = exposure_host(q.avg_lum, q.key);
        assert!(
            close(r.log2_value, golden_log2),
            "log2 mismatch: gpu={} golden={} (ratio={})",
            r.log2_value,
            golden_log2,
            q.ratio
        );
        assert!(
            close(r.exposure, golden_exposure),
            "exposure mismatch: gpu={} golden={} (avg_lum={}, key={})",
            r.exposure,
            golden_exposure,
            q.avg_lum,
            q.key
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping exposure_log2 parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn powers_of_two_are_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let queries = [
        query(1.0, 1.0, 1.0),
        query(2.0, 1.0, 1.0),
        query(4.0, 1.0, 1.0),
        query(8.0, 1.0, 1.0),
        query(16.0, 1.0, 1.0),
        query(0.5, 1.0, 1.0),
        query(0.25, 1.0, 1.0),
        query(0.125, 1.0, 1.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn non_power_ratios_track_the_cheap_approx() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let queries = [
        query(1.5, 1.0, 1.0),
        query(3.0, 1.0, 1.0),
        query(6.0, 1.0, 1.0),
        query(0.75, 1.0, 1.0),
        query(0.1, 1.0, 1.0),
        query(1000.0, 1.0, 1.0),
        query(0.01, 1.0, 1.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn exposure_black_frame_hits_the_eps_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let queries = [
        query(1.0, 0.0, 0.18),
        query(1.0, 0.0000001, 0.18),
        query(2.0, 0.0, 1.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn exposure_normal_and_varying_key() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let queries = [
        query(1.0, 0.18, 0.18),
        query(1.0, 1.0, 0.5),
        query(1.0, 10.0, 2.0),
        query(1.0, 0.5, 0.0),
        query(1.0, 4.0, 1.0),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let queries = [
        query(2.0, 0.0, 0.18),
        query(1.5, 0.18, 0.5),
        query(0.25, 10.0, 2.0),
        query(1000.0, 1.0, 1.0),
        query(0.01, 0.5, 0.3),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuExposureLog2::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_0f01;
    let mut queries = Vec::with_capacity(384);
    for _ in 0..384 {
        let ratio = draw(&mut state, 0.01, 500.0);
        let avg_lum = draw(&mut state, 0.0, 10.0);
        let key = draw(&mut state, 0.0, 2.0);
        queries.push(query(ratio, avg_lum, key));
    }
    check_batch(&ctx, &gpu, &queries);
}
