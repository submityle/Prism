//! Real-device parity for the underwater-visibility twin:
//! [`GpuWaterUnderwaterTransmittance`](prism_volumetric_gpu::water_underwater_transmittance::GpuWaterUnderwaterTransmittance)
//! must reproduce the stateless outputs of the `CPU` goldens
//! [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
//! and [`is_visible`](prism_render_architecture::water::underwater::is_visible)
//! — the floored, clamped `Beer-Lambert` transmittance and the visibility
//! predicate — across the clear-water, near/far attenuation, visible/occluded,
//! threshold-clamping and saturating-extinction cases plus a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference functions are public, so they drive the oracle directly: each
//! query's extinction and distance feed
//! [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
//! for the expected transmittance, and its threshold additionally feeds
//! [`is_visible`](prism_render_architecture::water::underwater::is_visible) for
//! the expected flag. A passing `GPU == oracle` run is direct evidence the
//! kernel computes the same attenuation.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical twelve-squaring
//! [`exp_approx`](prism_render_architecture::water::exp_approx) recurrence over
//! the identical floored product, so the transmittance agrees to within
//! floating-point tolerance (`abs <= 1e-4` or `rel <= 1e-3`, with a `REL_FLOOR`
//! of `1e-6`); the `visible` flag is a discrete `u32` asserted with an exact
//! `==`. The threshold crossing is kept clear of the transmittance by the
//! fixtures and by reject-sampling the random sweep, so a last-place rounding
//! difference cannot flip the flag.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::underwater`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::underwater::{beer_lambert_transmittance, is_visible};
use prism_volumetric_gpu::water_underwater_transmittance::{
    GpuWaterUnderwaterTransmittance, WaterUnderwaterTransmittanceQuery,
    WaterUnderwaterTransmittanceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance. Values that both land near zero pass through the absolute
/// bound; larger values use the relative bound against the larger magnitude.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference response for one query by calling the goldens
/// directly.
fn oracle(q: &WaterUnderwaterTransmittanceQuery) -> WaterUnderwaterTransmittanceResult {
    let transmittance = beer_lambert_transmittance(q.extinction, q.distance);
    let visible = u32::from(is_visible(q.extinction, q.distance, q.threshold));
    WaterUnderwaterTransmittanceResult {
        transmittance,
        visible,
    }
}

/// Pins one `GPU` result against the oracle: transmittance within tolerance and
/// the visibility flag exactly.
fn check_result(
    idx: usize,
    got: &WaterUnderwaterTransmittanceResult,
    want: &WaterUnderwaterTransmittanceResult,
) {
    assert!(
        close(got.transmittance, want.transmittance),
        "query {idx} transmittance: gpu {} vs cpu {}",
        got.transmittance,
        want.transmittance
    );
    assert_eq!(
        got.visible, want.visible,
        "query {idx} visible: gpu {} vs cpu {}",
        got.visible, want.visible
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterUnderwaterTransmittance,
    queries: &[WaterUnderwaterTransmittanceQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_underwater_transmittance parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn clear_water_transmits_fully() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // Zero extinction (or zero distance) leaves the beam untouched: exp_approx(0)
    // is exactly 1, so the transmittance saturates and the target is visible.
    let queries = [
        WaterUnderwaterTransmittanceQuery::new(0.0, 100.0, 0.5),
        WaterUnderwaterTransmittanceQuery::new(1.5, 0.0, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].transmittance, 1.0),
        "no extinction transmits fully"
    );
    assert!(
        close(got[1].transmittance, 1.0),
        "no distance transmits fully"
    );
    assert_eq!(got[0].visible, 1, "fully transmitting water is visible");
    assert_eq!(got[1].visible, 1, "fully transmitting water is visible");
    check(&ctx, &gpu, &queries);
}

#[test]
fn attenuation_falls_with_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // Transmittance decays monotonically as the path grows at fixed extinction.
    let queries = [
        WaterUnderwaterTransmittanceQuery::new(0.5, 1.0, 0.0),
        WaterUnderwaterTransmittanceQuery::new(0.5, 10.0, 0.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        got[0].transmittance > got[1].transmittance,
        "longer path attenuates more"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn visibility_predicate_both_ways() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // Clear-enough water keeps the target visible at the 0.5 threshold; turbid
    // water hides it. Both are kept well clear of the threshold crossing.
    let queries = [
        WaterUnderwaterTransmittanceQuery::new(0.05, 5.0, 0.5),
        WaterUnderwaterTransmittanceQuery::new(1.0, 5.0, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].visible, 1, "clear water stays visible");
    assert_eq!(got[1].visible, 0, "turbid water occludes the target");
    check(&ctx, &gpu, &queries);
}

#[test]
fn threshold_clamps_into_unit_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // A negative threshold clamps to zero, so any non-negative transmittance is
    // visible; a threshold above one clamps to one, so only fully transmitting
    // water (here attenuated, so below one) is occluded.
    let queries = [
        WaterUnderwaterTransmittanceQuery::new(2.0, 3.0, -0.5),
        WaterUnderwaterTransmittanceQuery::new(0.5, 2.0, 1.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].visible, 1, "a negative threshold is always met");
    assert_eq!(
        got[1].visible, 0,
        "an above-one threshold rejects attenuated water"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn saturating_extinction_goes_dark() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // A very large extinction-distance product drives the base negative, so the
    // floored exp_approx saturates the transmittance to zero.
    let queries = [WaterUnderwaterTransmittanceQuery::new(50.0, 500.0, 0.01)];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].transmittance, 0.0),
        "deep turbid water is opaque"
    );
    assert_eq!(got[0].visible, 0, "an opaque column hides the target");
    check(&ctx, &gpu, &queries);
}

#[test]
fn negative_inputs_floor_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    // Negative extinction and distance both floor to zero, so the product is
    // zero and the transmittance saturates to one, matching the reference.
    let queries = [
        WaterUnderwaterTransmittanceQuery::new(-2.0, 5.0, 0.5),
        WaterUnderwaterTransmittanceQuery::new(0.5, -5.0, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].transmittance, 1.0),
        "negative extinction floors"
    );
    assert!(close(got[1].transmittance, 1.0), "negative distance floors");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterTransmittance::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries spanning clear-to-turbid water and
    // near-to-far paths. The threshold is reject-sampled to stay at least 0.03
    // away from the resolved transmittance so the hard visibility decision is
    // unambiguous and cannot flip on a last-place rounding difference. A share
    // of queries also carry negative inputs to exercise the flooring branch.
    while queries.len() < 300 {
        let extinction = if lcg(&mut state).is_multiple_of(7) {
            uniform(lcg(&mut state), -1.0, 0.0)
        } else {
            uniform(lcg(&mut state), 0.0, 3.0)
        };
        let distance = if lcg(&mut state).is_multiple_of(11) {
            uniform(lcg(&mut state), -4.0, 0.0)
        } else {
            uniform(lcg(&mut state), 0.0, 12.0)
        };
        let t = beer_lambert_transmittance(extinction, distance);
        let threshold = uniform(lcg(&mut state), 0.0, 1.0);
        // Reject when the threshold sits within the no-flip margin of the
        // transmittance; also skip the saturated endpoints where t pins to 0/1.
        if (t - threshold).abs() < 0.03 {
            continue;
        }
        queries.push(WaterUnderwaterTransmittanceQuery::new(
            extinction, distance, threshold,
        ));
    }
    check(&ctx, &gpu, &queries);
}
