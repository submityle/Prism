//! Real-device parity for the shoreline wetness-response per-point twin:
//! [`GpuWaterWetnessResponse`](prism_volumetric_gpu::water_wetness_response::GpuWaterWetnessResponse)
//! must reproduce the `CPU` golden
//! [`wetness`](prism_render_architecture::water::wetness) closed forms —
//! [`wet_albedo_scale`](prism_render_architecture::water::wetness::wet_albedo_scale),
//! [`capillary_height`](prism_render_architecture::water::wetness::capillary_height),
//! [`puddle_depth`](prism_render_architecture::water::wetness::puddle_depth) and
//! [`is_puddle`](prism_render_architecture::water::wetness::is_puddle) — across
//! hand-computed fixtures, the degenerate zero-reach / zero-threshold branches,
//! a mixed batch, and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The four twinned functions are all `pub`, so the golden is called directly
//! as the oracle and the `GPU` output asserted equal to it. The host-only
//! time-stepping envelopes (`absorb`, `dry`, `update_wetness`, `step_moisture`)
//! thread through the shared `exp_approx` and are intentionally not twinned, so
//! they are not exercised here.
//!
//! # Parity criterion
//!
//! The puddle predicate is a magnitude comparison, so its boolean outcome is
//! exact and asserted with `==`. The three continuous kernels thread only
//! through `clamp`, `min`, `max`, a divide and a multiply-add, so for fixtures
//! chosen clear of a threshold tie the `CPU` and `GPU` agree within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Every fixture is deliberately away from a discrete tie: puddle depths are
//! kept a comfortable margin clear of the threshold so the boolean never
//! straddles it, and capillary distances are kept away from the exact reach so
//! the falloff clamp stays on one side.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wetness`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::wetness::{
    capillary_height, is_puddle, puddle_depth, wet_albedo_scale, WetnessParams,
};
use prism_volumetric_gpu::water_wetness_response::{
    GpuWaterWetnessResponse, WaterWetnessResponseParams, WaterWetnessResponseQuery,
    WaterWetnessResponseResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous value. The kernel has no `sqrt` or
/// transcendental, so the only slack is a last-place divide difference; `1e-4`
/// admits that while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Maximum capillary rise / reach of the shared fixture params.
const MAX_CAP: f32 = 0.5;
/// Absorption rate of the shared fixture params (host-only envelope driver).
const ABSORB: f32 = 2.0;
/// Drying rate of the shared fixture params (host-only envelope driver).
const DRY: f32 = 0.5;
/// Peak albedo darkening of the shared fixture params.
const DARKEN: f32 = 0.4;
/// Puddle threshold of the shared fixture params.
const PUDDLE_THR: f32 = 0.02;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// The shared fixture params as the reference `WetnessParams` (oracle input).
fn wetness_params() -> WetnessParams {
    WetnessParams {
        max_capillary_height: MAX_CAP,
        absorb_rate: ABSORB,
        dry_rate: DRY,
        darkening_strength: DARKEN,
        puddle_threshold: PUDDLE_THR,
    }
}

/// The shared fixture params as the `GPU` [`WaterWetnessResponseParams`].
fn gpu_params() -> WaterWetnessResponseParams {
    WaterWetnessResponseParams::new(MAX_CAP, ABSORB, DRY, DARKEN, PUDDLE_THR)
}

/// Evaluates the oracle for one query under `params`, mirroring the four `CPU`
/// golden closed forms.
fn golden(params: WetnessParams, query: &WaterWetnessResponseQuery) -> WaterWetnessResponseResult {
    match *query {
        WaterWetnessResponseQuery::WetAlbedoScale { wetness } => {
            WaterWetnessResponseResult::Scalar(wet_albedo_scale(wetness, params))
        }
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness,
            dist_above_water,
        } => {
            WaterWetnessResponseResult::Scalar(capillary_height(wetness, dist_above_water, params))
        }
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated,
            rain_rate,
            drain_rate,
            dt,
        } => {
            WaterWetnessResponseResult::Scalar(puddle_depth(accumulated, rain_rate, drain_rate, dt))
        }
        WaterWetnessResponseQuery::IsPuddle { depth } => {
            WaterWetnessResponseResult::Puddle(is_puddle(depth, params))
        }
    }
}

/// Pins one `GPU` result against the oracle: a scalar within tolerance, a puddle
/// predicate exactly.
fn check_one(idx: usize, got: &WaterWetnessResponseResult, want: &WaterWetnessResponseResult) {
    match (got, want) {
        (WaterWetnessResponseResult::Scalar(a), WaterWetnessResponseResult::Scalar(b)) => {
            assert!(close(*a, *b), "query {idx} scalar: gpu {a} vs cpu {b}");
        }
        (WaterWetnessResponseResult::Puddle(a), WaterWetnessResponseResult::Puddle(b)) => {
            assert_eq!(*a, *b, "query {idx} puddle: gpu {a} vs cpu {b}");
        }
        _ => panic!("query {idx} result shape mismatch: gpu {got:?} vs cpu {want:?}"),
    }
}

/// Dispatches every query under `gp` / `wp` and pins each result against the
/// oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterWetnessResponse,
    wp: WetnessParams,
    gp: WaterWetnessResponseParams,
    queries: &[WaterWetnessResponseQuery],
) {
    let got = gpu.evaluate(ctx, gp, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = golden(wp, q);
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

/// Draws a value in `[0, 1)` from `state` at micro resolution.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1_000_000) as f32 / 1_000_000.0
}

/// Builds one random, well-conditioned query. Puddle depths are kept clear of
/// the threshold tie and capillary distances clear of the exact reach so no
/// discrete decision straddles.
fn random_query(state: &mut u64) -> WaterWetnessResponseQuery {
    match lcg(state) % 4 {
        0 => WaterWetnessResponseQuery::WetAlbedoScale {
            wetness: unit(state),
        },
        1 => WaterWetnessResponseQuery::CapillaryHeight {
            wetness: unit(state),
            // Distances in [0, 0.45] stay a margin below the 0.5 reach.
            dist_above_water: unit(state) * 0.45,
        },
        2 => WaterWetnessResponseQuery::PuddleDepth {
            accumulated: unit(state) * 0.1,
            rain_rate: unit(state) * 0.05,
            drain_rate: unit(state) * 0.05,
            dt: unit(state) * 2.0,
        },
        _ => loop {
            let depth = unit(state) * 0.1;
            // Reject depths within a comfortable band of the 0.02 threshold.
            if (depth - PUDDLE_THR).abs() > 0.003 {
                break WaterWetnessResponseQuery::IsPuddle { depth };
            }
        },
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_wetness_response parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, gpu_params(), &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn wet_albedo_scale_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // Dry -> 1.0; fully wet -> 1 - darkening = 0.6; mid -> 1 - 0.4 * 0.5 = 0.8.
    let queries = [
        WaterWetnessResponseQuery::WetAlbedoScale { wetness: 0.0 },
        WaterWetnessResponseQuery::WetAlbedoScale { wetness: 1.0 },
        WaterWetnessResponseQuery::WetAlbedoScale { wetness: 0.5 },
    ];
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}

#[test]
fn capillary_height_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // Waterline soaked -> reach = 0.5; beyond reach -> 0; mid -> 0.5*0.5*0.8=0.2.
    let queries = [
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 1.0,
            dist_above_water: 0.0,
        },
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 1.0,
            dist_above_water: 1.0,
        },
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 0.5,
            dist_above_water: 0.1,
        },
    ];
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}

#[test]
fn capillary_height_zero_reach_degenerates_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // A non-positive reach disables the effect: both sides must return 0.
    let wp = WetnessParams {
        max_capillary_height: 0.0,
        ..wetness_params()
    };
    let gp = WaterWetnessResponseParams::new(0.0, ABSORB, DRY, DARKEN, PUDDLE_THR);
    let queries = [
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 1.0,
            dist_above_water: 0.0,
        },
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 0.5,
            dist_above_water: 0.2,
        },
    ];
    check(&ctx, &gpu, wp, gp, &queries);
    // Grounding: the oracle itself returns zero here, matched within tolerance.
    let got = gpu.evaluate(&ctx, gp, &queries);
    for result in &got {
        match result {
            WaterWetnessResponseResult::Scalar(value) => {
                assert!(
                    close(*value, 0.0),
                    "zero reach must yield zero height: {value}"
                );
            }
            WaterWetnessResponseResult::Puddle(_) => {
                panic!("capillary height must return a scalar");
            }
        }
    }
}

#[test]
fn puddle_depth_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // Rain fills (0.01); drainage empties to zero; more rain fills more (0.02).
    let queries = [
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated: 0.0,
            rain_rate: 0.01,
            drain_rate: 0.0,
            dt: 1.0,
        },
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated: 0.05,
            rain_rate: 0.0,
            drain_rate: 1.0,
            dt: 1.0,
        },
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated: 0.0,
            rain_rate: 0.02,
            drain_rate: 0.0,
            dt: 1.0,
        },
    ];
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}

#[test]
fn is_puddle_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // Above the 0.02 threshold -> true; below -> false.
    let queries = [
        WaterWetnessResponseQuery::IsPuddle { depth: 0.03 },
        WaterWetnessResponseQuery::IsPuddle { depth: 0.01 },
    ];
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}

#[test]
fn is_puddle_zero_threshold_admits_any_positive_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // A non-positive threshold means any strictly positive depth is a puddle.
    let wp = WetnessParams {
        puddle_threshold: 0.0,
        ..wetness_params()
    };
    let gp = WaterWetnessResponseParams::new(MAX_CAP, ABSORB, DRY, DARKEN, 0.0);
    let queries = [
        WaterWetnessResponseQuery::IsPuddle { depth: 0.001 },
        WaterWetnessResponseQuery::IsPuddle { depth: 0.0 },
    ];
    check(&ctx, &gpu, wp, gp, &queries);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    // All four operations dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    let queries = [
        WaterWetnessResponseQuery::WetAlbedoScale { wetness: 0.3 },
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness: 0.7,
            dist_above_water: 0.15,
        },
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated: 0.02,
            rain_rate: 0.03,
            drain_rate: 0.01,
            dt: 0.5,
        },
        WaterWetnessResponseQuery::IsPuddle { depth: 0.08 },
        WaterWetnessResponseQuery::WetAlbedoScale { wetness: 0.9 },
        WaterWetnessResponseQuery::IsPuddle { depth: 0.005 },
    ];
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterWetnessResponse::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    // Several workgroups' worth of mixed-op queries pin every operation across a
    // wide span of drivers in a single dispatch.
    let mut queries = Vec::new();
    for _ in 0..256 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, wetness_params(), gpu_params(), &queries);
}
