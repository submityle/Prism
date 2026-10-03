//! Real-device parity for the water caustics route-selector twin:
//! [`GpuWaterCausticsSelect`](prism_volumetric_gpu::water_caustics_select::GpuWaterCausticsSelect)
//! must reproduce the discrete method index of the `CPU` golden
//! [`select_caustics`](prism_render_architecture::water::caustics::select_caustics)
//! across every band, both exact band edges, the collapsed-ray-band case, the
//! bias-expansion endpoints, clamped-negative inputs, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected method index comes straight from the public golden
//! [`select_caustics`](prism_render_architecture::water::caustics::select_caustics):
//! its returned `CausticsMethod` is reduced to the same discrete rank the twin
//! emits via
//! [`CausticsMethod::cost_rank`](prism_render_architecture::water::caustics::CausticsMethod::cost_rank)
//! (`JacobianProjection` is `0`, `RayTraced` is `1`, `PhotonMapped` is `2`).
//! A `GPU` `==` oracle pass is therefore direct evidence the ported kernel
//! routes identically to the reference.
//!
//! # Parity criterion
//!
//! The method index is a *discrete classification*, so the `CPU` and `GPU`
//! agree exactly and every assertion is an integer `==` (tolerance `0`).
//! Fixtures that sit a receiver exactly on a band edge are included
//! deliberately: because the reference uses `<=` and the twin uses the same
//! ordered compare on identically rounded operands, the boundary resolves the
//! same way on both sides, so no tie-break slack is required.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::caustics::select_caustics`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::caustics::{select_caustics, CausticsThresholds};
use prism_volumetric_gpu::water_caustics_select::{
    select_caustics_index, GpuWaterCausticsSelect, WaterCausticsSelectQuery,
    WaterCausticsSelectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Computes the expected discrete method index from the public golden, reducing
/// its `CausticsMethod` to the same `cost_rank` the twin emits.
fn oracle(q: &WaterCausticsSelectQuery) -> u32 {
    let thresholds = CausticsThresholds {
        photon_max_distance: q.photon_max_distance,
        ray_max_distance: q.ray_max_distance,
    };
    let gold = select_caustics(q.camera_distance, q.quality_bias, thresholds).cost_rank();
    let independent = select_caustics_index(
        q.camera_distance,
        q.quality_bias,
        q.photon_max_distance,
        q.ray_max_distance,
    );
    assert_eq!(
        independent, gold,
        "independent CPU reimplementation must match the golden cost_rank"
    );
    gold
}

/// Dispatches every query and pins each `GPU` method index against the golden
/// oracle with an exact integer `==`.
fn check(ctx: &GpuContext, gpu: &GpuWaterCausticsSelect, queries: &[WaterCausticsSelectQuery]) {
    let got: Vec<WaterCausticsSelectResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert_eq!(
            result.method_index,
            want,
            "query {idx} method_index: gpu {} vs cpu {} \
             (camera_distance {}, quality_bias {}, photon_max {}, ray_max {})",
            result.method_index,
            want,
            q.camera_distance,
            q.quality_bias,
            q.photon_max_distance,
            q.ray_max_distance,
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

/// The deterministic edge fixtures exercising every band, both exact band
/// edges, the collapsed ray band, the bias-expansion endpoints, and clamped
/// negative inputs. Band edges land on the twin and reference's shared `<=`, so
/// they resolve identically and need no slack.
fn edge_fixtures() -> Vec<WaterCausticsSelectQuery> {
    vec![
        // Deep in the photon band: close receiver, costliest route (index 2).
        WaterCausticsSelectQuery::new(5.0, 0.0, 20.0, 60.0),
        // Mid ray-traced band: past photon_max, within ray_max (index 1).
        WaterCausticsSelectQuery::new(40.0, 0.0, 20.0, 60.0),
        // Far Jacobian fallback: beyond ray_max (index 0).
        WaterCausticsSelectQuery::new(90.0, 0.0, 20.0, 60.0),
        // Receiver exactly on the photon edge: `<=` keeps it photon-mapped.
        WaterCausticsSelectQuery::new(20.0, 0.0, 20.0, 60.0),
        // Receiver exactly on the ray edge: `<=` keeps it ray-traced.
        WaterCausticsSelectQuery::new(60.0, 0.0, 20.0, 60.0),
        // Collapsed ray band (ray_max < photon_max): the ray branch is bounded
        // by max(ray, photon) = photon, so it never fires; photon then Jacobian.
        WaterCausticsSelectQuery::new(10.0, 0.0, 30.0, 5.0),
        WaterCausticsSelectQuery::new(50.0, 0.0, 30.0, 5.0),
        // Bias = 0: no expansion, receiver just past the raw photon edge.
        WaterCausticsSelectQuery::new(21.0, 0.0, 20.0, 60.0),
        // Bias = 1: full 50% expansion lifts photon_max to 30, pulling the same
        // receiver back into the photon band.
        WaterCausticsSelectQuery::new(21.0, 1.0, 20.0, 60.0),
        // Negative camera distance clamps to 0, so any non-negative photon_max
        // keeps it photon-mapped (index 2).
        WaterCausticsSelectQuery::new(-15.0, 0.3, 8.0, 40.0),
        // Negative thresholds clamp to 0: photon_max = ray_max = 0, so any
        // positive receiver falls through to Jacobian (index 0).
        WaterCausticsSelectQuery::new(3.0, 0.5, -10.0, -4.0),
        // Zero distance with zero thresholds: 0 <= 0 keeps it photon-mapped.
        WaterCausticsSelectQuery::new(0.0, 0.0, 0.0, 0.0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_caustics_select parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn photon_band_selects_photon_mapped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    let q = WaterCausticsSelectQuery::new(5.0, 0.0, 20.0, 60.0);
    assert_eq!(oracle(&q), 2, "fixture must land in the photon band");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn ray_band_selects_ray_traced() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    let q = WaterCausticsSelectQuery::new(40.0, 0.0, 20.0, 60.0);
    assert_eq!(oracle(&q), 1, "fixture must land in the ray band");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn far_band_selects_jacobian() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    let q = WaterCausticsSelectQuery::new(90.0, 0.0, 20.0, 60.0);
    assert_eq!(oracle(&q), 0, "fixture must fall back to Jacobian");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn collapsed_ray_band_skips_ray_route() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    // ray_max < photon_max: the ray branch is bounded by max(ray, photon) =
    // photon, so a receiver past photon jumps straight to Jacobian.
    let near = WaterCausticsSelectQuery::new(10.0, 0.0, 30.0, 5.0);
    let far = WaterCausticsSelectQuery::new(50.0, 0.0, 30.0, 5.0);
    assert_eq!(oracle(&near), 2, "near receiver stays photon-mapped");
    assert_eq!(oracle(&far), 0, "far receiver skips the collapsed ray band");
    check(&ctx, &gpu, &[near, far]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCausticsSelect::new(&ctx);
    let mut state: u64 = 0x5EED_CA05_71C5_1EC7;
    let mut queries = Vec::with_capacity(256);
    for _ in 0..256 {
        // Distances and thresholds span negative (clamped) through well past the
        // bands; biases span the whole expansion range. Band edges carry no
        // tie-flip risk here (shared `<=` on identical operands), so no
        // rejection sampling is needed for a discrete classifier.
        let camera_distance = uniform(&mut state, -20.0, 120.0);
        let quality_bias = uniform(&mut state, -0.5, 1.5);
        let photon_max_distance = uniform(&mut state, -10.0, 60.0);
        let ray_max_distance = uniform(&mut state, -10.0, 100.0);
        queries.push(WaterCausticsSelectQuery::new(
            camera_distance,
            quality_bias,
            photon_max_distance,
            ray_max_distance,
        ));
    }
    check(&ctx, &gpu, &queries);
}
