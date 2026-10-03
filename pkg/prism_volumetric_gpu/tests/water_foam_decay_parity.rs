//! Real-device parity for the foam-decay twin:
//! [`GpuWaterFoamDecay`](prism_volumetric_gpu::water_foam_decay::GpuWaterFoamDecay)
//! must reproduce the `CPU` golden
//! [`foam_decay_rate`](prism_render_architecture::water::foam::foam_decay_rate)
//! and
//! [`decay_foam`](prism_render_architecture::water::foam::decay_foam) across the
//! still-to-churning decay-rate curve, an exponential decay sweep, a
//! negative-density fold, the persistence-floor clamp and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden `foam_decay_rate` and
//! `decay_foam` directly, so the test pins `GPU == golden`, not merely that the
//! shader compiles.
//!
//! # Parity criterion
//!
//! The kernel performs clamps, a guarded divide and the shared squaring
//! `exp_approx`, so a `GPU` reciprocal may land a few units in the last place
//! from the scalar reference. Every continuous output is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. Fixtures keep `reference_speed`
//! well above `EPS` and bound `rate * dt` so the squaring approximation stays
//! accurate.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::foam::{decay_foam, foam_decay_rate, FoamConfig};
use prism_volumetric_gpu::water_foam_decay::{
    GpuWaterFoamDecay, WaterFoamDecayQuery, WaterFoamDecayResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` reciprocal may land a few units in the last
/// place from the scalar reference; `1e-4` admits that legal slack while still
/// failing a wrong port.
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

/// Builds a reference `FoamConfig` from a query. The grid fields (`nx`, `nz`,
/// `dx`) are unused by the two twinned scalars, so they take valid dummies.
fn config_of(q: &WaterFoamDecayQuery) -> FoamConfig {
    FoamConfig {
        nx: 1,
        nz: 1,
        dx: 1.0,
        base_decay: q.base_decay,
        persistence_floor: q.persistence_floor,
        reference_speed: q.reference_speed,
    }
}

/// Reconstructs the golden result in-host by calling the reference functions
/// directly.
fn oracle(q: &WaterFoamDecayQuery) -> WaterFoamDecayResult {
    WaterFoamDecayResult {
        decay_rate: foam_decay_rate(q.flow_speed, config_of(q)),
        decayed_density: decay_foam(q.density, q.rate, q.dt),
    }
}

/// Pins one `GPU` result against the in-host oracle: both continuous outputs
/// within tolerance.
fn check_query(idx: usize, got: &WaterFoamDecayResult, want: &WaterFoamDecayResult) {
    assert!(
        close(got.decay_rate, want.decay_rate),
        "query {idx} decay_rate: gpu {} vs cpu {}",
        got.decay_rate,
        want.decay_rate
    );
    assert!(
        close(got.decayed_density, want.decayed_density),
        "query {idx} decayed_density: gpu {} vs cpu {}",
        got.decayed_density,
        want.decayed_density
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterFoamDecay, queries: &[WaterFoamDecayQuery]) {
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

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn decay_rate_curve_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    // Flow speeds spanning still -> at reference -> beyond, with base_decay 1.0,
    // persistence_floor 0.1 and reference_speed 2.0 (well above EPS). The decay
    // step inputs are inert here (density 0 keeps the second output at 0).
    let queries = [
        WaterFoamDecayQuery::new(0.0, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0),
        WaterFoamDecayQuery::new(0.5, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0),
        WaterFoamDecayQuery::new(1.0, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0),
        WaterFoamDecayQuery::new(2.0, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0),
        WaterFoamDecayQuery::new(5.0, 1.0, 0.1, 2.0, 0.0, 0.0, 0.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn exponential_decay_step_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    // A fixed starting density decayed at rate 1.0 over a sweep of timesteps,
    // keeping rate * dt bounded so the squaring approximation stays accurate.
    let queries = [
        WaterFoamDecayQuery::new(1.0, 1.0, 0.2, 2.0, 0.8, 1.0, 0.0),
        WaterFoamDecayQuery::new(1.0, 1.0, 0.2, 2.0, 0.8, 1.0, 0.5),
        WaterFoamDecayQuery::new(1.0, 1.0, 0.2, 2.0, 0.8, 1.0, 1.0),
        WaterFoamDecayQuery::new(1.0, 1.0, 0.2, 2.0, 0.8, 1.0, 2.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn negative_density_folds_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    // A negative density folds to zero before decaying, so the decayed density
    // is exactly zero regardless of the rate/dt pair.
    let queries = [
        WaterFoamDecayQuery::new(1.0, 1.0, 0.2, 2.0, -0.5, 1.0, 1.0),
        WaterFoamDecayQuery::new(0.5, 2.0, 0.3, 1.5, -2.0, 0.5, 0.5),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn persistence_floor_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    // Persistence floors inside, below and above the unit range exercise the
    // clamp. Flow speed is held mid-ramp (0.5 of reference) and well away from
    // the clamp edges of the speed ratio.
    let queries = [
        WaterFoamDecayQuery::new(1.0, 1.0, 0.5, 2.0, 0.6, 0.4, 0.5),
        WaterFoamDecayQuery::new(1.0, 1.0, -0.3, 2.0, 0.6, 0.4, 0.5),
        WaterFoamDecayQuery::new(1.0, 1.0, 1.4, 2.0, 0.6, 0.4, 0.5),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamDecay::new(&ctx);
    let mut state = 0x1d8e_43b7_90f2_6c15_u64;
    let mut queries: Vec<WaterFoamDecayQuery> = Vec::new();
    // Several workgroups' worth of random queries. reference_speed stays well
    // above EPS (rule), and rate * dt stays bounded (<= 6) so the squaring
    // exp_approx holds the shared continuous tolerance.
    while queries.len() < 512 {
        let flow_speed = uniform(lcg(&mut state), 0.0, 6.0);
        let base_decay = uniform(lcg(&mut state), 0.1, 3.0);
        let persistence_floor = uniform(lcg(&mut state), 0.0, 1.0);
        let reference_speed = uniform(lcg(&mut state), 0.5, 4.0);
        let density = uniform(lcg(&mut state), -0.5, 2.0);
        let rate = uniform(lcg(&mut state), 0.0, 3.0);
        let dt = uniform(lcg(&mut state), 0.0, 2.0);
        queries.push(WaterFoamDecayQuery::new(
            flow_speed,
            base_decay,
            persistence_floor,
            reference_speed,
            density,
            rate,
            dt,
        ));
    }
    check(&ctx, &gpu, &queries);
}
