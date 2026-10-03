//! Real-device parity for the `PBF` density-solve scheduler twin:
//! [`GpuWaterPbfPlan`](prism_volumetric_gpu::water_pbf_plan::GpuWaterPbfPlan)
//! must reproduce the `CPU` golden
//! [`plan_solve`](prism_render_architecture::water::pbf::plan_solve) across
//! hand-picked boundary fixtures plus a randomized batch compared
//! value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`plan_solve`](prism_render_architecture::water::pbf::plan_solve) is `pub`,
//! so each `GPU` plan is pinned directly against the golden run on the same
//! [`PbfParams`](prism_render_architecture::water::pbf::PbfParams).
//!
//! # Parity criterion
//!
//! Both outputs (`iterations`, `artificial_pressure`) are discrete `u32`
//! values, asserted with exact `==`. The iteration count is an integer `max`,
//! and the flag is a single strict ordered comparison against `EPS = 1.0e-6`;
//! fixtures straddle that threshold (including a strength exactly equal to
//! `EPS`, which must disable the term) so the strict `>` boundary is covered.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::{plan_solve, PbfParams, PbfSolvePlan};
use prism_volumetric_gpu::water_pbf_plan::{
    GpuWaterPbfPlan, WaterPbfPlanQuery, WaterPbfPlanResult,
};
use prism_volumetric_gpu::GpuContext;

/// Rest threshold matching `water::EPS`; fixtures reference it to straddle the
/// strict `>` boundary used by the artificial-pressure flag.
const EPS: f32 = 1.0e-6;

/// Builds the full golden [`PbfParams`] for one query, filling the
/// schedule-irrelevant fields with legal positive values so [`plan_solve`]
/// sees a well-formed parameter set. Only `solver_iterations` and
/// `artificial_pressure_k` affect the plan.
fn params_for(q: &WaterPbfPlanQuery) -> PbfParams {
    PbfParams {
        rest_density: 1000.0,
        particle_mass: 1.0,
        smoothing_radius: 0.1,
        relaxation_epsilon: 1.0e-3,
        artificial_pressure_k: q.artificial_pressure_k,
        artificial_pressure_n: 4,
        artificial_pressure_delta_q: 0.2,
        solver_iterations: q.solver_iterations,
    }
}

/// Computes the golden plan for one query.
fn expected(q: &WaterPbfPlanQuery) -> WaterPbfPlanResult {
    let PbfSolvePlan {
        iterations,
        artificial_pressure,
    } = plan_solve(params_for(q));
    WaterPbfPlanResult {
        iterations,
        artificial_pressure: u32::from(artificial_pressure),
    }
}

/// Pins one `GPU` result against the golden oracle (both fields exact).
fn assert_result(idx: usize, got: &WaterPbfPlanResult, want: &WaterPbfPlanResult) {
    assert_eq!(got.iterations, want.iterations, "result {idx}: iterations");
    assert_eq!(
        got.artificial_pressure, want.artificial_pressure,
        "result {idx}: artificial_pressure"
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterPbfPlanQuery]) {
    let gpu = GpuWaterPbfPlan::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_pbf_plan parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterPbfPlan::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn pbf_plan_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // solver_iterations = 0 -> clamps up to one; strength well above EPS.
        WaterPbfPlanQuery {
            solver_iterations: 0,
            artificial_pressure_k: 0.5,
        },
        // solver_iterations = 1 -> stays one; strength zero -> disabled.
        WaterPbfPlanQuery {
            solver_iterations: 1,
            artificial_pressure_k: 0.0,
        },
        // Several iterations; strength well above EPS -> enabled.
        WaterPbfPlanQuery {
            solver_iterations: 8,
            artificial_pressure_k: 2.5,
        },
        // Strength exactly EPS -> strict `>` means disabled.
        WaterPbfPlanQuery {
            solver_iterations: 4,
            artificial_pressure_k: EPS,
        },
        // Strength just below EPS -> disabled.
        WaterPbfPlanQuery {
            solver_iterations: 3,
            artificial_pressure_k: EPS * 0.5,
        },
        // Strength just above EPS -> enabled.
        WaterPbfPlanQuery {
            solver_iterations: 2,
            artificial_pressure_k: EPS * 2.0,
        },
        // Large iteration count passes through unchanged.
        WaterPbfPlanQuery {
            solver_iterations: 64,
            artificial_pressure_k: 0.125,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;

    let mut queries: Vec<WaterPbfPlanQuery> = Vec::new();
    while queries.len() < 256 {
        // Spread the strength across both sides of EPS, keeping values clear of
        // the exact threshold so CPU/GPU agree on the strict comparison; the
        // exact-EPS case is pinned by the fixture suite above.
        let k = if lcg(&mut state) & 1 == 0 {
            // Below EPS by a comfortable margin -> disabled.
            ranged(&mut state, 0.0, EPS * 0.25)
        } else {
            // Above EPS by a comfortable margin -> enabled.
            ranged(&mut state, EPS * 4.0, 5.0)
        };
        queries.push(WaterPbfPlanQuery {
            solver_iterations: lcg(&mut state) % 8,
            artificial_pressure_k: k,
        });
    }
    run_and_check(&ctx, &queries);
}
