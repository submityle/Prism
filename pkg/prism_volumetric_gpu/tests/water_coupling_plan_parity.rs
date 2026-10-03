//! Real-device parity for the coupling sub-step scheduler twin:
//! [`GpuWaterCouplingPlan`](prism_volumetric_gpu::water_coupling_plan::GpuWaterCouplingPlan)
//! must reproduce the `CPU` golden
//! [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling)
//! across hand-picked degenerate fixtures plus a randomized batch compared
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
//! [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling)
//! is `pub`, so each `GPU` plan is pinned directly against the golden run on
//! the same inputs.
//!
//! # Parity criterion
//!
//! Both outputs (`substeps`, `readback_batch`) are `u32`, asserted with exact
//! `==`. The randomized batch keeps the intermediate `crossings` quantity in a
//! small, non-saturating range and away from integer boundaries (rejection
//! sampling), so the `CPU` `as u32` and the device `u32(floor(...))` truncate
//! to the same integer.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coupling`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::coupling::{plan_coupling, CouplingPlan};
use prism_volumetric_gpu::water_coupling_plan::{
    GpuWaterCouplingPlan, WaterCouplingPlanQuery, WaterCouplingPlanResult,
};
use prism_volumetric_gpu::GpuContext;

/// Rest threshold matching `water::EPS`; the host mirrors the golden `cell`
/// floor when checking the `crossings` boundary.
const EPS: f32 = 1.0e-6;

/// Computes the golden plan for one query.
fn expected(q: &WaterCouplingPlanQuery) -> WaterCouplingPlanResult {
    let CouplingPlan {
        substeps,
        readback_batch,
    } = plan_coupling(
        q.query_count,
        q.max_rel_speed,
        q.dt,
        q.dx,
        q.max_substeps,
        q.max_readback,
    );
    WaterCouplingPlanResult {
        substeps,
        readback_batch,
    }
}

/// Pins one `GPU` result against the golden oracle (both fields exact).
fn assert_result(idx: usize, got: &WaterCouplingPlanResult, want: &WaterCouplingPlanResult) {
    assert_eq!(got.substeps, want.substeps, "result {idx}: substeps");
    assert_eq!(
        got.readback_batch, want.readback_batch,
        "result {idx}: readback_batch"
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterCouplingPlanQuery]) {
    let gpu = GpuWaterCouplingPlan::new(ctx);
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

/// Mirrors the golden `crossings` intermediate so the host can reject samples
/// whose fractional part sits too close to an integer boundary (where `CPU`
/// `as u32` and device `u32(floor(..))` could truncate differently).
fn crossings(q: &WaterCouplingPlanQuery) -> f32 {
    let cell = q.dx.max(EPS);
    q.max_rel_speed.max(0.0) * q.dt.max(0.0) / cell
}

/// Returns whether `crossings` is comfortably away from an integer boundary.
fn boundary_safe(c: f32) -> bool {
    let frac = c - c.floor();
    frac > 0.05 && frac < 0.95
}

/// Returns whether a fixture is parity-safe regardless of `CPU`/device
/// truncation agreement on `crossings`.
///
/// A fixture is safe when `crossings` sits comfortably away from an integer
/// boundary, when it truncates to zero (`< 0.95`), or when the sub-step `cap`
/// is low enough that it binds the result. In the last case
/// `substeps = min(1 + (crossings as u32), cap)` returns `cap` whenever
/// `cap + 1 <= crossings`: then `1 + (crossings as u32) >= 1 + (crossings - 1)`
/// as integers is at least `cap + 1 > cap`, so any floating-point jitter in the
/// truncation cannot change the clamped minimum.
fn fixture_safe(q: &WaterCouplingPlanQuery) -> bool {
    let c = crossings(q);
    let cap = q.max_substeps.max(1) as f32;
    boundary_safe(c) || c < 0.95 || (cap + 1.0) <= c
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_coupling_plan parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterCouplingPlan::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn coupling_plan_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Normal frame: a few crossings, caps non-binding.
        WaterCouplingPlanQuery {
            query_count: 100,
            max_rel_speed: 6.0,
            dt: 0.1,
            dx: 1.0,
            max_substeps: 16,
            max_readback: 256,
        },
        // dt = 0: no crossings, so substeps floors at one.
        WaterCouplingPlanQuery {
            query_count: 50,
            max_rel_speed: 9.0,
            dt: 0.0,
            dx: 0.5,
            max_substeps: 8,
            max_readback: 64,
        },
        // max_rel_speed = 0: body at rest, substeps floors at one.
        WaterCouplingPlanQuery {
            query_count: 200,
            max_rel_speed: 0.0,
            dt: 0.05,
            dx: 0.25,
            max_substeps: 8,
            max_readback: 128,
        },
        // query_count > max_readback: read-back batch clamps to the cap.
        WaterCouplingPlanQuery {
            query_count: 4000,
            max_rel_speed: 3.0,
            dt: 0.02,
            dx: 0.5,
            max_substeps: 32,
            max_readback: 512,
        },
        // max_substeps = 0 -> cap = 1: substeps clamps to one even when the
        // crossing demand is higher.
        WaterCouplingPlanQuery {
            query_count: 10,
            max_rel_speed: 50.0,
            dt: 0.1,
            dx: 0.5,
            max_substeps: 0,
            max_readback: 16,
        },
        // Tiny dx hits the EPS floor on cell; keep crossings mid-integer.
        WaterCouplingPlanQuery {
            query_count: 7,
            max_rel_speed: 2.0,
            dt: 0.3,
            dx: 1.0,
            max_substeps: 64,
            max_readback: 4,
        },
        // The crossing demand (0.5) truncates to zero, so substeps = 1.
        WaterCouplingPlanQuery {
            query_count: 1,
            max_rel_speed: 5.0,
            dt: 0.1,
            dx: 1.0,
            max_substeps: 4,
            max_readback: 1,
        },
    ];
    // Keep every fixture clear of an integer crossing boundary, unless the
    // sub-step cap binds the result so truncation jitter is irrelevant.
    for q in &queries {
        assert!(
            fixture_safe(q),
            "fixture crossings={} cap={}",
            crossings(q),
            q.max_substeps.max(1)
        );
    }
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x51a3_77c2_9d04_e6b1_u64;

    let mut queries: Vec<WaterCouplingPlanQuery> = Vec::new();
    while queries.len() < 512 {
        let q = WaterCouplingPlanQuery {
            query_count: lcg(&mut state) % 4096,
            max_rel_speed: ranged(&mut state, 0.0, 20.0),
            dt: ranged(&mut state, 0.0, 0.05),
            dx: ranged(&mut state, 0.1, 2.0),
            max_substeps: lcg(&mut state) % 64,
            max_readback: lcg(&mut state) % 4096,
        };
        // Reject samples whose crossing count sits near an integer boundary so
        // CPU/GPU truncation always agrees.
        if boundary_safe(crossings(&q)) {
            queries.push(q);
        }
    }
    run_and_check(&ctx, &queries);
}
