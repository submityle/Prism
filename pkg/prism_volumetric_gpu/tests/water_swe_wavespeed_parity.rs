//! Real-device parity for the shallow-water wave-speed twin:
//! [`GpuWaterSweWavespeed`](prism_volumetric_gpu::water_swe_wavespeed::GpuWaterSweWavespeed)
//! must reproduce the `CPU` goldens
//! [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
//! [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
//! and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep)
//! across hand-picked fixtures (including the at-rest `f32::MAX` sentinel) plus
//! a randomized batch compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The goldens
//! [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
//! [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
//! and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep)
//! are `pub`, so each `GPU` result is pinned directly against the golden run on
//! the same inputs.
//!
//! # Parity criterion
//!
//! The per-cell speeds and the grid maximum are continuous `f32` quantities,
//! asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The timestep is
//! asserted with the same tolerance when it is a finite ratio, but the at-rest
//! sentinel `f32::MAX` is asserted with exact `==` on both sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::swe::{
    cell_wave_speed, cfl_timestep, max_wave_speed, SweConfig, SweState,
};
use prism_volumetric_gpu::water_swe_wavespeed::{
    GpuWaterSweWavespeed, WaterSweWavespeedQuery, WaterSweWavespeedResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on each continuous quantity.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Builds a `SweConfig` with the only fields that affect the twinned goldens
/// (`gravity`, `dx`); the remaining fields take valid, irrelevant dummies.
fn config(gravity: f32, dx: f32) -> SweConfig {
    SweConfig {
        nx: 1,
        nz: 1,
        dx,
        gravity,
        damping: 0.0,
    }
}

/// Computes the golden result for one query via the three `pub` oracles.
fn expected(q: &WaterSweWavespeedQuery) -> WaterSweWavespeedResult {
    let n = q.h.len().min(q.u.len()).min(q.v.len());
    let mut speeds = Vec::with_capacity(n);
    for i in 0..n {
        speeds.push(cell_wave_speed(q.h[i], q.u[i], q.v[i], q.gravity));
    }
    let state = SweState {
        h: q.h.clone(),
        u: q.u.clone(),
        v: q.v.clone(),
    };
    let cfg = config(q.gravity, q.dx);
    let max_speed = max_wave_speed(&state, cfg);
    let dt = cfl_timestep(max_speed, q.dx, q.cfl);
    WaterSweWavespeedResult {
        speeds,
        max_speed,
        dt,
    }
}

/// Pins one `GPU` result against the golden oracle.
fn assert_result(idx: usize, got: &WaterSweWavespeedResult, want: &WaterSweWavespeedResult) {
    assert_eq!(
        got.speeds.len(),
        want.speeds.len(),
        "result {idx}: speeds length"
    );
    for (j, (g, w)) in got.speeds.iter().zip(want.speeds.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "result {idx} speeds[{j}]: gpu {g} vs cpu {w}"
        );
    }
    assert!(
        close(got.max_speed, want.max_speed),
        "result {idx} max_speed: gpu {} vs cpu {}",
        got.max_speed,
        want.max_speed
    );
    // The at-rest sentinel is an exact f32::MAX on both sides; a finite
    // timestep uses the continuous tolerance.
    if want.dt == f32::MAX {
        assert!(
            got.dt == f32::MAX,
            "result {idx} dt sentinel: gpu {} vs cpu {}",
            got.dt,
            want.dt
        );
    } else {
        assert!(
            close(got.dt, want.dt),
            "result {idx} dt: gpu {} vs cpu {}",
            got.dt,
            want.dt
        );
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterSweWavespeedQuery]) {
    let gpu = GpuWaterSweWavespeed::new(ctx);
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
        eprintln!("skipping water_swe_wavespeed parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterSweWavespeed::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn wavespeed_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Normal flow with positive depth across all cells.
        WaterSweWavespeedQuery {
            h: vec![1.0, 2.0, 0.5, 3.0],
            u: vec![0.3, -0.4, 0.1, 0.9],
            v: vec![0.2, 0.5, -0.3, 0.1],
            gravity: 9.81,
            dx: 1.0,
            cfl: 0.5,
        },
        // Some dry (zero/negative) depth cells fold the celerity term to zero.
        WaterSweWavespeedQuery {
            h: vec![0.0, -1.0, 2.5, 0.0],
            u: vec![1.2, 0.0, -0.6, 0.4],
            v: vec![-0.7, 0.3, 0.8, 0.0],
            gravity: 9.81,
            dx: 0.5,
            cfl: 0.7,
        },
        // Low gravity and a single cell.
        WaterSweWavespeedQuery {
            h: vec![4.0],
            u: vec![2.0],
            v: vec![1.0],
            gravity: 1.62,
            dx: 2.0,
            cfl: 0.3,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn at_rest_hits_the_max_sentinel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A body fully at rest with zero depth: every cell speed is zero, so the
    // grid maximum is <= EPS and the timestep is the f32::MAX sentinel.
    let queries = vec![WaterSweWavespeedQuery {
        h: vec![0.0; 8],
        u: vec![0.0; 8],
        v: vec![0.0; 8],
        gravity: 9.81,
        dx: 1.0,
        cfl: 0.5,
    }];
    // Sanity: the oracle really produces the sentinel for this input.
    assert!(expected(&queries[0]).dt == f32::MAX);
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x0fe1_dc23_ab45_6789_u64;

    let mut queries: Vec<WaterSweWavespeedQuery> = Vec::new();
    while queries.len() < 512 {
        // Keep at least one cell so the grid maximum stays clearly above EPS
        // (depth and gravity are bounded away from zero), avoiding any
        // sentinel-flip ambiguity in the random path.
        let n = 1 + (lcg(&mut state) % 32) as usize;
        let mut h = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            h.push(ranged(&mut state, 0.5, 10.0));
            u.push(ranged(&mut state, -5.0, 5.0));
            v.push(ranged(&mut state, -5.0, 5.0));
        }
        queries.push(WaterSweWavespeedQuery {
            h,
            u,
            v,
            gravity: ranged(&mut state, 1.0, 15.0),
            dx: ranged(&mut state, 0.1, 2.0),
            cfl: ranged(&mut state, 0.1, 0.9),
        });
    }
    run_and_check(&ctx, &queries);
}
