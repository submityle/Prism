//! Real-device parity for the shallow-water `CFL` wave-speed twin:
//! [`GpuWaterSweCfl`](prism_volumetric_gpu::water_swe_cfl::GpuWaterSweCfl) must
//! reproduce the three stateless numeric kernels of the `CPU` golden
//! [`swe`](prism_render_architecture::water::swe) — the per-cell
//! [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
//! the reduced
//! [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
//! and the stability predicate
//! [`is_cfl_stable`](prism_render_architecture::water::swe::is_cfl_stable) —
//! across hand fixtures, a mixed batch, and a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The three golden functions are public and pure, so they are called directly
//! as the oracle: for each query a [`SweState`] of length `count` and a
//! [`SweConfig`] carrying the same `gravity` are assembled, then
//! `max_wave_speed` and `is_cfl_stable` are evaluated and packed into a
//! [`WaterSweCflResult`]. A `GPU == golden` pass is therefore direct evidence
//! the ported kernel computes the same stability answer the reference does.
//!
//! # Parity criterion
//!
//! The two speeds thread through a `sqrt` and a max reduction, so a `GPU`
//! built-in may land a few units in the last place from the scalar reference;
//! they are asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The
//! stability flag is a discrete decision, asserted exactly.
//!
//! # Conditioning
//!
//! Every fixture keeps `dt * max_speed` well clear of `cfl_number * dx` (by a
//! comfortable margin far beyond the `f32` slack), so a last-place `sqrt`
//! difference between the `CPU` and the `GPU` cannot flip the `CFL` verdict.
//! Depths include negative (dry) cells to exercise the clamp.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::swe::{
    cell_wave_speed, is_cfl_stable, max_wave_speed, SweConfig, SweState,
};
use prism_volumetric_gpu::water_swe_cfl::{
    GpuWaterSweCfl, WaterSweCflCell, WaterSweCflQuery, WaterSweCflResult, MAX_CELLS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a wave speed. A `GPU` `sqrt` may land a few units in
/// the last place from the scalar reference; `1e-4` admits that legal slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Clearance margin kept between `dt * max_speed` and `cfl_number * dx` so a
/// last-place `sqrt` difference cannot flip the discrete `CFL` verdict.
const BOUNDARY_MARGIN: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Assembles the `CPU` golden state and config for one query and evaluates the
/// three closed forms: the faithful oracle the `GPU` is pinned against.
fn oracle(q: &WaterSweCflQuery) -> WaterSweCflResult {
    let n = (q.count as usize).min(MAX_CELLS);
    let mut h = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for cell in q.cells.iter().take(n) {
        h.push(cell.h);
        u.push(cell.u);
        v.push(cell.v);
    }
    let state = SweState { h, u, v };
    // A grid at least `n` cells large with the query's gravity; nx * nz >= n.
    let cfg = SweConfig {
        nx: n.max(1) as u32,
        nz: 1,
        dx: q.dx,
        gravity: q.gravity,
        damping: 0.0,
    };
    let max_speed = max_wave_speed(&state, cfg);
    let first_cell_speed = if n > 0 {
        cell_wave_speed(q.cells[0].h, q.cells[0].u, q.cells[0].v, q.gravity)
    } else {
        0.0
    };
    let stable = is_cfl_stable(q.dt, q.dx, max_speed, q.cfl_number);
    WaterSweCflResult {
        max_speed,
        first_cell_speed,
        cfl_stable: u32::from(stable),
    }
}

/// Pins one `GPU` result against the oracle: both speeds within tolerance, the
/// stability flag exactly.
fn check_one(idx: usize, got: &WaterSweCflResult, want: &WaterSweCflResult) {
    assert!(
        close(got.max_speed, want.max_speed),
        "query {idx} max_speed: gpu {} vs cpu {}",
        got.max_speed,
        want.max_speed
    );
    assert!(
        close(got.first_cell_speed, want.first_cell_speed),
        "query {idx} first_cell_speed: gpu {} vs cpu {}",
        got.first_cell_speed,
        want.first_cell_speed
    );
    assert_eq!(
        got.cfl_stable, want.cfl_stable,
        "query {idx} cfl_stable: gpu {} vs cpu {}",
        got.cfl_stable, want.cfl_stable
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterSweCfl, queries: &[WaterSweCflQuery]) {
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

/// Draws a value in `[0.0, 1.0)` at milli resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0
}

/// Builds a zero-filled cell array for a query.
fn empty_cells() -> [WaterSweCflCell; MAX_CELLS] {
    [WaterSweCflCell::default(); MAX_CELLS]
}

/// Whether `dt * max_speed` is clear of the `cfl_number * dx` boundary so the
/// discrete `CFL` verdict is stable against a last-place `sqrt` difference.
fn boundary_clear(q: &WaterSweCflQuery) -> bool {
    let want = oracle(q);
    (q.dt * want.max_speed - q.cfl_number * q.dx).abs() > BOUNDARY_MARGIN
}

/// Draws a well-conditioned random query: random count in `0..=MAX_CELLS`,
/// random depths (including negative dry cells), velocities, gravity, timestep
/// and cell size, resampled until clear of the `CFL` boundary.
fn random_query(state: &mut u64) -> WaterSweCflQuery {
    loop {
        let count = lcg(state) % (MAX_CELLS as u32 + 1);
        let mut cells = empty_cells();
        for cell in cells.iter_mut().take(count as usize) {
            // Depth in [-1, 4): negatives exercise the dry clamp.
            cell.h = unit(state) * 5.0 - 1.0;
            // Velocities in [-2, 2).
            cell.u = unit(state) * 4.0 - 2.0;
            cell.v = unit(state) * 4.0 - 2.0;
        }
        let q = WaterSweCflQuery {
            // Gravity in [5, 15).
            gravity: 5.0 + unit(state) * 10.0,
            // Timestep in [0.001, 0.051).
            dt: 0.001 + unit(state) * 0.05,
            // Cell size in [0.25, 1.25).
            dx: 0.25 + unit(state),
            // CFL number in [0.2, 0.8).
            cfl_number: 0.2 + unit(state) * 0.6,
            count,
            cells,
        };
        if boundary_clear(&q) {
            return q;
        }
    }
}

/// Builds a query from a list of `(h, u, v)` cells and the physical constants.
fn make_query(
    gravity: f32,
    dt: f32,
    dx: f32,
    cfl_number: f32,
    cells_in: &[(f32, f32, f32)],
) -> WaterSweCflQuery {
    let mut cells = empty_cells();
    for (slot, &(h, u, v)) in cells.iter_mut().zip(cells_in.iter()) {
        slot.h = h;
        slot.u = u;
        slot.v = v;
    }
    WaterSweCflQuery {
        gravity,
        dt,
        dx,
        cfl_number,
        count: cells_in.len() as u32,
        cells,
    }
}

/// The deterministic hand fixtures: still cells, flowing cells, a dry cell, and
/// both a comfortably stable and a comfortably unstable timestep.
fn fixture_queries() -> Vec<WaterSweCflQuery> {
    vec![
        // Still water, generous timestep -> stable.
        make_query(9.81, 0.004, 0.5, 0.5, &[(2.0, 0.0, 0.0), (1.5, 0.0, 0.0)]),
        // Flowing water, small timestep -> stable.
        make_query(
            9.81,
            0.002,
            0.5,
            0.4,
            &[(1.0, 3.0, 4.0), (0.8, 1.0, 0.0), (1.2, 0.0, 2.0)],
        ),
        // Fast flow, large timestep -> unstable.
        make_query(9.81, 0.2, 0.5, 0.3, &[(3.0, 5.0, 5.0), (2.5, 4.0, 3.0)]),
        // A dry (negative depth) cell is treated as zero depth.
        make_query(9.0, 0.003, 0.6, 0.5, &[(-1.0, 0.0, 0.0), (2.0, 0.5, 0.5)]),
        // Single cell.
        make_query(9.81, 0.002, 0.5, 0.5, &[(2.0, 1.0, 1.0)]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_swe_cfl parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn zero_cell_query_reports_zero_speed_and_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    // No active cells: max_speed and first_cell_speed are zero, and the CFL
    // predicate holds vacuously (0 <= cfl*dx + EPS).
    let q = make_query(9.81, 0.01, 0.5, 0.5, &[]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
}

#[test]
fn stable_fixture_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    let q = fixture_queries()[0];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.cfl_stable, 1, "fixture must be stable");
    check_one(0, &got[0], &want);
}

#[test]
fn unstable_fixture_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    let q = fixture_queries()[2];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.cfl_stable, 0, "fixture must be unstable");
    check_one(0, &got[0], &want);
}

#[test]
fn dry_cell_is_clamped_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    // The first cell has negative depth: cell_wave_speed must treat it as dry,
    // so its speed is just the flow magnitude (here zero).
    let q = fixture_queries()[3];
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.first_cell_speed, 0.0),
        "dry still cell has zero wave speed, got {}",
        want.first_cell_speed
    );
    check_one(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    // Every hand fixture dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweCfl::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random well-conditioned queries pin every
    // output across a wide span of cell counts and drivers.
    for _ in 0..256 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
