//! Real-device parity for the shallow-water explicit step twin:
//! [`GpuWaterSweStep`](prism_volumetric_gpu::water_swe_step::GpuWaterSweStep)
//! must reproduce the next `SweState` of the `CPU` golden
//! [`step`](prism_render_architecture::water::swe::step) — conservative
//! continuity with zeroed reflective boundary faces, the central depth
//! gradient, first-order upwind self-advection and linear damping — across
//! hand-reasoned, fixed-size and randomized grids compared cell-for-cell.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference function is public, so it drives the oracle directly: each
//! query rebuilds the active [`SweState`](prism_render_architecture::water::swe::SweState)
//! and [`SweConfig`](prism_render_architecture::water::swe::SweConfig) from its
//! logical lengths, calls
//! [`step`](prism_render_architecture::water::swe::step) and zero-pads the
//! returned field arrays to the fixed cap for a cell-for-cell comparison. A
//! passing `GPU == oracle` run is direct evidence the kernel computes the same
//! next state.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical row-major index arithmetic, the identical
//! boundary selects and the identical upwind sign choice, so the next state
//! matches to within floating-point tolerance. The longer multiply/add chain of
//! the momentum update is absorbed by a slightly relaxed bound (`abs <= 2e-4`
//! or `rel <= 2e-3`, with a `REL_FLOOR` of `1e-6`). The branch decisions —
//! interior versus boundary face and the upwind direction — are evaluated on
//! identical `f32` bits on both sides, so no branch flips between host and
//! device.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::swe::{step, SweConfig, SweState};
use prism_volumetric_gpu::water_swe_step::{
    GpuWaterSweStep, WaterSweStepQuery, WaterSweStepResult, MAX_CELLS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison, relaxed for the longer
/// momentum multiply/add chain.
const ABS: f32 = 2e-4;
/// Relative closeness bound for the parity comparison, relaxed for the longer
/// momentum multiply/add chain.
const REL: f32 = 2e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference next state for one query by rebuilding its active
/// `SweState`/`SweConfig` from the logical lengths, calling the golden `step`
/// and zero-padding the returned fields to the fixed cap. The reported active
/// cell count mirrors the kernel's unconditional `nx * nz`.
fn oracle(q: &WaterSweStepQuery) -> WaterSweStepResult {
    let cfg = SweConfig {
        nx: q.nx,
        nz: q.nz,
        dx: q.dx,
        gravity: q.gravity,
        damping: q.damping,
    };
    let state = SweState {
        h: q.h[..q.h_len as usize].to_vec(),
        u: q.u[..q.u_len as usize].to_vec(),
        v: q.v[..q.v_len as usize].to_vec(),
    };
    let out = step(&state, cfg, q.dt);
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    h[..out.h.len()].copy_from_slice(&out.h);
    u[..out.u.len()].copy_from_slice(&out.u);
    v[..out.v.len()].copy_from_slice(&out.v);
    WaterSweStepResult {
        h,
        u,
        v,
        valid_n: q.nx * q.nz,
    }
}

/// Pins one `GPU` result against the oracle: every cell within tolerance and
/// the active cell count exactly equal.
fn check_result(idx: usize, got: &WaterSweStepResult, want: &WaterSweStepResult) {
    assert_eq!(
        got.valid_n, want.valid_n,
        "query {idx}: active cell count must match exactly"
    );
    for cell in 0..MAX_CELLS {
        assert!(
            close(got.h[cell], want.h[cell]),
            "query {idx} cell {cell} h: gpu {} vs cpu {}",
            got.h[cell],
            want.h[cell]
        );
        assert!(
            close(got.u[cell], want.u[cell]),
            "query {idx} cell {cell} u: gpu {} vs cpu {}",
            got.u[cell],
            want.u[cell]
        );
        assert!(
            close(got.v[cell], want.v[cell]),
            "query {idx} cell {cell} v: gpu {} vs cpu {}",
            got.v[cell],
            want.v[cell]
        );
    }
}

/// Runs `queries` on the `GPU` and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterSweStep, queries: &[WaterSweStepQuery]) {
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

/// Builds a non-degenerate query over an `nx * nz` grid, filling the active
/// cells from the generator with positive depths and signed velocities and
/// leaving the padding at `0`. The step scalars stay well away from zero so the
/// damping factor and the operator scale remain finite.
fn random_query(state: &mut u64, nx: u32, nz: u32) -> WaterSweStepQuery {
    let n = (nx * nz) as usize;
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    for i in 0..n {
        h[i] = uniform(lcg(state), 0.1, 3.0);
        u[i] = uniform(lcg(state), -2.0, 2.0);
        v[i] = uniform(lcg(state), -2.0, 2.0);
    }
    let dx = uniform(lcg(state), 0.5, 2.0);
    let gravity = uniform(lcg(state), 1.0, 10.0);
    let damping = uniform(lcg(state), 0.0, 0.5);
    let dt = uniform(lcg(state), 0.001, 0.02);
    WaterSweStepQuery::new(
        h, u, v, nx, nz, n as u32, n as u32, n as u32, dx, gravity, damping, dt,
    )
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_swe_step parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn single_cell_damps_velocity_and_keeps_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // A 1x1 grid is all boundary: every face is zeroed and every neighbour folds
    // back to the cell, so new_h stays at h and the velocity only decays by the
    // damping factor. With damping 0.5 and dt 0.1, damp = 1 - 0.05 = 0.95, so
    // un = 1.2 * 0.95 and vn = -0.8 * 0.95.
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    h[0] = 2.5;
    u[0] = 1.2;
    v[0] = -0.8;
    let queries = [WaterSweStepQuery::new(
        h, u, v, 1, 1, 1, 1, 1, 1.0, 9.81, 0.5, 0.1,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].h[0], 2.5), "a lone cell keeps its depth");
    assert!(
        close(got[0].u[0], 1.2 * 0.95),
        "u decays by the damping factor"
    );
    assert!(
        close(got[0].v[0], -0.8 * 0.95),
        "v decays by the damping factor"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn still_water_stays_at_rest() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // Uniform depth and zero velocity: the momentum flux is zero everywhere so
    // continuity leaves the depth untouched, and the velocity update stays at
    // zero. A body at rest remains at rest.
    let n = 16usize;
    let mut h = [0.0f32; MAX_CELLS];
    let u = [0.0f32; MAX_CELLS];
    let v = [0.0f32; MAX_CELLS];
    for value in h.iter_mut().take(n) {
        *value = 1.5;
    }
    let queries = [WaterSweStepQuery::new(
        h, u, v, 4, 4, 16, 16, 16, 1.0, 9.81, 0.25, 0.01,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    for cell in 0..n {
        assert!(close(got[0].h[cell], 1.5), "still depth is preserved");
        assert!(close(got[0].u[cell], 0.0), "still u stays zero");
        assert!(close(got[0].v[cell], 0.0), "still v stays zero");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn depth_peak_diffuses_on_a_small_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // A single raised cell in the middle of a 3x3 grid drives gravity-fed
    // outflow into its neighbours; this exercises interior, edge and corner
    // cells together. The oracle is the sole arbiter of the exact values.
    let mut h = [0.0f32; MAX_CELLS];
    let u = [0.0f32; MAX_CELLS];
    let v = [0.0f32; MAX_CELLS];
    for value in h.iter_mut().take(9) {
        *value = 1.0;
    }
    h[4] = 2.0;
    let queries = [WaterSweStepQuery::new(
        h, u, v, 3, 3, 9, 9, 9, 1.0, 9.81, 0.1, 0.005,
    )];
    check(&ctx, &gpu, &queries);
}

#[test]
fn flowing_field_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // A non-trivial depth and velocity field on a 5x4 grid exercises both the
    // conservative flux and the upwind self-advection in both axes.
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    for i in 0..20usize {
        let fi = i as f32;
        h[i] = 1.0 + 0.1 * fi;
        u[i] = 0.3 - 0.02 * fi;
        v[i] = -0.2 + 0.03 * fi;
    }
    let queries = [WaterSweStepQuery::new(
        h, u, v, 5, 4, 20, 20, 20, 1.25, 9.81, 0.2, 0.008,
    )];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_lengths_return_input_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    // nx * nz = 9 but the depth array only holds 4 logical cells, so the
    // degenerate guard fires and the whole input state is returned unchanged.
    // The arrays stay zero beyond their logical lengths, so the padded oracle
    // and the kernel's pass-through agree cell-for-cell.
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    for i in 0..4usize {
        h[i] = 0.5 + i as f32;
        u[i] = 0.1 * i as f32;
        v[i] = -0.1 * i as f32;
    }
    let queries = [WaterSweStepQuery::new(
        h, u, v, 3, 3, 4, 4, 4, 1.0, 9.81, 0.3, 0.01,
    )];
    let got = gpu.evaluate(&ctx, &queries);
    for i in 0..4usize {
        assert!(close(got[0].h[i], 0.5 + i as f32), "degenerate h unchanged");
        assert!(close(got[0].u[i], 0.1 * i as f32), "degenerate u unchanged");
        assert!(
            close(got[0].v[i], -0.1 * i as f32),
            "degenerate v unchanged"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixed_grids_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let queries = [
        random_query(&mut state, 1, 1),
        random_query(&mut state, 2, 1),
        random_query(&mut state, 8, 8),
        random_query(&mut state, 16, 4),
        random_query(&mut state, 64, 64),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweStep::new(&ctx);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut queries = Vec::new();
    // A broad sweep: random grid shapes within the cap and random positive
    // depths with signed velocities. Every branch is evaluated on identical
    // f32 bits on both sides, so no reject-sampling is needed. The query count
    // stays modest because each query carries three padded field arrays.
    while queries.len() < 128 {
        let nx = 1 + (lcg(&mut state) % 16);
        let nz = 1 + (lcg(&mut state) % 16);
        queries.push(random_query(&mut state, nx, nz));
    }
    check(&ctx, &gpu, &queries);
}
