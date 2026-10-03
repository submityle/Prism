//! Real-device parity for the damped-`Jacobi` smoother twin:
//! [`GpuWaterMgSmooth`](prism_volumetric_gpu::water_mg_smooth::GpuWaterMgSmooth)
//! must reproduce the `CPU` golden
//! [`smooth`](prism_render_architecture::water::pressure_multigrid::smooth)
//! across several grid sizes, damping factors, sweep counts, and a randomized
//! sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected field is produced by cloning each query's `p0` prefix and
//! calling the golden `smooth` in place, so the test pins `GPU == golden`, not
//! merely that the shader compiles. Only the first `n^3` entries of the padded
//! field are compared, since the padding beyond a grid's own cells is
//! meaningless.
//!
//! # Parity criterion
//!
//! Each sweep accumulates a reciprocal rounding, so the continuous tolerance is
//! widened to `abs_diff <= 2e-4` or `rel_diff <= 2e-3`, and the sweep count is
//! held to `<= 4` to bound the drift.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pressure_multigrid::smooth;
use prism_volumetric_gpu::water_mg_smooth::{
    GpuWaterMgSmooth, WaterMgSmoothQuery, WaterMgSmoothResult, MAX_N_CUBED,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. Each damped-`Jacobi` sweep goes through a reciprocal
/// and a residual sum, so a handful of sweeps can drift a few units in the last
/// place from the scalar reference; `2e-4` admits that legal slack while still
/// failing a wrong port.
const EPS: f32 = 2.0e-4;

/// Relative parity bound, applied for larger magnitudes where the accumulated
/// units in the last place exceed the absolute floor.
const REL: f32 = 2.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Reconstructs the golden result in-host by cloning the `p0` prefix and
/// running the reference `smooth` in place. The right-hand side is the `b`
/// prefix of matching length, and the result is padded back to `MAX_N_CUBED`.
fn oracle(q: &WaterMgSmoothQuery) -> WaterMgSmoothResult {
    let ncells = (q.n as usize).pow(3);
    let mut p = q.p0[..ncells].to_vec();
    let b = q.b[..ncells].to_vec();
    smooth(&mut p, &b, q.n as usize, q.h, q.omega, q.iters);
    let mut out = [0.0f32; MAX_N_CUBED];
    out[..ncells].copy_from_slice(&p);
    WaterMgSmoothResult { p: out }
}

/// Pins one `GPU` result against the in-host oracle over the meaningful
/// `n^3`-cell prefix.
fn check_query(idx: usize, n: u32, got: &WaterMgSmoothResult, want: &WaterMgSmoothResult) {
    let ncells = (n as usize).pow(3);
    for cell in 0..ncells {
        assert!(
            close(got.p[cell], want.p[cell]),
            "query {idx} cell {cell}: gpu {} vs cpu {}",
            got.p[cell],
            want.p[cell]
        );
    }
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterMgSmooth, queries: &[WaterMgSmoothQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, q.n, g, &want);
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

/// Builds a query of per-axis size `n` with random `p0` and `b` in `[-2, 2]`.
fn random_query(state: &mut u64, n: u32, h: f32, omega: f32, iters: u32) -> WaterMgSmoothQuery {
    let ncells = (n as usize).pow(3);
    let mut p0 = vec![0.0f32; ncells];
    let mut b = vec![0.0f32; ncells];
    for slot in &mut p0 {
        *slot = uniform(lcg(state), -2.0, 2.0);
    }
    for slot in &mut b {
        *slot = uniform(lcg(state), -2.0, 2.0);
    }
    WaterMgSmoothQuery::new(n, h, omega, iters, &p0, &b)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgSmooth::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixed_grids_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgSmooth::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut queries: Vec<WaterMgSmoothQuery> = Vec::new();
    // Every combination of grid size, damping, and sweep count the task
    // mandates, each with an independent random field pair. Spacing stays well
    // above zero so the reciprocal is well-conditioned.
    for n in [3u32, 5, 9] {
        for omega in [0.6f32, 0.8, 1.0] {
            for iters in [1u32, 2, 4] {
                let h = uniform(lcg(&mut state), 0.5, 2.0);
                queries.push(random_query(&mut state, n, h, omega, iters));
            }
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn single_sweep_zero_rhs_relaxes_toward_neighbours() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgSmooth::new(&ctx);
    // A 3x3x3 grid has exactly one interior node. With b = 0, one sweep nudges
    // that node by -factor * (6 * p_c) / h^2 since all its neighbours are the
    // zero boundary. The oracle confirms the exact value either way; this
    // fixture simply pins the smallest non-trivial grid.
    let mut p0 = [0.0f32; 27];
    // Interior node index idx(3, 1, 1, 1) = (1*3 + 1)*3 + 1 = 13.
    p0[13] = 1.0;
    let b = [0.0f32; 27];
    let q = WaterMgSmoothQuery::new(3, 1.0, 0.8, 1, &p0, &b);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgSmooth::new(&ctx);
    let mut state = 0x2f6b_18d4_c7e0_519au64;
    let sizes = [3u32, 5, 9];
    let omegas = [0.6f32, 0.8, 1.0];
    let iter_counts = [1u32, 2, 4];
    let mut queries: Vec<WaterMgSmoothQuery> = Vec::new();
    // Several workgroups' worth of random grids spanning the mandated size,
    // damping, and sweep-count bands, each with its own random field pair and a
    // healthy spacing. The sweep count stays at or below 4 to bound drift.
    while queries.len() < 512 {
        let n = sizes[(lcg(&mut state) as usize) % sizes.len()];
        let omega = omegas[(lcg(&mut state) as usize) % omegas.len()];
        let iters = iter_counts[(lcg(&mut state) as usize) % iter_counts.len()];
        let h = uniform(lcg(&mut state), 0.5, 2.0);
        queries.push(random_query(&mut state, n, h, omega, iters));
    }
    check(&ctx, &gpu, &queries);
}
