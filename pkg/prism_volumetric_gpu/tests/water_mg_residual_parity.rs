//! Real-device parity for the single-level multigrid residual twin:
//! [`GpuWaterMgResidual`](prism_volumetric_gpu::water_mg_residual::GpuWaterMgResidual)
//! must reproduce the stateless residual field of the `CPU` golden
//! [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
//! — the interior `7`-point negative-`Laplacian` stencil and the zeroed
//! `Dirichlet` boundary layer — across hand-computed, fixed-size and randomized
//! grids compared cell-for-cell.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference function is public, so it drives the oracle directly: each
//! query feeds its active `n^3` pressure and right-hand-side slices to
//! [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
//! and the returned field is zero-padded to the fixed cap for a cell-for-cell
//! comparison. A passing `GPU == oracle` run is direct evidence the kernel
//! computes the same residual.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical integer index arithmetic and the identical
//! stencil, so the residual field matches to within floating-point tolerance
//! (`abs <= 1e-4` or `rel <= 1e-3`, with a `REL_FLOOR` of `1e-6`). The one
//! branch decision — interior versus boundary — is a pure integer comparison,
//! so there is no floating-point crossing to flip.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pressure_multigrid::residual;
use prism_volumetric_gpu::water_mg_residual::{
    GpuWaterMgResidual, WaterMgResidualQuery, WaterMgResidualResult, MAX_N,
};
use prism_volumetric_gpu::GpuContext;

/// Fixed per-query cell cap `MAX_N^3`, matching the module's padded array
/// length.
const CELLS: usize = MAX_N * MAX_N * MAX_N;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference residual field for one query by feeding its active
/// `n^3` slices to the golden and zero-padding the result to the fixed cap.
fn oracle(q: &WaterMgResidualQuery) -> WaterMgResidualResult {
    let n = q.n as usize;
    let ncells = n * n * n;
    let r = residual(&q.p[..ncells], &q.b[..ncells], n, q.h);
    let mut out = [0.0f32; CELLS];
    out[..ncells].copy_from_slice(&r);
    WaterMgResidualResult { r: out }
}

/// Pins one `GPU` result against the oracle: every cell within tolerance.
fn check_result(idx: usize, got: &WaterMgResidualResult, want: &WaterMgResidualResult) {
    for cell in 0..CELLS {
        assert!(
            close(got.r[cell], want.r[cell]),
            "query {idx} cell {cell}: gpu {} vs cpu {}",
            got.r[cell],
            want.r[cell]
        );
    }
}

/// Runs `queries` on the `GPU` and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterMgResidual, queries: &[WaterMgResidualQuery]) {
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

/// Builds a query whose active `n^3` pressure and right-hand-side cells are
/// filled from the generator and whose padding stays `0`.
fn random_query(state: &mut u64, n: u32, h: f32) -> WaterMgResidualQuery {
    let ncells = (n * n * n) as usize;
    let mut p = [0.0f32; CELLS];
    let mut b = [0.0f32; CELLS];
    for value in p.iter_mut().take(ncells) {
        *value = uniform(lcg(state), -4.0, 4.0);
    }
    for value in b.iter_mut().take(ncells) {
        *value = uniform(lcg(state), -4.0, 4.0);
    }
    WaterMgResidualQuery::new(p, b, n, h)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_mg_residual parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterMgResidual::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hand_computed_single_interior_cell() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgResidual::new(&ctx);
    // n = 3 has exactly one interior node at (1, 1, 1), flat index
    // (1*3+1)*3+1 = 13. A unit pressure spike there with zero neighbors and a
    // zero right-hand side gives A*p = (6*1)/1 = 6, so r = 0 - 6 = -6; every
    // other (boundary) cell stays 0.
    let mut p = [0.0f32; CELLS];
    p[13] = 1.0;
    let b = [0.0f32; CELLS];
    let queries = [WaterMgResidualQuery::new(p, b, 3, 1.0)];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].r[13], -6.0),
        "the lone interior residual is -6"
    );
    for (cell, value) in got[0].r.iter().enumerate() {
        if cell != 13 {
            assert!(close(*value, 0.0), "cell {cell} is on the boundary and 0");
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn hand_computed_spacing_scales_operator() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgResidual::new(&ctx);
    // Same n = 3 spike, now with h = 2.0 so inv_h2 = 0.25: A*p = 6*0.25 = 1.5
    // and r = 3.0 - 1.5 = 1.5 at the lone interior cell.
    let mut p = [0.0f32; CELLS];
    p[13] = 1.0;
    let mut b = [0.0f32; CELLS];
    b[13] = 3.0;
    let queries = [WaterMgResidualQuery::new(p, b, 3, 2.0)];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].r[13], 1.5),
        "the spacing-scaled residual is 1.5"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixed_sizes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgResidual::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // The valid vertex-centred level sizes 2^L + 1 up to the cap.
    let queries = [
        random_query(&mut state, 3, 1.0),
        random_query(&mut state, 5, 0.5),
        random_query(&mut state, 9, 1.5),
        random_query(&mut state, 9, 0.75),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgResidual::new(&ctx);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let sizes = [3u32, 5u32, 9u32];
    let mut queries = Vec::new();
    // A broad sweep: random valid level sizes, random pressure and right-hand
    // sides in [-4, 4], and a grid spacing in [0.5, 2.0] kept well away from
    // zero so the operator scale stays finite. Every branch is a pure integer
    // comparison, so no reject-sampling is needed.
    while queries.len() < 512 {
        let n = sizes[(lcg(&mut state) % 3) as usize];
        let h = uniform(lcg(&mut state), 0.5, 2.0);
        queries.push(random_query(&mut state, n, h));
    }
    check(&ctx, &gpu, &queries);
}
