//! Real-device parity for the trilinear multigrid prolongation twin:
//! [`GpuWaterMgProlong`](prism_volumetric_gpu::water_mg_prolong::GpuWaterMgProlong)
//! must reproduce the dense fine field of the `CPU` golden
//! [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear)
//! across hand-sized grids, both supported edge lengths, an empty batch, and a
//! randomized sweep compared cell-for-cell.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear)
//! is `pub`, so it is called directly as the oracle: each `GPU` fine field is
//! pinned against `prolong_trilinear(&coarse, nc)` entry by entry.
//!
//! # Parity criterion
//!
//! Each fine value is a weighted corner sum with exact binary weights (`1.0`,
//! `0.5`); only the floating accumulation order can diverge, so values are
//! asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pressure_multigrid::prolong_trilinear;
use prism_volumetric_gpu::water_mg_prolong::{
    GpuWaterMgProlong, WaterMgProlongQuery, WaterMgProlongResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a fine value.
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

/// Fine edge length for a coarse edge `nc`.
fn fine_edge(nc: usize) -> usize {
    (nc - 1) * 2 + 1
}

/// Pins one `GPU` fine field against the direct golden oracle, cell for cell.
fn check(ctx: &GpuContext, gpu: &GpuWaterMgProlong, queries: &[WaterMgProlongQuery]) {
    let got: Vec<WaterMgProlongResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (qi, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = prolong_trilinear(&q.coarse, q.nc);
        let nf = fine_edge(q.nc);
        assert_eq!(g.nf, nf, "query {qi} nf mismatch");
        assert_eq!(
            g.fine.len(),
            want.len(),
            "query {qi} fine length: gpu {} vs cpu {}",
            g.fine.len(),
            want.len()
        );
        for (i, (&a, &b)) in g.fine.iter().zip(want.iter()).enumerate() {
            assert!(close(a, b), "query {qi} fine[{i}]: gpu {a} vs cpu {b}");
        }
    }
}

/// A 64-bit linear congruential generator for host-side fixture synthesis; no
/// transcendental math is used.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// A pseudo-random coarse value in `[-4, 4)` with three decimal digits of
/// resolution.
fn coarse_value(state: &mut u64) -> f32 {
    let unit = (lcg(state) % 1000) as f32 / 1000.0;
    unit * 8.0 - 4.0
}

/// Builds a coarse grid of edge `nc` filled from the generator.
fn random_coarse(state: &mut u64, nc: usize) -> Vec<f32> {
    let len = nc * nc * nc;
    (0..len).map(|_| coarse_value(state)).collect()
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    // An empty batch short-circuits (a storage buffer cannot be zero-sized) and
    // returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn tiny_grid_nc3_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    // A deterministic ramp coarse grid of edge 3 (27 samples), fine edge 5.
    let coarse: Vec<f32> = (0..27).map(|i| i as f32 * 0.5 - 3.0).collect();
    check(&ctx, &gpu, &[WaterMgProlongQuery::new(coarse, 3)]);
}

#[test]
fn full_grid_nc5_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    // A deterministic coarse grid of edge 5 (125 samples), fine edge 9.
    let coarse: Vec<f32> = (0..125).map(|i| i as f32 * 0.17 - 10.0).collect();
    check(&ctx, &gpu, &[WaterMgProlongQuery::new(coarse, 5)]);
}

#[test]
fn constant_field_prolongs_to_constant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    // A constant coarse field must prolong to the same constant everywhere,
    // since the per-axis weights sum to one on both the copy and the average
    // branches.
    let coarse = vec![2.75f32; 27];
    check(&ctx, &gpu, &[WaterMgProlongQuery::new(coarse, 3)]);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    // Both edge lengths dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    let queries = vec![
        WaterMgProlongQuery::new(random_coarse(&mut state, 3), 3),
        WaterMgProlongQuery::new(random_coarse(&mut state, 5), 5),
        WaterMgProlongQuery::new(random_coarse(&mut state, 3), 3),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterMgProlong::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries: Vec<WaterMgProlongQuery> = Vec::new();
    // Many random grids (several workgroups' worth) pin every reported fine
    // cell across a span of coarse data.
    for _ in 0..512 {
        let nc = if (lcg(&mut state) & 1u32) == 0 { 3 } else { 5 };
        queries.push(WaterMgProlongQuery::new(random_coarse(&mut state, nc), nc));
    }
    check(&ctx, &gpu, &queries);
}
