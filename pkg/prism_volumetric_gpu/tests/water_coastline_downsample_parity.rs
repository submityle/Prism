//! Real-device parity for the coastline heightfield-downsample twin:
//! [`GpuWaterCoastlineDownsample`](prism_volumetric_gpu::water_coastline_downsample::GpuWaterCoastlineDownsample)
//! must reproduce the dependency-free `CPU` golden
//! [`downsample_heightfield`](prism_render_architecture::water::coastline::downsample_heightfield)
//! — the block-average decimation an authoring layer runs to coarsen an
//! over-resolved terrain tile to the water solver-grid budget before flooding
//! the coast — across strides, non-square and non-divisible grids, the
//! identity stride and the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`downsample_heightfield`] is public and pure, so the expected
//! field is built in-host and compared cell by cell against the `GPU` readback.
//! A `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each thread sums its source block in the golden's row-major order, so the
//! summation order matches and the only residual is the last-place slack of a
//! `GPU` fused multiply-add in the running sum. Each coarsened sample is
//! asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coastline`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::coastline::downsample_heightfield;
use prism_volumetric_gpu::water_coastline_downsample::{
    GpuWaterCoastlineDownsample, WaterCoastlineDownsample,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one coarsened sample.
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

/// A deterministic ramp-plus-ripple terrain of `nx * nz` samples.
fn terrain(nx: u32, nz: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity((nx * nz) as usize);
    for row in 0..nz {
        for col in 0..nx {
            let r = row as f32;
            let c = col as f32;
            // A tilted plane plus a mild checker so block means are non-trivial.
            v.push(0.5 * c - 0.3 * r + if (row + col) % 2 == 0 { 0.25 } else { -0.25 });
        }
    }
    v
}

/// Pins one `GPU` downsample against the `CPU` golden, cell by cell.
fn check(ctx: &GpuContext, gpu: &GpuWaterCoastlineDownsample, nx: u32, nz: u32, stride: u32) {
    let heights = terrain(nx, nz);
    let (want, want_w, want_h) =
        downsample_heightfield(&heights, nx, nz, stride).expect("golden coarsens");
    let got = gpu.evaluate(ctx, &heights, nx, nz, stride);
    let label = format!("nx={nx} nz={nz} stride={stride}");
    assert_eq!(got.out_nx, want_w, "{label}: out_nx");
    assert_eq!(got.out_nz, want_h, "{label}: out_nz");
    assert_eq!(got.heights.len(), want.len(), "{label}: len");
    for (i, (&g, &w)) in got.heights.iter().zip(want.iter()).enumerate() {
        assert!(close(g, w), "{label}: cell[{i}] gpu {g} vs cpu {w}");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_coastline_downsample parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    let empty = WaterCoastlineDownsample {
        out_nx: 0,
        out_nz: 0,
        heights: Vec::new(),
    };
    // Zero stride, zero dimension, and a length mismatch are all honest no-ops,
    // exactly where the golden returns `None`.
    assert!(downsample_heightfield(&[0.0], 1, 1, 0).is_none());
    assert_eq!(gpu.evaluate(&ctx, &[0.0], 1, 1, 0), empty, "zero stride");
    assert!(downsample_heightfield(&[], 0, 0, 2).is_none());
    assert_eq!(gpu.evaluate(&ctx, &[], 0, 0, 2), empty, "zero dimension");
    assert!(downsample_heightfield(&[0.0, 0.0], 3, 3, 2).is_none());
    assert_eq!(
        gpu.evaluate(&ctx, &[0.0, 0.0], 3, 3, 2),
        empty,
        "length mismatch"
    );
}

#[test]
fn identity_stride_reproduces_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    // Stride one coarsens nothing: every output cell is a one-sample block.
    for &(nx, nz) in &[(1u32, 1u32), (4, 3), (7, 5)] {
        check(&ctx, &gpu, nx, nz, 1);
        let heights = terrain(nx, nz);
        let got = gpu.evaluate(&ctx, &heights, nx, nz, 1);
        assert_eq!(got.out_nx, nx);
        assert_eq!(got.out_nz, nz);
        for (&g, &w) in got.heights.iter().zip(heights.iter()) {
            assert!(close(g, w), "identity must preserve the sample");
        }
    }
}

#[test]
fn matches_golden_over_divisible_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    // Grids whose dimensions divide the stride evenly (full blocks only).
    for &stride in &[2u32, 3, 4] {
        check(&ctx, &gpu, 12, 12, stride);
    }
    check(&ctx, &gpu, 16, 8, 2);
    check(&ctx, &gpu, 8, 16, 4);
}

#[test]
fn matches_golden_over_non_divisible_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    // Grids that leave partial edge blocks (the twin must average only the
    // samples that exist, exactly as the golden does).
    check(&ctx, &gpu, 3, 3, 2);
    check(&ctx, &gpu, 5, 7, 3);
    check(&ctx, &gpu, 9, 4, 4);
    check(&ctx, &gpu, 13, 11, 5);
}

#[test]
fn matches_golden_for_large_block() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    // A single output cell averaging the whole grid, and a stride larger than
    // the grid (one partial block covering everything).
    check(&ctx, &gpu, 8, 8, 8);
    check(&ctx, &gpu, 6, 10, 16);
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineDownsample::new(&ctx);
    let heights = terrain(13, 11);
    let a = gpu.evaluate(&ctx, &heights, 13, 11, 3);
    let b = gpu.evaluate(&ctx, &heights, 13, 11, 3);
    assert_eq!(a, b, "the same request coarsens identically");
}
