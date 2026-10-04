//! Real-device parity for the coastline flood-depth twin:
//! [`GpuWaterCoastlineFloodDepth`](prism_volumetric_gpu::water_coastline_flood_depth::GpuWaterCoastlineFloodDepth)
//! must reproduce the dependency-free `CPU` golden
//! [`flood_depth`](prism_render_architecture::water::coastline::flood_depth)
//! — the clamped `max(0, sea_level - height)` subtraction `UE5` Water uses to
//! flood a terrain under a still sea level — across all-dry, all-wet, mixed and
//! exact-shoreline terrains, and the degenerate empty request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`flood_depth`] is public and pure, so the expected map is built
//! in-host cell by cell and compared against the `GPU` readback. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each cell is a single subtraction and clamp, the identical operation the
//! golden performs, so the only residual is the last-place slack of a `GPU`
//! fused multiply-add. Each cell is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coastline`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::coastline::flood_depth;
use prism_volumetric_gpu::water_coastline_flood_depth::{
    GpuWaterCoastlineFloodDepth, WaterCoastlineFloodDepth,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one flood-depth sample.
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

/// A deterministic ramp-plus-ripple terrain of `n` samples spanning both sides
/// of a typical sea level.
fn terrain(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f32;
        // A tilted ramp from below to above the sea level plus a mild ripple so
        // some cells flood and some stay dry.
        v.push(-3.0 + 0.1 * x + if i % 3 == 0 { 0.4 } else { -0.2 });
    }
    v
}

/// Pins one `GPU` flood-depth map against the `CPU` golden, cell by cell.
fn check(ctx: &GpuContext, gpu: &GpuWaterCoastlineFloodDepth, heights: &[f32], sea_level: f32) {
    let want: Vec<f32> = heights.iter().map(|&h| flood_depth(h, sea_level)).collect();
    let got = gpu.evaluate(ctx, heights, sea_level);
    assert_eq!(got.depth.len(), want.len(), "sea_level={sea_level}: len");
    for (i, (&g, &w)) in got.depth.iter().zip(want.iter()).enumerate() {
        assert!(
            close(g, w),
            "sea_level={sea_level}: cell[{i}] gpu {g} vs cpu {w}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_coastline_flood_depth parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    // An empty terrain is an honest no-op field.
    assert_eq!(
        gpu.evaluate(&ctx, &[], 0.0),
        WaterCoastlineFloodDepth { depth: Vec::new() },
        "empty terrain"
    );
}

#[test]
fn matches_golden_when_all_dry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    // Terrain entirely above the sea level: every cell is dry (depth 0).
    let heights: Vec<f32> = (0..64).map(|i| 5.0 + 0.25 * i as f32).collect();
    check(&ctx, &gpu, &heights, 0.0);
}

#[test]
fn matches_golden_when_all_wet() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    // Terrain entirely below the sea level: every cell floods positively.
    let heights: Vec<f32> = (0..64).map(|i| -10.0 - 0.1 * i as f32).collect();
    check(&ctx, &gpu, &heights, 2.0);
}

#[test]
fn matches_golden_over_mixed_terrain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    // A ramp crossing the sea level plus a ripple: a realistic coast with both
    // wet and dry cells, swept over several sea levels including the exact
    // shoreline value.
    for &sea_level in &[-5.0, -1.0, 0.0, 1.5, 4.0] {
        let heights = terrain(200);
        check(&ctx, &gpu, &heights, sea_level);
    }
}

#[test]
fn matches_golden_over_a_partial_workgroup() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    // Sizes that are not multiples of the 256-wide workgroup exercise the tail
    // guard in the kernel.
    for &n in &[1usize, 2, 255, 257, 513, 1000] {
        let heights = terrain(n);
        check(&ctx, &gpu, &heights, 0.5);
    }
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCoastlineFloodDepth::new(&ctx);
    let heights = terrain(300);
    let a = gpu.evaluate(&ctx, &heights, 0.75);
    let b = gpu.evaluate(&ctx, &heights, 0.75);
    assert_eq!(a, b, "the same request floods identically");
}
