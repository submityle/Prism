//! Real-device parity for the foam-reactive-mask twin:
//! [`GpuWaterFoamReactiveMask`](prism_volumetric_gpu::water_foam_reactive_mask::GpuWaterFoamReactiveMask)
//! must reproduce the dependency-free `CPU` golden
//! [`reactive_mask_into`](prism_render_architecture::water::foam::reactive_mask_into)
//! — the per-cell `coverage >= threshold` visibility predicate — across varied
//! coverage fields, several thresholds (including boundary values that land
//! exactly on a cell's coverage), cell counts that are not multiples of the
//! `256`-wide workgroup, and the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`reactive_mask_into`] is public and pure, so the expected mask
//! is built in-host into a `bool` buffer sized to the field and compared cell
//! by cell against the `GPU` readback. A `GPU == oracle` pass is therefore
//! directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! The predicate is a single floating-point comparison with no arithmetic, so
//! the `GPU` result is exactly bit-identical to the golden — the masks must be
//! equal, with no tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::foam::reactive_mask_into;
use prism_volumetric_gpu::water_foam_reactive_mask::{
    GpuWaterFoamReactiveMask, WaterFoamReactiveMask,
};
use prism_volumetric_gpu::GpuContext;

/// Pins one `GPU` mask against the `CPU` golden, cell by cell.
fn check(ctx: &GpuContext, gpu: &GpuWaterFoamReactiveMask, field: &[f32], threshold: f32) {
    let mut want = vec![false; field.len()];
    let written = reactive_mask_into(field, threshold, &mut want);
    assert_eq!(written, field.len(), "oracle should fill the whole field");
    let got = gpu.evaluate(ctx, field, threshold);
    let label = format!("n={} threshold={threshold}", field.len());
    assert_eq!(got.mask.len(), want.len(), "{label}: len");
    assert_eq!(got.mask, want, "{label}: mask");
}

/// A deterministic coverage field over `n` cells spanning `0..=1`, with a few
/// values placed exactly on thresholds the tests probe so the `>=` boundary is
/// exercised.
fn coverage(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match i % 6 {
            0 => 0.0,
            1 => 0.25,
            2 => 0.5,
            3 => 0.75,
            4 => 1.0,
            _ => (i as f32 * 0.013) % 1.0,
        })
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_foam_reactive_mask parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterFoamReactiveMask::new(&ctx);
    assert_eq!(
        gpu.evaluate(&ctx, &[], 0.5),
        WaterFoamReactiveMask { mask: Vec::new() },
        "empty field"
    );
}

#[test]
fn matches_golden_on_threshold_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamReactiveMask::new(&ctx);
    let field = coverage(60);
    // Thresholds that land exactly on cell coverages (`>=` keeps those cells),
    // plus below-all and above-all extremes.
    for &t in &[-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5] {
        check(&ctx, &gpu, &field, t);
    }
}

#[test]
fn matches_golden_across_workgroup_tails() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamReactiveMask::new(&ctx);
    // Cell counts straddling the 256-wide workgroup boundary exercise the tail
    // guard: single cell, below, exactly one group, and just over two groups.
    for &n in &[1usize, 255, 256, 257, 513] {
        let field = coverage(n);
        check(&ctx, &gpu, &field, 0.5);
    }
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamReactiveMask::new(&ctx);
    let field = coverage(400);
    let a = gpu.evaluate(&ctx, &field, 0.3);
    let b = gpu.evaluate(&ctx, &field, 0.3);
    assert_eq!(a, b, "the same request masks identically");
}
