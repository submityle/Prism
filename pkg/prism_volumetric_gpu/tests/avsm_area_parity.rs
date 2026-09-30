//! Real-device parity test for the AVSM curve-area twin.
//!
//! Builds real [`AvsmCurve`](prism_render_architecture::volumetric::avsm::AvsmCurve)
//! instances through the sequential `insert`/`compress` state machine (small
//! and large budgets so compression actually fires, plus sub-two-node curves),
//! then confirms the `wgpu` [`GpuAvsmArea`] kernel reproduces
//! [`AvsmCurve::area`](prism_render_architecture::volumetric::avsm::AvsmCurve::area)
//! for every curve in one batched dispatch. It also checks the all-empty batch
//! (all `0.0`) and empty-batch fast paths.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::avsm::AvsmCurve;
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{AvsmSampleNode, GpuAvsmArea};

/// Extracts the curve's control points into the GPU node type.
fn snapshot(curve: &AvsmCurve) -> Vec<AvsmSampleNode> {
    curve
        .nodes()
        .iter()
        .map(|n| AvsmSampleNode {
            depth: n.depth,
            transmittance: n.transmittance,
        })
        .collect()
}

#[test]
fn avsm_area_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_area_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmArea::new(&ctx);

    // A small-budget curve that self-compresses as samples pile up.
    let mut tight = AvsmCurve::new(4);
    for i in 0..24 {
        let depth = i as f32 * 0.37;
        let seg = 0.05 + 0.03 * (i as f32 % 5.0);
        tight.insert(depth, seg);
    }

    // A roomy-budget curve with irregular, out-of-order, duplicate and
    // out-of-range inputs.
    let mut wide = AvsmCurve::new(64);
    let samples = [
        (0.0, 0.2),
        (5.0, 0.4),
        (2.5, 0.1),
        (2.5, 0.3),
        (10.0, 1.5),
        (7.5, 0.0),
        (-3.0, 0.6),
        (12.0, 2.0),
        (1.0, -1.0),
    ];
    for &(d, s) in &samples {
        wide.insert(d, s);
    }

    // A two-node minimal curve (a single trapezoid).
    let mut minimal = AvsmCurve::new(2);
    minimal.insert(1.0, 0.5);
    minimal.insert(4.0, 0.8);

    // A single-node curve: fewer than two nodes, so zero area.
    let mut single = AvsmCurve::new(4);
    single.insert(2.0, 0.9);

    // An untouched, empty curve: zero area.
    let empty = AvsmCurve::new(4);

    let curves_src = [&tight, &wide, &minimal, &single, &empty];
    let batch: Vec<Vec<AvsmSampleNode>> = curves_src.iter().map(|c| snapshot(c)).collect();

    let gpu = kernel.eval(&ctx, &batch);
    assert_eq!(gpu.len(), curves_src.len(), "length mismatch");
    for (curve, &g) in curves_src.iter().zip(gpu.iter()) {
        let cpu = curve.area();
        let abs = (g - cpu).abs();
        let rel = abs / cpu.abs().max(1.0);
        assert!(
            abs < 1e-4 || rel < 1e-3,
            "area mismatch: gpu {g} vs cpu {cpu} (abs {abs}, rel {rel})"
        );
    }
}

#[test]
fn avsm_area_all_empty_is_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_area_parity all-empty: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmArea::new(&ctx);
    let batch: Vec<Vec<AvsmSampleNode>> = vec![Vec::new(), Vec::new(), Vec::new()];
    let gpu = kernel.eval(&ctx, &batch);
    assert_eq!(gpu.len(), batch.len());
    for &g in &gpu {
        assert_eq!(g, 0.0);
    }
    // Cross-check against a real untouched curve.
    let cpu_curve = AvsmCurve::new(4);
    assert_eq!(cpu_curve.area(), 0.0);
}

#[test]
fn avsm_area_empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_area_parity empty batch: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmArea::new(&ctx);
    let gpu = kernel.eval(&ctx, &[]);
    assert!(gpu.is_empty());
}
