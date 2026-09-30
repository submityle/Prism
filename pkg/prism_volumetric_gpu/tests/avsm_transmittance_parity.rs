//! Real-device parity test for the AVSM transmittance-sampling twin.
//!
//! Builds real [`AvsmCurve`](prism_render_architecture::volumetric::avsm::AvsmCurve)
//! instances through the sequential `insert`/`compress` state machine (small
//! and large budgets so compression actually fires), then confirms the `wgpu`
//! [`GpuAvsmTransmittance`] kernel reproduces
//! [`AvsmCurve::transmittance_at`](prism_render_architecture::volumetric::avsm::AvsmCurve::transmittance_at)
//! for a dense depth sweep that spans before the first node, exactly on nodes,
//! between nodes, and beyond the last node. It also checks the empty-curve
//! (all `1.0`) and empty-query fast paths.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::avsm::AvsmCurve;
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{AvsmSampleNode, GpuAvsmTransmittance};

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

/// Builds a dense depth sweep covering the full node range plus the flanks.
fn depth_sweep(curve: &AvsmCurve) -> Vec<f32> {
    let nodes = curve.nodes();
    let mut depths = Vec::new();
    let (lo, hi) = if nodes.is_empty() {
        (0.0f32, 1.0f32)
    } else {
        (nodes[0].depth, nodes[nodes.len() - 1].depth)
    };
    let span = (hi - lo).max(1.0);
    // Start well before the first node, march past the last.
    let start = lo - 0.5 * span;
    let end = hi + 0.5 * span;
    let steps = 400;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        depths.push(start + (end - start) * t);
    }
    // Land exactly on each node depth too.
    for n in nodes {
        depths.push(n.depth);
    }
    depths
}

fn check_curve(ctx: &GpuContext, kernel: &GpuAvsmTransmittance, curve: &AvsmCurve, label: &str) {
    let nodes = snapshot(curve);
    let depths = depth_sweep(curve);
    let gpu = kernel.eval(ctx, &nodes, &depths);
    assert_eq!(gpu.len(), depths.len(), "{label}: length mismatch");
    for (&d, &g) in depths.iter().zip(gpu.iter()) {
        let cpu = curve.transmittance_at(d);
        assert!(
            (g - cpu).abs() < 1e-6,
            "{label}: transmittance mismatch at depth {d}: gpu {g} vs cpu {cpu}"
        );
    }
}

#[test]
fn avsm_transmittance_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_transmittance_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmTransmittance::new(&ctx);

    // A small-budget curve that self-compresses as samples pile up.
    let mut tight = AvsmCurve::new(4);
    for i in 0..24 {
        let depth = i as f32 * 0.37;
        let seg = 0.05 + 0.03 * (i as f32 % 5.0);
        tight.insert(depth, seg);
    }
    check_curve(&ctx, &kernel, &tight, "tight");

    // A roomy-budget curve with irregular, out-of-order and duplicate depths.
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
    check_curve(&ctx, &kernel, &wide, "wide");

    // A two-node minimal curve (only endpoints).
    let mut minimal = AvsmCurve::new(2);
    minimal.insert(1.0, 0.5);
    minimal.insert(4.0, 0.8);
    check_curve(&ctx, &kernel, &minimal, "minimal");
}

#[test]
fn avsm_transmittance_empty_curve_is_all_ones() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_transmittance_parity empty curve: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmTransmittance::new(&ctx);
    let depths = [-1.0f32, 0.0, 3.5, 100.0];
    let gpu = kernel.eval(&ctx, &[], &depths);
    assert_eq!(gpu.len(), depths.len());
    let cpu_curve = AvsmCurve::new(4);
    for (&d, &g) in depths.iter().zip(gpu.iter()) {
        assert_eq!(g, 1.0);
        assert_eq!(cpu_curve.transmittance_at(d), 1.0);
    }
}

#[test]
fn avsm_transmittance_empty_queries_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_transmittance_parity empty queries: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmTransmittance::new(&ctx);
    let nodes = [
        AvsmSampleNode {
            depth: 0.0,
            transmittance: 1.0,
        },
        AvsmSampleNode {
            depth: 1.0,
            transmittance: 0.5,
        },
    ];
    let gpu = kernel.eval(&ctx, &nodes, &[]);
    assert!(gpu.is_empty());
}
