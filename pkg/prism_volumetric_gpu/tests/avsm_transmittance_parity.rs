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

/// Directly asserts the AVSM physical invariant (design section 16): the
/// transmittance sampled on-device is monotonically non-increasing along
/// increasing depth and stays in `[0, 1]`.
///
/// The parity tests above only prove `gpu == cpu` within a tolerance, so a
/// device interpolation bug that stays inside that band around a monotone `CPU`
/// value could still introduce a tiny non-monotone bump. This test builds a
/// compressed curve, sweeps a *strictly increasing* depth range entirely on the
/// device, and checks the sampled curve never rises (beyond one-`ULP` join
/// rounding) — light can only be attenuated with depth, never restored.
#[test]
fn avsm_transmittance_is_monotone_non_increasing_on_gpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping avsm_transmittance_parity monotonicity: no GPU adapter available");
        return;
    };
    let kernel = GpuAvsmTransmittance::new(&ctx);

    // A small budget so `compress` fires and the surviving nodes are re-forced
    // monotone; irregular segments give a non-trivial descending curve.
    let mut curve = AvsmCurve::new(6);
    for i in 0..40 {
        let depth = i as f32 * 0.23;
        let seg = 0.02 + 0.05 * ((i * 7 % 11) as f32 / 11.0);
        curve.insert(depth, seg);
    }

    let nodes = snapshot(&curve);
    assert!(nodes.len() >= 2, "curve must retain at least two nodes");
    let lo = nodes[0].depth;
    let hi = nodes[nodes.len() - 1].depth;
    let span = (hi - lo).max(1.0);
    let start = lo - 0.5 * span;
    let end = hi + 0.5 * span;

    // A strictly increasing depth sweep (so index order == depth order),
    // covering the constant flat before the first node, the descending body and
    // the constant tail past the last node.
    let steps = 800usize;
    let depths: Vec<f32> = (0..=steps)
        .map(|i| start + (end - start) * (i as f32 / steps as f32))
        .collect();
    for w in depths.windows(2) {
        assert!(w[1] > w[0], "depth sweep must be strictly increasing");
    }

    let gpu = kernel.eval(&ctx, &nodes, &depths);
    assert_eq!(gpu.len(), depths.len());

    // One-ULP-scale slack absorbs f32 rounding at segment joins; the underlying
    // node sequence is exactly monotone, so any real violation dwarfs this.
    const MONO_SLACK: f32 = 1e-6;
    let mut prev = f32::INFINITY;
    for (&d, &g) in depths.iter().zip(gpu.iter()) {
        assert!(
            (0.0..=1.0).contains(&g),
            "transmittance must stay in [0, 1], got {g} at depth {d}",
        );
        assert!(
            g <= prev + MONO_SLACK,
            "transmittance rose with depth at {d}: {g} > previous {prev}",
        );
        prev = g;
    }
}
