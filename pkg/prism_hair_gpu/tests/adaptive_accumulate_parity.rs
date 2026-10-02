//! Real-device parity for the adaptive transmittance *accumulation* twin:
//! [`GpuHairAdaptiveAccumulate`] must reproduce the `CPU` golden
//! [`accumulate`](prism_render_architecture::hair::adaptive_transmittance::accumulate)
//! for a batch of light rays, each a slice of
//! [`TransmittanceSample`](prism_render_architecture::hair::adaptive_transmittance::TransmittanceSample),
//! emitting one [`TransmittanceCurve`](prism_render_architecture::hair::adaptive_transmittance::TransmittanceCurve)
//! per ray in input order.
//!
//! The suite covers the running composite `T = product(1 - alpha)` over a
//! multi-node ray (asserting a genuine, strictly decreasing occlusion so no
//! constant/no-op kernel could pass), the same-depth node fold (coincident
//! depths compose into one node), the host-side stable depth sort (verified with
//! shuffled input), the public-field re-sanitise (an out-of-range struct-literal
//! sample is clamped before it touches the product), the empty batch (no
//! dispatch), an all-empty-rays batch (every ray fully transmissive), a mixed
//! multi-ray batch, and a 100-ray batch that crosses the 64-wide dispatch
//! boundary so threads in distinct workgroups each recover their own curve.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The composite is a closed-form running product with no transcendental call,
//! so the `CPU` and `GPU` evaluate the same expressions in the same order and
//! diverge only through legal fused-multiply-add contraction. Node depths and
//! node counts are host passthrough and are asserted bit-exact; the running
//! transmittance is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` —
//! tight enough to fail a genuinely wrong port (a swapped factor, a missing
//! clamp, a dropped host sort), loose enough to admit the contraction. All
//! sample data are explicit decimal literals (never `sin`/`cos`).
//!
//! Provenance: standard alpha-composite running product plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::adaptive_accumulate::GpuHairAdaptiveAccumulate;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::adaptive_transmittance::{accumulate, TransmittanceSample};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// A sample from explicit decimal literals (no transcendental input).
fn s(depth: f32, alpha: f32) -> TransmittanceSample {
    TransmittanceSample::new(depth, alpha)
}

/// Runs the twin and the golden on the same `rays`, asserting one curve per ray
/// with bit-exact node depths/counts and tolerance-close transmittance.
fn assert_batch_matches(
    ctx: &GpuContext,
    twin: &GpuHairAdaptiveAccumulate,
    rays: &[Vec<TransmittanceSample>],
) {
    let refs: Vec<&[TransmittanceSample]> = rays.iter().map(Vec::as_slice).collect();
    let gpu = twin.eval(ctx, &refs);
    assert_eq!(gpu.len(), rays.len(), "one curve per ray");
    for (r, ray) in rays.iter().enumerate() {
        let cpu = accumulate(ray);
        assert_eq!(gpu[r].node_count(), cpu.node_count(), "ray {r} node count",);
        for (n, (g, c)) in gpu[r].nodes.iter().zip(&cpu.nodes).enumerate() {
            assert_eq!(
                g.depth.to_bits(),
                c.depth.to_bits(),
                "ray {r} node {n} depth must be bit-exact host passthrough",
            );
            assert!(
                close(g.transmittance, c.transmittance),
                "ray {r} node {n} transmittance: gpu {} vs cpu {}",
                g.transmittance,
                c.transmittance,
            );
        }
    }
}

#[test]
fn gpu_running_product_is_monotone_and_golden() {
    let Some(ctx) = context_or_skip("gpu_running_product_is_monotone_and_golden") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Four occluders at distinct increasing depths: the running product
    // decreases at every node and ends well below 1.
    let rays = vec![vec![s(0.0, 0.2), s(1.0, 0.3), s(2.0, 0.4), s(3.0, 0.5)]];
    assert_batch_matches(&ctx, &twin, &rays);

    let gpu = twin.eval(&ctx, &[rays[0].as_slice()]);
    let nodes = &gpu[0].nodes;
    assert_eq!(nodes.len(), 4, "one node per distinct depth");
    // Strictly decreasing (genuine occlusion, not a no-op) and in range.
    let mut prev = 1.0_f32;
    for node in nodes {
        assert!(
            node.transmittance < prev,
            "transmittance must strictly decrease at each occluder",
        );
        assert!((0.0..=1.0).contains(&node.transmittance));
        prev = node.transmittance;
    }
    // Last node composites all four: (1-.2)(1-.3)(1-.4)(1-.5)=0.168.
    assert!(
        close(nodes[3].transmittance, 0.168),
        "final composite must be 0.168, got {}",
        nodes[3].transmittance,
    );
}

#[test]
fn gpu_coincident_depths_fold_into_one_node() {
    let Some(ctx) = context_or_skip("gpu_coincident_depths_fold_into_one_node") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Three (near-)coincident samples collapse to one node compositing all
    // three: (1-0.5)^3 = 0.125.
    let rays = vec![vec![s(2.0, 0.5), s(2.0, 0.5), s(2.0 + 1e-9, 0.5)]];
    assert_batch_matches(&ctx, &twin, &rays);
    let gpu = twin.eval(&ctx, &[rays[0].as_slice()]);
    assert_eq!(gpu[0].node_count(), 1, "coincident depths fold to one node");
    assert!(
        close(gpu[0].nodes[0].transmittance, 0.125),
        "folded node must composite all three alphas, got {}",
        gpu[0].nodes[0].transmittance,
    );
}

#[test]
fn gpu_shuffled_input_sorts_on_host() {
    let Some(ctx) = context_or_skip("gpu_shuffled_input_sorts_on_host") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Deliberately out-of-order depths: the host stable sort must reorder them
    // so the curve matches the golden exactly (nodes ascending in depth).
    let rays = vec![vec![s(3.0, 0.5), s(0.0, 0.2), s(2.0, 0.4), s(1.0, 0.3)]];
    assert_batch_matches(&ctx, &twin, &rays);
    let gpu = twin.eval(&ctx, &[rays[0].as_slice()]);
    let depths: Vec<f32> = gpu[0].nodes.iter().map(|n| n.depth).collect();
    assert_eq!(
        depths,
        vec![0.0, 1.0, 2.0, 3.0],
        "nodes must be depth-sorted"
    );
}

#[test]
fn gpu_out_of_range_fields_are_resanitized() {
    let Some(ctx) = context_or_skip("gpu_out_of_range_fields_are_resanitized") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Public-field struct literals outside the valid range: alpha > 1 and a
    // negative depth. The host re-sanitise clamps alpha to 1 (full occlusion)
    // and depth to 0 before the product, matching the golden.
    let rays = vec![vec![
        TransmittanceSample {
            depth: -5.0,
            alpha: 2.0,
        },
        TransmittanceSample {
            depth: 4.0,
            alpha: 0.5,
        },
    ]];
    assert_batch_matches(&ctx, &twin, &rays);
    let gpu = twin.eval(&ctx, &[rays[0].as_slice()]);
    // Clamped alpha=1 at depth 0 drives transmittance to exactly 0 there, and it
    // stays 0 afterward (monotone, clamped).
    assert!(
        close(gpu[0].nodes[0].depth, 0.0),
        "negative depth must clamp to 0",
    );
    assert!(
        close(gpu[0].nodes[0].transmittance, 0.0),
        "alpha clamped to 1 must zero the transmittance, got {}",
        gpu[0].nodes[0].transmittance,
    );
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    let gpu = twin.eval(&ctx, &[]);
    assert!(gpu.is_empty(), "an empty batch must return no curves");
}

#[test]
fn gpu_all_empty_rays_are_fully_transmissive() {
    let Some(ctx) = context_or_skip("gpu_all_empty_rays_are_fully_transmissive") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Every ray carries no samples: each curve is empty (fully transmissive),
    // and the batch returns without a dispatch.
    let rays: Vec<Vec<TransmittanceSample>> = vec![Vec::new(), Vec::new(), Vec::new()];
    assert_batch_matches(&ctx, &twin, &rays);
    let refs: Vec<&[TransmittanceSample]> = rays.iter().map(Vec::as_slice).collect();
    let gpu = twin.eval(&ctx, &refs);
    assert_eq!(gpu.len(), 3, "one empty curve per empty ray");
    for curve in &gpu {
        assert!(
            curve.is_empty(),
            "an empty ray must read fully transmissive"
        );
    }
}

#[test]
fn gpu_mixed_multi_ray_batch() {
    let Some(ctx) = context_or_skip("gpu_mixed_multi_ray_batch") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // Rays of different lengths, including an empty one in the middle, so the
    // per-ray node ranges and the running-product reset are exercised together.
    let rays = vec![
        vec![s(0.0, 0.1), s(1.0, 0.2), s(2.5, 0.6)],
        Vec::new(),
        vec![s(0.5, 0.9)],
        vec![s(1.0, 0.25), s(1.0, 0.25), s(3.0, 0.5)],
    ];
    assert_batch_matches(&ctx, &twin, &rays);
}

#[test]
fn gpu_many_rays_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_rays_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuHairAdaptiveAccumulate::new(&ctx);
    // 100 rays, each two occluders whose depths/alphas vary with the index, so
    // every thread owns a distinct curve across the 64-wide dispatch boundary.
    let rays: Vec<Vec<TransmittanceSample>> = (0..100)
        .map(|i| {
            let f = i as f32;
            vec![
                s(0.1 + f * 0.03, 0.2 + (f * 0.004).min(0.5)),
                s(1.0 + f * 0.05, 0.3 + (f * 0.003).min(0.4)),
            ]
        })
        .collect();
    assert_batch_matches(&ctx, &twin, &rays);
}
