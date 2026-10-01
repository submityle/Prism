//! Real-device parity for the adaptive variable-node transmittance curve twin:
//! [`GpuHairAdaptiveTransmittance`] must reproduce the `CPU` golden
//! [`reference_adaptive_transmittance_sample`](prism_hair_gpu::adaptive_transmittance::reference_adaptive_transmittance_sample)
//! (built on
//! [`sample_transmittance`](prism_render_architecture::hair::adaptive_transmittance::sample_transmittance))
//! for a batch of receiver depths sharing one compressed curve, mapping each
//! depth to its surviving transmittance independently. The suite drives a
//! deterministic sweep (front clamp, exact nodes, interior midpoints, back
//! clamp), a stable repeat, a single-node curve, the empty curve (fully
//! transmissive everywhere), a curve with a coincident-depth span, the
//! monotone-bounded property, the empty no-op, and a large batch crossing the
//! 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The interior lookup is a subtract, a divide and a mul/add a `GPU` may fuse,
//! so each transmittance is asserted within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3` rather than bit-for-bit; the front/back clamp branches are
//! exact. Every result is asserted finite. The golden does not sanitize the
//! node data or the query depth, so the batches use only finite depths and
//! finite monotone curves; no `sin`/`cos` appears anywhere.
//!
//! Provenance: standard deep-shadow transmittance curve lookup and `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::adaptive_transmittance::{
    reference_adaptive_transmittance_sample, GpuHairAdaptiveTransmittance,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::adaptive_transmittance::{CompressedCurve, TransmittanceNode};

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

/// Builds a compressed curve from explicit `(depth, transmittance)` pairs.
fn curve(pairs: &[(f32, f32)]) -> CompressedCurve {
    CompressedCurve {
        nodes: pairs
            .iter()
            .map(|&(depth, transmittance)| TransmittanceNode {
                depth,
                transmittance,
            })
            .collect(),
    }
}

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch matches the `CPU` golden depth by depth and that every
/// value is finite.
fn assert_batch(got: &[f32], curve: &CompressedCurve, depths: &[f32]) {
    assert_eq!(got.len(), depths.len(), "one transmittance per depth");
    for (i, (&out, &depth)) in got.iter().zip(depths.iter()).enumerate() {
        let want = reference_adaptive_transmittance_sample(curve, depth);
        assert_close(out, want, &format!("depth {i} ({depth})"));
        assert!(
            out.is_finite(),
            "depth {i} ({depth}) must be finite, got {out}"
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, curve: &CompressedCurve, depths: &[f32]) -> Vec<f32> {
    GpuHairAdaptiveTransmittance::new(ctx).eval(ctx, curve, depths)
}

/// A representative monotone compressed curve: fully lit at the front, four
/// occluders thinning transmittance down to `0.1`.
fn sample_curve() -> CompressedCurve {
    curve(&[(0.0, 1.0), (2.0, 0.7), (5.0, 0.4), (8.0, 0.2), (12.0, 0.1)])
}

/// Depths that exercise every branch: before the front node, exactly on nodes,
/// interior midpoints, and beyond the deepest node.
fn sample_depths() -> Vec<f32> {
    vec![-3.0, 0.0, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 10.0, 12.0, 20.0]
}

#[test]
fn deterministic_batch_matches_golden() {
    let Some(ctx) = context_or_skip("deterministic_batch_matches_golden") else {
        return;
    };
    let c = sample_curve();
    let depths = sample_depths();
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
}

#[test]
fn repeat_run_is_stable() {
    let Some(ctx) = context_or_skip("repeat_run_is_stable") else {
        return;
    };
    let c = sample_curve();
    let depths = sample_depths();
    let first = run(&ctx, &c, &depths);
    let second = run(&ctx, &c, &depths);
    assert_eq!(first, second, "repeat runs must be bit-identical");
    assert_batch(&first, &c, &depths);
}

#[test]
fn front_and_back_are_clamped() {
    let Some(ctx) = context_or_skip("front_and_back_are_clamped") else {
        return;
    };
    let c = sample_curve();
    // Depths in front of the frontmost node read fully lit; depths at or beyond
    // the deepest node read the deepest node's transmittance.
    let depths = vec![-100.0, -1.0, 0.0, 12.0, 50.0, 1_000.0];
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
    // Front clamp is exactly 1.0.
    assert_eq!(got[0], 1.0, "deep front read must be fully lit");
    // Back clamp is exactly the last node's transmittance.
    assert_eq!(got[5], 0.1, "beyond-last read must equal the deepest node");
}

#[test]
fn interior_midpoints_interpolate_linearly() {
    let Some(ctx) = context_or_skip("interior_midpoints_interpolate_linearly") else {
        return;
    };
    // A simple two-node ramp so the midpoint transmittance is the exact mean.
    let c = curve(&[(0.0, 1.0), (10.0, 0.0)]);
    let depths = vec![0.0, 2.5, 5.0, 7.5, 10.0];
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
    // Midpoint of a 1.0 -> 0.0 ramp is 0.5.
    assert_close(got[2], 0.5, "midpoint of linear ramp");
}

#[test]
fn single_node_curve_is_a_step() {
    let Some(ctx) = context_or_skip("single_node_curve_is_a_step") else {
        return;
    };
    // One node: fully lit in front of it, its value at or beyond it.
    let c = curve(&[(4.0, 0.3)]);
    let depths = vec![-1.0, 0.0, 3.999, 4.0, 4.001, 100.0];
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
    assert_eq!(got[0], 1.0, "in front of the only node is fully lit");
    assert_eq!(got[5], 0.3, "beyond the only node reads its transmittance");
}

#[test]
fn empty_curve_is_fully_transmissive() {
    let Some(ctx) = context_or_skip("empty_curve_is_fully_transmissive") else {
        return;
    };
    let c = curve(&[]);
    let depths = vec![-5.0, 0.0, 3.0, 50.0];
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
    for (i, &v) in got.iter().enumerate() {
        assert_eq!(v, 1.0, "empty curve must read fully lit at depth {i}");
    }
}

#[test]
fn coincident_depth_span_matches_golden() {
    let Some(ctx) = context_or_skip("coincident_depth_span_matches_golden") else {
        return;
    };
    // A zero-width interior span (two nodes at depth 5) plus a normal ramp; the
    // kernel must agree with the golden's span == 0 guard across the step.
    let c = curve(&[(0.0, 1.0), (5.0, 0.6), (5.0, 0.3), (10.0, 0.1)]);
    let depths = vec![0.0, 2.5, 4.999, 5.0, 5.001, 7.5, 10.0];
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
}

#[test]
fn transmittance_is_monotone_and_bounded() {
    let Some(ctx) = context_or_skip("transmittance_is_monotone_and_bounded") else {
        return;
    };
    let c = sample_curve();
    // A dense increasing depth sweep: transmittance must stay in 0..=1 and never
    // increase as depth grows (each occluder only removes light).
    let depths: Vec<f32> = (0..=120).map(|i| -2.0 + (i as f32) * 0.15).collect();
    let got = run(&ctx, &c, &depths);
    assert_batch(&got, &c, &depths);
    let mut prev = f32::INFINITY;
    for (i, &v) in got.iter().enumerate() {
        assert!((0.0..=1.0).contains(&v), "value {i} out of range: {v}");
        assert!(
            v <= prev + 1e-4,
            "transmittance must be non-increasing at {i}: {v} > {prev}"
        );
        prev = v;
    }
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let c = sample_curve();
    let got = run(&ctx, &c, &[]);
    assert!(got.is_empty(), "empty depth batch yields an empty result");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    let c = sample_curve();
    // 130 depths (> two 64-wide workgroups) spanning well before the front node
    // to well beyond the last, so the batch exercises both clamps and every
    // interior bracket across the dispatch boundary.
    let depths: Vec<f32> = (0..130).map(|i| -4.0 + (i as f32) * 0.14).collect();
    let got = run(&ctx, &c, &depths);
    assert_eq!(got.len(), 130, "one transmittance per depth");
    assert_batch(&got, &c, &depths);
}
