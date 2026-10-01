//! Real-device parity for the isolated near-/far-field scatter-blend twin:
//! [`GpuHairScatterBlend`] must reproduce the `CPU` golden
//! [`reference_scatter_blend`](prism_hair_gpu::scatter_blend::reference_scatter_blend)
//! (which forwards to
//! [`scatter_blend`](prism_render_architecture::hair::scatter_lod::scatter_blend))
//! for a batch of projected fibre widths under one shared threshold pair.
//!
//! # Parity criterion
//!
//! The hard `0.0`/`1.0` saturation returns are exact; only the in-band
//! `(near - w) / span` ramp divides, which the `GPU` may round a few `ULP`
//! differently from the scalar reference. Each returned factor is therefore
//! compared against a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not by
//! raw bit pattern.
//!
//! The suite drives widths at and beyond both thresholds, a width inside the
//! transition band, a degenerate band (`near == far`, a hard step), sanitised
//! non-finite widths (which collapse to `0`, blend `1`), the empty no-op batch,
//! and a large multi-workgroup batch that crosses the 64-wide dispatch boundary.
//! It also asserts the structural invariants: the blend is monotone
//! non-increasing in the width and always clamped to `[0, 1]`.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: `d'Eon` 2011 near-/far-field hair scattering split plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::scatter_blend::{reference_scatter_blend, GpuHairScatterBlend};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::scatter_lod::ScatterLodThresholds;

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

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, widths: &[f32], thresholds: ScatterLodThresholds) -> Vec<f32> {
    GpuHairScatterBlend::new(ctx).eval(ctx, widths, thresholds)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a whole batch matches the `CPU` golden within tolerance.
fn assert_batch_close(got: &[f32], widths: &[f32], thresholds: ScatterLodThresholds) {
    assert_eq!(
        got.len(),
        widths.len(),
        "one blend factor per width (got {}, want {})",
        got.len(),
        widths.len()
    );
    for (i, (&g, &w)) in got.iter().zip(widths.iter()).enumerate() {
        let want = reference_scatter_blend(w, thresholds);
        assert!(
            close(g, want),
            "width {i} ({w}): device {g} must match golden {want}"
        );
    }
}

#[test]
fn regular_widths_match_golden() {
    let Some(ctx) = context_or_skip("regular_widths_match_golden") else {
        return;
    };
    let thresholds = ScatterLodThresholds::new(1.0, 0.25);
    // Below far, at far, inside band, at near, above near.
    let widths = [0.1, 0.25, 0.5, 0.625, 0.75, 1.0, 2.0];
    let got = run(&ctx, &widths, thresholds);
    assert_batch_close(&got, &widths, thresholds);

    // Spot-check the known endpoints and the band midpoint (0.625 -> 0.5).
    assert!(close(got[0], 1.0), "below far is pure far-field");
    assert!(close(got[1], 1.0), "at far is pure far-field");
    assert!(close(got[3], 0.5), "band midpoint blends to 0.5");
    assert!(close(got[5], 0.0), "at near is pure near-field");
    assert!(close(got[6], 0.0), "above near is pure near-field");
}

#[test]
fn blend_is_monotone_non_increasing() {
    let Some(ctx) = context_or_skip("blend_is_monotone_non_increasing") else {
        return;
    };
    let thresholds = ScatterLodThresholds::new(2.0, 0.5);
    let mut widths = Vec::with_capacity(40);
    for i in 0..40u32 {
        widths.push(i as f32 * 0.1);
    }
    let got = run(&ctx, &widths, thresholds);
    assert_batch_close(&got, &widths, thresholds);
    for w in &got {
        assert!((0.0..=1.0).contains(w), "blend stays in [0,1]: {w}");
    }
    for pair in got.windows(2) {
        assert!(
            pair[1] <= pair[0] + 1.0e-4,
            "blend is non-increasing in width: {} then {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn degenerate_band_is_a_hard_step() {
    let Some(ctx) = context_or_skip("degenerate_band_is_a_hard_step") else {
        return;
    };
    // near == far collapses the transition to a single step at that width.
    let thresholds = ScatterLodThresholds::new(0.5, 0.5);
    let widths = [0.25, 0.5, 0.75];
    let got = run(&ctx, &widths, thresholds);
    assert_batch_close(&got, &widths, thresholds);
    assert!(close(got[0], 1.0), "below the step is far-field");
    assert!(close(got[2], 0.0), "above the step is near-field");
}

#[test]
fn nonfinite_widths_sanitise_to_far_field() {
    let Some(ctx) = context_or_skip("nonfinite_widths_sanitise_to_far_field") else {
        return;
    };
    let thresholds = ScatterLodThresholds::new(1.0, 0.25);
    // NaN / +inf / -inf collapse to width 0 (finest footprint -> blend 1); a
    // negative width also clamps to 0.
    let widths = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0];
    let got = run(&ctx, &widths, thresholds);
    assert_batch_close(&got, &widths, thresholds);
    for b in &got {
        assert!(close(*b, 1.0), "non-finite/negative width -> blend 1: {b}");
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[], ScatterLodThresholds::new(1.0, 0.25));
    assert!(got.is_empty(), "an empty batch yields no blend factors");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 widths > 3 full 64-wide workgroups: every width index must map to its
    // own blend independent of the dispatch tiling.
    let thresholds = ScatterLodThresholds::new(1.5, 0.3);
    let mut widths = Vec::with_capacity(200);
    for i in 0..200u32 {
        widths.push(i as f32 * 0.01);
    }
    let got = run(&ctx, &widths, thresholds);
    assert_batch_close(&got, &widths, thresholds);
}
