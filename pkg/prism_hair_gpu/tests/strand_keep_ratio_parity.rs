//! Real-device parity for the isolated continuous strand keep-ratio twin:
//! [`GpuHairStrandKeepRatio`] must reproduce the `CPU` golden
//! [`reference_strand_keep_ratio`](prism_hair_gpu::strand_keep_ratio::reference_strand_keep_ratio)
//! (which forwards to
//! [`strand_keep_ratio`](prism_render_architecture::hair::cluster::strand_keep_ratio))
//! for a batch of cluster projected pixel footprints under one shared
//! [`DecimationThresholds`].
//!
//! # Parity criterion
//!
//! The hard saturation returns (`1.0` at or above `full_px`, `min_ratio` at or
//! below `cull_px`) are exact; only the in-band `(px - cull) / span` ramp
//! divides, which the `GPU` may round a few `ULP` differently, so keep ratios
//! are compared against a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`)
//! rather than bit-for-bit. Every ratio must also stay inside `[min_ratio, 1]`.
//!
//! The suite drives the saturated upper plateau, the saturated lower floor, the
//! linear band (including the exact midpoint), a degenerate `full == cull` band
//! that collapses to a hard step, non-finite and negative footprints that
//! sanitise to `0` (and thus to `min_ratio`), the empty no-op batch, and a large
//! multi-workgroup batch crossing the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: continuous screen-footprint strand decimation ramp plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::strand_keep_ratio::{reference_strand_keep_ratio, GpuHairStrandKeepRatio};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::cluster::DecimationThresholds;

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

/// Dispatches one batch of footprints through the device twin.
fn run(ctx: &GpuContext, footprints: &[f32], thresholds: DecimationThresholds) -> Vec<f32> {
    GpuHairStrandKeepRatio::new(ctx).eval(ctx, footprints, thresholds)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a whole batch matches the `CPU` golden within tolerance and stays
/// inside the `[min_ratio, 1]` invariant.
fn assert_batch_close(got: &[f32], footprints: &[f32], thresholds: DecimationThresholds) {
    assert_eq!(
        got.len(),
        footprints.len(),
        "one keep ratio per footprint (got {}, want {})",
        got.len(),
        footprints.len()
    );
    let floor = thresholds.sanitized().min_ratio;
    for (i, (&g, &px)) in got.iter().zip(footprints.iter()).enumerate() {
        let want = reference_strand_keep_ratio(px, thresholds);
        assert!(
            close(g, want),
            "footprint {i} ({px}): device ratio {g} must match golden {want}"
        );
        assert!(
            g >= floor - 1.0e-4 && g <= 1.0 + 1.0e-4,
            "footprint {i} ({px}): ratio {g} must stay inside [{floor}, 1]"
        );
    }
}

#[test]
fn saturated_upper_plateau_is_full() {
    let Some(ctx) = context_or_skip("saturated_upper_plateau_is_full") else {
        return;
    };
    let t = DecimationThresholds::new(64.0, 4.0, 0.05);
    // At or above full_px every strand is kept.
    let footprints = [64.0, 64.5, 128.0, 1000.0];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
    for &g in &got {
        assert!(
            close(g, 1.0),
            "an at-or-above-full footprint keeps every strand"
        );
    }
}

#[test]
fn saturated_lower_floor_is_min_ratio() {
    let Some(ctx) = context_or_skip("saturated_lower_floor_is_min_ratio") else {
        return;
    };
    let t = DecimationThresholds::new(64.0, 4.0, 0.05);
    // At or below cull_px only min_ratio survive.
    let footprints = [4.0, 3.0, 1.0, 0.0];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
    for &g in &got {
        assert!(
            close(g, 0.05),
            "an at-or-below-cull footprint keeps only min_ratio"
        );
    }
}

#[test]
fn in_band_ramp_is_linear() {
    let Some(ctx) = context_or_skip("in_band_ramp_is_linear") else {
        return;
    };
    let t = DecimationThresholds::new(64.0, 4.0, 0.05);
    // Span is 60; the midpoint 34 is halfway, so ratio = 0.05 + 0.95*0.5.
    let footprints = [10.0, 19.0, 34.0, 49.0, 60.0];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
    // Midpoint sanity: 0.05 + 0.95 * (34 - 4) / 60 = 0.525.
    assert!(close(got[2], 0.525), "the band midpoint ramps to 0.525");
}

#[test]
fn degenerate_band_is_a_hard_step() {
    let Some(ctx) = context_or_skip("degenerate_band_is_a_hard_step") else {
        return;
    };
    // full == cull collapses the ramp to a hard step at that footprint.
    let t = DecimationThresholds::new(16.0, 16.0, 0.1);
    let footprints = [15.9, 16.0, 16.1, 0.0, 100.0];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
}

#[test]
fn non_finite_and_negative_footprints_sanitise() {
    let Some(ctx) = context_or_skip("non_finite_and_negative_footprints_sanitise") else {
        return;
    };
    let t = DecimationThresholds::new(64.0, 4.0, 0.05);
    // NaN / +inf / -inf / negatives all collapse to 0, hence min_ratio.
    let footprints = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -5.0, -0.001];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
    for &g in &got {
        assert!(
            close(g, 0.05),
            "a non-finite or negative footprint keeps only min_ratio"
        );
    }
}

#[test]
fn default_thresholds_ramp() {
    let Some(ctx) = context_or_skip("default_thresholds_ramp") else {
        return;
    };
    let t = DecimationThresholds::default();
    let footprints = [0.0, 4.0, 20.0, 34.0, 64.0, 200.0];
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[], DecimationThresholds::default());
    assert!(got.is_empty(), "an empty batch yields no keep ratios");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 footprints > 3 full 64-wide workgroups: every footprint index must map
    // to its own ratio independent of the dispatch tiling.
    let t = DecimationThresholds::new(64.0, 4.0, 0.05);
    let mut footprints = Vec::with_capacity(200);
    for i in 0..200u32 {
        // Sweep the full range from below cull through the band past full.
        footprints.push((i as f32) * 0.5);
    }
    let got = run(&ctx, &footprints, t);
    assert_batch_close(&got, &footprints, t);
}
