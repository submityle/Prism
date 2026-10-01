//! Real-device parity for the isolated hair `Zinke` dual-scattering factor twin:
//! [`GpuHairDualScatterFactors`] must reproduce the `CPU` golden
//! [`reference_dual_scatter_factors`](prism_hair_gpu::dual_scatter_factors::reference_dual_scatter_factors)
//! (built on
//! [`dual_scatter_factors`](prism_render_architecture::hair::dual_scatter_sh::dual_scatter_factors))
//! for a batch of directional transmittance samples split by hemisphere into
//! the forward factor `a_f` (`+z`) and the backward factor `a_b` (`-z`). This is
//! a hemisphere-split averaging reduction, genuinely distinct from the `SH`
//! projection twin (`project_sh`) and the integer-power twin
//! (`forward_scatter_power`).
//!
//! The suite drives a mixed two-hemisphere batch, a repeated dispatch that must
//! agree bit-for-bit with itself, forward-only and backward-only batches whose
//! empty hemisphere must yield `0`, the empty no-op short-circuit, non-finite
//! sample values sanitising to zero, degenerate / non-finite / equatorial
//! (`z == 0`) directions dropping out of both hemispheres, out-of-range values
//! clamping into `[0, 1]`, and a 130-sample batch exercising the long
//! per-hemisphere accumulation.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each factor is a sequential hemisphere sum, one divide and a clamp a `GPU`
//! may fuse, so both factors are asserted within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`, and both are asserted finite and within `[0, 1]`. No
//! `sin`/`cos` appears anywhere; all directions are explicit literals or
//! integer-derived fractions and all floating comparisons use a tolerance,
//! never a float `==`/`!=`.
//!
//! Provenance: `Zinke` 2008 dual-scattering averaged factors plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::dual_scatter_factors::{
    reference_dual_scatter_factors, GpuHairDualScatterFactors,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dual_scatter_sh::{ScatterFactors, TransmittanceSample, Vec3};

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

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts both factors match the `CPU` golden and are finite within `[0, 1]`.
fn assert_factors(got: ScatterFactors, samples: &[TransmittanceSample]) {
    let want = reference_dual_scatter_factors(samples);
    assert_close(got.forward, want.forward, "forward");
    assert_close(got.backward, want.backward, "backward");
    assert!(
        got.forward.is_finite() && (0.0..=1.0).contains(&got.forward),
        "forward must be finite in [0, 1], got {}",
        got.forward
    );
    assert!(
        got.backward.is_finite() && (0.0..=1.0).contains(&got.backward),
        "backward must be finite in [0, 1], got {}",
        got.backward
    );
}

/// A convenience sample from explicit components.
fn sample(x: f32, y: f32, z: f32, value: f32) -> TransmittanceSample {
    TransmittanceSample::new(Vec3::new(x, y, z), value)
}

/// Dispatches one factor estimate through the device twin.
fn run(ctx: &GpuContext, samples: &[TransmittanceSample]) -> ScatterFactors {
    GpuHairDualScatterFactors::new(ctx).factors(ctx, samples)
}

#[test]
fn mixed_hemispheres_match_golden() {
    let Some(ctx) = context_or_skip("mixed_hemispheres_match_golden") else {
        return;
    };
    // A spread of +z and -z samples with varied values exercises both the
    // forward and backward averages.
    let samples = [
        sample(0.0, 0.0, 1.0, 0.9),
        sample(1.0, 0.0, 2.0, 0.6),
        sample(0.0, 1.0, 3.0, 0.3),
        sample(0.0, 0.0, -1.0, 0.2),
        sample(2.0, 0.0, -1.0, 0.8),
        sample(-1.0, -1.0, -4.0, 0.5),
    ];
    assert_factors(run(&ctx, &samples), &samples);
}

#[test]
fn repeat_dispatch_is_bit_stable() {
    let Some(ctx) = context_or_skip("repeat_dispatch_is_bit_stable") else {
        return;
    };
    // The kernel is deterministic, so two dispatches of the same batch must
    // produce bit-identical factors (a raw bit compare, no tolerance).
    let samples = [
        sample(0.0, 0.0, 1.0, 0.75),
        sample(1.0, 0.0, 2.0, 0.5),
        sample(0.0, 0.0, -1.0, 0.25),
        sample(-2.0, 1.0, -3.0, 0.9),
    ];
    let a = run(&ctx, &samples);
    let b = run(&ctx, &samples);
    assert_eq!(
        a.forward.to_bits(),
        b.forward.to_bits(),
        "forward must be bit-stable across dispatches"
    );
    assert_eq!(
        a.backward.to_bits(),
        b.backward.to_bits(),
        "backward must be bit-stable across dispatches"
    );
}

#[test]
fn forward_only_leaves_backward_zero() {
    let Some(ctx) = context_or_skip("forward_only_leaves_backward_zero") else {
        return;
    };
    // Every sample sits in the +z hemisphere, so backward has no samples and the
    // golden yields exactly 0 there.
    let samples = [
        sample(0.0, 0.0, 1.0, 0.8),
        sample(1.0, 0.0, 1.0, 0.4),
        sample(0.0, 2.0, 1.0, 0.6),
    ];
    let got = run(&ctx, &samples);
    assert_factors(got, &samples);
    assert_close(got.backward, 0.0, "empty backward hemisphere");
}

#[test]
fn backward_only_leaves_forward_zero() {
    let Some(ctx) = context_or_skip("backward_only_leaves_forward_zero") else {
        return;
    };
    // Every sample sits in the -z hemisphere, so forward has no samples and the
    // golden yields exactly 0 there.
    let samples = [
        sample(0.0, 0.0, -1.0, 0.7),
        sample(1.0, 0.0, -1.0, 0.3),
        sample(0.0, -2.0, -1.0, 0.5),
    ];
    let got = run(&ctx, &samples);
    assert_factors(got, &samples);
    assert_close(got.forward, 0.0, "empty forward hemisphere");
}

#[test]
fn empty_batch_is_default() {
    let Some(ctx) = context_or_skip("empty_batch_is_default") else {
        return;
    };
    // The host short-circuits empty input (no dispatch) and returns the all-zero
    // default, matching the golden.
    let got = run(&ctx, &[]);
    assert_close(got.forward, 0.0, "empty forward");
    assert_close(got.backward, 0.0, "empty backward");
}

#[test]
fn non_finite_values_sanitize() {
    let Some(ctx) = context_or_skip("non_finite_values_sanitize") else {
        return;
    };
    // NaN/+inf/-inf values sanitise to 0 then clamp into [0, 1] exactly like the
    // golden; those samples still count toward their hemisphere.
    let samples = [
        sample(0.0, 0.0, 1.0, f32::NAN),
        sample(1.0, 0.0, 1.0, 0.6),
        sample(0.0, 0.0, -1.0, f32::INFINITY),
        sample(0.0, 1.0, -2.0, 0.4),
        sample(0.0, 0.0, -1.0, f32::NEG_INFINITY),
    ];
    assert_factors(run(&ctx, &samples), &samples);
}

#[test]
fn equatorial_and_degenerate_directions_drop_out() {
    let Some(ctx) = context_or_skip("equatorial_and_degenerate_directions_drop_out") else {
        return;
    };
    // z == 0 (equatorial), zero-length and non-finite directions all fall into
    // neither hemisphere (the golden tests dir.z > 0 and dir.z < 0 strictly), so
    // only the genuine +z/-z samples drive the averages.
    let samples = [
        sample(1.0, 0.0, 0.0, 0.9),
        sample(0.0, 1.0, 0.0, 0.1),
        sample(0.0, 0.0, 0.0, 0.5),
        sample(f32::NAN, 1.0, f32::INFINITY, 0.7),
        sample(0.0, 0.0, 1.0, 0.6),
        sample(0.0, 0.0, -1.0, 0.3),
    ];
    assert_factors(run(&ctx, &samples), &samples);
}

#[test]
fn out_of_range_values_clamp() {
    let Some(ctx) = context_or_skip("out_of_range_values_clamp") else {
        return;
    };
    // Values above 1 clamp to 1 and negative values clamp to 0 before averaging,
    // on both sides, exactly like the golden.
    let samples = [
        sample(0.0, 0.0, 1.0, 5.0),
        sample(1.0, 0.0, 2.0, -3.0),
        sample(0.0, 0.0, -1.0, 2.5),
        sample(0.0, 1.0, -1.0, -0.5),
    ];
    assert_factors(run(&ctx, &samples), &samples);
}

#[test]
fn large_batch_matches_golden() {
    let Some(ctx) = context_or_skip("large_batch_matches_golden") else {
        return;
    };
    // 130 samples drive long per-hemisphere accumulations over a deterministic
    // sweep of integer-derived directions and values, with every 17th slot
    // forced non-finite to exercise the sanitiser inside the loop. The two
    // factor lanes still fit one 64-wide workgroup.
    let mut samples = Vec::new();
    for k in 0u32..130 {
        if k % 17 == 0 {
            samples.push(sample(f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN));
        } else {
            let x = ((k % 7) as f32) - 3.0;
            let y = ((k % 5) as f32) - 2.0;
            let z = ((k % 9) as f32) - 4.0;
            let value = ((k % 11) as f32) / 10.0;
            samples.push(sample(x, y, z, value));
        }
    }
    assert_factors(run(&ctx, &samples), &samples);
}
