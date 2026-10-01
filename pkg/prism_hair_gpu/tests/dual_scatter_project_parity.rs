//! Real-device parity for the isolated hair second-order `SH` transmittance
//! projection twin: [`GpuHairProjectSh`] must reproduce the `CPU` golden
//! [`reference_project_sh`](prism_hair_gpu::dual_scatter_project::reference_project_sh)
//! (built on
//! [`project_sh`](prism_render_architecture::hair::dual_scatter_sh::project_sh)
//! composed with
//! [`sh_basis`](prism_render_architecture::hair::dual_scatter_sh::sh_basis))
//! for a batch of directional transmittance samples projected onto the nine
//! band-major coefficients. This is the inverse of the `eval_sh` twin
//! (reconstruct from coefficients): projection fits coefficients from samples,
//! a reduction over the whole batch rather than an element-wise map.
//!
//! The suite drives a deterministic multi-sample batch, a repeated dispatch
//! that must agree bit-for-bit with itself, a single sample, the empty no-op
//! short-circuit, non-finite sample values sanitising to zero, degenerate /
//! non-finite directions collapsing through `normalize_or_zero`, two distinct
//! sample sets that must project to distinct coefficients, a constant forward
//! lobe that chiefly activates `Y00`/`Y20`, and a 130-sample batch (whose
//! host-computed weight is `4π / 130`) exercising the long per-coefficient
//! accumulation loop.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each coefficient is a long multiply-add chain a `GPU` may fuse, so every
//! coefficient is asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3`, and
//! every coefficient is asserted finite. No `sin`/`cos` appears anywhere; all
//! directions are explicit literals or integer-derived fractions and all
//! floating comparisons use a tolerance, never a float `==`/`!=`.
//!
//! Provenance: standard orthonormal real spherical-harmonic projection plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::dual_scatter_project::{reference_project_sh, GpuHairProjectSh};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dual_scatter_sh::{ShCoeffs, TransmittanceSample, Vec3};

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

/// Asserts the nine projected coefficients match the `CPU` golden coefficient by
/// coefficient, and that each coefficient is finite.
fn assert_coeffs(got: ShCoeffs, samples: &[TransmittanceSample]) {
    let want = reference_project_sh(samples);
    for k in 0..9 {
        assert_close(got.c[k], want.c[k], &format!("coefficient {k}"));
        assert!(
            got.c[k].is_finite(),
            "coefficient {k} must be finite, got {}",
            got.c[k]
        );
    }
}

/// A convenience sample from explicit components.
fn sample(x: f32, y: f32, z: f32, value: f32) -> TransmittanceSample {
    TransmittanceSample::new(Vec3::new(x, y, z), value)
}

/// Dispatches one projection through the device twin.
fn run(ctx: &GpuContext, samples: &[TransmittanceSample]) -> ShCoeffs {
    GpuHairProjectSh::new(ctx).project(ctx, samples)
}

#[test]
fn deterministic_batch_matches_golden() {
    let Some(ctx) = context_or_skip("deterministic_batch_matches_golden") else {
        return;
    };
    // A spread of signed axes and oblique directions with varied transmittance
    // values exercises every linear and quadratic basis term in the projection.
    let samples = [
        sample(0.0, 0.0, 1.0, 0.9),
        sample(0.0, 0.0, -1.0, 0.1),
        sample(1.0, 0.0, 0.0, 0.6),
        sample(-1.0, 0.0, 0.0, 0.4),
        sample(0.0, 1.0, 0.0, 0.7),
        sample(0.0, -1.0, 0.0, 0.3),
        sample(3.0, 4.0, 0.0, 0.5),
        sample(1.0, 2.0, 2.0, 0.8),
    ];
    assert_coeffs(run(&ctx, &samples), &samples);
}

#[test]
fn repeat_dispatch_is_bit_stable() {
    let Some(ctx) = context_or_skip("repeat_dispatch_is_bit_stable") else {
        return;
    };
    // The kernel is deterministic, so two dispatches of the same batch must
    // produce bit-identical coefficients (a raw bit compare, no tolerance).
    let samples = [
        sample(0.0, 0.0, 1.0, 0.75),
        sample(1.0, 1.0, 0.0, 0.5),
        sample(-2.0, 1.0, 3.0, 0.25),
        sample(0.0, -1.0, 1.0, 0.9),
    ];
    let a = run(&ctx, &samples);
    let b = run(&ctx, &samples);
    for k in 0..9 {
        assert_eq!(
            a.c[k].to_bits(),
            b.c[k].to_bits(),
            "coefficient {k} must be bit-stable across dispatches"
        );
    }
}

#[test]
fn single_sample_matches_golden() {
    let Some(ctx) = context_or_skip("single_sample_matches_golden") else {
        return;
    };
    // One sample: the weight is 4*pi and the coefficients are a single scaled
    // basis evaluation.
    let samples = [sample(0.0, 0.0, 1.0, 0.5)];
    assert_coeffs(run(&ctx, &samples), &samples);
}

#[test]
fn empty_batch_is_default() {
    let Some(ctx) = context_or_skip("empty_batch_is_default") else {
        return;
    };
    // The host short-circuits empty input (no dispatch; storage buffers cannot
    // be zero-sized) and returns the all-zero default, matching the golden.
    let got = run(&ctx, &[]);
    for (k, &v) in got.c.iter().enumerate() {
        assert_close(v, 0.0, &format!("empty coefficient {k}"));
    }
}

#[test]
fn non_finite_values_sanitize() {
    let Some(ctx) = context_or_skip("non_finite_values_sanitize") else {
        return;
    };
    // NaN/+inf/-inf transmittance values sanitise to 0 before the weight
    // multiply exactly like the golden, so they contribute nothing and the
    // coefficients stay finite.
    let samples = [
        sample(0.0, 0.0, 1.0, f32::NAN),
        sample(1.0, 0.0, 0.0, f32::INFINITY),
        sample(0.0, 1.0, 0.0, f32::NEG_INFINITY),
        sample(0.0, 0.0, -1.0, 0.5),
        sample(1.0, 1.0, 1.0, 0.3),
    ];
    assert_coeffs(run(&ctx, &samples), &samples);
}

#[test]
fn degenerate_directions_collapse() {
    let Some(ctx) = context_or_skip("degenerate_directions_collapse") else {
        return;
    };
    // Zero-length and non-finite directions normalise to zero on both sides,
    // leaving only the Y00 and Y20 basis terms non-zero for those samples; the
    // device must reproduce the golden within tolerance.
    let samples = [
        sample(0.0, 0.0, 0.0, 0.8),
        sample(f32::NAN, 1.0, 0.0, 0.6),
        sample(f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.4),
        sample(0.0, 0.0, 1.0, 0.5),
    ];
    assert_coeffs(run(&ctx, &samples), &samples);
}

#[test]
fn distinct_sample_sets_project_distinctly() {
    let Some(ctx) = context_or_skip("distinct_sample_sets_project_distinctly") else {
        return;
    };
    // Two different sample sets must project to different coefficient sets, and
    // each must match its own golden.
    let a = [sample(0.0, 0.0, 1.0, 0.9), sample(1.0, 0.0, 0.0, 0.2)];
    let b = [sample(0.0, 1.0, 0.0, 0.1), sample(-1.0, 0.0, 0.0, 0.8)];
    let ga = run(&ctx, &a);
    let gb = run(&ctx, &b);
    assert_coeffs(ga, &a);
    assert_coeffs(gb, &b);
    let differs = (0..9).any(|k| (ga.c[k] - gb.c[k]).abs() > 1e-3);
    assert!(
        differs,
        "distinct sample sets must project to distinct coefficients"
    );
}

#[test]
fn constant_forward_lobe_activates_y00_y20() {
    let Some(ctx) = context_or_skip("constant_forward_lobe_activates_y00_y20") else {
        return;
    };
    // Every sample along +z with the same value: the x/y-dependent basis terms
    // cancel to near zero while the z-aligned terms Y00 (index 0), Y10 (index 2,
    // linear in z) and Y20 (index 6, quadratic in z) stay strongly activated. We
    // first assert golden parity, then the qualitative lobe shape.
    let samples: Vec<TransmittanceSample> = (0..6).map(|_| sample(0.0, 0.0, 1.0, 0.7)).collect();
    let got = run(&ctx, &samples);
    assert_coeffs(got, &samples);
    for &k in &[0usize, 2, 6] {
        assert!(
            got.c[k].abs() > 1e-2,
            "z-aligned coefficient {k} should dominate a constant +z lobe, got {}",
            got.c[k]
        );
    }
    for &k in &[1usize, 3, 4, 5, 7, 8] {
        assert!(
            got.c[k].abs() < 1e-3,
            "transverse coefficient {k} should be near zero, got {}",
            got.c[k]
        );
    }
}

#[test]
fn large_batch_matches_golden() {
    let Some(ctx) = context_or_skip("large_batch_matches_golden") else {
        return;
    };
    // 130 samples drive a long per-coefficient accumulation loop (weight
    // 4*pi / 130) over a deterministic sweep of integer-derived directions and
    // values, with every 17th slot forced non-finite to exercise the sanitiser
    // inside the loop. The nine coefficients still fit one 64-wide workgroup.
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
    assert_coeffs(run(&ctx, &samples), &samples);
}
