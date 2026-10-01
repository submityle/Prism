//! Real-device parity for the per-wavelength spectral melanin-absorption sampler
//! twin: [`GpuHairSpectrumSample`] must reproduce the `CPU` golden
//! [`reference_spectrum_sample_map`](prism_hair_gpu::spectrum_sample_map::reference_spectrum_sample_map)
//! (built on
//! [`spectrum_sample_map`](prism_render_architecture::hair::spectral_absorption::spectrum_sample_map))
//! for a batch of wavelengths sharing one `(eumelanin, pheomelanin)` pigment
//! pair, mapping each wavelength to its scalar `sigma_a` independently. The
//! suite drives exact `LUT` grid points, an interpolated midpoint, below-range
//! and above-range wavelengths clamping to the endpoints, non-finite wavelengths
//! collapsing to the short endpoint, negative and non-finite concentrations
//! collapsing to zero, eumelanin-only / pheomelanin-only / mixed pigments, the
//! empty no-op, and a large batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each `sigma_a` is a two-term interpolation multiply-add a `GPU` may fuse, so
//! every value is asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Beyond
//! matching the golden element by element, every result is asserted finite and
//! non-negative. No `sin`/`cos` appears anywhere; all inputs are explicit
//! literals or integer-derived fractions.
//!
//! Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
//! spectral sampling and `wgpu` compute dispatch; no third-party engine source
//! or derived code.

use prism_hair_gpu::spectrum_sample_map::{reference_spectrum_sample_map, GpuHairSpectrumSample};
use prism_hair_gpu::GpuContext;

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

/// Asserts a whole batch matches the `CPU` golden element by element and that
/// every result is finite and non-negative.
fn assert_batch(got: &[f32], eumelanin: f32, pheomelanin: f32, wavelengths: &[f32]) {
    let want = reference_spectrum_sample_map(eumelanin, pheomelanin, wavelengths);
    assert_eq!(
        got.len(),
        wavelengths.len(),
        "one sigma_a per queried wavelength"
    );
    assert_eq!(got.len(), want.len(), "golden length matches query length");
    for (i, (&out, &reference)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(out, reference, &format!("wavelength {i}"));
        assert!(
            out.is_finite() && out >= 0.0,
            "wavelength {i} must be finite and non-negative, got {out}"
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, eumelanin: f32, pheomelanin: f32, wavelengths: &[f32]) -> Vec<f32> {
    GpuHairSpectrumSample::new(ctx).eval(ctx, eumelanin, pheomelanin, wavelengths)
}

#[test]
fn exact_grid_points_match_golden() {
    let Some(ctx) = context_or_skip("exact_grid_points_match_golden") else {
        return;
    };
    // The band is sampled every 25nm from 380 to 730; querying exact grid points
    // hits individual table entries with no interpolation (frac = 0).
    let wavelengths = [380.0, 405.0, 430.0, 530.0, 580.0, 705.0, 730.0];
    assert_batch(&run(&ctx, 1.0, 1.0, &wavelengths), 1.0, 1.0, &wavelengths);
}

#[test]
fn interpolated_midpoints_match_golden() {
    let Some(ctx) = context_or_skip("interpolated_midpoints_match_golden") else {
        return;
    };
    // Off-grid wavelengths exercise the floor-index + linear-interpolation path
    // (frac in (0, 1)); the golden and the device twin must agree within fma
    // tolerance.
    let wavelengths = [392.5, 450.0, 517.3, 601.1, 666.6, 718.25];
    assert_batch(&run(&ctx, 0.8, 0.4, &wavelengths), 0.8, 0.4, &wavelengths);
}

#[test]
fn out_of_range_wavelengths_clamp_to_endpoints() {
    let Some(ctx) = context_or_skip("out_of_range_wavelengths_clamp_to_endpoints") else {
        return;
    };
    // Below 380 clamps to the violet endpoint, above 730 clamps to the red
    // endpoint (the table is clamped, not extrapolated), matching the golden.
    let wavelengths = [100.0, 379.999, 730.001, 2000.0];
    assert_batch(&run(&ctx, 1.2, 0.6, &wavelengths), 1.2, 0.6, &wavelengths);
}

#[test]
fn non_finite_wavelengths_map_to_short_endpoint() {
    let Some(ctx) = context_or_skip("non_finite_wavelengths_map_to_short_endpoint") else {
        return;
    };
    // NaN and +/-inf wavelengths all collapse to the 380nm endpoint on both the
    // CPU and the GPU, so each maps to the first table entry's absorption.
    let wavelengths = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 500.0];
    assert_batch(&run(&ctx, 0.9, 0.3, &wavelengths), 0.9, 0.3, &wavelengths);
}

#[test]
fn negative_and_non_finite_concentration_is_zero() {
    let Some(ctx) = context_or_skip("negative_and_non_finite_concentration_is_zero") else {
        return;
    };
    // Negative, NaN and +/-inf concentrations each sanitize to exactly 0, so the
    // whole batch reads back all-zero regardless of wavelength.
    let wavelengths = [400.0, 500.0, 600.0, 700.0];
    for (eu, pheo) in [
        (-1.0, -2.0),
        (f32::NAN, 0.5),
        (0.5, f32::NAN),
        (f32::INFINITY, f32::NEG_INFINITY),
    ] {
        let got = run(&ctx, eu, pheo, &wavelengths);
        assert_batch(&got, eu, pheo, &wavelengths);
    }
    // The all-non-finite case must be exactly zero absorption everywhere.
    let got = run(&ctx, f32::NAN, f32::NEG_INFINITY, &wavelengths);
    for (i, &v) in got.iter().enumerate() {
        assert_close(v, 0.0, &format!("non-finite concentration wavelength {i}"));
    }
}

#[test]
fn eumelanin_only_matches_golden() {
    let Some(ctx) = context_or_skip("eumelanin_only_matches_golden") else {
        return;
    };
    // Pheomelanin zeroed isolates the eumelanin spectrum (brown-black cast).
    let wavelengths = [420.0, 480.0, 555.0, 640.0];
    assert_batch(&run(&ctx, 1.5, 0.0, &wavelengths), 1.5, 0.0, &wavelengths);
}

#[test]
fn pheomelanin_only_matches_golden() {
    let Some(ctx) = context_or_skip("pheomelanin_only_matches_golden") else {
        return;
    };
    // Eumelanin zeroed isolates the pheomelanin spectrum (red-ginger cast).
    let wavelengths = [420.0, 480.0, 555.0, 640.0];
    assert_batch(&run(&ctx, 0.0, 1.1, &wavelengths), 0.0, 1.1, &wavelengths);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, 1.0, 1.0, &[]);
    assert!(got.is_empty(), "empty batch yields no samples");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 wavelengths span three 64-wide workgroups over a deterministic sweep
    // across (and beyond) the sampled band, with every 13th slot forced
    // non-finite to exercise the wavelength sanitizer across the dispatch
    // boundary.
    let mut wavelengths = Vec::new();
    for k in 0u32..130 {
        if k % 13 == 0 {
            wavelengths.push(f32::NAN);
        } else {
            // Sweep roughly 300nm..=820nm so some samples fall below 380 and
            // above 730 to exercise the endpoint clamp as well.
            wavelengths.push(300.0 + (k as f32) * 4.0);
        }
    }
    assert_batch(&run(&ctx, 0.7, 0.5, &wavelengths), 0.7, 0.5, &wavelengths);
}
