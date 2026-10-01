//! Real-device parity for the `Hero-wavelength` stratified spectral-sampling
//! twin: [`GpuHairHeroWavelengths`] must reproduce the `CPU` golden pairing of
//! [`hero_wavelengths`](prism_render_architecture::hair::spectral_absorption::hero_wavelengths)
//! with
//! [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a)
//! (re-exported as
//! [`reference_hero_sample`](prism_hair_gpu::hero_wavelengths::reference_hero_sample))
//! for a batch of `(u, rotate, eumelanin, pheomelanin)` samples, deriving four
//! quarter-band cyclic wavelengths per sample and folding each into `sigma_a`.
//! The suite drives a regular stratified sample, a coordinate that wraps the
//! red->violet boundary, non-finite / out-of-range coordinates folding into
//! `[0, 1)`, negative / non-finite concentrations collapsing to zero,
//! eumelanin-only / pheomelanin-only / mixed pigments, the empty no-op, and a
//! large batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each wavelength is a wrap plus one multiply-add and each `sigma_a` a two-term
//! interpolation multiply-add a `GPU` may fuse, so every value is asserted
//! within `abs_diff < 1e-4` or `rel_diff < 1e-3`. Beyond matching the golden,
//! every wavelength is asserted finite and inside `[380, 730]`nm and every
//! `sigma_a` finite and non-negative.
//!
//! Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
//! hero-wavelength spectral sampling and `wgpu` compute dispatch; no third-party
//! engine source or derived code.

use prism_hair_gpu::hero_wavelengths::{reference_hero_sample, GpuHairHeroWavelengths, HeroSample};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::spectral_absorption::{SPECTRUM_MAX_NM, SPECTRUM_MIN_NM};

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

/// Asserts one evaluated sample matches the `CPU` golden and that every
/// wavelength is finite and in `[380, 730]`nm with a finite, non-negative
/// `sigma_a`.
fn assert_sample(got: &HeroSample, u: f32, rotate: f32, eumelanin: f32, pheomelanin: f32) {
    let want = reference_hero_sample(u, rotate, eumelanin, pheomelanin);
    for j in 0..4 {
        assert_close(got.lambdas[j], want.lambdas[j], "lambda");
        assert_close(got.sigma_a[j], want.sigma_a[j], "sigma_a");
        assert!(
            got.lambdas[j].is_finite(),
            "wavelength must be finite, got {}",
            got.lambdas[j]
        );
        assert!(
            got.lambdas[j] >= SPECTRUM_MIN_NM && got.lambdas[j] <= SPECTRUM_MAX_NM,
            "wavelength {} out of band [{SPECTRUM_MIN_NM}, {SPECTRUM_MAX_NM}]",
            got.lambdas[j]
        );
        assert!(
            got.sigma_a[j].is_finite() && got.sigma_a[j] >= 0.0,
            "sigma_a must be finite and non-negative, got {}",
            got.sigma_a[j]
        );
    }
}

/// Asserts a whole batch matches the golden sample by sample.
fn assert_batch(got: &[HeroSample], samples: &[(f32, f32, f32, f32)]) {
    assert_eq!(got.len(), samples.len(), "one HeroSample per input");
    for (sample, &(u, rotate, eu, pheo)) in got.iter().zip(samples.iter()) {
        assert_sample(sample, u, rotate, eu, pheo);
    }
}

#[test]
fn regular_sample_matches_golden() {
    let Some(ctx) = context_or_skip("regular_sample_matches_golden") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let samples = [(0.3, 0.1, 1.2, 0.4)];
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
    // The hero wavelength is the only one driven solely by `base`; a mid-band
    // stratified coord must land it strictly inside the band (non-trivial).
    assert!(got[0].lambdas[0] > SPECTRUM_MIN_NM && got[0].lambdas[0] < SPECTRUM_MAX_NM);
}

#[test]
fn wavelengths_wrap_the_band_boundary() {
    let Some(ctx) = context_or_skip("wavelengths_wrap_the_band_boundary") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    // base near 0.9 forces companions at 0.9, 1.15, 1.4, 1.65 — three of which
    // wrap past the red end back through violet.
    let samples = [(0.6, 0.3, 0.9, 0.6)];
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
}

#[test]
fn repeat_run_is_bit_identical() {
    let Some(ctx) = context_or_skip("repeat_run_is_bit_identical") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let samples = [(0.42, 0.17, 1.5, 0.3)];
    let a = twin.eval(&ctx, &samples);
    let b = twin.eval(&ctx, &samples);
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(b.iter()) {
        for j in 0..4 {
            assert_eq!(sa.lambdas[j].to_bits(), sb.lambdas[j].to_bits());
            assert_eq!(sa.sigma_a[j].to_bits(), sb.sigma_a[j].to_bits());
        }
    }
}

#[test]
fn non_finite_coordinates_fold_into_unit_range() {
    let Some(ctx) = context_or_skip("non_finite_coordinates_fold_into_unit_range") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let samples = [
        (f32::NAN, f32::INFINITY, 1.0, 0.5),
        (f32::NEG_INFINITY, f32::NAN, 0.8, 0.2),
        (12.7, -3.4, 1.1, 0.6),
    ];
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
    // NaN/inf coords collapse to base 0 -> hero wavelength is the violet
    // endpoint exactly.
    assert_close(got[0].lambdas[0], SPECTRUM_MIN_NM, "folded hero lambda");
}

#[test]
fn bad_concentrations_clamp_to_zero_absorption() {
    let Some(ctx) = context_or_skip("bad_concentrations_clamp_to_zero_absorption") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let samples = [(0.5, 0.0, f32::NAN, f32::INFINITY), (0.25, 0.1, -4.0, -2.0)];
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
    for sample in &got {
        for j in 0..4 {
            assert_close(sample.sigma_a[j], 0.0, "clamped sigma_a");
        }
    }
}

#[test]
fn single_pigment_channels_match_golden() {
    let Some(ctx) = context_or_skip("single_pigment_channels_match_golden") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let samples = [
        (0.35, 0.2, 1.4, 0.0), // eumelanin only
        (0.35, 0.2, 0.0, 1.4), // pheomelanin only
    ];
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
    // Both channels still absorb somewhere in the band (non-trivial sigma).
    assert!(got[0].sigma_a.iter().any(|&s| s > 0.0));
    assert!(got[1].sigma_a.iter().any(|&s| s > 0.0));
}

#[test]
fn empty_batch_is_empty_without_dispatch() {
    let Some(ctx) = context_or_skip("empty_batch_is_empty_without_dispatch") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let got = twin.eval(&ctx, &[]);
    assert!(got.is_empty());
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    let twin = GpuHairHeroWavelengths::new(&ctx);
    let mut samples = Vec::with_capacity(130);
    let mut i = 0usize;
    while i < 130 {
        let fi = i as f32;
        let u = fi / 130.0;
        let rotate = (fi * 0.013) % 1.0;
        let eu = 0.2 + (fi * 0.01);
        let pheo = (fi * 0.007) % 0.9;
        // Sprinkle non-finite inputs to exercise the guards across the boundary.
        let (u, eu) = if i.is_multiple_of(13) {
            (f32::NAN, -eu)
        } else {
            (u, eu)
        };
        samples.push((u, rotate, eu, pheo));
        i += 1;
    }
    let got = twin.eval(&ctx, &samples);
    assert_batch(&got, &samples);
}
