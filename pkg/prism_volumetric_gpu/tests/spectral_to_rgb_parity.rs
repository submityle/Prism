//! Real-device parity for the spectral-to-RGB twin: [`GpuSpectralToRgb`] must
//! reproduce the `CPU` golden
//! [`spectral_to_rgb`](prism_render_architecture::volumetric::spectral::spectral_to_rgb)
//! for spectral band distributions of arbitrary bucket count, including the
//! three-bucket `RGB` case, longer variable-length spectra, all-zero (uniform)
//! sets, and the empty set.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The per-band Gaussian response uses the same hand-rolled `exp_approx` the
//! reference uses, and the `lerp` is expanded to the same closed form, so `CPU`
//! and `GPU` evaluate identical algebra. Values are asserted per channel to
//! within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong
//! port (a dropped normalisation, a wrong centre wavelength). The scenes also
//! assert every channel is non-negative and that the channels sum to the band
//! weight total (one for a normalised, non-empty set; zero for an empty one),
//! so a degenerate kernel could not pass.
//!
//! Provenance: standard CIE-flavoured spectral-to-RGB collapse; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::spectral::{spectral_to_rgb, SpectralBands};
use prism_volumetric_gpu::{GpuContext, GpuSpectralToRgb, SpectralRgb};

/// Asserts every `gpu` triple matches the `CPU` golden per channel to within
/// the documented tolerance, stays non-negative, and conserves weight (its
/// channels sum to the band's weight total).
fn assert_parity(bands: &[SpectralBands], gpu: &[SpectralRgb]) {
    assert_eq!(gpu.len(), bands.len(), "one RGB triple per band set");
    for (i, band) in bands.iter().enumerate() {
        let exp = spectral_to_rgb(band);
        let got = gpu[i];
        for (channel, (g, e)) in [(got.r, exp.x), (got.g, exp.y), (got.b, exp.z)]
            .into_iter()
            .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "spectral-to-RGB mismatch for band {i} channel {channel}: \
                 gpu {g}, cpu {e} (abs {abs_diff}, rel {rel_diff})"
            );
            assert!(
                g >= 0.0,
                "gpu spectral-to-RGB channel {channel} of band {i} must be \
                 non-negative: {g}"
            );
        }
        let channel_sum = got.r + got.g + got.b;
        let weight_sum = band.sum();
        assert!(
            (channel_sum - weight_sum).abs() < 1e-5,
            "gpu channels of band {i} must sum to the band weight total: \
             channels {channel_sum}, weights {weight_sum}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_spectral_to_rgb_matches_cpu_golden_across_distributions() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping spectral-to-RGB parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSpectralToRgb::new(&ctx);

    // A deterministic mix of bucket counts and shapes: pure RGB primaries and
    // mixes, longer variable-length spectra, an all-zero (uniform) set, and a
    // single-bucket set.
    let mut bands: Vec<SpectralBands> = vec![
        SpectralBands::from_rgb(1.0, 0.0, 0.0),
        SpectralBands::from_rgb(0.0, 1.0, 0.0),
        SpectralBands::from_rgb(0.0, 0.0, 1.0),
        SpectralBands::from_rgb(2.0, 1.0, 1.0),
        SpectralBands::from_rgb(0.0, 0.0, 0.0),
        SpectralBands::new(vec![0.2, 0.5, 0.1, 0.7, 0.3]),
        SpectralBands::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        SpectralBands::new(vec![0.0, 0.0, 0.0, 0.0]),
        SpectralBands::new(vec![1.0]),
    ];
    // A deterministic sweep of monotone ramps of increasing bucket count.
    for n in 2u32..=16 {
        let weights: Vec<f32> = (0..n).map(|k| 0.1 + (k as f32) * 0.05).collect();
        bands.push(SpectralBands::new(weights));
    }

    let gpu = gpu_kernel.eval(&ctx, &bands);
    assert_parity(&bands, &gpu);
}

#[test]
fn empty_band_slice_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSpectralToRgb::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty band slice yields no values");
}

#[test]
fn empty_band_sets_map_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSpectralToRgb::new(&ctx);

    // Empty band sets (no buckets) must collapse to the zero vector, and they
    // must coexist with non-empty ones in the same flattened dispatch.
    let bands = vec![
        SpectralBands::new(vec![]),
        SpectralBands::from_rgb(0.3, 0.4, 0.3),
        SpectralBands::new(vec![]),
    ];

    let gpu = gpu_kernel.eval(&ctx, &bands);
    assert_parity(&bands, &gpu);

    for idx in [0usize, 2] {
        let z = gpu[idx];
        assert!(
            z.r.abs() < 1e-6 && z.g.abs() < 1e-6 && z.b.abs() < 1e-6,
            "an empty band set maps to zero: {z:?}"
        );
    }
}
