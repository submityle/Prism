//! Real-device parity for the precipitation-classification twin:
//! [`GpuClassifyPrecip`] must reproduce the `CPU` golden
//! [`classify_precip`](prism_render_architecture::volumetric::weather::classify_precip)
//! across a deterministic grid of precip / coverage channels, temperatures and
//! cloud kinds, including the dry/wet trigger, the freezing edge and the
//! Cumulonimbus gain.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The phase is asserted to match the `CPU` golden exactly and the intensity
//! within a tight absolute tolerance. The test builds the `WeatherSample`
//! through [`WeatherSample::from_rgba`] with the same in-range channels fed to
//! the GPU, so both sides see identical inputs.
//!
//! Provenance: standard threshold-cascade precipitation classification; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::weather::{classify_precip, PrecipKind};
use prism_render_architecture::volumetric::CloudKind;
use prism_render_architecture::volumetric::WeatherSample;
use prism_volumetric_gpu::{ClassifyPrecipQuery, GpuClassifyPrecip, GpuContext};

/// Absolute tolerance for the intensity: one saturated product, so agreement is
/// to the last few ULPs.
const TOL: f32 = 1e-6;

const KINDS: [CloudKind; 4] = [
    CloudKind::Cumulus,
    CloudKind::Stratus,
    CloudKind::Cirrus,
    CloudKind::Cumulonimbus,
];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_classify_precip_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping classify-precip parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuClassifyPrecip::new(&ctx);

    // Precip channel straddling the 0.5 trigger, coverage across the range,
    // temperatures straddling the freezing edge (exactly 0.0 must be Snow).
    let precips = [0.0_f32, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0];
    let coverages = [0.0_f32, 0.2, 0.5, 0.8, 1.0];
    let temperatures = [-10.0_f32, -0.5, 0.0, 0.5, 20.0];

    let mut queries: Vec<ClassifyPrecipQuery> = Vec::new();
    for &kind in &KINDS {
        for &precipitation in &precips {
            for &coverage in &coverages {
                for &temperature in &temperatures {
                    queries.push(ClassifyPrecipQuery {
                        precipitation,
                        coverage,
                        temperature,
                        kind,
                    });
                }
            }
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");

    for (i, q) in queries.iter().enumerate() {
        // Same in-range channels on both sides (from_rgba is identity here).
        let sample = WeatherSample::from_rgba(q.coverage, 0.5, q.precipitation, 0.0);
        let exp = classify_precip(&sample, q.kind, q.temperature);

        assert_eq!(
            gpu[i].kind, exp.kind,
            "phase mismatch for query {i} (precip {}, cov {}, temp {}, kind {:?}): \
             gpu {:?}, cpu {:?}",
            q.precipitation, q.coverage, q.temperature, q.kind, gpu[i].kind, exp.kind
        );
        assert!(
            (gpu[i].intensity - exp.intensity).abs() <= TOL,
            "intensity mismatch for query {i}: gpu {}, cpu {}, |diff| {}",
            gpu[i].intensity,
            exp.intensity,
            (gpu[i].intensity - exp.intensity).abs()
        );

        // Intensity stays in `0..=1`; None phase carries exactly zero intensity.
        assert!(
            gpu[i].intensity >= -TOL && gpu[i].intensity <= 1.0 + TOL,
            "intensity out of [0,1] for query {i}: {}",
            gpu[i].intensity
        );
        if gpu[i].kind == PrecipKind::None {
            assert_eq!(
                gpu[i].intensity, 0.0,
                "None phase must carry zero intensity for query {i}"
            );
        }
    }

    // Freezing edge exactly at 0.0 C must classify as Snow (deterministic).
    let edge = ClassifyPrecipQuery {
        precipitation: 0.9,
        coverage: 1.0,
        temperature: 0.0,
        kind: CloudKind::Cumulonimbus,
    };
    let out = gpu_kernel.eval(&ctx, &[edge]);
    assert_eq!(
        out[0].kind,
        PrecipKind::Snow,
        "temperature == 0 C must be Snow"
    );

    // Cumulonimbus (full gain) precipitates harder than a non-capable kind at
    // the same wet inputs.
    let wet_cb = ClassifyPrecipQuery {
        precipitation: 0.8,
        coverage: 1.0,
        temperature: 10.0,
        kind: CloudKind::Cumulonimbus,
    };
    let wet_cu = ClassifyPrecipQuery {
        kind: CloudKind::Cumulus,
        ..wet_cb
    };
    let pair = gpu_kernel.eval(&ctx, &[wet_cb, wet_cu]);
    assert!(
        pair[0].intensity > pair[1].intensity,
        "Cumulonimbus gain must exceed the default kind gain: {} vs {}",
        pair[0].intensity,
        pair[1].intensity
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuClassifyPrecip::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
