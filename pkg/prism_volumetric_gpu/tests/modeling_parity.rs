//! Real-device parity for the cloud-modelling twin: [`GpuModeling`] must
//! reproduce the `CPU` golden
//! [`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)
//! across all four [`CloudKind`]s, the full `height` band and a spread of
//! authored coverage / cloud-type / erosion controls.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The composition is entirely `saturate` / `lerp` / `remap` / Hermite
//! `smoothstep`, transcendental-free, so `CPU` and `GPU` evaluate the identical
//! arithmetic. Values are asserted to within `abs_diff < 1e-6` or
//! `rel_diff < 1e-5` — far tighter than any physically meaningful density
//! difference and enough to fail a wrong port (a swapped profile edge, a missing
//! saturation, an inverted coverage `remap`). The scenes also assert the
//! range (`0..=1`), coverage monotonicity and erosion non-brightening
//! contracts, so a degenerate constant kernel could not pass.
//!
//! Provenance: standard `Nubis`-style procedural cloud modelling; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::modeling::compose_from_modeling;
use prism_render_architecture::volumetric::{CloudKind, CloudModeling};
use prism_volumetric_gpu::{GpuContext, GpuModeling, ModelingQuery};

/// The four kinds, for exhaustive per-kind scans.
const KINDS: [CloudKind; 4] = [
    CloudKind::Cumulus,
    CloudKind::Stratus,
    CloudKind::Cirrus,
    CloudKind::Cumulonimbus,
];

/// Builds an authored modelling contract from the shape-relevant controls; the
/// frequency/curl fields do not affect [`compose_from_modeling`], so they take
/// representative constants.
fn modeling(coverage: f32, cloud_type: f32, detail_erosion: f32) -> CloudModeling {
    CloudModeling {
        coverage,
        cloud_type,
        base_frequency: 4.0,
        detail_frequency: 16.0,
        detail_erosion,
        curl_strength: 12.0,
    }
}

/// Asserts every `gpu` density matches the `CPU` golden to within the
/// documented tolerance.
fn assert_parity(queries: &[ModelingQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one density per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = compose_from_modeling(
            q.modeling,
            q.kind,
            q.base_low,
            q.base_high,
            q.height_fraction,
            q.weather_coverage,
            q.detail_noise,
        );
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "density mismatch for query {q:?}: gpu {got}, cpu {exp} (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu density must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_modeling_matches_cpu_golden_across_kinds_and_height() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping modeling parity: no wgpu adapter on this host");
        return;
    };
    let gpu_modeling = GpuModeling::new(&ctx);

    // Every kind swept across the full height band on a partly-cloudy contract
    // with meaningful base shape and detail.
    let mut queries: Vec<ModelingQuery> = Vec::new();
    for kind in KINDS {
        for step in 0..=10 {
            let hf = step as f32 / 10.0;
            queries.push(ModelingQuery {
                modeling: modeling(0.6, 0.5, 0.35),
                kind,
                base_low: 0.55,
                base_high: 0.7,
                height_fraction: hf,
                weather_coverage: 0.8,
                detail_noise: 0.4,
            });
        }
    }

    let gpu = gpu_modeling.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Each kind's vertical profile must vanish at both band edges (height
    // fraction 0.0 and 1.0), proving a real height gradient ran rather than a
    // constant.
    for (k, kind) in KINDS.iter().enumerate() {
        let base = k * 11;
        assert!(
            gpu[base] <= 1e-6,
            "{kind:?} density must vanish at the band floor: {}",
            gpu[base]
        );
        assert!(
            gpu[base + 10] <= 1e-6,
            "{kind:?} density must vanish at the band ceiling: {}",
            gpu[base + 10]
        );
    }
}

#[test]
fn gpu_modeling_is_monotonic_in_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_modeling = GpuModeling::new(&ctx);

    // Same cell, rising coverage: density must be non-decreasing and match the
    // reference, exercising the coverage remap direction.
    let queries: Vec<ModelingQuery> = (0..=10)
        .map(|step| ModelingQuery {
            modeling: modeling(step as f32 / 10.0, 0.5, 0.3),
            kind: CloudKind::Cumulus,
            base_low: 0.5,
            base_high: 0.6,
            height_fraction: 0.5,
            weather_coverage: 1.0,
            detail_noise: 0.2,
        })
        .collect();

    let gpu = gpu_modeling.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
    for w in gpu.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "density must not shrink as coverage grows: {} -> {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn gpu_modeling_is_non_brightening_in_erosion() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_modeling = GpuModeling::new(&ctx);

    // Same cell, rising erosion strength with fixed detail: density must be
    // non-increasing (erosion only eats, never brightens) and match the golden.
    let queries: Vec<ModelingQuery> = (0..=10)
        .map(|step| ModelingQuery {
            modeling: modeling(0.7, 0.5, step as f32 / 10.0),
            kind: CloudKind::Cumulonimbus,
            base_low: 0.6,
            base_high: 0.8,
            height_fraction: 0.5,
            weather_coverage: 0.9,
            detail_noise: 0.7,
        })
        .collect();

    let gpu = gpu_modeling.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "density must not brighten as erosion grows: {} -> {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn gpu_modeling_matches_golden_for_out_of_range_inputs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_modeling = GpuModeling::new(&ctx);

    // Out-of-range inputs the reference saturates/clamps: negative and
    // above-one shapes, coverage and height fraction, plus every kind.
    let queries: Vec<ModelingQuery> = KINDS
        .iter()
        .map(|&kind| ModelingQuery {
            modeling: modeling(1.5, -0.3, 2.0),
            kind,
            base_low: -0.4,
            base_high: 1.7,
            height_fraction: 1.9,
            weather_coverage: -0.2,
            detail_noise: 1.3,
        })
        .collect();

    let gpu = gpu_modeling.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_modeling = GpuModeling::new(&ctx);
    let out = gpu_modeling.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no densities");
}
