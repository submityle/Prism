//! Cross-module integration tests for the volumetric subsystem.
//!
//! Every other file in `volumetric/` ships its own unit tests for the property
//! obligations in design-doc section 16. This module instead exercises the
//! *seams* between modules end to end, asserting that data produced by one
//! stage flows into the next without violating the subsystem-wide invariants:
//!
//! - `noise` -> `modeling`: procedural noise feeds density composition and the
//!   result stays in `0..=1` and is deterministic.
//! - `weather` -> `modeling`: sampled coverage drives density monotonically.
//! - `modeling`/`noise` -> `raymarch` -> `scatter`: a full density field is
//!   integrated with an `HG` phase, conserving energy (`transmittance` in
//!   `0..=1`, non-increasing with distance; scattered radiance finite and
//!   non-negative).
//! - `coupling` -> `raymarch`: carving holes only ever *raises* transmittance.
//! - `avsm` -> `raymarch`: the self-shadow `transmittance` curve is a valid
//!   light-visibility function for the march.
//! - `multiscatter`, `fog`, `atmosphere`: energy stays bounded when composed
//!   with the ray-march result.
//! - `reference` -> `multiscatter`: the Monte-Carlo single-scatter oracle and
//!   the closed form agree, and folding the LUT energy gain into the resolve
//!   composition `single * (1 + gain)` only ever adds bounded energy (never
//!   removes it, never more than doubles it) and stays monotone in optical
//!   depth and albedo.
//! - `cloud_lod`, `temporal`, `budget`: scheduling and selection are
//!   deterministic and stay within their declared limits.
//!
//! These tests only consume the public contracts of each module, so they also
//! act as a compile-time guard that the cross-module API surface stays stable.

use super::atmosphere::{aerial_perspective_weight, blend_with_atmosphere};
use super::avsm::AvsmCurve;
use super::budget::{plan_volumetric, VolumetricJobKind, VolumetricJobRequest};
use super::cloud_lod::{bin_by_distance, select_lod, CloudLodThresholds};
use super::coupling::{apply_carve, density_delta, CarveBrush};
use super::fog::{fog_transmittance, height_fog_density, HeightFogParams};
use super::math::{saturate, EPS};
use super::modeling::compose_from_modeling;
use super::multiscatter::MultiScatterLut;
use super::noise::{perlin_worley, worley_fbm};
use super::raymarch::{march, RaymarchConfig};
use super::reference::{analytic_single_scatter, single_scatter_reference};
use super::scatter::{hg_phase, octave_scatter, OctaveParams};
use super::temporal::{active_pixel, clamp_history, UpscaleMode};
use super::weather::{WeatherField, WindField};
use super::{
    CloudKind, CloudLayerHandle, CloudModeling, Vec2, Vec3, VolumetricBudget, WeatherMapHandle,
    WeatherSample,
};

/// The four authored cloud kinds, scanned exhaustively in the seam tests.
const KINDS: [CloudKind; 4] = [
    CloudKind::Cumulus,
    CloudKind::Stratus,
    CloudKind::Cirrus,
    CloudKind::Cumulonimbus,
];

/// A representative authored `CloudModeling` preset used across the tests.
fn sample_modeling() -> CloudModeling {
    CloudModeling {
        coverage: 0.55,
        cloud_type: 0.6,
        base_frequency: 0.9,
        detail_frequency: 3.5,
        detail_erosion: 0.4,
        curl_strength: 12.0,
    }
}

/// Composes a single normalized density from the `noise` and `modeling`
/// modules at world position `p`, given a sampled `weather_coverage`.
///
/// This mirrors the real density pipeline: low-frequency `perlin_worley` forms
/// the base shape, higher-frequency `worley_fbm` supplies the detail-erosion
/// noise, and `p.y` (assumed already in `0..=1`) acts as the height fraction.
fn field_density(
    modeling: CloudModeling,
    kind: CloudKind,
    p: Vec3,
    weather_coverage: f32,
    seed: u32,
) -> f32 {
    let base_low = perlin_worley(p.scale(modeling.base_frequency), seed);
    let base_high = perlin_worley(p.scale(modeling.base_frequency * 1.7), seed ^ 0x9E37_79B9);
    let detail = worley_fbm(p.scale(modeling.detail_frequency), seed ^ 0x0123_4567, 3);
    let height_fraction = saturate(p.y);
    compose_from_modeling(
        modeling,
        kind,
        base_low,
        base_high,
        height_fraction,
        weather_coverage,
        detail,
    )
}

/// Maps a normalized density to `(sigma_t, sigma_s)` extinction/scattering
/// coefficients using a fixed albedo, for the ray-march seam tests.
fn sigma_from_density(density: f32) -> (f32, f32) {
    let sigma_t = density * 4.0;
    let sigma_s = sigma_t * 0.9;
    (sigma_t, sigma_s)
}

#[test]
fn noise_feeds_modeling_density_in_unit_range_and_is_deterministic() {
    let modeling = sample_modeling();
    for kind in KINDS {
        let mut z = 0u32;
        while z < 6 {
            let p = Vec3::new(z as f32 * 0.37, (z as f32 * 0.19) % 1.0, z as f32 * 0.53);
            let d0 = field_density(modeling, kind, p, 0.5, 1337);
            let d1 = field_density(modeling, kind, p, 0.5, 1337);
            assert_eq!(d0.to_bits(), d1.to_bits(), "density must be deterministic");
            assert!(
                (0.0..=1.0).contains(&d0),
                "density out of range for {kind:?}: {d0}"
            );
            z += 1;
        }
    }
}

#[test]
fn weather_coverage_drives_modeling_monotonically() {
    let modeling = sample_modeling();
    let p = Vec3::new(2.0, 0.5, 3.0);
    let mut prev = -1.0_f32;
    for step in 0..=10 {
        let weather_coverage = step as f32 / 10.0;
        let d = field_density(modeling, CloudKind::Cumulus, p, weather_coverage, 42);
        assert!(
            d + 1.0e-4 >= prev,
            "density must be non-decreasing in weather coverage: {d} < {prev}"
        );
        prev = d;
    }
}

#[test]
fn weather_field_sampling_feeds_density_pipeline() {
    // A 2x2 coverage tile: sampling anywhere yields a coverage in range that
    // the density pipeline accepts.
    let handle = WeatherMapHandle(7);
    let cells = alloc::vec![
        WeatherSample::from_rgba(0.2, 0.6, 0.0, 0.1),
        WeatherSample::from_rgba(0.8, 0.6, 0.0, 0.1),
        WeatherSample::from_rgba(0.5, 0.6, 0.0, 0.1),
        WeatherSample::from_rgba(0.9, 0.6, 0.0, 0.1),
    ];
    let field = WeatherField::from_cells(handle, 2, 2, cells).expect("valid 2x2 tile");
    let modeling = sample_modeling();
    for i in 0..=8 {
        let u = i as f32 / 8.0;
        let sample = field.sample_bilinear(u, 0.5);
        assert!((0.0..=1.0).contains(&sample.coverage));
        let d = field_density(
            modeling,
            CloudKind::Stratus,
            Vec3::new(u * 4.0, 0.5, 1.0),
            sample.coverage,
            9,
        );
        assert!((0.0..=1.0).contains(&d));
    }
}

#[test]
fn density_field_through_raymarch_conserves_energy() {
    let modeling = sample_modeling();
    let cfg = RaymarchConfig::default();
    let phase = hg_phase(0.6, 0.4);
    let density_fn = |t: f32| {
        let p = Vec3::new(t * 0.05, saturate(t * 0.01), 1.0);
        field_density(modeling, CloudKind::Cumulus, p, 0.7, 555)
    };
    let state = march(density_fn, sigma_from_density, phase, |_| 1.0, 400.0, cfg);
    assert!(
        (0.0..=1.0).contains(&state.transmittance),
        "transmittance out of range: {}",
        state.transmittance
    );
    assert!(state.scattered.is_finite() && state.scattered >= 0.0);
    assert!(state.optical_depth >= 0.0);
    assert!(state.steps_taken <= cfg.max_steps);
}

#[test]
fn raymarch_transmittance_is_non_increasing_with_distance() {
    let cfg = RaymarchConfig::default();
    // A uniformly dense column: farther marches can only occlude more.
    let density_fn = |_t: f32| 0.8_f32;
    let phase = hg_phase(0.3, 0.5);
    let mut prev = 2.0_f32;
    for steps in 1..=8 {
        let distance = steps as f32 * 20.0;
        let state = march(
            density_fn,
            sigma_from_density,
            phase,
            |_| 1.0,
            distance,
            cfg,
        );
        assert!(
            state.transmittance <= prev + 1.0e-6,
            "transmittance rose with distance: {} > {prev}",
            state.transmittance
        );
        prev = state.transmittance;
    }
}

#[test]
fn carving_holes_only_raises_transmittance() {
    // Uniform fine steps with no early-termination cutoff, so the march is a
    // faithful Riemann sum of the optical depth: lower density everywhere then
    // strictly implies less extinction (a fair apples-to-apples comparison
    // that is not confounded by adaptive-step / early-out discretization).
    let cfg = RaymarchConfig {
        base_step: 2.0,
        max_step: 2.0,
        min_step: 2.0,
        density_threshold: 1.0e-4,
        transmittance_cutoff: 0.0,
        max_steps: 1000,
        powder_strength: 0.0,
    };
    let phase = hg_phase(0.2, 0.5);
    let base_fn = |_t: f32| 0.5_f32;
    let brush = CarveBrush {
        center: Vec3::new(100.0, 0.0, 0.0),
        radius: 400.0,
        strength: 0.8,
    };
    let carved_fn = |t: f32| {
        let pos = Vec3::new(t, 0.0, 0.0);
        apply_carve(0.5, density_delta(pos, brush))
    };
    let base = march(base_fn, sigma_from_density, phase, |_| 1.0, 200.0, cfg);
    let carved = march(carved_fn, sigma_from_density, phase, |_| 1.0, 200.0, cfg);
    // Carving only removes density, so extinction can only fall.
    assert!(
        carved.optical_depth <= base.optical_depth + 1.0e-6,
        "carving must not raise optical depth: {} > {}",
        carved.optical_depth,
        base.optical_depth
    );
    assert!(
        carved.transmittance + 1.0e-6 >= base.transmittance,
        "carving must not reduce transmittance: {} < {}",
        carved.transmittance,
        base.transmittance
    );
}

#[test]
fn avsm_curve_is_a_valid_raymarch_light_function() {
    let mut curve = AvsmCurve::new(8);
    // Insert monotonically deepening extinction segments.
    let mut depth = 10.0_f32;
    while depth <= 80.0 {
        curve.insert(depth, 0.15);
        depth += 10.0;
    }
    // Light visibility is monotone non-increasing with depth.
    let mut prev = 2.0_f32;
    for d in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0] {
        let t = curve.transmittance_at(d);
        assert!(
            (0.0..=1.0).contains(&t),
            "avsm transmittance out of range: {t}"
        );
        assert!(t <= prev + 1.0e-6, "avsm transmittance rose: {t} > {prev}");
        prev = t;
    }
    // Feed it into a march as the light visibility function.
    let cfg = RaymarchConfig::default();
    let phase = hg_phase(0.5, 0.4);
    let state = march(
        |_t| 0.6,
        sigma_from_density,
        phase,
        |t| curve.transmittance_at(t),
        200.0,
        cfg,
    );
    assert!((0.0..=1.0).contains(&state.transmittance));
    assert!(state.scattered.is_finite() && state.scattered >= 0.0);
}

#[test]
fn multiscatter_lut_never_amplifies_and_octaves_decay() {
    let lut = MultiScatterLut::build_energy_gain([8, 8, 8], OctaveParams::DEFAULT);
    for &cos in &[-1.0_f32, -0.3, 0.0, 0.4, 1.0] {
        for &depth in &[0.0_f32, 1.5, 5.0] {
            for &albedo in &[0.0_f32, 0.5, 1.0] {
                let g = lut.sample(cos, depth, albedo);
                assert!(
                    (0.0..=1.0).contains(&g),
                    "multiscatter gain out of range: {g}"
                );
            }
        }
    }
    // Octave scattering energy is non-increasing in the octave index.
    let mut prev = f32::INFINITY;
    for i in 0..5 {
        let (sigma_s, _sigma_t, _g) = octave_scatter(1.0, 1.0, 0.8, i, OctaveParams::DEFAULT);
        assert!(
            sigma_s <= prev + 1.0e-6,
            "octave energy grew: {sigma_s} > {prev}"
        );
        prev = sigma_s;
    }
}

#[test]
fn reference_single_scatter_matches_multiscatter_composition_bounded() {
    // Shared LUT and octave-derived forward anisotropy, so the phase used for
    // the reference oracle is the same lobe the multi-scatter table is built
    // around.
    let lut = MultiScatterLut::build_energy_gain([16, 16, 16], OctaveParams::DEFAULT);
    let (_ss0, _st0, g) = octave_scatter(1.0, 1.0, 0.7, 0, OctaveParams::DEFAULT);
    let cos = 0.5_f32;
    let phase_value = hg_phase(cos, g);
    let phase = |_t: f32| phase_value;
    // Unit extinction keeps optical_depth == distance, so the LUT depth axis
    // lines up exactly with the reference march length.
    let sigma_t = 1.0_f32;
    let light = 1.0_f32;

    // (1) reference oracle self-consistency: the Monte-Carlo single-scatter
    //     estimator converges to the closed form for the octave phase.
    for &(albedo, distance, seed) in &[
        (0.4_f32, 2.0_f32, 11_u32),
        (0.8, 4.0, 4242),
        (1.0, 6.0, 909),
    ] {
        let sigma_s = albedo * sigma_t;
        let analytic = analytic_single_scatter(sigma_t, sigma_s, phase_value, light, distance);
        let mc = single_scatter_reference(sigma_t, sigma_s, light, phase, distance, seed, 40_000);
        assert!(analytic > 0.0, "analytic single scatter should be positive");
        assert!(
            (mc - analytic).abs() < 0.02,
            "reference MC did not track closed form: {mc} vs {analytic}"
        );

        // (2) bounded parity: folding the LUT gain into the resolve composition
        //     `single * (1 + gain)` only ever adds energy (gain >= 0) and never
        //     more than doubles it (gain <= 1). This is the design-doc section 5
        //     "reference vs multiscatter error is bounded" obligation.
        let optical_depth = sigma_t * distance;
        let gain = lut.sample(cos, optical_depth, albedo);
        assert!(
            (0.0..=1.0).contains(&gain),
            "LUT gain escaped [0,1]: {gain}"
        );
        let resolved = analytic * (1.0 + gain);
        assert!(
            resolved >= analytic - EPS,
            "multiscatter removed energy: {resolved} < {analytic}"
        );
        assert!(
            resolved <= 2.0 * analytic + EPS,
            "multiscatter amplified past 2x single scatter: {resolved} > {}",
            2.0 * analytic
        );
    }

    // (3) monotone in optical depth: at fixed albedo, a longer march raises
    //     both the single-scatter integral and the LUT gain, so the composed
    //     radiance is non-decreasing.
    let albedo = 0.7_f32;
    let sigma_s = albedo * sigma_t;
    let mut prev = f32::NEG_INFINITY;
    for &distance in &[0.5_f32, 1.0, 2.0, 4.0, 7.0] {
        let analytic = analytic_single_scatter(sigma_t, sigma_s, phase_value, light, distance);
        let gain = lut.sample(cos, sigma_t * distance, albedo);
        let resolved = analytic * (1.0 + gain);
        assert!(
            resolved >= prev - EPS,
            "composed radiance dropped with optical depth: {resolved} < {prev}"
        );
        prev = resolved;
    }

    // (4) monotone in albedo: at fixed march length, a brighter medium raises
    //     both the single-scatter coefficient and the LUT gain, so the composed
    //     radiance is non-decreasing.
    let distance = 3.0_f32;
    let optical_depth = sigma_t * distance;
    let mut prev = f32::NEG_INFINITY;
    for &albedo in &[0.1_f32, 0.3, 0.6, 0.85, 1.0] {
        let sigma_s = albedo * sigma_t;
        let analytic = analytic_single_scatter(sigma_t, sigma_s, phase_value, light, distance);
        let gain = lut.sample(cos, optical_depth, albedo);
        let resolved = analytic * (1.0 + gain);
        assert!(
            resolved >= prev - EPS,
            "composed radiance dropped with albedo: {resolved} < {prev}"
        );
        prev = resolved;
    }
}

#[test]
fn fog_and_cloud_transmittance_compose_bounded() {
    let params = HeightFogParams {
        density_at_sea_level: 0.02,
        falloff: 0.1,
        max_height: 500.0,
    };
    let cfg = RaymarchConfig::default();
    let cloud = march(
        |_t| 0.5,
        sigma_from_density,
        hg_phase(0.4, 0.4),
        |_| 1.0,
        150.0,
        cfg,
    );
    for altitude in [0.0_f32, 50.0, 200.0, 600.0] {
        let density = height_fog_density(altitude, params);
        assert!(density >= 0.0);
        let fog_t = fog_transmittance(300.0, density);
        assert!((0.0..=1.0).contains(&fog_t));
        let composed = cloud.transmittance * fog_t;
        assert!(
            (0.0..=1.0).contains(&composed),
            "composed transmittance out of range: {composed}"
        );
    }
}

#[test]
fn atmosphere_blend_stays_bounded_over_ray_march_output() {
    let cfg = RaymarchConfig::default();
    let cloud = march(
        |_t| 0.7,
        sigma_from_density,
        hg_phase(0.5, 0.4),
        |_| 1.0,
        300.0,
        cfg,
    );
    let cloud_color = Vec3::new(0.8, 0.85, 0.95);
    let inscatter = Vec3::new(0.3, 0.4, 0.6);
    let max_distance = 1000.0;
    let mut prev_weight = -1.0_f32;
    for step in 0..=10 {
        let distance = step as f32 / 10.0 * max_distance;
        let weight = aerial_perspective_weight(distance, max_distance);
        assert!((0.0..=1.0).contains(&weight));
        assert!(weight + 1.0e-6 >= prev_weight, "aerial weight not monotone");
        prev_weight = weight;
        let blended = blend_with_atmosphere(cloud_color, cloud.transmittance, inscatter, weight);
        for c in [blended.x, blended.y, blended.z] {
            assert!(c.is_finite() && c >= 0.0, "blended channel invalid: {c}");
        }
    }
}

#[test]
fn cloud_lod_selection_and_binning_track_distance_deterministically() {
    let thresholds = CloudLodThresholds {
        mid_beyond: 100.0,
        far_beyond: 400.0,
        imposter_beyond: 1200.0,
    };
    // Rank is monotone non-decreasing in distance (coarser when farther).
    let mut prev_rank = 0u8;
    for distance in [0.0_f32, 50.0, 150.0, 500.0, 2000.0] {
        let rank = select_lod(distance, thresholds).rank();
        assert!(rank >= prev_rank, "LOD rank decreased with distance");
        prev_rank = rank;
    }
    // Binning is deterministic and partitions every in-range layer.
    let layers = [0u32, 1, 2, 3];
    let distances = [10.0_f32, 200.0, 600.0, 5000.0];
    let a = bin_by_distance(&layers, &distances, thresholds);
    let b = bin_by_distance(&layers, &distances, thresholds);
    assert_eq!(a.total(), 4);
    assert_eq!(a.total(), b.total());
}

#[test]
fn temporal_active_pixel_and_history_clamp_are_deterministic_and_bounded() {
    // Active-pixel mask is deterministic and, over a full period, covers pixels.
    let mode = UpscaleMode::QuarterRes;
    let period = mode.period();
    assert!(period >= 1);
    for frame in 0..(period * 2) {
        let a = active_pixel(frame, 3, 5, mode);
        let b = active_pixel(frame, 3, 5, mode);
        assert_eq!(a, b, "active_pixel must be deterministic");
    }
    // History clamp keeps the sample within the neighbourhood window.
    for sample in [-5.0_f32, 0.0, 0.5, 2.0] {
        let clamped = clamp_history(sample, 0.1, 0.9);
        assert!(
            (0.1..=0.9).contains(&clamped),
            "history clamp escaped window: {clamped}"
        );
    }
}

#[test]
fn budget_arbitration_stays_within_quota_with_forward_progress() {
    let budget = VolumetricBudget {
        raymarch_samples_per_frame: 100,
        modeling_voxels_per_frame: 100,
        upsample_pixels_per_frame: 100,
        multiscatter_cells_per_frame: 100,
    };
    let requests = [
        VolumetricJobRequest {
            handle: CloudLayerHandle(0),
            kind: VolumetricJobKind::Raymarch,
            cost: 60,
            priority: 10,
        },
        VolumetricJobRequest {
            handle: CloudLayerHandle(1),
            kind: VolumetricJobKind::Raymarch,
            cost: 60,
            priority: 5,
        },
        VolumetricJobRequest {
            handle: CloudLayerHandle(2),
            kind: VolumetricJobKind::Modeling,
            cost: 250,
            priority: 20,
        },
    ];
    let plan = plan_volumetric(&requests, budget);
    // Raymarch quota (100) admits the first 60 job, defers the second 60 job.
    assert_eq!(plan.count_of_kind(VolumetricJobKind::Raymarch), 1);
    assert!(plan.used_of_kind(VolumetricJobKind::Raymarch) <= budget.raymarch_samples_per_frame);
    // Oversized first job is still admitted (forward-progress guarantee).
    assert_eq!(plan.count_of_kind(VolumetricJobKind::Modeling), 1);
    // Determinism: replanning yields the same schedule size.
    let plan2 = plan_volumetric(&requests, budget);
    assert_eq!(plan.scheduled_count(), plan2.scheduled_count());
}

#[test]
fn wind_field_advection_preserves_weather_range() {
    let handle = WeatherMapHandle(1);
    let field = WeatherField::filled(handle, 4, 4, WeatherSample::from_rgba(0.5, 0.5, 0.2, 0.3));
    let wind = WindField::new(Vec2::new(1.0, 0.0), 5.0, 0.2);
    let advected = super::weather::advect_with_wind(&field, &wind, 0.1);
    for cell in advected.cells() {
        assert!((0.0..=1.0).contains(&cell.coverage));
        assert!((0.0..=1.0).contains(&cell.cloud_type));
        assert!((0.0..=1.0).contains(&cell.precipitation));
    }
}
