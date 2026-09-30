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
//! - `spectral` -> `atmosphere`: the spectral night-sky/twilight model
//!   (`rayleigh_phase`, `ozone_absorption`, `spectral_to_rgb`,
//!   `sunset_reddening`) feeds the aerial-perspective coupling so a low sun
//!   only ever *warms* (reddens) the sampled airlight, staying energy-bounded.
//! - `reference` -> `raymarch`: the unbiased delta-/ratio-tracking
//!   transmittance oracles and the analytic Beer-Lambert form agree with the
//!   ray-march's accumulated transmittance on a homogeneous column (design
//!   section 9c reference cross-check).
//! - `reference` -> `multiscatter`: the Monte-Carlo single-scatter oracle and
//!   the closed form agree, and folding the LUT energy gain into the resolve
//!   composition `single * (1 + gain)` only ever adds bounded energy (never
//!   removes it, never more than doubles it) and stays monotone in optical
//!   depth and albedo.
//! - `storm` -> `modeling`: the cumulonimbus vertical-development state
//!   machine folds through its own anvil/overshoot curves into a `0..=1`
//!   weight that only ever *adds* vertical development to the base height
//!   gradient, staying bounded and growing with storm maturity.
//! - `cloud_lod`, `temporal`, `budget`: scheduling and selection are
//!   deterministic and stay within their declared limits.
//! - `coupling` (two-way, section 9d): terrain occlusion only ever *carves*
//!   cloud density, and the cloud-shadow modulation feeding ground/GI bounce
//!   grows monotonically with cloud transmittance, both staying in `0..=1`.
//! - `fog` (unified volumetric fog, section 9f): the contrail diffusion kernel
//!   conserves cross-section mass, spread grows with age, and froxel injection
//!   weights taper monotonically without overwriting deeper slices.
//! - `storm` authored primitives (section 9b): the anvil/overshoot/virga/
//!   pyrocumulus curves fold into the modelling height gradient as bounded,
//!   only-additive vertical development.
//! - `multiscatter` probe grid: trilinear irradiance sampling is a convex
//!   blend of the shared probes and agrees with the lattice accessor.
//!
//! These tests only consume the public contracts of each module, so they also
//! act as a compile-time guard that the cross-module API surface stays stable.

use super::atmosphere::{
    aerial_perspective_weight, blend_with_atmosphere, sunset_inscatter_tint,
    AerialPerspectiveParams, AtmosphereCoupling,
};
use super::avsm::AvsmCurve;
use super::budget::{plan_volumetric, VolumetricJobKind, VolumetricJobRequest};
use super::cloud_lod::{bin_by_distance, select_lod, CloudLodThresholds};
use super::coupling::{
    apply_carve, cloud_shadow_modulation, density_delta, terrain_occlusion, CarveBrush,
};
use super::fog::{
    contrail_kernel, contrail_spread, fog_transmittance, froxel_injection_weight,
    height_fog_density, Contrail, HeightFogParams,
};
use super::math::{ln_approx, saturate, EPS};
use super::modeling::{compose_from_modeling, height_gradient};
use super::multiscatter::{MultiScatterLut, ProbeGrid, PROBE_BANDS};
use super::noise::{perlin_worley, worley_fbm};
use super::raymarch::{march, RaymarchConfig};
use super::reference::{
    analytic_single_scatter, analytic_transmittance, delta_tracking_transmittance,
    ratio_tracking_transmittance, single_scatter_reference,
};
use super::scatter::{
    dual_lobe_draine_phase, hg_phase, isotropic_phase, octave_scatter, powder, OctaveParams,
    DEFAULT_BACKWARD_G, DEFAULT_DRAINE_ALPHA, DEFAULT_FORWARD_G, DEFAULT_HG_DRAINE_WEIGHT,
    DEFAULT_LOBE_BLEND, DEFAULT_POWDER_STRENGTH,
};
use super::shadow::{
    accumulate_shadow, god_ray_weight, scattering_mask, CloudShadowConfig, GodRayConfig,
};
use super::spectral::{
    ozone_absorption, rayleigh_phase, spectral_to_rgb, sunset_reddening, SpectralBands,
};
use super::storm::{
    anvil_profile, gravity_wave, overshooting_bump, pyrocumulus_buoyancy, virga_fade, StormState,
};
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
fn storm_vertical_development_modulates_cumulonimbus_density_bounded() {
    // A barely-developed cell versus one driven hard to maturity.
    let mut young = StormState::default();
    young.advance_storm(0.1, 0.7);
    let mut mature = StormState::default();
    for _ in 0..80 {
        mature.advance_storm(0.1, 0.95);
    }

    let mut h = 0.0_f32;
    while h <= 1.0 {
        // Base deep-convective column from the modelling height gradient.
        let base = height_gradient(h, CloudKind::Cumulonimbus);
        assert!((0.0..=1.0).contains(&base));

        // The storm state machine adds vertical development through its own
        // authored curves; union it with the base so density stays bounded.
        let young_dev = young.vertical_profile(h);
        let mature_dev = mature.vertical_profile(h);
        let young_density = saturate(base + young_dev - base * young_dev);
        let mature_density = saturate(base + mature_dev - base * mature_dev);

        assert!(
            (0.0..=1.0).contains(&young_density),
            "young storm density out of range at {h}"
        );
        assert!(
            (0.0..=1.0).contains(&mature_density),
            "mature storm density out of range at {h}"
        );
        // Storm development only ever raises density above the bare column.
        assert!(
            young_density + EPS >= base,
            "storm removed column density at {h}"
        );
        // A more mature storm never has less development than a younger one.
        assert!(
            mature_density + EPS >= young_density,
            "mature storm weaker than young at {h}: {mature_density} < {young_density}"
        );
        h += 0.05;
    }

    // The trailing virga veil is a bounded precipitation curtain beneath the
    // base that only appears once the storm has matured.
    assert!(
        young.virga_veil(1.0) <= mature.virga_veil(1.0) + EPS,
        "virga veil should not shrink as the storm matures"
    );
    let mut f = 0.0_f32;
    while f <= 1.0 {
        assert!((0.0..=1.0).contains(&mature.virga_veil(f)));
        f += 0.1;
    }

    // The stable-layer gravity-wave ripple stays bounded for the mature cell's
    // evolving phase across a horizontal sweep (lateral anvil undulation).
    let mut x = -8.0_f32;
    while x <= 8.0 {
        let ripple = gravity_wave(mature.gravity_wave_phase, x);
        assert!(
            ripple.abs() <= 1.0 + 1.0e-4,
            "gravity wave unbounded: {ripple}"
        );
        x += 0.5;
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

#[test]
fn spectral_night_sky_base_color_conserves_energy() {
    // A bluish twilight distribution maps to a non-negative RGB whose channels
    // sum to the (unit) band energy: `spectral_to_rgb` conserves weight and
    // never manufactures negative light, so it is a valid sky base colour.
    for (r, g, b) in [
        (0.10_f32, 0.15, 0.60),
        (0.05, 0.20, 0.75),
        (0.33, 0.33, 0.34),
        (0.00, 0.00, 1.00),
    ] {
        let bands = SpectralBands::from_rgb(r, g, b);
        assert!(
            (bands.sum() - 1.0).abs() < 1.0e-5,
            "band set not normalised"
        );
        let rgb = spectral_to_rgb(&bands);
        for c in [rgb.x, rgb.y, rgb.z] {
            assert!(c.is_finite() && c >= 0.0, "sky channel invalid: {c}");
            assert!(c <= 1.0 + 1.0e-5, "sky channel exceeds unit energy: {c}");
        }
        let total = rgb.x + rgb.y + rgb.z;
        assert!(
            (total - 1.0).abs() < 1.0e-4,
            "spectral_to_rgb did not conserve band energy: {total}"
        );
    }
}

#[test]
fn rayleigh_phase_is_symmetric_and_bounds_the_hg_product() {
    // Rayleigh scattering is forward/back symmetric, and multiplying it by the
    // cloud HG lobe (the airlight-through-cloud product used along the march)
    // stays finite and non-negative for every scattering angle.
    let g = 0.4_f32;
    for step in 0..=20 {
        let mu = -1.0 + step as f32 / 10.0;
        let fwd = rayleigh_phase(mu);
        let bwd = rayleigh_phase(-mu);
        assert!(
            fwd.is_finite() && fwd >= 0.0,
            "rayleigh not finite/positive"
        );
        assert_eq!(
            fwd.to_bits(),
            bwd.to_bits(),
            "rayleigh not symmetric at mu={mu}"
        );
        let product = fwd * hg_phase(mu, g);
        assert!(
            product.is_finite() && product >= 0.0,
            "rayleigh*hg product invalid at mu={mu}: {product}"
        );
    }
}

#[test]
fn ozone_absorption_peaks_in_the_chappuis_band_and_stays_non_negative() {
    // The Chappuis band centre absorbs more than the red tail, and the
    // coefficient is non-negative across (and beyond) the visible span.
    let chappuis_center = ozone_absorption(602.0);
    let red_tail = ozone_absorption(700.0);
    assert!(
        chappuis_center > red_tail,
        "Chappuis centre should absorb more than the red tail: {chappuis_center} vs {red_tail}"
    );
    for nm in [200.0_f32, 380.0, 500.0, 602.0, 700.0, 900.0] {
        assert!(
            ozone_absorption(nm) >= 0.0,
            "ozone absorption went negative at {nm}nm"
        );
    }
}

#[test]
fn sunset_coupling_warms_airlight_relative_to_neutral_over_the_march() {
    // End-to-end spectral -> atmosphere seam: march a near-opaque cloud column,
    // fade it into sampled airlight with the aerial-perspective weight, and
    // compare a low (sunset) sun against a high (neutral) sun. The low sun must
    // redden the airlight -- i.e. attenuate blue relative to red -- while every
    // channel stays finite and non-negative.
    let cfg = RaymarchConfig::default();
    let cloud = march(
        |_t| 0.7,
        sigma_from_density,
        hg_phase(0.5, 0.4),
        |_| 1.0,
        300.0,
        cfg,
    );
    // Airlight dominates when the cloud is near-opaque and the view distance is
    // large, so the sunset tint is actually visible in the composite.
    let cloud_color = Vec3::new(0.05, 0.05, 0.06);
    let inscatter = Vec3::new(0.20, 0.35, 0.80);
    let weight = aerial_perspective_weight(950.0, 1000.0);

    let sunset = AtmosphereCoupling::new(1.0, 0.0).with_sun_altitude(0.0);
    let neutral = AtmosphereCoupling::new(1.0, 0.0);
    // `new` leaves the sun high (neutral tint); only `with_sun_altitude` arms it.
    assert_eq!(
        neutral.sun_altitude,
        super::atmosphere::NO_TWILIGHT_ALTITUDE
    );

    let warm = sunset.apply(cloud_color, cloud.transmittance, inscatter, weight);
    let cool = neutral.apply(cloud_color, cloud.transmittance, inscatter, weight);

    for c in [warm.x, warm.y, warm.z, cool.x, cool.y, cool.z] {
        assert!(c.is_finite() && c >= 0.0, "coupled channel invalid: {c}");
    }
    // The neutral coupling reproduces the plain, untinted blend exactly.
    let plain = blend_with_atmosphere(cloud_color, cloud.transmittance, inscatter, weight);
    for (a, b) in [(cool.x, plain.x), (cool.y, plain.y), (cool.z, plain.z)] {
        assert!(
            (a - b).abs() < 1.0e-6,
            "neutral coupling drifted from plain blend"
        );
    }
    // Reddening: blue/red ratio strictly drops under the sunset tint.
    let warm_ratio = warm.z / warm.x.max(EPS);
    let cool_ratio = cool.z / cool.x.max(EPS);
    assert!(
        warm_ratio < cool_ratio,
        "sunset did not warm the airlight (blue/red {warm_ratio} !< {cool_ratio})"
    );
    // The tint only attenuates, so warm channels never exceed the neutral ones.
    for (w, c) in [(warm.x, cool.x), (warm.y, cool.y), (warm.z, cool.z)] {
        assert!(
            w <= c + 1.0e-6,
            "sunset tint amplified a channel: {w} > {c}"
        );
    }
    // And the underlying reddening driver is monotone: a low sun reddens more.
    assert!(sunset_reddening(0.0) > sunset_reddening(super::atmosphere::NO_TWILIGHT_ALTITUDE));
    // The exposed tint agrees with what the coupling applied.
    let tint = sunset_inscatter_tint(0.0);
    assert!(
        tint.z < tint.x,
        "twilight tint should suppress blue below red"
    );
}

#[test]
fn reference_tracking_estimators_agree_with_march_transmittance() {
    // Homogeneous column so every transmittance model has the same closed form.
    // A constant density of 0.25 maps (via `sigma_from_density`) to sigma_t = 1,
    // and over `distance` the analytic Beer-Lambert transmittance is exp(-1.2).
    let density = 0.25_f32;
    let (sigma_t, _sigma_s) = sigma_from_density(density);
    let distance = 1.2_f32;

    // 1) The ray-march accumulates exact per-segment Beer-Lambert factors, so it
    //    must match the analytic transmittance tightly for a homogeneous field.
    let cfg = RaymarchConfig::default();
    let cloud = march(
        |_t| density,
        sigma_from_density,
        hg_phase(0.4, 0.3),
        |_| 1.0,
        distance,
        cfg,
    );
    let analytic = analytic_transmittance(sigma_t, distance);
    assert!(
        (cloud.transmittance - analytic).abs() < 1.0e-3,
        "march transmittance {} diverged from analytic {analytic}",
        cloud.transmittance
    );

    // 2) The unbiased tracking oracles converge to the same analytic value. A
    //    majorant above sigma_t exercises real null-collisions in both walks.
    let majorant = 2.0_f32;
    let samples = 8192;
    let seed = 0xC0FF_EE01;
    let delta = delta_tracking_transmittance(|_| sigma_t, majorant, distance, seed, samples);
    let ratio = ratio_tracking_transmittance(|_| sigma_t, majorant, distance, seed, samples);
    for (name, est) in [("delta", delta), ("ratio", ratio)] {
        assert!(
            (0.0..=1.0).contains(&est),
            "{name} tracking out of range: {est}"
        );
        assert!(
            (est - analytic).abs() < 0.05,
            "{name} tracking {est} did not converge to analytic {analytic}"
        );
    }

    // 3) Both estimators are deterministic for a fixed seed (reproducible oracle).
    assert_eq!(
        delta.to_bits(),
        delta_tracking_transmittance(|_| sigma_t, majorant, distance, seed, samples).to_bits(),
        "delta tracking not deterministic"
    );
    assert_eq!(
        ratio.to_bits(),
        ratio_tracking_transmittance(|_| sigma_t, majorant, distance, seed, samples).to_bits(),
        "ratio tracking not deterministic"
    );
}

#[test]
fn terrain_and_cloud_shadow_close_the_two_way_coupling() {
    // Section 9d, downward half: terrain punching into the cloud layer only ever
    // *carves* density, and the occlusion is monotone non-increasing with height.
    let terrain_height = 100.0_f32;
    let base_density = 0.8_f32;
    let mut prev_occ = f32::INFINITY;
    let mut prev_density = f32::NEG_INFINITY;
    let mut y = 90.0_f32;
    while y <= 130.0 {
        let occ = terrain_occlusion(y, terrain_height);
        assert!(
            (0.0..=1.0).contains(&occ),
            "occlusion out of range at y={y}"
        );
        assert!(
            occ <= prev_occ + EPS,
            "occlusion rose with altitude at y={y}"
        );
        prev_occ = occ;
        // Fold the occlusion into the density through the shared carve choke.
        let carved = apply_carve(base_density, -occ);
        assert!(
            carved <= base_density + EPS,
            "terrain added density at y={y}"
        );
        assert!(
            carved >= prev_density - EPS,
            "carved density not monotone in height"
        );
        prev_density = carved;
        y += 2.5;
    }
    // Deep below terrain is fully occluded (density fully removed); well above is clear.
    assert!(apply_carve(base_density, -terrain_occlusion(0.0, terrain_height)) < EPS);
    assert_eq!(
        apply_carve(base_density, -terrain_occlusion(1000.0, terrain_height)).to_bits(),
        base_density.to_bits()
    );

    // Section 9d, upward half: cloud shadow modulates the lit ground that bounces
    // back into the aerial-perspective/GI sky-light. More sky shows through the
    // cloud (higher transmittance) => strictly more lit ground, bounded by albedo.
    let albedo = 0.6_f32;
    let mut prev_lit = f32::NEG_INFINITY;
    for step in 0..=10 {
        let transmittance = step as f32 / 10.0;
        let lit = cloud_shadow_modulation(transmittance, albedo);
        assert!(
            (0.0..=albedo + EPS).contains(&lit),
            "lit ground out of range: {lit}"
        );
        assert!(
            lit >= prev_lit - EPS,
            "lit ground not monotone in transmittance"
        );
        prev_lit = lit;
    }
    // Opaque cloud kills the ground bounce; clear sky reflects the full albedo.
    assert!(cloud_shadow_modulation(0.0, albedo) < EPS);
    assert!((cloud_shadow_modulation(1.0, albedo) - albedo).abs() < EPS);
}

#[test]
fn contrail_and_froxel_injection_conserve_mass_and_taper() {
    // Section 9f: the contrail diffusion kernel is a unit-area Gaussian, so a
    // fine numeric integral across its cross-section recovers (near) unit mass
    // regardless of age -- injecting a contrail conserves total condensate.
    for age in [0.0_f32, 5.0, 30.0] {
        let contrail = Contrail {
            age,
            width: 2.0,
            diffusion: 0.5,
        };
        let dx = 0.05_f32;
        let mut mass = 0.0_f32;
        let mut off = -160.0_f32;
        while off <= 160.0 {
            let k = contrail_kernel(off, contrail);
            assert!(k >= 0.0, "contrail kernel went negative");
            mass += k * dx;
            off += dx;
        }
        assert!(
            (mass - 1.0).abs() < 2.0e-2,
            "contrail mass not conserved: {mass}"
        );
    }
    // Spread widens monotonically with age (diffusion + wind shear).
    let mut prev_spread = f32::NEG_INFINITY;
    for age in [0.0_f32, 1.0, 10.0, 60.0] {
        let sp = contrail_spread(age);
        assert!(sp >= prev_spread, "contrail spread shrank with age");
        prev_spread = sp;
    }

    // Froxel injection only adds near-field energy and tapers to zero far away,
    // and the injected fog density still composes into a valid transmittance.
    let slices = 16u32;
    let mut prev_w = f32::INFINITY;
    for slice in 0..=slices {
        let w = froxel_injection_weight(slice, slices);
        assert!(
            (0.0..=1.0).contains(&w),
            "froxel weight out of range at {slice}"
        );
        assert!(
            w <= prev_w + EPS,
            "froxel weight not monotone non-increasing"
        );
        prev_w = w;
        let params = HeightFogParams {
            density_at_sea_level: 0.5,
            falloff: 0.1,
            max_height: 500.0,
        };
        let injected = height_fog_density(10.0, params) * w;
        let t = fog_transmittance(200.0, injected);
        assert!(
            (0.0..=1.0).contains(&t),
            "fog transmittance escaped unit range: {t}"
        );
    }
    assert!(
        froxel_injection_weight(slices, slices) < EPS,
        "far slice should taper to zero"
    );
}

#[test]
fn storm_authored_profiles_fold_into_bounded_vertical_development() {
    // Section 9b: the individual authored cumulonimbus curves are bounded and
    // monotone in their maturity drivers, independent of the StormState machine.
    // Anvil flares more with spread near the band top.
    let mut prev_anvil = f32::NEG_INFINITY;
    for spread in [0.0_f32, 0.3, 0.6, 1.0] {
        let a = anvil_profile(0.95, spread);
        assert!((0.0..=1.0).contains(&a), "anvil out of range");
        assert!(a >= prev_anvil - EPS, "anvil not monotone in spread");
        prev_anvil = a;
    }
    // Overshooting dome peaks at the band top and is bounded above/below it.
    let top = 0.8_f32;
    assert!(overshooting_bump(1.0, top) >= overshooting_bump(0.7, top));
    assert!(overshooting_bump(1.0, top) >= overshooting_bump(1.3, top));
    for h in [0.0_f32, 0.5, 1.0, 1.4] {
        assert!((0.0..=1.0).contains(&overshooting_bump(h, top)));
    }
    // Virga veil and pyrocumulus buoyancy are bounded, monotone ramps.
    assert!(virga_fade(1.0) >= virga_fade(0.5) && virga_fade(0.5) >= virga_fade(0.0));
    assert!(pyrocumulus_buoyancy(2.0) >= pyrocumulus_buoyancy(0.5));
    for q in [-1.0_f32, 0.0, 1.0, 5.0] {
        assert!((0.0..=1.0).contains(&pyrocumulus_buoyancy(q)));
    }

    // Seam: union the authored anvil + overshoot development onto the modelling
    // height gradient for a deep-convective column. It only ever *adds* density
    // and stays bounded, matching the state-machine seam's invariant.
    let spread = 0.8_f32;
    let mut h = 0.0_f32;
    while h <= 1.0 {
        let base = height_gradient(h, CloudKind::Cumulonimbus);
        let dev = saturate(anvil_profile(h, spread) + overshooting_bump(h, top));
        let density = saturate(base + dev - base * dev);
        assert!(
            (0.0..=1.0).contains(&density),
            "storm density out of range at {h}"
        );
        assert!(
            density + EPS >= base,
            "authored storm curves removed density at {h}"
        );
        h += 0.05;
    }
}

#[test]
fn multiscatter_probe_grid_blends_shared_irradiance_convexly() {
    // The probe grid only *consumes* shared irradiance; sampling must be a convex
    // blend of the stored probes and agree with the lattice accessor.
    let min_corner = Vec3::new(0.0, 0.0, 0.0);
    let max_corner = Vec3::new(1.0, 1.0, 1.0);
    // Fill each probe's bands with its own x coordinate: a known linear ramp.
    let grid = ProbeGrid::from_fn([2, 2, 2], min_corner, max_corner, |pos| {
        [pos.x; PROBE_BANDS]
    });
    assert_eq!(grid.dims(), [2, 2, 2]);

    // The lattice accessor reports the ramp endpoints exactly.
    for j in 0..2 {
        for k in 0..2 {
            assert!((grid.probe_at(0, j, k).irradiance[0] - 0.0).abs() < EPS);
            assert!((grid.probe_at(1, j, k).irradiance[0] - 1.0).abs() < EPS);
        }
    }

    // An interior query is the convex (here linear) blend of the corners.
    let mid = grid.sample(Vec3::new(0.5, 0.5, 0.5));
    for &v in mid.iter() {
        assert!(
            (v - 0.5).abs() < 1.0e-4,
            "probe blend not linear at midpoint"
        );
        assert!((0.0..=1.0).contains(&v), "probe blend left convex hull");
    }

    // Queries outside the box clamp to the boundary probes (no out-of-range read).
    let clamped = grid.sample(Vec3::new(5.0, 0.5, 0.5));
    for &v in clamped.iter() {
        assert!(
            (v - 1.0).abs() < 1.0e-4,
            "out-of-box query did not clamp high"
        );
    }
    let clamped_low = grid.sample(Vec3::new(-5.0, 0.5, 0.5));
    for &v in clamped_low.iter() {
        assert!(v.abs() < 1.0e-4, "out-of-box query did not clamp low");
    }

    // Degenerate dims are floored to at least one probe per axis (no panic).
    let thin = ProbeGrid::new([0, 3, 1], min_corner, max_corner);
    assert_eq!(thin.dims(), [1, 3, 1]);
}

/// Section 8 seam: the altitude-aware aerial-perspective weight
/// ([`AerialPerspectiveParams`]) feeds [`blend_with_atmosphere`] over a real
/// ray-march output. Thinner air aloft yields a strictly smaller weight, so a
/// high cloud sample retains more of its own radiance (is washed toward the
/// airlight less) than an otherwise identical ground-level sample, while both
/// composites stay inside the convex span of cloud radiance and airlight.
#[test]
fn altitude_aware_aerial_perspective_washes_low_clouds_more_than_high_ones() {
    let modeling = sample_modeling();
    let cfg = RaymarchConfig::default();
    let phase = hg_phase(0.4, 0.5);
    let seed = 0x5EED_1234;
    let state = march(
        |t| {
            field_density(
                modeling,
                CloudKind::Cumulus,
                Vec3::new(0.3, saturate(t / 200.0), 0.7),
                0.6,
                seed,
            )
        },
        sigma_from_density,
        phase,
        |_t| 1.0,
        200.0,
        cfg,
    );
    assert!((0.0..=1.0).contains(&state.transmittance));

    let cloud_color = Vec3::splat(saturate(state.scattered));
    // A distinct bluish airlight so the wash direction is observable.
    let inscatter = Vec3::new(0.35, 0.5, 0.85);
    let max_distance = 20_000.0;
    let distance = 12_000.0;

    let ground = AerialPerspectiveParams::new(distance, 500.0);
    let aloft = AerialPerspectiveParams::new(distance, 9_000.0);

    // Thinner air aloft => strictly smaller aerial-perspective weight.
    let wg = ground.weight(max_distance);
    let wa = aloft.weight(max_distance);
    assert!((0.0..=1.0).contains(&wg) && (0.0..=1.0).contains(&wa));
    assert!(
        wa < wg,
        "aloft weight {wa} should sit below ground weight {wg}"
    );

    let bg = blend_with_atmosphere(cloud_color, state.transmittance, inscatter, wg);
    let ba = blend_with_atmosphere(cloud_color, state.transmittance, inscatter, wa);

    // Both stay within the convex span of cloud_color..inscatter, channel-wise.
    for (c, i, g, a) in [
        (cloud_color.x, inscatter.x, bg.x, ba.x),
        (cloud_color.y, inscatter.y, bg.y, ba.y),
        (cloud_color.z, inscatter.z, bg.z, ba.z),
    ] {
        let lo = c.min(i) - EPS;
        let hi = c.max(i) + EPS;
        assert!((lo..=hi).contains(&g), "ground blend left convex hull: {g}");
        assert!((lo..=hi).contains(&a), "aloft blend left convex hull: {a}");
    }

    // Less wash aloft => composite sits farther from the airlight (retains more
    // cloud radiance) on the blue channel, where the two sources differ most.
    let dist_from_air = |b: f32| (b - inscatter.z).abs();
    assert!(
        dist_from_air(ba.z) + EPS >= dist_from_air(bg.z),
        "aloft composite should retain more cloud radiance: aloft {} ground {}",
        ba.z,
        bg.z
    );

    // The combined weight is monotone non-decreasing in distance at fixed
    // altitude (distance fade only ever grows).
    let mut prev = AerialPerspectiveParams::new(0.0, 3_000.0).weight(max_distance);
    let mut d = 0.0;
    while d <= max_distance {
        let cur = AerialPerspectiveParams::new(d, 3_000.0).weight(max_distance);
        assert!(cur + EPS >= prev, "aerial weight dropped with distance");
        prev = cur;
        d += 1_000.0;
    }
}

/// Section 6b seam: an [`AvsmCurve`] built from a real ray-march extinction
/// profile self-compresses to a small node budget yet still reconstructs the
/// self-shadow transmittance of an uncompressed reference within a bounded
/// error, preserves the area under the curve (the adaptive-merge error metric),
/// and remains a valid monotone light-visibility function for [`march`].
#[test]
fn avsm_compression_from_march_extinction_preserves_self_shadow_within_budget() {
    let modeling = sample_modeling();
    let seed = 0x00C0_FFEE;
    // Deterministic per-segment optical thickness sampled front-to-back along
    // the light ray through a tall cumulonimbus column.
    let seg_at = |i: usize| -> (f32, f32) {
        let depth = i as f32 * 3.0;
        let frac = saturate(depth / 200.0);
        let density = field_density(
            modeling,
            CloudKind::Cumulonimbus,
            Vec3::new(0.4, frac, 0.6),
            0.7,
            seed,
        );
        let (sigma_t, _sigma_s) = sigma_from_density(density);
        (depth, sigma_t * 3.0)
    };

    let n = 64usize;
    let budget = 8usize;
    // Fine reference keeps every node; budgeted curve compresses on insert.
    let mut fine = AvsmCurve::new(n + 2);
    let mut budgeted = AvsmCurve::new(budget);
    for i in 0..n {
        let (d, seg) = seg_at(i);
        fine.insert(d, seg);
        budgeted.insert(d, seg);
    }

    // Node budget is honored and compression actually dropped nodes.
    assert_eq!(budgeted.max_nodes(), budget);
    assert!(budgeted.len() <= budgeted.max_nodes());
    assert!(
        fine.len() > budgeted.len(),
        "compression should drop nodes: fine {} budgeted {}",
        fine.len(),
        budgeted.len()
    );

    // Reconstructed transmittance tracks the fine reference and stays a monotone
    // non-increasing, in-range light function.
    let mut prev = 2.0_f32;
    let mut max_err = 0.0_f32;
    let mut d = 0.0;
    while d <= 190.0 {
        let tf = fine.transmittance_at(d);
        let tb = budgeted.transmittance_at(d);
        assert!(
            (0.0..=1.0).contains(&tb),
            "budgeted transmittance out of range: {tb}"
        );
        assert!(
            tb <= prev + 1.0e-6,
            "budgeted transmittance rose: {tb} > {prev}"
        );
        prev = tb;
        max_err = max_err.max((tf - tb).abs());
        d += 5.0;
    }
    assert!(
        max_err < 0.12,
        "compression transmittance error too large: {max_err}"
    );

    // Area under the curve (the compression error metric) is preserved.
    let area_err = (fine.area() - budgeted.area()).abs();
    let rel = area_err / fine.area().max(EPS);
    assert!(rel < 0.1, "compression area drifted too far: rel {rel}");

    // The compressed curve is still a valid march light-visibility function.
    let cfg = RaymarchConfig::default();
    let phase = hg_phase(0.3, 0.4);
    let lit = march(
        |_t| 0.5,
        sigma_from_density,
        phase,
        |t| budgeted.transmittance_at(t),
        190.0,
        cfg,
    );
    assert!((0.0..=1.0).contains(&lit.transmittance));
    assert!(lit.scattered.is_finite() && lit.scattered >= 0.0);
}

/// Section 5/7 seam: the production anisotropic dual-lobe HG+Draine cloud phase
/// (parameterized entirely by the `scatter::DEFAULT_*` presets) drives a real
/// ray-march, and the Nubis powder dark-edge term composes onto its scattered
/// radiance. Asserts the silver-lining anisotropy (forward >> side >> back, and
/// the forward peak beats isotropic), that the march stays energy-bounded, and
/// that powder only ever darkens the edge while keeping the result in `[0, 1]`.
#[test]
fn production_dual_lobe_draine_phase_and_powder_drive_a_bounded_march() {
    // Silver-lining anisotropy of the authored default phase.
    let phase_at = |cos: f32| {
        dual_lobe_draine_phase(
            cos,
            DEFAULT_FORWARD_G,
            DEFAULT_BACKWARD_G,
            DEFAULT_DRAINE_ALPHA,
            DEFAULT_HG_DRAINE_WEIGHT,
            DEFAULT_LOBE_BLEND,
        )
    };
    let forward = phase_at(0.95);
    let side = phase_at(0.0);
    let back = phase_at(-0.9);
    let iso = isotropic_phase();
    // The Draine-sharpened forward lobe dominates every other direction (the
    // silver-lining peak).
    assert!(
        forward > side,
        "forward lobe {forward} should beat side {side}"
    );
    assert!(
        forward > back,
        "forward lobe {forward} should beat back {back}"
    );
    assert!(
        forward > iso,
        "forward peak {forward} should exceed isotropic {iso}"
    );
    // The soft negative-g backward lobe lifts back-scatter above the side
    // minimum: the dual-lobe wrap-around ambient fill.
    assert!(
        back > side,
        "backward wrap lobe {back} should lift above the side minimum {side}"
    );
    for c in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
        assert!(phase_at(c) >= 0.0, "phase went negative at cos {c}");
    }

    // Feed the forward-scattering phase value into a real march over a density
    // field; the accumulated transmittance and scattered radiance stay bounded.
    let modeling = sample_modeling();
    let cfg = RaymarchConfig::default();
    let seed = 0x00A1_1CE0;
    let state = march(
        |t| {
            field_density(
                modeling,
                CloudKind::Cumulus,
                Vec3::new(0.6, saturate(t / 200.0), 0.35),
                0.65,
                seed,
            )
        },
        sigma_from_density,
        forward,
        |_t| 1.0,
        200.0,
        cfg,
    );
    assert!((0.0..=1.0).contains(&state.transmittance));
    assert!(state.scattered.is_finite() && state.scattered >= 0.0);

    // Powder darkens the edge: applying `1 - powder` to the scattered radiance
    // only ever removes energy and stays in range, and the darkening deepens as
    // the view-ray optical depth grows.
    let optical_depth = -ln_approx(state.transmittance.max(EPS));
    let dark = powder(optical_depth, DEFAULT_POWDER_STRENGTH);
    assert!((0.0..=1.0).contains(&dark));
    let radiance = saturate(state.scattered);
    let darkened = radiance * (1.0 - dark);
    assert!((0.0..=1.0).contains(&darkened));
    assert!(
        darkened <= radiance + EPS,
        "powder should not brighten: {darkened} > {radiance}"
    );
    let mut prev = 0.0_f32;
    let mut d = 0.0;
    while d <= 4.0 {
        let cur = powder(d, DEFAULT_POWDER_STRENGTH);
        assert!(cur + EPS >= prev, "powder darkening should grow with depth");
        prev = cur;
        d += 0.5;
    }
}

/// Section 12 seam: the cloud self-shadow and crepuscular (god-ray) helpers
/// compose with the real density pipeline. A density profile sampled along the
/// sun ray accumulates into a Beer-Lambert shadow transmittance that (a) is a
/// valid monotone light-visibility function fed back into [`march`], and only
/// ever darkens as more cloud is added; the god-ray sample weights taper
/// geometrically with bounded partial sums; and the screen-space scattering
/// mask vanishes in full shadow or clear air while growing with both light and
/// medium.
#[test]
fn cloud_shadow_and_god_rays_compose_with_the_density_pipeline() {
    let modeling = sample_modeling();
    let shadow_cfg = CloudShadowConfig::default();
    let seed = 0x5AD0_0011;
    let steps = shadow_cfg.step_count as usize;
    let step = shadow_cfg.max_shadow_distance / shadow_cfg.step_count as f32;

    // Sample the density profile toward the sun through a cumulonimbus column.
    let mut samples = [0.0_f32; 16];
    for (i, slot) in samples.iter_mut().enumerate().take(steps) {
        let frac = saturate(i as f32 / steps as f32);
        let density = field_density(
            modeling,
            CloudKind::Cumulonimbus,
            Vec3::new(0.5, frac, 0.5),
            0.7,
            seed,
        );
        *slot = density * shadow_cfg.density_scale;
    }

    // Adding more cloud along the ray only ever darkens the shadow.
    let half = accumulate_shadow(&samples[..steps / 2], step);
    let full = accumulate_shadow(&samples[..steps], step);
    assert!((0.0..=1.0).contains(&half) && (0.0..=1.0).contains(&full));
    assert!(
        full <= half + EPS,
        "more cloud should not brighten: {full} > {half}"
    );

    // The self-shadow transmittance is a valid light-visibility function: fed as
    // the march light term it keeps the accumulated transmittance/radiance bounded.
    let sun_visibility = full;
    let cfg = RaymarchConfig::default();
    let phase = hg_phase(0.5, DEFAULT_FORWARD_G);
    let lit = march(
        |_t| 0.5,
        sigma_from_density,
        phase,
        |_t| sun_visibility,
        200.0,
        cfg,
    );
    assert!((0.0..=1.0).contains(&lit.transmittance));
    assert!(lit.scattered.is_finite() && lit.scattered >= 0.0);

    // God-ray weights taper monotonically and their partial sum stays under the
    // closed-form geometric bound weight / (1 - decay).
    let ray_cfg = GodRayConfig::default();
    let mut prev = 2.0_f32;
    let mut sum = 0.0_f32;
    for i in 0..ray_cfg.sample_count {
        let w = god_ray_weight(i, ray_cfg);
        assert!((0.0..=1.0).contains(&w), "god-ray weight out of range: {w}");
        assert!(w <= prev + EPS, "god-ray weight rose at index {i}");
        prev = w;
        sum += w;
    }
    let bound = saturate(ray_cfg.weight) / (1.0 - saturate(ray_cfg.decay));
    assert!(
        sum <= bound + 1.0e-3,
        "god-ray partial sum {sum} exceeded bound {bound}"
    );

    // The scattering mask vanishes in full shadow and in clear air, and rises
    // when both light reaches the fragment and there is medium to scatter off.
    assert_eq!(scattering_mask(0.0, 0.8), 0.0);
    assert_eq!(scattering_mask(0.9, 0.0), 0.0);
    let dim = scattering_mask(0.3, 0.3);
    let bright = scattering_mask(0.9, 0.9);
    assert!((0.0..=1.0).contains(&dim) && (0.0..=1.0).contains(&bright));
    assert!(
        bright > dim,
        "mask should grow with light and medium: {bright} <= {dim}"
    );
}
