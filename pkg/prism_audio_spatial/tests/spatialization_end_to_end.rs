//! End-to-end integration coverage for the spatial audio layer.
//!
//! Drives the public spatialisation entry point
//! [`prism_audio_spatial::spatializer::resolve`] with real [`Listener`] /
//! [`Emitter`] poses and [`SourceDescriptor`] configuration, then asserts the
//! perceptual invariants a game/renderer relies on: a front source is centred
//! and audible, distance monotonically attenuates, azimuth/elevation signs
//! follow the right-handed `+X` right / `+Y` up / `-Z` forward convention,
//! Doppler raises pitch on approach and lowers it on recession, occlusion
//! darkens and quietens the direct path, air absorption rolls off the top end
//! with distance, and the whole resolve is bit-reproducible. It also exercises
//! the first-order Ambisonic encode/decode/rotate helpers that the panning
//! stage builds on.
//!
//! # Provenance
//!
//! Original test code written for Prism. It contains no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Dolby, MPEG, Google Resonance, Web Audio,
//! or Project Acoustics source or derived code, and no AI/ML. It exercises only
//! the crate's own public API and relies on standard, publicly documented
//! acoustics (inverse-distance attenuation, the Doppler ratio, ISO-style air
//! absorption, and ACN/SN3D first-order Ambisonics) at the level of ideas only.
//!
//! # Relationship
//!
//! Integration test for `prism_audio_spatial` (design doc chapters on
//! spatialisation and Ambisonics). Depends on `prism_audio_spatial` and
//! `bevy_math` as ordinary dev-visible dependencies and treats them as black
//! boxes through their public interfaces.

use bevy_math::{Quat, Vec3};
use core::f32::consts::FRAC_PI_2;
use prism_audio_spatial::ambisonics::{
    decode_foa, encode_foa_gains, encode_foa_sample, rotate_foa, FOA_CHANNELS, IDX_W, IDX_X,
    IDX_Y, IDX_Z,
};
use prism_audio_spatial::attenuation::{Attenuation, DistanceModel};
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::spatializer::{resolve, SourceDescriptor};

const SR: u32 = 48_000;

/// Branch-free absolute value, avoiding `f32::abs` so the determinism lints
/// that forbid `std` float intrinsics in this crate's tests stay satisfied.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Approximate scalar equality within an absolute tolerance.
fn approx(a: f32, b: f32, eps: f32) -> bool {
    fabs(a - b) <= eps
}

/// A descriptor whose distance model is a steep, clamped inverse curve so the
/// distance tests have a wide, monotone dynamic range to inspect.
fn descriptor_with_attenuation(att: Attenuation) -> SourceDescriptor {
    SourceDescriptor {
        attenuation: att,
        ..SourceDescriptor::default()
    }
}

#[test]
fn front_source_is_centred_and_audible() {
    let listener = Listener::default();
    let emitter = Emitter::point(Vec3::new(0.0, 0.0, -4.0), Vec3::ZERO);
    let params = resolve(
        &listener,
        &emitter,
        &SourceDescriptor::default(),
        OcclusionFactors::OPEN,
        SR,
    );

    assert!(approx(params.azimuth, 0.0, 1e-4), "azimuth {}", params.azimuth);
    assert!(
        approx(params.elevation, 0.0, 1e-4),
        "elevation {}",
        params.elevation
    );
    assert!(
        params.direct_gain > 0.0 && params.direct_gain <= 1.0,
        "direct_gain {}",
        params.direct_gain
    );
    // An open line of sight leaves the full wet send and no Doppler shift.
    assert!(approx(params.pitch_ratio, 1.0, 1e-6));
    assert!(params.wet_gain > 0.0 && params.wet_gain <= 1.0);
    // Air absorption at a few metres must still leave a wide-open corner.
    assert!(params.direct_cutoff_hz > 1_000.0);
    assert!(params.direct_cutoff_hz.is_finite());
}

#[test]
fn direct_gain_decreases_monotonically_with_distance() {
    let att = Attenuation::new(DistanceModel::Inverse, 1.0, 1_000.0, 1.0);
    let descriptor = descriptor_with_attenuation(att);
    let listener = Listener::default();

    let distances = [1.0_f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    let mut previous = f32::INFINITY;
    for d in distances {
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -d), Vec3::ZERO);
        let params = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
        assert!(
            params.direct_gain > 0.0 && params.direct_gain <= 1.0,
            "gain out of range at {} m: {}",
            d,
            params.direct_gain
        );
        assert!(
            params.direct_gain < previous,
            "gain not strictly decreasing at {} m: {} !< {}",
            d,
            params.direct_gain,
            previous
        );
        previous = params.direct_gain;
    }
}

#[test]
fn azimuth_sign_follows_left_right() {
    let listener = Listener::default();
    let descriptor = SourceDescriptor::default();

    let right = Emitter::point(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO);
    let left = Emitter::point(Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO);

    let r = resolve(&listener, &right, &descriptor, OcclusionFactors::OPEN, SR);
    let l = resolve(&listener, &left, &descriptor, OcclusionFactors::OPEN, SR);

    assert!(r.azimuth > 0.0, "right azimuth should be positive: {}", r.azimuth);
    assert!(l.azimuth < 0.0, "left azimuth should be negative: {}", l.azimuth);
    // Mirror sources are symmetric in azimuth magnitude.
    assert!(approx(r.azimuth, -l.azimuth, 1e-4));
}

#[test]
fn elevation_sign_follows_up_down() {
    let listener = Listener::default();
    let descriptor = SourceDescriptor::default();

    let up = Emitter::point(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
    let down = Emitter::point(Vec3::new(0.0, -5.0, 0.0), Vec3::ZERO);

    let u = resolve(&listener, &up, &descriptor, OcclusionFactors::OPEN, SR);
    let d = resolve(&listener, &down, &descriptor, OcclusionFactors::OPEN, SR);

    assert!(u.elevation > 0.0, "up elevation should be positive: {}", u.elevation);
    assert!(d.elevation < 0.0, "down elevation should be negative: {}", d.elevation);
    // Directly overhead is a quarter turn up.
    assert!(approx(u.elevation, FRAC_PI_2, 1e-3));
    assert!(approx(d.elevation, -FRAC_PI_2, 1e-3));
}

#[test]
fn doppler_raises_pitch_on_approach_and_lowers_on_recession() {
    let listener = Listener::default();
    let descriptor = SourceDescriptor::default();

    // Source dead ahead; velocity toward the listener (`+Z`) is approaching,
    // away from it (`-Z`) is receding.
    let approaching = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 30.0));
    let receding = Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, -30.0));

    let a = resolve(&listener, &approaching, &descriptor, OcclusionFactors::OPEN, SR);
    let r = resolve(&listener, &receding, &descriptor, OcclusionFactors::OPEN, SR);

    assert!(a.pitch_ratio > 1.0, "approach should raise pitch: {}", a.pitch_ratio);
    assert!(r.pitch_ratio < 1.0, "recession should lower pitch: {}", r.pitch_ratio);
    assert!(a.pitch_ratio.is_finite() && r.pitch_ratio.is_finite());
}

#[test]
fn occlusion_darkens_and_quietens_the_direct_path() {
    let listener = Listener::default();
    let descriptor = SourceDescriptor::default();
    let emitter = Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);

    let open = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
    let blocked = resolve(
        &listener,
        &emitter,
        &descriptor,
        OcclusionFactors::new(0.9, 0.9),
        SR,
    );

    assert!(
        blocked.direct_gain < open.direct_gain,
        "occlusion must reduce direct gain: {} !< {}",
        blocked.direct_gain,
        open.direct_gain
    );
    assert!(
        blocked.direct_cutoff_hz < open.direct_cutoff_hz,
        "occlusion must lower the direct corner: {} !< {}",
        blocked.direct_cutoff_hz,
        open.direct_cutoff_hz
    );
    assert!(
        blocked.wet_gain < open.wet_gain,
        "occlusion must reduce the wet send: {} !< {}",
        blocked.wet_gain,
        open.wet_gain
    );
}

#[test]
fn air_absorption_rolls_off_with_distance() {
    let listener = Listener::default();
    let descriptor = SourceDescriptor::default();

    let near = Emitter::point(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO);
    let far = Emitter::point(Vec3::new(0.0, 0.0, -200.0), Vec3::ZERO);

    let n = resolve(&listener, &near, &descriptor, OcclusionFactors::OPEN, SR);
    let f = resolve(&listener, &far, &descriptor, OcclusionFactors::OPEN, SR);

    assert!(
        f.direct_cutoff_hz <= n.direct_cutoff_hz,
        "distant source must not be brighter: {} > {}",
        f.direct_cutoff_hz,
        n.direct_cutoff_hz
    );
    // The corner is clamped below Nyquist.
    assert!(n.direct_cutoff_hz <= (SR as f32) * 0.5 + 1.0);
}

#[test]
fn resolve_is_bit_identical_across_calls() {
    let listener = Listener::new(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::from_rotation_y(0.3),
        Vec3::new(0.5, 0.0, -0.5),
    );
    let emitter = Emitter::new(
        Vec3::new(-4.0, 1.0, -9.0),
        Vec3::new(2.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    let descriptor = SourceDescriptor::default();
    let factors = OcclusionFactors::new(0.4, 0.25);

    let a = resolve(&listener, &emitter, &descriptor, factors, SR);
    let b = resolve(&listener, &emitter, &descriptor, factors, SR);

    // Every scalar field must be bit-for-bit identical, not just approximately.
    assert_eq!(a.direct_gain.to_bits(), b.direct_gain.to_bits());
    assert_eq!(a.pitch_ratio.to_bits(), b.pitch_ratio.to_bits());
    assert_eq!(a.azimuth.to_bits(), b.azimuth.to_bits());
    assert_eq!(a.elevation.to_bits(), b.elevation.to_bits());
    assert_eq!(a.direct_cutoff_hz.to_bits(), b.direct_cutoff_hz.to_bits());
    assert_eq!(a.wet_gain.to_bits(), b.wet_gain.to_bits());
}

#[test]
fn foa_encode_puts_omni_in_w_and_points_decode() {
    // Encode a unit source straight ahead (`-Z` forward).
    let gains = encode_foa_gains(Vec3::new(0.0, 0.0, -1.0));
    // The omnidirectional `W` channel carries unit SN3D weight.
    assert!(approx(gains[IDX_W], 1.0, 1e-6), "W gain {}", gains[IDX_W]);

    // Decoding toward the encoded direction must beat decoding away from it.
    let field = gains;
    let toward = decode_foa(&field, Vec3::new(0.0, 0.0, -1.0));
    let away = decode_foa(&field, Vec3::new(0.0, 0.0, 1.0));
    assert!(
        toward > away,
        "decode toward source must dominate: {} !> {}",
        toward,
        away
    );
}

#[test]
fn foa_encode_sample_scales_the_gains() {
    let direction = Vec3::new(0.3, -0.7, 0.6);
    let sample = 0.42_f32;
    let gains = encode_foa_gains(direction);

    let mut out = [0.0_f32; FOA_CHANNELS];
    encode_foa_sample(sample, direction, &mut out);

    for k in 0..FOA_CHANNELS {
        // `encode_foa_sample` is documented as `out[k] = sample * gains[k]`;
        // the product is computed identically so it must match bit-for-bit.
        assert_eq!(out[k].to_bits(), (sample * gains[k]).to_bits());
    }
}

#[test]
fn foa_rotation_preserves_w_and_velocity_energy() {
    let mut field = encode_foa_gains(Vec3::new(0.2, 0.5, -0.84));
    let w_before = field[IDX_W];
    let energy_before =
        field[IDX_X] * field[IDX_X] + field[IDX_Y] * field[IDX_Y] + field[IDX_Z] * field[IDX_Z];

    rotate_foa(&mut field, Quat::from_rotation_y(0.9));

    let energy_after =
        field[IDX_X] * field[IDX_X] + field[IDX_Y] * field[IDX_Y] + field[IDX_Z] * field[IDX_Z];

    // `W` is rotation invariant and orthonormal rotation preserves the velocity
    // energy of the first-order components.
    assert_eq!(field[IDX_W].to_bits(), w_before.to_bits());
    assert!(
        approx(energy_after, energy_before, 1e-5),
        "velocity energy drifted: {} vs {}",
        energy_after,
        energy_before
    );
}

#[test]
fn foa_identity_rotation_is_a_no_op() {
    let mut field = encode_foa_gains(Vec3::new(-0.6, 0.1, -0.79));
    let before = field;
    rotate_foa(&mut field, Quat::IDENTITY);
    for k in 0..FOA_CHANNELS {
        assert_eq!(field[k].to_bits(), before[k].to_bits());
    }
}
