//! End-to-end golden: geometry produces a banded direct arrival, the spatial
//! real-time shaper consumes it, and the rendered output matches an
//! independent hand computation.
//!
//! This integration test stitches the two crates together across their real
//! public API, exactly as a running engine would:
//!
//! 1. [`resolve_direct`](prism_audio_geometry::direct_path::resolve_direct)
//!    marches a straight emitter-to-listener segment through an
//!    [`AcousticScene`](prism_audio_geometry::AcousticScene) and folds every
//!    partition's per-band transmission into a
//!    [`PropagationPath`](prism_audio_spatial::propagation::PropagationPath),
//!    whose broadband [`gain`](prism_audio_spatial::propagation::PropagationPath::gain)
//!    and relative [`bands`](prism_audio_spatial::propagation::PropagationPath::bands)
//!    recombine via
//!    [`effective_bands`](prism_audio_spatial::propagation::PropagationPath::effective_bands)
//!    into the true per-band transmission.
//! 2. [`BandedPropagationShaper`](prism_audio_spatial::BandedPropagationShaper)
//!    splits a test tone into the three propagation bands with the shared
//!    Linkwitz-Riley crossover, scales each band by that spectrum, and sums
//!    them back.
//! 3. The per-band tail RMS of the rendered output is asserted against the
//!    independently derived per-band transmission (the golden).
//!
//! The point of a *cross-crate* golden is to prove the produce/consume contract
//! is lossless: the colour geometry computes is the colour the voice renders,
//! and the broadband/colour factorisation
//! ([`split_peak`](prism_audio_spatial::BandGains::split_peak)) round-trips
//! through the path and back.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.

use bevy_math::{ops, Quat, Vec3};
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;
use prism_audio_geometry::direct_path::resolve_direct;
use prism_audio_geometry::material_map::MaterialTable;
use prism_audio_geometry::scene::AcousticScene;
use prism_audio_geometry::GeometricConfig;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::material_spectrum::BandedAcousticMaterial;
use prism_audio_spatial::propagation::PathKind;
use prism_audio_spatial::{BandGains, BandedPropagationShaper};

const SR: u32 = 48_000;
const FRAMES: usize = 4096;

/// A low/mid/high test tone, each comfortably inside one propagation band so
/// the band crossover routes it to a single gain. The edges are 800 Hz and
/// 8 kHz, so 150 Hz is low, 2.5 kHz is mid, and 13 kHz is high.
const BAND_TONES_HZ: [Sample; 3] = [150.0, 2_500.0, 13_000.0];

/// Builds a scene whose single wall (two triangles in the plane `x = 0`)
/// carries the given per-band amplitude transmission. The listener and emitter
/// sit on opposite sides, so the direct segment crosses the wall exactly once.
fn wall_scene(transmission: BandGains) -> AcousticScene {
    let vertices = vec![
        Vec3::new(0.0, -4.0, -4.0),
        Vec3::new(0.0, 4.0, -4.0),
        Vec3::new(0.0, 4.0, 4.0),
        Vec3::new(0.0, -4.0, 4.0),
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];
    // Reflection and scattering are irrelevant to the direct arrival; only the
    // per-band transmission colours the transmitted path.
    let material = BandedAcousticMaterial::new(BandGains::SILENT, transmission, 0.0);
    AcousticScene::new(vertices, indices, MaterialTable::uniform(material)).unwrap()
}

fn empty_scene() -> AcousticScene {
    AcousticScene::new(vec![], vec![], MaterialTable::default()).unwrap()
}

fn fill_sine(buf: &mut AudioBuffer, freq: Sample) {
    buf.set_active_frames(buf.capacity_frames());
    let step = 2.0 * core::f32::consts::PI * freq / SR as Sample;
    let channels = buf.channels();
    for ch in 0..channels {
        let data = buf.channel_mut(ch);
        let mut phase = 0.0;
        for d in data.iter_mut() {
            *d = ops::sin(phase);
            phase += step;
        }
    }
}

fn tail_rms(buf: &AudioBuffer, channel: usize, skip: usize) -> Sample {
    let data = buf.channel(channel);
    let start = skip.min(data.len());
    let slice = &data[start..];
    if slice.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for &s in slice {
        acc += s * s;
    }
    ops::sqrt(acc / slice.len() as Sample)
}

/// Renders a steady tone through a shaper carrying `colour` and returns the
/// output/input tail RMS ratio (the realised per-band gain at that frequency).
fn rendered_ratio(colour: BandGains, freq: Sample) -> Sample {
    let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
    shaper.set_bands_immediate(colour);
    let mut input = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
    fill_sine(&mut input, freq);
    let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
    shaper.process_block(&input, &mut output);
    let skip = FRAMES / 2;
    tail_rms(&output, 0, skip) / tail_rms(&input, 0, skip)
}

#[test]
fn clear_line_renders_transparently() {
    // Geometry: an open scene yields a unity, full-band direct arrival.
    let scene = empty_scene();
    let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
    let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
    let cfg = GeometricConfig::new(SR);
    let result = resolve_direct(&scene, &listener, &emitter, &cfg);
    assert!(result.audible);
    assert_eq!(result.path.kind, PathKind::Direct);

    let effective = result.path.effective_bands();
    for (band, &g) in effective.bands().iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "clear line band {band} should be unity (got {g})"
        );
    }

    // Spatial: the shaper driven by that spectrum is magnitude transparent at
    // every band tone.
    for &freq in &BAND_TONES_HZ {
        let ratio = rendered_ratio(effective, freq);
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "clear line should render transparently at {freq} Hz (ratio {ratio})"
        );
    }
}

#[test]
fn absorptive_window_colours_the_arrival() {
    // A window that passes most low energy, half the mid, and little high.
    let transmission = BandGains::new([0.9, 0.5, 0.1]);
    let scene = wall_scene(transmission);
    let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
    let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
    let cfg = GeometricConfig::new(SR);
    let result = resolve_direct(&scene, &listener, &emitter, &cfg);

    assert!(result.audible);
    assert_eq!(result.path.kind, PathKind::Transmission);

    // Independent golden: crossing one wall multiplies unity by the wall's
    // per-band transmission, so the effective per-band gain is exactly the
    // material transmission.
    let golden = BandGains::UNITY.combine(transmission);
    let effective = result.path.effective_bands();
    for band in 0..3 {
        assert!(
            (effective.band(band) - golden.band(band)).abs() < 1e-5,
            "effective band {band} ({}) should match golden ({})",
            effective.band(band),
            golden.band(band)
        );
    }

    // Spatial render: each band tone comes out at its golden transmission.
    for (band, &freq) in BAND_TONES_HZ.iter().enumerate() {
        let ratio = rendered_ratio(effective, freq);
        let expected = golden.band(band);
        assert!(
            (ratio - expected).abs() < 0.08,
            "band {band} at {freq} Hz should render at {expected} (got {ratio})"
        );
    }
}

#[test]
fn broadband_colour_factorisation_round_trips() {
    // The path stores a flat broadband gain plus a relative colour; recombining
    // them must reproduce the original transmitted spectrum end to end.
    let transmission = BandGains::new([0.72, 0.33, 0.08]);
    let scene = wall_scene(transmission);
    let listener = Listener::new(Vec3::new(-2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
    let emitter = Emitter::point(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
    let cfg = GeometricConfig::new(SR);
    let result = resolve_direct(&scene, &listener, &emitter, &cfg);
    assert!(result.audible);

    // The stored colour is normalised to its peak, so the loudest band is at
    // unity while the broadband gain carries the peak transmission.
    let peak = transmission.peak();
    assert!(
        (result.path.gain - peak).abs() < 1e-5,
        "broadband gain {} should equal peak transmission {peak}",
        result.path.gain
    );
    assert!(
        (result.path.bands.peak() - 1.0).abs() < 1e-5,
        "stored colour should be peak-normalised (peak {})",
        result.path.bands.peak()
    );

    // gain * colour reconstructs the true per-band transmission.
    let effective = result.path.effective_bands();
    for band in 0..3 {
        assert!(
            (effective.band(band) - transmission.band(band)).abs() < 1e-5,
            "round-trip band {band} ({}) should recover transmission ({})",
            effective.band(band),
            transmission.band(band)
        );
    }
}

#[test]
fn disabled_transmission_silences_the_voice() {
    // A heavily blocking wall with transmission disabled yields no audible
    // path, so there is nothing for the shaper to render.
    let transmission = BandGains::new([0.02, 0.01, 0.005]);
    let scene = wall_scene(transmission);
    let listener = Listener::new(Vec3::new(-3.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ZERO);
    let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
    let cfg = GeometricConfig {
        transmission_enabled: false,
        ..GeometricConfig::new(SR)
    };
    let result = resolve_direct(&scene, &listener, &emitter, &cfg);
    assert!(!result.audible);
    assert!(result.occlusion.obstruction > 0.9);

    // The path gain is zero, so the effective colour is silent and the shaper
    // renders silence regardless of tone.
    let effective = result.path.effective_bands();
    for band in 0..3 {
        assert!(effective.band(band) <= 1e-6);
    }
    for &freq in &BAND_TONES_HZ {
        let ratio = rendered_ratio(effective, freq);
        assert!(ratio < 1e-3, "silent colour should render silence at {freq} Hz");
    }
}
