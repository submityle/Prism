//! Production-grade end-to-end integration test for the real public API of the
//! `prism_audio_accessibility` crate.
//!
//! This suite drives the full accessibility accommodation chain exactly as a
//! runtime or editor would: it builds an aggregate `AccessibilityProfile` from
//! the individual accommodations (`CaptionMetadata` / `CaptionReport`,
//! `MonoDownmix` / `DownmixMatrix`, `DialogueBoost`, `CompressionProfile`, and
//! `VisualCue`), stamps the resulting settings onto the real `prism_audio_core`
//! parameter structs (`DuckingParams`, `CompressorParams`) and buffers
//! (`AudioBuffer`), and asserts the semantic contracts, determinism, boundary
//! handling, and cross-component invariants of the whole pipeline. Only the
//! genuine public API is exercised; nothing is mocked or stubbed.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics source
//! or derived code; no AI/ML. Public acoustic standards (equal-power downmix,
//! dynamic-range compression) inform only the ideas, not any borrowed code.
//!
//! # Relationship
//! Validates design section 23 (accessibility) across its five accommodations
//! as composed by the `profile` module: captions and visual cues gate telemetry
//! reports, the mono downmix feeds the output stage, and the dialogue boost and
//! compression profile feed the dynamics stage. The test treats the crate as a
//! settings producer and checks that its outputs are internally consistent with
//! the `prism_audio_core` primitives it reuses.

use bevy_math::ops;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::dynamics::compressor::CompressorParams;
use prism_audio_core::nodes::dynamics::ducking::DuckingParams;

use prism_audio_accessibility::caption::{CaptionMetadata, CaptionReportBuilder};
use prism_audio_accessibility::compression::{CompressionPreset, CompressionProfile};
use prism_audio_accessibility::dialogue::{BoostStrength, DialogueBoost};
use prism_audio_accessibility::downmix::{DownmixMatrix, MonoDownmix};
use prism_audio_accessibility::profile::AccessibilityProfile;
use prism_audio_accessibility::visual_cue::{CueKind, VisualCue};

/// Absolute value without pulling in std float math (`f32::abs`).
///
/// Keeping the helper local lets every tolerance comparison use the same
/// branch-only implementation.
#[inline]
fn fabs(x: Sample) -> Sample {
    if x < 0.0 { -x } else { x }
}

/// Returns `true` when `a` and `b` are within `eps` of each other.
#[inline]
fn close(a: Sample, b: Sample, eps: Sample) -> bool {
    fabs(a - b) < eps
}

/// Tolerance for control-rate parameter comparisons.
const EPS: Sample = 1e-6;

/// Builds the canonical "fully assisted" profile used by several tests: every
/// accommodation on at its strongest setting.
fn fully_assisted_profile() -> AccessibilityProfile {
    AccessibilityProfile::new()
        .with_captions(true)
        .with_visual_cues(true)
        .with_downmix(MonoDownmix::new(true))
        .with_dialogue_boost(DialogueBoost::new(BoostStrength::High))
        .with_compression(CompressionProfile::from_preset(CompressionPreset::HardOfHearing))
}

/// The complete main chain: compose a profile, then derive every downstream
/// artifact (caption report, visual cue, downmix matrix, ducking params,
/// compressor params) and check they are mutually consistent.
#[test]
fn full_chain_end_to_end_produces_consistent_settings() {
    let profile = fully_assisted_profile();
    assert!(profile.is_any_active());

    // Caption half: author metadata, assemble the telemetry report.
    let caption = CaptionMetadata::new("Reactor critical".to_string(), "en-US".to_string(), 1_200)
        .with_speaker("Computer".to_string());
    let report = CaptionReportBuilder::new(caption.clone())
        .event_id(7)
        .timestamp_samples(48_000)
        .build();
    assert!(profile.captions_enabled());
    assert_eq!(report.event_id(), 7);
    assert_eq!(report.timestamp_samples(), 48_000);
    assert_eq!(report.caption(), &caption);
    assert!(report.caption().has_speaker());

    // Visual-cue half: an alert to the right of the listener.
    assert!(profile.visual_cues_enabled());
    let cue = VisualCue::new(CueKind::Alert, core::f32::consts::FRAC_PI_2, 0.0, 1.0);
    let dir = cue.direction_vector();
    assert!(close(dir.x, 1.0, 1e-5));
    assert!(close(dir.z, 0.0, 1e-5));

    // Output stage: the mono fold matrix for the current layout.
    let matrix = profile
        .downmix_matrix(ChannelLayout::Stereo)
        .expect("downmix enabled must yield a matrix");
    assert_eq!(matrix.source_layout(), ChannelLayout::Stereo);

    // Dynamics stage: dialogue boost and compression stamped onto core params.
    let base_duck = DuckingParams::default();
    let ducked = profile.apply_ducking(base_duck);
    assert!(close(ducked.range_db, base_duck.range_db + 10.0, EPS));
    assert!(close(ducked.threshold_db, base_duck.threshold_db - 12.0, EPS));

    let base_comp = CompressorParams::default();
    let compressed = profile.apply_compressor(base_comp);
    assert!(close(compressed.ratio, 5.0, EPS));
    assert!(close(compressed.threshold_db, -30.0, EPS));
    // Timing is preserved from the base compressor, not from the profile.
    assert!(close(compressed.attack_ms, base_comp.attack_ms, EPS));
    assert!(close(compressed.release_ms, base_comp.release_ms, EPS));
}

/// Running the same chain twice must yield bit-identical parameters, verified
/// through `to_bits()` rather than tolerant float comparison.
#[test]
fn full_chain_is_bit_deterministic() {
    let run = || {
        let profile = fully_assisted_profile();
        let duck = profile.apply_ducking(DuckingParams::default());
        let comp = profile.apply_compressor(CompressorParams::default());
        let matrix = profile
            .downmix_matrix(ChannelLayout::Surround5_1)
            .expect("downmix enabled");
        (duck, comp, matrix)
    };

    let (duck_a, comp_a, matrix_a) = run();
    let (duck_b, comp_b, matrix_b) = run();

    assert_eq!(duck_a.range_db.to_bits(), duck_b.range_db.to_bits());
    assert_eq!(duck_a.threshold_db.to_bits(), duck_b.threshold_db.to_bits());
    assert_eq!(comp_a.ratio.to_bits(), comp_b.ratio.to_bits());
    assert_eq!(comp_a.threshold_db.to_bits(), comp_b.threshold_db.to_bits());
    assert_eq!(comp_a.makeup_db.to_bits(), comp_b.makeup_db.to_bits());

    assert_eq!(matrix_a.coefficients().len(), matrix_b.coefficients().len());
    for (ca, cb) in matrix_a.coefficients().iter().zip(matrix_b.coefficients().iter()) {
        assert_eq!(ca.to_bits(), cb.to_bits());
    }
}

/// Caption metadata and its report carry the full authoring context toward the
/// telemetry ring, including the audio-description flag and timeline position.
#[test]
fn caption_report_rides_telemetry_context() {
    let description = CaptionMetadata::new("distant thunder".to_string(), "en-GB".to_string(), 900)
        .with_speaker("Ambience".to_string())
        .as_audio_description()
        .with_duration_ms(1_000);
    assert!(description.is_audio_description());
    assert!(description.has_speaker());
    assert_eq!(description.speaker(), "Ambience");
    assert_eq!(description.language(), "en-GB");
    assert_eq!(description.duration_ms(), 1_000);
    assert_eq!(description.text(), "distant thunder");

    let report = CaptionReportBuilder::new(description.clone())
        .event_id(900_001)
        .timestamp_samples(192_000)
        .build();
    assert_eq!(report.event_id(), 900_001);
    assert_eq!(report.timestamp_samples(), 192_000);
    assert!(report.caption().is_audio_description());
}

/// The equal-power downmix preserves acoustic power: the sum of squared
/// coefficients is unity for every layout, and every contributing channel
/// shares the same gain.
#[test]
fn mono_downmix_equal_power_preserves_acoustic_power() {
    let layouts = [
        ChannelLayout::Mono,
        ChannelLayout::Stereo,
        ChannelLayout::Quad,
        ChannelLayout::Surround5_1,
        ChannelLayout::Surround7_1,
        ChannelLayout::AmbisonicFoa,
    ];
    for layout in layouts {
        let matrix = DownmixMatrix::equal_power(layout);
        let power: Sample = matrix.coefficients().iter().map(|c| c * c).sum();
        assert!(close(power, 1.0, 1e-5), "power not unity for {layout:?}");
    }

    // Surround 5.1 excludes the LFE channel (index 3) from the fold.
    let surround = DownmixMatrix::equal_power(ChannelLayout::Surround5_1);
    assert!(fabs(surround.coefficients()[3]) < EPS);

    // First-order ambisonics folds to the omnidirectional `W` component only.
    let foa = DownmixMatrix::equal_power(ChannelLayout::AmbisonicFoa);
    assert!(close(foa.coefficients()[0], 1.0, EPS));
    for &c in &foa.coefficients()[1..] {
        assert!(fabs(c) < EPS);
    }
}

/// Folding a real stereo `AudioBuffer` through the matrix matches the
/// per-frame dot product and lands in a mono buffer with the same active span.
#[test]
fn mono_downmix_folds_buffer_to_mono() {
    let frames = 6usize;
    let mut src = AudioBuffer::new(ChannelLayout::Stereo, frames);
    src.set_active_frames(frames);
    for f in 0..frames {
        src.channel_mut(0)[f] = 0.5;
        src.channel_mut(1)[f] = 0.5;
    }

    let matrix = DownmixMatrix::equal_power(ChannelLayout::Stereo);
    let mono = matrix.fold_buffer(&src);
    assert_eq!(mono.layout(), ChannelLayout::Mono);
    assert_eq!(mono.active_frames(), frames);

    let gain = 1.0 / ops::sqrt(2.0);
    let expected = 0.5 * gain + 0.5 * gain;
    let per_frame = matrix.fold_frame(&[0.5, 0.5]);
    assert!(close(per_frame, expected, EPS));
    for (f, &sample) in mono.channel(0).iter().enumerate().take(frames) {
        assert!(close(sample, expected, EPS), "frame {f} mismatch");
    }
}

/// The dialogue boost strengthens ducking monotonically: a stronger level adds
/// more range and lowers the key threshold further.
#[test]
fn dialogue_boost_strengthens_ducking_monotonically() {
    let base = DuckingParams::default();
    let strengths = [
        BoostStrength::Off,
        BoostStrength::Low,
        BoostStrength::Medium,
        BoostStrength::High,
    ];

    let mut prev_range = Sample::NEG_INFINITY;
    let mut prev_threshold = Sample::INFINITY;
    for (idx, &strength) in strengths.iter().enumerate() {
        let out = DialogueBoost::new(strength).apply(base);
        if idx == 0 {
            // `Off` is the exact identity.
            assert!(close(out.range_db, base.range_db, EPS));
            assert!(close(out.threshold_db, base.threshold_db, EPS));
        } else {
            assert!(out.range_db > prev_range, "range not increasing at {strength:?}");
            assert!(out.threshold_db < prev_threshold, "threshold not lowering at {strength:?}");
        }
        prev_range = out.range_db;
        prev_threshold = out.threshold_db;
    }
}

/// Compression presets narrow the dynamic-range window with increasing
/// strength, and `apply` overwrites only the window while preserving timing.
#[test]
fn compression_presets_narrow_dynamic_range() {
    let off = CompressionProfile::from_preset(CompressionPreset::Off);
    let night = CompressionProfile::from_preset(CompressionPreset::Night);
    let hoh = CompressionProfile::from_preset(CompressionPreset::HardOfHearing);

    assert!(!off.is_active());
    assert!(night.is_active());
    assert!(hoh.is_active());
    assert!(hoh.ratio() > night.ratio());
    assert!(hoh.makeup_db() > night.makeup_db());
    assert!(hoh.threshold_db() < night.threshold_db());

    let base = CompressorParams {
        attack_ms: 12.0,
        release_ms: 240.0,
        ..CompressorParams::default()
    };
    let out = night.apply(base);
    assert!(close(out.attack_ms, 12.0, EPS));
    assert!(close(out.release_ms, 240.0, EPS));
    assert!(close(out.threshold_db, -24.0, EPS));
    assert!(close(out.ratio, 3.0, EPS));
    assert!(close(out.knee_db, 8.0, EPS));
    assert!(close(out.makeup_db, 3.0, EPS));
}

/// A visual cue's direction vector is unit length and oriented per the engine
/// convention: forward is `-Z`, right azimuth is `+X`, and intensity is
/// clamped into `[0, 1]`.
#[test]
fn visual_cue_direction_is_unit_and_oriented() {
    let forward = VisualCue::new(CueKind::Voice, 0.0, 0.0, 0.8);
    let fdir = forward.direction_vector();
    assert!(fabs(fdir.x) < 1e-5);
    assert!(fabs(fdir.y) < 1e-5);
    assert!(close(fdir.z, -1.0, 1e-5));
    assert_eq!(forward.kind(), CueKind::Voice);

    let oblique = VisualCue::new(CueKind::Impact, 0.7, 0.3, 1.5).with_intensity(2.0);
    assert!(close(oblique.intensity(), 1.0, EPS));
    let d = oblique.direction_vector();
    let len_sq = d.x * d.x + d.y * d.y + d.z * d.z;
    assert!(close(len_sq, 1.0, 1e-5));

    let clamped_low = oblique.with_intensity(-3.0);
    assert!(fabs(clamped_low.intensity()) < EPS);
}

/// Disabled accommodations gate their outputs: a profile with nothing enabled
/// reports inactive, produces no downmix matrix, and leaves the dynamics
/// params as the identity.
#[test]
fn disabled_accommodations_gate_outputs() {
    let profile = AccessibilityProfile::default();
    assert!(!profile.is_any_active());
    assert!(!profile.captions_enabled());
    assert!(!profile.visual_cues_enabled());
    assert!(profile.downmix_matrix(ChannelLayout::Stereo).is_none());

    let base_duck = DuckingParams::default();
    let duck = profile.apply_ducking(base_duck);
    assert!(close(duck.range_db, base_duck.range_db, EPS));
    assert!(close(duck.threshold_db, base_duck.threshold_db, EPS));

    let base_comp = CompressorParams::default();
    let comp = profile.apply_compressor(base_comp);
    // `Off` forces unity ratio regardless of the base ratio.
    assert!(close(comp.ratio, 1.0, EPS));

    // A toggled-off downmix also gates its own matrix helper.
    let mut toggle = MonoDownmix::new(true);
    assert!(toggle.matrix(ChannelLayout::Stereo).is_some());
    toggle.set_enabled(false);
    assert!(toggle.matrix(ChannelLayout::Stereo).is_none());
}

/// Composing two profiles merges each accommodation to its stronger operand:
/// toggles OR together, the dialogue boost and compression take the higher
/// intensity, and the downmix enables when either side does.
#[test]
fn compose_merges_to_strongest_accommodation() {
    let a = AccessibilityProfile::new()
        .with_captions(true)
        .with_downmix(MonoDownmix::new(true))
        .with_dialogue_boost(DialogueBoost::new(BoostStrength::Low))
        .with_compression(CompressionProfile::from_preset(CompressionPreset::Night));
    let b = AccessibilityProfile::new()
        .with_visual_cues(true)
        .with_dialogue_boost(DialogueBoost::new(BoostStrength::High))
        .with_compression(CompressionProfile::from_preset(CompressionPreset::Off));

    let merged = a.compose(&b);
    assert!(merged.captions_enabled());
    assert!(merged.visual_cues_enabled());
    assert!(merged.downmix().is_enabled());
    assert_eq!(merged.dialogue_boost().strength(), BoostStrength::High);
    assert_eq!(merged.compression().preset(), CompressionPreset::Night);

    // Compose is symmetric for the merged outcome.
    let merged_rev = b.compose(&a);
    assert_eq!(
        merged_rev.dialogue_boost().strength(),
        merged.dialogue_boost().strength()
    );
    assert_eq!(merged_rev.compression().preset(), merged.compression().preset());
    assert_eq!(merged_rev.downmix().is_enabled(), merged.downmix().is_enabled());
}

/// Boundary: the dialogue boost clamps the lowered key threshold to a `-80 dB`
/// floor rather than running off to arbitrarily negative values.
#[test]
fn dialogue_threshold_respects_floor() {
    let base = DuckingParams {
        threshold_db: -75.0,
        ..DuckingParams::default()
    };
    let out = DialogueBoost::new(BoostStrength::High).apply(base);
    assert!(out.threshold_db >= -80.0);
    // The range still grows normally even when the threshold is clamped.
    assert!(close(out.range_db, base.range_db + 10.0, EPS));
}

/// The aggregate profile's `apply` helpers must be exactly equivalent to
/// applying the underlying components directly, proving the profile only
/// delegates and does not re-derive the window.
#[test]
fn profile_apply_helpers_match_component_apply() {
    let boost = DialogueBoost::new(BoostStrength::Medium);
    let compression = CompressionProfile::from_preset(CompressionPreset::Night);
    let profile = AccessibilityProfile::new()
        .with_dialogue_boost(boost)
        .with_compression(compression);

    let base_duck = DuckingParams::default();
    let via_profile_duck = profile.apply_ducking(base_duck);
    let via_component_duck = boost.apply(base_duck);
    assert_eq!(via_profile_duck.range_db.to_bits(), via_component_duck.range_db.to_bits());
    assert_eq!(
        via_profile_duck.threshold_db.to_bits(),
        via_component_duck.threshold_db.to_bits()
    );

    let base_comp = CompressorParams::default();
    let via_profile_comp = profile.apply_compressor(base_comp);
    let via_component_comp = compression.apply(base_comp);
    assert_eq!(via_profile_comp.ratio.to_bits(), via_component_comp.ratio.to_bits());
    assert_eq!(
        via_profile_comp.threshold_db.to_bits(),
        via_component_comp.threshold_db.to_bits()
    );
    assert_eq!(via_profile_comp.knee_db.to_bits(), via_component_comp.knee_db.to_bits());
    assert_eq!(via_profile_comp.makeup_db.to_bits(), via_component_comp.makeup_db.to_bits());
}
