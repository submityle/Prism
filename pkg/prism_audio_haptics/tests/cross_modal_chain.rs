//! AAA end-to-end integration coverage for the audio-sourced haptic chain.
//!
//! This suite drives the public surface of `prism_audio_haptics` as a single
//! cross-modal pipeline: a dual-band haptic signal (the analogue of what the
//! `HapticTranscoder` emits -- channel zero carries the low-band envelope and
//! channel one the high-band envelope) is scaled by the `HapticGovernor`,
//! retimed with `HapticWaveform::resample`, spatially weighted by
//! `SpatialWeighting`, rendered through the `WideBandBackend`,
//! `DualMotorBackend`, and `SilentBackend`, and finally planned against the
//! group's slowest device by `LatencyAligner`. Each test fixes one concern:
//! determinism, tier scaling, amplitude clamping, directional/distance
//! weighting, empty-input handling, latency compensation, config sanitisation,
//! and resample fidelity.
//!
//! The audio-input stages (`HapticTranscoder::transcode` and `HapticBus`) take
//! a `prism_audio_core` `AudioBuffer`, which is a non-dev dependency that is not
//! re-exported and therefore not constructible from an integration test; the
//! transcoder output is modelled directly as a `HapticWaveform` so the felt
//! side of the chain is exercised with realistic dual-band data. Likewise
//! `SpatialWeighting::weight_direction` needs a `bevy_math` vector that is not
//! reachable here, so the scalar `weight_pan`/`weight_azimuth` entry points are
//! used instead.
//!
//! # Provenance
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics
//! source or derived code; no AI/ML is involved; only the ideas behind public
//! DSP standards (band envelopes, inverse-distance rolloff, constant-power
//! panning) are drawn upon, never third-party source.
//!
//! # Relationship
//! Exercises design section 36 (haptics and cross-modal output) across its
//! `waveform`, `transcode`, `spatial`, `backend`, `latency`, and `governor`
//! modules. It depends only on the `prism_audio_haptics` crate itself (with its
//! `serialize` dev-dependency feature); it deliberately avoids the
//! `prism_audio_core` and `bevy_math` types that the audio-input and
//! vector-direction entry points require because those crates are not available
//! to an integration test target.

use prism_audio_haptics::backend::{
    DualMotorBackend, HapticBackend, SilentBackend, WideBandBackend,
};
use prism_audio_haptics::governor::{HapticGovernor, HapticTier};
use prism_audio_haptics::latency::LatencyAligner;
use prism_audio_haptics::spatial::SpatialWeighting;
use prism_audio_haptics::transcode::TranscodeConfig;
use prism_audio_haptics::waveform::{ActuatorLayout, HapticWaveform};

/// Absolute value without pulling in `std` floating-point intrinsics.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Approximate equality for felt-domain levels.
fn approx(a: f32, b: f32) -> bool {
    fabs(a - b) <= 1.0e-6
}

/// A tiny deterministic linear congruential generator mapped to `[0, 1)`.
///
/// Uses the well-known 64-bit multiplier/increment pair; the top 24 state bits
/// are taken so the result never relies on floating-point math to synthesise a
/// noise-like envelope.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    bits as f32 / 16_777_216.0
}

/// Builds a dual-band haptic waveform standing in for the transcoder output.
///
/// Channel zero is a deterministic rising ramp in `[0, 1)` (a low-band
/// envelope) and channel one is LCG noise in `[0, 1)` (a high-band envelope),
/// so downstream stages see non-trivial, reproducible data.
fn synth_dual(frames: usize, rate_hz: u32) -> HapticWaveform {
    let mut wf = HapticWaveform::new(rate_hz, ActuatorLayout::Dual);
    let span = frames.max(1) as f32;
    let mut seed = 0x1234_5678_9abc_def0_u64;
    for i in 0..frames {
        let low = i as f32 / span;
        let high = lcg(&mut seed);
        wf.push_frame(&[low, high]);
    }
    wf
}

/// Captured end state of one full pipeline run.
struct PipelineOut {
    low_motor: f32,
    high_motor: f32,
    wide_peak: f32,
    wide_frames: u64,
    silent_submissions: u64,
    retimed_len: usize,
    compensation: u32,
}

/// Runs the whole audio-sourced haptic chain for one block.
///
/// Returns `None` when the governor bypasses haptics entirely, mirroring the
/// production contract where no waveform is generated at the bypass tier.
fn run_pipeline(frames: usize, pan: f32, distance: f32, tier: HapticTier) -> Option<PipelineOut> {
    let config = TranscodeConfig::default();
    let scaled = HapticGovernor::new(tier).apply(config)?;

    // Transcoder-output analogue at the base haptic rate, then retimed to the
    // governor-selected rate just as the backend would retime it.
    let source = synth_dual(frames, config.haptic_rate_hz);
    let retimed = source.resample(scaled.haptic_rate_hz);
    let retimed_len = retimed.len();

    // Directional/distance weighting splits the energy across the actuator pair.
    let gains = SpatialWeighting::new(1.0, 100.0, 1.0).weight_pan(pan, distance);
    let mut spatial_wf = HapticWaveform::new(retimed.rate_hz(), ActuatorLayout::Dual);
    let low_ch = retimed.channel(0);
    let high_ch = retimed.channel(1);
    for (lo_in, hi_in) in low_ch.iter().zip(high_ch) {
        spatial_wf.push_frame(&[*lo_in * gains.left, *hi_in * gains.right]);
    }

    // Render through all three backend kinds.
    let mut dual = DualMotorBackend::new(8, scaled.haptic_rate_hz);
    let mut wide = WideBandBackend::new(32, scaled.haptic_rate_hz);
    let mut silent = SilentBackend::new();
    dual.submit(&spatial_wf);
    wide.submit(&spatial_wf);
    silent.submit(&spatial_wf);
    let (low_motor, high_motor) = dual.motors();

    // Plan compensation so the fastest active backend lines up with the slowest.
    let aligner = LatencyAligner::new(48_000, scaled.haptic_rate_hz);
    let dual_audio = aligner.haptic_to_audio(dual.intrinsic_latency_samples());
    let wide_audio = aligner.haptic_to_audio(wide.intrinsic_latency_samples());
    let target = dual_audio.max(wide_audio);
    let compensation = aligner.compensation_samples(target, dual_audio);

    Some(PipelineOut {
        low_motor,
        high_motor,
        wide_peak: wide.last_peak(),
        wide_frames: wide.frames_received(),
        silent_submissions: silent.submissions(),
        retimed_len,
        compensation,
    })
}

#[test]
fn pipeline_is_deterministic() {
    let first = run_pipeline(400, -0.5, 3.0, HapticTier::Full).expect("full tier runs");
    let second = run_pipeline(400, -0.5, 3.0, HapticTier::Full).expect("full tier runs");

    // Identical inputs must yield bit-identical felt output and bookkeeping.
    assert_eq!(first.low_motor.to_bits(), second.low_motor.to_bits());
    assert_eq!(first.high_motor.to_bits(), second.high_motor.to_bits());
    assert_eq!(first.wide_peak.to_bits(), second.wide_peak.to_bits());
    assert_eq!(first.wide_frames, second.wide_frames);
    assert_eq!(first.silent_submissions, second.silent_submissions);
    assert_eq!(first.retimed_len, second.retimed_len);
    assert_eq!(first.compensation, second.compensation);

    // The chain actually did work: 400 frames in, 400 felt frames out.
    assert_eq!(first.wide_frames, 400);
    assert_eq!(first.silent_submissions, 1);
    assert!(first.wide_peak > 0.0);
}

#[test]
fn governor_tiers_scale_the_chain() {
    let config = TranscodeConfig::default();
    assert_eq!(config.haptic_rate_hz, 1_000);

    let full = HapticGovernor::new(HapticTier::Full)
        .apply(config)
        .expect("full keeps a config");
    let reduced = HapticGovernor::new(HapticTier::Reduced)
        .apply(config)
        .expect("reduced keeps a config");
    assert_eq!(full.haptic_rate_hz, 1_000);
    assert_eq!(reduced.haptic_rate_hz, 500);
    assert!(HapticGovernor::new(HapticTier::Bypass).apply(config).is_none());

    // The reduced rate halves the retimed frame count end-to-end.
    let full_run = run_pipeline(200, 0.0, 1.0, HapticTier::Full).expect("full runs");
    let reduced_run = run_pipeline(200, 0.0, 1.0, HapticTier::Reduced).expect("reduced runs");
    assert_eq!(full_run.retimed_len, 200);
    assert_eq!(reduced_run.retimed_len, 100);

    // Bypass severs the chain entirely.
    assert!(run_pipeline(200, 0.0, 1.0, HapticTier::Bypass).is_none());

    // A high CPU load maps straight to the bypass tier.
    let mut governor = HapticGovernor::default();
    assert_eq!(governor.tier(), HapticTier::Full);
    governor.set_from_load(0.95);
    assert!(governor.is_bypassed());
    assert!(approx(governor.quality_scale(), 0.0));
}

#[test]
fn dual_motor_clamps_overdriven_bands() {
    // An overdriven, partly negative dual-band block must rectify and clamp.
    let mut overdriven = HapticWaveform::new(1_000, ActuatorLayout::Dual);
    for _ in 0..16 {
        overdriven.push_frame(&[5.0, -9.0]);
    }
    let mut dual = DualMotorBackend::new(0, 1_000);
    dual.submit(&overdriven);
    let (low, high) = dual.motors();
    assert!(approx(low, 1.0), "low motor should clamp to unity, got {low}");
    assert!(approx(high, 1.0), "high motor should clamp to unity, got {high}");
    assert_eq!(dual.capabilities().actuator_count, 2);
    assert!(!dual.capabilities().wide_band);

    // The wide-band backend reports the true (unclamped) peak magnitude.
    let mut wide = WideBandBackend::new(16, 1_000);
    wide.submit(&overdriven);
    assert!(approx(wide.last_peak(), 9.0), "peak={}", wide.last_peak());
    assert!(wide.capabilities().wide_band);

    // A mono block drives both motors from its single channel.
    let mut mono = HapticWaveform::new(1_000, ActuatorLayout::Mono);
    for _ in 0..8 {
        mono.push_frame(&[0.6]);
    }
    let mut mono_pad = DualMotorBackend::new(4, 1_000);
    mono_pad.submit(&mono);
    let (mono_low, mono_high) = mono_pad.motors();
    assert!(approx(mono_low, mono_high));
    assert!(approx(mono_low, 0.6));
}

#[test]
fn spatial_weighting_biases_and_attenuates_actuators() {
    let weighting = SpatialWeighting::new(1.0, 100.0, 1.0);

    // Centre is balanced; a left pan biases the left actuator and vice versa.
    let centre = weighting.weight_pan(0.0, 0.0);
    assert!(approx(centre.left, centre.right));
    let left = weighting.weight_pan(-1.0, 0.0);
    assert!(left.left > left.right);
    let right = weighting.weight_pan(1.0, 0.0);
    assert!(right.right > right.left);

    // Azimuth to the right (+pi/2) favours the right actuator.
    let az_right = weighting.weight_azimuth(core::f32::consts::FRAC_PI_2, 0.0);
    assert!(az_right.right > az_right.left);

    // Within the reference distance attenuation is unity; it then falls off but
    // never reaches zero inside the maximum distance.
    assert!(approx(weighting.attenuation(0.0), 1.0));
    assert!(approx(weighting.attenuation(1.0), 1.0));
    let near = weighting.attenuation(2.0);
    let far = weighting.attenuation(40.0);
    assert!(far < near);
    assert!(far > 0.0);

    // Fed through the pipeline, a left pan leaves more energy on the low (left)
    // actuator channel than on the high (right) channel.
    let panned = run_pipeline(300, -1.0, 2.0, HapticTier::Full).expect("runs");
    assert!(panned.low_motor > panned.high_motor);

    // Pushing the source far out attenuates the felt energy versus up close.
    let close = run_pipeline(300, 0.0, 2.0, HapticTier::Full).expect("runs");
    let distant = run_pipeline(300, 0.0, 90.0, HapticTier::Full).expect("runs");
    assert!(distant.low_motor < close.low_motor);
    assert!(distant.wide_peak < close.wide_peak);
}

#[test]
fn empty_waveform_flows_through_backends() {
    let empty = HapticWaveform::new(1_000, ActuatorLayout::Dual);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    // Resampling empty input stays empty but keeps the requested rate.
    let resampled = empty.resample(500);
    assert!(resampled.is_empty());
    assert_eq!(resampled.rate_hz(), 500);

    // Each backend handles the empty block without panicking.
    let mut dual = DualMotorBackend::new(16, 1_000);
    dual.submit(&empty);
    let (low, high) = dual.motors();
    assert!(approx(low, 0.0));
    assert!(approx(high, 0.0));

    let mut wide = WideBandBackend::new(32, 1_000);
    wide.submit(&empty);
    assert!(approx(wide.last_peak(), 0.0));
    assert_eq!(wide.frames_received(), 0);

    let mut silent = SilentBackend::new();
    silent.submit(&empty);
    silent.submit(&empty);
    assert_eq!(silent.submissions(), 2);
    assert_eq!(silent.intrinsic_latency_samples(), 0);
    assert_eq!(silent.capabilities().actuator_count, 0);
}

#[test]
fn latency_alignment_compensates_slowest_backend() {
    let aligner = LatencyAligner::new(48_000, 1_000);
    assert_eq!(aligner.audio_rate(), 48_000);
    assert_eq!(aligner.haptic_rate(), 1_000);

    // Three devices with distinct intrinsic latencies (in haptic samples).
    let wide = WideBandBackend::new(32, 1_000);
    let dual = DualMotorBackend::new(8, 1_000);
    let silent = SilentBackend::new();

    let wide_audio = aligner.haptic_to_audio(wide.intrinsic_latency_samples());
    let dual_audio = aligner.haptic_to_audio(dual.intrinsic_latency_samples());
    let silent_audio = aligner.haptic_to_audio(silent.intrinsic_latency_samples());
    assert_eq!(wide_audio, 1_536);
    assert_eq!(dual_audio, 384);
    assert_eq!(silent_audio, 0);

    // The slowest device defines the group target; it needs no compensation.
    let target = wide_audio.max(dual_audio).max(silent_audio);
    assert_eq!(target, 1_536);
    assert_eq!(aligner.compensation_samples(target, wide_audio), 0);
    assert!(aligner.compensation_samples(target, dual_audio) > 0);
    assert_eq!(aligner.compensation_samples(target, silent_audio), target);

    // Retarding the playhead saturates at zero rather than underflowing.
    let comp = aligner.compensation_samples(target, silent_audio);
    assert_eq!(aligner.compensated_playhead(10_000, comp), 10_000 - u64::from(comp));
    assert_eq!(aligner.compensated_playhead(100, target), 0);

    // Unit conversion round-trips cleanly on evenly divisible rates.
    let round_trip = aligner.audio_to_haptic(aligner.haptic_to_audio(21));
    assert_eq!(round_trip, 21);
}

#[test]
fn transcode_config_sanitises_then_governor_scales() {
    let bad = TranscodeConfig {
        crossover_hz: f32::NAN,
        attack_ms: f32::INFINITY,
        release_ms: -5.0,
        low_gain: f32::NAN,
        high_gain: 2.0,
        haptic_rate_hz: 0,
    };
    let sane = bad.sanitised();
    assert!(sane.crossover_hz.is_finite());
    assert!(sane.crossover_hz >= 1.0);
    assert!(sane.attack_ms.is_finite());
    assert!(approx(sane.release_ms, 0.0));
    assert!(approx(sane.high_gain, 2.0));
    assert_eq!(sane.haptic_rate_hz, 1);

    // The reduced tier never drives the rate below one hertz.
    let reduced_floor = HapticGovernor::new(HapticTier::Reduced)
        .apply(sane)
        .expect("reduced keeps config");
    assert_eq!(reduced_floor.haptic_rate_hz, 1);

    // A healthy default halves cleanly under the reduced tier.
    let reduced_default = HapticGovernor::new(HapticTier::Reduced)
        .apply(TranscodeConfig::default())
        .expect("reduced keeps config");
    assert_eq!(reduced_default.haptic_rate_hz, 500);

    // Mapping thresholds select the right tiers.
    assert_eq!(HapticGovernor::tier_for_load(0.10), HapticTier::Full);
    assert_eq!(HapticGovernor::tier_for_load(0.80), HapticTier::Reduced);
    assert_eq!(HapticGovernor::tier_for_load(0.95), HapticTier::Bypass);
    assert_eq!(HapticGovernor::tier_for_load(f32::NAN), HapticTier::Bypass);
}

#[test]
fn resample_preserves_order_and_peak_through_backend() {
    // A rising ramp downsamples to half length while staying monotonic.
    let mut ramp = HapticWaveform::new(2_000, ActuatorLayout::Mono);
    for i in 0..100 {
        ramp.push_frame(&[i as f32 / 100.0]);
    }
    let downsampled = ramp.resample(1_000);
    assert_eq!(downsampled.len(), 50);
    let track = downsampled.channel(0);
    assert!(track[0] < track[track.len() - 1]);

    // Upsampling a two-sample ramp interpolates the midpoint exactly.
    let mut edge = HapticWaveform::new(2, ActuatorLayout::Mono);
    edge.push_frame(&[0.0]);
    edge.push_frame(&[1.0]);
    let upsampled = edge.resample(4);
    assert!(approx(upsampled.channel(0)[1], 0.5));

    // The wide-band backend reports the ramp's true peak and frame count.
    let mut wide = WideBandBackend::new(0, 2_000);
    wide.submit(&ramp);
    assert!(approx(wide.last_peak(), 0.99));
    assert_eq!(wide.frames_received(), 100);
}
