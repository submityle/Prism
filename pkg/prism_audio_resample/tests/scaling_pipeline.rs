//! End-to-end integration coverage for the decoupled resampling and
//! time/pitch-scaling ladder (section 34 of the engine design doc).
//!
//! These tests drive the public streaming API of the whole crate as a user
//! would: pumping block-sized input through each quality grade of
//! [`Resampler`] and each grade of [`TimeStretcher`] and asserting the
//! cross-module invariants the design promises — unity-ratio transparency,
//! DC-gain preservation, output-rate scaling, phase-continuous determinism,
//! decoupled time-stretch vs pitch-shift, and silence-in/silence-out.
//!
//! # Provenance
//!
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Microsoft Project
//! Acoustics source or derived code, and no AI/ML. Classic DSP only; public
//! standards informed ideas, not code.
//!
//! # Relationship
//!
//! Exercises section 34 (`Resampler`, `TimeStretcher`) across the
//! `prism_audio_resample` modules `linear`, `polyphase_sinc`,
//! `high_order_sinc`, `wsola`, and `phase_vocoder`, over the public
//! `prism_audio_core::math::Sample` scalar.

use prism_audio_resample::{
    clamp_ratio, semitones_to_ratio, HighOrderSincResampler, LinearResampler,
    PhaseVocoderStretcher, PolyphaseSincResampler, ResampleQuality, Resampler, TimeStretcher,
    WsolaStretcher, MAX_PITCH, MAX_RATIO, MAX_STRETCH, MIN_PITCH, MIN_RATIO, MIN_STRETCH,
};

type Sample = f32;

/// Absolute value without pulling in `std` float intrinsics.
fn fabs(x: Sample) -> Sample {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// A constant-valued (DC) block of length `n`.
fn dc(n: usize, value: Sample) -> Vec<Sample> {
    vec![value; n]
}

/// A deterministic broadband signal in `[-0.5, 0.5]` from a 32-bit LCG, so
/// determinism assertions compare meaningful non-trivial content.
fn broadband(n: usize, seed: u32) -> Vec<Sample> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (state >> 8) as Sample / (1_u32 << 24) as Sample;
        out.push(unit - 0.5);
    }
    out
}

/// Pumps `input` through a resampler in `block`-sized output windows, draining
/// the tail with empty presentations, and returns the concatenated output.
fn pump_resampler(r: &mut dyn Resampler, input: &[Sample], block: usize) -> Vec<Sample> {
    let mut scratch = vec![0.0; block];
    let mut out = Vec::new();
    let mut offset = 0_usize;
    for _ in 0..4_000_000 {
        let progress = r.process(&input[offset.min(input.len())..], &mut scratch);
        offset += progress.consumed;
        out.extend_from_slice(&scratch[..progress.produced]);
        if progress.consumed == 0 && progress.produced == 0 {
            break;
        }
    }
    out
}

/// Pumps `input` through a time stretcher the same way.
fn pump_stretcher(s: &mut dyn TimeStretcher, input: &[Sample], block: usize) -> Vec<Sample> {
    let mut scratch = vec![0.0; block];
    let mut out = Vec::new();
    let mut offset = 0_usize;
    for _ in 0..4_000_000 {
        let progress = s.process(&input[offset.min(input.len())..], &mut scratch);
        offset += progress.consumed;
        out.extend_from_slice(&scratch[..progress.produced]);
        if progress.consumed == 0 && progress.produced == 0 {
            break;
        }
    }
    out
}

/// Mean of the central half of a buffer, used to read steady-state DC gain
/// while ignoring edge warm-up and flush transients.
fn central_mean(signal: &[Sample]) -> Sample {
    if signal.len() < 4 {
        return 0.0;
    }
    let lo = signal.len() / 4;
    let hi = signal.len() - signal.len() / 4;
    let mut sum = 0.0;
    for &x in &signal[lo..hi] {
        sum += x;
    }
    sum / (hi - lo) as Sample
}

/// Each grade reports its own quality tag and the tags form the documented
/// performance-governed ladder `Linear < Sinc < HighOrderSinc`.
#[test]
fn quality_grades_report_and_order_on_the_ladder() {
    assert_eq!(LinearResampler::new().quality(), ResampleQuality::Linear);
    assert_eq!(PolyphaseSincResampler::new().quality(), ResampleQuality::Sinc);
    assert_eq!(
        HighOrderSincResampler::new().quality(),
        ResampleQuality::HighOrderSinc
    );

    assert!(ResampleQuality::Linear < ResampleQuality::Sinc);
    assert!(ResampleQuality::Sinc < ResampleQuality::HighOrderSinc);
    assert!(ResampleQuality::Linear < ResampleQuality::HighOrderSinc);
}

/// At unity ratio every grade passes a DC block through with unity gain, so the
/// steady-state output matches the input level.
#[test]
fn unity_ratio_preserves_dc_level_for_all_grades() {
    let level = 0.5;
    let input = dc(4096, level);

    let mut lin = LinearResampler::new();
    let mut sinc = PolyphaseSincResampler::new();
    let mut high = HighOrderSincResampler::new();

    for r in [
        &mut lin as &mut dyn Resampler,
        &mut sinc as &mut dyn Resampler,
        &mut high as &mut dyn Resampler,
    ] {
        r.set_ratio(1.0);
        let out = pump_resampler(r, &input, 256);
        assert!(!out.is_empty(), "unity ratio must produce output");
        let gain = central_mean(&out);
        assert!(
            fabs(gain - level) < 1.0e-2,
            "steady-state DC gain should stay near unity, got {gain}"
        );
    }
}

/// Upsampling by two roughly doubles the produced sample count and downsampling
/// by half roughly halves it, which is the defining property of sample-rate
/// conversion. The whole input is always consumed.
#[test]
fn ratio_scales_the_produced_rate() {
    let n = 4096;
    let input = dc(n, 0.25);

    let mut up = PolyphaseSincResampler::new();
    up.set_ratio(2.0);
    let out_up = pump_resampler(&mut up, &input, 256);
    let expected_up = 2 * n;
    assert!(
        out_up.len() as isize > expected_up as isize - 512
            && (out_up.len() as isize) < expected_up as isize + 512,
        "upsample by 2 should yield about {expected_up} samples, got {}",
        out_up.len()
    );

    let mut down = PolyphaseSincResampler::new();
    down.set_ratio(0.5);
    let out_down = pump_resampler(&mut down, &input, 256);
    let expected_down = n / 2;
    assert!(
        out_down.len() as isize > expected_down as isize - 512
            && (out_down.len() as isize) < expected_down as isize + 512,
        "downsample by 2 should yield about {expected_down} samples, got {}",
        out_down.len()
    );
}

/// Two freshly reset instances of the top grade produce bit-identical output
/// for identical broadband input and ratio: the streamed pipeline is fully
/// deterministic and reproducible.
#[test]
fn resampling_is_deterministic() {
    let input = broadband(8192, 0x1234_5678);

    let mut a = HighOrderSincResampler::new();
    a.set_ratio(1.5);
    let mut b = HighOrderSincResampler::new();
    b.set_ratio(1.5);

    let out_a = pump_resampler(&mut a, &input, 256);
    let out_b = pump_resampler(&mut b, &input, 256);

    assert_eq!(
        out_a, out_b,
        "identical input and ratio must give bit-identical output after reset"
    );
}

/// `set_ratio` and the public `clamp_ratio` helper fold out-of-range and
/// non-finite ratios into the supported window instead of panicking.
#[test]
fn ratios_clamp_into_the_supported_window() {
    let mut r = PolyphaseSincResampler::new();
    r.set_ratio(1_000.0);
    assert!(fabs(r.ratio() - MAX_RATIO) < 1.0e-6);
    r.set_ratio(0.0);
    assert!(fabs(r.ratio() - MIN_RATIO) < 1.0e-6);
    r.set_ratio(Sample::NAN);
    assert!(fabs(r.ratio() - 1.0) < 1.0e-6);

    assert!(fabs(clamp_ratio(1_000.0) - MAX_RATIO) < 1.0e-6);
    assert!(fabs(clamp_ratio(-5.0) - MIN_RATIO) < 1.0e-6);
    assert!(fabs(clamp_ratio(Sample::INFINITY) - 1.0) < 1.0e-6);
}

/// WSOLA (voice grade) and the phase vocoder (music grade) scale output
/// duration by the time-stretch factor while leaving the pitch ratio at unity:
/// stretch and pitch are independent controls.
#[test]
fn time_stretch_scales_duration_independently_of_pitch() {
    let n = 16_384;
    let input = broadband(n, 0x0bad_f00d);

    for grade in 0..2 {
        let mut wsola;
        let mut vocoder;
        let stretcher: &mut dyn TimeStretcher = if grade == 0 {
            wsola = WsolaStretcher::new();
            &mut wsola
        } else {
            vocoder = PhaseVocoderStretcher::new();
            &mut vocoder
        };

        stretcher.set_pitch_shift(1.0);
        stretcher.set_time_stretch(2.0);
        let stretched = pump_stretcher(stretcher, &input, 512);
        assert!(
            stretched.len() > (n as f32 * 1.5) as usize
                && stretched.len() < (n as f32 * 2.5) as usize,
            "stretch 2x should roughly double duration, got {} from {n}",
            stretched.len()
        );

        stretcher.reset();
        stretcher.set_time_stretch(0.5);
        let compressed = pump_stretcher(stretcher, &input, 512);
        assert!(
            compressed.len() > (n as f32 * 0.3) as usize
                && compressed.len() < (n as f32 * 0.7) as usize,
            "stretch 0.5x should roughly halve duration, got {} from {n}",
            compressed.len()
        );
    }
}

/// A pitch shift with unity time stretch preserves the output duration even
/// though the pipeline resamples internally: the two quantities stay decoupled.
#[test]
fn pitch_shift_preserves_duration() {
    let n = 16_384;
    let input = broadband(n, 0x5eed_cafe);

    let mut wsola = WsolaStretcher::new();
    wsola.set_time_stretch(1.0);
    wsola.set_pitch_shift(semitones_to_ratio(7.0));
    assert!(wsola.pitch_shift() > 1.0, "a positive semitone shift raises pitch");

    let shifted = pump_stretcher(&mut wsola, &input, 512);
    assert!(
        shifted.len() > (n as f32 * 0.8) as usize && shifted.len() < (n as f32 * 1.25) as usize,
        "pitch-only shift should preserve duration, got {} from {n}",
        shifted.len()
    );
}

/// Stretcher controls clamp into their published ranges and silence maps to
/// silence (non-finite input is treated as silence by the hot path).
#[test]
fn stretcher_clamps_and_silence_stays_silent() {
    let mut s = PhaseVocoderStretcher::new();
    s.set_time_stretch(100.0);
    assert!(fabs(s.time_stretch() - MAX_STRETCH) < 1.0e-6);
    s.set_time_stretch(0.0);
    assert!(fabs(s.time_stretch() - MIN_STRETCH) < 1.0e-6);
    s.set_pitch_shift(100.0);
    assert!(fabs(s.pitch_shift() - MAX_PITCH) < 1.0e-6);
    s.set_pitch_shift(0.0);
    assert!(fabs(s.pitch_shift() - MIN_PITCH) < 1.0e-6);

    s.reset();
    s.set_time_stretch(1.5);
    s.set_pitch_shift(1.0);
    let silence = dc(8192, 0.0);
    let out = pump_stretcher(&mut s, &silence, 512);
    for &x in &out {
        assert!(fabs(x) < 1.0e-9, "silence in must yield silence out, saw {x}");
    }
}

/// Two reset stretchers of the same grade and settings produce bit-identical
/// output: the music-grade STFT path is fully deterministic.
#[test]
fn stretching_is_deterministic() {
    let input = broadband(12_288, 0x00c0_ffee);

    let mut a = PhaseVocoderStretcher::new();
    a.set_time_stretch(1.25);
    let mut b = PhaseVocoderStretcher::new();
    b.set_time_stretch(1.25);

    let out_a = pump_stretcher(&mut a, &input, 256);
    let out_b = pump_stretcher(&mut b, &input, 1024);

    assert_eq!(out_a, out_b, "stretch output must be block-size independent");
}
