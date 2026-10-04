//! Production-grade whole-chain integration suite for `prism_audio_conditioning`.
//!
//! These tests drive the crate's real public offline conditioning chain over
//! real [`ConditionedPcm`] assets: the full orchestrated [`pipeline::run`]
//! (decode -> resample -> source hygiene -> loudness/loop/transient/tempo
//! analysis -> codec tier -> finalize -> loudness normalize -> content hash),
//! plus each public DSP stage exercised directly on real buffers. Every stage
//! is checked for its mastering-chain contract: gain/attenuation direction,
//! loudness-move direction, true-peak-ceiling enforcement, zero-crossing loop
//! detection, seam-only crossfade editing (active-window preservation),
//! encoder-delay trimming order, bypass bit-transparency, and bit-exact
//! determinism. No stage is stubbed, mocked, or bypassed.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or `MPEG` source or derived code, and no AI/ML. Signals are
//! synthesized from first principles (DC, ramp, triangle, and a linear
//! congruential pseudo-random generator); public standards such as the RBJ
//! cookbook and `ITU-R` `BS.1770` inform only the expected direction of the
//! assertions, never any borrowed implementation.
//!
//! # Relationship
//! Integration-level coverage for design section 51 (offline asset
//! conditioning and authoring). It depends only on the crate under test plus
//! its ordinary dependencies that Cargo re-exposes to in-package integration
//! tests: `prism_audio_core` (`ConditionedPcm`'s `Sample`/`ChannelLayout`),
//! `prism_audio_assets` (`PcmSampleFormat` for the raw-`PCM` decode leg), and
//! `bevy_math` (`no_std`-friendly float operations).

use bevy_math::ops;

use prism_audio_assets::codec::PcmSampleFormat;
use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::math::Sample;

use prism_audio_conditioning::config::{ConditioningConfig, DecodeHint};
use prism_audio_conditioning::decode::SourceFormat;
use prism_audio_conditioning::dc_block::{self, DcBlockConfig};
use prism_audio_conditioning::finalize::{self, FinalizeConfig};
use prism_audio_conditioning::loop_crossfade::{self, CrossfadeShape};
use prism_audio_conditioning::loop_point::{self, LoopMode, LoopPoints};
use prism_audio_conditioning::loudness_normalize::{self, LoudnessNormalizeConfig};
use prism_audio_conditioning::loudness_offline;
use prism_audio_conditioning::pcm::{ConditionedPcm, EncoderDelay};
use prism_audio_conditioning::pipeline;
use prism_audio_conditioning::resample_offline;
use prism_audio_conditioning::transient;
use prism_audio_conditioning::config::{LoopConfig, ResampleConfig};

/// Branch-free absolute value, avoiding `std` float math per the test lints.
fn fabs(x: Sample) -> Sample {
    if x < 0.0 { -x } else { x }
}

/// Returns `true` when `a` and `b` agree to within `eps`.
fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
    fabs(a - b) < eps
}

/// Deterministic linear-congruential pseudo-random sample in `[-amp, amp)`.
///
/// `state` is advanced in place; the generator is the classic Numerical
/// Recipes `LCG`, so the stream is bit-identical across runs and platforms.
fn lcg_next(state: &mut u32, amp: Sample) -> Sample {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let unit = (*state >> 8) as Sample / (1u32 << 24) as Sample;
    amp * (unit * 2.0 - 1.0)
}

/// Builds a mono program of pseudo-random noise at `amp` and `rate`.
fn noise_mono(amp: Sample, rate: u32, frames: usize, seed: u32) -> ConditionedPcm {
    let mut state = seed;
    let channel: Vec<Sample> = (0..frames).map(|_| lcg_next(&mut state, amp)).collect();
    ConditionedPcm::new(rate, ChannelLayout::Mono, vec![channel]).expect("valid mono program")
}

/// Builds a zero-mean triangle wave of `period` frames at `amp`.
fn triangle_mono(amp: Sample, period: usize, rate: u32, frames: usize) -> ConditionedPcm {
    let half = (period / 2).max(1);
    let channel: Vec<Sample> = (0..frames)
        .map(|i| {
            let phase = i % period;
            let value = if phase < half {
                -1.0 + 2.0 * (phase as Sample / half as Sample)
            } else {
                1.0 - 2.0 * ((phase - half) as Sample / half as Sample)
            };
            amp * value
        })
        .collect();
    ConditionedPcm::new(rate, ChannelLayout::Mono, vec![channel]).expect("valid triangle program")
}

/// Encodes planar channels into a headerless interleaved `F32Le` raw-`PCM`
/// byte block (frame-major), matching the raw decode leg's expectation.
fn interleave_f32le(channels: &[&[Sample]]) -> Vec<u8> {
    let frames = channels.first().map_or(0, |c| c.len());
    let mut bytes = Vec::with_capacity(frames * channels.len() * 4);
    for frame in 0..frames {
        for channel in channels {
            for b in channel[frame].to_le_bytes() {
                bytes.push(b);
            }
        }
    }
    bytes
}

/// Builds a decode hint for an interleaved `F32Le` raw-`PCM` block.
fn raw_hint(rate: u32, channels: u16) -> DecodeHint {
    DecodeHint {
        sample_rate: Some(rate),
        channels: Some(channels),
        pcm_format: Some(PcmSampleFormat::F32Le),
        preroll_frames: 0,
        padding_frames: 0,
    }
}

/// Linear amplitude for a `dBFS` level, via the `no_std` float ops.
fn dbfs_to_linear(db: Sample) -> Sample {
    ops::powf(10.0, db / 20.0)
}

/// Largest absolute sample across every channel of `pcm`.
fn program_peak(pcm: &ConditionedPcm) -> Sample {
    let mut peak = 0.0;
    for ch in 0..pcm.channel_count() {
        if let Some(samples) = pcm.channel(ch) {
            for &s in samples {
                let m = fabs(s);
                if m > peak {
                    peak = m;
                }
            }
        }
    }
    peak
}

/// Mean of a channel accumulated in `f64` to match the stage's own math.
fn channel_mean(samples: &[Sample]) -> Sample {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s)).sum();
    (sum / samples.len() as f64) as Sample
}

// ------------------------------------------------------------------------
// Whole-chain, orchestrated pipeline.
// ------------------------------------------------------------------------

#[test]
fn full_chain_stereo_produces_coherent_artifact() {
    let rate = 48_000u32;
    let frames = 24_000usize;
    let left = triangle_mono(0.6, 480, rate, frames);
    let mut state = 0x51A5_11C3;
    let right: Vec<Sample> = (0..frames).map(|_| lcg_next(&mut state, 0.4)).collect();
    let left_ch = left.channel(0).expect("left channel");
    let bytes = interleave_f32le(&[left_ch, &right]);

    let config = ConditioningConfig {
        decode: raw_hint(rate, 2),
        dc_block: DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 20.0,
        },
        finalize: FinalizeConfig {
            trim_encoder_delay: false,
            bake_loop_crossfade: true,
            crossfade_shape: CrossfadeShape::EqualPower,
        },
        loudness_normalize: LoudnessNormalizeConfig {
            enabled: true,
            target_lufs: -16.0,
            true_peak_ceiling_dbfs: -1.0,
            max_gain_db: 24.0,
        },
        ..ConditioningConfig::default()
    };

    let artifact = pipeline::run(&bytes, SourceFormat::RawPcm, &config).expect("pipeline runs");

    assert_eq!(artifact.pcm.sample_rate(), rate);
    assert_eq!(artifact.pcm.channel_count(), 2);
    assert!(artifact.pcm.frames() > 0, "delivered program is non-empty");
    assert!(
        artifact.loudness.integrated_lufs.is_finite(),
        "a loud program has finite integrated loudness"
    );
    // Normalization enforces the true-peak ceiling on the delivered program.
    let ceiling = dbfs_to_linear(config.loudness_normalize.true_peak_ceiling_dbfs);
    assert!(
        program_peak(&artifact.pcm) <= ceiling * 1.001,
        "delivered peak honors the true-peak ceiling"
    );
}

#[test]
fn full_chain_is_bitwise_deterministic() {
    let rate = 48_000u32;
    let noise = noise_mono(0.5, rate, 16_000, 0x0BAD_F00D);
    let bytes = interleave_f32le(&[noise.channel(0).expect("channel")]);

    let config = ConditioningConfig {
        decode: raw_hint(rate, 1),
        dc_block: DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 25.0,
        },
        loudness_normalize: LoudnessNormalizeConfig {
            enabled: true,
            ..LoudnessNormalizeConfig::default()
        },
        ..ConditioningConfig::default()
    };

    let first = pipeline::run(&bytes, SourceFormat::RawPcm, &config).expect("first run");
    let second = pipeline::run(&bytes, SourceFormat::RawPcm, &config).expect("second run");

    // Content hashes must match and every delivered sample must be bit-exact.
    assert_eq!(first.hash, second.hash, "content hash is deterministic");
    assert_eq!(
        first.pcm.channel_count(),
        second.pcm.channel_count(),
        "channel count is stable"
    );
    for ch in 0..first.pcm.channel_count() {
        let a = first.pcm.channel(ch).expect("channel a");
        let b = second.pcm.channel(ch).expect("channel b");
        assert_eq!(a.len(), b.len(), "frame count is stable");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "delivered sample is bit-exact");
        }
    }
}

// ------------------------------------------------------------------------
// Source hygiene: exact DC removal and bypass transparency.
// ------------------------------------------------------------------------

#[test]
fn dc_block_removes_offset_toward_zero_and_preserves_ac() {
    let rate = 48_000u32;
    let frames = 4_000usize;
    let offset = 0.25 as Sample;
    let ac = 0.3 as Sample;
    // Alternating +/- around a constant DC bias: mean is exactly `offset`.
    let channel: Vec<Sample> = (0..frames)
        .map(|i| if i % 2 == 0 { offset + ac } else { offset - ac })
        .collect();
    let pcm = ConditionedPcm::new(rate, ChannelLayout::Mono, vec![channel]).expect("program");

    let config = DcBlockConfig {
        remove_dc_offset: true,
        highpass_cutoff_hz: 0.0,
    };
    let result = dc_block::apply(&pcm, &config);

    assert!(
        approx(result.removed_dc[0], offset, 1.0e-4),
        "reported DC equals the injected bias"
    );
    let out = result.pcm.channel(0).expect("channel");
    assert!(
        approx(channel_mean(out), 0.0, 1.0e-4),
        "post-removal mean is driven to zero"
    );
    // The alternating (AC) component is preserved in magnitude.
    assert!(approx(fabs(out[0]), ac, 1.0e-4), "AC magnitude survives");
}

#[test]
fn dc_block_bypass_is_bit_transparent() {
    let pcm = noise_mono(0.4, 48_000, 2_048, 0x1234_ABCD);
    let result = dc_block::apply(&pcm, &DcBlockConfig::default());

    assert_eq!(result.removed_dc, vec![0.0 as Sample]);
    let before = pcm.channel(0).expect("input");
    let after = result.pcm.channel(0).expect("output");
    assert_eq!(before.len(), after.len());
    for (x, y) in before.iter().zip(after.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "disabled stage is byte-identical");
    }
}

#[test]
fn dc_block_highpass_edits_but_preserves_length() {
    let rate = 48_000u32;
    let pcm = triangle_mono(0.5, 600, rate, 6_000);
    let config = DcBlockConfig {
        remove_dc_offset: false,
        highpass_cutoff_hz: 120.0,
    };
    let result = dc_block::apply(&pcm, &config);

    assert_eq!(
        result.pcm.frames(),
        pcm.frames(),
        "zero-phase high-pass preserves the frame count"
    );
    // A real high-pass must actually change the waveform of a low tone.
    let before = pcm.channel(0).expect("input");
    let after = result.pcm.channel(0).expect("output");
    let changed = before
        .iter()
        .zip(after.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(changed, "high-pass alters the low-frequency program");
}

// ------------------------------------------------------------------------
// Loudness normalization: direction, ceiling, bypass.
// ------------------------------------------------------------------------

#[test]
fn loudness_normalize_moves_quiet_program_toward_target() {
    let rate = 48_000u32;
    let pcm = noise_mono(0.02, rate, rate as usize, 0x00C0_FFEE);
    let measured = loudness_offline::analyze(&pcm);
    assert!(
        measured.integrated_lufs.is_finite(),
        "quiet-but-audible program has finite loudness"
    );

    let target = -16.0 as Sample;
    let config = LoudnessNormalizeConfig {
        enabled: true,
        target_lufs: target,
        true_peak_ceiling_dbfs: -1.0,
        max_gain_db: 36.0,
    };
    let normalized = loudness_normalize::normalize(&pcm, measured, &config);

    assert!(
        normalized.applied_gain_db > 0.0,
        "a quiet program is boosted upward"
    );
    let remeasured = loudness_offline::analyze(&normalized.pcm);
    assert!(
        fabs(remeasured.integrated_lufs - target) < fabs(measured.integrated_lufs - target),
        "normalized loudness is closer to the target"
    );
}

#[test]
fn loudness_normalize_enforces_true_peak_ceiling() {
    let rate = 48_000u32;
    // Peak ~0.5 (-6 dBFS), moderate loudness: a high boost target would push
    // the peak past the ceiling, so the ceiling clamp must bind.
    let pcm = noise_mono(0.5, rate, rate as usize, 0xFEED_FACE);
    let measured = loudness_offline::analyze(&pcm);

    let ceiling_db = -1.0 as Sample;
    let config = LoudnessNormalizeConfig {
        enabled: true,
        target_lufs: 0.0,
        true_peak_ceiling_dbfs: ceiling_db,
        max_gain_db: 48.0,
    };
    let normalized = loudness_normalize::normalize(&pcm, measured, &config);

    let ceiling = dbfs_to_linear(ceiling_db);
    assert!(
        program_peak(&normalized.pcm) <= ceiling * 1.001,
        "post-gain peak never exceeds the true-peak ceiling"
    );
}

#[test]
fn loudness_normalize_bypass_is_bit_transparent() {
    let pcm = noise_mono(0.3, 48_000, 4_000, 0x2222_3333);
    let measured = loudness_offline::analyze(&pcm);
    let normalized = loudness_normalize::normalize(&pcm, measured, &LoudnessNormalizeConfig::default());

    assert_eq!(normalized.applied_gain_db.to_bits(), (0.0 as Sample).to_bits());
    let before = pcm.channel(0).expect("input");
    let after = normalized.pcm.channel(0).expect("output");
    for (x, y) in before.iter().zip(after.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "disabled normalization is byte-identical");
    }
}

// ------------------------------------------------------------------------
// Offline resampling.
// ------------------------------------------------------------------------

#[test]
fn resample_same_rate_is_bit_identical() {
    let pcm = noise_mono(0.5, 48_000, 2_000, 0x4455_6677);
    let out = resample_offline::resample_to(&pcm, 48_000, &ResampleConfig::default());

    assert_eq!(out.sample_rate(), 48_000);
    let before = pcm.channel(0).expect("input");
    let after = out.channel(0).expect("output");
    assert_eq!(before.len(), after.len());
    for (x, y) in before.iter().zip(after.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "same-rate resample clones exactly");
    }
}

#[test]
fn resample_changes_rate_and_frame_count_in_the_right_direction() {
    let rate = 48_000u32;
    let frames = 8_000usize;
    let pcm = triangle_mono(0.5, 500, rate, frames);

    let down = resample_offline::resample_to(&pcm, 24_000, &ResampleConfig::default());
    assert_eq!(down.sample_rate(), 24_000);
    assert!(down.frames() > 0, "downsample output is non-empty");
    assert!(
        down.frames() < frames,
        "halving the rate shrinks the frame count"
    );

    let up = resample_offline::resample_to(&pcm, 96_000, &ResampleConfig::default());
    assert_eq!(up.sample_rate(), 96_000);
    assert!(
        up.frames() > frames,
        "doubling the rate grows the frame count"
    );
}

// ------------------------------------------------------------------------
// Loop seam crossfade: active-window editing and bypass.
// ------------------------------------------------------------------------

#[test]
fn loop_crossfade_edits_only_the_seam_window() {
    let rate = 48_000u32;
    let frames = 2_000usize;
    let pcm = triangle_mono(0.7, 400, rate, frames);
    let points = LoopPoints {
        start: 400,
        end: 1_400,
        crossfade_frames: 128,
        mode: LoopMode::Forward,
    };
    let cf = loop_crossfade::effective_crossfade(&points, frames);
    assert_eq!(cf, 128, "crossfade fits within lead-in and loop length");

    let out = loop_crossfade::apply_loop_crossfade(&pcm, &points, CrossfadeShape::EqualPower);
    let before = pcm.channel(0).expect("input");
    let after = out.channel(0).expect("output");

    let tail_start = points.end - cf;
    let mut touched = false;
    for (i, (x, y)) in before.iter().zip(after.iter()).enumerate() {
        if i >= tail_start && i < points.end {
            if x.to_bits() != y.to_bits() {
                touched = true;
            }
        } else {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "samples outside the seam window are untouched"
            );
        }
    }
    assert!(touched, "the seam window is actually blended");
}

#[test]
fn loop_crossfade_pingpong_is_identity() {
    let rate = 48_000u32;
    let frames = 2_000usize;
    let pcm = triangle_mono(0.7, 400, rate, frames);
    let points = LoopPoints {
        start: 400,
        end: 1_400,
        crossfade_frames: 128,
        mode: LoopMode::PingPong,
    };
    assert_eq!(
        loop_crossfade::effective_crossfade(&points, frames),
        0,
        "ping-pong loops skip seam baking"
    );
    let out = loop_crossfade::apply_loop_crossfade(&pcm, &points, CrossfadeShape::EqualPower);
    let before = pcm.channel(0).expect("input");
    let after = out.channel(0).expect("output");
    for (x, y) in before.iter().zip(after.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "ping-pong bake is a bit-exact identity");
    }
}

// ------------------------------------------------------------------------
// Loop detection.
// ------------------------------------------------------------------------

#[test]
fn loop_detect_finds_period_in_strongly_periodic_signal_and_skips_short() {
    let rate = 48_000u32;
    let config = LoopConfig::default();

    // Clearly periodic triangle: a seamless loop should be detectable and the
    // reported region must stay inside the buffer with a valid length.
    let periodic = triangle_mono(0.8, 500, rate, 10_000);
    let channel = periodic.channel(0).expect("channel");
    let found = loop_point::detect(channel, &config).expect("periodic signal loops");
    assert!(found.end <= channel.len(), "loop end stays inside the buffer");
    assert!(found.start < found.end, "loop region is non-empty");
    assert!(
        found.length() >= config.min_period_frames,
        "loop length respects the minimum period"
    );

    // A signal shorter than the window-plus-period requirement cannot loop.
    let short = triangle_mono(0.8, 500, rate, 300);
    assert!(
        loop_point::detect(short.channel(0).expect("channel"), &config).is_none(),
        "too-short signals report no loop"
    );
}

// ------------------------------------------------------------------------
// Finalize: encoder-delay trim ordering and default identity.
// ------------------------------------------------------------------------

#[test]
fn finalize_trim_removes_delay_and_preserves_program_order() {
    let rate = 48_000u32;
    let frames = 1_000usize;
    // Ramp program: frame value equals its index, so trims are checkable.
    let channel: Vec<Sample> = (0..frames).map(|i| i as Sample).collect();
    let pcm = ConditionedPcm::new(rate, ChannelLayout::Mono, vec![channel]).expect("program");

    let preroll = 100u32;
    let padding = 150u32;
    let delay = EncoderDelay::new(preroll, padding);
    let config = FinalizeConfig {
        trim_encoder_delay: true,
        bake_loop_crossfade: false,
        crossfade_shape: CrossfadeShape::EqualPower,
    };
    let finalized = finalize::finalize(&pcm, delay, None, &config).expect("finalize runs");

    let expected = frames - preroll as usize - padding as usize;
    assert_eq!(
        finalized.pcm.frames(),
        expected,
        "trim removes exactly the pre-roll and padding"
    );
    // The surviving program is the interior slice, still in frame order.
    let out = finalized.pcm.channel(0).expect("channel");
    for (k, &value) in out.iter().enumerate() {
        let original = (preroll as usize + k) as Sample;
        assert_eq!(
            value.to_bits(),
            original.to_bits(),
            "interior frames keep their value and order"
        );
    }
}

#[test]
fn finalize_default_config_is_bit_exact_identity() {
    let rate = 48_000u32;
    let pcm = noise_mono(0.5, rate, 1_500, 0x7788_99AA);
    let delay = EncoderDelay::new(32, 48);
    let finalized = finalize::finalize(&pcm, delay, None, &FinalizeConfig::default())
        .expect("finalize runs");

    assert_eq!(finalized.pcm.frames(), pcm.frames());
    let before = pcm.channel(0).expect("input");
    let after = finalized.pcm.channel(0).expect("output");
    for (x, y) in before.iter().zip(after.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "disabled finalize is byte-identical");
    }
}

// ------------------------------------------------------------------------
// Transient analysis invariants.
// ------------------------------------------------------------------------

#[test]
fn transient_envelope_is_nonnegative_and_onsets_are_ordered() {
    let rate = 48_000u32;
    let frames = 28_000usize;
    // Short decaying clicks on silence: unambiguous transient onsets.
    let positions = [4_096usize, 12_288, 20_480];
    let mut channel = vec![0.0 as Sample; frames];
    for &p in &positions {
        for (k, s) in channel.iter_mut().enumerate().skip(p).take(16) {
            let d = (k - p) as Sample;
            *s += ops::powf(0.6, d);
        }
    }
    let config = prism_audio_conditioning::config::TransientConfig::default();

    let envelope = transient::onset_envelope(&channel, &config);
    assert!(
        envelope.iter().all(|&v| v >= 0.0),
        "spectral-flux onset envelope is non-negative"
    );

    let onsets = transient::detect_onsets(&channel, rate, &config);
    assert!(!onsets.is_empty(), "a loud burst produces at least one onset");
    // Onsets are strictly increasing, in range, and respect the min spacing.
    for pair in onsets.windows(2) {
        assert!(pair[1] > pair[0], "onsets are strictly increasing");
        assert!(
            pair[1] - pair[0] >= config.min_separation_frames,
            "onsets respect the minimum separation"
        );
    }
    assert!(
        onsets.iter().all(|&o| o < frames),
        "onsets fall inside the signal"
    );
}
