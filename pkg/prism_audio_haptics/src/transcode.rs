//! Audio-to-haptic transcoding: band split, rectify, envelope follow, decimate.
//!
//! The transcoder turns the same audio material that drives the speakers into a
//! felt vibration. It downmixes the input to mono, splits it into a low and a
//! high band with two biquad filters (a crossover), rectifies each band, tracks
//! its amplitude envelope with split attack/release ballistics, and finally
//! resamples the full-rate envelopes down to the haptic rate. The low band maps
//! to actuator channel zero and the high band to channel one, which lines up
//! with the low-frequency and high-frequency motors of a dual-motor backend.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the transcode stage of design section 36. The band split reuses
//! the biquad primitive of `prism_audio_core::nodes::biquad` and the envelope
//! ballistics reuse the time-constant helper of
//! `prism_audio_core::nodes::dynamics::detector`, so no filter or detector math
//! is re-implemented here. It emits a [`crate::waveform::HapticWaveform`] for a
//! [`crate::backend::HapticBackend`].

use bevy_math::ops;
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::biquad::{Biquad, BiquadKind};
use prism_audio_core::nodes::dynamics::detector::time_to_coef;

use crate::waveform::{ActuatorLayout, HapticWaveform};

/// Butterworth quality factor for a maximally flat crossover section.
const CROSSOVER_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Tunable parameters for a [`HapticTranscoder`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TranscodeConfig {
    /// Crossover frequency in hertz separating the low and high bands.
    pub crossover_hz: Sample,
    /// Envelope attack time in milliseconds (rising edge).
    pub attack_ms: Sample,
    /// Envelope release time in milliseconds (falling edge).
    pub release_ms: Sample,
    /// Linear gain applied to the low-band envelope.
    pub low_gain: Sample,
    /// Linear gain applied to the high-band envelope.
    pub high_gain: Sample,
    /// Output haptic sample rate in hertz.
    pub haptic_rate_hz: u32,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            crossover_hz: 250.0,
            attack_ms: 3.0,
            release_ms: 60.0,
            low_gain: 1.0,
            high_gain: 1.0,
            haptic_rate_hz: 1_000,
        }
    }
}

impl TranscodeConfig {
    /// Returns a copy with every field forced into a valid range.
    ///
    /// Non-finite values fall back to the default for that field; the haptic
    /// rate is clamped to at least one hertz.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let fix = |v: Sample, def: Sample, lo: Sample| {
            if v.is_finite() {
                v.max(lo)
            } else {
                def
            }
        };
        Self {
            crossover_hz: fix(self.crossover_hz, d.crossover_hz, 1.0),
            attack_ms: fix(self.attack_ms, d.attack_ms, 0.0),
            release_ms: fix(self.release_ms, d.release_ms, 0.0),
            low_gain: fix(self.low_gain, d.low_gain, 0.0),
            high_gain: fix(self.high_gain, d.high_gain, 0.0),
            haptic_rate_hz: self.haptic_rate_hz.max(1),
        }
    }
}

/// Converts audio buffers into haptic waveforms.
///
/// The transcoder owns its filter and envelope state so successive blocks of a
/// stream join seamlessly without clicks. Call [`transcode`](Self::transcode)
/// once per audio block.
#[derive(Debug, Clone)]
pub struct HapticTranscoder {
    sample_rate: u32,
    config: TranscodeConfig,
    low: Biquad,
    high: Biquad,
    low_env: Sample,
    high_env: Sample,
    attack_coef: Sample,
    release_coef: Sample,
}

/// Advances a one-pole amplitude envelope by a single rectified sample.
#[inline]
fn follow(env: Sample, rectified: Sample, attack: Sample, release: Sample) -> Sample {
    let coef = if rectified > env { attack } else { release };
    coef * env + (1.0 - coef) * rectified
}

impl HapticTranscoder {
    /// Builds a transcoder for audio arriving at `sample_rate` hertz.
    ///
    /// The configuration is sanitised, the crossover biquads are designed, and
    /// the envelope coefficients are derived from the attack/release times.
    #[must_use]
    pub fn new(sample_rate: u32, config: TranscodeConfig) -> Self {
        let sr = sample_rate.max(1);
        let config = config.sanitised();
        let low = Biquad::from_params(BiquadKind::LowPass, sr, config.crossover_hz, CROSSOVER_Q, 0.0, 1);
        let high = Biquad::from_params(BiquadKind::HighPass, sr, config.crossover_hz, CROSSOVER_Q, 0.0, 1);
        Self {
            sample_rate: sr,
            config,
            low,
            high,
            low_env: 0.0,
            high_env: 0.0,
            attack_coef: time_to_coef(config.attack_ms, sr),
            release_coef: time_to_coef(config.release_ms, sr),
        }
    }

    /// Returns the active configuration (post-sanitisation).
    #[inline]
    #[must_use]
    pub fn config(&self) -> TranscodeConfig {
        self.config
    }

    /// Returns the audio sample rate the transcoder expects.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Clears the filter and envelope memory for a hard restart.
    pub fn reset(&mut self) {
        self.low.reset();
        self.high.reset();
        self.low_env = 0.0;
        self.high_env = 0.0;
    }

    /// Transcodes one audio block into a haptic waveform at the haptic rate.
    ///
    /// The input is downmixed to mono, split into two bands, envelope-followed
    /// at the audio rate, and then resampled down to
    /// [`TranscodeConfig::haptic_rate_hz`].
    #[must_use]
    pub fn transcode(&mut self, input: &AudioBuffer) -> HapticWaveform {
        let frames = input.active_frames();
        if frames == 0 {
            return HapticWaveform::new(self.config.haptic_rate_hz, ActuatorLayout::Dual);
        }

        let channels = input.channels().max(1);
        let scale = 1.0 / channels as Sample;
        let mut mono = AudioBuffer::new(ChannelLayout::Mono, frames);
        mono.set_active_frames(frames);
        {
            let dst = mono.channel_mut(0);
            for value in dst.iter_mut() {
                *value = 0.0;
            }
            for ch in 0..input.channels() {
                let src = input.channel(ch);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += *s * scale;
                }
            }
        }

        let mut low_buf = mono.clone();
        let mut high_buf = mono;
        self.low.process_inplace(&mut low_buf);
        self.high.process_inplace(&mut high_buf);

        let mut full = HapticWaveform::new(self.sample_rate, ActuatorLayout::Dual);
        {
            let low = low_buf.channel(0);
            let high = high_buf.channel(0);
            for (low_in, high_in) in low.iter().zip(high) {
                self.low_env = follow(
                    self.low_env,
                    ops::abs(*low_in),
                    self.attack_coef,
                    self.release_coef,
                );
                self.high_env = follow(
                    self.high_env,
                    ops::abs(*high_in),
                    self.attack_coef,
                    self.release_coef,
                );
                full.push_frame(&[
                    self.low_env * self.config.low_gain,
                    self.high_env * self.config.high_gain,
                ]);
            }
        }

        full.resample(self.config.haptic_rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn tone(freq: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        let ch = buf.channel_mut(0);
        for (i, s) in ch.iter_mut().enumerate() {
            let t = i as Sample / SR as Sample;
            *s = ops::sin(2.0 * core::f32::consts::PI * freq * t);
        }
        buf
    }

    fn mean(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().copied().sum();
        sum / samples.len() as Sample
    }

    #[test]
    fn default_config_is_sane() {
        let c = TranscodeConfig::default().sanitised();
        assert!(c.crossover_hz > 0.0);
        assert_eq!(c.haptic_rate_hz, 1_000);
    }

    #[test]
    fn sanitise_fixes_non_finite() {
        let bad = TranscodeConfig {
            crossover_hz: Sample::NAN,
            attack_ms: Sample::INFINITY,
            release_ms: -5.0,
            low_gain: Sample::NAN,
            high_gain: 2.0,
            haptic_rate_hz: 0,
        };
        let c = bad.sanitised();
        assert!(c.crossover_hz.is_finite());
        assert!(c.attack_ms.is_finite());
        assert!((c.release_ms - 0.0).abs() < 1e-6);
        assert!((c.high_gain - 2.0).abs() < 1e-6);
        assert_eq!(c.haptic_rate_hz, 1);
    }

    #[test]
    fn empty_input_yields_empty_waveform() {
        let mut tc = HapticTranscoder::new(SR, TranscodeConfig::default());
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 16);
        buf.set_active_frames(0);
        let wf = tc.transcode(&buf);
        assert!(wf.is_empty());
        assert_eq!(wf.rate_hz(), 1_000);
    }

    #[test]
    fn output_rate_and_layout_match_config() {
        let mut tc = HapticTranscoder::new(SR, TranscodeConfig::default());
        let buf = tone(80.0, 4_800);
        let wf = tc.transcode(&buf);
        assert_eq!(wf.rate_hz(), 1_000);
        assert_eq!(wf.channel_count(), 2);
        // 4800 audio frames at 48 kHz -> ~100 ms -> ~100 haptic frames.
        assert_eq!(wf.len(), 100);
    }

    #[test]
    fn low_tone_drives_low_band() {
        let config = TranscodeConfig {
            crossover_hz: 300.0,
            ..TranscodeConfig::default()
        };
        let mut tc = HapticTranscoder::new(SR, config);
        let buf = tone(60.0, 9_600);
        let wf = tc.transcode(&buf);
        let low = mean(wf.channel(0));
        let high = mean(wf.channel(1));
        assert!(low > high * 2.0, "low={low} high={high}");
    }

    #[test]
    fn high_tone_drives_high_band() {
        let config = TranscodeConfig {
            crossover_hz: 300.0,
            ..TranscodeConfig::default()
        };
        let mut tc = HapticTranscoder::new(SR, config);
        let buf = tone(4_000.0, 9_600);
        let wf = tc.transcode(&buf);
        let low = mean(wf.channel(0));
        let high = mean(wf.channel(1));
        assert!(high > low * 2.0, "low={low} high={high}");
    }

    #[test]
    fn silence_stays_silent() {
        let mut tc = HapticTranscoder::new(SR, TranscodeConfig::default());
        let buf = AudioBuffer::new(ChannelLayout::Mono, 1_000);
        let wf = tc.transcode(&buf);
        for ch in 0..wf.channel_count() {
            assert!(mean(wf.channel(ch)).abs() < 1e-6);
        }
    }

    #[test]
    fn reset_clears_envelope() {
        let mut tc = HapticTranscoder::new(SR, TranscodeConfig::default());
        let buf = tone(60.0, 4_800);
        let _ = tc.transcode(&buf);
        tc.reset();
        let silence = AudioBuffer::new(ChannelLayout::Mono, 2_000);
        let wf = tc.transcode(&silence);
        assert!(mean(wf.channel(0)).abs() < 1e-3);
    }
}
