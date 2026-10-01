//! Envelope follower: tracks the moving amplitude of a signal and emits that
//! envelope as an audio-rate control signal.
//!
//! A follower rectifies the input (peak or RMS), then smooths it with separate
//! attack and release time constants: the envelope rises quickly when the
//! signal grows and eases back slowly when it fades. Unlike the engine's
//! dynamics processors, which consume an internal envelope to compute a gain,
//! this node *outputs the envelope itself* so it can be routed as a modulation
//! or side-chain control -- driving a filter cutoff, an amplitude, a ducking
//! trigger, or any other parameter in a modular patch.
//!
//! The output is a per-channel, non-negative linear amplitude in the same
//! channel layout as the input (channel 0 follows input channel 0, and so on).
//!
//! # Relationship
//!
//! This processor *composes* the dynamics family's shared detection stage
//! rather than duplicating it: measurement reuses
//! [`LevelDetector`](crate::nodes::dynamics::detector::LevelDetector) with its
//! [`DetectionMode`](crate::nodes::dynamics::detector::DetectionMode) taxonomy,
//! and the attack / release coefficients reuse
//! [`time_to_coef`](crate::nodes::dynamics::detector::time_to_coef). The
//! difference from `compressor`/`limiter`/`gate`/`ducking` is purely in the
//! routing: those feed the detected level into a gain computer and apply the
//! result to audio, whereas this node converts the detected level back to a
//! linear amplitude, applies attack / release ballistics, and emits that
//! envelope as its output signal.
//!
//! # Real-time contract
//!
//! The per-channel detectors and envelope state are allocated at construction.
//! [`EnvelopeFollowerNode::process`] performs no allocation, locking, or
//! panicking on the audio thread. Non-finite inputs are treated as silence and
//! the envelope state is flushed of denormals, so the output stays finite.
//!
//! # Provenance
//!
//! Pure classic DSP. The rectify-and-smooth envelope follower with split
//! attack / release ballistics is a textbook building block. There is no AI/ML
//! of any kind, and no UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance
//! Audio/Web Audio source or derived code.

use alloc::vec;
use alloc::vec::Vec;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::dynamics::detector::{DetectionMode, LevelDetector, time_to_coef};

/// Largest attack / release / window time the follower accepts, in ms.
pub const MAX_ENVELOPE_TIME_MS: Sample = 2_000.0;

/// Default attack time in milliseconds.
pub const DEFAULT_ENVELOPE_ATTACK_MS: Sample = 5.0;

/// Default release time in milliseconds.
pub const DEFAULT_ENVELOPE_RELEASE_MS: Sample = 100.0;

/// Default RMS averaging window in milliseconds (used in RMS mode).
pub const DEFAULT_ENVELOPE_RMS_WINDOW_MS: Sample = 10.0;

/// Construction parameters for an [`EnvelopeFollowerNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvelopeFollowerParams {
    /// How the instantaneous level is measured before smoothing.
    pub mode: DetectionMode,
    /// RMS averaging window in milliseconds (ignored in peak mode).
    pub rms_window_ms: Sample,
    /// Attack time in milliseconds (rising edge of the envelope).
    pub attack_ms: Sample,
    /// Release time in milliseconds (falling edge of the envelope).
    pub release_ms: Sample,
}

impl Default for EnvelopeFollowerParams {
    fn default() -> Self {
        Self {
            mode: DetectionMode::Peak,
            rms_window_ms: DEFAULT_ENVELOPE_RMS_WINDOW_MS,
            attack_ms: DEFAULT_ENVELOPE_ATTACK_MS,
            release_ms: DEFAULT_ENVELOPE_RELEASE_MS,
        }
    }
}

impl EnvelopeFollowerParams {
    /// Returns a copy with every time clamped to `[0, MAX_ENVELOPE_TIME_MS]`;
    /// any non-finite time falls back to its default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let fix = |v: Sample, def: Sample| {
            if v.is_finite() {
                v.clamp(0.0, MAX_ENVELOPE_TIME_MS)
            } else {
                def
            }
        };
        Self {
            mode: self.mode,
            rms_window_ms: fix(self.rms_window_ms, d.rms_window_ms),
            attack_ms: fix(self.attack_ms, d.attack_ms),
            release_ms: fix(self.release_ms, d.release_ms),
        }
    }
}

/// An amplitude envelope follower (input port 0 -> output port 0).
///
/// Each channel is followed independently; the output carries the per-channel
/// linear amplitude envelope.
#[derive(Debug, Clone)]
pub struct EnvelopeFollowerNode {
    /// Sample rate in Hz, cached for cold-path coefficient recomputation.
    sample_rate: u32,
    /// Number of channels processed.
    channels: usize,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Maximum block size, in frames, this node can process.
    max_block_frames: usize,
    /// Per-channel level detectors (peak or RMS).
    detectors: Vec<LevelDetector>,
    /// Per-channel smoothed linear envelope state.
    env: Vec<Sample>,
    /// Attack one-pole coefficient (rising edge).
    attack_coef: Sample,
    /// Release one-pole coefficient (falling edge).
    release_coef: Sample,
}

impl EnvelopeFollowerNode {
    /// Builds a follower for `layout` at `sample_rate` Hz that can process up to
    /// `max_block_frames` frames per call. All detectors and envelope state are
    /// allocated here.
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::effects::envelope_follower::{
    ///     EnvelopeFollowerNode, EnvelopeFollowerParams,
    /// };
    ///
    /// let params = EnvelopeFollowerParams::default();
    /// let mut follower =
    ///     EnvelopeFollowerNode::new(48_000, ChannelLayout::Mono, 256, params);
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
    /// let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
    /// input.set_active_frames(256);
    /// let inputs = [input];
    /// let mut output = AudioBuffer::new(ChannelLayout::Mono, 256);
    /// output.set_active_frames(256);
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// follower.process(&ctx, &mut io);
    /// assert_eq!(follower.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_block_frames: usize,
        params: EnvelopeFollowerParams,
    ) -> Self {
        let p = params.sanitised();
        let sr = sample_rate.max(1);
        let channels = layout.channel_count().max(1);
        let cap = max_block_frames.max(1);
        let detectors = vec![LevelDetector::new(p.mode, p.rms_window_ms, sr); channels];
        Self {
            sample_rate: sr,
            channels,
            layout,
            max_block_frames: cap,
            detectors,
            env: vec![0.0; channels],
            attack_coef: time_to_coef(p.attack_ms, sr),
            release_coef: time_to_coef(p.release_ms, sr),
        }
    }

    /// Returns the number of channels this node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the channel layout reported to the host.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the maximum block size, in frames, this node can process.
    #[inline]
    #[must_use]
    pub fn max_block_frames(&self) -> usize {
        self.max_block_frames
    }

    /// Returns the current linear envelope of `channel`, or `0.0` if the
    /// channel index is out of range.
    #[inline]
    #[must_use]
    pub fn envelope(&self, channel: usize) -> Sample {
        self.env.get(channel).copied().unwrap_or(0.0)
    }

    /// Updates every parameter in place (allocation-free). The detectors are
    /// rebuilt (clearing their running average) and the ballistics coefficients
    /// are recomputed. This is a control-thread operation, not called from
    /// [`AudioNode::process`].
    pub fn set_params(&mut self, params: EnvelopeFollowerParams) {
        let p = params.sanitised();
        for det in &mut self.detectors {
            *det = LevelDetector::new(p.mode, p.rms_window_ms, self.sample_rate);
        }
        self.attack_coef = time_to_coef(p.attack_ms, self.sample_rate);
        self.release_coef = time_to_coef(p.release_ms, self.sample_rate);
    }
}

impl AudioNode for EnvelopeFollowerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output
            .active_frames()
            .min(input.active_frames())
            .min(self.max_block_frames);
        if frames == 0 || out_channels == 0 {
            return;
        }
        let ch_n = self.channels;
        let a = self.attack_coef;
        let r = self.release_coef;
        for ch in 0..out_channels {
            if ch >= ch_n {
                // No detector exists for surplus output channels: emit silence.
                for s in output.channel_mut(ch)[..frames].iter_mut() {
                    *s = 0.0;
                }
                continue;
            }
            let det = &mut self.detectors[ch];
            let mut env = self.env[ch];
            let have_in = ch < in_channels;
            for n in 0..frames {
                let x = if have_in {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                } else {
                    0.0
                };
                let level = db_to_linear(det.level_db(x));
                let coef = if level > env { a } else { r };
                env = flush_denormal(coef * env + (1.0 - coef) * level);
                output.channel_mut(ch)[n] = env;
            }
            self.env[ch] = env;
        }
    }

    fn reset(&mut self) {
        for det in &mut self.detectors {
            det.reset();
        }
        for e in &mut self.env {
            *e = 0.0;
        }
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn sine(freq: Sample, amp: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| amp * ops_sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    fn ops_sin(x: Sample) -> Sample {
        bevy_math::ops::sin(x)
    }

    fn run_mono(node: &mut EnvelopeFollowerNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(signal);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0)[..len].to_vec()
    }

    fn mean(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&x| f64::from(x)).sum();
        (sum / samples.len() as f64) as Sample
    }

    #[test]
    fn latency_is_zero() {
        let node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 256, EnvelopeFollowerParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn reports_requested_geometry() {
        let node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Stereo,
            512,
            EnvelopeFollowerParams::default(),
        );
        assert_eq!(node.channels(), 2);
        assert_eq!(node.layout(), ChannelLayout::Stereo);
        assert_eq!(node.max_block_frames(), 512);
    }

    #[test]
    fn default_params_within_domain() {
        let d = EnvelopeFollowerParams::default();
        assert_eq!(d, d.sanitised());
        assert!(d.attack_ms >= 0.0 && d.attack_ms <= MAX_ENVELOPE_TIME_MS);
        assert!(d.release_ms >= 0.0 && d.release_ms <= MAX_ENVELOPE_TIME_MS);
        assert!(d.rms_window_ms >= 0.0 && d.rms_window_ms <= MAX_ENVELOPE_TIME_MS);
    }

    #[test]
    fn sanitise_clamps_and_replaces_non_finite() {
        let p = EnvelopeFollowerParams {
            mode: DetectionMode::Rms,
            rms_window_ms: 1.0e9,
            attack_ms: -5.0,
            release_ms: Sample::NAN,
        }
        .sanitised();
        assert_eq!(p.mode, DetectionMode::Rms);
        assert!((p.rms_window_ms - MAX_ENVELOPE_TIME_MS).abs() < 1e-3);
        assert!((p.attack_ms - 0.0).abs() < 1e-6);
        assert!((p.release_ms - DEFAULT_ENVELOPE_RELEASE_MS).abs() < 1e-6);
    }

    #[test]
    fn silence_produces_zero_envelope() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 1_024, EnvelopeFollowerParams::default());
        let out = run_mono(&mut node, &vec![0.0; 4_096]);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn envelope_is_non_negative() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 4_096, EnvelopeFollowerParams::default());
        let out = run_mono(&mut node, &sine(220.0, 0.7, 4_096));
        assert!(out.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn tracks_constant_amplitude() {
        // Peak mode on a DC level converges to that level.
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, SR as usize, EnvelopeFollowerParams::default());
        let out = run_mono(&mut node, &vec![0.5; SR as usize / 2]);
        let settled = mean(&out[out.len() - 2_000..]);
        assert!((settled - 0.5).abs() < 1e-2, "settled near 0.5: {settled}");
    }

    #[test]
    fn peak_mode_reads_higher_than_rms_mode() {
        let amp = 0.8;
        let signal = sine(1_000.0, amp, SR as usize / 2);
        let mut peak = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Mono,
            SR as usize,
            EnvelopeFollowerParams {
                mode: DetectionMode::Peak,
                ..EnvelopeFollowerParams::default()
            },
        );
        let mut rms = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Mono,
            SR as usize,
            EnvelopeFollowerParams {
                mode: DetectionMode::Rms,
                ..EnvelopeFollowerParams::default()
            },
        );
        let out_peak = run_mono(&mut peak, &signal);
        let out_rms = run_mono(&mut rms, &signal);
        let tail = out_peak.len() - 4_000;
        let peak_level = mean(&out_peak[tail..]);
        let rms_level = mean(&out_rms[tail..]);
        // Peak tracks ~amp; RMS tracks ~amp/sqrt(2) ~= 0.566.
        assert!(peak_level > rms_level + 0.1, "peak {peak_level} vs rms {rms_level}");
        assert!(rms_level > 0.4 && rms_level < 0.7, "rms near amp/sqrt(2): {rms_level}");
    }

    #[test]
    fn attack_rises_toward_signal() {
        let mut node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Mono,
            4_096,
            EnvelopeFollowerParams {
                attack_ms: 10.0,
                release_ms: 500.0,
                ..EnvelopeFollowerParams::default()
            },
        );
        let out = run_mono(&mut node, &vec![0.5; 4_096]);
        // Monotone-ish rise from 0 toward 0.5.
        assert!(out[0] < out[100], "envelope rising");
        assert!(out[4_095] > 0.4, "approaches target: {}", out[4_095]);
    }

    #[test]
    fn shorter_attack_rises_faster() {
        let fast_params = EnvelopeFollowerParams {
            attack_ms: 1.0,
            release_ms: 200.0,
            ..EnvelopeFollowerParams::default()
        };
        let slow_params = EnvelopeFollowerParams {
            attack_ms: 100.0,
            ..fast_params
        };
        let signal = vec![0.5; 4_096];
        let mut fast = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 4_096, fast_params);
        let mut slow = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 4_096, slow_params);
        let out_fast = run_mono(&mut fast, &signal);
        let out_slow = run_mono(&mut slow, &signal);
        // At a fixed early time the fast attack is further along.
        assert!(out_fast[500] > out_slow[500], "fast {} slow {}", out_fast[500], out_slow[500]);
    }

    #[test]
    fn shorter_release_decays_faster() {
        let total = SR as usize;
        let onset = total / 2;
        let mut signal = vec![0.5; total];
        for s in signal.iter_mut().skip(onset) {
            *s = 0.0;
        }
        let fast_params = EnvelopeFollowerParams {
            attack_ms: 2.0,
            release_ms: 20.0,
            ..EnvelopeFollowerParams::default()
        };
        let slow_params = EnvelopeFollowerParams {
            release_ms: 500.0,
            ..fast_params
        };
        let mut fast = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, total, fast_params);
        let mut slow = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, total, slow_params);
        let out_fast = run_mono(&mut fast, &signal);
        let out_slow = run_mono(&mut slow, &signal);
        // A short time after the signal stops, the fast release has fallen more.
        let probe = onset + 2_000;
        assert!(out_fast[probe] < out_slow[probe], "fast {} slow {}", out_fast[probe], out_slow[probe]);
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 512, EnvelopeFollowerParams::default());
        let mut signal = vec![0.0; 2_048];
        signal[0] = Sample::NAN;
        signal[1] = Sample::INFINITY;
        signal[2] = Sample::NEG_INFINITY;
        signal[100] = 0.4;
        let out = run_mono(&mut node, &signal);
        assert!(out.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn tone_output_is_finite() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 512, EnvelopeFollowerParams::default());
        let out = run_mono(&mut node, &sine(330.0, 0.5, SR as usize));
        assert!(out.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 256, EnvelopeFollowerParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn stereo_channels_independent() {
        let len = 4_096;
        let mut node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Stereo,
            len,
            EnvelopeFollowerParams::default(),
        );
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.6;
        }
        // Right channel stays silent.
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert!(outputs[0].channel(0)[len - 1] > 0.4, "left tracks the signal");
        assert!(outputs[0].channel(1)[len - 1] < 1e-6, "right stays silent");
    }

    #[test]
    fn mono_input_into_stereo_is_finite() {
        let mut node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Stereo,
            256,
            EnvelopeFollowerParams::default(),
        );
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
        input.set_active_frames(256);
        output.set_active_frames(256);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        assert!(outputs[0].channel(0).iter().all(|&x| x.is_finite()));
        // The surplus right channel has no detector and stays silent.
        assert!(outputs[0].channel(1).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = EnvelopeFollowerParams {
            mode: DetectionMode::Rms,
            attack_ms: 5.0,
            release_ms: 80.0,
            ..EnvelopeFollowerParams::default()
        };
        let mut node = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 1_024, params);
        let _ = run_mono(&mut node, &sine(440.0, 0.7, 4_096));
        node.reset();
        let probe = sine(330.0, 0.5, 4_096);
        let after = run_mono(&mut node, &probe);
        let mut fresh = EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, 1_024, params);
        let baseline = run_mono(&mut fresh, &probe);
        let max_err = after
            .iter()
            .zip(baseline.iter())
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(max_err < 1e-6, "reset should match a fresh node: {max_err}");
    }

    #[test]
    fn set_params_updates_ballistics() {
        let mut node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Mono,
            4_096,
            EnvelopeFollowerParams {
                attack_ms: 100.0,
                ..EnvelopeFollowerParams::default()
            },
        );
        node.set_params(EnvelopeFollowerParams {
            attack_ms: 1.0,
            ..EnvelopeFollowerParams::default()
        });
        let out = run_mono(&mut node, &vec![0.5; 4_096]);
        // With the now-fast attack the envelope climbs quickly.
        assert!(out[500] > 0.4, "fast attack after set_params: {}", out[500]);
    }

    #[test]
    fn envelope_getter_reports_state() {
        let mut node =
            EnvelopeFollowerNode::new(SR, ChannelLayout::Mono, SR as usize, EnvelopeFollowerParams::default());
        let _ = run_mono(&mut node, &vec![0.5; SR as usize / 2]);
        assert!((node.envelope(0) - 0.5).abs() < 1e-2, "getter: {}", node.envelope(0));
        assert_eq!(node.envelope(99), 0.0);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut node = EnvelopeFollowerNode::new(
            SR,
            ChannelLayout::Stereo,
            256,
            EnvelopeFollowerParams {
                mode: DetectionMode::Rms,
                rms_window_ms: 1.0e9,
                attack_ms: -100.0,
                release_ms: Sample::NAN,
            },
        );
        let out = run_mono_stereo(&mut node, 2_048);
        assert!(out.iter().all(|&x| x.is_finite()));
    }

    fn run_mono_stereo(node: &mut EnvelopeFollowerNode, len: usize) -> Vec<Sample> {
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        let mut v = outputs[0].channel(0).to_vec();
        v.extend_from_slice(outputs[0].channel(1));
        v
    }
}
