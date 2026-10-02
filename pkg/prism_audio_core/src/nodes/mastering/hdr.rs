//! Classic HDR-audio dynamic window: a loudness-tracking automatic level
//! control that maps a wide input dynamic range into a bounded output window.
//!
//! "HDR audio" borrows the high-dynamic-range idea from imaging: a scene can
//! span a far wider range of intensities than the output medium can reproduce,
//! so a *window* slides over the full range and continuously remaps whatever is
//! currently loudest onto the top of the output's limited range. For audio that
//! means tracking the loudest recent signal, treating it as the top of a window
//! `window_db` wide, and deriving a broadband gain that places that loudest
//! sound at a fixed output `target`. Everything quieter is lifted or lowered by
//! the same gain, so quiet sounds are relatively *lifted* while loud sounds are
//! *attenuated* -- exactly the perceptual "make everything audible without ever
//! clipping" behaviour the technique is used for.
//!
//! # Model
//!
//! An attack/release envelope tracks the loudest recent level in decibels,
//! `env_db` (the window top). The broadband makeup that places it on the target
//! is simply `target_db - env_db`:
//!
//! - When the input is **loud**, `env_db` is high, so the makeup is negative --
//!   the signal is attenuated. A loud transient pulls the window up (raises
//!   `env_db`) quickly on the attack, then the window drifts back down over the
//!   release.
//! - When the input is **quiet**, `env_db` is low, so the makeup is positive --
//!   the signal is lifted toward the target.
//! - A quiet sound arriving **just after** a loud one is attenuated relative to
//!   the same quiet sound heard in isolation, because the still-elevated
//!   `env_db` (slow release) holds the window up: the hallmark of HDR-style
//!   dynamic windowing / loudness ducking.
//!
//! The gain is clamped to `+/- window_db` so the output stays bounded: material
//! more than `window_db` below the window top is not lifted past the window
//! floor, and the makeup never attenuates more than the window either. The gain
//! is derived from a smoothly moving envelope, so it needs no extra smoothing
//! to stay click-free.
//!
//! # Real-time contract
//!
//! The only state is the scalar envelope and a [`LevelDetector`]; nothing is
//! allocated after construction. [`HdrNode::process`] performs no allocation,
//! takes no locks, cannot panic, and sanitises non-finite input to silence. All
//! transcendental math routes through [`bevy_math::ops`] via the shared
//! [`math`](crate::math) helpers, so the result is bit-reproducible.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The dynamic
//! windowing / loudness-tracking automatic-gain law is a textbook dynamics
//! construction described in public audio-engineering literature. It is pure
//! classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! This node reuses the dynamics family's
//! [`LevelDetector`](crate::nodes::dynamics::detector::LevelDetector) and
//! [`time_to_coef`](crate::nodes::dynamics::detector::time_to_coef) for its
//! level measurement and envelope coefficients rather than re-deriving them. It
//! differs from the static
//! [`LoudnessNormalizerNode`](crate::nodes::mastering::loudness_normalizer::LoudnessNormalizerNode),
//! which applies one measured makeup gain, by *continuously* re-deriving its
//! gain from a moving loudness window; and from the
//! [`CompressorNode`](crate::nodes::dynamics::compressor::CompressorNode),
//! which reduces range around a fixed threshold, by *re-centring* the whole
//! signal on a target as the loudest content moves.

use crate::buffer::AudioBuffer;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{db_to_linear, Sample};
use crate::nodes::dynamics::detector::{time_to_coef, DetectionMode, LevelDetector};

/// Default output level the loudest recent signal is mapped to, in dBFS.
pub const DEFAULT_TARGET_DB: Sample = -12.0;

/// Default dynamic window width, in decibels.
pub const DEFAULT_WINDOW_DB: Sample = 24.0;

/// Default attack time (window rises to a louder signal), in milliseconds.
pub const DEFAULT_ATTACK_MS: Sample = 5.0;

/// Default release time (window falls after a loud signal), in milliseconds.
pub const DEFAULT_RELEASE_MS: Sample = 300.0;

/// Level floor fed to the envelope follower when the input is silent (or
/// otherwise below the audible floor). Silence reports `-inf` dB, which
/// would freeze the follower; clamping to a finite floor lets the window
/// release cleanly toward silence.
const LEVEL_FLOOR_DB: Sample = -120.0;

/// Construction parameters for an [`HdrWindow`] / [`HdrNode`].
///
/// All fields are plain values and the struct is [`Copy`]; it is a pure
/// description of the window policy, not processing state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HdrParams {
    /// Output level, in dBFS, that the loudest recent signal is mapped to.
    pub target_db: Sample,
    /// Dynamic window width, in decibels. Also bounds the makeup gain to
    /// `+/- window_db`, keeping the output bounded.
    pub window_db: Sample,
    /// Attack time in milliseconds: how fast the window rises toward a louder
    /// signal.
    pub attack_ms: Sample,
    /// Release time in milliseconds: how slowly the window falls after a loud
    /// signal (the longer it is, the more a quiet-after-loud sound is held
    /// down).
    pub release_ms: Sample,
    /// How the recent level is measured (peak or RMS).
    pub detection: DetectionMode,
    /// Averaging window for RMS detection, in milliseconds (ignored for peak).
    pub rms_window_ms: Sample,
}

impl Default for HdrParams {
    fn default() -> Self {
        Self {
            target_db: DEFAULT_TARGET_DB,
            window_db: DEFAULT_WINDOW_DB,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            detection: DetectionMode::Peak,
            rms_window_ms: 10.0,
        }
    }
}

/// The HDR dynamic-window DSP core (format-agnostic, operates on
/// [`AudioBuffer`]s).
///
/// Construct one with [`new`](Self::new); drive it with
/// [`process`](Self::process). The [`HdrNode`] wrapper adapts it to the
/// [`AudioNode`] graph interface and adds nothing but plumbing.
#[derive(Debug, Clone)]
pub struct HdrWindow {
    channels: usize,
    /// Level detector shared with the dynamics family.
    detector: LevelDetector,
    /// Detection mode (cached: `LevelDetector` keeps its mode private).
    detection: DetectionMode,
    /// Attack coefficient for the envelope (one-pole).
    attack_coef: Sample,
    /// Release coefficient for the envelope (one-pole).
    release_coef: Sample,
    /// The tracked loudest-recent level, in dB (the window top).
    env_db: Sample,
    /// Output target level, in dB.
    target_db: Sample,
    /// Window width / gain bound, in dB.
    window_db: Sample,
}

impl HdrWindow {
    /// Builds an HDR window for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: HdrParams) -> Self {
        let channels = channels.max(1);
        let sr = sample_rate.max(1);
        Self {
            channels,
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sr),
            detection: params.detection,
            attack_coef: time_to_coef(params.attack_ms, sr),
            release_coef: time_to_coef(params.release_ms, sr),
            // Start the window at the target so the initial makeup is unity.
            env_db: params.target_db,
            target_db: params.target_db,
            window_db: params.window_db.max(0.0),
        }
    }

    /// Returns the number of channels this window was built for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the current window top (tracked loudest-recent level) in dB.
    #[inline]
    #[must_use]
    pub fn reference_db(&self) -> Sample {
        self.env_db
    }

    /// Returns the broadband makeup gain currently applied, in dB.
    #[inline]
    #[must_use]
    pub fn gain_db(&self) -> Sample {
        self.current_gain_db()
    }

    /// The makeup that places the window top on the target, bounded by the
    /// window width.
    #[inline]
    fn current_gain_db(&self) -> Sample {
        (self.target_db - self.env_db).clamp(-self.window_db, self.window_db)
    }

    /// Sanitises one input sample: non-finite values become silence.
    #[inline]
    fn clean(x: Sample) -> Sample {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    /// Processes one block, writing the windowed result to `output`.
    ///
    /// A single broadband gain is applied to every channel per frame so the
    /// stereo/surround image stays coherent, driven by the combined channel
    /// level.
    pub fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.channels);
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || channels == 0 {
            return;
        }

        for f in 0..frames {
            // Combine channels into one detector feed: the loudest sample for a
            // peak follower, the summed signal for RMS power.
            let mut combined = 0.0;
            for ch in 0..channels {
                let x = Self::clean(input.channel(ch)[f]);
                match self.detector_mode() {
                    DetectionMode::Peak => {
                        let a = x.abs();
                        if a > combined {
                            combined = a;
                        }
                    }
                    DetectionMode::Rms => combined += x,
                }
            }

            // Silence reports `-inf` dB; clamp to a finite floor so the
            // follower keeps releasing instead of freezing.
            let raw_db = self.detector.level_db(combined);
            let level_db = if raw_db.is_finite() {
                raw_db.max(LEVEL_FLOOR_DB)
            } else {
                LEVEL_FLOOR_DB
            };

            // Attack when the signal is louder than the window top, release
            // otherwise -- the classic peak-follower envelope.
            let coef = if level_db > self.env_db {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.env_db = coef * self.env_db + (1.0 - coef) * level_db;

            let gain = db_to_linear(self.current_gain_db());
            for ch in 0..channels {
                let x = Self::clean(input.channel(ch)[f]);
                output.channel_mut(ch)[f] = x * gain;
            }
        }
    }

    #[inline]
    fn detector_mode(&self) -> DetectionMode {
        self.detection
    }

    /// Clears the envelope back to the target (unity makeup) and resets the
    /// level detector.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.env_db = self.target_db;
    }
}

/// An HDR dynamic-window graph node (input port 0 -> output port 0).
///
/// Thin [`AudioNode`] adapter over the [`HdrWindow`] DSP core; it adds no DSP
/// of its own.
#[derive(Debug, Clone)]
pub struct HdrNode {
    engine: HdrWindow,
}

impl HdrNode {
    /// Creates an HDR window node for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: HdrParams) -> Self {
        Self {
            engine: HdrWindow::new(sample_rate, channels, params),
        }
    }

    /// Borrows the underlying window engine.
    #[must_use]
    pub fn engine(&self) -> &HdrWindow {
        &self.engine
    }

    /// Mutably borrows the underlying window engine.
    pub fn engine_mut(&mut self) -> &mut HdrWindow {
        &mut self.engine
    }

    /// Returns the broadband makeup gain currently applied, in dB.
    #[inline]
    #[must_use]
    pub fn gain_db(&self) -> Sample {
        self.engine.gain_db()
    }
}

impl AudioNode for HdrNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        self.engine.process(input, output);
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use bevy_math::ops;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    fn run_mono(node: &mut HdrNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = mono(frames);
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    /// Fills a mono buffer with a sine at amplitude `amp` and frequency `f`.
    fn tone(buf: &mut AudioBuffer, amp: Sample, f: Sample) {
        for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample / SR as Sample;
            *s = amp * ops::sin(2.0 * core::f32::consts::PI * f * t);
        }
    }

    #[test]
    fn loud_transient_pulls_window_up_then_releases() {
        let mut node = HdrNode::new(SR, 1, HdrParams::default());
        // Quiet lead-in settles the window low.
        let mut quiet = mono(4_000);
        tone(&mut quiet, 0.05, 1_000.0);
        let _ = run_mono(&mut node, &quiet);
        let low_ref = node.engine().reference_db();

        // A loud blast pulls the window up.
        let mut loud = mono(4_000);
        tone(&mut loud, 0.9, 1_000.0);
        let _ = run_mono(&mut node, &loud);
        let high_ref = node.engine().reference_db();
        assert!(
            high_ref > low_ref + 10.0,
            "window did not rise: {low_ref} -> {high_ref}"
        );

        // Silence lets it release back down.
        let silence = mono(SR as usize); // ~1 s
        let _ = run_mono(&mut node, &silence);
        let released_ref = node.engine().reference_db();
        assert!(
            released_ref < high_ref - 10.0,
            "window did not release: {high_ref} -> {released_ref}"
        );
    }

    #[test]
    fn quiet_after_loud_is_attenuated_relative_to_standalone() {
        let params = HdrParams::default();
        let quiet_amp = 0.05;

        // Standalone: the quiet tone heard in isolation.
        let mut standalone = HdrNode::new(SR, 1, params);
        let mut quiet = mono(6_000);
        tone(&mut quiet, quiet_amp, 1_000.0);
        let out_standalone = run_mono(&mut standalone, &quiet);

        // After-loud: the same quiet tone immediately after a loud blast.
        let mut after = HdrNode::new(SR, 1, params);
        let mut loud = mono(6_000);
        tone(&mut loud, 0.9, 1_000.0);
        let _ = run_mono(&mut after, &loud);
        let out_after = run_mono(&mut after, &quiet);

        // Compare steady-state RMS near the end of the quiet block.
        let rms = |b: &AudioBuffer| -> Sample {
            let s = &b.channel(0)[4_000..];
            let sum: Sample = s.iter().map(|x| x * x).sum();
            ops::sqrt(sum / s.len() as Sample)
        };
        let r_standalone = rms(&out_standalone);
        let r_after = rms(&out_after);
        assert!(
            r_after < r_standalone * 0.9,
            "quiet-after-loud not attenuated: after={r_after} standalone={r_standalone}"
        );
    }

    #[test]
    fn output_is_bounded() {
        // Even for a hot full-scale input the output stays within a bound set
        // by the window / target, never blowing up.
        let params = HdrParams::default();
        let mut node = HdrNode::new(SR, 1, params);
        let mut input = mono(16_000);
        tone(&mut input, 1.0, 1_000.0);
        let out = run_mono(&mut node, &input);
        // Bound: unity-ceiling input times the maximum possible boost.
        let bound = db_to_linear(params.window_db) + 1e-3;
        for &y in out.channel(0) {
            assert!(y.abs() <= bound, "unbounded output: {y}");
        }
        // Once settled the makeup should be attenuating a hot input.
        assert!(node.gain_db() < 0.0, "hot input not attenuated: {}", node.gain_db());
    }

    #[test]
    fn quiet_signal_is_lifted() {
        // A quiet steady tone should be boosted toward the target.
        let params = HdrParams::default();
        let mut node = HdrNode::new(SR, 1, params);
        let mut input = mono(SR as usize);
        tone(&mut input, 0.02, 1_000.0);
        let _ = run_mono(&mut node, &input);
        assert!(node.gain_db() > 1.0, "quiet signal not lifted: {}", node.gain_db());
    }

    #[test]
    fn deterministic_across_runs() {
        let params = HdrParams::default();
        let n = 4_000;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.6 * ops::sin(0.19 * i as Sample) + 0.3 * ops::sin(0.41 * i as Sample);
        }
        let mut a = HdrNode::new(SR, 1, params);
        let mut b = HdrNode::new(SR, 1, params);
        let out_a = run_mono(&mut a, &input);
        let out_b = run_mono(&mut b, &input);
        for i in 0..n {
            assert_eq!(out_a.channel(0)[i], out_b.channel(0)[i], "nondeterministic at {i}");
        }
    }

    #[test]
    fn non_finite_input_produces_no_nan() {
        let mut node = HdrNode::new(SR, 1, HdrParams::default());
        let n = 512;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = match i % 4 {
                0 => Sample::NAN,
                1 => Sample::INFINITY,
                2 => Sample::NEG_INFINITY,
                _ => 0.3,
            };
        }
        let out = run_mono(&mut node, &input);
        for &y in out.channel(0) {
            assert!(y.is_finite(), "non-finite output: {y}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let params = HdrParams::default();
        let mut node = HdrNode::new(SR, 1, params);
        let mut loud = mono(8_000);
        tone(&mut loud, 0.9, 1_000.0);
        let _ = run_mono(&mut node, &loud);
        node.reset();
        // Envelope back at the target => unity makeup.
        assert!((node.engine().reference_db() - params.target_db).abs() < 1e-6);
        assert!(node.gain_db().abs() < 1e-6);
    }

    #[test]
    fn stereo_gain_is_coherent() {
        let params = HdrParams::default();
        let mut node = HdrNode::new(SR, 2, params);
        let n = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, n);
        for i in 0..n {
            let t = i as Sample / SR as Sample;
            let v = 0.5 * ops::sin(2.0 * core::f32::consts::PI * 1_000.0 * t);
            input.channel_mut(0)[i] = v;
            input.channel_mut(1)[i] = v;
        }
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, n);
        let inputs = [input];
        let mut outputs = [out.clone()];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        out = outputs.into_iter().next().unwrap();
        for i in 0..n {
            assert_eq!(out.channel(0)[i], out.channel(1)[i], "image not coherent at {i}");
        }
    }
}
