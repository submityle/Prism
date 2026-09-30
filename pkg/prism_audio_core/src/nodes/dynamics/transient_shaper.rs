//! Transient shaper: a level-independent attack / sustain designer that
//! reshapes a signal's envelope instead of clamping its dynamic range.
//!
//! A compressor reacts to *absolute* level relative to a threshold. A
//! transient shaper instead reacts to how fast the envelope is *moving*: it
//! runs two envelope followers on the same signal, a fast one that snaps to
//! onsets and a slow one that lags behind, and drives the gain from their
//! difference. On a rising edge the fast follower leads the slow one, marking
//! an attack; while the sound decays the fast follower falls below the slow
//! one, marking the sustain / body. Two independent controls then emphasise or
//! soften each region -- snappier drums, longer room tails, tighter plucks --
//! all without a threshold, so the effect tracks the material at any level.
//!
//! # The model (differential envelope)
//!
//! For each frame a stereo-linked side-chain level `L` (the loudest rectified
//! channel) feeds two attack/release one-pole followers, `fast` and `slow`.
//! Their difference is normalised by the larger of the two envelopes:
//!
//! ```text
//! ratio = (fast - slow) / max(fast, slow)   in (-1, 1)
//! ```
//!
//! `ratio > 0` is an onset (fast leads), `ratio < 0` is decay (fast trails).
//! The applied linear gain is `1 + attack * ratio` in the onset region and
//! `1 + sustain * (-ratio)` in the decay region, so a positive `attack`
//! emphasises transients and a positive `sustain` lengthens the body; negative
//! values do the opposite. Because the difference is normalised by the peak
//! envelope, the gain is bounded and level-independent, and the followers
//! themselves supply the smoothing, so no zipper noise is introduced.
//!
//! # Real-time contract
//!
//! Every follower and smoother is allocated in [`TransientShaperNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length
//! blocks degrade gracefully, and the gain is clamped to a finite range.
//!
//! # Provenance
//!
//! The differential-envelope transient designer is a classic studio topology
//! (two envelope followers, difference-driven gain) documented across the
//! digital-audio-effects literature (e.g. Zoelzer, "DAFX"). It reuses this
//! crate's own [`time_to_coef`](super::detector::time_to_coef) ballistics
//! helper. This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented signal-processing theory.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::param::{Ramp, Smoothed};

use super::detector::time_to_coef;

/// Smallest envelope used as a normalisation divisor, so a silent input never
/// produces a division by zero.
const MIN_DIVISOR: Sample = 1e-9;

/// A single attack / release one-pole envelope follower on a rectified level.
#[derive(Debug, Clone, Copy)]
struct EnvFollower {
    /// Attack smoothing coefficient (applied while the input rises).
    attack_coef: Sample,
    /// Release smoothing coefficient (applied while the input falls).
    release_coef: Sample,
    /// Current envelope estimate (linear amplitude).
    env: Sample,
}

impl EnvFollower {
    /// Builds a follower from attack / release times in milliseconds.
    fn new(attack_ms: Sample, release_ms: Sample, sample_rate: u32) -> Self {
        Self {
            attack_coef: time_to_coef(attack_ms, sample_rate),
            release_coef: time_to_coef(release_ms, sample_rate),
            env: 0.0,
        }
    }

    /// Advances one sample with rectified input `level` and returns the new
    /// envelope value.
    #[inline]
    fn process(&mut self, level: Sample) -> Sample {
        let coef = if level > self.env {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.env = coef * self.env + (1.0 - coef) * level;
        self.env
    }

    /// Clears the envelope estimate.
    #[inline]
    fn reset(&mut self) {
        self.env = 0.0;
    }
}

/// Configuration for a [`TransientShaperNode`].
///
/// `attack` and `sustain` are dimensionless amounts in `[-1, 1]`: positive
/// emphasises the region, negative softens it, zero leaves it untouched. The
/// four time constants set the two followers; the fast follower must be faster
/// than the slow one for the difference to mark transients.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransientShaperParams {
    /// Attack (onset) emphasis in `[-1, 1]`.
    pub attack: Sample,
    /// Sustain (decay / body) emphasis in `[-1, 1]`.
    pub sustain: Sample,
    /// Fast-follower attack time in milliseconds.
    pub fast_attack_ms: Sample,
    /// Fast-follower release time in milliseconds.
    pub fast_release_ms: Sample,
    /// Slow-follower attack time in milliseconds.
    pub slow_attack_ms: Sample,
    /// Slow-follower release time in milliseconds.
    pub slow_release_ms: Sample,
    /// Symmetric clamp on the applied gain magnitude, in decibels.
    pub max_gain_db: Sample,
    /// Processed (shaped) mix fraction.
    pub wet: Sample,
    /// Dry (unprocessed) mix fraction.
    pub dry: Sample,
}

impl Default for TransientShaperParams {
    fn default() -> Self {
        Self {
            attack: 0.0,
            sustain: 0.0,
            fast_attack_ms: 1.0,
            fast_release_ms: 40.0,
            slow_attack_ms: 15.0,
            slow_release_ms: 180.0,
            max_gain_db: 18.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

/// A differential-envelope transient shaper.
///
/// Stereo-linked: one gain is computed per frame from the loudest channel and
/// applied to every channel, preserving the stereo image.
#[derive(Debug)]
pub struct TransientShaperNode {
    /// Fast envelope follower (leads onsets).
    fast: EnvFollower,
    /// Slow envelope follower (lags behind).
    slow: EnvFollower,
    /// Onset emphasis amount, clamped to `[-1, 1]`.
    attack: Sample,
    /// Decay / body emphasis amount, clamped to `[-1, 1]`.
    sustain: Sample,
    /// Lower gain clamp (linear).
    gain_min: Sample,
    /// Upper gain clamp (linear).
    gain_max: Sample,
    /// Smoothed processed mix fraction.
    wet: Smoothed,
    /// Smoothed dry mix fraction.
    dry: Smoothed,
    /// Most recent applied gain (linear), for metering.
    last_gain: Sample,
}

impl TransientShaperNode {
    /// Builds a transient shaper for `channels` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, _channels: usize, params: TransientShaperParams) -> Self {
        let max_db = params.max_gain_db.max(0.0);
        Self {
            fast: EnvFollower::new(params.fast_attack_ms, params.fast_release_ms, sample_rate),
            slow: EnvFollower::new(params.slow_attack_ms, params.slow_release_ms, sample_rate),
            attack: params.attack.clamp(-1.0, 1.0),
            sustain: params.sustain.clamp(-1.0, 1.0),
            gain_min: db_to_linear(-max_db),
            gain_max: db_to_linear(max_db),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
            last_gain: 1.0,
        }
    }

    /// Sets the onset emphasis amount (clamped to `[-1, 1]`).
    pub fn set_attack(&mut self, attack: Sample) {
        self.attack = attack.clamp(-1.0, 1.0);
    }

    /// Sets the decay / body emphasis amount (clamped to `[-1, 1]`).
    pub fn set_sustain(&mut self, sustain: Sample) {
        self.sustain = sustain.clamp(-1.0, 1.0);
    }

    /// Retargets the processed mix fraction along `ramp`.
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Retargets the dry mix fraction along `ramp`.
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }

    /// Returns the most recent applied gain in decibels (positive = boost).
    #[must_use]
    pub fn gain_db(&self) -> Sample {
        crate::math::linear_to_db(self.last_gain)
    }
}

impl AudioNode for TransientShaperNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let mut level = 0.0;
            for ch in 0..channels {
                let a = input.channel(ch)[f].abs();
                if a > level {
                    level = a;
                }
            }

            let fast = self.fast.process(level);
            let slow = self.slow.process(level);
            let denom = fast.max(slow).max(MIN_DIVISOR);
            let ratio = (fast - slow) / denom;
            let gain = if ratio >= 0.0 {
                1.0 + self.attack * ratio
            } else {
                1.0 + self.sustain * (-ratio)
            }
            .clamp(self.gain_min, self.gain_max);
            self.last_gain = gain;

            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();
            for ch in 0..channels {
                let x = input.channel(ch)[f];
                output.channel_mut(ch)[f] = dry * x + wet * (gain * x);
            }
        }
    }

    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.last_gain = 1.0;
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
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

    // Builds a mono buffer holding one decaying "drum hit": an instantaneous
    // onset followed by an exponential tail.
    fn drum_hit(frames: usize, decay_tau: Sample) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.set_active_frames(frames);
        for (i, s) in buf.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample;
            let env = ops::exp(-t / decay_tau);
            // A tone under the envelope so peaks are well defined.
            let tone = ops::sin(0.30 * t);
            *s = env * tone;
        }
        buf
    }

    fn run(node: &mut TransientShaperNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    fn peak(buf: &AudioBuffer) -> Sample {
        let mut p = 0.0;
        for s in buf.channel(0) {
            let a = s.abs();
            if a > p {
                p = a;
            }
        }
        p
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        let mut e = 0.0;
        for s in buf.channel(0) {
            e += s * s;
        }
        e
    }

    #[test]
    fn neutral_settings_are_transparent() {
        let params = TransientShaperParams::default();
        let mut node = TransientShaperNode::new(SR, 1, params);
        let input = drum_hit(2000, 300.0);
        let out = run(&mut node, &input);
        for (a, b) in input.channel(0).iter().zip(out.channel(0)) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn positive_attack_raises_the_onset_peak() {
        let input = drum_hit(2000, 300.0);
        let base = peak(&input);
        let params = TransientShaperParams {
            attack: 0.8,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &input);
        assert!(peak(&out) > base * 1.05, "shaped peak {} base {base}", peak(&out));
    }

    #[test]
    fn negative_attack_softens_the_onset_peak() {
        let input = drum_hit(2000, 300.0);
        let base = peak(&input);
        let params = TransientShaperParams {
            attack: -0.8,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &input);
        assert!(peak(&out) < base, "shaped peak {} base {base}", peak(&out));
    }

    #[test]
    fn positive_sustain_adds_tail_energy() {
        let input = drum_hit(4000, 600.0);
        let base = energy(&input);
        let params = TransientShaperParams {
            sustain: 0.9,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &input);
        assert!(energy(&out) > base, "shaped {} base {base}", energy(&out));
    }

    #[test]
    fn negative_sustain_removes_tail_energy() {
        let input = drum_hit(4000, 600.0);
        let base = energy(&input);
        let params = TransientShaperParams {
            sustain: -0.9,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &input);
        assert!(energy(&out) < base, "shaped {} base {base}", energy(&out));
    }

    #[test]
    fn gain_is_clamped_to_the_configured_range() {
        let input = drum_hit(2000, 300.0);
        let params = TransientShaperParams {
            attack: 1.0,
            sustain: 1.0,
            max_gain_db: 3.0,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let _ = run(&mut node, &input);
        let ceil = db_to_linear(3.0);
        assert!(node.last_gain <= ceil + 1e-4, "gain {} ceil {ceil}", node.last_gain);
        assert!(node.last_gain >= db_to_linear(-3.0) - 1e-4);
    }

    #[test]
    fn output_is_finite_and_reset_clears_state() {
        let input = drum_hit(1000, 200.0);
        let params = TransientShaperParams {
            attack: 0.7,
            sustain: 0.5,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &input);
        for s in out.channel(0) {
            assert!(s.is_finite());
        }
        node.reset();
        assert!((node.last_gain - 1.0).abs() < 1e-9);
    }

    #[test]
    fn silent_input_is_safe_and_unity() {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 256);
        buf.set_active_frames(256);
        let params = TransientShaperParams {
            attack: 1.0,
            sustain: 1.0,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 1, params);
        let out = run(&mut node, &buf);
        for s in out.channel(0) {
            assert!(s.abs() < 1e-9);
        }
    }

    #[test]
    fn zero_frame_block_does_not_panic() {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, 16);
        buf.set_active_frames(0);
        let params = TransientShaperParams::default();
        let mut node = TransientShaperNode::new(SR, 1, params);
        let _ = run(&mut node, &buf);
    }

    #[test]
    fn stereo_gain_is_linked_across_channels() {
        let frames = 1500;
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, frames);
        buf.set_active_frames(frames);
        let hit = drum_hit(frames, 300.0);
        // Left is the full hit; right is a quiet copy. A linked shaper applies
        // the same gain to both, so the right/left ratio is preserved.
        for i in 0..frames {
            let v = hit.channel(0)[i];
            buf.channel_mut(0)[i] = v;
            buf.channel_mut(1)[i] = 0.25 * v;
        }
        let params = TransientShaperParams {
            attack: 0.8,
            ..TransientShaperParams::default()
        };
        let mut node = TransientShaperNode::new(SR, 2, params);
        let frames_n = buf.active_frames();
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames_n);
        out.set_active_frames(frames_n);
        let inputs = [buf];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames_n), &mut io);
        let [o] = outputs;
        for i in 0..frames_n {
            let l = o.channel(0)[i];
            let r = o.channel(1)[i];
            if l.abs() > 1e-4 {
                assert!((r - 0.25 * l).abs() < 1e-4, "l={l} r={r}");
            }
        }
    }
}
