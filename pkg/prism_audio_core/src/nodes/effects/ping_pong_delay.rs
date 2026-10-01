//! Ping-pong delay: a cross-coupled stereo echo whose repeats bounce from one
//! side to the other, with independent per-side delay times, a damping low-pass
//! inside the feedback loop, and dry / wet mixing.
//!
//! A plain delay feeds each channel's output back into its own line, so the
//! echoes stay put. A ping-pong delay instead feeds each side's delayed signal
//! back into the *other* side's line, so a single hit alternates left, right,
//! left, right as it decays, the signature "bouncing" echo of stereo tape and
//! digital delays. Channels are paired `(0, 1)`, `(2, 3)`, and so on; even
//! channels use the left delay time and odd channels the right, and each pair
//! cross-feeds. A lone trailing channel (or a mono buffer) degrades gracefully
//! to a self-feeding delay.
//!
//! # Relationship
//!
//! This node reuses the crate's standard per-channel fractional ring-buffer
//! delay line (linear-interpolated read behind a shared write head, identical
//! in spirit to [`DelayNode`](crate::nodes::effects::delay::DelayNode)) but
//! differs in topology: [`DelayNode`] feeds every channel back into itself,
//! whereas this node routes each side's damped feedback into its partner side,
//! which no per-channel delay can produce. It also differs from
//! [`MultiTapDelayNode`](crate::nodes::effects::multi_tap_delay::MultiTapDelayNode)
//! (several taps off one shared mono line, no cross-channel feedback) and from
//! [`CombResonatorNode`](crate::nodes::effects::comb_resonator::CombResonatorNode)
//! (a single short tuned feedback comb that rings at a pitch rather than echoing
//! at a musically audible delay time). The in-loop damping is a one-pole
//! low-pass so successive bounces lose high frequencies, the way repeated tape
//! or analog repeats darken.
//!
//! # Real-time contract
//!
//! Every ring buffer, the per-channel damping state, the per-channel delay-time
//! table, and the per-frame read / feedback scratch are allocated once at
//! construction (sized for the maximum delay and channel count).
//! [`PingPongDelayNode::process`] performs no allocation, locking, or panic on
//! the hot path; non-finite input samples are treated as silence, and every
//! value written back into a ring or damping state is denormal-flushed so the
//! feedback loop cannot stall on denormals. The node adds no latency
//! ([`PingPongDelayNode::latency_frames`] returns zero); the shortest delay is
//! clamped to one frame so the cross-feedback loop always has a delay in it.
//!
//! # Provenance
//!
//! The ping-pong (cross-feedback stereo) delay, the fractional-delay ring
//! buffer with linear interpolation, and the one-pole low-pass damping term in
//! a feedback loop are standard, publicly documented classic DSP building
//! blocks (for example Zoelzer's DAFX delay chapter and the Schroeder / Moorer
//! feedback-delay literature). This is pure classic DSP with no AI or ML. This
//! module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, or Web Audio source or derived code**; only the
//! widely documented delay-line and one-pole formulas are used.

use alloc::{vec, vec::Vec};

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};

/// Largest per-side delay time, in milliseconds, the node can address.
pub const MAX_PING_PONG_DELAY_MS: Sample = 2_000.0;

/// Largest stable feedback coefficient, kept below unity so the bouncing tail
/// decays instead of building without bound.
pub const MAX_PING_PONG_FEEDBACK: Sample = 0.95;

/// Default left-side delay time in milliseconds.
pub const DEFAULT_LEFT_DELAY_MS: Sample = 250.0;

/// Default right-side delay time in milliseconds.
pub const DEFAULT_RIGHT_DELAY_MS: Sample = 375.0;

/// Default feedback coefficient.
pub const DEFAULT_PING_PONG_FEEDBACK: Sample = 0.4;

/// Default in-loop damping amount in `[0, 1]`.
pub const DEFAULT_PING_PONG_DAMPING: Sample = 0.3;

/// Default dry / wet mix.
pub const DEFAULT_PING_PONG_MIX: Sample = 0.35;

/// Parameters shared by every channel pair.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PingPongDelayParams {
    /// Left-side (even channel) delay time in milliseconds.
    pub left_delay_ms: Sample,
    /// Right-side (odd channel) delay time in milliseconds.
    pub right_delay_ms: Sample,
    /// Feedback coefficient in `[0, MAX_PING_PONG_FEEDBACK]`: how much of each
    /// bounce is routed into the opposite side.
    pub feedback: Sample,
    /// In-loop damping in `[0, 1]`: `0` keeps repeats bright, `1` applies the
    /// darkest low-pass to each successive bounce.
    pub damping: Sample,
    /// Dry / wet mix in `[0, 1]`: `0` is the dry input, `1` is the echo only.
    pub mix: Sample,
}

impl Default for PingPongDelayParams {
    fn default() -> Self {
        Self {
            left_delay_ms: DEFAULT_LEFT_DELAY_MS,
            right_delay_ms: DEFAULT_RIGHT_DELAY_MS,
            feedback: DEFAULT_PING_PONG_FEEDBACK,
            damping: DEFAULT_PING_PONG_DAMPING,
            mix: DEFAULT_PING_PONG_MIX,
        }
    }
}

impl PingPongDelayParams {
    /// Returns the parameters with every field clamped into range and any
    /// non-finite field replaced by its default.
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let clamp = |v: Sample, lo: Sample, hi: Sample, fallback: Sample| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        Self {
            left_delay_ms: clamp(self.left_delay_ms, 0.0, MAX_PING_PONG_DELAY_MS, d.left_delay_ms),
            right_delay_ms: clamp(
                self.right_delay_ms,
                0.0,
                MAX_PING_PONG_DELAY_MS,
                d.right_delay_ms,
            ),
            feedback: clamp(self.feedback, 0.0, MAX_PING_PONG_FEEDBACK, d.feedback),
            damping: clamp(self.damping, 0.0, 1.0, d.damping),
            mix: clamp(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// A cross-coupled stereo ping-pong delay.
#[derive(Clone, Debug)]
pub struct PingPongDelayNode {
    sample_rate: u32,
    channels: usize,
    /// Ring length in frames (`max_delay + 2`), shared by every channel.
    ring_len: usize,
    /// Largest addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// One ring buffer per channel.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every ring.
    write_pos: usize,
    /// One-pole low-pass state per channel (the damping memory in each loop).
    damp_state: Vec<Sample>,
    /// Resolved delay in frames per channel (even -> left, odd -> right).
    delay_frames: Vec<Sample>,
    /// Per-frame scratch holding each channel's freshly read delayed sample.
    read_scratch: Vec<Sample>,
    /// Per-frame scratch holding each channel's damped feedback sample.
    fb_scratch: Vec<Sample>,
    feedback: Sample,
    /// One-pole coefficient `1 - damping` (`1` passes feedback unfiltered).
    damp_coef: Sample,
    wet: Sample,
    dry: Sample,
}

impl PingPongDelayNode {
    /// Builds a ping-pong delay for `channels` channels at `sample_rate` Hz.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::ping_pong_delay::{
    ///     PingPongDelayNode, PingPongDelayParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = PingPongDelayNode::new(48_000, 2, PingPongDelayParams::default());
    /// // A delay effect adds no reported processing latency.
    /// assert_eq!(node.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: PingPongDelayParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let max_delay_frames =
            ops::floor(MAX_PING_PONG_DELAY_MS * sample_rate as Sample / 1_000.0) as usize + 1;
        let ring_len = max_delay_frames + 2;
        let mut node = Self {
            sample_rate,
            channels,
            ring_len,
            max_delay: max_delay_frames as Sample,
            rings: vec![vec![0.0; ring_len]; channels],
            write_pos: 0,
            damp_state: vec![0.0; channels],
            delay_frames: vec![1.0; channels],
            read_scratch: vec![0.0; channels],
            fb_scratch: vec![0.0; channels],
            feedback: 0.0,
            damp_coef: 1.0,
            wet: 0.0,
            dry: 1.0,
        };
        node.set_params(params);
        node
    }

    /// Number of channels this node was built for.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Converts a delay time in milliseconds to a clamped delay in frames.
    fn ms_to_frames(&self, ms: Sample) -> Sample {
        let frames = ms * self.sample_rate as Sample / 1_000.0;
        frames.clamp(1.0, self.max_delay)
    }

    /// Replaces the delay times, feedback, damping, and mix. Delay-line and
    /// damping state are preserved so automation stays click-free.
    pub fn set_params(&mut self, params: PingPongDelayParams) {
        let params = params.sanitised();
        let left = self.ms_to_frames(params.left_delay_ms);
        let right = self.ms_to_frames(params.right_delay_ms);
        for (ch, slot) in self.delay_frames.iter_mut().enumerate() {
            *slot = if ch % 2 == 0 { left } else { right };
        }
        self.feedback = params.feedback;
        self.damp_coef = 1.0 - params.damping;
        self.wet = params.mix;
        self.dry = 1.0 - params.mix;
    }
}

impl AudioNode for PingPongDelayNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());
        let ring_len = self.ring_len;
        let len_i = ring_len as isize;

        let feedback = self.feedback;
        let damp_coef = self.damp_coef;
        let wet = self.wet;
        let dry = self.dry;

        for f in 0..frames {
            let w = self.write_pos;

            // Pass 1: read each channel's delayed sample and update its damping
            // one-pole, caching both so the cross-feedback write can read any
            // partner channel without a borrow conflict.
            for ch in 0..channels {
                let delay = self.delay_frames[ch];
                let read_pos = w as Sample - delay;
                let base = ops::floor(read_pos);
                let frac = read_pos - base;
                let base_i = base as isize;
                let i0 = base_i.rem_euclid(len_i) as usize;
                let i1 = (base_i + 1).rem_euclid(len_i) as usize;
                let ring = &self.rings[ch];
                let delayed = lerp(ring[i0], ring[i1], frac);
                self.read_scratch[ch] = delayed;

                let damped = self.damp_state[ch] + damp_coef * (delayed - self.damp_state[ch]);
                let damped = flush_denormal(damped);
                self.damp_state[ch] = damped;
                self.fb_scratch[ch] = damped;
            }

            // Pass 2: cross-feed each side's damped feedback into its partner
            // line, emit the dry / wet mix.
            for ch in 0..channels {
                let x_raw = input.channel(ch)[f];
                let x = if x_raw.is_finite() { x_raw } else { 0.0 };
                let partner = if (ch ^ 1) < channels { ch ^ 1 } else { ch };
                let write_val = x + feedback * self.fb_scratch[partner];
                self.rings[ch][w] = flush_denormal(write_val);
                output.channel_mut(ch)[f] = dry * x + wet * self.read_scratch[ch];
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }

        // Pass surplus channels (beyond the processed set) through untouched.
        for ch in channels..available {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        for s in &mut self.damp_state {
            *s = 0.0;
        }
        self.write_pos = 0;
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    /// Streams a mono `signal` through the node in one block.
    fn run_mono(node: &mut PingPongDelayNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        input.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(signal);
        let output = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        let inputs = [input];
        let mut outputs = [output];
        outputs[0].set_active_frames(len);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0)[..len].to_vec()
    }

    /// Streams a stereo pair through the node in one block.
    fn run_stereo(
        node: &mut PingPongDelayNode,
        left: &[Sample],
        right: &[Sample],
    ) -> (Vec<Sample>, Vec<Sample>) {
        let len = left.len().min(right.len());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len.max(1));
        input.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(&left[..len]);
        input.channel_mut(1)[..len].copy_from_slice(&right[..len]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len.max(1));
        let inputs = [input];
        let mut outputs = [output];
        outputs[0].set_active_frames(len);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        (
            outputs[0].channel(0)[..len].to_vec(),
            outputs[0].channel(1)[..len].to_vec(),
        )
    }

    fn impulse(len: usize) -> Vec<Sample> {
        let mut v = vec![0.0; len];
        if !v.is_empty() {
            v[0] = 1.0;
        }
        v
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().map(|&x| x * x).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    #[test]
    fn latency_is_zero() {
        let node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn channels_getter_reports_build_width() {
        let node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        assert_eq!(node.channels(), 2);
    }

    #[test]
    fn default_params_are_in_range() {
        let p = PingPongDelayParams::default();
        assert!(p.left_delay_ms >= 0.0 && p.left_delay_ms <= MAX_PING_PONG_DELAY_MS);
        assert!(p.right_delay_ms >= 0.0 && p.right_delay_ms <= MAX_PING_PONG_DELAY_MS);
        assert!(p.feedback >= 0.0 && p.feedback <= MAX_PING_PONG_FEEDBACK);
        assert!(p.damping >= 0.0 && p.damping <= 1.0);
        assert!(p.mix >= 0.0 && p.mix <= 1.0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        let (l, r) = run_stereo(&mut node, &vec![0.0; 256], &vec![0.0; 256]);
        assert!(l.iter().all(|&x| x == 0.0));
        assert!(r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn dry_passthrough_when_mix_zero() {
        let params = PingPongDelayParams {
            mix: 0.0,
            feedback: 0.6,
            ..PingPongDelayParams::default()
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let left = impulse(128);
        let mut right = vec![0.0; 128];
        right[3] = -0.5;
        let (lo, ro) = run_stereo(&mut node, &left, &right);
        for (o, i) in lo.iter().zip(left.iter()) {
            assert!((o - i).abs() < 1e-6, "left dry mismatch: {o} vs {i}");
        }
        for (o, i) in ro.iter().zip(right.iter()) {
            assert!((o - i).abs() < 1e-6, "right dry mismatch: {o} vs {i}");
        }
    }

    #[test]
    fn integer_delay_shifts_impulse() {
        // 2 ms at 48 kHz is exactly 96 frames.
        let params = PingPongDelayParams {
            left_delay_ms: 2.0,
            right_delay_ms: 2.0,
            feedback: 0.0,
            damping: 0.0,
            mix: 1.0,
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let left = impulse(256);
        let (lo, _ro) = run_stereo(&mut node, &left, &vec![0.0; 256]);
        assert!((lo[96] - 1.0).abs() < 1e-5, "echo should land at frame 96: {}", lo[96]);
        for (f, &v) in lo.iter().enumerate().take(96) {
            assert!(v.abs() < 1e-6, "no echo before frame 96 (frame {f} = {v})");
        }
    }

    #[test]
    fn cross_feed_puts_energy_on_silent_partner() {
        let params = PingPongDelayParams {
            left_delay_ms: 1.0,
            right_delay_ms: 1.0,
            feedback: 0.6,
            damping: 0.2,
            mix: 1.0,
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        // Only the left channel is excited; the right input stays silent.
        let left = impulse(512);
        let (_lo, ro) = run_stereo(&mut node, &left, &vec![0.0; 512]);
        assert!(
            rms(&ro) > 1e-4,
            "cross-feedback must deposit energy on the silent right channel: rms={}",
            rms(&ro)
        );
    }

    #[test]
    fn first_echo_appears_on_input_side() {
        let params = PingPongDelayParams {
            left_delay_ms: 2.0,
            right_delay_ms: 3.0,
            feedback: 0.5,
            damping: 0.0,
            mix: 1.0,
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let left = impulse(512);
        let (lo, _ro) = run_stereo(&mut node, &left, &vec![0.0; 512]);
        // 2 ms at 48 kHz is 96 frames: the first repeat lands on the input side.
        assert!(lo[96].abs() > 0.1, "first echo on input side at frame 96: {}", lo[96]);
    }

    #[test]
    fn feedback_decays_and_stays_finite() {
        let params = PingPongDelayParams {
            left_delay_ms: 1.0,
            right_delay_ms: 1.0,
            feedback: MAX_PING_PONG_FEEDBACK,
            damping: 0.3,
            mix: 1.0,
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let left = impulse(8_000);
        let (lo, ro) = run_stereo(&mut node, &left, &vec![0.0; 8_000]);
        assert!(lo.iter().all(|x| x.is_finite()));
        assert!(ro.iter().all(|x| x.is_finite()));
        let early = rms(&lo[..2_000]);
        let late = rms(&lo[6_000..]);
        assert!(late < early, "tail must decay: early={early} late={late}");
    }

    #[test]
    fn damping_reduces_tail_energy() {
        let base = PingPongDelayParams {
            left_delay_ms: 1.0,
            right_delay_ms: 1.0,
            feedback: 0.7,
            mix: 1.0,
            ..PingPongDelayParams::default()
        };
        let mut bright = PingPongDelayNode::new(
            SR,
            2,
            PingPongDelayParams {
                damping: 0.0,
                ..base
            },
        );
        let mut dark = PingPongDelayNode::new(
            SR,
            2,
            PingPongDelayParams {
                damping: 0.9,
                ..base
            },
        );
        let left = impulse(4_000);
        let (bl, _) = run_stereo(&mut bright, &left, &vec![0.0; 4_000]);
        let (dl, _) = run_stereo(&mut dark, &left, &vec![0.0; 4_000]);
        assert!(
            rms(&dl) < rms(&bl),
            "heavier damping must shed feedback energy: dark={} bright={}",
            rms(&dl),
            rms(&bl)
        );
    }

    #[test]
    fn non_finite_input_becomes_finite() {
        let mut node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        let mut left = vec![0.0; 256];
        left[0] = Sample::NAN;
        left[1] = Sample::INFINITY;
        left[2] = Sample::NEG_INFINITY;
        let (lo, ro) = run_stereo(&mut node, &left, &vec![0.0; 256]);
        assert!(lo.iter().all(|x| x.is_finite()));
        assert!(ro.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 1);
        let mut inputs = [input];
        inputs[0].set_active_frames(0);
        let mut outputs = [output.clone()];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        output = outputs[0].clone();
        assert_eq!(output.active_frames(), 0);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let params = PingPongDelayParams {
            left_delay_ms: 10_000.0,
            right_delay_ms: -5.0,
            feedback: 10.0,
            damping: 4.0,
            mix: 2.0,
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let (lo, ro) = run_stereo(&mut node, &impulse(512), &vec![0.0; 512]);
        assert!(lo.iter().all(|x| x.is_finite()));
        assert!(ro.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn non_finite_params_fall_back_to_default() {
        let params = PingPongDelayParams {
            left_delay_ms: Sample::NAN,
            right_delay_ms: Sample::INFINITY,
            feedback: Sample::NAN,
            damping: Sample::NEG_INFINITY,
            mix: Sample::NAN,
        };
        let sane = params.sanitised();
        let d = PingPongDelayParams::default();
        assert_eq!(sane.left_delay_ms, d.left_delay_ms);
        assert_eq!(sane.right_delay_ms, d.right_delay_ms);
        assert_eq!(sane.feedback, d.feedback);
        assert_eq!(sane.damping, d.damping);
        assert_eq!(sane.mix, d.mix);
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = PingPongDelayNode::new(SR, 2, PingPongDelayParams::default());
        let len = 128;
        let mut input = AudioBuffer::new(ChannelLayout::Quad, len);
        for ch in 0..4 {
            for f in 0..len {
                input.channel_mut(ch)[f] = (ch as Sample + 1.0) * 0.1 * f as Sample;
            }
        }
        let output = AudioBuffer::new(ChannelLayout::Quad, len);
        let inputs = [input.clone()];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        // Channels 2 and 3 are beyond the processed stereo pair: verbatim copy.
        for ch in 2..4 {
            for f in 0..len {
                assert_eq!(outputs[0].channel(ch)[f], inputs[0].channel(ch)[f]);
            }
        }
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = PingPongDelayParams {
            left_delay_ms: 1.0,
            right_delay_ms: 1.0,
            feedback: 0.6,
            ..PingPongDelayParams::default()
        };
        let mut node = PingPongDelayNode::new(SR, 2, params);
        let left = impulse(512);
        let first = run_stereo(&mut node, &left, &vec![0.0; 512]);
        node.reset();
        let second = run_stereo(&mut node, &left, &vec![0.0; 512]);
        let mut max_err = 0.0_f32;
        for (a, b) in first.0.iter().zip(second.0.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        for (a, b) in first.1.iter().zip(second.1.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-6, "reset must reproduce the fresh response: {max_err}");
    }

    #[test]
    fn set_params_changes_delay_time() {
        let mut node = PingPongDelayNode::new(
            SR,
            2,
            PingPongDelayParams {
                left_delay_ms: 2.0,
                right_delay_ms: 2.0,
                feedback: 0.0,
                damping: 0.0,
                mix: 1.0,
            },
        );
        let short = run_stereo(&mut node, &impulse(512), &vec![0.0; 512]).0;
        node.reset();
        node.set_params(PingPongDelayParams {
            left_delay_ms: 4.0,
            right_delay_ms: 4.0,
            feedback: 0.0,
            damping: 0.0,
            mix: 1.0,
        });
        let long = run_stereo(&mut node, &impulse(512), &vec![0.0; 512]).0;
        // 2 ms -> 96 frames, 4 ms -> 192 frames.
        assert!((short[96] - 1.0).abs() < 1e-5);
        assert!((long[192] - 1.0).abs() < 1e-5);
        assert!(long[96].abs() < 1e-6, "longer delay must not echo at the old time");
    }

    #[test]
    fn mono_degrades_to_self_feedback() {
        let params = PingPongDelayParams {
            left_delay_ms: 1.0,
            right_delay_ms: 1.0,
            feedback: 0.6,
            damping: 0.2,
            mix: 1.0,
        };
        let mut node = PingPongDelayNode::new(SR, 1, params);
        let out = run_mono(&mut node, &impulse(1_024));
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(rms(&out) > 1e-4, "a mono line must still echo back on itself");
    }
}
