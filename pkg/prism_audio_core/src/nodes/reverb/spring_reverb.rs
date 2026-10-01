//! Spring reverb: a physically motivated model of the dispersive metal-spring
//! reverberation tanks found in guitar amplifiers and vintage studio units.
//!
//! A real spring tank sends the signal as a torsional wave down a coiled metal
//! spring. The spring is strongly *dispersive*: different frequencies travel at
//! different speeds, so a sharp transient smears into the characteristic
//! metallic chirp (the "boing"), and the wave reflects back and forth from the
//! transducers at each end to build a decaying tail.
//!
//! # Model
//!
//! Each channel runs one recirculating delay line (the spring's travel time)
//! whose feedback loop contains two shaping stages:
//!
//! - a cascade of first-order all-pass filters that imparts the
//!   frequency-dependent group delay (the dispersion / chirp), and
//! - a one-pole low-pass that damps the brightest partials a little more on
//!   every pass, as the metal loses high-frequency energy.
//!
//! A scalar `decay` term closes the loop to set the tail length, and a dry /
//! wet control blends the dispersive wash against the untouched input. The
//! first-order all-pass used for dispersion is
//! `y[n] = a*x[n] + x[n-1] - a*y[n-1]` with `|a| < 1`, which has unity
//! magnitude at every frequency (so it only re-phases, never boosts) and a
//! group delay that rises toward low frequencies -- the essence of the spring
//! chirp. Cascading several stages deepens the dispersion.
//!
//! # Relationship
//!
//! Unlike the room-ambience reverbs in this family --
//! [`FdnReverb`](crate::nodes::reverb::fdn::FdnReverb) (a feedback delay
//! network), [`AlgorithmicRoom`](crate::nodes::reverb::algorithmic::AlgorithmicRoom)
//! (Schroeder combs plus diffusers),
//! [`PlateReverb`](crate::nodes::reverb::plate::PlateReverb) (a Dattorro plate
//! tank), and [`Convolver`](crate::nodes::reverb::convolver::Convolver) (a
//! measured impulse response) -- this node's defining feature is the all-pass
//! dispersion chain inside a short single delay loop, which produces the
//! chirped, metallic timbre no diffuse-field room model targets.
//!
//! # Real-time contract
//!
//! Every delay line and filter memory is allocated once in
//! [`SpringReverbNode::new`], sized for the maximum spring length and channel
//! count. [`SpringReverbNode::process`] performs no allocation, locking, or
//! panic on the hot path; non-finite input samples are treated as silence and
//! every value fed back into the loop is denormal-flushed so the tail cannot
//! stall on denormals. The dry path passes straight through, so the node
//! reports no compensating latency ([`SpringReverbNode::latency_frames`]
//! returns zero).
//!
//! # Provenance
//!
//! Modelling a spring reverb as a short feedback delay with a cascade of
//! dispersive all-pass filters and an in-loop damping low-pass is a standard,
//! publicly documented classic-DSP technique (for example the parametric
//! spring-reverb literature by Valimaki, Parker, and Abel, and the general
//! all-pass dispersion material in Zoelzer's DAFX). This is pure classic DSP
//! with no AI or ML. This module contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web Audio source or
//! derived code**; only the publicly documented all-pass, one-pole low-pass,
//! and feedback-delay formulas are used.

use alloc::{vec, vec::Vec};

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Largest spring-length (loop delay) in milliseconds.
pub const MAX_SPRING_SIZE_MS: Sample = 100.0;

/// Smallest spring-length (loop delay) in milliseconds.
pub const MIN_SPRING_SIZE_MS: Sample = 5.0;

/// Largest stable loop feedback, kept below unity so the tail always decays.
pub const MAX_SPRING_FEEDBACK: Sample = 0.98;

/// Number of cascaded first-order all-pass dispersion stages in the loop.
pub const SPRING_ALLPASS_STAGES: usize = 8;

/// Default spring length in milliseconds.
pub const DEFAULT_SPRING_SIZE_MS: Sample = 30.0;

/// Default loop feedback (tail length).
pub const DEFAULT_SPRING_DECAY: Sample = 0.85;

/// Default dispersion amount.
pub const DEFAULT_SPRING_DISPERSION: Sample = 0.6;

/// Default in-loop damping.
pub const DEFAULT_SPRING_DAMPING: Sample = 0.3;

/// Default dry / wet mix.
pub const DEFAULT_SPRING_MIX: Sample = 0.3;

/// Parameters shared by every channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpringReverbParams {
    /// Spring length as the loop delay in milliseconds (`[5, 100]`).
    pub size_ms: Sample,
    /// Loop feedback in `[0, MAX_SPRING_FEEDBACK]`: higher is a longer tail.
    pub decay: Sample,
    /// Dispersion amount in `[0, 1]`: `0` is a plain delay, `1` is the deepest
    /// metallic chirp.
    pub dispersion: Sample,
    /// In-loop damping in `[0, 1]`: `0` keeps the tail bright, `1` is darkest.
    pub damping: Sample,
    /// Dry / wet mix in `[0, 1]`.
    pub mix: Sample,
}

impl Default for SpringReverbParams {
    fn default() -> Self {
        Self {
            size_ms: DEFAULT_SPRING_SIZE_MS,
            decay: DEFAULT_SPRING_DECAY,
            dispersion: DEFAULT_SPRING_DISPERSION,
            damping: DEFAULT_SPRING_DAMPING,
            mix: DEFAULT_SPRING_MIX,
        }
    }
}

impl SpringReverbParams {
    /// Returns the parameters with every field clamped into range and any
    /// non-finite field replaced by its default.
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let clamp = |v: Sample, lo: Sample, hi: Sample, fallback: Sample| {
            if v.is_finite() { v.clamp(lo, hi) } else { fallback }
        };
        Self {
            size_ms: clamp(self.size_ms, MIN_SPRING_SIZE_MS, MAX_SPRING_SIZE_MS, d.size_ms),
            decay: clamp(self.decay, 0.0, MAX_SPRING_FEEDBACK, d.decay),
            dispersion: clamp(self.dispersion, 0.0, 1.0, d.dispersion),
            damping: clamp(self.damping, 0.0, 1.0, d.damping),
            mix: clamp(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// A dispersive spring-tank reverb.
#[derive(Clone, Debug)]
pub struct SpringReverbNode {
    sample_rate: u32,
    channels: usize,
    /// Ring length in frames, shared by every channel.
    ring_len: usize,
    /// Largest addressable loop delay in frames (`ring_len - 2`).
    max_delay: usize,
    /// Active loop delay in frames.
    delay: usize,
    /// One ring buffer per channel.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every ring.
    write_pos: usize,
    /// In-loop one-pole low-pass memory per channel.
    lp_state: Vec<Sample>,
    /// All-pass `x[n-1]` memory per channel per stage.
    ap_x1: Vec<Vec<Sample>>,
    /// All-pass `y[n-1]` memory per channel per stage.
    ap_y1: Vec<Vec<Sample>>,
    /// Loop feedback coefficient.
    decay: Sample,
    /// First-order all-pass coefficient `a` derived from dispersion.
    ap_coef: Sample,
    /// One-pole low-pass coefficient in `(0, 1]` (`1` passes unfiltered).
    damp_coef: Sample,
    wet: Sample,
    dry: Sample,
}

impl SpringReverbNode {
    /// Builds a spring reverb for `channels` channels at `sample_rate` Hz.
    ///
    /// ```
    /// use prism_audio_core::nodes::reverb::spring_reverb::{
    ///     SpringReverbNode, SpringReverbParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = SpringReverbNode::new(48_000, 2, SpringReverbParams::default());
    /// // The dry path passes through, so the node adds no reported latency.
    /// assert_eq!(node.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: SpringReverbParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let max_delay =
            ops::floor(MAX_SPRING_SIZE_MS * sample_rate as Sample / 1_000.0) as usize + 1;
        let ring_len = max_delay + 2;
        let mut node = Self {
            sample_rate,
            channels,
            ring_len,
            max_delay,
            delay: 1,
            rings: vec![vec![0.0; ring_len]; channels],
            write_pos: 0,
            lp_state: vec![0.0; channels],
            ap_x1: vec![vec![0.0; SPRING_ALLPASS_STAGES]; channels],
            ap_y1: vec![vec![0.0; SPRING_ALLPASS_STAGES]; channels],
            decay: 0.0,
            ap_coef: 0.0,
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

    /// Replaces the spring length, decay, dispersion, damping, and mix. All
    /// delay-line and filter memory is preserved so automation stays
    /// click-free.
    pub fn set_params(&mut self, params: SpringReverbParams) {
        let p = params.sanitised();
        let frames = ops::floor(p.size_ms * self.sample_rate as Sample / 1_000.0) as usize;
        self.delay = frames.clamp(1, self.max_delay);
        self.decay = p.decay;
        // Map dispersion to an all-pass coefficient below unity for stability.
        self.ap_coef = p.dispersion * 0.9;
        // damping 0 -> coef 1 (bright, no filtering); damping 1 -> coef 0.1.
        self.damp_coef = 1.0 - 0.9 * p.damping;
        self.wet = p.mix;
        self.dry = 1.0 - p.mix;
    }
}

impl AudioNode for SpringReverbNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let ring_len = self.ring_len;
        let delay = self.delay;
        let decay = self.decay;
        let a = self.ap_coef;
        let damp_coef = self.damp_coef;
        let wet = self.wet;
        let dry = self.dry;

        for f in 0..frames {
            let write_pos = self.write_pos;
            let read_pos = (write_pos + ring_len - delay) % ring_len;

            for ch in 0..channels {
                let x_raw = input.channel(ch)[f];
                let x = if x_raw.is_finite() { x_raw } else { 0.0 };

                let delayed = self.rings[ch][read_pos];

                // In-loop one-pole damping low-pass.
                let lp = self.lp_state[ch] + damp_coef * (delayed - self.lp_state[ch]);
                self.lp_state[ch] = flush_denormal(lp);

                // Dispersion all-pass cascade.
                let mut d = lp;
                let xs = &mut self.ap_x1[ch];
                let ys = &mut self.ap_y1[ch];
                for (x1, y1) in xs.iter_mut().zip(ys.iter_mut()) {
                    let y = a * d + *x1 - a * *y1;
                    *x1 = d;
                    *y1 = flush_denormal(y);
                    d = y;
                }

                let write_val = x + decay * d;
                self.rings[ch][write_pos] = flush_denormal(write_val);
                output.channel_mut(ch)[f] = dry * x + wet * d;
            }

            self.write_pos = if write_pos + 1 == ring_len { 0 } else { write_pos + 1 };
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
        for s in &mut self.lp_state {
            *s = 0.0;
        }
        for stage in &mut self.ap_x1 {
            for s in stage.iter_mut() {
                *s = 0.0;
            }
        }
        for stage in &mut self.ap_y1 {
            for s in stage.iter_mut() {
                *s = 0.0;
            }
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

    fn run_mono(node: &mut SpringReverbNode, signal: &[Sample]) -> Vec<Sample> {
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
        let sum: Sample = samples.iter().map(|&v| v * v).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    #[test]
    fn reports_zero_latency() {
        let node = SpringReverbNode::new(SR, 2, SpringReverbParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn channels_getter_reports_build_width() {
        let node = SpringReverbNode::new(SR, 2, SpringReverbParams::default());
        assert_eq!(node.channels(), 2);
    }

    #[test]
    fn default_params_are_in_domain() {
        let d = SpringReverbParams::default();
        assert_eq!(d, d.sanitised());
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = SpringReverbNode::new(SR, 1, SpringReverbParams::default());
        let out = run_mono(&mut node, &vec![0.0; 8_000]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dry_mix_passes_input_through_exactly() {
        let params = SpringReverbParams {
            mix: 0.0,
            ..SpringReverbParams::default()
        };
        let mut node = SpringReverbNode::new(SR, 1, params);
        let sig: Vec<Sample> = (0..6_000).map(|i| ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &sig);
        for (o, s) in out.iter().zip(sig.iter()) {
            assert!((o - s).abs() < 1e-6);
        }
    }

    #[test]
    fn impulse_produces_a_decaying_tail() {
        let params = SpringReverbParams {
            mix: 1.0,
            ..SpringReverbParams::default()
        };
        let mut node = SpringReverbNode::new(SR, 1, params);
        let out = run_mono(&mut node, &impulse(16_000));
        let delay = node.delay;
        // Energy appears after the first loop traversal.
        let early = rms(&out[delay..delay + 4_000]);
        let late = rms(&out[12_000..16_000]);
        assert!(early > 1e-4, "early tail should carry energy: {early}");
        assert!(late < early, "tail should decay: early {early}, late {late}");
        assert!(late > 0.0, "tail should still be audible mid-decay");
    }

    #[test]
    fn higher_decay_lengthens_the_tail() {
        let short = SpringReverbParams {
            mix: 1.0,
            decay: 0.5,
            ..SpringReverbParams::default()
        };
        let long = SpringReverbParams {
            mix: 1.0,
            decay: 0.95,
            ..SpringReverbParams::default()
        };
        let mut node_short = SpringReverbNode::new(SR, 1, short);
        let mut node_long = SpringReverbNode::new(SR, 1, long);
        let tail_short = rms(&run_mono(&mut node_short, &impulse(20_000))[12_000..]);
        let tail_long = rms(&run_mono(&mut node_long, &impulse(20_000))[12_000..]);
        assert!(tail_long > tail_short, "short {tail_short}, long {tail_long}");
    }

    #[test]
    fn dispersion_smears_the_impulse_response() {
        // With no dispersion the first echo is a near-isolated tap; with
        // dispersion the energy spreads across more samples around that echo.
        let plain = SpringReverbParams {
            mix: 1.0,
            dispersion: 0.0,
            damping: 0.0,
            decay: 0.0,
            ..SpringReverbParams::default()
        };
        let dispersed = SpringReverbParams {
            dispersion: 0.85,
            ..plain
        };
        let mut node_plain = SpringReverbNode::new(SR, 1, plain);
        let mut node_disp = SpringReverbNode::new(SR, 1, dispersed);
        let out_plain = run_mono(&mut node_plain, &impulse(4_000));
        let out_disp = run_mono(&mut node_disp, &impulse(4_000));
        let count_above = |v: &[Sample]| v.iter().filter(|&&s| s.abs() > 1e-3).count();
        assert!(
            count_above(&out_disp) > count_above(&out_plain),
            "dispersed {} should spread wider than plain {}",
            count_above(&out_disp),
            count_above(&out_plain)
        );
    }

    #[test]
    fn minimum_size_is_clamped() {
        let params = SpringReverbParams {
            size_ms: 0.0,
            ..SpringReverbParams::default()
        };
        let node = SpringReverbNode::new(SR, 1, params);
        let expected = ops::floor(MIN_SPRING_SIZE_MS * SR as Sample / 1_000.0) as usize;
        assert_eq!(node.delay, expected);
    }

    #[test]
    fn maximum_size_is_clamped() {
        let params = SpringReverbParams {
            size_ms: 10_000.0,
            ..SpringReverbParams::default()
        };
        let node = SpringReverbNode::new(SR, 1, params);
        assert!(node.delay <= node.max_delay);
        let expected = ops::floor(MAX_SPRING_SIZE_MS * SR as Sample / 1_000.0) as usize;
        assert_eq!(node.delay, expected);
    }

    #[test]
    fn feedback_is_clamped() {
        let params = SpringReverbParams {
            decay: 9.0,
            ..SpringReverbParams::default()
        };
        let node = SpringReverbNode::new(SR, 1, params);
        assert!(node.decay <= MAX_SPRING_FEEDBACK);
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = SpringReverbNode::new(SR, 1, SpringReverbParams::default());
        let mut sig = vec![0.5; 8_000];
        sig[10] = Sample::INFINITY;
        sig[20] = Sample::NAN;
        sig[30] = Sample::NEG_INFINITY;
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = SpringReverbNode::new(SR, 1, SpringReverbParams::default());
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
    fn extreme_params_do_not_panic() {
        let params = SpringReverbParams {
            size_ms: Sample::INFINITY,
            decay: 1e9,
            dispersion: -5.0,
            damping: 42.0,
            mix: 100.0,
        };
        let mut node = SpringReverbNode::new(SR, 2, params);
        let _ = run_mono(&mut node, &vec![0.3; 10_000]);
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let params = SpringReverbParams {
            size_ms: Sample::NAN,
            decay: Sample::INFINITY,
            dispersion: Sample::NAN,
            damping: Sample::NEG_INFINITY,
            mix: 0.4,
        };
        let s = params.sanitised();
        let d = SpringReverbParams::default();
        assert_eq!(s.size_ms, d.size_ms);
        assert_eq!(s.decay, d.decay);
        assert_eq!(s.dispersion, d.dispersion);
        assert_eq!(s.damping, d.damping);
        assert_eq!(s.mix, 0.4);
    }

    #[test]
    fn high_feedback_impulse_stays_bounded() {
        let params = SpringReverbParams {
            mix: 1.0,
            decay: MAX_SPRING_FEEDBACK,
            ..SpringReverbParams::default()
        };
        let mut node = SpringReverbNode::new(SR, 1, params);
        let out = run_mono(&mut node, &impulse(48_000));
        assert!(out.iter().all(|v| v.is_finite()));
        let peak = out.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        assert!(peak < 100.0, "tail should stay bounded, peak {peak}");
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = SpringReverbNode::new(SR, 2, SpringReverbParams::default());
        let len = 2_000;
        let mut input = AudioBuffer::new(ChannelLayout::Quad, len);
        let mut output = AudioBuffer::new(ChannelLayout::Quad, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for ch in 0..4 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                *s = ops::sin(i as Sample * 0.01 + ch as Sample);
            }
        }
        let expected2 = input.channel(2).to_vec();
        let expected3 = input.channel(3).to_vec();
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert_eq!(outputs[0].channel(2), expected2.as_slice());
        assert_eq!(outputs[0].channel(3), expected3.as_slice());
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = SpringReverbParams {
            mix: 1.0,
            ..SpringReverbParams::default()
        };
        let mut node = SpringReverbNode::new(SR, 1, params);
        let sig = impulse(8_000);
        let first = run_mono(&mut node, &sig);
        node.reset();
        let second = run_mono(&mut node, &sig);
        let max_err = first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-6);
    }

    #[test]
    fn set_params_changes_size() {
        let mut node = SpringReverbNode::new(SR, 1, SpringReverbParams::default());
        let before = node.delay;
        node.set_params(SpringReverbParams {
            size_ms: 80.0,
            ..SpringReverbParams::default()
        });
        let after = node.delay;
        let expected = ops::floor(80.0 * SR as Sample / 1_000.0) as usize;
        assert_eq!(after, expected);
        assert_ne!(before, after);
        let out = run_mono(&mut node, &vec![0.3; 10_000]);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
