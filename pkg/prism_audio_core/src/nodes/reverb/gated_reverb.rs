//! Gated reverb: a dense reverb tail hard-cut by a key-driven noise gate.
//!
//! The gated reverb is the signature "big, punchy, truncated" ambience of
//! 1980s record production (famously the exploding drum sound of that era).
//! The recipe is deliberately simple: run a source through a bright, dense
//! room reverb, then feed the wet tail into a noise gate whose side-chain key
//! is the *dry* source. While the source is loud the gate stays open and the
//! full reverb blooms; the instant the source stops the gate holds briefly and
//! then slams shut, chopping the natural decay off abruptly. The ear hears a
//! huge burst of ambience with none of the long, muddy ring-out, which both
//! enlarges a transient and keeps dense arrangements clean.
//!
//! This node realises that chain from two reusable pieces:
//!
//! 1. A fully-wet [`AlgorithmicRoom`] produces the dense reverberant tail. It
//!    is driven with `wet = 1`, `dry = 0`, so this node owns the final dry /
//!    wet balance.
//! 2. A per-block **gate envelope** tracks the dry input's peak level. When the
//!    peak crosses `threshold_db` the gate opens with an `attack_ms` ramp; once
//!    the peak falls back below threshold a `hold_ms` timer keeps it open to
//!    avoid chattering on decaying material, after which it closes with a short
//!    `release_ms` ramp. The resulting gain multiplies the wet tail only.
//!
//! The output is `(1 - mix) * dry + mix * gate * wet_tail`, so the dry source
//! is always passed through and `mix` sets how much gated ambience is layered
//! on top. The gate gain is a single mono envelope applied uniformly to every
//! channel, which keeps the stereo tail coherent as it is cut.
//!
//! # Relationship
//!
//! This processor *composes* existing engine building blocks rather than
//! duplicating any DSP. The reverberant field is produced verbatim by
//! [`AlgorithmicRoom`](crate::nodes::reverb::algorithmic::AlgorithmicRoom); the
//! gate envelope is a small local attack/hold/release follower in the spirit of
//! the dynamics-family
//! [`ExpanderGateNode`](crate::nodes::dynamics::gate::GateParams), but applied
//! as a *multiplicative tail gate keyed from the dry source* instead of a
//! downward expander on its own input. Compared with the plain
//! [`AlgorithmicRoom`] it differs only by the key-driven gate on the wet path;
//! compared with a standalone gate it differs by gating a reverb tail with the
//! dry signal as the side-chain key. No other reverb in the family truncates
//! its own decay this way.
//!
//! # Real-time contract
//!
//! The inner reverb, the wet scratch buffer, and all envelope state are
//! allocated at construction. [`GatedReverbNode::process`] performs no
//! allocation, locking, or panicking on the audio thread: it only runs the
//! pre-allocated reverb, advances the scalar gate envelope, and mixes. Wet
//! samples are flushed of denormals before mixing, and non-finite inputs are
//! treated as silence.
//!
//! # Provenance
//!
//! Pure classic DSP. The gated-reverb technique is a widely and publicly
//! documented 1980s studio practice (reverb into a noise gate keyed by the
//! source); the room model is the engine's own Schroeder/Freeverb-style
//! [`AlgorithmicRoom`] and the gate is a textbook attack/hold/release envelope
//! follower (Zoelzer, DAFX). There is no AI/ML of any kind, and no UE/Unity/
//! Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio source or
//! derived code.

use bevy_math::ops;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::reverb::algorithmic::{AlgorithmicRoom, AlgorithmicRoomParams};
use crate::param::{Ramp, Smoothed};

/// Largest pre-delay the gated reverb accepts, in milliseconds.
pub const MAX_GATED_PRE_DELAY_MS: Sample = 200.0;

/// Largest gate hold time, in milliseconds.
pub const MAX_GATED_HOLD_MS: Sample = 2_000.0;

/// Smallest gate attack or release time, in milliseconds, to keep the one-pole
/// coefficient well-defined.
pub const MIN_GATED_TIME_MS: Sample = 0.01;

/// Default room size (large, for a dense bloom).
pub const DEFAULT_GATED_ROOM_SIZE: Sample = 0.85;

/// Default high-frequency damping of the tail.
pub const DEFAULT_GATED_DAMPING: Sample = 0.3;

/// Default pre-delay in milliseconds.
pub const DEFAULT_GATED_PRE_DELAY_MS: Sample = 8.0;

/// Default gate open threshold in dBFS.
pub const DEFAULT_GATED_THRESHOLD_DB: Sample = -40.0;

/// Default gate attack (open) time in milliseconds.
pub const DEFAULT_GATED_ATTACK_MS: Sample = 1.0;

/// Default gate hold time in milliseconds.
pub const DEFAULT_GATED_HOLD_MS: Sample = 120.0;

/// Default gate release (close) time in milliseconds; short for a hard cut.
pub const DEFAULT_GATED_RELEASE_MS: Sample = 8.0;

/// Default dry / wet mix.
pub const DEFAULT_GATED_MIX: Sample = 0.5;

/// Construction parameters for a [`GatedReverbNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GatedReverbParams {
    /// Room size in `[0, 1]`; larger yields a denser, longer pre-gate tail.
    pub room_size: Sample,
    /// High-frequency damping in `[0, 1]`; higher darkens the tail.
    pub damping: Sample,
    /// Pre-delay before the tail, in milliseconds (clamped to
    /// `[0, MAX_GATED_PRE_DELAY_MS]`).
    pub pre_delay_ms: Sample,
    /// Gate open threshold in dBFS, measured on the dry side-chain key.
    pub threshold_db: Sample,
    /// Gate attack (open) time in milliseconds.
    pub attack_ms: Sample,
    /// Gate hold time in milliseconds after the key drops below threshold.
    pub hold_ms: Sample,
    /// Gate release (close) time in milliseconds; short values chop the tail.
    pub release_ms: Sample,
    /// Dry / wet mix in `[0, 1]`: `0` is dry only, `1` is fully gated ambience.
    pub mix: Sample,
}

impl Default for GatedReverbParams {
    fn default() -> Self {
        Self {
            room_size: DEFAULT_GATED_ROOM_SIZE,
            damping: DEFAULT_GATED_DAMPING,
            pre_delay_ms: DEFAULT_GATED_PRE_DELAY_MS,
            threshold_db: DEFAULT_GATED_THRESHOLD_DB,
            attack_ms: DEFAULT_GATED_ATTACK_MS,
            hold_ms: DEFAULT_GATED_HOLD_MS,
            release_ms: DEFAULT_GATED_RELEASE_MS,
            mix: DEFAULT_GATED_MIX,
        }
    }
}

impl GatedReverbParams {
    /// Returns a copy with every field clamped to its valid domain; any
    /// non-finite field falls back to its default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let fix = |v: Sample, lo: Sample, hi: Sample, def: Sample| {
            if v.is_finite() { v.clamp(lo, hi) } else { def }
        };
        Self {
            room_size: fix(self.room_size, 0.0, 1.0, d.room_size),
            damping: fix(self.damping, 0.0, 1.0, d.damping),
            pre_delay_ms: fix(self.pre_delay_ms, 0.0, MAX_GATED_PRE_DELAY_MS, d.pre_delay_ms),
            threshold_db: fix(self.threshold_db, -120.0, 0.0, d.threshold_db),
            attack_ms: fix(self.attack_ms, MIN_GATED_TIME_MS, 1_000.0, d.attack_ms),
            hold_ms: fix(self.hold_ms, 0.0, MAX_GATED_HOLD_MS, d.hold_ms),
            release_ms: fix(self.release_ms, MIN_GATED_TIME_MS, 2_000.0, d.release_ms),
            mix: fix(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// One-pole smoothing coefficient for a `ms` time constant at `sr` Hz. A
/// non-positive time yields `0` (an instantaneous transition).
#[inline]
fn smoothing_coef(ms: Sample, sr: Sample) -> Sample {
    if ms <= 0.0 || sr <= 0.0 {
        0.0
    } else {
        ops::exp(-1.0 / (ms * 0.001 * sr))
    }
}

/// Builds the fully-wet inner-reverb parameters from the gated-reverb controls.
#[inline]
fn room_params(p: &GatedReverbParams) -> AlgorithmicRoomParams {
    AlgorithmicRoomParams {
        room_size: p.room_size,
        damping: p.damping,
        width: 1.0,
        pre_delay_ms: p.pre_delay_ms,
        early_level: 0.5,
        // The tank is a pure wet send; this node owns the dry / wet mix.
        wet: 1.0,
        dry: 0.0,
    }
}

/// A gated reverb (input port 0 -> output port 0).
///
/// A fully-wet [`AlgorithmicRoom`] feeds a key-driven gate so the dense tail is
/// cut abruptly once the dry source stops, producing the punchy, truncated
/// ambience of 1980s productions.
#[derive(Debug, Clone)]
pub struct GatedReverbNode {
    /// Sample rate in Hz, cached for cold-path coefficient recomputation.
    sample_rate: u32,
    /// Number of channels processed.
    channels: usize,
    /// Channel layout shared by the wet scratch buffer.
    layout: ChannelLayout,
    /// Maximum block size, in frames, the scratch buffer can hold.
    max_block_frames: usize,
    /// Fully-wet inner reverb producing the tail.
    reverb: AlgorithmicRoom,
    /// Scratch buffer holding a finite-sanitised copy of the dry input fed
    /// to the inner reverb (keeps non-finite samples out of its state).
    in_buf: AudioBuffer,
    /// Scratch buffer holding the reverb's wet output for one block.
    wet_buf: AudioBuffer,
    /// Cached pre-delay so `set_params` only rebuilds the reverb when it moves.
    pre_delay_ms: Sample,
    /// Linear gate-open threshold applied to the dry side-chain key.
    threshold_lin: Sample,
    /// One-pole coefficient for the opening (attack) transition.
    attack_coef: Sample,
    /// One-pole coefficient for the closing (release) transition.
    release_coef: Sample,
    /// Hold length in frames once the key falls below threshold.
    hold_frames: usize,
    /// Current gate gain in `[0, 1]`.
    gate_gain: Sample,
    /// Remaining hold frames before the gate is allowed to close.
    hold_counter: usize,
    /// Smoothed dry / wet mix.
    mix: Smoothed,
}

impl GatedReverbNode {
    /// Builds a gated reverb for `layout` at `sample_rate` Hz that can process
    /// up to `max_block_frames` frames per call. The inner reverb and the wet
    /// scratch buffer are allocated here; parameters start settled.
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::reverb::gated_reverb::{
    ///     GatedReverbNode, GatedReverbParams,
    /// };
    ///
    /// let params = GatedReverbParams::default();
    /// let mut verb = GatedReverbNode::new(48_000, ChannelLayout::Stereo, 256, params);
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
    /// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// input.set_active_frames(256);
    /// let inputs = [input];
    /// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// output.set_active_frames(256);
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// verb.process(&ctx, &mut io);
    /// assert_eq!(verb.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_block_frames: usize,
        params: GatedReverbParams,
    ) -> Self {
        let p = params.sanitised();
        let sr = sample_rate.max(1);
        let channels = layout.channel_count().max(1);
        let cap = max_block_frames.max(1);
        let srf = sr as Sample;

        let reverb = AlgorithmicRoom::new(sr, channels, room_params(&p));

        Self {
            sample_rate: sr,
            channels,
            layout,
            max_block_frames: cap,
            reverb,
            in_buf: AudioBuffer::new(layout, cap),
            wet_buf: AudioBuffer::new(layout, cap),
            pre_delay_ms: p.pre_delay_ms,
            threshold_lin: db_to_linear(p.threshold_db),
            attack_coef: smoothing_coef(p.attack_ms, srf),
            release_coef: smoothing_coef(p.release_ms, srf),
            hold_frames: ops::round(p.hold_ms * 0.001 * srf) as usize,
            gate_gain: 0.0,
            hold_counter: 0,
            mix: Smoothed::new(p.mix),
        }
    }

    /// Returns the number of channels this reverb processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the channel layout of the wet scratch buffer.
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

    /// Updates every parameter. The gate coefficients and threshold are
    /// recomputed in place; the inner reverb is rebuilt only when the pre-delay
    /// changes (room size and damping are updated in place otherwise). This is
    /// a control-thread operation, not called from [`AudioNode::process`].
    pub fn set_params(&mut self, params: GatedReverbParams) {
        let p = params.sanitised();
        let srf = self.sample_rate as Sample;

        if (p.pre_delay_ms - self.pre_delay_ms).abs() > Sample::EPSILON {
            self.reverb = AlgorithmicRoom::new(self.sample_rate, self.channels, room_params(&p));
            self.pre_delay_ms = p.pre_delay_ms;
        } else {
            self.reverb.set_room_size(p.room_size);
            self.reverb.set_damping(p.damping);
        }

        self.threshold_lin = db_to_linear(p.threshold_db);
        self.attack_coef = smoothing_coef(p.attack_ms, srf);
        self.release_coef = smoothing_coef(p.release_ms, srf);
        self.hold_frames = ops::round(p.hold_ms * 0.001 * srf) as usize;
        self.mix.set_target(p.mix, Ramp::Immediate);
    }

    /// Advances the mono gate envelope one frame for a dry-key peak `key` and
    /// returns the gate gain to apply to the wet tail.
    #[inline]
    fn gate_step(&mut self, key: Sample) -> Sample {
        let open = if key >= self.threshold_lin {
            self.hold_counter = self.hold_frames;
            true
        } else if self.hold_counter > 0 {
            self.hold_counter -= 1;
            true
        } else {
            false
        };
        let target = if open { 1.0 } else { 0.0 };
        let coef = if target > self.gate_gain {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.gate_gain = target + (self.gate_gain - target) * coef;
        self.gate_gain
    }
}

impl AudioNode for GatedReverbNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let cap = self.max_block_frames;
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output
            .active_frames()
            .min(input.active_frames())
            .min(cap);
        if frames == 0 || out_channels == 0 {
            return;
        }
        let rev_ch = self.channels;
        self.wet_buf.set_active_frames(frames);
        self.in_buf.set_active_frames(frames);

        // 1. Copy a finite-sanitised dry input into the reverb-feed scratch so
        //    non-finite samples never corrupt the inner reverb state.
        for ch in 0..rev_ch {
            let src_ch = if ch < in_channels { ch } else { usize::MAX };
            let dst = self.in_buf.channel_mut(ch);
            for (n, d) in dst[..frames].iter_mut().enumerate() {
                let x = if src_ch == usize::MAX { 0.0 } else { input.channel(src_ch)[n] };
                *d = if x.is_finite() { x } else { 0.0 };
            }
        }

        // 2. Run the fully-wet reverb: sanitised dry input -> wet scratch.
        {
            let inputs = core::slice::from_ref(&self.in_buf);
            let outputs = core::slice::from_mut(&mut self.wet_buf);
            let mut rio = ProcessIo::new(inputs, outputs);
            self.reverb.process(ctx, &mut rio);
        }

        // 3. For each frame, derive the dry-key peak, advance the gate, and mix
        //    the dry source with the gated wet tail.
        for n in 0..frames {
            let mut key = 0.0;
            for ch in 0..in_channels {
                let x = input.channel(ch)[n];
                let x = if x.is_finite() { x } else { 0.0 };
                let a = x.abs();
                if a > key {
                    key = a;
                }
            }
            let gate = self.gate_step(key);
            let mix = self.mix.next_sample();
            for ch in 0..out_channels {
                let x = if ch < in_channels {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                } else {
                    0.0
                };
                let wet = if ch < rev_ch {
                    flush_denormal(self.wet_buf.channel(ch)[n])
                } else {
                    0.0
                };
                output.channel_mut(ch)[n] = (1.0 - mix) * x + mix * gate * wet;
            }
        }
    }

    fn reset(&mut self) {
        self.reverb.reset();
        self.in_buf.clear();
        self.wet_buf.clear();
        self.gate_gain = 0.0;
        self.hold_counter = 0;
        self.mix = Smoothed::new(self.mix.target());
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

    fn run_mono(node: &mut GatedReverbNode, signal: &[Sample]) -> Vec<Sample> {
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

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().map(|&v| v * v).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    /// A short burst of full-scale tone followed by silence; the burst keys the
    /// gate open and the tail should be audible then cut.
    fn burst(burst_len: usize, total: usize) -> Vec<Sample> {
        (0..total)
            .map(|i| {
                if i < burst_len {
                    ops::sin(i as Sample * 0.08)
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn reports_zero_latency() {
        let node = GatedReverbNode::new(SR, ChannelLayout::Stereo, 256, GatedReverbParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn channels_getter_reports_build_width() {
        let node = GatedReverbNode::new(SR, ChannelLayout::Stereo, 256, GatedReverbParams::default());
        assert_eq!(node.channels(), 2);
    }

    #[test]
    fn layout_and_block_getters_report_build_values() {
        let node = GatedReverbNode::new(SR, ChannelLayout::Stereo, 128, GatedReverbParams::default());
        assert_eq!(node.layout(), ChannelLayout::Stereo);
        assert_eq!(node.max_block_frames(), 128);
    }

    #[test]
    fn default_params_are_in_domain() {
        let d = GatedReverbParams::default();
        assert_eq!(d, d.sanitised());
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 4_096, GatedReverbParams::default());
        let out = run_mono(&mut node, &vec![0.0; 8_000]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dry_only_mix_passes_input_through_exactly() {
        let params = GatedReverbParams {
            mix: 0.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 8_192, params);
        let sig: Vec<Sample> = (0..6_000).map(|i| ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &sig);
        for (o, s) in out.iter().zip(sig.iter()) {
            assert!((o - s).abs() < 1e-6, "dry mix should be identity: {o} vs {s}");
        }
    }

    #[test]
    fn gate_cuts_the_tail_after_hold() {
        // A loud burst then a long silence. With a short hold and release the
        // tail present right after the burst must be far louder than the tail
        // deep in the silence, proving the gate chopped the decay.
        let params = GatedReverbParams {
            mix: 1.0,
            hold_ms: 40.0,
            release_ms: 5.0,
            threshold_db: -30.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, params);
        let burst_len = 2_400; // 50 ms
        let out = run_mono(&mut node, &burst(burst_len, 48_000));
        // Window just after the burst ends (gate still open / holding).
        let early = rms(&out[burst_len..burst_len + 2_000]);
        // Window ~0.5 s later (gate fully closed well past hold + release).
        let late = rms(&out[30_000..32_000]);
        assert!(early > 1e-3, "tail should bloom right after the burst: {early}");
        assert!(
            late < early * 0.05,
            "gate should cut the tail: early {early}, late {late}"
        );
    }

    #[test]
    fn ungated_room_keeps_a_long_tail() {
        // With a very low threshold and a long hold the gate essentially never
        // closes, so a comparable room produces a tail that is still audible
        // where the gated version was silent -- confirming the gate (not the
        // reverb) is what truncates the decay.
        let open = GatedReverbParams {
            mix: 1.0,
            threshold_db: -120.0,
            hold_ms: MAX_GATED_HOLD_MS,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, open);
        let out = run_mono(&mut node, &burst(2_400, 48_000));
        let late = rms(&out[30_000..32_000]);
        assert!(late > 1e-4, "open gate should keep a tail: {late}");
    }

    #[test]
    fn quiet_input_below_threshold_stays_gated() {
        // A tone well below the open threshold never opens the gate, so the wet
        // path is suppressed and only the dry remainder (mix<1) survives.
        let params = GatedReverbParams {
            mix: 1.0,
            threshold_db: -6.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, params);
        let quiet: Vec<Sample> = (0..20_000).map(|i| 0.01 * ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &quiet);
        // mix=1 means dry is fully removed; a never-opening gate keeps output ~0.
        assert!(rms(&out) < 1e-3, "closed gate should suppress the wet tail: {}", rms(&out));
    }

    #[test]
    fn non_finite_input_is_treated_as_silence() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 4_096, GatedReverbParams::default());
        let mut sig = vec![0.0; 4_000];
        sig[10] = Sample::INFINITY;
        sig[20] = Sample::NAN;
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|&v| v.is_finite()), "output must stay finite");
    }

    #[test]
    fn tone_output_is_finite() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, GatedReverbParams::default());
        let sig: Vec<Sample> = (0..20_000).map(|i| ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 256, GatedReverbParams::default());
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
    fn stereo_channels_are_processed_independently() {
        let params = GatedReverbParams {
            mix: 1.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Stereo, 8_192, params);
        let len = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for i in 0..len {
            input.channel_mut(0)[i] = if i < 100 { ops::sin(i as Sample * 0.08) } else { 0.0 };
            input.channel_mut(1)[i] = 0.0;
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        // The left channel was excited; the two tails should differ.
        let left = rms(outputs[0].channel(0));
        let right = rms(outputs[0].channel(1));
        assert!(left > right, "excited left should exceed silent right: {left} vs {right}");
    }

    #[test]
    fn surplus_output_channels_are_filled() {
        // A quad output from a mono reverb width: extra channels should be
        // silent rather than garbage.
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Stereo, 1_024, GatedReverbParams::default());
        let len = 512;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Quad, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for i in 0..len {
            input.channel_mut(0)[i] = ops::sin(i as Sample * 0.08);
            input.channel_mut(1)[i] = ops::sin(i as Sample * 0.08);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert!(outputs[0].channel(2).iter().all(|&v| v.is_finite()));
        assert!(outputs[0].channel(3).iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = GatedReverbParams {
            mix: 1.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, params);
        let sig = burst(2_400, 20_000);
        let first = run_mono(&mut node, &sig);
        node.reset();
        let second = run_mono(&mut node, &sig);
        let max_err = first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-6, "reset should reproduce a fresh run: {max_err}");
    }

    #[test]
    fn set_params_updates_mix_and_gate() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 8_192, GatedReverbParams::default());
        node.set_params(GatedReverbParams {
            mix: 0.0,
            ..GatedReverbParams::default()
        });
        let sig: Vec<Sample> = (0..4_000).map(|i| ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &sig);
        // mix=0 now behaves as dry passthrough.
        for (o, s) in out.iter().zip(sig.iter()).skip(64) {
            assert!((o - s).abs() < 1e-5, "mix=0 should pass dry: {o} vs {s}");
        }
    }

    #[test]
    fn set_params_rebuilds_reverb_on_pre_delay_change() {
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 8_192, GatedReverbParams::default());
        node.set_params(GatedReverbParams {
            pre_delay_ms: 60.0,
            ..GatedReverbParams::default()
        });
        assert!((node.pre_delay_ms - 60.0).abs() < 1e-3);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let params = GatedReverbParams {
            room_size: 9.0,
            damping: -3.0,
            pre_delay_ms: 10_000.0,
            threshold_db: 50.0,
            attack_ms: -1.0,
            hold_ms: 1e9,
            release_ms: -5.0,
            mix: 7.0,
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 4_096, params);
        let out = run_mono(&mut node, &burst(1_000, 8_000));
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let params = GatedReverbParams {
            room_size: Sample::NAN,
            damping: Sample::INFINITY,
            pre_delay_ms: Sample::NAN,
            threshold_db: Sample::NAN,
            attack_ms: Sample::NAN,
            hold_ms: Sample::NAN,
            release_ms: Sample::NAN,
            mix: Sample::NAN,
        };
        assert_eq!(params.sanitised(), GatedReverbParams::default());
    }

    #[test]
    fn longer_hold_sustains_the_tail_longer() {
        let short = GatedReverbParams {
            mix: 1.0,
            hold_ms: 10.0,
            release_ms: 3.0,
            threshold_db: -30.0,
            ..GatedReverbParams::default()
        };
        let long = GatedReverbParams {
            hold_ms: 400.0,
            ..short
        };
        let mut node_short = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, short);
        let mut node_long = GatedReverbNode::new(SR, ChannelLayout::Mono, 48_000, long);
        let sig = burst(2_400, 48_000);
        // Window ~0.3 s after the burst: inside the long hold, past the short one.
        let s = rms(&run_mono(&mut node_short, &sig)[16_000..18_000]);
        let l = rms(&run_mono(&mut node_long, &sig)[16_000..18_000]);
        assert!(l > s, "longer hold should sustain the tail: short {s}, long {l}");
    }

    #[test]
    fn gate_opens_within_attack_on_a_loud_onset() {
        let params = GatedReverbParams {
            mix: 1.0,
            attack_ms: 1.0,
            threshold_db: -30.0,
            ..GatedReverbParams::default()
        };
        let mut node = GatedReverbNode::new(SR, ChannelLayout::Mono, 4_096, params);
        let _ = run_mono(&mut node, &burst(2_000, 4_000));
        // After a 2000-frame loud burst the gate must be essentially fully open.
        assert!(node.gate_gain > 0.9, "gate should open on a loud onset: {}", node.gate_gain);
    }
}
