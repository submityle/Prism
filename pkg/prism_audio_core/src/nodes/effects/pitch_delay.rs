//! Pitch delay: a feedback echo whose recirculating tail is pitch-shifted on
//! every pass, so repeats spiral up or down in pitch.
//!
//! An ordinary feedback delay replays each echo at the same pitch, fading out.
//! A pitch delay inserts a pitch shifter *inside the feedback loop*: the signal
//! read out of the delay line is transposed by `pitch_semitones` before being
//! fed back in, so the first echo is shifted once, the second echo twice, and
//! so on. The tail therefore climbs (for a positive shift) or descends (for a
//! negative shift) in a shimmering, cascading spiral -- the classic "pitch
//! echo" of ambient and sound-design production. With a shift of zero it
//! degrades gracefully into a plain feedback delay.
//!
//! The structure mirrors the engine's shimmer reverb but with a single delay
//! line in place of the reverberant tank:
//!
//! 1. A per-channel integer delay line (ring buffer) provides the echo. The
//!    *wet output tap is read straight from the line*, so the node reports zero
//!    latency even though the shifter inside the loop is latent.
//! 2. The delayed block is transposed by a
//!    [`PitchShifterNode`](crate::nodes::effects::pitch_shifter::PitchShifterNode)
//!    and the result is re-injected (scaled by `feedback`) into the line on the
//!    next block, so each recirculation accumulates another transposition.
//!
//! The output is `(1 - mix) * dry + mix * delayed`.
//!
//! # Relationship
//!
//! This processor *composes* existing building blocks rather than duplicating
//! DSP. The transposition is produced verbatim by the phase-vocoder
//! [`PitchShifterNode`]; only the delay line and the feedback mixing are owned
//! here. Compared with the plain time-domain
//! [`delay::DelayNode`](crate::nodes::effects::delay::DelayNode) it differs by
//! transposing the feedback path; compared with the
//! [`shimmer::ShimmerReverb`](crate::nodes::reverb::shimmer::ShimmerReverb) it
//! differs by recirculating through a discrete delay line (producing distinct,
//! rhythmic echoes) rather than a diffuse reverb tank.
//!
//! # Real-time contract
//!
//! The delay rings, the inner shifter, and the scratch buffers are allocated at
//! construction. [`PitchDelayNode::process`] performs no allocation, locking,
//! or panicking on the audio thread: it advances the ring, runs the
//! pre-allocated shifter, and mixes. Feedback is flushed of denormals and
//! non-finite inputs are treated as silence.
//!
//! # Provenance
//!
//! Pure classic DSP. The pitch-shifted feedback delay is a widely and publicly
//! documented effect topology; the transposition is the engine's own
//! phase-vocoder shifter. There is no AI/ML of any kind, and no UE/Unity/Godot/
//! Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio source or derived
//! code.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::effects::pitch_shifter::{PitchShifterNode, PitchShifterParams, semitones_to_ratio};
use crate::param::{Ramp, Smoothed};

/// Largest delay time the pitch delay accepts, in milliseconds.
pub const MAX_PITCH_DELAY_MS: Sample = 2_000.0;

/// Smallest delay time, in milliseconds.
pub const MIN_PITCH_DELAY_MS: Sample = 1.0;

/// Largest stable feedback gain, kept below unity so the spiral always decays.
pub const MAX_PITCH_DELAY_FEEDBACK: Sample = 0.95;

/// Largest absolute transposition per pass, in semitones.
pub const MAX_PITCH_DELAY_SEMITONES: Sample = 24.0;

/// FFT size of the inner phase-vocoder shifter.
pub const PITCH_DELAY_FFT_SIZE: usize = 1_024;

/// Default delay time in milliseconds.
pub const DEFAULT_PITCH_DELAY_MS: Sample = 350.0;

/// Default feedback gain.
pub const DEFAULT_PITCH_DELAY_FEEDBACK: Sample = 0.4;

/// Default transposition per pass, in semitones (neutral: a plain delay).
pub const DEFAULT_PITCH_DELAY_SEMITONES: Sample = 0.0;

/// Default dry / wet mix.
pub const DEFAULT_PITCH_DELAY_MIX: Sample = 0.4;

/// Construction parameters for a [`PitchDelayNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PitchDelayParams {
    /// Delay time in milliseconds (clamped to
    /// `[MIN_PITCH_DELAY_MS, MAX_PITCH_DELAY_MS]`).
    pub delay_ms: Sample,
    /// Feedback gain in `[0, MAX_PITCH_DELAY_FEEDBACK]`.
    pub feedback: Sample,
    /// Transposition applied on every feedback pass, in semitones.
    pub pitch_semitones: Sample,
    /// Dry / wet mix in `[0, 1]`.
    pub mix: Sample,
}

impl Default for PitchDelayParams {
    fn default() -> Self {
        Self {
            delay_ms: DEFAULT_PITCH_DELAY_MS,
            feedback: DEFAULT_PITCH_DELAY_FEEDBACK,
            pitch_semitones: DEFAULT_PITCH_DELAY_SEMITONES,
            mix: DEFAULT_PITCH_DELAY_MIX,
        }
    }
}

impl PitchDelayParams {
    /// Returns a copy with every field clamped to its valid domain; any
    /// non-finite field falls back to its default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let fix = |v: Sample, lo: Sample, hi: Sample, def: Sample| {
            if v.is_finite() { v.clamp(lo, hi) } else { def }
        };
        Self {
            delay_ms: fix(self.delay_ms, MIN_PITCH_DELAY_MS, MAX_PITCH_DELAY_MS, d.delay_ms),
            feedback: fix(self.feedback, 0.0, MAX_PITCH_DELAY_FEEDBACK, d.feedback),
            pitch_semitones: fix(
                self.pitch_semitones,
                -MAX_PITCH_DELAY_SEMITONES,
                MAX_PITCH_DELAY_SEMITONES,
                d.pitch_semitones,
            ),
            mix: fix(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// Converts a delay time in milliseconds to an integer frame count in
/// `[1, max_delay]`.
#[inline]
fn delay_to_frames(delay_ms: Sample, sr: Sample, max_delay: usize) -> usize {
    let frames = ops::round(delay_ms * 0.001 * sr) as usize;
    frames.clamp(1, max_delay)
}

/// A pitch-shifting feedback delay (input port 0 -> output port 0).
///
/// Each channel owns an independent integer delay line; the delayed block is
/// transposed by a shared [`PitchShifterNode`] and re-injected on the next
/// block, so repeats spiral in pitch.
#[derive(Debug, Clone)]
pub struct PitchDelayNode {
    /// Sample rate in Hz, cached for cold-path delay recomputation.
    sample_rate: u32,
    /// Number of channels processed.
    channels: usize,
    /// Channel layout shared by the scratch buffers.
    layout: ChannelLayout,
    /// Maximum block size, in frames, the scratch buffers can hold.
    max_block_frames: usize,
    /// Per-channel delay ring buffers (length `max_delay + 1`).
    rings: Vec<Vec<Sample>>,
    /// Ring length in frames, shared by every channel.
    ring_len: usize,
    /// Largest integer delay the rings can hold.
    max_delay: usize,
    /// Shared write head into every ring.
    write_pos: usize,
    /// Current integer delay in frames.
    delay_frames: usize,
    /// Inner phase-vocoder shifter transposing the feedback path.
    shifter: PitchShifterNode,
    /// Scratch buffer holding one block of the delayed signal to transpose.
    delayed_buf: AudioBuffer,
    /// Scratch buffer holding the transposed feedback for the next block.
    feedback_buf: AudioBuffer,
    /// Number of valid feedback frames carried from the previous block.
    feedback_frames: usize,
    /// Smoothed feedback gain.
    feedback: Smoothed,
    /// Smoothed dry / wet mix.
    mix: Smoothed,
}

impl PitchDelayNode {
    /// Builds a pitch delay for `layout` at `sample_rate` Hz that can process up
    /// to `max_block_frames` frames per call. All rings, the inner shifter, and
    /// the scratch buffers are allocated here; parameters start settled.
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::effects::pitch_delay::{
    ///     PitchDelayNode, PitchDelayParams,
    /// };
    ///
    /// let params = PitchDelayParams::default();
    /// let mut delay = PitchDelayNode::new(48_000, ChannelLayout::Stereo, 256, params);
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
    /// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// input.set_active_frames(256);
    /// let inputs = [input];
    /// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// output.set_active_frames(256);
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// delay.process(&ctx, &mut io);
    /// assert_eq!(delay.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_block_frames: usize,
        params: PitchDelayParams,
    ) -> Self {
        let p = params.sanitised();
        let sr = sample_rate.max(1);
        let channels = layout.channel_count().max(1);
        let cap = max_block_frames.max(1);
        let srf = sr as Sample;

        let max_delay = delay_to_frames(MAX_PITCH_DELAY_MS, srf, usize::MAX / 4).max(1);
        let ring_len = max_delay + 1;
        let rings = vec![vec![0.0; ring_len]; channels];
        let delay_frames = delay_to_frames(p.delay_ms, srf, max_delay);

        let shifter = PitchShifterNode::new(
            sr,
            channels,
            PITCH_DELAY_FFT_SIZE,
            PitchShifterParams {
                pitch_ratio: semitones_to_ratio(p.pitch_semitones),
            },
        );

        Self {
            sample_rate: sr,
            channels,
            layout,
            max_block_frames: cap,
            rings,
            ring_len,
            max_delay,
            write_pos: 0,
            delay_frames,
            shifter,
            delayed_buf: AudioBuffer::new(layout, cap),
            feedback_buf: AudioBuffer::new(layout, cap),
            feedback_frames: 0,
            feedback: Smoothed::new(p.feedback),
            mix: Smoothed::new(p.mix),
        }
    }

    /// Returns the number of channels this node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the channel layout of the scratch buffers.
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

    /// Returns the current integer delay in frames.
    #[inline]
    #[must_use]
    pub fn delay_frames(&self) -> usize {
        self.delay_frames
    }

    /// Updates every parameter in place (allocation-free). The inner shifter's
    /// ratio, the delay length, feedback, and mix are all refreshed. This is a
    /// control-thread operation, not called from [`AudioNode::process`].
    pub fn set_params(&mut self, params: PitchDelayParams) {
        let p = params.sanitised();
        let srf = self.sample_rate as Sample;
        self.delay_frames = delay_to_frames(p.delay_ms, srf, self.max_delay);
        self.shifter.set_semitones(p.pitch_semitones);
        self.feedback.set_target(p.feedback, Ramp::Immediate);
        self.mix.set_target(p.mix, Ramp::Immediate);
    }
}

impl AudioNode for PitchDelayNode {
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
        let ch_n = self.channels;
        let ring_len = self.ring_len;
        let delay = self.delay_frames;
        self.delayed_buf.set_active_frames(frames);
        let fb_frames = self.feedback_frames.min(frames);

        // 1. Advance the delay rings. Read the delayed tap, capture it for later
        //    transposition, and write input + feedback (the transposed delayed
        //    signal carried from the previous block) back into the line.
        for n in 0..frames {
            let fb_gain = self.feedback.next_sample();
            let w = self.write_pos;
            let read_pos = if w >= delay { w - delay } else { w + ring_len - delay };
            for ch in 0..ch_n {
                let d = self.rings[ch][read_pos];
                self.delayed_buf.channel_mut(ch)[n] = d;
                let x = if ch < in_channels {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                } else {
                    0.0
                };
                let fb = if n < fb_frames {
                    self.feedback_buf.channel(ch)[n]
                } else {
                    0.0
                };
                self.rings[ch][w] = flush_denormal(x + fb_gain * fb);
            }
            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }

        // 2. Mix dry input with the delayed (wet) tap into the output.
        for n in 0..frames {
            let mix = self.mix.next_sample();
            for ch in 0..out_channels {
                let x = if ch < in_channels {
                    let v = input.channel(ch)[n];
                    if v.is_finite() { v } else { 0.0 }
                } else {
                    0.0
                };
                let wet = if ch < ch_n {
                    self.delayed_buf.channel(ch)[n]
                } else {
                    0.0
                };
                output.channel_mut(ch)[n] = (1.0 - mix) * x + mix * wet;
            }
        }

        // 3. Transpose the delayed block to form the next block's feedback.
        self.feedback_buf.set_active_frames(frames);
        {
            let inputs = core::slice::from_ref(&self.delayed_buf);
            let outputs = core::slice::from_mut(&mut self.feedback_buf);
            let mut sio = ProcessIo::new(inputs, outputs);
            self.shifter.process(ctx, &mut sio);
        }
        for ch in 0..ch_n {
            let dst = self.feedback_buf.channel_mut(ch);
            for d in dst[..frames].iter_mut() {
                *d = flush_denormal(*d);
            }
        }
        self.feedback_frames = frames;
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.iter_mut().for_each(|s| *s = 0.0);
        }
        self.write_pos = 0;
        self.shifter.reset();
        self.delayed_buf.clear();
        self.feedback_buf.clear();
        self.feedback_frames = 0;
        self.feedback = Smoothed::new(self.feedback.target());
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
            .map(|n| amp * ops::sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    fn burst(freq: Sample, amp: Sample, burst_len: usize, total: usize) -> Vec<Sample> {
        (0..total)
            .map(|n| {
                if n < burst_len {
                    amp * ops::sin(TAU * freq * n as Sample / SR as Sample)
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn impulse(len: usize) -> Vec<Sample> {
        let mut v = vec![0.0; len.max(1)];
        v[0] = 1.0;
        v
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        ops::sqrt((sum / samples.len() as f64) as Sample)
    }

    /// Single-frequency DFT magnitude (Goertzel), normalised by length.
    fn goertzel(signal: &[Sample], freq: Sample) -> Sample {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let w = TAU * freq / SR as Sample;
        let (sin_w, cos_w) = ops::sin_cos(w);
        let coeff = 2.0 * cos_w;
        let mut s_prev = 0.0f32;
        let mut s_prev2 = 0.0f32;
        for &x in signal {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        let real = s_prev - s_prev2 * cos_w;
        let imag = s_prev2 * sin_w;
        ops::sqrt(real * real + imag * imag) / n as Sample
    }

    /// Processes `signal` through the node in a single block (feedback does not
    /// recirculate within one call). The node must accept `signal.len()` frames.
    fn run_mono(node: &mut PitchDelayNode, signal: &[Sample]) -> Vec<Sample> {
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

    /// Processes `signal` through the node in fixed-size blocks so the
    /// block-granular feedback path recirculates realistically.
    fn run_blocks(node: &mut PitchDelayNode, signal: &[Sample], block: usize) -> Vec<Sample> {
        let block = block.max(1);
        let mut out = Vec::with_capacity(signal.len());
        let mut i = 0;
        while i < signal.len() {
            let n = block.min(signal.len() - i);
            let mut input = AudioBuffer::new(ChannelLayout::Mono, block);
            let mut output = AudioBuffer::new(ChannelLayout::Mono, block);
            input.set_active_frames(n);
            output.set_active_frames(n);
            input.channel_mut(0)[..n].copy_from_slice(&signal[i..i + n]);
            let inputs = [input];
            let mut outputs = [output];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
            out.extend_from_slice(&outputs[0].channel(0)[..n]);
            i += n;
        }
        out
    }

    #[test]
    fn latency_is_zero() {
        let node = PitchDelayNode::new(SR, ChannelLayout::Stereo, 256, PitchDelayParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn reports_requested_geometry() {
        let node = PitchDelayNode::new(SR, ChannelLayout::Stereo, 512, PitchDelayParams::default());
        assert_eq!(node.channels(), 2);
        assert_eq!(node.layout(), ChannelLayout::Stereo);
        assert_eq!(node.max_block_frames(), 512);
    }

    #[test]
    fn delay_frames_matches_params() {
        let params = PitchDelayParams {
            delay_ms: 100.0,
            ..PitchDelayParams::default()
        };
        let node = PitchDelayNode::new(SR, ChannelLayout::Mono, 256, params);
        assert_eq!(node.delay_frames(), 4_800);
    }

    #[test]
    fn default_params_within_domain() {
        let d = PitchDelayParams::default();
        assert_eq!(d, d.sanitised());
        assert!(d.delay_ms >= MIN_PITCH_DELAY_MS && d.delay_ms <= MAX_PITCH_DELAY_MS);
        assert!(d.feedback >= 0.0 && d.feedback <= MAX_PITCH_DELAY_FEEDBACK);
        assert!(d.pitch_semitones.abs() <= MAX_PITCH_DELAY_SEMITONES);
        assert!(d.mix >= 0.0 && d.mix <= 1.0);
    }

    #[test]
    fn sanitise_clamps_out_of_range() {
        let p = PitchDelayParams {
            delay_ms: 1.0e9,
            feedback: 5.0,
            pitch_semitones: 100.0,
            mix: 5.0,
        }
        .sanitised();
        assert!((p.delay_ms - MAX_PITCH_DELAY_MS).abs() < 1e-3);
        assert!((p.feedback - MAX_PITCH_DELAY_FEEDBACK).abs() < 1e-6);
        assert!((p.pitch_semitones - MAX_PITCH_DELAY_SEMITONES).abs() < 1e-6);
        assert!((p.mix - 1.0).abs() < 1e-6);

        let q = PitchDelayParams {
            delay_ms: -10.0,
            feedback: -1.0,
            pitch_semitones: -100.0,
            mix: -1.0,
        }
        .sanitised();
        assert!((q.delay_ms - MIN_PITCH_DELAY_MS).abs() < 1e-6);
        assert!((q.feedback - 0.0).abs() < 1e-6);
        assert!((q.pitch_semitones + MAX_PITCH_DELAY_SEMITONES).abs() < 1e-6);
        assert!((q.mix - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sanitise_replaces_non_finite() {
        let p = PitchDelayParams {
            delay_ms: Sample::NAN,
            feedback: Sample::INFINITY,
            pitch_semitones: Sample::NEG_INFINITY,
            mix: Sample::NAN,
        }
        .sanitised();
        assert_eq!(p, PitchDelayParams::default());
    }

    #[test]
    fn silence_produces_silence() {
        let mut node =
            PitchDelayNode::new(SR, ChannelLayout::Mono, 512, PitchDelayParams::default());
        let out = run_blocks(&mut node, &vec![0.0; 4_096], 512);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn mix_zero_is_dry_passthrough() {
        let params = PitchDelayParams {
            mix: 0.0,
            feedback: 0.8,
            pitch_semitones: 7.0,
            ..PitchDelayParams::default()
        };
        let mut node = PitchDelayNode::new(SR, ChannelLayout::Mono, 2_048, params);
        let input = sine(440.0, 0.5, 2_048);
        let out = run_mono(&mut node, &input);
        for (o, i) in out.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-6, "dry passthrough: {o} vs {i}");
        }
    }

    #[test]
    fn impulse_appears_at_delay_frames() {
        let params = PitchDelayParams {
            delay_ms: 10.0,
            feedback: 0.0,
            pitch_semitones: 0.0,
            mix: 1.0,
        };
        let mut node = PitchDelayNode::new(SR, ChannelLayout::Mono, 2_048, params);
        let delay = node.delay_frames();
        let out = run_mono(&mut node, &impulse(2_048));
        // The wet tap is read straight from the line, so the impulse re-emerges
        // exactly `delay_frames` later with no shift (feedback is zero).
        assert!((out[delay] - 1.0).abs() < 1e-6, "peak at delay: {}", out[delay]);
        let pre: Sample = out[..delay].iter().map(|&x| x.abs()).sum();
        assert!(pre < 1e-5, "no energy before the delay: {pre}");
    }

    #[test]
    fn feedback_extends_tail() {
        let total = SR as usize * 2;
        let block = 480;
        let low_params = PitchDelayParams {
            delay_ms: 80.0,
            feedback: 0.2,
            pitch_semitones: 0.0,
            mix: 1.0,
        };
        let high_params = PitchDelayParams {
            feedback: 0.85,
            ..low_params
        };
        let signal = burst(500.0, 0.5, SR as usize / 10, total);
        let mut low = PitchDelayNode::new(SR, ChannelLayout::Mono, block, low_params);
        let mut high = PitchDelayNode::new(SR, ChannelLayout::Mono, block, high_params);
        let out_low = run_blocks(&mut low, &signal, block);
        let out_high = run_blocks(&mut high, &signal, block);
        // Measure a late window well after the burst has stopped.
        let start = SR as usize + SR as usize / 2;
        let tail_low = rms(&out_low[start..]);
        let tail_high = rms(&out_high[start..]);
        assert!(
            tail_high > tail_low * 2.0,
            "stronger feedback should sustain a louder tail: low {tail_low} high {tail_high}"
        );
    }

    #[test]
    fn positive_shift_migrates_energy_upward() {
        let f0 = 1_200.0;
        let total = SR as usize * 3 / 2;
        let block = 480;
        let signal = burst(f0, 0.6, SR as usize / 4, total);
        let base = PitchDelayParams {
            delay_ms: 120.0,
            feedback: 0.75,
            pitch_semitones: 0.0,
            mix: 1.0,
        };
        let shifted_params = PitchDelayParams {
            pitch_semitones: 7.0,
            ..base
        };
        let mut flat = PitchDelayNode::new(SR, ChannelLayout::Mono, block, base);
        let mut up = PitchDelayNode::new(SR, ChannelLayout::Mono, block, shifted_params);
        let out_flat = run_blocks(&mut flat, &signal, block);
        let out_up = run_blocks(&mut up, &signal, block);
        // Analyse the tail after the first (unshifted) echo has passed.
        let start = SR as usize * 55 / 100;
        let flat_tail = &out_flat[start..];
        let up_tail = &out_up[start..];
        // With no shift the tail keeps repeating at f0.
        assert!(
            goertzel(flat_tail, f0) > goertzel(flat_tail, f0 * 1.5),
            "unshifted tail should stay at f0"
        );
        // A positive shift spirals the recirculating energy above f0.
        assert!(
            goertzel(up_tail, f0 * 1.5) > goertzel(up_tail, f0),
            "shifted tail should move energy above f0"
        );
        // And it should hold less energy at f0 than the unshifted control.
        assert!(
            goertzel(flat_tail, f0) > goertzel(up_tail, f0),
            "shift should drain the original pitch from the tail"
        );
    }

    #[test]
    fn high_feedback_stays_bounded() {
        let params = PitchDelayParams {
            delay_ms: 60.0,
            feedback: 0.95,
            pitch_semitones: 0.0,
            mix: 1.0,
        };
        let mut node = PitchDelayNode::new(SR, ChannelLayout::Mono, 512, params);
        let signal = burst(440.0, 0.8, SR as usize / 20, SR as usize * 3);
        let out = run_blocks(&mut node, &signal, 512);
        assert!(out.iter().all(|&x| x.is_finite()), "tail stays finite");
        let peak = out.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(peak < 8.0, "tail stays bounded: peak {peak}");
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = PitchDelayNode::new(
            SR,
            ChannelLayout::Mono,
            256,
            PitchDelayParams {
                feedback: 0.7,
                pitch_semitones: 5.0,
                mix: 0.5,
                ..PitchDelayParams::default()
            },
        );
        let mut signal = vec![0.0; 2_048];
        signal[0] = Sample::NAN;
        signal[1] = Sample::INFINITY;
        signal[2] = Sample::NEG_INFINITY;
        signal[100] = 0.4;
        let out = run_blocks(&mut node, &signal, 256);
        assert!(out.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn tone_output_is_finite() {
        let mut node =
            PitchDelayNode::new(SR, ChannelLayout::Mono, 512, PitchDelayParams::default());
        let out = run_blocks(&mut node, &sine(330.0, 0.5, SR as usize), 512);
        assert!(out.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node =
            PitchDelayNode::new(SR, ChannelLayout::Mono, 256, PitchDelayParams::default());
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
        let params = PitchDelayParams {
            delay_ms: 50.0,
            feedback: 0.6,
            pitch_semitones: 0.0,
            mix: 1.0,
        };
        let mut node = PitchDelayNode::new(SR, ChannelLayout::Stereo, 512, params);
        let total = SR as usize;
        let block = 512;
        let mut out_l = Vec::with_capacity(total);
        let mut out_r = Vec::with_capacity(total);
        let mut i = 0;
        while i < total {
            let n = block.min(total - i);
            let mut input = AudioBuffer::new(ChannelLayout::Stereo, block);
            let mut output = AudioBuffer::new(ChannelLayout::Stereo, block);
            input.set_active_frames(n);
            output.set_active_frames(n);
            // Drive the left channel only.
            if i == 0 {
                input.channel_mut(0)[0] = 1.0;
            }
            let inputs = [input];
            let mut outputs = [output];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(n), &mut io);
            out_l.extend_from_slice(&outputs[0].channel(0)[..n]);
            out_r.extend_from_slice(&outputs[0].channel(1)[..n]);
            i += n;
        }
        assert!(rms(&out_l) > 1e-4, "left channel echoes");
        assert!(rms(&out_r) < 1e-6, "right channel stays silent");
    }

    #[test]
    fn mono_input_into_stereo_is_finite() {
        let mut node =
            PitchDelayNode::new(SR, ChannelLayout::Stereo, 256, PitchDelayParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
        input.set_active_frames(256);
        output.set_active_frames(256);
        for (n, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(TAU * 220.0 * n as Sample / SR as Sample);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        assert!(outputs[0].channel(0).iter().all(|&x| x.is_finite()));
        assert!(outputs[0].channel(1).iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = PitchDelayParams {
            delay_ms: 40.0,
            feedback: 0.7,
            pitch_semitones: 4.0,
            mix: 0.5,
        };
        let mut node = PitchDelayNode::new(SR, ChannelLayout::Mono, 256, params);
        // Prime the node with a signal, then reset it.
        let _ = run_blocks(&mut node, &sine(440.0, 0.5, 4_096), 256);
        node.reset();

        let probe = sine(330.0, 0.5, 4_096);
        let after = run_blocks(&mut node, &probe, 256);

        let mut fresh = PitchDelayNode::new(SR, ChannelLayout::Mono, 256, params);
        let baseline = run_blocks(&mut fresh, &probe, 256);

        let max_err = after
            .iter()
            .zip(baseline.iter())
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(max_err < 1e-6, "reset should match a fresh node: {max_err}");
    }

    #[test]
    fn set_params_mix_zero_passes_dry() {
        let mut node = PitchDelayNode::new(
            SR,
            ChannelLayout::Mono,
            2_048,
            PitchDelayParams {
                feedback: 0.6,
                mix: 1.0,
                ..PitchDelayParams::default()
            },
        );
        node.set_params(PitchDelayParams {
            mix: 0.0,
            feedback: 0.6,
            ..PitchDelayParams::default()
        });
        let input = sine(440.0, 0.5, 2_048);
        let out = run_mono(&mut node, &input);
        for (o, i) in out.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-6, "mix=0 after set_params: {o} vs {i}");
        }
    }

    #[test]
    fn set_params_updates_delay_frames() {
        let mut node =
            PitchDelayNode::new(SR, ChannelLayout::Mono, 256, PitchDelayParams::default());
        node.set_params(PitchDelayParams {
            delay_ms: 200.0,
            ..PitchDelayParams::default()
        });
        assert_eq!(node.delay_frames(), 9_600);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut node = PitchDelayNode::new(
            SR,
            ChannelLayout::Stereo,
            256,
            PitchDelayParams {
                delay_ms: 1.0e9,
                feedback: 10.0,
                pitch_semitones: 1_000.0,
                mix: 10.0,
            },
        );
        let out = run_blocks(&mut node, &impulse(2_048), 256);
        assert!(out.iter().all(|&x| x.is_finite()));
    }
}
