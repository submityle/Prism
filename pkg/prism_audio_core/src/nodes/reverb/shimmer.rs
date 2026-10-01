//! M1 reverb-family node: a shimmer reverb that recirculates a pitch-shifted
//! copy of its own tail to synthesise the ethereal, upward-"climbing" wash
//! popularised by ambient and cinematic sound design.
//!
//! A plain reverb adds a dense, decaying cloud of reflections to a dry source.
//! A *shimmer* reverb inserts a pitch shifter into the reverb's own feedback
//! path: each time the tail recirculates it is transposed (classically up one
//! octave) and re-injected, so energy perpetually migrates toward higher
//! frequencies and the tail appears to rise and bloom rather than simply fade.
//! The upward migration is self-limiting -- transposed energy eventually
//! reaches the damped high-frequency region of the tank and is absorbed -- so a
//! sub-unity feedback gain combined with the reverb's own decay keeps the loop
//! bounded.
//!
//! # Signal flow
//!
//! ```text
//!   dry --------------------------------------------+--> dry * dry_gain --+
//!    |                                              |                     |
//!    v                                              |                     v
//!   (+)<-- shimmer_feedback * shifted_tail(prev) ---+                   (sum) --> out
//!    |                                                                    ^
//!    v                                                                    |
//!   FDN reverb tank (wet=1, dry=0) --> reverb tail --> wet * wet_gain ----+
//!    |
//!    v
//!   pitch shifter (+semitones) --> shifted tail (feeds the next block)
//! ```
//!
//! The pitch shifter sits strictly inside the feedback loop, so its latency
//! lengthens the recirculation delay (a musically useful pre-bloom) but adds
//! no latency to the direct dry or first-pass wet output; the node therefore
//! reports zero processing latency.
//!
//! # Provenance
//!
//! Original implementation for Prism. The architecture -- a feedback reverb
//! with an in-loop octave pitch shifter -- is a widely documented classic
//! studio technique (e.g. Eventide/Lexicon-style "shimmer" patches and the
//! many software emulations that followed). This module was written from that
//! public description using only Prism's own primitives and contains no
//! source or derived code from UE, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Web Audio, or any other audio engine or plug-in.
//! It is pure classic DSP with no AI/ML content.
//!
//! # Relationship
//!
//! This node is a *composition*: it owns an internal
//! [`FdnReverb`](crate::nodes::reverb::fdn::FdnReverb) tank (configured
//! fully-wet so its output is the bare reverberant signal) and an internal
//! [`PitchShifterNode`](crate::nodes::effects::pitch_shifter::PitchShifterNode)
//! driving the feedback transposition. It reuses -- and never duplicates --
//! their DSP. Compared with [`FdnReverb`] alone it differs only by the
//! pitch-shifted self-feedback path; compared with a bare
//! [`PitchShifterNode`] it differs by embedding the shifter in a decaying
//! recirculating tank rather than applying it once to the dry input.

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::effects::pitch_shifter::{
    PitchShifterNode, PitchShifterParams, semitones_to_ratio,
};
use crate::nodes::reverb::fdn::{FdnReverb, FdnReverbParams};
use crate::param::{Ramp, Smoothed};

/// Largest permitted shimmer feedback gain.
///
/// Kept below unity so the recirculating loop -- already attenuated by the
/// tank's own decay and high-frequency damping -- cannot sustain or grow
/// indefinitely.
pub const MAX_SHIMMER_FEEDBACK: Sample = 0.85;

/// Default transposition applied in the feedback loop, in semitones
/// (`+12` == one octave up, the canonical shimmer interval).
pub const DEFAULT_SHIMMER_SEMITONES: Sample = 12.0;

/// Default analysis/synthesis window the internal pitch shifter is built with.
///
/// A larger window yields smoother transposition of the diffuse tail at the
/// cost of a longer feedback pre-bloom; `2048` is a good shimmer compromise.
pub const DEFAULT_SHIMMER_FFT_SIZE: usize = 2048;

/// Construction parameters for a [`ShimmerReverb`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ShimmerReverbParams {
    /// Scales the tank's delay-line lengths; larger models a bigger space.
    pub room_size: Sample,
    /// Target reverberation time (`RT60`) of the tank, in seconds.
    pub decay_rt60_seconds: Sample,
    /// High-frequency damping of the tank in `[0.0, 0.999]`.
    pub damping: Sample,
    /// Transposition applied to the recirculated tail, in semitones.
    pub pitch_semitones: Sample,
    /// Gain applied to the pitch-shifted tail before re-injection, clamped to
    /// `[0.0, MAX_SHIMMER_FEEDBACK]`.
    pub shimmer_feedback: Sample,
    /// Wet (reverberated) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed input) mix gain.
    pub dry: Sample,
}

impl Default for ShimmerReverbParams {
    fn default() -> Self {
        Self {
            room_size: 1.0,
            decay_rt60_seconds: 2.8,
            damping: 0.4,
            pitch_semitones: DEFAULT_SHIMMER_SEMITONES,
            shimmer_feedback: 0.5,
            wet: 0.3,
            dry: 1.0,
        }
    }
}

impl ShimmerReverbParams {
    /// Returns a copy with every field forced finite and into its valid range,
    /// so hostile or denormal automation can never destabilise the loop.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let fix = |v: Sample, fallback: Sample| if v.is_finite() { v } else { fallback };
        Self {
            room_size: fix(self.room_size, 1.0).clamp(0.1, 4.0),
            decay_rt60_seconds: fix(self.decay_rt60_seconds, 2.8).max(0.01),
            damping: fix(self.damping, 0.4).clamp(0.0, 0.999),
            pitch_semitones: fix(self.pitch_semitones, DEFAULT_SHIMMER_SEMITONES).clamp(-24.0, 24.0),
            shimmer_feedback: fix(self.shimmer_feedback, 0.5).clamp(0.0, MAX_SHIMMER_FEEDBACK),
            wet: fix(self.wet, 0.3).clamp(0.0, 4.0),
            dry: fix(self.dry, 1.0).clamp(0.0, 4.0),
        }
    }
}

/// A shimmer reverb (input port 0 -> output port 0).
///
/// See the [module documentation](self) for the architecture, provenance, and
/// relationship to the primitives it composes.
#[derive(Debug)]
pub struct ShimmerReverb {
    /// Internal fully-wet feedback-delay-network tank.
    reverb: FdnReverb,
    /// Internal pitch shifter driving the feedback transposition.
    shifter: PitchShifterNode,
    /// Channel count every internal stage and scratch buffer uses.
    channels: usize,
    /// Channel layout shared by the scratch buffers.
    layout: ChannelLayout,
    /// Maximum block size, in frames, the scratch buffers can hold.
    max_block_frames: usize,
    /// Scratch: dry input plus the re-injected shimmer feedback (tank input).
    reverb_in: AudioBuffer,
    /// Scratch: bare reverberant tail produced by the tank.
    reverb_out: AudioBuffer,
    /// Scratch: pitch-shifted tail produced this block.
    shifted: AudioBuffer,
    /// Carried feedback: the previous block's pitch-shifted tail.
    feedback_buf: AudioBuffer,
    /// Number of valid frames currently held in `feedback_buf`.
    feedback_frames: usize,
    /// Smoothed shimmer feedback gain (click-free automation).
    feedback_gain: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

impl ShimmerReverb {
    /// Builds a shimmer reverb for a `layout`-wide signal at `sample_rate` Hz
    /// that can process up to `max_block_frames` frames per call.
    ///
    /// The internal tank is forced fully wet (so its output is the bare
    /// reverberant signal); the dry/wet blend is applied by this node. All
    /// scratch and delay state is allocated here so
    /// [`process`](AudioNode::process) stays allocation-free.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::reverb::{ShimmerReverb, ShimmerReverbParams};
    ///
    /// let params = ShimmerReverbParams::default();
    /// let mut verb = ShimmerReverb::new(48_000, ChannelLayout::Stereo, 512, params);
    /// assert_eq!(verb.latency_frames(), 0);
    ///
    /// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// input.set_active_frames(256);
    /// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// output.set_active_frames(256);
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
    /// let inputs = [input];
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// verb.process(&ctx, &mut io);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_block_frames: usize,
        params: ShimmerReverbParams,
    ) -> Self {
        let p = params.sanitised();
        let channels = layout.channel_count().max(1);
        let cap = max_block_frames.max(1);

        let reverb = FdnReverb::new(
            sample_rate,
            channels,
            FdnReverbParams {
                room_size: p.room_size,
                decay_rt60_seconds: p.decay_rt60_seconds,
                damping: p.damping,
                // The tank is a pure wet send; this node owns the dry/wet mix.
                wet: 1.0,
                dry: 0.0,
                ..FdnReverbParams::default()
            },
        );

        let shifter = PitchShifterNode::new(
            sample_rate,
            channels,
            DEFAULT_SHIMMER_FFT_SIZE,
            PitchShifterParams {
                pitch_ratio: semitones_to_ratio(p.pitch_semitones),
            },
        );

        Self {
            reverb,
            shifter,
            channels,
            layout,
            max_block_frames: cap,
            reverb_in: AudioBuffer::new(layout, cap),
            reverb_out: AudioBuffer::new(layout, cap),
            shifted: AudioBuffer::new(layout, cap),
            feedback_buf: AudioBuffer::new(layout, cap),
            feedback_frames: 0,
            feedback_gain: Smoothed::new(p.shimmer_feedback),
            wet: Smoothed::new(p.wet),
            dry: Smoothed::new(p.dry),
        }
    }

    /// Returns the channel layout the reverb was built for.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the channel count every internal stage uses.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the maximum block size, in frames, the reverb can process.
    #[inline]
    #[must_use]
    pub fn max_block_frames(&self) -> usize {
        self.max_block_frames
    }

    /// Returns the internal pitch shifter's analysis window size (the power of
    /// two actually in use).
    #[inline]
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.shifter.fft_size()
    }

    /// Returns the current target shimmer feedback gain.
    #[inline]
    #[must_use]
    pub fn shimmer_feedback(&self) -> Sample {
        self.feedback_gain.target()
    }

    /// Returns the current feedback transposition ratio.
    #[inline]
    #[must_use]
    pub fn pitch_ratio(&self) -> Sample {
        self.shifter.pitch_ratio()
    }

    /// Sets the shimmer feedback gain (clamped to
    /// `[0.0, MAX_SHIMMER_FEEDBACK]`), gliding with `ramp`.
    #[inline]
    pub fn set_shimmer_feedback(&mut self, amount: Sample, ramp: Ramp) {
        let a = if amount.is_finite() { amount } else { 0.0 };
        self.feedback_gain
            .set_target(a.clamp(0.0, MAX_SHIMMER_FEEDBACK), ramp);
    }

    /// Sets the feedback transposition in semitones (takes effect on the next
    /// shifter frame, so a sounding tail is not interrupted).
    #[inline]
    pub fn set_pitch_semitones(&mut self, semitones: Sample) {
        self.shifter.set_semitones(semitones);
    }

    /// Sets the tank's target `RT60` decay time in seconds.
    #[inline]
    pub fn set_decay(&mut self, rt60_seconds: Sample) {
        self.reverb.set_decay(rt60_seconds);
    }

    /// Sets the tank's high-frequency damping, gliding with `ramp`.
    #[inline]
    pub fn set_damping(&mut self, damping: Sample, ramp: Ramp) {
        self.reverb.set_damping(damping, ramp);
    }

    /// Sets the wet mix gain, gliding with `ramp`.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry mix gain, gliding with `ramp`.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for ShimmerReverb {
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

        self.reverb_in.set_active_frames(frames);
        self.reverb_out.set_active_frames(frames);
        self.shifted.set_active_frames(frames);

        // 1. Build the tank input: dry signal plus the re-injected, gain-scaled
        //    shimmer feedback carried from the previous block.
        let fb_frames = self.feedback_frames.min(frames);
        for n in 0..frames {
            let g = self.feedback_gain.next_sample();
            for ch in 0..rev_ch {
                let x = if ch < in_channels {
                    input.channel(ch)[n]
                } else {
                    0.0
                };
                let fb = if n < fb_frames {
                    self.feedback_buf.channel(ch)[n]
                } else {
                    0.0
                };
                self.reverb_in.channel_mut(ch)[n] = x + g * fb;
            }
        }

        // 2. Run the fully-wet tank: reverb_in -> reverb_out.
        {
            let inputs = core::slice::from_ref(&self.reverb_in);
            let outputs = core::slice::from_mut(&mut self.reverb_out);
            let mut rio = ProcessIo::new(inputs, outputs);
            self.reverb.process(ctx, &mut rio);
        }

        // 3. Blend dry input with the wet tail into the output.
        for n in 0..frames {
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();
            for ch in 0..out_channels {
                let x = if ch < in_channels {
                    input.channel(ch)[n]
                } else {
                    0.0
                };
                let r = if ch < rev_ch {
                    self.reverb_out.channel(ch)[n]
                } else {
                    0.0
                };
                output.channel_mut(ch)[n] = dry * x + wet * r;
            }
        }

        // 4. Pitch-shift the tail to form the next block's feedback source.
        {
            let inputs = core::slice::from_ref(&self.reverb_out);
            let outputs = core::slice::from_mut(&mut self.shifted);
            let mut sio = ProcessIo::new(inputs, outputs);
            self.shifter.process(ctx, &mut sio);
        }

        // 5. Latch the shifted tail (denormal-flushed) for the next block.
        self.feedback_buf.set_active_frames(frames);
        for ch in 0..rev_ch {
            let (src, dst) = (self.shifted.channel(ch), self.feedback_buf.channel_mut(ch));
            for (d, &s) in dst[..frames].iter_mut().zip(src[..frames].iter()) {
                *d = flush_denormal(s);
            }
        }
        self.feedback_frames = frames;
    }

    fn reset(&mut self) {
        self.reverb.reset();
        self.shifter.reset();
        self.reverb_in.clear();
        self.reverb_out.clear();
        self.shifted.clear();
        self.feedback_buf.clear();
        self.feedback_frames = 0;
        self.feedback_gain = Smoothed::new(self.feedback_gain.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use bevy_math::ops;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;
    const BLOCK: usize = 512;

    fn ctx(frames: usize, playhead: u64) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead,
        }
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

    /// Runs `verb` over `input_mono` (duplicated to both stereo channels) in
    /// `BLOCK`-sized blocks and returns channel-0 output concatenated.
    fn run(verb: &mut ShimmerReverb, input_mono: &[Sample]) -> Vec<Sample> {
        let total = input_mono.len();
        let mut out = Vec::with_capacity(total);
        let mut i = 0;
        while i < total {
            let f = (total - i).min(BLOCK);
            let mut inb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
            inb.set_active_frames(f);
            for ch in 0..2 {
                let dst = inb.channel_mut(ch);
                dst[..f].copy_from_slice(&input_mono[i..i + f]);
            }
            let mut outb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
            outb.set_active_frames(f);
            let inputs = [inb];
            let mut outputs = [outb];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            verb.process(&ctx(f, i as u64), &mut io);
            out.extend_from_slice(&outputs[0].channel(0)[..f]);
            i += f;
        }
        out
    }

    fn tone_then_silence(freq: Sample, amp: Sample, tone_len: usize, total: usize) -> Vec<Sample> {
        (0..total)
            .map(|n| {
                if n < tone_len {
                    amp * ops::sin(TAU * freq * n as Sample / SR as Sample)
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn reports_zero_latency() {
        let verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        assert_eq!(verb.latency_frames(), 0);
    }

    #[test]
    fn default_pitch_is_one_octave_up() {
        let verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        assert!((verb.pitch_ratio() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        let out = run(&mut verb, &vec![0.0; 4 * BLOCK]);
        assert!(out.iter().all(|&s| s.abs() < 1e-7), "silence must stay silent");
    }

    #[test]
    fn dry_passthrough_when_wet_zero_and_no_feedback() {
        let params = ShimmerReverbParams {
            wet: 0.0,
            dry: 1.0,
            shimmer_feedback: 0.0,
            ..ShimmerReverbParams::default()
        };
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, params);
        let input = tone_then_silence(440.0, 0.5, BLOCK, 2 * BLOCK);
        let out = run(&mut verb, &input);
        for (o, i) in out.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-6, "wet=0 dry=1 must pass the dry signal");
        }
    }

    #[test]
    fn tail_rings_after_input_stops() {
        let params = ShimmerReverbParams {
            wet: 1.0,
            dry: 0.0,
            ..ShimmerReverbParams::default()
        };
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, params);
        let total = 24_576;
        let input = tone_then_silence(500.0, 0.6, 4_096, total);
        let out = run(&mut verb, &input);
        // Energy must remain in the silent tail region (reverb decay).
        let tail = &out[16_384..];
        let energy: f64 = tail.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        assert!(energy > 1e-3, "reverb tail must carry energy, got {energy}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn shimmer_feedback_adds_octave_energy() {
        let total = 32_768;
        let tone = tone_then_silence(1_000.0, 0.6, 8_192, total);
        let base = ShimmerReverbParams {
            wet: 1.0,
            dry: 0.0,
            damping: 0.2,
            decay_rt60_seconds: 2.5,
            pitch_semitones: 12.0,
            ..ShimmerReverbParams::default()
        };

        let mut with_shimmer = ShimmerReverb::new(
            SR,
            ChannelLayout::Stereo,
            BLOCK,
            ShimmerReverbParams {
                shimmer_feedback: 0.8,
                ..base
            },
        );
        let out_on = run(&mut with_shimmer, &tone);

        let mut no_shimmer = ShimmerReverb::new(
            SR,
            ChannelLayout::Stereo,
            BLOCK,
            ShimmerReverbParams {
                shimmer_feedback: 0.0,
                ..base
            },
        );
        let out_off = run(&mut no_shimmer, &tone);

        // Measure the octave-up partial (2 kHz) in the tail region.
        let octave_on = goertzel(&out_on[16_384..], 2_000.0);
        let octave_off = goertzel(&out_off[16_384..], 2_000.0);
        assert!(
            octave_on > octave_off * 4.0 + 1e-6,
            "shimmer must raise octave energy: on={octave_on}, off={octave_off}"
        );
        assert!(out_on.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn stays_bounded_under_max_feedback() {
        let params = ShimmerReverbParams {
            wet: 1.0,
            dry: 1.0,
            shimmer_feedback: MAX_SHIMMER_FEEDBACK,
            decay_rt60_seconds: 4.0,
            damping: 0.2,
            ..ShimmerReverbParams::default()
        };
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, params);
        // One second of full-scale excitation.
        let input: Vec<Sample> = (0..SR as usize)
            .map(|n| ops::sin(TAU * 300.0 * n as Sample / SR as Sample))
            .collect();
        let out = run(&mut verb, &input);
        assert!(out.iter().all(|s| s.is_finite()), "loop must not produce NaN/inf");
        let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak < 50.0, "loop must stay bounded, peak={peak}");
    }

    #[test]
    fn reset_clears_the_tail() {
        let params = ShimmerReverbParams {
            wet: 1.0,
            dry: 0.0,
            shimmer_feedback: 0.7,
            ..ShimmerReverbParams::default()
        };
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, params);
        let _ = run(&mut verb, &tone_then_silence(400.0, 0.7, 2_048, 8 * BLOCK));
        verb.reset();
        let out = run(&mut verb, &vec![0.0; 4 * BLOCK]);
        assert!(out.iter().all(|&s| s.abs() < 1e-7), "reset must silence the tail");
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        let mut inb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
        inb.set_active_frames(0);
        let mut outb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
        outb.set_active_frames(0);
        let inputs = [inb];
        let mut outputs = [outb];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        verb.process(&ctx(0, 0), &mut io);
    }

    #[test]
    fn non_finite_params_are_sanitised() {
        let params = ShimmerReverbParams {
            room_size: Sample::NAN,
            decay_rt60_seconds: Sample::INFINITY,
            damping: Sample::NAN,
            pitch_semitones: Sample::NAN,
            shimmer_feedback: Sample::INFINITY,
            wet: Sample::NAN,
            dry: Sample::NAN,
        };
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, params);
        let out = run(&mut verb, &tone_then_silence(440.0, 0.5, BLOCK, 4 * BLOCK));
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn both_channels_are_finite() {
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        let input = tone_then_silence(440.0, 0.5, 2_048, 8 * BLOCK);
        let total = input.len();
        let mut i = 0;
        while i < total {
            let f = (total - i).min(BLOCK);
            let mut inb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
            inb.set_active_frames(f);
            inb.channel_mut(0)[..f].copy_from_slice(&input[i..i + f]);
            // Opposite phase in the right channel.
            for n in 0..f {
                inb.channel_mut(1)[n] = -input[i + n];
            }
            let mut outb = AudioBuffer::new(ChannelLayout::Stereo, BLOCK);
            outb.set_active_frames(f);
            let inputs = [inb];
            let mut outputs = [outb];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            verb.process(&ctx(f, i as u64), &mut io);
            assert!(outputs[0].channel(0)[..f].iter().all(|s| s.is_finite()));
            assert!(outputs[0].channel(1)[..f].iter().all(|s| s.is_finite()));
            i += f;
        }
    }

    #[test]
    fn sanitised_clamps_into_range() {
        let p = ShimmerReverbParams {
            room_size: 100.0,
            decay_rt60_seconds: -5.0,
            damping: 9.0,
            pitch_semitones: 999.0,
            shimmer_feedback: 9.0,
            wet: -1.0,
            dry: 99.0,
        }
        .sanitised();
        assert!((0.1..=4.0).contains(&p.room_size));
        assert!(p.decay_rt60_seconds >= 0.01);
        assert!((0.0..=0.999).contains(&p.damping));
        assert!((-24.0..=24.0).contains(&p.pitch_semitones));
        assert!((0.0..=MAX_SHIMMER_FEEDBACK).contains(&p.shimmer_feedback));
        assert!((0.0..=4.0).contains(&p.wet));
        assert!((0.0..=4.0).contains(&p.dry));
    }

    #[test]
    fn set_shimmer_feedback_clamps() {
        let mut verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        verb.set_shimmer_feedback(5.0, Ramp::Immediate);
        assert!((verb.shimmer_feedback() - MAX_SHIMMER_FEEDBACK).abs() < 1e-6);
        verb.set_shimmer_feedback(Sample::NAN, Ramp::Immediate);
        assert!(verb.shimmer_feedback() <= MAX_SHIMMER_FEEDBACK);
        assert!(verb.shimmer_feedback() >= 0.0);
    }

    #[test]
    fn getters_report_configuration() {
        let verb = ShimmerReverb::new(SR, ChannelLayout::Stereo, BLOCK, ShimmerReverbParams::default());
        assert_eq!(verb.layout(), ChannelLayout::Stereo);
        assert_eq!(verb.channels(), 2);
        assert_eq!(verb.max_block_frames(), BLOCK);
        // fft_size rounds the request up to a power of two.
        assert_eq!(verb.fft_size(), DEFAULT_SHIMMER_FFT_SIZE);
    }
}
