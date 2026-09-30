//! Tuned feedback comb resonator (Karplus-Strong / lowpass-feedback comb).
//!
//! A comb resonator recirculates its input through a delay line whose length is
//! tuned to a musical pitch, so a broadband excitation (a pluck, a noise burst,
//! or any input transient) rings at a fundamental of `frequency_hz` with a
//! decaying harmonic series. A one-pole lowpass in the feedback loop damps the
//! recirculating signal so higher partials die away faster than the
//! fundamental, which is exactly what makes plucked strings, struck bars, and
//! resonant bodies sound natural rather than metallic and endless.
//!
//! The recurrence is
//!
//! ```text
//! filtered[n] = (1 - damping) * y[n - D] + damping * filtered[n - 1]
//! y[n]        = x[n] + feedback * filtered[n]
//! ```
//!
//! where `D = sample_rate / frequency_hz` is the (fractional) loop delay read
//! with linear interpolation so the tuning is continuous rather than quantized
//! to integer taps. `feedback` sets the decay time and `damping` sets how much
//! brighter partials are attenuated on each pass through the loop.
//!
//! This differs from the sibling effects: a
//! [`DelayNode`](crate::nodes::effects::DelayNode) produces discrete echoes at
//! audible spacings, a [`FlangerNode`](crate::nodes::effects::FlangerNode)
//! sweeps a short feedback delay to drag comb notches through the spectrum, and
//! a [`ChorusNode`](crate::nodes::effects::ChorusNode) sums several *feedback
//! free* modulated taps. A comb resonator instead uses a *pitched, damped*
//! feedback loop as a sustained resonant voice.
//!
//! # Real-time contract
//!
//! One ring buffer and one filter state per channel are allocated in
//! [`CombResonatorNode::new`]. [`process`](crate::graph::AudioNode::process)
//! performs no allocation, takes no locks, and cannot panic: mismatched channel
//! counts and zero-length blocks degrade gracefully, feedback is clamped just
//! below unity so the loop always decays, and every recirculated sample is
//! denormal-flushed to avoid subnormal CPU stalls.
//!
//! # Provenance
//!
//! The tuned feedback comb is a classic physical-modeling and reverberation
//! primitive: the plucked-string algorithm of Karplus and Strong ("Digital
//! Synthesis of Plucked-String and Drum Timbres", 1983), its extensions by
//! Jaffe and Smith ("Extensions of the Karplus-Strong Plucked-String
//! Algorithm", 1983), and the lowpass-in-the-loop comb of Schroeder and Moorer
//! reverberators (Moorer, "About This Reverberation Business", 1979). This
//! module reuses only this crate's own [`Sample`] type, linear-interpolation
//! [`lerp`], and denormal-flushing primitive. It contains **no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**; it is implemented purely from that publicly documented
//! theory.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};

/// Lowest tunable fundamental in hertz. This bounds the pre-allocated ring
/// length (`sample_rate / MIN_FREQUENCY_HZ` frames of maximum delay).
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Largest stable feedback coefficient. Kept just below unity so the resonant
/// loop always decays instead of building without bound.
pub const MAX_FEEDBACK: Sample = 0.999;

/// Returns `value` when finite, otherwise `fallback`. Guards the public setters
/// against `NaN`/infinity leaking into the loop state (a `NaN` would otherwise
/// survive `clamp`).
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Configuration for a [`CombResonatorNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CombResonatorParams {
    /// Resonant fundamental in hertz, clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`. The loop delay is
    /// `sample_rate / frequency_hz` frames.
    pub frequency_hz: Sample,
    /// Feedback coefficient in `[0, MAX_FEEDBACK]`. Higher values ring longer;
    /// at `MAX_FEEDBACK` the tone decays slowly but never sustains forever.
    pub feedback: Sample,
    /// Loop damping in `[0, 1]`: the pole of the one-pole lowpass in the
    /// feedback path. `0` is a bright, undamped comb; larger values roll off
    /// high partials so the timbre darkens and higher harmonics decay first.
    pub damping: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` is the fully
    /// resonated signal.
    pub mix: Sample,
}

impl Default for CombResonatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: 220.0,
            feedback: 0.9,
            damping: 0.2,
            mix: 1.0,
        }
    }
}

/// A tuned feedback comb resonator (input port 0 -> output port 0).
///
/// Each channel owns an independent delay line and lowpass state, but all
/// channels share the same tuning, feedback, damping, and mix so a stereo or
/// surround signal resonates coherently.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{CombResonatorNode, CombResonatorParams};
///
/// let mut node = CombResonatorNode::new(
///     48_000,
///     ChannelLayout::Mono,
///     CombResonatorParams { frequency_hz: 480.0, feedback: 0.9, damping: 0.0, mix: 1.0 },
/// );
///
/// // Feed a single-sample impulse into a 256-frame block.
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
/// input.set_active_frames(256);
/// input.channel_mut(0)[0] = 1.0;
///
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 256);
/// output.set_active_frames(256);
///
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A 480 Hz resonator at 48 kHz has a 100-frame loop; with no damping the
/// // first repeat is the impulse scaled by the feedback coefficient.
/// let out = &outputs[0];
/// assert!((out.channel(0)[100] - 0.9).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct CombResonatorNode {
    /// Ring length in frames (`max_delay + 2`), shared by every channel.
    ring_len: usize,
    /// One delay line per channel; each holds exactly `ring_len` samples.
    rings: Vec<Vec<Sample>>,
    /// One-pole lowpass state per channel (the `filtered[n - 1]` term).
    lp_state: Vec<Sample>,
    /// Shared write cursor into every channel's ring buffer.
    write_pos: usize,
    /// Maximum addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Loop delay in frames (`sample_rate / frequency_hz`), possibly fractional.
    delay: Sample,
    /// Feedback coefficient in `[0, MAX_FEEDBACK]`.
    feedback: Sample,
    /// Loop lowpass pole in `[0, 1]`.
    damping: Sample,
    /// Wet/dry blend in `[0, 1]`.
    mix: Sample,
}

impl CombResonatorNode {
    /// Builds a comb resonator for `layout`'s channels running at
    /// `sample_rate` Hz.
    ///
    /// The ring buffer is sized so that a fundamental as low as
    /// [`MIN_FREQUENCY_HZ`] fits. All parameters are sanitized: non-finite
    /// values fall back to safe defaults, `frequency_hz` is clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`, `feedback` to `[0, MAX_FEEDBACK]`,
    /// and `damping`/`mix` to `[0, 1]`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: CombResonatorParams) -> Self {
        let channels = layout.channel_count();
        let sr = sample_rate as Sample;
        // Longest delay we ever need (frames at the lowest supported pitch).
        let max_delay_frames = ops::round(sr / MIN_FREQUENCY_HZ) as usize;
        let ring_len = max_delay_frames + 2;

        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }
        let mut lp_state = Vec::with_capacity(channels);
        lp_state.resize(channels, 0.0);

        let max_delay = max_delay_frames as Sample;
        let nyquist = sr * 0.5;
        let hz = finite_or(params.frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist);
        let delay = (sr / hz).clamp(1.0, max_delay);

        Self {
            ring_len,
            rings,
            lp_state,
            write_pos: 0,
            max_delay,
            delay,
            feedback: finite_or(params.feedback, 0.0).clamp(0.0, MAX_FEEDBACK),
            damping: finite_or(params.damping, 0.0).clamp(0.0, 1.0),
            mix: finite_or(params.mix, 1.0).clamp(0.0, 1.0),
        }
    }

    /// Returns the number of channels this resonator processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Returns the current loop delay in frames.
    #[inline]
    #[must_use]
    pub fn delay_frames(&self) -> Sample {
        self.delay
    }

    /// Returns the current feedback coefficient.
    #[inline]
    #[must_use]
    pub fn feedback(&self) -> Sample {
        self.feedback
    }

    /// Returns the current loop damping.
    #[inline]
    #[must_use]
    pub fn damping(&self) -> Sample {
        self.damping
    }

    /// Returns the current wet/dry blend.
    #[inline]
    #[must_use]
    pub fn mix(&self) -> Sample {
        self.mix
    }

    /// Retunes the resonator to `frequency_hz`, clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`.
    #[inline]
    pub fn set_frequency_hz(&mut self, sample_rate: u32, frequency_hz: Sample) {
        let sr = sample_rate as Sample;
        let nyquist = sr * 0.5;
        let hz = finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist);
        self.delay = (sr / hz).clamp(1.0, self.max_delay);
    }

    /// Sets the feedback coefficient, clamped to `[0, MAX_FEEDBACK]`.
    #[inline]
    pub fn set_feedback(&mut self, feedback: Sample) {
        self.feedback = finite_or(feedback, 0.0).clamp(0.0, MAX_FEEDBACK);
    }

    /// Sets the loop damping, clamped to `[0, 1]`.
    #[inline]
    pub fn set_damping(&mut self, damping: Sample) {
        self.damping = finite_or(damping, 0.0).clamp(0.0, 1.0);
    }

    /// Sets the wet/dry blend, clamped to `[0, 1]`.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample) {
        self.mix = finite_or(mix, 1.0).clamp(0.0, 1.0);
    }
}

impl AudioNode for CombResonatorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels()).min(self.rings.len());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }

        let ring_len = self.ring_len;
        let len_i = ring_len as isize;
        let delay = self.delay;
        let feedback = self.feedback;
        let damping = self.damping;
        let mix = self.mix;
        let dry = 1.0 - mix;

        for f in 0..frames {
            let w = self.write_pos;
            // Fractional read position `delay` frames behind the write head.
            let read_pos = w as Sample - delay;
            let base = ops::floor(read_pos);
            let frac = read_pos - base;
            let base_i = base as isize;
            let i0 = base_i.rem_euclid(len_i) as usize;
            let i1 = (base_i + 1).rem_euclid(len_i) as usize;

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let delayed = {
                    let ring = &self.rings[ch];
                    lerp(ring[i0], ring[i1], frac)
                };
                // One-pole lowpass in the feedback path: higher `damping`
                // lowers the cutoff so upper partials decay faster.
                let filtered = (1.0 - damping) * delayed + damping * self.lp_state[ch];
                self.lp_state[ch] = filtered;
                let recirculated = flush_denormal(x + feedback * filtered);
                self.rings[ch][w] = recirculated;
                output.channel_mut(ch)[f] = dry * x + mix * recirculated;
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
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
        self.write_pos = 0;
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

    /// Builds an interleaved-per-channel input buffer from per-channel slices.
    fn signal(layout: ChannelLayout, frames: usize, per_channel: &[&[Sample]]) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for (ch, data) in per_channel.iter().enumerate() {
            let dst = buf.channel_mut(ch);
            for (d, &s) in dst.iter_mut().zip(data.iter()) {
                *d = s;
            }
        }
        buf
    }

    fn impulse(layout: ChannelLayout, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..buf.channels() {
            buf.channel_mut(ch)[0] = 1.0;
        }
        buf
    }

    fn run(node: &mut CombResonatorNode, input: &AudioBuffer) -> AudioBuffer {
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

    #[test]
    fn mix_zero_is_bypass() {
        let input = signal(ChannelLayout::Mono, 64, &[&[0.3; 64]]);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 300.0,
                feedback: 0.9,
                damping: 0.2,
                mix: 0.0,
            },
        );
        let out = run(&mut node, &input);
        for (o, i) in out.channel(0).iter().zip(input.channel(0)) {
            assert!((o - i).abs() < 1e-6, "not bypassed: {o} vs {i}");
        }
    }

    #[test]
    fn silence_in_silence_out() {
        let input = signal(ChannelLayout::Mono, 128, &[&[0.0; 128]]);
        let mut node = CombResonatorNode::new(SR, ChannelLayout::Mono, CombResonatorParams::default());
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.abs() < 1e-9, "expected silence, got {s}");
        }
    }

    #[test]
    fn undamped_impulse_repeats_at_the_loop_period() {
        // 480 Hz at 48 kHz => 100-frame loop. With damping 0 the interpolation
        // is exact at integer taps, so the k-th repeat is feedback^k.
        let input = impulse(ChannelLayout::Mono, 512);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 480.0,
                feedback: 0.9,
                damping: 0.0,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-6, "dc {}", out.channel(0)[0]);
        assert!((out.channel(0)[100] - 0.9).abs() < 1e-5, "1st {}", out.channel(0)[100]);
        assert!((out.channel(0)[200] - 0.81).abs() < 1e-5, "2nd {}", out.channel(0)[200]);
        assert!((out.channel(0)[300] - 0.729).abs() < 1e-4, "3rd {}", out.channel(0)[300]);
        // Between repeats the response is silent.
        assert!(out.channel(0)[150].abs() < 1e-6, "gap {}", out.channel(0)[150]);
    }

    #[test]
    fn frequency_sets_the_loop_period() {
        // 240 Hz at 48 kHz => 200-frame loop.
        let input = impulse(ChannelLayout::Mono, 512);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 240.0,
                feedback: 0.8,
                damping: 0.0,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        assert!((out.channel(0)[200] - 0.8).abs() < 1e-5, "repeat {}", out.channel(0)[200]);
        assert!(out.channel(0)[100].abs() < 1e-6, "no early repeat {}", out.channel(0)[100]);
    }

    #[test]
    fn retuning_changes_the_period() {
        let input = impulse(ChannelLayout::Mono, 512);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 480.0,
                feedback: 0.8,
                damping: 0.0,
                mix: 1.0,
            },
        );
        node.set_frequency_hz(SR, 240.0);
        let out = run(&mut node, &input);
        // Now a 200-frame loop, not 100.
        assert!(out.channel(0)[100].abs() < 1e-6, "stale period {}", out.channel(0)[100]);
        assert!((out.channel(0)[200] - 0.8).abs() < 1e-5, "new period {}", out.channel(0)[200]);
    }

    #[test]
    fn damping_attenuates_the_repeat() {
        // With damping the first repeat is feedback * (1 - damping) because the
        // lowpass state is still zero when the impulse first recirculates.
        let input = impulse(ChannelLayout::Mono, 256);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 480.0,
                feedback: 0.9,
                damping: 0.5,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        assert!(
            (out.channel(0)[100] - 0.9 * 0.5).abs() < 1e-5,
            "damped repeat {}",
            out.channel(0)[100]
        );
    }

    #[test]
    fn higher_feedback_sustains_longer() {
        let input = impulse(ChannelLayout::Mono, 1024);
        let mut low = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 240.0,
                feedback: 0.5,
                damping: 0.1,
                mix: 1.0,
            },
        );
        let mut high = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 240.0,
                feedback: 0.95,
                damping: 0.1,
                mix: 1.0,
            },
        );
        let out_low = run(&mut low, &input);
        let out_high = run(&mut high, &input);
        // Measure energy in the tail (after the initial impulse).
        let energy = |b: &AudioBuffer| -> Sample {
            b.channel(0)[8..].iter().map(|&s| s * s).sum()
        };
        assert!(
            energy(&out_high) > 4.0 * energy(&out_low),
            "high {} vs low {}",
            energy(&out_high),
            energy(&out_low)
        );
    }

    #[test]
    fn full_damping_kills_resonance() {
        // damping = 1 freezes the lowpass at its initial zero, so nothing
        // recirculates and the output equals the (impulse) input.
        let input = impulse(ChannelLayout::Mono, 256);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 480.0,
                feedback: 0.99,
                damping: 1.0,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-6);
        for &s in &out.channel(0)[1..] {
            assert!(s.abs() < 1e-9, "unexpected ring {s}");
        }
    }

    #[test]
    fn parameters_are_clamped() {
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 5.0,      // below MIN_FREQUENCY_HZ
                feedback: 4.0,          // above MAX_FEEDBACK
                damping: -1.0,          // below 0
                mix: 9.0,               // above 1
            },
        );
        assert!((node.feedback() - MAX_FEEDBACK).abs() < 1e-6);
        assert!(node.damping().abs() < 1e-6);
        assert!((node.mix() - 1.0).abs() < 1e-6);
        // 20 Hz at 48 kHz => 2400-frame loop (the maximum).
        assert!((node.delay_frames() - 2400.0).abs() < 1e-3, "delay {}", node.delay_frames());

        node.set_feedback(-2.0);
        node.set_damping(3.0);
        node.set_mix(-5.0);
        assert!(node.feedback().abs() < 1e-6);
        assert!((node.damping() - 1.0).abs() < 1e-6);
        assert!(node.mix().abs() < 1e-6);
    }

    #[test]
    fn non_finite_parameters_fall_back() {
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: Sample::NAN,
                feedback: Sample::INFINITY,
                damping: Sample::NAN,
                mix: Sample::NEG_INFINITY,
            },
        );
        assert!(node.feedback().is_finite());
        assert!(node.damping().is_finite());
        assert!(node.mix().is_finite());
        assert!(node.delay_frames().is_finite());
        node.set_frequency_hz(SR, Sample::NAN);
        assert!(node.delay_frames().is_finite());
    }

    #[test]
    fn channels_resonate_independently() {
        // Impulse only in the left channel; the right stays silent.
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
        input.set_active_frames(256);
        input.channel_mut(0)[0] = 1.0;
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Stereo,
            CombResonatorParams {
                frequency_hz: 480.0,
                feedback: 0.9,
                damping: 0.0,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        assert!((out.channel(0)[100] - 0.9).abs() < 1e-5, "left {}", out.channel(0)[100]);
        for &s in out.channel(1) {
            assert!(s.abs() < 1e-9, "right leaked {s}");
        }
    }

    #[test]
    fn output_stays_finite_at_max_feedback() {
        let input = impulse(ChannelLayout::Mono, 4096);
        let mut node = CombResonatorNode::new(
            SR,
            ChannelLayout::Mono,
            CombResonatorParams {
                frequency_hz: 110.0,
                feedback: MAX_FEEDBACK,
                damping: 0.05,
                mix: 1.0,
            },
        );
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite(), "non-finite {s}");
            assert!(s.abs() <= 4.0, "unbounded {s}");
        }
    }

    #[test]
    fn reset_is_reproducible() {
        let input = impulse(ChannelLayout::Mono, 512);
        let mut node = CombResonatorNode::new(SR, ChannelLayout::Mono, CombResonatorParams::default());
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        for (a, b) in first.channel(0).iter().zip(second.channel(0)) {
            assert!((a - b).abs() < 1e-6, "reset not reproducible: {a} vs {b}");
        }
    }

    #[test]
    fn zero_frames_do_not_panic() {
        let mut node = CombResonatorNode::new(SR, ChannelLayout::Mono, CombResonatorParams::default());
        let input = impulse(ChannelLayout::Mono, 16);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 16);
        out.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        let [o] = outputs;
        assert_eq!(o.active_frames(), 0);
    }
}
