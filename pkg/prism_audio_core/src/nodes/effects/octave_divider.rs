//! Analog-style octave divider (sub-octave generator) built from a classic
//! flip-flop frequency divider.
//!
//! A rising-edge detector with hysteresis watches the input waveform. Each time
//! the signal crosses from below `-hysteresis` to above `+hysteresis` a new
//! "period" is declared and a toggle flip-flop (`flop1`) flips state. A toggle
//! that flips once per input period produces a square wave at exactly half the
//! input frequency, i.e. one octave down. A second flip-flop (`flop2`) toggles
//! on every other rising edge, dividing by four for a tone two octaves down.
//!
//! ```text
//! rising edge  : x crosses up through +hysteresis  (with Schmitt hysteresis)
//! flop1        : toggles on every rising edge        -> f / 2   (-1 octave)
//! flop2        : toggles on every 2nd rising edge     -> f / 4   (-2 octaves)
//! env          : peak follower, env = max(|x|, env * release_coeff)
//! y = dry*x + sub1*(sign(flop1)*env) + sub2*(sign(flop2)*env)
//! ```
//!
//! The square waves are scaled by a per-channel peak envelope so the generated
//! sub-octaves fade in and out with the dynamics of the input instead of
//! sitting at a constant level. The Schmitt-trigger hysteresis band keeps the
//! edge detector from chattering on noise or on the small wiggles near a zero
//! crossing, which is what gives the classic divider its stable, glitch-free
//! tracking on clean monophonic sources.
//!
//! This is intrinsically a monophonic effect: frequency division relies on
//! counting the periods of a single fundamental, so a chord (several
//! fundamentals at once) has no single period to divide and the detector locks
//! onto whichever partial dominates the waveform. Each channel runs its own
//! independent divider, so a correlated stereo input yields a coherent stereo
//! sub-octave while uncorrelated channels divide independently.
//!
//! # Relationship
//!
//! This node sits beside the other pitch-domain effects but works on a
//! completely different principle:
//!
//! - [`pitch_shifter::PitchShifterNode`](crate::nodes::effects::pitch_shifter)
//!   and [`pitch_delay::PitchDelayNode`](crate::nodes::effects::pitch_delay)
//!   resample / time-stretch the input to shift *every* partial by an arbitrary
//!   ratio while preserving the waveform's spectral shape. The octave divider
//!   does not resample anything; it synthesises brand-new square waves whose
//!   frequency is a hard integer division of the detected fundamental, so it is
//!   locked to exact octaves and adds its own harmonic series.
//! - [`ring_modulator::RingModulatorNode`](crate::nodes::effects::ring_modulator)
//!   and
//!   [`frequency_shifter::FrequencyShifterNode`](crate::nodes::effects::frequency_shifter)
//!   move energy by multiplying with (or single-sideband shifting against) a
//!   carrier, producing generally *inharmonic* sum / difference tones. The
//!   divider instead produces strictly sub-harmonic, musically octave-related
//!   tones.
//!
//! Because the sub-octave levels add on top of the dry signal, the summed
//! output can exceed unity magnitude; like the other `effects` nodes this stage
//! deliberately does not clamp its output and leaves headroom management to a
//! downstream gain or limiter.
//!
//! # Real-time contract
//!
//! All per-channel state (the two flip-flops, the rising-edge counter, the
//! edge-detector arm/disarm flag, and the peak envelope) is allocated once in
//! [`OctaveDividerNode::new`]. [`OctaveDividerNode::process`] performs no
//! allocation, takes no locks, and cannot panic: non-finite inputs are treated
//! as silence, the peak envelope is seeded from a magnitude so it stays
//! non-negative and finite, and every output sample is flushed of denormals.
//! The dry and two sub levels are [`Smoothed`] so automation never clicks; the
//! shared smoothers are advanced exactly once per frame. Latency is zero.
//!
//! # Provenance
//!
//! Pure classic DSP. A toggle flip-flop driven by a comparator is the textbook
//! analog frequency divider, and using one (or two cascaded) to synthesise
//! octave-down square waves is the long-standing design of analog "sub-octave"
//! / "octave divider" effects. Only the elementary edge-detect-and-toggle logic
//! and a one-pole peak follower shown above are used. There is no AI/ML of any
//! kind, and no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, or Web Audio source or derived code.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Default dry (unprocessed) level: the input passes through at unity.
pub const DEFAULT_DRY: Sample = 1.0;

/// Default level of the one-octave-down square wave.
pub const DEFAULT_SUB1: Sample = 0.7;

/// Default level of the two-octave-down square wave (off by default).
pub const DEFAULT_SUB2: Sample = 0.0;

/// Default Schmitt-trigger hysteresis half-width, in input amplitude units.
pub const DEFAULT_HYSTERESIS: Sample = 0.02;

/// Smallest accepted hysteresis half-width. A tiny but non-zero band is kept so
/// the edge detector never chatters exactly at the zero crossing.
pub const MIN_HYSTERESIS: Sample = 0.001;

/// Largest accepted hysteresis half-width.
pub const MAX_HYSTERESIS: Sample = 0.5;

/// Maximum level accepted for the dry and sub-octave mix controls.
pub const MAX_LEVEL: Sample = 4.0;

/// Release time, in milliseconds, of the peak envelope that scales the
/// generated square waves. Short enough to follow note decays, long enough to
/// avoid zipper noise within a single period.
pub const SUB_RELEASE_MS: Sample = 30.0;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Converts the fixed [`SUB_RELEASE_MS`] release time to a one-pole decay
/// coefficient `exp(-1 / (release_seconds * fs))` in `[0, 1)`.
#[inline]
fn release_coefficient(sample_rate: u32) -> Sample {
    let fs = sample_rate.max(1) as Sample;
    let release_seconds = SUB_RELEASE_MS / 1_000.0;
    ops::exp(-1.0 / (release_seconds * fs))
}

/// Configuration for an [`OctaveDividerNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OctaveDividerParams {
    /// Level of the unprocessed input in the mix.
    pub dry: Sample,
    /// Level of the one-octave-down square wave.
    pub sub1: Sample,
    /// Level of the two-octave-down square wave.
    pub sub2: Sample,
    /// Half-width of the Schmitt-trigger hysteresis band, in amplitude units.
    pub hysteresis: Sample,
}

impl Default for OctaveDividerParams {
    fn default() -> Self {
        Self {
            dry: DEFAULT_DRY,
            sub1: DEFAULT_SUB1,
            sub2: DEFAULT_SUB2,
            hysteresis: DEFAULT_HYSTERESIS,
        }
    }
}

impl OctaveDividerParams {
    /// Returns a copy with every field clamped to its supported range and any
    /// non-finite value replaced by the corresponding default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            dry: finite_or(self.dry, DEFAULT_DRY).clamp(0.0, MAX_LEVEL),
            sub1: finite_or(self.sub1, DEFAULT_SUB1).clamp(0.0, MAX_LEVEL),
            sub2: finite_or(self.sub2, DEFAULT_SUB2).clamp(0.0, MAX_LEVEL),
            hysteresis: finite_or(self.hysteresis, DEFAULT_HYSTERESIS)
                .clamp(MIN_HYSTERESIS, MAX_HYSTERESIS),
        }
    }
}

/// Per-channel state of the frequency divider.
#[derive(Debug, Clone, Copy, Default)]
struct ChannelState {
    /// `true` while the detector is "armed high" (has seen the rising edge and
    /// is waiting for the signal to fall back below `-hysteresis`).
    high: bool,
    /// Toggle flip-flop dividing by two (one octave down).
    flop1: bool,
    /// Toggle flip-flop dividing by four (two octaves down).
    flop2: bool,
    /// Count of rising edges seen so far, used to clock `flop2` every other
    /// edge. Wraps harmlessly; only its parity matters.
    rising_count: u32,
    /// Peak-tracking envelope used to scale the square waves.
    env: Sample,
}

/// A per-channel analog-style octave divider (input port 0 -> output port 0).
///
/// Each channel owns an independent flip-flop divider and peak envelope, but
/// the dry and sub-octave levels are shared so a multi-channel signal is mixed
/// coherently.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{OctaveDividerNode, OctaveDividerParams};
///
/// // Sub-octave only, so the output is the synthesised divided square wave.
/// let mut node = OctaveDividerNode::new(
///     48_000,
///     ChannelLayout::Mono,
///     OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.02 },
/// );
///
/// // One second of a 200 Hz sine.
/// let frames = 48_000;
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
/// input.set_active_frames(frames);
/// for (n, s) in input.channel_mut(0).iter_mut().enumerate() {
///     let t = n as f32 / 48_000.0;
///     *s = (2.0 * std::f32::consts::PI * 200.0 * t).sin();
/// }
///
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, frames);
/// output.set_active_frames(frames);
///
/// let ctx = RenderContext { sample_rate: 48_000, frames, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The divider emits a real, non-silent sub-octave once it has locked on.
/// let peak = outputs[0]
///     .channel(0)
///     .iter()
///     .skip(1_000)
///     .fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.3);
/// ```
#[derive(Debug)]
pub struct OctaveDividerNode {
    /// Sample rate in Hz, retained so [`OctaveDividerNode`] can report it.
    sample_rate: u32,
    /// Channel layout reported to the host.
    layout: ChannelLayout,
    /// Number of independently divided channels.
    channels: usize,
    /// Schmitt-trigger hysteresis half-width (already sanitised).
    hysteresis: Sample,
    /// One-pole decay coefficient of the peak envelope.
    release_coeff: Sample,
    /// Smoothed dry level.
    dry: Smoothed,
    /// Smoothed one-octave-down level.
    sub1: Smoothed,
    /// Smoothed two-octaves-down level.
    sub2: Smoothed,
    /// Independent divider state, one entry per channel.
    state: Vec<ChannelState>,
}

impl OctaveDividerNode {
    /// Builds an octave divider for `layout`'s channels running at
    /// `sample_rate` Hz. All parameters are sanitised.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: OctaveDividerParams) -> Self {
        let channels = layout.channel_count();
        let p = params.sanitised();
        Self {
            sample_rate: sample_rate.max(1),
            layout,
            channels,
            hysteresis: p.hysteresis,
            release_coeff: release_coefficient(sample_rate),
            dry: Smoothed::new(p.dry),
            sub1: Smoothed::new(p.sub1),
            sub2: Smoothed::new(p.sub2),
            state: vec![ChannelState::default(); channels],
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

    /// Returns the sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the target dry level the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn dry(&self) -> Sample {
        self.dry.target()
    }

    /// Returns the target one-octave-down level the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn sub1(&self) -> Sample {
        self.sub1.target()
    }

    /// Returns the target two-octaves-down level the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn sub2(&self) -> Sample {
        self.sub2.target()
    }

    /// Returns the current Schmitt-trigger hysteresis half-width.
    #[inline]
    #[must_use]
    pub fn hysteresis(&self) -> Sample {
        self.hysteresis
    }

    /// Sets a new target dry level, gliding with `ramp`. Clamped to
    /// `[0, MAX_LEVEL]`.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        let v = finite_or(dry, self.dry.target()).clamp(0.0, MAX_LEVEL);
        self.dry.set_target(v, ramp);
    }

    /// Sets a new target one-octave-down level, gliding with `ramp`. Clamped to
    /// `[0, MAX_LEVEL]`.
    #[inline]
    pub fn set_sub1(&mut self, sub1: Sample, ramp: Ramp) {
        let v = finite_or(sub1, self.sub1.target()).clamp(0.0, MAX_LEVEL);
        self.sub1.set_target(v, ramp);
    }

    /// Sets a new target two-octaves-down level, gliding with `ramp`. Clamped
    /// to `[0, MAX_LEVEL]`.
    #[inline]
    pub fn set_sub2(&mut self, sub2: Sample, ramp: Ramp) {
        let v = finite_or(sub2, self.sub2.target()).clamp(0.0, MAX_LEVEL);
        self.sub2.set_target(v, ramp);
    }

    /// Updates the Schmitt-trigger hysteresis half-width in place
    /// (allocation-free), clamped to `[MIN_HYSTERESIS, MAX_HYSTERESIS]`. This is
    /// a control-thread operation, not called from [`AudioNode::process`].
    pub fn set_hysteresis(&mut self, hysteresis: Sample) {
        self.hysteresis =
            finite_or(hysteresis, self.hysteresis).clamp(MIN_HYSTERESIS, MAX_HYSTERESIS);
    }
}

impl AudioNode for OctaveDividerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 {
            return;
        }
        let channels = out_channels.min(in_channels).min(self.state.len());

        let hyst = self.hysteresis;
        let release = self.release_coeff;

        for f in 0..frames {
            // Advance the shared smoothed levels exactly once per frame.
            let dry = self.dry.next_sample();
            let sub1 = self.sub1.next_sample();
            let sub2 = self.sub2.next_sample();

            for ch in 0..channels {
                let x = {
                    let v = input.channel(ch)[f];
                    if v.is_finite() { v } else { 0.0 }
                };
                let st = &mut self.state[ch];

                // Peak follower: instant attack, exponential release. Seeded
                // from a magnitude so it is always non-negative.
                st.env = x.abs().max(st.env * release);

                // Schmitt-trigger edge detector. Arm on an upward crossing of
                // +hysteresis (counting a new period and clocking the toggles),
                // disarm only after falling below -hysteresis.
                if !st.high && x > hyst {
                    st.high = true;
                    st.rising_count = st.rising_count.wrapping_add(1);
                    st.flop1 = !st.flop1;
                    if st.rising_count.is_multiple_of(2) {
                        st.flop2 = !st.flop2;
                    }
                } else if st.high && x < -hyst {
                    st.high = false;
                }

                let square1 = if st.flop1 { st.env } else { -st.env };
                let square2 = if st.flop2 { st.env } else { -st.env };
                let y = dry * x + sub1 * square1 + sub2 * square2;
                output.channel_mut(ch)[f] = flush_denormal(y);
            }
        }

        // Any output channels without a matching input / state are silenced.
        for ch in channels..out_channels {
            for s in output.channel_mut(ch)[..frames].iter_mut() {
                *s = 0.0;
            }
        }
    }

    fn reset(&mut self) {
        for st in &mut self.state {
            *st = ChannelState::default();
        }
        self.dry = Smoothed::new(self.dry.target());
        self.sub1 = Smoothed::new(self.sub1.target());
        self.sub2 = Smoothed::new(self.sub2.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    #[inline]
    fn ops_sin(x: Sample) -> Sample {
        ops::sin(x)
    }

    /// Fills a mono buffer with `frames` samples of a sine at `freq` Hz and
    /// amplitude `amp`.
    fn sine_buffer(freq: Sample, amp: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.set_active_frames(frames);
        for (n, s) in buf.channel_mut(0).iter_mut().enumerate() {
            let t = n as Sample / SR as Sample;
            *s = amp * ops_sin(TAU * freq * t);
        }
        buf
    }

    /// Processes `input` through `node` and returns the mono output.
    fn run(node: &mut OctaveDividerNode, input: AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let layout = input.layout();
        let mut output = AudioBuffer::new(layout, frames.max(1));
        output.set_active_frames(frames);
        let c = ctx(frames);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        let [out] = outputs;
        out
    }

    /// Counts rising sign changes (strictly negative to strictly positive) over
    /// a channel, after an initial warm-up region.
    fn rising_sign_changes(samples: &[Sample], skip: usize) -> usize {
        let mut count = 0;
        let mut prev = 0.0_f32;
        for &s in &samples[skip..] {
            if prev <= 0.0 && s > 0.0 {
                count += 1;
            }
            if s != 0.0 {
                prev = s;
            }
        }
        count
    }

    #[test]
    fn default_params_are_sane() {
        let p = OctaveDividerParams::default();
        assert_eq!(p, p.sanitised());
        assert_eq!(p.dry, DEFAULT_DRY);
        assert_eq!(p.sub1, DEFAULT_SUB1);
        assert_eq!(p.sub2, DEFAULT_SUB2);
        assert_eq!(p.hysteresis, DEFAULT_HYSTERESIS);
    }

    #[test]
    fn sanitise_clamps_ranges() {
        let p = OctaveDividerParams {
            dry: 100.0,
            sub1: -1.0,
            sub2: 9.0,
            hysteresis: 10.0,
        }
        .sanitised();
        assert_eq!(p.dry, MAX_LEVEL);
        assert_eq!(p.sub1, 0.0);
        assert_eq!(p.sub2, MAX_LEVEL);
        assert_eq!(p.hysteresis, MAX_HYSTERESIS);
    }

    #[test]
    fn sanitise_replaces_non_finite() {
        let p = OctaveDividerParams {
            dry: Sample::NAN,
            sub1: Sample::INFINITY,
            sub2: Sample::NEG_INFINITY,
            hysteresis: Sample::NAN,
        }
        .sanitised();
        assert_eq!(p.dry, DEFAULT_DRY);
        // Infinity is finite()==false -> fallback default, then clamp.
        assert_eq!(p.sub1, DEFAULT_SUB1);
        assert_eq!(p.sub2, DEFAULT_SUB2);
        assert_eq!(p.hysteresis, DEFAULT_HYSTERESIS);
    }

    #[test]
    fn new_reports_channels_and_layout() {
        let node = OctaveDividerNode::new(SR, ChannelLayout::Stereo, OctaveDividerParams::default());
        assert_eq!(node.channels(), 2);
        assert_eq!(node.layout(), ChannelLayout::Stereo);
        assert_eq!(node.sample_rate(), SR);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        let input = AudioBuffer::new(ChannelLayout::Mono, 256);
        let mut input = input;
        input.set_active_frames(256);
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn dry_only_is_passthrough() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 1.0, sub1: 0.0, sub2: 0.0, hysteresis: 0.02 },
        );
        let input = sine_buffer(220.0, 0.8, 2_000);
        let reference: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input);
        for (o, i) in out.channel(0).iter().zip(reference.iter()) {
            assert!((o - i).abs() < 1e-6, "o={o} i={i}");
        }
    }

    #[test]
    fn all_zero_levels_give_silence() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 0.0, sub2: 0.0, hysteresis: 0.02 },
        );
        let input = sine_buffer(220.0, 1.0, 2_000);
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn sub1_is_one_octave_down() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.02 },
        );
        let freq = 200.0;
        let frames = SR as usize; // one second
        let input = sine_buffer(freq, 1.0, frames);
        let out = run(&mut node, input);
        // Reference: how many input cycles in the measured region.
        let input_cycles = rising_sign_changes(
            &sine_buffer(freq, 1.0, frames).channel(0).to_vec(),
            2_000,
        );
        let sub_cycles = rising_sign_changes(out.channel(0), 2_000);
        // flop1 completes one full square cycle every two input periods.
        let ratio = sub_cycles as f32 / input_cycles as f32;
        assert!((ratio - 0.5).abs() < 0.05, "ratio={ratio}");
    }

    #[test]
    fn sub2_is_two_octaves_down() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 0.0, sub2: 1.0, hysteresis: 0.02 },
        );
        let freq = 200.0;
        let frames = SR as usize;
        let input = sine_buffer(freq, 1.0, frames);
        let out = run(&mut node, input);
        let input_cycles = rising_sign_changes(
            &sine_buffer(freq, 1.0, frames).channel(0).to_vec(),
            2_000,
        );
        let sub_cycles = rising_sign_changes(out.channel(0), 2_000);
        let ratio = sub_cycles as f32 / input_cycles as f32;
        assert!((ratio - 0.25).abs() < 0.03, "ratio={ratio}");
    }

    #[test]
    fn envelope_tracks_input_amplitude() {
        let make = || {
            OctaveDividerNode::new(
                SR,
                ChannelLayout::Mono,
                OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.01 },
            )
        };
        let frames = SR as usize / 2;
        let peak = |amp: Sample| {
            let mut node = make();
            let out = run(&mut node, sine_buffer(200.0, amp, frames));
            out.channel(0)
                .iter()
                .skip(2_000)
                .fold(0.0_f32, |m, s| m.max(s.abs()))
        };
        let loud = peak(1.0);
        let quiet = peak(0.4);
        assert!(loud > quiet * 1.5, "loud={loud} quiet={quiet}");
    }

    #[test]
    fn envelope_decays_on_silence() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.01 },
        );
        // Loud burst, then silence.
        let frames = SR as usize / 2;
        let mut input = sine_buffer(200.0, 1.0, frames);
        let half = frames / 2;
        for s in input.channel_mut(0)[half..].iter_mut() {
            *s = 0.0;
        }
        let out = run(&mut node, input);
        let tail_peak = out.channel(0)[frames - 1_000..]
            .iter()
            .fold(0.0_f32, |m, s| m.max(s.abs()));
        assert!(tail_peak < 0.05, "tail_peak={tail_peak}");
    }

    #[test]
    fn output_can_exceed_unity() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 1.0, sub1: 1.0, sub2: 1.0, hysteresis: 0.01 },
        );
        let out = run(&mut node, sine_buffer(200.0, 1.0, SR as usize / 2));
        let peak = out.channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
        assert!(peak > 1.0, "peak={peak}");
    }

    #[test]
    fn non_finite_input_treated_as_silence() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        let mut input = sine_buffer(200.0, 1.0, 512);
        input.channel_mut(0)[100] = Sample::NAN;
        input.channel_mut(0)[200] = Sample::INFINITY;
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn hysteresis_prevents_chatter() {
        // Tiny sub-threshold wiggle around zero must never clock the divider.
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.1 },
        );
        let frames = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        input.set_active_frames(frames);
        for (n, s) in input.channel_mut(0).iter_mut().enumerate() {
            let t = n as Sample / SR as Sample;
            *s = 0.05 * ops_sin(TAU * 500.0 * t); // amplitude below hysteresis
        }
        let out = run(&mut node, input);
        // No rising edge ever arms -> flop1 stays false -> output is -env only,
        // never crosses up through zero.
        let rises = rising_sign_changes(out.channel(0), 0);
        assert_eq!(rises, 0, "rises={rises}");
    }

    #[test]
    fn reset_clears_state() {
        let mut node = OctaveDividerNode::new(
            SR,
            ChannelLayout::Mono,
            OctaveDividerParams { dry: 0.0, sub1: 1.0, sub2: 0.0, hysteresis: 0.02 },
        );
        let _ = run(&mut node, sine_buffer(200.0, 1.0, 2_000));
        node.reset();
        // After reset, feeding silence must give silence (env and flops cleared).
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        input.set_active_frames(256);
        let out = run(&mut node, input);
        assert!(out.channel(0).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn set_levels_update_targets() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        node.set_dry(0.5, Ramp::Immediate);
        node.set_sub1(0.25, Ramp::Immediate);
        node.set_sub2(0.1, Ramp::Immediate);
        assert!((node.dry() - 0.5).abs() < 1e-6);
        assert!((node.sub1() - 0.25).abs() < 1e-6);
        assert!((node.sub2() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn set_levels_clamp() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        node.set_dry(100.0, Ramp::Immediate);
        node.set_sub1(-5.0, Ramp::Immediate);
        assert_eq!(node.dry(), MAX_LEVEL);
        assert_eq!(node.sub1(), 0.0);
    }

    #[test]
    fn set_hysteresis_clamps() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        node.set_hysteresis(100.0);
        assert_eq!(node.hysteresis(), MAX_HYSTERESIS);
        node.set_hysteresis(0.0);
        assert_eq!(node.hysteresis(), MIN_HYSTERESIS);
        node.set_hysteresis(Sample::NAN);
        assert_eq!(node.hysteresis(), MIN_HYSTERESIS);
    }

    #[test]
    fn extra_output_channels_are_silenced() {
        // Mono state, stereo output: channel 1 has no state and must be zeroed.
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        let input = sine_buffer(200.0, 1.0, 512);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 512);
        output.set_active_frames(512);
        let c = ctx(512);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        assert!(outputs[0].channel(1).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn stereo_correlated_input_is_coherent() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Stereo, OctaveDividerParams {
                dry: 0.0,
                sub1: 1.0,
                sub2: 0.0,
                hysteresis: 0.02,
            });
        let frames = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        input.set_active_frames(frames);
        for n in 0..frames {
            let t = n as Sample / SR as Sample;
            let v = ops_sin(TAU * 200.0 * t);
            input.channel_mut(0)[n] = v;
            input.channel_mut(1)[n] = v;
        }
        let out = run(&mut node, input);
        for n in 0..frames {
            assert!((out.channel(0)[n] - out.channel(1)[n]).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_buffer_is_a_no_op() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let c = ctx(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn latency_is_zero() {
        let node = OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn output_is_finite_and_denormal_free() {
        let mut node =
            OctaveDividerNode::new(SR, ChannelLayout::Mono, OctaveDividerParams::default());
        let out = run(&mut node, sine_buffer(200.0, 1.0, 4_000));
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s == 0.0 || s.abs() >= f32::MIN_POSITIVE);
        }
    }
}
