//! Formant-wave-function (FOF) source node: a pitch-synchronous stream of
//! formant grains that synthesizes voiced, vowel-like tones from scratch.
//!
//! A [`FofSourceNode`] is a zero-input, one-output source. On every fundamental
//! period it fires one short *formant wave function* grain for each configured
//! formant; summed across many overlapping periods the grains reconstruct a
//! spectrum with sharp harmonic structure (from the periodic triggering) under
//! broad formant peaks (from the per-grain damped resonances), the hallmark of
//! sung vowels, choirs, and expressive synthetic voices.
//!
//! # Model
//!
//! A single FOF grain is a damped sinusoid at a formant centre frequency `fc`
//! shaped by a two-stage amplitude envelope: a raised-cosine *attack skirt*
//! that opens the grain smoothly, followed by an exponential decay whose rate
//! sets the formant bandwidth:
//!
//! ```text
//! grain(t) = a * env(t) * sin(2*pi * fc * t)
//! env(t)   = attack(t) * exp(-pi * bw * t)
//! attack(t)= 0.5 - 0.5*cos(pi * t / tex)   (0 <= t < tex, then clamped to 1)
//! ```
//!
//! where `bw` is the formant bandwidth in hertz (the exponential time constant
//! `alpha = pi * bw` gives the -3 dB resonance width), `tex` is the shared
//! excitation (skirt) time that controls how far the formant's spectral skirts
//! extend, and `a` is the formant's linear amplitude. A grain retires once its
//! decay has fallen below [`FOF_RETIRE_LEVEL`] and its attack has completed, so
//! each grain lives for a bounded number of samples.
//!
//! A fundamental phase accumulator advances by `f0 / sample_rate` each sample;
//! every time it wraps past `1` the node triggers one fresh grain per active
//! formant (a glottal pulse), each allocated from a shared fixed-size pool
//! sized by [`MAX_FOF_GRAINS`]. Because the trigger is strictly periodic the
//! output is perfectly pitched at `f0`, while each formant contributes a
//! resonant lobe centred at its `fc`: the classic independent control of pitch
//! and spectral envelope that makes FOF synthesis so expressive for voice.
//!
//! The summed grains are scaled by a [`Smoothed`] overall `amplitude`. As with
//! the other `sources`, overlapping grains can momentarily sum above unity, so
//! route the output through a limiter when a hard ceiling is required.
//!
//! # Relationship
//!
//! This node *synthesizes* a voiced tone, so it is a true zero-input source.
//! That distinguishes it from [`crate::nodes::effects::formant_filter`], which
//! is a subtractive *effect* that imposes formant resonances on an external
//! input, and from [`crate::nodes::effects::vocoder`], which cross-maps the
//! spectral envelope of one signal onto another. Unlike the stochastic grain
//! cloud of [`super::granular_source::GranularSourceNode`] (randomized,
//! decorrelated atoms) the FOF grain stream is deterministic and
//! pitch-synchronous: each grain is a formant wave function locked to the
//! fundamental. For a single partial use
//! [`super::oscillator::OscillatorNode`]; for an arbitrary fixed harmonic sum
//! use [`super::additive_oscillator::AdditiveOscillatorNode`].
//!
//! # Determinism
//!
//! The grain schedule is purely periodic and carries no randomness, so two
//! [`FofSourceNode`]s built with identical parameters emit bit-identical
//! streams on every platform, and [`AudioNode::reset`] replays the same tone.
//!
//! # Real-time contract
//!
//! The grain pool is a compile-time fixed array, so
//! [`FofSourceNode::process`] performs no allocation, locking, or panicking:
//! triggering reuses retired slots (or drops the grain if the pool is
//! saturated) and every per-sample step is a bounded set of multiply-adds plus
//! one sine per active grain. Pitch, formant, bandwidth, and skirt controls are
//! plain scalars (they only influence future grains), while `amplitude` is
//! smoothed so level automation never zippers.
//!
//! # Provenance
//!
//! Implemented from first principles from the public formant-synthesis
//! literature: X. Rodet's formant-wave-function (`FOF`, *Fonction d'Onde
//! Formantique*) method and the `CHANT` voice synthesizer (X. Rodet,
//! Y. Potard, J.-B. Barriere, "The CHANT Project: From the Synthesis of the
//! Singing Voice to Synthesis in General", *Computer Music Journal*, 1984),
//! with the standard raised-cosine skirt and exponential-decay grain envelope.
//! Nothing here is derived from any AI/ML technique, nor from the source code
//! of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, or Web Audio; only the shared mathematical ideas are referenced.

use bevy_math::ops;
use core::f32::consts::{PI, TAU};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Number of formants the node tracks. A formant with zero gain is inert.
pub const MAX_FORMANTS: usize = 5;

/// Maximum number of simultaneously sounding grains across all formants.
/// Triggers beyond this are dropped, bounding the per-sample work.
pub const MAX_FOF_GRAINS: usize = 128;

/// Default fundamental (pitch) frequency in hertz.
pub const DEFAULT_FOF_FUNDAMENTAL_HZ: Sample = 110.0;

/// Maximum fundamental frequency in hertz.
pub const MAX_FOF_FUNDAMENTAL_HZ: Sample = 2_000.0;

/// Default excitation (attack skirt) time in milliseconds.
pub const DEFAULT_FOF_SKIRT_MS: Sample = 2.0;

/// Minimum excitation time in milliseconds (keeps the attack well-defined).
pub const MIN_FOF_SKIRT_MS: Sample = 0.1;

/// Maximum excitation time in milliseconds.
pub const MAX_FOF_SKIRT_MS: Sample = 50.0;

/// Minimum formant bandwidth in hertz (guarantees every grain decays).
pub const MIN_FOF_BANDWIDTH_HZ: Sample = 10.0;

/// Maximum formant bandwidth in hertz.
pub const MAX_FOF_BANDWIDTH_HZ: Sample = 2_000.0;

/// Default linear output amplitude.
pub const DEFAULT_FOF_AMPLITUDE: Sample = 1.0;

/// Envelope level below which a decaying grain retires (about -80 dB).
pub const FOF_RETIRE_LEVEL: Sample = 1.0e-4;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Wraps a normalized phase accumulator back into `[0, 1)` without a transcendental.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "phase is small and bounded; the i64 floor is exact for audio phases"
)]
fn wrap01(phase: Sample) -> Sample {
    let p = phase - (phase as i64 as Sample);
    if p < 0.0 { p + 1.0 } else { p }
}

/// One formant's steady spectral description (centre, width, and level).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Formant {
    /// Formant centre frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Formant bandwidth in hertz. Clamped to `[MIN, MAX]`.
    pub bandwidth_hz: Sample,
    /// Linear amplitude (relative weight). Clamped non-negative; `0` disables.
    pub gain: Sample,
}

impl Formant {
    /// Returns a copy with every field coerced into its valid range.
    #[must_use]
    fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, 0.0).max(0.0),
            bandwidth_hz: finite_or(self.bandwidth_hz, MIN_FOF_BANDWIDTH_HZ)
                .clamp(MIN_FOF_BANDWIDTH_HZ, MAX_FOF_BANDWIDTH_HZ),
            gain: finite_or(self.gain, 0.0).max(0.0),
        }
    }
}

/// A single live FOF grain: a damped sine with a raised-cosine attack skirt.
#[derive(Debug, Clone, Copy, Default)]
struct FofGrain {
    /// Whether this slot is currently sounding.
    active: bool,
    /// Carrier phase in `[0, 1)`.
    phase: Sample,
    /// Carrier phase increment per sample (`fc / sample_rate`).
    phase_inc: Sample,
    /// Current exponential-decay envelope value (starts at `1`).
    decay_env: Sample,
    /// Per-sample decay multiplier (`exp(-pi * bw / sample_rate)`).
    decay_mult: Sample,
    /// Attack-skirt phase in `[0, 1]` (`1` once the attack has completed).
    attack_phase: Sample,
    /// Attack-skirt phase increment per sample (`1 / tex_samples`).
    attack_inc: Sample,
    /// Baked formant amplitude for this grain.
    amp: Sample,
}

/// Construction parameters for a [`FofSourceNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FofSourceParams {
    /// Fundamental (pitch) frequency in hertz. Clamped to `[0, MAX]`.
    pub fundamental_hz: Sample,
    /// Excitation (attack skirt) time in milliseconds. Clamped to `[MIN, MAX]`.
    pub skirt_ms: Sample,
    /// The formant bank (centre/bandwidth/gain triples).
    pub formants: [Formant; MAX_FORMANTS],
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

/// The default formant bank: an open `/a/`-like vowel.
const DEFAULT_FORMANTS: [Formant; MAX_FORMANTS] = [
    Formant { frequency_hz: 800.0, bandwidth_hz: 80.0, gain: 1.0 },
    Formant { frequency_hz: 1_150.0, bandwidth_hz: 90.0, gain: 0.5 },
    Formant { frequency_hz: 2_900.0, bandwidth_hz: 120.0, gain: 0.25 },
    Formant { frequency_hz: 3_900.0, bandwidth_hz: 130.0, gain: 0.2 },
    Formant { frequency_hz: 4_950.0, bandwidth_hz: 140.0, gain: 0.1 },
];

impl Default for FofSourceParams {
    fn default() -> Self {
        Self {
            fundamental_hz: DEFAULT_FOF_FUNDAMENTAL_HZ,
            skirt_ms: DEFAULT_FOF_SKIRT_MS,
            formants: DEFAULT_FORMANTS,
            amplitude: DEFAULT_FOF_AMPLITUDE,
        }
    }
}

impl FofSourceParams {
    /// Returns a copy with every field coerced into its valid range and all
    /// non-finite values replaced by their defaults.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let mut formants = self.formants;
        for f in &mut formants {
            *f = f.sanitised();
        }
        Self {
            fundamental_hz: finite_or(self.fundamental_hz, DEFAULT_FOF_FUNDAMENTAL_HZ)
                .clamp(0.0, MAX_FOF_FUNDAMENTAL_HZ),
            skirt_ms: finite_or(self.skirt_ms, DEFAULT_FOF_SKIRT_MS)
                .clamp(MIN_FOF_SKIRT_MS, MAX_FOF_SKIRT_MS),
            formants,
            amplitude: finite_or(self.amplitude, DEFAULT_FOF_AMPLITUDE),
        }
    }
}

/// A formant-wave-function (FOF) voice source node (0 inputs, 1 output).
///
/// The mono grain stream is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{FofSourceNode, FofSourceParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = FofSourceNode::from_params(FofSourceParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4_800)];
/// outputs[0].set_active_frames(4_800);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4_800, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A voiced vowel is not silent and stays finite.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct FofSourceNode {
    /// Fundamental frequency in hertz, in `[0, MAX]`.
    fundamental: Sample,
    /// Excitation (skirt) time in milliseconds, in `[MIN, MAX]`.
    skirt_ms: Sample,
    /// The formant bank.
    formants: [Formant; MAX_FORMANTS],
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Fixed-size grain pool shared by every formant.
    grains: [FofGrain; MAX_FOF_GRAINS],
    /// Fundamental phase accumulator in `[0, 1)`; a wrap triggers a glottal pulse.
    f0_phase: Sample,
}

impl FofSourceNode {
    /// Creates an FOF source from a [`FofSourceParams`] bundle.
    ///
    /// All parameters are sanitized via [`FofSourceParams::sanitised`].
    #[must_use]
    pub fn from_params(params: FofSourceParams) -> Self {
        let p = params.sanitised();
        Self {
            fundamental: p.fundamental_hz,
            skirt_ms: p.skirt_ms,
            formants: p.formants,
            amplitude: Smoothed::new(p.amplitude),
            grains: [FofGrain::default(); MAX_FOF_GRAINS],
            f0_phase: 0.0,
        }
    }

    /// Fundamental (pitch) frequency in hertz.
    #[inline]
    #[must_use]
    pub fn fundamental(&self) -> Sample {
        self.fundamental
    }

    /// Excitation (attack skirt) time in milliseconds.
    #[inline]
    #[must_use]
    pub fn skirt_ms(&self) -> Sample {
        self.skirt_ms
    }

    /// Target output amplitude.
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns the formant description at `index` (clamped to the bank).
    #[inline]
    #[must_use]
    pub fn formant(&self, index: usize) -> Formant {
        self.formants[index.min(MAX_FORMANTS - 1)]
    }

    /// Sets the fundamental frequency in hertz (clamped to `[0, MAX]`).
    ///
    /// Click-free: it only affects grains triggered after the change.
    #[inline]
    pub fn set_fundamental(&mut self, hz: Sample) {
        self.fundamental = finite_or(hz, self.fundamental).clamp(0.0, MAX_FOF_FUNDAMENTAL_HZ);
    }

    /// Sets the excitation (skirt) time in milliseconds (clamped to `[MIN, MAX]`).
    #[inline]
    pub fn set_skirt_ms(&mut self, skirt_ms: Sample) {
        self.skirt_ms =
            finite_or(skirt_ms, self.skirt_ms).clamp(MIN_FOF_SKIRT_MS, MAX_FOF_SKIRT_MS);
    }

    /// Replaces the formant at `index` (ignored when `index` is out of range),
    /// sanitizing the supplied description.
    #[inline]
    pub fn set_formant(&mut self, index: usize, formant: Formant) {
        if index < MAX_FORMANTS {
            self.formants[index] = formant.sanitised();
        }
    }

    /// Sets the target output amplitude, smoothing over `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, amplitude: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(amplitude, self.amplitude.target()), ramp);
    }

    /// Writes a grain into the first free pool slot (dropped if the pool is full).
    #[inline]
    fn spawn_one(&mut self, grain: FofGrain) {
        for slot in &mut self.grains {
            if !slot.active {
                *slot = grain;
                return;
            }
        }
        // Pool saturated: the grain is dropped.
    }

    /// Fires one grain per active formant: a glottal pulse at the fundamental.
    fn trigger(&mut self, sample_rate: Sample) {
        let tex_samples = (self.skirt_ms * 0.001 * sample_rate).max(1.0);
        let attack_inc = 1.0 / tex_samples;
        for i in 0..MAX_FORMANTS {
            let f = self.formants[i];
            if f.gain <= 0.0 || f.frequency_hz <= 0.0 {
                continue;
            }
            self.spawn_one(FofGrain {
                active: true,
                phase: 0.0,
                phase_inc: f.frequency_hz / sample_rate,
                decay_env: 1.0,
                decay_mult: ops::exp(-PI * f.bandwidth_hz / sample_rate),
                attack_phase: 0.0,
                attack_inc,
                amp: f.gain,
            });
        }
    }

    /// Renders one mono sample, advancing the fundamental schedule, every active
    /// grain, and the smoothed amplitude exactly once.
    #[inline]
    fn render_sample(&mut self, sample_rate: Sample) -> Sample {
        self.f0_phase += self.fundamental / sample_rate;
        while self.f0_phase >= 1.0 {
            self.f0_phase -= 1.0;
            self.trigger(sample_rate);
        }

        let mut acc = 0.0;
        for grain in &mut self.grains {
            if !grain.active {
                continue;
            }
            let attack = 0.5 - 0.5 * ops::cos(PI * grain.attack_phase);
            let env = grain.decay_env * attack;
            acc += grain.amp * env * ops::sin(TAU * grain.phase);

            grain.phase = wrap01(grain.phase + grain.phase_inc);
            grain.decay_env *= grain.decay_mult;
            if grain.attack_phase < 1.0 {
                grain.attack_phase = (grain.attack_phase + grain.attack_inc).min(1.0);
            }
            if grain.decay_env < FOF_RETIRE_LEVEL && grain.attack_phase >= 1.0 {
                grain.active = false;
            }
        }

        flush_denormal(acc * self.amplitude.next_sample())
    }
}

impl AudioNode for FofSourceNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }
        let frames = io.output(0).active_frames();
        if frames == 0 {
            return;
        }
        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(sample_rate);
            }
        }
        // Replicate the mono signal into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.grains = [FofGrain::default(); MAX_FOF_GRAINS];
        self.f0_phase = 0.0;
        self.amplitude = Smoothed::new(self.amplitude.target());
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

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext { sample_rate, frames, playhead: 0 }
    }

    fn render(node: &mut FofSourceNode, layout: ChannelLayout, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    fn peak(buf: &AudioBuffer, ch: usize) -> Sample {
        buf.channel(ch).iter().fold(0.0_f32, |m, s| m.max(s.abs()))
    }

    fn rms(buf: &AudioBuffer, ch: usize) -> Sample {
        let data = buf.channel(ch);
        let sum: Sample = data.iter().map(|s| s * s).sum();
        ops::sqrt(sum / data.len() as Sample)
    }

    fn default_node() -> FofSourceNode {
        FofSourceNode::from_params(FofSourceParams::default())
    }

    #[test]
    fn default_vowel_is_not_silent() {
        let mut node = default_node();
        let buf = render(&mut node, ChannelLayout::Mono, 4_800);
        assert!(peak(&buf, 0) > 0.0, "default /a/ vowel must produce output");
    }

    #[test]
    fn output_is_finite_and_bounded() {
        let mut node = default_node();
        let buf = render(&mut node, ChannelLayout::Mono, 9_600);
        for &s in buf.channel(0) {
            assert!(s.is_finite(), "sample must stay finite");
            assert!(s.abs() < 64.0, "overlapping grains must stay bounded");
        }
    }

    #[test]
    fn same_params_are_bit_identical() {
        let mut a = default_node();
        let mut b = default_node();
        let ba = render(&mut a, ChannelLayout::Mono, 4_800);
        let bb = render(&mut b, ChannelLayout::Mono, 4_800);
        assert_eq!(ba.channel(0), bb.channel(0), "schedule carries no randomness");
    }

    #[test]
    fn reset_replays_same_tone() {
        let mut node = default_node();
        let first = render(&mut node, ChannelLayout::Mono, 4_800);
        node.reset();
        let second = render(&mut node, ChannelLayout::Mono, 4_800);
        assert_eq!(first.channel(0), second.channel(0), "reset must replay the tone");
    }

    #[test]
    fn zero_fundamental_is_silent() {
        let params = FofSourceParams { fundamental_hz: 0.0, ..FofSourceParams::default() };
        let mut node = FofSourceNode::from_params(params);
        let buf = render(&mut node, ChannelLayout::Mono, 4_800);
        assert_eq!(peak(&buf, 0), 0.0, "no fundamental means no glottal pulses");
    }

    #[test]
    fn all_formant_gains_zero_is_silent() {
        let mut params = FofSourceParams::default();
        for f in &mut params.formants {
            f.gain = 0.0;
        }
        let mut node = FofSourceNode::from_params(params);
        let buf = render(&mut node, ChannelLayout::Mono, 4_800);
        assert_eq!(peak(&buf, 0), 0.0, "silent formant bank emits nothing");
    }

    #[test]
    fn amplitude_scales_output_linearly() {
        let mut quiet = FofSourceNode::from_params(FofSourceParams {
            amplitude: 0.25,
            ..FofSourceParams::default()
        });
        let mut loud = FofSourceNode::from_params(FofSourceParams {
            amplitude: 0.5,
            ..FofSourceParams::default()
        });
        let bq = render(&mut quiet, ChannelLayout::Mono, 4_800);
        let bl = render(&mut loud, ChannelLayout::Mono, 4_800);
        for (q, l) in bq.channel(0).iter().zip(bl.channel(0).iter()) {
            assert!((2.0 * q - l).abs() <= 1.0e-5, "amplitude is a linear scalar");
        }
    }

    #[test]
    fn mono_core_replicates_to_every_channel() {
        let mut node = default_node();
        let buf = render(&mut node, ChannelLayout::Surround5_1, 2_400);
        let base = buf.channel(0).to_vec();
        for ch in 1..buf.channels() {
            assert_eq!(buf.channel(ch), base.as_slice(), "channels must be identical");
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = default_node();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 16);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0, "zero-frame render is a no-op");
    }

    #[test]
    fn getters_report_constructor_values() {
        let node = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 220.0,
            skirt_ms: 3.0,
            amplitude: 0.75,
            ..FofSourceParams::default()
        });
        assert!((node.fundamental() - 220.0).abs() < 1.0e-6);
        assert!((node.skirt_ms() - 3.0).abs() < 1.0e-6);
        assert!((node.amplitude() - 0.75).abs() < 1.0e-6);
        assert!((node.formant(0).frequency_hz - 800.0).abs() < 1.0e-6);
    }

    #[test]
    fn set_fundamental_clamps_and_rejects_nonfinite() {
        let mut node = default_node();
        node.set_fundamental(10_000.0);
        assert!((node.fundamental() - MAX_FOF_FUNDAMENTAL_HZ).abs() < 1.0e-6);
        node.set_fundamental(-5.0);
        assert_eq!(node.fundamental(), 0.0);
        node.set_fundamental(123.0);
        node.set_fundamental(Sample::NAN);
        assert!((node.fundamental() - 123.0).abs() < 1.0e-6, "NaN keeps the old value");
    }

    #[test]
    fn set_skirt_ms_clamps() {
        let mut node = default_node();
        node.set_skirt_ms(1_000.0);
        assert!((node.skirt_ms() - MAX_FOF_SKIRT_MS).abs() < 1.0e-6);
        node.set_skirt_ms(0.0);
        assert!((node.skirt_ms() - MIN_FOF_SKIRT_MS).abs() < 1.0e-6);
    }

    #[test]
    fn set_amplitude_updates_target() {
        let mut node = default_node();
        node.set_amplitude(0.3, Ramp::Immediate);
        assert!((node.amplitude() - 0.3).abs() < 1.0e-6);
    }

    #[test]
    fn set_formant_out_of_range_is_ignored() {
        let mut node = default_node();
        let before = node.formant(0);
        node.set_formant(
            MAX_FORMANTS,
            Formant { frequency_hz: 1.0, bandwidth_hz: 50.0, gain: 1.0 },
        );
        assert_eq!(node.formant(0).frequency_hz, before.frequency_hz);
    }

    #[test]
    fn set_formant_sanitises_bandwidth() {
        let mut node = default_node();
        node.set_formant(
            0,
            Formant { frequency_hz: 1_000.0, bandwidth_hz: 1.0, gain: 2.0 },
        );
        let f = node.formant(0);
        assert!((f.frequency_hz - 1_000.0).abs() < 1.0e-6);
        assert!((f.bandwidth_hz - MIN_FOF_BANDWIDTH_HZ).abs() < 1.0e-6);
        assert!((f.gain - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn formant_getter_clamps_index() {
        let node = default_node();
        let last = node.formant(MAX_FORMANTS - 1);
        assert_eq!(node.formant(999).frequency_hz, last.frequency_hz);
    }

    #[test]
    fn params_sanitise_replaces_nonfinite() {
        let params = FofSourceParams {
            fundamental_hz: Sample::INFINITY,
            skirt_ms: Sample::NAN,
            amplitude: Sample::NAN,
            formants: [Formant { frequency_hz: -1.0, bandwidth_hz: -5.0, gain: -2.0 };
                MAX_FORMANTS],
        }
        .sanitised();
        assert!((params.fundamental_hz - DEFAULT_FOF_FUNDAMENTAL_HZ).abs() < 1.0e-6);
        assert!((params.skirt_ms - DEFAULT_FOF_SKIRT_MS).abs() < 1.0e-6);
        assert!((params.amplitude - DEFAULT_FOF_AMPLITUDE).abs() < 1.0e-6);
        assert_eq!(params.formants[0].frequency_hz, 0.0);
        assert_eq!(params.formants[0].gain, 0.0);
        assert!((params.formants[0].bandwidth_hz - MIN_FOF_BANDWIDTH_HZ).abs() < 1.0e-6);
    }

    #[test]
    fn fundamental_changes_the_output() {
        let mut low = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 110.0,
            ..FofSourceParams::default()
        });
        let mut high = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 220.0,
            ..FofSourceParams::default()
        });
        let bl = render(&mut low, ChannelLayout::Mono, 4_800);
        let bh = render(&mut high, ChannelLayout::Mono, 4_800);
        assert_ne!(bl.channel(0), bh.channel(0), "pitch must affect the signal");
    }

    #[test]
    fn higher_fundamental_increases_energy() {
        // More glottal pulses per second deposit more grain energy.
        let mut low = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 100.0,
            ..FofSourceParams::default()
        });
        let mut high = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 400.0,
            ..FofSourceParams::default()
        });
        let bl = render(&mut low, ChannelLayout::Mono, 9_600);
        let bh = render(&mut high, ChannelLayout::Mono, 9_600);
        assert!(rms(&bh, 0) > rms(&bl, 0), "denser pulses raise RMS");
    }

    #[test]
    fn formant_frequency_affects_output() {
        let mut a = default_node();
        let mut b = default_node();
        b.set_formant(0, Formant { frequency_hz: 1_600.0, bandwidth_hz: 80.0, gain: 1.0 });
        let ba = render(&mut a, ChannelLayout::Mono, 4_800);
        let bb = render(&mut b, ChannelLayout::Mono, 4_800);
        assert_ne!(ba.channel(0), bb.channel(0), "formant centre shapes the timbre");
    }

    #[test]
    fn long_run_stays_bounded_after_pool_recycling() {
        let mut node = FofSourceNode::from_params(FofSourceParams {
            fundamental_hz: 500.0,
            ..FofSourceParams::default()
        });
        let buf = render(&mut node, ChannelLayout::Mono, 48_000);
        for &s in buf.channel(0) {
            assert!(s.is_finite() && s.abs() < 64.0, "retired slots must recycle cleanly");
        }
    }

    #[test]
    fn latency_is_zero() {
        assert_eq!(default_node().latency_frames(), 0);
    }

    #[test]
    fn zero_sample_rate_is_guarded() {
        let mut node = default_node();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 128);
        out.set_active_frames(128);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0, 128), &mut io);
        for &s in outputs[0].channel(0) {
            assert!(s.is_finite(), "a zero sample rate must not produce NaN");
        }
    }
}
