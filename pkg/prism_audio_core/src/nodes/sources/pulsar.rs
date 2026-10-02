//! Pulsar synthesis source node.
//!
//! [`PulsarNode`] emits, once per fundamental period, a single windowed
//! sinusoidal burst (a "pulsaret") followed by silence for the rest of the
//! period. The fundamental rate sets the pitch, the pulsaret's internal
//! sinusoid places a movable formant peak, and the fraction of the period the
//! pulsaret occupies (its duty cycle) sets the spectral width -- a short
//! pulsaret with a long silent gap is bright and buzzy, a pulsaret filling the
//! whole period is a smooth windowed tone. This is the classic pulsar-synthesis
//! texture: a formant that glides independently of pitch with a continuously
//! variable brightness.
//!
//! # Model
//!
//! One fundamental period has normalized phase `p` in `[0, 1)` advancing by
//! `f0 / sample_rate` per sample. A second phase `c` advances independently at
//! `formant / sample_rate` and is re-locked to zero at every period boundary,
//! so each pulsaret always starts at carrier phase zero. The pulsaret occupies
//! `p` in `[0, duty)` and the remainder of the period is silent:
//!
//! ```text
//!   p < duty  ->  u       = p / duty             (phase within pulsaret)
//!                 window  = sin(pi * u)^2        (Hann window)
//!                 carrier = sin(2*pi * c)        (formant sinusoid)
//!                 x       = window * carrier
//!   p >= duty ->  x       = 0
//! ```
//!
//! The pulsaret is a sinusoid at `formant` hertz multiplied by a raised-sine
//! (Hann) window `sin(pi * u)^2`, whose value and first derivative both vanish
//! at `u = 0` and `u = 1`. The window therefore glues the pulsaret smoothly
//! into the trailing silence and across the period boundary (`C1`-continuous),
//! so no step is injected and the output is free of discontinuity clicks. The
//! windowed sinusoid places a spectral peak at `formant` whose bandwidth is
//! roughly the reciprocal of the pulsaret duration (`f0 / duty` hertz); the
//! periodic repetition at `f0` quantizes that envelope onto the harmonic
//! series of `f0`, so the formant center and brightness glide independently of
//! pitch. Since `|window| <= 1` and `|carrier| <= 1`, the raw waveform stays
//! within `[-1, 1]` and only `amplitude` scales it; no normalization is
//! required.
//!
//! Re-locking the carrier phase to zero at each period boundary (where the
//! Hann window is already zero) keeps the pulsaret strictly periodic at `f0`
//! and lets the fundamental frequency change mid-period without a carrier-phase
//! jump: both phases stay continuous, so the output is click-free. The formant
//! is clamped to a fraction of Nyquist each sample; at very small `duty` the
//! wide pulsaret skirt can push a little energy toward Nyquist, but the center
//! stays band-limited and the musical range is clean.
//!
//! # Relationship
//!
//! Like [`super::vosim::VosimNode`] and [`super::fof_source::FofSourceNode`]
//! this is a periodically triggered formant source, but the pulsaret differs
//! from both grain shapes: VOSIM lays down a train of `N` squared-sine pulses
//! scaled by a decay factor, and FOF fires an exponentially damped sine grain
//! per period, whereas a pulsar emits exactly one symmetric Hann-windowed
//! sinusoid per period and exposes an explicit silent-gap duty control for its
//! brightness. Unlike [`super::granular_source::GranularSourceNode`], whose
//! grains are drawn asynchronously from a buffer and overlap freely, the
//! pulsaret is synthesized and strictly synchronous, one per fundamental
//! period. Unlike the flat-spectrum band-limited train of
//! [`super::impulse_train::ImpulseTrainNode`], the duty and formant shape a
//! continuous formant region rather than an impulse.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so [`PulsarNode::process`]
//! performs no allocation, no locking, and no panicking: it is a pure
//! per-sample state machine. `formant`, `duty`, and `amplitude` are driven
//! through [`Smoothed`] values so automation (including the signature formant
//! glide and brightness sweep) never produces zipper clicks, while the two
//! continuous phase accumulators make fundamental-frequency changes click-free
//! without smoothing. Reproducible across platforms via [`bevy_math::ops`].
//!
//! # Provenance
//!
//! Implemented from first principles from the public pulsar-synthesis
//! technique (Curtis Roads, "Microsound", 2001): one windowed pulsaret per
//! fundamental period with an adjustable silent duty gap. It contains no code,
//! data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, the Web Audio API, the Synthesis Toolkit, or
//! any other audio engine or toolkit; only the shared mathematical ideas are
//! used. There is no AI or machine learning of any kind.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

const PI: Sample = core::f32::consts::PI;
const TAU: Sample = core::f32::consts::TAU;

/// Minimum fundamental frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Default fundamental frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum fundamental frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum formant frequency in hertz.
pub const MIN_FORMANT_HZ: Sample = 50.0;

/// Default formant frequency in hertz.
pub const DEFAULT_FORMANT_HZ: Sample = 1_200.0;

/// Maximum formant frequency in hertz.
pub const MAX_FORMANT_HZ: Sample = 8_000.0;

/// Minimum pulsaret duty cycle (fraction of the period occupied).
pub const MIN_DUTY: Sample = 0.05;

/// Default pulsaret duty cycle.
pub const DEFAULT_DUTY: Sample = 0.5;

/// Maximum pulsaret duty cycle (the pulsaret fills the whole period).
pub const MAX_DUTY: Sample = 1.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fraction of the sample rate treated as the usable Nyquist ceiling.
pub const NYQUIST_GUARD: Sample = 0.49;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Construction parameters for a [`PulsarNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PulsarParams {
    /// Fundamental frequency in hertz. Clamped to `[MIN, MAX]`.
    pub frequency_hz: Sample,
    /// Formant frequency in hertz. Clamped to `[MIN, MAX]`.
    pub formant_hz: Sample,
    /// Pulsaret duty cycle in `[MIN_DUTY, MAX_DUTY]`.
    pub duty: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for PulsarParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            formant_hz: DEFAULT_FORMANT_HZ,
            duty: DEFAULT_DUTY,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl PulsarParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant_hz: finite_or(self.formant_hz, DEFAULT_FORMANT_HZ)
                .clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            duty: finite_or(self.duty, DEFAULT_DUTY).clamp(MIN_DUTY, MAX_DUTY),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A pulsar synthesis source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::PulsarNode;
///
/// let mut node = PulsarNode::new(110.0, 1_200.0, 0.5, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.duty(), 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct PulsarNode {
    /// Clamped fundamental frequency in hertz (drives the fundamental phase).
    frequency_hz: Sample,
    /// Smoothed formant frequency in hertz.
    formant: Smoothed,
    /// Smoothed pulsaret duty cycle.
    duty: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized fundamental phase in `[0, 1)`.
    phase: Sample,
    /// Normalized carrier phase in `[0, 1)`, re-locked at each period boundary.
    carrier_phase: Sample,
}

impl PulsarNode {
    /// Creates a pulsar source at `frequency_hz` with the given `formant_hz`,
    /// `duty`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; every value is clamped to its
    /// documented domain.
    #[must_use]
    pub fn new(frequency_hz: Sample, formant_hz: Sample, duty: Sample, amplitude: Sample) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant: Smoothed::new(
                finite_or(formant_hz, DEFAULT_FORMANT_HZ).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ),
            duty: Smoothed::new(finite_or(duty, DEFAULT_DUTY).clamp(MIN_DUTY, MAX_DUTY)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
            carrier_phase: 0.0,
        }
    }

    /// Builds a pulsar source from a [`PulsarParams`] bundle.
    #[must_use]
    pub fn from_params(params: PulsarParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.formant_hz, p.duty, p.amplitude)
    }

    /// Sets a new fundamental frequency in hertz (clamped), keeping the phase
    /// continuous so the change is click-free without smoothing.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target formant frequency in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_formant_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.formant.set_target(
            finite_or(hz, self.formant.target()).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ramp,
        );
    }

    /// Sets a new target pulsaret duty cycle, gliding with `ramp`.
    #[inline]
    pub fn set_duty(&mut self, duty: Sample, ramp: Ramp) {
        self.duty.set_target(
            finite_or(duty, self.duty.target()).clamp(MIN_DUTY, MAX_DUTY),
            ramp,
        );
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the fundamental frequency the node is running at (clamped).
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target formant frequency the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn formant_hz(&self) -> Sample {
        self.formant.target()
    }

    /// Returns the target pulsaret duty cycle the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn duty(&self) -> Sample {
        self.duty.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample, advancing both phases and the smoothed
    /// controls. `inv_sr` is the reciprocal of the sample rate and `nyquist` is
    /// the per-sample formant ceiling in hertz.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample, nyquist: Sample) -> Sample {
        let formant = self.formant.next_sample().min(nyquist);
        let duty = self.duty.next_sample().clamp(MIN_DUTY, MAX_DUTY);
        let amp = self.amplitude.next_sample();

        let raw = if self.phase < duty {
            let u = self.phase / duty;
            let w = ops::sin(PI * u);
            let window = w * w;
            let carrier = ops::sin(TAU * self.carrier_phase);
            window * carrier
        } else {
            0.0
        };

        // Advance the carrier first; re-lock it to zero whenever the
        // fundamental period wraps so each pulsaret begins at carrier phase 0.
        self.carrier_phase += formant * inv_sr;
        if self.carrier_phase >= 1.0 {
            self.carrier_phase -= (self.carrier_phase as u32) as Sample;
        }
        self.phase += self.frequency_hz * inv_sr;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
            self.carrier_phase = 0.0;
        }

        raw * amp
    }
}

impl AudioNode for PulsarNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let inv_sr = 1.0 / sr;
        let nyquist = sr * NYQUIST_GUARD;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(inv_sr, nyquist);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.carrier_phase = 0.0;
        self.formant = Smoothed::new(self.formant.target());
        self.duty = Smoothed::new(self.duty.target());
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
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut PulsarNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut PulsarNode,
        sample_rate: u32,
        frames: usize,
        layout: ChannelLayout,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().map(|s| s * s).sum()
    }

    /// Energy of a single Goertzel bin at `freq` over the channel.
    fn goertzel(buf: &AudioBuffer, sample_rate: u32, freq: Sample) -> Sample {
        let samples = buf.channel(0);
        let n = samples.len();
        if n == 0 {
            return 0.0;
        }
        let omega = TAU * freq / sample_rate as Sample;
        let coeff = 2.0 * ops::cos(omega);
        let mut s_prev = 0.0;
        let mut s_prev2 = 0.0;
        for &x in samples {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    }

    /// Energy accumulated across harmonics of `f0` from `lo` to `hi` hertz.
    fn band_energy(buf: &AudioBuffer, f0: Sample, lo: Sample, hi: Sample) -> Sample {
        let mut e = 0.0;
        let mut h = 1u32;
        loop {
            let f = f0 * h as Sample;
            if f >= SR as Sample * NYQUIST_GUARD {
                break;
            }
            if f >= lo && f <= hi {
                e += goertzel(buf, SR, f);
            }
            h += 1;
        }
        e
    }

    #[test]
    fn renders_bounded_finite() {
        for &freq in &[20.0, 110.0, 440.0, 4_000.0] {
            for &formant in &[50.0, 1_200.0, 8_000.0] {
                for &duty in &[0.05, 0.5, 1.0] {
                    let mut node = PulsarNode::new(freq, formant, duty, 0.9);
                    let out = render(&mut node, SR, 8_192);
                    for &s in out.channel(0) {
                        assert!(
                            s.is_finite() && s.abs() <= 1.0 + 1e-3,
                            "freq={freq} formant={formant} duty={duty} s={s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = PulsarNode::new(220.0, 1_500.0, 0.5, 0.0);
        let out = render(&mut node, SR, 4_096);
        assert_eq!(energy(&out), 0.0);
    }

    #[test]
    fn not_silent() {
        let mut node = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let out = render(&mut node, SR, 4_096);
        assert!(energy(&out) > 0.0);
    }

    #[test]
    fn smaller_duty_has_more_silence() {
        let mut narrow = PulsarNode::new(100.0, 1_000.0, 0.1, 0.8);
        let mut wide = PulsarNode::new(100.0, 1_000.0, 0.9, 0.8);
        let narrow_out = render(&mut narrow, SR, 48_000);
        let wide_out = render(&mut wide, SR, 48_000);
        let narrow_zeros = narrow_out.channel(0).iter().filter(|&&s| s == 0.0).count();
        let wide_zeros = wide_out.channel(0).iter().filter(|&&s| s == 0.0).count();
        assert!(
            narrow_zeros > wide_zeros,
            "narrow duty should leave more silent samples: narrow={narrow_zeros} wide={wide_zeros}"
        );
    }

    #[test]
    fn energy_concentrates_on_harmonics() {
        let mut node = PulsarNode::new(100.0, 1_000.0, 0.5, 0.8);
        let out = render(&mut node, SR, 48_000);
        let harmonic: Sample = [100.0, 200.0, 300.0, 1_000.0, 2_000.0]
            .iter()
            .map(|&f| goertzel(&out, SR, f))
            .sum();
        let between: Sample = [150.0, 250.0, 350.0, 1_050.0]
            .iter()
            .map(|&f| goertzel(&out, SR, f))
            .sum();
        assert!(
            harmonic > between * 50.0,
            "spectrum should sit on harmonics of f0: harmonic={harmonic} between={between}"
        );
    }

    #[test]
    fn higher_formant_increases_brightness() {
        let mut low = PulsarNode::new(100.0, 600.0, 0.5, 0.8);
        let mut high = PulsarNode::new(100.0, 3_000.0, 0.5, 0.8);
        let low_out = render(&mut low, SR, 48_000);
        let high_out = render(&mut high, SR, 48_000);
        let low_band = band_energy(&low_out, 100.0, 2_600.0, 3_400.0);
        let high_band = band_energy(&high_out, 100.0, 2_600.0, 3_400.0);
        assert!(
            high_band > low_band * 4.0,
            "a higher formant should brighten the 3 kHz band: low={low_band} high={high_band}"
        );
    }

    #[test]
    fn formant_centers_spectral_peak() {
        // The windowed carrier concentrates energy near the formant frequency.
        let mut node = PulsarNode::new(100.0, 2_000.0, 0.5, 0.8);
        let out = render(&mut node, SR, 48_000);
        let near = band_energy(&out, 100.0, 1_700.0, 2_300.0);
        let far = band_energy(&out, 100.0, 5_000.0, 6_000.0);
        assert!(
            near > far * 4.0,
            "energy should peak near the formant: near={near} far={far}"
        );
    }

    #[test]
    fn fundamental_frequency_change_is_click_free() {
        // With both phases continuous, an abrupt fundamental change must not
        // inject a step larger than the carrier's natural per-sample slope.
        let mut node = PulsarNode::new(110.0, 200.0, 0.9, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.set_frequency_hz(130.0);
        let second = render(&mut node, SR, 4_096);
        let boundary = (second.channel(0)[0] - first.channel(0)[first.channel(0).len() - 1]).abs();
        assert!(
            boundary < 0.1,
            "frequency change should be click-free: boundary step={boundary}"
        );
    }

    #[test]
    fn formant_sweep_is_click_free() {
        let mut node = PulsarNode::new(110.0, 300.0, 0.9, 0.8);
        node.set_formant_hz(700.0, Ramp::Linear { samples: 4_096 });
        let out = render(&mut node, SR, 4_096);
        let ch = out.channel(0);
        let mut max_step = 0.0;
        for w in ch.windows(2) {
            let d = (w[1] - w[0]).abs();
            if d > max_step {
                max_step = d;
            }
        }
        assert!(max_step < 0.1, "formant sweep should be click-free: {max_step}");
    }

    #[test]
    fn duty_change_takes_effect() {
        let mut node = PulsarNode::new(100.0, 1_000.0, 0.1, 0.8);
        let narrow = render(&mut node, SR, 48_000);
        node.set_duty(0.9, Ramp::Immediate);
        let wide = render(&mut node, SR, 48_000);
        let narrow_zeros = narrow.channel(0).iter().filter(|&&s| s == 0.0).count();
        let wide_zeros = wide.channel(0).iter().filter(|&&s| s == 0.0).count();
        assert!(
            narrow_zeros > wide_zeros,
            "raising duty should reduce the silent gap"
        );
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = PulsarNode::new(123.0, 1_300.0, 0.4, 0.7);
        let mut b = PulsarNode::new(123.0, 1_300.0, 0.4, 0.7);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = PulsarNode::new(123.0, 1_300.0, 0.4, 0.7);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = PulsarNode::new(220.0, 1_500.0, 0.5, 0.25);
        let mut loud = PulsarNode::new(220.0, 1_500.0, 0.5, 0.5);
        let e_quiet = energy(&render(&mut quiet, SR, 8_192));
        let e_loud = energy(&render(&mut loud, SR, 8_192));
        assert!(e_quiet > 0.0);
        let ratio = e_loud / e_quiet;
        assert!(
            (ratio - 4.0).abs() < 1e-2,
            "energy should scale quadratically: ratio={ratio}"
        );
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let mono_out = render(&mut mono, SR, 2_048);
        let mut stereo = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);
        let mut quad = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let quad_out = render_layout(&mut quad, SR, 2_048, ChannelLayout::Quad);
        assert_eq!(mono_out.channel(0), stereo_out.channel(0));
        assert_eq!(stereo_out.channel(0), stereo_out.channel(1));
        assert_eq!(mono_out.channel(0), quad_out.channel(0));
        assert_eq!(quad_out.channel(0), quad_out.channel(3));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut idle = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let mut empty = AudioBuffer::new(ChannelLayout::Mono, 64);
        empty.set_active_frames(0);
        {
            let inputs: [AudioBuffer; 0] = [];
            let mut outputs = [empty];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            idle.process(&ctx(SR, 0), &mut io);
        }
        let after_idle = render(&mut idle, SR, 2_048);
        let mut fresh = PulsarNode::new(220.0, 1_500.0, 0.5, 0.8);
        let fresh_out = render(&mut fresh, SR, 2_048);
        assert_eq!(after_idle.channel(0), fresh_out.channel(0));
    }

    #[test]
    fn latency_is_zero() {
        let node = PulsarNode::new(110.0, 1_200.0, 0.5, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = PulsarNode::new(123.0, 1_300.0, 0.4, 0.6);
        assert_eq!(node.frequency_hz(), 123.0);
        assert_eq!(node.formant_hz(), 1_300.0);
        assert_eq!(node.duty(), 0.4);
        assert_eq!(node.amplitude(), 0.6);
    }

    #[test]
    fn default_params_in_domain() {
        let p = PulsarParams::default();
        assert_eq!(p.frequency_hz, DEFAULT_FREQUENCY_HZ);
        assert_eq!(p.formant_hz, DEFAULT_FORMANT_HZ);
        assert_eq!(p.duty, DEFAULT_DUTY);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
        let s = p.sanitised();
        assert!(s.frequency_hz >= MIN_FREQUENCY_HZ && s.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(s.duty >= MIN_DUTY && s.duty <= MAX_DUTY);
    }

    #[test]
    fn from_params_matches_new() {
        let params = PulsarParams {
            frequency_hz: 180.0,
            formant_hz: 2_200.0,
            duty: 0.3,
            amplitude: 0.7,
        };
        let mut from = PulsarNode::from_params(params);
        let mut direct = PulsarNode::new(180.0, 2_200.0, 0.3, 0.7);
        let from_out = render(&mut from, SR, 2_048);
        let direct_out = render(&mut direct, SR, 2_048);
        assert_eq!(from_out.channel(0), direct_out.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = PulsarNode::new(1.0e9, 1.0e9, 5.0, 0.8);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        assert_eq!(node.duty(), MAX_DUTY);
        let node = PulsarNode::new(0.0, 0.0, 0.0, 0.8);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), MIN_FORMANT_HZ);
        assert_eq!(node.duty(), MIN_DUTY);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = PulsarNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN, Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), DEFAULT_FORMANT_HZ);
        assert_eq!(node.duty(), DEFAULT_DUTY);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = PulsarNode::new(110.0, 1_200.0, 0.5, 0.8);
        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 110.0);
        node.set_frequency_hz(1.0e9);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        node.set_formant_hz(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.formant_hz(), 1_200.0);
        node.set_formant_hz(1.0e9, Ramp::Immediate);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        node.set_duty(5.0, Ramp::Immediate);
        assert_eq!(node.duty(), MAX_DUTY);
        node.set_duty(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.duty(), MAX_DUTY);
        node.set_amplitude(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn frequency_change_shifts_pitch() {
        let mut low = PulsarNode::new(100.0, 1_000.0, 0.5, 0.8);
        let low_out = render(&mut low, SR, 48_000);
        let mut high = PulsarNode::new(200.0, 1_000.0, 0.5, 0.8);
        let high_out = render(&mut high, SR, 48_000);
        // The 100 Hz bin is a harmonic for f0=100 but not for f0=200.
        assert!(goertzel(&low_out, SR, 100.0) > goertzel(&high_out, SR, 100.0) * 10.0);
        // The 200 Hz bin is a harmonic for both; energy stays present.
        assert!(goertzel(&high_out, SR, 200.0) > 0.0);
    }

    #[test]
    fn carrier_relocks_each_period() {
        // With duty < 1 the silent gap must contain exact zeros every period.
        let mut node = PulsarNode::new(100.0, 1_000.0, 0.5, 0.8);
        let out = render(&mut node, SR, 48_000);
        let zeros = out.channel(0).iter().filter(|&&s| s == 0.0).count();
        assert!(zeros > 20_000, "half-duty should silence about half the stream: zeros={zeros}");
    }
}
