//! VOSIM (voice-simulation) formant source node.
//!
//! [`VosimNode`] synthesizes a voiced, formant-peaked tone by emitting, once
//! per fundamental period, a short train of `N` squared-sine pulses whose
//! successive amplitudes decay by a factor `decay`, followed by silence for the
//! rest of the period. The pulse repetition rate sets a formant peak, the
//! fundamental sets the pitch, and the pulse count and decay shape the formant
//! bandwidth, so a single voice can glide its formant independently of its
//! pitch -- the classic VOSIM vowel-like timbre.
//!
//! # Model
//!
//! One fundamental period has length `T0 = 1 / f0`. Inside it the node lays
//! down `N` identical pulses of width `T = 1 / formant`, pulse `k` scaled by
//! `decay^k`, then holds zero until the period wraps:
//!
//! ```text
//!   position = phase * (formant / f0)        (phase in [0, 1))
//!   k        = floor(position)               (pulse index)
//!   u        = position - k                  (phase within the pulse)
//!   x        = decay^k * sin(pi * u)^2       if k < N
//!            = 0                             otherwise
//! ```
//!
//! Each pulse is a squared sine `sin(pi * u)^2`, a single raised hump whose
//! value and first derivative both vanish at `u = 0` and `u = 1`, so adjacent
//! pulses -- and the join into the trailing silence -- are `C1`-continuous and
//! the period boundary carries no step. The squared-sine shape concentrates
//! spectral energy around the pulse rate `formant`, placing an adjustable
//! formant peak over the harmonic series of `f0`.
//!
//! The effective pulse count is `min(N, floor(formant / f0))` so the pulses
//! always fit inside one period `N * T <= T0`; a `formant` below `f0` fits no
//! complete pulse and the voice falls silent. The pulse peak is `decay^0 = 1`
//! and `sin(pi * 0.5)^2 = 1`, so the raw waveform never exceeds `1` and only
//! the output `amplitude` scales it; no normalization is required.
//!
//! # Relationship
//!
//! Like [`super::fof_source::FofSourceNode`] this is a periodically triggered
//! formant source, but the two shape their grains differently: FOF fires a
//! single exponentially damped sine grain (with a raised-cosine attack skirt)
//! per period per formant, whereas VOSIM emits a train of `N` fixed-width
//! squared-sine pulses geometrically scaled by `decay`. Where
//! [`super::glottal_pulse::GlottalPulseNode`] models the physical glottal flow
//! derivative and [`super::impulse_train::ImpulseTrainNode`] emits a
//! flat-spectrum band-limited impulse train, this node builds its spectrum
//! purely from the pulse width, count, and decay.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so [`VosimNode::process`]
//! performs no allocation, no locking, and no panicking: it is a pure
//! per-sample state machine. `formant`, `decay`, and `amplitude` are driven
//! through [`Smoothed`] values so automation (including the signature formant
//! glide) never produces zipper clicks, while the single continuous phase
//! accumulator makes frequency changes click-free without smoothing. The pulse
//! count is a structural control applied immediately. Reproducible across
//! platforms via [`bevy_math::ops`].
//!
//! # Provenance
//!
//! Implemented from first principles from the public VOSIM
//! (voice-simulation) technique for formant synthesis (Kaegi and Tempelaars,
//! 1978): a periodic train of amplitude-decaying squared-sine pulses. It
//! contains no code, data, or derivative of Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web Audio API, the
//! Synthesis Toolkit, or any other audio engine or toolkit; only the shared
//! mathematical ideas are used. There is no AI or machine learning of any kind.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

const PI: Sample = core::f32::consts::PI;

/// Minimum fundamental frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Default fundamental frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum fundamental frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum formant (pulse-repetition) frequency in hertz.
pub const MIN_FORMANT_HZ: Sample = 50.0;

/// Default formant (pulse-repetition) frequency in hertz.
pub const DEFAULT_FORMANT_HZ: Sample = 1_400.0;

/// Maximum formant (pulse-repetition) frequency in hertz.
pub const MAX_FORMANT_HZ: Sample = 8_000.0;

/// Minimum number of pulses emitted per fundamental period.
pub const MIN_PULSE_COUNT: u32 = 1;

/// Default number of pulses emitted per fundamental period.
pub const DEFAULT_PULSE_COUNT: u32 = 3;

/// Maximum number of pulses emitted per fundamental period.
pub const MAX_PULSE_COUNT: u32 = 8;

/// Default inter-pulse amplitude decay factor in `[0, 1]`.
pub const DEFAULT_DECAY: Sample = 0.6;

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

/// Construction parameters for a [`VosimNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VosimParams {
    /// Fundamental frequency in hertz. Clamped to `[MIN, MAX]`.
    pub frequency_hz: Sample,
    /// Formant (pulse-repetition) frequency in hertz. Clamped to `[MIN, MAX]`.
    pub formant_hz: Sample,
    /// Number of pulses per period. Clamped to `[MIN, MAX]`.
    pub pulse_count: u32,
    /// Inter-pulse amplitude decay factor in `[0, 1]`.
    pub decay: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for VosimParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            formant_hz: DEFAULT_FORMANT_HZ,
            pulse_count: DEFAULT_PULSE_COUNT,
            decay: DEFAULT_DECAY,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl VosimParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant_hz: finite_or(self.formant_hz, DEFAULT_FORMANT_HZ)
                .clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            pulse_count: self.pulse_count.clamp(MIN_PULSE_COUNT, MAX_PULSE_COUNT),
            decay: finite_or(self.decay, DEFAULT_DECAY).clamp(0.0, 1.0),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A VOSIM formant source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::VosimNode;
///
/// let mut node = VosimNode::new(110.0, 1400.0, 3, 0.6, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.pulse_count(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct VosimNode {
    /// Fundamental frequency in hertz, clamped to `[MIN, MAX]`. Stored as a
    /// plain scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency_hz: Sample,
    /// Smoothed formant (pulse-repetition) frequency in hertz.
    formant_hz: Smoothed,
    /// Number of pulses per period, clamped to `[MIN, MAX]`.
    pulse_count: u32,
    /// Smoothed inter-pulse amplitude decay factor in `[0, 1]`.
    decay: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl VosimNode {
    /// Creates a VOSIM source at `frequency_hz` with the given `formant_hz`,
    /// `pulse_count`, `decay`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; frequency and formant are
    /// clamped to their documented ranges, `pulse_count` to
    /// `[MIN_PULSE_COUNT, MAX_PULSE_COUNT]`, and `decay` to `[0, 1]`.
    #[must_use]
    pub fn new(
        frequency_hz: Sample,
        formant_hz: Sample,
        pulse_count: u32,
        decay: Sample,
        amplitude: Sample,
    ) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant_hz: Smoothed::new(
                finite_or(formant_hz, DEFAULT_FORMANT_HZ).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ),
            pulse_count: pulse_count.clamp(MIN_PULSE_COUNT, MAX_PULSE_COUNT),
            decay: Smoothed::new(finite_or(decay, DEFAULT_DECAY).clamp(0.0, 1.0)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds a VOSIM source from a [`VosimParams`] bundle.
    #[must_use]
    pub fn from_params(params: VosimParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.formant_hz, p.pulse_count, p.decay, p.amplitude)
    }

    /// Sets the fundamental frequency in hertz (clamped to `[MIN, MAX]`).
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target formant frequency in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_formant_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.formant_hz.set_target(
            finite_or(hz, self.formant_hz.target()).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ramp,
        );
    }

    /// Sets the number of pulses per period (clamped to `[MIN, MAX]`).
    ///
    /// Applied immediately; it is a structural control, not a smoothed one.
    #[inline]
    pub fn set_pulse_count(&mut self, count: u32) {
        self.pulse_count = count.clamp(MIN_PULSE_COUNT, MAX_PULSE_COUNT);
    }

    /// Sets a new target inter-pulse decay in `[0, 1]`, gliding with `ramp`.
    #[inline]
    pub fn set_decay(&mut self, decay: Sample, ramp: Ramp) {
        self.decay
            .set_target(finite_or(decay, self.decay.target()).clamp(0.0, 1.0), ramp);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target formant frequency the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn formant_hz(&self) -> Sample {
        self.formant_hz.target()
    }

    /// Returns the number of pulses emitted per period.
    #[inline]
    #[must_use]
    pub fn pulse_count(&self) -> u32 {
        self.pulse_count
    }

    /// Returns the target inter-pulse decay the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn decay(&self) -> Sample {
        self.decay.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample at normalized phase increment `dt`, advancing
    /// the phase accumulator and the smoothed controls.
    ///
    /// `f0` is the block-constant fundamental (already clamped to the guard
    /// ceiling) and `guard_hz` the usable Nyquist ceiling the per-sample
    /// formant is clamped to.
    #[inline]
    fn render_sample(&mut self, dt: Sample, f0: Sample, guard_hz: Sample) -> Sample {
        let formant = self.formant_hz.next_sample().min(guard_hz);
        let decay = self.decay.next_sample();
        let amp = self.amplitude.next_sample();

        let ratio = formant / f0;
        let position = self.phase * ratio;
        let k = ops::floor(position);
        // Only pulses that fully fit inside one period are emitted.
        let fit = ops::floor(ratio).min(self.pulse_count as Sample);

        let value = if k < fit {
            let u = position - k;
            let s = ops::sin(PI * u);
            // decay^0 == 1 even when decay == 0, avoiding a 0^0 NaN from powf.
            let env = if k < 0.5 { 1.0 } else { ops::powf(decay, k) };
            env * s * s
        } else {
            0.0
        };

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value * amp
    }
}

impl AudioNode for VosimNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let guard_hz = sr * NYQUIST_GUARD;
        let f0 = self.frequency_hz.min(guard_hz);
        let dt = f0 / sr;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(dt, f0, guard_hz);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.formant_hz = Smoothed::new(self.formant_hz.target());
        self.decay = Smoothed::new(self.decay.target());
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

    fn render(node: &mut VosimNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut VosimNode,
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
        let omega = core::f32::consts::TAU * freq / sample_rate as Sample;
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

    /// Energy accumulated in a band of harmonics of `f0` from `lo` to `hi` hertz.
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
        for &formant in &[300.0, 1_400.0, 4_000.0, 8_000.0] {
            for &decay in &[0.0, 0.5, 1.0] {
                let mut node = VosimNode::new(120.0, formant, 4, decay, 1.0);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "formant={formant} decay={decay} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn not_silent() {
        let mut node = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.8);
        let out = render(&mut node, SR, 4_096);
        assert!(energy(&out) > 1e-3);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.0);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn fundamental_locks_to_frequency() {
        let mut node = VosimNode::new(150.0, 3_000.0, 3, 0.6, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 150.0);
        let off = goertzel(&out, SR, 150.0 * 1.5);
        assert!(fund > off * 4.0, "fund={fund} off={off}");
    }

    #[test]
    fn formant_shifts_spectral_energy_upward() {
        // Raising the pulse rate moves the spectral peak upward, so the share
        // of energy in the upper harmonics must rise.
        let f0 = 120.0;
        let mut low = VosimNode::new(f0, 1_200.0, 4, 0.7, 0.8);
        let mut high = VosimNode::new(f0, 4_800.0, 4, 0.7, 0.8);
        let low_out = render(&mut low, SR, 16_384);
        let high_out = render(&mut high, SR, 16_384);
        let upper_low = band_energy(&low_out, f0, 3_000.0, 20_000.0);
        let upper_high = band_energy(&high_out, f0, 3_000.0, 20_000.0);
        assert!(upper_high > upper_low * 2.0, "low={upper_low} high={upper_high}");
    }

    #[test]
    fn decay_zero_leaves_single_pulse() {
        // With decay == 0 only pulse k == 0 survives, so each period has exactly
        // one hump; the rest of the period is silence.
        let f0 = 120.0;
        let formant = 2_400.0;
        let mut node = VosimNode::new(f0, formant, 4, 0.0, 1.0);
        let out = render(&mut node, SR, 400); // one full period at 120 Hz
        // Pulse width in samples: T = 1 / formant -> sr / formant samples.
        let pulse_len = (SR as Sample / formant) as usize; // 20 samples
        let samples = out.channel(0);
        // Beyond the first pulse the output must be silent for this period.
        for (n, &s) in samples.iter().enumerate().skip(pulse_len + 2) {
            assert!(s.abs() < 1e-4, "n={n} s={s} should be silent after one pulse");
        }
        // The first pulse carries energy.
        let first: Sample = samples[..pulse_len].iter().map(|s| s * s).sum();
        assert!(first > 1e-3, "first pulse energy={first}");
    }

    #[test]
    fn more_pulses_fill_more_of_the_period() {
        // More pulses occupy a longer active fraction of each period, so the
        // total energy (all pulses have the same shape) grows with the count.
        let f0 = 120.0;
        let formant = 3_600.0;
        let mut few = VosimNode::new(f0, formant, 1, 1.0, 0.8);
        let mut many = VosimNode::new(f0, formant, 6, 1.0, 0.8);
        let few_e = energy(&render(&mut few, SR, 4_000));
        let many_e = energy(&render(&mut many, SR, 4_000));
        assert!(many_e > few_e * 2.0, "few={few_e} many={many_e}");
    }

    #[test]
    fn is_periodic_at_fundamental() {
        // With an integer number of samples per period the waveform repeats.
        let f0 = 120.0; // 48000 / 120 = 400 samples per period
        let period = 400usize;
        let mut node = VosimNode::new(f0, 2_400.0, 3, 0.7, 0.8);
        let out = render(&mut node, SR, period * 4);
        let s = out.channel(0);
        for n in 0..period {
            assert!(
                (s[n] - s[n + period]).abs() < 2e-3,
                "n={n} a={} b={}",
                s[n],
                s[n + period]
            );
        }
    }

    #[test]
    fn formant_sweep_is_click_free() {
        // Keep the pulse count below floor(formant / f0) across the whole
        // sweep so the active pulse count never toggles, and keep the formant
        // moderate so each squared-sine pulse stays several samples wide; the
        // only remaining sample-to-sample change is the honest waveform slope.
        let mut node = VosimNode::new(55.0, 400.0, 3, 0.7, 0.8);
        node.set_formant_hz(1_500.0, Ramp::Linear { samples: 8_192 });
        let out = render(&mut node, SR, 8_192);
        let s = out.channel(0);
        for n in 1..s.len() {
            assert!((s[n] - s[n - 1]).abs() < 0.1, "n={n} jump={}", s[n] - s[n - 1]);
        }
    }

    #[test]
    fn deterministic() {
        let mut a = VosimNode::new(130.0, 2_200.0, 3, 0.6, 0.8);
        let mut b = VosimNode::new(130.0, 2_200.0, 3, 0.6, 0.8);
        let oa = render(&mut a, SR, 2_048);
        let ob = render(&mut b, SR, 2_048);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_output() {
        let mut node = VosimNode::new(130.0, 2_200.0, 3, 0.6, 0.8);
        let first = render(&mut node, SR, 1_024).channel(0).to_vec();
        node.reset();
        let second = render(&mut node, SR, 1_024).channel(0).to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = VosimNode::new(130.0, 2_200.0, 3, 0.6, 0.8);
        let _ = render(&mut node, SR, 777);
        node.reset();
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.25);
        let mut loud = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.5);
        let eq = energy(&render(&mut quiet, SR, 4_096));
        let el = energy(&render(&mut loud, SR, 4_096));
        assert!((el / eq - 4.0).abs() < 0.1, "ratio={}", el / eq);
    }

    #[test]
    fn mono_core_copied_to_stereo_and_quad() {
        for layout in [ChannelLayout::Stereo, ChannelLayout::Quad] {
            let mut node = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.8);
            let out = render_layout(&mut node, SR, 512, layout);
            let ch0 = out.channel(0);
            for ch in 1..out.channels() {
                assert_eq!(out.channel(ch), ch0, "channel {ch} must mirror mono core");
            }
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn latency_is_zero() {
        let node = VosimNode::new(120.0, 2_400.0, 3, 0.7, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = VosimNode::new(123.0, 1_800.0, 5, 0.4, 0.7);
        assert_eq!(node.frequency_hz(), 123.0);
        assert_eq!(node.formant_hz(), 1_800.0);
        assert_eq!(node.pulse_count(), 5);
        assert_eq!(node.decay(), 0.4);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = VosimParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.formant_hz >= MIN_FORMANT_HZ && p.formant_hz <= MAX_FORMANT_HZ);
        assert!(p.pulse_count >= MIN_PULSE_COUNT && p.pulse_count <= MAX_PULSE_COUNT);
        assert!(p.decay >= 0.0 && p.decay <= 1.0);
        assert_eq!(p.sanitised().frequency_hz, p.frequency_hz);
    }

    #[test]
    fn from_params_matches_new() {
        let p = VosimParams {
            frequency_hz: 140.0,
            formant_hz: 2_100.0,
            pulse_count: 4,
            decay: 0.5,
            amplitude: 0.7,
        };
        let mut a = VosimNode::from_params(p);
        let mut b = VosimNode::new(140.0, 2_100.0, 4, 0.5, 0.7);
        assert_eq!(render(&mut a, SR, 512).channel(0), render(&mut b, SR, 512).channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = VosimNode::new(5.0, 20_000.0, 99, 2.0, 0.8);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        assert_eq!(node.pulse_count(), MAX_PULSE_COUNT);
        assert_eq!(node.decay(), 1.0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = VosimNode::new(
            Sample::NAN,
            Sample::INFINITY,
            3,
            Sample::NAN,
            Sample::NAN,
        );
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), DEFAULT_FORMANT_HZ.clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ));
        assert_eq!(node.decay(), DEFAULT_DECAY);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = VosimNode::new(120.0, 2_400.0, 3, 0.6, 0.8);
        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 120.0);
        node.set_formant_hz(100_000.0, Ramp::Immediate);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        node.set_pulse_count(0);
        assert_eq!(node.pulse_count(), MIN_PULSE_COUNT);
        node.set_decay(-1.0, Ramp::Immediate);
        assert_eq!(node.decay(), 0.0);
        node.set_amplitude(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn nyquist_guard_keeps_super_nyquist_bounded() {
        let mut node = VosimNode::new(3_900.0, 8_000.0, 6, 0.8, 1.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn frequency_change_shifts_fundamental() {
        let mut low = VosimNode::new(110.0, 2_200.0, 3, 0.6, 0.8);
        let mut high = VosimNode::new(220.0, 2_200.0, 3, 0.6, 0.8);
        let low_out = render(&mut low, SR, 8_192);
        let high_out = render(&mut high, SR, 8_192);
        assert!(goertzel(&low_out, SR, 110.0) > goertzel(&high_out, SR, 110.0) * 4.0);
        // 330 Hz is not a harmonic of 220 Hz, so the high node must peak at its
        // own fundamental rather than that off-harmonic bin.
        assert!(goertzel(&high_out, SR, 220.0) > goertzel(&high_out, SR, 330.0) * 4.0);
    }

    #[test]
    fn pulse_count_change_takes_effect() {
        let mut node = VosimNode::new(120.0, 3_600.0, 1, 1.0, 0.8);
        let one = energy(&render(&mut node, SR, 4_000));
        node.set_pulse_count(6);
        node.reset();
        let six = energy(&render(&mut node, SR, 4_000));
        assert!(six > one * 2.0, "one={one} six={six}");
    }
}
