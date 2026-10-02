//! Phase-aligned formant (PAF) source node.
//!
//! [`PafNode`] synthesizes a single formant centered on an arbitrary frequency
//! while keeping every partial locked to the harmonic series of a separate
//! fundamental. It multiplies a phase-aligned carrier -- a crossfade between
//! the two harmonics of `f0` that bracket the formant center -- by a smooth,
//! strictly periodic bell window whose width sets the formant bandwidth. A
//! single voice can therefore glide its formant center and bandwidth
//! independently of its pitch, the signature of phase-aligned formant
//! synthesis.
//!
//! # Model
//!
//! A normalized phase `p` runs in `[0, 1)` at the fundamental `f0`. Let
//! `r = formant / f0`, `k = floor(r)`, and `frac = r - k`. The carrier
//! crossfades the two harmonics that bracket the formant center so the
//! spectral peak slides continuously as the center sweeps:
//!
//! ```text
//!   carrier = (1 - frac) * cos(2*pi * k * p) + frac * cos(2*pi * (k + 1) * p)
//! ```
//!
//! Both cosines share the fundamental phase `p`, so every partial of the
//! carrier is an exact harmonic of `f0` (the "phase-aligned" property) and the
//! crossfade is continuous across a harmonic boundary (as `frac -> 1` at `k`
//! the carrier equals the `k + 1`, `frac = 0` carrier). The carrier is scaled
//! by a von Mises bell window, a strictly periodic Gaussian-like hump of unit
//! peak:
//!
//! ```text
//!   kappa  = (bandwidth / f0)^2
//!   window = exp(kappa * (cos(2*pi * p) - 1))
//!   output = amplitude * window * carrier
//! ```
//!
//! Near `p = 0` the window behaves like `exp(-kappa * (2*pi*p)^2 / 2)`, a
//! Gaussian whose time width shrinks as `kappa` grows; a narrower time pulse
//! means a wider formant, so the formant's spectral standard deviation is
//! approximately `f0 * sqrt(kappa) = bandwidth` hertz. Both the window and the
//! carrier are `C`-infinity and genuinely periodic in `p`, so the waveform is
//! smooth and the period boundary carries no step. The window never exceeds
//! `1` and the carrier is a convex combination of two unit cosines, so the raw
//! waveform stays within `[-1, 1]`; only `amplitude` scales it and no
//! normalization is required.
//!
//! Because the formant has a Gaussian spectral envelope rather than a hard
//! band limit, a little energy always sits beyond the formant center; it rolls
//! off quickly and stays clean across the musical range, and the carrier
//! center is clamped below the Nyquist guard.
//!
//! # Relationship
//!
//! Like [`super::fof_source::FofSourceNode`] and
//! [`super::vosim::VosimNode`] this is a formant source that decouples formant
//! center from pitch, but the mechanism differs: FOF fires an exponentially
//! damped sine grain per period, VOSIM emits a train of squared-sine pulses,
//! whereas PAF multiplies a harmonic-locked crossfaded carrier by a periodic
//! bell window. Where [`super::glottal_pulse::GlottalPulseNode`] models the
//! physical glottal flow derivative and
//! [`super::impulse_train::ImpulseTrainNode`] emits a flat-spectrum
//! band-limited impulse train, this node shapes one isolated Gaussian formant
//! over the harmonic series.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so [`PafNode::process`] performs
//! no allocation, no locking, and no panicking: it is a pure per-sample state
//! machine. `formant`, `bandwidth`, and `amplitude` are driven through
//! [`Smoothed`] values so automation (including the signature formant glide)
//! never produces zipper clicks, while the single continuous phase accumulator
//! makes frequency changes click-free without smoothing. Reproducible across
//! platforms via [`bevy_math::ops`].
//!
//! # Provenance
//!
//! Implemented from first principles from the public phase-aligned formant
//! (PAF) synthesis technique (Miller Puckette, 1995): a harmonic-locked
//! crossfaded carrier shaped by a periodic bell window. It contains no code,
//! data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, the Web Audio API, the Synthesis Toolkit, or
//! any other audio engine or toolkit; only the shared mathematical ideas are
//! used. There is no AI or machine learning of any kind.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

const TAU: Sample = core::f32::consts::TAU;

/// Minimum fundamental frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Default fundamental frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum fundamental frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum formant center frequency in hertz.
pub const MIN_FORMANT_HZ: Sample = 50.0;

/// Default formant center frequency in hertz.
pub const DEFAULT_FORMANT_HZ: Sample = 1_200.0;

/// Maximum formant center frequency in hertz.
pub const MAX_FORMANT_HZ: Sample = 8_000.0;

/// Minimum formant bandwidth in hertz.
pub const MIN_BANDWIDTH_HZ: Sample = 20.0;

/// Default formant bandwidth in hertz.
pub const DEFAULT_BANDWIDTH_HZ: Sample = 200.0;

/// Maximum formant bandwidth in hertz.
pub const MAX_BANDWIDTH_HZ: Sample = 4_000.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Upper clamp on the window concentration to keep the time pulse from
/// collapsing toward a delta (which would broaden the spectrum without bound).
pub const MAX_CONCENTRATION: Sample = 2_000.0;

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

/// Non-negative fractional part, avoiding `Sample::fract` which is unavailable
/// under `no_std`.
#[inline]
fn fract_nonneg(x: Sample) -> Sample {
    x - ops::floor(x)
}

/// Construction parameters for a [`PafNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PafParams {
    /// Fundamental frequency in hertz. Clamped to `[MIN, MAX]`.
    pub frequency_hz: Sample,
    /// Formant center frequency in hertz. Clamped to `[MIN, MAX]`.
    pub formant_hz: Sample,
    /// Formant bandwidth in hertz. Clamped to `[MIN, MAX]`.
    pub bandwidth_hz: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for PafParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            formant_hz: DEFAULT_FORMANT_HZ,
            bandwidth_hz: DEFAULT_BANDWIDTH_HZ,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl PafParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant_hz: finite_or(self.formant_hz, DEFAULT_FORMANT_HZ)
                .clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            bandwidth_hz: finite_or(self.bandwidth_hz, DEFAULT_BANDWIDTH_HZ)
                .clamp(MIN_BANDWIDTH_HZ, MAX_BANDWIDTH_HZ),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A phase-aligned formant source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::PafNode;
///
/// let mut node = PafNode::new(110.0, 1200.0, 200.0, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.formant_hz(), 1200.0);
/// ```
#[derive(Debug, Clone)]
pub struct PafNode {
    /// Fundamental frequency in hertz, clamped to `[MIN, MAX]`. Stored as a
    /// plain scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency_hz: Sample,
    /// Smoothed formant center frequency in hertz.
    formant_hz: Smoothed,
    /// Smoothed formant bandwidth in hertz.
    bandwidth_hz: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl PafNode {
    /// Creates a PAF source at `frequency_hz` with the given `formant_hz`,
    /// `bandwidth_hz`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; every frequency is clamped to
    /// its documented range.
    #[must_use]
    pub fn new(
        frequency_hz: Sample,
        formant_hz: Sample,
        bandwidth_hz: Sample,
        amplitude: Sample,
    ) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            formant_hz: Smoothed::new(
                finite_or(formant_hz, DEFAULT_FORMANT_HZ).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ),
            bandwidth_hz: Smoothed::new(
                finite_or(bandwidth_hz, DEFAULT_BANDWIDTH_HZ)
                    .clamp(MIN_BANDWIDTH_HZ, MAX_BANDWIDTH_HZ),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds a PAF source from a [`PafParams`] bundle.
    #[must_use]
    pub fn from_params(params: PafParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.formant_hz, p.bandwidth_hz, p.amplitude)
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

    /// Sets a new target formant center frequency in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_formant_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.formant_hz.set_target(
            finite_or(hz, self.formant_hz.target()).clamp(MIN_FORMANT_HZ, MAX_FORMANT_HZ),
            ramp,
        );
    }

    /// Sets a new target formant bandwidth in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_bandwidth_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.bandwidth_hz.set_target(
            finite_or(hz, self.bandwidth_hz.target()).clamp(MIN_BANDWIDTH_HZ, MAX_BANDWIDTH_HZ),
            ramp,
        );
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

    /// Returns the target formant center frequency the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn formant_hz(&self) -> Sample {
        self.formant_hz.target()
    }

    /// Returns the target formant bandwidth the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn bandwidth_hz(&self) -> Sample {
        self.bandwidth_hz.target()
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
    /// formant center is clamped to.
    #[inline]
    fn render_sample(&mut self, dt: Sample, f0: Sample, guard_hz: Sample) -> Sample {
        let formant = self.formant_hz.next_sample().min(guard_hz);
        let bandwidth = self.bandwidth_hz.next_sample();
        let amp = self.amplitude.next_sample();

        let ratio = formant / f0;
        let k = ops::floor(ratio);
        let frac = ratio - k;

        // Phase-aligned carrier: both harmonics read the same fundamental
        // phase. Fold each harmonic phase into [0, 1) before scaling by TAU so
        // the cosine argument stays small and precise for high harmonics.
        let lower = ops::cos(TAU * fract_nonneg(k * self.phase));
        let upper = ops::cos(TAU * fract_nonneg((k + 1.0) * self.phase));
        let carrier = (1.0 - frac) * lower + frac * upper;

        // Von Mises bell window of unit peak at phase 0.
        let conc = bandwidth / f0;
        let kappa = (conc * conc).min(MAX_CONCENTRATION);
        let window = ops::exp(kappa * (ops::cos(TAU * self.phase) - 1.0));

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        window * carrier * amp
    }
}

impl AudioNode for PafNode {
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
        self.bandwidth_hz = Smoothed::new(self.bandwidth_hz.target());
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

    fn render(node: &mut PafNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut PafNode,
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

    /// Energy accumulated across harmonics of `f0` whose index lies in
    /// `[lo_h, hi_h]`, skipping any harmonic in `skip`.
    fn harmonic_band(buf: &AudioBuffer, f0: Sample, lo_h: u32, hi_h: u32, skip: &[u32]) -> Sample {
        let mut e = 0.0;
        for h in lo_h..=hi_h {
            if skip.contains(&h) {
                continue;
            }
            let f = f0 * h as Sample;
            if f >= SR as Sample * NYQUIST_GUARD {
                break;
            }
            e += goertzel(buf, SR, f);
        }
        e
    }

    #[test]
    fn renders_bounded_finite() {
        for &formant in &[300.0, 1_200.0, 4_000.0, 8_000.0] {
            for &bandwidth in &[20.0, 200.0, 2_000.0, 4_000.0] {
                let mut node = PafNode::new(120.0, formant, bandwidth, 1.0);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "formant={formant} bandwidth={bandwidth} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn not_silent() {
        let mut node = PafNode::new(120.0, 1_200.0, 200.0, 0.8);
        let out = render(&mut node, SR, 4_096);
        assert!(energy(&out) > 1e-3);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = PafNode::new(120.0, 1_200.0, 200.0, 0.0);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn partials_lock_to_fundamental() {
        // All carrier partials are exact harmonics of f0, so an inter-harmonic
        // bin carries far less energy than the formant-centered harmonic.
        let f0 = 120.0;
        let mut node = PafNode::new(f0, 1_200.0, 200.0, 0.8);
        let out = render(&mut node, SR, 16_384);
        let on = goertzel(&out, SR, 1_200.0); // 10th harmonic
        let off = goertzel(&out, SR, 1_260.0); // 10.5th, not a harmonic
        assert!(on > off * 8.0, "on={on} off={off}");
    }

    #[test]
    fn formant_peaks_near_center() {
        let f0 = 120.0;
        let mut node = PafNode::new(f0, 1_200.0, 200.0, 0.8);
        let out = render(&mut node, SR, 16_384);
        let peak = goertzel(&out, SR, 1_200.0); // 10th harmonic == center
        let low = goertzel(&out, SR, 240.0); // 2nd harmonic, far below
        let high = goertzel(&out, SR, 3_600.0); // 30th harmonic, far above
        assert!(peak > low * 4.0 && peak > high * 4.0, "peak={peak} low={low} high={high}");
    }

    #[test]
    fn formant_shifts_spectral_energy_upward() {
        let f0 = 120.0;
        let mut low = PafNode::new(f0, 1_200.0, 300.0, 0.8);
        let mut high = PafNode::new(f0, 4_800.0, 300.0, 0.8);
        let low_out = render(&mut low, SR, 16_384);
        let high_out = render(&mut high, SR, 16_384);
        // Harmonics 30..60 (3600..7200 Hz).
        let upper_low = harmonic_band(&low_out, f0, 30, 60, &[]);
        let upper_high = harmonic_band(&high_out, f0, 30, 60, &[]);
        assert!(upper_high > upper_low * 2.0, "low={upper_low} high={upper_high}");
    }

    #[test]
    fn bandwidth_widens_spectrum() {
        let f0 = 120.0;
        // Center on the 10th harmonic; compare neighbor energy either side,
        // skipping the peak harmonic itself.
        let mut narrow = PafNode::new(f0, 1_200.0, 60.0, 0.8);
        let mut wide = PafNode::new(f0, 1_200.0, 900.0, 0.8);
        let narrow_out = render(&mut narrow, SR, 16_384);
        let wide_out = render(&mut wide, SR, 16_384);
        // The far band (harmonics 20..40, well above the 10th-harmonic center)
        // only fills in as the window narrows in time, i.e. as bandwidth grows.
        let far_narrow = harmonic_band(&narrow_out, f0, 20, 40, &[]);
        let far_wide = harmonic_band(&wide_out, f0, 20, 40, &[]);
        assert!(far_wide > far_narrow * 4.0, "narrow={far_narrow} wide={far_wide}");
    }

    #[test]
    fn narrow_bandwidth_approaches_single_harmonic() {
        let f0 = 120.0;
        let mut node = PafNode::new(f0, 1_200.0, 30.0, 0.8);
        let out = render(&mut node, SR, 16_384);
        let peak = goertzel(&out, SR, 1_200.0);
        let neigh = goertzel(&out, SR, 1_080.0) + goertzel(&out, SR, 1_320.0);
        assert!(peak > neigh * 4.0, "peak={peak} neigh={neigh}");
    }

    #[test]
    fn formant_between_harmonics_splits_energy() {
        let f0 = 120.0;
        // 1260 Hz == 10.5 * f0, exactly between harmonics 10 and 11.
        let mut node = PafNode::new(f0, 1_260.0, 150.0, 0.8);
        let out = render(&mut node, SR, 16_384);
        let h10 = goertzel(&out, SR, 1_200.0);
        let h11 = goertzel(&out, SR, 1_320.0);
        let far_lo = goertzel(&out, SR, 960.0); // harmonic 8
        let far_hi = goertzel(&out, SR, 1_560.0); // harmonic 13
        let bracket = h10.min(h11);
        let far = far_lo.max(far_hi);
        assert!(bracket > far * 2.0, "h10={h10} h11={h11} far={far}");
        // Equal crossfade weights keep the two brackets within ~2x of each other.
        assert!(h10 < h11 * 2.5 && h11 < h10 * 2.5, "h10={h10} h11={h11}");
    }

    #[test]
    fn waveform_is_click_free() {
        // Moderate bandwidth keeps the pulse wide enough that adjacent samples
        // stay close -- a direct witness of C-infinity continuity.
        let mut node = PafNode::new(120.0, 600.0, 150.0, 0.8);
        let out = render(&mut node, SR, 4_096);
        let s = out.channel(0);
        for w in s.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.1, "diff={}", w[1] - w[0]);
        }
    }

    #[test]
    fn formant_sweep_is_click_free() {
        let mut node = PafNode::new(120.0, 300.0, 150.0, 0.8);
        node.set_formant_hz(700.0, Ramp::Linear { samples: 4_096 });
        let out = render(&mut node, SR, 4_096);
        let s = out.channel(0);
        for w in s.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.1, "diff={}", w[1] - w[0]);
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = PafNode::new(130.0, 1_300.0, 220.0, 0.7);
        let mut b = PafNode::new(130.0, 1_300.0, 220.0, 0.7);
        assert_eq!(render(&mut a, SR, 1_024).channel(0), render(&mut b, SR, 1_024).channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = PafNode::new(130.0, 1_300.0, 220.0, 0.7);
        let first = render(&mut node, SR, 1_024).channel(0).to_vec();
        node.reset();
        assert_eq!(node.phase, 0.0);
        let second = render(&mut node, SR, 1_024).channel(0).to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = PafNode::new(120.0, 1_200.0, 200.0, 0.25);
        let mut loud = PafNode::new(120.0, 1_200.0, 200.0, 0.5);
        let eq = energy(&render(&mut quiet, SR, 4_096));
        let el = energy(&render(&mut loud, SR, 4_096));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 0.1, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        for layout in [ChannelLayout::Stereo, ChannelLayout::Quad] {
            let mut node = PafNode::new(120.0, 1_200.0, 200.0, 0.8);
            let out = render_layout(&mut node, SR, 512, layout);
            let ch0 = out.channel(0);
            for ch in 1..out.channels() {
                assert_eq!(out.channel(ch), ch0, "channel {ch} must mirror mono core");
            }
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = PafNode::new(120.0, 1_200.0, 200.0, 0.8);
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
        let node = PafNode::new(120.0, 1_200.0, 200.0, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = PafNode::new(123.0, 1_800.0, 250.0, 0.7);
        assert_eq!(node.frequency_hz(), 123.0);
        assert_eq!(node.formant_hz(), 1_800.0);
        assert_eq!(node.bandwidth_hz(), 250.0);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = PafParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.formant_hz >= MIN_FORMANT_HZ && p.formant_hz <= MAX_FORMANT_HZ);
        assert!(p.bandwidth_hz >= MIN_BANDWIDTH_HZ && p.bandwidth_hz <= MAX_BANDWIDTH_HZ);
        assert_eq!(p.sanitised().frequency_hz, p.frequency_hz);
    }

    #[test]
    fn from_params_matches_new() {
        let p = PafParams {
            frequency_hz: 140.0,
            formant_hz: 2_100.0,
            bandwidth_hz: 300.0,
            amplitude: 0.7,
        };
        let mut a = PafNode::from_params(p);
        let mut b = PafNode::new(140.0, 2_100.0, 300.0, 0.7);
        assert_eq!(render(&mut a, SR, 512).channel(0), render(&mut b, SR, 512).channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = PafNode::new(5.0, 20_000.0, 20_000.0, 0.8);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        assert_eq!(node.bandwidth_hz(), MAX_BANDWIDTH_HZ);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = PafNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN, Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.formant_hz(), DEFAULT_FORMANT_HZ);
        assert_eq!(node.bandwidth_hz(), DEFAULT_BANDWIDTH_HZ);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = PafNode::new(120.0, 1_200.0, 200.0, 0.8);
        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 120.0);
        node.set_formant_hz(100_000.0, Ramp::Immediate);
        assert_eq!(node.formant_hz(), MAX_FORMANT_HZ);
        node.set_bandwidth_hz(-1.0, Ramp::Immediate);
        assert_eq!(node.bandwidth_hz(), MIN_BANDWIDTH_HZ);
        node.set_amplitude(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn nyquist_guard_keeps_super_nyquist_bounded() {
        let mut node = PafNode::new(3_900.0, 8_000.0, 4_000.0, 1.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn frequency_change_shifts_fundamental() {
        let mut low = PafNode::new(110.0, 1_100.0, 200.0, 0.8);
        let mut high = PafNode::new(130.0, 1_300.0, 200.0, 0.8);
        let low_out = render(&mut low, SR, 8_192);
        let high_out = render(&mut high, SR, 8_192);
        // Low node peaks on its own 10th harmonic (1100), high node on 1300.
        assert!(goertzel(&low_out, SR, 1_100.0) > goertzel(&high_out, SR, 1_100.0) * 4.0);
        assert!(goertzel(&high_out, SR, 1_300.0) > goertzel(&low_out, SR, 1_300.0) * 4.0);
    }

    #[test]
    fn bandwidth_change_takes_effect() {
        let f0 = 120.0;
        let mut node = PafNode::new(f0, 1_200.0, 60.0, 0.8);
        let narrow = harmonic_band(&render(&mut node, SR, 16_384), f0, 20, 40, &[]);
        node.set_bandwidth_hz(900.0, Ramp::Immediate);
        node.reset();
        let wide = harmonic_band(&render(&mut node, SR, 16_384), f0, 20, 40, &[]);
        assert!(wide > narrow * 4.0, "narrow={narrow} wide={wide}");
    }
}
