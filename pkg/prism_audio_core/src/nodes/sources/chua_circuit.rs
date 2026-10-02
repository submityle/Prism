//! Continuous chaotic-attractor source node (Chua's circuit).
//!
//! [`ChuaCircuitNode`] sonifies the state of Chua's circuit, the canonical
//! electronic oscillator whose only nonlinear element is a piecewise-linear
//! resistor (the "Chua diode"). Unlike the smooth polynomial or sinusoidal
//! chaotic flows, its vector field has three linear regions joined at two
//! breakpoints, so the trajectory winds around two unstable foci and hops
//! between them across a flat separatrix, drawing the famous double-scroll
//! attractor. The first state coordinate (a capacitor voltage) is read out as a
//! bipolar, broadband audio texture with a sharp, buzzy edge that the smooth
//! attractors lack.
//!
//! # Model
//!
//! In dimensionless form, with a fixed `beta` and the two diode slopes `m0`
//! (inner) and `m1` (outer), Chua's circuit is
//!
//! ```text
//!   dx/dt = alpha * (y - x - g(x))
//!   dy/dt = x - y + z
//!   dz/dt = -beta * y
//!   g(x)  = m1 * x + 0.5 * (m0 - m1) * (|x + 1| - |x - 1|)
//! ```
//!
//! The nonlinearity `g(x)` is a three-segment piecewise-linear curve: it has
//! slope `m0` in the central region `|x| < 1` and slope `m1` outside it, with
//! breakpoints at `x = +-1`. This odd, continuous-but-not-smooth characteristic
//! is what gives the double scroll its sharp folds. The control `alpha` is the
//! natural bifurcation parameter: raising it walks the circuit from a stable
//! focus through a period-doubling cascade into a single scroll and finally the
//! two-lobe double scroll (the canonical value is `alpha = 15.6`), so `alpha`
//! behaves like a continuous order-to-chaos morph.
//!
//! It is advanced with the classic fourth-order Runge-Kutta integrator (`RK4`),
//! which evaluates the field four times per step and is far more stable than
//! forward Euler at the step sizes used here. Each audio sample runs
//! `OVERSAMPLE` integrator substeps; the total simulation time advanced per
//! sample is `rate_hz * RATE_TO_TIMESTEP / sample_rate`, so the `rate_hz`
//! control sets how fast the trajectory moves and therefore the nominal
//! brightness/pitch of the texture (the spectrum is broadband, so this is a
//! nominal rate, not a precise fundamental). Each integrator substep is clamped
//! to `H_MAX` for stability, so once `rate_hz` grows large enough that the
//! per-substep timestep saturates (near the top of its range at 48 kHz) further
//! increases stop speeding up the clock and the brightness plateaus.
//!
//! The output coordinate is scaled and soft-limited through a hyperbolic
//! tangent, keeping the normalised readout within `+-1` (so the output stays
//! bounded to `+-amplitude`) with gentle saturation on the sharp scroll
//! transitions:
//!
//! ```text
//!   raw = tanh(x * OUTPUT_INV_SCALE)
//!   out = amplitude * decimation_lowpass(raw)
//! ```
//!
//! Because the substeps run at `OVERSAMPLE` times the audio rate, a one-pole
//! lowpass is applied across the substeps before the final value is taken as
//! the output sample; it acts as a decimation / anti-image filter that
//! attenuates energy above the audio Nyquist produced by the oversampled
//! integration, and doubles as a de-click smoother.
//!
//! # Relationship
//!
//! - Unlike [`super::lorenz_attractor::LorenzAttractorNode`],
//!   [`super::rossler_attractor::RosslerAttractorNode`], and
//!   [`super::thomas_attractor::ThomasAttractorNode`], whose nonlinearities are
//!   smooth (polynomial products or sines), Chua's only nonlinearity is a
//!   piecewise-linear diode: the field is continuous but has slope
//!   discontinuities at the breakpoints, so the double scroll folds sharply and
//!   the readout carries a buzzier, more electronic edge.
//! - Unlike [`super::duffing_oscillator::DuffingOscillatorNode`], which is a
//!   *driven* oscillator with an external periodic forcing term, Chua's circuit
//!   is *autonomous*: it has no external clock and generates its own recurrence
//!   from the diode feedback.
//! - Unlike [`super::chaotic_oscillator::ChaoticOscillatorNode`], which iterates
//!   a discrete one-dimensional map once per waveform period, this node
//!   integrates a continuous three-dimensional flow every sample and has no
//!   hard period.
//! - Unlike [`super::noise::NoiseNode`], whose stream is drawn from a
//!   pseudo-random generator (`PRNG`), this source is fully deterministic: its
//!   roughness comes from deterministic chaos and is reproduced exactly from the
//!   fixed initial state.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`ChuaCircuitNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every integrator substep
//! clamps the state to a generous bounding box and falls back to the initial
//! seed if a non-finite value or runaway ever appears, so the normalised
//! readout stays bounded to `+-1` and the output to `+-amplitude`. `alpha` and
//! `amplitude` glide through [`Smoothed`] values and
//! `rate_hz` only scales the timestep, so automation never produces zipper
//! clicks. Two nodes built with the same parameters produce bit-identical
//! output, and [`ChuaCircuitNode::reset`] restarts the exact same trajectory.
//!
//! # Provenance
//!
//! Implemented from first principles from public-domain nonlinear dynamics:
//! Chua's circuit (Leon Chua, 1983) and its dimensionless double-scroll
//! formulation, together with the classic fourth-order Runge-Kutta method from
//! public-domain numerical analysis. Only the shared mathematical idea of
//! sonifying a continuous chaotic attractor is used. This file contains no
//! code, data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, the Web Audio API, the Synthesis Toolkit, or
//! any other audio engine or toolkit; only the shared mathematical ideas are
//! used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};
use bevy_math::ops;
use core::f32::consts::TAU;

/// Minimum nominal rate in hertz (a frozen trajectory).
pub const MIN_RATE_HZ: Sample = 0.0;

/// Default nominal rate in hertz.
pub const DEFAULT_RATE_HZ: Sample = 110.0;

/// Maximum nominal rate in hertz.
pub const MAX_RATE_HZ: Sample = 2_000.0;

/// Minimum `alpha` (a stable focus before the period-doubling cascade).
pub const MIN_ALPHA: Sample = 8.0;

/// Default `alpha` (the canonical double-scroll attractor).
pub const DEFAULT_ALPHA: Sample = 15.6;

/// Maximum `alpha` (a large, violently folded double scroll).
pub const MAX_ALPHA: Sample = 18.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fixed `beta` coefficient of the dimensionless Chua system.
const BETA: Sample = 28.0;

/// Inner slope of the piecewise-linear Chua diode (region `|x| < 1`).
const M0: Sample = -8.0 / 7.0;

/// Outer slope of the piecewise-linear Chua diode (region `|x| > 1`).
const M1: Sample = -5.0 / 7.0;

/// Scales the nominal `rate_hz` into the per-sample simulation timestep.
const RATE_TO_TIMESTEP: Sample = 4.0;

/// Number of integrator substeps per audio sample (Chua is comparatively
/// stiff, so more substeps are used than for the smooth attractors).
const OVERSAMPLE: usize = 8;

/// Hard upper bound on a single integrator substep for stability.
const H_MAX: Sample = 0.02;

/// Reciprocal readout scale feeding the output `tanh` soft-limiter.
const OUTPUT_INV_SCALE: Sample = 1.0 / 2.5;

/// Deterministic initial state coordinates (a point near the saddle origin that
/// quickly spirals out onto the attractor).
const INITIAL_X: Sample = 0.1;
/// See [`INITIAL_X`].
const INITIAL_Y: Sample = 0.0;
/// See [`INITIAL_X`].
const INITIAL_Z: Sample = 0.0;

/// Generous bounding box; a substep that leaves it (or goes non-finite) is
/// treated as numerical runaway and the state is reseeded.
const SAFETY_BOUND: Sample = 50.0;

/// Decimation-lowpass cutoff as a fraction of the audio sample rate.
const DECIM_CUTOFF_FRACTION: Sample = 0.45;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Evaluates the piecewise-linear Chua diode characteristic `g(x)`.
#[inline]
fn chua_diode(x: Sample) -> Sample {
    M1 * x + 0.5 * (M0 - M1) * (ops::abs(x + 1.0) - ops::abs(x - 1.0))
}

/// Evaluates the Chua vector field at `(x, y, z)` for the given `alpha`.
#[inline]
fn chua_derivative(x: Sample, y: Sample, z: Sample, alpha: Sample) -> (Sample, Sample, Sample) {
    (alpha * (y - x - chua_diode(x)), x - y + z, -BETA * y)
}

/// Construction parameters for a [`ChuaCircuitNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChuaCircuitParams {
    /// Nominal rate in hertz. Clamped to `[MIN_RATE_HZ, MAX_RATE_HZ]`.
    pub rate_hz: Sample,
    /// Chua `alpha`. Clamped to `[MIN_ALPHA, MAX_ALPHA]`.
    pub alpha: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for ChuaCircuitParams {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_RATE_HZ,
            alpha: DEFAULT_ALPHA,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl ChuaCircuitParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            rate_hz: finite_or(self.rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            alpha: finite_or(self.alpha, DEFAULT_ALPHA).clamp(MIN_ALPHA, MAX_ALPHA),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A continuous chaotic-attractor (Chua's circuit) source node (0 inputs, 1
/// output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::ChuaCircuitNode;
///
/// let mut node = ChuaCircuitNode::new(110.0, 15.6, 0.8);
/// assert_eq!(node.rate_hz(), 110.0);
/// assert_eq!(node.alpha(), 15.6);
/// ```
#[derive(Debug, Clone)]
pub struct ChuaCircuitNode {
    /// Nominal rate in hertz; scales the per-sample simulation timestep.
    rate_hz: Sample,
    /// Smoothed Chua `alpha` in `[MIN_ALPHA, MAX_ALPHA]`.
    alpha: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// First state coordinate (read out as the audio signal).
    x: Sample,
    /// Second state coordinate.
    y: Sample,
    /// Third state coordinate.
    z: Sample,
    /// Decimation-lowpass memory running at the oversampled rate.
    lp: Sample,
    /// Decimation-lowpass pole coefficient (one-pole feedback gain).
    lp_pole: Sample,
}

impl ChuaCircuitNode {
    /// Creates a Chua's circuit source at the given nominal `rate_hz`, Chua
    /// `alpha`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `rate_hz` is clamped to
    /// `[MIN_RATE_HZ, MAX_RATE_HZ]` and `alpha` to `[MIN_ALPHA, MAX_ALPHA]`.
    #[must_use]
    pub fn new(rate_hz: Sample, alpha: Sample, amplitude: Sample) -> Self {
        Self {
            rate_hz: finite_or(rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            alpha: Smoothed::new(finite_or(alpha, DEFAULT_ALPHA).clamp(MIN_ALPHA, MAX_ALPHA)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            x: INITIAL_X,
            y: INITIAL_Y,
            z: INITIAL_Z,
            lp: 0.0,
            lp_pole: ops::exp(-TAU * (DECIM_CUTOFF_FRACTION / OVERSAMPLE as Sample)),
        }
    }

    /// Builds a Chua's circuit source from a [`ChuaCircuitParams`] bundle.
    #[must_use]
    pub fn from_params(params: ChuaCircuitParams) -> Self {
        let p = params.sanitised();
        Self::new(p.rate_hz, p.alpha, p.amplitude)
    }

    /// Sets a new nominal rate in hertz (applied immediately; only the timestep
    /// scales, so the trajectory stays continuous and the change is
    /// click-free).
    #[inline]
    pub fn set_rate_hz(&mut self, hz: Sample) {
        self.rate_hz = finite_or(hz, self.rate_hz).clamp(MIN_RATE_HZ, MAX_RATE_HZ);
    }

    /// Sets a new target Chua `alpha`, gliding with `ramp`.
    #[inline]
    pub fn set_alpha(&mut self, alpha: Sample, ramp: Ramp) {
        self.alpha.set_target(
            finite_or(alpha, self.alpha.target()).clamp(MIN_ALPHA, MAX_ALPHA),
            ramp,
        );
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the nominal rate in hertz.
    #[inline]
    #[must_use]
    pub fn rate_hz(&self) -> Sample {
        self.rate_hz
    }

    /// Returns the target Chua `alpha` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn alpha(&self) -> Sample {
        self.alpha.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Advances the state by one `RK4` substep of size `h` for the given
    /// `alpha`, reseeding on any non-finite value or runaway excursion.
    #[inline]
    fn integrate_step(&mut self, h: Sample, alpha: Sample) {
        let (x, y, z) = (self.x, self.y, self.z);
        let (k1x, k1y, k1z) = chua_derivative(x, y, z, alpha);
        let h2 = h * 0.5;
        let (k2x, k2y, k2z) = chua_derivative(x + h2 * k1x, y + h2 * k1y, z + h2 * k1z, alpha);
        let (k3x, k3y, k3z) = chua_derivative(x + h2 * k2x, y + h2 * k2y, z + h2 * k2z, alpha);
        let (k4x, k4y, k4z) = chua_derivative(x + h * k3x, y + h * k3y, z + h * k3z, alpha);
        let sixth = h / 6.0;
        let nx = x + sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        let ny = y + sixth * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        let nz = z + sixth * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);

        if nx.is_finite()
            && ny.is_finite()
            && nz.is_finite()
            && nx.abs() <= SAFETY_BOUND
            && ny.abs() <= SAFETY_BOUND
            && nz.abs() <= SAFETY_BOUND
        {
            self.x = nx;
            self.y = ny;
            self.z = nz;
        } else {
            self.x = INITIAL_X;
            self.y = INITIAL_Y;
            self.z = INITIAL_Z;
        }
    }

    /// Produces one output sample by running `OVERSAMPLE` integrator substeps
    /// and decimating through the one-pole lowpass. `inv_sr` is the reciprocal
    /// of the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let alpha = self.alpha.next_sample();
        let amp = self.amplitude.next_sample();

        let dt_total = (self.rate_hz * RATE_TO_TIMESTEP * inv_sr).max(0.0);
        let h = (dt_total / OVERSAMPLE as Sample).min(H_MAX);
        let one_minus_pole = 1.0 - self.lp_pole;

        for _ in 0..OVERSAMPLE {
            self.integrate_step(h, alpha);
            let raw = ops::tanh(self.x * OUTPUT_INV_SCALE);
            self.lp += one_minus_pole * (raw - self.lp);
        }

        self.lp * amp
    }
}

impl AudioNode for ChuaCircuitNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let inv_sr = 1.0 / sr;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(inv_sr);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.x = INITIAL_X;
        self.y = INITIAL_Y;
        self.z = INITIAL_Z;
        self.lp = 0.0;
        self.alpha = Smoothed::new(self.alpha.target());
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

    fn render(node: &mut ChuaCircuitNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut ChuaCircuitNode,
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

    /// Sum of squared first differences: a monotone proxy for high-frequency
    /// (fast-motion) energy.
    fn hf_energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0)
            .windows(2)
            .map(|w| (w[1] - w[0]) * (w[1] - w[0]))
            .sum()
    }

    #[test]
    fn renders_bounded_finite() {
        for &rate in &[0.0, 55.0, 440.0, 2_000.0] {
            for &alpha in &[8.0, 12.0, 15.6, 18.0] {
                let mut node = ChuaCircuitNode::new(rate, alpha, 0.9);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "rate={rate} alpha={alpha} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, DEFAULT_AMPLITUDE);
        let out = render(&mut node, SR, 16_384);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn rate_zero_freezes_output() {
        let mut node = ChuaCircuitNode::new(0.0, DEFAULT_ALPHA, 0.8);
        let out = render(&mut node, SR, 4_096);
        let ch = out.channel(0);
        let tail = &ch[ch.len() - 512..];
        let first = tail[0];
        for &s in tail {
            assert!((s - first).abs() < 1e-6, "s={s} first={first}");
        }
    }

    #[test]
    fn higher_rate_increases_high_frequency_energy() {
        let mut low = ChuaCircuitNode::new(110.0, DEFAULT_ALPHA, 0.8);
        let mut high = ChuaCircuitNode::new(880.0, DEFAULT_ALPHA, 0.8);
        let _ = render(&mut low, SR, 8_192);
        let _ = render(&mut high, SR, 8_192);
        let hl = hf_energy(&render(&mut low, SR, 16_384));
        let hh = hf_energy(&render(&mut high, SR, 16_384));
        assert!(hh > 1.5 * hl, "high={hh} low={hl}");
    }

    #[test]
    fn alpha_changes_alter_output() {
        let mut tame = ChuaCircuitNode::new(DEFAULT_RATE_HZ, 9.0, 0.8);
        let mut wild = ChuaCircuitNode::new(DEFAULT_RATE_HZ, 16.0, 0.8);
        let a = render(&mut tame, SR, 8_192);
        let b = render(&mut wild, SR, 8_192);
        let differing = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing * 10 > a.channel(0).len(), "differing={differing}");
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.25);
        let mut loud = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.5);
        let eq = energy(&render(&mut quiet, SR, 16_384));
        let el = energy(&render(&mut loud, SR, 16_384));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let mono_out = render(&mut mono, SR, 2_048);

        let mut stereo = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);

        let mut quad = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let quad_out = render_layout(&mut quad, SR, 2_048, ChannelLayout::Quad);

        for ch in 0..stereo_out.channels() {
            assert_eq!(stereo_out.channel(ch), mono_out.channel(0));
        }
        for ch in 0..quad_out.channels() {
            assert_eq!(quad_out.channel(ch), mono_out.channel(0));
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let mut b = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn rate_change_is_click_free() {
        let mut node = ChuaCircuitNode::new(110.0, DEFAULT_ALPHA, 0.8);
        let before = render(&mut node, SR, 2_048);
        node.set_rate_hz(880.0);
        let after = render(&mut node, SR, 2_048);
        let last = *before.channel(0).last().unwrap();
        let first = after.channel(0)[0];
        assert!((first - last).abs() < 0.2, "step={}", (first - last).abs());
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(node.x, INITIAL_X);
        assert_eq!(node.y, INITIAL_Y);
        assert_eq!(node.z, INITIAL_Z);
    }

    #[test]
    fn latency_is_zero() {
        let node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = ChuaCircuitNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.rate_hz(), DEFAULT_RATE_HZ);
        assert_eq!(node.alpha(), DEFAULT_ALPHA);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn getters_report_state() {
        let node = ChuaCircuitNode::new(220.0, 14.0, 0.7);
        assert_eq!(node.rate_hz(), 220.0);
        assert_eq!(node.alpha(), 14.0);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = ChuaCircuitParams::default();
        assert!(p.rate_hz >= MIN_RATE_HZ && p.rate_hz <= MAX_RATE_HZ);
        assert!(p.alpha >= MIN_ALPHA && p.alpha <= MAX_ALPHA);
        assert_eq!(p.sanitised().alpha, p.alpha);
    }

    #[test]
    fn from_params_matches_new() {
        let params = ChuaCircuitParams {
            rate_hz: 330.0,
            alpha: 13.0,
            amplitude: 0.6,
        };
        let mut a = ChuaCircuitNode::from_params(params);
        let mut b = ChuaCircuitNode::new(330.0, 13.0, 0.6);
        let out_a = render(&mut a, SR, 2_048);
        let out_b = render(&mut b, SR, 2_048);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = ChuaCircuitNode::new(99_000.0, 99.0, 0.8);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);
        assert_eq!(node.alpha(), MAX_ALPHA);

        let node = ChuaCircuitNode::new(-5.0, -1.0, 0.8);
        assert_eq!(node.rate_hz(), MIN_RATE_HZ);
        assert_eq!(node.alpha(), MIN_ALPHA);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = ChuaCircuitParams {
            rate_hz: 440.0,
            alpha: 15.0,
            amplitude: 0.75,
        };
        let s = params.sanitised();
        assert_eq!(s.rate_hz, params.rate_hz);
        assert_eq!(s.alpha, params.alpha);
        assert_eq!(s.amplitude, params.amplitude);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = ChuaCircuitNode::new(220.0, 14.0, 0.8);

        node.set_rate_hz(Sample::NAN);
        assert_eq!(node.rate_hz(), 220.0);
        node.set_rate_hz(99_000.0);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);

        node.set_alpha(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.alpha(), 14.0);
        node.set_alpha(99.0, Ramp::Immediate);
        assert_eq!(node.alpha(), MAX_ALPHA);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = ChuaCircuitNode::new(MAX_RATE_HZ, MAX_ALPHA, 1.0);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn runaway_state_is_reseeded_and_recovers() {
        let mut node = ChuaCircuitNode::new(DEFAULT_RATE_HZ, DEFAULT_ALPHA, 0.8);
        // Force the integrator state far outside the safety bounding box to
        // exercise the runaway reseed branch inside the substep integrator.
        node.x = SAFETY_BOUND * 10.0;
        node.y = SAFETY_BOUND * 10.0;
        node.z = SAFETY_BOUND * 10.0;
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }
}
