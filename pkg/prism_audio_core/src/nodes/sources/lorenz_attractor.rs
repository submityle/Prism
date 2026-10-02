//! Continuous chaotic-attractor source node (the Lorenz system).
//!
//! [`LorenzAttractorNode`] sonifies the trajectory of the Lorenz system, a
//! three-dimensional set of coupled ordinary differential equations whose
//! solution never settles and never exactly repeats. The equations are
//! integrated in real time and the `x` coordinate is read out as a bipolar,
//! broadband, slowly drifting audio texture. Where
//! [`super::chaotic_oscillator::ChaoticOscillatorNode`] iterates a discrete
//! one-dimensional map once per waveform period, this node integrates a
//! continuous three-dimensional flow at every sample, so it produces an
//! aperiodic drone rather than a pitched tone.
//!
//! # Model
//!
//! The Lorenz system, with fixed Prandtl number `sigma` and geometric factor
//! `beta` and a user-controlled `rho`, is
//!
//! ```text
//!   dx/dt = sigma * (y - x)
//!   dy/dt = x * (rho - z) - y
//!   dz/dt = x * y - beta * z
//! ```
//!
//! It is advanced with the classic fourth-order Runge-Kutta integrator (`RK4`),
//! which evaluates the field four times per step and is far more stable than
//! forward Euler at the step sizes used here. Each audio sample runs
//! `OVERSAMPLE` integrator substeps of size `h`; the total simulation time
//! advanced per sample is `rate_hz * RATE_TO_TIMESTEP / sample_rate`, so the
//! `rate_hz` control sets how fast the trajectory moves and therefore the
//! nominal brightness/pitch of the texture (the spectrum is broadband, so this
//! is a nominal rate, not a precise fundamental).
//!
//! The output coordinate is scaled and soft-limited through a hyperbolic
//! tangent, guaranteeing a strictly bounded signal and gentle saturation when
//! the attractor grows at high `rho`:
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
//! `rho` morphs the character: just above the Hopf bifurcation (near `24.74`
//! for these `sigma`/`beta`) the nonzero equilibria are unstable so the flow
//! sustains forever; the canonical `rho = 28` gives the familiar two-lobe
//! butterfly, and larger values enlarge and energise the attractor.
//!
//! # Relationship
//!
//! - Unlike [`super::chaotic_oscillator::ChaoticOscillatorNode`], which iterates
//!   a discrete one-dimensional logistic map once per waveform period and is
//!   phase-locked to a pitch, this node integrates a continuous
//!   three-dimensional flow every sample and has no hard period: it is
//!   continuous-flow chaos versus discrete-map chaos.
//! - Unlike [`super::noise::NoiseNode`], whose stream is drawn from a
//!   pseudo-random generator (`PRNG`), this source is fully deterministic: its
//!   roughness comes from deterministic chaos and is reproduced exactly from the
//!   fixed initial state.
//! - Unlike the fixed periodic waveshape of
//!   [`super::oscillator::OscillatorNode`], the trajectory never exactly
//!   repeats, so the timbre drifts continuously.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`LorenzAttractorNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every integrator substep
//! clamps the state to a generous bounding box and falls back to the initial
//! seed if a non-finite value or runaway ever appears, so the output can never
//! blow up. `rho` and `amplitude` glide through [`Smoothed`] values and
//! `rate_hz` only scales the timestep, so automation never produces zipper
//! clicks. Two nodes built with the same parameters produce bit-identical
//! output, and [`LorenzAttractorNode::reset`] restarts the exact same
//! trajectory.
//!
//! # Provenance
//!
//! Implemented from first principles from public-domain nonlinear dynamics: the
//! Lorenz system (Edward Lorenz, 1963, "Deterministic Nonperiodic Flow") and
//! the classic fourth-order Runge-Kutta method from public-domain numerical
//! analysis. Only the shared mathematical idea of sonifying a continuous
//! chaotic attractor is used. This file contains no code, data, or derivative
//! of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, the Web Audio API, the Synthesis Toolkit, or any other audio engine
//! or toolkit; only the shared mathematical ideas are used. There is no AI or
//! machine learning of any kind.

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

/// Minimum `rho` (just above the Hopf bifurcation, so the flow sustains).
pub const MIN_RHO: Sample = 25.0;

/// Default `rho` (the canonical two-lobe butterfly attractor).
pub const DEFAULT_RHO: Sample = 28.0;

/// Maximum `rho` (a large, energetic attractor).
pub const MAX_RHO: Sample = 100.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fixed Prandtl number of the Lorenz system.
const SIGMA: Sample = 10.0;

/// Fixed geometric factor of the Lorenz system.
const BETA: Sample = 8.0 / 3.0;

/// Scales the nominal `rate_hz` into the per-sample simulation timestep.
const RATE_TO_TIMESTEP: Sample = 0.77;

/// Number of integrator substeps per audio sample.
const OVERSAMPLE: usize = 4;

/// Hard upper bound on a single integrator substep for stability.
const H_MAX: Sample = 0.02;

/// Reciprocal readout scale feeding the output `tanh` soft-limiter.
const OUTPUT_INV_SCALE: Sample = 1.0 / 12.0;

/// Deterministic initial state coordinates (a point near the origin that
/// quickly falls onto the attractor).
const INITIAL_X: Sample = 0.1;
/// See [`INITIAL_X`].
const INITIAL_Y: Sample = 0.0;
/// See [`INITIAL_X`].
const INITIAL_Z: Sample = 0.0;

/// Generous bounding box; a substep that leaves it (or goes non-finite) is
/// treated as numerical runaway and the state is reseeded.
const SAFETY_BOUND: Sample = 500.0;

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

/// Evaluates the Lorenz vector field at `(x, y, z)` for the given `rho`.
#[inline]
fn lorenz_derivative(x: Sample, y: Sample, z: Sample, rho: Sample) -> (Sample, Sample, Sample) {
    (SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z)
}

/// Construction parameters for a [`LorenzAttractorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LorenzAttractorParams {
    /// Nominal rate in hertz. Clamped to `[MIN_RATE_HZ, MAX_RATE_HZ]`.
    pub rate_hz: Sample,
    /// Lorenz `rho`. Clamped to `[MIN_RHO, MAX_RHO]`.
    pub rho: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for LorenzAttractorParams {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_RATE_HZ,
            rho: DEFAULT_RHO,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl LorenzAttractorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            rate_hz: finite_or(self.rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            rho: finite_or(self.rho, DEFAULT_RHO).clamp(MIN_RHO, MAX_RHO),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A continuous chaotic-attractor (Lorenz) source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::LorenzAttractorNode;
///
/// let mut node = LorenzAttractorNode::new(110.0, 28.0, 0.8);
/// assert_eq!(node.rate_hz(), 110.0);
/// assert_eq!(node.rho(), 28.0);
/// ```
#[derive(Debug, Clone)]
pub struct LorenzAttractorNode {
    /// Nominal rate in hertz; scales the per-sample simulation timestep.
    rate_hz: Sample,
    /// Smoothed Lorenz `rho` in `[MIN_RHO, MAX_RHO]`.
    rho: Smoothed,
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

impl LorenzAttractorNode {
    /// Creates a Lorenz attractor source at the given nominal `rate_hz`, Lorenz
    /// `rho`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `rate_hz` is clamped to
    /// `[MIN_RATE_HZ, MAX_RATE_HZ]` and `rho` to `[MIN_RHO, MAX_RHO]`.
    #[must_use]
    pub fn new(rate_hz: Sample, rho: Sample, amplitude: Sample) -> Self {
        Self {
            rate_hz: finite_or(rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            rho: Smoothed::new(finite_or(rho, DEFAULT_RHO).clamp(MIN_RHO, MAX_RHO)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            x: INITIAL_X,
            y: INITIAL_Y,
            z: INITIAL_Z,
            lp: 0.0,
            lp_pole: ops::exp(-TAU * (DECIM_CUTOFF_FRACTION / OVERSAMPLE as Sample)),
        }
    }

    /// Builds a Lorenz attractor source from a [`LorenzAttractorParams`] bundle.
    #[must_use]
    pub fn from_params(params: LorenzAttractorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.rate_hz, p.rho, p.amplitude)
    }

    /// Sets a new nominal rate in hertz (applied immediately; only the timestep
    /// scales, so the trajectory stays continuous and the change is
    /// click-free).
    #[inline]
    pub fn set_rate_hz(&mut self, hz: Sample) {
        self.rate_hz = finite_or(hz, self.rate_hz).clamp(MIN_RATE_HZ, MAX_RATE_HZ);
    }

    /// Sets a new target Lorenz `rho`, gliding with `ramp`.
    #[inline]
    pub fn set_rho(&mut self, rho: Sample, ramp: Ramp) {
        self.rho
            .set_target(finite_or(rho, self.rho.target()).clamp(MIN_RHO, MAX_RHO), ramp);
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

    /// Returns the target Lorenz `rho` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn rho(&self) -> Sample {
        self.rho.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Advances the state by one `RK4` substep of size `h` for the given `rho`,
    /// reseeding on any non-finite value or runaway excursion.
    #[inline]
    fn integrate_step(&mut self, h: Sample, rho: Sample) {
        let (x, y, z) = (self.x, self.y, self.z);
        let (k1x, k1y, k1z) = lorenz_derivative(x, y, z, rho);
        let h2 = h * 0.5;
        let (k2x, k2y, k2z) = lorenz_derivative(x + h2 * k1x, y + h2 * k1y, z + h2 * k1z, rho);
        let (k3x, k3y, k3z) = lorenz_derivative(x + h2 * k2x, y + h2 * k2y, z + h2 * k2z, rho);
        let (k4x, k4y, k4z) = lorenz_derivative(x + h * k3x, y + h * k3y, z + h * k3z, rho);
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
        let rho = self.rho.next_sample();
        let amp = self.amplitude.next_sample();

        let dt_total = (self.rate_hz * RATE_TO_TIMESTEP * inv_sr).max(0.0);
        let h = (dt_total / OVERSAMPLE as Sample).min(H_MAX);
        let one_minus_pole = 1.0 - self.lp_pole;

        for _ in 0..OVERSAMPLE {
            self.integrate_step(h, rho);
            let raw = ops::tanh(self.x * OUTPUT_INV_SCALE);
            self.lp += one_minus_pole * (raw - self.lp);
        }

        self.lp * amp
    }
}

impl AudioNode for LorenzAttractorNode {
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
        self.rho = Smoothed::new(self.rho.target());
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

    fn render(node: &mut LorenzAttractorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut LorenzAttractorNode,
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

    fn mean(buf: &AudioBuffer) -> Sample {
        let ch = buf.channel(0);
        if ch.is_empty() {
            return 0.0;
        }
        ch.iter().sum::<Sample>() / ch.len() as Sample
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
            for &rho in &[25.0, 28.0, 60.0, 100.0] {
                let mut node = LorenzAttractorNode::new(rate, rho, 0.9);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "rate={rate} rho={rho} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = LorenzAttractorNode::new(110.0, 28.0, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = LorenzAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_RHO, DEFAULT_AMPLITUDE);
        let out = render(&mut node, SR, 8_192);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = LorenzAttractorNode::new(220.0, 40.0, 0.7);
        let mut b = LorenzAttractorNode::new(220.0, 40.0, 0.7);
        let out_a = render(&mut a, SR, 8_192);
        let out_b = render(&mut b, SR, 8_192);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = LorenzAttractorNode::new(330.0, 28.0, 0.8);
        let first = render(&mut node, SR, 8_192);
        node.reset();
        let second = render(&mut node, SR, 8_192);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = LorenzAttractorNode::new(110.0, 28.0, 0.25);
        let mut loud = LorenzAttractorNode::new(110.0, 28.0, 0.5);
        let e_quiet = energy(&render(&mut quiet, SR, 4_096));
        let e_loud = energy(&render(&mut loud, SR, 4_096));
        assert!(e_quiet > 0.0);
        let ratio = e_loud / e_quiet;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        for layout in [ChannelLayout::Stereo, ChannelLayout::Quad] {
            let mut node = LorenzAttractorNode::new(140.0, 32.0, 0.8);
            let out = render_layout(&mut node, SR, 2_048, layout);
            let base = out.channel(0).to_vec();
            for ch in 1..out.channels() {
                assert_eq!(out.channel(ch), base.as_slice(), "channel {ch}");
            }
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut idle = LorenzAttractorNode::new(110.0, 28.0, 0.8);
        {
            let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
            out.set_active_frames(0);
            let inputs: [AudioBuffer; 0] = [];
            let mut outputs = [out];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            idle.process(&ctx(SR, 0), &mut io);
        }
        let after_idle = render(&mut idle, SR, 1_024).channel(0).to_vec();

        let mut fresh = LorenzAttractorNode::new(110.0, 28.0, 0.8);
        let fresh_out = render(&mut fresh, SR, 1_024).channel(0).to_vec();
        assert_eq!(after_idle, fresh_out);
    }

    #[test]
    fn latency_is_zero() {
        let node = LorenzAttractorNode::new(110.0, 28.0, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = LorenzAttractorNode::new(275.0, 44.0, 0.6);
        assert_eq!(node.rate_hz(), 275.0);
        assert_eq!(node.rho(), 44.0);
        assert_eq!(node.amplitude(), 0.6);
    }

    #[test]
    fn default_params_in_domain() {
        let p = LorenzAttractorParams::default();
        assert!(p.rate_hz >= MIN_RATE_HZ && p.rate_hz <= MAX_RATE_HZ);
        assert!(p.rho >= MIN_RHO && p.rho <= MAX_RHO);
        assert!(p.amplitude.is_finite());
    }

    #[test]
    fn from_params_matches_new() {
        let params = LorenzAttractorParams {
            rate_hz: 180.0,
            rho: 36.0,
            amplitude: 0.75,
        };
        let mut via_params = LorenzAttractorNode::from_params(params);
        let mut via_new = LorenzAttractorNode::new(180.0, 36.0, 0.75);
        let a = render(&mut via_params, SR, 4_096);
        let b = render(&mut via_new, SR, 4_096);
        assert_eq!(a.channel(0), b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = LorenzAttractorNode::new(1.0e9, 1.0e9, 0.5);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);
        assert_eq!(node.rho(), MAX_RHO);
        let low = LorenzAttractorNode::new(-50.0, -50.0, 0.5);
        assert_eq!(low.rate_hz(), MIN_RATE_HZ);
        assert_eq!(low.rho(), MIN_RHO);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = LorenzAttractorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.rate_hz(), DEFAULT_RATE_HZ);
        assert_eq!(node.rho(), DEFAULT_RHO);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = LorenzAttractorNode::new(110.0, 28.0, 0.8);
        node.set_rate_hz(Sample::NAN);
        assert_eq!(node.rate_hz(), 110.0);
        node.set_rate_hz(1.0e9);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);
        node.set_rho(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.rho(), 28.0);
        node.set_rho(1.0e9, Ramp::Immediate);
        assert_eq!(node.rho(), MAX_RHO);
        node.set_rho(0.0, Ramp::Immediate);
        assert_eq!(node.rho(), MIN_RHO);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn higher_rate_increases_high_frequency_energy() {
        let mut slow = LorenzAttractorNode::new(20.0, 28.0, 0.8);
        let mut fast = LorenzAttractorNode::new(1_500.0, 28.0, 0.8);
        let slow_out = render(&mut slow, SR, 16_384);
        let fast_out = render(&mut fast, SR, 16_384);
        assert!(
            hf_energy(&fast_out) > hf_energy(&slow_out),
            "slow={} fast={}",
            hf_energy(&slow_out),
            hf_energy(&fast_out)
        );
    }

    #[test]
    fn rho_changes_alter_output() {
        // Discard a warm-up so both trajectories are on the attractor.
        let mut low = LorenzAttractorNode::new(110.0, 26.0, 0.8);
        let mut high = LorenzAttractorNode::new(110.0, 95.0, 0.8);
        let _ = render(&mut low, SR, 4_096);
        let _ = render(&mut high, SR, 4_096);
        let a = render(&mut low, SR, 8_192);
        let b = render(&mut high, SR, 8_192);
        let differ = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differ > a.channel(0).len() / 10, "differ={differ}");
    }

    #[test]
    fn output_is_bounded_under_extreme_rate() {
        let mut node = LorenzAttractorNode::new(MAX_RATE_HZ, MAX_RHO, 0.9);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 0.9 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn rate_change_is_click_free() {
        let mut node = LorenzAttractorNode::new(110.0, 28.0, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.set_rate_hz(400.0);
        let second = render(&mut node, SR, 4_096);
        let last = *first.channel(0).last().unwrap();
        let next = second.channel(0)[0];
        assert!((next - last).abs() < 0.2, "join step {}", (next - last).abs());
    }

    #[test]
    fn long_run_mean_is_near_zero() {
        let mut node = LorenzAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_RHO, 0.9);
        let _ = render(&mut node, SR, 8_192);
        let out = render(&mut node, SR, 200_000);
        assert!(mean(&out).abs() < 0.1, "mean={}", mean(&out));
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = LorenzAttractorParams {
            rate_hz: 300.0,
            rho: 50.0,
            amplitude: 0.7,
        };
        let s = params.sanitised();
        assert_eq!(s.rate_hz, 300.0);
        assert_eq!(s.rho, 50.0);
        assert_eq!(s.amplitude, 0.7);
    }
}
