//! Continuous chaotic-attractor source node (the Rossler system).
//!
//! [`RosslerAttractorNode`] sonifies the trajectory of the Rossler system, a
//! three-dimensional set of coupled ordinary differential equations with a
//! single quadratic nonlinearity. Its solution spirals outward in a nearly flat
//! plane and is periodically folded back by one sharp vertical excursion,
//! producing a single-scroll "funnel" attractor whose sound is smoother and
//! more tonal than the two-lobe Lorenz butterfly. The `x` coordinate is read
//! out as a bipolar, slowly drifting audio texture.
//!
//! # Model
//!
//! The Rossler system, with fixed `A` and `B` and a user-controlled `c`, is
//!
//! ```text
//!   dx/dt = -y - z
//!   dy/dt = x + A * y
//!   dz/dt = B + z * (x - c)
//! ```
//!
//! Only the `z * x` product is nonlinear, so between folds the flow is almost a
//! linear spiral and the waveform is comparatively clean; the `z` variable
//! stays near zero until `x` exceeds `c`, when it spikes and yanks the
//! trajectory back toward the center. The control `c` sets the character: at
//! small `c` the orbit is a simple limit cycle, and raising `c` walks the
//! system through a period-doubling cascade into developed chaos (the canonical
//! `c = 5.7` is firmly chaotic), so `c` behaves like a continuous
//! order-to-chaos morph.
//!
//! It is advanced with the classic fourth-order Runge-Kutta integrator (`RK4`),
//! which evaluates the field four times per step and is far more stable than
//! forward Euler at the step sizes used here. Each audio sample runs
//! `OVERSAMPLE` integrator substeps; the total simulation time advanced per
//! sample is `rate_hz * RATE_TO_TIMESTEP / sample_rate`, so the `rate_hz`
//! control sets how fast the trajectory moves and therefore the nominal
//! brightness/pitch of the texture (the spectrum is broadband, so this is a
//! nominal rate, not a precise fundamental).
//!
//! The output coordinate is scaled and soft-limited through a hyperbolic
//! tangent, guaranteeing a strictly bounded signal and gentle saturation when
//! the attractor grows at high `c`:
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
//! - Unlike [`super::lorenz_attractor::LorenzAttractorNode`], whose two
//!   nonlinear products build a symmetric two-lobe butterfly, the Rossler flow
//!   has a single nonlinearity and a single scroll, so between its sparse folds
//!   the waveform is a near-linear spiral: it is smoother and more tonal than
//!   the broadband Lorenz drone.
//! - Unlike [`super::duffing_oscillator::DuffingOscillatorNode`], which is a
//!   *driven* oscillator with an external periodic forcing term, the Rossler
//!   system is *autonomous*: it has no external clock and generates its own
//!   recurrence from the internal fold.
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
//! [`RosslerAttractorNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every integrator substep
//! clamps the state to a generous bounding box and falls back to the initial
//! seed if a non-finite value or runaway ever appears, so the output can never
//! blow up. `c` and `amplitude` glide through [`Smoothed`] values and `rate_hz`
//! only scales the timestep, so automation never produces zipper clicks. Two
//! nodes built with the same parameters produce bit-identical output, and
//! [`RosslerAttractorNode::reset`] restarts the exact same trajectory.
//!
//! # Provenance
//!
//! Implemented from first principles from public-domain nonlinear dynamics: the
//! Rossler system (Otto Rossler, 1976, "An Equation for Continuous Chaos") and
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

/// Minimum `c` (a simple limit cycle before the period-doubling cascade).
pub const MIN_C: Sample = 2.5;

/// Default `c` (the canonical single-scroll chaotic attractor).
pub const DEFAULT_C: Sample = 5.7;

/// Maximum `c` (a large, violently folded funnel attractor).
pub const MAX_C: Sample = 18.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fixed `A` coefficient of the Rossler system.
const A: Sample = 0.2;

/// Fixed `B` coefficient of the Rossler system.
const B: Sample = 0.2;

/// Scales the nominal `rate_hz` into the per-sample simulation timestep.
const RATE_TO_TIMESTEP: Sample = 2.0;

/// Number of integrator substeps per audio sample.
const OVERSAMPLE: usize = 4;

/// Hard upper bound on a single integrator substep for stability.
const H_MAX: Sample = 0.03;

/// Reciprocal readout scale feeding the output `tanh` soft-limiter.
const OUTPUT_INV_SCALE: Sample = 1.0 / 10.0;

/// Deterministic initial state coordinates (a point near the origin that
/// quickly falls onto the attractor).
const INITIAL_X: Sample = 0.1;
/// See [`INITIAL_X`].
const INITIAL_Y: Sample = 0.0;
/// See [`INITIAL_X`].
const INITIAL_Z: Sample = 0.0;

/// Generous bounding box; a substep that leaves it (or goes non-finite) is
/// treated as numerical runaway and the state is reseeded.
const SAFETY_BOUND: Sample = 2_000.0;

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

/// Evaluates the Rossler vector field at `(x, y, z)` for the given `c`.
#[inline]
fn rossler_derivative(x: Sample, y: Sample, z: Sample, c: Sample) -> (Sample, Sample, Sample) {
    (-y - z, x + A * y, B + z * (x - c))
}

/// Construction parameters for a [`RosslerAttractorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RosslerAttractorParams {
    /// Nominal rate in hertz. Clamped to `[MIN_RATE_HZ, MAX_RATE_HZ]`.
    pub rate_hz: Sample,
    /// Rossler `c`. Clamped to `[MIN_C, MAX_C]`.
    pub c: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for RosslerAttractorParams {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_RATE_HZ,
            c: DEFAULT_C,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl RosslerAttractorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            rate_hz: finite_or(self.rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            c: finite_or(self.c, DEFAULT_C).clamp(MIN_C, MAX_C),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A continuous chaotic-attractor (Rossler) source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::RosslerAttractorNode;
///
/// let mut node = RosslerAttractorNode::new(110.0, 5.7, 0.8);
/// assert_eq!(node.rate_hz(), 110.0);
/// assert_eq!(node.c(), 5.7);
/// ```
#[derive(Debug, Clone)]
pub struct RosslerAttractorNode {
    /// Nominal rate in hertz; scales the per-sample simulation timestep.
    rate_hz: Sample,
    /// Smoothed Rossler `c` in `[MIN_C, MAX_C]`.
    c: Smoothed,
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

impl RosslerAttractorNode {
    /// Creates a Rossler attractor source at the given nominal `rate_hz`,
    /// Rossler `c`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `rate_hz` is clamped to
    /// `[MIN_RATE_HZ, MAX_RATE_HZ]` and `c` to `[MIN_C, MAX_C]`.
    #[must_use]
    pub fn new(rate_hz: Sample, c: Sample, amplitude: Sample) -> Self {
        Self {
            rate_hz: finite_or(rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            c: Smoothed::new(finite_or(c, DEFAULT_C).clamp(MIN_C, MAX_C)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            x: INITIAL_X,
            y: INITIAL_Y,
            z: INITIAL_Z,
            lp: 0.0,
            lp_pole: ops::exp(-TAU * (DECIM_CUTOFF_FRACTION / OVERSAMPLE as Sample)),
        }
    }

    /// Builds a Rossler attractor source from a [`RosslerAttractorParams`]
    /// bundle.
    #[must_use]
    pub fn from_params(params: RosslerAttractorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.rate_hz, p.c, p.amplitude)
    }

    /// Sets a new nominal rate in hertz (applied immediately; only the timestep
    /// scales, so the trajectory stays continuous and the change is
    /// click-free).
    #[inline]
    pub fn set_rate_hz(&mut self, hz: Sample) {
        self.rate_hz = finite_or(hz, self.rate_hz).clamp(MIN_RATE_HZ, MAX_RATE_HZ);
    }

    /// Sets a new target Rossler `c`, gliding with `ramp`.
    #[inline]
    pub fn set_c(&mut self, c: Sample, ramp: Ramp) {
        self.c
            .set_target(finite_or(c, self.c.target()).clamp(MIN_C, MAX_C), ramp);
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

    /// Returns the target Rossler `c` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn c(&self) -> Sample {
        self.c.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Advances the state by one `RK4` substep of size `h` for the given `c`,
    /// reseeding on any non-finite value or runaway excursion.
    #[inline]
    fn integrate_step(&mut self, h: Sample, c: Sample) {
        let (x, y, z) = (self.x, self.y, self.z);
        let (k1x, k1y, k1z) = rossler_derivative(x, y, z, c);
        let h2 = h * 0.5;
        let (k2x, k2y, k2z) = rossler_derivative(x + h2 * k1x, y + h2 * k1y, z + h2 * k1z, c);
        let (k3x, k3y, k3z) = rossler_derivative(x + h2 * k2x, y + h2 * k2y, z + h2 * k2z, c);
        let (k4x, k4y, k4z) = rossler_derivative(x + h * k3x, y + h * k3y, z + h * k3z, c);
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
        let c = self.c.next_sample();
        let amp = self.amplitude.next_sample();

        let dt_total = (self.rate_hz * RATE_TO_TIMESTEP * inv_sr).max(0.0);
        let h = (dt_total / OVERSAMPLE as Sample).min(H_MAX);
        let one_minus_pole = 1.0 - self.lp_pole;

        for _ in 0..OVERSAMPLE {
            self.integrate_step(h, c);
            let raw = ops::tanh(self.x * OUTPUT_INV_SCALE);
            self.lp += one_minus_pole * (raw - self.lp);
        }

        self.lp * amp
    }
}

impl AudioNode for RosslerAttractorNode {
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
        self.c = Smoothed::new(self.c.target());
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

    fn render(node: &mut RosslerAttractorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut RosslerAttractorNode,
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
            for &c in &[2.5, 5.7, 12.0, 18.0] {
                let mut node = RosslerAttractorNode::new(rate, c, 0.9);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "rate={rate} c={c} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node =
            RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, DEFAULT_AMPLITUDE);
        let out = render(&mut node, SR, 16_384);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn rate_zero_freezes_output() {
        let mut node = RosslerAttractorNode::new(0.0, DEFAULT_C, 0.8);
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
        let mut low = RosslerAttractorNode::new(110.0, DEFAULT_C, 0.8);
        let mut high = RosslerAttractorNode::new(880.0, DEFAULT_C, 0.8);
        let _ = render(&mut low, SR, 8_192);
        let _ = render(&mut high, SR, 8_192);
        let hl = hf_energy(&render(&mut low, SR, 16_384));
        let hh = hf_energy(&render(&mut high, SR, 16_384));
        assert!(hh > hl, "high={hh} low={hl}");
    }

    #[test]
    fn c_changes_alter_output() {
        let mut tame = RosslerAttractorNode::new(DEFAULT_RATE_HZ, 3.0, 0.8);
        let mut wild = RosslerAttractorNode::new(DEFAULT_RATE_HZ, 9.0, 0.8);
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
        let mut quiet = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.25);
        let mut loud = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.5);
        let eq = energy(&render(&mut quiet, SR, 16_384));
        let el = energy(&render(&mut loud, SR, 16_384));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        let mono_out = render(&mut mono, SR, 2_048);

        let mut stereo = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);

        let mut quad = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
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
        let mut a = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        let mut b = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn rate_change_is_click_free() {
        let mut node = RosslerAttractorNode::new(110.0, DEFAULT_C, 0.8);
        let before = render(&mut node, SR, 2_048);
        node.set_rate_hz(880.0);
        let after = render(&mut node, SR, 2_048);
        let last = *before.channel(0).last().unwrap();
        let first = after.channel(0)[0];
        assert!((first - last).abs() < 0.2, "step={}", (first - last).abs());
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
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
        let node = RosslerAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_C, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = RosslerAttractorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.rate_hz(), DEFAULT_RATE_HZ);
        assert_eq!(node.c(), DEFAULT_C);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn getters_report_state() {
        let node = RosslerAttractorNode::new(220.0, 7.0, 0.7);
        assert_eq!(node.rate_hz(), 220.0);
        assert_eq!(node.c(), 7.0);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = RosslerAttractorParams::default();
        assert!(p.rate_hz >= MIN_RATE_HZ && p.rate_hz <= MAX_RATE_HZ);
        assert!(p.c >= MIN_C && p.c <= MAX_C);
        assert_eq!(p.sanitised().c, p.c);
    }

    #[test]
    fn from_params_matches_new() {
        let params = RosslerAttractorParams {
            rate_hz: 330.0,
            c: 8.0,
            amplitude: 0.6,
        };
        let mut a = RosslerAttractorNode::from_params(params);
        let mut b = RosslerAttractorNode::new(330.0, 8.0, 0.6);
        let out_a = render(&mut a, SR, 2_048);
        let out_b = render(&mut b, SR, 2_048);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = RosslerAttractorNode::new(99_000.0, 99.0, 0.8);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);
        assert_eq!(node.c(), MAX_C);

        let node = RosslerAttractorNode::new(-5.0, -1.0, 0.8);
        assert_eq!(node.rate_hz(), MIN_RATE_HZ);
        assert_eq!(node.c(), MIN_C);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = RosslerAttractorParams {
            rate_hz: 440.0,
            c: 6.0,
            amplitude: 0.75,
        };
        let s = params.sanitised();
        assert_eq!(s.rate_hz, params.rate_hz);
        assert_eq!(s.c, params.c);
        assert_eq!(s.amplitude, params.amplitude);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = RosslerAttractorNode::new(220.0, 6.0, 0.8);

        node.set_rate_hz(Sample::NAN);
        assert_eq!(node.rate_hz(), 220.0);
        node.set_rate_hz(99_000.0);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);

        node.set_c(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.c(), 6.0);
        node.set_c(99.0, Ramp::Immediate);
        assert_eq!(node.c(), MAX_C);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = RosslerAttractorNode::new(MAX_RATE_HZ, MAX_C, 1.0);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }
}
