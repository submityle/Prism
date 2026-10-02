//! Continuous chaotic-attractor source node (the Thomas cyclically symmetric
//! system).
//!
//! [`ThomasAttractorNode`] sonifies the trajectory of the Thomas system, a
//! three-dimensional set of coupled ordinary differential equations that is
//! invariant under the cyclic permutation `x -> y -> z -> x`. Each coordinate
//! is driven by the sine of the next and bled off by a single dissipation
//! constant, so the vector field is intrinsically bounded (its drive never
//! exceeds unit magnitude) and the attractor is a gently labyrinthine,
//! space-filling tangle rather than a sharp-folded scroll. The `x` coordinate
//! is read out as a bipolar, slowly drifting audio texture.
//!
//! # Model
//!
//! The Thomas system, with a single user-controlled dissipation `b`, is
//!
//! ```text
//!   dx/dt = sin(y) - b * x
//!   dy/dt = sin(z) - b * y
//!   dz/dt = sin(x) - b * z
//! ```
//!
//! The three equations are identical up to a cyclic relabelling of the
//! coordinates, so the dynamics have no preferred axis and the sound has no
//! sharp recurring landmark. The nonlinearity is sinusoidal, so the forcing on
//! each axis is confined to `[-1, 1]` and the orbit is confined to roughly
//! `|coordinate| <= 1 / b`. The control `b` sets the character in the opposite
//! sense to a polynomial attractor: small `b` leaves the system nearly
//! conservative and strongly chaotic (a dense, diffusing walk), while raising
//! `b` damps the motion until, past the canonical threshold near `0.33`, the
//! orbit collapses onto a quiet fixed point. The canonical chaotic value is
//! `b = 0.19`, so `b` behaves like a continuous chaos-to-order morph.
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
//! the attractor swells at small `b`:
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
//!   polynomial products build a sharply folded two-lobe butterfly that must be
//!   scaled down before readout, the Thomas field is sinusoidal and cyclically
//!   symmetric: it is self-bounding, has no preferred axis, and sounds like a
//!   slow drifting labyrinth rather than a turbulent drone.
//! - Unlike [`super::rossler_attractor::RosslerAttractorNode`], whose single
//!   quadratic nonlinearity builds an asymmetric single-scroll funnel and whose
//!   `c` runs order to chaos as it increases, the Thomas system is fully
//!   symmetric and its `b` runs chaos to order as it increases (small `b` is
//!   the chaotic regime).
//! - Unlike [`super::duffing_oscillator::DuffingOscillatorNode`], which is a
//!   *driven* oscillator with an external periodic forcing term, the Thomas
//!   system is *autonomous*: it has no external clock and generates its own
//!   recurrence from the symmetric feedback loop.
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
//! [`ThomasAttractorNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every integrator substep
//! clamps the state to a generous bounding box and falls back to the initial
//! seed if a non-finite value or runaway ever appears, so the output can never
//! blow up. `b` and `amplitude` glide through [`Smoothed`] values and `rate_hz`
//! only scales the timestep, so automation never produces zipper clicks. Two
//! nodes built with the same parameters produce bit-identical output, and
//! [`ThomasAttractorNode::reset`] restarts the exact same trajectory.
//!
//! # Provenance
//!
//! Implemented from first principles from public-domain nonlinear dynamics: the
//! Thomas cyclically symmetric system (Rene Thomas, 1999, "Deterministic chaos
//! seen in terms of feedback circuits") and the classic fourth-order
//! Runge-Kutta method from public-domain numerical analysis. Only the shared
//! mathematical idea of sonifying a continuous chaotic attractor is used. This
//! file contains no code, data, or derivative of Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web Audio API, the
//! Synthesis Toolkit, or any other audio engine or toolkit; only the shared
//! mathematical ideas are used. There is no AI or machine learning of any kind.

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

/// Minimum dissipation `b` (a near-conservative, densely chaotic regime).
pub const MIN_B: Sample = 0.08;

/// Default dissipation `b` (the canonical cyclically symmetric chaos).
pub const DEFAULT_B: Sample = 0.19;

/// Maximum dissipation `b` (heavy damping collapsing toward a fixed point).
pub const MAX_B: Sample = 0.6;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Scales the nominal `rate_hz` into the per-sample simulation timestep.
const RATE_TO_TIMESTEP: Sample = 2.0;

/// Number of integrator substeps per audio sample.
const OVERSAMPLE: usize = 4;

/// Hard upper bound on a single integrator substep for stability.
const H_MAX: Sample = 0.03;

/// Reciprocal readout scale feeding the output `tanh` soft-limiter.
const OUTPUT_INV_SCALE: Sample = 1.0 / 5.0;

/// Deterministic initial state coordinates (a point near the origin that
/// quickly falls onto the attractor).
const INITIAL_X: Sample = 0.1;
/// See [`INITIAL_X`].
const INITIAL_Y: Sample = 0.0;
/// See [`INITIAL_X`].
const INITIAL_Z: Sample = 0.0;

/// Generous bounding box; a substep that leaves it (or goes non-finite) is
/// treated as numerical runaway and the state is reseeded.
const SAFETY_BOUND: Sample = 100.0;

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

/// Evaluates the Thomas vector field at `(x, y, z)` for the given `b`.
#[inline]
fn thomas_derivative(x: Sample, y: Sample, z: Sample, b: Sample) -> (Sample, Sample, Sample) {
    (
        ops::sin(y) - b * x,
        ops::sin(z) - b * y,
        ops::sin(x) - b * z,
    )
}

/// Construction parameters for a [`ThomasAttractorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ThomasAttractorParams {
    /// Nominal rate in hertz. Clamped to `[MIN_RATE_HZ, MAX_RATE_HZ]`.
    pub rate_hz: Sample,
    /// Dissipation `b`. Clamped to `[MIN_B, MAX_B]`.
    pub b: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for ThomasAttractorParams {
    fn default() -> Self {
        Self {
            rate_hz: DEFAULT_RATE_HZ,
            b: DEFAULT_B,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl ThomasAttractorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            rate_hz: finite_or(self.rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            b: finite_or(self.b, DEFAULT_B).clamp(MIN_B, MAX_B),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A continuous chaotic-attractor (Thomas) source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::ThomasAttractorNode;
///
/// let mut node = ThomasAttractorNode::new(110.0, 0.19, 0.8);
/// assert_eq!(node.rate_hz(), 110.0);
/// assert_eq!(node.b(), 0.19);
/// ```
#[derive(Debug, Clone)]
pub struct ThomasAttractorNode {
    /// Nominal rate in hertz; scales the per-sample simulation timestep.
    rate_hz: Sample,
    /// Smoothed dissipation `b` in `[MIN_B, MAX_B]`.
    b: Smoothed,
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

impl ThomasAttractorNode {
    /// Creates a Thomas attractor source at the given nominal `rate_hz`,
    /// dissipation `b`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `rate_hz` is clamped to
    /// `[MIN_RATE_HZ, MAX_RATE_HZ]` and `b` to `[MIN_B, MAX_B]`.
    #[must_use]
    pub fn new(rate_hz: Sample, b: Sample, amplitude: Sample) -> Self {
        Self {
            rate_hz: finite_or(rate_hz, DEFAULT_RATE_HZ).clamp(MIN_RATE_HZ, MAX_RATE_HZ),
            b: Smoothed::new(finite_or(b, DEFAULT_B).clamp(MIN_B, MAX_B)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            x: INITIAL_X,
            y: INITIAL_Y,
            z: INITIAL_Z,
            lp: 0.0,
            lp_pole: ops::exp(-TAU * (DECIM_CUTOFF_FRACTION / OVERSAMPLE as Sample)),
        }
    }

    /// Builds a Thomas attractor source from a [`ThomasAttractorParams`] bundle.
    #[must_use]
    pub fn from_params(params: ThomasAttractorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.rate_hz, p.b, p.amplitude)
    }

    /// Sets a new nominal rate in hertz (applied immediately; only the timestep
    /// scales, so the trajectory stays continuous and the change is
    /// click-free).
    #[inline]
    pub fn set_rate_hz(&mut self, hz: Sample) {
        self.rate_hz = finite_or(hz, self.rate_hz).clamp(MIN_RATE_HZ, MAX_RATE_HZ);
    }

    /// Sets a new target dissipation `b`, gliding with `ramp`.
    #[inline]
    pub fn set_b(&mut self, b: Sample, ramp: Ramp) {
        self.b
            .set_target(finite_or(b, self.b.target()).clamp(MIN_B, MAX_B), ramp);
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

    /// Returns the target dissipation `b` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn b(&self) -> Sample {
        self.b.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Advances the state by one `RK4` substep of size `h` for the given `b`,
    /// reseeding on any non-finite value or runaway excursion.
    #[inline]
    fn integrate_step(&mut self, h: Sample, b: Sample) {
        let (x, y, z) = (self.x, self.y, self.z);
        let (k1x, k1y, k1z) = thomas_derivative(x, y, z, b);
        let h2 = h * 0.5;
        let (k2x, k2y, k2z) = thomas_derivative(x + h2 * k1x, y + h2 * k1y, z + h2 * k1z, b);
        let (k3x, k3y, k3z) = thomas_derivative(x + h2 * k2x, y + h2 * k2y, z + h2 * k2z, b);
        let (k4x, k4y, k4z) = thomas_derivative(x + h * k3x, y + h * k3y, z + h * k3z, b);
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
        let b = self.b.next_sample();
        let amp = self.amplitude.next_sample();

        let dt_total = (self.rate_hz * RATE_TO_TIMESTEP * inv_sr).max(0.0);
        let h = (dt_total / OVERSAMPLE as Sample).min(H_MAX);
        let one_minus_pole = 1.0 - self.lp_pole;

        for _ in 0..OVERSAMPLE {
            self.integrate_step(h, b);
            let raw = ops::tanh(self.x * OUTPUT_INV_SCALE);
            self.lp += one_minus_pole * (raw - self.lp);
        }

        self.lp * amp
    }
}

impl AudioNode for ThomasAttractorNode {
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
        self.b = Smoothed::new(self.b.target());
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

    fn render(node: &mut ThomasAttractorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut ThomasAttractorNode,
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
            for &b in &[0.08, 0.19, 0.4, 0.6] {
                let mut node = ThomasAttractorNode::new(rate, b, 0.9);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "rate={rate} b={b} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, DEFAULT_AMPLITUDE);
        let out = render(&mut node, SR, 16_384);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn rate_zero_freezes_output() {
        let mut node = ThomasAttractorNode::new(0.0, DEFAULT_B, 0.8);
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
        let mut low = ThomasAttractorNode::new(110.0, DEFAULT_B, 0.8);
        let mut high = ThomasAttractorNode::new(880.0, DEFAULT_B, 0.8);
        let _ = render(&mut low, SR, 8_192);
        let _ = render(&mut high, SR, 8_192);
        let hl = hf_energy(&render(&mut low, SR, 16_384));
        let hh = hf_energy(&render(&mut high, SR, 16_384));
        assert!(hh > hl, "high={hh} low={hl}");
    }

    #[test]
    fn b_changes_alter_output() {
        let mut chaotic = ThomasAttractorNode::new(DEFAULT_RATE_HZ, 0.12, 0.8);
        let mut damped = ThomasAttractorNode::new(DEFAULT_RATE_HZ, 0.45, 0.8);
        let a = render(&mut chaotic, SR, 8_192);
        let b = render(&mut damped, SR, 8_192);
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
        let mut quiet = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.25);
        let mut loud = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.5);
        let eq = energy(&render(&mut quiet, SR, 16_384));
        let el = energy(&render(&mut loud, SR, 16_384));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        let mono_out = render(&mut mono, SR, 2_048);

        let mut stereo = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);

        let mut quad = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
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
        let mut a = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        let mut b = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn rate_change_is_click_free() {
        let mut node = ThomasAttractorNode::new(110.0, DEFAULT_B, 0.8);
        let before = render(&mut node, SR, 2_048);
        node.set_rate_hz(880.0);
        let after = render(&mut node, SR, 2_048);
        let last = *before.channel(0).last().unwrap();
        let first = after.channel(0)[0];
        assert!((first - last).abs() < 0.2, "step={}", (first - last).abs());
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
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
        let node = ThomasAttractorNode::new(DEFAULT_RATE_HZ, DEFAULT_B, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = ThomasAttractorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.rate_hz(), DEFAULT_RATE_HZ);
        assert_eq!(node.b(), DEFAULT_B);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn getters_report_state() {
        let node = ThomasAttractorNode::new(220.0, 0.3, 0.7);
        assert_eq!(node.rate_hz(), 220.0);
        assert_eq!(node.b(), 0.3);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = ThomasAttractorParams::default();
        assert!(p.rate_hz >= MIN_RATE_HZ && p.rate_hz <= MAX_RATE_HZ);
        assert!(p.b >= MIN_B && p.b <= MAX_B);
        assert_eq!(p.sanitised().b, p.b);
    }

    #[test]
    fn from_params_matches_new() {
        let params = ThomasAttractorParams {
            rate_hz: 330.0,
            b: 0.25,
            amplitude: 0.6,
        };
        let mut a = ThomasAttractorNode::from_params(params);
        let mut b = ThomasAttractorNode::new(330.0, 0.25, 0.6);
        let out_a = render(&mut a, SR, 2_048);
        let out_b = render(&mut b, SR, 2_048);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = ThomasAttractorNode::new(99_000.0, 9.0, 0.8);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);
        assert_eq!(node.b(), MAX_B);

        let node = ThomasAttractorNode::new(-5.0, -1.0, 0.8);
        assert_eq!(node.rate_hz(), MIN_RATE_HZ);
        assert_eq!(node.b(), MIN_B);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = ThomasAttractorParams {
            rate_hz: 440.0,
            b: 0.3,
            amplitude: 0.75,
        };
        let s = params.sanitised();
        assert_eq!(s.rate_hz, params.rate_hz);
        assert_eq!(s.b, params.b);
        assert_eq!(s.amplitude, params.amplitude);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = ThomasAttractorNode::new(220.0, 0.3, 0.8);

        node.set_rate_hz(Sample::NAN);
        assert_eq!(node.rate_hz(), 220.0);
        node.set_rate_hz(99_000.0);
        assert_eq!(node.rate_hz(), MAX_RATE_HZ);

        node.set_b(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.b(), 0.3);
        node.set_b(9.0, Ramp::Immediate);
        assert_eq!(node.b(), MAX_B);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = ThomasAttractorNode::new(MAX_RATE_HZ, MIN_B, 1.0);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }
}
