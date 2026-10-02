//! Driven Duffing-oscillator source node (a periodically forced double-well
//! flow).
//!
//! [`DuffingOscillatorNode`] sonifies the trajectory of the Duffing equation, a
//! single mechanical oscillator with a cubic (hardening) restoring force that
//! is shaken by an external periodic drive. Unlike the autonomous
//! [`super::lorenz_attractor::LorenzAttractorNode`], whose three-dimensional
//! flow has no external clock and no pitch, this oscillator is *driven*: a
//! sinusoidal forcing term at the control `frequency` anchors a perceptible
//! pitch, and the `drive` strength morphs the response continuously from a
//! clean forced tone, through period-doubling, into deterministic chaos. The
//! position coordinate is read out as a bipolar, pitched-but-restless audio
//! waveform.
//!
//! # Model
//!
//! The classic double-well Duffing oscillator, written as a first-order system
//! in position `x`, velocity `v` and drive phase `phase`, is
//!
//! ```text
//!   dx/dt     = v
//!   dv/dt     = -damping * v + x - x^3 + drive * cos(phase)
//!   dphase/dt = OMEGA
//! ```
//!
//! The `+x - x^3` restoring term is the gradient of the twin-well potential
//! `V(x) = -x^2 / 2 + x^4 / 4`, which has stable minima at `x = +1` and
//! `x = -1` separated by an unstable hill at the origin. With no drive the
//! oscillator decays into one well; the periodic forcing pumps energy back in
//! and, for the right `drive` and `damping`, the trajectory hops chaotically
//! between the two wells so the tone never exactly repeats.
//!
//! It is advanced with the classic fourth-order Runge-Kutta integrator (`RK4`),
//! which evaluates the field four times per step and is far more stable than
//! forward Euler at the step sizes used here. Each audio sample runs
//! `OVERSAMPLE` integrator substeps. The whole simulation clock is scaled by
//! the control `frequency`, so that the dimensionless drive at the fixed
//! angular rate `OMEGA` completes `frequency` cycles per real second:
//!
//! ```text
//!   dt_total = frequency * (TAU / OMEGA) / sample_rate
//! ```
//!
//! Scaling every derivative by this timestep makes the internal dynamics run
//! proportionally faster at higher `frequency`, so the chaotic texture tracks
//! the drive pitch instead of smearing across it. At `frequency = 0` the clock
//! stops and the oscillator freezes.
//!
//! The output coordinate is scaled and soft-limited through a hyperbolic
//! tangent, guaranteeing a strictly bounded signal and gentle saturation when
//! the orbit swings wide:
//!
//! ```text
//!   raw = tanh(x * OUTPUT_DRIVE)
//!   out = amplitude * decimation_lowpass(raw)
//! ```
//!
//! Because the substeps run at `OVERSAMPLE` times the audio rate, a one-pole
//! lowpass is applied across the substeps before the final value is taken as
//! the output sample; it acts as a decimation / anti-image filter that
//! attenuates energy above the audio Nyquist produced by the oversampled
//! integration, and doubles as a de-click smoother.
//!
//! `drive` sets the character: a small value lets the oscillator ring quietly
//! inside one well, moderate values drive sustained large swings across both
//! wells, and large values push the system into fully developed chaos. `damping`
//! sets how fast energy bleeds away between drive kicks, so a low `damping`
//! sustains a long, lively, chaotic ring while a high `damping` settles toward
//! the plain forced response.
//!
//! # Relationship
//!
//! - Unlike [`super::lorenz_attractor::LorenzAttractorNode`], which integrates
//!   an *autonomous* three-dimensional flow with no external clock and no
//!   pitch, this node is a *driven* two-dimensional oscillator whose forcing
//!   term anchors a definite pitch: it is forced chaos versus free-running
//!   chaos.
//! - Unlike [`super::chaotic_oscillator::ChaoticOscillatorNode`], which iterates
//!   a discrete one-dimensional map once per waveform period, this node
//!   integrates a continuous forced flow every sample, so between the periodic
//!   drive kicks the waveform is a smooth mechanical trajectory, not a sequence
//!   of discrete map iterates.
//! - Unlike [`super::noise::NoiseNode`], whose stream is drawn from a
//!   pseudo-random generator (`PRNG`), this source is fully deterministic: its
//!   restlessness comes from deterministic chaos and is reproduced exactly from
//!   the fixed initial state.
//! - Unlike the fixed periodic waveshape of
//!   [`super::oscillator::OscillatorNode`], the trajectory only loosely locks to
//!   the drive, so the timbre breathes and drifts around the nominal pitch.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`DuffingOscillatorNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every integrator substep
//! clamps the state to a generous bounding box and falls back to the initial
//! seed if a non-finite value or runaway ever appears, so the output can never
//! blow up (the quartic potential already confines the orbit, and the enforced
//! lower bound on `damping` keeps the forced response bounded-input
//! bounded-output). `drive`, `damping` and `amplitude` glide through
//! [`Smoothed`] values and `frequency` only scales the timestep, so automation
//! never produces zipper clicks. Two nodes built with the same parameters
//! produce bit-identical output, and [`DuffingOscillatorNode::reset`] restarts
//! the exact same trajectory.
//!
//! # Provenance
//!
//! Implemented from first principles from public-domain nonlinear dynamics: the
//! Duffing oscillator (Georg Duffing, 1918) and its chaotic forced regime
//! studied by Ueda, together with the classic fourth-order Runge-Kutta method
//! from public-domain numerical analysis. Only the shared mathematical idea of
//! sonifying a driven chaotic oscillator is used. This file contains no code,
//! data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, the Web Audio API, the Synthesis Toolkit, or
//! any other audio engine or toolkit; only the shared mathematical ideas are
//! used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};
use bevy_math::ops;
use core::f32::consts::TAU;

/// Minimum drive frequency in hertz (a frozen oscillator).
pub const MIN_FREQUENCY_HZ: Sample = 0.0;

/// Default drive frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum drive frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum forcing amplitude (the oscillator rings quietly inside one well).
pub const MIN_DRIVE: Sample = 0.0;

/// Default forcing amplitude (a lively near-chaotic regime).
pub const DEFAULT_DRIVE: Sample = 0.42;

/// Maximum forcing amplitude (fully developed, violent chaos).
pub const MAX_DRIVE: Sample = 2.0;

/// Minimum viscous damping (kept above zero so the forced response stays
/// bounded-input bounded-output).
pub const MIN_DAMPING: Sample = 0.02;

/// Default viscous damping (the canonical lightly damped chaotic regime).
pub const DEFAULT_DAMPING: Sample = 0.3;

/// Maximum viscous damping (the orbit settles quickly toward the forced
/// response).
pub const MAX_DAMPING: Sample = 1.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fixed dimensionless drive angular frequency (the canonical Ueda value).
const OMEGA: Sample = 1.2;

/// Number of integrator substeps per audio sample.
const OVERSAMPLE: usize = 4;

/// Hard upper bound on a single integrator substep for stability.
const H_MAX: Sample = 0.05;

/// Readout scale feeding the output `tanh` soft-limiter.
const OUTPUT_DRIVE: Sample = 0.9;

/// Deterministic initial position (near the unstable hilltop so the orbit falls
/// into a well and then responds to the drive).
const INITIAL_X: Sample = 0.1;
/// Deterministic initial velocity.
const INITIAL_V: Sample = 0.0;
/// Deterministic initial drive phase in radians.
const INITIAL_PHASE: Sample = 0.0;

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

/// Evaluates the Duffing vector field at `(x, v, phase)` for the given `drive`
/// and `damping`, returning `(dx, dv, dphase)`.
#[inline]
fn duffing_derivative(
    x: Sample,
    v: Sample,
    phase: Sample,
    drive: Sample,
    damping: Sample,
) -> (Sample, Sample, Sample) {
    let dx = v;
    let dv = -damping * v + x - x * x * x + drive * ops::cos(phase);
    (dx, dv, OMEGA)
}

/// Construction parameters for a [`DuffingOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DuffingOscillatorParams {
    /// Drive frequency in hertz. Clamped to `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`.
    pub frequency_hz: Sample,
    /// Forcing amplitude. Clamped to `[MIN_DRIVE, MAX_DRIVE]`.
    pub drive: Sample,
    /// Viscous damping. Clamped to `[MIN_DAMPING, MAX_DAMPING]`.
    pub damping: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for DuffingOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            drive: DEFAULT_DRIVE,
            damping: DEFAULT_DAMPING,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl DuffingOscillatorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            drive: finite_or(self.drive, DEFAULT_DRIVE).clamp(MIN_DRIVE, MAX_DRIVE),
            damping: finite_or(self.damping, DEFAULT_DAMPING).clamp(MIN_DAMPING, MAX_DAMPING),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A driven Duffing-oscillator source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::DuffingOscillatorNode;
///
/// let mut node = DuffingOscillatorNode::new(110.0, 0.42, 0.3, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.drive(), 0.42);
/// ```
#[derive(Debug, Clone)]
pub struct DuffingOscillatorNode {
    /// Drive frequency in hertz; scales the per-sample simulation timestep.
    frequency_hz: Sample,
    /// Smoothed forcing amplitude in `[MIN_DRIVE, MAX_DRIVE]`.
    drive: Smoothed,
    /// Smoothed viscous damping in `[MIN_DAMPING, MAX_DAMPING]`.
    damping: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Position coordinate (read out as the audio signal).
    x: Sample,
    /// Velocity coordinate.
    v: Sample,
    /// Drive phase in radians, wrapped into `[0, TAU)`.
    phase: Sample,
    /// Decimation-lowpass memory running at the oversampled rate.
    lp: Sample,
    /// Decimation-lowpass pole coefficient (one-pole feedback gain).
    lp_pole: Sample,
}

impl DuffingOscillatorNode {
    /// Creates a Duffing oscillator source at the given drive `frequency_hz`,
    /// forcing `drive`, viscous `damping`, and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `frequency_hz` is clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`, `drive` to
    /// `[MIN_DRIVE, MAX_DRIVE]`, and `damping` to `[MIN_DAMPING, MAX_DAMPING]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, drive: Sample, damping: Sample, amplitude: Sample) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            drive: Smoothed::new(finite_or(drive, DEFAULT_DRIVE).clamp(MIN_DRIVE, MAX_DRIVE)),
            damping: Smoothed::new(
                finite_or(damping, DEFAULT_DAMPING).clamp(MIN_DAMPING, MAX_DAMPING),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            x: INITIAL_X,
            v: INITIAL_V,
            phase: INITIAL_PHASE,
            lp: 0.0,
            lp_pole: ops::exp(-TAU * (DECIM_CUTOFF_FRACTION / OVERSAMPLE as Sample)),
        }
    }

    /// Builds a Duffing oscillator source from a [`DuffingOscillatorParams`]
    /// bundle.
    #[must_use]
    pub fn from_params(params: DuffingOscillatorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.drive, p.damping, p.amplitude)
    }

    /// Sets a new drive frequency in hertz (applied immediately; only the
    /// timestep scales, so the trajectory stays continuous and the change is
    /// click-free).
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target forcing `drive`, gliding with `ramp`.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample, ramp: Ramp) {
        self.drive.set_target(
            finite_or(drive, self.drive.target()).clamp(MIN_DRIVE, MAX_DRIVE),
            ramp,
        );
    }

    /// Sets a new target viscous `damping`, gliding with `ramp`.
    #[inline]
    pub fn set_damping(&mut self, damping: Sample, ramp: Ramp) {
        self.damping.set_target(
            finite_or(damping, self.damping.target()).clamp(MIN_DAMPING, MAX_DAMPING),
            ramp,
        );
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the drive frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target forcing `drive` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive.target()
    }

    /// Returns the target viscous `damping` the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn damping(&self) -> Sample {
        self.damping.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Advances the state by one `RK4` substep of size `h` for the given
    /// `drive` and `damping`, reseeding on any non-finite value or runaway
    /// excursion.
    #[inline]
    fn integrate_step(&mut self, h: Sample, drive: Sample, damping: Sample) {
        let (x, v, p) = (self.x, self.v, self.phase);
        let (k1x, k1v, k1p) = duffing_derivative(x, v, p, drive, damping);
        let h2 = h * 0.5;
        let (k2x, k2v, k2p) = duffing_derivative(
            x + h2 * k1x,
            v + h2 * k1v,
            p + h2 * k1p,
            drive,
            damping,
        );
        let (k3x, k3v, k3p) = duffing_derivative(
            x + h2 * k2x,
            v + h2 * k2v,
            p + h2 * k2p,
            drive,
            damping,
        );
        let (k4x, k4v, k4p) =
            duffing_derivative(x + h * k3x, v + h * k3v, p + h * k3p, drive, damping);
        let sixth = h / 6.0;
        let nx = x + sixth * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        let nv = v + sixth * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        let mut np = p + sixth * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);

        if nx.is_finite()
            && nv.is_finite()
            && np.is_finite()
            && nx.abs() <= SAFETY_BOUND
            && nv.abs() <= SAFETY_BOUND
        {
            // Keep the drive phase in a tight range so it never loses precision.
            if np >= TAU {
                np -= TAU;
            }
            self.x = nx;
            self.v = nv;
            self.phase = np;
        } else {
            self.x = INITIAL_X;
            self.v = INITIAL_V;
            self.phase = INITIAL_PHASE;
        }
    }

    /// Produces one output sample by running `OVERSAMPLE` integrator substeps
    /// and decimating through the one-pole lowpass. `inv_sr` is the reciprocal
    /// of the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let drive = self.drive.next_sample();
        let damping = self.damping.next_sample();
        let amp = self.amplitude.next_sample();

        let dt_total = (self.frequency_hz * (TAU / OMEGA) * inv_sr).max(0.0);
        let h = (dt_total / OVERSAMPLE as Sample).min(H_MAX);
        let one_minus_pole = 1.0 - self.lp_pole;

        for _ in 0..OVERSAMPLE {
            self.integrate_step(h, drive, damping);
            let raw = ops::tanh(self.x * OUTPUT_DRIVE);
            self.lp += one_minus_pole * (raw - self.lp);
        }

        self.lp * amp
    }
}

impl AudioNode for DuffingOscillatorNode {
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
        self.v = INITIAL_V;
        self.phase = INITIAL_PHASE;
        self.lp = 0.0;
        self.drive = Smoothed::new(self.drive.target());
        self.damping = Smoothed::new(self.damping.target());
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

    fn render(node: &mut DuffingOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut DuffingOscillatorNode,
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
        for &freq in &[0.0, 55.0, 440.0, 4_000.0] {
            for &drive in &[0.0, 0.42, 1.0, 2.0] {
                for &damping in &[0.02, 0.3, 1.0] {
                    let mut node = DuffingOscillatorNode::new(freq, drive, damping, 0.9);
                    let out = render(&mut node, SR, 4_096);
                    for &s in out.channel(0) {
                        assert!(
                            s.is_finite() && s.abs() <= 1.0 + 1e-3,
                            "freq={freq} drive={drive} damping={damping} s={s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = DuffingOscillatorNode::new(
            DEFAULT_FREQUENCY_HZ,
            DEFAULT_DRIVE,
            DEFAULT_DAMPING,
            DEFAULT_AMPLITUDE,
        );
        let out = render(&mut node, SR, 16_384);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn frozen_when_frequency_zero() {
        let mut node = DuffingOscillatorNode::new(0.0, DEFAULT_DRIVE, 0.3, 0.8);
        let out = render(&mut node, SR, 4_096);
        let ch = out.channel(0);
        // After the decimation lowpass settles, a frozen clock yields a constant.
        let tail = &ch[ch.len() - 512..];
        let first = tail[0];
        for &s in tail {
            assert!((s - first).abs() < 1e-6, "s={s} first={first}");
        }
    }

    #[test]
    fn higher_drive_increases_high_frequency_energy() {
        let mut tame = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, 0.2, 0.3, 0.8);
        let mut wild = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, 1.2, 0.3, 0.8);
        let _ = render(&mut tame, SR, 8_192);
        let _ = render(&mut wild, SR, 8_192);
        let ht = hf_energy(&render(&mut tame, SR, 16_384));
        let hw = hf_energy(&render(&mut wild, SR, 16_384));
        assert!(hw > ht, "wild={hw} tame={ht}");
    }

    #[test]
    fn damping_changes_alter_output() {
        let mut light = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.1, 0.8);
        let mut heavy = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 1.0, 0.8);
        let a = render(&mut light, SR, 4_096);
        let b = render(&mut heavy, SR, 4_096);
        let differing = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing * 10 > a.channel(0).len(), "differing={differing}");
    }

    #[test]
    fn higher_frequency_increases_high_frequency_energy() {
        let mut low = DuffingOscillatorNode::new(110.0, DEFAULT_DRIVE, 0.3, 0.8);
        let mut high = DuffingOscillatorNode::new(880.0, DEFAULT_DRIVE, 0.3, 0.8);
        let _ = render(&mut low, SR, 8_192);
        let _ = render(&mut high, SR, 8_192);
        let hl = hf_energy(&render(&mut low, SR, 16_384));
        let hh = hf_energy(&render(&mut high, SR, 16_384));
        assert!(hh > hl, "high={hh} low={hl}");
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.25);
        let mut loud = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.5);
        let eq = energy(&render(&mut quiet, SR, 16_384));
        let el = energy(&render(&mut loud, SR, 16_384));
        // 0.5^2 / 0.25^2 == 4; the amplitude multiply sits after the tanh.
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let mono_out = render(&mut mono, SR, 2_048);

        let mut stereo = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);

        let mut quad = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
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
        let mut a = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let mut b = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn frequency_change_is_click_free() {
        let mut node = DuffingOscillatorNode::new(110.0, DEFAULT_DRIVE, 0.3, 0.8);
        let before = render(&mut node, SR, 2_048);
        node.set_frequency_hz(880.0);
        let after = render(&mut node, SR, 2_048);
        let last = *before.channel(0).last().unwrap();
        let first = after.channel(0)[0];
        assert!((first - last).abs() < 0.2, "step={}", (first - last).abs());
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        // State must not advance when there are no active frames.
        assert_eq!(node.x, INITIAL_X);
        assert_eq!(node.v, INITIAL_V);
        assert_eq!(node.phase, INITIAL_PHASE);
    }

    #[test]
    fn latency_is_zero() {
        let node = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, DEFAULT_DRIVE, 0.3, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = DuffingOscillatorNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NEG_INFINITY,
            Sample::NAN,
        );
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.damping(), DEFAULT_DAMPING);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn getters_report_state() {
        let node = DuffingOscillatorNode::new(220.0, 0.6, 0.5, 0.7);
        assert_eq!(node.frequency_hz(), 220.0);
        assert_eq!(node.drive(), 0.6);
        assert_eq!(node.damping(), 0.5);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = DuffingOscillatorParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.drive >= MIN_DRIVE && p.drive <= MAX_DRIVE);
        assert!(p.damping >= MIN_DAMPING && p.damping <= MAX_DAMPING);
        assert_eq!(p.sanitised().frequency_hz, p.frequency_hz);
    }

    #[test]
    fn from_params_matches_new() {
        let params = DuffingOscillatorParams {
            frequency_hz: 330.0,
            drive: 0.8,
            damping: 0.25,
            amplitude: 0.6,
        };
        let mut a = DuffingOscillatorNode::from_params(params);
        let mut b = DuffingOscillatorNode::new(330.0, 0.8, 0.25, 0.6);
        let out_a = render(&mut a, SR, 2_048);
        let out_b = render(&mut b, SR, 2_048);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = DuffingOscillatorNode::new(99_000.0, 9.0, 9.0, 0.8);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(node.drive(), MAX_DRIVE);
        assert_eq!(node.damping(), MAX_DAMPING);

        let node = DuffingOscillatorNode::new(-5.0, -1.0, -1.0, 0.8);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(node.drive(), MIN_DRIVE);
        assert_eq!(node.damping(), MIN_DAMPING);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = DuffingOscillatorParams {
            frequency_hz: 440.0,
            drive: 0.5,
            damping: 0.4,
            amplitude: 0.75,
        };
        let s = params.sanitised();
        assert_eq!(s.frequency_hz, params.frequency_hz);
        assert_eq!(s.drive, params.drive);
        assert_eq!(s.damping, params.damping);
        assert_eq!(s.amplitude, params.amplitude);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = DuffingOscillatorNode::new(220.0, 0.5, 0.4, 0.8);

        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 220.0);
        node.set_frequency_hz(99_000.0);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);

        node.set_drive(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.drive(), 0.5);
        node.set_drive(9.0, Ramp::Immediate);
        assert_eq!(node.drive(), MAX_DRIVE);

        node.set_damping(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.damping(), 0.4);
        node.set_damping(-1.0, Ramp::Immediate);
        assert_eq!(node.damping(), MIN_DAMPING);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = DuffingOscillatorNode::new(MAX_FREQUENCY_HZ, MAX_DRIVE, MIN_DAMPING, 1.0);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn drive_changes_alter_output() {
        let mut low = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, 0.2, 0.3, 0.8);
        let mut high = DuffingOscillatorNode::new(DEFAULT_FREQUENCY_HZ, 1.2, 0.3, 0.8);
        let a = render(&mut low, SR, 4_096);
        let b = render(&mut high, SR, 4_096);
        let differing = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing * 10 > a.channel(0).len(), "differing={differing}");
    }
}
