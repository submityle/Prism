//! Wave-terrain synthesis source node (a static two-dimensional surface read
//! along a moving orbit).
//!
//! [`WaveTerrainNode`] defines a fixed analytic height surface `z = f(x, y)`
//! over the square `[-1, 1] x [-1, 1]` and traces a closed orbit
//! `(x(t), y(t))` across it. The instantaneous terrain height sampled along the
//! orbit is the output waveform. The orbit is a Lissajous figure whose angular
//! rate sets the musical pitch, while the surface shape and the orbit radius
//! set the timbre: a small orbit stays near the flat center and sounds pure,
//! and a large orbit sweeps the rippled outer surface and sounds bright and
//! harmonically rich.
//!
//! # Model
//!
//! Two independent phase accumulators drive the orbit, one for each axis, so
//! the Lissajous ratio can be any positive value without introducing a wrap
//! discontinuity:
//!
//! ```text
//!   x = radius * cos(TAU * phase_x)
//!   y = radius * sin(TAU * phase_y)
//!   phase_x += frequency / sample_rate                 (wrapped into [0, 1))
//!   phase_y += ratio * frequency / sample_rate         (wrapped into [0, 1))
//! ```
//!
//! When `ratio` is an integer the orbit is a closed Lissajous curve and the
//! tone is exactly periodic at `frequency`; a non-integer `ratio` gives a
//! quasi-periodic orbit that slowly fills the surface and makes the tone
//! shimmer, but it is still click-free because both axes stay continuous
//! sinusoids.
//!
//! The terrain is a convex blend, controlled by `warp`, of a gentle diagonal
//! swell and a non-separable cross-term ripple:
//!
//! ```text
//!   swell  = sin(PI * (x + y))
//!   ripple = sin(TAU * RIDGES * x * y)
//!   z      = (1 - warp) * swell + warp * ripple
//! ```
//!
//! Both terms are bounded in `[-1, 1]`, so their convex blend is bounded in
//! `[-1, 1]` for any `warp` in `[0, 1]`. The `x * y` cross term is the
//! essential non-separable interaction that gives wave-terrain synthesis its
//! signature: as the orbit sweeps the surface, the product term folds the two
//! axes together and sprays harmonics that a one-dimensional lookup table
//! cannot produce. Raising `warp` shifts weight toward that ripple and
//! brightens the tone; raising `radius` sends the orbit further into the
//! steep, rippled outer surface, which also brightens and loudens it, so a tiny
//! orbit near the origin is quiet and nearly sinusoidal.
//!
//! The output soft-limits the terrain height and only `amplitude` scales it:
//!
//! ```text
//!   out = amplitude * tanh(z * OUTPUT_DRIVE)
//! ```
//!
//! Because the gain sits after the `tanh`, output level is a clean square-law
//! of `amplitude`.
//!
//! # Relationship
//!
//! - Unlike [`super::wavetable_oscillator::WavetableOscillatorNode`], which
//!   reads a one-dimensional precomputed table at a single phase, this node
//!   samples a two-dimensional analytic surface along a moving orbit, so the
//!   non-separable `x * y` interaction creates timbres no single table scan can.
//! - Unlike [`super::scanned_synthesis::ScannedSynthesisNode`], whose table is
//!   a one-dimensional displacement profile that a mass-spring lattice morphs
//!   over time, the surface here is a fixed closed-form function and all the
//!   motion lives in the deterministic orbit, so there is no dynamical state to
//!   integrate and nothing that can diverge.
//! - Unlike [`super::lorenz_attractor::LorenzAttractorNode`] and
//!   [`super::chaotic_oscillator::ChaoticOscillatorNode`], which sonify a state
//!   variable of a nonlinear differential system and are aperiodic or
//!   pitched-but-noisy, the orbit here is a smooth closed curve and the surface
//!   is fixed, so an integer `ratio` yields a cleanly periodic, pitched tone.
//! - Unlike [`super::additive_oscillator::AdditiveOscillatorNode`], which sums a
//!   specified set of harmonics, the harmonic content here emerges from the
//!   geometry of the orbit crossing the surface rather than being prescribed.
//!
//! # Real-time contract
//!
//! The entire state is two scalar phase accumulators, so
//! [`WaveTerrainNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. The output cannot blow up
//! because the terrain is a blend of bounded sinusoids passed through a `tanh`,
//! and the orbit is bounded trigonometry of finite phases. `frequency`,
//! `radius`, `warp`, and `amplitude` glide through [`Smoothed`] values and
//! `ratio` only scales a phase increment, so automation never produces zipper
//! clicks. Two nodes built with the same parameters produce bit-identical
//! output, and [`WaveTerrainNode::reset`] restarts the exact same orbit.
//!
//! # Provenance
//!
//! Implemented from first principles. Wave-terrain synthesis (tracing an orbit
//! across a two-dimensional height surface and reading its elevation as a
//! waveform) is the public-domain technique introduced by Mitsuhashi and
//! developed by Borgonovo and Haus and described by Roads; the Lissajous orbit
//! and the particular analytic terrain used here are public-domain mathematics
//! chosen by the author. Only those shared mathematical ideas are used. This
//! file contains no code, data, or derivative of Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web Audio API, the
//! Synthesis Toolkit, or any other audio engine or toolkit; only the shared
//! mathematical ideas are used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};
use bevy_math::ops;
use core::f32::consts::{PI, TAU};

/// Minimum orbit rate in hertz (direct current, a frozen orbit).
pub const MIN_FREQUENCY_HZ: Sample = 0.0;

/// Default orbit rate in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum orbit rate in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum orbit radius (a point at the surface center).
pub const MIN_RADIUS: Sample = 0.0;

/// Default orbit radius.
pub const DEFAULT_RADIUS: Sample = 0.7;

/// Maximum orbit radius (the orbit reaches the surface corners).
pub const MAX_RADIUS: Sample = 1.0;

/// Minimum normalized warp (pure diagonal swell terrain).
pub const MIN_WARP: Sample = 0.0;

/// Default normalized warp.
pub const DEFAULT_WARP: Sample = 0.5;

/// Maximum normalized warp (pure cross-term ripple terrain).
pub const MAX_WARP: Sample = 1.0;

/// Minimum Lissajous ratio of the `y` axis rate to the `x` axis rate.
pub const MIN_RATIO: Sample = 0.5;

/// Default Lissajous ratio (a circular orbit).
pub const DEFAULT_RATIO: Sample = 1.0;

/// Maximum Lissajous ratio of the `y` axis rate to the `x` axis rate.
pub const MAX_RATIO: Sample = 8.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Spatial frequency of the cross-term ripple in the terrain surface.
const RIDGES: Sample = 2.0;

/// Input scale feeding the output `tanh` soft-limiter.
const OUTPUT_DRIVE: Sample = 1.2;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Evaluates the terrain height at `(x, y)` for the given `warp`.
///
/// The result is bounded in `[-1, 1]` because it is a convex blend of two
/// bounded sinusoids.
#[inline]
fn terrain(x: Sample, y: Sample, warp: Sample) -> Sample {
    let swell = ops::sin(PI * (x + y));
    let ripple = ops::sin(TAU * RIDGES * x * y);
    (1.0 - warp) * swell + warp * ripple
}

/// Construction parameters for a [`WaveTerrainNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveTerrainParams {
    /// Orbit rate in hertz. Clamped to `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`.
    pub frequency_hz: Sample,
    /// Orbit radius. Clamped to `[MIN_RADIUS, MAX_RADIUS]`.
    pub radius: Sample,
    /// Normalized warp. Clamped to `[MIN_WARP, MAX_WARP]`.
    pub warp: Sample,
    /// Lissajous ratio. Clamped to `[MIN_RATIO, MAX_RATIO]`.
    pub ratio: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for WaveTerrainParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            radius: DEFAULT_RADIUS,
            warp: DEFAULT_WARP,
            ratio: DEFAULT_RATIO,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl WaveTerrainParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            radius: finite_or(self.radius, DEFAULT_RADIUS).clamp(MIN_RADIUS, MAX_RADIUS),
            warp: finite_or(self.warp, DEFAULT_WARP).clamp(MIN_WARP, MAX_WARP),
            ratio: finite_or(self.ratio, DEFAULT_RATIO).clamp(MIN_RATIO, MAX_RATIO),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A wave-terrain synthesis source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::WaveTerrainNode;
///
/// let mut node = WaveTerrainNode::new(110.0, 0.7, 0.5, 1.0, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.radius(), 0.7);
/// ```
#[derive(Debug, Clone)]
pub struct WaveTerrainNode {
    /// Smoothed orbit rate in hertz.
    frequency: Smoothed,
    /// Smoothed orbit radius.
    radius: Smoothed,
    /// Smoothed normalized warp (terrain blend).
    warp: Smoothed,
    /// Lissajous ratio of the `y` axis rate to the `x` axis rate.
    ratio: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Orbit phase on the `x` axis in `[0, 1)`.
    phase_x: Sample,
    /// Orbit phase on the `y` axis in `[0, 1)`.
    phase_y: Sample,
}

impl WaveTerrainNode {
    /// Creates a wave-terrain source.
    ///
    /// Non-finite inputs fall back to defaults; every value is clamped to its
    /// documented domain.
    #[must_use]
    pub fn new(
        frequency_hz: Sample,
        radius: Sample,
        warp: Sample,
        ratio: Sample,
        amplitude: Sample,
    ) -> Self {
        Self {
            frequency: Smoothed::new(
                finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                    .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            ),
            radius: Smoothed::new(
                finite_or(radius, DEFAULT_RADIUS).clamp(MIN_RADIUS, MAX_RADIUS),
            ),
            warp: Smoothed::new(finite_or(warp, DEFAULT_WARP).clamp(MIN_WARP, MAX_WARP)),
            ratio: finite_or(ratio, DEFAULT_RATIO).clamp(MIN_RATIO, MAX_RATIO),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase_x: 0.0,
            phase_y: 0.0,
        }
    }

    /// Builds a wave-terrain source from a [`WaveTerrainParams`] bundle.
    #[must_use]
    pub fn from_params(params: WaveTerrainParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.radius, p.warp, p.ratio, p.amplitude)
    }

    /// Sets a new target orbit rate in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.frequency.set_target(
            finite_or(hz, self.frequency.target()).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            ramp,
        );
    }

    /// Sets a new target orbit radius, gliding with `ramp`.
    #[inline]
    pub fn set_radius(&mut self, radius: Sample, ramp: Ramp) {
        self.radius.set_target(
            finite_or(radius, self.radius.target()).clamp(MIN_RADIUS, MAX_RADIUS),
            ramp,
        );
    }

    /// Sets a new target warp, gliding with `ramp`.
    #[inline]
    pub fn set_warp(&mut self, warp: Sample, ramp: Ramp) {
        self.warp.set_target(
            finite_or(warp, self.warp.target()).clamp(MIN_WARP, MAX_WARP),
            ramp,
        );
    }

    /// Sets a new Lissajous ratio (applied immediately; it only scales the
    /// `y` axis phase increment, so the change is click-free).
    #[inline]
    pub fn set_ratio(&mut self, ratio: Sample) {
        self.ratio = finite_or(ratio, self.ratio).clamp(MIN_RATIO, MAX_RATIO);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the target orbit rate in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency.target()
    }

    /// Returns the target orbit radius the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn radius(&self) -> Sample {
        self.radius.target()
    }

    /// Returns the target warp the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn warp(&self) -> Sample {
        self.warp.target()
    }

    /// Returns the Lissajous ratio.
    #[inline]
    #[must_use]
    pub fn ratio(&self) -> Sample {
        self.ratio
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample: samples the terrain at the current orbit
    /// position, then advances both orbit phases. `inv_sr` is the reciprocal of
    /// the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let freq = self.frequency.next_sample();
        let radius = self.radius.next_sample();
        let warp = self.warp.next_sample();
        let amp = self.amplitude.next_sample();

        let x = radius * ops::cos(TAU * self.phase_x);
        let y = radius * ops::sin(TAU * self.phase_y);
        let z = terrain(x, y, warp);

        let step = freq * inv_sr;
        self.phase_x += step;
        self.phase_x -= ops::floor(self.phase_x);
        self.phase_y += self.ratio * step;
        self.phase_y -= ops::floor(self.phase_y);

        amp * ops::tanh(z * OUTPUT_DRIVE)
    }
}

impl AudioNode for WaveTerrainNode {
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
        self.phase_x = 0.0;
        self.phase_y = 0.0;
        self.frequency = Smoothed::new(self.frequency.target());
        self.radius = Smoothed::new(self.radius.target());
        self.warp = Smoothed::new(self.warp.target());
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

    fn render(node: &mut WaveTerrainNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut WaveTerrainNode,
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
        for &freq in &[0.0, 110.0, 1_000.0, 4_000.0] {
            for &radius in &[0.0, 0.5, 1.0] {
                for &warp in &[0.0, 0.5, 1.0] {
                    for &ratio in &[0.5, 1.0, 8.0] {
                        let mut node = WaveTerrainNode::new(freq, radius, warp, ratio, 0.9);
                        let out = render(&mut node, SR, 4_096);
                        for &s in out.channel(0) {
                            assert!(
                                s.is_finite() && s.abs() <= 0.9 + 1e-3,
                                "freq={freq} radius={radius} warp={warp} ratio={ratio} s={s}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn silent_when_radius_zero() {
        // A zero-radius orbit sits at the flat surface center, where every
        // terrain term is zero.
        let mut node = WaveTerrainNode::new(220.0, 0.0, 0.5, 1.0, 0.8);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = WaveTerrainNode::new(
            DEFAULT_FREQUENCY_HZ,
            DEFAULT_RADIUS,
            DEFAULT_WARP,
            DEFAULT_RATIO,
            DEFAULT_AMPLITUDE,
        );
        let out = render(&mut node, SR, 8_192);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = WaveTerrainNode::new(180.0, 0.6, 0.4, 2.0, 0.8);
        let mut b = WaveTerrainNode::new(180.0, 0.6, 0.4, 2.0, 0.8);
        let oa = render(&mut a, SR, 8_192);
        let ob = render(&mut b, SR, 8_192);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = WaveTerrainNode::new(180.0, 0.6, 0.4, 2.0, 0.8);
        let first = render(&mut node, SR, 8_192);
        node.reset();
        let second = render(&mut node, SR, 8_192);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.25);
        let mut loud = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.5);
        let eq = energy(&render(&mut quiet, SR, 8_192));
        let el = energy(&render(&mut loud, SR, 8_192));
        assert!((el / eq - 4.0).abs() < 1e-2, "ratio={}", el / eq);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono_node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let mono = render(&mut mono_node, SR, 4_096);
        let mut stereo_node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let stereo = render_layout(&mut stereo_node, SR, 4_096, ChannelLayout::Stereo);
        let mut quad_node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let quad = render_layout(&mut quad_node, SR, 4_096, ChannelLayout::Quad);
        assert_eq!(mono.channel(0), stereo.channel(0));
        assert_eq!(mono.channel(0), stereo.channel(1));
        assert_eq!(mono.channel(0), quad.channel(0));
        assert_eq!(mono.channel(0), quad.channel(3));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        // State must not have advanced: a fresh render matches a fresh node.
        let a = render(&mut node, SR, 2_048);
        let mut fresh = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let b = render(&mut fresh, SR, 2_048);
        assert_eq!(a.channel(0), b.channel(0));
    }

    #[test]
    fn latency_is_zero() {
        let node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = WaveTerrainNode::new(330.0, 0.6, 0.7, 3.0, 0.6);
        assert_eq!(node.frequency_hz(), 330.0);
        assert_eq!(node.radius(), 0.6);
        assert_eq!(node.warp(), 0.7);
        assert_eq!(node.ratio(), 3.0);
        assert_eq!(node.amplitude(), 0.6);
    }

    #[test]
    fn default_params_in_domain() {
        let p = WaveTerrainParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.radius >= MIN_RADIUS && p.radius <= MAX_RADIUS);
        assert!(p.warp >= MIN_WARP && p.warp <= MAX_WARP);
        assert!(p.ratio >= MIN_RATIO && p.ratio <= MAX_RATIO);
    }

    #[test]
    fn from_params_matches_new() {
        let params = WaveTerrainParams {
            frequency_hz: 240.0,
            radius: 0.55,
            warp: 0.65,
            ratio: 2.5,
            amplitude: 0.75,
        };
        let mut via_params = WaveTerrainNode::from_params(params);
        let mut via_new = WaveTerrainNode::new(240.0, 0.55, 0.65, 2.5, 0.75);
        let a = render(&mut via_params, SR, 4_096);
        let b = render(&mut via_new, SR, 4_096);
        assert_eq!(a.channel(0), b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = WaveTerrainNode::new(1.0e9, 9.0, 9.0, 99.0, 0.5);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(node.radius(), MAX_RADIUS);
        assert_eq!(node.warp(), MAX_WARP);
        assert_eq!(node.ratio(), MAX_RATIO);
        let low = WaveTerrainNode::new(-50.0, -9.0, -9.0, -9.0, 0.5);
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(low.radius(), MIN_RADIUS);
        assert_eq!(low.warp(), MIN_WARP);
        assert_eq!(low.ratio(), MIN_RATIO);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = WaveTerrainNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NAN,
            Sample::NAN,
            Sample::NAN,
        );
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.radius(), DEFAULT_RADIUS);
        assert_eq!(node.warp(), DEFAULT_WARP);
        assert_eq!(node.ratio(), DEFAULT_RATIO);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        node.set_frequency_hz(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), 220.0);
        node.set_frequency_hz(1.0e9, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        node.set_radius(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.radius(), 0.7);
        node.set_radius(9.0, Ramp::Immediate);
        assert_eq!(node.radius(), MAX_RADIUS);
        node.set_warp(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.warp(), 0.5);
        node.set_warp(-9.0, Ramp::Immediate);
        assert_eq!(node.warp(), MIN_WARP);
        node.set_ratio(Sample::NAN);
        assert_eq!(node.ratio(), 1.0);
        node.set_ratio(99.0);
        assert_eq!(node.ratio(), MAX_RATIO);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = WaveTerrainParams {
            frequency_hz: 300.0,
            radius: 0.6,
            warp: 0.4,
            ratio: 3.0,
            amplitude: 0.7,
        };
        let s = params.sanitised();
        assert_eq!(s.frequency_hz, 300.0);
        assert_eq!(s.radius, 0.6);
        assert_eq!(s.warp, 0.4);
        assert_eq!(s.ratio, 3.0);
        assert_eq!(s.amplitude, 0.7);
    }

    #[test]
    fn higher_frequency_increases_high_frequency_energy() {
        let mut slow = WaveTerrainNode::new(80.0, 0.7, 0.5, 1.0, 0.8);
        let mut fast = WaveTerrainNode::new(2_000.0, 0.7, 0.5, 1.0, 0.8);
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
    fn larger_radius_increases_energy() {
        let mut small = WaveTerrainNode::new(220.0, 0.2, 0.5, 1.0, 0.8);
        let mut large = WaveTerrainNode::new(220.0, 0.9, 0.5, 1.0, 0.8);
        let es = energy(&render(&mut small, SR, 8_192));
        let el = energy(&render(&mut large, SR, 8_192));
        assert!(el > es, "small={es} large={el}");
    }

    #[test]
    fn higher_ratio_increases_high_frequency_energy() {
        let mut low = WaveTerrainNode::new(220.0, 0.8, 0.5, 1.0, 0.8);
        let mut high = WaveTerrainNode::new(220.0, 0.8, 0.5, 6.0, 0.8);
        let low_out = render(&mut low, SR, 16_384);
        let high_out = render(&mut high, SR, 16_384);
        assert!(
            hf_energy(&high_out) > hf_energy(&low_out),
            "low={} high={}",
            hf_energy(&low_out),
            hf_energy(&high_out)
        );
    }

    #[test]
    fn integer_ratio_is_periodic() {
        // 48000 / 375 = 128 samples per cycle exactly; a circular orbit over a
        // fixed surface repeats every fundamental period.
        let period = 128;
        let mut node = WaveTerrainNode::new(375.0, 0.7, 0.5, 1.0, 0.8);
        let out = render(&mut node, SR, 4 * period);
        let ch = out.channel(0);
        for n in 0..(3 * period) {
            assert!(
                (ch[n] - ch[n + period]).abs() < 1e-5,
                "n={n} a={} b={}",
                ch[n],
                ch[n + period]
            );
        }
    }

    #[test]
    fn warp_changes_alter_output() {
        let mut swell = WaveTerrainNode::new(220.0, 0.8, 0.0, 1.0, 0.8);
        let mut ripple = WaveTerrainNode::new(220.0, 0.8, 1.0, 1.0, 0.8);
        let a = render(&mut swell, SR, 8_192);
        let b = render(&mut ripple, SR, 8_192);
        let differ = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differ > a.channel(0).len() / 10, "differ={differ}");
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = WaveTerrainNode::new(MAX_FREQUENCY_HZ, MAX_RADIUS, MAX_WARP, MAX_RATIO, 0.9);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 0.9 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn frequency_change_is_click_free() {
        let mut node = WaveTerrainNode::new(220.0, 0.7, 0.5, 1.0, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.set_frequency_hz(660.0, Ramp::Immediate);
        let second = render(&mut node, SR, 4_096);
        let last = *first.channel(0).last().unwrap();
        let next = second.channel(0)[0];
        assert!((next - last).abs() < 0.2, "join step {}", (next - last).abs());
    }
}
