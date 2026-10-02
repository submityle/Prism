//! Scanned-synthesis source node (a slowly evolving mass-spring lattice read as
//! a live wavetable).
//!
//! [`ScannedSynthesisNode`] couples a ring of `N` masses with springs into a
//! dynamical system whose displacement profile is treated as a single-cycle
//! wavetable. The lattice evolves slowly in its own "scan time" under spring,
//! centering, damping, and a gentle sustaining drive force, while a phase
//! accumulator scans the instantaneous displacement profile at audio rate to
//! set the musical pitch. Because the table keeps morphing, the scanned tone is
//! pitched yet never static: it breathes and drifts the way a bowed or struck
//! resonant body does, but the slow lattice "boil" is fully decoupled from the
//! audio pitch.
//!
//! # Model
//!
//! The state is a circular chain of `N` mass displacements `pos[i]` and
//! velocities `vel[i]`. Each scan step evaluates the discrete Laplacian of the
//! ring (nearest-neighbor spring coupling), a weak centering spring that pulls
//! every mass back toward equilibrium, viscous damping proportional to
//! velocity, and a slow sinusoidal drive injected at one node to replace the
//! energy that damping removes:
//!
//! ```text
//!   lap[i]   = pos[i-1] - 2*pos[i] + pos[i+1]      (indices wrap mod N)
//!   accel[i] = tension*lap[i] - CENTERING*pos[i] - damp*vel[i] (+ drive at i=0)
//!   vel[i]  += accel[i] * dt
//!   pos[i]  += vel[i]   * dt
//! ```
//!
//! This is the classic semi-implicit (symplectic) Euler update: velocities are
//! advanced from the old positions, then positions from the new velocities,
//! which is markedly more stable for oscillatory systems than plain forward
//! Euler. The step size `dt = scan_rate * SCAN_TO_DT / sample_rate` (clamped to
//! `DT_MAX`) is tiny, so the lattice modes oscillate at only tens of hertz in
//! real time: the table evolves slowly relative to the audio sample rate. The
//! `scan_rate` control sets both this evolution speed and the frequency of the
//! sustaining drive oscillator, so a low `scan_rate` yields a slow, glassy boil
//! and a higher one a faster, more agitated morph. At `scan_rate = 0` the
//! lattice freezes and the node behaves as a fixed wavetable oscillator.
//!
//! The output reads the displacement profile with periodic linear
//! interpolation at the readout `phase` and soft-limits it:
//!
//! ```text
//!   s   = lerp(pos[i0], pos[i0+1], frac)
//!   out = amplitude * tanh(s * OUTPUT_DRIVE)
//!   phase += frequency / sample_rate     (wrapped into [0, 1))
//! ```
//!
//! `brightness` maps to the spring tension (stiffer coupling propagates sharper
//! spatial features, so the waveform carries more harmonics) and `damping` maps
//! to the viscous loss (low damping lets the lattice ring and evolve richly,
//! high damping settles it toward a smooth shape). The `tanh` keeps the signal
//! strictly bounded and only `amplitude` scales it, so output level is a clean
//! square-law of `amplitude`.
//!
//! # Relationship
//!
//! - Unlike [`super::wavetable_oscillator::WavetableOscillatorNode`], whose
//!   octave-mipmap tables are precomputed and band-limited, this node scans a
//!   *live* table that cannot be pre-mipmapped, so it reads a modest `N`-point
//!   table with linear interpolation and is intentionally not band-limited; the
//!   lattice's damping and coupling smooth the profile, which keeps aliasing
//!   mild, and the tradeoff buys a continuously evolving timbre.
//! - Unlike [`super::additive_oscillator::AdditiveOscillatorNode`], which sums a
//!   fixed set of harmonics, the harmonic content here emerges from and drifts
//!   with the lattice dynamics rather than being specified directly.
//! - Unlike [`super::lorenz_attractor::LorenzAttractorNode`] and
//!   [`super::chaotic_oscillator::ChaoticOscillatorNode`], which sonify a single
//!   state variable of a nonlinear system and are respectively aperiodic or
//!   pitched-but-noisy, this node reads the whole *spatial* displacement profile
//!   as one cycle scanned at a definite `frequency`, so it is cleanly pitched.
//! - Unlike struck/plucked physical models such as
//!   [`super::struck_bar::StruckBarNode`], whose excitation and resonator form a
//!   single audio-rate feedback loop that rings and decays, here the dynamical
//!   lattice runs at a slow sub-audio scan rate fully decoupled from the audio
//!   pitch and is continuously re-energized, so the tone sustains and morphs
//!   instead of decaying.
//!
//! # Real-time contract
//!
//! All state lives in fixed-size arrays sized at compile time, so
//! [`ScannedSynthesisNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Every scan step clamps the
//! lattice to a generous bounding box and reseeds the initial profile if a
//! non-finite value or runaway excursion ever appears, so the output can never
//! blow up. `frequency`, `brightness`, `damping`, and `amplitude` glide through
//! [`Smoothed`] values and `scan_rate` only scales the timestep and drive, so
//! automation never produces zipper clicks. Two nodes built with the same
//! parameters produce bit-identical output, and [`ScannedSynthesisNode::reset`]
//! restarts the exact same evolution.
//!
//! # Provenance
//!
//! Implemented from first principles. The scanned-synthesis concept (slowly
//! evolving a dynamical system and reading its state as a wavetable) is the
//! public-domain idea developed by Bill Verplank, Max Mathews, and Rob Shaw;
//! the mass-spring lattice is public-domain classical mechanics and the
//! semi-implicit (symplectic) Euler integrator is from public-domain numerical
//! analysis. Only those shared mathematical ideas are used. This file contains
//! no code, data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, Google Resonance Audio, the Web Audio API, the Synthesis
//! Toolkit, or any other audio engine or toolkit; only the shared mathematical
//! ideas are used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};
use bevy_math::ops;
use core::f32::consts::TAU;

/// Number of masses in the lattice ring (and samples in the scanned table).
///
/// A power of two so the circular index wrap is a cheap bitmask.
pub const TABLE_SIZE: usize = 64;

/// Bitmask for wrapping a lattice index into `[0, TABLE_SIZE)`.
const MASK: usize = TABLE_SIZE - 1;

/// Minimum readout frequency in hertz (direct current, a frozen scan).
pub const MIN_FREQUENCY_HZ: Sample = 0.0;

/// Default readout frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum readout frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum lattice scan rate in hertz (a frozen table).
pub const MIN_SCAN_RATE_HZ: Sample = 0.0;

/// Default lattice scan rate in hertz (a slow, glassy boil).
pub const DEFAULT_SCAN_RATE_HZ: Sample = 5.0;

/// Maximum lattice scan rate in hertz (a fast, agitated morph).
pub const MAX_SCAN_RATE_HZ: Sample = 50.0;

/// Minimum normalized brightness (softest spring coupling).
pub const MIN_BRIGHTNESS: Sample = 0.0;

/// Default normalized brightness.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Maximum normalized brightness (stiffest spring coupling).
pub const MAX_BRIGHTNESS: Sample = 1.0;

/// Minimum normalized damping (lattice rings longest).
pub const MIN_DAMPING: Sample = 0.0;

/// Default normalized damping.
pub const DEFAULT_DAMPING: Sample = 0.3;

/// Maximum normalized damping (lattice settles fastest).
pub const MAX_DAMPING: Sample = 1.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Weak centering spring constant pulling each mass toward equilibrium.
const CENTERING: Sample = 0.1;

/// Spring tension (coupling) at `brightness = 0`.
const TENSION_MIN: Sample = 0.05;

/// Spring tension (coupling) at `brightness = 1`.
const TENSION_MAX: Sample = 0.6;

/// Viscous damping coefficient at `damping = 0` (small but strictly positive so
/// the driven lattice stays bounded).
const DAMP_MIN: Sample = 0.01;

/// Viscous damping coefficient at `damping = 1`.
const DAMP_MAX: Sample = 0.15;

/// Scales the `scan_rate` (hertz) into the per-sample integration timestep.
const SCAN_TO_DT: Sample = 10.0;

/// Hard upper bound on the integration timestep for stability.
const DT_MAX: Sample = 0.05;

/// Peak magnitude of the sustaining drive force injected at the drive node.
const DRIVE_AMOUNT: Sample = 0.3;

/// Lattice index that receives the sustaining drive force.
const DRIVE_INDEX: usize = 0;

/// Input scale feeding the output `tanh` soft-limiter.
const OUTPUT_DRIVE: Sample = 1.5;

/// Generous bounding box; a mass leaving it (or going non-finite) is treated as
/// numerical runaway and the whole lattice is reseeded.
const SAFETY_BOUND: Sample = 60.0;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Linear interpolation between `a` and `b` by `t` in `[0, 1]`.
#[inline]
fn lerp(a: Sample, b: Sample, t: Sample) -> Sample {
    a + (b - a) * t
}

/// Fills `pos` with the deterministic initial displacement profile: a
/// raised-cosine bump over the first quarter of the ring, made zero-mean so the
/// frozen waveform has no direct-current offset.
fn seed_profile(pos: &mut [Sample; TABLE_SIZE]) {
    let width = TABLE_SIZE / 4;
    let mut sum = 0.0;
    let mut i = 0;
    while i < TABLE_SIZE {
        let v = if i < width {
            0.5 * (1.0 - ops::cos(TAU * i as Sample / width as Sample))
        } else {
            0.0
        };
        pos[i] = v;
        sum += v;
        i += 1;
    }
    let mean = sum / TABLE_SIZE as Sample;
    let mut j = 0;
    while j < TABLE_SIZE {
        pos[j] -= mean;
        j += 1;
    }
}

/// Construction parameters for a [`ScannedSynthesisNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ScannedSynthesisParams {
    /// Readout frequency in hertz. Clamped to `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`.
    pub frequency_hz: Sample,
    /// Lattice scan rate in hertz. Clamped to `[MIN_SCAN_RATE_HZ, MAX_SCAN_RATE_HZ]`.
    pub scan_rate_hz: Sample,
    /// Normalized brightness. Clamped to `[MIN_BRIGHTNESS, MAX_BRIGHTNESS]`.
    pub brightness: Sample,
    /// Normalized damping. Clamped to `[MIN_DAMPING, MAX_DAMPING]`.
    pub damping: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for ScannedSynthesisParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            scan_rate_hz: DEFAULT_SCAN_RATE_HZ,
            brightness: DEFAULT_BRIGHTNESS,
            damping: DEFAULT_DAMPING,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl ScannedSynthesisParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            scan_rate_hz: finite_or(self.scan_rate_hz, DEFAULT_SCAN_RATE_HZ)
                .clamp(MIN_SCAN_RATE_HZ, MAX_SCAN_RATE_HZ),
            brightness: finite_or(self.brightness, DEFAULT_BRIGHTNESS)
                .clamp(MIN_BRIGHTNESS, MAX_BRIGHTNESS),
            damping: finite_or(self.damping, DEFAULT_DAMPING).clamp(MIN_DAMPING, MAX_DAMPING),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A scanned-synthesis source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::ScannedSynthesisNode;
///
/// let mut node = ScannedSynthesisNode::new(110.0, 5.0, 0.5, 0.3, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.scan_rate_hz(), 5.0);
/// ```
#[derive(Debug, Clone)]
pub struct ScannedSynthesisNode {
    /// Smoothed readout frequency in hertz.
    frequency: Smoothed,
    /// Lattice scan rate in hertz; scales the timestep and drive oscillator.
    scan_rate_hz: Sample,
    /// Smoothed normalized brightness (mapped to spring tension).
    brightness: Smoothed,
    /// Smoothed normalized damping (mapped to viscous loss).
    damping: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Mass displacements (the live scanned table).
    pos: [Sample; TABLE_SIZE],
    /// Mass velocities.
    vel: [Sample; TABLE_SIZE],
    /// Readout phase in `[0, 1)`.
    phase: Sample,
    /// Slow drive-oscillator phase in `[0, 1)`.
    drive_phase: Sample,
}

impl ScannedSynthesisNode {
    /// Creates a scanned-synthesis source.
    ///
    /// Non-finite inputs fall back to defaults; every value is clamped to its
    /// documented domain.
    #[must_use]
    pub fn new(
        frequency_hz: Sample,
        scan_rate_hz: Sample,
        brightness: Sample,
        damping: Sample,
        amplitude: Sample,
    ) -> Self {
        let mut pos = [0.0; TABLE_SIZE];
        seed_profile(&mut pos);
        Self {
            frequency: Smoothed::new(
                finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                    .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            ),
            scan_rate_hz: finite_or(scan_rate_hz, DEFAULT_SCAN_RATE_HZ)
                .clamp(MIN_SCAN_RATE_HZ, MAX_SCAN_RATE_HZ),
            brightness: Smoothed::new(
                finite_or(brightness, DEFAULT_BRIGHTNESS).clamp(MIN_BRIGHTNESS, MAX_BRIGHTNESS),
            ),
            damping: Smoothed::new(
                finite_or(damping, DEFAULT_DAMPING).clamp(MIN_DAMPING, MAX_DAMPING),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            pos,
            vel: [0.0; TABLE_SIZE],
            phase: 0.0,
            drive_phase: 0.0,
        }
    }

    /// Builds a scanned-synthesis source from a [`ScannedSynthesisParams`] bundle.
    #[must_use]
    pub fn from_params(params: ScannedSynthesisParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.scan_rate_hz, p.brightness, p.damping, p.amplitude)
    }

    /// Sets a new target readout frequency in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.frequency.set_target(
            finite_or(hz, self.frequency.target()).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            ramp,
        );
    }

    /// Sets a new lattice scan rate in hertz (applied immediately; only the
    /// timestep and drive scale, so the change is click-free).
    #[inline]
    pub fn set_scan_rate_hz(&mut self, hz: Sample) {
        self.scan_rate_hz =
            finite_or(hz, self.scan_rate_hz).clamp(MIN_SCAN_RATE_HZ, MAX_SCAN_RATE_HZ);
    }

    /// Sets a new target brightness, gliding with `ramp`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample, ramp: Ramp) {
        self.brightness.set_target(
            finite_or(brightness, self.brightness.target()).clamp(MIN_BRIGHTNESS, MAX_BRIGHTNESS),
            ramp,
        );
    }

    /// Sets a new target damping, gliding with `ramp`.
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

    /// Returns the target readout frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency.target()
    }

    /// Returns the lattice scan rate in hertz.
    #[inline]
    #[must_use]
    pub fn scan_rate_hz(&self) -> Sample {
        self.scan_rate_hz
    }

    /// Returns the target brightness the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness.target()
    }

    /// Returns the target damping the node is gliding toward.
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

    /// Advances the lattice by one semi-implicit Euler step of size `dt`,
    /// reseeding the whole profile on any non-finite value or runaway.
    #[inline]
    fn integrate_step(&mut self, dt: Sample, tension: Sample, damp: Sample, drive: Sample) {
        // Advance velocities from the current positions.
        let mut i = 0;
        while i < TABLE_SIZE {
            let left = self.pos[(i + MASK) & MASK];
            let right = self.pos[(i + 1) & MASK];
            let lap = left - 2.0 * self.pos[i] + right;
            let mut accel = tension * lap - CENTERING * self.pos[i] - damp * self.vel[i];
            if i == DRIVE_INDEX {
                accel += drive;
            }
            self.vel[i] += accel * dt;
            i += 1;
        }

        // Advance positions from the new velocities, watching for runaway.
        let mut runaway = false;
        let mut j = 0;
        while j < TABLE_SIZE {
            let p = self.pos[j] + self.vel[j] * dt;
            if !p.is_finite() || p.abs() > SAFETY_BOUND {
                runaway = true;
                break;
            }
            self.pos[j] = p;
            j += 1;
        }

        if runaway {
            seed_profile(&mut self.pos);
            self.vel = [0.0; TABLE_SIZE];
        }
    }

    /// Produces one output sample: evolves the lattice one step, then scans the
    /// displacement profile at the readout phase. `inv_sr` is the reciprocal of
    /// the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let freq = self.frequency.next_sample();
        let brightness = self.brightness.next_sample();
        let damping = self.damping.next_sample();
        let amp = self.amplitude.next_sample();

        let tension = TENSION_MIN + brightness * (TENSION_MAX - TENSION_MIN);
        let damp = DAMP_MIN + damping * (DAMP_MAX - DAMP_MIN);
        let dt = (self.scan_rate_hz * SCAN_TO_DT * inv_sr).min(DT_MAX);

        // Slow sustaining drive, phase-locked to the scan rate.
        self.drive_phase += self.scan_rate_hz * inv_sr;
        self.drive_phase -= ops::floor(self.drive_phase);
        let drive = DRIVE_AMOUNT * ops::sin(TAU * self.drive_phase);

        self.integrate_step(dt, tension, damp, drive);

        // Scan the profile with periodic linear interpolation.
        let idx = self.phase * TABLE_SIZE as Sample;
        let i0f = ops::floor(idx);
        let frac = idx - i0f;
        let i0 = (i0f as usize) & MASK;
        let i1 = (i0 + 1) & MASK;
        let s = lerp(self.pos[i0], self.pos[i1], frac);

        self.phase += freq * inv_sr;
        self.phase -= ops::floor(self.phase);

        amp * ops::tanh(s * OUTPUT_DRIVE)
    }
}

impl AudioNode for ScannedSynthesisNode {
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
        seed_profile(&mut self.pos);
        self.vel = [0.0; TABLE_SIZE];
        self.phase = 0.0;
        self.drive_phase = 0.0;
        self.frequency = Smoothed::new(self.frequency.target());
        self.brightness = Smoothed::new(self.brightness.target());
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

    fn render(node: &mut ScannedSynthesisNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut ScannedSynthesisNode,
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
            for &scan in &[0.0, 5.0, 50.0] {
                for &bright in &[0.0, 0.5, 1.0] {
                    let mut node = ScannedSynthesisNode::new(freq, scan, bright, 0.3, 0.9);
                    let out = render(&mut node, SR, 8_192);
                    for &s in out.channel(0) {
                        assert!(
                            s.is_finite() && s.abs() <= 0.9 + 1e-3,
                            "freq={freq} scan={scan} bright={bright} s={s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = ScannedSynthesisNode::new(
            DEFAULT_FREQUENCY_HZ,
            DEFAULT_SCAN_RATE_HZ,
            DEFAULT_BRIGHTNESS,
            DEFAULT_DAMPING,
            DEFAULT_AMPLITUDE,
        );
        let out = render(&mut node, SR, 8_192);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ScannedSynthesisNode::new(180.0, 7.0, 0.6, 0.2, 0.8);
        let mut b = ScannedSynthesisNode::new(180.0, 7.0, 0.6, 0.2, 0.8);
        let oa = render(&mut a, SR, 8_192);
        let ob = render(&mut b, SR, 8_192);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = ScannedSynthesisNode::new(180.0, 7.0, 0.6, 0.2, 0.8);
        let first = render(&mut node, SR, 8_192);
        node.reset();
        let second = render(&mut node, SR, 8_192);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.25);
        let mut loud = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.5);
        let eq = energy(&render(&mut quiet, SR, 8_192));
        let el = energy(&render(&mut loud, SR, 8_192));
        assert!((el / eq - 4.0).abs() < 1e-2, "ratio={}", el / eq);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono_node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let mono = render(&mut mono_node, SR, 4_096);
        let mut stereo_node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let stereo = render_layout(&mut stereo_node, SR, 4_096, ChannelLayout::Stereo);
        let mut quad_node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let quad = render_layout(&mut quad_node, SR, 4_096, ChannelLayout::Quad);
        assert_eq!(mono.channel(0), stereo.channel(0));
        assert_eq!(mono.channel(0), stereo.channel(1));
        assert_eq!(mono.channel(0), quad.channel(0));
        assert_eq!(mono.channel(0), quad.channel(3));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        // State must not have advanced: a fresh render matches a fresh node.
        let a = render(&mut node, SR, 2_048);
        let mut fresh = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let b = render(&mut fresh, SR, 2_048);
        assert_eq!(a.channel(0), b.channel(0));
    }

    #[test]
    fn latency_is_zero() {
        let node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = ScannedSynthesisNode::new(330.0, 12.0, 0.7, 0.4, 0.6);
        assert_eq!(node.frequency_hz(), 330.0);
        assert_eq!(node.scan_rate_hz(), 12.0);
        assert_eq!(node.brightness(), 0.7);
        assert_eq!(node.damping(), 0.4);
        assert_eq!(node.amplitude(), 0.6);
    }

    #[test]
    fn default_params_in_domain() {
        let p = ScannedSynthesisParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.scan_rate_hz >= MIN_SCAN_RATE_HZ && p.scan_rate_hz <= MAX_SCAN_RATE_HZ);
        assert!(p.brightness >= MIN_BRIGHTNESS && p.brightness <= MAX_BRIGHTNESS);
        assert!(p.damping >= MIN_DAMPING && p.damping <= MAX_DAMPING);
    }

    #[test]
    fn from_params_matches_new() {
        let params = ScannedSynthesisParams {
            frequency_hz: 240.0,
            scan_rate_hz: 9.0,
            brightness: 0.65,
            damping: 0.25,
            amplitude: 0.75,
        };
        let mut via_params = ScannedSynthesisNode::from_params(params);
        let mut via_new = ScannedSynthesisNode::new(240.0, 9.0, 0.65, 0.25, 0.75);
        let a = render(&mut via_params, SR, 4_096);
        let b = render(&mut via_new, SR, 4_096);
        assert_eq!(a.channel(0), b.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = ScannedSynthesisNode::new(1.0e9, 1.0e9, 9.0, 9.0, 0.5);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(node.scan_rate_hz(), MAX_SCAN_RATE_HZ);
        assert_eq!(node.brightness(), MAX_BRIGHTNESS);
        assert_eq!(node.damping(), MAX_DAMPING);
        let low = ScannedSynthesisNode::new(-50.0, -50.0, -9.0, -9.0, 0.5);
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(low.scan_rate_hz(), MIN_SCAN_RATE_HZ);
        assert_eq!(low.brightness(), MIN_BRIGHTNESS);
        assert_eq!(low.damping(), MIN_DAMPING);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = ScannedSynthesisNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NAN,
            Sample::NAN,
            Sample::NAN,
        );
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.scan_rate_hz(), DEFAULT_SCAN_RATE_HZ);
        assert_eq!(node.brightness(), DEFAULT_BRIGHTNESS);
        assert_eq!(node.damping(), DEFAULT_DAMPING);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        node.set_frequency_hz(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), 220.0);
        node.set_frequency_hz(1.0e9, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        node.set_scan_rate_hz(Sample::NAN);
        assert_eq!(node.scan_rate_hz(), 5.0);
        node.set_scan_rate_hz(1.0e9);
        assert_eq!(node.scan_rate_hz(), MAX_SCAN_RATE_HZ);
        node.set_brightness(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.brightness(), 0.5);
        node.set_brightness(9.0, Ramp::Immediate);
        assert_eq!(node.brightness(), MAX_BRIGHTNESS);
        node.set_damping(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.damping(), 0.3);
        node.set_damping(-9.0, Ramp::Immediate);
        assert_eq!(node.damping(), MIN_DAMPING);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn higher_frequency_increases_high_frequency_energy() {
        let mut slow = ScannedSynthesisNode::new(80.0, 5.0, 0.5, 0.3, 0.8);
        let mut fast = ScannedSynthesisNode::new(2_000.0, 5.0, 0.5, 0.3, 0.8);
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
    fn frozen_scan_rate_is_periodic() {
        // 48000 / 375 = 128 samples per cycle exactly; a frozen table repeats.
        let period = 128;
        let mut node = ScannedSynthesisNode::new(375.0, 0.0, 0.5, 0.3, 0.8);
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
    fn evolving_scan_rate_breaks_periodicity() {
        let period = 128;
        let mut node = ScannedSynthesisNode::new(375.0, 20.0, 0.5, 0.05, 0.8);
        let out = render(&mut node, SR, 64 * period);
        let ch = out.channel(0);
        // Compare an early cycle to a much later one: the morph makes them differ.
        let early = &ch[period..2 * period];
        let late = &ch[60 * period..61 * period];
        let diff: Sample = early
            .iter()
            .zip(late)
            .map(|(a, b)| (a - b).abs())
            .sum::<Sample>()
            / period as Sample;
        assert!(diff > 1e-3, "diff={diff}");
    }

    #[test]
    fn drive_sustains_energy() {
        let mut node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        // Discard a long warm-up, then confirm the tone has not died away.
        let _ = render(&mut node, SR, 48_000);
        let out = render(&mut node, SR, 8_192);
        assert!(energy(&out) > 0.5, "sustained energy={}", energy(&out));
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let mut node = ScannedSynthesisNode::new(MAX_FREQUENCY_HZ, MAX_SCAN_RATE_HZ, 1.0, 0.0, 0.9);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 0.9 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn frequency_change_is_click_free() {
        let mut node = ScannedSynthesisNode::new(220.0, 5.0, 0.5, 0.3, 0.8);
        let first = render(&mut node, SR, 4_096);
        node.set_frequency_hz(660.0, Ramp::Immediate);
        let second = render(&mut node, SR, 4_096);
        let last = *first.channel(0).last().unwrap();
        let next = second.channel(0)[0];
        assert!((next - last).abs() < 0.2, "join step {}", (next - last).abs());
    }

    #[test]
    fn brightness_changes_alter_output() {
        let mut dark = ScannedSynthesisNode::new(220.0, 10.0, 0.0, 0.1, 0.8);
        let mut bright = ScannedSynthesisNode::new(220.0, 10.0, 1.0, 0.1, 0.8);
        let a = render(&mut dark, SR, 8_192);
        let b = render(&mut bright, SR, 8_192);
        let differ = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differ > a.channel(0).len() / 10, "differ={differ}");
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = ScannedSynthesisParams {
            frequency_hz: 300.0,
            scan_rate_hz: 20.0,
            brightness: 0.6,
            damping: 0.4,
            amplitude: 0.7,
        };
        let s = params.sanitised();
        assert_eq!(s.frequency_hz, 300.0);
        assert_eq!(s.scan_rate_hz, 20.0);
        assert_eq!(s.brightness, 0.6);
        assert_eq!(s.damping, 0.4);
        assert_eq!(s.amplitude, 0.7);
    }
}
