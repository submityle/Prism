//! Helmholtz blown-bottle resonator physical-modeling source node.
//!
//! [`HelmholtzResonatorNode`] is a *source* (zero inputs, one output) that
//! synthesizes the breathy, near-sinusoidal tone of a blown bottle, ocarina, or
//! vessel flute (the Helmholtz-resonator family). Like its wind siblings
//! [`super::air_jet_flute::AirJetFluteNode`],
//! [`super::bowed_string::BowedStringNode`], and
//! [`super::reed_woodwind::ReedWoodwindNode`] it is *continuously driven*: a
//! steady breath pumps energy into the resonance every sample, so the tone
//! sustains for as long as the player blows. Unlike those delay-line waveguides,
//! though, the pitch here is set by the *vessel*, not by a travelling-wave
//! round trip.
//!
//! # Model: a lumped Helmholtz mode with a Van der Pol self-oscillation
//!
//! A bottle (or ocarina) is a *Helmholtz resonator*: a lump of air in the neck
//! springs against the compliant volume of air in the body, giving a single
//! low-order resonance rather than a harmonic series of bore modes. Acoustically
//! it behaves as one damped mass-spring oscillator at
//! `f0 = (c / 2*pi) * sqrt(A / (V * L))`, exactly the lumped-element picture in
//! Fletcher and Rossing's acoustics. We model that single mode directly with a
//! Chamberlin state-variable filter (two integrators carrying the band-pass
//! state `bp` and low-pass state `lp`) tuned to `f0`:
//!
//! ```text
//! f     = 2 * sin(pi * f0 / sample_rate)      (SVF frequency coefficient)
//! hp    = drive - lp - damp * bp              (band-pass "velocity" input)
//! bp   += f * hp
//! lp   += f * bp
//! ```
//!
//! The air jet grazing the vessel's lip is an *edge tone*: it feeds energy back
//! into the resonance in phase, i.e. it acts as a *negative* resistance. The
//! classic compact description of any such edge-/jet-sustained near-sinusoidal
//! oscillator is the Van der Pol equation, whose cubic term pumps energy at
//! small amplitude and removes it at large amplitude so the oscillation settles
//! into a stable limit cycle:
//!
//! ```text
//! nl    = mu_eff * (1 - (bp*bp) / (v_sat*v_sat)) * bp
//! drive = nl + breath * noise_amount * turbulence
//! ```
//!
//! where `mu_eff = mu * breath` makes the breath pressure control the negative
//! resistance. Below the threshold `breath < damp / mu` the net damping is
//! positive and the vessel stays quiet; above it the oscillation grows until the
//! cubic saturates it, so loudness and onset follow the breath exactly as a real
//! bottle "speaks" only once you blow hard enough.
//!
//! # Pitch: pinned by the vessel, not the breath
//!
//! Because the resonance is a *lumped* mode rather than a delay-line round trip,
//! the sounding pitch is the resonator's `f0` and does not drift with breath
//! pressure or jet geometry: blowing a bottle harder makes it louder and
//! brighter, not sharper. This is the acoustic opposite of the open-bore
//! [`super::air_jet_flute::AirJetFluteNode`], whose pitch *is* the delay-line
//! length. A fixed reference-frequency compensation (`comp`) scales `mu` and the
//! damping so the onset time stays roughly constant across the pitch range
//! instead of racing at high `f0` and crawling at low `f0`.
//!
//! # Timbre and brightness
//!
//! At rest the tone is nearly a pure sine (a single mode), matching the hollow,
//! flute-like quality of a blown bottle. The `brightness` control blends in a
//! `tanh` waveshaper applied to the band-pass state *outside* the resonance
//! loop, which adds odd harmonics (3rd, 5th, ...) without introducing even
//! harmonics or DC and without affecting loop stability. `breath_noise` mixes a
//! deterministic turbulence stream (scaled by the breath) into the drive for the
//! airy breathiness of a real vessel flute.
//!
//! # Determinism
//!
//! The breath turbulence is a self-contained seeded `xorshift64` stream, so two
//! [`HelmholtzResonatorNode`]s built with the same sample rate, parameters, and
//! seed emit bit-identical streams on every platform via [`bevy_math::ops`]. A
//! tiny deterministic initial band-pass displacement guarantees the oscillator
//! self-starts even with the turbulence disabled, and [`AudioNode::reset`]
//! restores that displacement, clears the filter, and reseeds the generator to
//! restart the identical attack.
//!
//! # Real-time contract
//!
//! Construction sizes nothing dynamically and `process` allocates nothing, takes
//! no locks, and never panics. Every state update is a handful of multiplies and
//! adds, denormals are flushed, and the clamped cubic plus positive baseline
//! damping bound the limit cycle so the output stays finite and within `[-1, 1]`
//! across the whole parameter grid.
//!
//! # Relationship
//!
//! This is the *lumped-resonance* member of the breath-driven source family. It
//! differs fundamentally from [`super::air_jet_flute::AirJetFluteNode`], which is
//! a *distributed* bidirectional waveguide whose delay line both sets the pitch
//! and gives a full harmonic series; here a single Chamberlin resonance pins a
//! near-sinusoidal pitch that breath cannot bend. It differs from the
//! delay-loop-driven [`super::bowed_string::BowedStringNode`] and
//! [`super::reed_woodwind::ReedWoodwindNode`] for the same reason: those sustain
//! a tuned waveguide via friction or a reed valve, whereas this sustains a
//! single lumped mode via a Van der Pol negative resistance. Unlike the
//! struck/plucked modal voices [`super::struck_bar::StruckBarNode`] and
//! [`super::membrane_drum::MembraneDrumNode`], which ring down after an impulse,
//! this voice is continuously blown. And unlike the geometric-waveform
//! [`super::oscillator::OscillatorNode`], the tone emerges from a physically
//! driven resonance rather than a sampled wave shape.
//!
//! # Provenance
//!
//! The design borrows only *ideas* from the public-domain acoustics and
//! signal-processing literature: the lumped Helmholtz-resonator model of a
//! blown vessel (Fletcher and Rossing, *The Physics of Musical Instruments*);
//! the Van der Pol self-oscillator (Van der Pol, 1920s) as the compact model of
//! an edge-/jet-sustained negative-resistance oscillation; the Chamberlin
//! state-variable filter topology; and public-domain `tanh` waveshaping for odd
//! harmonics. No source code or derivative of any audio engine or toolkit was
//! consulted or copied, including Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, Google Resonance Audio, the Web Audio API, or STK. This is a
//! pure classical-DSP implementation with no AI or machine-learning component.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable resonance frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 50.0;

/// Highest tunable resonance frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Default resonance (pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Default normalized breath pressure in `[0, 1]` (loudness / drive).
pub const DEFAULT_BREATH_PRESSURE: Sample = 0.5;

/// Default normalized resonance in `[0, 1]` (vessel Q / ring).
pub const DEFAULT_RESONANCE: Sample = 0.6;

/// Default brightness in `[0, 1]` (odd-harmonic shaping).
pub const DEFAULT_BRIGHTNESS: Sample = 0.3;

/// Default normalized breath turbulence in `[0, 1]` (breathiness).
pub const DEFAULT_BREATH_NOISE: Sample = 0.25;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default PRNG seed for the breath-turbulence stream.
pub const DEFAULT_SEED: u64 = 0x8807_7E50_11CE_A5EF;

/// Negative-resistance gain of the Van der Pol pump, relative to the frequency
/// compensation. Sized so a default breath comfortably exceeds the threshold
/// while the cubic still saturates well below unit amplitude.
const MU_RATIO: Sample = 0.025;

/// Baseline SVF damping at `resonance = 0` (lowest Q, hardest to speak).
const DAMP_MAX: Sample = 0.016;

/// SVF damping at `resonance = 1` (highest Q, easiest to speak, longest ring).
const DAMP_MIN: Sample = 0.0015;

/// Band-pass amplitude at which the Van der Pol cubic cancels the pump, setting
/// the limit-cycle size.
const V_SAT: Sample = 0.7;

/// Scales `breath_noise` into the turbulence amplitude mixed into the drive.
const NOISE_SCALE: Sample = 0.08;

/// Normalizes the limit-cycle band-pass amplitude to a safe peak before the user
/// amplitude so a default voice stays well within `[-1, 1]`.
const OUTPUT_GAIN: Sample = 0.5;

/// Reference frequency (hertz) at which the frequency compensation is unity.
const F_REF: Sample = 440.0;

/// Tiny deterministic initial band-pass displacement so the oscillator
/// self-starts even when `breath_noise` is zero. Negative net damping grows it
/// to the limit cycle; positive net damping (breath below threshold) decays it.
const BP_INIT: Sample = 1.0e-4;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Clamps `frequency_hz` to `[MIN_FREQUENCY_HZ, min(MAX_FREQUENCY_HZ, sr/2)]`,
/// falling back to [`DEFAULT_FREQUENCY_HZ`] for non-finite input.
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    let upper = MAX_FREQUENCY_HZ.min(nyquist);
    finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, upper)
}

/// Self-contained deterministic PRNG (Marsaglia xorshift64) seeded via
/// `SplitMix64`, matching the generator in [`super::noise`].
#[derive(Debug, Clone, Copy)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed_to_state(seed) }
    }

    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let bits = (x >> 32) as u32;
        let unit = (bits >> 8) as Sample * (1.0 / 16_777_216.0);
        unit * 2.0 - 1.0
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
#[inline]
fn seed_to_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
}

/// Construction parameters for a [`HelmholtzResonatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HelmholtzResonatorParams {
    /// Resonance (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized breath pressure in `[0, 1]` (loudness / drive).
    pub breath_pressure: Sample,
    /// Normalized resonance in `[0, 1]` (vessel Q / ring).
    pub resonance: Sample,
    /// Brightness in `[0, 1]` (odd-harmonic `tanh` shaping).
    pub brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]` (breathiness).
    pub breath_noise: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
    /// Seed for the deterministic breath-turbulence stream.
    pub seed: u64,
}

impl Default for HelmholtzResonatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            resonance: DEFAULT_RESONANCE,
            brightness: DEFAULT_BRIGHTNESS,
            breath_noise: DEFAULT_BREATH_NOISE,
            amplitude: DEFAULT_AMPLITUDE,
            seed: DEFAULT_SEED,
        }
    }
}

/// A Helmholtz blown-bottle resonator voice source node (0 inputs, 1 output).
///
/// The mono resonance signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{HelmholtzResonatorNode, HelmholtzResonatorParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = HelmholtzResonatorNode::new(48_000, HelmholtzResonatorParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A steady breath pumps the lumped resonance into a sustained tone.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct HelmholtzResonatorNode {
    /// Sample rate the coefficients were computed for.
    sample_rate: Sample,
    /// Resonance (pitch) frequency in hertz.
    frequency_hz: Sample,
    /// Normalized resonance in `[0, 1]`.
    resonance: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]`.
    breath_noise: Sample,
    /// Smoothed normalized breath pressure in `[0, 1]`.
    breath_pressure: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Chamberlin SVF low-pass integrator state.
    lp: Sample,
    /// Chamberlin SVF band-pass integrator state (the resonating mode).
    bp: Sample,

    /// Deterministic breath-turbulence generator.
    rng: Xorshift64,
    /// Seed the turbulence generator was constructed/reseeded with.
    seed: u64,

    /// Latched SVF frequency coefficient `f = 2 * sin(pi * f0 / sr)`.
    coeff_f: Sample,
    /// Latched baseline SVF damping `damp = damp_ratio * comp`.
    damp0: Sample,
    /// Latched Van der Pol pump gain `mu = MU_RATIO * comp`.
    mu: Sample,
    /// Latched brightness waveshaper drive `1 + brightness * 4`.
    shape_drive: Sample,
    /// Latched waveshaper normalization `tanh(shape_drive)`.
    shape_norm: Sample,
}

impl HelmholtzResonatorNode {
    /// Builds a Helmholtz resonator for `sample_rate` Hz from a parameter
    /// bundle.
    ///
    /// All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to
    /// `[MIN_FREQUENCY_HZ, min(MAX_FREQUENCY_HZ, sample_rate / 2)]`, and
    /// `breath_pressure`/`resonance`/`brightness`/`breath_noise` to `[0, 1]`.
    #[must_use]
    pub fn new(sample_rate: u32, params: HelmholtzResonatorParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let resonance = finite_or(params.resonance, DEFAULT_RESONANCE).clamp(0.0, 1.0);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let breath_noise = finite_or(params.breath_noise, DEFAULT_BREATH_NOISE).clamp(0.0, 1.0);
        let breath_pressure = finite_or(params.breath_pressure, DEFAULT_BREATH_PRESSURE).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            resonance,
            brightness,
            breath_noise,
            breath_pressure: Smoothed::new(breath_pressure),
            amplitude: Smoothed::new(amplitude),
            lp: 0.0,
            bp: BP_INIT,
            rng: Xorshift64::new(params.seed),
            seed: params.seed,
            coeff_f: 1.0,
            damp0: DAMP_MAX,
            mu: MU_RATIO,
            shape_drive: 1.0,
            shape_norm: 1.0,
        };
        node.recompute();
        node
    }

    /// Retunes the resonance to `frequency_hz` (clamped to
    /// `[MIN_FREQUENCY_HZ, min(MAX_FREQUENCY_HZ, sample_rate / 2)]`).
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        let requested = finite_or(frequency_hz, self.frequency_hz);
        self.frequency_hz = sanitize_frequency(requested, self.sample_rate);
        self.recompute();
    }

    /// Sets the normalized resonance in `[0, 1]` (vessel Q / ring).
    #[inline]
    pub fn set_resonance(&mut self, resonance: Sample) {
        self.resonance = finite_or(resonance, self.resonance).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the brightness in `[0, 1]` (odd-harmonic shaping).
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the normalized breath turbulence in `[0, 1]`.
    #[inline]
    pub fn set_breath_noise(&mut self, breath_noise: Sample) {
        self.breath_noise = finite_or(breath_noise, self.breath_noise).clamp(0.0, 1.0);
    }

    /// Sets the normalized breath pressure in `[0, 1]`, smoothing over `ramp`.
    #[inline]
    pub fn set_breath_pressure(&mut self, breath_pressure: Sample, ramp: Ramp) {
        let v = finite_or(breath_pressure, self.breath_pressure.target()).clamp(0.0, 1.0);
        self.breath_pressure.set_target(v, ramp);
    }

    /// Sets a new target output amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Reseeds the breath-turbulence generator, restarting the stream.
    #[inline]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Xorshift64::new(seed);
    }

    /// Returns the resonance (pitch) frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the normalized resonance in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn resonance(&self) -> Sample {
        self.resonance
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the normalized breath turbulence in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn breath_noise(&self) -> Sample {
        self.breath_noise
    }

    /// Returns the target normalized breath pressure in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn breath_pressure(&self) -> Sample {
        self.breath_pressure.target()
    }

    /// Returns the target output amplitude (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns the current turbulence seed.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Recomputes the latched coefficients from the user-facing parameters.
    ///
    /// The frequency compensation `comp` holds the net small-signal loop gain
    /// (and hence the onset time) roughly constant across the pitch range by
    /// scaling both the pump `mu` and the damping with the inverse of the SVF
    /// frequency coefficient, referenced to [`F_REF`].
    fn recompute(&mut self) {
        let sr = self.sample_rate;
        let f = 2.0 * ops::sin(core::f32::consts::PI * self.frequency_hz / sr);
        // Guard against a zero coefficient at pathologically low f0/sr.
        let f = if f > 1.0e-6 { f } else { 1.0e-6 };
        self.coeff_f = f;

        let f_ref = 2.0 * ops::sin(core::f32::consts::PI * F_REF / sr);
        let comp = f_ref / f;

        let damp_ratio = DAMP_MAX + (DAMP_MIN - DAMP_MAX) * self.resonance;
        self.damp0 = damp_ratio * comp;
        self.mu = MU_RATIO * comp;

        self.shape_drive = 1.0 + self.brightness * 4.0;
        self.shape_norm = ops::tanh(self.shape_drive);
    }

    /// Advances the lumped resonance by one sample and returns the radiated
    /// output.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let breath = self.breath_pressure.next_sample();
        let noise_amount = self.breath_noise * NOISE_SCALE;

        // Van der Pol negative resistance: pumps small amplitudes and saturates
        // large ones so the band-pass settles into a stable limit cycle. The
        // breath scales the pump, so the vessel only speaks above threshold.
        let v = self.bp;
        let mu_eff = self.mu * breath;
        let nl = mu_eff * (1.0 - (v * v) / (V_SAT * V_SAT)) * v;
        let turbulence = breath * noise_amount * self.rng.next_bipolar();
        let drive = nl + turbulence;

        // Chamberlin state-variable filter tuned to f0 (the lumped mode).
        let hp = drive - self.lp - self.damp0 * self.bp;
        self.bp = flush_denormal(self.bp + self.coeff_f * hp);
        self.lp = flush_denormal(self.lp + self.coeff_f * self.bp);

        // Brightness: odd-harmonic tanh shaping outside the resonance loop, so
        // it never affects stability and adds no even harmonics or DC.
        let y = if self.brightness > 0.0 {
            let shaped = ops::tanh(self.shape_drive * self.bp / V_SAT) / self.shape_norm * V_SAT;
            (1.0 - self.brightness) * self.bp + self.brightness * shaped
        } else {
            self.bp
        };

        flush_denormal(y * OUTPUT_GAIN * self.amplitude.next_sample())
    }
}

impl AudioNode for HelmholtzResonatorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }
        let frames = io.output(0).active_frames();
        if frames == 0 {
            return;
        }

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample();
            }
        }
        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.lp = 0.0;
        self.bp = BP_INIT;
        self.rng = Xorshift64::new(self.seed);
        self.breath_pressure = Smoothed::new(self.breath_pressure.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute();
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut HelmholtzResonatorNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut HelmholtzResonatorNode,
        frames: usize,
        layout: ChannelLayout,
    ) -> Vec<Vec<Sample>> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let channels = outputs[0].channels();
        (0..channels)
            .map(|ch| outputs[0].channel(ch).to_vec())
            .collect()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(s.abs()))
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }

    /// Magnitude of the Goertzel estimate at frequency `f` over `block`.
    fn goertzel(block: &[Sample], f: Sample, sr: Sample) -> f64 {
        let w = core::f32::consts::TAU * f / sr;
        let c = 2.0 * ops::cos(w);
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for &x in block {
            let s0 = f64::from(x) + f64::from(c) * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - f64::from(c) * s1 * s2).sqrt()
    }

    #[test]
    fn default_self_oscillates() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let block = render(&mut node, 2 * SR as usize);
        // A steady breath pumps the lumped mode into a sustained tone, not a
        // decaying ring.
        assert!(peak(&block[SR as usize..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_bounded() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let block = render(&mut node, 4 * SR as usize);
        assert!(block.iter().all(|s| s.is_finite()));
        // The clamped Van der Pol cubic plus positive baseline damping bound the
        // limit cycle; it must never run away.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn worst_case_grid_stays_below_unity() {
        // Across the whole parameter grid at full amplitude the output must stay
        // finite and within [-1, 1].
        for &f0 in &[MIN_FREQUENCY_HZ, 120.0, 440.0, 2_000.0, MAX_FREQUENCY_HZ] {
            for &br in &[0.25, 1.0] {
                for &res in &[0.0, 1.0] {
                    for &bri in &[0.0, 1.0] {
                        let params = HelmholtzResonatorParams {
                            frequency_hz: f0,
                            breath_pressure: br,
                            resonance: res,
                            brightness: bri,
                            breath_noise: 1.0,
                            amplitude: 1.0,
                            ..HelmholtzResonatorParams::default()
                        };
                        let mut node = HelmholtzResonatorNode::new(SR, params);
                        let block = render(&mut node, SR as usize);
                        assert!(block.iter().all(|s| s.is_finite()), "nan at {f0} Hz");
                        assert!(peak(&block) < 1.0, "runaway at {f0} Hz");
                    }
                }
            }
        }
    }

    #[test]
    fn pitch_locks_to_resonance_frequency() {
        let params = HelmholtzResonatorParams {
            frequency_hz: 220.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut node = HelmholtzResonatorNode::new(SR, params);
        let block = render(&mut node, 2 * SR as usize);
        let steady = &block[SR as usize..];
        let f0 = goertzel(steady, 220.0, SR as Sample);
        // The fundamental dominates its octave neighbours by a wide margin.
        assert!(f0 > 10.0 * goertzel(steady, 110.0, SR as Sample));
        assert!(f0 > 10.0 * goertzel(steady, 440.0, SR as Sample));
    }

    #[test]
    fn breath_does_not_bend_pitch() {
        // Blowing harder makes the vessel louder and brighter, never sharper:
        // the lumped resonance pins the fundamental regardless of breath.
        let soft = HelmholtzResonatorParams {
            breath_pressure: 0.4,
            ..HelmholtzResonatorParams::default()
        };
        let hard = HelmholtzResonatorParams {
            breath_pressure: 1.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, soft);
        let mut b = HelmholtzResonatorNode::new(SR, hard);
        let ba = render(&mut a, 2 * SR as usize);
        let bb = render(&mut b, 2 * SR as usize);
        let sa = &ba[SR as usize..];
        let sb = &bb[SR as usize..];
        let f0a = goertzel(sa, 220.0, SR as Sample);
        let f0b = goertzel(sb, 220.0, SR as Sample);
        assert!(f0a > 10.0 * goertzel(sa, 247.0, SR as Sample));
        assert!(f0b > 10.0 * goertzel(sb, 247.0, SR as Sample));
    }

    #[test]
    fn default_tone_is_nearly_sinusoidal() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let block = render(&mut node, 2 * SR as usize);
        let steady = &block[SR as usize..];
        let h1 = goertzel(steady, 220.0, SR as Sample);
        let h3 = goertzel(steady, 660.0, SR as Sample);
        // At the default brightness the third harmonic is a small fraction of
        // the fundamental: a hollow, flute-like bottle tone.
        assert!(h3 < 0.2 * h1);
    }

    #[test]
    fn no_even_harmonics() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let block = render(&mut node, 2 * SR as usize);
        let steady = &block[SR as usize..];
        let h1 = goertzel(steady, 220.0, SR as Sample);
        let h2 = goertzel(steady, 440.0, SR as Sample);
        // The odd-only tanh shaper introduces no second harmonic.
        assert!(h2 < 0.05 * h1);
    }

    #[test]
    fn brightness_adds_odd_harmonics() {
        let dark = HelmholtzResonatorParams {
            brightness: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let bright = HelmholtzResonatorParams {
            brightness: 0.8,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, dark);
        let mut b = HelmholtzResonatorNode::new(SR, bright);
        let ba = render(&mut a, 2 * SR as usize);
        let bb = render(&mut b, 2 * SR as usize);
        let h3_dark = goertzel(&ba[SR as usize..], 660.0, SR as Sample);
        let h3_bright = goertzel(&bb[SR as usize..], 660.0, SR as Sample);
        // Brightness markedly increases the odd-harmonic content.
        assert!(h3_bright > 5.0 * h3_dark);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let mut b = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        assert_eq!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let first = render(&mut node, 8192);
        node.reset();
        let second = render(&mut node, 8192);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_breath_is_silent() {
        let params = HelmholtzResonatorParams {
            breath_pressure: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut node = HelmholtzResonatorNode::new(SR, params);
        // With no breath the Van der Pol pump vanishes and the positive baseline
        // damping decays the tiny initial displacement to silence.
        let block = render(&mut node, SR as usize);
        assert!(peak(&block) < 1e-3);
    }

    #[test]
    fn low_breath_is_near_silent() {
        let params = HelmholtzResonatorParams {
            breath_pressure: 0.1,
            ..HelmholtzResonatorParams::default()
        };
        let mut node = HelmholtzResonatorNode::new(SR, params);
        let block = render(&mut node, 2 * SR as usize);
        // Below the speaking threshold the vessel barely sounds.
        assert!(peak(&block[SR as usize..]) < 1e-2);
    }

    #[test]
    fn starts_without_turbulence() {
        // Even with breath noise disabled, the deterministic initial condition
        // lets the oscillator self-start.
        let params = HelmholtzResonatorParams {
            breath_noise: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut node = HelmholtzResonatorNode::new(SR, params);
        let block = render(&mut node, 2 * SR as usize);
        assert!(peak(&block[SR as usize..]) > 1e-2);
    }

    #[test]
    fn amplitude_scales_output_energy() {
        // Disable breath noise so both voices share the same deterministic drive
        // and only the output gain differs.
        let loud = HelmholtzResonatorParams {
            amplitude: 1.0,
            breath_noise: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let soft = HelmholtzResonatorParams {
            amplitude: 0.5,
            breath_noise: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, loud);
        let mut b = HelmholtzResonatorNode::new(SR, soft);
        let ea = energy(&render(&mut a, 2 * SR as usize)[SR as usize..]);
        let eb = energy(&render(&mut b, 2 * SR as usize)[SR as usize..]);
        // Energy scales with amplitude squared: (1.0 / 0.5)^2 = 4.
        assert!((ea / eb - 4.0).abs() < 0.2);
    }

    #[test]
    fn resonance_affects_speaking() {
        // A high-Q vessel speaks far more readily than a low-Q one at the same
        // breath.
        let low = HelmholtzResonatorParams {
            resonance: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let high = HelmholtzResonatorParams {
            resonance: 1.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, low);
        let mut b = HelmholtzResonatorNode::new(SR, high);
        let pa = peak(&render(&mut a, 2 * SR as usize)[SR as usize..]);
        let pb = peak(&render(&mut b, 2 * SR as usize)[SR as usize..]);
        assert!(pb > pa);
    }

    #[test]
    fn breath_noise_changes_output() {
        let dry = HelmholtzResonatorParams {
            breath_noise: 0.0,
            ..HelmholtzResonatorParams::default()
        };
        let airy = HelmholtzResonatorParams {
            breath_noise: 1.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, dry);
        let mut b = HelmholtzResonatorNode::new(SR, airy);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn seed_changes_turbulence_stream() {
        let a_params = HelmholtzResonatorParams {
            seed: 0x1111_2222_3333_4444,
            ..HelmholtzResonatorParams::default()
        };
        let b_params = HelmholtzResonatorParams {
            seed: 0xAAAA_BBBB_CCCC_DDDD,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, a_params);
        let mut b = HelmholtzResonatorNode::new(SR, b_params);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn mono_core_replicates_into_all_channels() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let channels = render_layout(&mut node, 4096, ChannelLayout::Quad);
        assert_eq!(channels.len(), 4);
        for ch in 1..4 {
            assert_eq!(channels[0], channels[ch]);
        }
    }

    #[test]
    fn zero_frames_is_a_no_op() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = HelmholtzResonatorParams {
            frequency_hz: 330.0,
            breath_pressure: 0.7,
            resonance: 0.8,
            brightness: 0.4,
            breath_noise: 0.3,
            amplitude: 0.6,
            seed: 0x1234_5678_9ABC_DEF0,
        };
        let node = HelmholtzResonatorNode::new(SR, params);
        assert!((node.frequency_hz() - 330.0).abs() < 1e-4);
        assert!((node.breath_pressure() - 0.7).abs() < 1e-6);
        assert!((node.resonance() - 0.8).abs() < 1e-6);
        assert!((node.brightness() - 0.4).abs() < 1e-6);
        assert!((node.breath_noise() - 0.3).abs() < 1e-6);
        assert!((node.amplitude() - 0.6).abs() < 1e-6);
        assert_eq!(node.seed(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn frequency_is_clamped() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        node.set_frequency_hz(1.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1e-4);
        node.set_frequency_hz(1_000_000.0);
        assert!((node.frequency_hz() - MAX_FREQUENCY_HZ).abs() < 1e-4);
    }

    #[test]
    fn normalized_params_are_clamped() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        node.set_resonance(5.0);
        node.set_brightness(5.0);
        node.set_breath_noise(5.0);
        node.set_breath_pressure(5.0, Ramp::Immediate);
        assert!((node.resonance() - 1.0).abs() < 1e-6);
        assert!((node.brightness() - 1.0).abs() < 1e-6);
        assert!((node.breath_noise() - 1.0).abs() < 1e-6);
        assert!((node.breath_pressure() - 1.0).abs() < 1e-6);
        node.set_resonance(-5.0);
        node.set_brightness(-5.0);
        node.set_breath_noise(-5.0);
        node.set_breath_pressure(-5.0, Ramp::Immediate);
        assert!((node.resonance() - 0.0).abs() < 1e-6);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
        assert!((node.breath_noise() - 0.0).abs() < 1e-6);
        assert!((node.breath_pressure() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        let f = node.frequency_hz();
        let r = node.resonance();
        let b = node.brightness();
        let bn = node.breath_noise();
        let bp = node.breath_pressure();
        let a = node.amplitude();
        node.set_frequency_hz(Sample::NAN);
        node.set_resonance(Sample::INFINITY);
        node.set_brightness(Sample::NEG_INFINITY);
        node.set_breath_noise(Sample::NAN);
        node.set_breath_pressure(Sample::NAN, Ramp::Immediate);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.resonance(), r);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.breath_noise(), bn);
        assert_eq!(node.breath_pressure(), bp);
        assert_eq!(node.amplitude(), a);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = HelmholtzResonatorParams {
            frequency_hz: Sample::NAN,
            breath_pressure: Sample::INFINITY,
            resonance: Sample::NAN,
            brightness: Sample::NEG_INFINITY,
            breath_noise: Sample::NAN,
            amplitude: Sample::NAN,
            seed: DEFAULT_SEED,
        };
        let node = HelmholtzResonatorNode::new(SR, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.breath_pressure().is_finite());
        assert!(node.resonance().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.breath_noise().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let low = HelmholtzResonatorParams {
            frequency_hz: 180.0,
            ..HelmholtzResonatorParams::default()
        };
        let high = HelmholtzResonatorParams {
            frequency_hz: 660.0,
            ..HelmholtzResonatorParams::default()
        };
        let mut a = HelmholtzResonatorNode::new(SR, low);
        let mut b = HelmholtzResonatorNode::new(SR, high);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn high_and_low_frequencies_both_oscillate() {
        for &f in &[MIN_FREQUENCY_HZ, 120.0, 1_500.0, MAX_FREQUENCY_HZ] {
            let params = HelmholtzResonatorParams {
                frequency_hz: f,
                ..HelmholtzResonatorParams::default()
            };
            let mut node = HelmholtzResonatorNode::new(SR, params);
            let block = render(&mut node, 2 * SR as usize);
            assert!(peak(&block[SR as usize..]) > 1e-3, "silent at {f} Hz");
            assert!(block.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn breath_pressure_target_tracks_setter() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        node.set_breath_pressure(0.8, Ramp::linear_seconds(0.01, SR));
        assert!((node.breath_pressure() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        node.set_amplitude(0.25, Ramp::linear_seconds(0.01, SR));
        assert!((node.amplitude() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = HelmholtzResonatorNode::new(SR, HelmholtzResonatorParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
