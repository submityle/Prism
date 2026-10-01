//! Dattorro figure-eight plate reverberator.
//!
//! Where the [`AlgorithmicRoom`](super::algorithmic::AlgorithmicRoom) models a
//! physical room from parallel Freeverb combs, the
//! [`FdnReverb`](super::fdn::FdnReverb) recirculates a Hadamard-matrixed delay
//! bank, and the [`Convolver`](super::convolver::Convolver) replays a measured
//! impulse response, the [`PlateReverb`] synthesises the lush, dense, metallic
//! wash of a vintage electro-mechanical *plate* (the Lexicon 224 / EMT 140
//! lineage) using a single cross-coupled all-pass feedback loop.
//!
//! The signal path has three stages:
//!
//! 1. **Input conditioning** -- a pre-delay, a one-pole `bandwidth` low-pass
//!    (rolling off the harsh top before it enters the loop), then four series
//!    all-pass diffusers that smear the input into a dense, phase-scrambled
//!    excitation.
//! 2. **The tank** -- a figure-eight of two half-loops that feed each other.
//!    Each half runs a slowly *modulated* all-pass (whose LFO-wobbled delay
//!    stops the loop ringing on fixed eigenfrequencies), a long delay, a
//!    one-pole `damping` low-pass (shortening the high-frequency tail like air
//!    absorption), a `decay` scaler, a second fixed all-pass, and a final
//!    delay. The halves cross-couple so energy circulates between them.
//! 3. **Output taps** -- seven signed taps are read from the two halves'
//!    internal delays for the left output and seven more for the right,
//!    producing a wide, decorrelated stereo image from a mono excitation.
//!
//! All delay/all-pass lengths and output-tap offsets are quoted at Dattorro's
//! reference rate and rescaled to the runtime rate at construction, mirroring
//! how [`AlgorithmicRoom`](super::algorithmic::AlgorithmicRoom) rescales the
//! Freeverb constants from 44.1 kHz. Every buffer is pre-allocated, so
//! [`PlateReverb::process`](crate::graph::AudioNode::process) never allocates,
//! locks, or panics on the audio thread.
//!
//! # Model
//!
//! The topology is Dattorro's figure-eight "tank": two all-pass feedback
//! half-loops with cross-coupling, fed by a four-stage input diffuser. The
//! modulated all-pass uses a ~1 Hz LFO with a few samples of excursion to keep
//! the tail from settling onto audible resonant modes. With `decay < 1` and
//! every all-pass coefficient `|g| < 1`, the loop gain stays below unity, so
//! the impulse response decays and the output is bounded.
//!
//! # Real-time contract
//!
//! Every delay line, all-pass, and one-pole filter is sized and allocated in
//! [`PlateReverb::new`]. `process` only reads and writes those buffers, advances
//! integer indices, and evaluates one sine per sample, so it is allocation-,
//! lock-, and panic-free. Denormals are flushed on every delay write, filter
//! state, and feedback register.
//!
//! # Provenance
//!
//! Written from the published description of the structure in J. Dattorro,
//! "Effect Design, Part 1: Reverberator and Other Filters", Journal of the
//! Audio Engineering Society, vol. 45, no. 9, 1997 (the figure-eight tank of
//! figure 8 and its tuning table). The all-pass, delay-line, and one-pole
//! filter primitives are standard textbook difference equations implemented
//! from scratch here. This file contains no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, Resonance Audio, or any other third-party audio-engine
//! source code or derivative code; only the classical DSP structure is used.
//!
//! # Relationship
//!
//! Complements the other reverb nodes rather than duplicating them: the plate
//! is a single cross-coupled all-pass loop (metallic, dense, fast-building),
//! distinct from Freeverb's parallel combs in `algorithmic`, Jot's matrixed
//! feedback network in `fdn`, and the measured-response playback in `convolver`.
//! The all-pass here is a true lattice (`y = d - g*w`), unlike the Freeverb
//! approximation (`output = -input + buffered`) used in `algorithmic`.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::{FRAC_PI_2, TAU};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Sample rate the tuning table below is expressed at (Dattorro's reference).
/// Lengths are rescaled to the runtime rate at construction.
const REFERENCE_RATE: Sample = 29_761.0;

/// Series input-diffuser all-pass lengths (frames at [`REFERENCE_RATE`]). The
/// first pair uses `input_diffusion_1`, the second pair `input_diffusion_2`.
const INPUT_DIFFUSION_DELAYS: [usize; 4] = [142, 107, 379, 277];

/// Base delays of the two modulated tank all-passes (left, right).
const TANK_MOD_AP_BASE: [usize; 2] = [672, 908];

/// First long tank delay in each half (left, right).
const TANK_DELAY_1: [usize; 2] = [4453, 4217];

/// Second fixed tank all-pass in each half (left, right).
const TANK_AP_2: [usize; 2] = [1800, 2656];

/// Second long tank delay in each half (left, right).
const TANK_DELAY_2: [usize; 2] = [3720, 3163];

/// Left-output tap offsets (frames at [`REFERENCE_RATE`]), read in the order
/// `delay_r1`, `delay_r1`, `ap_r2`, `delay_r2`, `delay_l1`, `ap_l2`, `delay_l2`
/// with signs `+ + - + - - -`.
const YL_TAP_REFERENCE: [usize; 7] = [266, 2974, 1913, 1996, 1990, 187, 1066];

/// Right-output tap offsets (frames at [`REFERENCE_RATE`]), read in the order
/// `delay_l1`, `delay_l1`, `ap_l2`, `delay_l2`, `delay_r1`, `ap_r2`, `delay_r2`
/// with signs `+ + - + - - -`.
const YR_TAP_REFERENCE: [usize; 7] = [353, 3627, 1228, 2673, 2111, 335, 121];

/// Peak excursion of the modulated-all-pass LFO (frames at [`REFERENCE_RATE`]).
const MOD_EXCURSION_FRAMES: Sample = 8.0;

/// Frequency of the modulated-all-pass LFO in hertz (slow, sub-audio).
const MOD_LFO_HZ: Sample = 1.0;

/// Largest `decay`/all-pass coefficient magnitude; keeps the loop gain below
/// unity so the tail always decays and the output stays bounded.
const MAX_COEFFICIENT: Sample = 0.9999;

/// Largest channel count supported; covers every
/// [`ChannelLayout`](crate::buffer::ChannelLayout) the engine ships.
const MAX_CHANNELS: usize = 8;

/// Rounds a reference-rate frame count to the runtime rate, never below one.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded product of a non-negative length and ratio fits usize"
)]
fn scale_frames(reference: usize, rate_scale: Sample) -> usize {
    let scaled = ops::round(reference as Sample * rate_scale);
    (scaled as usize).max(1)
}

/// Converts milliseconds to frames at `sample_rate`, flooring to zero when the
/// requested time is non-positive.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded non-negative frame count fits usize"
)]
fn ms_to_frames(ms: Sample, sample_rate: u32) -> usize {
    if ms <= 0.0 {
        return 0;
    }
    let frames = ops::round(ms * 0.001 * sample_rate as Sample);
    frames as usize
}

/// Replaces a non-finite control with a fallback.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// A circular delay line. `read(d)` returns the sample written `d` steps ago
/// for `d` in `[1, len]`; `read(len)` is the oldest (full-length) sample.
#[derive(Debug, Clone)]
struct DelayLine {
    buf: Vec<Sample>,
    pos: usize,
}

impl DelayLine {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.buf.len()
    }

    /// Reads the sample `delay` frames ago, clamped to `[1, len]`.
    #[inline]
    fn read(&self, delay: usize) -> Sample {
        let len = self.buf.len();
        let d = delay.clamp(1, len);
        let idx = (self.pos + len - d) % len;
        self.buf[idx]
    }

    /// Reads a fractional delay (linear interpolation) clamped to `[1, len-1]`.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "floor of a value clamped to [1, len-1] is a valid in-range index"
    )]
    fn read_frac(&self, delay: Sample) -> Sample {
        let len = self.buf.len();
        let max_delay = (len - 1) as Sample;
        let d = delay.clamp(1.0, max_delay.max(1.0));
        let floor = ops::floor(d);
        let frac = d - floor;
        let whole = floor as usize;
        let i0 = (self.pos + len - whole) % len;
        let i1 = (self.pos + len - whole - 1) % len;
        self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac
    }

    /// Writes `x` at the head and advances; flushes denormals.
    #[inline]
    fn write(&mut self, x: Sample) {
        self.buf[self.pos] = flush_denormal(x);
        self.pos = (self.pos + 1) % self.buf.len();
    }

    #[inline]
    fn clear(&mut self) {
        for sample in &mut self.buf {
            *sample = 0.0;
        }
        self.pos = 0;
    }
}

/// A true lattice all-pass: `H(z) = (z^-M - g) / (1 - g z^-M)`, `|H| = 1`.
#[derive(Debug, Clone)]
struct Allpass {
    line: DelayLine,
    g: Sample,
}

impl Allpass {
    #[inline]
    fn new(len: usize, g: Sample) -> Self {
        Self {
            line: DelayLine::new(len),
            g,
        }
    }

    /// Processes one sample through the lattice.
    #[inline]
    fn process(&mut self, x: Sample) -> Sample {
        let d = self.line.read(self.line.len());
        let w = x + self.g * d;
        self.line.write(w);
        d - self.g * w
    }

    /// Reads the internal delay memory `off` frames ago (for output taps).
    #[inline]
    fn tap(&self, off: usize) -> Sample {
        self.line.read(off)
    }

    #[inline]
    fn set_g(&mut self, g: Sample) {
        self.g = g;
    }

    #[inline]
    fn clear(&mut self) {
        self.line.clear();
    }
}

/// A lattice all-pass whose read position is wobbled by a slow sine LFO. The
/// fractional delay keeps the tank from ringing on fixed eigenfrequencies.
#[derive(Debug, Clone)]
struct ModulatedAllpass {
    line: DelayLine,
    base: Sample,
    excursion: Sample,
    g: Sample,
    phase: Sample,
    init_phase: Sample,
    phase_inc: Sample,
}

impl ModulatedAllpass {
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ceil of a non-negative delay bound fits usize"
    )]
    fn new(base: Sample, excursion: Sample, g: Sample, sample_rate: Sample, init_phase: Sample) -> Self {
        let span = ops::round(base + excursion.abs()) as usize + 2;
        Self {
            line: DelayLine::new(span),
            base,
            excursion,
            g,
            phase: init_phase,
            init_phase,
            phase_inc: TAU * MOD_LFO_HZ / sample_rate,
        }
    }

    /// Processes one sample, advancing the LFO.
    #[inline]
    fn process(&mut self, x: Sample) -> Sample {
        let delay = self.base + self.excursion * ops::sin(self.phase);
        let d = self.line.read_frac(delay);
        let w = x + self.g * d;
        self.line.write(w);
        self.phase += self.phase_inc;
        if self.phase >= TAU {
            self.phase -= TAU;
        }
        d - self.g * w
    }

    #[inline]
    fn set_g(&mut self, g: Sample) {
        self.g = g;
    }

    #[inline]
    fn set_excursion(&mut self, excursion: Sample) {
        self.excursion = excursion;
    }

    #[inline]
    fn clear(&mut self) {
        self.line.clear();
        self.phase = self.init_phase;
    }
}

/// A one-pole low-pass `y[n] = y[n-1] + coeff * (x[n] - y[n-1])`.
#[derive(Debug, Clone)]
struct OnePole {
    state: Sample,
    coeff: Sample,
}

impl OnePole {
    #[inline]
    fn new(coeff: Sample) -> Self {
        Self { state: 0.0, coeff }
    }

    #[inline]
    fn process(&mut self, x: Sample) -> Sample {
        self.state = flush_denormal(self.state + self.coeff * (x - self.state));
        self.state
    }

    #[inline]
    fn set_coeff(&mut self, coeff: Sample) {
        self.coeff = coeff;
    }

    #[inline]
    fn clear(&mut self) {
        self.state = 0.0;
    }
}

/// Construction-time controls for a [`PlateReverb`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlateReverbParams {
    /// Pre-delay before the loop, in milliseconds.
    pub pre_delay_ms: Sample,
    /// Input low-pass bandwidth in `[0, 1]`; `1` passes full bandwidth.
    pub bandwidth: Sample,
    /// Tail decay in `[0, MAX_COEFFICIENT]`; higher is longer.
    pub decay: Sample,
    /// Modulated-all-pass coefficient (decay diffusion 1).
    pub decay_diffusion_1: Sample,
    /// Fixed tank all-pass coefficient (decay diffusion 2).
    pub decay_diffusion_2: Sample,
    /// First input-diffuser coefficient (first two all-passes).
    pub input_diffusion_1: Sample,
    /// Second input-diffuser coefficient (last two all-passes).
    pub input_diffusion_2: Sample,
    /// High-frequency damping in `[0, 1]`; higher shortens the bright tail.
    pub damping: Sample,
    /// Modulation depth in `[0, 1]` scaling the all-pass LFO excursion.
    pub mod_depth: Sample,
    /// Wet (reverberated) output gain.
    pub wet: Sample,
    /// Dry (unprocessed input) output gain.
    pub dry: Sample,
}

impl Default for PlateReverbParams {
    fn default() -> Self {
        Self {
            pre_delay_ms: 0.0,
            bandwidth: 0.9995,
            decay: 0.5,
            decay_diffusion_1: 0.70,
            decay_diffusion_2: 0.50,
            input_diffusion_1: 0.750,
            input_diffusion_2: 0.625,
            damping: 0.0005,
            mod_depth: 1.0,
            wet: 0.4,
            dry: 1.0,
        }
    }
}

impl PlateReverbParams {
    /// Clamps every control into its valid range, replacing non-finite values
    /// with the default.
    fn sanitised(self) -> Self {
        let d = Self::default();
        Self {
            pre_delay_ms: finite_or(self.pre_delay_ms, d.pre_delay_ms).max(0.0),
            bandwidth: finite_or(self.bandwidth, d.bandwidth).clamp(0.0, 1.0),
            decay: finite_or(self.decay, d.decay).clamp(0.0, MAX_COEFFICIENT),
            decay_diffusion_1: finite_or(self.decay_diffusion_1, d.decay_diffusion_1)
                .clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT),
            decay_diffusion_2: finite_or(self.decay_diffusion_2, d.decay_diffusion_2)
                .clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT),
            input_diffusion_1: finite_or(self.input_diffusion_1, d.input_diffusion_1)
                .clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT),
            input_diffusion_2: finite_or(self.input_diffusion_2, d.input_diffusion_2)
                .clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT),
            damping: finite_or(self.damping, d.damping).clamp(0.0, 1.0),
            mod_depth: finite_or(self.mod_depth, d.mod_depth).clamp(0.0, 1.0),
            wet: finite_or(self.wet, d.wet),
            dry: finite_or(self.dry, d.dry),
        }
    }
}

/// A Dattorro figure-eight plate reverberator (mono excitation, stereo tail).
///
/// See the [module documentation](self) for the signal path and provenance.
#[derive(Debug, Clone)]
pub struct PlateReverb {
    channels: usize,
    rate_scale: Sample,
    pre_delay_len: usize,
    pre_delay: DelayLine,
    input_lp: OnePole,
    in_ap: [Allpass; 4],
    mod_ap_l: ModulatedAllpass,
    mod_ap_r: ModulatedAllpass,
    delay_l1: DelayLine,
    delay_r1: DelayLine,
    damp_l: OnePole,
    damp_r: OnePole,
    ap_l2: Allpass,
    ap_r2: Allpass,
    delay_l2: DelayLine,
    delay_r2: DelayLine,
    left_out: Sample,
    right_out: Sample,
    decay: Sample,
    bandwidth: Sample,
    damping: Sample,
    yl_taps: [usize; 7],
    yr_taps: [usize; 7],
    wet: Smoothed,
    dry: Smoothed,
}

impl PlateReverb {
    /// Builds a plate reverberator for `sample_rate`, `channels`, and `params`.
    ///
    /// All delay lengths are rescaled from [`REFERENCE_RATE`] to `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: PlateReverbParams) -> Self {
        let channels = channels.clamp(1, MAX_CHANNELS);
        let sr = sample_rate as Sample;
        let rate_scale = sr / REFERENCE_RATE;
        let p = params.sanitised();

        let pre_delay_len = ms_to_frames(p.pre_delay_ms, sample_rate);
        let pre_delay = DelayLine::new(pre_delay_len.max(1));

        let in_ap = [
            Allpass::new(scale_frames(INPUT_DIFFUSION_DELAYS[0], rate_scale), p.input_diffusion_1),
            Allpass::new(scale_frames(INPUT_DIFFUSION_DELAYS[1], rate_scale), p.input_diffusion_1),
            Allpass::new(scale_frames(INPUT_DIFFUSION_DELAYS[2], rate_scale), p.input_diffusion_2),
            Allpass::new(scale_frames(INPUT_DIFFUSION_DELAYS[3], rate_scale), p.input_diffusion_2),
        ];

        let excursion = MOD_EXCURSION_FRAMES * rate_scale * p.mod_depth;
        let mod_ap_l = ModulatedAllpass::new(
            TANK_MOD_AP_BASE[0] as Sample * rate_scale,
            excursion,
            p.decay_diffusion_1,
            sr,
            0.0,
        );
        let mod_ap_r = ModulatedAllpass::new(
            TANK_MOD_AP_BASE[1] as Sample * rate_scale,
            excursion,
            p.decay_diffusion_1,
            sr,
            FRAC_PI_2,
        );

        let delay_l1 = DelayLine::new(scale_frames(TANK_DELAY_1[0], rate_scale));
        let delay_r1 = DelayLine::new(scale_frames(TANK_DELAY_1[1], rate_scale));
        let damp_coeff = 1.0 - p.damping;
        let damp_l = OnePole::new(damp_coeff);
        let damp_r = OnePole::new(damp_coeff);
        let ap_l2 = Allpass::new(scale_frames(TANK_AP_2[0], rate_scale), p.decay_diffusion_2);
        let ap_r2 = Allpass::new(scale_frames(TANK_AP_2[1], rate_scale), p.decay_diffusion_2);
        let delay_l2 = DelayLine::new(scale_frames(TANK_DELAY_2[0], rate_scale));
        let delay_r2 = DelayLine::new(scale_frames(TANK_DELAY_2[1], rate_scale));

        let mut yl_taps = [0usize; 7];
        for (dst, &reference) in yl_taps.iter_mut().zip(YL_TAP_REFERENCE.iter()) {
            *dst = scale_frames(reference, rate_scale);
        }
        let mut yr_taps = [0usize; 7];
        for (dst, &reference) in yr_taps.iter_mut().zip(YR_TAP_REFERENCE.iter()) {
            *dst = scale_frames(reference, rate_scale);
        }

        Self {
            channels,
            rate_scale,
            pre_delay_len,
            pre_delay,
            input_lp: OnePole::new(p.bandwidth),
            in_ap,
            mod_ap_l,
            mod_ap_r,
            delay_l1,
            delay_r1,
            damp_l,
            damp_r,
            ap_l2,
            ap_r2,
            delay_l2,
            delay_r2,
            left_out: 0.0,
            right_out: 0.0,
            decay: p.decay,
            bandwidth: p.bandwidth,
            damping: p.damping,
            yl_taps,
            yr_taps,
            wet: Smoothed::new(p.wet),
            dry: Smoothed::new(p.dry),
        }
    }

    /// Returns the channel count the reverb was built for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the cached tail-decay control.
    #[inline]
    #[must_use]
    pub fn decay(&self) -> Sample {
        self.decay
    }

    /// Returns the cached high-frequency damping control.
    #[inline]
    #[must_use]
    pub fn damping(&self) -> Sample {
        self.damping
    }

    /// Returns the cached input-bandwidth control.
    #[inline]
    #[must_use]
    pub fn bandwidth(&self) -> Sample {
        self.bandwidth
    }

    /// Returns the current wet-mix target.
    #[inline]
    #[must_use]
    pub fn wet(&self) -> Sample {
        self.wet.target()
    }

    /// Returns the current dry-mix target.
    #[inline]
    #[must_use]
    pub fn dry(&self) -> Sample {
        self.dry.target()
    }

    /// Sets the tail decay in `[0, MAX_COEFFICIENT]` (ignored if non-finite).
    #[inline]
    pub fn set_decay(&mut self, decay: Sample) {
        if decay.is_finite() {
            self.decay = decay.clamp(0.0, MAX_COEFFICIENT);
        }
    }

    /// Sets the high-frequency damping in `[0, 1]`, recomputing both tank
    /// low-pass coefficients in place (ignored if non-finite).
    #[inline]
    pub fn set_damping(&mut self, damping: Sample) {
        if damping.is_finite() {
            self.damping = damping.clamp(0.0, 1.0);
            let coeff = 1.0 - self.damping;
            self.damp_l.set_coeff(coeff);
            self.damp_r.set_coeff(coeff);
        }
    }

    /// Sets the input bandwidth in `[0, 1]` (ignored if non-finite).
    #[inline]
    pub fn set_bandwidth(&mut self, bandwidth: Sample) {
        if bandwidth.is_finite() {
            self.bandwidth = bandwidth.clamp(0.0, 1.0);
            self.input_lp.set_coeff(self.bandwidth);
        }
    }

    /// Sets the decay-diffusion-1 coefficient on both modulated all-passes.
    #[inline]
    pub fn set_decay_diffusion_1(&mut self, g: Sample) {
        if g.is_finite() {
            let g = g.clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT);
            self.mod_ap_l.set_g(g);
            self.mod_ap_r.set_g(g);
        }
    }

    /// Sets the decay-diffusion-2 coefficient on both fixed tank all-passes.
    #[inline]
    pub fn set_decay_diffusion_2(&mut self, g: Sample) {
        if g.is_finite() {
            let g = g.clamp(-MAX_COEFFICIENT, MAX_COEFFICIENT);
            self.ap_l2.set_g(g);
            self.ap_r2.set_g(g);
        }
    }

    /// Sets the modulation depth in `[0, 1]`, rescaling both LFO excursions
    /// (ignored if non-finite).
    #[inline]
    pub fn set_mod_depth(&mut self, depth: Sample) {
        if depth.is_finite() {
            let excursion = MOD_EXCURSION_FRAMES * self.rate_scale * depth.clamp(0.0, 1.0);
            self.mod_ap_l.set_excursion(excursion);
            self.mod_ap_r.set_excursion(excursion);
        }
    }

    /// Sets the wet mix gain, gliding with `ramp` (ignored if non-finite).
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        if wet.is_finite() {
            self.wet.set_target(wet, ramp);
        }
    }

    /// Sets the dry mix gain, gliding with `ramp` (ignored if non-finite).
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        if dry.is_finite() {
            self.dry.set_target(dry, ramp);
        }
    }

    /// Advances the tank by one input sample, returning the stereo tap pair.
    #[inline]
    fn tick(&mut self, xin: Sample) -> (Sample, Sample) {
        // Pre-delay (bypassed when zero frames were requested).
        let pre = if self.pre_delay_len == 0 {
            xin
        } else {
            let delayed = self.pre_delay.read(self.pre_delay_len);
            self.pre_delay.write(xin);
            delayed
        };

        // Input bandwidth low-pass, then four series diffusers.
        let mut diffused = self.input_lp.process(pre);
        for ap in &mut self.in_ap {
            diffused = ap.process(diffused);
        }

        let decay = self.decay;

        // Cross-coupled half-loop inputs (previous sample's opposite outputs).
        let left_in = diffused + self.right_out;
        let right_in = diffused + self.left_out;

        // Left half.
        let a_l = self.mod_ap_l.process(left_in);
        self.delay_l1.write(a_l);
        let d1_l = self.delay_l1.read(self.delay_l1.len());
        let damped_l = self.damp_l.process(d1_l) * decay;
        let b_l = self.ap_l2.process(damped_l);
        self.delay_l2.write(b_l);
        let d2_l = self.delay_l2.read(self.delay_l2.len());
        let new_left_out = flush_denormal(d2_l * decay);

        // Right half.
        let a_r = self.mod_ap_r.process(right_in);
        self.delay_r1.write(a_r);
        let d1_r = self.delay_r1.read(self.delay_r1.len());
        let damped_r = self.damp_r.process(d1_r) * decay;
        let b_r = self.ap_r2.process(damped_r);
        self.delay_r2.write(b_r);
        let d2_r = self.delay_r2.read(self.delay_r2.len());
        let new_right_out = flush_denormal(d2_r * decay);

        self.left_out = new_left_out;
        self.right_out = new_right_out;

        // Seven signed taps per output drawn from the two halves' delays.
        let yl = self.delay_r1.read(self.yl_taps[0])
            + self.delay_r1.read(self.yl_taps[1])
            - self.ap_r2.tap(self.yl_taps[2])
            + self.delay_r2.read(self.yl_taps[3])
            - self.delay_l1.read(self.yl_taps[4])
            - self.ap_l2.tap(self.yl_taps[5])
            - self.delay_l2.read(self.yl_taps[6]);
        let yr = self.delay_l1.read(self.yr_taps[0])
            + self.delay_l1.read(self.yr_taps[1])
            - self.ap_l2.tap(self.yr_taps[2])
            + self.delay_l2.read(self.yr_taps[3])
            - self.delay_r1.read(self.yr_taps[4])
            - self.ap_r2.tap(self.yr_taps[5])
            - self.delay_r2.read(self.yr_taps[6]);

        (yl, yr)
    }
}

impl AudioNode for PlateReverb {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let in_channels = input.channels();
        let out_channels = output.channels().min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        if in_channels == 0 || out_channels == 0 {
            return;
        }

        let norm = 1.0 / in_channels as Sample;
        for f in 0..frames {
            // Downmix the input to the mono tank excitation.
            let mut sum = 0.0;
            for ch in 0..in_channels {
                sum += input.channel(ch)[f];
            }
            let xin = sum * norm;

            let (yl, yr) = self.tick(xin);

            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            if out_channels == 1 {
                let x = input.channel(0)[f];
                output.channel_mut(0)[f] = dry * x + wet * (yl + yr) * 0.5;
            } else {
                for ch in 0..out_channels {
                    let x = input.channel(ch)[f];
                    let tap = if ch % 2 == 0 { yl } else { yr };
                    output.channel_mut(ch)[f] = dry * x + wet * tap;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.pre_delay.clear();
        self.input_lp.clear();
        for ap in &mut self.in_ap {
            ap.clear();
        }
        self.mod_ap_l.clear();
        self.mod_ap_r.clear();
        self.delay_l1.clear();
        self.delay_r1.clear();
        self.damp_l.clear();
        self.damp_r.clear();
        self.ap_l2.clear();
        self.ap_r2.clear();
        self.delay_l2.clear();
        self.delay_r2.clear();
        self.left_out = 0.0;
        self.right_out = 0.0;
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    /// Feeds a single unit impulse, then `frames - 1` zeros, returning the
    /// left-channel impulse response of a default stereo plate.
    fn impulse_response(frames: usize, params: PlateReverbParams) -> Vec<Sample> {
        let mut plate = PlateReverb::new(48_000, 2, params);
        let block = 256;
        let mut out = Vec::with_capacity(frames);
        let mut remaining = frames;
        let mut first = true;
        while remaining > 0 {
            let n = block.min(remaining);
            let mut input = stereo(n);
            let mut output = stereo(n);
            input.set_active_frames(n);
            output.set_active_frames(n);
            if first {
                input.channel_mut(0)[0] = 1.0;
                input.channel_mut(1)[0] = 1.0;
                first = false;
            }
            let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
            plate.process(&ctx(n), &mut io);
            out.extend_from_slice(output.channel(0));
            remaining -= n;
        }
        out
    }

    fn energy(samples: &[Sample]) -> f64 {
        samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }

    #[test]
    fn impulse_response_decays_and_is_bounded() {
        let ir = impulse_response(48_000, PlateReverbParams::default());
        let total = energy(&ir);
        assert!(total > 0.0, "plate must produce a non-trivial tail");
        let head = energy(&ir[..12_000]);
        let tail = energy(&ir[36_000..]);
        assert!(tail < head, "late energy {tail} must fall below early {head}");
        let peak = ir.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!(peak.is_finite() && peak < 8.0, "output must stay bounded: {peak}");
    }

    #[test]
    fn longer_decay_lengthens_the_tail() {
        let short = PlateReverbParams {
            decay: 0.3,
            ..PlateReverbParams::default()
        };
        let long = PlateReverbParams {
            decay: 0.85,
            ..PlateReverbParams::default()
        };
        let ir_short = impulse_response(48_000, short);
        let ir_long = impulse_response(48_000, long);
        let tail_short = energy(&ir_short[36_000..]);
        let tail_long = energy(&ir_long[36_000..]);
        assert!(
            tail_long > tail_short,
            "longer decay should retain more late energy: {tail_long} vs {tail_short}"
        );
    }

    #[test]
    fn more_damping_darkens_the_tail() {
        fn hf_energy(ir: &[Sample]) -> f64 {
            // First-difference energy as a proxy for high-frequency content.
            ir.windows(2)
                .map(|w| {
                    let d = f64::from(w[1]) - f64::from(w[0]);
                    d * d
                })
                .sum()
        }
        let bright = PlateReverbParams {
            damping: 0.0,
            ..PlateReverbParams::default()
        };
        let dark = PlateReverbParams {
            damping: 0.9,
            ..PlateReverbParams::default()
        };
        let ir_bright = impulse_response(24_000, bright);
        let ir_dark = impulse_response(24_000, dark);
        let hf_bright = hf_energy(&ir_bright[8_000..]);
        let hf_dark = hf_energy(&ir_dark[8_000..]);
        assert!(
            hf_dark < hf_bright,
            "more damping should remove high-frequency energy: {hf_dark} vs {hf_bright}"
        );
    }

    #[test]
    fn stereo_outputs_are_decorrelated() {
        let mut plate = PlateReverb::new(48_000, 2, PlateReverbParams::default());
        let n = 8_000;
        let mut input = stereo(n);
        let mut output = stereo(n);
        input.set_active_frames(n);
        output.set_active_frames(n);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        plate.process(&ctx(n), &mut io);
        // Compare the two output channels past the dry impulse.
        let left = &output.channel(0)[64..];
        let right = &output.channel(1)[64..];
        let mut diff = 0.0_f64;
        for (&l, &r) in left.iter().zip(right.iter()) {
            diff += f64::from(l - r).abs();
        }
        assert!(diff > 1e-3, "left and right tails must differ: {diff}");
    }

    #[test]
    fn wet_zero_is_pure_dry() {
        let params = PlateReverbParams {
            wet: 0.0,
            dry: 1.0,
            pre_delay_ms: 0.0,
            ..PlateReverbParams::default()
        };
        let mut plate = PlateReverb::new(48_000, 2, params);
        let n = 128;
        let mut input = stereo(n);
        let mut output = stereo(n);
        input.set_active_frames(n);
        output.set_active_frames(n);
        for f in 0..n {
            let v = 0.1 * (f as Sample);
            input.channel_mut(0)[f] = v;
            input.channel_mut(1)[f] = -v;
        }
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        plate.process(&ctx(n), &mut io);
        for f in 0..n {
            assert!((output.channel(0)[f] - input.channel(0)[f]).abs() < 1e-6);
            assert!((output.channel(1)[f] - input.channel(1)[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn dry_zero_is_pure_wet() {
        let params = PlateReverbParams {
            wet: 1.0,
            dry: 0.0,
            ..PlateReverbParams::default()
        };
        let ir = impulse_response(4_000, params);
        assert!(energy(&ir) > 0.0, "wet-only plate still produces a tail");
    }

    #[test]
    fn deterministic_across_instances() {
        let a = impulse_response(6_000, PlateReverbParams::default());
        let b = impulse_response(6_000, PlateReverbParams::default());
        assert_eq!(a, b, "identical configs must be bit-for-bit reproducible");
    }

    #[test]
    fn reset_restores_initial_state() {
        let params = PlateReverbParams::default();
        let mut plate = PlateReverb::new(48_000, 2, params);
        let n = 1_000;
        // Prime with an impulse.
        let mut input = stereo(n);
        let mut output = stereo(n);
        input.set_active_frames(n);
        output.set_active_frames(n);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        {
            let mut io =
                ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
            plate.process(&ctx(n), &mut io);
        }
        plate.reset();
        // After reset, a fresh impulse must match a brand-new instance.
        let fresh = impulse_response(n, params);
        let mut input2 = stereo(n);
        let mut output2 = stereo(n);
        input2.set_active_frames(n);
        output2.set_active_frames(n);
        input2.channel_mut(0)[0] = 1.0;
        input2.channel_mut(1)[0] = 1.0;
        let mut io =
            ProcessIo::new(core::slice::from_ref(&input2), core::slice::from_mut(&mut output2));
        plate.process(&ctx(n), &mut io);
        for f in 0..n {
            assert!((output2.channel(0)[f] - fresh[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn pre_delay_holds_off_the_wet_tail() {
        let params = PlateReverbParams {
            pre_delay_ms: 20.0,
            wet: 1.0,
            dry: 0.0,
            ..PlateReverbParams::default()
        };
        let ir = impulse_response(8_000, params);
        // 20 ms at 48 kHz is 960 frames; the wet tail cannot arrive earlier.
        let pre = ms_to_frames(20.0, 48_000);
        let early = energy(&ir[..pre / 2]);
        assert!(early < 1e-9, "no wet energy before the pre-delay: {early}");
    }

    #[test]
    fn mono_output_is_finite_and_active() {
        let mut plate = PlateReverb::new(48_000, 1, PlateReverbParams::default());
        let n = 2_000;
        let mut input = mono(n);
        let mut output = mono(n);
        input.set_active_frames(n);
        output.set_active_frames(n);
        input.channel_mut(0)[0] = 1.0;
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        plate.process(&ctx(n), &mut io);
        assert!(output.channel(0).iter().all(|s| s.is_finite()));
        assert!(energy(output.channel(0)) > 0.0);
    }

    #[test]
    fn non_finite_params_are_rejected() {
        let params = PlateReverbParams {
            decay: f32::NAN,
            damping: f32::INFINITY,
            bandwidth: f32::NAN,
            wet: f32::NAN,
            ..PlateReverbParams::default()
        };
        let plate = PlateReverb::new(48_000, 2, params);
        assert!(plate.decay().is_finite());
        assert!(plate.damping().is_finite());
        assert!(plate.bandwidth().is_finite());
        assert!(plate.wet().is_finite());
    }

    #[test]
    fn setters_clamp_and_reject() {
        let mut plate = PlateReverb::new(48_000, 2, PlateReverbParams::default());
        plate.set_decay(5.0);
        assert!(plate.decay() <= MAX_COEFFICIENT);
        let before = plate.decay();
        plate.set_decay(f32::NAN);
        assert_eq!(plate.decay(), before, "non-finite decay is ignored");
        plate.set_damping(2.0);
        assert!(plate.damping() <= 1.0);
        plate.set_bandwidth(-1.0);
        assert!(plate.bandwidth() >= 0.0);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut plate = PlateReverb::new(48_000, 2, PlateReverbParams::default());
        let input = stereo(16);
        let mut output = stereo(16);
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        plate.process(&ctx(0), &mut io);
    }
}
