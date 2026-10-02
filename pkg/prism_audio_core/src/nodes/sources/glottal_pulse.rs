//! Rosenberg-model glottal pulse source node.
//!
//! [`GlottalPulseNode`] is a *source* (zero inputs, one output) that synthesizes
//! the glottal volume-velocity pulse of the human voice -- the periodic airflow
//! waveform produced by the vibrating vocal folds *before* it is coloured by the
//! vocal tract. In the classic **source-filter** model of speech, this node is
//! the *source*: its buzz is meant to be filtered by a formant resonator bank
//! (see [`super::super::effects::formant_filter`]) to produce vowels. On its own
//! it is a warm, buzzy, vocal-sounding drone whose spectral slope and brightness
//! track the glottal shape controls.
//!
//! # Model
//!
//! The pulse follows the **Rosenberg "C" model** of glottal flow: within each
//! fundamental period the glottis opens, reaches peak flow, closes, and then
//! stays shut for the remainder of the period. With a normalized phase
//! `phi in [0, 1)` advancing by `f0 / fs` each sample, the open interval spans
//! `[0, Oq)` where `Oq` is the **open quotient**, split into an opening segment
//! of width `Op` and a closing segment of width `Cl = Oq - Op`. The opening /
//! closing split is set by the **speed quotient** `Sq = Op / Cl` (how much
//! faster the fold closes than it opens -- a sharper close is brighter):
//!
//! ```text
//!   Op = Oq * Sq / (Sq + 1)        (opening-segment width in phase)
//!   Cl = Oq / (Sq + 1)             (closing-segment width in phase)
//!
//!   flow(phi) =
//!     0.5 * (1 - cos(pi * phi / Op))              for 0  <= phi < Op   (open)
//!     cos(pi * (phi - Op) / (2 * Cl))             for Op <= phi < Oq   (close)
//!     0                                           for Oq <= phi < 1    (shut)
//! ```
//!
//! `flow` rises smoothly from `0` to its peak `1` at `phi == Op`, falls back to
//! `0` at `phi == Oq`, and is identically zero through the closed phase, so the
//! waveform is continuous across the whole period. The glottal-flow *derivative*
//! -- the quantity that actually excites the vocal tract once lip radiation is
//! folded in -- is the closed-form slope of the above:
//!
//! ```text
//!   flow'(phi) =
//!      (pi / (2 Op)) * sin(pi * phi / Op)         (open)
//!     -(pi / (2 Cl)) * sin(pi * (phi - Op) / (2 Cl))   (close)
//!      0                                          (shut)
//! ```
//!
//! The derivative is continuous everywhere except at the **glottal closure
//! instant** `phi == Oq`, where it jumps from its sharp negative peak back to
//! zero. That single step is the dominant excitation of voiced speech and the
//! sole aliasing source, so it is band-limited with a two-sample `PolyBLEP`
//! correction (shared with [`super::oscillator`]); every other feature is
//! smooth. [`GlottalOutput`] selects whether the node emits the band-limited
//! derivative (the DC-free excitation, the default) or the raw continuous flow.
//!
//! # Determinism
//!
//! The pulse is a closed-form composition of [`bevy_math::ops`] trigonometry and
//! the polynomial `PolyBLEP`; there are no iterative solvers and no table
//! lookups, so a given `(sample_rate, params)` reproduces bit-identical audio on
//! every platform. Any fundamental above the Nyquist guard is muted.
//!
//! # Real-time contract
//!
//! `process` performs no allocation, no locking, and no panics. The pulse shape
//! (`Op`, `Cl`, and the derivative normalization) is recomputed off the audio
//! hot path whenever the open or speed quotient changes; `f0` and amplitude are
//! [`Smoothed`] so sweeps are click-free.
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], denormal-flush, and the
//! crate-internal [`super::oscillator::poly_blep`] primitive, and pairs with the
//! [`super::super::effects::formant_filter::FormantFilterNode`] to form a full
//! source-filter voice. It is distinct from
//! [`super::fof_source::FofSourceNode`], which synthesizes the *filtered* vowel
//! directly by depositing one formant impulse-response grain per period
//! (source and tract fused), whereas this node emits only the *unfiltered*
//! glottal source. It differs from [`super::impulse_train::ImpulseTrainNode`],
//! an ideal band-limited Dirichlet spike train of equal-amplitude harmonics,
//! and from the geometric virtual-analog waveforms of
//! [`super::oscillator::OscillatorNode`]: the glottal pulse is a physiologically
//! shaped volume-velocity waveform whose spectral envelope and brightness are
//! governed by the open and speed quotients, not a mathematical spike or a
//! textbook saw/square.
//!
//! # Provenance
//!
//! Classic public-domain DSP only; no third-party engine, library, or toolkit
//! source or derivative was consulted or copied. The glottal-pulse flow shape is
//! the Rosenberg "C" model (A. E. Rosenberg, "Effect of Glottal Pulse Shape on
//! the Quality of Natural Vowels", JASA 1971), a textbook voice-source model
//! (e.g. Rabiner & Schafer, "Digital Processing of Speech Signals"); the
//! source-filter decomposition of voice is Fant's classic acoustic theory. The
//! open/speed quotient parameterization and the `PolyBLEP` band-limiting of a
//! step discontinuity are long-published public techniques. No code from Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web
//! Audio, or STK was referenced.

use bevy_math::ops;
use core::f32::consts::PI;

use super::oscillator::poly_blep;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;
/// Highest tunable fundamental in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 1_000.0;
/// Default fundamental in hertz (a low male-register pitch).
pub const DEFAULT_FREQUENCY_HZ: Sample = 120.0;

/// Smallest open quotient (shortest glottal open phase, brightest/thinnest).
pub const MIN_OPEN_QUOTIENT: Sample = 0.2;
/// Largest open quotient (longest open phase, breathiest).
pub const MAX_OPEN_QUOTIENT: Sample = 0.95;
/// Default open quotient (a relaxed modal phonation).
pub const DEFAULT_OPEN_QUOTIENT: Sample = 0.6;

/// Smallest speed quotient (symmetric open/close pulse).
pub const MIN_SPEED_QUOTIENT: Sample = 1.0;
/// Largest speed quotient (very sharp closure, brightest).
pub const MAX_SPEED_QUOTIENT: Sample = 7.0;
/// Default speed quotient (close roughly twice as fast as the open).
pub const DEFAULT_SPEED_QUOTIENT: Sample = 2.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Fraction of Nyquist above which the pulse is muted (anti-aliasing guard).
pub const NYQUIST_GUARD: Sample = 0.49;

/// Overall output scale keeping the band-limited pulse below full scale.
///
/// Calibrated so the deterministic module parameter-grid test (every shape
/// across the frequency range at `amplitude == 1`) peaks near `0.9`, leaving
/// headroom to full scale after the `PolyBLEP` closure correction.
pub const OUTPUT_GAIN: Sample = 0.9;

/// Which glottal waveform the node emits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum GlottalOutput {
    /// Band-limited glottal-flow derivative: the DC-free voiced excitation that
    /// drives a vocal-tract filter. This is the default.
    #[default]
    FlowDerivative,
    /// Raw continuous glottal flow (volume velocity). Unipolar, so it carries a
    /// shape-dependent DC offset; route through a
    /// [`super::super::effects::DcBlockerNode`] when a DC-free flow is required.
    Flow,
}

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a fundamental to the tunable range.
#[inline]
fn clamp_frequency(freq_hz: Sample) -> Sample {
    freq_hz.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
}

/// Construction parameters for a [`GlottalPulseNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GlottalPulseParams {
    /// Fundamental frequency (voice pitch) in hertz.
    pub frequency_hz: Sample,
    /// Open quotient: fraction of the period the glottis is open, in
    /// `[MIN_OPEN_QUOTIENT, MAX_OPEN_QUOTIENT]`.
    pub open_quotient: Sample,
    /// Speed quotient: ratio of opening to closing duration, in
    /// `[MIN_SPEED_QUOTIENT, MAX_SPEED_QUOTIENT]`. Larger closes faster.
    pub speed_quotient: Sample,
    /// Which glottal waveform to emit.
    pub output: GlottalOutput,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for GlottalPulseParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            open_quotient: DEFAULT_OPEN_QUOTIENT,
            speed_quotient: DEFAULT_SPEED_QUOTIENT,
            output: GlottalOutput::FlowDerivative,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl GlottalPulseParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        Self {
            frequency_hz: clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz)),
            open_quotient: finite_or(self.open_quotient, d.open_quotient)
                .clamp(MIN_OPEN_QUOTIENT, MAX_OPEN_QUOTIENT),
            speed_quotient: finite_or(self.speed_quotient, d.speed_quotient)
                .clamp(MIN_SPEED_QUOTIENT, MAX_SPEED_QUOTIENT),
            output: self.output,
            amplitude: finite_or(self.amplitude, d.amplitude),
        }
    }
}

/// A Rosenberg-model glottal pulse source.
///
/// See the [module documentation](self) for the model, the band-limiting of the
/// closure instant, the determinism guarantee, and the real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{GlottalPulseNode, GlottalPulseParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = GlottalPulseNode::new(48_000, GlottalPulseParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The default voice source is audible, bounded, and finite.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak < 1.0 && peak.is_finite());
/// ```
pub struct GlottalPulseNode {
    sample_rate: u32,
    frequency: Smoothed,
    amplitude: Smoothed,
    open_quotient: Sample,
    speed_quotient: Sample,
    output: GlottalOutput,
    /// Normalized phase in `[0, 1)`.
    phase: Sample,
    /// Opening-segment width in phase units.
    open_frac: Sample,
    /// Closing-segment width in phase units.
    close_frac: Sample,
    /// Peak magnitude of the raw flow derivative, used to normalize to unit peak.
    deriv_peak: Sample,
}

impl GlottalPulseNode {
    /// Builds a glottal source for `sample_rate` from `params`, sanitising every
    /// field. The default shape sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: GlottalPulseParams) -> Self {
        let p = params.sanitised();
        let mut node = Self {
            sample_rate,
            frequency: Smoothed::new(p.frequency_hz),
            amplitude: Smoothed::new(p.amplitude),
            open_quotient: p.open_quotient,
            speed_quotient: p.speed_quotient,
            output: p.output,
            phase: 0.0,
            open_frac: 0.0,
            close_frac: 0.0,
            deriv_peak: 1.0,
        };
        node.recompute_shape();
        node
    }

    /// Returns the target fundamental frequency in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency.target()
    }

    /// Returns the open quotient.
    #[must_use]
    pub fn open_quotient(&self) -> Sample {
        self.open_quotient
    }

    /// Returns the speed quotient.
    #[must_use]
    pub fn speed_quotient(&self) -> Sample {
        self.speed_quotient
    }

    /// Returns the selected output waveform.
    #[must_use]
    pub fn output(&self) -> GlottalOutput {
        self.output
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the target fundamental, clamped to the tunable range, gliding over
    /// `ramp`.
    pub fn set_frequency(&mut self, frequency_hz: Sample, ramp: Ramp) {
        let target = clamp_frequency(finite_or(frequency_hz, self.frequency.target()));
        self.frequency.set_target(target, ramp);
    }

    /// Sets the open quotient, clamped to its range, and recomputes the shape.
    pub fn set_open_quotient(&mut self, open_quotient: Sample) {
        self.open_quotient = finite_or(open_quotient, self.open_quotient)
            .clamp(MIN_OPEN_QUOTIENT, MAX_OPEN_QUOTIENT);
        self.recompute_shape();
    }

    /// Sets the speed quotient, clamped to its range, and recomputes the shape.
    pub fn set_speed_quotient(&mut self, speed_quotient: Sample) {
        self.speed_quotient = finite_or(speed_quotient, self.speed_quotient)
            .clamp(MIN_SPEED_QUOTIENT, MAX_SPEED_QUOTIENT);
        self.recompute_shape();
    }

    /// Selects the output waveform.
    pub fn set_output(&mut self, output: GlottalOutput) {
        self.output = output;
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Recomputes the opening/closing widths and the derivative normalization
    /// from the open and speed quotients. Never runs on the audio hot path.
    fn recompute_shape(&mut self) {
        let oq = self.open_quotient;
        let sq = self.speed_quotient;
        self.open_frac = oq * sq / (sq + 1.0);
        self.close_frac = oq - self.open_frac;
        // Peak |flow'| is pi/(2*Op) on the open side and pi/(2*Cl) on the close
        // side; the sharper (narrower) segment dominates.
        let open_slope = PI / (2.0 * self.open_frac);
        let close_slope = PI / (2.0 * self.close_frac);
        self.deriv_peak = open_slope.max(close_slope);
    }

    /// Raw glottal flow at normalized phase `phi` (peak `1`).
    #[inline]
    fn flow(&self, phi: Sample) -> Sample {
        let op = self.open_frac;
        let oq = self.open_quotient;
        if phi < op {
            0.5 * (1.0 - ops::cos(PI * phi / op))
        } else if phi < oq {
            ops::cos(PI * (phi - op) / (2.0 * self.close_frac))
        } else {
            0.0
        }
    }

    /// Raw glottal-flow derivative at normalized phase `phi` (not yet
    /// normalized or band-limited).
    #[inline]
    fn flow_derivative(&self, phi: Sample) -> Sample {
        let op = self.open_frac;
        let oq = self.open_quotient;
        if phi < op {
            (PI / (2.0 * op)) * ops::sin(PI * phi / op)
        } else if phi < oq {
            -(PI / (2.0 * self.close_frac)) * ops::sin(PI * (phi - op) / (2.0 * self.close_frac))
        } else {
            0.0
        }
    }

    /// Renders one mono output sample and advances the glottal phase.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let f0 = self.frequency.next_sample();
        let amp = self.amplitude.next_sample();
        let nyquist = sr * NYQUIST_GUARD;
        // Mute (but keep advancing time) for a non-positive, non-finite, or
        // above-guard fundamental.
        if !f0.is_finite() || f0 <= 0.0 || f0 >= nyquist {
            return 0.0;
        }
        let dt = f0 / sr;
        let phi = self.phase;

        let value = match self.output {
            GlottalOutput::Flow => self.flow(phi),
            GlottalOutput::FlowDerivative => {
                let raw = self.flow_derivative(phi) / self.deriv_peak;
                // The only step is the rising jump at the glottal closure
                // instant (phi == open_quotient); band-limit it with a PolyBLEP.
                // Its normalized height is the closing peak over deriv_peak.
                let step_height = (PI / (2.0 * self.close_frac)) / self.deriv_peak;
                let mut ts = phi - self.open_quotient;
                if ts < 0.0 {
                    ts += 1.0;
                }
                raw + step_height * poly_blep(ts, dt)
            }
        };

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        flush_denormal(value * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for GlottalPulseNode {
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
        self.phase = 0.0;
        self.frequency = Smoothed::new(self.frequency.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute_shape();
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
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut GlottalPulseNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut GlottalPulseNode,
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
        block.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    /// Single-frequency magnitude via the Goertzel sum (test-only analysis).
    fn goertzel(block: &[Sample], freq: Sample) -> f64 {
        let w = TAU as f64 * (freq as f64) / (SR as f64);
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &s) in block.iter().enumerate() {
            re += (s as f64) * (w * n as f64).cos();
            im -= (s as f64) * (w * n as f64).sin();
        }
        (re * re + im * im).sqrt()
    }

    fn params(frequency_hz: Sample, open_quotient: Sample, speed_quotient: Sample) -> GlottalPulseParams {
        GlottalPulseParams {
            frequency_hz,
            open_quotient,
            speed_quotient,
            output: GlottalOutput::FlowDerivative,
            amplitude: 1.0,
        }
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let out = render(&mut node, SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p < 1.0, "expected a bounded, audible peak: {p}");
        assert!(out.iter().all(|s| s.is_finite()), "all samples must be finite");
    }

    #[test]
    fn peak_grid_stays_below_full_scale() {
        // Worst-case headroom check across the shape grid at amplitude 1.
        let mut worst = 0.0_f32;
        for &freq in &[MIN_FREQUENCY_HZ, 55.0, DEFAULT_FREQUENCY_HZ, 220.0, 440.0, MAX_FREQUENCY_HZ] {
            for &oq in &[MIN_OPEN_QUOTIENT, 0.4, DEFAULT_OPEN_QUOTIENT, MAX_OPEN_QUOTIENT] {
                for &sq in &[MIN_SPEED_QUOTIENT, DEFAULT_SPEED_QUOTIENT, MAX_SPEED_QUOTIENT] {
                    for &output in &[GlottalOutput::FlowDerivative, GlottalOutput::Flow] {
                        let mut p = params(freq, oq, sq);
                        p.output = output;
                        let mut node = GlottalPulseNode::new(SR, p);
                        let out = render(&mut node, SR as usize);
                        worst = worst.max(peak(&out));
                    }
                }
            }
        }
        assert!(worst < 1.0, "grid peak should stay below full scale: {worst}");
        assert!(worst > 0.4, "grid peak should use the headroom: {worst}");
    }

    #[test]
    fn flow_mode_is_unipolar() {
        // Raw glottal flow is a volume velocity: it never goes negative.
        let mut p = params(DEFAULT_FREQUENCY_HZ, DEFAULT_OPEN_QUOTIENT, DEFAULT_SPEED_QUOTIENT);
        p.output = GlottalOutput::Flow;
        let mut node = GlottalPulseNode::new(SR, p);
        let out = render(&mut node, SR as usize);
        assert!(
            out.iter().all(|&s| s >= -1e-6),
            "glottal flow must be non-negative (unipolar)"
        );
        assert!(peak(&out) > 0.1, "flow should be audible");
    }

    #[test]
    fn derivative_mode_is_dc_free() {
        // The flow derivative integrates to zero over a period, so its mean is
        // negligible relative to its peak.
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let out = render(&mut node, SR as usize);
        let mean = (out.iter().map(|&s| s as f64).sum::<f64>() / out.len() as f64).abs();
        let p = peak(&out) as f64;
        assert!(mean < 0.02 * p, "derivative mean {mean} should be << peak {p}");
        // And it genuinely swings both ways.
        assert!(out.iter().any(|&s| s > 0.0) && out.iter().any(|&s| s < 0.0));
    }

    #[test]
    fn sharper_speed_quotient_is_brighter() {
        // A faster closure (larger speed quotient) injects more high-harmonic
        // energy, so a high harmonic is stronger relative to the fundamental.
        let f0 = DEFAULT_FREQUENCY_HZ;
        let brightness = |sq: Sample| {
            let mut node = GlottalPulseNode::new(SR, params(f0, DEFAULT_OPEN_QUOTIENT, sq));
            let out = render(&mut node, SR as usize);
            let high = goertzel(&out, f0 * 16.0);
            let fund = goertzel(&out, f0).max(1e-9);
            high / fund
        };
        let dull = brightness(MIN_SPEED_QUOTIENT);
        let sharp = brightness(MAX_SPEED_QUOTIENT);
        assert!(sharp > dull, "larger speed quotient should be brighter: {sharp} vs {dull}");
    }

    #[test]
    fn open_quotient_changes_spectrum() {
        let f0 = DEFAULT_FREQUENCY_HZ;
        let harmonic = |oq: Sample| {
            let mut node = GlottalPulseNode::new(SR, params(f0, oq, DEFAULT_SPEED_QUOTIENT));
            let out = render(&mut node, SR as usize);
            goertzel(&out, f0 * 8.0)
        };
        let tight = harmonic(MIN_OPEN_QUOTIENT);
        let wide = harmonic(MAX_OPEN_QUOTIENT);
        let rel = (tight - wide).abs() / tight.max(wide).max(1e-9);
        assert!(rel > 0.05, "open quotient should reshape the spectrum: {tight} vs {wide}");
    }

    #[test]
    fn fundamental_present() {
        let f0 = DEFAULT_FREQUENCY_HZ;
        let mut node = GlottalPulseNode::new(SR, params(f0, DEFAULT_OPEN_QUOTIENT, DEFAULT_SPEED_QUOTIENT));
        let out = render(&mut node, SR as usize);
        let fund = goertzel(&out, f0);
        let off = goertzel(&out, f0 * 1.5);
        assert!(fund > off * 4.0, "fundamental {fund} should dominate off-harmonic {off}");
    }

    #[test]
    fn frequency_change_shifts_spectrum() {
        let mut node = GlottalPulseNode::new(SR, params(120.0, DEFAULT_OPEN_QUOTIENT, DEFAULT_SPEED_QUOTIENT));
        let _ = render(&mut node, SR as usize);
        node.set_frequency(240.0, Ramp::Immediate);
        let out = render(&mut node, SR as usize);
        let old = goertzel(&out, 120.0);
        let new = goertzel(&out, 240.0);
        assert!(new > old, "spectrum should follow the new fundamental: {new} vs {old}");
    }

    #[test]
    fn deterministic() {
        let mut a = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let mut b = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        assert_eq!(render(&mut a, 4096), render(&mut b, 4096));
    }

    #[test]
    fn reset_replays() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let first = render(&mut node, 4096);
        node.reset();
        let second = render(&mut node, 4096);
        assert_eq!(first, second, "reset should replay the identical pulse");
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut p = GlottalPulseParams::default();
        p.amplitude = 0.0;
        let mut node = GlottalPulseNode::new(SR, p);
        let out = render(&mut node, 2048);
        assert!(out.iter().all(|&s| s == 0.0), "zero amplitude must be silent");
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let quiet = {
            let mut p = GlottalPulseParams::default();
            p.amplitude = 0.25;
            let mut node = GlottalPulseNode::new(SR, p);
            energy(&render(&mut node, SR as usize))
        };
        let loud = {
            let mut p = GlottalPulseParams::default();
            p.amplitude = 0.5;
            let mut node = GlottalPulseNode::new(SR, p);
            energy(&render(&mut node, SR as usize))
        };
        let ratio = loud / quiet.max(1e-12);
        assert!((ratio - 4.0).abs() < 0.05, "doubling amplitude should quadruple energy: {ratio}");
    }

    #[test]
    fn mono_core_copied_to_channels() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let chans = render_layout(&mut node, 2048, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch], "every channel should mirror the mono core");
        }
    }

    #[test]
    fn zero_frames_noop() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty(), "zero frames must render nothing");
    }

    #[test]
    fn latency_is_zero() {
        let node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_construction() {
        let p = params(200.0, 0.5, 3.0);
        let node = GlottalPulseNode::new(SR, p);
        assert!((node.frequency_hz() - 200.0).abs() < 1e-3);
        assert!((node.open_quotient() - 0.5).abs() < 1e-6);
        assert!((node.speed_quotient() - 3.0).abs() < 1e-6);
        assert_eq!(node.output(), GlottalOutput::FlowDerivative);
        assert!((node.amplitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn default_params_in_domain() {
        let d = GlottalPulseParams::default();
        assert!(d.frequency_hz >= MIN_FREQUENCY_HZ && d.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(d.open_quotient >= MIN_OPEN_QUOTIENT && d.open_quotient <= MAX_OPEN_QUOTIENT);
        assert!(d.speed_quotient >= MIN_SPEED_QUOTIENT && d.speed_quotient <= MAX_SPEED_QUOTIENT);
        assert_eq!(d, d.sanitised());
    }

    #[test]
    fn sanitise_clamps_and_repairs() {
        let dirty = GlottalPulseParams {
            frequency_hz: 10_000.0,
            open_quotient: 2.0,
            speed_quotient: 0.1,
            output: GlottalOutput::Flow,
            amplitude: f32::NAN,
        };
        let clean = dirty.sanitised();
        assert!((clean.frequency_hz - MAX_FREQUENCY_HZ).abs() < 1e-3);
        assert!((clean.open_quotient - MAX_OPEN_QUOTIENT).abs() < 1e-6);
        assert!((clean.speed_quotient - MIN_SPEED_QUOTIENT).abs() < 1e-6);
        assert_eq!(clean.output, GlottalOutput::Flow);
        assert!((clean.amplitude - DEFAULT_AMPLITUDE).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        node.set_frequency(f32::INFINITY, Ramp::Immediate);
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1e-3);
        node.set_frequency(10_000.0, Ramp::Immediate);
        assert!((node.frequency_hz() - MAX_FREQUENCY_HZ).abs() < 1e-3);

        node.set_open_quotient(f32::NAN);
        assert!((node.open_quotient() - DEFAULT_OPEN_QUOTIENT).abs() < 1e-6);
        node.set_open_quotient(0.0);
        assert!((node.open_quotient() - MIN_OPEN_QUOTIENT).abs() < 1e-6);

        node.set_speed_quotient(f32::NAN);
        assert!((node.speed_quotient() - DEFAULT_SPEED_QUOTIENT).abs() < 1e-6);
        node.set_speed_quotient(100.0);
        assert!((node.speed_quotient() - MAX_SPEED_QUOTIENT).abs() < 1e-6);

        node.set_amplitude(f32::NAN, Ramp::Immediate);
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn set_output_switches_waveform() {
        let mut node = GlottalPulseNode::new(SR, GlottalPulseParams::default());
        assert_eq!(node.output(), GlottalOutput::FlowDerivative);
        node.set_output(GlottalOutput::Flow);
        assert_eq!(node.output(), GlottalOutput::Flow);
        let out = render(&mut node, SR as usize);
        assert!(out.iter().all(|&s| s >= -1e-6), "flow output must be unipolar");
    }

    #[test]
    fn muted_above_nyquist_guard() {
        // A low sample rate pushes the tunable fundamental past the guard.
        let low_sr = 2_000;
        let mut node = GlottalPulseNode::new(low_sr, params(990.0, DEFAULT_OPEN_QUOTIENT, DEFAULT_SPEED_QUOTIENT));
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 1024);
        out.set_active_frames(1024);
        let mut outputs = [out];
        let ctx = RenderContext { sample_rate: low_sr, frames: 1024, playhead: 0 };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert!(
            outputs[0].channel(0).iter().all(|&s| s == 0.0),
            "fundamental above the Nyquist guard must be muted"
        );
    }
}
