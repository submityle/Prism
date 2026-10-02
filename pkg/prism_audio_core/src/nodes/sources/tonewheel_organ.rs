//! Tonewheel drawbar-organ source node.
//!
//! [`TonewheelOrganNode`] is a *source* (zero inputs, one output) that
//! synthesizes the classic electromechanical drawbar organ (Hammond-family)
//! tone. The instrument's sound comes from a bank of [`NUM_DRAWBARS`]
//! independently rotating tonewheels, each tapped at a fixed musical "footage"
//! above the played key and mixed by its own drawbar slider. This node models
//! each footage as its own phase accumulator running at the historically
//! tempered gear ratio, summed under per-drawbar smoothed levels, with the
//! octave *foldback* that a finite tonewheel generator imposes on the highest
//! footages.
//!
//! # Model
//!
//! The nine drawbars correspond to the fixed organ footages
//!
//! ```text
//! index  footage  interval         semitones  nominal ratio
//!   0     16'      sub-octave        -12         0.5
//!   1     5 1/3'   perfect twelfth    +7         ~1.498 (quint)
//!   2     8'       unison              0         1.0
//!   3     4'       octave            +12         2.0
//!   4     2 2/3'   twelfth           +19         ~2.997
//!   5     2'       fifteenth         +24         4.0
//!   6     1 3/5'   seventeenth       +28         ~5.040 (sharp 17th)
//!   7     1 1/3'   nineteenth        +31         ~5.997
//!   8     1'       twenty-second     +36         8.0
//! ```
//!
//! Each ratio is `2^(semitones / 12)` evaluated as `exp(semitones/12 * ln 2)`,
//! so the quint, twelfth, and especially the seventeenth are *tempered*, not
//! exact integer harmonics. Because every footage advances on its own phase
//! accumulator at that slightly-inharmonic ratio, the partials drift in and out
//! of phase and produce the gentle, ever-shifting shimmer that distinguishes a
//! real tonewheel generator from a phase-locked harmonic series.
//!
//! # Foldback
//!
//! A real tonewheel generator has a finite number of wheels, so the highest
//! footages of high keys cannot be produced and are instead taken an octave (or
//! more) lower -- the audible "foldback". A footage whose frequency exceeds
//! [`FOLDBACK_HZ`] is repeatedly halved until it falls below that ceiling, so
//! the brightest footages roll back into the generator's range exactly as the
//! hardware does.
//!
//! # Level normalization
//!
//! The summed drawbars are divided by the running sum of their active levels
//! whenever that sum exceeds one, so the mix stays within the single-drawbar
//! envelope regardless of how many drawbars are pulled out, exactly like a
//! normalized additive partial set. Per-drawbar levels and the master amplitude
//! are [`Smoothed`], so moving a drawbar while a chord holds is click-free.
//!
//! # Determinism
//!
//! Every footage is a plain phase accumulator advanced by closed-form
//! [`bevy_math::ops`] trigonometry, and the tempered ratios use
//! `exp(x * ln 2)` from the exact [`core::f32::consts::LN_2`] constant, so a
//! given `(sample_rate, params)` reproduces bit-identical audio on every
//! platform. Any footage still above the Nyquist guard after foldback is muted.
//!
//! # Real-time contract
//!
//! `process` performs no allocation, no locking, and no panics. All footage
//! state lives in fixed-size arrays sized by the compile-time [`NUM_DRAWBARS`];
//! frequency changes recompute the (bounded) foldback off the audio hot path.
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], and denormal-flush primitives,
//! and pairs naturally with the [`super::super::effects::leslie::LeslieNode`]
//! rotary cabinet. It differs structurally from
//! [`super::additive_oscillator::AdditiveOscillatorNode`]: that node sums
//! *integer* harmonics (`k >= 1`) that are all phase-locked to one fundamental
//! phase, so it can represent neither a sub-octave (`0.5x`) footage nor the
//! tempered, non-integer quint/seventeenth ratios, nor the octave foldback, and
//! its perfectly locked partials never beat. Here each footage is an
//! independent, tempered, foldback-limited accumulator -- the electromechanical
//! generator, not a Fourier series. It also differs from the single-waveform
//! [`super::oscillator::OscillatorNode`] and from the static-table
//! [`super::wavetable_oscillator::WavetableOscillatorNode`].
//!
//! # Provenance
//!
//! Classic public-domain DSP only; no third-party engine, library, or toolkit
//! source or derivative was consulted or copied. Additive/tonewheel drawbar
//! synthesis -- a sum of sinusoids at fixed musical footages -- is textbook
//! signal processing; the specific footage intervals and the tempered gear
//! ratios (notably the sharp seventeenth) and the finite-generator octave
//! foldback are long-documented public facts about electromechanical drawbar
//! organs (Fletcher & Rossing, "The Physics of Musical Instruments"). The
//! phase-accumulator oscillator is standard. No code from Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK
//! was referenced.

use bevy_math::ops;
use core::f32::consts::{LN_2, TAU};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of drawbar footages in the tonewheel generator.
pub const NUM_DRAWBARS: usize = 9;

/// Semitone offset of each footage above the played unison (8') key.
///
/// Indices follow the classic drawbar order
/// `16', 5 1/3', 8', 4', 2 2/3', 2', 1 3/5', 1 1/3', 1'`.
pub const DRAWBAR_SEMITONES: [i32; NUM_DRAWBARS] = [-12, 7, 0, 12, 19, 24, 28, 31, 36];

/// Lowest tunable key fundamental in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;
/// Highest tunable key fundamental in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;
/// Default key fundamental in hertz (A3).
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Ceiling above which a footage folds back an octave (top-tonewheel limit).
pub const FOLDBACK_HZ: Sample = 5_920.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Fraction of Nyquist above which a footage is muted (anti-aliasing guard).
pub const NYQUIST_GUARD: Sample = 0.49;

/// Default drawbar registration (`88 8800 000`): sub, quint, unison, and octave
/// pulled fully out for a warm, full organ voice.
pub const DEFAULT_DRAWBARS: [Sample; NUM_DRAWBARS] =
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Overall output scale keeping the normalized drawbar sum below full scale.
///
/// Calibrated so the deterministic module parameter-grid test (every drawbar
/// at full with `amplitude == 1`) peaks near `0.77`, leaving headroom to full
/// scale while the normalized partial sum is bounded by one.
pub const OUTPUT_GAIN: Sample = 0.8;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a key fundamental to the tunable range.
#[inline]
fn clamp_frequency(freq_hz: Sample) -> Sample {
    freq_hz.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
}

/// Construction parameters for a [`TonewheelOrganNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TonewheelOrganParams {
    /// Played key fundamental in hertz (the 8' unison pitch).
    pub frequency_hz: Sample,
    /// Per-drawbar levels in `[0, 1]`, in the classic footage order.
    pub drawbars: [Sample; NUM_DRAWBARS],
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for TonewheelOrganParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            drawbars: DEFAULT_DRAWBARS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl TonewheelOrganParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let frequency_hz = clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz));
        let mut drawbars = [0.0; NUM_DRAWBARS];
        for (i, slot) in drawbars.iter_mut().enumerate() {
            *slot = finite_or(self.drawbars[i], d.drawbars[i]).clamp(0.0, 1.0);
        }
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            drawbars,
            amplitude,
        }
    }
}

/// A tonewheel drawbar-organ source.
///
/// See the [module documentation](self) for the model, the foldback and
/// normalization rules, the determinism guarantee, and the real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{TonewheelOrganNode, TonewheelOrganParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = TonewheelOrganNode::new(48_000, TonewheelOrganParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The default registration produces a sustained, bounded, finite tone.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak < 1.0 && peak.is_finite());
/// ```
pub struct TonewheelOrganNode {
    sample_rate: u32,
    frequency_hz: Sample,
    drawbars: [Smoothed; NUM_DRAWBARS],
    amplitude: Smoothed,
    /// Per-footage phase accumulators in radians.
    phase: [Sample; NUM_DRAWBARS],
    /// Per-footage frequency in hertz after tempering and foldback.
    freq: [Sample; NUM_DRAWBARS],
    /// Whether each footage is audible (below the Nyquist guard).
    enabled: [bool; NUM_DRAWBARS],
}

impl TonewheelOrganNode {
    /// Builds an organ voice for `sample_rate` from `params`, sanitising every
    /// field. The default registration makes it sound immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: TonewheelOrganParams) -> Self {
        let p = params.sanitised();
        let drawbars = core::array::from_fn(|i| Smoothed::new(p.drawbars[i]));
        let mut node = Self {
            sample_rate,
            frequency_hz: p.frequency_hz,
            drawbars,
            amplitude: Smoothed::new(p.amplitude),
            phase: [0.0; NUM_DRAWBARS],
            freq: [0.0; NUM_DRAWBARS],
            enabled: [false; NUM_DRAWBARS],
        };
        node.recompute();
        node
    }

    /// Returns the played key fundamental in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target level of drawbar `index` in `[0, 1]`, or `0` if the
    /// index is out of range.
    #[must_use]
    pub fn drawbar(&self, index: usize) -> Sample {
        self.drawbars
            .get(index)
            .map(Smoothed::target)
            .unwrap_or(0.0)
    }

    /// Returns the sounding frequency of footage `index` in hertz (after
    /// tempering and foldback), or `0` if the index is out of range.
    #[must_use]
    pub fn drawbar_frequency(&self, index: usize) -> Sample {
        self.freq.get(index).copied().unwrap_or(0.0)
    }

    /// Reports whether footage `index` is currently audible (below Nyquist).
    #[must_use]
    pub fn mode_enabled(&self, index: usize) -> bool {
        self.enabled.get(index).copied().unwrap_or(false)
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the played key fundamental, clamped to the tunable range.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz = clamp_frequency(finite_or(frequency_hz, self.frequency_hz));
        self.recompute();
    }

    /// Sets the target level of drawbar `index`, clamped to `[0, 1]`, gliding
    /// over `ramp`. Out-of-range indices are ignored.
    pub fn set_drawbar(&mut self, index: usize, level: Sample, ramp: Ramp) {
        if let Some(bar) = self.drawbars.get_mut(index) {
            let target = finite_or(level, bar.target()).clamp(0.0, 1.0);
            bar.set_target(target, ramp);
        }
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Recomputes every footage frequency from the key fundamental, applying the
    /// tempered gear ratio, octave foldback, and Nyquist guard. Never runs on
    /// the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;
        let ceiling = FOLDBACK_HZ.min(nyquist);
        for (i, &semitones) in DRAWBAR_SEMITONES.iter().enumerate() {
            // ratio = 2^(semitones / 12) = exp(semitones/12 * ln 2).
            let ratio = ops::exp((semitones as Sample / 12.0) * LN_2);
            let mut f = self.frequency_hz * ratio;
            // Finite-generator octave foldback: fold the brightest footages down
            // until they fall under the top-tonewheel ceiling (bounded loop).
            while f > ceiling && f > 0.0 {
                f *= 0.5;
            }
            if f > 0.0 && f < nyquist {
                self.freq[i] = f;
                self.enabled[i] = true;
            } else {
                self.freq[i] = 0.0;
                self.enabled[i] = false;
            }
        }
    }

    /// Renders one mono output sample, advancing every footage by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let mut acc = 0.0;
        let mut level_sum = 0.0;
        for i in 0..NUM_DRAWBARS {
            // Advance every smoother each sample so levels stay time-aligned.
            let level = self.drawbars[i].next_sample();
            if !self.enabled[i] {
                continue;
            }
            level_sum += level;
            // Step < TAU * NYQUIST_GUARD < TAU, so one fold keeps phase in
            // [0, TAU).
            let mut ph = self.phase[i] + TAU * self.freq[i] / sr;
            if ph >= TAU {
                ph -= TAU;
            }
            self.phase[i] = ph;
            acc += level * ops::sin(ph);
        }
        // Normalize by the active level sum when it exceeds one, keeping the mix
        // inside the single-drawbar envelope without altering relative timbre.
        let norm = if level_sum > 1.0 { 1.0 / level_sum } else { 1.0 };
        let amp = self.amplitude.next_sample();
        flush_denormal(acc * norm * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for TonewheelOrganNode {
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
        self.phase = [0.0; NUM_DRAWBARS];
        for bar in &mut self.drawbars {
            *bar = Smoothed::new(bar.target());
        }
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
    fn render(node: &mut TonewheelOrganNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut TonewheelOrganNode,
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

    /// Registration with every drawbar fully out.
    fn all_drawbars_params(frequency_hz: Sample, amplitude: Sample) -> TonewheelOrganParams {
        TonewheelOrganParams {
            frequency_hz,
            drawbars: [1.0; NUM_DRAWBARS],
            amplitude,
        }
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let out = render(&mut node, SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p < 1.0, "expected a bounded, audible peak: {p}");
        assert!(out.iter().all(|s| s.is_finite()), "all samples must be finite");
    }

    #[test]
    fn peak_grid_stays_below_full_scale() {
        // Worst-case headroom check: every drawbar fully out, amplitude 1,
        // across the tunable fundamental grid.
        let mut worst = 0.0_f32;
        for &freq in &[MIN_FREQUENCY_HZ, 55.0, DEFAULT_FREQUENCY_HZ, 440.0, 1000.0, MAX_FREQUENCY_HZ]
        {
            let mut node = TonewheelOrganNode::new(SR, all_drawbars_params(freq, 1.0));
            let out = render(&mut node, SR as usize);
            worst = worst.max(peak(&out));
        }
        assert!(worst < 1.0, "grid peak should stay below full scale: {worst}");
        assert!(worst > 0.4, "grid peak should use the headroom: {worst}");
    }

    #[test]
    fn tempered_seventeenth_is_sharp() {
        // Footage 6 (1 3/5', seventeenth) sounds a *tempered* ~5.0397x, which is
        // noticeably sharper than the just 5x. This temperament is the signature
        // of the real tonewheel gear ratios.
        let node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let ratio = node.drawbar_frequency(6) / node.frequency_hz();
        assert!(
            ratio > 5.02 && ratio < 5.06,
            "seventeenth should be tempered-sharp (~5.04x): {ratio}"
        );
    }

    #[test]
    fn sub_octave_present() {
        // Footage 0 (16') sounds an octave below the played key.
        let node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let ratio = node.drawbar_frequency(0) / node.frequency_hz();
        assert!(
            (ratio - 0.5).abs() < 1e-3,
            "16' footage should be the sub-octave (0.5x): {ratio}"
        );
    }

    #[test]
    fn foldback_halves_high_footage() {
        // At a high key the brightest footage exceeds the generator ceiling and
        // folds down an octave at a time until it is below FOLDBACK_HZ.
        let freq = 2_000.0;
        let node = TonewheelOrganNode::new(SR, all_drawbars_params(freq, 0.5));
        let raw = freq * ops::exp((DRAWBAR_SEMITONES[8] as Sample / 12.0) * LN_2);
        assert!(raw > FOLDBACK_HZ, "setup: raw footage should exceed ceiling");
        let mut expected = raw;
        while expected > FOLDBACK_HZ {
            expected *= 0.5;
        }
        let got = node.drawbar_frequency(8);
        assert!(
            (got - expected).abs() < 1e-2 && got <= FOLDBACK_HZ,
            "footage 8 should fold back to {expected}: {got}"
        );
    }

    #[test]
    fn independent_phases_produce_nonharmonic_partials() {
        // Each footage runs on its own accumulator at a tempered ratio, so a
        // sharp seventeenth injects spectral energy at a non-integer multiple of
        // the fundamental that a phase-locked harmonic series could not.
        let node_freq = DEFAULT_FREQUENCY_HZ;
        let mut node = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: node_freq,
                // Only unison + seventeenth so the tempered partial is isolated.
                drawbars: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                amplitude: 0.8,
            },
        );
        let out = render(&mut node, SR as usize);
        let tempered = node.drawbar_frequency(6);
        let just = node_freq * 5.0;
        let at_tempered = goertzel(&out, tempered);
        let at_just = goertzel(&out, just);
        assert!(
            at_tempered > at_just * 4.0,
            "energy should sit at the tempered 17th {tempered}, not the just 5x {just}: \
             tempered={at_tempered} just={at_just}"
        );
    }

    #[test]
    fn drawbar_changes_timbre() {
        // Two registrations with the same fundamental differ spectrally.
        let mut unison = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: DEFAULT_FREQUENCY_HZ,
                drawbars: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                amplitude: 0.8,
            },
        );
        let mut bright = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: DEFAULT_FREQUENCY_HZ,
                drawbars: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                amplitude: 0.8,
            },
        );
        let a = render(&mut unison, SR as usize);
        let b = render(&mut bright, SR as usize);
        let fifteenth = DEFAULT_FREQUENCY_HZ * 4.0;
        let ga = goertzel(&a, fifteenth);
        let gb = goertzel(&b, fifteenth);
        assert!(
            gb > ga * 8.0,
            "pulling the 2' footage should add fifteenth energy: unison={ga} bright={gb}"
        );
    }

    #[test]
    fn normalization_keeps_full_registration_bounded() {
        // Nine drawbars out must not peak proportionally louder than one.
        let mut one = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: DEFAULT_FREQUENCY_HZ,
                drawbars: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                amplitude: 1.0,
            },
        );
        let mut full = TonewheelOrganNode::new(SR, all_drawbars_params(DEFAULT_FREQUENCY_HZ, 1.0));
        let p_one = peak(&render(&mut one, SR as usize));
        let p_full = peak(&render(&mut full, SR as usize));
        assert!(p_full < 1.0, "full registration must stay bounded: {p_full}");
        assert!(
            p_full < p_one * 2.0,
            "normalization should keep the full mix comparable: one={p_one} full={p_full}"
        );
    }

    #[test]
    fn deterministic() {
        let mut a = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let mut b = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        assert_eq!(render(&mut a, SR as usize), render(&mut b, SR as usize));
    }

    #[test]
    fn reset_replays() {
        let mut node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let first = render(&mut node, SR as usize);
        node.reset();
        let second = render(&mut node, SR as usize);
        assert_eq!(first, second, "reset must replay the identical tone");
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                amplitude: 0.0,
                ..TonewheelOrganParams::default()
            },
        );
        let out = render(&mut node, SR as usize);
        assert_eq!(peak(&out), 0.0, "zero amplitude must be silent");
    }

    #[test]
    fn silent_when_all_drawbars_closed() {
        let mut node = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                drawbars: [0.0; NUM_DRAWBARS],
                ..TonewheelOrganParams::default()
            },
        );
        let out = render(&mut node, SR as usize);
        assert_eq!(peak(&out), 0.0, "closed drawbars must be silent");
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut loud = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                amplitude: 1.0,
                ..TonewheelOrganParams::default()
            },
        );
        let mut soft = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                amplitude: 0.5,
                ..TonewheelOrganParams::default()
            },
        );
        let el = energy(&render(&mut loud, SR as usize));
        let es = energy(&render(&mut soft, SR as usize));
        assert!(
            (el / es - 4.0).abs() < 0.05,
            "halving amplitude should quarter energy: ratio={}",
            el / es
        );
    }

    #[test]
    fn mono_core_copied_to_channels() {
        let mut node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let chans = render_layout(&mut node, SR as usize, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..4 {
            assert_eq!(chans[0], chans[ch], "channel {ch} must mirror the mono core");
        }
    }

    #[test]
    fn zero_frames_noop() {
        let mut node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty(), "zero active frames must render nothing");
    }

    #[test]
    fn latency_is_zero() {
        let node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_configuration() {
        let node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
        for i in 0..NUM_DRAWBARS {
            assert_eq!(node.drawbar(i), DEFAULT_DRAWBARS[i]);
        }
        // Out-of-range indices are reported as inert rather than panicking.
        assert_eq!(node.drawbar(NUM_DRAWBARS), 0.0);
        assert_eq!(node.drawbar_frequency(NUM_DRAWBARS), 0.0);
        assert!(!node.mode_enabled(NUM_DRAWBARS));
    }

    #[test]
    fn default_params_in_domain() {
        let p = TonewheelOrganParams::default();
        assert_eq!(p, p.sanitised(), "defaults must already be in-domain");
    }

    #[test]
    fn sanitise_clamps_and_replaces_non_finite() {
        let p = TonewheelOrganParams {
            frequency_hz: Sample::NAN,
            drawbars: [2.0, -1.0, Sample::INFINITY, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            amplitude: Sample::NAN,
        }
        .sanitised();
        assert_eq!(p.frequency_hz, DEFAULT_FREQUENCY_HZ);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
        assert_eq!(p.drawbars[0], 1.0, "2.0 clamps to 1");
        assert_eq!(p.drawbars[1], 0.0, "-1.0 clamps to 0");
        assert_eq!(p.drawbars[2], DEFAULT_DRAWBARS[2], "inf falls back to default");
        assert_eq!(p.drawbars[3], 0.5);
    }

    #[test]
    fn out_of_range_frequency_is_clamped() {
        let low = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: 1.0,
                ..TonewheelOrganParams::default()
            },
        );
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        let high = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: 1.0e6,
                ..TonewheelOrganParams::default()
            },
        );
        assert_eq!(high.frequency_hz(), MAX_FREQUENCY_HZ);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        node.set_frequency(Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ, "NaN keeps prior freq");
        node.set_frequency(1.0e9);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ, "huge freq clamps");

        node.set_drawbar(2, 5.0, Ramp::Immediate);
        assert_eq!(node.drawbar(2), 1.0, "level clamps to 1");
        node.set_drawbar(2, Sample::NAN, Ramp::Immediate);
        assert_eq!(node.drawbar(2), 1.0, "NaN level keeps prior target");
        // Out-of-range drawbar index is ignored, not a panic.
        node.set_drawbar(NUM_DRAWBARS, 1.0, Ramp::Immediate);

        node.set_amplitude(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE, "inf amplitude keeps prior");
        node.set_amplitude(0.25, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.25);
    }

    #[test]
    fn frequency_change_shifts_spectrum() {
        let mut node = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                frequency_hz: 220.0,
                drawbars: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                amplitude: 0.8,
            },
        );
        let before = render(&mut node, SR as usize);
        node.set_frequency(330.0);
        assert_eq!(node.drawbar_frequency(2), 330.0, "unison footage tracks the key");
        let after = render(&mut node, SR as usize);
        assert!(goertzel(&before, 220.0) > goertzel(&after, 220.0) * 4.0);
        assert!(goertzel(&after, 330.0) > goertzel(&before, 330.0) * 4.0);
    }

    #[test]
    fn set_drawbar_tracks_target() {
        let mut node = TonewheelOrganNode::new(
            SR,
            TonewheelOrganParams {
                drawbars: [0.0; NUM_DRAWBARS],
                ..TonewheelOrganParams::default()
            },
        );
        assert_eq!(peak(&render(&mut node, 16)), 0.0, "starts closed");
        node.set_drawbar(2, 1.0, Ramp::Immediate);
        assert_eq!(node.drawbar(2), 1.0);
        let out = render(&mut node, SR as usize);
        assert!(peak(&out) > 0.0, "opening a drawbar should produce sound");
    }

    #[test]
    fn mode_enabled_reports_audible_footages() {
        // Default-key footages are all well below Nyquist and audible.
        let node = TonewheelOrganNode::new(SR, TonewheelOrganParams::default());
        for i in 0..NUM_DRAWBARS {
            assert!(node.mode_enabled(i), "footage {i} should be audible at A3");
        }
    }
}
