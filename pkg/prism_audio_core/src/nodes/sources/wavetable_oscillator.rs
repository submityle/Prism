//! Band-limited mipmap wavetable oscillator source node.
//!
//! [`WavetableOscillatorNode`] is a *source* (zero inputs, one output) that
//! reads a single-cycle waveform from a precomputed table. Unlike the geometric
//! [`OscillatorNode`](super::oscillator::OscillatorNode) (which traces
//! `PolyBLEP`-corrected sine/saw/square/triangle shapes directly), a wavetable
//! oscillator can reproduce *any* periodic timbre described by its harmonic
//! spectrum, so it is the workhorse voice for rich, spectrally designed tones
//! (the family of synthesis popularized by instruments such as Serum and
//! Vital), while remaining a plain table lookup on the audio thread.
//!
//! # The model
//!
//! A naive wavetable aliases badly: a table that contains, say, 1024 harmonics
//! played at a high fundamental folds every partial above Nyquist back into the
//! audible band. The classic cure is a *mipmap* of band-limited tables, one per
//! octave: for each table only the harmonics that stay below Nyquist are
//! synthesized, and the oscillator picks the richest table whose highest
//! harmonic still fits under Nyquist at the current pitch.
//!
//! Each table is built by **additive synthesis** from a caller-supplied array
//! of harmonic amplitudes (`harmonic_amplitudes[k - 1]` is the linear amplitude
//! of the `k`-th sine partial):
//!
//! ```text
//! table_m[i] = sum over k in 1..=m of  a[k] * sin(2*pi * k * i / TABLE_SIZE)
//! ```
//!
//! Table `m` (which contains harmonics `1..=m`) is only ever played when the
//! normalized fundamental `nf = f0 / sample_rate` satisfies `m * nf <= 0.5`, so
//! no partial it carries ever crosses Nyquist. The tables are indexed by that
//! ceiling frequency `top_freq = 0.5 / m` and the oscillator selects, once per
//! block, the first table whose `top_freq` is at least `nf`. Because the
//! amplitude of every shared harmonic is identical across tables, and because
//! all tables are scaled by **one global normalization factor** (derived from
//! the peak of the richest table), switching tables as the pitch sweeps never
//! produces a level jump.
//!
//! Reading the table uses a four-point Catmull-Rom interpolation with periodic
//! (wrap-around) indexing, which gives smoothly interpolated output without the
//! high-frequency roll-off of plain linear interpolation. The amplitude is
//! driven through a [`Smoothed`] value, so gain automation is click-free.
//!
//! # Relationship
//!
//! This is the spectral counterpart to the geometric
//! [`OscillatorNode`](super::oscillator::OscillatorNode): that node band-limits
//! a fixed set of analytic shapes with `PolyBLEP` step/slope corrections,
//! whereas this node band-limits an *arbitrary* harmonic spectrum with an
//! octave mipmap of additively synthesized tables. It shares no DSP math with
//! the geometric oscillator; it reuses only this crate's own [`Sample`] type,
//! [`Smoothed`] parameter smoother, and the same four-point Catmull-Rom kernel
//! used by [`SamplePlayerNode`](super::sample_player::SamplePlayerNode) (there
//! applied to recorded PCM, here to a synthetic single cycle). The convenience
//! constructors expose classic saw/square/triangle spectra so it can also serve
//! as a drop-in band-limited geometric oscillator.
//!
//! # Real-time contract
//!
//! The entire mipmap is built once in the constructors; nothing is allocated,
//! locked, or recomputed on the audio thread.
//! [`process`](crate::graph::AudioNode::process) selects a table by index,
//! reads it with a bounded interpolation, advances a wrapped phase accumulator,
//! and cannot panic: an empty or all-zero spectrum yields a single silent
//! table, the table index is always in range, and non-finite parameters are
//! rejected at the setters and the constructor.
//!
//! # Provenance
//!
//! The band-limited mipmap wavetable technique (one additively synthesized
//! table per octave, selected so the highest retained harmonic stays below
//! Nyquist) is the standard treatment described in Nigel Redmon's public
//! "earlevel engineering" wavetable articles and in the classic computer-music
//! literature on additive synthesis and the discrete Fourier series. The
//! Catmull-Rom interpolation kernel follows E. Catmull and R. Rom ("A class of
//! local interpolating splines", 1974). This module contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; it is implemented purely from that publicly
//! documented theory.

use alloc::{vec, vec::Vec};

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Two-pi, the full-cycle argument for the additive sine partials.
const TAU: Sample = core::f32::consts::TAU;

/// Lowest fundamental, as a whole number of hertz, the table set is
/// band-limited for. Used both as the [`Sample`] clamp bound and (as an
/// integer) to cap the additive harmonic count so the richest table never
/// aliases at that pitch.
const MIN_FREQUENCY_HZ_INT: u32 = 20;

/// Lowest tunable fundamental in hertz. Kept module-private so it does not
/// collide with the sibling source nodes' own frequency floors at re-export.
const MIN_FREQUENCY_HZ: Sample = MIN_FREQUENCY_HZ_INT as Sample;

/// Number of samples in every single-cycle table. A power of two keeps the
/// phase-to-index scaling exact and leaves ample room (Nyquist at
/// `TABLE_SIZE / 2` harmonics) for rich spectra.
pub const TABLE_SIZE: usize = 2048;

/// Amplitudes with magnitude at or below this are treated as absent partials,
/// and a peak at or below this is treated as a silent table (no normalization).
const AMPLITUDE_EPSILON: Sample = 1.0e-12;

/// Returns `value` when finite, otherwise `fallback`. Guards the public setters
/// and the constructor against `NaN`/infinity leaking into the oscillator state
/// (a `NaN` would otherwise survive `clamp`).
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Clamps `frequency_hz` to the playable range `[MIN_FREQUENCY_HZ,
/// sample_rate / 2]`, substituting [`MIN_FREQUENCY_HZ`] for non-finite input.
#[inline]
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

/// One band-limited single-cycle table in the octave mipmap.
#[derive(Debug, Clone)]
struct Mipmap {
    /// Highest normalized fundamental `f / sample_rate` at which this table is
    /// alias-free (`0.5 / highest_harmonic`).
    top_freq: Sample,
    /// `TABLE_SIZE` samples spanning exactly one period.
    data: Vec<Sample>,
}

/// Builds the saw spectrum `a[k] = 1 / k` for `count` harmonics (a descending
/// sawtooth: `sum sin(k x) / k`).
#[must_use]
fn saw_harmonics(count: usize) -> Vec<Sample> {
    let mut harmonics = vec![0.0; count];
    for (k, slot) in harmonics.iter_mut().enumerate() {
        let n = (k + 1) as Sample;
        *slot = 1.0 / n;
    }
    harmonics
}

/// Builds the square spectrum `a[k] = 1 / k` for odd `k`, zero otherwise.
#[must_use]
fn square_harmonics(count: usize) -> Vec<Sample> {
    let mut harmonics = vec![0.0; count];
    for (k, slot) in harmonics.iter_mut().enumerate() {
        let n = k + 1;
        if n % 2 == 1 {
            *slot = 1.0 / n as Sample;
        }
    }
    harmonics
}

/// Builds the triangle spectrum `a[k] = (-1)^((k-1)/2) / k^2` for odd `k`, zero
/// otherwise.
#[must_use]
fn triangle_harmonics(count: usize) -> Vec<Sample> {
    let mut harmonics = vec![0.0; count];
    for (k, slot) in harmonics.iter_mut().enumerate() {
        let n = k + 1;
        if n % 2 == 1 {
            let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };
            let nf = n as Sample;
            *slot = sign / (nf * nf);
        }
    }
    harmonics
}

/// Additively synthesizes one `TABLE_SIZE` table containing harmonics
/// `1..=count` from the amplitude array. Uses exact integer phase reduction
/// `(k * i) % TABLE_SIZE` so the partials stay perfectly periodic in `f32`.
#[must_use]
fn synthesize(harmonic_amplitudes: &[Sample], count: usize) -> Vec<Sample> {
    let n = TABLE_SIZE;
    let mut data = vec![0.0; n];
    let limit = count.min(harmonic_amplitudes.len());
    for k in 1..=limit {
        let a = harmonic_amplitudes[k - 1];
        if ops::abs(a) <= AMPLITUDE_EPSILON {
            continue;
        }
        for (i, slot) in data.iter_mut().enumerate() {
            let idx = (k * i) % n;
            *slot += a * ops::sin(TAU * idx as Sample / n as Sample);
        }
    }
    data
}

/// Builds the full octave mipmap for `harmonic_amplitudes` at `sample_rate`.
///
/// Returns a single silent table for an empty or all-zero spectrum. Otherwise
/// it caps the harmonic count so the richest table aliases neither at the table
/// Nyquist (`TABLE_SIZE / 2`) nor at [`MIN_FREQUENCY_HZ`], then builds one table
/// per octave (halving the harmonic count each step down to one). All tables
/// share a single normalization factor taken from the richest table, and are
/// returned sorted by ascending `top_freq`.
#[must_use]
fn build_tables(sample_rate: u32, harmonic_amplitudes: &[Sample]) -> Vec<Mipmap> {
    let n = TABLE_SIZE;

    // Highest harmonic index carrying audible energy.
    let mut max_h = 0usize;
    for (k, &a) in harmonic_amplitudes.iter().enumerate() {
        if ops::abs(a) > AMPLITUDE_EPSILON {
            max_h = k + 1;
        }
    }
    if max_h == 0 {
        return vec![Mipmap {
            top_freq: 0.5,
            data: vec![0.0; n],
        }];
    }

    // Nothing above the table Nyquist, and nothing that would alias at the
    // lowest supported pitch (`sample_rate / 2 / MIN_FREQUENCY_HZ` harmonics).
    let nyquist_harmonics = ((sample_rate / 2) / MIN_FREQUENCY_HZ_INT) as usize;
    let top_h = max_h.min(nyquist_harmonics.max(1)).min(n / 2);

    let mut tables: Vec<Mipmap> = Vec::new();
    let mut norm = 1.0;
    let mut harmonics_count = top_h;
    loop {
        let data = synthesize(harmonic_amplitudes, harmonics_count);
        if tables.is_empty() {
            // The first (richest) table sets the global normalization so every
            // table shares one scale and table switches are level-matched.
            let peak = data.iter().fold(0.0, |m: Sample, &x| m.max(ops::abs(x)));
            norm = if peak > AMPLITUDE_EPSILON {
                1.0 / peak
            } else {
                1.0
            };
        }
        // `harmonics_count` descends, so `top_freq = 0.5 / harmonics_count`
        // ascends: tables are pushed already sorted by ascending `top_freq`.
        let top_freq = 0.5 / harmonics_count as Sample;
        tables.push(Mipmap { top_freq, data });
        if harmonics_count == 1 {
            break;
        }
        harmonics_count = (harmonics_count / 2).max(1);
    }

    for table in &mut tables {
        for sample in &mut table.data {
            *sample *= norm;
        }
    }
    tables
}

/// Selects the index of the first table whose `top_freq` covers the normalized
/// fundamental `nf`, falling back to the most band-limited (last) table.
#[inline]
fn select(tables: &[Mipmap], nf: Sample) -> usize {
    for (i, table) in tables.iter().enumerate() {
        if table.top_freq >= nf {
            return i;
        }
    }
    tables.len().saturating_sub(1)
}

/// Reads `data` at normalized `phase` in `[0, 1)` with periodic four-point
/// Catmull-Rom interpolation. A free function (not a method) so the hot loop
/// can hold an immutable borrow of the selected table while mutably advancing
/// the node's phase and amplitude fields.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the normalized phase is in [0, 1) so pos is in [0, data.len()); truncating to usize is an exact floor that never wraps"
)]
fn sample_table(data: &[Sample], phase: Sample) -> Sample {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let pos = phase * n as Sample;
    let i1 = (pos as usize).min(n - 1);
    let t = pos - i1 as Sample;
    let i0 = if i1 == 0 { n - 1 } else { i1 - 1 };
    let i2 = if i1 + 1 >= n { 0 } else { i1 + 1 };
    let i3 = if i2 + 1 >= n { 0 } else { i2 + 1 };
    let p0 = data[i0];
    let p1 = data[i1];
    let p2 = data[i2];
    let p3 = data[i3];
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Construction parameters for a [`WavetableOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WavetableOscillatorParams {
    /// Fundamental frequency in hertz, clamped to `[MIN_FREQUENCY_HZ,
    /// sample_rate / 2]`.
    pub frequency_hz: Sample,
    /// Linear output amplitude (smoothed to avoid zipper noise).
    pub amplitude: Sample,
}

impl Default for WavetableOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: 220.0,
            amplitude: 1.0,
        }
    }
}

/// Band-limited mipmap wavetable oscillator (0 inputs, 1 output).
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::WavetableOscillatorNode;
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = WavetableOscillatorNode::saw(48_000, 220.0, 1.0);
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
/// outputs[0].set_active_frames(256);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A plucked saw is not silent and stays bounded.
/// let peak = outputs[0]
///     .channel(0)
///     .iter()
///     .fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak <= 1.2);
/// ```
#[derive(Debug, Clone)]
pub struct WavetableOscillatorNode {
    /// Sample rate the tables were band-limited against (set at construction).
    sample_rate: Sample,
    /// Octave mipmap, sorted by ascending `top_freq`; never empty.
    tables: Vec<Mipmap>,
    /// Phase accumulator in `[0, 1)`.
    phase: Sample,
    /// User-facing fundamental in hertz.
    frequency_hz: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
}

impl WavetableOscillatorNode {
    /// Builds a wavetable oscillator for `sample_rate` Hz from an explicit
    /// harmonic spectrum (`harmonic_amplitudes[k - 1]` is the linear amplitude
    /// of the `k`-th sine partial).
    ///
    /// All parameters are sanitized: non-finite values fall back to safe
    /// defaults and `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ,
    /// sample_rate / 2]`. An empty or all-zero spectrum yields a silent (but
    /// still valid) oscillator.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        harmonic_amplitudes: &[Sample],
        frequency_hz: Sample,
        amplitude: Sample,
    ) -> Self {
        let sr_u32 = sample_rate.max(1);
        let sr = sr_u32 as Sample;
        let tables = build_tables(sr_u32, harmonic_amplitudes);
        let frequency_hz = sanitize_frequency(frequency_hz, sr);
        let amplitude = finite_or(amplitude, 1.0);
        Self {
            sample_rate: sr,
            tables,
            phase: 0.0,
            frequency_hz,
            amplitude: Smoothed::new(amplitude),
        }
    }

    /// Builds a band-limited sawtooth (`a[k] = 1 / k`).
    #[must_use]
    pub fn saw(sample_rate: u32, frequency_hz: Sample, amplitude: Sample) -> Self {
        Self::new(
            sample_rate,
            &saw_harmonics(TABLE_SIZE / 2),
            frequency_hz,
            amplitude,
        )
    }

    /// Builds a band-limited square (`a[k] = 1 / k` for odd `k`).
    #[must_use]
    pub fn square(sample_rate: u32, frequency_hz: Sample, amplitude: Sample) -> Self {
        Self::new(
            sample_rate,
            &square_harmonics(TABLE_SIZE / 2),
            frequency_hz,
            amplitude,
        )
    }

    /// Builds a band-limited triangle (`a[k] = (-1)^((k-1)/2) / k^2` for odd
    /// `k`).
    #[must_use]
    pub fn triangle(sample_rate: u32, frequency_hz: Sample, amplitude: Sample) -> Self {
        Self::new(
            sample_rate,
            &triangle_harmonics(TABLE_SIZE / 2),
            frequency_hz,
            amplitude,
        )
    }

    /// Builds a wavetable oscillator from a [`WavetableOscillatorParams`] bundle
    /// and explicit spectrum.
    #[must_use]
    pub fn from_params(
        sample_rate: u32,
        harmonic_amplitudes: &[Sample],
        params: WavetableOscillatorParams,
    ) -> Self {
        Self::new(
            sample_rate,
            harmonic_amplitudes,
            params.frequency_hz,
            params.amplitude,
        )
    }

    /// Sets the fundamental in hertz (clamped; non-finite input is ignored).
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            sanitize_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
    }

    /// Sets the target output amplitude, ramped over `ramp_samples` frames to
    /// avoid zipper noise. Non-finite input is ignored.
    pub fn set_amplitude(&mut self, amplitude: Sample, ramp_samples: u32) {
        if !amplitude.is_finite() {
            return;
        }
        let ramp = if ramp_samples == 0 {
            Ramp::Immediate
        } else {
            Ramp::Linear {
                samples: ramp_samples,
            }
        };
        self.amplitude.set_target(amplitude, ramp);
    }

    /// Returns the current fundamental in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the current (possibly mid-ramp) output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.current()
    }

    /// Returns the current phase accumulator in `[0, 1)`.
    #[must_use]
    pub fn phase(&self) -> Sample {
        self.phase
    }

    /// Returns the number of band-limited tables in the octave mipmap.
    #[must_use]
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

impl AudioNode for WavetableOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();
        if channels == 0 {
            return;
        }

        // Per-sample phase increment and table selection for this block.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let nf = self.frequency_hz / sample_rate;
        let inc = nf;
        let table_idx = select(&self.tables, nf);

        // Field-level disjoint borrows: `data` borrows `self.tables`, while the
        // loop mutates `self.phase` and `self.amplitude`.
        {
            let data = &self.tables[table_idx].data;
            let buf = out.channel_mut(0);
            for sample in buf.iter_mut() {
                let amp = self.amplitude.next_sample();
                let value = sample_table(data, self.phase);
                *sample = flush_denormal(value * amp);
                self.phase += inc;
                if self.phase >= 1.0 {
                    self.phase -= 1.0;
                }
            }
        }

        // Replicate the mono voice into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = out.channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
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

    /// Renders `frames` of a mono voice into a flat vector.
    fn render(node: &mut WavetableOscillatorNode, frames: usize) -> Vec<Sample> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        outputs[0].channel(0).to_vec()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(ops::abs(s)))
    }

    /// Zero-mean normalized cross-correlation of two equal-length signals.
    fn normalized_xcorr(a: &[Sample], b: &[Sample]) -> f64 {
        let n = a.len().min(b.len());
        let ma: f64 = a[..n].iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let mb: f64 = b[..n].iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let mut num = 0.0f64;
        let mut da = 0.0f64;
        let mut db = 0.0f64;
        for i in 0..n {
            let x = a[i] as f64 - ma;
            let y = b[i] as f64 - mb;
            num += x * y;
            da += x * x;
            db += y * y;
        }
        if da <= 0.0 || db <= 0.0 {
            return 0.0;
        }
        num / (da.sqrt() * db.sqrt())
    }

    #[test]
    fn saw_autocorrelation_matches_period() {
        // 100 Hz at 48 kHz -> period of 480 samples.
        let mut node = WavetableOscillatorNode::saw(SR, 100.0, 1.0);
        let signal = render(&mut node, 4096);
        let win = 2048;
        let mut best_lag = 0usize;
        let mut best = f64::NEG_INFINITY;
        for lag in 200..=800 {
            let mut acc = 0.0f64;
            for i in 0..win {
                acc += (signal[i] as f64) * (signal[i + lag] as f64);
            }
            if acc > best {
                best = acc;
                best_lag = lag;
            }
        }
        assert!(
            (best_lag as i64 - 480).unsigned_abs() <= 3,
            "autocorrelation period {best_lag} should be near 480"
        );
    }

    #[test]
    fn high_frequency_is_band_limited_to_a_sine() {
        // 13 kHz at 48 kHz: only the fundamental fits below Nyquist, so the
        // selected table carries a single partial and the output is a pure sine.
        let f0 = 13_000.0;
        let mut node = WavetableOscillatorNode::saw(SR, f0, 1.0);
        let signal = render(&mut node, 2048);
        let nf = f0 / SR as Sample;
        let reference: Vec<Sample> = (0..signal.len())
            .map(|n| ops::sin(TAU * n as Sample * nf))
            .collect();
        let corr = normalized_xcorr(&signal, &reference);
        assert!(
            corr > 0.995,
            "band-limited 13 kHz output should match a pure sine (corr = {corr})"
        );
    }

    #[test]
    fn amplitude_scales_output() {
        let mut loud = WavetableOscillatorNode::saw(SR, 220.0, 1.0);
        let mut quiet = WavetableOscillatorNode::saw(SR, 220.0, 0.25);
        let a = render(&mut loud, 1024);
        let b = render(&mut quiet, 1024);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x * 0.25 - y).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn deterministic_bit_for_bit() {
        let mut a = WavetableOscillatorNode::triangle(SR, 330.0, 0.8);
        let mut b = WavetableOscillatorNode::triangle(SR, 330.0, 0.8);
        let sa = render(&mut a, 2048);
        let sb = render(&mut b, 2048);
        assert_eq!(sa, sb);
    }

    #[test]
    fn empty_spectrum_is_silent() {
        let mut node = WavetableOscillatorNode::new(SR, &[], 220.0, 1.0);
        let block = render(&mut node, 512);
        assert!(block.iter().all(|&s| s == 0.0));
        assert_eq!(node.table_count(), 1);
    }

    #[test]
    fn all_zero_spectrum_is_silent() {
        let mut node = WavetableOscillatorNode::new(SR, &[0.0, 0.0, 0.0], 220.0, 1.0);
        let block = render(&mut node, 512);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn stereo_channels_are_identical() {
        let mut node = WavetableOscillatorNode::square(SR, 220.0, 0.9);
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 512);
        out.set_active_frames(512);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 512,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let left = outputs[0].channel(0).to_vec();
        let right = outputs[0].channel(1).to_vec();
        assert_eq!(left, right);
    }

    #[test]
    fn non_finite_frequency_is_clamped() {
        let mut node = WavetableOscillatorNode::saw(SR, Sample::NAN, 1.0);
        assert!(node.frequency_hz().is_finite());
        assert!(node.frequency_hz() >= MIN_FREQUENCY_HZ);
        node.set_frequency_hz(Sample::INFINITY);
        assert!(node.frequency_hz().is_finite());
        // Ignored, so the previous valid value is retained.
        node.set_frequency_hz(440.0);
        node.set_frequency_hz(Sample::NAN);
        assert!((node.frequency_hz() - 440.0).abs() <= 1.0e-3);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = WavetableOscillatorNode::saw(SR, 220.0, 1.0);
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn set_frequency_changes_pitch() {
        let mut low = WavetableOscillatorNode::saw(SR, 110.0, 1.0);
        let mut high = WavetableOscillatorNode::saw(SR, 110.0, 1.0);
        high.set_frequency_hz(440.0);
        assert!(high.frequency_hz() > low.frequency_hz());

        // The higher pitch advances phase four times as fast, so after the same
        // number of frames it has wrapped more cycles.
        let _ = render(&mut low, 480);
        let _ = render(&mut high, 480);
        // 110 Hz -> ~1.1 cycles in 480 frames; 440 Hz -> ~4.4 cycles.
        // Both wrap into [0, 1); the test only asserts frequency ordering above,
        // and that rendering at the new pitch stays bounded.
        let hi_block = render(&mut high, 1024);
        assert!(peak(&hi_block) <= 1.2);
    }

    #[test]
    fn output_is_bounded() {
        for &shape in &[0u8, 1, 2] {
            let mut node = match shape {
                0 => WavetableOscillatorNode::saw(SR, 220.0, 1.0),
                1 => WavetableOscillatorNode::square(SR, 220.0, 1.0),
                _ => WavetableOscillatorNode::triangle(SR, 220.0, 1.0),
            };
            let block = render(&mut node, 4096);
            assert!(
                peak(&block) <= 1.1,
                "shape {shape} should stay near unity after normalization"
            );
        }
    }

    #[test]
    fn mipmap_has_multiple_octave_tables() {
        let node = WavetableOscillatorNode::saw(SR, 220.0, 1.0);
        // 1024 harmonics halving to 1 -> 11 tables (1024,512,...,2,1).
        assert_eq!(node.table_count(), 11);
    }

    #[test]
    fn table_selection_is_monotonic_in_frequency() {
        let tables = build_tables(SR, &saw_harmonics(TABLE_SIZE / 2));
        // Ascending top_freq ordering.
        for pair in tables.windows(2) {
            assert!(pair[0].top_freq < pair[1].top_freq);
        }
        // Higher normalized frequency never selects a richer (lower-index) table.
        let low = select(&tables, 0.0001);
        let mid = select(&tables, 0.05);
        let high = select(&tables, 0.4);
        assert!(low <= mid && mid <= high);
        // The richest table (index 0, finest top_freq) serves the lowest pitches.
        assert_eq!(low, 0);
        // The most band-limited table (last index) serves pitches above every
        // table's ceiling.
        assert_eq!(select(&tables, 1.0), tables.len() - 1);
    }

    #[test]
    fn default_params_round_trip() {
        let params = WavetableOscillatorParams::default();
        assert!((params.frequency_hz - 220.0).abs() <= 1.0e-6);
        assert!((params.amplitude - 1.0).abs() <= 1.0e-6);
        let node = WavetableOscillatorNode::from_params(SR, &saw_harmonics(64), params);
        assert!((node.frequency_hz() - 220.0).abs() <= 1.0e-3);
    }

    #[test]
    fn reset_clears_phase() {
        let mut node = WavetableOscillatorNode::saw(SR, 220.0, 1.0);
        let _ = render(&mut node, 123);
        assert!(node.phase() > 0.0);
        node.reset();
        assert_eq!(node.phase(), 0.0);
    }
}
