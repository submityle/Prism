//! Reusable polyphase oversampler for anti-aliased non-linear processing.
//!
//! Any instantaneous non-linearity -- a soft clipper, a wave folder, a
//! rectifier -- creates harmonics above its input frequency. Once those
//! harmonics pass the Nyquist limit they *fold back* into the audible band as
//! inharmonic aliasing. The classic cure is to run the non-linearity at an
//! integer multiple of the host sample rate so the new harmonics land further
//! above Nyquist, then low-pass filter them away before decimating back down.
//!
//! This module factors that machinery out of any single effect into one shared
//! primitive so every non-linear node reuses the identical, well-tested filter
//! design instead of copying it. The pipeline for a single input sample is:
//!
//! 1. **Upsample** by inserting `factor - 1` zeros between input samples and
//!    reconstructing with a polyphase windowed-sinc interpolation filter, which
//!    yields `factor` oversampled values per host-rate input.
//! 2. **Shape** each oversampled value with a caller-supplied closure (the
//!    non-linearity).
//! 3. **Decimate** back to the host rate through the same half-band-style
//!    low-pass so the freshly created out-of-band harmonics are removed before
//!    they can alias.
//!
//! The up/down filter pair is linear phase, so it introduces a fixed group
//! delay reported by [`Oversampler::latency_frames`]; a node that blends a dry
//! path should delay it by the same amount with a [`DryDelay`] to stay
//! phase-coherent.
//!
//! # Real-time contract
//!
//! The filter coefficients live once inside the [`Oversampler`]; the per-channel
//! histories live in [`OversamplerState`] values the caller allocates up front
//! with [`Oversampler::make_state`]. [`Oversampler::process_sample`] and
//! [`DryDelay::push`] perform no allocation, take no locks, and cannot panic;
//! the output is denormal-flushed so a decaying tail cannot stall the CPU on
//! subnormals. All transcendental math routes through [`bevy_math::ops`], so
//! the designed filter is bit-reproducible across platforms.
//!
//! # Provenance
//!
//! Polyphase interpolation/decimation and the windowed-sinc (Hann) low-pass are
//! textbook multirate-signal-processing constructions described in every
//! digital-signal-processing reference; oversampling a non-linearity to control
//! aliasing is the standard anti-aliased-waveshaping technique. This module
//! reuses only this crate's own [`Sample`] scalar. It is pure classic DSP with
//! no AI or ML and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from those publicly documented algorithms.
//!
//! # Relationship
//!
//! This primitive is consumed by the non-linear effect nodes in
//! [`nodes::effects`](crate::nodes::effects) -- for example the
//! [`waveshaper`](crate::nodes::effects::waveshaper) soft clipper and the
//! [`wavefolder`](crate::nodes::effects::wavefolder) -- so they share one
//! anti-aliasing implementation rather than each carrying a private copy. It
//! shares no code with the frequency-domain
//! [`SpectrumAnalyzer`](crate::nodes::analysis::spectrum::SpectrumAnalyzer);
//! that transforms a signal, whereas this resamples one.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::PI;

use crate::math::{Sample, flush_denormal};

/// Default number of filter taps per polyphase branch.
///
/// The full prototype length is `TAPS_PER_PHASE * factor`, trading a little
/// latency and CPU for strong stop-band rejection.
pub const DEFAULT_TAPS_PER_PHASE: usize = 16;

/// A linear-phase polyphase oversampler shared by non-linear effect nodes.
///
/// Construct one per node, then allocate one [`OversamplerState`] per channel
/// with [`make_state`](Self::make_state). A `factor` of 1 is a valid identity
/// pipeline (no filtering, zero latency) so callers can treat the "no
/// oversampling" case uniformly.
#[derive(Debug, Clone)]
pub struct Oversampler {
    factor: usize,
    taps_per_phase: usize,
    /// Decimation prototype filter (unity DC gain). Empty when `factor == 1`.
    proto: Vec<Sample>,
    /// Polyphase interpolation branches derived from [`Oversampler::proto`].
    /// Empty when `factor == 1`.
    phases: Vec<Vec<Sample>>,
    /// Reported processing latency in host-rate frames.
    latency: u32,
}

impl Oversampler {
    /// Designs an oversampler for the given integer `factor` (clamped to `>= 1`)
    /// and taps-per-phase (clamped to `>= 1`).
    #[must_use]
    pub fn new(factor: usize, taps_per_phase: usize) -> Self {
        let factor = factor.max(1);
        let taps_per_phase = taps_per_phase.max(1);

        if factor == 1 {
            return Self {
                factor,
                taps_per_phase,
                proto: Vec::new(),
                phases: Vec::new(),
                latency: 0,
            };
        }

        let taps = taps_per_phase * factor;
        let proto = design_lowpass(taps, factor);
        let mut phases = Vec::with_capacity(factor);
        for p in 0..factor {
            let mut phase = Vec::with_capacity(taps_per_phase);
            let mut k = 0usize;
            while p + k * factor < taps {
                // Scale by `factor` to compensate for the energy lost to
                // zero-stuffing during interpolation.
                phase.push(factor as Sample * proto[p + k * factor]);
                k += 1;
            }
            phases.push(phase);
        }

        // Linear-phase group delay of the up/down filter pair is `taps - 1`
        // samples at the oversampled rate, i.e. `(taps - 1) / factor` frames.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded non-negative latency fits comfortably in u32"
        )]
        let latency = ops::round((taps as Sample - 1.0) / factor as Sample) as u32;

        Self {
            factor,
            taps_per_phase,
            proto,
            phases,
            latency,
        }
    }

    /// Convenience constructor using [`DEFAULT_TAPS_PER_PHASE`].
    #[must_use]
    pub fn with_default_taps(factor: usize) -> Self {
        Self::new(factor, DEFAULT_TAPS_PER_PHASE)
    }

    /// Returns the integer oversampling factor (`>= 1`).
    #[inline]
    #[must_use]
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// Returns the length of the prototype FIR filter (`0` when `factor == 1`).
    #[inline]
    #[must_use]
    pub fn filter_taps(&self) -> usize {
        self.proto.len()
    }

    /// Returns the reported processing latency in host-rate frames.
    #[inline]
    #[must_use]
    pub fn latency_frames(&self) -> u32 {
        self.latency
    }

    /// Allocates a zeroed per-channel history for this oversampler.
    #[must_use]
    pub fn make_state(&self) -> OversamplerState {
        let (up_len, down_len) = if self.factor > 1 {
            (self.taps_per_phase, self.proto.len())
        } else {
            (0, 0)
        };
        OversamplerState {
            up_hist: zeroed(up_len),
            up_pos: 0,
            down_hist: zeroed(down_len),
            down_pos: 0,
        }
    }

    /// Shapes one host-rate input sample through the oversampled, anti-aliased
    /// path and returns the resulting host-rate sample.
    ///
    /// `shape` is the non-linearity applied to every oversampled value. When
    /// `factor == 1` the input is shaped directly with no filtering (and
    /// `state` is untouched), so callers need no special-case branch.
    #[inline]
    pub fn process_sample<F>(&self, state: &mut OversamplerState, x: Sample, mut shape: F) -> Sample
    where
        F: FnMut(Sample) -> Sample,
    {
        if self.factor == 1 {
            return flush_denormal(shape(x));
        }

        // Advance the interpolation input history.
        let up_len = state.up_hist.len();
        state.up_pos = (state.up_pos + 1) % up_len;
        state.up_hist[state.up_pos] = x;

        let down_len = state.down_hist.len();
        // Synthesise `factor` oversampled samples, shape each, and push them
        // into the decimation history in temporal order.
        for phase in &self.phases {
            let mut acc = 0.0;
            for (j, &coeff) in phase.iter().enumerate() {
                let idx = (state.up_pos + up_len - j) % up_len;
                acc += coeff * state.up_hist[idx];
            }
            let shaped = shape(acc);
            state.down_pos = (state.down_pos + 1) % down_len;
            state.down_hist[state.down_pos] = shaped;
        }

        // Decimate: one host-rate output per `factor` oversampled samples.
        let mut out = 0.0;
        for (j, &coeff) in self.proto.iter().enumerate() {
            let idx = (state.down_pos + down_len - j) % down_len;
            out += coeff * state.down_hist[idx];
        }
        flush_denormal(out)
    }
}

/// Per-channel filter memory for an [`Oversampler`].
///
/// Allocate one with [`Oversampler::make_state`]; it is empty (and cheap) when
/// the owning oversampler has `factor == 1`.
#[derive(Debug, Clone)]
pub struct OversamplerState {
    /// Ring buffer of recent inputs feeding the polyphase interpolation filter.
    up_hist: Vec<Sample>,
    up_pos: usize,
    /// Ring buffer of recent shaped oversampled values feeding decimation.
    down_hist: Vec<Sample>,
    down_pos: usize,
}

impl OversamplerState {
    /// Clears every history back to silence.
    pub fn reset(&mut self) {
        for s in &mut self.up_hist {
            *s = 0.0;
        }
        for s in &mut self.down_hist {
            *s = 0.0;
        }
        self.up_pos = 0;
        self.down_pos = 0;
    }
}

/// A fixed-length delay line that aligns a dry path with oversampler latency.
///
/// Push one sample per host-rate frame; the value returned is the input delayed
/// by the configured number of frames. A length of zero is the identity.
#[derive(Debug, Clone)]
pub struct DryDelay {
    line: Vec<Sample>,
    pos: usize,
}

impl DryDelay {
    /// Allocates a zeroed dry-delay line of `frames` samples.
    #[must_use]
    pub fn new(frames: usize) -> Self {
        Self {
            line: zeroed(frames),
            pos: 0,
        }
    }

    /// Pushes `x` and returns the value delayed by the line length.
    #[inline]
    pub fn push(&mut self, x: Sample) -> Sample {
        let len = self.line.len();
        if len == 0 {
            return x;
        }
        let out = self.line[self.pos];
        self.line[self.pos] = x;
        self.pos = (self.pos + 1) % len;
        out
    }

    /// Clears the delay line back to silence.
    pub fn reset(&mut self) {
        for s in &mut self.line {
            *s = 0.0;
        }
        self.pos = 0;
    }

    /// Returns the delay length in frames.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.line.len()
    }

    /// Returns `true` when the delay length is zero (the identity).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.line.is_empty()
    }
}

/// Allocates a zero-filled sample vector of length `n`.
fn zeroed(n: usize) -> Vec<Sample> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

/// Designs a `taps`-long, unity-DC-gain, Hann-windowed sinc low-pass prototype.
///
/// The cutoff is placed at the host-rate Nyquist, i.e. a normalised cutoff of
/// `1/(2*factor)` cycles/sample at the oversampled rate, so the filter both
/// reconstructs interpolated samples and rejects the images/aliases created by
/// the non-linearity.
fn design_lowpass(taps: usize, factor: usize) -> Vec<Sample> {
    let center = (taps - 1) as Sample / 2.0;
    // `cutoff` is twice the cutoff frequency (the sinc argument scale):
    // 2 * (1 / (2 * factor)) = 1 / factor.
    let cutoff = 1.0 / factor as Sample;
    let denom = (taps - 1) as Sample;

    let mut h = Vec::with_capacity(taps);
    let mut sum = 0.0;
    for n in 0..taps {
        let t = n as Sample - center;
        let ideal = if t == 0.0 {
            cutoff
        } else {
            let arg = PI * cutoff * t;
            cutoff * (ops::sin(arg) / arg)
        };
        // Hann window keeps the stop-band clean with a gentle main-lobe trade.
        let window = 0.5 - 0.5 * ops::cos(2.0 * PI * n as Sample / denom);
        let coeff = ideal * window;
        h.push(coeff);
        sum += coeff;
    }

    // Normalise to unity DC gain so the decimation stage preserves level.
    let inv = if sum != 0.0 { 1.0 / sum } else { 1.0 };
    for c in &mut h {
        *c *= inv;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure identity shaper, so `process_sample` only exercises the
    /// resampling filters, not any non-linearity.
    fn identity(v: Sample) -> Sample {
        v
    }

    #[test]
    fn identity_factor_is_zero_latency_passthrough() {
        let os = Oversampler::new(1, DEFAULT_TAPS_PER_PHASE);
        assert_eq!(os.factor(), 1);
        assert_eq!(os.latency_frames(), 0);
        assert_eq!(os.filter_taps(), 0);

        let mut state = os.make_state();
        // factor == 1 must pass the shaped input through bit-exactly (identity
        // shaper here) with no state dependence.
        for i in 0..32 {
            let x = (i as Sample) * 0.1 - 1.5;
            assert_eq!(os.process_sample(&mut state, x, identity), x);
        }
    }

    #[test]
    fn clamps_degenerate_arguments() {
        let os = Oversampler::new(0, 0);
        assert_eq!(os.factor(), 1);
        assert_eq!(os.filter_taps(), 0);
    }

    #[test]
    fn filter_length_matches_taps_times_factor() {
        for &factor in &[2usize, 4, 8] {
            let taps_per_phase = 12;
            let os = Oversampler::new(factor, taps_per_phase);
            assert_eq!(os.factor(), factor);
            assert_eq!(os.filter_taps(), taps_per_phase * factor);
        }
    }

    #[test]
    fn latency_is_non_negative_and_grows_with_filter() {
        let o1 = Oversampler::with_default_taps(1);
        let o2 = Oversampler::with_default_taps(2);
        let o4 = Oversampler::with_default_taps(4);
        assert_eq!(o1.latency_frames(), 0);
        assert!(o2.latency_frames() >= 1);
        assert!(o4.latency_frames() >= 1);
    }

    #[test]
    fn dc_input_survives_oversampling() {
        // A constant input is pure DC: the unity-DC-gain filters must reproduce
        // it almost exactly once the pipeline has filled past its latency.
        let os = Oversampler::with_default_taps(4);
        let mut state = os.make_state();
        let dc = 0.37;
        let mut last = 0.0;
        for _ in 0..256 {
            last = os.process_sample(&mut state, dc, identity);
        }
        assert!((last - dc).abs() < 1.0e-3, "DC not preserved: {last} vs {dc}");
    }

    #[test]
    fn low_frequency_tone_passes_with_preserved_level() {
        // A tone well below Nyquist lives deep inside the filter pass-band, so
        // its energy must survive the resampling almost intact. Comparing RMS
        // in the steady-state region sidesteps the fractional group delay (the
        // reported latency is rounded to an integer frame).
        let os = Oversampler::with_default_taps(2);
        let mut state = os.make_state();
        let sr = 48_000.0;
        let f = 500.0;
        let n = 2_048usize;
        let mut in_energy = 0.0;
        let mut out_energy = 0.0;
        for i in 0..n {
            let t = i as Sample / sr;
            let x = 0.5 * ops::sin(2.0 * PI * f * t);
            let y = os.process_sample(&mut state, x, identity);
            // Skip the transient while the filter history fills.
            if i >= n / 2 {
                in_energy += x * x;
                out_energy += y * y;
            }
        }
        let ratio = out_energy / in_energy;
        assert!(
            (ratio - 1.0).abs() < 2.0e-2,
            "pass-band level not preserved: ratio={ratio}"
        );
    }

    #[test]
    fn reset_restores_fresh_behaviour() {
        let os = Oversampler::with_default_taps(4);
        let mut state = os.make_state();
        for i in 0..64 {
            let x = 0.6 * ops::sin(0.3 * i as Sample);
            let _ = os.process_sample(&mut state, x, identity);
        }
        state.reset();
        // After reset, silence in must yield exact silence out.
        for _ in 0..64 {
            assert_eq!(os.process_sample(&mut state, 0.0, identity), 0.0);
        }
    }

    #[test]
    fn dry_delay_delays_by_length() {
        let mut d = DryDelay::new(3);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
        // The first `len` outputs are the zero-fill, then the inputs appear.
        assert_eq!(d.push(1.0), 0.0);
        assert_eq!(d.push(2.0), 0.0);
        assert_eq!(d.push(3.0), 0.0);
        assert_eq!(d.push(4.0), 1.0);
        assert_eq!(d.push(5.0), 2.0);
    }

    #[test]
    fn dry_delay_zero_length_is_identity() {
        let mut d = DryDelay::new(0);
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        for i in 0..8 {
            let x = i as Sample * 0.25;
            assert_eq!(d.push(x), x);
        }
    }

    #[test]
    fn dry_delay_reset_clears() {
        let mut d = DryDelay::new(2);
        let _ = d.push(1.0);
        let _ = d.push(2.0);
        d.reset();
        assert_eq!(d.push(9.0), 0.0);
        assert_eq!(d.push(8.0), 0.0);
    }
}
