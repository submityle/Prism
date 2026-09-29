//! Signal / `Audio` spectrum sampler — the `CPU` reference for design §8.3's
//! `DataInterface` (design §8).
//!
//! Production `VFX` stacks let audio *drive* effects: Unreal `Niagara`'s audio
//! spectrum/oscilloscope data-interfaces, Unity `VFX Graph`'s audio-reactive
//! samples, and `Houdini`'s `CHOP`-driven parameters all expose "given the
//! current sound, hand me a few scalars I can wire into emission rate, force
//! strength, or colour". This module owns the `CPU`-verifiable *maths* of that
//! contract: it consumes an already-computed magnitude/energy spectrum and maps
//! it into signals that can drive emission and forces — band aggregation,
//! attack/release envelope following, spectral-flux onset detection, and linear
//! parameter remapping.
//!
//! # The `FFT` is upstream — this file does no `FFT`
//!
//! A Fast Fourier Transform (`FFT`) is intrinsically built from trigonometric
//! twiddle factors (`sin`/`cos`), which are transcendental and therefore out of
//! range for this determinism-locked layer. The `FFT` (and any windowing) is an
//! *upstream* `DSP` concern: by the time a [`Spectrum`] reaches this module the
//! per-bin magnitudes/energies have already been computed elsewhere. Everything
//! here is pure linear arithmetic (add / subtract / multiply / scalar divide
//! guarded by [`EPS`], plus `floor`/`abs` for bin location). No `sin`, `cos`,
//! `exp`, `ln`, or `pow` is called, and the analysis deliberately stays in the
//! *energy / mean-square* domain so it never needs `sqrt` either. The `CPU`
//! reference therefore stays bit-reproducible against a future `GPU` sampler.
//!
//! # Orthogonality
//!
//! This module is deliberately orthogonal to its two determinism-layer
//! neighbours:
//!
//! * [`super::curves`] owns *over-life* shaping: a 1-D look-up table (`LUT`)
//!   sampled by a particle's normalized age. It answers "how does this
//!   parameter change as the particle ages", independent of any live input.
//! * [`super::determinism`] owns the global *random stream* and *time*: a
//!   stateless hash `RNG` keyed by stable integers and the fixed `dt` clock.
//!   It answers "what pseudo-random value belongs to this draw".
//!
//! This module is the third, independent axis: given a live audio spectrum, it
//! produces *deterministic external-signal* scalars. A higher layer composes
//! the three (an audio band can pick a `LUT`, seed a random stream, or scale a
//! force); none of them depends on another's internals.

use super::Vec3;
use alloc::vec::Vec;

/// Absolute tolerance for `f32` equality/degeneracy decisions.
///
/// Denominators (bin widths, envelope time constants, remap spans, reference
/// energies) at or below this magnitude are treated as zero so no routine ever
/// divides by (near) zero or propagates `NaN`. Comparisons against this
/// tolerance replace bare `==`/`!=` on floating-point values throughout.
pub const EPS: f32 = 1e-9;

/// Default low/mid crossover in `Hz` used by [`Spectrum::low_mid_high`].
///
/// Below this frequency sits the "low" band (kick/bass energy); it is a
/// conventional three-band split, exposed as a constant so call sites and tests
/// can reference the exact boundary rather than a magic number.
pub const LOW_MID_CROSSOVER_HZ: f32 = 250.0;

/// Default mid/high crossover in `Hz` used by [`Spectrum::low_mid_high`].
///
/// Above this frequency sits the "high" band (presence/air); between it and
/// [`LOW_MID_CROSSOVER_HZ`] is the "mid" band (vocals/instruments).
pub const MID_HIGH_CROSSOVER_HZ: f32 = 4000.0;

/// Clamps `x` into the inclusive range `[lo, hi]`.
///
/// A local helper rather than `f32::clamp` so the `lo <= hi` precondition is
/// handled explicitly and the intent is visible at every call site; `f32::clamp`
/// panics when `lo > hi`, which this layer must never do on live input.
#[must_use]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// A borrowed view over one already-computed audio magnitude/energy spectrum.
///
/// `bins` holds one non-negative amplitude (or energy) per frequency bin, as
/// produced by an upstream `FFT` (this module performs no `FFT` itself). The
/// `sample_rate_hz` and `fft_size` fields let the view translate frequencies in
/// `Hz` to bin indices. The struct is `Copy` because it only borrows the bin
/// slice; it holds no owned buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectrum<'a> {
    /// Per-bin magnitude or energy, non-negative, `bins[0]` is the `DC` bin.
    pub bins: &'a [f32],
    /// The audio sample rate in `Hz` the spectrum was computed at.
    pub sample_rate_hz: f32,
    /// The `FFT` window size (number of time-domain samples transformed).
    pub fft_size: u32,
}

impl<'a> Spectrum<'a> {
    /// Wraps an existing magnitude/energy slice with its `FFT` parameters.
    #[must_use]
    pub fn new(bins: &'a [f32], sample_rate_hz: f32, fft_size: u32) -> Self {
        Self {
            bins,
            sample_rate_hz,
            fft_size,
        }
    }

    /// Frequency span of a single bin in `Hz`, `sample_rate / fft_size`.
    ///
    /// Returns `0.0` when `fft_size` is zero so a malformed view never divides
    /// by zero (scalar divide guarded by [`EPS`]).
    #[must_use]
    pub fn bin_width_hz(self) -> f32 {
        let denom = self.fft_size as f32;
        if denom > EPS {
            self.sample_rate_hz / denom
        } else {
            0.0
        }
    }

    /// Highest representable frequency in `Hz` (the Nyquist limit, half the
    /// sample rate). Used as the upper edge of the top band.
    #[must_use]
    pub fn nyquist_hz(self) -> f32 {
        self.sample_rate_hz * 0.5
    }

    /// Number of frequency bins in the view.
    #[must_use]
    pub fn bin_count(self) -> usize {
        self.bins.len()
    }

    /// Maps a frequency in `Hz` to a bin index, locating the bin with `floor`
    /// and clamping into `[0, bin_count - 1]`.
    ///
    /// Negative or sub-bin frequencies map to bin `0`; frequencies at or above
    /// Nyquist clamp to the last bin. An empty or zero-width spectrum yields `0`.
    #[must_use]
    pub fn hz_to_bin(self, hz: f32) -> usize {
        let count = self.bins.len();
        if count == 0 {
            return 0;
        }
        let width = self.bin_width_hz();
        if width <= EPS {
            return 0;
        }
        let located = (hz / width).floor();
        if located <= 0.0 {
            0
        } else {
            // `as usize` saturates a huge float to `usize::MAX`, which the
            // upper clamp below then folds onto the last valid bin.
            let idx = located as usize;
            if idx >= count {
                count - 1
            } else {
                idx
            }
        }
    }

    /// Energy (magnitude) of bin `i`, or `0.0` when `i` is out of range.
    #[must_use]
    pub fn bin_energy(self, i: usize) -> f32 {
        if i < self.bins.len() {
            self.bins[i]
        } else {
            0.0
        }
    }

    /// Summed energy of every bin whose centre falls in `[lo_hz, hi_hz]`.
    ///
    /// Boundaries are located with `floor` via [`Spectrum::hz_to_bin`] and
    /// clamped into range; the two endpoints are swapped when passed reversed so
    /// the range is always well-formed. Summation (rather than averaging) keeps
    /// wider bands proportionally louder, matching how audio-reactive rate
    /// modules weight bass versus treble.
    #[must_use]
    pub fn band_energy(self, lo_hz: f32, hi_hz: f32) -> f32 {
        let count = self.bins.len();
        if count == 0 {
            return 0.0;
        }
        let (a, b) = if lo_hz <= hi_hz {
            (lo_hz, hi_hz)
        } else {
            (hi_hz, lo_hz)
        };
        let lo = self.hz_to_bin(a);
        let hi = self.hz_to_bin(b);
        let mut acc = 0.0;
        let mut i = lo;
        while i <= hi {
            acc += self.bins[i];
            i += 1;
        }
        acc
    }

    /// Splits the spectrum into contiguous bands at the given frequency
    /// `edges_hz`, returning the summed energy of each `[edges[k], edges[k+1]]`
    /// segment.
    ///
    /// `n` edges yield `n - 1` bands; fewer than two edges yield an empty
    /// [`Vec`]. Edges need not be sorted — [`Spectrum::band_energy`] orders each
    /// pair — but sorted, ascending edges are the intended use.
    #[must_use]
    pub fn bands(self, edges_hz: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        if edges_hz.len() < 2 {
            return out;
        }
        let mut i = 0;
        while i + 1 < edges_hz.len() {
            out.push(self.band_energy(edges_hz[i], edges_hz[i + 1]));
            i += 1;
        }
        out
    }

    /// Convenience three-band split (low / mid / high) packed into a [`Vec3`].
    ///
    /// Uses the [`LOW_MID_CROSSOVER_HZ`] and [`MID_HIGH_CROSSOVER_HZ`] crossovers
    /// and the view's [`Spectrum::nyquist_hz`] as the top edge. The three
    /// components (`x` = low, `y` = mid, `z` = high) can drive a colour or a
    /// three-axis force directly through the shared [`Vec3`] algebra.
    #[must_use]
    pub fn low_mid_high(self) -> Vec3 {
        let low = self.band_energy(0.0, LOW_MID_CROSSOVER_HZ);
        let mid = self.band_energy(LOW_MID_CROSSOVER_HZ, MID_HIGH_CROSSOVER_HZ);
        let high = self.band_energy(MID_HIGH_CROSSOVER_HZ, self.nyquist_hz());
        Vec3::new(low, mid, high)
    }
}

/// Attack/release time constants for a linear one-pole envelope follower.
///
/// `attack_seconds` governs how fast the envelope rises toward a larger target;
/// `release_seconds` how fast it falls toward a smaller one. Both are expressed
/// in seconds so the per-step coefficient scales with the frame `dt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvelopeParams {
    /// Seconds to (linearly) approach a rising target.
    pub attack_seconds: f32,
    /// Seconds to (linearly) approach a falling target.
    pub release_seconds: f32,
}

impl EnvelopeParams {
    /// Builds envelope parameters from attack/release times in seconds.
    #[must_use]
    pub fn new(attack_seconds: f32, release_seconds: f32) -> Self {
        Self {
            attack_seconds,
            release_seconds,
        }
    }
}

/// Mutable state of a running envelope follower: the last smoothed value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvelopeState {
    /// The current smoothed envelope value.
    pub value: f32,
}

impl EnvelopeState {
    /// Creates an envelope state seeded to `value`.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self { value }
    }

    /// Advances the envelope one step toward `target` and returns the new value.
    ///
    /// This is a *linear* one-pole follower (no `exp`): the blend coefficient is
    /// `clamp(dt / time_constant, 0, 1)`, using `attack_seconds` while rising
    /// and `release_seconds` while falling, and the value moves
    /// `value += (target - value) * coeff`. Because `coeff` is clamped to
    /// `[0, 1]`, the value approaches `target` monotonically and never
    /// overshoots; when `dt >= time_constant` the coefficient is `1` and the
    /// value snaps to `target` in a single step. A zero (or sub-[`EPS`]) time
    /// constant also snaps immediately.
    pub fn advance(&mut self, target: f32, params: EnvelopeParams, dt: f32) -> f32 {
        let rising = target > self.value;
        let time_constant = if rising {
            params.attack_seconds
        } else {
            params.release_seconds
        };
        let coeff = if time_constant > EPS {
            clamp(dt / time_constant, 0.0, 1.0)
        } else {
            1.0
        };
        self.value += (target - self.value) * coeff;
        self.value
    }
}

/// Tuning for the spectral-flux onset (transient) detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetParams {
    /// Seconds over which the running-average energy is linearly smoothed.
    pub smoothing_seconds: f32,
    /// Minimum (sensitivity-scaled) positive flux that counts as an onset.
    pub threshold: f32,
    /// Multiplier applied to raw flux before the threshold test, so one
    /// threshold can serve inputs of different overall loudness.
    pub sensitivity: f32,
}

impl OnsetParams {
    /// Builds onset parameters from smoothing time, threshold, and sensitivity.
    #[must_use]
    pub fn new(smoothing_seconds: f32, threshold: f32, sensitivity: f32) -> Self {
        Self {
            smoothing_seconds,
            threshold,
            sensitivity,
        }
    }
}

/// One onset-detector evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetResult {
    /// Whether this step crossed the onset threshold.
    pub is_onset: bool,
    /// Half-wave-rectified spectral flux (always `>= 0`): the positive part of
    /// `energy - running_average`.
    pub flux: f32,
    /// Amount by which the sensitivity-scaled flux exceeded the threshold, or
    /// `0.0` when no onset fired. Usable directly as a pulse strength.
    pub trigger: f32,
}

/// A spectral-flux onset detector over a single band's energy stream.
///
/// It compares the incoming band energy against a linearly smoothed running
/// average, half-wave rectifies the difference (only *rises* in energy count),
/// scales by sensitivity, and fires when that exceeds the threshold. The running
/// average is then advanced with a linear one-pole step (no `exp`), keeping the
/// detector transcendental-free and deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnsetDetector {
    /// The linearly smoothed running-average energy baseline.
    pub running_avg: f32,
    /// The detector tuning.
    pub params: OnsetParams,
}

impl OnsetDetector {
    /// Creates a detector with the given tuning and an initial energy baseline.
    #[must_use]
    pub fn new(params: OnsetParams, initial_avg: f32) -> Self {
        Self {
            running_avg: initial_avg,
            params,
        }
    }

    /// Feeds one band-energy sample and returns the onset decision.
    ///
    /// The flux is measured against the *current* baseline before the baseline
    /// is updated, so a sudden energy jump is detected on the very step it
    /// arrives. The baseline then tracks the input with coefficient
    /// `clamp(dt / smoothing_seconds, 0, 1)` (a zero/sub-[`EPS`] smoothing time
    /// snaps the baseline to the input immediately).
    pub fn update(&mut self, energy: f32, dt: f32) -> OnsetResult {
        let diff = energy - self.running_avg;
        let flux = if diff > 0.0 { diff } else { 0.0 };
        let scaled = flux * self.params.sensitivity;
        let is_onset = scaled > self.params.threshold;
        let trigger = if is_onset {
            scaled - self.params.threshold
        } else {
            0.0
        };
        let coeff = if self.params.smoothing_seconds > EPS {
            clamp(dt / self.params.smoothing_seconds, 0.0, 1.0)
        } else {
            1.0
        };
        self.running_avg += (energy - self.running_avg) * coeff;
        OnsetResult {
            is_onset,
            flux,
            trigger,
        }
    }
}

/// Linearly remaps `x` from `[in_lo, in_hi]` onto `[out_lo, out_hi]`, clamped.
///
/// The normalized position is clamped to `[0, 1]`, so the result never leaves
/// `[out_lo, out_hi]` (or `[out_hi, out_lo]` for a descending output range). A
/// degenerate input span (`|in_hi - in_lo| <= EPS`) returns `out_lo`, guarding
/// the scalar divide against zero.
#[must_use]
pub fn map_range(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    if span.abs() <= EPS {
        return out_lo;
    }
    let t = clamp((x - in_lo) / span, 0.0, 1.0);
    out_lo + (out_hi - out_lo) * t
}

/// Normalizes an energy against a reference into `[0, 1]`.
///
/// Returns `0.0` when `ref_energy` is at or below [`EPS`] in magnitude (guarding
/// the scalar divide); otherwise `clamp(energy / ref_energy, 0, 1)`.
#[must_use]
pub fn normalize(energy: f32, ref_energy: f32) -> f32 {
    if ref_energy.abs() <= EPS {
        return 0.0;
    }
    clamp(energy / ref_energy, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only absolute tolerance for comparing computed `f32` results
    /// without bare `==`.
    const TOL: f32 = 1e-5;

    /// Returns `true` when `a` and `b` agree to within [`TOL`].
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn bin_width_and_count() {
        // 8-bin spectrum at 16 kHz over a 16-sample window → 1000 Hz per bin.
        let bins = [0.0f32; 8];
        let s = Spectrum::new(&bins, 16_000.0, 16);
        assert!(approx(s.bin_width_hz(), 1000.0));
        assert_eq!(s.bin_count(), 8);
        assert!(approx(s.nyquist_hz(), 8000.0));
    }

    #[test]
    fn bin_width_guards_zero_fft_size() {
        let bins = [0.0f32; 4];
        let s = Spectrum::new(&bins, 48_000.0, 0);
        assert!(approx(s.bin_width_hz(), 0.0));
        // A zero-width spectrum maps every frequency to bin 0.
        assert_eq!(s.hz_to_bin(1234.0), 0);
    }

    #[test]
    fn hz_to_bin_floor_and_clamp() {
        let bins = [0.0f32; 8];
        let s = Spectrum::new(&bins, 16_000.0, 16); // 1000 Hz/bin
        assert_eq!(s.hz_to_bin(0.0), 0);
        assert_eq!(s.hz_to_bin(999.0), 0); // floor
        assert_eq!(s.hz_to_bin(1000.0), 1); // exact edge
        assert_eq!(s.hz_to_bin(1500.0), 1); // floor
        assert_eq!(s.hz_to_bin(-50.0), 0); // negative clamps low
        assert_eq!(s.hz_to_bin(1_000_000.0), 7); // huge clamps to last bin
    }

    #[test]
    fn hz_to_bin_empty_is_zero() {
        let bins: [f32; 0] = [];
        let s = Spectrum::new(&bins, 44_100.0, 1024);
        assert_eq!(s.hz_to_bin(440.0), 0);
    }

    #[test]
    fn bin_energy_in_and_out_of_range() {
        let bins = [1.0, 2.0, 3.0, 4.0];
        let s = Spectrum::new(&bins, 8000.0, 8); // 1000 Hz/bin
        assert!(approx(s.bin_energy(2), 3.0));
        assert!(approx(s.bin_energy(4), 0.0)); // out of range
    }

    #[test]
    fn band_energy_sums_interval() {
        // 1000 Hz/bin, bins 0..4 hold 1,2,3,4.
        let bins = [1.0, 2.0, 3.0, 4.0];
        let s = Spectrum::new(&bins, 8000.0, 8);
        // [1000, 3000] → bins 1..=3 → 2+3+4 = 9.
        assert!(approx(s.band_energy(1000.0, 3000.0), 9.0));
        // Reversed arguments give the same interval.
        assert!(approx(s.band_energy(3000.0, 1000.0), 9.0));
        // Sub-bin range still includes the located bin 0.
        assert!(approx(s.band_energy(0.0, 500.0), 1.0));
    }

    #[test]
    fn bands_splits_by_edges() {
        // 1000 Hz/bin, bins 0..8 hold their own index energy.
        let bins = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let s = Spectrum::new(&bins, 16_000.0, 16);
        // Edges at 0, 2000, 5000 Hz → two bands.
        // Band A: [0,2000] → bins 0..=2 → 0+1+2 = 3.
        // Band B: [2000,5000] → bins 2..=5 → 2+3+4+5 = 14.
        let out = s.bands(&[0.0, 2000.0, 5000.0]);
        assert_eq!(out.len(), 2);
        assert!(approx(out[0], 3.0));
        assert!(approx(out[1], 14.0));
    }

    #[test]
    fn bands_needs_two_edges() {
        let bins = [1.0, 2.0, 3.0];
        let s = Spectrum::new(&bins, 6000.0, 6);
        assert!(s.bands(&[]).is_empty());
        assert!(s.bands(&[1000.0]).is_empty());
    }

    #[test]
    fn low_mid_high_three_band_split() {
        // 100 Hz/bin (44100/441 ≈ 100), 6 bins covering 0..600 Hz would be too
        // narrow; use a wider window so crossovers land cleanly.
        // 8000 Hz sample rate, fft 16 → 500 Hz/bin, 8 bins → 0..4000 Hz.
        let bins = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let s = Spectrum::new(&bins, 8000.0, 16); // 500 Hz/bin, nyquist 4000
        let v = s.low_mid_high();
        // low: [0,250] → bin 0 → 1.
        assert!(approx(v.x, 1.0));
        // mid: [250,4000] → bins 0..=8 clamp → 0..=7 → 8 bins.
        assert!(v.y > v.x);
        // high: [4000,4000] → bin 7 → 1.
        assert!(approx(v.z, 1.0));
    }

    #[test]
    fn envelope_converges_without_overshoot() {
        let params = EnvelopeParams::new(0.1, 0.1);
        let mut env = EnvelopeState::new(0.0);
        let mut last = 0.0;
        // dt = 0.02 → coeff = 0.2 per step; approach 1.0 monotonically.
        for _ in 0..100 {
            let v = env.advance(1.0, params, 0.02);
            assert!(v >= last - TOL); // monotically rising
            assert!(v <= 1.0 + TOL); // never overshoots the target
            last = v;
        }
        assert!(approx(last, 1.0));
    }

    #[test]
    fn envelope_attack_faster_than_release() {
        // Fast attack, slow release.
        let params = EnvelopeParams::new(0.01, 1.0);
        let mut rising = EnvelopeState::new(0.0);
        let up = rising.advance(1.0, params, 0.02); // dt >= attack → snap
        assert!(approx(up, 1.0));

        let mut falling = EnvelopeState::new(1.0);
        // Slow release: coeff = 0.02 / 1.0 = 0.02 → value = 1 - 0.02 = 0.98.
        let down = falling.advance(0.0, params, 0.02);
        assert!(approx(down, 0.98));
        assert!(down > 0.5); // release is much slower than attack
    }

    #[test]
    fn envelope_snaps_when_dt_exceeds_time() {
        let params = EnvelopeParams::new(0.1, 0.1);
        let mut env = EnvelopeState::new(0.0);
        // dt >= attack_seconds → coeff clamps to 1 → one-step to target.
        let v = env.advance(0.75, params, 0.5);
        assert!(approx(v, 0.75));
    }

    #[test]
    fn envelope_zero_time_constant_snaps() {
        let params = EnvelopeParams::new(0.0, 0.0);
        let mut env = EnvelopeState::new(0.2);
        let v = env.advance(0.9, params, 0.016);
        assert!(approx(v, 0.9));
    }

    #[test]
    fn onset_fires_on_energy_jump() {
        let params = OnsetParams::new(0.2, 0.5, 1.0);
        let mut det = OnsetDetector::new(params, 0.0);
        // Baseline 0, sudden energy 2.0 → flux 2.0 > threshold 0.5 → onset.
        let r = det.update(2.0, 0.016);
        assert!(r.is_onset);
        assert!(approx(r.flux, 2.0));
        assert!(approx(r.trigger, 1.5)); // 2.0 - 0.5
    }

    #[test]
    fn onset_quiet_when_steady() {
        let params = OnsetParams::new(0.2, 0.5, 1.0);
        // Baseline already at the steady level → no rise → no onset.
        let mut det = OnsetDetector::new(params, 1.0);
        let r = det.update(1.0, 0.016);
        assert!(!r.is_onset);
        assert!(approx(r.flux, 0.0));
        assert!(approx(r.trigger, 0.0));
    }

    #[test]
    fn onset_flux_is_half_wave_rectified() {
        let params = OnsetParams::new(0.2, 0.5, 1.0);
        // Energy below baseline → negative diff → flux clamped to 0.
        let mut det = OnsetDetector::new(params, 5.0);
        let r = det.update(1.0, 0.016);
        assert!(approx(r.flux, 0.0));
        assert!(!r.is_onset);
    }

    #[test]
    fn onset_sensitivity_scales_flux() {
        // Same raw flux, but sensitivity 3.0 pushes it over the threshold.
        let params = OnsetParams::new(0.2, 1.0, 3.0);
        let mut det = OnsetDetector::new(params, 0.0);
        let r = det.update(0.5, 0.016); // scaled flux = 1.5 > 1.0
        assert!(r.is_onset);
        assert!(approx(r.flux, 0.5)); // raw flux unscaled
        assert!(approx(r.trigger, 0.5)); // 1.5 - 1.0
    }

    #[test]
    fn map_range_linear_endpoints_and_clamp() {
        // Map [0,10] → [0,100].
        assert!(approx(map_range(0.0, 0.0, 10.0, 0.0, 100.0), 0.0));
        assert!(approx(map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0));
        assert!(approx(map_range(10.0, 0.0, 10.0, 0.0, 100.0), 100.0));
        // Below/above input range clamps to output endpoints.
        assert!(approx(map_range(-5.0, 0.0, 10.0, 0.0, 100.0), 0.0));
        assert!(approx(map_range(15.0, 0.0, 10.0, 0.0, 100.0), 100.0));
    }

    #[test]
    fn map_range_descending_output() {
        // [0,1] → [10, 0]: input 0.25 → 7.5.
        assert!(approx(map_range(0.25, 0.0, 1.0, 10.0, 0.0), 7.5));
    }

    #[test]
    fn map_range_degenerate_span_returns_out_lo() {
        assert!(approx(map_range(5.0, 3.0, 3.0, -1.0, 1.0), -1.0));
    }

    #[test]
    fn normalize_clamps_to_unit_interval() {
        assert!(approx(normalize(0.0, 4.0), 0.0));
        assert!(approx(normalize(2.0, 4.0), 0.5));
        assert!(approx(normalize(4.0, 4.0), 1.0));
        assert!(approx(normalize(9.0, 4.0), 1.0)); // clamps high
        assert!(approx(normalize(-1.0, 4.0), 0.0)); // clamps low
        assert!(approx(normalize(1.0, 0.0), 0.0)); // guarded reference
    }
}
