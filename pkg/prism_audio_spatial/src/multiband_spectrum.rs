//! Configurable multi-band spectral gains for high-fidelity propagation.
//!
//! The real-time render path carries a fixed three-band [`BandGains`] EQ (see
//! [`band_spectrum`](crate::band_spectrum)), the compact representation every
//! shipping engine renders with a two-crossover filterbank. That fixed layout
//! is deliberate: it is cheap, allocation free, and matches the runtime EQ of
//! Steam Audio. It is, however, too coarse for *authoring* and *baking*, where
//! a geometry or wave solver wants to describe an arrival's colour at arbitrary
//! spectral resolution before it is resolved down to the runtime bands.
//!
//! This module adds that high-resolution layer. A [`MultibandLayout`] describes
//! an arbitrary number of contiguous, log-spaced frequency bands; a
//! [`MultibandGains`] carries a per-band linear amplitude gain aligned to a
//! layout. Both are **control-rate / authoring-time** objects that allocate on
//! construction and therefore never run on the audio callback thread; the one
//! bridge to the hot path is [`MultibandGains::to_band_gains`], which resolves
//! the high-resolution colour down to the fixed three-band runtime EQ while
//! preserving per-band energy.
//!
//! # Why configurable bands
//!
//! A single global band count is a false economy. A dull cloth drape only needs
//! three bands; a resonant stairwell or a tuned Helmholtz cavity needs a dozen
//! to describe its notches without smearing them. Steam Audio, Wwise and FMOD
//! all let a project pick the propagation band count as a quality knob. This
//! module makes the same choice a first-class, validated type rather than a
//! hard-coded `3`.
//!
//! # Relationship
//!
//! [`MultibandGains`] is the high-resolution sibling of [`BandGains`]:
//! [`from_band_gains`](MultibandGains::from_band_gains) up-samples the runtime
//! EQ onto any layout, and [`to_band_gains`](MultibandGains::to_band_gains)
//! resolves any layout back down to the runtime EQ by integrating energy over
//! each runtime band. Material and low-pass constructors mirror the
//! [`BandGains`] ones so a backend can build colour at either resolution with
//! the same vocabulary.
//!
//! [`BandGains`]: crate::band_spectrum::BandGains

use alloc::vec::Vec;

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::band_spectrum::{
    BandGains, PROPAGATION_BAND_EDGES, PROPAGATION_BAND_HIGH_HZ, PROPAGATION_BAND_LOW_HZ,
};
use crate::material_library::{MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// Smallest band edge frequency (Hz) a layout will accept, keeping the
/// log-frequency math finite and the lowest band audible.
pub const MIN_BAND_FREQUENCY_HZ: Sample = 1.0;

/// Largest band edge frequency (Hz) a layout will accept, a decade above the
/// nominal audible top so an ultrasonic band is still representable.
pub const MAX_BAND_FREQUENCY_HZ: Sample = 192_000.0;

/// A contiguous set of log-spaced frequency bands covering `[low, high]`.
///
/// A layout owns `band_count` representative centre frequencies (the geometric
/// mean of each band's bounds) and the `band_count - 1` crossover edges between
/// them. All frequencies are strictly increasing and lie in
/// `[MIN_BAND_FREQUENCY_HZ, MAX_BAND_FREQUENCY_HZ]`. Construct one with
/// [`log_spaced`](Self::log_spaced) for an even geometric split,
/// [`from_edges`](Self::from_edges) for explicit crossovers,
/// [`octave_bands`](Self::octave_bands) for one band per octave, or
/// [`propagation`](Self::propagation) to mirror the fixed runtime layout.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MultibandLayout {
    /// Low bound (Hz) of the lowest band.
    low_hz: Sample,
    /// High bound (Hz) of the highest band.
    high_hz: Sample,
    /// Interior crossover frequencies (Hz), strictly increasing, length
    /// `band_count - 1`.
    edges: Vec<Sample>,
    /// Representative centre frequency (Hz) of each band: the geometric mean of
    /// its bounds. Length `band_count`, strictly increasing.
    centers: Vec<Sample>,
}

impl MultibandLayout {
    /// Builds a layout of `band_count` bands evenly spaced in log-frequency
    /// across `[low_hz, high_hz]`.
    ///
    /// `band_count` is clamped to at least `1`. The bounds are sanitised into
    /// `[MIN_BAND_FREQUENCY_HZ, MAX_BAND_FREQUENCY_HZ]` and ordered, so a
    /// degenerate or reversed request still yields a valid single-decade
    /// layout. Each interior edge sits at an equal log-frequency step and each
    /// centre at the geometric mean of its band's bounds.
    ///
    /// ```
    /// use prism_audio_spatial::multiband_spectrum::MultibandLayout;
    ///
    /// let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 8);
    /// assert_eq!(layout.band_count(), 8);
    /// assert_eq!(layout.edges().len(), 7);
    /// // Centres share one constant log-frequency ratio.
    /// let centres = layout.centers();
    /// let ratio = centres[1] / centres[0];
    /// assert!((centres[2] / centres[1] - ratio).abs() < 1e-3);
    /// ```
    #[must_use]
    pub fn log_spaced(low_hz: Sample, high_hz: Sample, band_count: usize) -> Self {
        let (low, high) = sanitise_bounds(low_hz, high_hz);
        let count = band_count.max(1);
        let log_lo = ops::ln(low);
        let log_hi = ops::ln(high);
        let step = (log_hi - log_lo) / count as Sample;
        let mut edges = Vec::with_capacity(count - 1);
        for k in 1..count {
            edges.push(ops::exp(log_lo + step * k as Sample));
        }
        let mut centers = Vec::with_capacity(count);
        for k in 0..count {
            centers.push(ops::exp(log_lo + step * (k as Sample + 0.5)));
        }
        Self {
            low_hz: low,
            high_hz: high,
            edges,
            centers,
        }
    }

    /// Builds a layout from explicit interior crossover edges spanning
    /// `[low_hz, high_hz]`.
    ///
    /// The edges are sanitised: non-finite values are dropped, every value is
    /// clamped strictly inside `(low, high)`, the list is sorted ascending and
    /// de-duplicated. The band count is one more than the number of surviving
    /// edges, and each centre is the geometric mean of its band's bounds. An
    /// empty edge list yields a single band spanning the whole range.
    #[must_use]
    pub fn from_edges(low_hz: Sample, high_hz: Sample, edges: &[Sample]) -> Self {
        let (low, high) = sanitise_bounds(low_hz, high_hz);
        let mut clean: Vec<Sample> = Vec::with_capacity(edges.len());
        for &e in edges {
            if e.is_finite() && e > low && e < high {
                clean.push(e);
            }
        }
        clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        clean.dedup_by(|a, b| ops::abs(*a - *b) <= Sample::EPSILON * *b);
        let mut bounds = Vec::with_capacity(clean.len() + 2);
        bounds.push(low);
        bounds.extend_from_slice(&clean);
        bounds.push(high);
        let mut centers = Vec::with_capacity(bounds.len() - 1);
        for pair in bounds.windows(2) {
            centers.push(ops::sqrt(pair[0] * pair[1]));
        }
        Self {
            low_hz: low,
            high_hz: high,
            edges: clean,
            centers,
        }
    }

    /// Builds a layout with one band per octave across `[low_hz, high_hz]`.
    ///
    /// The number of octaves is derived from the sanitised bounds and used as
    /// the band count for a [`log_spaced`](Self::log_spaced) layout, so bands
    /// land on even one-octave log-frequency steps.
    #[must_use]
    pub fn octave_bands(low_hz: Sample, high_hz: Sample) -> Self {
        let (low, high) = sanitise_bounds(low_hz, high_hz);
        let octaves = ops::log2(high / low);
        let count = (octaves + 0.5) as usize;
        Self::log_spaced(low, high, count.max(1))
    }

    /// Builds the layout that mirrors the fixed runtime three-band EQ
    /// ([`PROPAGATION_BAND_EDGES`] over `[20, 20000] Hz`).
    ///
    /// Resolving a [`MultibandGains`] built on this layout to a [`BandGains`]
    /// is an identity up to the energy integration tolerance.
    #[must_use]
    pub fn propagation() -> Self {
        Self::from_edges(
            PROPAGATION_BAND_LOW_HZ,
            PROPAGATION_BAND_HIGH_HZ,
            &PROPAGATION_BAND_EDGES,
        )
    }

    /// The number of bands in the layout (always at least `1`).
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.centers.len()
    }

    /// The representative centre frequencies (Hz), strictly increasing.
    #[inline]
    #[must_use]
    pub fn centers(&self) -> &[Sample] {
        &self.centers
    }

    /// The interior crossover edges (Hz), length `band_count - 1`.
    #[inline]
    #[must_use]
    pub fn edges(&self) -> &[Sample] {
        &self.edges
    }

    /// The low bound (Hz) of the lowest band.
    #[inline]
    #[must_use]
    pub fn low_hz(&self) -> Sample {
        self.low_hz
    }

    /// The high bound (Hz) of the highest band.
    #[inline]
    #[must_use]
    pub fn high_hz(&self) -> Sample {
        self.high_hz
    }

    /// The `(low, high)` frequency bounds (Hz) of band `index`, saturating to
    /// the last band for an out-of-range index (never panics).
    #[must_use]
    pub fn band_bounds(&self, index: usize) -> (Sample, Sample) {
        let last = self.band_count() - 1;
        let i = index.min(last);
        let lo = if i == 0 { self.low_hz } else { self.edges[i - 1] };
        let hi = if i == last {
            self.high_hz
        } else {
            self.edges[i]
        };
        (lo, hi)
    }
}

impl Default for MultibandLayout {
    /// The fixed runtime propagation layout.
    #[inline]
    fn default() -> Self {
        Self::propagation()
    }
}

/// A per-band linear amplitude gain in `[0, 1]`, aligned to a
/// [`MultibandLayout`].
///
/// Like [`BandGains`], `1.0` passes a band untouched and `0.0` silences it, and
/// every band is clamped into `[0, 1]` on construction so the spectrum is
/// always a physically valid attenuation. Unlike [`BandGains`] the band count
/// is whatever the owning layout declares.
///
/// [`BandGains`]: crate::band_spectrum::BandGains
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MultibandGains {
    /// The frequency layout the gains are aligned to.
    layout: MultibandLayout,
    /// Per-band linear amplitude gains in `[0, 1]`, length `band_count`.
    gains: Vec<Sample>,
}

impl MultibandGains {
    /// Builds a spectrum from per-band gains aligned to `layout`.
    ///
    /// The input is clamped into `[0, 1]` (non-finite becomes `0`) and resized
    /// to the layout's band count: a short input repeats its last value (or
    /// unity if empty) and a long input is truncated. This never panics on a
    /// length mismatch.
    #[must_use]
    pub fn from_bands(layout: MultibandLayout, gains: &[Sample]) -> Self {
        let count = layout.band_count();
        let mut out = Vec::with_capacity(count);
        let mut last = 1.0;
        for i in 0..count {
            let raw = if i < gains.len() {
                gains[i]
            } else if gains.is_empty() {
                1.0
            } else {
                last
            };
            let clamped = if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            };
            last = clamped;
            out.push(clamped);
        }
        Self {
            layout,
            gains: out,
        }
    }

    /// Builds a flat spectrum with the same gain in every band.
    #[must_use]
    pub fn uniform(layout: MultibandLayout, gain: Sample) -> Self {
        let g = if gain.is_finite() {
            gain.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let count = layout.band_count();
        Self {
            layout,
            gains: vec_filled(g, count),
        }
    }

    /// A flat, full-band spectrum on `layout`: every band passes untouched.
    #[must_use]
    pub fn unity(layout: MultibandLayout) -> Self {
        Self::uniform(layout, 1.0)
    }

    /// A fully attenuated spectrum on `layout`: every band silenced.
    #[must_use]
    pub fn silent(layout: MultibandLayout) -> Self {
        Self::uniform(layout, 0.0)
    }

    /// The layout the gains are aligned to.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> &MultibandLayout {
        &self.layout
    }

    /// The number of bands.
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.gains.len()
    }

    /// The raw per-band gains, aligned to [`layout`](Self::layout)'s centres.
    #[inline]
    #[must_use]
    pub fn bands(&self) -> &[Sample] {
        &self.gains
    }

    /// The gain of band `index`, saturating to the last band for an
    /// out-of-range index (never panics).
    #[inline]
    #[must_use]
    pub fn band(&self, index: usize) -> Sample {
        self.gains[index.min(self.gains.len() - 1)]
    }

    /// The spectrum gain at an arbitrary frequency, interpolated linearly in
    /// `log(frequency)` between the two bracketing band centres.
    ///
    /// Frequencies at or below the lowest centre return the lowest band and
    /// frequencies at or above the highest centre return the highest band (the
    /// spectrum is clamped, never extrapolated). A non-finite frequency falls
    /// back to the lowest band. Mirrors [`BandGains::at`].
    ///
    /// [`BandGains::at`]: crate::band_spectrum::BandGains::at
    #[must_use]
    pub fn at(&self, freq_hz: Sample) -> Sample {
        let centers = self.layout.centers();
        let last = centers.len() - 1;
        if !freq_hz.is_finite() || freq_hz <= centers[0] {
            return self.gains[0];
        }
        if freq_hz >= centers[last] {
            return self.gains[last];
        }
        let log_f = ops::ln(freq_hz);
        for i in 0..last {
            let c_hi = centers[i + 1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(centers[i]);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > 0.0 {
                    (log_f - log_lo) / span
                } else {
                    0.0
                };
                let v = self.gains[i] + (self.gains[i + 1] - self.gains[i]) * t;
                return v.clamp(0.0, 1.0);
            }
        }
        self.gains[last]
    }

    /// Scales every band by a scalar broadband gain, re-clamping into `[0, 1]`.
    #[must_use]
    pub fn scaled(&self, gain: Sample) -> Self {
        let mut out = self.gains.clone();
        for g in &mut out {
            let scaled = *g * gain;
            *g = if scaled.is_finite() {
                scaled.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        Self {
            layout: self.layout.clone(),
            gains: out,
        }
    }

    /// Combines two spectra by multiplying each band, modelling a signal that
    /// passes through both colourations in series.
    ///
    /// If `other` uses a different layout it is first resampled onto this
    /// spectrum's layout (via [`at`](Self::at) at each centre), so the result
    /// always keeps this spectrum's resolution and the operation never panics
    /// on a layout mismatch.
    #[must_use]
    pub fn combine(&self, other: &Self) -> Self {
        let resampled;
        let other_bands: &[Sample] = if other.layout == self.layout {
            &other.gains
        } else {
            resampled = other.resample_onto(&self.layout);
            &resampled.gains
        };
        let mut out = self.gains.clone();
        for (o, &b) in out.iter_mut().zip(other_bands.iter()) {
            *o *= b;
        }
        Self {
            layout: self.layout.clone(),
            gains: out,
        }
    }

    /// Resamples this spectrum onto `target`, evaluating [`at`](Self::at) at
    /// each of the target layout's band centres.
    #[must_use]
    pub fn resample_onto(&self, target: &MultibandLayout) -> Self {
        let mut out = Vec::with_capacity(target.band_count());
        for &centre in target.centers() {
            out.push(self.at(centre));
        }
        Self {
            layout: target.clone(),
            gains: out,
        }
    }

    /// The root-mean-square of the band gains: a single broadband amplitude
    /// that preserves total energy when the spectrum is collapsed to one
    /// number. Always in `[0, 1]`.
    #[must_use]
    pub fn broadband_rms(&self) -> Sample {
        let mut sum_sq = 0.0;
        for &g in &self.gains {
            sum_sq += g * g;
        }
        ops::sqrt(sum_sq / self.gains.len() as Sample)
    }

    /// The largest band gain: the brightest frequency the arrival still passes.
    #[must_use]
    pub fn peak(&self) -> Sample {
        self.gains
            .iter()
            .copied()
            .fold(0.0, |acc, g| if g > acc { g } else { acc })
    }

    /// Returns `true` when every band is within `tolerance` of unity.
    #[must_use]
    pub fn is_full_band(&self, tolerance: Sample) -> bool {
        let tol = tolerance.max(0.0);
        self.gains.iter().all(|&g| ops::abs(g - 1.0) <= tol)
    }

    /// Returns `true` when every band is at or below `tolerance`.
    #[must_use]
    pub fn is_silent(&self, tolerance: Sample) -> bool {
        let tol = tolerance.max(0.0);
        self.gains.iter().all(|&g| g <= tol)
    }

    /// Approximates a first-order (`6 dB`/octave) low-pass with corner
    /// `cutoff_hz` on `layout`, sampling `1 / sqrt(1 + (f / fc)^2)` at each band
    /// centre. Mirrors [`BandGains::from_lowpass_cutoff`].
    ///
    /// [`BandGains::from_lowpass_cutoff`]: crate::band_spectrum::BandGains::from_lowpass_cutoff
    #[must_use]
    pub fn from_lowpass_cutoff(layout: MultibandLayout, cutoff_hz: Sample) -> Self {
        if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 {
            return Self::silent(layout);
        }
        let mut gains = Vec::with_capacity(layout.band_count());
        for &centre in layout.centers() {
            let ratio = centre / cutoff_hz;
            gains.push(1.0 / ops::sqrt(1.0 + ratio * ratio));
        }
        Self::from_bands(layout, &gains)
    }

    /// Converts an octave-band [`MaterialAbsorption`] into a per-band amplitude
    /// reflection coefficient on `layout`.
    ///
    /// Each band averages the octave-band energy-absorption coefficients whose
    /// ISO centre falls inside the band's bounds (falling back to the nearest
    /// populated band so the result is always defined), then converts reflected
    /// energy `1 - alpha` into an amplitude gain `sqrt(1 - alpha)`. This is the
    /// arbitrary-resolution sibling of
    /// [`BandGains::reflection_from_absorption`].
    ///
    /// [`BandGains::reflection_from_absorption`]: crate::band_spectrum::BandGains::reflection_from_absorption
    #[must_use]
    pub fn reflection_from_absorption(
        layout: MultibandLayout,
        absorption: &MaterialAbsorption,
    ) -> Self {
        let octaves = absorption.bands();
        let count = layout.band_count();
        let mut gains = Vec::with_capacity(count);
        let mut last_defined = ops::sqrt((1.0 - octaves[0]).max(0.0));
        for b in 0..count {
            let (lo, hi) = layout.band_bounds(b);
            let mut sum = 0.0;
            let mut n = 0u32;
            for i in 0..OCTAVE_BAND_COUNT {
                let centre = OCTAVE_BAND_CENTERS[i];
                if centre >= lo && centre < hi {
                    sum += octaves[i];
                    n += 1;
                }
            }
            if n > 0 {
                let alpha = (sum / n as Sample).clamp(0.0, 1.0);
                let gain = ops::sqrt((1.0 - alpha).max(0.0));
                last_defined = gain;
                gains.push(gain);
            } else {
                gains.push(last_defined);
            }
        }
        Self::from_bands(layout, &gains)
    }

    /// Up-samples a fixed runtime [`BandGains`] onto `layout` by sampling the
    /// runtime EQ response at each of the layout's band centres.
    ///
    /// [`BandGains`]: crate::band_spectrum::BandGains
    #[must_use]
    pub fn from_band_gains(layout: MultibandLayout, gains: &BandGains) -> Self {
        let mut out = Vec::with_capacity(layout.band_count());
        for &centre in layout.centers() {
            let runtime_band = if centre < PROPAGATION_BAND_EDGES[0] {
                0
            } else if centre < PROPAGATION_BAND_EDGES[1] {
                1
            } else {
                2
            };
            out.push(gains.band(runtime_band));
        }
        Self::from_bands(layout, &out)
    }

    /// Resolves the high-resolution spectrum down to the fixed runtime
    /// three-band [`BandGains`], preserving per-band energy.
    ///
    /// For each runtime band spanning `[f_lo, f_hi]`, the mean-square gain is
    /// integrated over log-frequency (so energy, not amplitude, is averaged)
    /// and the band amplitude is the square root of that mean. This is the
    /// bridge from the authoring layer to the audio callback thread: the
    /// returned [`BandGains`] is the compact EQ the real-time voice renders.
    ///
    ///
    /// ```
    /// use prism_audio_spatial::{MultibandGains, MultibandLayout};
    /// let authored = MultibandGains::from_bands(MultibandLayout::propagation(), &[1.0, 0.1, 1.0]);
    /// let runtime = authored.to_band_gains();
    /// // The authored mid-band notch survives the resolve to the runtime EQ.
    /// assert!(runtime.mid() < runtime.low());
    /// assert!(runtime.mid() < runtime.high());
    /// ```
    ///
    /// [`BandGains`]: crate::band_spectrum::BandGains
    #[must_use]
    pub fn to_band_gains(&self) -> BandGains {
        let runtime_bounds = [
            (PROPAGATION_BAND_LOW_HZ, PROPAGATION_BAND_EDGES[0]),
            (PROPAGATION_BAND_EDGES[0], PROPAGATION_BAND_EDGES[1]),
            (PROPAGATION_BAND_EDGES[1], PROPAGATION_BAND_HIGH_HZ),
        ];
        let mut bands = [0.0; 3];
        for (out, &(lo, hi)) in bands.iter_mut().zip(runtime_bounds.iter()) {
            *out = self.band_energy_rms(lo, hi);
        }
        BandGains::new(bands)
    }

    /// The root-mean-square gain of the spectrum over `[lo, hi]`, summing each
    /// band's mean-square gain weighted by its log-frequency overlap with the
    /// query range.
    ///
    /// Each band is a flat gain over its own bounds (a piecewise-constant
    /// filterbank), so the energy over `[lo, hi]` is the exact sum of every
    /// band's `gain^2` times the log-width it covers inside the range. This
    /// confines a band's energy to its own bounds, so a notch in one band never
    /// bleeds into an adjacent runtime band. Returns the single-centre gain
    /// when the span is degenerate so a zero-width band is still defined.
    #[must_use]
    fn band_energy_rms(&self, lo: Sample, hi: Sample) -> Sample {
        let log_lo = ops::ln(lo);
        let log_hi = ops::ln(hi);
        let span = log_hi - log_lo;
        if span <= 0.0 {
            return self.at(ops::sqrt(lo * hi));
        }
        let mut energy = 0.0;
        for b in 0..self.band_count() {
            let (b_lo, b_hi) = self.layout.band_bounds(b);
            let o_lo = b_lo.max(lo);
            let o_hi = b_hi.min(hi);
            if o_hi > o_lo {
                let width = ops::ln(o_hi) - ops::ln(o_lo);
                let g = self.gains[b];
                energy += g * g * width;
            }
        }
        let mean_sq = energy / span;
        ops::sqrt(mean_sq.max(0.0)).clamp(0.0, 1.0)
    }
}

/// Clamps a `(low, high)` frequency pair into the representable range and
/// orders it so `low < high`, keeping at least a hair of span for the
/// log-frequency math.
fn sanitise_bounds(low_hz: Sample, high_hz: Sample) -> (Sample, Sample) {
    let a = if low_hz.is_finite() {
        low_hz.clamp(MIN_BAND_FREQUENCY_HZ, MAX_BAND_FREQUENCY_HZ)
    } else {
        MIN_BAND_FREQUENCY_HZ
    };
    let b = if high_hz.is_finite() {
        high_hz.clamp(MIN_BAND_FREQUENCY_HZ, MAX_BAND_FREQUENCY_HZ)
    } else {
        MAX_BAND_FREQUENCY_HZ
    };
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if hi > lo {
        (lo, hi)
    } else {
        // Degenerate equal bounds: open a one-octave window upward (or
        // downward at the ceiling) so the layout spans a real range.
        let widened = (hi * 2.0).min(MAX_BAND_FREQUENCY_HZ);
        if widened > lo {
            (lo, widened)
        } else {
            ((lo * 0.5).max(MIN_BAND_FREQUENCY_HZ), hi)
        }
    }
}

/// Builds a `Vec` of `count` copies of `value` without an intrinsic `f32`
/// method, keeping the determinism contract.
fn vec_filled(value: Sample, count: usize) -> Vec<Sample> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(value);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_spectrum::PROPAGATION_BAND_CENTERS;
    use crate::material_library::Material;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        ops::abs(a - b) <= tol
    }

    #[test]
    fn log_spaced_bands_are_geometrically_even() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 10);
        assert_eq!(layout.band_count(), 10);
        assert_eq!(layout.edges().len(), 9);
        // Successive centres share a constant log ratio.
        let centers = layout.centers();
        let first_ratio = centers[1] / centers[0];
        for pair in centers.windows(2) {
            assert!(approx(pair[1] / pair[0], first_ratio, 1e-3));
        }
    }

    #[test]
    fn log_spaced_clamps_degenerate_requests() {
        // Zero band count clamps to one band; reversed bounds get ordered.
        let layout = MultibandLayout::log_spaced(20_000.0, 20.0, 0);
        assert_eq!(layout.band_count(), 1);
        assert!(layout.low_hz() < layout.high_hz());
    }

    #[test]
    fn from_edges_sanitises_and_sorts() {
        // Out-of-range, duplicate, reversed and non-finite edges are dropped
        // or ordered into a clean strictly-increasing list.
        let layout = MultibandLayout::from_edges(
            20.0,
            20_000.0,
            &[8000.0, 800.0, 10.0, 40_000.0, Sample::NAN, 800.0],
        );
        assert_eq!(layout.edges(), &[800.0, 8000.0]);
        assert_eq!(layout.band_count(), 3);
    }

    #[test]
    fn band_bounds_cover_the_range_contiguously() {
        let layout = MultibandLayout::from_edges(20.0, 20_000.0, &[800.0, 8000.0]);
        assert_eq!(layout.band_bounds(0), (20.0, 800.0));
        assert_eq!(layout.band_bounds(1), (800.0, 8000.0));
        assert_eq!(layout.band_bounds(2), (8000.0, 20_000.0));
        // Out-of-range index saturates to the last band.
        assert_eq!(layout.band_bounds(99), (8000.0, 20_000.0));
    }

    #[test]
    fn propagation_layout_matches_runtime_centres() {
        let layout = MultibandLayout::propagation();
        assert_eq!(layout.band_count(), 3);
        for (c, &p) in layout.centers().iter().zip(PROPAGATION_BAND_CENTERS.iter()) {
            assert!(approx(*c, p, 0.5));
        }
    }

    #[test]
    fn octave_bands_span_the_requested_decades() {
        // 20 Hz..20 kHz is ~9.97 octaves, so ~10 octave bands.
        let layout = MultibandLayout::octave_bands(20.0, 20_480.0);
        assert_eq!(layout.band_count(), 10);
    }

    #[test]
    fn from_bands_clamps_and_resizes() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 4);
        // Short input repeats its last value; out-of-range values clamp.
        let g = MultibandGains::from_bands(layout, &[1.5, -0.2]);
        assert_eq!(g.bands(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn uniform_sentinels() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 6);
        assert!(MultibandGains::unity(layout.clone()).is_full_band(1e-6));
        assert!(MultibandGains::silent(layout).is_silent(1e-6));
    }

    #[test]
    fn at_interpolates_in_log_frequency() {
        let layout = MultibandLayout::propagation();
        let g = MultibandGains::from_bands(layout.clone(), &[0.2, 0.6, 1.0]);
        assert!(approx(g.at(1.0), 0.2, 1e-6));
        assert!(approx(g.at(40_000.0), 1.0, 1e-6));
        let centers = layout.centers();
        assert!(approx(g.at(centers[1]), 0.6, 1e-6));
        let midpoint = ops::sqrt(centers[0] * centers[1]);
        assert!(approx(g.at(midpoint), 0.4, 1e-3));
    }

    #[test]
    fn scaled_reclamps() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 3);
        let g = MultibandGains::from_bands(layout, &[0.6, 0.4, 0.2]).scaled(0.5);
        assert!(approx(g.band(0), 0.3, 1e-6));
        assert!(approx(g.band(1), 0.2, 1e-6));
        assert!(approx(g.band(2), 0.1, 1e-6));
    }

    #[test]
    fn combine_resamples_mismatched_layouts() {
        let fine = MultibandLayout::log_spaced(20.0, 20_000.0, 8);
        let coarse = MultibandLayout::log_spaced(20.0, 20_000.0, 3);
        let a = MultibandGains::uniform(fine.clone(), 0.5);
        let b = MultibandGains::uniform(coarse, 0.5);
        let c = a.combine(&b);
        // Result keeps the finer layout and multiplies band-for-band.
        assert_eq!(c.band_count(), 8);
        for &g in c.bands() {
            assert!(approx(g, 0.25, 1e-6));
        }
    }

    #[test]
    fn broadband_rms_and_peak() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 3);
        let g = MultibandGains::from_bands(layout, &[0.0, 0.0, 1.0]);
        assert!(approx(g.broadband_rms(), ops::sqrt(1.0 / 3.0), 1e-6));
        assert!(approx(g.peak(), 1.0, 1e-6));
    }

    #[test]
    fn lowpass_rolls_off_above_the_corner() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 24);
        let corner = 1000.0;
        let g = MultibandGains::from_lowpass_cutoff(layout, corner);
        // At the corner the magnitude is -3 dB; well above it is much darker.
        assert!(approx(g.at(corner), 1.0 / ops::sqrt(2.0), 1e-3));
        assert!(g.at(100.0) > g.at(corner));
        assert!(g.at(10_000.0) < g.at(corner));
    }

    #[test]
    fn full_band_resolves_to_unity() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 16);
        let g = MultibandGains::unity(layout);
        let runtime = g.to_band_gains();
        assert!(runtime.is_full_band(1e-3));
    }

    #[test]
    fn silence_resolves_to_silence() {
        let layout = MultibandLayout::log_spaced(20.0, 20_000.0, 16);
        let runtime = MultibandGains::silent(layout).to_band_gains();
        assert!(runtime.is_silent(1e-3));
    }

    #[test]
    fn resolution_round_trip_is_near_identity_for_flat_runtime() {
        // A runtime EQ up-sampled onto a fine layout and resolved back must
        // reproduce the original bands closely (energy is preserved).
        let runtime = BandGains::new([0.3, 0.6, 0.9]);
        let fine = MultibandLayout::log_spaced(20.0, 20_000.0, 32);
        let up = MultibandGains::from_band_gains(fine, &runtime);
        let back = up.to_band_gains();
        assert!(approx(back.low(), runtime.low(), 0.05));
        assert!(approx(back.mid(), runtime.mid(), 0.05));
        assert!(approx(back.high(), runtime.high(), 0.05));
    }

    #[test]
    fn to_band_gains_preserves_energy_of_a_notch() {
        // A deep notch confined to the mid band must darken the runtime mid
        // band while leaving low and high essentially untouched.
        let layout = MultibandLayout::from_edges(20.0, 20_000.0, &[800.0, 8000.0]);
        let g = MultibandGains::from_bands(layout, &[1.0, 0.1, 1.0]);
        let runtime = g.to_band_gains();
        assert!(runtime.low() > 0.9);
        assert!(runtime.high() > 0.9);
        assert!(runtime.mid() < 0.2);
    }

    #[test]
    fn reflection_from_absorption_tracks_energy_balance() {
        let layout = MultibandLayout::log_spaced(63.0, 8000.0, 8);
        let mirror = MaterialAbsorption::new([0.0; OCTAVE_BAND_COUNT]);
        assert!(MultibandGains::reflection_from_absorption(layout.clone(), &mirror).is_full_band(1e-6));
        let sink = MaterialAbsorption::new([1.0; OCTAVE_BAND_COUNT]);
        assert!(MultibandGains::reflection_from_absorption(layout, &sink).is_silent(1e-6));
    }

    #[test]
    fn carpet_reflection_is_darker_in_the_highs() {
        let layout = MultibandLayout::log_spaced(63.0, 8000.0, 8);
        let carpet = Material::Carpet.absorption();
        let g = MultibandGains::reflection_from_absorption(layout, &carpet);
        assert!(g.band(0) > g.band(g.band_count() - 1));
    }

    #[test]
    fn resample_onto_preserves_a_flat_spectrum() {
        let coarse = MultibandLayout::log_spaced(20.0, 20_000.0, 3);
        let fine = MultibandLayout::log_spaced(20.0, 20_000.0, 20);
        let flat = MultibandGains::uniform(coarse, 0.42);
        let up = flat.resample_onto(&fine);
        assert_eq!(up.band_count(), 20);
        for &g in up.bands() {
            assert!(approx(g, 0.42, 1e-6));
        }
    }

    #[test]
    fn sanitise_bounds_handles_non_finite_and_equal() {
        let layout = MultibandLayout::log_spaced(Sample::NAN, Sample::INFINITY, 4);
        assert!(layout.low_hz() >= MIN_BAND_FREQUENCY_HZ);
        assert!(layout.high_hz() <= MAX_BAND_FREQUENCY_HZ);
        assert!(layout.low_hz() < layout.high_hz());
        let equal = MultibandLayout::log_spaced(1000.0, 1000.0, 2);
        assert!(equal.low_hz() < equal.high_hz());
    }
}
