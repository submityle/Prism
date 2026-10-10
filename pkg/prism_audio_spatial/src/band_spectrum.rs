//! Compact three-band spectral gain for real-time propagation paths.
//!
//! A geometric-acoustics backend colours each arrival with a frequency
//! response: a reflection off carpet is dull, air absorption rolls off highs
//! with distance, and an edge diffraction behaves like a low-pass. Carrying a
//! full octave-band spectrum (see [`MaterialAbsorption`]) per arrival into the
//! audio thread is wasteful; carrying a single low-pass corner (the historical
//! [`PropagationPath::cutoff_hz`]) throws away the shape. This module defines
//! the middle ground every shipping engine converges on: a **three-band EQ**
//! (low / mid / high) the real-time voice renders with a two-crossover
//! filterbank.
//!
//! # Band layout
//!
//! The two crossovers in [`PROPAGATION_BAND_EDGES`] (`800 Hz` and `8 kHz`)
//! split the audible range into three contiguous bands. These are the same
//! crossovers Valve's Steam Audio uses for its runtime material and
//! air-absorption EQ, chosen because `800 Hz` roughly separates the
//! omnidirectional low end from the directional mid, and `8 kHz` isolates the
//! "air" band most sensitive to absorption and absorptive materials.
//!
//! | Band | Range              | Representative centre            |
//! |------|--------------------|----------------------------------|
//! | low  | `[20, 800) Hz`     | [`PROPAGATION_BAND_CENTERS`]`[0]` |
//! | mid  | `[800, 8000) Hz`   | [`PROPAGATION_BAND_CENTERS`]`[1]` |
//! | high | `[8000, 20000) Hz` | [`PROPAGATION_BAND_CENTERS`]`[2]` |
//!
//! The representative centres are the geometric means of each band's nominal
//! bounds, so a value sampled there is the log-frequency midpoint of the band.
//!
//! # Relationship
//!
//! [`BandGains`] is produced at control rate by the geometry backends in
//! `prism_audio_geometry` (direct / reflection / diffraction) and the GPU
//! backend in `prism_audio_geometry_gpu`, then consumed by the real-time voice.
//! It converts both ways with the existing acoustics vocabulary:
//! [`BandGains::from_lowpass_cutoff`] maps a legacy
//! [`PropagationPath::cutoff_hz`] low-pass colour onto three bands, and
//! [`BandGains::reflection_from_absorption`] turns an octave-band
//! [`MaterialAbsorption`] into a per-band amplitude reflection coefficient. The
//! single-number collapse [`BandGains::at`] mirrors [`MaterialAbsorption::at`]
//! so a scalar consumer can still read one frequency.
//!
//! [`MaterialAbsorption`]: crate::material_library::MaterialAbsorption
//! [`MaterialAbsorption::at`]: crate::material_library::MaterialAbsorption::at
//! [`PropagationPath::cutoff_hz`]: crate::propagation::PropagationPath

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::material_library::{MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// Number of frequency bands in a real-time propagation spectrum.
pub const PROPAGATION_BAND_COUNT: usize = 3;

/// Crossover frequencies (Hz) separating the low/mid and mid/high bands.
///
/// These two corners partition the audible range into the three contiguous
/// bands documented on the module. They match Steam Audio's runtime EQ
/// crossovers.
pub const PROPAGATION_BAND_EDGES: [Sample; PROPAGATION_BAND_COUNT - 1] = [800.0, 8000.0];

/// Nominal lower bound (Hz) of the low band, used only to derive its
/// representative centre.
pub const PROPAGATION_BAND_LOW_HZ: Sample = 20.0;

/// Nominal upper bound (Hz) of the high band, used only to derive its
/// representative centre.
pub const PROPAGATION_BAND_HIGH_HZ: Sample = 20_000.0;

/// Representative centre frequency (Hz) of each band: the geometric mean of its
/// nominal bounds (`[20, 800]`, `[800, 8000]`, `[8000, 20000]`).
///
/// Sampling a filter response here evaluates it at the log-frequency midpoint
/// of the band. The literals equal the runtime geometric means to within
/// floating-point tolerance (asserted by the module tests).
pub const PROPAGATION_BAND_CENTERS: [Sample; PROPAGATION_BAND_COUNT] =
    [126.491_1, 2529.822, 12649.111];

/// A per-band linear amplitude gain in `[0, 1]`, aligned with
/// [`PROPAGATION_BAND_CENTERS`].
///
/// `1.0` in a band passes it untouched; `0.0` silences it. The flat
/// [`UNITY`](Self::UNITY) spectrum means "no colour", matching the historical
/// full-band path. Values are clamped into `[0, 1]` on construction, so a
/// [`BandGains`] is always a physically valid attenuation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BandGains {
    bands: [Sample; PROPAGATION_BAND_COUNT],
}

impl BandGains {
    /// A flat, full-band spectrum: every band passes untouched.
    pub const UNITY: Self = Self {
        bands: [1.0; PROPAGATION_BAND_COUNT],
    };

    /// A fully attenuated spectrum: every band silenced.
    pub const SILENT: Self = Self {
        bands: [0.0; PROPAGATION_BAND_COUNT],
    };

    /// Builds a spectrum from three per-band gains, clamping each into the
    /// physical `[0, 1]` amplitude range. A non-finite input becomes `0`.
    #[must_use]
    pub fn new(bands: [Sample; PROPAGATION_BAND_COUNT]) -> Self {
        let mut clamped = bands;
        for b in &mut clamped {
            *b = if b.is_finite() {
                b.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        Self { bands: clamped }
    }

    /// Builds a flat spectrum with the same gain in every band.
    #[must_use]
    pub fn uniform(gain: Sample) -> Self {
        Self::new([gain; PROPAGATION_BAND_COUNT])
    }

    /// Returns the raw per-band gains aligned with
    /// [`PROPAGATION_BAND_CENTERS`].
    #[inline]
    #[must_use]
    pub fn bands(&self) -> [Sample; PROPAGATION_BAND_COUNT] {
        self.bands
    }

    /// Returns the gain of band `index`, saturating to the last band for an
    /// out-of-range index (never panics).
    #[inline]
    #[must_use]
    pub fn band(&self, index: usize) -> Sample {
        self.bands[index.min(PROPAGATION_BAND_COUNT - 1)]
    }

    /// The low-band gain (below the first crossover).
    #[inline]
    #[must_use]
    pub fn low(&self) -> Sample {
        self.bands[0]
    }

    /// The mid-band gain (between the two crossovers).
    #[inline]
    #[must_use]
    pub fn mid(&self) -> Sample {
        self.bands[1]
    }

    /// The high-band gain (above the second crossover).
    #[inline]
    #[must_use]
    pub fn high(&self) -> Sample {
        self.bands[PROPAGATION_BAND_COUNT - 1]
    }

    /// Returns `true` when every band is within `tolerance` of unity, i.e. the
    /// spectrum applies no audible colour and a filterbank can be bypassed.
    #[must_use]
    pub fn is_full_band(&self, tolerance: Sample) -> bool {
        let tol = tolerance.max(0.0);
        self.bands.iter().all(|&g| ops::abs(g - 1.0) <= tol)
    }

    /// Returns `true` when every band is at or below `tolerance`, i.e. the
    /// arrival is effectively silent and can be dropped.
    #[must_use]
    pub fn is_silent(&self, tolerance: Sample) -> bool {
        let tol = tolerance.max(0.0);
        self.bands.iter().all(|&g| g <= tol)
    }

    /// Combines two spectra by multiplying each band, modelling a signal that
    /// passes through both colourations in series (e.g. a reflection followed
    /// by air absorption). The product of two `[0, 1]` gains stays in `[0, 1]`.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        let mut out = self.bands;
        for (o, &b) in out.iter_mut().zip(other.bands.iter()) {
            *o *= b;
        }
        Self { bands: out }
    }

    /// Scales every band by a scalar broadband gain, re-clamping into `[0, 1]`.
    #[must_use]
    pub fn scaled(self, gain: Sample) -> Self {
        Self::new([
            self.bands[0] * gain,
            self.bands[1] * gain,
            self.bands[2] * gain,
        ])
    }

    /// The root-mean-square of the three band gains: a single broadband
    /// amplitude that preserves total energy when the spectrum is collapsed to
    /// one number. Always in `[0, 1]`.
    #[must_use]
    pub fn broadband_rms(&self) -> Sample {
        let mut sum_sq = 0.0;
        for &g in &self.bands {
            sum_sq += g * g;
        }
        ops::sqrt(sum_sq / PROPAGATION_BAND_COUNT as Sample)
    }

    /// The largest band gain: the brightest frequency the arrival still passes.
    #[must_use]
    pub fn peak(&self) -> Sample {
        self.bands
            .iter()
            .copied()
            .fold(0.0, |acc, g| if g > acc { g } else { acc })
    }

    /// Factors the spectrum into a broadband scalar gain and a normalised
    /// colour such that `scalar * colour` reproduces `self` exactly.
    ///
    /// The scalar is the [`peak`](Self::peak) band gain and the colour is the
    /// spectrum divided by that peak, so every band of the returned colour lies
    /// in `[0, 1]` with at least one band at unity. This is the inverse of
    /// [`scaled`](Self::scaled): `colour.scaled(scalar)` recovers `self` with no
    /// clamping loss, because dividing by the peak can never push a band above
    /// one. It is the canonical way to hand a coloured arrival to a consumer
    /// that stores a flat broadband gain alongside a relative
    /// [`BandGains`](Self) colour, matching the
    /// [`PropagationPath`](crate::propagation::PropagationPath) split between
    /// `gain` and `bands`.
    ///
    /// A fully silent spectrum has no colour to recover, so it factors into a
    /// zero scalar and the [`SILENT`](Self::SILENT) colour.
    #[must_use]
    pub fn split_peak(self) -> (Sample, Self) {
        let peak = self.peak();
        if peak <= 0.0 {
            (0.0, Self::SILENT)
        } else {
            (peak, self.scaled(1.0 / peak))
        }
    }

    /// The spectrum gain at an arbitrary frequency, interpolated linearly in
    /// `log(frequency)` between the two bracketing band centres.
    ///
    /// Frequencies at or below the low centre return the low band; frequencies
    /// at or above the high centre return the high band (the spectrum is
    /// clamped, never extrapolated). A non-finite frequency falls back to the
    /// low band. This mirrors
    /// [`MaterialAbsorption::at`](crate::material_library::MaterialAbsorption::at)
    /// so a scalar consumer can collapse the spectrum at one frequency.
    #[must_use]
    pub fn at(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= PROPAGATION_BAND_CENTERS[0] {
            return self.bands[0];
        }
        let last = PROPAGATION_BAND_COUNT - 1;
        if freq_hz >= PROPAGATION_BAND_CENTERS[last] {
            return self.bands[last];
        }
        let log_f = ops::ln(freq_hz);
        for i in 0..last {
            let c_hi = PROPAGATION_BAND_CENTERS[i + 1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(PROPAGATION_BAND_CENTERS[i]);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > 0.0 {
                    (log_f - log_lo) / span
                } else {
                    0.0
                };
                let v = self.bands[i] + (self.bands[i + 1] - self.bands[i]) * t;
                return v.clamp(0.0, 1.0);
            }
        }
        self.bands[last]
    }

    /// Approximates a first-order (6 dB/octave) low-pass with corner `cutoff_hz`
    /// as a three-band spectrum, sampling the analogue magnitude response
    /// `1 / sqrt(1 + (f / fc)^2)` at each band centre.
    ///
    /// This converts the legacy single-corner
    /// [`PropagationPath::cutoff_hz`](crate::propagation::PropagationPath::cutoff_hz)
    /// colour (used by diffraction and air absorption) into the band
    /// representation without changing the perceived roll-off. A very large
    /// corner (such as
    /// [`FULL_BAND_CUTOFF_HZ`](crate::propagation::FULL_BAND_CUTOFF_HZ))
    /// yields an essentially flat [`UNITY`](Self::UNITY) spectrum.
    #[must_use]
    pub fn from_lowpass_cutoff(cutoff_hz: Sample) -> Self {
        if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 {
            return Self::SILENT;
        }
        let mut bands = [0.0; PROPAGATION_BAND_COUNT];
        for (b, &centre) in bands.iter_mut().zip(PROPAGATION_BAND_CENTERS.iter()) {
            let ratio = centre / cutoff_hz;
            *b = 1.0 / ops::sqrt(1.0 + ratio * ratio);
        }
        Self::new(bands)
    }

    /// Converts an octave-band [`MaterialAbsorption`] into the per-band
    /// amplitude reflection coefficient of a surface made of that material.
    ///
    /// Each band averages the octave-band energy-absorption coefficients whose
    /// ISO centre falls inside the band's range, then converts reflected energy
    /// `1 - alpha` into an amplitude gain `sqrt(1 - alpha)`. The result is the
    /// frequency-dependent sibling of
    /// [`AcousticMaterial::reflection_gain`](crate::propagation::AcousticMaterial::reflection_gain).
    ///
    /// [`MaterialAbsorption`]: crate::material_library::MaterialAbsorption
    #[must_use]
    pub fn reflection_from_absorption(absorption: &MaterialAbsorption) -> Self {
        let alpha = band_average_absorption(absorption);
        let mut bands = [0.0; PROPAGATION_BAND_COUNT];
        for (b, &a) in bands.iter_mut().zip(alpha.iter()) {
            *b = ops::sqrt((1.0 - a).max(0.0));
        }
        Self::new(bands)
    }
}

impl Default for BandGains {
    #[inline]
    fn default() -> Self {
        Self::UNITY
    }
}

/// Averages the octave-band absorption coefficients into the three propagation
/// bands by partitioning the ISO octave centres against
/// [`PROPAGATION_BAND_EDGES`].
///
/// A band with no octave centre inside it (which the standard
/// `[63, 8000] Hz` grid never produces for the documented edges) falls back to
/// the nearest populated band so the result is always defined.
fn band_average_absorption(absorption: &MaterialAbsorption) -> [Sample; PROPAGATION_BAND_COUNT] {
    let octaves = absorption.bands();
    let mut sums = [0.0; PROPAGATION_BAND_COUNT];
    let mut counts = [0u32; PROPAGATION_BAND_COUNT];
    for i in 0..OCTAVE_BAND_COUNT {
        let centre = OCTAVE_BAND_CENTERS[i];
        let band = if centre < PROPAGATION_BAND_EDGES[0] {
            0
        } else if centre < PROPAGATION_BAND_EDGES[1] {
            1
        } else {
            2
        };
        sums[band] += octaves[i];
        counts[band] += 1;
    }
    let mut out = [0.0; PROPAGATION_BAND_COUNT];
    let mut last_defined = 0.0;
    for b in 0..PROPAGATION_BAND_COUNT {
        if counts[b] > 0 {
            out[b] = (sums[b] / counts[b] as Sample).clamp(0.0, 1.0);
            last_defined = out[b];
        } else {
            out[b] = last_defined;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material_library::Material;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        ops::abs(a - b) <= tol
    }

    #[test]
    fn centres_are_geometric_means_of_band_bounds() {
        let lo = ops::sqrt(PROPAGATION_BAND_LOW_HZ * PROPAGATION_BAND_EDGES[0]);
        let mid = ops::sqrt(PROPAGATION_BAND_EDGES[0] * PROPAGATION_BAND_EDGES[1]);
        let hi = ops::sqrt(PROPAGATION_BAND_EDGES[1] * PROPAGATION_BAND_HIGH_HZ);
        assert!(approx(PROPAGATION_BAND_CENTERS[0], lo, 0.1));
        assert!(approx(PROPAGATION_BAND_CENTERS[1], mid, 0.1));
        assert!(approx(PROPAGATION_BAND_CENTERS[2], hi, 0.1));
    }

    #[test]
    fn unity_and_silent_sentinels() {
        assert_eq!(BandGains::UNITY.bands(), [1.0, 1.0, 1.0]);
        assert_eq!(BandGains::SILENT.bands(), [0.0, 0.0, 0.0]);
        assert!(BandGains::UNITY.is_full_band(1e-6));
        assert!(BandGains::SILENT.is_silent(1e-6));
        assert!(!BandGains::UNITY.is_silent(1e-6));
    }

    #[test]
    fn new_clamps_into_unit_range() {
        let g = BandGains::new([1.5, -0.2, Sample::NAN]);
        assert_eq!(g.bands(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn accessors_match_bands() {
        let g = BandGains::new([0.2, 0.5, 0.8]);
        assert!(approx(g.low(), 0.2, 1e-6));
        assert!(approx(g.mid(), 0.5, 1e-6));
        assert!(approx(g.high(), 0.8, 1e-6));
        assert!(approx(g.band(0), 0.2, 1e-6));
        assert!(approx(g.band(99), 0.8, 1e-6));
    }

    #[test]
    fn combine_is_elementwise_series_combination() {
        let a = BandGains::new([0.5, 0.4, 0.25]);
        let b = BandGains::new([0.5, 0.5, 0.8]);
        let c = a.combine(b);
        assert!(approx(c.low(), 0.25, 1e-6));
        assert!(approx(c.mid(), 0.2, 1e-6));
        assert!(approx(c.high(), 0.2, 1e-6));
    }

    #[test]
    fn scaled_reclamps() {
        let g = BandGains::new([0.6, 0.6, 0.6]).scaled(2.0);
        assert_eq!(g.bands(), [1.0, 1.0, 1.0]);
        let h = BandGains::new([0.6, 0.4, 0.2]).scaled(0.5);
        assert!(approx(h.low(), 0.3, 1e-6));
        assert!(approx(h.mid(), 0.2, 1e-6));
        assert!(approx(h.high(), 0.1, 1e-6));
    }

    #[test]
    fn broadband_rms_and_peak() {
        let g = BandGains::new([0.0, 0.0, 1.0]);
        assert!(approx(g.broadband_rms(), ops::sqrt(1.0 / 3.0), 1e-6));
        assert!(approx(g.peak(), 1.0, 1e-6));
        assert!(approx(BandGains::UNITY.broadband_rms(), 1.0, 1e-6));
    }

    #[test]
    fn at_clamps_and_interpolates_in_log_frequency() {
        let g = BandGains::new([0.2, 0.6, 1.0]);
        // Below the low centre clamps to the low band.
        assert!(approx(g.at(20.0), 0.2, 1e-6));
        // Above the high centre clamps to the high band.
        assert!(approx(g.at(20_000.0), 1.0, 1e-6));
        // Exactly at a centre returns that band.
        assert!(approx(g.at(PROPAGATION_BAND_CENTERS[1]), 0.6, 1e-6));
        // Geometric midpoint between low and mid centres is the average.
        let mid_point = ops::sqrt(PROPAGATION_BAND_CENTERS[0] * PROPAGATION_BAND_CENTERS[1]);
        assert!(approx(g.at(mid_point), 0.4, 1e-4));
    }

    #[test]
    fn full_band_lowpass_is_essentially_unity() {
        let g = BandGains::from_lowpass_cutoff(crate::propagation::FULL_BAND_CUTOFF_HZ);
        assert!(g.is_full_band(1e-3));
    }

    #[test]
    fn lowpass_rolls_off_above_the_corner() {
        // Corner at the mid centre: mid band sits at -3 dB (gain 1/sqrt(2)),
        // the low band is brighter, the high band darker.
        let g = BandGains::from_lowpass_cutoff(PROPAGATION_BAND_CENTERS[1]);
        assert!(approx(g.mid(), 1.0 / ops::sqrt(2.0), 1e-4));
        assert!(g.low() > g.mid());
        assert!(g.high() < g.mid());
    }

    #[test]
    fn zero_or_negative_cutoff_is_silent() {
        assert!(BandGains::from_lowpass_cutoff(0.0).is_silent(1e-6));
        assert!(BandGains::from_lowpass_cutoff(-5.0).is_silent(1e-6));
    }

    #[test]
    fn reflection_from_absorption_tracks_energy_balance() {
        // A perfect mirror (zero absorption) reflects unity amplitude.
        let mirror = MaterialAbsorption::new([0.0; OCTAVE_BAND_COUNT]);
        let g = BandGains::reflection_from_absorption(&mirror);
        assert!(g.is_full_band(1e-6));

        // A perfect absorber reflects nothing.
        let sink = MaterialAbsorption::new([1.0; OCTAVE_BAND_COUNT]);
        assert!(BandGains::reflection_from_absorption(&sink).is_silent(1e-6));
    }

    #[test]
    fn carpet_reflection_is_darker_in_the_high_band() {
        // Carpet absorbs strongly at high frequencies, so its high-band
        // reflection gain must be well below its low-band gain.
        let carpet = Material::Carpet.absorption();
        let g = BandGains::reflection_from_absorption(&carpet);
        assert!(g.low() > g.high());
        assert!(g.high() < 0.7);
    }

    #[test]
    fn reflection_band_average_matches_manual_octave_grouping() {
        let carpet = Material::Carpet.absorption();
        let octaves = carpet.bands();
        // Low band averages the first four octave centres (63..500 Hz).
        let low_alpha = (octaves[0] + octaves[1] + octaves[2] + octaves[3]) / 4.0;
        let expected_low = ops::sqrt((1.0 - low_alpha).max(0.0));
        let g = BandGains::reflection_from_absorption(&carpet);
        assert!(approx(g.low(), expected_low, 1e-5));
    }

    #[test]
    fn split_peak_is_the_inverse_of_scaled() {
        // A coloured spectrum factors into its peak and a normalised colour
        // whose brightest band is unity, and multiplying them back is lossless.
        let g = BandGains::new([0.3, 0.6, 0.15]);
        let (scalar, colour) = g.split_peak();
        assert!(approx(scalar, 0.6, 1e-6));
        assert!(approx(colour.peak(), 1.0, 1e-6));
        let recovered = colour.scaled(scalar);
        assert!(approx(recovered.low(), g.low(), 1e-6));
        assert!(approx(recovered.mid(), g.mid(), 1e-6));
        assert!(approx(recovered.high(), g.high(), 1e-6));
    }

    #[test]
    fn split_peak_of_uniform_is_flat_colour() {
        // A frequency-flat spectrum keeps all its energy in the scalar and
        // yields a unity colour, so a uniform material is unchanged.
        let (scalar, colour) = BandGains::uniform(0.42).split_peak();
        assert!(approx(scalar, 0.42, 1e-6));
        assert!(colour.is_full_band(1e-6));
    }

    #[test]
    fn split_peak_of_silence_has_no_colour() {
        let (scalar, colour) = BandGains::SILENT.split_peak();
        assert!(approx(scalar, 0.0, 1e-6));
        assert!(colour.is_silent(1e-6));
    }
}
