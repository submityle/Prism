//! Frequency-dependent source radiation directivity.
//!
//! Real acoustic sources (voices, instruments, loudspeakers) do not radiate
//! uniformly: they grow more directional with frequency, beaming high
//! frequencies forward while remaining nearly omnidirectional at low
//! frequencies. This module is a control-rate description of that radiation
//! pattern; it performs no per-sample DSP. It produces a scalar radiation gain
//! for a given off-axis angle and frequency, plus the classic directivity
//! factor `Q` and directivity index `DI` per octave band.
//!
//! # Weighted first-order model
//!
//! For an angle `theta` between the source forward axis and the listener
//! direction, the per-band radiation gain is the weighted first-order pattern
//!
//! `d(band, cos_theta) = (1 - s_b) + s_b * cos_theta`
//!
//! clamped to be non-negative (rear-hemisphere cut-off). Here `s_b` in `[0, 1]`
//! is the per-octave-band directivity sharpness: `s = 0` is a fully
//! omnidirectional source, `s = 1` is a pure cardioid. On axis (`cos_theta = 1`)
//! every band has unit gain.
//!
//! # Directivity factor and index
//!
//! Averaging `d^2` over the sphere, and using `<cos> = 0` and `<cos^2> = 1/3`,
//!
//! `<d^2> = (1 - s)^2 + s^2 / 3`
//!
//! With the on-axis value `d(0) = 1`, the directivity factor is
//!
//! `Q_b = 1 / ((1 - s_b)^2 + s_b^2 / 3)`
//!
//! so `s = 0` gives `Q = 1` (`DI = 0 dB`) and `s = 1` gives `Q = 3`
//! (`DI` approximately `4.77 dB`), matching the known cardioid directivity
//! index. The directivity index is `DI_b = 10 * log10(Q_b)`.
//!
//! # Control rate, not audio rate
//!
//! All queries operate on stack scalars and fixed eight-element arrays; there
//! is no heap allocation, no locking, and no panicking. Out-of-range band
//! indices and frequencies clamp to the end bands, and non-finite inputs return
//! safe finite values. All transcendental math routes through
//! [`bevy_math::ops`], never through `f32` intrinsics.
//!
//! # Provenance
//!
//! This is the textbook electro-acoustics description of source directivity:
//! the weighted first-order (omni-to-cardioid) radiation pattern and the
//! directivity factor / index relations as presented in L. Beranek's
//! *Acoustics* and H. Olson's *Acoustical Engineering*. This module is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics knowledge.

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// The natural logarithm of ten, used to convert `ln` into `log10`.
const LN_10: Sample = core::f32::consts::LN_10;

/// Smallest denominator used to keep the directivity factor finite.
const MIN_DIVISOR: Sample = 1e-9;

/// A named per-octave-band directivity sharpness preset.
///
/// Each preset maps to an eight-element sharpness spectrum aligned with
/// [`OCTAVE_BAND_CENTERS`], where `0` is omnidirectional and `1` is a pure
/// cardioid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DirectivityPreset {
    /// Omnidirectional: uniform radiation at all angles and frequencies.
    Omni,
    /// Pure cardioid at every band: strong forward bias, rear null.
    Cardioid,
    /// A human voice: nearly omnidirectional low frequencies, moderately
    /// directional highs.
    Voice,
    /// A trumpet or other horn: sharply beamed highs, more directional than a
    /// voice at every band.
    Trumpet,
}

impl DirectivityPreset {
    /// The per-octave-band sharpness spectrum for this preset.
    ///
    /// Every returned value lies in `[0, 1]` and, for the frequency-dependent
    /// presets, is monotonically non-decreasing with frequency (real sources
    /// grow more directional with frequency).
    #[must_use]
    pub fn sharpness(self) -> [Sample; OCTAVE_BAND_COUNT] {
        match self {
            // 63  125  250  500  1k   2k   4k   8k
            DirectivityPreset::Omni => [0.0; OCTAVE_BAND_COUNT],
            DirectivityPreset::Cardioid => [1.0; OCTAVE_BAND_COUNT],
            DirectivityPreset::Voice => {
                [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85]
            }
            DirectivityPreset::Trumpet => {
                [0.10, 0.20, 0.35, 0.55, 0.70, 0.85, 0.95, 1.00]
            }
        }
    }
}

/// A frequency-dependent source radiation directivity described by a per-octave
/// -band sharpness spectrum.
///
/// Build one with [`SourceDirectivity::from_preset`] or
/// [`SourceDirectivity::from_sharpness`], then query the radiation gain for an
/// off-axis cosine and frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceDirectivity {
    /// Per-octave-band directivity sharpness in `[0, 1]`, aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    sharpness: [Sample; OCTAVE_BAND_COUNT],
}

impl SourceDirectivity {
    /// Builds a directivity from an explicit per-band sharpness spectrum.
    ///
    /// Each value is clamped to `[0, 1]`; non-finite values become `0`
    /// (omnidirectional).
    #[must_use]
    pub fn from_sharpness(sharpness: [Sample; OCTAVE_BAND_COUNT]) -> Self {
        let mut clamped = [0.0; OCTAVE_BAND_COUNT];
        for (out, &s) in clamped.iter_mut().zip(sharpness.iter()) {
            *out = if s.is_finite() { s.clamp(0.0, 1.0) } else { 0.0 };
        }
        Self { sharpness: clamped }
    }

    /// Builds a directivity from a named [`DirectivityPreset`].
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::source_directivity::{DirectivityPreset, SourceDirectivity};
    ///
    /// // A cardioid source radiates fully forward and is silent to the rear.
    /// let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
    /// assert!((d.gain_at(0, 1.0) - 1.0).abs() < 1e-6); // on axis
    /// assert!(d.gain_at(0, -1.0).abs() < 1e-6); // rear null
    /// ```
    #[must_use]
    pub fn from_preset(preset: DirectivityPreset) -> Self {
        Self::from_sharpness(preset.sharpness())
    }

    /// The per-octave-band sharpness spectrum, aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    #[must_use]
    pub fn sharpness(&self) -> &[Sample; OCTAVE_BAND_COUNT] {
        &self.sharpness
    }

    /// The radiation gain for band `band_index` at off-axis cosine
    /// `cos_theta`.
    ///
    /// Computes `(1 - s) + s * cos_theta` clamped to be non-negative. The
    /// cosine is clamped to `[-1, 1]` for robustness; a band index past the end
    /// clamps to the highest band.
    #[must_use]
    pub fn gain_at(&self, band_index: usize, cos_theta: Sample) -> Sample {
        let idx = band_index.min(OCTAVE_BAND_COUNT - 1);
        let s = self.sharpness[idx];
        let c = clamp_cos(cos_theta);
        weighted_cardioid(s, c)
    }

    /// The radiation gain in every octave band at off-axis cosine
    /// `cos_theta`.
    #[must_use]
    pub fn band_gains(&self, cos_theta: Sample) -> [Sample; OCTAVE_BAND_COUNT] {
        let c = clamp_cos(cos_theta);
        let mut out = [0.0; OCTAVE_BAND_COUNT];
        for (g, &s) in out.iter_mut().zip(self.sharpness.iter()) {
            *g = weighted_cardioid(s, c);
        }
        out
    }

    /// The radiation gain at an arbitrary frequency and off-axis cosine.
    ///
    /// The sharpness is interpolated linearly in `log(frequency)` between the
    /// two neighbouring octave bands, then applied to the weighted cardioid
    /// pattern. Frequencies below the lowest band centre use the lowest band;
    /// frequencies above the highest use the highest.
    #[must_use]
    pub fn broadband_gain(&self, cos_theta: Sample, freq_hz: Sample) -> Sample {
        let s = self.sharpness_at(freq_hz);
        let c = clamp_cos(cos_theta);
        weighted_cardioid(s, c)
    }

    /// The directivity factor `Q = 1 / ((1 - s)^2 + s^2 / 3)` for band
    /// `band_index`.
    ///
    /// `s = 0` gives `Q = 1`; `s = 1` gives `Q = 3`. A band index past the end
    /// clamps to the highest band.
    #[must_use]
    pub fn directivity_factor(&self, band_index: usize) -> Sample {
        let idx = band_index.min(OCTAVE_BAND_COUNT - 1);
        let s = self.sharpness[idx];
        let one_minus = 1.0 - s;
        let denom = one_minus * one_minus + s * s / 3.0;
        if denom > MIN_DIVISOR { 1.0 / denom } else { 1.0 / MIN_DIVISOR }
    }

    /// The directivity index `DI = 10 * log10(Q)` in decibels for band
    /// `band_index`.
    ///
    /// `s = 0` gives `0 dB`; `s = 1` gives approximately `4.77 dB`.
    #[must_use]
    pub fn directivity_index_db(&self, band_index: usize) -> Sample {
        let q = self.directivity_factor(band_index);
        if q > MIN_DIVISOR {
            10.0 * ops::ln(q) / LN_10
        } else {
            0.0
        }
    }

    /// The sharpness interpolated in `log(frequency)` between neighbouring
    /// octave bands, clamped to the end bands outside the tabulated range.
    fn sharpness_at(&self, freq_hz: Sample) -> Sample {
        let last = OCTAVE_BAND_COUNT - 1;
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return self.sharpness[0];
        }
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return self.sharpness[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, values) in
            OCTAVE_BAND_CENTERS.windows(2).zip(self.sharpness.windows(2))
        {
            let c_lo = centres[0];
            let c_hi = centres[1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(c_lo);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > 0.0 { (log_f - log_lo) / span } else { 0.0 };
                let v = values[0] + (values[1] - values[0]) * t;
                return v.clamp(0.0, 1.0);
            }
        }
        self.sharpness[last]
    }
}

/// The weighted first-order radiation gain `(1 - s) + s * cos_theta`, clamped to
/// be non-negative.
fn weighted_cardioid(s: Sample, cos_theta: Sample) -> Sample {
    ((1.0 - s) + s * cos_theta).max(0.0)
}

/// Clamps a cosine to `[-1, 1]`, mapping non-finite input to the on-axis value.
fn clamp_cos(cos_theta: Sample) -> Sample {
    if cos_theta.is_finite() { cos_theta.clamp(-1.0, 1.0) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn omni_is_unity_at_every_angle() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Omni);
        for &cos in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            for band in 0..OCTAVE_BAND_COUNT {
                assert!(approx(d.gain_at(band, cos), 1.0, 1e-6), "cos {cos}");
            }
        }
    }

    #[test]
    fn cardioid_rear_is_null() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(approx(d.gain_at(band, -1.0), 0.0, 1e-6));
            // At s = 1 the pattern is d = cos, so the side (90 degrees) is
            // zero and the whole rear hemisphere is clamped to zero.
            assert!(approx(d.gain_at(band, 0.0), 0.0, 1e-6));
        }
    }

    #[test]
    fn on_axis_is_unity_for_all_presets() {
        for preset in [
            DirectivityPreset::Omni,
            DirectivityPreset::Cardioid,
            DirectivityPreset::Voice,
            DirectivityPreset::Trumpet,
        ] {
            let d = SourceDirectivity::from_preset(preset);
            for band in 0..OCTAVE_BAND_COUNT {
                assert!(approx(d.gain_at(band, 1.0), 1.0, 1e-6));
            }
        }
    }

    #[test]
    fn higher_sharpness_attenuates_the_side_more() {
        let voice = SourceDirectivity::from_preset(DirectivityPreset::Voice);
        let trumpet = SourceDirectivity::from_preset(DirectivityPreset::Trumpet);
        // At the top band the trumpet is sharper, so its side gain is lower.
        let last = OCTAVE_BAND_COUNT - 1;
        assert!(trumpet.gain_at(last, 0.0) <= voice.gain_at(last, 0.0));
    }

    #[test]
    fn presets_grow_more_directional_with_frequency() {
        for preset in [DirectivityPreset::Voice, DirectivityPreset::Trumpet] {
            let s = preset.sharpness();
            for w in s.windows(2) {
                assert!(w[1] >= w[0], "sharpness must be non-decreasing");
            }
            // Trumpet is at least as sharp as voice everywhere.
        }
        let voice = DirectivityPreset::Voice.sharpness();
        let trumpet = DirectivityPreset::Trumpet.sharpness();
        for (v, t) in voice.iter().zip(trumpet.iter()) {
            assert!(t >= v);
        }
    }

    #[test]
    fn directivity_factor_matches_closed_form() {
        let omni = SourceDirectivity::from_preset(DirectivityPreset::Omni);
        let cardioid = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(approx(omni.directivity_factor(band), 1.0, 1e-4));
            assert!(approx(cardioid.directivity_factor(band), 3.0, 1e-4));
        }
    }

    #[test]
    fn directivity_index_matches_known_values() {
        let omni = SourceDirectivity::from_preset(DirectivityPreset::Omni);
        let cardioid = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        assert!(approx(omni.directivity_index_db(0), 0.0, 1e-4));
        // 10 * log10(3) approximately 4.771 dB.
        assert!(approx(cardioid.directivity_index_db(0), 4.771, 1e-2));
    }

    #[test]
    fn broadband_matches_band_at_centre() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Voice);
        for (band, &centre) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let broad = d.broadband_gain(0.0, centre);
            let exact = d.gain_at(band, 0.0);
            assert!(approx(broad, exact, 1e-4), "band {band}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Voice);
        // A frequency between 1000 and 2000 Hz has sharpness between the two.
        let s = d.sharpness();
        let g = d.broadband_gain(0.0, 1414.0);
        let lo = ((1.0 - s[4]) + s[4] * 0.0).max(0.0);
        let hi = ((1.0 - s[5]) + s[5] * 0.0).max(0.0);
        let (top, bottom) = if lo >= hi { (lo, hi) } else { (hi, lo) };
        assert!(g <= top + 1e-4 && g >= bottom - 1e-4);
    }

    #[test]
    fn out_of_range_band_clamps_to_last() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Trumpet);
        let last = OCTAVE_BAND_COUNT - 1;
        assert!(approx(d.gain_at(999, 0.0), d.gain_at(last, 0.0), 1e-6));
        assert!(approx(
            d.directivity_factor(999),
            d.directivity_factor(last),
            1e-6
        ));
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let d = SourceDirectivity::from_preset(DirectivityPreset::Cardioid);
        // Non-finite cosine treated as on-axis.
        assert!(approx(d.gain_at(0, Sample::NAN), 1.0, 1e-6));
        // Non-finite frequency clamps to the lowest band.
        let g = d.broadband_gain(1.0, Sample::INFINITY);
        assert!(g.is_finite());
        // Non-finite sharpness input becomes omnidirectional.
        let weird = SourceDirectivity::from_sharpness([Sample::NAN; OCTAVE_BAND_COUNT]);
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(approx(weird.gain_at(band, -1.0), 1.0, 1e-6));
        }
    }

    #[test]
    fn from_sharpness_clamps_range() {
        let d = SourceDirectivity::from_sharpness([2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let s = d.sharpness();
        assert!(approx(s[0], 1.0, 1e-6));
        assert!(approx(s[1], 0.0, 1e-6));
        assert!(approx(s[2], 0.5, 1e-6));
    }
}
