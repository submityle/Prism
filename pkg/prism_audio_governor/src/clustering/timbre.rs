//! Compact timbre descriptor used to decide which voices sound alike enough to
//! merge into one representative source.
//!
//! Clustering purely by position would fuse a scream and a footstep that happen
//! to share a corner of the room; perceptually that is wrong. A small,
//! deterministic timbre fingerprint lets the assignment stage keep spectrally
//! dissimilar voices apart even when they are spatially adjacent. The
//! descriptor is derived from a per-band energy spectrum
//! ([`crate::masking::masking_model::VoiceSpectrum`]) and the band partition
//! that produced it, so it reuses the same critical-band analysis the masking
//! model already needs.
//!
//! Three scalars capture the gross spectral shape:
//!
//! - `brightness` -- the energy-weighted spectral centroid, mapped onto a
//!   logarithmic frequency axis and normalised to `[0, 1]`.
//! - `width` -- the energy-weighted spectral spread about that centroid, also
//!   normalised to `[0, 1]`.
//! - `flatness` -- the ratio of geometric to arithmetic mean band energy, a
//!   `[0, 1]` tonal-versus-noisy indicator.
//!
//! Distance in this three-dimensional space is a plain Euclidean metric;
//! similarity is a bounded, monotonically decreasing function of it. Both are
//! deterministic functions of the input energies.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Supports the source-clustering half of design section 33. Consumes the
//! critical-band spectra of [`crate::masking`].

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::masking::critical_bands::CriticalBands;
use crate::masking::masking_model::VoiceSpectrum;

/// Lowest frequency on the normalisation axis, in hertz.
pub const MIN_HZ: Sample = 20.0;

/// Highest frequency on the normalisation axis, in hertz.
pub const MAX_HZ: Sample = 20_000.0;

/// A three-scalar fingerprint of a voice's gross spectral shape.
///
/// All three fields live in `[0, 1]`; a zero/silent spectrum yields the neutral
/// descriptor returned by [`Timbre::neutral`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Timbre {
    /// Normalised log-frequency spectral centroid (`0` = low, `1` = high).
    pub brightness: Sample,
    /// Normalised spectral spread about the centroid (`0` = narrow, `1` = wide).
    pub width: Sample,
    /// Spectral flatness (`0` = tonal, `1` = noise-like).
    pub flatness: Sample,
}

impl Default for Timbre {
    fn default() -> Self {
        Self::neutral()
    }
}

impl Timbre {
    /// The neutral descriptor used for silent voices: mid brightness, zero
    /// width, fully tonal. Chosen so a silent voice is maximally distinct from
    /// nothing in particular and never dominates a cluster centroid.
    #[must_use]
    pub const fn neutral() -> Self {
        Self { brightness: 0.5, width: 0.0, flatness: 0.0 }
    }

    /// Derives a descriptor from a per-band energy spectrum and the band
    /// partition that produced it.
    ///
    /// Returns [`Timbre::neutral`] when the spectrum carries no finite positive
    /// energy or when the band partition is empty.
    #[must_use]
    pub fn from_spectrum(spectrum: &VoiceSpectrum, bands: &CriticalBands) -> Self {
        let band_count = bands.band_count();
        if band_count == 0 {
            return Self::neutral();
        }
        let usable = spectrum.bands.len().min(band_count);
        if usable == 0 {
            return Self::neutral();
        }

        let log_min = ops::log10(MIN_HZ);
        let log_span = (ops::log10(MAX_HZ) - log_min).max(Sample::EPSILON);

        // First pass: energy-weighted log-frequency centroid, total and peak.
        let mut total = 0.0;
        let mut weighted_log = 0.0;
        let mut peak = 0.0;
        for (i, &raw) in spectrum.bands.iter().take(usable).enumerate() {
            let e = if raw.is_finite() && raw > 0.0 { raw } else { 0.0 };
            if e <= 0.0 {
                continue;
            }
            let center = bands.band_center(i).max(MIN_HZ);
            total += e;
            weighted_log += e * ops::log10(center);
            if e > peak {
                peak = e;
            }
        }
        if total <= 0.0 || peak <= 0.0 {
            return Self::neutral();
        }

        let centroid_log = weighted_log / total;
        let brightness = ((centroid_log - log_min) / log_span).clamp(0.0, 1.0);

        // Second pass: energy-weighted spread about the centroid (log-frequency
        // units, normalised by the full span) and the spectral-flatness log sum
        // over all usable bands. The flatness log sum floors each band at a
        // tiny fraction of the peak so silent bands pull the geometric mean
        // toward zero (tonal) without diverging, and the result is invariant to
        // the overall energy scale.
        let floor = peak * 1.0e-9;
        let mut var = 0.0;
        let mut ln_sum = 0.0;
        for (i, &raw) in spectrum.bands.iter().take(usable).enumerate() {
            let e = if raw.is_finite() && raw > 0.0 { raw } else { 0.0 };
            if e > 0.0 {
                let center = bands.band_center(i).max(MIN_HZ);
                let d = ops::log10(center) - centroid_log;
                var += e * d * d;
            }
            ln_sum += ops::ln(e.max(floor));
        }
        let spread_log = ops::sqrt((var / total).max(0.0));
        let width = (spread_log / log_span).clamp(0.0, 1.0);

        let n = usable as Sample;
        let geo = ops::exp(ln_sum / n);
        let arith = total / n;
        let flatness = if arith > 0.0 { (geo / arith).clamp(0.0, 1.0) } else { 0.0 };

        Self { brightness, width, flatness }
    }

    /// Euclidean distance to another descriptor in `[0, sqrt(3)]`.
    #[must_use]
    pub fn distance(&self, other: &Self) -> Sample {
        let db = self.brightness - other.brightness;
        let dw = self.width - other.width;
        let df = self.flatness - other.flatness;
        ops::sqrt(db * db + dw * dw + df * df)
    }

    /// Similarity in `(0, 1]`, equal to `1` for identical descriptors and
    /// decreasing monotonically with [`Timbre::distance`].
    #[must_use]
    pub fn similarity(&self, other: &Self) -> Sample {
        1.0 / (1.0 + self.distance(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masking::critical_bands::CriticalBands;
    use crate::masking::masking_model::VoiceSpectrum;

    const EPS: Sample = 1.0e-5;

    fn bands() -> CriticalBands {
        CriticalBands::bark_default()
    }

    #[test]
    fn silent_spectrum_is_neutral() {
        let b = bands();
        let s = VoiceSpectrum::zeros(b.band_count());
        let t = Timbre::from_spectrum(&s, &b);
        assert!((t.brightness - 0.5).abs() < EPS);
        assert!(t.width.abs() < EPS);
        assert!(t.flatness.abs() < EPS);
    }

    #[test]
    fn empty_spectrum_is_neutral() {
        let b = bands();
        let s = VoiceSpectrum::zeros(b.band_count());
        // A spectrum with zero usable bands falls back to the neutral timbre.
        let empty = VoiceSpectrum::default();
        let t = Timbre::from_spectrum(&empty, &b);
        let n = Timbre::from_spectrum(&s, &b);
        assert!((t.brightness - n.brightness).abs() < EPS);
    }

    #[test]
    fn low_tone_is_darker_than_high_tone() {
        let b = bands();
        let low = VoiceSpectrum::tonal(&b, 80.0, 1.0);
        let high = VoiceSpectrum::tonal(&b, 8000.0, 1.0);
        let tl = Timbre::from_spectrum(&low, &b);
        let th = Timbre::from_spectrum(&high, &b);
        assert!(th.brightness > tl.brightness + 0.1);
    }

    #[test]
    fn narrow_tone_has_small_width() {
        let b = bands();
        let tone = VoiceSpectrum::tonal(&b, 1000.0, 1.0);
        let t = Timbre::from_spectrum(&tone, &b);
        assert!(t.width < 0.05);
    }

    #[test]
    fn broadband_is_wider_than_tone() {
        let b = bands();
        let tone = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 1000.0, 1.0), &b);
        let mut broad = VoiceSpectrum::zeros(b.band_count());
        for band in &mut broad.bands {
            *band = 1.0;
        }
        let wide = Timbre::from_spectrum(&broad, &b);
        assert!(wide.width > tone.width + 0.1);
    }

    #[test]
    fn flat_spectrum_has_high_flatness() {
        let b = bands();
        let mut broad = VoiceSpectrum::zeros(b.band_count());
        for band in &mut broad.bands {
            *band = 1.0;
        }
        let t = Timbre::from_spectrum(&broad, &b);
        assert!(t.flatness > 0.9);
    }

    #[test]
    fn tone_has_low_flatness() {
        let b = bands();
        let tone = VoiceSpectrum::tonal(&b, 1000.0, 1.0);
        let t = Timbre::from_spectrum(&tone, &b);
        assert!(t.flatness < 0.2);
    }

    #[test]
    fn identical_descriptors_have_unit_similarity() {
        let b = bands();
        let tone = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 1000.0, 1.0), &b);
        assert!((tone.similarity(&tone) - 1.0).abs() < EPS);
        assert!(tone.distance(&tone).abs() < EPS);
    }

    #[test]
    fn dissimilar_descriptors_have_lower_similarity() {
        let b = bands();
        let low = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 60.0, 1.0), &b);
        let high = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 12000.0, 1.0), &b);
        let self_sim = low.similarity(&low);
        let cross_sim = low.similarity(&high);
        assert!(cross_sim < self_sim);
        assert!(cross_sim > 0.0);
    }

    #[test]
    fn distance_is_symmetric() {
        let b = bands();
        let a = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 200.0, 1.0), &b);
        let c = Timbre::from_spectrum(&VoiceSpectrum::tonal(&b, 4000.0, 1.0), &b);
        assert!((a.distance(&c) - c.distance(&a)).abs() < EPS);
    }

    #[test]
    fn default_is_neutral() {
        let d = Timbre::default();
        let n = Timbre::neutral();
        assert!((d.brightness - n.brightness).abs() < EPS);
    }
}
