//! Surface scattering and diffusion: splitting a wall reflection into a
//! specular part and a Lambert-diffuse part.
//!
//! A real surface is never a perfect mirror. Roughness and relief scatter part
//! of the incident energy away from the specular direction. The ISO 17497
//! scattering coefficient `s(band)` is the fraction of reflected energy that
//! leaves *non-specularly*; the remaining `1 - s(band)` stays specular and
//! keeps the mirror-image direction used by [`crate::early_reflections`]. The
//! scattered fraction feeds a late diffuse field with a Lambert cosine
//! directivity.
//!
//! This module supplies:
//! - a per-octave-band scattering spectrum ([`ScatteringSpectrum`]) aligned
//!   with [`OCTAVE_BAND_CENTERS`], with the same log-frequency query and
//!   endpoint clamping as [`crate::material_library`];
//! - a small table of typical surfaces ([`SurfaceScatter`]);
//! - the energy split ([`specular_fraction`], [`diffuse_fraction`]);
//! - the Lambert directivity used to weight scattered energy by direction
//!   ([`lambert_weight`], [`lambert_directivity`]).
//!
//! # The model (classic geometrical/statistical acoustics)
//!
//! Scattering coefficients rise with frequency: at long wavelengths a surface
//! looks flat (little scattering), while at short wavelengths the same relief
//! scatters strongly. Given a reflected band energy `E` and a scattering
//! coefficient `s`, the specular energy is `(1 - s) * E` and the diffuse
//! energy is `s * E`. The diffuse energy radiates with Lambert's cosine law:
//! the radiant intensity in a direction making angle `theta` with the surface
//! normal is proportional to `cos(theta)`, normalised so the hemispherical
//! integral is unity, giving a per-steradian weight `cos(theta) / PI`.
//!
//! # Control rate, not audio rate
//!
//! Everything here is a small scalar computation over eight bands or a single
//! direction: no allocation, no locking, no panics, and no `f32` intrinsics
//! (all logarithms and roots route through [`bevy_math::ops`]). Degenerate
//! inputs (out-of-range coefficients, back-facing or zero-length directions)
//! clamp to safe finite values, never a `NaN` or an infinity.
//!
//! # Provenance
//!
//! This is the textbook treatment of surface scattering and diffuse reflection:
//! the scattering coefficient as defined by ISO 17497-1, Lambert's cosine law
//! for diffuse reflection as presented in H. Kuttruff's *Room Acoustics* and
//! T. Cox and P. D'Antonio's *Acoustic Absorbers and Diffusers*. This module is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics knowledge.

use bevy_math::Vec3;
use bevy_math::ops;
use core::f32::consts::PI;

use prism_audio_core::math::Sample;

use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// Smallest squared length treated as a usable direction vector.
const MIN_DIR_LEN_SQ: Sample = 1e-12;

// Octave-band scattering spectra, aligned with `OCTAVE_BAND_CENTERS`
// (63 Hz .. 8 kHz). Values are representative of the surface class and follow
// the universal trend of rising scattering with frequency. Sources: typical
// ISO 17497-1 measured ranges reported in Cox and D'Antonio and in Vorlander,
// rounded to convenient reference values.

/// A smooth flat surface (glass, polished plaster): almost specular.
const FLAT: [Sample; OCTAVE_BAND_COUNT] =
    [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12];

/// Painted or sealed brick: shallow relief, low-to-moderate scattering.
const PAINTED_BRICK: [Sample; OCTAVE_BAND_COUNT] =
    [0.05, 0.08, 0.12, 0.18, 0.25, 0.32, 0.40, 0.45];

/// Bare rough brickwork: pronounced mortar relief, moderate-to-high scattering.
const ROUGH_BRICK: [Sample; OCTAVE_BAND_COUNT] =
    [0.10, 0.15, 0.22, 0.32, 0.42, 0.52, 0.60, 0.65];

/// A filled bookshelf: deep irregular relief, strong broadband scattering.
const BOOKSHELF: [Sample; OCTAVE_BAND_COUNT] =
    [0.15, 0.25, 0.40, 0.55, 0.65, 0.72, 0.78, 0.80];

/// A dedicated acoustic diffuser (Schroeder/QRD type): very high scattering.
const DIFFUSER: [Sample; OCTAVE_BAND_COUNT] =
    [0.20, 0.35, 0.55, 0.75, 0.88, 0.92, 0.94, 0.95];

/// A hanging curtain in folds: soft irregular surface, moderate scattering.
const CURTAIN: [Sample; OCTAVE_BAND_COUNT] =
    [0.08, 0.14, 0.22, 0.30, 0.38, 0.44, 0.48, 0.50];

/// A per-octave-band scattering-coefficient spectrum.
///
/// Each coefficient is the ISO 17497 fraction of reflected energy that leaves
/// the surface non-specularly, in `[0, 1]`, aligned with
/// [`OCTAVE_BAND_CENTERS`] (63 Hz to 8 kHz). Well-behaved surfaces scatter more
/// at higher frequencies.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ScatteringSpectrum {
    /// Per-octave-band scattering coefficients, aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    s: [Sample; OCTAVE_BAND_COUNT],
}

impl ScatteringSpectrum {
    /// Builds a spectrum from eight coefficients, clamping each to `[0, 1]`.
    #[must_use]
    pub fn new(s: [Sample; OCTAVE_BAND_COUNT]) -> Self {
        let mut clamped = s;
        for c in &mut clamped {
            *c = c.clamp(0.0, 1.0);
        }
        Self { s: clamped }
    }

    /// The eight per-band scattering coefficients, aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    #[must_use]
    pub fn bands(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.s
    }

    /// Scattering coefficient at an arbitrary frequency, interpolated linearly
    /// in `log(frequency)` between the two bracketing octave bands.
    ///
    /// Frequencies at or below 63 Hz return the lowest band, at or above 8 kHz
    /// return the highest band (clamped, never extrapolated). A non-finite or
    /// non-positive frequency falls back to the lowest band. The result is
    /// clamped to `[0, 1]`.
    #[must_use]
    pub fn at(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return self.s[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return self.s[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, values) in OCTAVE_BAND_CENTERS.windows(2).zip(self.s.windows(2)) {
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
        self.s[last]
    }

    /// The unweighted arithmetic mean of the eight octave-band coefficients.
    ///
    /// A coarse single-number scattering coefficient. Always in `[0, 1]`.
    #[must_use]
    pub fn broadband_mean(&self) -> Sample {
        let mut sum = 0.0;
        for c in &self.s {
            sum += *c;
        }
        sum / OCTAVE_BAND_COUNT as Sample
    }
}

/// A named surface class with a representative scattering spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SurfaceScatter {
    /// Smooth flat surface (glass, polished plaster): near-specular.
    Flat,
    /// Painted or sealed brick: shallow relief.
    PaintedBrick,
    /// Bare rough brickwork: pronounced mortar relief.
    RoughBrick,
    /// A filled bookshelf: deep irregular relief.
    Bookshelf,
    /// A dedicated acoustic diffuser (Schroeder/QRD type).
    Diffuser,
    /// A hanging curtain in folds.
    Curtain,
}

impl SurfaceScatter {
    /// The representative per-octave-band scattering spectrum for this surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::scattering::{
    ///     diffuse_fraction, specular_fraction, SurfaceScatter,
    /// };
    ///
    /// // A diffuser scatters far more energy than smooth glass at 4 kHz.
    /// let diffuser = SurfaceScatter::Diffuser.scattering();
    /// let flat = SurfaceScatter::Flat.scattering();
    /// let s = diffuser.at(4000.0);
    /// assert!(s > flat.at(4000.0));
    ///
    /// // Specular and diffuse fractions always partition the reflected energy.
    /// assert!((specular_fraction(s) + diffuse_fraction(s) - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn scattering(self) -> ScatteringSpectrum {
        let bands = match self {
            SurfaceScatter::Flat => FLAT,
            SurfaceScatter::PaintedBrick => PAINTED_BRICK,
            SurfaceScatter::RoughBrick => ROUGH_BRICK,
            SurfaceScatter::Bookshelf => BOOKSHELF,
            SurfaceScatter::Diffuser => DIFFUSER,
            SurfaceScatter::Curtain => CURTAIN,
        };
        ScatteringSpectrum::new(bands)
    }

    /// The coarse single-number scattering coefficient for this surface.
    #[must_use]
    pub fn broadband_mean(self) -> Sample {
        self.scattering().broadband_mean()
    }
}

/// The specular fraction of reflected energy for a scattering coefficient `s`.
///
/// This is `1 - s`, clamped to `[0, 1]`. Multiply an incident reflected band
/// energy by this to obtain the energy that keeps the mirror-image direction.
#[must_use]
pub fn specular_fraction(s: Sample) -> Sample {
    (1.0 - s.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// The diffuse (scattered) fraction of reflected energy for a coefficient `s`.
///
/// This is `s`, clamped to `[0, 1]`. Multiply an incident reflected band energy
/// by this to obtain the energy that feeds the late diffuse field.
#[must_use]
pub fn diffuse_fraction(s: Sample) -> Sample {
    s.clamp(0.0, 1.0)
}

/// The Lambert cosine directivity weight for a diffuse reflection.
///
/// Given `cos_theta`, the cosine of the angle between the outgoing direction
/// and the surface normal, returns `cos_theta / PI` for a front-facing
/// direction and `0` for a back-facing one (`cos_theta <= 0`). Integrating this
/// weight over the forward hemisphere yields unity, so it is an energy-
/// preserving per-steradian directivity.
#[must_use]
pub fn lambert_weight(cos_theta: Sample) -> Sample {
    if !cos_theta.is_finite() || cos_theta <= 0.0 {
        return 0.0;
    }
    (cos_theta / PI).max(0.0)
}

/// The Lambert directivity for an outgoing direction relative to a normal.
///
/// `normal` is the surface normal and `direction` the outgoing direction; both
/// are normalised internally. Returns the [`lambert_weight`] of the cosine of
/// the angle between them. Zero-length inputs return `0` rather than a `NaN`.
#[must_use]
pub fn lambert_directivity(normal: Vec3, direction: Vec3) -> Sample {
    let n_len_sq = normal.dot(normal);
    let d_len_sq = direction.dot(direction);
    if n_len_sq <= MIN_DIR_LEN_SQ || d_len_sq <= MIN_DIR_LEN_SQ {
        return 0.0;
    }
    let inv_norm = 1.0 / (ops::sqrt(n_len_sq) * ops::sqrt(d_len_sq));
    let cos_theta = normal.dot(direction) * inv_norm;
    lambert_weight(cos_theta.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn all_surface_bands_are_in_unit_range() {
        for surface in [
            SurfaceScatter::Flat,
            SurfaceScatter::PaintedBrick,
            SurfaceScatter::RoughBrick,
            SurfaceScatter::Bookshelf,
            SurfaceScatter::Diffuser,
            SurfaceScatter::Curtain,
        ] {
            for c in surface.scattering().bands() {
                assert!((0.0..=1.0).contains(&c), "coeff out of range: {c}");
            }
        }
    }

    #[test]
    fn scattering_rises_with_frequency() {
        // The high band should scatter at least as much as the low band.
        for surface in [
            SurfaceScatter::Flat,
            SurfaceScatter::PaintedBrick,
            SurfaceScatter::RoughBrick,
            SurfaceScatter::Bookshelf,
            SurfaceScatter::Diffuser,
            SurfaceScatter::Curtain,
        ] {
            let b = surface.scattering().bands();
            assert!(b[OCTAVE_BAND_COUNT - 1] > b[0], "not rising for {surface:?}");
        }
    }

    #[test]
    fn rough_surfaces_scatter_more_than_flat() {
        let flat = SurfaceScatter::Flat.broadband_mean();
        let diffuser = SurfaceScatter::Diffuser.broadband_mean();
        let bookshelf = SurfaceScatter::Bookshelf.broadband_mean();
        assert!(diffuser > flat);
        assert!(bookshelf > flat);
        assert!(diffuser > bookshelf);
    }

    #[test]
    fn new_clamps_out_of_range_coefficients() {
        let spec = ScatteringSpectrum::new([-1.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let b = spec.bands();
        assert!(approx(b[0], 0.0, 1e-9));
        assert!(approx(b[1], 1.0, 1e-9));
        assert!(approx(b[2], 0.5, 1e-9));
    }

    #[test]
    fn at_band_center_returns_that_band() {
        let spec = SurfaceScatter::RoughBrick.scattering();
        let b = spec.bands();
        for (i, centre) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            assert!(approx(spec.at(*centre), b[i], 1e-4), "band {i}");
        }
    }

    #[test]
    fn at_clamps_beyond_range_and_survives_degenerate_input() {
        let spec = SurfaceScatter::Diffuser.scattering();
        let b = spec.bands();
        assert!(approx(spec.at(10.0), b[0], 1e-4));
        assert!(approx(spec.at(30000.0), b[OCTAVE_BAND_COUNT - 1], 1e-4));
        let _ = spec.at(Sample::NAN);
        let _ = spec.at(-3.0);
        let _ = spec.at(0.0);
    }

    #[test]
    fn at_interpolates_between_bands() {
        let spec = ScatteringSpectrum::new([0.0, 0.0, 0.0, 0.2, 0.6, 0.6, 0.6, 0.6]);
        // 707 Hz is the geometric midpoint of the 500 and 1000 Hz bands, whose
        // coefficients are 0.2 and 0.6; log-linear interpolation gives ~0.4.
        let v = spec.at(707.0);
        assert!(approx(v, 0.4, 0.02), "v={v}");
    }

    #[test]
    fn broadband_mean_matches_manual_average() {
        let spec = ScatteringSpectrum::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert!(approx(spec.broadband_mean(), 0.45, 1e-6));
    }

    #[test]
    fn specular_and_diffuse_fractions_partition_energy() {
        for &s in &[0.0, 0.25, 0.5, 0.9, 1.0] {
            let spec = specular_fraction(s);
            let diff = diffuse_fraction(s);
            assert!(approx(spec + diff, 1.0, 1e-6), "s={s}");
            assert!((0.0..=1.0).contains(&spec));
            assert!((0.0..=1.0).contains(&diff));
        }
        // Out-of-range input is clamped, still partitions to one.
        assert!(approx(specular_fraction(2.0) + diffuse_fraction(2.0), 1.0, 1e-6));
        assert!(approx(specular_fraction(-1.0) + diffuse_fraction(-1.0), 1.0, 1e-6));
    }

    #[test]
    fn lambert_weight_peaks_at_normal_and_vanishes_at_grazing() {
        // At the normal (cos = 1) the weight is 1/PI; at grazing (cos = 0) and
        // behind the surface (cos < 0) it is zero.
        assert!(approx(lambert_weight(1.0), 1.0 / PI, 1e-6));
        assert!(approx(lambert_weight(0.0), 0.0, 1e-9));
        assert!(approx(lambert_weight(-0.5), 0.0, 1e-9));
        assert!(approx(lambert_weight(Sample::NAN), 0.0, 1e-9));
        // Monotonic in cos_theta over the forward hemisphere.
        assert!(lambert_weight(0.8) > lambert_weight(0.3));
    }

    #[test]
    fn lambert_directivity_uses_angle_to_normal() {
        let n = Vec3::Y;
        // Straight up: cos = 1 -> 1/PI.
        assert!(approx(lambert_directivity(n, Vec3::Y), 1.0 / PI, 1e-6));
        // Sideways: cos = 0 -> 0.
        assert!(approx(lambert_directivity(n, Vec3::X), 0.0, 1e-6));
        // Downward (behind surface): 0.
        assert!(approx(lambert_directivity(n, Vec3::NEG_Y), 0.0, 1e-6));
        // Non-unit vectors are normalised: scaling does not change the weight.
        let scaled = lambert_directivity(n * 5.0, Vec3::Y * 3.0);
        assert!(approx(scaled, 1.0 / PI, 1e-6));
    }

    #[test]
    fn lambert_directivity_handles_zero_length_inputs() {
        assert!(approx(lambert_directivity(Vec3::ZERO, Vec3::Y), 0.0, 1e-9));
        assert!(approx(lambert_directivity(Vec3::Y, Vec3::ZERO), 0.0, 1e-9));
        assert!(lambert_directivity(Vec3::Y, Vec3::Y).is_finite());
    }
}
