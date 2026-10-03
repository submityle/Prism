//! Acoustic materials for the Encoder Input Format (EIF) scene model.
//!
//! An [`AcousticMaterial`] describes how a surface interacts with incident
//! sound across the eight ISO octave bands shared with the rest of the engine
//! ([`prism_audio_spatial::material_library`]). Each band carries three
//! independent, publicly standard coefficients:
//!
//! * **absorption** `alpha`: the fraction of incident energy removed by the
//!   surface (converted to heat or carried away). The complementary fraction
//!   `1 - alpha` is returned to the room as reflected energy.
//! * **scattering** `s`: of the reflected energy, the fraction redirected
//!   diffusely (Lambert-like) rather than specularly. `s = 0` is a perfect
//!   mirror; `s = 1` is fully diffuse. This matches the ISO 17497 scattering
//!   coefficient used by the engine's `SurfaceScatter`.
//! * **transmission** `tau`: the fraction of incident energy that passes
//!   through the surface into the space behind it (wall transmission loss).
//!
//! The three coefficients are stored and queried independently; the model does
//! not force `alpha + tau <= 1` because transmitted energy is accounted on the
//! far side of the surface rather than inside the absorption budget. Callers
//! that need a strict single-surface energy split can use
//! [`AcousticMaterial::reflected_fraction`] and
//! [`AcousticMaterial::transmitted_fraction`].
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The absorption / scattering / transmission triplet is the publicly
//! documented room-acoustics surface model (ISO 354 absorption, ISO 17497
//! scattering, mass-law transmission); the EIF names it as a declarative scene
//! material (MPEG-I Immersive Audio, ISO/IEC 23090-4, section 49.1).
//!
//! # Relationship
//!
//! Reuses the shared octave-band grid of
//! [`prism_audio_spatial::material_library`] (so EIF materials map directly
//! onto the engine's section 14 acoustic materials and the shared BVH) and can
//! be built from a stock [`prism_audio_spatial::material_library::Material`].
//! Consumed by [`crate::eif::geometry`] (per-primitive material reference) and
//! [`crate::eif::scene`] (the scene material table).

use bevy_math::ops;

use prism_audio_core::math::Sample;
use prism_audio_spatial::material_library::{
    Material, MaterialAbsorption, OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT,
};

/// Default diffuse scattering fraction assigned to a stock library material
/// when no measured scattering spectrum is supplied: a mildly diffusing
/// surface rather than a perfect mirror.
const DEFAULT_SCATTERING: Sample = 0.1;

/// Default energy transmission assigned to a stock library material: a nearly
/// opaque surface (most of the incident energy is reflected or absorbed, very
/// little passes through).
const DEFAULT_TRANSMISSION: Sample = 0.02;

/// A frequency-dependent acoustic surface material.
///
/// Each of the three spectra holds one coefficient per octave band aligned with
/// [`OCTAVE_BAND_CENTERS`]. Every stored value is clamped to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticMaterial {
    absorption: [Sample; OCTAVE_BAND_COUNT],
    scattering: [Sample; OCTAVE_BAND_COUNT],
    transmission: [Sample; OCTAVE_BAND_COUNT],
}

/// Clamps every element of a spectrum into `[0, 1]`, replacing non-finite
/// entries with `0`.
fn sanitize(mut spectrum: [Sample; OCTAVE_BAND_COUNT]) -> [Sample; OCTAVE_BAND_COUNT] {
    for v in &mut spectrum {
        *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    }
    spectrum
}

/// Interpolates a spectrum at an arbitrary frequency, linearly in
/// `log(frequency)` between the two bracketing octave-band centres, clamped at
/// the end bands (never extrapolated).
fn interpolate_at(spectrum: &[Sample; OCTAVE_BAND_COUNT], freq_hz: Sample) -> Sample {
    if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
        return spectrum[0];
    }
    let last = OCTAVE_BAND_COUNT - 1;
    if freq_hz >= OCTAVE_BAND_CENTERS[last] {
        return spectrum[last];
    }
    let log_f = ops::ln(freq_hz);
    for (centres, values) in OCTAVE_BAND_CENTERS.windows(2).zip(spectrum.windows(2)) {
        let c_lo = centres[0];
        let c_hi = centres[1];
        if freq_hz <= c_hi {
            let log_lo = ops::ln(c_lo);
            let log_hi = ops::ln(c_hi);
            let span = log_hi - log_lo;
            let t = if span > 0.0 {
                (log_f - log_lo) / span
            } else {
                0.0
            };
            let v = values[0] + (values[1] - values[0]) * t;
            return v.clamp(0.0, 1.0);
        }
    }
    spectrum[last]
}

impl AcousticMaterial {
    /// Builds a material from explicit absorption, scattering, and transmission
    /// spectra. Each spectrum is sanitised (non-finite entries become `0`, all
    /// values clamped to `[0, 1]`).
    #[must_use]
    pub fn new(
        absorption: [Sample; OCTAVE_BAND_COUNT],
        scattering: [Sample; OCTAVE_BAND_COUNT],
        transmission: [Sample; OCTAVE_BAND_COUNT],
    ) -> Self {
        Self {
            absorption: sanitize(absorption),
            scattering: sanitize(scattering),
            transmission: sanitize(transmission),
        }
    }

    /// Builds a material from a single broadband value for each coefficient
    /// (the same number repeated across all eight bands).
    #[must_use]
    pub fn broadband(absorption: Sample, scattering: Sample, transmission: Sample) -> Self {
        Self::new(
            [absorption; OCTAVE_BAND_COUNT],
            [scattering; OCTAVE_BAND_COUNT],
            [transmission; OCTAVE_BAND_COUNT],
        )
    }

    /// Builds a material from a stock engine library [`Material`], taking its
    /// published absorption spectrum and assigning the default mild scattering
    /// and near-opaque transmission. This is the bridge from the shared
    /// section 14 material library into the EIF scene model.
    #[must_use]
    pub fn from_library(material: Material) -> Self {
        Self::new(
            material.absorption().bands(),
            [DEFAULT_SCATTERING; OCTAVE_BAND_COUNT],
            [DEFAULT_TRANSMISSION; OCTAVE_BAND_COUNT],
        )
    }

    /// A perfectly rigid, lossless mirror: no absorption, no scattering, no
    /// transmission (all incident energy is specularly reflected).
    #[must_use]
    pub fn rigid() -> Self {
        Self::broadband(0.0, 0.0, 0.0)
    }

    /// The per-band absorption spectrum (`alpha`).
    #[must_use]
    pub fn absorption(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.absorption
    }

    /// The per-band scattering spectrum (`s`).
    #[must_use]
    pub fn scattering(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.scattering
    }

    /// The per-band transmission spectrum (`tau`).
    #[must_use]
    pub fn transmission(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.transmission
    }

    /// Absorption at an arbitrary frequency (log-frequency interpolation,
    /// clamped at the end bands).
    #[must_use]
    pub fn absorption_at(&self, freq_hz: Sample) -> Sample {
        interpolate_at(&self.absorption, freq_hz)
    }

    /// Scattering at an arbitrary frequency (log-frequency interpolation,
    /// clamped at the end bands).
    #[must_use]
    pub fn scattering_at(&self, freq_hz: Sample) -> Sample {
        interpolate_at(&self.scattering, freq_hz)
    }

    /// Transmission at an arbitrary frequency (log-frequency interpolation,
    /// clamped at the end bands).
    #[must_use]
    pub fn transmission_at(&self, freq_hz: Sample) -> Sample {
        interpolate_at(&self.transmission, freq_hz)
    }

    /// The reflected energy fraction in band `index`, `1 - alpha`. Out-of-range
    /// indices clamp to the end bands.
    #[must_use]
    pub fn reflected_fraction(&self, index: usize) -> Sample {
        let i = index.min(OCTAVE_BAND_COUNT - 1);
        (1.0 - self.absorption[i]).clamp(0.0, 1.0)
    }

    /// The transmitted energy fraction in band `index`, `tau`. Out-of-range
    /// indices clamp to the end bands.
    #[must_use]
    pub fn transmitted_fraction(&self, index: usize) -> Sample {
        let i = index.min(OCTAVE_BAND_COUNT - 1);
        self.transmission[i]
    }

    /// The specular reflection pressure coefficient in band `index`,
    /// `sqrt((1 - alpha) * (1 - s))`: the amplitude of the mirror-direction
    /// reflection after both absorption and the diffuse split are removed.
    #[must_use]
    pub fn specular_pressure(&self, index: usize) -> Sample {
        let i = index.min(OCTAVE_BAND_COUNT - 1);
        let reflected = (1.0 - self.absorption[i]).clamp(0.0, 1.0);
        let specular = reflected * (1.0 - self.scattering[i]).clamp(0.0, 1.0);
        ops::sqrt(specular.clamp(0.0, 1.0))
    }

    /// The engine-shared absorption view ([`MaterialAbsorption`]), so an EIF
    /// material can drive the section 14 reverberation and reflection models
    /// directly.
    #[must_use]
    pub fn as_absorption(&self) -> MaterialAbsorption {
        MaterialAbsorption::new(self.absorption)
    }
}

impl Default for AcousticMaterial {
    /// A neutral mildly-absorptive, mildly-diffusing, near-opaque surface.
    #[inline]
    fn default() -> Self {
        Self::broadband(0.1, DEFAULT_SCATTERING, DEFAULT_TRANSMISSION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn new_clamps_and_sanitizes() {
        let m = AcousticMaterial::new(
            [2.0; OCTAVE_BAND_COUNT],
            [-1.0; OCTAVE_BAND_COUNT],
            [Sample::NAN; OCTAVE_BAND_COUNT],
        );
        assert!(m.absorption().iter().all(|&a| close(a, 1.0)));
        assert!(m.scattering().iter().all(|&s| close(s, 0.0)));
        assert!(m.transmission().iter().all(|&t| close(t, 0.0)));
    }

    #[test]
    fn rigid_reflects_everything() {
        let m = AcousticMaterial::rigid();
        for b in 0..OCTAVE_BAND_COUNT {
            assert!(close(m.reflected_fraction(b), 1.0));
            assert!(close(m.specular_pressure(b), 1.0));
            assert!(close(m.transmitted_fraction(b), 0.0));
        }
    }

    #[test]
    fn from_library_tracks_absorption() {
        let m = AcousticMaterial::from_library(Material::Concrete);
        let a = Material::Concrete.absorption().bands();
        assert_eq!(m.absorption(), a);
    }

    #[test]
    fn interpolation_clamps_at_ends() {
        let m = AcousticMaterial::broadband(0.3, 0.0, 0.0);
        assert!(close(m.absorption_at(10.0), 0.3));
        assert!(close(m.absorption_at(20_000.0), 0.3));
        assert!(close(m.absorption_at(1000.0), 0.3));
    }

    #[test]
    fn specular_pressure_drops_with_scattering() {
        let m = AcousticMaterial::broadband(0.0, 0.75, 0.0);
        // reflected = 1, specular energy = 0.25, pressure = 0.5.
        assert!(close(m.specular_pressure(3), 0.5));
    }

    #[test]
    fn as_absorption_round_trips_bands() {
        let bands = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let m = AcousticMaterial::new(bands, [0.0; OCTAVE_BAND_COUNT], [0.0; OCTAVE_BAND_COUNT]);
        assert_eq!(m.as_absorption().bands(), bands);
    }
}
