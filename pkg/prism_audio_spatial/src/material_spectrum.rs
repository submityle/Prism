//! Frequency-dependent surface material for geometric acoustics.
//!
//! A real surface does not treat every frequency alike: a heavy curtain soaks
//! up highs while passing lows, a glass pane transmits bass through a wall and
//! reflects the treble, and a rough stone wall scatters a bright arrival into a
//! diffuse smear. The historical [`AcousticMaterial`] collapses all of that
//! into two broadband scalars (a single transmission loss and a single
//! reflection coefficient), which is all a mono low-pass arrival can carry.
//! Now that a [`PropagationPath`] carries a three-band [`BandGains`] envelope,
//! this module defines the matching three-band material every shipping engine
//! authors against.
//!
//! # Model
//!
//! [`BandedAcousticMaterial`] holds three per-band amplitude quantities on the
//! [`PROPAGATION_BAND_COUNT`]-band layout defined by [`BandGains`]:
//!
//! * `reflection` -- the amplitude a specular image source keeps when it
//!   bounces off the surface, per band. This is the frequency-dependent
//!   sibling of [`AcousticMaterial::reflection_gain`].
//! * `transmission` -- the amplitude a ray keeps when it passes *through* the
//!   surface, per band. This is the frequency-dependent sibling of
//!   [`AcousticMaterial::transmission_gain`].
//! * `scattering` -- a single broadband energy fraction in `[0, 1]` describing
//!   how much of the reflected energy leaves the surface diffusely rather than
//!   specularly, matching the single scattering coefficient Valve's Steam Audio
//!   and most room-acoustics tools expose. [`Self::specular_reflection`] and
//!   [`Self::diffuse_reflection`] split the reflection by this fraction with an
//!   energy-preserving `sqrt` weighting.
//!
//! Reflection is derived from the authoring vocabulary artists already use:
//! octave-band energy absorption (see [`MaterialAbsorption`] and the named
//! [`Material`] presets). Transmission is authored directly as a per-band loss
//! in decibels, because wall transmission loss is tabulated that way and is a
//! distinct physical property from surface absorption.
//!
//! # Relationship
//!
//! This type is the control-rate authoring source for the per-band colour a
//! geometry backend writes into [`PropagationPath::bands`]. It converts both
//! ways with the legacy broadband [`AcousticMaterial`]
//! ([`Self::from_scalar`] / [`Self::to_scalar`]) so a scene can mix upgraded
//! and legacy surfaces during migration, and it reuses
//! [`BandGains::reflection_from_absorption`] so a surface defined by its
//! absorption spectrum reflects exactly as the band spectrum module already
//! specifies.
//!
//! [`AcousticMaterial`]: crate::propagation::AcousticMaterial
//! [`AcousticMaterial::reflection_gain`]: crate::propagation::AcousticMaterial::reflection_gain
//! [`AcousticMaterial::transmission_gain`]: crate::propagation::AcousticMaterial::transmission_gain
//! [`PropagationPath`]: crate::propagation::PropagationPath
//! [`PropagationPath::bands`]: crate::propagation::PropagationPath::bands
//! [`MaterialAbsorption`]: crate::material_library::MaterialAbsorption
//! [`Material`]: crate::material_library::Material

use bevy_math::ops;
use prism_audio_core::math::{Sample, db_to_linear, linear_to_db, MIN_AUDIBLE_GAIN};

use crate::band_spectrum::{BandGains, PROPAGATION_BAND_COUNT};
use crate::material_library::{Material, MaterialAbsorption};
use crate::propagation::AcousticMaterial;

/// A finite transmission loss (decibels) used when a band is acoustically
/// opaque, so [`BandedAcousticMaterial::to_scalar`] never stores a non-finite
/// loss. `200 dB` is `1e-10` in amplitude: silent for every practical purpose.
const OPAQUE_LOSS_DB: Sample = 200.0;

/// A frequency-dependent surface material on the three-band propagation layout.
///
/// See the [module documentation](self) for the model. Construct one from an
/// octave-band absorption spectrum ([`Self::from_absorption`]), a named preset
/// ([`Self::from_material`]), explicit per-band values ([`Self::new`]), or a
/// legacy broadband [`AcousticMaterial`] ([`Self::from_scalar`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BandedAcousticMaterial {
    /// Per-band amplitude reflection coefficient in `[0, 1]`.
    reflection: BandGains,
    /// Per-band amplitude transmission gain in `[0, 1]`.
    transmission: BandGains,
    /// Broadband diffuse-scatter energy fraction in `[0, 1]`.
    scattering: Sample,
}

impl BandedAcousticMaterial {
    /// A fully open interface: lossless transmission, no reflection, no
    /// scattering. The three-band analogue of [`AcousticMaterial::OPEN`].
    pub const OPEN: Self = Self {
        reflection: BandGains::SILENT,
        transmission: BandGains::UNITY,
        scattering: 0.0,
    };

    /// A perfectly rigid, perfectly specular wall: full reflection, no
    /// transmission, no scattering.
    pub const RIGID: Self = Self {
        reflection: BandGains::UNITY,
        transmission: BandGains::SILENT,
        scattering: 0.0,
    };

    /// Builds a material from explicit per-band reflection and transmission
    /// amplitudes and a broadband scattering fraction.
    ///
    /// The `reflection` and `transmission` bands are taken as-is (each
    /// [`BandGains`] constructor already clamps to `[0, 1]`); `scattering` is
    /// clamped to `[0, 1]` and a non-finite value becomes `0`.
    #[inline]
    #[must_use]
    pub fn new(reflection: BandGains, transmission: BandGains, scattering: Sample) -> Self {
        Self {
            reflection,
            transmission,
            scattering: clamp_unit(scattering),
        }
    }

    /// Builds a material whose reflection comes from an octave-band absorption
    /// spectrum, with the given per-band transmission and scattering.
    ///
    /// Reflection is [`BandGains::reflection_from_absorption`], i.e. each band's
    /// amplitude reflection is `sqrt(1 - alpha)` for the band-averaged energy
    /// absorption `alpha`.
    #[inline]
    #[must_use]
    pub fn from_absorption(
        absorption: &MaterialAbsorption,
        transmission: BandGains,
        scattering: Sample,
    ) -> Self {
        Self::new(
            BandGains::reflection_from_absorption(absorption),
            transmission,
            scattering,
        )
    }

    /// Builds a material from a named [`Material`] preset, with the given
    /// per-band transmission and scattering.
    ///
    /// The preset supplies the octave-band absorption spectrum
    /// ([`Material::absorption`]); transmission and scattering are not part of
    /// the absorption library and so are supplied by the caller.
    #[inline]
    #[must_use]
    pub fn from_material(material: Material, transmission: BandGains, scattering: Sample) -> Self {
        Self::from_absorption(&material.absorption(), transmission, scattering)
    }

    /// Builds a three-band material from a legacy broadband [`AcousticMaterial`].
    ///
    /// Both the reflection coefficient and the transmission gain are spread
    /// flat across all three bands, and scattering is `0`. The result colours a
    /// path identically to the broadband material, so a legacy surface upgrades
    /// without changing its sound.
    #[inline]
    #[must_use]
    pub fn from_scalar(material: &AcousticMaterial) -> Self {
        Self {
            reflection: BandGains::uniform(material.reflection_gain()),
            transmission: BandGains::uniform(material.transmission_gain()),
            scattering: 0.0,
        }
    }

    /// The per-band amplitude reflection coefficient.
    #[inline]
    #[must_use]
    pub fn reflection(&self) -> BandGains {
        self.reflection
    }

    /// The per-band amplitude transmission gain.
    #[inline]
    #[must_use]
    pub fn transmission(&self) -> BandGains {
        self.transmission
    }

    /// The broadband diffuse-scatter energy fraction in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn scattering(&self) -> Sample {
        self.scattering
    }

    /// The specular share of the reflection, i.e. the part that leaves the
    /// surface as a mirror image source.
    ///
    /// Energy conservation splits the reflected energy as
    /// `specular_energy = (1 - scattering)` of the total, so the amplitude
    /// weight is `sqrt(1 - scattering)`.
    #[inline]
    #[must_use]
    pub fn specular_reflection(&self) -> BandGains {
        self.reflection.scaled(ops::sqrt((1.0 - self.scattering).max(0.0)))
    }

    /// The diffuse share of the reflection, i.e. the part scattered off the
    /// specular direction.
    ///
    /// Complementary to [`Self::specular_reflection`]: the amplitude weight is
    /// `sqrt(scattering)`, so specular and diffuse energy sum to the total
    /// reflected energy band by band.
    #[inline]
    #[must_use]
    pub fn diffuse_reflection(&self) -> BandGains {
        self.reflection.scaled(ops::sqrt(self.scattering.max(0.0)))
    }

    /// The single broadband reflection coefficient, as the root-mean-square of
    /// the per-band reflection amplitudes.
    #[inline]
    #[must_use]
    pub fn broadband_reflection(&self) -> Sample {
        self.reflection.broadband_rms()
    }

    /// The single broadband transmission gain, as the root-mean-square of the
    /// per-band transmission amplitudes.
    #[inline]
    #[must_use]
    pub fn broadband_transmission(&self) -> Sample {
        self.transmission.broadband_rms()
    }

    /// Collapses this material back into a legacy broadband [`AcousticMaterial`].
    ///
    /// The reflection coefficient is [`Self::broadband_reflection`] and the
    /// transmission loss is the decibel value whose
    /// [`AcousticMaterial::transmission_gain`] equals
    /// [`Self::broadband_transmission`]. Scattering has no broadband analogue
    /// and is dropped. Round-trips with [`Self::from_scalar`] to within
    /// floating-point tolerance for a flat material.
    #[inline]
    #[must_use]
    pub fn to_scalar(&self) -> AcousticMaterial {
        let gain = self.broadband_transmission();
        let loss_db = if gain <= MIN_AUDIBLE_GAIN {
            OPAQUE_LOSS_DB
        } else {
            -linear_to_db(gain)
        };
        AcousticMaterial::new(loss_db, self.broadband_reflection())
    }
}

impl Default for BandedAcousticMaterial {
    /// The default surface is [`BandedAcousticMaterial::OPEN`], matching
    /// [`AcousticMaterial::default`].
    #[inline]
    fn default() -> Self {
        Self::OPEN
    }
}

/// Builds a per-band transmission [`BandGains`] from per-band transmission loss
/// in decibels (low, mid, high), using the same `loss -> gain` law as
/// [`AcousticMaterial::transmission_gain`]: `gain = 10 ^ (-max(loss, 0) / 20)`.
///
/// A `0 dB` or negative-loss band passes unchanged; a larger positive loss
/// attenuates more, and `+infinity` is a fully opaque band (`gain = 0`). A
/// `NaN` loss is treated as `0 dB` so an unauthored field never silences a
/// path.
#[must_use]
pub fn transmission_from_loss_db(loss_db: [Sample; PROPAGATION_BAND_COUNT]) -> BandGains {
    let mut bands = [0.0; PROPAGATION_BAND_COUNT];
    for (gain, &loss) in bands.iter_mut().zip(loss_db.iter()) {
        *gain = if loss.is_nan() || loss <= 0.0 {
            1.0
        } else {
            // `db_to_linear(-infinity)` is `0`, so an infinite loss is opaque.
            db_to_linear(-loss)
        };
    }
    BandGains::new(bands)
}

/// Clamps a scattering fraction into `[0, 1]`, mapping a non-finite input to
/// `0` so an unauthored field never poisons later arithmetic.
fn clamp_unit(value: Sample) -> Sample {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_spectrum::PROPAGATION_BAND_CENTERS;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn open_passes_everything_and_reflects_nothing() {
        let open = BandedAcousticMaterial::OPEN;
        assert!(open.transmission().is_full_band(1e-6));
        assert!(open.reflection().is_silent(1e-6));
        assert!(approx(open.scattering(), 0.0, 1e-6));
    }

    #[test]
    fn rigid_reflects_everything_and_transmits_nothing() {
        let rigid = BandedAcousticMaterial::RIGID;
        assert!(rigid.reflection().is_full_band(1e-6));
        assert!(rigid.transmission().is_silent(1e-6));
    }

    #[test]
    fn default_is_open() {
        assert_eq!(BandedAcousticMaterial::default(), BandedAcousticMaterial::OPEN);
    }

    #[test]
    fn scattering_is_clamped() {
        let hot = BandedAcousticMaterial::new(BandGains::UNITY, BandGains::SILENT, 4.0);
        assert!(approx(hot.scattering(), 1.0, 1e-6));
        let cold = BandedAcousticMaterial::new(BandGains::UNITY, BandGains::SILENT, -1.0);
        assert!(approx(cold.scattering(), 0.0, 1e-6));
        let nan = BandedAcousticMaterial::new(BandGains::UNITY, BandGains::SILENT, Sample::NAN);
        assert!(approx(nan.scattering(), 0.0, 1e-6));
    }

    #[test]
    fn specular_and_diffuse_energy_sum_to_total() {
        let m = BandedAcousticMaterial::new(BandGains::uniform(0.8), BandGains::SILENT, 0.3);
        let specular = m.specular_reflection();
        let diffuse = m.diffuse_reflection();
        for i in 0..PROPAGATION_BAND_COUNT {
            let total_energy = m.reflection().band(i) * m.reflection().band(i);
            let split_energy = specular.band(i) * specular.band(i)
                + diffuse.band(i) * diffuse.band(i);
            assert!(approx(total_energy, split_energy, 1e-5));
        }
    }

    #[test]
    fn fully_scattering_has_no_specular_reflection() {
        let m = BandedAcousticMaterial::new(BandGains::uniform(0.9), BandGains::SILENT, 1.0);
        assert!(m.specular_reflection().is_silent(1e-6));
    }

    #[test]
    fn from_absorption_matches_band_spectrum_reflection() {
        let carpet = Material::Carpet.absorption();
        let m = BandedAcousticMaterial::from_material(Material::Carpet, BandGains::UNITY, 0.0);
        let expected = BandGains::reflection_from_absorption(&carpet);
        for i in 0..PROPAGATION_BAND_COUNT {
            assert!(approx(m.reflection().band(i), expected.band(i), 1e-6));
        }
        // Carpet absorbs far more high than low, so it reflects more low than
        // high.
        assert!(m.reflection().low() > m.reflection().high());
    }

    #[test]
    fn transmission_from_loss_matches_scalar_law() {
        let bands = transmission_from_loss_db([0.0, 6.020_6, OPAQUE_LOSS_DB]);
        // 0 dB passes unchanged.
        assert!(approx(bands.low(), 1.0, 1e-4));
        // ~6 dB halves the amplitude.
        assert!(approx(bands.mid(), 0.5, 1e-3));
        // A huge loss is effectively silent.
        assert!(bands.high() < 1e-4);
    }

    #[test]
    fn transmission_from_loss_rejects_non_finite_and_negative() {
        let bands = transmission_from_loss_db([-10.0, Sample::NAN, Sample::INFINITY]);
        // Negative and non-finite losses collapse to 0 dB -> unity gain, except
        // positive infinity which is a real (opaque) loss.
        assert!(approx(bands.low(), 1.0, 1e-4));
        assert!(approx(bands.mid(), 1.0, 1e-4));
        assert!(bands.high() < 1e-4);
    }

    #[test]
    fn scalar_round_trip_is_stable() {
        let scalar = AcousticMaterial::new(12.0, 0.6);
        let banded = BandedAcousticMaterial::from_scalar(&scalar);
        let back = banded.to_scalar();
        assert!(approx(back.reflection_gain(), scalar.reflection_gain(), 1e-4));
        assert!(approx(
            back.transmission_gain(),
            scalar.transmission_gain(),
            1e-4,
        ));
    }

    #[test]
    fn opaque_scalar_round_trip_stays_finite() {
        let silent = BandedAcousticMaterial::new(BandGains::UNITY, BandGains::SILENT, 0.0);
        let scalar = silent.to_scalar();
        assert!(scalar.transmission_loss_db.is_finite());
        assert!(scalar.transmission_gain() < 1e-4);
    }

    #[test]
    fn frequency_centres_are_used() {
        // Guard that the band layout this module reasons about has not drifted.
        assert_eq!(PROPAGATION_BAND_CENTERS.len(), PROPAGATION_BAND_COUNT);
    }
}
