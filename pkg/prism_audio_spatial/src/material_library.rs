//! Standard building-acoustics absorption library: octave-band absorption
//! coefficients for common surface materials, with frequency interpolation and
//! broadband aggregation.
//!
//! Upstream reverberation models want physically credible absorption, not
//! hand-tuned magic numbers. [`crate::early_reflections`] takes a per-face
//! `wall_absorption: [Sample; 6]` for a [`ShoeboxRoom`], and
//! [`crate::room_acoustics`] turns total absorption into Sabine/Eyring
//! reverberation times. This module supplies the authoritative input: named
//! [`Material`]s carry the textbook octave-band energy-absorption spectra, and
//! [`MaterialAbsorption`] queries them at an arbitrary frequency (log-frequency
//! interpolation) or collapses them to a single broadband mean.
//!
//! # The model (classic architectural acoustics)
//!
//! Every material is described by eight energy-absorption coefficients in
//! `[0, 1]`, one per standard octave band centred at
//! [`OCTAVE_BAND_CENTERS`] (63 Hz up to 8000 Hz). A coefficient of `0` is a
//! perfect reflector, `1` a perfect absorber. Interpolation between bands is
//! linear in `log(frequency)` -- octave bands are equally spaced on a
//! logarithmic axis, which is where absorption data is measured and tabulated.
//! Queries below 63 Hz or above 8000 Hz clamp to the nearest measured band
//! (no extrapolation), and every returned value is clamped back into `[0, 1]`.
//!
//! The broadband mean is the unweighted arithmetic average of the eight bands.
//! It is a coarse single-number summary suitable for driving a broadband
//! reverberation-time estimate; a real design would weight by the source
//! spectrum, but the plain mean is deterministic and adequate here.
//!
//! # Control rate, not audio rate
//!
//! Everything is a small, `const`-fed scalar computation: no allocation, no
//! locking, no panics, no `f32` intrinsics (all logarithms route through
//! [`bevy_math::ops`]). The absorption tables are compile-time constants, so a
//! [`Material`] lookup is a branch and a copy.
//!
//! # Provenance
//!
//! The absorption coefficients are representative values from the public
//! building-acoustics literature -- the octave-band absorption tables reproduced
//! across standard texts such as H. Kuttruff's *Room Acoustics*, L. Beranek's
//! *Acoustics*, and M. Vorlander's *Auralization*. They are round, typical
//! figures for a material class, not a specific manufacturer datasheet. This
//! module is engine-agnostic and contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented acoustics
//! knowledge.

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// Number of octave bands in every material spectrum.
pub const OCTAVE_BAND_COUNT: usize = 8;

/// Standard octave-band centre frequencies in hertz, from 63 Hz to 8 kHz.
///
/// These are the ISO preferred octave centres over the range where surface
/// absorption is routinely tabulated. Bands are equally spaced on a
/// logarithmic frequency axis (each centre is twice the previous), which is why
/// [`MaterialAbsorption::at`] interpolates in `log(frequency)`.
pub const OCTAVE_BAND_CENTERS: [Sample; OCTAVE_BAND_COUNT] =
    [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

/// An octave-band energy-absorption spectrum in `[0, 1]` per band.
///
/// The eight coefficients line up with [`OCTAVE_BAND_CENTERS`]. Construct one
/// directly with [`MaterialAbsorption::new`] (which clamps each band into
/// `[0, 1]`) or obtain a stock spectrum from [`Material::absorption`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialAbsorption {
    /// Per-octave-band energy absorption, aligned with [`OCTAVE_BAND_CENTERS`].
    bands: [Sample; OCTAVE_BAND_COUNT],
}

impl MaterialAbsorption {
    /// Builds a spectrum from eight octave-band coefficients, clamping each
    /// into the physical `[0, 1]` energy-absorption range.
    #[must_use]
    pub fn new(bands: [Sample; OCTAVE_BAND_COUNT]) -> Self {
        let mut clamped = bands;
        for b in &mut clamped {
            *b = b.clamp(0.0, 1.0);
        }
        Self { bands: clamped }
    }

    /// Returns the raw per-band coefficients aligned with
    /// [`OCTAVE_BAND_CENTERS`].
    #[must_use]
    pub fn bands(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        self.bands
    }

    /// Absorption at an arbitrary frequency in hertz.
    ///
    /// The value is interpolated linearly in `log(frequency)` between the two
    /// bracketing octave-band centres. Frequencies at or below 63 Hz return the
    /// lowest band; frequencies at or above 8 kHz return the highest band (the
    /// spectrum is clamped, never extrapolated). A non-finite or non-positive
    /// frequency falls back to the lowest band. The result is clamped to
    /// `[0, 1]`.
    #[must_use]
    pub fn at(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return self.bands[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return self.bands[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, values) in
            OCTAVE_BAND_CENTERS.windows(2).zip(self.bands.windows(2))
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
        self.bands[last]
    }

    /// The unweighted arithmetic mean of the eight octave-band coefficients.
    ///
    /// A coarse single-number absorption suitable for driving a broadband
    /// reverberation-time estimate. Always in `[0, 1]`.
    #[must_use]
    pub fn broadband_mean(&self) -> Sample {
        let mut sum = 0.0;
        for b in &self.bands {
            sum += *b;
        }
        sum / OCTAVE_BAND_COUNT as Sample
    }

    /// Six identical face coefficients (one per [`ShoeboxRoom`] wall) taken
    /// from the broadband mean, ready to fill `ShoeboxRoom::wall_absorption`.
    ///
    /// [`ShoeboxRoom`]: crate::early_reflections::ShoeboxRoom
    #[must_use]
    pub fn uniform_walls(&self) -> [Sample; 6] {
        [self.broadband_mean(); 6]
    }

    /// Six identical face coefficients evaluated at a single frequency, ready
    /// to fill `ShoeboxRoom::wall_absorption` for a narrow-band estimate.
    #[must_use]
    pub fn uniform_walls_at(&self, freq_hz: Sample) -> [Sample; 6] {
        [self.at(freq_hz); 6]
    }
}

/// A named surface material with a stock octave-band absorption spectrum.
///
/// Values are representative of a material *class* (see the module Provenance),
/// not a specific product. Hard, dense surfaces (concrete, glass, water) absorb
/// little across the band; porous or draped surfaces (carpet, curtain, acoustic
/// tile) absorb strongly, especially at high frequencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Material {
    /// Bare, sealed structural concrete: near-rigid across the band.
    Concrete,
    /// Painted concrete or block: slightly lower than bare concrete.
    PaintedConcrete,
    /// Unglazed brickwork: hard, with a gentle high-frequency rise.
    Brick,
    /// Plaster on a solid masonry backing: hard, low absorption.
    Plaster,
    /// Wooden floor or thin panelling: mild low-frequency absorption from
    /// panel flexing.
    Wood,
    /// Heavy sealed plate glazing: low absorption throughout (thick glass does
    /// not flex like a thin resonant window pane).
    Glass,
    /// Heavy carpet on a concrete subfloor: strong high-frequency absorption.
    Carpet,
    /// Heavy draped curtain (roughly half a kilogram per square metre): strong
    /// mid/high absorption.
    HeavyCurtain,
    /// Suspended mineral-fibre acoustic ceiling tile: broadband absorber.
    AcousticTile,
    /// Open water surface (pool, tank): essentially rigid, negligible
    /// absorption.
    Water,
}

// Octave-band energy-absorption spectra, aligned with `OCTAVE_BAND_CENTERS`
// (63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz). Representative textbook
// figures for each material class (see module Provenance).
const CONCRETE: [Sample; OCTAVE_BAND_COUNT] =
    [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03];
const PAINTED_CONCRETE: [Sample; OCTAVE_BAND_COUNT] =
    [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02];
const BRICK: [Sample; OCTAVE_BAND_COUNT] =
    [0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07];
const PLASTER: [Sample; OCTAVE_BAND_COUNT] =
    [0.01, 0.013, 0.015, 0.02, 0.03, 0.04, 0.05, 0.05];
const WOOD: [Sample; OCTAVE_BAND_COUNT] =
    [0.10, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07];
const GLASS: [Sample; OCTAVE_BAND_COUNT] =
    [0.02, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02];
const CARPET: [Sample; OCTAVE_BAND_COUNT] =
    [0.02, 0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65];
const HEAVY_CURTAIN: [Sample; OCTAVE_BAND_COUNT] =
    [0.05, 0.07, 0.31, 0.49, 0.75, 0.70, 0.60, 0.60];
const ACOUSTIC_TILE: [Sample; OCTAVE_BAND_COUNT] =
    [0.25, 0.29, 0.55, 0.75, 0.85, 0.80, 0.75, 0.75];
const WATER: [Sample; OCTAVE_BAND_COUNT] =
    [0.008, 0.008, 0.01, 0.013, 0.015, 0.02, 0.025, 0.025];

impl Material {
    /// The stock octave-band absorption spectrum for this material.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::material_library::Material;
    ///
    /// let carpet = Material::Carpet.absorption();
    /// // Heavy carpet absorbs far more at 4 kHz than at 125 Hz.
    /// assert!(carpet.at(4000.0) > carpet.at(125.0));
    /// // A query at a band centre returns that band exactly.
    /// let bands = carpet.bands();
    /// assert!((carpet.at(1000.0) - bands[4]).abs() < 1e-4);
    /// ```
    #[must_use]
    pub fn absorption(self) -> MaterialAbsorption {
        let bands = match self {
            Material::Concrete => CONCRETE,
            Material::PaintedConcrete => PAINTED_CONCRETE,
            Material::Brick => BRICK,
            Material::Plaster => PLASTER,
            Material::Wood => WOOD,
            Material::Glass => GLASS,
            Material::Carpet => CARPET,
            Material::HeavyCurtain => HEAVY_CURTAIN,
            Material::AcousticTile => ACOUSTIC_TILE,
            Material::Water => WATER,
        };
        MaterialAbsorption { bands }
    }

    /// Broadband-mean absorption for this material.
    #[must_use]
    pub fn broadband_mean(self) -> Sample {
        self.absorption().broadband_mean()
    }

    /// Six identical broadband-mean face coefficients for a [`ShoeboxRoom`].
    ///
    /// [`ShoeboxRoom`]: crate::early_reflections::ShoeboxRoom
    #[must_use]
    pub fn uniform_walls(self) -> [Sample; 6] {
        self.absorption().uniform_walls()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MATERIALS: [Material; 10] = [
        Material::Concrete,
        Material::PaintedConcrete,
        Material::Brick,
        Material::Plaster,
        Material::Wood,
        Material::Glass,
        Material::Carpet,
        Material::HeavyCurtain,
        Material::AcousticTile,
        Material::Water,
    ];

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn every_band_is_in_unit_range() {
        for m in ALL_MATERIALS {
            for band in m.absorption().bands() {
                assert!((0.0..=1.0).contains(&band), "material {m:?} band {band}");
            }
        }
    }

    #[test]
    fn at_band_center_returns_that_band() {
        let a = Material::Carpet.absorption();
        let bands = a.bands();
        for (i, centre) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            assert!(
                approx(a.at(*centre), bands[i], 1e-4),
                "band {i} at {centre} Hz"
            );
        }
    }

    #[test]
    fn interpolation_stays_between_neighbours() {
        let a = Material::AcousticTile.absorption();
        // 707 Hz sits between the 500 and 1000 Hz bands.
        let v = a.at(707.0);
        let lo = a.at(500.0);
        let hi = a.at(1000.0);
        let (min, max) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        assert!(v >= min - 1e-6 && v <= max + 1e-6, "v={v} lo={lo} hi={hi}");
        // Geometric midpoint of an octave interpolates near the arithmetic
        // midpoint of the two band values (log-linear).
        let mid = 0.5 * (lo + hi);
        assert!(approx(v, mid, 0.02), "v={v} mid={mid}");
    }

    #[test]
    fn below_and_above_range_clamp_to_endpoints() {
        let a = Material::Wood.absorption();
        let bands = a.bands();
        assert!(approx(a.at(20.0), bands[0], 1e-6));
        assert!(approx(a.at(10.0), bands[0], 1e-6));
        assert!(approx(a.at(16000.0), bands[OCTAVE_BAND_COUNT - 1], 1e-6));
    }

    #[test]
    fn absorbers_beat_reflectors_broadband() {
        let carpet = Material::Carpet.broadband_mean();
        let curtain = Material::HeavyCurtain.broadband_mean();
        let tile = Material::AcousticTile.broadband_mean();
        let concrete = Material::Concrete.broadband_mean();
        let glass = Material::Glass.broadband_mean();
        let water = Material::Water.broadband_mean();
        for absorber in [carpet, curtain, tile] {
            for reflector in [concrete, glass, water] {
                assert!(absorber > reflector, "{absorber} !> {reflector}");
            }
        }
    }

    #[test]
    fn glass_and_water_low_frequency_absorption_is_small() {
        assert!(Material::Glass.absorption().at(125.0) < 0.1);
        assert!(Material::Water.absorption().at(125.0) < 0.05);
        assert!(Material::Water.absorption().at(63.0) < 0.05);
    }

    #[test]
    fn broadband_mean_matches_manual_average() {
        let a = Material::Brick.absorption();
        let bands = a.bands();
        let mut sum = 0.0;
        for b in bands {
            sum += b;
        }
        let manual = sum / OCTAVE_BAND_COUNT as Sample;
        assert!(approx(a.broadband_mean(), manual, 1e-6));
    }

    #[test]
    fn constructor_clamps_out_of_range_bands() {
        let a = MaterialAbsorption::new([2.0, -1.0, 0.5, 1.5, -0.2, 0.3, 10.0, 0.0]);
        let bands = a.bands();
        assert!(approx(bands[0], 1.0, 1e-6));
        assert!(approx(bands[1], 0.0, 1e-6));
        assert!(approx(bands[2], 0.5, 1e-6));
        assert!(approx(bands[3], 1.0, 1e-6));
        assert!(approx(bands[4], 0.0, 1e-6));
        assert!(approx(bands[6], 1.0, 1e-6));
    }

    #[test]
    fn all_zero_material_has_zero_mean() {
        let a = MaterialAbsorption::new([0.0; OCTAVE_BAND_COUNT]);
        assert!(approx(a.broadband_mean(), 0.0, 1e-9));
        assert!(approx(a.at(1000.0), 0.0, 1e-9));
    }

    #[test]
    fn uniform_walls_are_six_equal_broadband_means() {
        let m = Material::Carpet;
        let walls = m.uniform_walls();
        let mean = m.broadband_mean();
        assert_eq!(walls.len(), 6);
        for w in walls {
            assert!(approx(w, mean, 1e-6));
        }
    }

    #[test]
    fn uniform_walls_at_uses_single_frequency() {
        let a = Material::HeavyCurtain.absorption();
        let walls = a.uniform_walls_at(1000.0);
        for w in walls {
            assert!(approx(w, a.at(1000.0), 1e-6));
        }
    }

    #[test]
    fn degenerate_frequency_is_safe() {
        let a = Material::Concrete.absorption();
        // Non-positive and non-finite frequencies fall back to the lowest band.
        let _ = a.at(0.0);
        let _ = a.at(-100.0);
        let _ = a.at(Sample::NAN);
        let _ = a.at(Sample::INFINITY);
        assert!(approx(a.at(0.0), a.bands()[0], 1e-6));
    }

    #[test]
    fn hard_room_reverberates_longer_than_soft_room() {
        use crate::early_reflections::ShoeboxRoom;
        use crate::room_acoustics::RoomAcoustics;
        use bevy_math::Vec3;

        let corner = Vec3::new(10.0, 4.0, 8.0);
        let hard = ShoeboxRoom::new(Vec3::ZERO, corner, Material::Concrete.uniform_walls());
        let soft = ShoeboxRoom::new(Vec3::ZERO, corner, Material::AcousticTile.uniform_walls());
        let rt_hard = RoomAcoustics::from_shoebox(&hard).rt60_sabine();
        let rt_soft = RoomAcoustics::from_shoebox(&soft).rt60_sabine();
        assert!(rt_hard > rt_soft, "hard {rt_hard} !> soft {rt_soft}");
    }
}
