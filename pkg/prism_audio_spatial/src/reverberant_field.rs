//! Steady-state reverberant field energy and direct-to-reverberant ratio.
//!
//! In a room excited by a steady point source, the sound energy density at a
//! listener is the sum of a direct component that falls off with distance and a
//! reverberant (diffuse) component that is nearly uniform throughout the room.
//! This module is the classic room-constant description of that split; it is a
//! control-rate estimator and performs no per-sample DSP.
//!
//! # Room constant
//!
//! The room constant collects the total absorbing power of the boundaries,
//!
//! `R = S * a_bar / (1 - a_bar)`
//!
//! where `S` is the total interior surface area in square metres and `a_bar` in
//! `(0, 1)` is the mean absorption coefficient. As `a_bar` approaches `1` the
//! room becomes fully absorbing (free field) and `R` grows without bound, so
//! the denominator is guarded.
//!
//! # Energy split
//!
//! For a point source of acoustic power `W` and on-axis directivity factor `Q`,
//! at distance `r` the steady-state energy densities are proportional to
//!
//! - direct: `Q / (4 * PI * r^2)`
//! - reverberant (diffuse, distance-independent): `4 / R`
//!
//! The proportionality constant (which carries `W` and the speed of sound)
//! cancels in every ratio this module reports, so the source power need never
//! be supplied. The direct-to-reverberant ratio is
//!
//! `DRR(r) = (Q / (4 * PI * r^2)) / (4 / R) = Q * R / (16 * PI * r^2)`
//!
//! and the critical distance where `DRR = 1` is
//!
//! `r_c = sqrt(Q * R / (16 * PI))`.
//!
//! # Relationship to [`crate::room_acoustics`]
//!
//! [`crate::room_acoustics::critical_distance`] estimates the same critical
//! distance from volume and reverberation time via the Sabine approximation
//! `r_c = 0.057 * sqrt(V / RT60)`. This module instead parameterises the
//! critical distance directly by the room constant `R` and the source
//! directivity factor `Q`, which is the equivalent formulation when the
//! absorbing area rather than the reverberation time is known. Use whichever
//! inputs are available; they describe the same physical quantity.
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars; there is no heap allocation, no
//! locking, and no panicking. Degenerate geometry (a non-positive area, a mean
//! absorption at `0` or `1`) and non-finite inputs return safe finite values.
//! All transcendental math routes through [`bevy_math::ops`], never through
//! `f32` intrinsics.
//!
//! # Provenance
//!
//! This is the textbook steady-state diffuse-field description of a room: the
//! room constant, the direct-plus-reverberant energy split, the
//! direct-to-reverberant ratio, and the critical distance as presented in
//! L. Beranek's *Acoustics* and H. Kuttruff's *Room Acoustics*. This module is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics knowledge.

use bevy_math::ops;

use core::f32::consts::{LN_10, PI};

use prism_audio_core::math::Sample;

/// Smallest divisor used to keep ratios finite for degenerate inputs.
const MIN_DIVISOR: Sample = 1e-9;

/// Largest mean absorption used when computing the room constant, keeping `R`
/// finite as `a_bar` approaches `1` (a fully absorbing room).
const MAX_MEAN_ABSORPTION: Sample = 0.999_999;

/// A steady-state reverberant field described by its room constant.
///
/// Build one with [`ReverberantField::from_surface_absorption`], then query the
/// direct, reverberant, and total energy densities, the direct-to-reverberant
/// ratio, and the critical distance for a source of directivity factor `Q`.
///
/// The directivity factor pairs naturally with
/// [`crate::source_directivity::SourceDirectivity::directivity_factor`], which
/// supplies a per-octave-band `Q`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReverberantField {
    /// The room constant `R = S * a_bar / (1 - a_bar)` in square metres.
    room_constant: Sample,
}

impl ReverberantField {
    /// Builds a reverberant field from the total interior surface area (square
    /// metres) and the mean absorption coefficient.
    ///
    /// The surface area is clamped to be non-negative and the mean absorption
    /// to `(0, MAX)` just below `1`; non-finite inputs fall back to safe
    /// values, yielding a finite room constant.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::reverberant_field::ReverberantField;
    ///
    /// // S = 100 m^2, a_bar = 0.5 -> R = 100 * 0.5 / 0.5 = 100.
    /// let field = ReverberantField::from_surface_absorption(100.0, 0.5);
    /// assert!((field.room_constant() - 100.0).abs() < 1e-3);
    /// ```
    #[must_use]
    pub fn from_surface_absorption(surface_area_m2: Sample, mean_absorption: Sample) -> Self {
        let s = if surface_area_m2.is_finite() {
            surface_area_m2.max(0.0)
        } else {
            0.0
        };
        let a = if mean_absorption.is_finite() {
            mean_absorption.clamp(0.0, MAX_MEAN_ABSORPTION)
        } else {
            0.0
        };
        let denom = (1.0 - a).max(MIN_DIVISOR);
        let room_constant = s * a / denom;
        Self { room_constant }
    }

    /// Builds a reverberant field directly from a known room constant (square
    /// metres). Non-finite or negative values become `0`.
    #[must_use]
    pub fn from_room_constant(room_constant: Sample) -> Self {
        let r = if room_constant.is_finite() {
            room_constant.max(0.0)
        } else {
            0.0
        };
        Self { room_constant: r }
    }

    /// The room constant `R` in square metres.
    #[must_use]
    pub fn room_constant(&self) -> Sample {
        self.room_constant
    }

    /// The relative direct energy density `Q / (4 * PI * r^2)` at distance `r`
    /// for a source of directivity factor `q`.
    ///
    /// The distance is guarded against zero; non-finite inputs return `0`.
    #[must_use]
    pub fn direct_energy(&self, q: Sample, r: Sample) -> Sample {
        if !q.is_finite() || !r.is_finite() {
            return 0.0;
        }
        let r2 = (r * r).max(MIN_DIVISOR);
        q / (4.0 * PI * r2)
    }

    /// The relative reverberant energy density `4 / R`, independent of
    /// distance.
    ///
    /// Returns `0` for a degenerate (zero) room constant.
    #[must_use]
    pub fn reverberant_energy(&self) -> Sample {
        if self.room_constant <= MIN_DIVISOR {
            0.0
        } else {
            4.0 / self.room_constant
        }
    }

    /// The total relative energy density: direct plus reverberant.
    #[must_use]
    pub fn total_energy(&self, q: Sample, r: Sample) -> Sample {
        self.direct_energy(q, r) + self.reverberant_energy()
    }

    /// The linear direct-to-reverberant ratio
    /// `DRR(r) = Q * R / (16 * PI * r^2)` at distance `r`.
    ///
    /// Equals `1` at the critical distance, falls with the square of distance,
    /// and rises with the directivity factor. Non-finite inputs return `0`; a
    /// degenerate room constant (no reverberation) returns a large finite
    /// ratio.
    #[must_use]
    pub fn direct_to_reverberant_ratio(&self, q: Sample, r: Sample) -> Sample {
        let reverb = self.reverberant_energy();
        let direct = self.direct_energy(q, r);
        if reverb <= MIN_DIVISOR {
            direct / MIN_DIVISOR
        } else {
            direct / reverb
        }
    }

    /// The direct-to-reverberant ratio in decibels, `10 * log10(DRR)`.
    ///
    /// A ratio at or below zero maps to a large negative floor rather than
    /// negative infinity.
    #[must_use]
    pub fn direct_to_reverberant_ratio_db(&self, q: Sample, r: Sample) -> Sample {
        let drr = self.direct_to_reverberant_ratio(q, r);
        if drr > MIN_DIVISOR {
            10.0 * ops::ln(drr) / LN_10
        } else {
            10.0 * ops::ln(MIN_DIVISOR) / LN_10
        }
    }

    /// The critical distance `r_c = sqrt(Q * R / (16 * PI))`, where the direct
    /// and reverberant energy densities are equal.
    ///
    /// See the module documentation for the relationship to
    /// [`crate::room_acoustics::critical_distance`], which computes the same
    /// quantity from volume and reverberation time. Non-finite or negative `q`
    /// is treated as `0`.
    #[must_use]
    pub fn critical_distance(&self, q: Sample) -> Sample {
        let qq = if q.is_finite() { q.max(0.0) } else { 0.0 };
        ops::sqrt(qq * self.room_constant / (16.0 * PI))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn room_constant_matches_closed_form() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.5);
        assert!(approx(field.room_constant(), 100.0, 1e-3));
        // S = 200, a = 0.2 -> R = 200 * 0.2 / 0.8 = 50.
        let f2 = ReverberantField::from_surface_absorption(200.0, 0.2);
        assert!(approx(f2.room_constant(), 50.0, 1e-3));
    }

    #[test]
    fn absorption_near_one_gives_large_finite_constant() {
        let field = ReverberantField::from_surface_absorption(100.0, 1.0);
        assert!(field.room_constant().is_finite());
        assert!(field.room_constant() > 1.0e6);
    }

    #[test]
    fn drr_is_unity_at_critical_distance() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.3);
        let q = 1.0;
        let rc = field.critical_distance(q);
        assert!(rc > 0.0);
        assert!(approx(field.direct_to_reverberant_ratio(q, rc), 1.0, 1e-3));
    }

    #[test]
    fn drr_falls_with_distance_squared() {
        let field = ReverberantField::from_surface_absorption(120.0, 0.4);
        let q = 1.0;
        let d1 = field.direct_to_reverberant_ratio(q, 1.0);
        let d2 = field.direct_to_reverberant_ratio(q, 2.0);
        // Doubling distance quarters the ratio.
        assert!(approx(d1 / d2, 4.0, 1e-3), "ratio {}", d1 / d2);
    }

    #[test]
    fn drr_db_matches_ten_log10() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.5);
        let q = 2.0;
        let r = 1.5;
        let lin = field.direct_to_reverberant_ratio(q, r);
        let db = field.direct_to_reverberant_ratio_db(q, r);
        let expected = 10.0 * ops::ln(lin) / LN_10;
        assert!(approx(db, expected, 1e-3));
    }

    #[test]
    fn direct_field_falls_six_db_per_distance_double() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.5);
        let q = 1.0;
        let e1 = field.direct_energy(q, 1.0);
        let e2 = field.direct_energy(q, 2.0);
        // Inverse-square law: a factor of four, i.e. about 6 dB.
        assert!(approx(e1 / e2, 4.0, 1e-3));
        let db = 10.0 * ops::ln(e1 / e2) / LN_10;
        assert!(approx(db, 6.0206, 1e-2));
    }

    #[test]
    fn reverberant_field_is_distance_independent() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.5);
        let a = field.reverberant_energy();
        let b = field.reverberant_energy();
        assert!(approx(a, b, 1e-9));
        // R = 100 -> reverberant energy = 4 / 100 = 0.04.
        assert!(approx(a, 0.04, 1e-6));
    }

    #[test]
    fn higher_q_raises_drr_and_pushes_critical_distance_out() {
        let field = ReverberantField::from_surface_absorption(100.0, 0.4);
        let r = 2.0;
        let low = field.direct_to_reverberant_ratio(1.0, r);
        let high = field.direct_to_reverberant_ratio(3.0, r);
        assert!(high > low);
        assert!(field.critical_distance(3.0) > field.critical_distance(1.0));
    }

    #[test]
    fn total_energy_is_direct_plus_reverberant() {
        let field = ReverberantField::from_surface_absorption(90.0, 0.35);
        let q = 1.5;
        let r = 1.2;
        let total = field.total_energy(q, r);
        let sum = field.direct_energy(q, r) + field.reverberant_energy();
        assert!(approx(total, sum, 1e-6));
    }

    #[test]
    fn degenerate_and_non_finite_inputs_are_safe() {
        // Zero surface area -> zero room constant -> no reverberant field.
        let none = ReverberantField::from_surface_absorption(0.0, 0.5);
        assert!(approx(none.room_constant(), 0.0, 1e-9));
        assert!(approx(none.reverberant_energy(), 0.0, 1e-9));
        // Zero absorption -> zero room constant.
        let dead = ReverberantField::from_surface_absorption(100.0, 0.0);
        assert!(approx(dead.room_constant(), 0.0, 1e-9));
        // Non-finite inputs stay finite.
        let weird = ReverberantField::from_surface_absorption(Sample::NAN, Sample::NAN);
        assert!(weird.room_constant().is_finite());
        let field = ReverberantField::from_surface_absorption(100.0, 0.5);
        assert!(field.direct_energy(Sample::NAN, 1.0).is_finite());
        assert!(field.direct_to_reverberant_ratio(1.0, 0.0).is_finite());
        assert!(field.direct_to_reverberant_ratio_db(1.0, 0.0).is_finite());
    }

    #[test]
    fn from_room_constant_round_trips() {
        let field = ReverberantField::from_room_constant(250.0);
        assert!(approx(field.room_constant(), 250.0, 1e-6));
        let clamped = ReverberantField::from_room_constant(-5.0);
        assert!(approx(clamped.room_constant(), 0.0, 1e-9));
    }
}
