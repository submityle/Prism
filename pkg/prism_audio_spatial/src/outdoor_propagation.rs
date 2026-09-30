//! ISO 9613-2 outdoor propagation budget aggregator.
//!
//! Outdoors, the attenuation of sound between a point source and a receiver is
//! the sum of several independent, additive terms measured in decibels. This
//! module is a control-rate one-stop evaluator that combines the four terms
//! already implemented elsewhere in this crate into a single per-octave-band
//! total attenuation (and the matching linear gains). It performs no per-sample
//! DSP and, crucially, re-implements none of the underlying physics: it is a
//! pure `DRY` combiner over existing modules.
//!
//! # The budget
//!
//! Following **ISO 9613-2:1996, section 7**, the octave-band attenuation is
//!
//! `A(f) = A_div + A_atm(f) + A_gr(f) + A_bar(f)`,
//!
//! summed in decibels (positive = attenuation, negative = gain), where:
//!
//! - `A_div` is geometric divergence, taken from [`crate::attenuation`]. The
//!   linear distance gain is converted to a decibel attenuation; it is
//!   frequency-independent.
//! - `A_atm(f)` is atmospheric absorption, taken from [`crate::air`] as the
//!   per-metre absorption coefficient at each octave-band centre multiplied by
//!   the propagation distance.
//! - `A_gr(f)` is the ground effect, taken from
//!   [`crate::ground_effect::GroundEffect`]; it may be negative over hard
//!   ground (a constructive reflection gain).
//! - `A_bar(f)` is the barrier / screen diffraction loss, taken from
//!   [`crate::diffraction::Diffraction`]. It is optional: a line-of-sight path
//!   with no obstruction contributes `0`.
//!
//! # Relationship
//!
//! This module owns no acoustics of its own; it is the assembler that sits on
//! top of four component modules, each of which is the authoritative source of
//! its term:
//!
//! - divergence: [`crate::attenuation`] (`Attenuation::gain`),
//! - atmospheric absorption: [`crate::air`] (`absorption_db_per_metre`),
//! - ground effect: [`crate::ground_effect`] (`GroundEffect::band_attenuations_db`),
//! - barrier diffraction: [`crate::diffraction`] (`Diffraction::band_losses_db`).
//!
//! All four share the eight octave bands of
//! [`crate::material_library::OCTAVE_BAND_CENTERS`], so the terms add band by
//! band without resampling.
//!
//! # Control rate, not audio rate
//!
//! Every query operates on stack scalars and fixed-size arrays; there is no
//! heap allocation, no locking, and no panicking. Non-finite and degenerate
//! inputs fall back to safe finite values, and the divergence term is capped at
//! a finite ceiling so a fully attenuated distance model never produces an
//! infinite total. All transcendental math routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is the textbook additive outdoor-propagation budget of ISO 9613-2:1996,
//! "Acoustics -- Attenuation of sound during propagation outdoors -- Part 2:
//! General method of calculation". This module is engine-agnostic and contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it merely composes terms defined
//! by that publicly documented acoustics standard.

use bevy_math::ops;

use prism_audio_core::math::{Sample, db_to_linear, linear_to_db};

use crate::air::{AtmosphericConditions, absorption_db_per_metre};
use crate::attenuation::Attenuation;
use crate::diffraction::Diffraction;
use crate::ground_effect::GroundEffect;
use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};

/// Smallest divisor used to keep ratios finite for degenerate inputs.
const MIN_DIVISOR: Sample = 1e-9;

/// Finite ceiling (in decibels) for the divergence term, used when the distance
/// model collapses the gain to silence (which would otherwise be an infinite
/// attenuation).
const MAX_DIVERGENCE_DB: Sample = 200.0;

/// An ISO 9613-2 outdoor propagation budget: a distance plus the four additive
/// attenuation terms (divergence, atmospheric absorption, ground effect, and an
/// optional barrier).
///
/// Build one with [`OutdoorPropagation::new`] (with a barrier) or
/// [`OutdoorPropagation::line_of_sight`] (no barrier), then query the total
/// per-octave-band attenuation in decibels, the matching linear gains, or a
/// broadband value at an arbitrary frequency.
///
/// The octave bands line up with [`OCTAVE_BAND_CENTERS`], so the total can be
/// applied to an octave-band graphic equaliser directly.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OutdoorPropagation {
    /// Straight-line source-to-receiver distance in metres (non-negative).
    distance_m: Sample,
    /// Geometric divergence descriptor.
    attenuation: Attenuation,
    /// Atmosphere used for the absorption term.
    conditions: AtmosphericConditions,
    /// Ground-effect term.
    ground: GroundEffect,
    /// Optional barrier / screen diffraction term (`None` = line of sight).
    barrier: Option<Diffraction>,
}

/// Clamps a distance to be finite and non-negative.
#[inline]
#[must_use]
fn clamp_distance(d: Sample) -> Sample {
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

impl OutdoorPropagation {
    /// Builds an outdoor propagation budget with an explicit optional barrier.
    ///
    /// The distance is clamped to be finite and non-negative; the component
    /// descriptors are used as given (each already sanitises its own inputs).
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::attenuation::{Attenuation, DistanceModel};
    /// use prism_audio_spatial::air::AtmosphericConditions;
    /// use prism_audio_spatial::ground_effect::GroundEffect;
    /// use prism_audio_spatial::outdoor_propagation::OutdoorPropagation;
    ///
    /// let divergence = Attenuation::new(DistanceModel::Inverse, 1.0, 10_000.0, 1.0);
    /// let air = AtmosphericConditions::default();
    /// let ground = GroundEffect::from_geometry(2.0, 2.0, 100.0, 1.0);
    /// let field = OutdoorPropagation::line_of_sight(100.0, divergence, air, ground);
    /// // High frequencies are attenuated more than low frequencies far out.
    /// assert!(field.total_attenuation_db(7) > field.total_attenuation_db(0));
    /// ```
    #[must_use]
    pub fn new(
        distance_m: Sample,
        attenuation: Attenuation,
        conditions: AtmosphericConditions,
        ground: GroundEffect,
        barrier: Option<Diffraction>,
    ) -> Self {
        Self {
            distance_m: clamp_distance(distance_m),
            attenuation,
            conditions,
            ground,
            barrier,
        }
    }

    /// Builds a line-of-sight budget (no barrier term).
    ///
    /// Equivalent to [`OutdoorPropagation::new`] with `barrier = None`.
    #[must_use]
    pub fn line_of_sight(
        distance_m: Sample,
        attenuation: Attenuation,
        conditions: AtmosphericConditions,
        ground: GroundEffect,
    ) -> Self {
        Self::new(distance_m, attenuation, conditions, ground, None)
    }

    /// The straight-line source-to-receiver distance in metres.
    #[must_use]
    pub fn distance(&self) -> Sample {
        self.distance_m
    }

    /// Whether a barrier diffraction term is present.
    #[must_use]
    pub fn has_barrier(&self) -> bool {
        self.barrier.is_some()
    }

    /// The ground-effect component.
    #[must_use]
    pub fn ground_effect(&self) -> &GroundEffect {
        &self.ground
    }

    /// The barrier diffraction component, if any.
    #[must_use]
    pub fn barrier(&self) -> Option<&Diffraction> {
        self.barrier.as_ref()
    }

    /// The geometric divergence attenuation in decibels (frequency-independent).
    ///
    /// This converts the linear distance gain from [`crate::attenuation`] into a
    /// decibel attenuation, `A_div = -20 * log10(gain)`. When the distance model
    /// collapses the gain to silence the result is capped at a finite ceiling so
    /// the total budget stays finite.
    #[must_use]
    pub fn divergence_db(&self) -> Sample {
        let gain = self.attenuation.gain(self.distance_m);
        let a = -linear_to_db(gain);
        if a.is_finite() { a } else { MAX_DIVERGENCE_DB }
    }

    /// The atmospheric absorption attenuation in decibels for octave band
    /// `band_index`.
    ///
    /// This is the per-metre absorption coefficient at the band centre times the
    /// propagation distance. The index is clamped to the last band.
    #[must_use]
    pub fn atmospheric_db(&self, band_index: usize) -> Sample {
        let band = band_index.min(OCTAVE_BAND_COUNT - 1);
        let alpha = absorption_db_per_metre(&self.conditions, OCTAVE_BAND_CENTERS[band]);
        alpha * self.distance_m
    }

    /// The barrier diffraction attenuation in decibels for octave band
    /// `band_index` (0 when there is no barrier).
    #[must_use]
    pub fn barrier_db(&self, band_index: usize) -> Sample {
        match &self.barrier {
            Some(d) => d.insertion_loss_db(band_index),
            None => 0.0,
        }
    }

    /// The total outdoor attenuation in decibels for octave band `band_index`
    /// (positive = attenuation, negative = net gain).
    ///
    /// The index is clamped to the last band, so out-of-range indices are safe.
    #[must_use]
    pub fn total_attenuation_db(&self, band_index: usize) -> Sample {
        let band = band_index.min(OCTAVE_BAND_COUNT - 1);
        let a_div = self.divergence_db();
        let a_atm = self.atmospheric_db(band);
        let a_gr = self.ground.attenuation_db(band);
        let a_bar = self.barrier_db(band);
        a_div + a_atm + a_gr + a_bar
    }

    /// The total outdoor attenuation in decibels for every octave band.
    #[must_use]
    pub fn band_attenuations_db(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [0.0; OCTAVE_BAND_COUNT];
        for (band, slot) in out.iter_mut().enumerate() {
            *slot = self.total_attenuation_db(band);
        }
        out
    }

    /// The total linear gain `10^(-A/20)` for every octave band.
    ///
    /// The gain exceeds `1` on bands where the ground reflection is
    /// constructive enough to overcome the other terms.
    #[must_use]
    pub fn band_gains(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [1.0; OCTAVE_BAND_COUNT];
        for (band, slot) in out.iter_mut().enumerate() {
            *slot = db_to_linear(-self.total_attenuation_db(band));
        }
        out
    }

    /// The broadband total attenuation in decibels at `freq_hz`, interpolated
    /// linearly in `log(frequency)` between the two bracketing octave-band
    /// values.
    ///
    /// Frequencies at or below 63 Hz return the lowest band; frequencies at or
    /// above 8 kHz return the highest band (the spectrum is clamped, never
    /// extrapolated). At a band centre it equals that band's value.
    #[must_use]
    pub fn broadband_attenuation_db(&self, freq_hz: Sample) -> Sample {
        let values = self.band_attenuations_db();
        if !freq_hz.is_finite() || freq_hz <= OCTAVE_BAND_CENTERS[0] {
            return values[0];
        }
        let last = OCTAVE_BAND_COUNT - 1;
        if freq_hz >= OCTAVE_BAND_CENTERS[last] {
            return values[last];
        }
        let log_f = ops::ln(freq_hz);
        for (centres, vals) in OCTAVE_BAND_CENTERS.windows(2).zip(values.windows(2)) {
            let c_hi = centres[1];
            if freq_hz <= c_hi {
                let log_lo = ops::ln(centres[0]);
                let log_hi = ops::ln(c_hi);
                let span = log_hi - log_lo;
                let t = if span > MIN_DIVISOR {
                    (log_f - log_lo) / span
                } else {
                    0.0
                };
                return vals[0] + (vals[1] - vals[0]) * t;
            }
        }
        values[last]
    }

    /// The broadband linear gain at `freq_hz`, i.e.
    /// `10^(-broadband_attenuation_db / 20)`.
    #[must_use]
    pub fn broadband_gain(&self, freq_hz: Sample) -> Sample {
        db_to_linear(-self.broadband_attenuation_db(freq_hz))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attenuation::DistanceModel;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn divergence() -> Attenuation {
        Attenuation::new(DistanceModel::Inverse, 1.0, 10_000.0, 1.0)
    }

    fn field(distance: Sample, g: Sample, barrier: Option<Diffraction>) -> OutdoorPropagation {
        let ground = GroundEffect::from_geometry(2.0, 2.0, distance, g);
        OutdoorPropagation::new(
            distance,
            divergence(),
            AtmosphericConditions::default(),
            ground,
            barrier,
        )
    }

    #[test]
    fn total_equals_component_sum() {
        let f = field(100.0, 1.0, None);
        for band in 0..OCTAVE_BAND_COUNT {
            let manual =
                f.divergence_db() + f.atmospheric_db(band) + f.ground_effect().attenuation_db(band);
            assert!(approx(f.total_attenuation_db(band), manual, 1e-4), "band {band}");
        }
    }

    #[test]
    fn total_with_barrier_equals_component_sum() {
        let barrier = Diffraction::from_path_difference(0.5, 343.0);
        let f = field(100.0, 0.5, Some(barrier));
        for band in 0..OCTAVE_BAND_COUNT {
            let manual = f.divergence_db()
                + f.atmospheric_db(band)
                + f.ground_effect().attenuation_db(band)
                + barrier.insertion_loss_db(band);
            assert!(approx(f.total_attenuation_db(band), manual, 1e-4), "band {band}");
        }
    }

    #[test]
    fn no_barrier_contributes_zero() {
        let f = field(100.0, 0.5, None);
        assert!(!f.has_barrier());
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(approx(f.barrier_db(band), 0.0, 1e-9), "band {band}");
        }
    }

    #[test]
    fn barrier_adds_positive_loss() {
        let bare = field(100.0, 0.5, None);
        let barrier = Diffraction::from_path_difference(1.0, 343.0);
        let blocked = field(100.0, 0.5, Some(barrier));
        assert!(blocked.has_barrier());
        // Barrier diffraction shadows the high band the most.
        assert!(blocked.total_attenuation_db(7) > bare.total_attenuation_db(7));
    }

    #[test]
    fn band_gains_match_attenuations() {
        let f = field(120.0, 0.7, None);
        let atts = f.band_attenuations_db();
        let gains = f.band_gains();
        for (att, gain) in atts.iter().zip(gains.iter()) {
            assert!(approx(*gain, db_to_linear(-att), 1e-6));
        }
    }

    #[test]
    fn divergence_grows_with_distance() {
        let near = field(50.0, 0.5, None);
        let far = field(500.0, 0.5, None);
        assert!(far.divergence_db() > near.divergence_db());
    }

    #[test]
    fn divergence_is_finite_even_at_silence() {
        // A linear model reaches zero gain at the maximum distance; the
        // divergence term must stay finite (capped), not become infinite.
        let att = Attenuation::new(DistanceModel::Linear, 1.0, 100.0, 1.0);
        let ground = GroundEffect::from_geometry(2.0, 2.0, 100.0, 0.5);
        let f = OutdoorPropagation::line_of_sight(
            100.0,
            att,
            AtmosphericConditions::default(),
            ground,
        );
        assert!(f.divergence_db().is_finite());
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(f.total_attenuation_db(band).is_finite(), "band {band}");
        }
    }

    #[test]
    fn atmospheric_grows_with_frequency() {
        let f = field(200.0, 0.5, None);
        assert!(f.atmospheric_db(7) > f.atmospheric_db(0));
    }

    #[test]
    fn atmospheric_scales_with_distance() {
        let near = field(100.0, 0.5, None);
        let far = field(300.0, 0.5, None);
        // Same atmosphere, three times the distance -> three times the loss.
        let ratio = far.atmospheric_db(7) / near.atmospheric_db(7).max(MIN_DIVISOR);
        assert!(approx(ratio, 3.0, 1e-2));
    }

    #[test]
    fn zero_distance_zero_divergence_and_atmospheric() {
        let f = field(0.0, 0.5, None);
        // At (clamped) zero distance the inverse model is unity gain -> 0 dB,
        // and the atmospheric term is zero.
        assert!(approx(f.divergence_db(), 0.0, 1e-4));
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(approx(f.atmospheric_db(band), 0.0, 1e-6), "band {band}");
        }
    }

    #[test]
    fn broadband_at_band_centre_equals_band_value() {
        let f = field(150.0, 1.0, None);
        for (band, &freq) in OCTAVE_BAND_CENTERS.iter().enumerate() {
            let via_band = f.total_attenuation_db(band);
            let via_broadband = f.broadband_attenuation_db(freq);
            assert!(approx(via_band, via_broadband, 1e-2), "band {band}");
        }
    }

    #[test]
    fn broadband_interpolates_between_bands() {
        let f = field(150.0, 1.0, None);
        let lo = f.total_attenuation_db(1);
        let hi = f.total_attenuation_db(2);
        let mid = f.broadband_attenuation_db(177.0);
        let (min, max) = if lo < hi { (lo, hi) } else { (hi, lo) };
        assert!(mid > min && mid < max, "mid {mid} lo {lo} hi {hi}");
    }

    #[test]
    fn broadband_clamps_endpoints() {
        let f = field(150.0, 1.0, None);
        let low = f.broadband_attenuation_db(20.0);
        let high = f.broadband_attenuation_db(20_000.0);
        assert!(approx(low, f.total_attenuation_db(0), 1e-3));
        assert!(approx(high, f.total_attenuation_db(OCTAVE_BAND_COUNT - 1), 1e-3));
    }

    #[test]
    fn non_finite_and_degenerate_inputs_are_safe() {
        let ground = GroundEffect::from_geometry(Sample::NAN, 2.0, Sample::NAN, Sample::NAN);
        let f = OutdoorPropagation::new(
            Sample::NAN,
            divergence(),
            AtmosphericConditions::default(),
            ground,
            Some(Diffraction::from_path_difference(Sample::INFINITY, Sample::NAN)),
        );
        assert!(f.distance().is_finite());
        for band in 0..OCTAVE_BAND_COUNT {
            assert!(f.total_attenuation_db(band).is_finite(), "band {band}");
            assert!(f.band_gains()[band].is_finite(), "band {band}");
        }
        assert!(f.broadband_gain(Sample::INFINITY).is_finite());
    }
}
