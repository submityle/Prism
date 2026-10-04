//! Bolton (1986) relative dilatancy index `I_R` for granular soils.
//!
//! Bolton's empirical framework correlates the peak strength and dilatancy of
//! sands and gravels to a single dimensionless state variable, the relative
//! dilatancy index:
//!
//! ```text
//! I_R = I_D * (Q - ln p') - R
//! ```
//!
//! where `I_D` is the relative density (expressed as a fraction in `0..=1`),
//! `p'` is the mean effective stress at failure (kPa), and `Q`, `R` are
//! dimensionless fitting constants. For quartz and feldspar sands Bolton
//! recommends `Q = 10` and `R = 1`.
//!
//! The index drives three widely used correlations:
//!
//! * triaxial peak friction gain: `phi_max - phi_cv = 3 * I_R` (degrees),
//! * plane-strain peak friction gain: `phi_max - phi_cv = 5 * I_R` (degrees),
//! * maximum rate of dilation: `(-d_eps_v / d_eps_1)_max = 0.3 * I_R`.
//!
//! The framework is calibrated for `0 <= I_R <= 4`; outside this range the
//! correlations lose physical meaning (negative values imply a contractive,
//! non-dilatant response and are conventionally truncated to zero).
//!
//! Reference: M. D. Bolton, "The strength and dilatancy of sands",
//! Geotechnique 36(1), 1986, pp. 65-78.
//!
//! This module is a self-contained diagnostic primitive. It performs no
//! coupling with the simulation pipeline and holds no simulation state.

/// Triaxial peak-friction multiplier `phi_max - phi_cv = 3 * I_R` (degrees).
const TRIAXIAL_FRICTION_GAIN: f32 = 3.0;

/// Plane-strain peak-friction multiplier `phi_max - phi_cv = 5 * I_R` (degrees).
const PLANE_STRAIN_FRICTION_GAIN: f32 = 5.0;

/// Maximum-dilatancy multiplier `(-d_eps_v / d_eps_1)_max = 0.3 * I_R`.
const MAX_DILATANCY_GAIN: f32 = 0.3;

/// Upper bound of the calibrated validity window for `I_R`.
const VALID_INDEX_MAX: f32 = 4.0;

/// Bolton's recommended fitting constant `Q` for quartz / feldspar sands.
pub const QUARTZ_Q: f32 = 10.0;

/// Bolton's recommended fitting constant `R` for quartz / feldspar sands.
pub const QUARTZ_R: f32 = 1.0;

/// Bolton (1986) relative dilatancy index `I_R`.
///
/// Stored as the raw (possibly negative) index value so that callers can
/// inspect the uncorrected magnitude; use [`BoltonDilatancyIndex::clamped_index`]
/// to obtain the conventionally truncated non-negative value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoltonDilatancyIndex {
    index: f32,
}

impl BoltonDilatancyIndex {
    /// Builds the index from relative density, mean effective stress, and the
    /// fitting constants `Q`, `R`.
    ///
    /// * `relative_density` is `I_D` as a fraction in `0..=1`.
    /// * `mean_effective_stress_kpa` is `p'` at failure (kPa), strictly
    ///   positive because of the logarithm.
    /// * `q`, `r` are the dimensionless fitting constants.
    ///
    /// Returns `None` if any argument is non-finite, if `I_D` is outside
    /// `0..=1`, or if `p'` is not strictly positive.
    pub fn from_relative_density(
        relative_density: f32,
        mean_effective_stress_kpa: f32,
        q: f32,
        r: f32,
    ) -> Option<Self> {
        if !relative_density.is_finite()
            || !mean_effective_stress_kpa.is_finite()
            || !q.is_finite()
            || !r.is_finite()
        {
            return None;
        }
        if !(0.0..=1.0).contains(&relative_density) {
            return None;
        }
        if mean_effective_stress_kpa <= 0.0 {
            return None;
        }
        // `ln` is computed in f64 to satisfy the f32 transcendental lint.
        let ln_p = f64::from(mean_effective_stress_kpa).ln() as f32;
        let index = relative_density * (q - ln_p) - r;
        if !index.is_finite() {
            return None;
        }
        Some(Self { index })
    }

    /// Builds the index with Bolton's quartz-sand defaults `Q = 10`, `R = 1`.
    pub fn from_relative_density_quartz(
        relative_density: f32,
        mean_effective_stress_kpa: f32,
    ) -> Option<Self> {
        Self::from_relative_density(
            relative_density,
            mean_effective_stress_kpa,
            QUARTZ_Q,
            QUARTZ_R,
        )
    }

    /// Wraps a directly supplied index value.
    ///
    /// Returns `None` if the value is non-finite.
    pub fn from_index(index: f32) -> Option<Self> {
        if !index.is_finite() {
            return None;
        }
        Some(Self { index })
    }

    /// Returns the raw index value, which may be negative for loose soils at
    /// high confining stress.
    pub fn index(&self) -> f32 {
        self.index
    }

    /// Returns the index truncated to the non-negative, calibrated window
    /// `0..=4`. Negative raw values collapse to zero (contractive response);
    /// values above the calibrated upper bound are capped at `4`.
    pub fn clamped_index(&self) -> f32 {
        self.index.clamp(0.0, VALID_INDEX_MAX)
    }

    /// Triaxial peak-friction gain `phi_max - phi_cv = 3 * I_R` in degrees,
    /// using the non-negative clamped index.
    pub fn peak_friction_increment_triaxial_degrees(&self) -> f32 {
        TRIAXIAL_FRICTION_GAIN * self.clamped_index()
    }

    /// Plane-strain peak-friction gain `phi_max - phi_cv = 5 * I_R` in degrees,
    /// using the non-negative clamped index.
    pub fn peak_friction_increment_plane_strain_degrees(&self) -> f32 {
        PLANE_STRAIN_FRICTION_GAIN * self.clamped_index()
    }

    /// Triaxial peak friction angle (degrees) given a critical-state friction
    /// angle `phi_cv` in degrees.
    pub fn peak_friction_angle_triaxial_degrees(&self, phi_cv_degrees: f32) -> f32 {
        phi_cv_degrees + self.peak_friction_increment_triaxial_degrees()
    }

    /// Plane-strain peak friction angle (degrees) given a critical-state
    /// friction angle `phi_cv` in degrees.
    pub fn peak_friction_angle_plane_strain_degrees(&self, phi_cv_degrees: f32) -> f32 {
        phi_cv_degrees + self.peak_friction_increment_plane_strain_degrees()
    }

    /// Maximum rate of dilation `(-d_eps_v / d_eps_1)_max = 0.3 * I_R`,
    /// dimensionless, using the non-negative clamped index.
    pub fn max_dilatancy_rate(&self) -> f32 {
        MAX_DILATANCY_GAIN * self.clamped_index()
    }

    /// Returns `true` if the raw index is strictly positive, i.e. the soil is
    /// expected to dilate at peak.
    pub fn is_dilative(&self) -> bool {
        self.index > 0.0
    }

    /// Returns `true` if the raw index lies within Bolton's calibrated window
    /// `0..=4`.
    pub fn is_within_valid_range(&self) -> bool {
        (0.0..=VALID_INDEX_MAX).contains(&self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn index_matches_hand_computation() {
        // I_D = 0.75, p' = 100 kPa, Q = 10, R = 1.
        // ln(100) = 4.60517; I_R = 0.75 * (10 - 4.60517) - 1 = 3.0461...
        let idx = BoltonDilatancyIndex::from_relative_density_quartz(0.75, 100.0).unwrap();
        assert_close(idx.index(), 3.046_1, 1.0e-3);
    }

    #[test]
    fn dense_low_stress_is_strongly_dilative() {
        // Very dense sand at low stress -> large positive index.
        let idx = BoltonDilatancyIndex::from_relative_density_quartz(1.0, 10.0).unwrap();
        // ln(10) = 2.302585; I_R = 1*(10-2.302585)-1 = 6.6974 (above valid cap).
        assert_close(idx.index(), 6.697_4, 1.0e-3);
        assert!(idx.is_dilative());
        assert!(!idx.is_within_valid_range());
        // Clamped into the calibrated window.
        assert_close(idx.clamped_index(), 4.0, 1.0e-6);
    }

    #[test]
    fn loose_high_stress_is_contractive() {
        // Loose sand at high stress -> negative raw index.
        let idx = BoltonDilatancyIndex::from_relative_density_quartz(0.2, 500.0).unwrap();
        // ln(500) = 6.214608; I_R = 0.2*(10-6.214608)-1 = -0.2429
        assert!(idx.index() < 0.0);
        assert!(!idx.is_dilative());
        assert!(!idx.is_within_valid_range());
        assert_close(idx.clamped_index(), 0.0, 1.0e-6);
        // Contractive soils gain no peak friction and do not dilate.
        assert_close(idx.peak_friction_increment_triaxial_degrees(), 0.0, 1.0e-6);
        assert_close(idx.max_dilatancy_rate(), 0.0, 1.0e-6);
    }

    #[test]
    fn friction_correlations_use_clamped_index() {
        let idx = BoltonDilatancyIndex::from_index(2.0).unwrap();
        assert_close(idx.peak_friction_increment_triaxial_degrees(), 6.0, 1.0e-6);
        assert_close(
            idx.peak_friction_increment_plane_strain_degrees(),
            10.0,
            1.0e-6,
        );
        assert_close(idx.peak_friction_angle_triaxial_degrees(33.0), 39.0, 1.0e-6);
        assert_close(
            idx.peak_friction_angle_plane_strain_degrees(33.0),
            43.0,
            1.0e-6,
        );
    }

    #[test]
    fn max_dilatancy_rate_scales_with_index() {
        let idx = BoltonDilatancyIndex::from_index(3.0).unwrap();
        assert_close(idx.max_dilatancy_rate(), 0.9, 1.0e-6);
    }

    #[test]
    fn plane_strain_gain_exceeds_triaxial_gain() {
        let idx = BoltonDilatancyIndex::from_index(1.5).unwrap();
        assert!(
            idx.peak_friction_increment_plane_strain_degrees()
                > idx.peak_friction_increment_triaxial_degrees()
        );
    }

    #[test]
    fn custom_fitting_constants_are_honoured() {
        // Q = 8, R = 0.5 (a softer calibration).
        let idx = BoltonDilatancyIndex::from_relative_density(0.5, 100.0, 8.0, 0.5).unwrap();
        // ln(100) = 4.60517; I_R = 0.5*(8-4.60517)-0.5 = 1.1974
        assert_close(idx.index(), 1.197_4, 1.0e-3);
    }

    #[test]
    fn in_window_index_is_valid() {
        let idx = BoltonDilatancyIndex::from_index(2.5).unwrap();
        assert!(idx.is_within_valid_range());
        assert!(idx.is_dilative());
    }

    #[test]
    fn rejects_out_of_range_relative_density() {
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(-0.1, 100.0).is_none());
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(1.1, 100.0).is_none());
    }

    #[test]
    fn rejects_non_positive_stress() {
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(0.5, 0.0).is_none());
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(0.5, -5.0).is_none());
    }

    #[test]
    fn rejects_non_finite_inputs() {
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(f32::NAN, 100.0).is_none());
        assert!(BoltonDilatancyIndex::from_relative_density_quartz(0.5, f32::INFINITY).is_none());
        assert!(BoltonDilatancyIndex::from_index(f32::NAN).is_none());
    }
}
