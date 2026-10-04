//! Overconsolidation ratio (OCR) and its derived state correlations.
//!
//! The overconsolidation ratio compares the maximum effective stress a soil has
//! ever experienced (the preconsolidation pressure `p_c`) to its current
//! effective stress `p'`:
//!
//! ```text
//! OCR = p_c / p'
//! ```
//!
//! Because `p_c` is the historical maximum, `OCR >= 1` always holds; `OCR = 1`
//! denotes a normally consolidated state and `OCR > 1` an overconsolidated one.
//!
//! Two widely used correlations are derived from the ratio:
//!
//! * at-rest earth pressure (Jaky for the normally consolidated baseline,
//!   Mayne & Kulhawy for the overconsolidated correction):
//!
//! ```text
//! K0_nc = 1 - sin(phi)
//! K0_oc = (1 - sin(phi)) * OCR^sin(phi)
//! ```
//!
//! * SHANSEP undrained-strength scaling with exponent `Lambda` (typically
//!   around `0.8`):
//!
//! ```text
//! (su / sigma_v')_OC = (su / sigma_v')_NC * OCR^Lambda
//! ```
//!
//! References: Jaky (1944); Mayne & Kulhawy, "K0-OCR relationships in soil",
//! ASCE 1982; Ladd & Foott SHANSEP, 1974.
//!
//! This module is a self-contained diagnostic primitive. It performs no
//! coupling with the simulation pipeline and holds no simulation state.

use std::f32::consts::FRAC_PI_2;

/// Tolerance within which an OCR is treated as normally consolidated.
const NC_TOLERANCE: f32 = 1.0e-4;

/// Overconsolidation ratio `OCR = p_c / p'`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverconsolidationRatio {
    ocr: f32,
}

impl OverconsolidationRatio {
    /// Builds the ratio from the preconsolidation pressure `p_c` and the
    /// current effective stress `p'`.
    ///
    /// Both pressures must be finite and strictly positive, and the computed
    /// ratio must satisfy the physical lower bound `OCR >= 1`; otherwise
    /// `None` is returned.
    pub fn from_pressures(
        preconsolidation_pressure: f32,
        current_effective_stress: f32,
    ) -> Option<Self> {
        if !preconsolidation_pressure.is_finite() || !current_effective_stress.is_finite() {
            return None;
        }
        if preconsolidation_pressure <= 0.0 || current_effective_stress <= 0.0 {
            return None;
        }
        let ocr = preconsolidation_pressure / current_effective_stress;
        Self::from_ratio(ocr)
    }

    /// Wraps a directly supplied ratio.
    ///
    /// Returns `None` if `ocr` is non-finite or below the physical lower bound
    /// `1`.
    pub fn from_ratio(ocr: f32) -> Option<Self> {
        if !ocr.is_finite() || ocr < 1.0 - NC_TOLERANCE {
            return None;
        }
        Some(Self { ocr: ocr.max(1.0) })
    }

    /// Returns the overconsolidation ratio.
    pub fn ratio(&self) -> f32 {
        self.ocr
    }

    /// Returns `true` if the soil is normally consolidated (`OCR ~= 1`).
    pub fn is_normally_consolidated(&self) -> bool {
        (self.ocr - 1.0).abs() <= NC_TOLERANCE
    }

    /// Returns `true` if the soil is overconsolidated (`OCR > 1`).
    pub fn is_overconsolidated(&self) -> bool {
        self.ocr > 1.0 + NC_TOLERANCE
    }

    /// Jaky's normally consolidated at-rest earth pressure coefficient
    /// `K0_nc = 1 - sin(phi)`.
    ///
    /// Returns `None` if `phi` is non-finite or outside `(0, pi/2)`.
    pub fn k0_normally_consolidated(phi_radians: f32) -> Option<f32> {
        if !phi_radians.is_finite() || phi_radians <= 0.0 || phi_radians >= FRAC_PI_2 {
            return None;
        }
        // `sin` is computed in f64 to satisfy the f32 transcendental lint.
        let sin_phi = f64::from(phi_radians).sin() as f32;
        Some(1.0 - sin_phi)
    }

    /// Mayne & Kulhawy overconsolidated at-rest earth pressure coefficient
    /// `K0_oc = (1 - sin(phi)) * OCR^sin(phi)`.
    ///
    /// Returns `None` if `phi` is non-finite or outside `(0, pi/2)`.
    pub fn k0_overconsolidated(&self, phi_radians: f32) -> Option<f32> {
        if !phi_radians.is_finite() || phi_radians <= 0.0 || phi_radians >= FRAC_PI_2 {
            return None;
        }
        // `sin` and `powf` are computed in f64 to satisfy the f32 lints.
        let sin_phi = f64::from(phi_radians).sin();
        let factor = f64::from(self.ocr).powf(sin_phi);
        Some(((1.0 - sin_phi) * factor) as f32)
    }

    /// SHANSEP undrained-strength scaling factor `OCR^Lambda`.
    ///
    /// Returns `None` if `lambda` is non-finite.
    pub fn undrained_strength_scaling(&self, lambda: f32) -> Option<f32> {
        if !lambda.is_finite() {
            return None;
        }
        // `powf` is computed in f64 to satisfy the f32 lint.
        Some(f64::from(self.ocr).powf(f64::from(lambda)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn ratio_from_pressures() {
        // p_c = 400 kPa, p' = 100 kPa -> OCR = 4.
        let ocr = OverconsolidationRatio::from_pressures(400.0, 100.0).unwrap();
        assert_close(ocr.ratio(), 4.0, 1.0e-6);
        assert!(ocr.is_overconsolidated());
        assert!(!ocr.is_normally_consolidated());
    }

    #[test]
    fn normally_consolidated_detection() {
        let ocr = OverconsolidationRatio::from_pressures(100.0, 100.0).unwrap();
        assert!(ocr.is_normally_consolidated());
        assert!(!ocr.is_overconsolidated());
    }

    #[test]
    fn k0_nc_matches_jaky() {
        // phi = 30 deg -> sin = 0.5 -> K0_nc = 0.5.
        let k0 = OverconsolidationRatio::k0_normally_consolidated(30.0_f32.to_radians()).unwrap();
        assert_close(k0, 0.5, 1.0e-5);
    }

    #[test]
    fn k0_oc_matches_mayne_kulhawy() {
        // OCR = 4, phi = 30 deg -> K0_oc = 0.5 * 4^0.5 = 1.0.
        let ocr = OverconsolidationRatio::from_ratio(4.0).unwrap();
        let k0 = ocr.k0_overconsolidated(30.0_f32.to_radians()).unwrap();
        assert_close(k0, 1.0, 1.0e-4);
    }

    #[test]
    fn k0_oc_reduces_to_nc_at_unit_ratio() {
        let ocr = OverconsolidationRatio::from_ratio(1.0).unwrap();
        let k0_oc = ocr.k0_overconsolidated(30.0_f32.to_radians()).unwrap();
        let k0_nc =
            OverconsolidationRatio::k0_normally_consolidated(30.0_f32.to_radians()).unwrap();
        assert_close(k0_oc, k0_nc, 1.0e-5);
    }

    #[test]
    fn shansep_scaling() {
        // OCR = 4, Lambda = 0.8 -> 4^0.8 = 3.0314.
        let ocr = OverconsolidationRatio::from_ratio(4.0).unwrap();
        assert_close(
            ocr.undrained_strength_scaling(0.8).unwrap(),
            3.031_4,
            1.0e-3,
        );
    }

    #[test]
    fn shansep_scaling_unit_ratio_is_one() {
        let ocr = OverconsolidationRatio::from_ratio(1.0).unwrap();
        assert_close(ocr.undrained_strength_scaling(0.8).unwrap(), 1.0, 1.0e-6);
    }

    #[test]
    fn rejects_sub_physical_ratio() {
        assert!(OverconsolidationRatio::from_ratio(0.5).is_none());
    }

    #[test]
    fn rejects_non_positive_pressures() {
        assert!(OverconsolidationRatio::from_pressures(0.0, 100.0).is_none());
        assert!(OverconsolidationRatio::from_pressures(100.0, 0.0).is_none());
        // Current stress above preconsolidation is non-physical (OCR < 1).
        assert!(OverconsolidationRatio::from_pressures(100.0, 200.0).is_none());
    }

    #[test]
    fn rejects_out_of_range_friction_angle() {
        assert!(OverconsolidationRatio::k0_normally_consolidated(0.0).is_none());
        assert!(OverconsolidationRatio::k0_normally_consolidated(FRAC_PI_2).is_none());
        let ocr = OverconsolidationRatio::from_ratio(2.0).unwrap();
        assert!(ocr.k0_overconsolidated(-0.1).is_none());
    }

    #[test]
    fn rejects_non_finite_inputs() {
        assert!(OverconsolidationRatio::from_ratio(f32::NAN).is_none());
        assert!(OverconsolidationRatio::from_pressures(f32::INFINITY, 100.0).is_none());
        let ocr = OverconsolidationRatio::from_ratio(2.0).unwrap();
        assert!(ocr.undrained_strength_scaling(f32::NAN).is_none());
    }
}
