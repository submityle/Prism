//! Rankine tension cutoff for brittle / cohesive-frictional materials.
//!
//! A Mohr-Coulomb cone closes in hydrostatic tension at its apex
//! `sigma_apex = c / tan(phi)`, but real geomaterials crack well before that
//! idealized limit. The Rankine (maximum principal stress) tension cutoff caps
//! the admissible tensile stress at a separate tensile strength `sigma_t`:
//!
//! ```text
//! f_t = sigma_1 - sigma_t      (tension-positive convention)
//! ```
//!
//! where `sigma_1` is the largest (most tensile) principal stress. A crack
//! opens once `f_t >= 0`. Because the cutoff cannot sit beyond the cone apex,
//! a physically consistent cutoff is `sigma_t <= c / tan(phi)`; the
//! [`RankineTensionCutoff::capped_with_mohr_coulomb_apex`] constructor enforces
//! that bound.
//!
//! Stresses use the crate's tension-positive convention: positive principal
//! stresses are tensile, so the governing stress is the arithmetic maximum of
//! the three principals.
//!
//! References: standard composite Mohr-Coulomb + tension-cutoff plasticity as
//! implemented in geotechnical codes (e.g. Plaxis, FLAC).
//!
//! This module is a self-contained diagnostic primitive. It performs no
//! coupling with the simulation pipeline and holds no simulation state.

use std::f32::consts::FRAC_PI_2;

/// A Rankine maximum-principal-stress tension cutoff with tensile strength
/// `sigma_t >= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankineTensionCutoff {
    tensile_strength: f32,
}

impl RankineTensionCutoff {
    /// Builds a cutoff from a tensile strength `sigma_t`.
    ///
    /// Returns `None` if `sigma_t` is non-finite or negative.
    pub fn new(tensile_strength: f32) -> Option<Self> {
        if !tensile_strength.is_finite() || tensile_strength < 0.0 {
            return None;
        }
        Some(Self { tensile_strength })
    }

    /// Builds a cutoff at the Mohr-Coulomb apex `sigma_t = c / tan(phi)`.
    ///
    /// Returns `None` if the inputs are non-finite, if `cohesion` is negative,
    /// if `phi` is outside `(0, pi/2)` (a cohesionless `phi = 0` has no finite
    /// apex), or if the apex is non-finite.
    pub fn from_mohr_coulomb_apex(cohesion: f32, friction_angle: f32) -> Option<Self> {
        if !cohesion.is_finite() || !friction_angle.is_finite() {
            return None;
        }
        if cohesion < 0.0 || friction_angle <= 0.0 || friction_angle >= FRAC_PI_2 {
            return None;
        }
        // `tan` is computed in f64 to satisfy the f32 transcendental lint.
        let tan_phi = f64::from(friction_angle).tan() as f32;
        if tan_phi <= 0.0 {
            return None;
        }
        let apex = cohesion / tan_phi;
        Self::new(apex)
    }

    /// Builds a cutoff from a requested tensile strength, capped so it never
    /// exceeds the Mohr-Coulomb apex `c / tan(phi)`.
    ///
    /// Returns `None` under the same conditions as
    /// [`RankineTensionCutoff::from_mohr_coulomb_apex`], or if
    /// `requested_tensile_strength` is non-finite or negative.
    pub fn capped_with_mohr_coulomb_apex(
        requested_tensile_strength: f32,
        cohesion: f32,
        friction_angle: f32,
    ) -> Option<Self> {
        if !requested_tensile_strength.is_finite() || requested_tensile_strength < 0.0 {
            return None;
        }
        let apex = Self::from_mohr_coulomb_apex(cohesion, friction_angle)?.tensile_strength;
        Self::new(requested_tensile_strength.min(apex))
    }

    /// Returns the tensile strength `sigma_t`.
    pub fn tensile_strength(&self) -> f32 {
        self.tensile_strength
    }

    /// Tension yield function `f_t = sigma_1 - sigma_t` for a given largest
    /// (most tensile) principal stress.
    pub fn yield_function(&self, max_principal_stress: f32) -> f32 {
        max_principal_stress - self.tensile_strength
    }

    /// Evaluates the cutoff against a principal-stress triple (tension-positive).
    ///
    /// Returns `None` if any component is non-finite.
    pub fn evaluate_principal(&self, principal: [f32; 3]) -> Option<RankineTensionState> {
        if principal.iter().any(|s| !s.is_finite()) {
            return None;
        }
        let max_principal = principal[0].max(principal[1]).max(principal[2]);
        Some(RankineTensionState {
            max_principal,
            tensile_strength: self.tensile_strength,
            yield_function: max_principal - self.tensile_strength,
        })
    }
}

/// Outcome of evaluating a [`RankineTensionCutoff`] at a stress state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankineTensionState {
    max_principal: f32,
    tensile_strength: f32,
    yield_function: f32,
}

impl RankineTensionState {
    /// The governing largest (most tensile) principal stress.
    pub fn max_principal_stress(&self) -> f32 {
        self.max_principal
    }

    /// The tensile strength the state was compared against.
    pub fn tensile_strength(&self) -> f32 {
        self.tensile_strength
    }

    /// The tension yield function `f_t = sigma_1 - sigma_t`.
    pub fn yield_function(&self) -> f32 {
        self.yield_function
    }

    /// Returns `true` if the state is at or beyond the cutoff within
    /// `tolerance` (a crack opens).
    pub fn is_cracking(&self, tolerance: f32) -> bool {
        self.yield_function >= -tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn apex_matches_hand_value() {
        // c = 10, phi = 30 deg -> tan = 0.57735 -> apex = 17.3205.
        let cut =
            RankineTensionCutoff::from_mohr_coulomb_apex(10.0, 30.0_f32.to_radians()).unwrap();
        assert_close(cut.tensile_strength(), 17.320_5, 1.0e-3);
    }

    #[test]
    fn capping_limits_to_apex() {
        // Requested 100 but apex is 17.32 -> cutoff clamps to apex.
        let cut =
            RankineTensionCutoff::capped_with_mohr_coulomb_apex(100.0, 10.0, 30.0_f32.to_radians())
                .unwrap();
        assert_close(cut.tensile_strength(), 17.320_5, 1.0e-3);
    }

    #[test]
    fn capping_keeps_smaller_request() {
        let cut =
            RankineTensionCutoff::capped_with_mohr_coulomb_apex(5.0, 10.0, 30.0_f32.to_radians())
                .unwrap();
        assert_close(cut.tensile_strength(), 5.0, 1.0e-6);
    }

    #[test]
    fn below_cutoff_is_not_cracking() {
        let cut = RankineTensionCutoff::new(5.0).unwrap();
        let st = cut.evaluate_principal([3.0, 1.0, 0.0]).unwrap();
        assert_close(st.max_principal_stress(), 3.0, 1.0e-6);
        assert_close(st.yield_function(), -2.0, 1.0e-6);
        assert!(!st.is_cracking(0.0));
    }

    #[test]
    fn above_cutoff_cracks() {
        let cut = RankineTensionCutoff::new(5.0).unwrap();
        let st = cut.evaluate_principal([6.0, 2.0, 1.0]).unwrap();
        assert_close(st.max_principal_stress(), 6.0, 1.0e-6);
        assert_close(st.yield_function(), 1.0, 1.0e-6);
        assert!(st.is_cracking(0.0));
    }

    #[test]
    fn governing_stress_is_the_arithmetic_maximum() {
        // Tension-positive: the most tensile (largest) principal governs.
        let cut = RankineTensionCutoff::new(2.0).unwrap();
        let st = cut.evaluate_principal([-5.0, 3.0, -1.0]).unwrap();
        assert_close(st.max_principal_stress(), 3.0, 1.0e-6);
        assert!(st.is_cracking(0.0));
    }

    #[test]
    fn pure_compression_never_cracks() {
        let cut = RankineTensionCutoff::new(0.0).unwrap();
        let st = cut.evaluate_principal([-10.0, -20.0, -30.0]).unwrap();
        assert!(!st.is_cracking(0.0));
    }

    #[test]
    fn tolerance_catches_near_cutoff_state() {
        let cut = RankineTensionCutoff::new(5.0).unwrap();
        let st = cut.evaluate_principal([4.999, 0.0, 0.0]).unwrap();
        assert!(!st.is_cracking(0.0));
        assert!(st.is_cracking(0.01));
    }

    #[test]
    fn standalone_yield_function() {
        let cut = RankineTensionCutoff::new(5.0).unwrap();
        assert_close(cut.yield_function(8.0), 3.0, 1.0e-6);
        assert_close(cut.yield_function(2.0), -3.0, 1.0e-6);
    }

    #[test]
    fn rejects_invalid_strength() {
        assert!(RankineTensionCutoff::new(-1.0).is_none());
        assert!(RankineTensionCutoff::new(f32::NAN).is_none());
    }

    #[test]
    fn rejects_invalid_apex_inputs() {
        // Cohesionless material has no finite apex.
        assert!(RankineTensionCutoff::from_mohr_coulomb_apex(10.0, 0.0).is_none());
        // Negative cohesion is rejected.
        assert!(
            RankineTensionCutoff::from_mohr_coulomb_apex(-1.0, 30.0_f32.to_radians()).is_none()
        );
        // phi >= pi/2 is rejected.
        assert!(RankineTensionCutoff::from_mohr_coulomb_apex(10.0, FRAC_PI_2).is_none());
    }

    #[test]
    fn rejects_non_finite_principal() {
        let cut = RankineTensionCutoff::new(5.0).unwrap();
        assert!(cut.evaluate_principal([f32::NAN, 0.0, 0.0]).is_none());
    }
}
