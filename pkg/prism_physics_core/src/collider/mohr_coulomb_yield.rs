//! Mohr–Coulomb shear-failure criterion for granular / frictional stress states.
//!
//! The Mohr–Coulomb criterion is the classical yield surface for cohesive
//! frictional materials (soils, powders, rock). On a plane carrying normal
//! stress `sigma_n` and shear stress `tau`, failure occurs when the mobilised
//! shear reaches the available shear strength
//!
//! ```text
//! tau_f = c + sigma_n * tan(phi)
//! ```
//!
//! where `c` is cohesion and `phi` is the internal friction angle. Expressed in
//! terms of the Mohr circle of a principal-stress state with major/minor
//! principal stresses `sigma1 >= sigma3`, the circle has centre
//! `p = (sigma1 + sigma3) / 2` and radius `R = (sigma1 - sigma3) / 2`; failure
//! occurs when the circle becomes tangent to the envelope:
//!
//! ```text
//! R = c * cos(phi) + p * sin(phi)
//! ```
//!
//! This module uses the **compression-positive** convention standard in soil
//! mechanics: a larger principal stress is more compressive, and the normal
//! stress on the critical plane increases shear strength. The criterion is a
//! pure analytic predicate over a supplied stress state and has no coupling to
//! the simulation pipeline, making it directly unit-verifiable.
//!
//! No Unreal Engine source or derived code.

use std::f32::consts::FRAC_PI_2;

/// A Mohr–Coulomb failure envelope defined by a friction angle and cohesion.
///
/// Construct with [`MohrCoulombCriterion::new`] (friction angle in radians) or
/// [`MohrCoulombCriterion::from_friction_coefficient`] (`tan(phi)`), then query
/// a stress state via [`MohrCoulombCriterion::evaluate_principal`] or
/// [`MohrCoulombCriterion::evaluate_plane`].
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MohrCoulombCriterion {
    /// Internal friction angle in radians, in the range `[0, pi/2)`.
    friction_angle: f32,
    /// Cohesion intercept of the envelope (`>= 0`), in stress units.
    cohesion: f32,
}

impl MohrCoulombCriterion {
    /// Builds a criterion from a friction angle (radians) and cohesion.
    ///
    /// Returns `None` unless the angle lies in `[0, pi/2)`, cohesion is
    /// non-negative, and both inputs are finite.
    #[must_use]
    pub fn new(friction_angle: f32, cohesion: f32) -> Option<Self> {
        if !friction_angle.is_finite() || !cohesion.is_finite() {
            return None;
        }
        if !(0.0..FRAC_PI_2).contains(&friction_angle) {
            return None;
        }
        if cohesion < 0.0 {
            return None;
        }
        Some(Self {
            friction_angle,
            cohesion,
        })
    }

    /// Builds a criterion from a friction coefficient `mu = tan(phi)` and cohesion.
    ///
    /// Returns `None` unless `mu >= 0`, cohesion is non-negative, and both inputs
    /// are finite.
    #[must_use]
    pub fn from_friction_coefficient(friction_coefficient: f32, cohesion: f32) -> Option<Self> {
        if !friction_coefficient.is_finite() || friction_coefficient < 0.0 {
            return None;
        }
        // atan is disallowed on f32 by the lint policy; compute in f64.
        let angle = (friction_coefficient as f64).atan() as f32;
        Self::new(angle, cohesion)
    }

    /// Internal friction angle in radians.
    #[must_use]
    pub fn friction_angle(&self) -> f32 {
        self.friction_angle
    }

    /// Friction coefficient `tan(phi)`.
    #[must_use]
    pub fn friction_coefficient(&self) -> f32 {
        let a = self.friction_angle as f64;
        (a.sin() / a.cos()) as f32
    }

    /// Cohesion intercept of the envelope.
    #[must_use]
    pub fn cohesion(&self) -> f32 {
        self.cohesion
    }

    /// Shear strength available on a plane carrying normal stress `sigma_n`.
    ///
    /// `tau_f = c + sigma_n * tan(phi)` (compression-positive `sigma_n`).
    #[must_use]
    pub fn shear_strength(&self, sigma_n: f32) -> f32 {
        self.cohesion + sigma_n * self.friction_coefficient()
    }

    /// Unconfined compressive strength: the major principal stress at failure
    /// when the minor principal stress is zero.
    ///
    /// `sigma_c = 2 c cos(phi) / (1 - sin(phi))`.
    #[must_use]
    pub fn unconfined_compressive_strength(&self) -> f32 {
        let a = self.friction_angle as f64;
        let (s, c) = a.sin_cos();
        (2.0 * (self.cohesion as f64) * c / (1.0 - s)) as f32
    }

    /// Isotropic tensile apex of the cone: the hydrostatic tension at which the
    /// envelope closes, `sigma_apex = c / tan(phi)`.
    ///
    /// Returns `None` for a cohesionless material (`phi = 0`), where the apex is
    /// unbounded.
    #[must_use]
    pub fn cohesion_apex(&self) -> Option<f32> {
        if self.friction_angle == 0.0 {
            return None;
        }
        let mu = self.friction_coefficient();
        if mu <= 0.0 {
            return None;
        }
        Some(self.cohesion / mu)
    }

    /// Evaluates the criterion against a principal-stress state.
    ///
    /// `principal` may be in any order; the major (most compressive) and minor
    /// values drive the classical criterion, while the intermediate principal
    /// stress does not enter. Returns `None` if any component is non-finite.
    #[must_use]
    pub fn evaluate_principal(&self, principal: [f32; 3]) -> Option<MohrCoulombYieldState> {
        if principal.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let sigma_max = principal[0].max(principal[1]).max(principal[2]);
        let sigma_min = principal[0].min(principal[1]).min(principal[2]);
        let center = 0.5 * (sigma_max + sigma_min);
        let radius = 0.5 * (sigma_max - sigma_min);

        let a = self.friction_angle as f64;
        let (s, c) = a.sin_cos();
        let strength = (self.cohesion as f64) * c + (center as f64) * s;
        let strength = strength as f32;

        Some(MohrCoulombYieldState {
            normal_stress: center,
            shear_stress: radius,
            shear_strength: strength,
            yield_function: radius - strength,
        })
    }

    /// Evaluates the criterion on a single plane with normal stress `sigma_n`
    /// and non-negative shear-stress magnitude `tau`.
    ///
    /// Returns `None` if either input is non-finite or if `tau < 0`.
    #[must_use]
    pub fn evaluate_plane(&self, sigma_n: f32, tau: f32) -> Option<MohrCoulombYieldState> {
        if !sigma_n.is_finite() || !tau.is_finite() || tau < 0.0 {
            return None;
        }
        let strength = self.shear_strength(sigma_n);
        Some(MohrCoulombYieldState {
            normal_stress: sigma_n,
            shear_stress: tau,
            shear_strength: strength,
            yield_function: tau - strength,
        })
    }
}

/// Result of evaluating a [`MohrCoulombCriterion`] against a stress state.
///
/// The sign of [`MohrCoulombYieldState::yield_function`] classifies the state:
/// negative is elastic/stable, zero is at yield, positive is beyond the
/// envelope (inadmissible for a perfectly plastic material).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MohrCoulombYieldState {
    normal_stress: f32,
    shear_stress: f32,
    shear_strength: f32,
    yield_function: f32,
}

impl MohrCoulombYieldState {
    /// Representative normal stress (plane normal stress, or Mohr-circle centre
    /// for a principal evaluation).
    #[must_use]
    pub fn normal_stress(&self) -> f32 {
        self.normal_stress
    }

    /// Mobilised shear stress (plane shear, or Mohr-circle radius for a
    /// principal evaluation).
    #[must_use]
    pub fn shear_stress(&self) -> f32 {
        self.shear_stress
    }

    /// Available shear strength at this state.
    #[must_use]
    pub fn shear_strength(&self) -> f32 {
        self.shear_strength
    }

    /// Yield function `f = shear_stress - shear_strength`.
    ///
    /// `f < 0` stable, `f = 0` at yield, `f > 0` beyond the envelope.
    #[must_use]
    pub fn yield_function(&self) -> f32 {
        self.yield_function
    }

    /// Whether the state lies on or beyond the failure envelope, within
    /// `tolerance` (`f >= -tolerance`).
    #[must_use]
    pub fn is_yielding(&self, tolerance: f32) -> bool {
        self.yield_function >= -tolerance
    }

    /// Whether the state lies strictly beyond the envelope by more than
    /// `tolerance` (`f > tolerance`).
    #[must_use]
    pub fn is_beyond_yield(&self, tolerance: f32) -> bool {
        self.yield_function > tolerance
    }

    /// Factor of safety: available strength divided by mobilised shear.
    ///
    /// Greater than one is stable, one is at yield, less than one is failed.
    /// Returns `None` when the mobilised shear is non-positive (undefined).
    #[must_use]
    pub fn factor_of_safety(&self) -> Option<f32> {
        if self.shear_stress <= 0.0 {
            return None;
        }
        Some(self.shear_strength / self.shear_stress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_6;

    const TOL: f32 = 1.0e-4;

    #[test]
    fn new_rejects_invalid_inputs() {
        assert!(MohrCoulombCriterion::new(-0.1, 1.0).is_none());
        assert!(MohrCoulombCriterion::new(FRAC_PI_2, 1.0).is_none());
        assert!(MohrCoulombCriterion::new(FRAC_PI_6, -1.0).is_none());
        assert!(MohrCoulombCriterion::new(f32::NAN, 1.0).is_none());
        assert!(MohrCoulombCriterion::new(FRAC_PI_6, f32::INFINITY).is_none());
    }

    #[test]
    fn from_friction_coefficient_recovers_angle() {
        // tan(30 deg) = 1/sqrt(3).
        let mu = 1.0 / 3.0_f32.sqrt();
        let crit = MohrCoulombCriterion::from_friction_coefficient(mu, 2.0).expect("valid");
        assert!((crit.friction_angle() - FRAC_PI_6).abs() < TOL);
        assert!((crit.friction_coefficient() - mu).abs() < TOL);
        assert!(MohrCoulombCriterion::from_friction_coefficient(-0.1, 0.0).is_none());
    }

    #[test]
    fn shear_strength_is_linear_in_normal_stress() {
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 5.0).expect("valid");
        let mu = crit.friction_coefficient();
        assert!((crit.shear_strength(0.0) - 5.0).abs() < TOL);
        assert!((crit.shear_strength(10.0) - (5.0 + 10.0 * mu)).abs() < 1.0e-3);
    }

    #[test]
    fn cohesionless_circle_tangent_is_at_yield() {
        // phi = 30 deg, c = 0 => yield when R = p * sin(phi) = 0.5 p.
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 0.0).expect("valid");
        let p = 10.0_f32;
        let r = 0.5 * p; // sin(30 deg) = 0.5
        let state = crit.evaluate_principal([p + r, p, p - r]).expect("finite");
        assert!(state.yield_function().abs() < 1.0e-3);
        assert!(state.is_yielding(TOL));
        let fos = state.factor_of_safety().expect("positive shear");
        assert!((fos - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn subcritical_circle_is_stable() {
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 0.0).expect("valid");
        let p = 10.0_f32;
        let r = 0.3 * p; // below the 0.5 p yield radius
        let state = crit.evaluate_principal([p + r, p, p - r]).expect("finite");
        assert!(state.yield_function() < 0.0);
        assert!(!state.is_yielding(TOL));
        assert!(state.factor_of_safety().expect("positive shear") > 1.0);
    }

    #[test]
    fn supercritical_circle_is_beyond_yield() {
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 0.0).expect("valid");
        let p = 10.0_f32;
        let r = 0.8 * p; // above the 0.5 p yield radius
        let state = crit.evaluate_principal([p + r, p, p - r]).expect("finite");
        assert!(state.yield_function() > 0.0);
        assert!(state.is_beyond_yield(TOL));
        assert!(state.factor_of_safety().expect("positive shear") < 1.0);
    }

    #[test]
    fn plane_at_strength_is_at_yield() {
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 3.0).expect("valid");
        let sigma_n = 8.0_f32;
        let tau = crit.shear_strength(sigma_n);
        let state = crit.evaluate_plane(sigma_n, tau).expect("finite");
        assert!(state.yield_function().abs() < 1.0e-3);
        assert!((state.factor_of_safety().expect("positive shear") - 1.0).abs() < 1.0e-3);
        assert!(crit.evaluate_plane(sigma_n, -1.0).is_none());
        assert!(crit
            .evaluate_plane(sigma_n, 0.0)
            .unwrap()
            .factor_of_safety()
            .is_none());
    }

    #[test]
    fn unconfined_strength_matches_closed_form() {
        // phi = 30 deg, c = 10 => sigma_c = 2 c cos30 / (1 - sin30)
        //                                 = 2*10*(sqrt(3)/2) / 0.5 = 20 sqrt(3).
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 10.0).expect("valid");
        let expected = 20.0 * 3.0_f32.sqrt();
        assert!((crit.unconfined_compressive_strength() - expected).abs() < 1.0e-2);
    }

    #[test]
    fn cohesion_apex_handles_frictionless_case() {
        let frictionless = MohrCoulombCriterion::new(0.0, 4.0).expect("valid");
        assert!(frictionless.cohesion_apex().is_none());
        let crit = MohrCoulombCriterion::new(FRAC_PI_6, 10.0).expect("valid");
        let mu = crit.friction_coefficient();
        let apex = crit.cohesion_apex().expect("defined");
        assert!((apex - 10.0 / mu).abs() < 1.0e-2);
    }
}
