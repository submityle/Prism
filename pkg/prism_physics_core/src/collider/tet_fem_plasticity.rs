//! Finite-strain (J2 / von Mises) elastoplasticity for tetrahedral FEM.
//!
//! Real soft bodies — flesh, clay, metal, dough — do not spring back fully:
//! past a yield point they *flow* and keep the new shape. This module adds that
//! behaviour on top of the crate's hyperelastic constitutive models
//! ([`tet_fem_constitutive`](super::tet_fem_constitutive)) via the standard
//! **multiplicative split**
//!
//! ```text
//! F = Fₑ · Fₚ
//! ```
//!
//! where `Fₚ` is a persistent *plastic* deformation that the solver carries
//! across steps and `Fₑ` is the recoverable *elastic* part that actually
//! generates stress. Each step we:
//!
//! 1. form the **elastic predictor** `Fₑ_trial = F · Fₚ⁻¹`,
//! 2. take its signed SVD `Fₑ_trial = U Σ Vᵀ` and move into principal
//!    **Hencky (logarithmic) strain** `ε = ln|Σ|`,
//! 3. test the deviatoric strain against a von Mises yield surface, and
//! 4. **radially return** any excess onto the surface (associative, isochoric
//!    plastic flow), reconstruct `Fₑ`, and fold the consumed strain into `Fₚ`.
//!
//! The plastic flow is volume preserving (`det Fₚ` is invariant), matching the
//! J2 assumption that hydrostatic pressure causes no yielding. Optional linear
//! isotropic **hardening** grows the yield threshold with accumulated plastic
//! strain.
//!
//! This is a per-element, read-mostly kernel with no coupling to any
//! integrator: feed it the total deformation gradient, get back the elastic
//! gradient to hand to a stress routine, and keep the mutated [`PlasticState`].
//!
//! # Attribution
//!
//! Clean-room implementation of textbook multiplicative J2 plasticity
//! (Hencky-strain radial return) over the crate's own [`svd3`]. No Unreal
//! Engine source or derived code.

use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed predictor never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// Deviatoric strains with Frobenius norm at or below this are treated as
/// purely hydrostatic (no yield direction), skipping plastic flow.
const MIN_DEVIATORIC_NORM: f32 = 1e-9;

/// Material yield parameters for von Mises plasticity in log-strain space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticModel {
    /// Deviatoric Hencky-strain magnitude at which plastic flow begins
    /// (`‖dev ε‖`, dimensionless). Larger == stiffer / later yielding.
    yield_strain: f32,
    /// Linear isotropic hardening coefficient `≥ 0`: the effective yield grows
    /// as `yield_strain + hardening · accumulated_plastic_strain`.
    hardening: f32,
}

impl PlasticModel {
    /// Builds a model, validating `yield_strain` is finite and strictly
    /// positive and `hardening` is finite and non-negative.
    #[must_use]
    pub fn new(yield_strain: f32, hardening: f32) -> Option<Self> {
        if !yield_strain.is_finite() || yield_strain <= 0.0 {
            return None;
        }
        if !hardening.is_finite() || hardening < 0.0 {
            return None;
        }
        Some(Self {
            yield_strain,
            hardening,
        })
    }

    /// A perfectly plastic model (no hardening).
    #[must_use]
    pub fn perfectly_plastic(yield_strain: f32) -> Option<Self> {
        Self::new(yield_strain, 0.0)
    }

    /// Base yield strain (before hardening).
    #[must_use]
    pub fn yield_strain(&self) -> f32 {
        self.yield_strain
    }

    /// Hardening coefficient.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }

    /// Effective yield strain after `accumulated` plastic strain.
    #[must_use]
    pub fn effective_yield(&self, accumulated: f32) -> f32 {
        self.yield_strain + self.hardening * accumulated.max(0.0)
    }
}

/// Persistent per-element plastic state carried across simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticState {
    /// Plastic part of the deformation gradient `Fₚ`.
    plastic_gradient: Mat3,
    /// Accumulated equivalent plastic strain (monotonically non-decreasing),
    /// driving isotropic hardening.
    accumulated_strain: f32,
}

impl PlasticState {
    /// The undeformed rest state: `Fₚ = I`, zero accumulated strain.
    #[must_use]
    pub fn rest() -> Self {
        Self {
            plastic_gradient: Mat3::IDENTITY,
            accumulated_strain: 0.0,
        }
    }

    /// Current plastic deformation gradient `Fₚ`.
    #[must_use]
    pub fn plastic_gradient(&self) -> Mat3 {
        self.plastic_gradient
    }

    /// Accumulated equivalent plastic strain so far.
    #[must_use]
    pub fn accumulated_strain(&self) -> f32 {
        self.accumulated_strain
    }
}

impl Default for PlasticState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Equivalent plastic strain consumed this step (`0` when still elastic).
    pub plastic_increment: f32,
    /// `true` when the predictor exceeded the yield surface and flowed.
    pub yielded: bool,
}

/// Performs the elastic-predictor / plastic-return update for one element.
///
/// Given the total deformation gradient `f_total` and the carried
/// [`PlasticState`], returns the elastic gradient to use for stress and mutates
/// `state` (updating `Fₚ` and accumulated strain when the material yields).
///
/// The returned elastic gradient always satisfies `f_total ≈ Fₑ · Fₚ` with the
/// updated `Fₚ`, and plastic flow preserves `det Fₚ`.
#[must_use]
pub fn return_map(f_total: Mat3, model: &PlasticModel, state: &mut PlasticState) -> PlasticStep {
    // Elastic predictor: strip the stored plastic deformation.
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma = svd.sigma;
    let signs = Vec3::new(sign_of(sigma.x), sign_of(sigma.y), sign_of(sigma.z));
    // Principal Hencky strains on clamped magnitudes.
    let eps = Vec3::new(hencky(sigma.x), hencky(sigma.y), hencky(sigma.z));

    let mean = (eps.x + eps.y + eps.z) / 3.0;
    let dev = eps - Vec3::splat(mean);
    let dev_norm = dev.length();

    let yield_eff = model.effective_yield(state.accumulated_strain);

    // Still inside the yield surface (or no deviatoric direction): fully elastic.
    if dev_norm <= MIN_DEVIATORIC_NORM || dev_norm <= yield_eff {
        return PlasticStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            yielded: false,
        };
    }

    // Radial return: project the deviatoric strain back onto the surface,
    // keeping the volumetric part (isochoric plastic flow).
    let increment = dev_norm - yield_eff;
    let dev_returned = dev * (yield_eff / dev_norm);
    let eps_elastic = Vec3::splat(mean) + dev_returned;

    let sigma_elastic = Vec3::new(
        signs.x * stretch_from_log(eps_elastic.x),
        signs.y * stretch_from_log(eps_elastic.y),
        signs.z * stretch_from_log(eps_elastic.z),
    );
    let fe_new = svd.u * Mat3::from_diagonal(sigma_elastic) * svd.v.transpose();

    // Fold the consumed deformation into the plastic gradient: Fₚ = Fₑ⁻¹ · F.
    state.plastic_gradient = fe_new.inverse() * f_total;
    state.accumulated_strain += increment;

    PlasticStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        yielded: true,
    }
}

/// Principal Hencky strain `ln|σ|` of a (clamped) signed singular value,
/// evaluated in `f64` for deterministic libm-free precision then narrowed.
#[inline]
fn hencky(sigma: f32) -> f32 {
    (f64::from(sigma.abs().max(MIN_STRETCH)).ln()) as f32
}

/// Inverse of [`hencky`]: the stretch magnitude `exp(ε)` for an elastic log
/// strain, evaluated in `f64` for precision.
#[inline]
fn stretch_from_log(eps: f32) -> f32 {
    (f64::from(eps).exp()) as f32
}

/// Sign helper that treats exact zero as `+1` so a collapsed axis reconstructs
/// to a positive stretch rather than vanishing.
#[inline]
fn sign_of(x: f32) -> f32 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_mat(a: Mat3, b: Mat3, tol: f32) -> bool {
        (0..3).all(|c| (a.col(c) - b.col(c)).length() < tol)
    }

    #[test]
    fn params_validate_ranges() {
        assert!(PlasticModel::new(0.1, 0.0).is_some());
        assert!(PlasticModel::new(0.1, 2.0).is_some());
        assert!(PlasticModel::new(0.0, 0.0).is_none(), "zero yield rejected");
        assert!(PlasticModel::new(-0.1, 0.0).is_none());
        assert!(PlasticModel::new(0.1, -0.5).is_none(), "negative hardening");
        assert!(PlasticModel::new(f32::NAN, 0.0).is_none());
        assert!(PlasticModel::new(0.1, f32::INFINITY).is_none());
    }

    #[test]
    fn rest_state_is_identity() {
        let s = PlasticState::rest();
        assert_eq!(s.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(s.accumulated_strain(), 0.0);
        assert_eq!(PlasticState::default(), s);
    }

    #[test]
    fn small_strain_stays_elastic() {
        let model = PlasticModel::new(0.2, 0.0).unwrap();
        let mut state = PlasticState::rest();
        // Shear/stretch whose deviatoric Hencky strain is below yield.
        let f = Mat3::from_diagonal(Vec3::new(1.05, 1.0, 0.96));
        let step = return_map(f, &model, &mut state);
        assert!(!step.yielded);
        assert_eq!(step.plastic_increment, 0.0);
        assert!(approx_mat(step.elastic_gradient, f, 1e-6));
        assert_eq!(state.plastic_gradient(), Mat3::IDENTITY, "Fp untouched");
        assert_eq!(state.accumulated_strain(), 0.0);
    }

    #[test]
    fn large_deviatoric_strain_yields_and_projects_to_surface() {
        let model = PlasticModel::new(0.1, 0.0).unwrap();
        let mut state = PlasticState::rest();
        // Strong uniaxial stretch => deviatoric strain well past yield.
        let f = Mat3::from_diagonal(Vec3::new(1.6, 0.9, 0.9));
        let step = return_map(f, &model, &mut state);
        assert!(step.yielded);
        assert!(step.plastic_increment > 0.0);
        assert!(state.accumulated_strain() > 0.0);

        // Multiplicative split must hold: F == Fe * Fp.
        let recon = step.elastic_gradient * state.plastic_gradient();
        assert!(approx_mat(recon, f, 1e-4), "F = Fe*Fp reconstruction");

        // Elastic deviatoric Hencky strain must sit exactly on the yield
        // surface (radius == yield_strain, since no hardening yet).
        let svd = svd3(step.elastic_gradient);
        let eps = Vec3::new(
            hencky(svd.sigma.x),
            hencky(svd.sigma.y),
            hencky(svd.sigma.z),
        );
        let mean = (eps.x + eps.y + eps.z) / 3.0;
        let dev_norm = (eps - Vec3::splat(mean)).length();
        assert!(
            (dev_norm - 0.1).abs() < 1e-4,
            "returned deviatoric norm {dev_norm} should equal yield 0.1"
        );
    }

    #[test]
    fn plastic_flow_is_volume_preserving() {
        let model = PlasticModel::new(0.05, 0.0).unwrap();
        let mut state = PlasticState::rest();
        let f = Mat3::from_diagonal(Vec3::new(1.5, 1.1, 0.8));
        let _ = return_map(f, &model, &mut state);
        let det_fp = state.plastic_gradient().determinant();
        assert!(
            (det_fp - 1.0).abs() < 1e-4,
            "isochoric plastic flow keeps det Fp == 1, got {det_fp}"
        );
    }

    #[test]
    fn pure_hydrostatic_strain_never_yields() {
        let model = PlasticModel::new(0.01, 0.0).unwrap();
        let mut state = PlasticState::rest();
        // Large uniform compression: huge volumetric strain, zero deviatoric.
        let f = Mat3::from_diagonal(Vec3::splat(0.5));
        let step = return_map(f, &model, &mut state);
        assert!(!step.yielded, "J2 ignores hydrostatic pressure");
        assert_eq!(state.accumulated_strain(), 0.0);
        assert_eq!(state.plastic_gradient(), Mat3::IDENTITY);
    }

    #[test]
    fn hardening_raises_the_yield_threshold() {
        // Soft (no hardening) flows more than a hardening material under the
        // same load.
        let soft = PlasticModel::new(0.05, 0.0).unwrap();
        let hard = PlasticModel::new(0.05, 5.0).unwrap();
        let f = Mat3::from_diagonal(Vec3::new(1.4, 0.95, 0.9));

        let mut s_soft = PlasticState::rest();
        let step_soft = return_map(f, &soft, &mut s_soft);
        let mut s_hard = PlasticState::rest();
        let step_hard = return_map(f, &hard, &mut s_hard);

        assert!(step_soft.yielded && step_hard.yielded);
        // Both yield on the first step with the same base threshold, but the
        // effective yield only diverges once strain accumulates. Verify the
        // model's effective-yield curve directly.
        assert_eq!(soft.effective_yield(0.3), 0.05);
        assert!(hard.effective_yield(0.3) > 0.05);

        // After the first increment the hardening model's next-step threshold
        // has grown, so a second identical load consumes less plastic strain.
        let inc_hard_2 = return_map(f, &hard, &mut s_hard).plastic_increment;
        let inc_soft_2 = return_map(f, &soft, &mut s_soft).plastic_increment;
        assert!(
            inc_hard_2 < inc_soft_2 + 1e-6,
            "hardening should not flow more than perfectly-plastic"
        );
    }

    #[test]
    fn repeated_loading_accumulates_monotonically() {
        let model = PlasticModel::new(0.08, 0.0).unwrap();
        let mut state = PlasticState::rest();
        let f = Mat3::from_diagonal(Vec3::new(1.3, 0.95, 0.9));
        let a = return_map(f, &model, &mut state).plastic_increment;
        let after_first = state.accumulated_strain();
        // Load further in the same direction.
        let f2 = Mat3::from_diagonal(Vec3::new(1.6, 0.9, 0.85));
        let _ = return_map(f2, &model, &mut state);
        assert!(a > 0.0);
        assert!(
            state.accumulated_strain() >= after_first,
            "accumulated plastic strain is non-decreasing"
        );
    }

    #[test]
    fn inverted_predictor_stays_finite() {
        let model = PlasticModel::new(0.05, 0.0).unwrap();
        let mut state = PlasticState::rest();
        // Reflection (det < 0) plus stretch: must not NaN.
        let f = Mat3::from_cols(
            Vec3::new(-1.5, 0.0, 0.0),
            Vec3::new(0.0, 1.1, 0.0),
            Vec3::new(0.0, 0.0, 0.9),
        );
        let step = return_map(f, &model, &mut state);
        assert!(step.elastic_gradient.col(0).is_finite());
        assert!(step.elastic_gradient.col(1).is_finite());
        assert!(step.elastic_gradient.col(2).is_finite());
        let recon = step.elastic_gradient * state.plastic_gradient();
        assert!(approx_mat(recon, f, 1e-3));
    }

    #[test]
    fn is_deterministic() {
        let model = PlasticModel::new(0.07, 1.0).unwrap();
        let f = Mat3::from_diagonal(Vec3::new(1.45, 0.92, 0.88));
        let mut a = PlasticState::rest();
        let mut b = PlasticState::rest();
        let sa = return_map(f, &model, &mut a);
        let sb = return_map(f, &model, &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
