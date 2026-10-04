//! Stomakhin-style snow elastoplasticity for a tetrahedral `FEM` element.
//!
//! Snow fractures in tension and compacts in compression, and once it has
//! yielded it keeps the new shape. The classic animation model (Stomakhin
//! et al., clean-room reimplementation here) captures this with a
//! **singular-value box projection** on top of the crate's multiplicative split
//!
//! ```text
//! F = Fₑ · Fₚ
//! ```
//!
//! Each step strips the stored plastic deformation to form the elastic
//! predictor `Fₑ_trial = F · Fₚ⁻¹`, takes its signed SVD `U Σ Vᵀ`, and clamps
//! every principal stretch back into the elastic box
//!
//! ```text
//! σᵢ ∈ [1 − θ_c, 1 + θ_s]
//! ```
//!
//! where `θ_c` is the *critical compression* and `θ_s` the *critical stretch*.
//! Any excess stretch past those limits is irreversible and folded into `Fₚ`,
//! so a tet squeezed past `1 − θ_c` stays compacted and one pulled past
//! `1 + θ_s` tears. Unlike the deviatoric J2 / Drucker–Prager models in this
//! crate, snow yields on the *volumetric* principal stretches directly.
//!
//! Snow also **hardens**: as it compacts (`det Fₚ < 1`) it becomes markedly
//! stiffer. [`hardened_lame`] exposes the Stomakhin exponential stiffening
//! `μ, λ ↦ (μ, λ) · exp(ξ (1 − det Fₚ))` so the caller can feed hardened Lamé
//! parameters to the stress routine.
//!
//! Clean-room implementation over the crate's own [`svd3`]; the hardening
//! exponential runs in `f64` for deterministic, libm-free precision. No Unreal
//! Engine source or derived code.

use crate::collider::tet_fem_constitutive::LameParameters;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before projection so an
/// inverted or collapsed predictor never produces a non-finite result.
const MIN_STRETCH: f32 = 1e-4;

/// Material limits for Stomakhin snow plasticity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowModel {
    /// Critical compression `θ_c ∈ (0, 1)`: principal stretches below
    /// `1 − θ_c` flow plastically. Larger == the snow compacts more easily.
    critical_compression: f32,
    /// Critical stretch `θ_s ≥ 0`: principal stretches above `1 + θ_s` flow
    /// plastically (the snow tears). `0` means it fractures immediately in
    /// tension.
    critical_stretch: f32,
    /// Hardening coefficient `ξ ≥ 0`: compacted snow (`det Fₚ < 1`) stiffens as
    /// `exp(ξ (1 − det Fₚ))`.
    hardening: f32,
}

impl SnowModel {
    /// Creates a snow model, validating `θ_c ∈ (0, 1)`, `θ_s ≥ 0`, `ξ ≥ 0`.
    #[must_use]
    pub fn new(critical_compression: f32, critical_stretch: f32, hardening: f32) -> Option<Self> {
        if critical_compression.is_finite()
            && critical_compression > 0.0
            && critical_compression < 1.0
            && critical_stretch.is_finite()
            && critical_stretch >= 0.0
            && hardening.is_finite()
            && hardening >= 0.0
        {
            Some(Self {
                critical_compression,
                critical_stretch,
                hardening,
            })
        } else {
            None
        }
    }

    /// Lower bound of the elastic stretch box, `1 − θ_c`.
    #[must_use]
    pub fn lower_stretch(&self) -> f32 {
        1.0 - self.critical_compression
    }

    /// Upper bound of the elastic stretch box, `1 + θ_s`.
    #[must_use]
    pub fn upper_stretch(&self) -> f32 {
        1.0 + self.critical_stretch
    }

    /// Critical compression `θ_c`.
    #[must_use]
    pub fn critical_compression(&self) -> f32 {
        self.critical_compression
    }

    /// Critical stretch `θ_s`.
    #[must_use]
    pub fn critical_stretch(&self) -> f32 {
        self.critical_stretch
    }

    /// Hardening coefficient `ξ`.
    #[must_use]
    pub fn hardening(&self) -> f32 {
        self.hardening
    }
}

/// Persistent per-element snow plastic state carried across simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowState {
    plastic_gradient: Mat3,
}

impl SnowState {
    /// The undeformed rest state: `Fₚ = I`.
    #[must_use]
    pub fn rest() -> Self {
        Self {
            plastic_gradient: Mat3::IDENTITY,
        }
    }

    /// Current plastic deformation gradient `Fₚ`.
    #[must_use]
    pub fn plastic_gradient(&self) -> Mat3 {
        self.plastic_gradient
    }

    /// Plastic volume ratio `det Fₚ` (`< 1` once the snow has compacted).
    #[must_use]
    pub fn plastic_volume_ratio(&self) -> f32 {
        self.plastic_gradient.determinant()
    }
}

impl Default for SnowState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single snow return-mapping step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowStep {
    /// Elastic deformation gradient `Fₑ` to feed the stress routine.
    pub elastic_gradient: Mat3,
    /// Total principal-stretch excess clamped away this step (`0` when elastic).
    pub plastic_increment: f32,
    /// `true` when at least one principal stretch left the elastic box.
    pub clamped: bool,
}

/// Performs the Stomakhin elastic-predictor / box-projection update for one
/// element.
///
/// Given the total deformation gradient `f_total` and the carried
/// [`SnowState`], clamps the predictor's principal stretches into the elastic
/// box and folds the removed stretch into `Fₚ`. The returned elastic gradient
/// always satisfies `f_total ≈ Fₑ · Fₚ` with the updated `Fₚ`.
#[must_use]
pub fn return_map_snow(f_total: Mat3, model: &SnowModel, state: &mut SnowState) -> SnowStep {
    let fp_inv = state.plastic_gradient.inverse();
    let fe_trial = f_total * fp_inv;

    let svd = svd3(fe_trial);
    let sigma = svd.sigma;

    let lo = model.lower_stretch();
    let hi = model.upper_stretch();

    let clamped = Vec3::new(
        clamp_stretch(sigma.x, lo, hi),
        clamp_stretch(sigma.y, lo, hi),
        clamp_stretch(sigma.z, lo, hi),
    );

    let increment =
        (clamped.x - sigma.x).abs() + (clamped.y - sigma.y).abs() + (clamped.z - sigma.z).abs();

    if increment == 0.0 {
        return SnowStep {
            elastic_gradient: fe_trial,
            plastic_increment: 0.0,
            clamped: false,
        };
    }

    let fe_new = svd.u * Mat3::from_diagonal(clamped) * svd.v.transpose();
    state.plastic_gradient = fe_new.inverse() * f_total;

    SnowStep {
        elastic_gradient: fe_new,
        plastic_increment: increment,
        clamped: true,
    }
}

/// Returns `base` scaled by the Stomakhin hardening factor
/// `exp(ξ (1 − det Fₚ))`, so compacted snow (`det Fₚ < 1`) stiffens.
///
/// Both Lamé parameters receive the same multiplier. The exponential is clamped
/// to a finite range so extreme compaction cannot overflow.
#[must_use]
pub fn hardened_lame(
    model: &SnowModel,
    state: &SnowState,
    base: &LameParameters,
) -> LameParameters {
    let jp = state.plastic_volume_ratio();
    let exponent = f64::from(model.hardening) * f64::from(1.0 - jp);
    // Guard against overflow from a near-singular plastic gradient.
    let factor = exponent.clamp(-40.0, 40.0).exp() as f32;
    LameParameters::new(base.mu * factor, base.lambda * factor)
}

/// Clamps a signed principal stretch's magnitude into `[lo, hi]`, preserving
/// its sign, after flooring the magnitude at [`MIN_STRETCH`].
#[inline]
fn clamp_stretch(sigma: f32, lo: f32, hi: f32) -> f32 {
    let sign = if sigma < 0.0 { -1.0 } else { 1.0 };
    let magnitude = sigma.abs().max(MIN_STRETCH).clamp(lo, hi);
    sign * magnitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.3).unwrap())
    }

    fn signed_stretches(f: Mat3) -> Vec3 {
        svd3(f).sigma
    }

    #[test]
    fn params_validate_ranges() {
        assert!(SnowModel::new(0.025, 0.0075, 10.0).is_some());
        assert!(SnowModel::new(0.0, 0.01, 1.0).is_none(), "zero compression");
        assert!(SnowModel::new(1.0, 0.01, 1.0).is_none(), "full compression");
        assert!(
            SnowModel::new(0.1, -0.01, 1.0).is_none(),
            "negative stretch"
        );
        assert!(
            SnowModel::new(0.1, 0.01, -1.0).is_none(),
            "negative hardening"
        );
        assert!(SnowModel::new(f32::NAN, 0.01, 1.0).is_none());
    }

    #[test]
    fn rest_state_is_identity() {
        let s = SnowState::rest();
        assert_eq!(s.plastic_gradient(), Mat3::IDENTITY);
        assert_eq!(s.plastic_volume_ratio(), 1.0);
        assert_eq!(SnowState::default(), s);
    }

    #[test]
    fn inside_the_box_stays_elastic() {
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        // All stretches within [0.8, 1.1].
        let f = Mat3::from_diagonal(Vec3::new(1.05, 0.9, 0.85));
        let step = return_map_snow(f, &model, &mut state);
        assert!(!step.clamped);
        assert_eq!(step.plastic_increment, 0.0);
        assert_eq!(state, SnowState::rest(), "elastic step leaves Fp = I");
        assert_eq!(step.elastic_gradient, f);
    }

    #[test]
    fn over_compression_clamps_to_lower_bound() {
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let lo = model.lower_stretch();
        let mut state = SnowState::rest();
        // 0.5 is below 1 - 0.2 = 0.8 ⇒ must be clamped to 0.8.
        let f = Mat3::from_diagonal(Vec3::new(0.5, 0.9, 0.9));
        let step = return_map_snow(f, &model, &mut state);
        assert!(step.clamped);
        let sigma = signed_stretches(step.elastic_gradient);
        let min_sigma = sigma.x.min(sigma.y).min(sigma.z);
        assert!(
            (min_sigma - lo).abs() < 1e-4,
            "smallest elastic stretch should sit on the lower bound {lo}, got {min_sigma}"
        );
        // F = Fe · Fp reconstruction.
        let recon = step.elastic_gradient * state.plastic_gradient();
        assert!(
            (0..3).all(|c| (recon.col(c) - f.col(c)).length() < 1e-3),
            "F = Fe·Fp reconstruction"
        );
        // Plastic compaction: det Fp < 1.
        assert!(state.plastic_volume_ratio() < 1.0);
    }

    #[test]
    fn over_stretch_clamps_to_upper_bound() {
        let model = SnowModel::new(0.2, 0.05, 10.0).unwrap();
        let hi = model.upper_stretch();
        let mut state = SnowState::rest();
        // 1.4 exceeds 1 + 0.05 = 1.05 ⇒ clamp to 1.05.
        let f = Mat3::from_diagonal(Vec3::new(1.4, 1.0, 0.95));
        let step = return_map_snow(f, &model, &mut state);
        assert!(step.clamped);
        let sigma = signed_stretches(step.elastic_gradient);
        let max_sigma = sigma.x.max(sigma.y).max(sigma.z);
        assert!(
            (max_sigma - hi).abs() < 1e-4,
            "largest elastic stretch should sit on the upper bound {hi}, got {max_sigma}"
        );
    }

    #[test]
    fn hardening_stiffens_compacted_snow() {
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let base = lame();
        let mut state = SnowState::rest();
        // At rest the hardening factor is exp(0) = 1.
        let rest_lame = hardened_lame(&model, &state, &base);
        assert!((rest_lame.mu - base.mu).abs() < 1.0);

        // Compact the snow, then the hardened moduli must exceed the base.
        let _ = return_map_snow(
            Mat3::from_diagonal(Vec3::new(0.5, 0.6, 0.9)),
            &model,
            &mut state,
        );
        assert!(state.plastic_volume_ratio() < 1.0);
        let hard = hardened_lame(&model, &state, &base);
        assert!(
            hard.mu > base.mu,
            "compacted snow should stiffen: {} !> {}",
            hard.mu,
            base.mu
        );
        assert!(hard.lambda > base.lambda);
    }

    #[test]
    fn no_hardening_leaves_moduli_unchanged() {
        let model = SnowModel::new(0.2, 0.1, 0.0).unwrap();
        let base = lame();
        let mut state = SnowState::rest();
        let _ = return_map_snow(
            Mat3::from_diagonal(Vec3::new(0.5, 0.6, 0.9)),
            &model,
            &mut state,
        );
        let out = hardened_lame(&model, &state, &base);
        assert!((out.mu - base.mu).abs() < 1e-1);
        assert!((out.lambda - base.lambda).abs() < 1e-1);
    }

    #[test]
    fn inverted_predictor_stays_finite() {
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        let f = Mat3::from_cols(
            Vec3::new(-1.5, 0.0, 0.0),
            Vec3::new(0.0, 1.1, 0.0),
            Vec3::new(0.0, 0.0, 0.9),
        );
        let step = return_map_snow(f, &model, &mut state);
        assert!((0..3).all(|c| step.elastic_gradient.col(c).is_finite()));
        assert!((0..3).all(|c| state.plastic_gradient().col(c).is_finite()));
    }

    #[test]
    fn is_deterministic() {
        let model = SnowModel::new(0.15, 0.08, 5.0).unwrap();
        let f = Mat3::from_diagonal(Vec3::new(1.45, 0.6, 0.78));
        let mut a = SnowState::rest();
        let mut b = SnowState::rest();
        let sa = return_map_snow(f, &model, &mut a);
        let sb = return_map_snow(f, &model, &mut b);
        assert_eq!(sa, sb);
        assert_eq!(a, b);
    }
}
