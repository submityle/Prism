//! Isotropic scalar continuum damage for tetrahedral FEM.
//!
//! Brittle and quasi-brittle materials — concrete, ceramics, bone, bonded
//! aggregates — do not yield and flow like metals; instead their stiffness
//! *degrades* as microcracks open under tensile strain. This module models that
//! with the classical **scalar damage** variable `D ∈ [0, 1)`: the effective
//! stress is the pristine (undamaged) stress scaled by a degradation factor
//!
//! ```text
//!   σ = (1 − D) · σ₀(F)
//! ```
//!
//! where `σ₀` is whatever hyperelastic stress the element would carry intact.
//! `D = 0` is virgin material and `D → 1` is fully cracked. Damage is driven by
//! a **tension-sensitive equivalent strain** (Mazars) built from the positive
//! parts of the principal Hencky strains,
//!
//! ```text
//!   ε_eq = sqrt( Σ ⟨εᵢ⟩₊² ),   ⟨x⟩₊ = max(x, 0),
//! ```
//!
//! so hydrostatic compression (all `εᵢ < 0`) produces no damage, matching the
//! observation that cracks are opened by tension, not confinement.
//!
//! Irreversibility is enforced through a monotone history variable
//! `κ = max_t ε_eq(t)`: damage never heals when the load is removed, it only
//! grows when a new strain peak is reached. The softening law is the standard
//! **exponential** form
//!
//! ```text
//!   D(κ) = 1 − (κ₀ / κ) · exp( −(κ − κ₀) / κ_f )   for κ > κ₀,   else 0,
//! ```
//!
//! continuous at onset (`D(κ₀) = 0`), monotonically increasing, and asymptotic
//! to full damage. A `residual_stiffness ∈ [0, 1)` floor clamps the degradation
//! factor `(1 − D)` from below so a fully cracked element keeps a small, stable
//! stiffness (avoiding singular systems in an implicit solver).
//!
//! This is a self-contained, per-element, integrator-agnostic kernel: feed it
//! the total deformation gradient, get back the degradation factor to multiply
//! into the element's stress / force / energy, and keep the mutated
//! [`DamageState`].
//!
//! # Attribution
//!
//! Clean-room implementation of the textbook Mazars-type isotropic scalar
//! damage model with exponential strain softening, in Hencky-strain space over
//! the crate's own [`svd3`]. No Unreal Engine source or derived code.

use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// Principal stretches are clamped to this magnitude before taking a log, so an
/// inverted or collapsed element never produces a non-finite strain.
const MIN_STRETCH: f32 = 1e-4;

/// Material parameters for isotropic scalar damage with exponential softening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageModel {
    /// Equivalent-strain threshold `κ₀ > 0` at which damage initiates. Below
    /// this the material is intact (`D = 0`).
    onset_strain: f32,
    /// Softening modulus `κ_f > 0` controlling how quickly damage grows past
    /// onset: larger `κ_f` == more ductile (slower stiffness loss).
    softening: f32,
    /// Retained degradation floor `(1 − D)_min ∈ [0, 1)`: a fully cracked
    /// element keeps at least this fraction of its stiffness. Equivalently it
    /// caps damage at `D_max = 1 − residual_stiffness`.
    residual_stiffness: f32,
}

impl DamageModel {
    /// Builds a model, validating that `onset_strain > 0`, `softening > 0`, and
    /// `residual_stiffness ∈ [0, 1)`, all finite.
    #[must_use]
    pub fn new(onset_strain: f32, softening: f32, residual_stiffness: f32) -> Option<Self> {
        if !onset_strain.is_finite() || onset_strain <= 0.0 {
            return None;
        }
        if !softening.is_finite() || softening <= 0.0 {
            return None;
        }
        if !residual_stiffness.is_finite() || !(0.0..1.0).contains(&residual_stiffness) {
            return None;
        }
        Some(Self {
            onset_strain,
            softening,
            residual_stiffness,
        })
    }

    /// Equivalent-strain damage-onset threshold `κ₀`.
    #[must_use]
    pub fn onset_strain(&self) -> f32 {
        self.onset_strain
    }

    /// Softening modulus `κ_f`.
    #[must_use]
    pub fn softening(&self) -> f32 {
        self.softening
    }

    /// Retained degradation floor `(1 − D)_min`.
    #[must_use]
    pub fn residual_stiffness(&self) -> f32 {
        self.residual_stiffness
    }

    /// Maximum attainable damage `D_max = 1 − residual_stiffness`.
    #[must_use]
    pub fn max_damage(&self) -> f32 {
        1.0 - self.residual_stiffness
    }

    /// Scalar damage `D(κ)` for a history peak equivalent strain `κ`,
    /// clamped to `[0, D_max]`.
    ///
    /// Uses the exponential softening law, continuous at onset (`D(κ₀) = 0`).
    #[must_use]
    pub fn damage_at(&self, kappa: f32) -> f32 {
        if kappa <= self.onset_strain {
            return 0.0;
        }
        // D = 1 − (κ₀/κ)·exp(−(κ−κ₀)/κ_f); evaluate exp in f64.
        let ratio = f64::from(self.onset_strain) / f64::from(kappa);
        let decay = (-f64::from(kappa - self.onset_strain) / f64::from(self.softening)).exp();
        let d = 1.0 - ratio * decay;
        (d as f32).clamp(0.0, self.max_damage())
    }

    /// Degradation factor `(1 − D(κ))`, clamped to `[residual_stiffness, 1]`.
    #[must_use]
    pub fn degradation_at(&self, kappa: f32) -> f32 {
        1.0 - self.damage_at(kappa)
    }
}

/// Persistent per-element damage state carried across simulation steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageState {
    /// History variable `κ`: the maximum equivalent strain ever experienced
    /// (monotonically non-decreasing, enforcing irreversibility).
    kappa: f32,
    /// Current scalar damage `D ∈ [0, D_max]`, a cached function of `κ`.
    damage: f32,
}

impl DamageState {
    /// The undamaged virgin state: `κ = 0`, `D = 0`.
    #[must_use]
    pub fn rest() -> Self {
        Self {
            kappa: 0.0,
            damage: 0.0,
        }
    }

    /// Maximum equivalent strain experienced so far.
    #[must_use]
    pub fn kappa(&self) -> f32 {
        self.kappa
    }

    /// Current scalar damage `D`.
    #[must_use]
    pub fn damage(&self) -> f32 {
        self.damage
    }

    /// Current degradation factor `(1 − D)`.
    #[must_use]
    pub fn degradation(&self) -> f32 {
        1.0 - self.damage
    }
}

impl Default for DamageState {
    fn default() -> Self {
        Self::rest()
    }
}

/// Outcome of a single damage update step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageStep {
    /// Tension-sensitive equivalent strain `ε_eq` evaluated this step.
    pub equivalent_strain: f32,
    /// Scalar damage `D` after the (possibly) updated history.
    pub damage: f32,
    /// Degradation factor `(1 − D)` to multiply into the element stress/force.
    pub degradation: f32,
    /// `true` when this step pushed the history peak `κ` higher (new damage
    /// was created rather than merely re-loading below the previous peak).
    pub advanced: bool,
}

/// The Mazars tension-sensitive equivalent strain of a deformation gradient.
///
/// Takes the signed SVD, maps the singular values to principal Hencky strains,
/// and returns `sqrt(Σ ⟨εᵢ⟩₊²)` using only the tensile (positive) parts.
#[must_use]
pub fn equivalent_strain(f_total: Mat3) -> f32 {
    let sigma = svd3(f_total).sigma;
    let eps = Vec3::new(hencky(sigma.x), hencky(sigma.y), hencky(sigma.z));
    let pos = Vec3::new(eps.x.max(0.0), eps.y.max(0.0), eps.z.max(0.0));
    pos.length()
}

/// Advances the damage history for one element given the total deformation
/// gradient and returns the resulting [`DamageStep`].
///
/// Computes the tension-sensitive equivalent strain, raises the irreversible
/// history peak `κ` if the strain is a new maximum, recomputes damage from the
/// softening law, and reports the degradation factor to apply to the element's
/// stress. Non-finite deformation gradients leave the state untouched and
/// report the current (cached) damage with zero equivalent strain.
#[must_use]
pub fn update_damage(f_total: Mat3, model: &DamageModel, state: &mut DamageState) -> DamageStep {
    let eq = equivalent_strain(f_total);

    if !eq.is_finite() {
        return DamageStep {
            equivalent_strain: 0.0,
            damage: state.damage,
            degradation: state.degradation(),
            advanced: false,
        };
    }

    let advanced = eq > state.kappa;
    if advanced {
        state.kappa = eq;
        state.damage = model.damage_at(state.kappa);
    }

    DamageStep {
        equivalent_strain: eq,
        damage: state.damage,
        degradation: state.degradation(),
        advanced,
    }
}

/// Principal Hencky strain `ln|σ|` of a (clamped) singular value, evaluated in
/// `f64` for deterministic libm-free precision then narrowed.
#[inline]
fn hencky(sigma: f32) -> f32 {
    (f64::from(sigma.abs().max(MIN_STRETCH)).ln()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal stretch matrix from principal log strains (`σ = exp(ε)`).
    fn from_log_strain(e: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(
            (f64::from(e.x).exp()) as f32,
            (f64::from(e.y).exp()) as f32,
            (f64::from(e.z).exp()) as f32,
        ))
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(DamageModel::new(0.01, 0.05, 0.0).is_some());
        assert!(DamageModel::new(0.01, 0.05, 0.1).is_some());
        assert!(DamageModel::new(0.0, 0.05, 0.0).is_none(), "zero onset");
        assert!(DamageModel::new(-0.01, 0.05, 0.0).is_none());
        assert!(DamageModel::new(0.01, 0.0, 0.0).is_none(), "zero softening");
        assert!(DamageModel::new(0.01, -0.05, 0.0).is_none());
        assert!(DamageModel::new(0.01, 0.05, 1.0).is_none(), "residual >= 1");
        assert!(DamageModel::new(0.01, 0.05, -0.1).is_none());
        assert!(DamageModel::new(f32::NAN, 0.05, 0.0).is_none());
        assert!(DamageModel::new(0.01, f32::INFINITY, 0.0).is_none());
    }

    #[test]
    fn rest_state_is_virgin() {
        let s = DamageState::rest();
        assert_eq!(s.kappa(), 0.0);
        assert_eq!(s.damage(), 0.0);
        assert_eq!(s.degradation(), 1.0);
        assert_eq!(DamageState::default(), s);
    }

    #[test]
    fn damage_law_is_continuous_at_onset() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        assert_eq!(model.damage_at(0.0), 0.0);
        assert_eq!(model.damage_at(0.02), 0.0, "D(κ₀) == 0");
        assert_eq!(model.damage_at(0.019), 0.0, "below onset");
        assert!(model.damage_at(0.03) > 0.0, "past onset damages");
    }

    #[test]
    fn damage_law_matches_closed_form() {
        let (k0, kf) = (0.02_f32, 0.1_f32);
        let model = DamageModel::new(k0, kf, 0.0).unwrap();
        let kappa = 0.08_f32;
        let ratio = f64::from(k0) / f64::from(kappa);
        let decay = (-f64::from(kappa - k0) / f64::from(kf)).exp();
        let expected = (1.0 - ratio * decay) as f32;
        assert!((model.damage_at(kappa) - expected).abs() < 1e-6);
    }

    #[test]
    fn damage_is_monotone_in_kappa() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut prev = -1.0_f32;
        for i in 0..50 {
            let kappa = 0.02 + 0.01 * i as f32;
            let d = model.damage_at(kappa);
            assert!(d >= prev, "damage must be non-decreasing in κ");
            prev = d;
        }
        assert!(prev > 0.9, "large strain approaches full damage");
    }

    #[test]
    fn compression_does_not_damage() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut state = DamageState::rest();
        // Hydrostatic compression: all principal strains negative.
        let f = from_log_strain(Vec3::new(-0.2, -0.2, -0.2));
        let step = update_damage(f, &model, &mut state);
        assert_eq!(step.equivalent_strain, 0.0, "no tensile strain");
        assert_eq!(step.damage, 0.0);
        assert!(!step.advanced);
        assert_eq!(step.degradation, 1.0);
    }

    #[test]
    fn tension_damages_more_than_equal_compression() {
        let tension = equivalent_strain(from_log_strain(Vec3::new(0.1, -0.05, -0.05)));
        let compression = equivalent_strain(from_log_strain(Vec3::new(-0.1, 0.05, 0.05)));
        // Mazars equivalent strain weights the tensile principal strains more.
        assert!(
            tension > compression,
            "tension-dominated {tension} should exceed compression-dominated {compression}"
        );
    }

    #[test]
    fn stretch_past_onset_degrades_stiffness() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut state = DamageState::rest();
        let f = from_log_strain(Vec3::new(0.15, -0.02, -0.02));
        let step = update_damage(f, &model, &mut state);
        assert!(step.advanced);
        assert!(step.damage > 0.0 && step.damage < 1.0);
        assert!((step.degradation - (1.0 - step.damage)).abs() < 1e-7);
        assert!(step.degradation < 1.0, "stiffness is reduced");
    }

    #[test]
    fn unloading_preserves_damage() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut state = DamageState::rest();

        let loaded = update_damage(
            from_log_strain(Vec3::new(0.15, -0.02, -0.02)),
            &model,
            &mut state,
        );
        assert!(loaded.advanced && loaded.damage > 0.0);
        let peak_kappa = state.kappa();
        let peak_damage = state.damage();

        // Unload to a smaller tensile strain: κ must not shrink, damage frozen.
        let unloaded = update_damage(
            from_log_strain(Vec3::new(0.05, -0.01, -0.01)),
            &model,
            &mut state,
        );
        assert!(
            !unloaded.advanced,
            "reloading below peak creates no new damage"
        );
        assert_eq!(state.kappa(), peak_kappa, "history peak is irreversible");
        assert_eq!(state.damage(), peak_damage, "damage does not heal");
        assert_eq!(unloaded.damage, peak_damage);
    }

    #[test]
    fn reloading_past_peak_grows_damage() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut state = DamageState::rest();

        let first = update_damage(
            from_log_strain(Vec3::new(0.1, 0.0, 0.0)),
            &model,
            &mut state,
        );
        let second = update_damage(
            from_log_strain(Vec3::new(0.2, 0.0, 0.0)),
            &model,
            &mut state,
        );
        assert!(second.advanced);
        assert!(
            second.damage > first.damage,
            "a new strain peak grows damage"
        );
    }

    #[test]
    fn residual_stiffness_floors_degradation() {
        let residual = 0.1_f32;
        let model = DamageModel::new(0.02, 0.05, residual).unwrap();
        // Huge strain -> damage pinned at D_max = 1 − residual.
        let huge = model.damage_at(100.0);
        assert!(
            (huge - model.max_damage()).abs() < 1e-6,
            "damage capped at D_max"
        );
        assert!(
            model.degradation_at(100.0) >= residual - 1e-6,
            "degradation never drops below the residual floor"
        );

        let mut state = DamageState::rest();
        let step = update_damage(
            from_log_strain(Vec3::new(2.0, 1.0, 1.0)),
            &model,
            &mut state,
        );
        assert!(step.degradation >= residual - 1e-6);
    }

    #[test]
    fn non_finite_gradient_is_inert() {
        let model = DamageModel::new(0.02, 0.1, 0.0).unwrap();
        let mut state = DamageState::rest();
        // Pre-damage so we can confirm the cached value is reported unchanged.
        let _ = update_damage(
            from_log_strain(Vec3::new(0.15, 0.0, 0.0)),
            &model,
            &mut state,
        );
        let before = state;

        let bad = Mat3::from_cols(
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let step = update_damage(bad, &model, &mut state);
        assert_eq!(step.equivalent_strain, 0.0);
        assert_eq!(step.damage, before.damage());
        assert_eq!(state, before, "state untouched by a non-finite gradient");
    }

    #[test]
    fn is_deterministic() {
        let model = DamageModel::new(0.02, 0.12, 0.05).unwrap();
        let f = from_log_strain(Vec3::new(0.18, -0.03, 0.04));

        let mut a = DamageState::rest();
        let step_a = update_damage(f, &model, &mut a);
        let mut b = DamageState::rest();
        let step_b = update_damage(f, &model, &mut b);

        assert_eq!(step_a, step_b);
        assert_eq!(a, b);
    }
}
