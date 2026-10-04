//! Degraded nodal internal forces for a tetrahedral `FEM` element.
//!
//! This module is the bridge between the scalar continuum-damage model in
//! [`tet_fem_damage`](super::tet_fem_damage) and the hyperelastic stress
//! response in [`tet_fem_internal_force`](super::tet_fem_internal_force), in
//! the same way that
//! [`tet_fem_elastoplastic_force`](super::tet_fem_elastoplastic_force) bridges
//! plasticity and stress.
//!
//! An undamaged element develops the full first Piola–Kirchhoff stress
//! `P(F)`. Isotropic continuum damage multiplies that stress by a scalar
//! degradation factor `(1 − D) ∈ (0, 1]`, so a cracking element gradually
//! loses its load-carrying capacity. The nodal force becomes
//!
//! ```text
//! fᵢ = -(1 − D) · V · P(F) · gᵢ
//! ```
//!
//! where `V` is the absolute rest volume, `P` is the first Piola–Kirchhoff
//! stress of a [`HyperelasticModel`](crate::collider::HyperelasticModel), and
//! `gᵢ` is the reference-space shape-function gradient stored on the element.
//! Unlike plasticity the deformation gradient is *not* split: damage keeps the
//! full elastic stress and simply scales it, which is why the energy is scaled
//! by the identical factor rather than evaluated on a reduced gradient.
//!
//! Evaluating the force advances the damage state exactly once (via
//! [`update_damage`]): the tensile equivalent strain is measured, the
//! irreversible history variable `κ` is updated, and the resulting degradation
//! is folded into the stress. The returned [`DamageStep`] exposes the current
//! equivalent strain, scalar damage, degradation factor, and whether the step
//! advanced the history, so callers can drive softening, visualisation, or
//! fracture heuristics without re-deriving the split.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`DamageState`] and performs no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_damage::{update_damage, DamageModel, DamageState, DamageStep};
use glam::{Mat3, Vec3};

/// Result of a single degraded element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamagedForce {
    /// Nodal degraded forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The damage-update outcome that produced these forces.
    pub step: DamageStep,
}

/// Nodal degraded forces `fᵢ = -(1 − D) · V · P(F) · gᵢ` for one tetrahedral
/// element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`update_damage`] against the carried `state` (which is mutated in place
/// whenever the tensile equivalent strain exceeds the stored history), and the
/// resulting degradation factor scales the hyperelastic stress. When the
/// material is undamaged the forces are identical (to within rounding) to
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because the degradation factor is `1`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding even when the element is damaged, so a degraded element
/// still exerts no net force on its centre of mass.
#[must_use]
pub fn element_damaged_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    damage_model: &DamageModel,
    state: &mut DamageState,
) -> DamagedForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = update_damage(f_total, damage_model, state);
    let piola = model.first_piola(f_total, lame);
    // Fold the degradation factor into the (scalar) volume weight so the stress
    // tensor is scaled uniformly without constructing a scaled `Mat3`.
    let scaled = element.rest_volume.abs() * step.degradation;
    let g = element.shape_gradients;
    let forces = [
        -scaled * (piola * g[0]),
        -scaled * (piola * g[1]),
        -scaled * (piola * g[2]),
        -scaled * (piola * g[3]),
    ];
    DamagedForce { forces, step }
}

/// Degraded elastic potential energy `U = (1 − D) · V · Ψ(F)` for an element
/// whose degradation factor was produced by a prior
/// [`element_damaged_force`] evaluation.
///
/// This is a pure read of the stored energy, scaled by the same degradation
/// factor applied to the force; it does not advance the damage state. Pass the
/// total deformation gradient together with [`DamageStep::degradation`] to keep
/// a single state advance per step.
#[must_use]
pub fn damaged_elastic_potential_energy(
    element: &TetFemElement,
    model: HyperelasticModel,
    lame: &LameParameters,
    f_total: Mat3,
    degradation: f32,
) -> f32 {
    let volume = element.rest_volume.abs();
    volume * degradation * model.strain_energy_density(f_total, lame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_internal_force::{element_energy, element_internal_force};
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    const REST: [Vec3; 4] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.4).unwrap())
    }

    fn element() -> TetFemElement {
        TetFemElement::from_rest(REST[0], REST[1], REST[2], REST[3], 1e-12).unwrap()
    }

    /// Apply an affine map `x = A·X` to the rest nodes so the resulting
    /// deformation gradient equals `A` exactly.
    fn affine(a: Mat3) -> [Vec3; 4] {
        [a * REST[0], a * REST[1], a * REST[2], a * REST[3]]
    }

    fn total(forces: [Vec3; 4]) -> Vec3 {
        forces[0] + forces[1] + forces[2] + forces[3]
    }

    /// A damage model that cracks early so stretch tests exercise softening,
    /// yet leaves tiny strains below onset for the pristine comparison.
    fn damage() -> DamageModel {
        DamageModel::new(0.05, 0.3, 0.0).unwrap()
    }

    #[test]
    fn virgin_element_matches_pristine_force() {
        // A tiny deformation stays below the damage onset, so the degraded
        // force must equal the pristine elastic force and leave the damage
        // state untouched.
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let pos = affine(Mat3::from_diagonal(Vec3::new(1.01, 0.995, 1.003)));

        let out = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        let pristine = element_internal_force(&el, pos, HyperelasticModel::StableNeoHookean, &lame);

        assert!(
            (out.step.degradation - 1.0).abs() < 1e-6,
            "sub-onset strain must stay fully intact, got {}",
            out.step.degradation
        );
        assert!(state.damage() == 0.0, "sub-onset strain accrues no damage");
        assert!(
            (state.degradation() - 1.0).abs() < 1e-6,
            "sub-onset strain stays fully intact"
        );
        for i in 0..4 {
            assert!(
                (out.forces[i] - pristine[i]).length() < 1e-1,
                "node {i}: damaged {:?} vs pristine {:?}",
                out.forces[i],
                pristine[i]
            );
        }
    }

    #[test]
    fn forces_sum_to_zero_even_when_damaged() {
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let out = element_damaged_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.4, 1.0, 1.0))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        assert!(
            out.step.degradation < 1.0,
            "large tensile stretch should damage"
        );
        assert!(
            total(out.forces).length() < 1e-1,
            "net internal force must vanish, got {:?}",
            total(out.forces)
        );
    }

    #[test]
    fn stretch_past_onset_scales_force_down() {
        // Each nodal force component equals the pristine force scaled by the
        // degradation factor reported for the same configuration.
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let pos = affine(Mat3::from_diagonal(Vec3::new(1.4, 1.0, 1.0)));

        let out = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        let pristine = element_internal_force(&el, pos, HyperelasticModel::StableNeoHookean, &lame);

        assert!(
            out.step.degradation < 1.0,
            "tensile stretch past onset must soften"
        );
        for i in 0..4 {
            let expected = pristine[i] * out.step.degradation;
            assert!(
                (out.forces[i] - expected).length() <= 1e-3 * pristine[i].length().max(1.0),
                "node {i}: {:?} != pristine*degradation {:?}",
                out.forces[i],
                expected
            );
        }
    }

    #[test]
    fn pure_compression_leaves_force_pristine() {
        // Damage is tension driven: positive principal strains only. A purely
        // compressive state produces no tensile equivalent strain, so the
        // element stays intact.
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let pos = affine(Mat3::from_diagonal(Vec3::new(0.6, 0.6, 0.6)));

        let out = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        let pristine = element_internal_force(&el, pos, HyperelasticModel::StableNeoHookean, &lame);

        assert!(
            (out.step.degradation - 1.0).abs() < 1e-6,
            "compression carries no tensile damage, got {}",
            out.step.degradation
        );
        for i in 0..4 {
            assert!(
                (out.forces[i] - pristine[i]).length() < 1e-1,
                "node {i}: damaged {:?} vs pristine {:?}",
                out.forces[i],
                pristine[i]
            );
        }
    }

    #[test]
    fn damage_advances_state() {
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let out = element_damaged_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.4, 1.0, 1.0))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        assert!(out.step.advanced, "crossing onset should advance history");
        assert!(
            state.kappa() > model.onset_strain(),
            "history variable must grow"
        );
        assert!(state.damage() > 0.0, "scalar damage must be positive");
    }

    #[test]
    fn unloading_uses_frozen_degradation() {
        // A large stretch damages the element; a subsequent small stretch must
        // reuse the frozen (irreversible) degradation rather than healing.
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();

        let big = element_damaged_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.5, 1.0, 1.0))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        let frozen = big.step.degradation;
        assert!(frozen < 1.0, "big stretch must damage");

        let small_pos = affine(Mat3::from_diagonal(Vec3::new(1.02, 1.0, 1.0)));
        let small = element_damaged_force(
            &el,
            small_pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        assert!(
            (small.step.degradation - frozen).abs() < 1e-6,
            "unloading must not heal: {} vs frozen {}",
            small.step.degradation,
            frozen
        );
        assert!(!small.step.advanced, "sub-history reload must not advance");

        let pristine =
            element_internal_force(&el, small_pos, HyperelasticModel::StableNeoHookean, &lame);
        for i in 0..4 {
            let expected = pristine[i] * frozen;
            assert!(
                (small.forces[i] - expected).length() <= 1e-3 * pristine[i].length().max(1.0),
                "node {i}: reload {:?} != pristine*frozen {:?}",
                small.forces[i],
                expected
            );
        }
    }

    #[test]
    fn energy_scales_by_degradation() {
        let el = element();
        let lame = lame();
        let model = damage();
        let mut state = DamageState::rest();
        let f = Mat3::from_diagonal(Vec3::new(1.4, 1.0, 1.0));
        let pos = affine(f);

        let out = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        let degraded = damaged_elastic_potential_energy(
            &el,
            HyperelasticModel::StableNeoHookean,
            &lame,
            f,
            out.step.degradation,
        );
        let pristine = element_energy(&el, pos, HyperelasticModel::StableNeoHookean, &lame);

        assert!(out.step.degradation < 1.0);
        let expected = pristine * out.step.degradation;
        assert!(
            (degraded - expected).abs() <= 1e-3 * pristine.abs().max(1.0),
            "degraded energy {degraded} != pristine*degradation {expected}"
        );
    }

    #[test]
    fn is_deterministic() {
        let el = element();
        let lame = lame();
        let model = damage();
        let pos = affine(Mat3::from_diagonal(Vec3::new(1.4, 1.0, 1.0)));

        let mut state_a = DamageState::rest();
        let a = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state_a,
        );
        let mut state_b = DamageState::rest();
        let b = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state_b,
        );
        assert_eq!(a, b, "identical inputs must produce identical output");
        assert_eq!(
            state_a, state_b,
            "identical inputs must leave identical state"
        );
    }

    #[test]
    fn fully_cracked_force_floors_at_residual() {
        // With a residual-stiffness floor the degradation can never reach zero,
        // so even a severely stretched element retains a scaled restoring
        // force proportional to that floor.
        let el = element();
        let lame = lame();
        let model = DamageModel::new(0.05, 0.1, 0.2).unwrap();
        let mut state = DamageState::rest();
        let pos = affine(Mat3::from_diagonal(Vec3::new(2.5, 1.0, 1.0)));

        let out = element_damaged_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &model,
            &mut state,
        );
        assert!(
            out.step.degradation >= model.residual_stiffness() - 1e-6,
            "degradation {} must not fall below residual {}",
            out.step.degradation,
            model.residual_stiffness()
        );
        assert!(
            total(out.forces).length() < 1e-1,
            "residual force must still be balanced, got {:?}",
            total(out.forces)
        );
    }
}
