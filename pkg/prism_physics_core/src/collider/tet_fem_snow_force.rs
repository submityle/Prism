//! Elastoplastic nodal internal forces for a snow tetrahedral `FEM` element.
//!
//! This module is the snow analogue of
//! [`tet_fem_elastoplastic_force`](super::tet_fem_elastoplastic_force): it
//! bridges the Stomakhin box-projection return mapping in
//! [`tet_fem_snow_plasticity`](super::tet_fem_snow_plasticity) with the
//! hyperelastic stress response in
//! [`tet_fem_internal_force`](super::tet_fem_internal_force).
//!
//! Only the elastic part `Fₑ = F · Fₚ⁻¹` of the deformation generates stress,
//! so the nodal force is
//!
//! ```text
//! fᵢ = -V · P(Fₑ; μ', λ') · gᵢ
//! ```
//!
//! where `V` is the absolute rest volume, `gᵢ` the reference-space
//! shape-function gradient, and `P` the first Piola–Kirchhoff stress of a
//! [`HyperelasticModel`](crate::collider::HyperelasticModel). The crucial
//! difference from the generic J2 driver is that snow **hardens as it
//! compacts**: the Lamé parameters fed to `P` are the base moduli scaled by the
//! Stomakhin factor `exp(ξ (1 − det Fₚ))` from
//! [`hardened_lame`](super::tet_fem_snow_plasticity::hardened_lame), evaluated
//! against the plastic state *after* this step's return mapping. A tet squeezed
//! into dense snow therefore stiffens, while one that only tears stays at its
//! base stiffness.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map_snow`]): the elastic predictor's principal stretches are
//! clamped back into the elastic box and any excess is folded into `Fₚ`. The
//! returned [`SnowForce`] exposes the raw [`SnowStep`] and the hardened Lamé
//! parameters so callers can drive visualisation or compaction heuristics
//! without re-deriving the split.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`SnowState`] and performs no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_snow_plasticity::{
    hardened_lame, return_map_snow, SnowModel, SnowState, SnowStep,
};
use glam::{Mat3, Vec3};

/// Result of a single snow elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnowForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The snow return-mapping outcome that produced these forces.
    pub step: SnowStep,
    /// The hardened Lamé parameters used to evaluate the stress this step.
    pub hardened_lame: LameParameters,
}

/// Nodal snow elastoplastic forces `fᵢ = -V · P(Fₑ; μ', λ') · gᵢ` for one
/// tetrahedral element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map_snow`] against the carried `state` (mutated in place when the
/// element yields), and the resulting elastic gradient `Fₑ` is fed to the
/// hyperelastic stress using Lamé parameters hardened by the *post-step*
/// plastic volume ratio. When the snow stays elastic and unyielded the forces
/// match [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions with `base_lame`, because `Fₑ = F · Fₚ⁻¹` reduces to
/// `F` and `det Fₚ = 1` leaves the moduli unscaled.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_snow_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    base_lame: &LameParameters,
    snow_model: &SnowModel,
    state: &mut SnowState,
) -> SnowForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map_snow(f_total, snow_model, state);
    let hardened = hardened_lame(snow_model, state, base_lame);
    let piola = model.first_piola(step.elastic_gradient, &hardened);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    SnowForce {
        forces,
        step,
        hardened_lame: hardened,
    }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ; μ', λ')` for a snow
/// element whose elastic gradient `Fₑ` and hardened Lamé parameters were
/// produced by a prior [`element_snow_force`] call.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`SnowStep::elastic_gradient`] and [`SnowForce::hardened_lame`] from
/// [`element_snow_force`] to keep a single state advance per step and stay
/// consistent with the stiffened stress used for the forces.
#[must_use]
pub fn snow_elastic_potential_energy(
    element: &TetFemElement,
    model: HyperelasticModel,
    hardened_lame: &LameParameters,
    elastic_gradient: Mat3,
) -> f32 {
    let volume = element.rest_volume.abs();
    volume * model.strain_energy_density(elastic_gradient, hardened_lame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_internal_force::element_internal_force;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    const REST: [Vec3; 4] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.3).unwrap())
    }

    fn element() -> TetFemElement {
        TetFemElement::from_rest(REST[0], REST[1], REST[2], REST[3], 1e-12).unwrap()
    }

    /// Applies an affine map `A` to the rest nodes so the resulting deformation
    /// gradient equals `A` exactly.
    fn deform(a: Mat3) -> [Vec3; 4] {
        [a * REST[0], a * REST[1], a * REST[2], a * REST[3]]
    }

    fn sum(forces: &[Vec3; 4]) -> Vec3 {
        forces[0] + forces[1] + forces[2] + forces[3]
    }

    #[test]
    fn rest_configuration_has_no_force() {
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        let out = element_snow_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(!out.step.clamped);
        for f in out.forces {
            assert!(f.length() < 1e-1, "rest force should vanish, got {f}");
        }
        assert_eq!(state, SnowState::rest());
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.5, 0.6, 0.9)));
        let out = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(sum(&out.forces).length() < 1e-2, "net force must vanish");
    }

    #[test]
    fn elastic_step_matches_hyperelastic_internal_force() {
        // A deformation that stays strictly inside the elastic box leaves
        // Fp = I and det Fp = 1, so the hardened moduli equal the base moduli
        // and the forces must match the plain hyperelastic internal force.
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.05, 0.9, 0.95)));
        let out = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(!out.step.clamped);
        assert_eq!(state, SnowState::rest());
        let reference =
            element_internal_force(&el, positions, HyperelasticModel::StableNeoHookean, &lame());
        for i in 0..4 {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "elastic snow force should match hyperelastic force at node {i}: {} vs {}",
                out.forces[i],
                reference[i]
            );
        }
    }

    #[test]
    fn yield_advances_plastic_state() {
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();
        let mut state = SnowState::rest();
        // 0.5 is well below the 0.8 lower bound ⇒ plastic compaction.
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.5, 0.9, 0.9)));
        let out = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(out.step.clamped);
        assert!(out.step.plastic_increment > 0.0);
        assert!(state.plastic_volume_ratio() < 1.0, "snow must compact");
    }

    #[test]
    fn hardening_increases_force_magnitude() {
        // Compacting past the box with a large hardening coefficient must
        // produce stiffer (larger) forces than the same deformation with no
        // hardening, because the compacted moduli are scaled up.
        let el = element();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.5, 0.5, 0.9)));

        let hard_model = SnowModel::new(0.2, 0.1, 15.0).unwrap();
        let mut hard_state = SnowState::rest();
        let hard = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &hard_model,
            &mut hard_state,
        );

        let soft_model = SnowModel::new(0.2, 0.1, 0.0).unwrap();
        let mut soft_state = SnowState::rest();
        let soft = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soft_model,
            &mut soft_state,
        );

        assert!(hard.step.clamped && soft.step.clamped);
        assert!(hard.hardened_lame.mu > soft.hardened_lame.mu);
        let hard_mag: f32 = hard.forces.iter().map(|f| f.length()).sum();
        let soft_mag: f32 = soft.forces.iter().map(|f| f.length()).sum();
        assert!(
            hard_mag > soft_mag,
            "hardened snow should push back harder: {hard_mag} !> {soft_mag}"
        );
    }

    #[test]
    fn hardened_lame_field_matches_direct_evaluation() {
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 12.0).unwrap();
        let mut state = SnowState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.5, 0.6, 0.9)));
        let out = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        let direct = hardened_lame(&model, &state, &lame());
        assert!((out.hardened_lame.mu - direct.mu).abs() < 1e-3 * direct.mu.max(1.0));
        assert!((out.hardened_lame.lambda - direct.lambda).abs() < 1e-3 * direct.lambda.max(1.0));
    }

    #[test]
    fn potential_energy_is_finite_and_matches_density() {
        let el = element();
        let model = SnowModel::new(0.2, 0.1, 10.0).unwrap();

        // A compacted element stores finite elastic energy and the helper
        // reproduces V · Ψ(Fe) with the hardened moduli exactly.
        let mut state = SnowState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.5, 0.6, 0.9)));
        let out = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        let energy = snow_elastic_potential_energy(
            &el,
            HyperelasticModel::StableNeoHookean,
            &out.hardened_lame,
            out.step.elastic_gradient,
        );
        assert!(energy.is_finite());
        let expected = el.rest_volume.abs()
            * HyperelasticModel::StableNeoHookean
                .strain_energy_density(out.step.elastic_gradient, &out.hardened_lame);
        assert!((energy - expected).abs() <= 1e-3 * expected.abs().max(1.0));
    }

    #[test]
    fn is_deterministic() {
        let el = element();
        let model = SnowModel::new(0.15, 0.08, 5.0).unwrap();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.45, 0.6, 0.78)));
        let mut a = SnowState::rest();
        let mut b = SnowState::rest();
        let out_a = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut a,
        );
        let out_b = element_snow_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut b,
        );
        assert_eq!(out_a, out_b);
        assert_eq!(a, b);
    }
}
