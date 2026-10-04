//! Elastoplastic nodal internal forces for a modified Cam-Clay tetrahedral
//! `FEM` element.
//!
//! This module is the Cam-Clay analogue of
//! [`tet_fem_sand_force`](super::tet_fem_sand_force): it bridges the
//! cap-plasticity return mapping in
//! [`tet_fem_camclay_plasticity`](super::tet_fem_camclay_plasticity) with the
//! hyperelastic stress response in
//! [`tet_fem_internal_force`](super::tet_fem_internal_force).
//!
//! Only the elastic part `Fₑ = F · Fₚ⁻¹` of the deformation generates stress,
//! so the nodal force is
//!
//! ```text
//! fᵢ = -V · P(Fₑ) · gᵢ
//! ```
//!
//! where `V` is the absolute rest volume, `gᵢ` the reference-space
//! shape-function gradient, and `P` the first Piola–Kirchhoff stress of a
//! [`HyperelasticModel`](crate::collider::HyperelasticModel). Unlike snow, the
//! elastic moduli do not change as the material flows: the only history the
//! return mapping writes back is the plastic gradient `Fₚ` and the scalar
//! pre-consolidation pressure `p_c` (the compression cap), so the Lamé
//! parameters handed to the stress stay constant across the step.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map_camclay`]): the log-strain predictor is returned onto the
//! Cam-Clay ellipse, the cap `p_c` is hardened/softened by the volumetric
//! plastic strain, and the consumed strain is folded into `Fₚ`. The returned
//! [`CamClayForce`] exposes the raw [`CamClayStep`] so callers can read whether
//! the element yielded and inspect the updated cap.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`CamClayState`] and performs no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_camclay_plasticity::{
    return_map_camclay, CamClayModel, CamClayState, CamClayStep,
};
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use glam::{Mat3, Vec3};

/// Result of a single Cam-Clay elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The Cam-Clay return-mapping outcome that produced these forces.
    pub step: CamClayStep,
}

/// Nodal Cam-Clay elastoplastic forces `fᵢ = -V · P(Fₑ) · gᵢ` for one
/// tetrahedral element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map_camclay`] against the carried `state` (mutated in place when the
/// element yields, updating both `Fₚ` and the cap `p_c`), and the resulting
/// elastic gradient `Fₑ` is fed to the hyperelastic stress with the constant
/// `lame` moduli. When the predictor stays inside the ellipse the forces match
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_camclay_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    camclay_model: &CamClayModel,
    state: &mut CamClayState,
) -> CamClayForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map_camclay(f_total, lame, camclay_model, state);
    let piola = model.first_piola(step.elastic_gradient, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    CamClayForce { forces, step }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ)` for a Cam-Clay element
/// whose elastic gradient `Fₑ` was produced by a prior
/// [`element_camclay_force`] call.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`CamClayStep::elastic_gradient`] from [`element_camclay_force`] to keep a
/// single state advance per step.
#[must_use]
pub fn camclay_elastic_potential_energy(
    element: &TetFemElement,
    model: HyperelasticModel,
    lame: &LameParameters,
    elastic_gradient: Mat3,
) -> f32 {
    let volume = element.rest_volume.abs();
    volume * model.strain_energy_density(elastic_gradient, lame)
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

    fn camclay() -> CamClayModel {
        CamClayModel::new(1.2, 0.05, 5.0e4).unwrap()
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
        let model = camclay();
        let mut state = model.rest_state();
        let out = element_camclay_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(!out.step.yielded);
        for f in out.forces {
            assert!(f.length() < 1e-1, "rest force should vanish, got {f}");
        }
        assert_eq!(state, model.rest_state());
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = element();
        let model = camclay();
        let mut state = model.rest_state();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.88, 0.9, 0.86)));
        let out = element_camclay_force(
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
    fn small_strain_stays_inside_ellipse() {
        // A tiny compressive strain is well inside the Cam-Clay ellipse, so the
        // response is elastic and matches the plain hyperelastic force.
        let el = element();
        let model = camclay();
        let mut state = model.rest_state();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.999, 0.999, 0.999)));
        let out = element_camclay_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(!out.step.yielded);
        assert_eq!(state, model.rest_state());
        let reference =
            element_internal_force(&el, positions, HyperelasticModel::StableNeoHookean, &lame());
        for i in 0..4 {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "elastic Cam-Clay force should match hyperelastic force at node {i}"
            );
        }
    }

    #[test]
    fn compression_past_cap_yields_and_hardens() {
        // Strong hydrostatic compression crosses the cap: the element yields and
        // the pre-consolidation pressure grows (volumetric hardening).
        let el = element();
        let model = camclay();
        let mut state = model.rest_state();
        let pc_before = state.pre_consolidation();
        let positions = deform(Mat3::from_diagonal(Vec3::splat(0.9)));
        let out = element_camclay_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        assert!(out.step.yielded);
        assert!(out.step.plastic_increment > 0.0);
        assert!(
            state.pre_consolidation() > pc_before,
            "cap must harden under compaction"
        );
        assert_eq!(state.pre_consolidation(), out.step.pre_consolidation);
    }

    #[test]
    fn potential_energy_is_finite_and_matches_density() {
        let el = element();
        let model = camclay();
        let mut state = model.rest_state();
        let positions = deform(Mat3::from_diagonal(Vec3::splat(0.9)));
        let out = element_camclay_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut state,
        );
        let energy = camclay_elastic_potential_energy(
            &el,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            out.step.elastic_gradient,
        );
        assert!(energy.is_finite());
        let expected = el.rest_volume.abs()
            * HyperelasticModel::StableNeoHookean
                .strain_energy_density(out.step.elastic_gradient, &lame());
        assert!((energy - expected).abs() <= 1e-3 * expected.abs().max(1.0));
    }

    #[test]
    fn is_deterministic() {
        let el = element();
        let model = camclay();
        let positions = deform(Mat3::from_diagonal(Vec3::splat(0.9)));
        let mut a = model.rest_state();
        let mut b = model.rest_state();
        let out_a = element_camclay_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &model,
            &mut a,
        );
        let out_b = element_camclay_force(
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
