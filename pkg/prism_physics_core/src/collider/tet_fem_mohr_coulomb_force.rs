//! Elastoplastic nodal internal forces for a Mohr–Coulomb tetrahedral `FEM`
//! element.
//!
//! This module is the faceted-pyramid analogue of
//! [`tet_fem_drucker_prager_force`](super::tet_fem_drucker_prager_force): it
//! bridges the multisurface Mohr–Coulomb return mapping in
//! [`tet_fem_mohr_coulomb_plasticity`](super::tet_fem_mohr_coulomb_plasticity)
//! with the hyperelastic stress response in
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
//! [`HyperelasticModel`](crate::collider::HyperelasticModel). The elastic Lamé
//! moduli stay constant across the step; the only history variable is the
//! accumulated plastic strain, which feeds the (hardening) cohesion back into
//! the yield surface.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map_mohr_coulomb`]): the principal-stress predictor is projected
//! onto the first admissible feature of the pyramid (plane, edge, or apex) and
//! the consumed strain is folded into `Fₚ`. The returned [`MohrCoulombForce`]
//! exposes the raw [`MohrCoulombStep`] so callers can inspect which feature was
//! taken.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`MohrCoulombState`] and performs no time
//! integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_mohr_coulomb_plasticity::{
    return_map_mohr_coulomb, MohrCoulombModel, MohrCoulombState, MohrCoulombStep,
};
use glam::{Mat3, Vec3};

/// Result of a single Mohr–Coulomb elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The Mohr–Coulomb return-mapping outcome that produced these forces.
    pub step: MohrCoulombStep,
}

/// Nodal Mohr–Coulomb elastoplastic forces `fᵢ = -V · P(Fₑ) · gᵢ` for one
/// tetrahedral element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map_mohr_coulomb`] against the carried `state` (mutated in place
/// when the element yields), and the resulting elastic gradient `Fₑ` is fed to
/// the hyperelastic stress with the constant `lame` moduli. When the material
/// stays inside the pyramid the forces match
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_mohr_coulomb_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    soil_model: &MohrCoulombModel,
    state: &mut MohrCoulombState,
) -> MohrCoulombForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map_mohr_coulomb(f_total, lame, soil_model, state);
    let piola = model.first_piola(step.elastic_gradient, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    MohrCoulombForce { forces, step }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ)` for a Mohr–Coulomb
/// element whose elastic gradient `Fₑ` was produced by a prior
/// [`element_mohr_coulomb_force`] call.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`MohrCoulombStep::elastic_gradient`] from [`element_mohr_coulomb_force`] to
/// keep a single state advance per step.
#[must_use]
pub fn mohr_coulomb_elastic_potential_energy(
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
    use crate::collider::tet_fem_mohr_coulomb_plasticity::MohrCoulombYield;
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

    // Cohesive clay-like model: 30° friction, associated flow, modest cohesion.
    fn soil() -> MohrCoulombModel {
        MohrCoulombModel::from_angles(30.0, 30.0, 2000.0, 0.0).unwrap()
    }

    fn element() -> TetFemElement {
        TetFemElement::from_rest(REST[0], REST[1], REST[2], REST[3], 1e-12).unwrap()
    }

    // Applies an affine map `A` to the rest nodes so the element's deformation
    // gradient is exactly `A`.
    fn deform(a: Mat3) -> [Vec3; 4] {
        [a * REST[0], a * REST[1], a * REST[2], a * REST[3]]
    }

    fn sum(forces: &[Vec3; 4]) -> Vec3 {
        forces[0] + forces[1] + forces[2] + forces[3]
    }

    #[test]
    fn rest_force_vanishes() {
        let el = element();
        let mut state = MohrCoulombState::rest();
        let out = element_mohr_coulomb_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, MohrCoulombYield::Elastic);
        for f in out.forces {
            assert!(f.length() < 1e-1, "rest force should vanish, got {f}");
        }
        assert_eq!(state, MohrCoulombState::rest());
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = element();
        let mut state = MohrCoulombState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.3, 0.7, 0.95)));
        let out = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert!(sum(&out.forces).length() < 1e-2, "net force must vanish");
    }

    #[test]
    fn small_strain_stays_elastic() {
        // A tiny compressive strain is well inside the pyramid, so the response
        // is elastic and matches the plain hyperelastic force.
        let el = element();
        let mut state = MohrCoulombState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.999, 0.999, 0.999)));
        let out = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, MohrCoulombYield::Elastic);
        assert_eq!(state, MohrCoulombState::rest());
        let reference =
            element_internal_force(&el, positions, HyperelasticModel::StableNeoHookean, &lame());
        for i in 0..4 {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "elastic force should match hyperelastic force at node {i}"
            );
        }
    }

    #[test]
    fn shearing_yields_on_a_plane() {
        // A strongly deviatoric (volume-preserving) strain pushes the predictor
        // outside the pyramid and must be projected onto a yield plane.
        let el = element();
        let mut state = MohrCoulombState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)));
        let out = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, MohrCoulombYield::MainPlane);
        assert!(out.step.plastic_increment > 0.0);
        assert!(
            state.accumulated_strain() > 0.0,
            "cohesive soil must accumulate strain"
        );
    }

    #[test]
    fn expansion_hits_apex() {
        // A purely expanding predictor blows through the tensile apex of the
        // pyramid and must be returned onto it.
        let el = element();
        let mut state = MohrCoulombState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.4, 1.4, 1.4)));
        let out = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, MohrCoulombYield::Apex);
        assert!(out.step.plastic_increment > 0.0);
    }

    #[test]
    fn potential_energy_is_finite_and_matches_density() {
        let el = element();
        let mut state = MohrCoulombState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)));
        let out = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        let energy = mohr_coulomb_elastic_potential_energy(
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
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.55, 0.65, 0.98)));
        let mut a = MohrCoulombState::rest();
        let mut b = MohrCoulombState::rest();
        let out_a = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut a,
        );
        let out_b = element_mohr_coulomb_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut b,
        );
        assert_eq!(out_a, out_b);
        assert_eq!(a, b);
    }
}
