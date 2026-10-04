//! Elastoplastic nodal internal forces for a cohesive Drucker–Prager
//! tetrahedral `FEM` element (with tension cutoff).
//!
//! This module is the cohesive-soil analogue of
//! [`tet_fem_sand_force`](super::tet_fem_sand_force): it bridges the cohesive
//! Drucker–Prager return mapping in
//! [`tet_fem_drucker_prager_plasticity`](super::tet_fem_drucker_prager_plasticity)
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
//! [`HyperelasticModel`](crate::collider::HyperelasticModel). Unlike snow, the
//! cohesive cone does **not** stiffen its moduli as it flows: the only history
//! variable is the accumulated plastic strain, which the return mapping feeds
//! back into the (hardening) friction slope. The Lamé parameters passed to the
//! stress therefore stay constant across the step.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map_drucker_prager`]): the log-strain predictor is projected either
//! radially onto the friction cone or back onto the tension-cutoff plane, and
//! the consumed strain is folded into `Fₚ`. The returned
//! [`DruckerPragerForce`] exposes the raw [`DruckerPragerStep`] so callers can
//! inspect which branch was taken.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`DruckerPragerState`] and performs no time
//! integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_drucker_prager_plasticity::{
    return_map_drucker_prager, DruckerPragerModel, DruckerPragerState, DruckerPragerStep,
};
use glam::{Mat3, Vec3};

/// Result of a single cohesive Drucker–Prager elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruckerPragerForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The cohesive Drucker–Prager return-mapping outcome that produced these
    /// forces.
    pub step: DruckerPragerStep,
}

/// Nodal cohesive Drucker–Prager elastoplastic forces `fᵢ = -V · P(Fₑ) · gᵢ`
/// for one tetrahedral element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map_drucker_prager`] against the carried `state` (mutated in place
/// when the element yields), and the resulting elastic gradient `Fₑ` is fed to
/// the hyperelastic stress with the constant `lame` moduli. When the material
/// stays inside both the friction cone and the tension cap the forces match
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_drucker_prager_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    soil_model: &DruckerPragerModel,
    state: &mut DruckerPragerState,
) -> DruckerPragerForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map_drucker_prager(f_total, lame, soil_model, state);
    let piola = model.first_piola(step.elastic_gradient, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    DruckerPragerForce { forces, step }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ)` for a cohesive
/// Drucker–Prager element whose elastic gradient `Fₑ` was produced by a prior
/// [`element_drucker_prager_force`] call.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`DruckerPragerStep::elastic_gradient`] from [`element_drucker_prager_force`]
/// to keep a single state advance per step.
#[must_use]
pub fn drucker_prager_elastic_potential_energy(
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
    use crate::collider::tet_fem_drucker_prager_plasticity::DruckerPragerYield;
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

    // Cohesive clay-like model: 30° friction, modest cohesion, small tension
    // cutoff, no hardening.
    fn soil() -> DruckerPragerModel {
        DruckerPragerModel::from_friction_angle(30.0, 0.05, 0.0, 0.02).unwrap()
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
        let mut state = DruckerPragerState::rest();
        let out = element_drucker_prager_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, DruckerPragerYield::Elastic);
        for f in out.forces {
            assert!(f.length() < 1e-1, "rest force should vanish, got {f}");
        }
        assert_eq!(state, DruckerPragerState::rest());
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = element();
        let mut state = DruckerPragerState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.3, 0.7, 0.95)));
        let out = element_drucker_prager_force(
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
        // A tiny compressive strain is well inside both the friction cone and
        // the tension cap, so the response is elastic and matches the plain
        // hyperelastic force.
        let el = element();
        let mut state = DruckerPragerState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(0.999, 0.999, 0.999)));
        let out = element_drucker_prager_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, DruckerPragerYield::Elastic);
        assert_eq!(state, DruckerPragerState::rest());
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
    fn shearing_yields_on_the_cone() {
        // A strongly deviatoric (volume-preserving) strain pushes the predictor
        // outside the cone and must be projected onto its surface.
        let el = element();
        let mut state = DruckerPragerState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)));
        let out = element_drucker_prager_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, DruckerPragerYield::ConeSurface);
        assert!(out.step.plastic_increment > 0.0);
        assert!(
            state.accumulated_strain() > 0.0,
            "cohesive soil must accumulate strain"
        );
    }

    #[test]
    fn expansion_hits_tension_cutoff() {
        // A purely expanding predictor blows through the tension cap and must be
        // returned onto the cutoff plane.
        let el = element();
        let mut state = DruckerPragerState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.4, 1.4, 1.4)));
        let out = element_drucker_prager_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        assert_eq!(out.step.mode, DruckerPragerYield::TensionCutoff);
        assert!(out.step.plastic_increment > 0.0);
    }

    #[test]
    fn potential_energy_is_finite_and_matches_density() {
        let el = element();
        let mut state = DruckerPragerState::rest();
        let positions = deform(Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)));
        let out = element_drucker_prager_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut state,
        );
        let energy = drucker_prager_elastic_potential_energy(
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
        let mut a = DruckerPragerState::rest();
        let mut b = DruckerPragerState::rest();
        let out_a = element_drucker_prager_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &soil(),
            &mut a,
        );
        let out_b = element_drucker_prager_force(
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
