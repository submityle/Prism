//! Elastoplastic nodal internal forces for a tetrahedral `FEM` element.
//!
//! This module is the bridge between the finite-strain return mapping in
//! [`tet_fem_plasticity`](super::tet_fem_plasticity) and the hyperelastic
//! stress response in [`tet_fem_internal_force`](super::tet_fem_internal_force).
//!
//! A purely elastic element stores all of its deformation as recoverable
//! strain, so stress is a function of the *total* deformation gradient `F`.
//! A plastic element instead carries a persistent plastic gradient `Fₚ` and
//! only the *elastic* part `Fₑ = F · Fₚ⁻¹` generates stress. The nodal force is
//! therefore
//!
//! ```text
//! fᵢ = -V · P(Fₑ) · gᵢ
//! ```
//!
//! where `V` is the absolute rest volume, `P` is the first Piola–Kirchhoff
//! stress of a [`HyperelasticModel`](crate::collider::HyperelasticModel), and
//! `gᵢ` is the reference-space shape-function gradient stored on the element.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map`]): the elastic predictor is projected back onto the yield
//! surface and any consumed strain is folded into `Fₚ`. The returned
//! [`PlasticStep`] exposes whether the element yielded and how much plastic
//! strain it consumed, so callers can drive hardening, visualisation, or
//! fracture heuristics without re-deriving the split.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`PlasticState`] and performs no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_plasticity::{return_map, PlasticModel, PlasticState, PlasticStep};
use glam::{Mat3, Vec3};

/// Result of a single elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElastoplasticForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The return-mapping outcome that produced these forces.
    pub step: PlasticStep,
}

/// Nodal elastoplastic forces `fᵢ = -V · P(Fₑ) · gᵢ` for one tetrahedral
/// element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map`] against the carried `state` (which is mutated in place when
/// the element yields), and the resulting elastic gradient `Fₑ` is fed to the
/// hyperelastic stress. When the material stays elastic the forces are
/// identical (to within rounding) to
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_elastoplastic_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    plastic_model: &PlasticModel,
    state: &mut PlasticState,
) -> ElastoplasticForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map(f_total, plastic_model, state);
    let piola = model.first_piola(step.elastic_gradient, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    ElastoplasticForce { forces, step }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ)` for an element whose
/// elastic gradient `Fₑ` was produced by a prior [`return_map`] / force
/// evaluation.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`PlasticStep::elastic_gradient`] from
/// [`element_elastoplastic_force`] to keep a single state advance per step.
#[must_use]
pub fn elastic_potential_energy(
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

    fn max_norm(forces: [Vec3; 4]) -> f32 {
        forces.iter().map(|f| f.length()).fold(0.0_f32, f32::max)
    }

    #[test]
    fn elastic_regime_matches_pure_elastic() {
        // A tiny deformation stays well inside the yield surface, so the
        // elastoplastic force must equal the purely elastic force and leave
        // the plastic state untouched.
        let el = element();
        let lame = lame();
        let plastic = PlasticModel::new(0.2, 0.0).unwrap();
        let mut state = PlasticState::rest();
        let pos = affine(Mat3::from_diagonal(Vec3::new(1.01, 0.995, 1.003)));

        let plastic_out = element_elastoplastic_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        let elastic = element_internal_force(&el, pos, HyperelasticModel::StableNeoHookean, &lame);

        assert!(!plastic_out.step.yielded, "small strain must stay elastic");
        assert_eq!(state, PlasticState::rest(), "elastic step leaves Fp = I");
        for i in 0..4 {
            assert!(
                (plastic_out.forces[i] - elastic[i]).length() < 1e-1,
                "node {i}: elastoplastic {:?} vs elastic {:?}",
                plastic_out.forces[i],
                elastic[i]
            );
        }
    }

    #[test]
    fn forces_sum_to_zero_even_when_yielding() {
        let el = element();
        let lame = lame();
        let plastic = PlasticModel::new(0.05, 0.0).unwrap();
        let mut state = PlasticState::rest();
        let out = element_elastoplastic_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.3, 1.0, 0.7))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        assert!(out.step.yielded, "large deviatoric strain should yield");
        assert!(
            total(out.forces).length() < 1e-1,
            "net internal force must vanish, got {:?}",
            total(out.forces)
        );
    }

    #[test]
    fn yielding_relaxes_the_restoring_force() {
        // A deviatoric-dominant stretch: plastic flow removes deviatoric strain
        // so the elastoplastic restoring force and stored energy are strictly
        // smaller than the purely elastic response on the same configuration.
        let el = element();
        let lame = lame();
        let f = Mat3::from_diagonal(Vec3::new(1.3, 1.0, 0.72));
        let pos = affine(f);

        let plastic = PlasticModel::new(0.05, 0.0).unwrap();
        let mut state = PlasticState::rest();
        let out = element_elastoplastic_force(
            &el,
            pos,
            HyperelasticModel::StVenantKirchhoff,
            &lame,
            &plastic,
            &mut state,
        );
        let elastic = element_internal_force(&el, pos, HyperelasticModel::StVenantKirchhoff, &lame);

        assert!(out.step.yielded);
        assert!(
            max_norm(out.forces) < max_norm(elastic),
            "plastic flow should relax the restoring force: {} !< {}",
            max_norm(out.forces),
            max_norm(elastic)
        );

        let u_plastic = elastic_potential_energy(
            &el,
            HyperelasticModel::StVenantKirchhoff,
            &lame,
            out.step.elastic_gradient,
        );
        let u_elastic = element_energy(&el, pos, HyperelasticModel::StVenantKirchhoff, &lame);
        assert!(
            u_plastic < u_elastic,
            "stored elastic energy should drop after yielding: {u_plastic} !< {u_elastic}"
        );
    }

    #[test]
    fn permanent_set_leaves_residual_force_at_rest() {
        // Yield the element, then return the nodes to their rest positions.
        // Because Fp != I, the elastic gradient Fe = I·Fp⁻¹ != I, so the
        // element still carries stress — a permanent set.
        let el = element();
        let lame = lame();
        let plastic = PlasticModel::new(0.04, 0.0).unwrap();
        let mut state = PlasticState::rest();

        let _ = element_elastoplastic_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.4, 1.0, 0.7))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        assert_ne!(state.plastic_gradient(), Mat3::IDENTITY, "material yielded");

        let residual = element_elastoplastic_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        assert!(
            max_norm(residual.forces) > 1e-3,
            "permanent set should leave residual stress at rest geometry"
        );
    }

    #[test]
    fn repeated_loading_accumulates_plastic_strain() {
        let el = element();
        let lame = lame();
        let plastic = PlasticModel::new(0.05, 0.0).unwrap();
        let mut state = PlasticState::rest();

        let first = element_elastoplastic_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.25, 1.0, 0.8))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        let after_first = state.accumulated_strain();
        let _ = element_elastoplastic_force(
            &el,
            affine(Mat3::from_diagonal(Vec3::new(1.5, 0.95, 0.75))),
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut state,
        );
        assert!(first.step.plastic_increment > 0.0);
        assert!(
            state.accumulated_strain() >= after_first,
            "accumulated plastic strain is non-decreasing"
        );
    }

    #[test]
    fn is_deterministic() {
        let el = element();
        let lame = lame();
        let plastic = PlasticModel::new(0.06, 1.0).unwrap();
        let pos = affine(Mat3::from_diagonal(Vec3::new(1.35, 0.95, 0.78)));

        let mut sa = PlasticState::rest();
        let mut sb = PlasticState::rest();
        let a = element_elastoplastic_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut sa,
        );
        let b = element_elastoplastic_force(
            &el,
            pos,
            HyperelasticModel::StableNeoHookean,
            &lame,
            &plastic,
            &mut sb,
        );
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }
}
