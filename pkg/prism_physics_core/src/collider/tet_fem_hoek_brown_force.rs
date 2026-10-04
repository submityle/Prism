//! Elastoplastic nodal internal forces for a Hoek–Brown tetrahedral `FEM`
//! element.
//!
//! This module is the curved-rock-mass analogue of
//! [`tet_fem_mohr_coulomb_force`](super::tet_fem_mohr_coulomb_force): it bridges
//! the nonlinear multisurface Hoek–Brown return mapping in
//! [`tet_fem_hoek_brown_plasticity`](super::tet_fem_hoek_brown_plasticity) with
//! the hyperelastic stress response in
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
//! accumulated plastic strain, which feeds the (hardening) intact strength back
//! into the curved yield surface.
//!
//! Evaluating the force advances the plastic state exactly once (via
//! [`return_map_hoek_brown`]): the principal-stress predictor is projected onto
//! the first admissible feature of the pyramid (main surface, compression or
//! extension edge, or apex) and the consumed strain is folded into `Fₚ`. The
//! returned [`HoekBrownForce`] exposes the raw [`HoekBrownStep`] so callers can
//! inspect which feature was taken.
//!
//! Everything here is a pure, per-element kernel: it holds no solver state
//! beyond the caller-owned [`HoekBrownState`] and performs no time integration.

use crate::collider::tet_fem_basis::TetFemElement;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_hoek_brown_plasticity::{
    return_map_hoek_brown, HoekBrownModel, HoekBrownState, HoekBrownStep,
};
use glam::{Mat3, Vec3};

/// Result of a single Hoek–Brown elastoplastic element evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoekBrownForce {
    /// Nodal elastic forces ordered to match the supplied positions.
    pub forces: [Vec3; 4],
    /// The Hoek–Brown return-mapping outcome that produced these forces.
    pub step: HoekBrownStep,
}

/// Nodal Hoek–Brown elastoplastic forces `fᵢ = -V · P(Fₑ) · gᵢ` for one
/// tetrahedral element at the supplied deformed node positions.
///
/// The total deformation gradient is derived from `positions`, run through
/// [`return_map_hoek_brown`] against the carried `state` (mutated in place when
/// the element yields), and the resulting elastic gradient `Fₑ` is fed to the
/// hyperelastic stress with the constant `lame` moduli. When the material stays
/// inside the surface the forces match
/// [`element_internal_force`](super::tet_fem_internal_force::element_internal_force)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
///
/// Because the shape gradients sum to zero, the returned forces sum to zero to
/// within rounding, so the element exerts no net force on its centre of mass.
#[must_use]
pub fn element_hoek_brown_force(
    element: &TetFemElement,
    positions: [Vec3; 4],
    model: HyperelasticModel,
    lame: &LameParameters,
    rock_model: &HoekBrownModel,
    state: &mut HoekBrownState,
) -> HoekBrownForce {
    let [p0, p1, p2, p3] = positions;
    let f_total = element.deformation_gradient(p0, p1, p2, p3);
    let step = return_map_hoek_brown(f_total, lame, rock_model, state);
    let piola = model.first_piola(step.elastic_gradient, lame);
    let volume = element.rest_volume.abs();
    let g = element.shape_gradients;
    let forces = [
        -volume * (piola * g[0]),
        -volume * (piola * g[1]),
        -volume * (piola * g[2]),
        -volume * (piola * g[3]),
    ];
    HoekBrownForce { forces, step }
}

/// Recoverable elastic potential energy `U = V · Ψ(Fₑ)` for a Hoek–Brown
/// element whose elastic gradient `Fₑ` was produced by a prior
/// [`element_hoek_brown_force`] call.
///
/// This is a pure read of the energy stored in the *elastic* part of the
/// deformation; it does not advance plasticity. Pass
/// [`HoekBrownStep::elastic_gradient`] from [`element_hoek_brown_force`] to keep
/// a single state advance per step.
#[must_use]
pub fn hoek_brown_elastic_potential_energy(
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
    use crate::collider::tet_fem_hoek_brown_plasticity::HoekBrownYield;
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

    // A soft rock mass: low intact strength so yielding is reachable at the
    // strain scale used elsewhere; original criterion (s = 1, a = 0.5) with
    // associated flow and no hardening.
    fn rock() -> HoekBrownModel {
        HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 0.0).unwrap()
    }

    fn element() -> TetFemElement {
        TetFemElement::from_rest(REST[0], REST[1], REST[2], REST[3], 1e-12).unwrap()
    }

    // Converts a principal log-strain into its stretch, in f64 to avoid the
    // disallowed f32 transcendental methods.
    fn stretch(log: f32) -> f32 {
        f64::from(log).exp() as f32
    }

    // Diagonal deformation gradient whose principal log-strains are exactly the
    // supplied (descending) vector.
    fn from_log_strain(e: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(stretch(e.x), stretch(e.y), stretch(e.z)))
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
        let mut state = HoekBrownState::rest();
        let out = element_hoek_brown_force(
            &el,
            REST,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        assert_eq!(out.step.mode, HoekBrownYield::Elastic);
        for f in out.forces {
            assert!(f.length() < 1e-1, "rest force should vanish, got {f}");
        }
        assert_eq!(state, HoekBrownState::rest());
    }

    #[test]
    fn forces_sum_to_zero() {
        let el = element();
        let mut state = HoekBrownState::rest();
        let positions = deform(from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let out = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        assert!(sum(&out.forces).length() < 1e-2, "net force must vanish");
    }

    #[test]
    fn small_strain_stays_elastic() {
        // A tiny shear is well inside the curved surface, so the response is
        // elastic and matches the plain hyperelastic force.
        let el = element();
        let mut state = HoekBrownState::rest();
        let positions = deform(from_log_strain(Vec3::new(0.0005, 0.0, -0.0005)));
        let out = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        assert_eq!(out.step.mode, HoekBrownYield::Elastic);
        assert_eq!(state, HoekBrownState::rest());
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
    fn confined_shear_yields_on_main_surface() {
        // Confined (compressive σ₁) triaxial load overshoots the curved surface
        // and must be returned onto the main Hoek–Brown surface.
        let el = element();
        let mut state = HoekBrownState::rest();
        let positions = deform(from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let out = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        assert_eq!(out.step.mode, HoekBrownYield::MainSurface);
        assert!(out.step.plastic_increment > 0.0);
        assert!(
            state.accumulated_strain() > 0.0,
            "yielding rock must accumulate strain"
        );
    }

    #[test]
    fn expansion_hits_apex() {
        // A purely expanding predictor drives hydrostatic tension far past the
        // apex of the surface and must be returned onto it.
        let el = element();
        let mut state = HoekBrownState::rest();
        let positions = deform(from_log_strain(Vec3::splat(0.05)));
        let out = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        assert_eq!(out.step.mode, HoekBrownYield::Apex);
        assert!(out.step.plastic_increment > 0.0);
    }

    #[test]
    fn potential_energy_is_finite_and_matches_density() {
        let el = element();
        let mut state = HoekBrownState::rest();
        let positions = deform(from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let out = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut state,
        );
        let energy = hoek_brown_elastic_potential_energy(
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
        let positions = deform(from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let mut a = HoekBrownState::rest();
        let mut b = HoekBrownState::rest();
        let out_a = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut a,
        );
        let out_b = element_hoek_brown_force(
            &el,
            positions,
            HyperelasticModel::StableNeoHookean,
            &lame(),
            &rock(),
            &mut b,
        );
        assert_eq!(out_a, out_b);
        assert_eq!(a, b);
    }
}
