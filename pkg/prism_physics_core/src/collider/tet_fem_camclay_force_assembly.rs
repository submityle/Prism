//! Global assembly of modified Cam-Clay (MCC) elastoplastic forces over a
//! tetrahedral `FEM` mesh.
//!
//! This is the cohesive-soil counterpart of
//! [`tet_fem_sand_force_assembly`](super::tet_fem_sand_force_assembly): instead
//! of the Drucker-Prager friction cone it carries one persistent
//! [`CamClayState`](crate::collider::tet_fem_camclay_plasticity::CamClayState)
//! per element and advances each exactly once per pass through the modified
//! Cam-Clay cap return mapping in
//! [`tet_fem_camclay_force`](super::tet_fem_camclay_force). Unlike the sand
//! model, each state additionally stores a per-element pre-consolidation
//! pressure `p_c` that *hardens* when the element is compacted past its cap, so
//! the carried state is genuinely history dependent.
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, the current
//! deformed vertex positions, a shared [`CamClayModel`], and a mutable slice of
//! per-element Cam-Clay states, [`assemble_camclay_forces`] evaluates every
//! element's nodal force `fᵢ = -V · P(Fₑ) · gᵢ`, folds any consumed strain into
//! that element's `Fₚ`, updates its cap `p_c`, and scatters the four nodal
//! forces into a global per-vertex vector. Each element's forces sum to zero,
//! so the assembled total force is zero to within rounding.
//!
//! The returned [`CamClayAssembly`] keeps the per-element
//! [`CamClayStep`](crate::collider::tet_fem_camclay_plasticity::CamClayStep)
//! alongside the global force vector so a caller can report how many elements
//! yielded, how much plastic strain was consumed, and recover the stored
//! elastic energy via [`total_camclay_elastic_potential_energy`] without
//! re-advancing plasticity. This module holds no solver state of its own and
//! performs no time integration, so it feeds directly into an explicit or
//! implicit integrator supplied elsewhere.
//!
//! No Unreal Engine source or derived code.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_camclay_force::{
    camclay_elastic_potential_energy, element_camclay_force,
};
use crate::collider::tet_fem_camclay_plasticity::{CamClayModel, CamClayState, CamClayStep};
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use glam::Vec3;

/// Aggregate result of one global Cam-Clay assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct CamClayAssembly {
    /// Per-vertex internal Cam-Clay force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-element return-mapping outcome, in basis / connectivity order.
    pub steps: Vec<CamClayStep>,
}

impl CamClayAssembly {
    /// Number of elements whose predictor left the ellipse interior this pass
    /// and had to be returned onto the cap.
    #[must_use]
    pub fn yielded_element_count(&self) -> usize {
        self.steps.iter().filter(|s| s.yielded).count()
    }

    /// Total plastic strain consumed across all elements this pass.
    #[must_use]
    pub fn total_plastic_increment(&self) -> f32 {
        self.steps.iter().map(|s| s.plastic_increment).sum()
    }
}

/// Builds `count` fresh rest Cam-Clay states, one per element, each seeded with
/// the model's initial pre-consolidation pressure `p_c0`.
#[must_use]
pub fn rest_camclay_states(model: &CamClayModel, count: usize) -> Vec<CamClayState> {
    vec![model.rest_state(); count]
}

/// Gathers the four deformed node positions of tet `t` from `positions`.
fn gather(positions: &[Vec3], t: [u32; 4]) -> Option<[Vec3; 4]> {
    let n = positions.len();
    if t.iter().any(|&vi| vi as usize >= n) {
        return None;
    }
    Some([
        positions[t[0] as usize],
        positions[t[1] as usize],
        positions[t[2] as usize],
        positions[t[3] as usize],
    ])
}

/// Validates that the basis, connectivity, position, and state arrays are
/// mutually consistent: one element and one Cam-Clay state per tet, and every
/// tet index addressable in `positions`.
fn is_consistent(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    states: &[CamClayState],
) -> bool {
    if basis.element_count() != tets.len() || states.len() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Assembles the per-vertex Cam-Clay force vector of the mesh at the supplied
/// deformed positions, advancing each element's Cam-Clay state once.
///
/// `states` must hold exactly one [`CamClayState`] per tet (see
/// [`rest_camclay_states`]); each is mutated in place when its element yields,
/// updating both `Fₚ` and the cap `p_c`. Returns `None` when the basis element
/// count, `tets`, and `states` lengths disagree or when any tet index is out of
/// range for `positions`; in that case no state is mutated.
///
/// When every element stays inside the ellipse the result matches
/// [`assemble_internal_forces`](super::tet_fem_force_assembly::assemble_internal_forces)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
#[must_use]
pub fn assemble_camclay_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
    camclay_model: &CamClayModel,
    states: &mut [CamClayState],
) -> Option<CamClayAssembly> {
    if !is_consistent(basis, tets, positions, states) {
        return None;
    }
    // Validate every index up front so a late failure cannot leave the
    // Cam-Clay states partially advanced.
    for &t in tets {
        gather(positions, t)?;
    }

    let mut forces = vec![Vec3::ZERO; positions.len()];
    let mut steps = Vec::with_capacity(tets.len());
    for ((element, &t), state) in basis
        .elements
        .iter()
        .zip(tets.iter())
        .zip(states.iter_mut())
    {
        let nodes = gather(positions, t)?;
        let out = element_camclay_force(element, nodes, model, lame, camclay_model, state);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += out.forces[local];
        }
        steps.push(out.step);
    }
    Some(CamClayAssembly { forces, steps })
}

/// Total recoverable elastic potential energy `U = Σ_e V_e · Ψ(Fₑ)` of the
/// mesh, using the per-element elastic gradients produced by a prior
/// [`assemble_camclay_forces`] pass with the same constant `lame` moduli.
///
/// This is a pure read of the stored elastic energy; it does not advance
/// plasticity. Returns `None` when `steps` does not hold exactly one entry per
/// element.
#[must_use]
pub fn total_camclay_elastic_potential_energy(
    basis: &TetFemBasis,
    model: HyperelasticModel,
    lame: &LameParameters,
    steps: &[CamClayStep],
) -> Option<f32> {
    if steps.len() != basis.element_count() {
        return None;
    }
    let mut energy = 0.0_f32;
    for (element, step) in basis.elements.iter().zip(steps.iter()) {
        energy += camclay_elastic_potential_energy(element, model, lame, step.elastic_gradient);
    }
    Some(energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_force_assembly::assemble_internal_forces;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;

    const MODEL: HyperelasticModel = HyperelasticModel::StableNeoHookean;

    fn lame() -> LameParameters {
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.3).unwrap())
    }

    fn camclay() -> CamClayModel {
        CamClayModel::new(1.2, 0.05, 5.0e4).unwrap()
    }

    // Five vertices forming two tetrahedra sharing a face.
    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::new(1e-12)).unwrap()
    }

    fn scaled(verts: &[Vec3], s: f32) -> Vec<Vec3> {
        verts.iter().map(|v| *v * s).collect()
    }

    #[test]
    fn rejects_inconsistent_state_length() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 1);
        let out =
            assemble_camclay_forces(&basis, &tets, &verts, MODEL, &lame(), &model, &mut states);
        assert!(out.is_none());
        assert_eq!(
            states,
            rest_camclay_states(&model, 1),
            "nothing may advance"
        );
    }

    #[test]
    fn rejects_out_of_range_index_without_mutating() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let truncated = verts[..4].to_vec();
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let out = assemble_camclay_forces(
            &basis,
            &tets,
            &truncated,
            MODEL,
            &lame(),
            &model,
            &mut states,
        );
        assert!(out.is_none());
        assert_eq!(
            states,
            rest_camclay_states(&model, 2),
            "nothing may advance"
        );
    }

    #[test]
    fn rest_pose_is_force_free_and_elastic() {
        // At F = I the mean and deviatoric stresses vanish, so the ellipse
        // value y <= 0 and no element may yield.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let out =
            assemble_camclay_forces(&basis, &tets, &verts, MODEL, &lame(), &model, &mut states)
                .unwrap();
        assert_eq!(out.yielded_element_count(), 0);
        for (i, f) in out.forces.iter().enumerate() {
            assert!(f.length() <= 1e-1, "vertex {i} rest force {f:?}");
        }
        assert_eq!(
            states,
            rest_camclay_states(&model, 2),
            "rest pose must not yield"
        );
    }

    #[test]
    fn elastic_pass_matches_hyperelastic_assembly() {
        // A tiny uniform compression (s = 0.999) keeps the mean pressure far
        // below the cap, so the Cam-Clay assembly must agree with the stateless
        // elastic assembly.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.999);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let out = assemble_camclay_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert_eq!(out.yielded_element_count(), 0);
        let reference = assemble_internal_forces(&basis, &tets, &deformed, MODEL, &lame()).unwrap();
        assert_eq!(out.forces.len(), reference.len());
        for i in 0..reference.len() {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "elastic Cam-Clay assembly must match hyperelastic at vertex {i}"
            );
        }
    }

    #[test]
    fn compression_yields_and_sums_to_zero() {
        // A strong uniform compression (s = 0.9) drives the mean pressure well
        // past the pre-consolidation cap, so the predictor must be returned
        // onto the ellipse.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.9);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let out = assemble_camclay_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert!(
            out.yielded_element_count() > 0,
            "compression past the cap should yield"
        );
        assert!(out.total_plastic_increment() > 0.0);
        let net: Vec3 = out.forces.iter().copied().sum();
        assert!(net.length() <= 1e-1, "net force {net:?}");
    }

    #[test]
    fn yielding_advances_states() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.9);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let _ = assemble_camclay_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert_ne!(
            states,
            rest_camclay_states(&model, 2),
            "yielded elements must accumulate strain and harden the cap"
        );
    }

    #[test]
    fn energy_helper_is_finite_and_matches_sum() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.9);
        let model = camclay();
        let mut states = rest_camclay_states(&model, 2);
        let out = assemble_camclay_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        let energy =
            total_camclay_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps).unwrap();
        assert!(energy.is_finite());
        let manual: f32 = basis
            .elements
            .iter()
            .zip(out.steps.iter())
            .map(|(e, s)| camclay_elastic_potential_energy(e, MODEL, &lame(), s.elastic_gradient))
            .sum();
        assert!((energy - manual).abs() <= 1e-3 * manual.abs().max(1.0));
        // Wrong step count is rejected.
        assert!(
            total_camclay_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps[..1])
                .is_none()
        );
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.92);
        let model = camclay();
        let mut a = rest_camclay_states(&model, 2);
        let mut b = rest_camclay_states(&model, 2);
        let oa = assemble_camclay_forces(&basis, &tets, &deformed, MODEL, &lame(), &model, &mut a)
            .unwrap();
        let ob = assemble_camclay_forces(&basis, &tets, &deformed, MODEL, &lame(), &model, &mut b)
            .unwrap();
        assert_eq!(oa, ob);
        assert_eq!(a, b);
    }
}
