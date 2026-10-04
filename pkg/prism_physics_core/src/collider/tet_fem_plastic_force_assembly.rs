//! Global assembly of elastoplastic forces over a tetrahedral `FEM` mesh.
//!
//! This is the plastic counterpart of
//! [`tet_fem_force_assembly`](super::tet_fem_force_assembly): instead of a
//! stateless elastic scatter, it carries one persistent
//! [`PlasticState`](crate::collider::tet_fem_plasticity::PlasticState) per
//! element and advances each exactly once per pass through the finite-strain
//! J2 return mapping in
//! [`tet_fem_elastoplastic_force`](super::tet_fem_elastoplastic_force).
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, the current
//! deformed vertex positions, and a mutable slice of per-element plastic
//! states, [`assemble_elastoplastic_forces`] evaluates every element's nodal
//! elastoplastic force `fᵢ = -V · P(Fₑ) · gᵢ`, folds any consumed strain into
//! that element's `Fₚ`, and scatters the four nodal forces into a global
//! per-vertex vector. Because each element's forces sum to zero, the assembled
//! total force is zero to within rounding.
//!
//! The returned [`ElastoplasticAssembly`] keeps the per-element
//! [`PlasticStep`] alongside the global force vector so a caller can report how
//! many elements yielded, how much plastic strain was consumed, and recover the
//! stored elastic energy via [`total_elastic_potential_energy`] without
//! re-advancing plasticity. This module holds no solver state of its own and
//! performs no time integration, so it feeds directly into an explicit or
//! implicit integrator supplied elsewhere.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_elastoplastic_force::{
    elastic_potential_energy, element_elastoplastic_force,
};
use crate::collider::tet_fem_plasticity::{PlasticModel, PlasticState, PlasticStep};
use glam::Vec3;

/// Aggregate result of one global elastoplastic assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ElastoplasticAssembly {
    /// Per-vertex internal elastoplastic force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-element return-mapping outcome, in basis / connectivity order.
    pub steps: Vec<PlasticStep>,
}

impl ElastoplasticAssembly {
    /// Number of elements that yielded (left the elastic region) this pass.
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

/// Builds `count` fresh rest plastic states, one per element.
#[must_use]
pub fn rest_states(count: usize) -> Vec<PlasticState> {
    vec![PlasticState::rest(); count]
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
/// mutually consistent: one element and one plastic state per tet, and every
/// tet index addressable in `positions`.
fn is_consistent(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    states: &[PlasticState],
) -> bool {
    if basis.element_count() != tets.len() || states.len() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Assembles the per-vertex elastoplastic force vector of the mesh at the
/// supplied deformed positions, advancing each element's plastic state once.
///
/// `states` must hold exactly one [`PlasticState`] per tet (see
/// [`rest_states`]); each is mutated in place when its element yields. Returns
/// `None` when the basis element count, `tets`, and `states` lengths disagree
/// or when any tet index is out of range for `positions`; in that case no state
/// is mutated.
///
/// When every element stays elastic the result matches
/// [`assemble_internal_forces`](super::tet_fem_force_assembly::assemble_internal_forces)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
#[must_use]
pub fn assemble_elastoplastic_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
    plastic_model: &PlasticModel,
    states: &mut [PlasticState],
) -> Option<ElastoplasticAssembly> {
    if !is_consistent(basis, tets, positions, states) {
        return None;
    }
    // Validate every index up front so a late failure cannot leave the plastic
    // states partially advanced.
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
        let out = element_elastoplastic_force(element, nodes, model, lame, plastic_model, state);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += out.forces[local];
        }
        steps.push(out.step);
    }
    Some(ElastoplasticAssembly { forces, steps })
}

/// Total recoverable elastic potential energy `U = Σ_e V_e · Ψ(Fₑ)` of the
/// mesh, using the per-element elastic gradients produced by a prior
/// [`assemble_elastoplastic_forces`] pass.
///
/// This is a pure read of the stored elastic energy; it does not advance
/// plasticity. Returns `None` when `steps` does not hold exactly one entry per
/// element.
#[must_use]
pub fn total_elastic_potential_energy(
    basis: &TetFemBasis,
    model: HyperelasticModel,
    lame: &LameParameters,
    steps: &[PlasticStep],
) -> Option<f32> {
    if steps.len() != basis.element_count() {
        return None;
    }
    let mut energy = 0.0_f32;
    for (element, step) in basis.elements.iter().zip(steps.iter()) {
        energy += elastic_potential_energy(element, model, lame, step.elastic_gradient);
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
        LameParameters::from_isotropic(&IsotropicElasticity::new(1.0e6, 0.4).unwrap())
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

    /// Applies an affine map to every vertex to inject deviatoric,
    /// shape-changing strain that drives J2 yielding.
    fn affine(verts: &[Vec3], a: glam::Mat3) -> Vec<Vec3> {
        verts.iter().map(|v| a * *v).collect()
    }

    #[test]
    fn rejects_inconsistent_state_length() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.05).unwrap();
        // One state short of the two elements.
        let mut states = rest_states(1);
        let out = assemble_elastoplastic_forces(
            &basis,
            &tets,
            &verts,
            MODEL,
            &lame(),
            &model,
            &mut states,
        );
        assert!(out.is_none());
        // Nothing advanced.
        assert_eq!(states, rest_states(1));
    }

    #[test]
    fn rejects_out_of_range_index_without_mutating() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.05).unwrap();
        let mut states = rest_states(2);
        // A positions array too short to address vertex 4.
        let truncated = &verts[..4];
        let out = assemble_elastoplastic_forces(
            &basis,
            &tets,
            truncated,
            MODEL,
            &lame(),
            &model,
            &mut states,
        );
        assert!(out.is_none());
        assert_eq!(
            states,
            rest_states(2),
            "failed pass must not advance states"
        );
    }

    #[test]
    fn rest_pose_is_force_free_and_elastic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.05).unwrap();
        let mut states = rest_states(2);
        let out = assemble_elastoplastic_forces(
            &basis,
            &tets,
            &verts,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert_eq!(out.yielded_element_count(), 0);
        for (i, f) in out.forces.iter().enumerate() {
            assert!(f.length() <= 1e-1, "vertex {i} rest force {f:?}");
        }
        assert_eq!(states, rest_states(2), "rest pose must not yield");
    }

    #[test]
    fn elastic_pass_matches_hyperelastic_assembly() {
        // A high yield strain keeps a modest deformation purely elastic, so the
        // plastic assembly must agree with the stateless elastic assembly.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(10.0).unwrap();
        let deformed = scaled(&verts, 1.02);
        let mut states = rest_states(2);
        let out = assemble_elastoplastic_forces(
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
                "elastic assembly must match hyperelastic at vertex {i}"
            );
        }
    }

    #[test]
    fn global_force_sums_to_zero_after_yield() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.01).unwrap();
        let deformed = affine(
            &verts,
            glam::Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)),
        );
        let mut states = rest_states(2);
        let out = assemble_elastoplastic_forces(
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
            "deviatoric strain should yield"
        );
        assert!(out.total_plastic_increment() > 0.0);
        let net: Vec3 = out.forces.iter().copied().sum();
        assert!(net.length() <= 1e-1, "net force {net:?}");
    }

    #[test]
    fn yielding_advances_states() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.01).unwrap();
        let deformed = affine(
            &verts,
            glam::Mat3::from_diagonal(Vec3::new(1.6, 0.625, 1.0)),
        );
        let mut states = rest_states(2);
        let _ = assemble_elastoplastic_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert_ne!(states, rest_states(2), "yielded elements must advance Fp");
    }

    #[test]
    fn energy_helper_is_finite_and_matches_sum() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.01).unwrap();
        let deformed = scaled(&verts, 1.3);
        let mut states = rest_states(2);
        let out = assemble_elastoplastic_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        let energy = total_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps).unwrap();
        assert!(energy.is_finite());
        let manual: f32 = basis
            .elements
            .iter()
            .zip(out.steps.iter())
            .map(|(e, s)| elastic_potential_energy(e, MODEL, &lame(), s.elastic_gradient))
            .sum();
        assert!((energy - manual).abs() <= 1e-3 * manual.abs().max(1.0));
        // Wrong step count is rejected.
        assert!(total_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps[..1]).is_none());
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = PlasticModel::perfectly_plastic(0.02).unwrap();
        let deformed = scaled(&verts, 1.4);
        let mut a = rest_states(2);
        let mut b = rest_states(2);
        let oa =
            assemble_elastoplastic_forces(&basis, &tets, &deformed, MODEL, &lame(), &model, &mut a)
                .unwrap();
        let ob =
            assemble_elastoplastic_forces(&basis, &tets, &deformed, MODEL, &lame(), &model, &mut b)
                .unwrap();
        assert_eq!(oa, ob);
        assert_eq!(a, b);
    }
}
