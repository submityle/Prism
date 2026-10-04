//! Global assembly of damaged (degraded) forces over a tetrahedral `FEM` mesh.
//!
//! This is the continuum-damage counterpart of
//! [`tet_fem_force_assembly`](super::tet_fem_force_assembly) and the sibling of
//! [`tet_fem_plastic_force_assembly`](super::tet_fem_plastic_force_assembly):
//! instead of a stateless elastic scatter, it carries one persistent
//! [`DamageState`](crate::collider::tet_fem_damage::DamageState) per element and
//! advances each exactly once per pass through the scalar damage model in
//! [`tet_fem_damage_force`](super::tet_fem_damage_force).
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, the current
//! deformed vertex positions, and a mutable slice of per-element damage states,
//! [`assemble_damaged_forces`] evaluates every element's degraded nodal force
//! `fᵢ = -(1 − D) · V · P(F) · gᵢ`, updates that element's irreversible damage
//! history, and scatters the four nodal forces into a global per-vertex vector.
//! Because each element's forces sum to zero, the assembled total force is zero
//! to within rounding even when elements are cracking.
//!
//! The returned [`DamageAssembly`] keeps the per-element
//! [`DamageStep`] alongside the global force vector so a caller can report how
//! many elements are damaged and recover the degraded stored energy via
//! [`total_degraded_potential_energy`] without re-advancing the damage state.
//! This module holds no solver state of its own and performs no time
//! integration, so it feeds directly into an explicit or implicit integrator
//! supplied elsewhere.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_damage::{DamageModel, DamageState, DamageStep};
use crate::collider::tet_fem_damage_force::{
    damaged_elastic_potential_energy, element_damaged_force,
};
use glam::Vec3;

/// Aggregate result of one global damaged-force assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct DamageAssembly {
    /// Per-vertex internal degraded force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-element damage-update outcome, in basis / connectivity order.
    pub steps: Vec<DamageStep>,
}

impl DamageAssembly {
    /// Number of elements that carry any damage (degradation below one) after
    /// this pass.
    #[must_use]
    pub fn damaged_element_count(&self) -> usize {
        self.steps.iter().filter(|s| s.damage > 0.0).count()
    }

    /// Number of elements whose irreversible history advanced this pass.
    #[must_use]
    pub fn advanced_element_count(&self) -> usize {
        self.steps.iter().filter(|s| s.advanced).count()
    }

    /// Mean degradation factor `(1 − D)` across all elements, or `1.0` for an
    /// empty mesh. A value of one means the mesh is fully intact.
    #[must_use]
    pub fn mean_degradation(&self) -> f32 {
        if self.steps.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.steps.iter().map(|s| s.degradation).sum();
        sum / self.steps.len() as f32
    }
}

/// Builds `count` fresh rest damage states, one per element.
#[must_use]
pub fn rest_states(count: usize) -> Vec<DamageState> {
    vec![DamageState::rest(); count]
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
/// mutually consistent: one element and one damage state per tet, and every
/// tet index addressable in `positions`.
fn is_consistent(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    states: &[DamageState],
) -> bool {
    if basis.element_count() != tets.len() || states.len() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Assembles the per-vertex degraded force vector of the mesh at the supplied
/// deformed positions, advancing each element's damage state once.
///
/// `states` must hold exactly one [`DamageState`] per tet (see
/// [`rest_states`]); each is updated in place whenever its element's tensile
/// equivalent strain exceeds the stored history. Returns `None` when the basis
/// element count, `tets`, and `states` lengths disagree or when any tet index
/// is out of range for `positions`; in that case no state is mutated.
///
/// When every element is undamaged the result matches
/// [`assemble_internal_forces`](super::tet_fem_force_assembly::assemble_internal_forces)
/// on the same positions, because the degradation factor is `1`.
#[must_use]
pub fn assemble_damaged_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
    damage_model: &DamageModel,
    states: &mut [DamageState],
) -> Option<DamageAssembly> {
    if !is_consistent(basis, tets, positions, states) {
        return None;
    }
    // Validate every index up front so a late failure cannot leave the damage
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
        let out = element_damaged_force(element, nodes, model, lame, damage_model, state);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += out.forces[local];
        }
        steps.push(out.step);
    }
    Some(DamageAssembly { forces, steps })
}

/// Total degraded potential energy `U = Σ_e (1 − D_e) · V_e · Ψ(F_e)` of the
/// mesh at the supplied deformed positions, using the per-element degradation
/// factors produced by a prior [`assemble_damaged_forces`] pass.
///
/// This is a pure read of the degraded stored energy; it does not advance the
/// damage state. The deformation gradient is recomputed from `positions`, which
/// must match the positions used to produce `steps`. Returns `None` when
/// `steps`, `tets`, and the basis element count disagree, or when any tet index
/// is out of range for `positions`.
#[must_use]
pub fn total_degraded_potential_energy(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
    steps: &[DamageStep],
) -> Option<f32> {
    if steps.len() != basis.element_count() || tets.len() != basis.element_count() {
        return None;
    }
    let mut energy = 0.0_f32;
    for ((element, &t), step) in basis.elements.iter().zip(tets.iter()).zip(steps.iter()) {
        let [p0, p1, p2, p3] = gather(positions, t)?;
        let f_total = element.deformation_gradient(p0, p1, p2, p3);
        energy += damaged_elastic_potential_energy(element, model, lame, f_total, step.degradation);
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

    /// Applies an affine map to every vertex to inject tensile, shape-changing
    /// strain that drives damage.
    fn affine(verts: &[Vec3], a: glam::Mat3) -> Vec<Vec3> {
        verts.iter().map(|v| a * *v).collect()
    }

    fn damage() -> DamageModel {
        DamageModel::new(0.05, 0.3, 0.0).unwrap()
    }

    #[test]
    fn rejects_inconsistent_state_length() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let mut states = rest_states(1);
        let out =
            assemble_damaged_forces(&basis, &tets, &verts, MODEL, &lame(), &model, &mut states);
        assert!(out.is_none());
        assert_eq!(
            states,
            rest_states(1),
            "failed pass must not advance states"
        );
    }

    #[test]
    fn rejects_out_of_range_index_without_mutating() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let mut states = rest_states(2);
        let truncated = &verts[..4];
        let out = assemble_damaged_forces(
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
    fn rest_pose_is_force_free_and_intact() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let mut states = rest_states(2);
        let out =
            assemble_damaged_forces(&basis, &tets, &verts, MODEL, &lame(), &model, &mut states)
                .unwrap();
        assert_eq!(out.damaged_element_count(), 0);
        assert_eq!(out.advanced_element_count(), 0);
        assert!((out.mean_degradation() - 1.0).abs() < 1e-6);
        for (i, f) in out.forces.iter().enumerate() {
            assert!(f.length() <= 1e-1, "vertex {i} rest force {f:?}");
        }
    }

    #[test]
    fn intact_pass_matches_hyperelastic_assembly() {
        // A tiny deformation stays below the damage onset, so the degraded
        // assembly must agree with the stateless elastic assembly.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let deformed = scaled(&verts, 1.01);
        let mut states = rest_states(2);
        let out = assemble_damaged_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert_eq!(out.damaged_element_count(), 0);
        let reference = assemble_internal_forces(&basis, &tets, &deformed, MODEL, &lame()).unwrap();
        assert_eq!(out.forces.len(), reference.len());
        for i in 0..reference.len() {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "intact assembly must match hyperelastic at vertex {i}"
            );
        }
    }

    #[test]
    fn global_force_sums_to_zero_after_damage() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let deformed = affine(&verts, glam::Mat3::from_diagonal(Vec3::new(1.6, 1.0, 1.0)));
        let mut states = rest_states(2);
        let out = assemble_damaged_forces(
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
            out.damaged_element_count() > 0,
            "tensile stretch should damage"
        );
        let net: Vec3 = out.forces.iter().copied().fold(Vec3::ZERO, |a, b| a + b);
        assert!(
            net.length() < 1e-1,
            "assembled net force must vanish, got {net:?}"
        );
    }

    #[test]
    fn damage_reduces_energy_versus_intact() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let deformed = affine(&verts, glam::Mat3::from_diagonal(Vec3::new(1.6, 1.0, 1.0)));
        let mut states = rest_states(2);
        let out = assemble_damaged_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states,
        )
        .unwrap();
        assert!(out.damaged_element_count() > 0);

        let degraded =
            total_degraded_potential_energy(&basis, &tets, &deformed, MODEL, &lame(), &out.steps)
                .unwrap();

        // Energy with all degradation factors forced to one (fully intact).
        let intact_steps: Vec<DamageStep> = out
            .steps
            .iter()
            .map(|s| DamageStep {
                degradation: 1.0,
                ..*s
            })
            .collect();
        let intact = total_degraded_potential_energy(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &intact_steps,
        )
        .unwrap();

        assert!(degraded > 0.0 && intact > 0.0);
        assert!(
            degraded < intact,
            "damage must lower stored energy: {degraded} vs {intact}"
        );
    }

    #[test]
    fn energy_rejects_mismatched_steps() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let steps = vec![DamageStep {
            equivalent_strain: 0.0,
            damage: 0.0,
            degradation: 1.0,
            advanced: false,
        }];
        let out = total_degraded_potential_energy(&basis, &tets, &verts, MODEL, &lame(), &steps);
        assert!(out.is_none(), "one step for two elements must be rejected");
    }

    #[test]
    fn assembly_is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let deformed = affine(&verts, glam::Mat3::from_diagonal(Vec3::new(1.6, 1.0, 1.0)));

        let mut states_a = rest_states(2);
        let a = assemble_damaged_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states_a,
        )
        .unwrap();
        let mut states_b = rest_states(2);
        let b = assemble_damaged_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &model,
            &mut states_b,
        )
        .unwrap();
        assert_eq!(a, b, "identical inputs must assemble identically");
        assert_eq!(
            states_a, states_b,
            "identical inputs must leave identical state"
        );
    }

    #[test]
    fn unloading_preserves_irreversible_damage() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let model = damage();
        let mut states = rest_states(2);

        let big = affine(&verts, glam::Mat3::from_diagonal(Vec3::new(1.7, 1.0, 1.0)));
        let first =
            assemble_damaged_forces(&basis, &tets, &big, MODEL, &lame(), &model, &mut states)
                .unwrap();
        assert!(first.damaged_element_count() > 0);
        let frozen: Vec<f32> = first.steps.iter().map(|s| s.degradation).collect();

        let small = scaled(&verts, 1.01);
        let second =
            assemble_damaged_forces(&basis, &tets, &small, MODEL, &lame(), &model, &mut states)
                .unwrap();
        assert_eq!(
            second.advanced_element_count(),
            0,
            "unloading must not advance history"
        );
        for (i, s) in second.steps.iter().enumerate() {
            assert!(
                (s.degradation - frozen[i]).abs() < 1e-6,
                "element {i} must retain frozen degradation {} vs {}",
                s.degradation,
                frozen[i]
            );
        }
    }
}
