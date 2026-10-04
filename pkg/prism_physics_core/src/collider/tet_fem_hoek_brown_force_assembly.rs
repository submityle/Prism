//! Global assembly of Hoek–Brown elastoplastic forces over a tetrahedral `FEM`
//! mesh.
//!
//! This is the curved-rock-mass counterpart of
//! [`tet_fem_mohr_coulomb_force_assembly`](super::tet_fem_mohr_coulomb_force_assembly):
//! instead of the straight-faceted pyramid it carries one persistent
//! [`HoekBrownState`](crate::collider::tet_fem_hoek_brown_plasticity::HoekBrownState)
//! per element and advances each exactly once per pass through the nonlinear
//! multisurface elastic-predictor / main-surface–edge–apex return update in
//! [`tet_fem_hoek_brown_force`](super::tet_fem_hoek_brown_force).
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, the current
//! deformed vertex positions, and a mutable slice of per-element states,
//! [`assemble_hoek_brown_forces`] evaluates every element's nodal force
//! `fᵢ = -V · P(Fₑ) · gᵢ`, folds any consumed strain into that element's `Fₚ`,
//! and scatters the four nodal forces into a global per-vertex vector. Each
//! element's forces sum to zero, so the assembled total force is zero to within
//! rounding.
//!
//! The stress uses the constant `lame` moduli (the yield surface, not the
//! moduli, carries the rock-mass strength), so there is no per-element modulus
//! hardening to store; the sole history variable is the accumulated plastic
//! strain, which feeds the (hardening) intact strength back into the surface.
//! The returned [`HoekBrownAssembly`] keeps the per-element [`HoekBrownStep`]
//! alongside the global force vector so a caller can report how many elements
//! yielded, how much plastic strain was consumed, and recover the stored
//! elastic energy via [`total_hoek_brown_elastic_potential_energy`] without
//! re-advancing plasticity. This module holds no solver state of its own and
//! performs no time integration, so it feeds directly into an explicit or
//! implicit integrator supplied elsewhere.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_hoek_brown_force::{
    element_hoek_brown_force, hoek_brown_elastic_potential_energy,
};
use crate::collider::tet_fem_hoek_brown_plasticity::{
    HoekBrownModel, HoekBrownState, HoekBrownStep, HoekBrownYield,
};
use glam::Vec3;

/// Aggregate result of one global Hoek–Brown assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct HoekBrownAssembly {
    /// Per-vertex internal force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-element return-mapping outcome, in basis / connectivity order.
    pub steps: Vec<HoekBrownStep>,
}

impl HoekBrownAssembly {
    /// Number of elements that left the elastic interior this pass (projected
    /// onto the main surface, an edge, or the apex of the surface).
    #[must_use]
    pub fn yielded_element_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.mode != HoekBrownYield::Elastic)
            .count()
    }

    /// Total plastic strain consumed across all elements this pass.
    #[must_use]
    pub fn total_plastic_increment(&self) -> f32 {
        self.steps.iter().map(|s| s.plastic_increment).sum()
    }
}

/// Builds `count` fresh rest states, one per element.
#[must_use]
pub fn rest_hoek_brown_states(count: usize) -> Vec<HoekBrownState> {
    vec![HoekBrownState::rest(); count]
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
/// mutually consistent: one element and one state per tet, and every tet index
/// addressable in `positions`.
fn is_consistent(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    states: &[HoekBrownState],
) -> bool {
    if basis.element_count() != tets.len() || states.len() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Assembles the per-vertex Hoek–Brown force vector of the mesh at the supplied
/// deformed positions, advancing each element's state once.
///
/// `states` must hold exactly one [`HoekBrownState`] per tet (see
/// [`rest_hoek_brown_states`]); each is mutated in place when its element
/// yields. Returns `None` when the basis element count, `tets`, and `states`
/// lengths disagree or when any tet index is out of range for `positions`; in
/// that case no state is mutated.
///
/// When every element stays inside the surface the result matches
/// [`assemble_internal_forces`](super::tet_fem_force_assembly::assemble_internal_forces)
/// on the same positions, because `Fₑ = F · Fₚ⁻¹` reduces to `F` while
/// `Fₚ = I`.
#[must_use]
pub fn assemble_hoek_brown_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    lame: &LameParameters,
    rock_model: &HoekBrownModel,
    states: &mut [HoekBrownState],
) -> Option<HoekBrownAssembly> {
    if !is_consistent(basis, tets, positions, states) {
        return None;
    }
    // Validate every index up front so a late failure cannot leave the states
    // partially advanced.
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
        let out = element_hoek_brown_force(element, nodes, model, lame, rock_model, state);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += out.forces[local];
        }
        steps.push(out.step);
    }
    Some(HoekBrownAssembly { forces, steps })
}

/// Total recoverable elastic potential energy `U = Σ_e V_e · Ψ(Fₑ)` of the
/// mesh, using the per-element elastic gradients produced by a prior
/// [`assemble_hoek_brown_forces`] pass with the same constant `lame` moduli.
///
/// This is a pure read of the stored elastic energy; it does not advance
/// plasticity. Returns `None` when `steps` does not hold exactly one entry per
/// element.
#[must_use]
pub fn total_hoek_brown_elastic_potential_energy(
    basis: &TetFemBasis,
    model: HyperelasticModel,
    lame: &LameParameters,
    steps: &[HoekBrownStep],
) -> Option<f32> {
    if steps.len() != basis.element_count() {
        return None;
    }
    let mut energy = 0.0_f32;
    for (element, step) in basis.elements.iter().zip(steps.iter()) {
        energy += hoek_brown_elastic_potential_energy(element, model, lame, step.elastic_gradient);
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

    fn rock() -> HoekBrownModel {
        HoekBrownModel::new(5000.0, 1.0, 1.0, 0.5, 1.0, 0.0).unwrap()
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

    /// Applies an affine map to every vertex so each element's deformation
    /// gradient equals `a` exactly.
    fn affine(verts: &[Vec3], a: glam::Mat3) -> Vec<Vec3> {
        verts.iter().map(|v| a * *v).collect()
    }

    // Converts a principal log-strain into its stretch, in f64 to avoid the
    // disallowed f32 transcendental methods.
    fn stretch(log: f32) -> f32 {
        f64::from(log).exp() as f32
    }

    // Diagonal deformation gradient whose principal log-strains are exactly the
    // supplied (descending) vector.
    fn from_log_strain(e: Vec3) -> glam::Mat3 {
        glam::Mat3::from_diagonal(Vec3::new(stretch(e.x), stretch(e.y), stretch(e.z)))
    }

    #[test]
    fn rejects_inconsistent_state_length() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut states = rest_hoek_brown_states(1);
        let out =
            assemble_hoek_brown_forces(&basis, &tets, &verts, MODEL, &lame(), &rock(), &mut states);
        assert!(out.is_none());
        assert_eq!(states, rest_hoek_brown_states(1), "nothing may advance");
    }

    #[test]
    fn rejects_out_of_range_index_without_mutating() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let truncated = verts[..4].to_vec();
        let mut states = rest_hoek_brown_states(2);
        let out = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &truncated,
            MODEL,
            &lame(),
            &rock(),
            &mut states,
        );
        assert!(out.is_none());
        assert_eq!(states, rest_hoek_brown_states(2), "nothing may advance");
    }

    #[test]
    fn rest_pose_is_force_free_and_elastic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut states = rest_hoek_brown_states(2);
        let out =
            assemble_hoek_brown_forces(&basis, &tets, &verts, MODEL, &lame(), &rock(), &mut states)
                .unwrap();
        assert_eq!(out.yielded_element_count(), 0);
        for (i, f) in out.forces.iter().enumerate() {
            assert!(f.length() <= 1e-1, "vertex {i} rest force {f:?}");
        }
        assert_eq!(
            states,
            rest_hoek_brown_states(2),
            "rest pose must not yield"
        );
    }

    #[test]
    fn elastic_pass_matches_hyperelastic_assembly() {
        // A tiny uniform compression is hydrostatic and well inside the surface,
        // so the assembly must agree with the stateless elastic assembly.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.999);
        let mut states = rest_hoek_brown_states(2);
        let out = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &rock(),
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
    fn confined_shear_yields_and_sums_to_zero() {
        // A confined (compressive σ₁) triaxial strain overshoots the curved
        // surface and must be projected onto a yield feature on every element.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = affine(&verts, from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let mut states = rest_hoek_brown_states(2);
        let out = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &rock(),
            &mut states,
        )
        .unwrap();
        assert!(
            out.yielded_element_count() > 0,
            "confined overshoot should yield"
        );
        assert!(out.total_plastic_increment() > 0.0);
        let net: Vec3 = out.forces.iter().copied().sum();
        assert!(net.length() <= 1e-1, "net force {net:?}");
    }

    #[test]
    fn expansion_hits_apex_and_sums_to_zero() {
        // A purely expanding predictor drives hydrostatic tension far past the
        // apex of the surface and must be returned onto it on every element.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = affine(&verts, from_log_strain(Vec3::splat(0.05)));
        let mut states = rest_hoek_brown_states(2);
        let out = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &rock(),
            &mut states,
        )
        .unwrap();
        assert_eq!(out.yielded_element_count(), 2);
        for s in &out.steps {
            assert_eq!(s.mode, HoekBrownYield::Apex);
        }
        let net: Vec3 = out.forces.iter().copied().sum();
        assert!(net.length() <= 1e-1, "net force {net:?}");
    }

    #[test]
    fn yielding_advances_states() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = affine(&verts, from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let mut states = rest_hoek_brown_states(2);
        let _ = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &rock(),
            &mut states,
        )
        .unwrap();
        assert_ne!(
            states,
            rest_hoek_brown_states(2),
            "yielded elements must accumulate strain"
        );
    }

    #[test]
    fn energy_helper_is_finite_and_matches_sum() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = affine(&verts, from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let mut states = rest_hoek_brown_states(2);
        let out = assemble_hoek_brown_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &rock(),
            &mut states,
        )
        .unwrap();
        let energy =
            total_hoek_brown_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps).unwrap();
        assert!(energy.is_finite());
        let manual: f32 = basis
            .elements
            .iter()
            .zip(out.steps.iter())
            .map(|(e, s)| {
                hoek_brown_elastic_potential_energy(e, MODEL, &lame(), s.elastic_gradient)
            })
            .sum();
        assert!((energy - manual).abs() <= 1e-3 * manual.abs().max(1.0));
        // Wrong step count is rejected.
        assert!(
            total_hoek_brown_elastic_potential_energy(&basis, MODEL, &lame(), &out.steps[..1])
                .is_none()
        );
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = affine(&verts, from_log_strain(Vec3::new(-0.005, -0.02, -0.035)));
        let mut a = rest_hoek_brown_states(2);
        let mut b = rest_hoek_brown_states(2);
        let oa =
            assemble_hoek_brown_forces(&basis, &tets, &deformed, MODEL, &lame(), &rock(), &mut a)
                .unwrap();
        let ob =
            assemble_hoek_brown_forces(&basis, &tets, &deformed, MODEL, &lame(), &rock(), &mut b)
                .unwrap();
        assert_eq!(oa, ob);
        assert_eq!(a, b);
    }
}
