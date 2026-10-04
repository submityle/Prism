//! Global assembly of snow elastoplastic forces over a tetrahedral `FEM` mesh.
//!
//! This is the snow counterpart of
//! [`tet_fem_plastic_force_assembly`](super::tet_fem_plastic_force_assembly):
//! instead of the finite-strain J2 return mapping it carries one persistent
//! [`SnowState`](crate::collider::tet_fem_snow_plasticity::SnowState) per
//! element and advances each exactly once per pass through the Stomakhin
//! elastic-predictor / box-projection update in
//! [`tet_fem_snow_force`](super::tet_fem_snow_force).
//!
//! Given a rest-pose [`TetFemBasis`], the tet connectivity, the current
//! deformed vertex positions, and a mutable slice of per-element snow states,
//! [`assemble_snow_forces`] evaluates every element's nodal snow force
//! `fᵢ = -V · P(Fₑ; μ', λ') · gᵢ`, folds any clamped principal stretch into
//! that element's `Fₚ`, and scatters the four nodal forces into a global
//! per-vertex vector. Each element's forces sum to zero, so the assembled total
//! force is zero to within rounding.
//!
//! Snow stiffens as it compacts, so unlike the J2 assembly the Lamé parameters
//! used to evaluate the stress are hardened per element by the *post-step*
//! plastic volume ratio. The returned [`SnowAssembly`] therefore keeps both the
//! per-element [`SnowStep`] and the hardened [`LameParameters`] alongside the
//! global force vector so a caller can report how many elements clamped, how
//! much stretch was consumed, and recover the stored elastic energy via
//! [`total_snow_elastic_potential_energy`] without re-advancing plasticity.
//! This module holds no solver state of its own and performs no time
//! integration, so it feeds directly into an explicit or implicit integrator
//! supplied elsewhere.

use crate::collider::tet_fem_basis::TetFemBasis;
use crate::collider::tet_fem_constitutive::{HyperelasticModel, LameParameters};
use crate::collider::tet_fem_snow_force::{element_snow_force, snow_elastic_potential_energy};
use crate::collider::tet_fem_snow_plasticity::{SnowModel, SnowState, SnowStep};
use glam::Vec3;

/// Aggregate result of one global snow assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct SnowAssembly {
    /// Per-vertex internal snow force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-element return-mapping outcome, in basis / connectivity order.
    pub steps: Vec<SnowStep>,
    /// Per-element hardened Lamé parameters used to evaluate the stress this
    /// pass, in basis / connectivity order. Needed to recover elastic energy
    /// consistently with the stiffened stress that produced the forces.
    pub hardened: Vec<LameParameters>,
}

impl SnowAssembly {
    /// Number of elements that clamped (left the elastic box) this pass.
    #[must_use]
    pub fn clamped_element_count(&self) -> usize {
        self.steps.iter().filter(|s| s.clamped).count()
    }

    /// Total principal-stretch excess clamped away across all elements.
    #[must_use]
    pub fn total_plastic_increment(&self) -> f32 {
        self.steps.iter().map(|s| s.plastic_increment).sum()
    }
}

/// Builds `count` fresh rest snow states, one per element.
#[must_use]
pub fn rest_snow_states(count: usize) -> Vec<SnowState> {
    vec![SnowState::rest(); count]
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
/// mutually consistent: one element and one snow state per tet, and every tet
/// index addressable in `positions`.
fn is_consistent(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    states: &[SnowState],
) -> bool {
    if basis.element_count() != tets.len() || states.len() != tets.len() {
        return false;
    }
    let n = positions.len();
    tets.iter().all(|t| t.iter().all(|&vi| (vi as usize) < n))
}

/// Assembles the per-vertex snow force vector of the mesh at the supplied
/// deformed positions, advancing each element's snow state once.
///
/// `states` must hold exactly one [`SnowState`] per tet (see
/// [`rest_snow_states`]); each is mutated in place when its element clamps.
/// Returns `None` when the basis element count, `tets`, and `states` lengths
/// disagree or when any tet index is out of range for `positions`; in that case
/// no state is mutated.
///
/// When every element stays elastic (no stretch leaves the box, so `det Fₚ = 1`
/// and the moduli are unscaled) the result matches
/// [`assemble_internal_forces`](super::tet_fem_force_assembly::assemble_internal_forces)
/// on the same positions with `base_lame`.
#[must_use]
pub fn assemble_snow_forces(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &[Vec3],
    model: HyperelasticModel,
    base_lame: &LameParameters,
    snow_model: &SnowModel,
    states: &mut [SnowState],
) -> Option<SnowAssembly> {
    if !is_consistent(basis, tets, positions, states) {
        return None;
    }
    // Validate every index up front so a late failure cannot leave the snow
    // states partially advanced.
    for &t in tets {
        gather(positions, t)?;
    }

    let mut forces = vec![Vec3::ZERO; positions.len()];
    let mut steps = Vec::with_capacity(tets.len());
    let mut hardened = Vec::with_capacity(tets.len());
    for ((element, &t), state) in basis
        .elements
        .iter()
        .zip(tets.iter())
        .zip(states.iter_mut())
    {
        let nodes = gather(positions, t)?;
        let out = element_snow_force(element, nodes, model, base_lame, snow_model, state);
        for (local, &vi) in t.iter().enumerate() {
            forces[vi as usize] += out.forces[local];
        }
        steps.push(out.step);
        hardened.push(out.hardened_lame);
    }
    Some(SnowAssembly {
        forces,
        steps,
        hardened,
    })
}

/// Total recoverable elastic potential energy `U = Σ_e V_e · Ψ(Fₑ; μ', λ')` of
/// the mesh, using the per-element elastic gradients and hardened Lamé
/// parameters produced by a prior [`assemble_snow_forces`] pass.
///
/// This is a pure read of the stored elastic energy; it does not advance
/// plasticity. Returns `None` when `steps` and `hardened` do not each hold
/// exactly one entry per element.
#[must_use]
pub fn total_snow_elastic_potential_energy(
    basis: &TetFemBasis,
    model: HyperelasticModel,
    steps: &[SnowStep],
    hardened: &[LameParameters],
) -> Option<f32> {
    let count = basis.element_count();
    if steps.len() != count || hardened.len() != count {
        return None;
    }
    let mut energy = 0.0_f32;
    for ((element, step), lame) in basis.elements.iter().zip(steps.iter()).zip(hardened.iter()) {
        energy += snow_elastic_potential_energy(element, model, lame, step.elastic_gradient);
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

    // A wide elastic box so a modest deformation stays elastic.
    fn wide_snow() -> SnowModel {
        SnowModel::new(0.2, 0.1, 10.0).unwrap()
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
        // One state short of the two elements.
        let mut states = rest_snow_states(1);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &verts,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        );
        assert!(out.is_none());
        assert_eq!(states, rest_snow_states(1), "nothing may advance");
    }

    #[test]
    fn rejects_out_of_range_index_without_mutating() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        // Truncate positions so tet index 4 is unaddressable.
        let truncated = verts[..4].to_vec();
        let mut states = rest_snow_states(2);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &truncated,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        );
        assert!(out.is_none());
        assert_eq!(states, rest_snow_states(2), "nothing may advance");
    }

    #[test]
    fn rest_pose_is_force_free_and_unclamped() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let mut states = rest_snow_states(2);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &verts,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        )
        .unwrap();
        assert_eq!(out.clamped_element_count(), 0);
        for (i, f) in out.forces.iter().enumerate() {
            assert!(f.length() <= 1e-1, "vertex {i} rest force {f:?}");
        }
        assert_eq!(states, rest_snow_states(2), "rest pose must not clamp");
    }

    #[test]
    fn elastic_pass_matches_hyperelastic_assembly() {
        // A modest stretch stays inside the box, so det Fp = 1 leaves the Lamé
        // parameters unscaled and the snow assembly must agree with the
        // stateless elastic assembly on base_lame.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 1.02);
        let mut states = rest_snow_states(2);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        )
        .unwrap();
        assert_eq!(out.clamped_element_count(), 0);
        let reference = assemble_internal_forces(&basis, &tets, &deformed, MODEL, &lame()).unwrap();
        assert_eq!(out.forces.len(), reference.len());
        for i in 0..reference.len() {
            assert!(
                (out.forces[i] - reference[i]).length() < 1e-1,
                "elastic snow assembly must match hyperelastic at vertex {i}"
            );
        }
    }

    #[test]
    fn compression_clamps_and_global_force_sums_to_zero() {
        // Uniform volumetric compression below 1 - critical_compression clamps
        // every principal stretch, which (unlike J2) does yield for snow.
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.5);
        let mut states = rest_snow_states(2);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        )
        .unwrap();
        assert!(
            out.clamped_element_count() > 0,
            "volumetric compression should clamp"
        );
        assert!(out.total_plastic_increment() > 0.0);
        let net: Vec3 = out.forces.iter().copied().sum();
        assert!(net.length() <= 1e-1, "net force {net:?}");
    }

    #[test]
    fn clamping_advances_states() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.5);
        let mut states = rest_snow_states(2);
        let _ = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        )
        .unwrap();
        assert_ne!(
            states,
            rest_snow_states(2),
            "clamped elements must advance Fp"
        );
    }

    #[test]
    fn energy_helper_is_finite_and_matches_sum() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.5);
        let mut states = rest_snow_states(2);
        let out = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut states,
        )
        .unwrap();
        let energy =
            total_snow_elastic_potential_energy(&basis, MODEL, &out.steps, &out.hardened).unwrap();
        assert!(energy.is_finite());
        let manual: f32 = basis
            .elements
            .iter()
            .zip(out.steps.iter())
            .zip(out.hardened.iter())
            .map(|((e, s), l)| snow_elastic_potential_energy(e, MODEL, l, s.elastic_gradient))
            .sum();
        assert!((energy - manual).abs() <= 1e-3 * manual.abs().max(1.0));
        // Wrong step/hardened count is rejected.
        assert!(
            total_snow_elastic_potential_energy(&basis, MODEL, &out.steps[..1], &out.hardened)
                .is_none()
        );
        assert!(
            total_snow_elastic_potential_energy(&basis, MODEL, &out.steps, &out.hardened[..1])
                .is_none()
        );
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = basis_of(&verts, &tets);
        let deformed = scaled(&verts, 0.6);
        let mut a = rest_snow_states(2);
        let mut b = rest_snow_states(2);
        let oa = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut a,
        )
        .unwrap();
        let ob = assemble_snow_forces(
            &basis,
            &tets,
            &deformed,
            MODEL,
            &lame(),
            &wide_snow(),
            &mut b,
        )
        .unwrap();
        assert_eq!(oa, ob);
        assert_eq!(a, b);
    }
}
