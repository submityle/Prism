//! Fixed-rate multi-substep driver for the implicit corotational Newmark-β
//! integrator *with prescribed (inhomogeneous) Dirichlet velocities* and
//! all-or-nothing rollback.
//!
//! [`step_newmark_substeps`](super::tet_fem_substep::step_newmark_substeps)
//! splits a frame into equal implicit substeps but supports only homogeneous
//! boundary conditions (a `pinned` vertex is frozen). Animation-driven
//! simulation needs the substep robustness *and* kinematic anchors driven to a
//! prescribed end-of-step velocity `v̄` — a skeleton-dragged handle, a scripted
//! control point, a contact-resolved vertex. This module provides that by
//! substepping [`step_newmark_prescribed`] instead of the plain step.
//!
//! The prescribed velocities and the external load are both held constant
//! across the substeps of a frame. Because each substep drives a prescribed
//! vertex so its end-of-substep velocity equals `v̄` exactly, the vertex reaches
//! `v̄` on the first substep and holds it, so the end-of-frame velocity of a
//! driven vertex is `v̄` regardless of the substep count.
//!
//! The two guarantees of the homogeneous driver carry over unchanged:
//!
//! * **All-or-nothing.** The caller's [`NewmarkState`] is never mutated. A
//!   local copy is advanced and `Some` is returned only once *every* substep
//!   has succeeded, so a mid-frame breakdown leaves the previous frame intact.
//! * **Finiteness guard.** A substep that produces any `NaN`/∞ aborts the whole
//!   frame (`None`) rather than poisoning the next substep.
//!
//! When every `prescribed` entry is `None` this reduces exactly to
//! [`step_newmark_substeps`](super::tet_fem_substep::step_newmark_substeps)
//! with the same `pinned` mask.
//!
//! # Attribution
//!
//! Clean-room implementation built on the crate's own prescribed-Dirichlet
//! Newmark step. No Unreal Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_dirichlet::step_newmark_prescribed;
use super::tet_fem_newmark::NewmarkState;
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_fem_substep::{SubstepParams, SubstepResult};
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

fn all_finite(values: &[Vec3]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn state_is_finite(state: &NewmarkState) -> bool {
    all_finite(&state.positions)
        && all_finite(&state.velocities)
        && all_finite(&state.accelerations)
}

/// Advances a corotational tetrahedral FEM body by one frame split into
/// `params.substeps` equal implicit Newmark substeps while imposing prescribed
/// (inhomogeneous) Dirichlet velocities.
///
/// `state` holds `(x_n, v_n, a_n)`, `external_forces` is the per-vertex
/// external load (held constant across the frame), `pinned` optionally freezes
/// vertices, and `prescribed` has one entry per vertex: `Some(v̄)` drives that
/// vertex to end-of-substep velocity `v̄`, `None` leaves it free. A vertex that
/// is both pinned and prescribed is frozen (pinning wins), matching
/// [`step_newmark_prescribed`].
///
/// Returns the final state plus the per-substep solver diagnostics, or `None`
/// on dimension mismatch (`prescribed.len() != n`), a failed substep, or a
/// non-finite state. On `None` the caller's `state` is left untouched.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a prescribed-motion substepped Newmark frame is parameterised by \
              its mesh, material, mass, rest/current state, external load, the \
              pinned mask and the per-vertex prescribed velocities; bundling \
              them would only hide the explicit per-frame inputs"
)]
pub fn step_newmark_prescribed_substeps(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    state: &NewmarkState,
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    prescribed: &[Option<Vec3>],
    params: &SubstepParams,
) -> Option<SubstepResult> {
    if prescribed.len() != state.positions.len() {
        return None;
    }

    let sub_params = params.substep_params();
    // Advance a local copy so a failed substep leaves the caller untouched.
    let mut current = state.clone();
    let mut reports = Vec::with_capacity(params.substeps as usize);

    for _ in 0..params.substeps {
        let step = step_newmark_prescribed(
            basis,
            tets,
            material,
            mass,
            rest,
            &current,
            external_forces,
            pinned,
            prescribed,
            &sub_params,
        )?;
        current = NewmarkState {
            positions: step.positions,
            velocities: step.velocities,
            accelerations: step.accelerations,
        };
        if !state_is_finite(&current) {
            return None;
        }
        reports.push(step.solver);
    }

    Some(SubstepResult {
        state: current,
        reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_newmark::NewmarkParams;
    use crate::collider::tet_fem_substep::step_newmark_substeps;
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;

    fn mesh() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    fn material() -> IsotropicElasticity {
        IsotropicElasticity::new(2.0e4, 0.3).unwrap()
    }

    fn mass_of(verts: &[Vec3], tets: &[[u32; 4]]) -> LumpedMass {
        build_lumped_mass_from_mesh(verts, tets, &TetMassParams { density: 1200.0 }).unwrap()
    }

    fn gravity(mass: &LumpedMass, n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| Vec3::new(0.0, -9.81, 0.0) * mass.get(3 * i))
            .collect()
    }

    fn rest_state(verts: &[Vec3]) -> NewmarkState {
        let n = verts.len();
        NewmarkState {
            positions: verts.to_vec(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: vec![Vec3::ZERO; n],
        }
    }

    #[test]
    fn rejects_prescribed_length_mismatch() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = SubstepParams::new(3, base).unwrap();

        let short = vec![None; n - 1];
        assert!(step_newmark_prescribed_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &short, &params,
        )
        .is_none());
    }

    #[test]
    fn all_none_matches_homogeneous_substeps() {
        // With no prescribed vertices the driver must reproduce the plain
        // homogeneous substep driver bit-for-bit (both reduce to step_newmark).
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = SubstepParams::new(4, base).unwrap();

        let homogeneous = step_newmark_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();

        let none = vec![None; n];
        let prescribed = step_newmark_prescribed_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &none, &params,
        )
        .unwrap();

        assert_eq!(prescribed.reports.len(), homogeneous.reports.len());
        for i in 0..n {
            assert!(
                (prescribed.state.positions[i] - homogeneous.state.positions[i]).length() < 1e-6
            );
            assert!(
                (prescribed.state.velocities[i] - homogeneous.state.velocities[i]).length() < 1e-6
            );
        }
    }

    #[test]
    fn drives_prescribed_velocity_independent_of_substeps() {
        // Vertex 4 is driven to a fixed velocity. Its end-of-frame velocity must
        // equal v̄ exactly, independent of the substep count.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let vbar = Vec3::new(0.3, -0.2, 0.1);
        let mut prescribed = vec![None; n];
        prescribed[4] = Some(vbar);

        for substeps in [1u32, 2, 5] {
            let params = SubstepParams::new(substeps, base).unwrap();
            let out = step_newmark_prescribed_substeps(
                &basis,
                &tets,
                &mat,
                &mass,
                &verts,
                &state,
                &f,
                None,
                &prescribed,
                &params,
            )
            .unwrap();
            assert_eq!(out.reports.len(), substeps as usize);
            assert!(
                (out.state.velocities[4] - vbar).length() < 1e-5,
                "substeps={substeps}: v4 = {:?}, expected {vbar:?}",
                out.state.velocities[4]
            );
        }
    }

    #[test]
    fn pinning_wins_over_prescription() {
        // A vertex that is both pinned and prescribed must stay frozen.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = SubstepParams::new(3, base).unwrap();

        let mut pinned = vec![false; n];
        pinned[0] = true;
        let mut prescribed = vec![None; n];
        prescribed[0] = Some(Vec3::new(5.0, 5.0, 5.0));

        let out = step_newmark_prescribed_substeps(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &f,
            Some(&pinned),
            &prescribed,
            &params,
        )
        .unwrap();

        assert!((out.state.positions[0] - verts[0]).length() < 1e-6);
        assert!(out.state.velocities[0].length() < 1e-6);
    }

    #[test]
    fn does_not_mutate_caller_state() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let snapshot = state.clone();
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = SubstepParams::new(3, base).unwrap();

        let mut prescribed = vec![None; n];
        prescribed[4] = Some(Vec3::new(0.1, 0.0, 0.0));

        let _ = step_newmark_prescribed_substeps(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &f,
            None,
            &prescribed,
            &params,
        )
        .unwrap();
        assert_eq!(state, snapshot);
    }

    #[test]
    fn deterministic() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = SubstepParams::new(4, base).unwrap();

        let mut prescribed = vec![None; n];
        prescribed[4] = Some(Vec3::new(0.2, 0.1, -0.1));

        let a = step_newmark_prescribed_substeps(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &f,
            None,
            &prescribed,
            &params,
        )
        .unwrap();
        let b = step_newmark_prescribed_substeps(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &f,
            None,
            &prescribed,
            &params,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
