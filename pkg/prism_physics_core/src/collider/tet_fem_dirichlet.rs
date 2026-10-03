//! Prescribed (inhomogeneous) Dirichlet boundary conditions for the implicit
//! corotational Newmark-β integrator.
//!
//! The plain [`step_newmark`](super::tet_fem_newmark::step_newmark) supports
//! only *homogeneous* Dirichlet conditions: a `pinned` vertex is frozen in
//! place. Animation-driven simulation also needs *inhomogeneous* conditions,
//! where a vertex is forced to follow a prescribed end-of-step velocity
//! `v̄` (a kinematic anchor dragged by a skeleton, a scripted handle, a
//! collision-resolved contact, …). This module adds that capability without
//! duplicating the assembly: it reuses the shared
//! [`NewmarkSystem`](super::tet_fem_newmark::NewmarkSystem) and imposes the
//! prescribed motion by RHS lifting.
//!
//! # Lifting
//!
//! One implicit Newmark step solves `A a = b` for the end-of-step
//! acceleration `a`, where `A = c_m M + c_k K` is symmetric positive definite
//! and `b` is the assembled right-hand side. Partition the degrees of freedom
//! into *free* (`f`) and *known* (`k`, the driven and pinned vertices):
//!
//! ```text
//! | A_ff  A_fk | | a_f |   | b_f |
//! | A_kf  A_kk | | a_k | = | b_k |
//! ```
//!
//! With `a_k` fixed to the known accelerations, the free block reads
//! `A_ff a_f = b_f - A_fk a_k`. Rather than extracting sub-blocks we lift the
//! known part through the *full* operator and reuse the existing filtered SPD
//! solver: form `b' = b - A a_k`, zero the known rows of `b'`, and solve
//! `A_ff a_f = b'_f`. The known rows of `A a_k` are discarded by the filter,
//! so only the coupling term `A_fk a_k` survives on the free rows — exactly
//! the condensation above. The final acceleration is `a = a_f + a_k`.
//!
//! For a vertex driven to velocity `v̄`, the Newmark velocity update
//! `v_{n+1} = v* + γh·a` inverts to the known acceleration
//! `ā = (v̄ − v*) / (γh)`, which reproduces `v_{n+1} = v̄` exactly.
//!
//! Pinned vertices take priority over prescribed ones: a vertex that is both
//! pinned and prescribed is frozen (known acceleration zero), matching the
//! homogeneous convention of [`step_newmark`](super::tet_fem_newmark::step_newmark).
//!
//! # Attribution
//!
//! Clean-room implementation from the standard Newmark-β time-integration and
//! Dirichlet static-condensation literature. No Unreal Engine source or
//! derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_implicit::{solve_filtered_spd, unflatten};
use super::tet_fem_newmark::{
    assemble_newmark_system, NewmarkParams, NewmarkState, NewmarkStepResult,
};
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// Advances a corotational tetrahedral FEM body by one implicit Newmark-β step
/// while imposing prescribed (inhomogeneous) Dirichlet velocities.
///
/// `state` holds `(x_n, v_n, a_n)` and `external_forces` is the per-vertex
/// external load in Newtons. `pinned` optionally freezes vertices as in
/// [`step_newmark`](super::tet_fem_newmark::step_newmark). `prescribed` has one
/// entry per vertex: `Some(v̄)` drives that vertex so that its end-of-step
/// velocity equals `v̄` exactly, while `None` leaves it free. A vertex that is
/// both pinned and prescribed is frozen (pinning wins).
///
/// Returns the updated `(x_{n+1}, v_{n+1}, a_{n+1})` and the linear-solve
/// diagnostics, or `None` on dimension mismatch (`prescribed.len() != n`),
/// failed assembly, singular preconditioner, or solver breakdown.
///
/// When every entry of `prescribed` is `None` this reduces exactly to
/// [`step_newmark`](super::tet_fem_newmark::step_newmark) with the same
/// `pinned` mask.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a prescribed-motion Newmark step is parameterised by its mesh, \
              material, mass, rest/current state, external load, the pinned mask \
              and the per-vertex prescribed velocities; bundling them would only \
              hide the explicit per-step inputs"
)]
pub fn step_newmark_prescribed(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    state: &NewmarkState,
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    prescribed: &[Option<Vec3>],
    params: &NewmarkParams,
) -> Option<NewmarkStepResult> {
    let n = state.positions.len();
    if prescribed.len() != n {
        return None;
    }

    let sys = assemble_newmark_system(
        basis,
        tets,
        material,
        mass,
        rest,
        state,
        external_forces,
        pinned,
        params,
    )?;

    // Known end-of-step acceleration per degree of freedom. Zero on free and
    // pinned vertices; `ā = (v̄ − v*) / (γh)` on driven ones.
    let gamma_h = sys.gamma * sys.h;
    let mut a_known = vec![0.0_f32; 3 * n];
    // Rows held fixed by the solve: pinned ∪ driven.
    let mut combined = vec![false; n];
    let mut any_driven = false;

    for i in 0..n {
        if pinned.is_some_and(|m| m[i]) {
            // Pinning wins over any prescribed velocity: frozen, ā = 0.
            combined[i] = true;
            continue;
        }
        if let Some(v_bar) = prescribed[i] {
            if gamma_h == 0.0 {
                // γh > 0 for validated params, but guard against a degenerate
                // division rather than emitting a non-finite acceleration.
                return None;
            }
            let a_bar = (v_bar - sys.v_star[i]) / (gamma_h as f32);
            a_known[3 * i] = a_bar.x;
            a_known[3 * i + 1] = a_bar.y;
            a_known[3 * i + 2] = a_bar.z;
            combined[i] = true;
            any_driven = true;
        }
    }

    // Lift the known accelerations through the full operator `A = c_m M + c_k K`
    // so the filtered solve condenses the coupling onto the free rows.
    let mut rhs = sys.rhs.clone();
    if any_driven {
        let m_ak = mass.apply(&a_known)?;
        let k_ak = sys.stiffness.apply(&a_known)?;
        for i in 0..3 * n {
            rhs[i] -= (sys.c_m * f64::from(m_ak[i]) + sys.c_k * f64::from(k_ak[i])) as f32;
        }
    }

    let (a_free_flat, solver) = solve_filtered_spd(
        &sys.stiffness,
        mass,
        sys.c_m,
        sys.c_k,
        &rhs,
        Some(&combined),
        params.preconditioner,
        &params.cg,
    )?;

    // a = a_f + a_k: free rows carry the solved free acceleration (known rows
    // zeroed by the filter), known rows carry the prescribed acceleration.
    let mut a_next_flat = a_free_flat;
    for i in 0..3 * n {
        a_next_flat[i] += a_known[i];
    }
    let a_next = unflatten(&a_next_flat);

    let x = &state.positions;
    let c_v = (sys.gamma * sys.h) as f32;
    let c_x = (sys.beta * sys.h * sys.h) as f32;
    let mut positions_next = Vec::with_capacity(n);
    let mut velocities_next = Vec::with_capacity(n);
    let mut accelerations_next = Vec::with_capacity(n);
    for i in 0..n {
        if pinned.is_some_and(|m| m[i]) {
            positions_next.push(x[i]);
            velocities_next.push(Vec3::ZERO);
            accelerations_next.push(Vec3::ZERO);
        } else {
            velocities_next.push(sys.v_star[i] + c_v * a_next[i]);
            positions_next.push(x[i] + sys.d_star[i] + c_x * a_next[i]);
            accelerations_next.push(a_next[i]);
        }
    }

    Some(NewmarkStepResult {
        positions: positions_next,
        velocities: velocities_next,
        accelerations: accelerations_next,
        solver,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_newmark::step_newmark;
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
    fn prescribed_velocity_matched_exactly() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let v_bar = Vec3::new(0.3, -0.2, 0.1);
        let mut prescribed = vec![None; n];
        prescribed[4] = Some(v_bar);

        let out = step_newmark_prescribed(
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

        assert!(
            (out.velocities[4] - v_bar).length() < 1e-5,
            "driven velocity = {:?}, expected {v_bar:?}",
            out.velocities[4]
        );
    }

    #[test]
    fn all_none_reproduces_step_newmark() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let prescribed = vec![None; n];
        let driven = step_newmark_prescribed(
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
        let plain = step_newmark(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();

        for i in 0..n {
            assert!(
                (driven.positions[i] - plain.positions[i]).length() < 1e-6,
                "position {i} diverged"
            );
            assert!(
                (driven.velocities[i] - plain.velocities[i]).length() < 1e-6,
                "velocity {i} diverged"
            );
            assert!(
                (driven.accelerations[i] - plain.accelerations[i]).length() < 1e-6,
                "acceleration {i} diverged"
            );
        }
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let prescribed = vec![None; n - 1];
        assert!(step_newmark_prescribed(
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
        .is_none());
    }

    #[test]
    fn pinned_takes_priority_over_prescribed() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let mut pinned = vec![false; n];
        pinned[0] = true;
        let mut prescribed = vec![None; n];
        // Vertex 0 is pinned AND prescribed: pinning must win (frozen).
        prescribed[0] = Some(Vec3::new(5.0, 5.0, 5.0));

        let out = step_newmark_prescribed(
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

        assert!((out.positions[0] - verts[0]).length() < 1e-7);
        assert!(out.velocities[0].length() < 1e-7);
        assert!(out.accelerations[0].length() < 1e-7);
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
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let mut prescribed = vec![None; n];
        prescribed[3] = Some(Vec3::new(-0.1, 0.4, 0.2));

        let a = step_newmark_prescribed(
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
        let b = step_newmark_prescribed(
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

        for i in 0..n {
            assert_eq!(a.positions[i], b.positions[i]);
            assert_eq!(a.velocities[i], b.velocities[i]);
            assert_eq!(a.accelerations[i], b.accelerations[i]);
        }
    }
}
