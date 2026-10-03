//! Fixed-rate multi-substep driver for the implicit corotational Newmark-β
//! integrator with all-or-nothing rollback.
//!
//! A single [`step_newmark`] advances the body by the full frame step `h`. For
//! stiff materials, large external loads or fast prescribed motion, splitting
//! the frame into several smaller implicit substeps improves accuracy and
//! robustness without changing the time integrator. This module provides that
//! substepping loop with two guarantees a bare loop does not give:
//!
//! * **All-or-nothing.** The caller's [`NewmarkState`] is never mutated. The
//!   loop advances a local copy and only returns `Some` once *every* substep
//!   has succeeded, so a mid-frame solver breakdown leaves the simulation on
//!   the previous frame instead of a half-advanced, inconsistent state.
//! * **Finiteness guard.** After each substep the produced positions,
//!   velocities and accelerations are checked for `NaN`/∞. A non-finite
//!   substep aborts the whole frame (returns `None`) rather than propagating
//!   poison into the next substep.
//!
//! The external load is held constant across the substeps of a frame, matching
//! the usual explicit/implicit frame-load convention.
//!
//! # Attribution
//!
//! Clean-room implementation built on the crate's own Newmark step. No Unreal
//! Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_cg::CgReport;
use super::tet_fem_newmark::{step_newmark, NewmarkParams, NewmarkState};
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// Parameters controlling a fixed-rate substepped Newmark frame.
#[derive(Clone, Copy, Debug)]
pub struct SubstepParams {
    /// Number of equal implicit substeps per frame; must be at least one.
    pub substeps: u32,
    /// Base Newmark parameters. `base.dt` is the *frame* step; each substep
    /// uses `base.dt / substeps` while reusing `beta`, `gamma`, the linear
    /// solver settings and the Rayleigh damping unchanged.
    pub base: NewmarkParams,
}

impl SubstepParams {
    /// Builds substep parameters, returning `None` when `substeps == 0`.
    #[must_use]
    pub fn new(substeps: u32, base: NewmarkParams) -> Option<Self> {
        if substeps == 0 {
            return None;
        }
        Some(Self { substeps, base })
    }

    /// The per-substep Newmark parameters (`dt` divided by the substep count).
    #[must_use]
    pub fn substep_params(&self) -> NewmarkParams {
        let mut sub = self.base;
        sub.dt = self.base.dt / self.substeps as f32;
        sub
    }
}

/// Result of a substepped Newmark frame.
#[derive(Clone, Debug, PartialEq)]
pub struct SubstepResult {
    /// Final state `(x_{n+1}, v_{n+1}, a_{n+1})` after all substeps.
    pub state: NewmarkState,
    /// Per-substep linear-solve diagnostics, in execution order.
    pub reports: Vec<CgReport>,
}

fn all_finite(values: &[Vec3]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn state_is_finite(state: &NewmarkState) -> bool {
    all_finite(&state.positions)
        && all_finite(&state.velocities)
        && all_finite(&state.accelerations)
}

/// Advances a corotational tetrahedral FEM body by one frame split into
/// `params.substeps` equal implicit Newmark substeps.
///
/// `state` holds `(x_n, v_n, a_n)`, `external_forces` is the per-vertex
/// external load (held constant across the frame), and `pinned` optionally
/// freezes vertices. Returns the final state plus the per-substep solver
/// diagnostics, or `None` if any substep fails to assemble/solve or produces a
/// non-finite state. On `None` the caller's `state` is left untouched.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a substepped Newmark frame is parameterised by its mesh, material, \
              mass, rest/current state, external load and boundary mask; bundling \
              them would only hide the explicit per-frame inputs"
)]
pub fn step_newmark_substeps(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    state: &NewmarkState,
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    params: &SubstepParams,
) -> Option<SubstepResult> {
    let sub_params = params.substep_params();
    // Advance a local copy so a failed substep leaves the caller untouched.
    let mut current = state.clone();
    let mut reports = Vec::with_capacity(params.substeps as usize);

    for _ in 0..params.substeps {
        let step = step_newmark(
            basis,
            tets,
            material,
            mass,
            rest,
            &current,
            external_forces,
            pinned,
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
    use crate::collider::tet_fem_integrator::RayleighDamping;
    use crate::collider::tet_fem_newmark::initial_acceleration;
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
    fn rejects_zero_substeps() {
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        assert!(SubstepParams::new(0, params).is_none());
        assert!(SubstepParams::new(1, params).is_some());
    }

    #[test]
    fn substep_params_divide_the_frame_step() {
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let p = SubstepParams::new(4, base).unwrap();
        let sub = p.substep_params();
        assert!((sub.dt - base.dt / 4.0).abs() < 1e-9);
        assert_eq!((sub.beta, sub.gamma), (base.beta, base.gamma));
    }

    #[test]
    fn single_substep_matches_direct_step() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let state = rest_state(&verts);
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();

        let direct =
            step_newmark(&basis, &tets, &mat, &mass, &verts, &state, &f, None, &base).unwrap();
        let params = SubstepParams::new(1, base).unwrap();
        let subbed = step_newmark_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();

        assert_eq!(subbed.reports.len(), 1);
        for i in 0..n {
            assert!((subbed.state.positions[i] - direct.positions[i]).length() < 1e-6);
            assert!((subbed.state.velocities[i] - direct.velocities[i]).length() < 1e-6);
        }
    }

    #[test]
    fn free_fall_velocity_is_substep_independent() {
        // Uniform gravity is in the null space of the corotational tangent, so a
        // body seeded with the consistent initial acceleration a_0 = g free-falls
        // rigidly: after a full frame the velocity is h·g regardless of how many
        // equal substeps the frame is split into.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let h = 1.0 / 60.0;
        let base = NewmarkParams::average_acceleration(h).unwrap();

        let v0 = vec![Vec3::ZERO; n];
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v0,
            &f,
            None,
            &RayleighDamping::NONE,
        )
        .unwrap();
        let state = NewmarkState {
            positions: verts.clone(),
            velocities: v0,
            accelerations: a0,
        };

        let expected = Vec3::new(0.0, -9.81 * h, 0.0);
        for substeps in [1u32, 3, 7] {
            let params = SubstepParams::new(substeps, base).unwrap();
            let out = step_newmark_substeps(
                &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
            )
            .unwrap();
            assert_eq!(out.reports.len(), substeps as usize);
            for v in &out.state.velocities {
                assert!(
                    (*v - expected).length() < 2e-3,
                    "substeps={substeps}: v = {v:?}, expected {expected:?}"
                );
            }
        }
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

        let _ = step_newmark_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
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

        let a = step_newmark_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();
        let b = step_newmark_substeps(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
