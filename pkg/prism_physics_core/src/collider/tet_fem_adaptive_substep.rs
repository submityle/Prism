//! Error-controlled adaptive substep driver for the implicit corotational
//! Newmark-β integrator via Richardson step-doubling.
//!
//! [`step_newmark_substeps`] advances a frame with a *fixed* number of equal
//! implicit substeps. Choosing that count by hand is a trade-off: too few and
//! stiff dynamics are integrated inaccurately, too many and every frame pays
//! for the worst case. This module removes the guesswork by estimating the
//! temporal discretisation error of the frame and refining the substep count
//! until the error falls under a caller-supplied tolerance.
//!
//! # Method
//!
//! The driver uses classic *step doubling* (Richardson extrapolation of the
//! error, not of the solution):
//!
//! 1. Integrate the whole frame with `n` equal substeps → a *coarse* state.
//! 2. Integrate the same frame with `2n` substeps → a *fine* state.
//! 3. Estimate the error as the largest per-vertex position discrepancy
//!    `max_i ‖x_coarse,i − x_fine,i‖`. For a convergent integrator the coarse
//!    and fine solutions bracket the exact one, so their difference is a
//!    conservative, cheap error proxy.
//! 4. If the estimate is within tolerance (or the substep budget is spent)
//!    accept the *fine* state — it is the more accurate of the two. Otherwise
//!    the fine state becomes the new coarse state, the count doubles, and the
//!    loop repeats.
//!
//! Because the substep count strictly increases and is clamped to
//! `max_substeps`, the loop always terminates. The returned state is never the
//! coarse one, so refinement never *degrades* the result it returns.
//!
//! # Guarantees inherited from the fixed driver
//!
//! Each candidate frame is produced by [`step_newmark_substeps`], so the whole
//! driver is all-or-nothing: the caller's [`NewmarkState`] is never mutated and
//! a solver breakdown or non-finite candidate aborts the entire frame
//! (`None`). The external load is held constant across the frame.
//!
//! # Attribution
//!
//! Clean-room implementation built on the crate's own fixed-rate substep
//! driver. No Unreal Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_cg::CgReport;
use super::tet_fem_newmark::{NewmarkParams, NewmarkState};
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_fem_substep::{step_newmark_substeps, SubstepParams};
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// Parameters controlling an error-controlled adaptive Newmark frame.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveSubstepParams {
    /// Base Newmark parameters. `base.dt` is the *frame* step; each candidate
    /// subdivides it into equal substeps while reusing `beta`, `gamma`, the
    /// linear-solver settings and the Rayleigh damping unchanged.
    pub base: NewmarkParams,
    /// Smallest substep count to try first; must be at least one.
    pub min_substeps: u32,
    /// Largest substep count the driver may refine to; must be strictly
    /// greater than `min_substeps` so the step-doubling loop has room to run.
    pub max_substeps: u32,
    /// Acceptance threshold on the per-vertex position error estimate, in the
    /// same length units as the mesh. Must be finite and strictly positive.
    pub tolerance: f32,
}

impl AdaptiveSubstepParams {
    /// Builds adaptive parameters.
    ///
    /// Returns `None` when the configuration cannot drive a step-doubling loop:
    ///
    /// * `min_substeps == 0` (a frame needs at least one substep),
    /// * `max_substeps <= min_substeps` (adaptive refinement needs headroom to
    ///   double — for genuinely fixed stepping use [`step_newmark_substeps`]),
    /// * `tolerance` is non-finite or not strictly positive,
    /// * `base.dt` is non-finite or not strictly positive.
    #[must_use]
    pub fn new(
        base: NewmarkParams,
        min_substeps: u32,
        max_substeps: u32,
        tolerance: f32,
    ) -> Option<Self> {
        if min_substeps == 0 || max_substeps <= min_substeps {
            return None;
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return None;
        }
        if !base.dt.is_finite() || base.dt <= 0.0 {
            return None;
        }
        Some(Self {
            base,
            min_substeps,
            max_substeps,
            tolerance,
        })
    }
}

/// Result of an error-controlled adaptive Newmark frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveSubstepResult {
    /// Accepted final state `(x_{n+1}, v_{n+1}, a_{n+1})`. Always the finest
    /// candidate computed, hence the most accurate.
    pub state: NewmarkState,
    /// Substep count used for the accepted (fine) candidate.
    pub substeps: u32,
    /// Final per-vertex position error estimate between the last coarse and
    /// fine candidates.
    pub error_estimate: f32,
    /// Whether the error estimate met the tolerance. `false` means the driver
    /// hit `max_substeps` before converging and returned the best effort.
    pub converged: bool,
    /// Per-substep linear-solve diagnostics of the accepted (fine) candidate,
    /// in execution order.
    pub reports: Vec<CgReport>,
}

/// Largest per-vertex position discrepancy between two states of equal length.
fn max_position_error(a: &NewmarkState, b: &NewmarkState) -> f32 {
    a.positions
        .iter()
        .zip(&b.positions)
        .map(|(pa, pb)| (*pa - *pb).length())
        .fold(0.0_f32, f32::max)
}

/// Advances a corotational tetrahedral FEM body by one frame, automatically
/// choosing the substep count so the estimated temporal error is within
/// `params.tolerance`.
///
/// `state` holds `(x_n, v_n, a_n)`, `external_forces` is the per-vertex
/// external load (held constant across the frame), and `pinned` optionally
/// freezes vertices. Returns the accepted (finest) state, the substep count
/// used, the error estimate, whether it converged and the fine candidate's
/// solver diagnostics — or `None` if any candidate frame fails to solve or
/// produces a non-finite state. On `None` the caller's `state` is untouched.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "an adaptive Newmark frame is parameterised by its mesh, material, \
              mass, rest/current state, external load and boundary mask; bundling \
              them would only hide the explicit per-frame inputs"
)]
pub fn step_newmark_adaptive(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    state: &NewmarkState,
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    params: &AdaptiveSubstepParams,
) -> Option<AdaptiveSubstepResult> {
    let run = |count: u32| {
        let sub = SubstepParams::new(count, params.base)?;
        step_newmark_substeps(
            basis,
            tets,
            material,
            mass,
            rest,
            state,
            external_forces,
            pinned,
            &sub,
        )
    };

    let mut coarse_n = params.min_substeps;
    let mut coarse = run(coarse_n)?;

    loop {
        let fine_n = (coarse_n * 2).min(params.max_substeps);
        let fine = run(fine_n)?;
        let error_estimate = max_position_error(&coarse.state, &fine.state);
        let converged = error_estimate <= params.tolerance;

        if converged || fine_n >= params.max_substeps {
            return Some(AdaptiveSubstepResult {
                state: fine.state,
                substeps: fine_n,
                error_estimate,
                converged,
                reports: fine.reports,
            });
        }

        coarse = fine;
        coarse_n = fine_n;
    }
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

    fn rest_state(verts: &[Vec3]) -> NewmarkState {
        let n = verts.len();
        NewmarkState {
            positions: verts.to_vec(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: vec![Vec3::ZERO; n],
        }
    }

    /// A stiff dynamic scenario where the substep count genuinely matters:
    /// vertex 0 is pinned, the remaining vertices are pulled by a strong
    /// constant load on a stiff material, started from rest with the consistent
    /// initial acceleration. This excites fast internal oscillations whose
    /// implicit integration is substep-count sensitive.
    fn stiff_scenario() -> (
        Vec<Vec3>,
        Vec<[u32; 4]>,
        TetFemBasis,
        IsotropicElasticity,
        LumpedMass,
        NewmarkState,
        Vec<Vec3>,
        Vec<bool>,
        f32,
    ) {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let h = 1.0 / 60.0;

        let mut pinned = vec![false; n];
        pinned[0] = true;

        // Strong pull on the free vertices only.
        let mut forces = vec![Vec3::ZERO; n];
        for (i, f) in forces.iter_mut().enumerate() {
            if !pinned[i] {
                *f = Vec3::new(0.0, -4.0e3, 0.0) * mass.get(3 * i);
            }
        }

        let v0 = vec![Vec3::ZERO; n];
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v0,
            &forces,
            Some(&pinned),
            &RayleighDamping::NONE,
        )
        .unwrap();
        let state = NewmarkState {
            positions: verts.clone(),
            velocities: v0,
            accelerations: a0,
        };
        (verts, tets, basis, mat, mass, state, forces, pinned, h)
    }

    #[test]
    fn rejects_invalid_params() {
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        // zero minimum
        assert!(AdaptiveSubstepParams::new(base, 0, 4, 1e-4).is_none());
        // no headroom to double
        assert!(AdaptiveSubstepParams::new(base, 4, 4, 1e-4).is_none());
        assert!(AdaptiveSubstepParams::new(base, 5, 4, 1e-4).is_none());
        // bad tolerance
        assert!(AdaptiveSubstepParams::new(base, 1, 4, 0.0).is_none());
        assert!(AdaptiveSubstepParams::new(base, 1, 4, -1.0).is_none());
        assert!(AdaptiveSubstepParams::new(base, 1, 4, f32::NAN).is_none());
        // valid
        assert!(AdaptiveSubstepParams::new(base, 1, 4, 1e-4).is_some());
    }

    #[test]
    fn converges_within_max() {
        let (verts, tets, basis, mat, mass, state, forces, pinned, h) = stiff_scenario();
        let base = NewmarkParams::average_acceleration(h).unwrap();
        // Generous tolerance and budget: the driver should converge below max.
        let params = AdaptiveSubstepParams::new(base, 1, 64, 5e-3).unwrap();

        let out = step_newmark_adaptive(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &forces,
            Some(&pinned),
            &params,
        )
        .unwrap();

        assert!(out.converged, "error = {}", out.error_estimate);
        assert!(out.error_estimate <= params.tolerance);
        assert!(out.substeps > params.min_substeps);
        assert!(out.substeps <= params.max_substeps);
        assert_eq!(out.reports.len(), out.substeps as usize);
        // Pinned vertex must not have moved.
        assert!((out.state.positions[0] - verts[0]).length() < 1e-6);
    }

    #[test]
    fn tiny_tolerance_hits_max() {
        let (verts, tets, basis, mat, mass, state, forces, pinned, h) = stiff_scenario();
        let base = NewmarkParams::average_acceleration(h).unwrap();
        // Unreachable tolerance: must spend the whole budget and report failure.
        let params = AdaptiveSubstepParams::new(base, 1, 8, 1e-12).unwrap();

        let out = step_newmark_adaptive(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &forces,
            Some(&pinned),
            &params,
        )
        .unwrap();

        assert!(!out.converged);
        assert_eq!(out.substeps, params.max_substeps);
        assert_eq!(out.reports.len(), out.substeps as usize);
    }

    #[test]
    fn rest_state_is_near_stationary() {
        // From rest with no external load the exact solution does not move, so
        // every substep count agrees and the error estimate is ~0.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let state = rest_state(&verts);
        let forces = vec![Vec3::ZERO; n];
        let base = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let params = AdaptiveSubstepParams::new(base, 1, 16, 1e-6).unwrap();

        let out = step_newmark_adaptive(
            &basis, &tets, &mat, &mass, &verts, &state, &forces, None, &params,
        )
        .unwrap();

        assert!(out.converged);
        assert!(out.error_estimate < 1e-6);
        // State essentially unchanged.
        for i in 0..n {
            assert!((out.state.positions[i] - verts[i]).length() < 1e-5);
            assert!(out.state.velocities[i].length() < 1e-5);
        }
        // Converged at the minimum count (coarse vs its double already agree).
        assert_eq!(out.substeps, 2 * params.min_substeps);
    }

    #[test]
    fn does_not_mutate_caller_state() {
        let (verts, tets, basis, mat, mass, state, forces, pinned, h) = stiff_scenario();
        let base = NewmarkParams::average_acceleration(h).unwrap();
        let params = AdaptiveSubstepParams::new(base, 1, 16, 5e-3).unwrap();
        let snapshot = state.clone();

        let _ = step_newmark_adaptive(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &forces,
            Some(&pinned),
            &params,
        )
        .unwrap();

        assert_eq!(state, snapshot);
    }

    #[test]
    fn deterministic() {
        let (verts, tets, basis, mat, mass, state, forces, pinned, h) = stiff_scenario();
        let base = NewmarkParams::average_acceleration(h).unwrap();
        let params = AdaptiveSubstepParams::new(base, 1, 16, 5e-3).unwrap();

        let a = step_newmark_adaptive(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &forces,
            Some(&pinned),
            &params,
        )
        .unwrap();
        let b = step_newmark_adaptive(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &forces,
            Some(&pinned),
            &params,
        )
        .unwrap();

        assert_eq!(a, b);
    }
}
