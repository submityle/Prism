//! Implicit Newmark-β corotational FEM time integrator.
//!
//! Backward Euler ([`step_implicit_corotational`](super::tet_fem_integrator::step_implicit_corotational))
//! is robust but strongly dissipative: it damps every mode, so bouncy or
//! oscillatory soft bodies lose energy artificially. The Newmark-β family is the
//! standard structural-dynamics alternative that lets the caller trade accuracy
//! against numerical damping. With `γ = 1/2` the scheme is second-order and
//! energy-conserving; the `β = 1/4` *average-acceleration* choice is additionally
//! unconditionally stable, and `β = 1/6` gives the *linear-acceleration* scheme.
//!
//! # Formulation
//!
//! For a body with lumped mass `M`, corotational tangent `K`, restoring force
//! `f_r(x)`, Rayleigh damping `C = η_M M + η_K K` and external force `f_ext`,
//! Newmark expresses the end-of-step position and velocity in terms of the
//! unknown end-of-step acceleration `a_{n+1}`:
//!
//! ```text
//!   x_{n+1} = x_n + h v_n + h²[(½ − β) a_n + β a_{n+1}]
//!   v_{n+1} = v_n + h[(1 − γ) a_n + γ a_{n+1}]
//! ```
//!
//! Substituting into the equation of motion `M a + C v = f_r(x) + f_ext` and
//! linearising the restoring force about `x_n`
//! (`f_r(x_{n+1}) ≈ f_r(x_n) − K (x_{n+1} − x_n)`) gives a symmetric
//! positive-definite system in `a_{n+1}`:
//!
//! ```text
//!   [(1 + γ h η_M) M + (γ h η_K + β h²) K] a_{n+1}
//!       = f_ext + f_r(x_n) − C v* − K d*
//!   v* = v_n + h(1 − γ) a_n,   d* = h v_n + h²(½ − β) a_n
//! ```
//!
//! which is solved with the shared filtered conjugate-gradient driver
//! ([`solve_filtered_spd`]). Pinned vertices use the same filtered-CG Dirichlet
//! construction as the Euler integrator (`a = v = 0`, position frozen).
//!
//! Because the update needs the previous acceleration `a_n`, the integrator
//! threads an acceleration through its [`NewmarkState`]; [`initial_acceleration`]
//! seeds a consistent `a_0` from `M a_0 = f_ext + f_r(x_0) − C v_0`.
//!
//! Newmark-β, Rayleigh damping and corotational linearisation are standard
//! finite-element constructions. This file contains no Unreal Engine source or
//! derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_cg::{CgParams, CgReport};
use super::tet_fem_corotational_assembly::{
    assemble_corotational_forces, assemble_corotational_stiffness,
};
use super::tet_fem_implicit::{flatten, solve_filtered_spd, unflatten, FemPreconditioner};
use super::tet_fem_integrator::RayleighDamping;
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// Parameters controlling one Newmark-β step.
#[derive(Clone, Copy, Debug)]
pub struct NewmarkParams {
    /// Time step `h > 0` in seconds.
    pub dt: f32,
    /// Newmark `β ∈ (0, ½]`: weights the end-of-step acceleration in the
    /// position update. `¼` is average-acceleration, `⅙` linear-acceleration.
    pub beta: f32,
    /// Newmark `γ ∈ [0, 1]`: weights the end-of-step acceleration in the
    /// velocity update. `½` is second-order and non-dissipative; `> ½` adds
    /// numerical damping.
    pub gamma: f32,
    /// Conjugate-gradient iteration cap and tolerance.
    pub cg: CgParams,
    /// Preconditioner for the linear solve.
    pub preconditioner: FemPreconditioner,
    /// Rayleigh damping coefficients `C = α M + β K`.
    pub damping: RayleighDamping,
}

impl NewmarkParams {
    /// Builds parameters, returning `None` unless `dt > 0`, `β ∈ (0, ½]` and
    /// `γ ∈ [0, 1]` are all finite. Defaults to scalar Jacobi and no damping.
    #[must_use]
    pub fn new(dt: f32, beta: f32, gamma: f32) -> Option<Self> {
        if !(dt.is_finite() && beta.is_finite() && gamma.is_finite())
            || dt <= 0.0
            || beta <= 0.0
            || beta > 0.5
            || !(0.0..=1.0).contains(&gamma)
        {
            return None;
        }
        Some(Self {
            dt,
            beta,
            gamma,
            cg: CgParams::default(),
            preconditioner: FemPreconditioner::default(),
            damping: RayleighDamping::NONE,
        })
    }

    /// The unconditionally stable, second-order *average-acceleration* scheme
    /// (`β = ¼`, `γ = ½`): the usual default for implicit dynamics.
    #[must_use]
    pub fn average_acceleration(dt: f32) -> Option<Self> {
        Self::new(dt, 0.25, 0.5)
    }

    /// The *linear-acceleration* scheme (`β = ⅙`, `γ = ½`): more accurate than
    /// average-acceleration but only conditionally stable.
    #[must_use]
    pub fn linear_acceleration(dt: f32) -> Option<Self> {
        Self::new(dt, 1.0 / 6.0, 0.5)
    }
}

/// The kinematic state carried between Newmark steps.
///
/// Unlike backward Euler, Newmark needs the previous acceleration `a_n`, so the
/// state bundles positions, velocities and accelerations together.
#[derive(Clone, Debug, PartialEq)]
pub struct NewmarkState {
    /// Positions `x_n`.
    pub positions: Vec<Vec3>,
    /// Velocities `v_n`.
    pub velocities: Vec<Vec3>,
    /// Accelerations `a_n`.
    pub accelerations: Vec<Vec3>,
}

/// Result of a single Newmark-β step.
#[derive(Clone, Debug, PartialEq)]
pub struct NewmarkStepResult {
    /// Updated positions `x_{n+1}`.
    pub positions: Vec<Vec3>,
    /// Updated velocities `v_{n+1}` (zero for pinned vertices).
    pub velocities: Vec<Vec3>,
    /// Updated accelerations `a_{n+1}` (zero for pinned vertices).
    pub accelerations: Vec<Vec3>,
    /// Diagnostics from the linear solve.
    pub solver: CgReport,
}

/// Weighted sum `a + s · b` of two equal-length vector slices.
fn axpy(a: &[Vec3], s: f32, b: &[Vec3]) -> Vec<Vec3> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai + s * bi)
        .collect()
}

/// Computes a consistent initial acceleration `a_0` from
/// `M a_0 = f_ext + f_r(x_0) − C v_0`, using the diagonal mass inverse.
///
/// Pinned vertices get `a_0 = 0`. Returns `None` on dimension mismatch or if the
/// corotational assemblies fail.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a consistent initial acceleration is defined by the full dynamic \
              configuration — mesh, material, mass, rest/current/velocity state, \
              external load, boundary mask and damping; bundling them would hide \
              the explicit inputs"
)]
pub fn initial_acceleration(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    positions: &[Vec3],
    velocities: &[Vec3],
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    damping: &RayleighDamping,
) -> Option<Vec<Vec3>> {
    let n = positions.len();
    if n == 0
        || rest.len() != n
        || velocities.len() != n
        || external_forces.len() != n
        || mass.n_dofs() != 3 * n
        || pinned.is_some_and(|m| m.len() != n)
    {
        return None;
    }
    let stiffness = assemble_corotational_stiffness(basis, tets, material, positions, n)?;
    let restoring = assemble_corotational_forces(basis, tets, material, rest, positions, n)?;

    let v_flat = flatten(velocities);
    let m_v = mass.apply(&v_flat)?;
    let k_v = stiffness.apply(&v_flat)?;
    let eta_m = f64::from(damping.alpha);
    let eta_k = f64::from(damping.beta);

    let f_r = flatten(&restoring);
    let f_ext = flatten(external_forces);
    let mut rhs = Vec::with_capacity(3 * n);
    for i in 0..3 * n {
        let value = f64::from(f_ext[i]) + f64::from(f_r[i])
            - eta_m * f64::from(m_v[i])
            - eta_k * f64::from(k_v[i]);
        rhs.push(value as f32);
    }
    zero_pinned(&mut rhs, pinned);
    let mut a_flat = mass.apply_inverse(&rhs)?;
    zero_pinned(&mut a_flat, pinned);
    Some(unflatten(&a_flat))
}

/// Zeroes the three degrees of freedom of every pinned vertex, in place.
fn zero_pinned(values: &mut [f32], pinned: Option<&[bool]>) {
    if let Some(mask) = pinned {
        for (v, &is_pinned) in mask.iter().enumerate() {
            if is_pinned {
                values[3 * v] = 0.0;
                values[3 * v + 1] = 0.0;
                values[3 * v + 2] = 0.0;
            }
        }
    }
}

/// Advances a corotational tetrahedral FEM body by one implicit Newmark-β step.
///
/// `state` holds `(x_n, v_n, a_n)`; `external_forces` is the per-vertex external
/// load in Newtons; `pinned` optionally freezes vertices. Returns the updated
/// `(x_{n+1}, v_{n+1}, a_{n+1})` and the linear-solve diagnostics, or `None` on
/// dimension mismatch, failed assembly, singular preconditioner, or solver
/// breakdown.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "an implicit Newmark step is parameterised by its mesh, material, \
              mass, rest/current state, external load and boundary mask; bundling \
              them would only hide the explicit per-step inputs"
)]
pub fn step_newmark(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    state: &NewmarkState,
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    params: &NewmarkParams,
) -> Option<NewmarkStepResult> {
    let n = state.positions.len();
    if n == 0
        || rest.len() != n
        || state.velocities.len() != n
        || state.accelerations.len() != n
        || external_forces.len() != n
        || mass.n_dofs() != 3 * n
        || pinned.is_some_and(|m| m.len() != n)
        || !params.dt.is_finite()
        || params.dt <= 0.0
    {
        return None;
    }

    let h = f64::from(params.dt);
    let beta = f64::from(params.beta);
    let gamma = f64::from(params.gamma);
    let eta_m = f64::from(params.damping.alpha);
    let eta_k = f64::from(params.damping.beta);

    let c_m = 1.0 + gamma * h * eta_m;
    let c_k = gamma * h * eta_k + beta * h * h;

    let x = &state.positions;
    let v = &state.velocities;
    let a = &state.accelerations;

    // Predictors v* = v_n + h(1-γ) a_n and d* = h v_n + h²(½-β) a_n.
    let c_vstar = (h * (1.0 - gamma)) as f32;
    let v_star = axpy(v, c_vstar, a);
    let c_dstar_a = (h * h * (0.5 - beta)) as f32;
    let h_f = params.dt;
    let mut d_star = Vec::with_capacity(n);
    for i in 0..n {
        d_star.push(h_f * v[i] + c_dstar_a * a[i]);
    }

    let stiffness = assemble_corotational_stiffness(basis, tets, material, x, n)?;
    let restoring = assemble_corotational_forces(basis, tets, material, rest, x, n)?;

    // b = f_ext + f_r0 - C v* - K d* = f_ext + f_r0 - η_M M v* - η_K K v* - K d*.
    let v_star_flat = flatten(&v_star);
    let d_star_flat = flatten(&d_star);
    let m_vstar = mass.apply(&v_star_flat)?;
    let k_vstar = stiffness.apply(&v_star_flat)?;
    let k_dstar = stiffness.apply(&d_star_flat)?;
    let f_r = flatten(&restoring);
    let f_ext = flatten(external_forces);
    let mut b = Vec::with_capacity(3 * n);
    for i in 0..3 * n {
        let value = f64::from(f_ext[i]) + f64::from(f_r[i])
            - eta_m * f64::from(m_vstar[i])
            - eta_k * f64::from(k_vstar[i])
            - f64::from(k_dstar[i]);
        b.push(value as f32);
    }

    let (a_next_flat, solver) = solve_filtered_spd(
        &stiffness,
        mass,
        c_m,
        c_k,
        &b,
        pinned,
        params.preconditioner,
        &params.cg,
    )?;
    let a_next = unflatten(&a_next_flat);

    let c_v = (gamma * h) as f32;
    let c_x = (beta * h * h) as f32;
    let mut positions_next = Vec::with_capacity(n);
    let mut velocities_next = Vec::with_capacity(n);
    let mut accelerations_next = Vec::with_capacity(n);
    for i in 0..n {
        if pinned.is_some_and(|m| m[i]) {
            positions_next.push(x[i]);
            velocities_next.push(Vec3::ZERO);
            accelerations_next.push(Vec3::ZERO);
        } else {
            velocities_next.push(v_star[i] + c_v * a_next[i]);
            positions_next.push(x[i] + d_star[i] + c_x * a_next[i]);
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

    fn kinetic_energy(mass: &LumpedMass, v: &[Vec3]) -> f32 {
        let mut e = 0.0_f64;
        for (i, vi) in v.iter().enumerate() {
            e += f64::from(mass.get(3 * i)) * f64::from(vi.length_squared());
        }
        (0.5 * e) as f32
    }

    #[test]
    fn params_validation() {
        assert!(NewmarkParams::new(1.0 / 60.0, 0.25, 0.5).is_some());
        assert!(NewmarkParams::new(0.0, 0.25, 0.5).is_none());
        assert!(NewmarkParams::new(f32::NAN, 0.25, 0.5).is_none());
        assert!(NewmarkParams::new(0.01, 0.0, 0.5).is_none());
        assert!(NewmarkParams::new(0.01, 0.6, 0.5).is_none());
        assert!(NewmarkParams::new(0.01, 0.25, -0.1).is_none());
        assert!(NewmarkParams::new(0.01, 0.25, 1.1).is_none());
        let avg = NewmarkParams::average_acceleration(0.01).unwrap();
        assert_eq!((avg.beta, avg.gamma), (0.25, 0.5));
        let lin = NewmarkParams::linear_acceleration(0.01).unwrap();
        assert!((lin.beta - 1.0 / 6.0).abs() < 1e-6 && lin.gamma == 0.5);
    }

    #[test]
    fn initial_acceleration_recovers_gravity_at_rest() {
        // At the rest pose the restoring force is zero, so M a_0 = f_ext = M g,
        // giving a_0 = g for every (free) vertex.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = vec![Vec3::ZERO; n];
        let f = gravity(&mass, n);
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v,
            &f,
            None,
            &RayleighDamping::NONE,
        )
        .unwrap();
        for ai in &a0 {
            assert!(
                (*ai - Vec3::new(0.0, -9.81, 0.0)).length() < 1e-2,
                "a0 = {ai:?}"
            );
        }
    }

    #[test]
    fn free_fall_matches_rigid_acceleration() {
        // Uniform gravity lies in the null space of the corotational tangent, so
        // one average-acceleration step from rest reproduces rigid free fall:
        // a_{n+1} = g, v_{n+1} = h g.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &vec![Vec3::ZERO; n],
            &f,
            None,
            &RayleighDamping::NONE,
        )
        .unwrap();
        let state = NewmarkState {
            positions: verts.clone(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: a0,
        };
        let out = step_newmark(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();
        assert!(out.solver.converged);
        let hg = (1.0 / 60.0) * -9.81;
        for (i, vi) in out.velocities.iter().enumerate() {
            assert!((vi.y - hg).abs() < 1e-2, "vertex {i} v_y = {}", vi.y);
            assert!((out.accelerations[i].y - (-9.81)).abs() < 1e-1);
        }
    }

    #[test]
    fn average_acceleration_is_stable_over_many_steps() {
        // Anchor one vertex, perturb another, and integrate undamped for many
        // steps: the unconditionally stable scheme must stay bounded and finite.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let mut pinned = vec![false; n];
        pinned[0] = true;
        let mut positions = verts.clone();
        positions[4] += Vec3::new(0.15, 0.0, 0.0);
        let f = vec![Vec3::ZERO; n];
        let params = NewmarkParams::average_acceleration(1.0 / 120.0).unwrap();
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &positions,
            &vec![Vec3::ZERO; n],
            &f,
            Some(&pinned),
            &RayleighDamping::NONE,
        )
        .unwrap();
        let mut state = NewmarkState {
            positions,
            velocities: vec![Vec3::ZERO; n],
            accelerations: a0,
        };
        let mut max_disp = 0.0_f32;
        for _ in 0..200 {
            let out = step_newmark(
                &basis,
                &tets,
                &mat,
                &mass,
                &verts,
                &state,
                &f,
                Some(&pinned),
                &params,
            )
            .unwrap();
            for (xi, ri) in out.positions.iter().zip(verts.iter()) {
                max_disp = max_disp.max(xi.distance(*ri));
                assert!(xi.is_finite(), "position diverged: {xi:?}");
            }
            state = NewmarkState {
                positions: out.positions,
                velocities: out.velocities,
                accelerations: out.accelerations,
            };
        }
        // Undamped oscillation about rest: displacement stays near the initial
        // perturbation scale rather than exploding.
        assert!(
            max_disp < 1.0,
            "undamped Newmark blew up: max_disp = {max_disp}"
        );
    }

    #[test]
    fn stiffness_damping_dissipates_energy() {
        // With β-Rayleigh (stiffness) damping, the kinetic energy of a perturbed
        // body must decay relative to the undamped integrator.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let mut positions = verts.clone();
        positions[4] += Vec3::new(0.1, 0.1, 0.0);
        let f = vec![Vec3::ZERO; n];

        let run = |damping: RayleighDamping| -> f32 {
            let params = NewmarkParams {
                damping,
                ..NewmarkParams::average_acceleration(1.0 / 120.0).unwrap()
            };
            let a0 = initial_acceleration(
                &basis,
                &tets,
                &mat,
                &mass,
                &verts,
                &positions,
                &vec![Vec3::ZERO; n],
                &f,
                None,
                &damping,
            )
            .unwrap();
            let mut state = NewmarkState {
                positions: positions.clone(),
                velocities: vec![Vec3::ZERO; n],
                accelerations: a0,
            };
            for _ in 0..40 {
                let out = step_newmark(
                    &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
                )
                .unwrap();
                state = NewmarkState {
                    positions: out.positions,
                    velocities: out.velocities,
                    accelerations: out.accelerations,
                };
            }
            kinetic_energy(&mass, &state.velocities)
        };

        let undamped = run(RayleighDamping::NONE);
        let damped = run(RayleighDamping::new(0.0, 0.05).unwrap());
        assert!(
            damped < undamped,
            "damped KE {damped} not below undamped {undamped}"
        );
    }

    #[test]
    fn pinned_vertices_are_frozen() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let pinned = vec![true, false, false, false, false];
        let f = gravity(&mass, n);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let a0 = initial_acceleration(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &vec![Vec3::ZERO; n],
            &f,
            Some(&pinned),
            &RayleighDamping::NONE,
        )
        .unwrap();
        let state = NewmarkState {
            positions: verts.clone(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: a0,
        };
        let out = step_newmark(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &state,
            &f,
            Some(&pinned),
            &params,
        )
        .unwrap();
        assert!((out.positions[0] - verts[0]).length() < 1e-6);
        assert!(out.velocities[0].length() < 1e-6);
        assert!(out.accelerations[0].length() < 1e-6);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let good = NewmarkState {
            positions: verts.clone(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: vec![Vec3::ZERO; n],
        };
        // Short external force.
        assert!(step_newmark(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &good,
            &vec![Vec3::ZERO; n - 1],
            None,
            &params
        )
        .is_none());
        // Mismatched acceleration length in state.
        let bad = NewmarkState {
            positions: verts.clone(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: vec![Vec3::ZERO; n - 1],
        };
        assert!(step_newmark(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &bad,
            &vec![Vec3::ZERO; n],
            None,
            &params
        )
        .is_none());
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let f = gravity(&mass, n);
        let params = NewmarkParams::average_acceleration(1.0 / 60.0).unwrap();
        let state = NewmarkState {
            positions: verts.clone(),
            velocities: vec![Vec3::ZERO; n],
            accelerations: vec![Vec3::ZERO; n],
        };
        let a = step_newmark(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();
        let b = step_newmark(
            &basis, &tets, &mat, &mass, &verts, &state, &f, None, &params,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
