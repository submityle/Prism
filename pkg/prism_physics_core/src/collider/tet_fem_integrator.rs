//! Implicit (backward-Euler) corotational FEM time integrator.
//!
//! This module ties the standalone FEM building blocks — the corotational
//! tangent stiffness ([`assemble_corotational_stiffness`]), the corotational
//! internal force ([`assemble_corotational_forces`]), the lumped mass matrix
//! ([`LumpedMass`]), and the preconditioned conjugate-gradient driver
//! ([`preconditioned_conjugate_gradient`]) — into a single implicit time step
//! that advances a tetrahedral soft body one frame.
//!
//! # Formulation
//!
//! For a body with lumped mass `M`, corotational tangent `K`, internal force
//! `f_int(x)`, and external force `f_ext`, backward Euler with the linearised
//! internal force `f_int(x_{n+1}) ≈ f_int(x_n) − h K v_{n+1}` and Rayleigh
//! damping `C = αM + βK` gives
//!
//! ```text
//!   [(1 + hα) M + (hβ + h²) K] v_{n+1} = M v_n + h (f_int(x_n) + f_ext)
//!   x_{n+1} = x_n + h v_{n+1}
//! ```
//!
//! The left-hand operator `A = c_m M + c_k K` with `c_m = 1 + hα` and
//! `c_k = hβ + h²` is symmetric positive-definite, so it is solved with the
//! generic conjugate-gradient driver using either a scalar- or `3×3`
//! block-Jacobi preconditioner.
//!
//! # Dirichlet boundary conditions
//!
//! Pinned vertices are fixed (`v = 0`, position frozen) with the filtered-CG
//! construction of Baraff & Witkin, *Large Steps in Cloth Simulation* (1998):
//! the operator, preconditioner, and right-hand side are all composed with a
//! projection that zeroes the pinned degrees of freedom, so the iteration stays
//! in the free subspace while the pinned block trivially contributes nothing.
//!
//! Backward Euler, Rayleigh damping, corotational linearisation and filtered CG
//! are standard finite-element and physically-based-animation constructions.
//! This file contains no Unreal Engine source or derived code.

use super::tet_fem_assembly::GlobalStiffness;
use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_cg::{CgParams, CgReport};
use super::tet_fem_corotational_assembly::{
    assemble_corotational_forces, assemble_corotational_stiffness,
};
use super::tet_fem_pcg::preconditioned_conjugate_gradient;
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_lumped_mass::LumpedMass;
use glam::{Mat3, Vec3};

/// Preconditioner used by the implicit solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FemPreconditioner {
    /// Diagonal (scalar) Jacobi: `P⁻¹ r_i = r_i / A_ii`.
    #[default]
    ScalarJacobi,
    /// `3×3` block Jacobi: inverts each vertex's diagonal block of `A`, which
    /// captures the intra-vertex coupling that scalar Jacobi discards.
    BlockJacobi,
}

/// Rayleigh damping coefficients for `C = α M + β K`.
///
/// `alpha` is mass-proportional damping (removes low-frequency / bulk motion)
/// and `beta` is stiffness-proportional damping (removes high-frequency
/// vibration). Both must be finite and non-negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayleighDamping {
    /// Mass-proportional coefficient `α ≥ 0`.
    pub alpha: f32,
    /// Stiffness-proportional coefficient `β ≥ 0`.
    pub beta: f32,
}

impl RayleighDamping {
    /// No damping (`α = β = 0`).
    pub const NONE: Self = Self {
        alpha: 0.0,
        beta: 0.0,
    };

    /// Creates damping coefficients, returning `None` if either is negative or
    /// non-finite.
    #[must_use]
    pub fn new(alpha: f32, beta: f32) -> Option<Self> {
        if alpha.is_finite() && beta.is_finite() && alpha >= 0.0 && beta >= 0.0 {
            Some(Self { alpha, beta })
        } else {
            None
        }
    }
}

impl Default for RayleighDamping {
    fn default() -> Self {
        Self::NONE
    }
}

/// Parameters controlling one implicit Euler step.
#[derive(Clone, Copy, Debug)]
pub struct ImplicitStepParams {
    /// Time step `h > 0` in seconds.
    pub dt: f32,
    /// Conjugate-gradient iteration cap and tolerance.
    pub cg: CgParams,
    /// Preconditioner for the linear solve.
    pub preconditioner: FemPreconditioner,
    /// Rayleigh damping coefficients.
    pub damping: RayleighDamping,
}

impl ImplicitStepParams {
    /// Convenience constructor from a time step; defaults to scalar Jacobi, the
    /// default CG parameters, and no damping.
    #[must_use]
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            cg: CgParams::default(),
            preconditioner: FemPreconditioner::default(),
            damping: RayleighDamping::NONE,
        }
    }
}

/// Result of a single implicit step.
#[derive(Clone, Debug, PartialEq)]
pub struct FemStepResult {
    /// Updated vertex positions `x_{n+1}`.
    pub positions: Vec<Vec3>,
    /// Updated vertex velocities `v_{n+1}` (zero for pinned vertices).
    pub velocities: Vec<Vec3>,
    /// Diagnostics from the linear solve.
    pub solver: CgReport,
}

/// Flattens a slice of vectors into interleaved `[x, y, z, …]` scalars.
fn flatten(v: &[Vec3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.push(p.x);
        out.push(p.y);
        out.push(p.z);
    }
    out
}

/// Reconstructs a slice of vectors from interleaved scalars.
fn unflatten(s: &[f32]) -> Vec<Vec3> {
    s.chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect()
}

/// Zeroes the three degrees of freedom of every pinned vertex, in place.
fn apply_filter(values: &mut [f32], pinned: Option<&[bool]>) {
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

/// Builds the scalar-Jacobi inverse diagonal `1 / A_ii`, with pinned degrees of
/// freedom set to a harmless unit entry (the filter zeroes them anyway).
///
/// Returns `None` when a free degree of freedom has a non-positive diagonal,
/// which would make the preconditioner indefinite.
fn scalar_jacobi_inverse(
    stiffness: &GlobalStiffness,
    mass: &LumpedMass,
    c_m: f64,
    c_k: f64,
    pinned: Option<&[bool]>,
) -> Option<Vec<f32>> {
    let n = stiffness.n_dofs();
    let mut inv = Vec::with_capacity(n);
    for i in 0..n {
        let is_pinned = pinned.is_some_and(|m| m[i / 3]);
        if is_pinned {
            inv.push(1.0);
            continue;
        }
        let d = c_m * f64::from(mass.get(i)) + c_k * f64::from(stiffness.get(i, i));
        if d <= 0.0 {
            return None;
        }
        inv.push((1.0 / d) as f32);
    }
    Some(inv)
}

/// Builds the `3×3` block-Jacobi inverse blocks of `A = c_m M + c_k K`, with
/// pinned vertices set to the identity (the filter zeroes them anyway).
///
/// Returns `None` when a free vertex's diagonal block is singular.
fn block_jacobi_inverse(
    stiffness: &GlobalStiffness,
    mass: &LumpedMass,
    c_m: f64,
    c_k: f64,
    pinned: Option<&[bool]>,
) -> Option<Vec<Mat3>> {
    let n = stiffness.n_dofs();
    let n_vertices = n / 3;
    let mut blocks = Vec::with_capacity(n_vertices);
    for v in 0..n_vertices {
        if pinned.is_some_and(|m| m[v]) {
            blocks.push(Mat3::IDENTITY);
            continue;
        }
        let base = 3 * v;
        let mut entry = [[0.0f32; 3]; 3];
        let mut max_abs = 0.0f64;
        for (a, row) in entry.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                let kab = c_k * f64::from(stiffness.get(base + a, base + b));
                let mab = if a == b {
                    c_m * f64::from(mass.get(base + a))
                } else {
                    0.0
                };
                let value = mab + kab;
                *cell = value as f32;
                max_abs = max_abs.max(value.abs());
            }
        }
        let block = Mat3::from_cols(
            Vec3::new(entry[0][0], entry[1][0], entry[2][0]),
            Vec3::new(entry[0][1], entry[1][1], entry[2][1]),
            Vec3::new(entry[0][2], entry[1][2], entry[2][2]),
        );
        let det = f64::from(block.determinant());
        if max_abs <= 0.0 || det.abs() <= 1e-6 * max_abs * max_abs * max_abs {
            return None;
        }
        let inv = block.inverse();
        if !inv.is_finite() {
            return None;
        }
        blocks.push(inv);
    }
    Some(blocks)
}

/// Advances a corotational tetrahedral FEM body by one implicit Euler step.
///
/// Inputs are per-vertex arrays of equal length `n`:
/// - `rest`: the undeformed reference positions (define `K` and `f_int`),
/// - `positions`: the current deformed positions `x_n`,
/// - `velocities`: the current velocities `v_n`,
/// - `external_forces`: per-vertex external force in Newtons (gravity, contacts…),
/// - `pinned`: optional Dirichlet mask; a `true` entry freezes that vertex.
///
/// `mass` must have `3n` degrees of freedom. Returns `None` when any length
/// disagrees, `dt` is not strictly positive and finite, the assemblies fail, a
/// preconditioner diagonal is non-positive/singular, or the solve breaks down.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "an implicit FEM step is parameterised by its mesh, material, mass, \
              rest/current/velocity state, external load and boundary mask; bundling \
              them would only hide the explicit per-step inputs"
)]
pub fn step_implicit_corotational(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    material: &IsotropicElasticity,
    mass: &LumpedMass,
    rest: &[Vec3],
    positions: &[Vec3],
    velocities: &[Vec3],
    external_forces: &[Vec3],
    pinned: Option<&[bool]>,
    params: &ImplicitStepParams,
) -> Option<FemStepResult> {
    let n = positions.len();
    if n == 0
        || rest.len() != n
        || velocities.len() != n
        || external_forces.len() != n
        || mass.n_dofs() != 3 * n
        || pinned.is_some_and(|m| m.len() != n)
        || !params.dt.is_finite()
        || params.dt <= 0.0
    {
        return None;
    }

    let h = f64::from(params.dt);
    let c_m = 1.0 + h * f64::from(params.damping.alpha);
    let c_k = h * f64::from(params.damping.beta) + h * h;

    // Corotational tangent and restoring force at the current configuration.
    let stiffness = assemble_corotational_stiffness(basis, tets, material, positions, n)?;
    let internal = assemble_corotational_forces(basis, tets, material, rest, positions, n)?;

    // Right-hand side b = M v_n + h (f_int + f_ext), then filter pinned rows.
    let v_flat = flatten(velocities);
    let m_v = mass.apply(&v_flat)?;
    let mut total_force = Vec::with_capacity(n);
    for i in 0..n {
        total_force.push(internal[i] + external_forces[i]);
    }
    let f_flat = flatten(&total_force);
    let mut b = Vec::with_capacity(3 * n);
    for i in 0..3 * n {
        b.push((f64::from(m_v[i]) + h * f64::from(f_flat[i])) as f32);
    }
    apply_filter(&mut b, pinned);

    // A y = c_m M y + c_k K y, projected onto the free subspace.
    let apply = |y: &[f32]| -> Vec<f32> {
        let mut yy = y.to_vec();
        apply_filter(&mut yy, pinned);
        let my = mass.apply(&yy).expect("mass dimension validated");
        let ky = stiffness.apply(&yy).expect("stiffness dimension validated");
        let mut out = Vec::with_capacity(my.len());
        for i in 0..my.len() {
            out.push((c_m * f64::from(my[i]) + c_k * f64::from(ky[i])) as f32);
        }
        apply_filter(&mut out, pinned);
        out
    };

    let scalar_inv;
    let block_inv;
    let precondition: Box<dyn Fn(&[f32]) -> Vec<f32>> = match params.preconditioner {
        FemPreconditioner::ScalarJacobi => {
            scalar_inv = scalar_jacobi_inverse(&stiffness, mass, c_m, c_k, pinned)?;
            Box::new(move |r: &[f32]| -> Vec<f32> {
                let mut out: Vec<f32> = r
                    .iter()
                    .zip(scalar_inv.iter())
                    .map(|(&ri, &di)| ri * di)
                    .collect();
                apply_filter(&mut out, pinned);
                out
            })
        }
        FemPreconditioner::BlockJacobi => {
            block_inv = block_jacobi_inverse(&stiffness, mass, c_m, c_k, pinned)?;
            Box::new(move |r: &[f32]| -> Vec<f32> {
                let mut out = vec![0.0f32; r.len()];
                for (v, inv) in block_inv.iter().enumerate() {
                    let base = 3 * v;
                    let rv = Vec3::new(r[base], r[base + 1], r[base + 2]);
                    let yv = *inv * rv;
                    out[base] = yv.x;
                    out[base + 1] = yv.y;
                    out[base + 2] = yv.z;
                }
                apply_filter(&mut out, pinned);
                out
            })
        }
    };

    let (v_next_flat, solver) =
        preconditioned_conjugate_gradient(apply, precondition.as_ref(), &b, &params.cg)?;

    let mut velocities_next = unflatten(&v_next_flat);
    let mut positions_next = Vec::with_capacity(n);
    for i in 0..n {
        if pinned.is_some_and(|m| m[i]) {
            velocities_next[i] = Vec3::ZERO;
            positions_next.push(positions[i]);
        } else {
            positions_next.push(positions[i] + velocities_next[i] * params.dt);
        }
    }

    Some(FemStepResult {
        positions: positions_next,
        velocities: velocities_next,
        solver,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasis, TetFemBasisParams};
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;

    /// A small two-tet mesh sharing a face.
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

    /// Per-vertex gravity force `f_i = m_i g`.
    fn gravity(mass: &LumpedMass, n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| Vec3::new(0.0, -9.81, 0.0) * mass.get(3 * i))
            .collect()
    }

    fn total_speed(v: &[Vec3]) -> f32 {
        v.iter().map(|x| x.length()).sum()
    }

    fn sq_dist_to_rest(x: &[Vec3], rest: &[Vec3]) -> f32 {
        x.iter()
            .zip(rest.iter())
            .map(|(a, b)| a.distance_squared(*b))
            .sum()
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = vec![Vec3::ZERO; n];
        let f = vec![Vec3::ZERO; n];
        let params = ImplicitStepParams::new(1.0 / 60.0);
        // velocities too short.
        let bad_v = vec![Vec3::ZERO; n - 1];
        assert!(step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &verts, &bad_v, &f, None, &params
        )
        .is_none());
        // external forces too short.
        let bad_f = vec![Vec3::ZERO; n - 1];
        assert!(step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &verts, &v, &bad_f, None, &params
        )
        .is_none());
        // pinned mask wrong length.
        let bad_pin = vec![false; n + 1];
        assert!(step_implicit_corotational(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v,
            &f,
            Some(&bad_pin),
            &params
        )
        .is_none());
    }

    #[test]
    fn rejects_nonpositive_or_nonfinite_dt() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = vec![Vec3::ZERO; n];
        let f = vec![Vec3::ZERO; n];
        for bad in [0.0f32, -0.01, f32::NAN, f32::INFINITY] {
            let params = ImplicitStepParams::new(bad);
            assert!(step_implicit_corotational(
                &basis, &tets, &mat, &mass, &verts, &verts, &v, &f, None, &params
            )
            .is_none());
        }
    }

    #[test]
    fn zero_force_uniform_velocity_translates_rigidly() {
        // At the rest configuration with no external force, a uniform velocity
        // lies in the null space of the corotational tangent, so the body keeps
        // its velocity and translates by h*v.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let drift = Vec3::new(0.3, -0.2, 0.5);
        let v = vec![drift; n];
        let f = vec![Vec3::ZERO; n];
        let params = ImplicitStepParams::new(1.0 / 120.0);
        let out = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &verts, &v, &f, None, &params,
        )
        .unwrap();
        assert!(out.solver.converged);
        for (i, vn) in out.velocities.iter().enumerate() {
            assert!(
                (*vn - drift).length() < 1e-3,
                "vertex {i} velocity drifted: {vn:?}"
            );
            let expected = verts[i] + drift * params.dt;
            assert!((out.positions[i] - expected).length() < 1e-4);
        }
    }

    #[test]
    fn all_pinned_vertices_are_frozen() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = vec![Vec3::ZERO; n];
        let f = gravity(&mass, n);
        let pin = vec![true; n];
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let out = step_implicit_corotational(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v,
            &f,
            Some(&pin),
            &params,
        )
        .unwrap();
        for i in 0..n {
            assert_eq!(out.velocities[i], Vec3::ZERO);
            assert_eq!(out.positions[i], verts[i]);
        }
    }

    #[test]
    fn single_anchor_lets_the_rest_fall() {
        // Pin vertex 0; under gravity the anchor stays put while at least one
        // free vertex acquires a downward velocity.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let v = vec![Vec3::ZERO; n];
        let f = gravity(&mass, n);
        let mut pin = vec![false; n];
        pin[0] = true;
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let out = step_implicit_corotational(
            &basis,
            &tets,
            &mat,
            &mass,
            &verts,
            &verts,
            &v,
            &f,
            Some(&pin),
            &params,
        )
        .unwrap();
        assert_eq!(out.velocities[0], Vec3::ZERO);
        assert_eq!(out.positions[0], verts[0]);
        let max_down = out.velocities[1..]
            .iter()
            .map(|v| v.y)
            .fold(f32::INFINITY, f32::min);
        assert!(max_down < -1e-3, "no vertex fell: {:?}", out.velocities);
    }

    #[test]
    fn stretched_body_relaxes_toward_rest() {
        // Stretch the free part along +x, release with zero velocity and no
        // external force: the internal restoring force must pull the body back,
        // so the squared distance to rest decreases over successive steps.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        // Anchor vertex 0 at its rest position and stretch only the free
        // vertices, so the rest shape is the equilibrium of the pinned body.
        let mut x = verts.clone();
        for xi in x.iter_mut().skip(1) {
            xi.x += 0.15;
        }
        let mut v = vec![Vec3::ZERO; n];
        let f = vec![Vec3::ZERO; n];
        let mut pin = vec![false; n];
        pin[0] = true;
        // Rayleigh stiffness damping makes the relaxation an over-damped,
        // monotone decay toward the equilibrium rest shape.
        let mut params = ImplicitStepParams::new(1.0 / 120.0);
        params.damping = RayleighDamping::new(0.0, 0.03).unwrap();

        let mut prev = sq_dist_to_rest(&x, &verts);
        for step in 0..40 {
            let out = step_implicit_corotational(
                &basis,
                &tets,
                &mat,
                &mass,
                &verts,
                &x,
                &v,
                &f,
                Some(&pin),
                &params,
            )
            .unwrap();
            x = out.positions;
            v = out.velocities;
            let now = sq_dist_to_rest(&x, &verts);
            // Monotone non-increase (allow a tiny numerical slack).
            assert!(
                now <= prev + 2e-5,
                "step {step}: distance grew {prev} -> {now}"
            );
            prev = now;
        }
        // And it actually relaxed substantially from the initial stretch
        // (free vertices 1..n each displaced by 0.15 along x).
        let initial = ((n - 1) as f32) * 0.15f32 * 0.15f32;
        assert!(
            prev < 0.5 * initial,
            "did not relax enough: {prev} vs {initial}"
        );
    }

    #[test]
    fn scalar_and_block_jacobi_agree() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let mut x = verts.clone();
        x[4] += Vec3::new(0.1, -0.05, 0.08);
        let v = vec![Vec3::new(0.0, 0.1, 0.0); n];
        let f = gravity(&mass, n);

        let mut scalar = ImplicitStepParams::new(1.0 / 90.0);
        scalar.cg = CgParams::new(1024, 1e-8);
        scalar.preconditioner = FemPreconditioner::ScalarJacobi;
        let mut block = scalar;
        block.preconditioner = FemPreconditioner::BlockJacobi;

        let a = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &scalar,
        )
        .unwrap();
        let b = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &block,
        )
        .unwrap();
        assert!(a.solver.converged && b.solver.converged);
        for i in 0..n {
            assert!(
                (a.velocities[i] - b.velocities[i]).length() < 1e-3,
                "preconditioners disagree at {i}: {:?} vs {:?}",
                a.velocities[i],
                b.velocities[i]
            );
        }
    }

    #[test]
    fn stiffness_damping_reduces_speed() {
        // From the same stretched, at-rest-velocity state, stiffness-proportional
        // damping must yield a smaller acquired speed than the undamped step.
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let mut x = verts.clone();
        for xi in x.iter_mut() {
            xi.x += 0.12;
        }
        let v = vec![Vec3::ZERO; n];
        let f = vec![Vec3::ZERO; n];

        let undamped = ImplicitStepParams::new(1.0 / 120.0);
        let mut damped = undamped;
        damped.damping = RayleighDamping::new(0.0, 0.02).unwrap();

        let a = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &undamped,
        )
        .unwrap();
        let b = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &damped,
        )
        .unwrap();
        let sa = total_speed(&a.velocities);
        let sb = total_speed(&b.velocities);
        assert!(sa > 0.0);
        assert!(sb < sa, "damping did not reduce speed: {sb} vs {sa}");
    }

    #[test]
    fn is_deterministic() {
        let (verts, tets) = mesh();
        let basis = basis_of(&verts, &tets);
        let mat = material();
        let mass = mass_of(&verts, &tets);
        let n = verts.len();
        let mut x = verts.clone();
        x[4] += Vec3::new(0.05, 0.05, -0.05);
        let v = vec![Vec3::new(0.01, 0.0, 0.0); n];
        let f = gravity(&mass, n);
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let a = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &params,
        )
        .unwrap();
        let b = step_implicit_corotational(
            &basis, &tets, &mat, &mass, &verts, &x, &v, &f, None, &params,
        )
        .unwrap();
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.velocities, b.velocities);
    }
}
