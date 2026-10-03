//! Shared machinery for implicit finite-element time integrators.
//!
//! Every implicit tetrahedral-FEM integrator in this crate — backward Euler
//! ([`step_implicit_corotational`](super::tet_fem_integrator::step_implicit_corotational))
//! and Newmark-β ([`step_newmark`](super::tet_fem_newmark::step_newmark)) — ends
//! up solving the same shape of symmetric-positive-definite system,
//!
//! ```text
//!   A x = b,   A = c_m M + c_k K,
//! ```
//!
//! where `M` is the lumped mass, `K` the (corotational) tangent stiffness and
//! `c_m, c_k > 0` are integrator-specific coefficients. Dirichlet boundary
//! conditions are imposed with the filtered-conjugate-gradient construction of
//! Baraff & Witkin, *Large Steps in Cloth Simulation* (1998): the operator, the
//! preconditioner and the right-hand side are all composed with a projection
//! that zeroes the pinned degrees of freedom, so the iteration stays in the free
//! subspace while the pinned block contributes nothing.
//!
//! This module factors that common code out of the individual integrators so
//! each integrator only has to assemble its own `c_m`, `c_k` and `b`. It holds
//! no time-stepping policy of its own.
//!
//! Mass lumping, Jacobi preconditioning and filtered CG are standard
//! finite-element and physically-based-animation constructions. This file
//! contains no Unreal Engine source or derived code.

use super::tet_fem_assembly::GlobalStiffness;
use super::tet_fem_cg::{CgParams, CgReport};
use super::tet_fem_pcg::preconditioned_conjugate_gradient;
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

/// Flattens a slice of vectors into interleaved `[x, y, z, …]` scalars.
pub(crate) fn flatten(v: &[Vec3]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.push(p.x);
        out.push(p.y);
        out.push(p.z);
    }
    out
}

/// Reconstructs a slice of vectors from interleaved scalars.
pub(crate) fn unflatten(s: &[f32]) -> Vec<Vec3> {
    s.chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect()
}

/// Zeroes the three degrees of freedom of every pinned vertex, in place.
pub(crate) fn apply_filter(values: &mut [f32], pinned: Option<&[bool]>) {
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
pub(crate) fn scalar_jacobi_inverse(
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
pub(crate) fn block_jacobi_inverse(
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

/// Solves the filtered symmetric-positive-definite system `A x = b` with
/// `A = c_m M + c_k K`, projected onto the subspace of free (non-pinned)
/// degrees of freedom.
///
/// `rhs` is the right-hand side `b`; it is filtered internally, so callers do
/// not have to pre-project it. The returned solution has zeros on every pinned
/// degree of freedom. Returns `None` when the chosen preconditioner is singular
/// on a free degree of freedom or the conjugate-gradient driver breaks down.
pub(crate) fn solve_filtered_spd(
    stiffness: &GlobalStiffness,
    mass: &LumpedMass,
    c_m: f64,
    c_k: f64,
    rhs: &[f32],
    pinned: Option<&[bool]>,
    preconditioner: FemPreconditioner,
    cg: &CgParams,
) -> Option<(Vec<f32>, CgReport)> {
    let mut b = rhs.to_vec();
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
    let precondition: Box<dyn Fn(&[f32]) -> Vec<f32>> = match preconditioner {
        FemPreconditioner::ScalarJacobi => {
            scalar_inv = scalar_jacobi_inverse(stiffness, mass, c_m, c_k, pinned)?;
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
            block_inv = block_jacobi_inverse(stiffness, mass, c_m, c_k, pinned)?;
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

    preconditioned_conjugate_gradient(apply, precondition.as_ref(), &b, cg)
}
