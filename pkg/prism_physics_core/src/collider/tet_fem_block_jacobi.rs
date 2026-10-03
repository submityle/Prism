//! Block-Jacobi (`3x3`) preconditioned conjugate gradient for the implicit
//! `FEM` system.
//!
//! Scalar Jacobi preconditioning (see
//! [`solve_implicit_system_jacobi`](super::tet_fem_pcg::solve_implicit_system_jacobi))
//! ignores the coupling between the three spatial components of a vertex. For
//! the vector-valued elasticity operator `A = M + h^2 K` that coupling is
//! significant, so inverting the full `3x3` diagonal block of each vertex is a
//! strictly better preconditioner at nearly the same cost: it captures the
//! intra-vertex anisotropy that scalar Jacobi discards and typically cuts the
//! iteration count further.
//!
//! This module extracts each vertex's `3x3` diagonal block of `A`, inverts it
//! once, and feeds the resulting block solve as the preconditioner to the
//! generic [`preconditioned_conjugate_gradient`] driver. It is stateless and
//! performs no time integration. The block-Jacobi preconditioner is a textbook
//! construction; nothing here is derived from Unreal Engine source.

use super::tet_fem_assembly::GlobalStiffness;
use super::tet_fem_cg::{CgParams, CgReport};
use super::tet_fem_pcg::preconditioned_conjugate_gradient;
use super::tet_lumped_mass::LumpedMass;
use glam::{Mat3, Vec3};

/// Inverts the `3x3` diagonal block of `A = M + h^2 K` for every vertex.
///
/// Returns one inverse block per vertex, or `None` when the degree-of-freedom
/// count is not a multiple of three, the dimensions disagree, or any block is
/// singular (so the preconditioner is undefined).
fn invert_diagonal_blocks(
    stiffness: &GlobalStiffness,
    mass: &LumpedMass,
    h: f32,
) -> Option<Vec<Mat3>> {
    let n = stiffness.n_dofs();
    if mass.n_dofs() != n || !n.is_multiple_of(3) {
        return None;
    }
    let h2 = f64::from(h) * f64::from(h);
    let n_vertices = n / 3;
    let mut blocks = Vec::with_capacity(n_vertices);
    for v in 0..n_vertices {
        let base = 3 * v;
        let mut entry = [[0.0f32; 3]; 3];
        let mut max_abs = 0.0f64;
        for (a, row) in entry.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                let kab = f64::from(stiffness.get(base + a, base + b));
                let mab = if a == b {
                    f64::from(mass.get(base + a))
                } else {
                    0.0
                };
                let value = mab + h2 * kab;
                *cell = value as f32;
                let mag = value.abs();
                if mag > max_abs {
                    max_abs = mag;
                }
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

/// Solves `(M + h^2 K) x = b` with a `3x3` block-Jacobi preconditioned
/// conjugate gradient.
///
/// `stiffness` and `mass` must share the same degree-of-freedom count, which
/// must be a multiple of three and equal to `b.len()`. Returns `None` when the
/// dimensions disagree or when a vertex's `3x3` diagonal block is singular. In
/// exact arithmetic the solution matches the unpreconditioned and
/// scalar-Jacobi solvers; preconditioning only changes the convergence rate.
#[must_use]
pub fn solve_implicit_system_block_jacobi(
    stiffness: &GlobalStiffness,
    mass: &LumpedMass,
    h: f32,
    b: &[f32],
    params: &CgParams,
) -> Option<(Vec<f32>, CgReport)> {
    let n = stiffness.n_dofs();
    if mass.n_dofs() != n || b.len() != n {
        return None;
    }
    let blocks = invert_diagonal_blocks(stiffness, mass, h)?;
    let h2 = f64::from(h) * f64::from(h);
    let apply = |y: &[f32]| -> Vec<f32> {
        let my = mass.apply(y).expect("mass dimension validated");
        let ky = stiffness.apply(y).expect("stiffness dimension validated");
        my.iter()
            .zip(ky.iter())
            .map(|(&m, &k)| (f64::from(m) + h2 * f64::from(k)) as f32)
            .collect()
    };
    let precondition = |r: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (v, inv) in blocks.iter().enumerate() {
            let base = 3 * v;
            let rv = Vec3::new(r[base], r[base + 1], r[base + 2]);
            let yv = *inv * rv;
            out[base] = yv.x;
            out[base + 1] = yv.y;
            out[base + 2] = yv.z;
        }
        out
    };
    preconditioned_conjugate_gradient(apply, precondition, b, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_assembly::assemble_global_stiffness;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_pcg::solve_implicit_system_jacobi;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use crate::collider::tet_lumped_mass::{build_lumped_mass, build_lumped_mass_from_mesh};
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

    fn system() -> (GlobalStiffness, LumpedMass) {
        let (verts, tets) = mesh();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mat = IsotropicElasticity::new(2.0e4, 0.3).unwrap();
        let k = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let m =
            build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams { density: 1200.0 }).unwrap();
        (k, m)
    }

    fn params() -> CgParams {
        CgParams::new(512, 1e-6)
    }

    fn deterministic_rhs(n: usize) -> Vec<f32> {
        let mut state = 0x51ed_2701_abcd_1234u64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn solution_satisfies_operator() {
        let (k, m) = system();
        let n = k.n_dofs();
        let b = deterministic_rhs(n);
        let h = 1.0 / 60.0;
        let (x, report) = solve_implicit_system_block_jacobi(&k, &m, h, &b, &params()).unwrap();
        assert!(report.converged);
        let h2 = f64::from(h) * f64::from(h);
        let mx = m.apply(&x).unwrap();
        let kx = k.apply(&x).unwrap();
        let mut err = 0.0f64;
        let mut scale = 0.0f64;
        for i in 0..n {
            let ax = f64::from(mx[i]) + h2 * f64::from(kx[i]);
            err += (ax - f64::from(b[i])).powi(2);
            scale += f64::from(b[i]).powi(2);
        }
        assert!(
            err.sqrt() <= 1e-4 * (1.0 + scale.sqrt()),
            "residual {}",
            err.sqrt()
        );
    }

    #[test]
    fn matches_scalar_jacobi_solution() {
        let (k, m) = system();
        let n = k.n_dofs();
        let b = deterministic_rhs(n);
        let h = 1.0 / 120.0;
        let (x_block, _) = solve_implicit_system_block_jacobi(&k, &m, h, &b, &params()).unwrap();
        let (x_scalar, _) = solve_implicit_system_jacobi(&k, &m, h, &b, &params()).unwrap();
        for i in 0..n {
            let d = (f64::from(x_block[i]) - f64::from(x_scalar[i])).abs();
            assert!(
                d <= 1e-3 * (1.0 + f64::from(x_scalar[i]).abs()),
                "dof {i}: {d}"
            );
        }
    }

    #[test]
    fn mass_only_system_converges_in_one_step() {
        // h = 0 => A = M (block diagonal); block Jacobi is exact in one step.
        let (k, m) = system();
        let n = k.n_dofs();
        let b = deterministic_rhs(n);
        let (x, report) = solve_implicit_system_block_jacobi(&k, &m, 0.0, &b, &params()).unwrap();
        assert_eq!(report.iterations, 1);
        for i in 0..n {
            let expected = f64::from(b[i]) / f64::from(m.get(i));
            assert!((f64::from(x[i]) - expected).abs() <= 1e-3 * (1.0 + expected.abs()));
        }
    }

    #[test]
    fn zero_rhs_is_trivial() {
        let (k, m) = system();
        let n = k.n_dofs();
        let b = vec![0.0f32; n];
        let (x, _) = solve_implicit_system_block_jacobi(&k, &m, 1.0 / 60.0, &b, &params()).unwrap();
        assert!(x.iter().all(|&xi| xi == 0.0));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let (k, m) = system();
        let n = k.n_dofs();
        let b = vec![1.0f32; n + 1];
        assert!(solve_implicit_system_block_jacobi(&k, &m, 1.0 / 60.0, &b, &params()).is_none());
    }

    #[test]
    fn rejects_singular_block() {
        // One vertex, zero stiffness, zero mass => A block is the zero matrix.
        let k = GlobalStiffness {
            n_dofs: 3,
            row_offsets: vec![0, 3, 6, 9],
            col_indices: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            values: vec![0.0; 9],
        };
        let m = build_lumped_mass(&[0.0]).unwrap();
        let b = vec![1.0f32, 2.0, 3.0];
        assert!(solve_implicit_system_block_jacobi(&k, &m, 0.0, &b, &params()).is_none());
    }

    #[test]
    fn is_deterministic() {
        let (k, m) = system();
        let n = k.n_dofs();
        let b = deterministic_rhs(n);
        let h = 1.0 / 90.0;
        let (a, _) = solve_implicit_system_block_jacobi(&k, &m, h, &b, &params()).unwrap();
        let (c, _) = solve_implicit_system_block_jacobi(&k, &m, h, &b, &params()).unwrap();
        for i in 0..n {
            assert_eq!(a[i].to_bits(), c[i].to_bits());
        }
    }
}
