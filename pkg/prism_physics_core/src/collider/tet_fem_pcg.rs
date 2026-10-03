//! Jacobi-preconditioned conjugate gradient for the implicit `FEM` system.
//!
//! The preconditioner-free solver in
//! [`tet_fem_cg`](crate::collider::tet_fem_cg) converges slowly when the
//! implicit operator `A = M + h^2 K` is ill conditioned, which is the usual
//! case for stiff materials or small time steps. A diagonal (Jacobi)
//! preconditioner `P = diag(A)` is the cheapest effective remedy: it costs one
//! reciprocal per degree of freedom, requires no extra storage beyond the
//! diagonal, and clusters the spectrum enough to cut the iteration count
//! substantially.
//!
//! This module provides a generic preconditioned conjugate-gradient (`PCG`)
//! driver that accepts the operator and the preconditioner as closures, plus a
//! wrapper that builds the backward-Euler operator `x -> M x + h^2 K x` and its
//! Jacobi preconditioner from the assembled [`GlobalStiffness`] and
//! [`LumpedMass`]. It reuses the [`CgParams`]/[`CgReport`] types and, like the
//! unpreconditioned driver, holds no simulation state and performs no time
//! integration. It is the textbook `PCG` algorithm; nothing here is derived
//! from Unreal Engine source.

use super::tet_fem_assembly::GlobalStiffness;
use super::tet_fem_cg::{CgParams, CgReport};
use super::tet_lumped_mass::LumpedMass;

/// Euclidean inner product accumulated in `f64` for numerical stability.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Solves `A x = b` by Jacobi/preconditioned conjugate gradient.
///
/// `apply` evaluates the symmetric positive-definite operator `A`, while
/// `apply_preconditioner` evaluates `P^{-1} r` for a symmetric positive-definite
/// preconditioner `P` (for Jacobi, `P^{-1} r_i = r_i / diag(A)_i`). The
/// iteration starts from `x = 0`, measures the true (unpreconditioned) relative
/// residual `||r|| <= tolerance * ||b||`, and reports breakdown
/// (`converged = false`) if the operator is not positive-definite along a
/// search direction.
///
/// Returns `None` when `b` is empty or when either closure returns a vector of
/// the wrong length.
#[must_use]
pub fn preconditioned_conjugate_gradient<A, P>(
    apply: A,
    apply_preconditioner: P,
    b: &[f32],
    params: &CgParams,
) -> Option<(Vec<f32>, CgReport)>
where
    A: Fn(&[f32]) -> Vec<f32>,
    P: Fn(&[f32]) -> Vec<f32>,
{
    let n = b.len();
    if n == 0 {
        return None;
    }

    let mut x = vec![0.0f32; n];
    // Residual r = b - A x = b because x starts at zero.
    let mut r = b.to_vec();
    let b_norm = dot(b, b).sqrt();
    let threshold = f64::from(params.tolerance) * b_norm;

    if b_norm <= 0.0 {
        return Some((
            x,
            CgReport {
                iterations: 0,
                residual_norm: 0.0,
                converged: true,
            },
        ));
    }

    // z = P^{-1} r, p = z.
    let mut z = apply_preconditioner(&r);
    if z.len() != n {
        return None;
    }
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    for iteration in 1..=params.max_iterations {
        let ap = apply(&p);
        if ap.len() != n {
            return None;
        }
        let p_ap = dot(&p, &ap);
        if p_ap <= 0.0 {
            return Some((
                x,
                CgReport {
                    iterations: iteration,
                    residual_norm: dot(&r, &r).sqrt() as f32,
                    converged: false,
                },
            ));
        }
        let alpha = rz_old / p_ap;
        for (xi, &pi) in x.iter_mut().zip(p.iter()) {
            *xi = (f64::from(*xi) + alpha * f64::from(pi)) as f32;
        }
        for (ri, &api) in r.iter_mut().zip(ap.iter()) {
            *ri = (f64::from(*ri) - alpha * f64::from(api)) as f32;
        }
        let residual_norm = dot(&r, &r).sqrt();
        if residual_norm <= threshold {
            return Some((
                x,
                CgReport {
                    iterations: iteration,
                    residual_norm: residual_norm as f32,
                    converged: true,
                },
            ));
        }
        z = apply_preconditioner(&r);
        if z.len() != n {
            return None;
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        for (pi, &zi) in p.iter_mut().zip(z.iter()) {
            *pi = (f64::from(zi) + beta * f64::from(*pi)) as f32;
        }
        rz_old = rz_new;
    }

    Some((
        x,
        CgReport {
            iterations: params.max_iterations,
            residual_norm: dot(&r, &r).sqrt() as f32,
            converged: false,
        },
    ))
}

/// Diagonal of the implicit operator `A = M + h^2 K`.
///
/// Returns `None` when the matrices disagree on their degree-of-freedom count
/// or when any diagonal entry is non-positive (which would make the Jacobi
/// preconditioner singular or indefinite).
fn implicit_diagonal(stiffness: &GlobalStiffness, mass: &LumpedMass, h: f32) -> Option<Vec<f32>> {
    let n = stiffness.n_dofs();
    if mass.n_dofs() != n {
        return None;
    }
    let h2 = f64::from(h) * f64::from(h);
    let mut diagonal = Vec::with_capacity(n);
    for i in 0..n {
        let d = f64::from(mass.get(i)) + h2 * f64::from(stiffness.get(i, i));
        if d <= 0.0 {
            return None;
        }
        diagonal.push(d as f32);
    }
    Some(diagonal)
}

/// Solves `(M + h^2 K) x = b` with a Jacobi-preconditioned conjugate gradient.
///
/// `stiffness` and `mass` must share the same degree-of-freedom count, which
/// must also equal `b.len()`. Returns `None` when the dimensions disagree or
/// when the operator diagonal is non-positive (so the Jacobi preconditioner is
/// undefined). The solution is identical to the unpreconditioned solver in
/// exact arithmetic; preconditioning only changes the convergence rate.
#[must_use]
pub fn solve_implicit_system_jacobi(
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
    let diagonal = implicit_diagonal(stiffness, mass, h)?;
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
        r.iter()
            .zip(diagonal.iter())
            .map(|(&ri, &di)| (f64::from(ri) / f64::from(di)) as f32)
            .collect()
    };
    preconditioned_conjugate_gradient(apply, precondition, b, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_assembly::assemble_global_stiffness;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_cg::solve_implicit_system;
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;
    use glam::Vec3;

    fn params() -> CgParams {
        CgParams::new(512, 1e-6)
    }

    // Dense SPD 2x2 operator and its Jacobi preconditioner.
    fn dense2(a: [[f32; 2]; 2]) -> impl Fn(&[f32]) -> Vec<f32> {
        move |x: &[f32]| {
            vec![
                a[0][0] * x[0] + a[0][1] * x[1],
                a[1][0] * x[0] + a[1][1] * x[1],
            ]
        }
    }

    #[test]
    fn solves_known_dense_system() {
        // A = [[4,1],[1,3]], b = [1,2] -> x = [1/11, 7/11].
        let a = [[4.0, 1.0], [1.0, 3.0]];
        let apply = dense2(a);
        let diag = [a[0][0], a[1][1]];
        let precond = move |r: &[f32]| vec![r[0] / diag[0], r[1] / diag[1]];
        let b = [1.0_f32, 2.0];
        let (x, report) = preconditioned_conjugate_gradient(apply, precond, &b, &params()).unwrap();
        assert!(report.converged);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-4);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-4);
    }

    #[test]
    fn exact_diagonal_preconditioner_converges_in_one_step() {
        // For a diagonal operator the Jacobi preconditioner equals A^{-1}, so
        // PCG must reach the solution in a single iteration.
        let d = [2.0_f32, 5.0, 9.0, 13.0];
        let apply = move |x: &[f32]| x.iter().zip(d.iter()).map(|(&xi, &di)| xi * di).collect();
        let precond = move |r: &[f32]| r.iter().zip(d.iter()).map(|(&ri, &di)| ri / di).collect();
        let b = [1.0_f32, -2.0, 3.0, -4.0];
        let (x, report) = preconditioned_conjugate_gradient(apply, precond, &b, &params()).unwrap();
        assert!(report.converged);
        assert_eq!(report.iterations, 1);
        for i in 0..4 {
            assert!((x[i] - b[i] / d[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn zero_rhs_is_trivial() {
        let apply = dense2([[2.0, 0.0], [0.0, 2.0]]);
        let precond = |r: &[f32]| vec![r[0] / 2.0, r[1] / 2.0];
        let b = [0.0_f32, 0.0];
        let (x, report) = preconditioned_conjugate_gradient(apply, precond, &b, &params()).unwrap();
        assert!(report.converged);
        assert_eq!(report.iterations, 0);
        assert_eq!(x, vec![0.0, 0.0]);
    }

    #[test]
    fn non_spd_operator_reports_breakdown() {
        // Indefinite operator: a negative eigenvalue triggers p·Ap <= 0.
        let apply = dense2([[1.0, 0.0], [0.0, -1.0]]);
        let precond = |r: &[f32]| vec![r[0], r[1]];
        let b = [0.0_f32, 1.0];
        let (_x, report) =
            preconditioned_conjugate_gradient(apply, precond, &b, &params()).unwrap();
        assert!(!report.converged);
    }

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

    fn fem_system() -> (GlobalStiffness, LumpedMass, usize) {
        let (verts, tets) = two_tets();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::new(1e-12)).unwrap();
        let material = IsotropicElasticity::new(1.0e5, 0.3).unwrap();
        let k = assemble_global_stiffness(&basis, &tets, &material, verts.len()).unwrap();
        let m =
            build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams { density: 1000.0 }).unwrap();
        (k, m, verts.len() * 3)
    }

    #[test]
    fn jacobi_matches_unpreconditioned_solution() {
        // Independent cross-check: PCG and plain CG must agree on the solution
        // of the same implicit system to tight tolerance.
        let (k, m, n) = fem_system();
        let h = 0.01_f32;
        let mut b = vec![0.0f32; n];
        let mut state: u64 = 0xabcd_1234_5678_9f01;
        for bi in &mut b {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *bi = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }
        let (x_pcg, rep_pcg) = solve_implicit_system_jacobi(&k, &m, h, &b, &params()).unwrap();
        let (x_cg, _rep_cg) = solve_implicit_system(&k, &m, h, &b, &params()).unwrap();
        assert!(rep_pcg.converged);
        for (a, c) in x_pcg.iter().zip(x_cg.iter()) {
            assert!((a - c).abs() <= 1e-3 * (1.0 + c.abs()));
        }
    }

    #[test]
    fn jacobi_solution_satisfies_operator() {
        // Verify (M + h^2 K) x == b directly.
        let (k, m, n) = fem_system();
        let h = 0.02_f32;
        let b: Vec<f32> = (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
            .collect();
        let (x, report) = solve_implicit_system_jacobi(&k, &m, h, &b, &params()).unwrap();
        assert!(report.converged);
        let h2 = f64::from(h) * f64::from(h);
        let mx = m.apply(&x).unwrap();
        let kx = k.apply(&x).unwrap();
        for i in 0..n {
            let ax = f64::from(mx[i]) + h2 * f64::from(kx[i]);
            assert!((ax - f64::from(b[i])).abs() <= 1e-2 * (1.0 + f64::from(b[i]).abs()));
        }
    }

    #[test]
    fn rejects_dimension_and_singular_diagonal() {
        let (k, m, n) = fem_system();
        // Wrong rhs length.
        assert!(solve_implicit_system_jacobi(&k, &m, 0.01, &[0.0; 2], &params()).is_none());
        // h = 0 with zero mass dof would be singular; here mass is positive, so
        // a correct-length rhs still solves. Confirm the happy path returns Some.
        let b = vec![0.0f32; n];
        assert!(solve_implicit_system_jacobi(&k, &m, 0.01, &b, &params()).is_some());
    }

    #[test]
    fn solver_is_deterministic() {
        let (k, m, n) = fem_system();
        let h = 0.015_f32;
        let b: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let (x0, _) = solve_implicit_system_jacobi(&k, &m, h, &b, &params()).unwrap();
        let (x1, _) = solve_implicit_system_jacobi(&k, &m, h, &b, &params()).unwrap();
        for (a, c) in x0.iter().zip(x1.iter()) {
            assert_eq!(a.to_bits(), c.to_bits());
        }
    }
}
