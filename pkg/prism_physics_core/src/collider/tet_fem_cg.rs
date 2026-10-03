//! Preconditioner-free conjugate-gradient solver for the implicit `FEM` system.
//!
//! Assembling the global stiffness matrix `K` and the lumped mass matrix `M` is
//! only half of an implicit solver; the other half is solving the symmetric
//! positive-definite linear system that a backward-Euler step produces,
//!
//! ```text
//! (M + h^2 K) dx = b,
//! ```
//!
//! for the nodal displacement increment `dx`. For the large sparse systems that
//! arise from a tetrahedral mesh the method of choice is the conjugate-gradient
//! (`CG`) iteration: it needs only matrix-vector products, converges in exact
//! arithmetic within `n` steps, and never forms or factorises the matrix.
//!
//! This module provides a generic `CG` driver that works against any symmetric
//! positive-definite operator supplied as a closure, plus a thin wrapper that
//! builds the implicit operator `x -> M x + h^2 K x` from the assembled
//! [`GlobalStiffness`] and [`LumpedMass`]. It holds no simulation state and
//! performs no time integration (`h` is a plain parameter), so it is fully
//! decoupled from any solver loop. It is the textbook `CG` algorithm; nothing
//! here is derived from Unreal Engine source.

use super::tet_fem_assembly::GlobalStiffness;
use super::tet_lumped_mass::LumpedMass;

/// Parameters controlling the conjugate-gradient iteration.
#[derive(Clone, Copy, Debug)]
pub struct CgParams {
    /// Maximum number of iterations before the solver gives up.
    pub max_iterations: usize,
    /// Relative residual tolerance: the iteration stops once the Euclidean norm
    /// of the residual drops to or below `tolerance * ||b||`.
    pub tolerance: f32,
}

impl CgParams {
    /// Creates parameters from an iteration cap and a relative tolerance.
    #[must_use]
    pub fn new(max_iterations: usize, tolerance: f32) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for CgParams {
    fn default() -> Self {
        Self {
            max_iterations: 256,
            tolerance: 1.0e-6,
        }
    }
}

/// Diagnostics describing how a conjugate-gradient solve terminated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgReport {
    /// Number of iterations actually performed.
    pub iterations: usize,
    /// Euclidean norm of the final residual `b - A x`.
    pub residual_norm: f32,
    /// Whether the relative residual tolerance was met.
    pub converged: bool,
}

/// Dot product of two equal-length slices, accumulated in `f64`.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Solves `A x = b` for a symmetric positive-definite operator `A` using the
/// conjugate-gradient method starting from `x = 0`.
///
/// `apply` must return `A y` for the given vector `y`; it is assumed symmetric
/// positive-definite. Returns `None` when `b` is empty. The iteration stops on
/// convergence, on the iteration cap, or on a non-positive curvature
/// `p . A p <= 0` (which signals that `A` is not positive-definite); the latter
/// two cases are reported with `converged = false`.
#[must_use]
pub fn conjugate_gradient<F>(apply: F, b: &[f32], params: &CgParams) -> Option<(Vec<f32>, CgReport)>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let n = b.len();
    if n == 0 {
        return None;
    }

    let mut x = vec![0.0f32; n];
    // Residual r = b - A x = b (since x = 0).
    let mut r = b.to_vec();
    let b_norm = dot(b, b).sqrt();
    let threshold = f64::from(params.tolerance) * b_norm;

    // A zero right-hand side has the trivial solution.
    if b_norm == 0.0 {
        return Some((
            x,
            CgReport {
                iterations: 0,
                residual_norm: 0.0,
                converged: true,
            },
        ));
    }

    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    for iteration in 1..=params.max_iterations {
        let ap = apply(&p);
        let p_ap = dot(&p, &ap);
        if p_ap <= 0.0 {
            // Breakdown: operator is not positive-definite along p.
            return Some((
                x,
                CgReport {
                    iterations: iteration,
                    residual_norm: rs_old.sqrt() as f32,
                    converged: false,
                },
            ));
        }
        let alpha = rs_old / p_ap;
        for (xi, &pi) in x.iter_mut().zip(p.iter()) {
            *xi = (f64::from(*xi) + alpha * f64::from(pi)) as f32;
        }
        for (ri, &api) in r.iter_mut().zip(ap.iter()) {
            *ri = (f64::from(*ri) - alpha * f64::from(api)) as f32;
        }
        let rs_new = dot(&r, &r);
        let residual_norm = rs_new.sqrt();
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
        let beta = rs_new / rs_old;
        for (pi, &ri) in p.iter_mut().zip(r.iter()) {
            *pi = (f64::from(ri) + beta * f64::from(*pi)) as f32;
        }
        rs_old = rs_new;
    }

    Some((
        x,
        CgReport {
            iterations: params.max_iterations,
            residual_norm: rs_old.sqrt() as f32,
            converged: false,
        },
    ))
}

/// Solves the implicit system `(M + h^2 K) x = b` by conjugate gradient.
///
/// `stiffness` and `mass` must share the same degree-of-freedom count, which
/// must also equal `b.len()`; otherwise `None` is returned. The operator is
/// symmetric positive-definite whenever `mass` is strictly positive-definite
/// (every vertex carries positive lumped mass), because `K` is positive
/// semi-definite and `h^2 >= 0`.
#[must_use]
pub fn solve_implicit_system(
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
    let h2 = f64::from(h) * f64::from(h);
    let apply = |y: &[f32]| -> Vec<f32> {
        let my = mass.apply(y).expect("mass dimension validated");
        let ky = stiffness.apply(y).expect("stiffness dimension validated");
        my.iter()
            .zip(ky.iter())
            .map(|(&m, &k)| (f64::from(m) + h2 * f64::from(k)) as f32)
            .collect()
    };
    conjugate_gradient(apply, b, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_assembly::assemble_global_stiffness;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_fem_stiffness::IsotropicElasticity;
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;
    use glam::Vec3;

    fn two_tets() -> (Vec<Vec3>, Vec<[u32; 4]>) {
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

    #[test]
    fn solves_a_small_dense_spd_system() {
        // A = [[4,1],[1,3]], b = [1,2]; exact solution x = [1/11, 7/11].
        let a = |y: &[f32]| -> Vec<f32> { vec![4.0 * y[0] + y[1], y[0] + 3.0 * y[1]] };
        let b = [1.0f32, 2.0];
        let (x, report) = conjugate_gradient(a, &b, &CgParams::default()).unwrap();
        assert!(report.converged, "did not converge");
        assert!((x[0] - 1.0 / 11.0).abs() <= 1e-5, "x0 = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() <= 1e-5, "x1 = {}", x[1]);
        // At most n iterations are needed for an n-dimensional SPD system.
        assert!(report.iterations <= 2, "iters = {}", report.iterations);
    }

    #[test]
    fn solves_a_diagonal_system() {
        let diag = [2.0f32, 5.0, 0.5, 10.0];
        let a =
            |y: &[f32]| -> Vec<f32> { diag.iter().zip(y.iter()).map(|(&d, &v)| d * v).collect() };
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let (x, report) = conjugate_gradient(a, &b, &CgParams::default()).unwrap();
        assert!(report.converged);
        for i in 0..4 {
            assert!((x[i] - b[i] / diag[i]).abs() <= 1e-5, "x{i} = {}", x[i]);
        }
    }

    #[test]
    fn zero_right_hand_side_is_trivial() {
        let a = |y: &[f32]| -> Vec<f32> { y.to_vec() };
        let b = [0.0f32; 4];
        let (x, report) = conjugate_gradient(a, &b, &CgParams::default()).unwrap();
        assert_eq!(report.iterations, 0);
        assert!(report.converged);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn non_positive_definite_operator_reports_breakdown() {
        // A = diag(1, -1) is indefinite; CG must stop without converging.
        let a = |y: &[f32]| -> Vec<f32> { vec![y[0], -y[1]] };
        let b = [1.0f32, 1.0];
        let (_x, report) = conjugate_gradient(a, &b, &CgParams::default()).unwrap();
        assert!(!report.converged);
    }

    #[test]
    fn recovers_a_known_displacement_on_the_implicit_fem_system() {
        let (verts, tets) = two_tets();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mat = IsotropicElasticity::new(2.0e3, 0.3).unwrap();
        let k = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let m =
            build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams { density: 1.0 }).unwrap();
        assert!(m.is_invertible());

        // Choose a target displacement, form b = (M + h^2 K) x, then recover x.
        let n = k.n_dofs();
        let mut x_true = vec![0.0f32; n];
        let mut state = 0x1234_5678u64;
        for v in &mut x_true {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0;
        }
        let h = 0.05f32;
        let b = implicit_forward(&k, &m, h, &x_true);
        let (x, report) = solve_implicit_system(&k, &m, h, &b, &CgParams::new(512, 1e-8)).unwrap();
        assert!(report.converged, "iters = {}", report.iterations);
        for i in 0..n {
            assert!(
                (x[i] - x_true[i]).abs() <= 1e-3 * (1.0 + x_true[i].abs()),
                "dof {i}: {} vs {}",
                x[i],
                x_true[i]
            );
        }
    }

    /// Applies the implicit operator forward: returns `(M + h^2 K) x`.
    fn implicit_forward(k: &GlobalStiffness, m: &LumpedMass, h: f32, x: &[f32]) -> Vec<f32> {
        let h2 = f64::from(h) * f64::from(h);
        let mx = m.apply(x).unwrap();
        let kx = k.apply(x).unwrap();
        mx.iter()
            .zip(kx.iter())
            .map(|(&a, &c)| (f64::from(a) + h2 * f64::from(c)) as f32)
            .collect()
    }

    #[test]
    fn mismatched_dimensions_are_rejected() {
        let (verts, tets) = two_tets();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mat = IsotropicElasticity::default();
        let k = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let m = build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams::default()).unwrap();
        let wrong = vec![0.0f32; k.n_dofs() + 3];
        assert!(solve_implicit_system(&k, &m, 0.01, &wrong, &CgParams::default()).is_none());
        assert!(conjugate_gradient(|y: &[f32]| y.to_vec(), &[], &CgParams::default()).is_none());
    }

    #[test]
    fn solve_is_deterministic() {
        let (verts, tets) = two_tets();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mat = IsotropicElasticity::new(1.5e3, 0.25).unwrap();
        let k = assemble_global_stiffness(&basis, &tets, &mat, verts.len()).unwrap();
        let m = build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams::default()).unwrap();
        let b = vec![0.5f32; k.n_dofs()];
        let r1 = solve_implicit_system(&k, &m, 0.02, &b, &CgParams::default()).unwrap();
        let r2 = solve_implicit_system(&k, &m, 0.02, &b, &CgParams::default()).unwrap();
        assert_eq!(r1.0, r2.0);
        assert_eq!(r1.1, r2.1);
    }
}
