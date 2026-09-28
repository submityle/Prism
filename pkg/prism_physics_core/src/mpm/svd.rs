//! Robust 3x3 singular-value decomposition and symmetric eigen-solve.
//!
//! MPM's plasticity and elasticity need the singular-value decomposition
//! `F = U Σ Vᵀ` (with `U`, `V` proper rotations and the smallest singular
//! value carrying the sign of `det F`) and the polar-decomposition rotation
//! `R = U Vᵀ`. Both are derived here from a symmetric eigen-solve of `FᵀF`
//! performed with a cyclic Jacobi sweep. Every rotation angle is obtained from
//! a `sqrt`-based tangent formula, so **no trigonometric functions are used**.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! cyclic Jacobi eigenvalue algorithm and the `sqrt`-based rotation formulas
//! are standard, publicly documented numerical-linear-algebra techniques
//! (Golub & Van Loan, *Matrix Computations*); the signed-SVD convention for
//! deformation gradients follows Stomakhin et al. 2013.

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

/// Number of cyclic Jacobi sweeps. Five sweeps already drive the off-diagonal
/// of a 3x3 symmetric matrix to well below single-precision noise; eight is
/// used for a comfortable determinism margin.
const JACOBI_SWEEPS: usize = 8;

/// The result of a signed 3x3 singular-value decomposition `F = U Σ Vᵀ`.
///
/// `u` and `v` are proper rotations (`det = +1`) and the entries of `sigma`
/// are the singular values in **descending** order. When `det F < 0` the
/// smallest singular value (`sigma.z`) is negative so that the reconstruction
/// `u * diag(sigma) * vᵀ` reproduces the reflection while keeping `u` and `v`
/// rotations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd3 {
    /// Left rotation factor (`det = +1`).
    pub u: Mat3,
    /// Signed singular values in descending order.
    pub sigma: Vec3,
    /// Right rotation factor (`det = +1`).
    pub v: Mat3,
}

/// Returns element `(row, col)` of `m` (glam matrices are column-major).
#[inline]
#[must_use]
fn at(m: &Mat3, row: usize, col: usize) -> Real {
    m.col(col)[row]
}

/// Builds the Givens rotation acting in the `(p, q)` plane with the given
/// cosine and sine, laid out so that `Jᵀ A J` zeroes the `(p, q)` entry.
#[must_use]
fn givens(p: usize, q: usize, c: Real, s: Real) -> Mat3 {
    let mut m = [[0.0 as Real; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m[p][p] = c;
    m[q][q] = c;
    m[p][q] = s;
    m[q][p] = -s;
    Mat3::from_cols(
        Vec3::new(m[0][0], m[1][0], m[2][0]),
        Vec3::new(m[0][1], m[1][1], m[2][1]),
        Vec3::new(m[0][2], m[1][2], m[2][2]),
    )
}

/// Computes the symmetric eigen-decomposition `A = V Λ Vᵀ` of a symmetric 3x3
/// matrix using a cyclic Jacobi sweep.
///
/// Returns the eigenvector matrix `V` (columns are unit eigenvectors) and the
/// eigenvalues, both sorted so the eigenvalues are in **descending** order.
/// Only `sqrt` (never a trigonometric call) is used to form the rotations.
#[must_use]
pub fn symmetric_eigen(a: Mat3) -> (Mat3, Vec3) {
    let mut a = a;
    let mut v = Mat3::IDENTITY;
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    for _ in 0..JACOBI_SWEEPS {
        for &(p, q) in &pairs {
            let apq = at(&a, p, q);
            if apq.abs() < 1.0e-20 {
                continue;
            }
            let app = at(&a, p, p);
            let aqq = at(&a, q, q);
            let tau = (aqq - app) / (2.0 * apq);
            // t = tan(theta) via the sqrt-based stable branch, no trig.
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            let j = givens(p, q, c, s);
            a = j.transpose() * a * j;
            v *= j;
        }
    }
    let eig = Vec3::new(at(&a, 0, 0), at(&a, 1, 1), at(&a, 2, 2));
    sort_descending(v, eig)
}

/// Reorders the three eigenpairs so the eigenvalues are in descending order,
/// carrying the matching eigenvector columns along.
#[must_use]
fn sort_descending(v: Mat3, eig: Vec3) -> (Mat3, Vec3) {
    let mut cols = [v.x_axis, v.y_axis, v.z_axis];
    let mut vals = [eig.x, eig.y, eig.z];
    // Simple descending insertion sort over three elements.
    for i in 1..3 {
        let mut j = i;
        while j > 0 && vals[j - 1] < vals[j] {
            vals.swap(j - 1, j);
            cols.swap(j - 1, j);
            j -= 1;
        }
    }
    (
        Mat3::from_cols(cols[0], cols[1], cols[2]),
        Vec3::new(vals[0], vals[1], vals[2]),
    )
}

/// Returns any unit vector orthogonal to `n` (which need not be normalized).
#[must_use]
fn orthogonal_unit(n: Vec3) -> Vec3 {
    // Cross with whichever axis is least aligned with `n` for stability.
    let a = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        Vec3::X
    } else if n.y.abs() <= n.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let c = n.cross(a);
    let len = c.length();
    if len > 1.0e-12 {
        c / len
    } else {
        Vec3::X
    }
}

/// Computes the signed singular-value decomposition of an arbitrary 3x3
/// matrix `f`, returning proper rotations `u`, `v` and descending singular
/// values (the smallest carrying `sign(det f)`).
///
/// Degenerate (near-zero) singular values are handled by completing `u` with
/// an orthonormal frame, so the routine never divides by zero and always
/// returns finite rotations.
#[must_use]
pub fn svd3(f: Mat3) -> Svd3 {
    let ata = f.transpose() * f;
    let (mut v, eig) = symmetric_eigen(ata);
    // Make V a proper rotation.
    if v.determinant() < 0.0 {
        v = Mat3::from_cols(v.x_axis, v.y_axis, -v.z_axis);
    }
    let mut sigma = Vec3::new(
        eig.x.max(0.0).sqrt(),
        eig.y.max(0.0).sqrt(),
        eig.z.max(0.0).sqrt(),
    );
    // B = F V, whose columns are sigma_i * u_i.
    let b = f * v;
    let cols_b = [b.x_axis, b.y_axis, b.z_axis];
    let sig = [sigma.x, sigma.y, sigma.z];
    let tol = 1.0e-9 * sigma.x.max(1.0);
    let mut valid = [false; 3];
    let mut u_cols = [Vec3::ZERO; 3];
    for i in 0..3 {
        if sig[i] > tol {
            u_cols[i] = cols_b[i] / sig[i];
            valid[i] = true;
        }
    }
    // Fill degenerate columns with an orthonormal completion (descending order
    // means invalid columns are the trailing ones).
    match (valid[0], valid[1], valid[2]) {
        (true, true, true) => {}
        (true, true, false) => {
            u_cols[2] = u_cols[0].cross(u_cols[1]).normalize_or_zero();
            if u_cols[2] == Vec3::ZERO {
                u_cols[2] = orthogonal_unit(u_cols[0]);
            }
        }
        (true, false, false) => {
            let e1 = orthogonal_unit(u_cols[0]);
            let e2 = u_cols[0].cross(e1).normalize_or_zero();
            u_cols[1] = e1;
            u_cols[2] = e2;
        }
        _ => {
            u_cols[0] = Vec3::X;
            u_cols[1] = Vec3::Y;
            u_cols[2] = Vec3::Z;
        }
    }
    let mut u = Mat3::from_cols(u_cols[0], u_cols[1], u_cols[2]);
    // If U is a reflection, flip the smallest singular value and its column so
    // that F = U diag(sigma) Vᵀ still holds while U becomes a proper rotation.
    if u.determinant() < 0.0 {
        u = Mat3::from_cols(u.x_axis, u.y_axis, -u.z_axis);
        sigma.z = -sigma.z;
    }
    Svd3 { u, sigma, v }
}

/// Returns the polar-decomposition rotation `R = U Vᵀ` of `f`.
///
/// `R` is the closest proper rotation to `f` in the Frobenius norm and is the
/// rotation used by the fixed-corotated elastic energy.
#[must_use]
pub fn polar_rotation(f: Mat3) -> Mat3 {
    let s = svd3(f);
    s.u * s.v.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_mat(a: Mat3, b: Mat3, eps: Real) -> bool {
        (a.x_axis - b.x_axis).length() <= eps
            && (a.y_axis - b.y_axis).length() <= eps
            && (a.z_axis - b.z_axis).length() <= eps
    }

    fn reconstruct(s: &Svd3) -> Mat3 {
        let sig = Mat3::from_cols(
            Vec3::new(s.sigma.x, 0.0, 0.0),
            Vec3::new(0.0, s.sigma.y, 0.0),
            Vec3::new(0.0, 0.0, s.sigma.z),
        );
        s.u * sig * s.v.transpose()
    }

    #[test]
    fn reconstructs_identity() {
        let s = svd3(Mat3::IDENTITY);
        assert!(approx_mat(reconstruct(&s), Mat3::IDENTITY, 1.0e-5));
        assert!((s.u.determinant() - 1.0).abs() < 1.0e-5);
        assert!((s.v.determinant() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn reconstructs_general_matrix() {
        let f = Mat3::from_cols(
            Vec3::new(1.2, 0.1, -0.3),
            Vec3::new(0.05, 0.9, 0.2),
            Vec3::new(-0.1, 0.15, 1.1),
        );
        let s = svd3(f);
        assert!(approx_mat(reconstruct(&s), f, 1.0e-4));
        assert!((s.u.determinant() - 1.0).abs() < 1.0e-4);
        assert!((s.v.determinant() - 1.0).abs() < 1.0e-4);
        assert!(s.sigma.x >= s.sigma.y && s.sigma.y >= s.sigma.z);
    }

    #[test]
    fn handles_reflection_sign() {
        // A pure reflection has det = -1; the signed SVD must reproduce it.
        let f = Mat3::from_cols(Vec3::new(-1.0, 0.0, 0.0), Vec3::Y, Vec3::Z);
        let s = svd3(f);
        assert!(approx_mat(reconstruct(&s), f, 1.0e-5));
        assert!((s.u.determinant() - 1.0).abs() < 1.0e-5);
        assert!((s.v.determinant() - 1.0).abs() < 1.0e-5);
        assert!(s.sigma.z < 0.0);
    }

    #[test]
    fn polar_rotation_is_orthonormal() {
        let f = Mat3::from_cols(
            Vec3::new(1.1, 0.2, 0.0),
            Vec3::new(-0.15, 1.05, 0.1),
            Vec3::new(0.0, -0.05, 0.98),
        );
        let r = polar_rotation(f);
        let should_be_i = r * r.transpose();
        assert!(approx_mat(should_be_i, Mat3::IDENTITY, 1.0e-4));
        assert!((r.determinant() - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn eigen_reconstructs_symmetric() {
        let a = Mat3::from_cols(
            Vec3::new(2.0, 0.3, 0.1),
            Vec3::new(0.3, 1.0, -0.2),
            Vec3::new(0.1, -0.2, 3.0),
        );
        let (v, eig) = symmetric_eigen(a);
        let diag = Mat3::from_cols(
            Vec3::new(eig.x, 0.0, 0.0),
            Vec3::new(0.0, eig.y, 0.0),
            Vec3::new(0.0, 0.0, eig.z),
        );
        let recon = v * diag * v.transpose();
        assert!(approx_mat(recon, a, 1.0e-4));
    }
}
