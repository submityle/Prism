//! A trig-free symmetric eigensolver for modal analysis.
//!
//! Reduced-order deformation needs the low-frequency vibration modes of a soft
//! body, which are the eigenvectors of the (generalised) stiffness eigenproblem.
//! This module solves the *standard* symmetric eigenproblem `A x = lambda x` for
//! a small dense symmetric matrix using the cyclic Jacobi rotation method.
//!
//! The classical Jacobi rotation angle is normally obtained with `atan`, but the
//! rotation's cosine and sine can be computed with square roots alone via the
//! standard identity
//!
//! ```text
//! theta = (a_qq - a_pp) / (2 a_pq)
//! t     = sign(theta) / (|theta| + sqrt(theta^2 + 1))
//! c     = 1 / sqrt(t^2 + 1),   s = t c
//! ```
//!
//! which keeps the whole solver free of transcendental calls (the engine's lint
//! policy forbids `sin`/`cos`/`atan` on `f32`). Rotations are applied to both a
//! working copy of `A` and an accumulating basis `V`, so the columns of `V` are
//! always eigenvectors of the original matrix by construction, independent of any
//! sign convention.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The cyclic
//! Jacobi eigenvalue algorithm and its square-root rotation formula are standard,
//! publicly documented linear-algebra results (Golub & Van Loan, "Matrix
//! Computations").

use crate::math::scalar::Real;

/// The eigenvalue / eigenvector decomposition of a symmetric matrix.
///
/// Pairs are sorted by ascending eigenvalue, which for a stiffness matrix orders
/// the modes from the softest (lowest frequency) upward — exactly the order a
/// reduced model keeps.
#[derive(Clone, PartialEq, Debug)]
pub struct SymmetricEigen {
    /// Eigenvalues in ascending order; `values[k]` matches column `k` of
    /// [`vectors`](Self::vectors).
    pub values: Vec<Real>,
    /// Column-major eigenvectors: `vectors[k]` is the unit eigenvector for
    /// `values[k]`, stored as an `n`-length column.
    pub vectors: Vec<Vec<Real>>,
}

/// A dense symmetric matrix stored row-major in an `n * n` buffer.
///
/// Only symmetric matrices are meaningful here; callers are expected to fill it
/// symmetrically (the Jacobi sweep reads the full buffer but assumes symmetry).
#[derive(Clone, PartialEq, Debug)]
pub struct SymmetricMatrix {
    /// Dimension `n`.
    n: usize,
    /// Row-major `n * n` entries.
    data: Vec<Real>,
}

impl SymmetricMatrix {
    /// Creates an `n * n` zero matrix.
    #[must_use]
    pub fn zeros(n: usize) -> SymmetricMatrix {
        SymmetricMatrix {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// Returns the dimension `n`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the entry at `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Real {
        self.data[row * self.n + col]
    }

    /// Sets the entry at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, value: Real) {
        self.data[row * self.n + col] = value;
    }

    /// Adds `value` to the entry at `(row, col)`.
    pub fn add(&mut self, row: usize, col: usize, value: Real) {
        self.data[row * self.n + col] += value;
    }

    /// Adds `value` symmetrically to `(row, col)` and `(col, row)`.
    ///
    /// For `row == col` the diagonal entry is incremented once.
    pub fn add_symmetric(&mut self, row: usize, col: usize, value: Real) {
        if row == col {
            self.add(row, col, value);
        } else {
            self.add(row, col, value);
            self.add(col, row, value);
        }
    }

    /// Computes the full eigendecomposition with cyclic Jacobi rotations.
    ///
    /// `max_sweeps` bounds the number of full off-diagonal sweeps; convergence is
    /// declared when the off-diagonal Frobenius norm falls below `tolerance`.
    /// Both are given sensible defaults by [`Self::eigen`].
    #[must_use]
    pub fn eigen_with(&self, max_sweeps: usize, tolerance: Real) -> SymmetricEigen {
        let n = self.n;
        let mut a = self.data.clone();
        // V starts as the identity and accumulates the rotations.
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        for _ in 0..max_sweeps {
            if off_diagonal_norm(&a, n) <= tolerance {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq.abs() <= Real::MIN_POSITIVE {
                        continue;
                    }
                    let app = a[p * n + p];
                    let aqq = a[q * n + q];
                    let theta = (aqq - app) / (2.0 * apq);
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    apply_rotation(&mut a, n, p, q, c, s);
                    accumulate(&mut v, n, p, q, c, s);
                }
            }
        }

        // Extract eigenvalues (diagonal of the rotated matrix) and eigenvectors
        // (columns of V), then sort ascending.
        let mut pairs: Vec<(Real, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(core::cmp::Ordering::Equal));

        let mut values = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n);
        for (value, col) in pairs {
            values.push(value);
            let mut vector = Vec::with_capacity(n);
            for row in 0..n {
                vector.push(v[row * n + col]);
            }
            vectors.push(vector);
        }
        SymmetricEigen { values, vectors }
    }

    /// Computes the eigendecomposition with default sweep and tolerance limits.
    #[must_use]
    pub fn eigen(&self) -> SymmetricEigen {
        // 100 sweeps and a tight tolerance converge tiny modal problems fully;
        // Jacobi converges quadratically once the matrix is nearly diagonal.
        self.eigen_with(100, 1.0e-9)
    }
}

/// Returns the Frobenius norm of the strictly upper off-diagonal entries.
fn off_diagonal_norm(a: &[Real], n: usize) -> Real {
    let mut sum = 0.0;
    for p in 0..n {
        for q in (p + 1)..n {
            let value = a[p * n + q];
            sum += value * value;
        }
    }
    sum.sqrt()
}

/// Applies the two-sided Jacobi rotation `J^T A J` in the `(p, q)` plane.
fn apply_rotation(a: &mut [Real], n: usize, p: usize, q: usize, c: Real, s: Real) {
    // Rotate columns p and q.
    for i in 0..n {
        let aip = a[i * n + p];
        let aiq = a[i * n + q];
        a[i * n + p] = c * aip - s * aiq;
        a[i * n + q] = s * aip + c * aiq;
    }
    // Rotate rows p and q.
    for j in 0..n {
        let apj = a[p * n + j];
        let aqj = a[q * n + j];
        a[p * n + j] = c * apj - s * aqj;
        a[q * n + j] = s * apj + c * aqj;
    }
}

/// Accumulates the rotation into the eigenvector matrix: `V <- V J`.
fn accumulate(v: &mut [Real], n: usize, p: usize, q: usize, c: Real, s: Real) {
    for i in 0..n {
        let vip = v[i * n + p];
        let viq = v[i * n + q];
        v[i * n + p] = c * vip - s * viq;
        v[i * n + q] = s * vip + c * viq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(m: &SymmetricMatrix, x: &[Real]) -> Vec<Real> {
        let n = m.dim();
        let mut y = vec![0.0; n];
        for row in 0..n {
            let mut acc = 0.0;
            for col in 0..n {
                acc += m.get(row, col) * x[col];
            }
            y[row] = acc;
        }
        y
    }

    #[test]
    fn diagonal_matrix_returns_sorted_diagonal() {
        let mut m = SymmetricMatrix::zeros(3);
        m.set(0, 0, 5.0);
        m.set(1, 1, 1.0);
        m.set(2, 2, 3.0);
        let e = m.eigen();
        assert!((e.values[0] - 1.0).abs() < 1e-5);
        assert!((e.values[1] - 3.0).abs() < 1e-5);
        assert!((e.values[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn two_by_two_symmetric_matches_closed_form() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let mut m = SymmetricMatrix::zeros(2);
        m.set(0, 0, 2.0);
        m.set(1, 1, 2.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        let e = m.eigen();
        assert!((e.values[0] - 1.0).abs() < 1e-5);
        assert!((e.values[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn eigenvectors_satisfy_a_x_equals_lambda_x() {
        let mut m = SymmetricMatrix::zeros(3);
        // A symmetric, non-diagonal test matrix.
        m.set(0, 0, 4.0);
        m.set(1, 1, 2.0);
        m.set(2, 2, 3.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 2, -1.0);
        m.set(2, 1, -1.0);
        let e = m.eigen();
        for (lambda, x) in e.values.iter().zip(e.vectors.iter()) {
            let ax = matvec(&m, x);
            for i in 0..3 {
                assert!(
                    (ax[i] - lambda * x[i]).abs() < 1e-4,
                    "residual too large at {i}"
                );
            }
            // Eigenvectors are unit length.
            let norm2: Real = x.iter().map(|v| v * v).sum();
            assert!((norm2 - 1.0).abs() < 1e-4, "not unit length: {norm2}");
        }
    }

    #[test]
    fn add_symmetric_writes_both_triangles() {
        let mut m = SymmetricMatrix::zeros(2);
        m.add_symmetric(0, 1, 2.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 2.0);
        m.add_symmetric(0, 0, 5.0);
        assert_eq!(m.get(0, 0), 5.0);
    }
}
