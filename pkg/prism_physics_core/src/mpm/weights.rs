//! Quadratic B-spline interpolation weights for the MLS-MPM transfers.
//!
//! MLS-MPM uses the quadratic B-spline kernel, whose support is exactly three
//! grid nodes per axis (27 nodes in 3D). For a particle at grid coordinate
//! `x/dx`, the base node is `floor(x/dx − 0.5)` and the three per-axis weights
//! are evaluated from the fractional offset `fx = x/dx − base ∈ [0.5, 1.5]`.
//! The kernel reproduces constants exactly (weights sum to one), which is what
//! makes the affine (APIC) transfer conservative.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! quadratic B-spline weights and their gradients are the standard MLS-MPM
//! kernels (Hu et al. 2018; Steffen et al. 2008).

use glam::Vec3;

use crate::math::scalar::Real;

/// The per-axis quadratic B-spline weights and their derivatives for one
/// particle, together with the integer base node and fractional offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadraticWeights {
    /// The base grid node index per axis (`floor(x/dx − 0.5)`).
    pub base: [i32; 3],
    /// The fractional offset `fx = x/dx − base` per axis, in `[0.5, 1.5]`.
    pub fx: Vec3,
    /// Per-axis weights for local offsets 0, 1, 2. `weight[axis][offset]`.
    pub weight: [[Real; 3]; 3],
    /// Per-axis weight derivatives w.r.t. world position, `dweight[axis][off]`.
    pub dweight: [[Real; 3]; 3],
}

impl QuadraticWeights {
    /// Computes the quadratic weights for a particle at world position `x` on
    /// a grid with spacing `dx` and origin `origin`.
    #[must_use]
    pub fn new(x: Vec3, origin: Vec3, dx: Real) -> QuadraticWeights {
        let inv_dx = 1.0 / dx;
        let cell = (x - origin) * inv_dx;
        // base = floor(cell - 0.5)
        let base = [
            (cell.x - 0.5).floor() as i32,
            (cell.y - 0.5).floor() as i32,
            (cell.z - 0.5).floor() as i32,
        ];
        let fx = Vec3::new(
            cell.x - base[0] as Real,
            cell.y - base[1] as Real,
            cell.z - base[2] as Real,
        );
        let mut weight = [[0.0 as Real; 3]; 3];
        let mut dweight = [[0.0 as Real; 3]; 3];
        let fxa = [fx.x, fx.y, fx.z];
        for axis in 0..3 {
            let f = fxa[axis];
            let a = 1.5 - f;
            let b = f - 1.0;
            let c = f - 0.5;
            weight[axis][0] = 0.5 * a * a;
            weight[axis][1] = 0.75 - b * b;
            weight[axis][2] = 0.5 * c * c;
            // Derivatives w.r.t. world position (chain rule: d/dx = inv_dx d/df).
            dweight[axis][0] = (f - 1.5) * inv_dx;
            dweight[axis][1] = -2.0 * b * inv_dx;
            dweight[axis][2] = c * inv_dx;
        }
        QuadraticWeights {
            base,
            fx,
            weight,
            dweight,
        }
    }

    /// Returns the scalar weight `N(x_i − x_p)` for local offset `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn value(&self, i: usize, j: usize, k: usize) -> Real {
        self.weight[0][i] * self.weight[1][j] * self.weight[2][k]
    }

    /// Returns the weight gradient `∇N` for local offset `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn gradient(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            self.dweight[0][i] * self.weight[1][j] * self.weight[2][k],
            self.weight[0][i] * self.dweight[1][j] * self.weight[2][k],
            self.weight[0][i] * self.weight[1][j] * self.dweight[2][k],
        )
    }

    /// Returns the vector from the particle to grid node `(i, j, k)` in world
    /// units: `x_node − x_p = (offset − fx) · dx`.
    #[inline]
    #[must_use]
    pub fn dpos(&self, i: usize, j: usize, k: usize, dx: Real) -> Vec3 {
        Vec3::new(
            (i as Real - self.fx.x) * dx,
            (j as Real - self.fx.y) * dx,
            (k as Real - self.fx.z) * dx,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_partition_unity() {
        let w = QuadraticWeights::new(Vec3::new(0.37, 1.23, -0.44), Vec3::ZERO, 0.1);
        for axis in 0..3 {
            let sum = w.weight[axis][0] + w.weight[axis][1] + w.weight[axis][2];
            assert!((sum - 1.0).abs() < 1.0e-6);
        }
        let mut total = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    total += w.value(i, j, k);
                }
            }
        }
        assert!((total - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn weight_gradients_sum_to_zero() {
        // A partition of unity has gradients summing to zero.
        let w = QuadraticWeights::new(Vec3::new(0.52, -0.13, 0.91), Vec3::ZERO, 0.2);
        let mut g = Vec3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    g += w.gradient(i, j, k);
                }
            }
        }
        assert!(g.length() < 1.0e-5);
    }

    #[test]
    fn dpos_matches_node_positions() {
        let dx = 0.1;
        let x = Vec3::new(0.34, 0.0, 0.0);
        let w = QuadraticWeights::new(x, Vec3::ZERO, dx);
        for i in 0..3 {
            let node_x = (w.base[0] + i as i32) as Real * dx;
            let expected = node_x - x.x;
            assert!((w.dpos(i, 0, 0, dx).x - expected).abs() < 1.0e-6);
        }
    }
}
