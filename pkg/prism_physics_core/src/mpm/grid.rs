//! The transient MLS-MPM background grid.
//!
//! The grid is a regular lattice of nodes storing accumulated mass and
//! momentum during the particle-to-grid (P2G) scatter. After scatter the
//! momentum is converted to velocity, external forces are applied, and wall
//! boundary conditions are enforced before the grid-to-particle (G2P) gather.
//! The grid holds no persistent state between steps: it is cleared and rebuilt
//! every step, which is the defining feature of the Material Point Method.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Eulerian background grid and its index/coordinate mapping are standard MPM
//! constructs (Sulsky et al. 1994; Hu et al. 2018).

use glam::Vec3;

use crate::math::scalar::Real;

/// A regular grid of MPM background nodes.
///
/// Node `(i, j, k)` sits at world position `origin + (i, j, k)·dx` and is
/// stored at the flat index `i + nx·(j + ny·k)`. The `mass` and `momentum`
/// columns are index-aligned and are reset to zero every step by
/// [`Grid::clear`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Grid {
    /// Number of nodes along the x axis.
    nx: usize,
    /// Number of nodes along the y axis.
    ny: usize,
    /// Number of nodes along the z axis.
    nz: usize,
    /// Node spacing (uniform in all axes).
    dx: Real,
    /// World-space position of node `(0, 0, 0)`.
    origin: Vec3,
    /// Accumulated node mass.
    mass: Vec<Real>,
    /// Accumulated node momentum during P2G; node velocity after normalization.
    momentum: Vec<Vec3>,
}

impl Grid {
    /// Creates a zeroed grid with `(nx, ny, nz)` nodes, spacing `dx` and the
    /// given world-space `origin`.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Real, origin: Vec3) -> Grid {
        let count = nx * ny * nz;
        Grid {
            nx,
            ny,
            nz,
            dx,
            origin,
            mass: vec![0.0; count],
            momentum: vec![Vec3::ZERO; count],
        }
    }

    /// Number of nodes along the x axis.
    #[inline]
    #[must_use]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Number of nodes along the y axis.
    #[inline]
    #[must_use]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Number of nodes along the z axis.
    #[inline]
    #[must_use]
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// The uniform node spacing.
    #[inline]
    #[must_use]
    pub fn dx(&self) -> Real {
        self.dx
    }

    /// The world-space position of node `(0, 0, 0)`.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Returns `true` if `(i, j, k)` is a valid node index.
    #[inline]
    #[must_use]
    pub fn in_bounds(&self, i: i32, j: i32, k: i32) -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && (i as usize) < self.nx
            && (j as usize) < self.ny
            && (k as usize) < self.nz
    }

    /// Maps a valid node index to its flat storage index.
    #[inline]
    #[must_use]
    pub fn flat(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// The world-space position of node `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn node_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin + Vec3::new(i as Real, j as Real, k as Real) * self.dx
    }

    /// Resets all node mass and momentum to zero for a new step.
    pub fn clear(&mut self) {
        for m in &mut self.mass {
            *m = 0.0;
        }
        for p in &mut self.momentum {
            *p = Vec3::ZERO;
        }
    }

    /// Scatters `mass` and `momentum` into node `(i, j, k)` if it is in bounds.
    #[inline]
    pub fn accumulate(&mut self, i: i32, j: i32, k: i32, mass: Real, momentum: Vec3) {
        if !self.in_bounds(i, j, k) {
            return;
        }
        let idx = self.flat(i as usize, j as usize, k as usize);
        self.mass[idx] += mass;
        self.momentum[idx] += momentum;
    }

    /// The accumulated mass at a valid node index.
    #[inline]
    #[must_use]
    pub fn mass_at(&self, i: usize, j: usize, k: usize) -> Real {
        self.mass[self.flat(i, j, k)]
    }

    /// The node velocity (valid only after [`Grid::finalize_velocity`]).
    #[inline]
    #[must_use]
    pub fn velocity_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.momentum[self.flat(i, j, k)]
    }

    /// Converts accumulated momentum to velocity (`v = p/m`) for every node
    /// with positive mass, leaving empty nodes at zero velocity.
    pub fn finalize_velocity(&mut self) {
        for idx in 0..self.mass.len() {
            let m = self.mass[idx];
            if m > 0.0 {
                self.momentum[idx] /= m;
            } else {
                self.momentum[idx] = Vec3::ZERO;
            }
        }
    }

    /// Adds `dv` to the velocity of every node with positive mass.
    pub fn add_velocity_to_active(&mut self, dv: Vec3) {
        for idx in 0..self.mass.len() {
            if self.mass[idx] > 0.0 {
                self.momentum[idx] += dv;
            }
        }
    }

    /// Overwrites the velocity stored at a valid node index.
    #[inline]
    pub fn set_velocity(&mut self, i: usize, j: usize, k: usize, v: Vec3) {
        let idx = self.flat(i, j, k);
        self.momentum[idx] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_index_roundtrip() {
        let g = Grid::new(4, 5, 6, 0.1, Vec3::ZERO);
        assert_eq!(g.flat(0, 0, 0), 0);
        assert_eq!(g.flat(3, 4, 5), 3 + 4 * (4 + 5 * 5));
        assert!(g.in_bounds(3, 4, 5));
        assert!(!g.in_bounds(4, 4, 5));
        assert!(!g.in_bounds(-1, 0, 0));
    }

    #[test]
    fn accumulate_and_finalize() {
        let mut g = Grid::new(3, 3, 3, 1.0, Vec3::ZERO);
        g.accumulate(1, 1, 1, 2.0, Vec3::new(4.0, 0.0, 0.0));
        g.accumulate(1, 1, 1, 2.0, Vec3::new(0.0, 8.0, 0.0));
        g.finalize_velocity();
        let v = g.velocity_at(1, 1, 1);
        assert!((v.x - 1.0).abs() < 1.0e-6);
        assert!((v.y - 2.0).abs() < 1.0e-6);
        assert_eq!(g.mass_at(1, 1, 1), 4.0);
    }

    #[test]
    fn out_of_bounds_accumulate_is_ignored() {
        let mut g = Grid::new(2, 2, 2, 1.0, Vec3::ZERO);
        g.accumulate(5, 5, 5, 1.0, Vec3::ONE);
        g.finalize_velocity();
        assert_eq!(g.mass_at(0, 0, 0), 0.0);
    }
}
