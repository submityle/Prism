//! The per-vertex local linear system solved by Vertex Block Descent.
//!
//! Vertex Block Descent advances one vertex at a time. For the vertex under
//! consideration it gathers the implicit-Euler inertial term and every incident
//! elastic-element contribution into a single `3x3` symmetric positive-definite
//! system
//!
//! ```text
//! H dx = f
//! ```
//!
//! where `f` is the total force (negative energy gradient) and `H` is the total
//! energy Hessian block for that vertex. Solving for `dx` and adding it to the
//! vertex position is one block-coordinate-descent update. Because the inertial
//! term contributes `(m / h^2) I` — a strictly positive multiple of the
//! identity — and every elastic Hessian block is projected to be
//! positive-semidefinite, `H` is always invertible, which is what makes the
//! descent unconditionally stable.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! per-vertex variational system (inertia plus summed elastic Hessians) is the
//! formulation published by Chen et al., "Vertex Block Descent" (2024).

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

use super::element::SpringContribution;

/// Accumulator for a single vertex's local `3x3` descent system.
///
/// Start from [`VertexSystem::new`], add the inertial target with
/// [`add_inertia`](Self::add_inertia), fold in each incident element with
/// [`add_spring`](Self::add_spring), then call [`solve`](Self::solve) to obtain
/// the position update `dx = H^{-1} f`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VertexSystem {
    /// Accumulated force (negative total energy gradient) on the vertex.
    force: Vec3,
    /// Accumulated `3x3` energy Hessian block for the vertex.
    hessian: Mat3,
}

impl VertexSystem {
    /// Creates an empty system with zero force and zero Hessian.
    #[must_use]
    pub const fn new() -> VertexSystem {
        VertexSystem {
            force: Vec3::ZERO,
            hessian: Mat3::ZERO,
        }
    }

    /// Adds the implicit-Euler inertial term for a vertex of mass `mass` over a
    /// substep of length `h`, pulling the current position `x` toward the
    /// inertial target `y = x_prev + h v + h^2 g`.
    ///
    /// The term contributes a force `-(m / h^2) (x - y)` and a Hessian
    /// `(m / h^2) I`. The `m / h^2` factor is strictly positive for a dynamic
    /// vertex, which is what guarantees the assembled system is invertible.
    pub fn add_inertia(&mut self, mass: Real, h: Real, x: Vec3, y: Vec3) {
        let coeff = mass / (h * h);
        self.force -= (x - y) * coeff;
        self.hessian += Mat3::IDENTITY * coeff;
    }

    /// Folds one incident spring's contribution into the system.
    pub fn add_spring(&mut self, contribution: SpringContribution) {
        self.force += contribution.force;
        self.hessian += contribution.hessian;
    }

    /// Returns the accumulated force on the vertex.
    #[must_use]
    pub const fn force(&self) -> Vec3 {
        self.force
    }

    /// Returns the accumulated Hessian block for the vertex.
    #[must_use]
    pub const fn hessian(&self) -> Mat3 {
        self.hessian
    }

    /// Solves the local system `H dx = f` for the position update `dx`.
    ///
    /// Returns [`Vec3::ZERO`] when the assembled Hessian is singular (which can
    /// only happen if no inertial term and no elastic element were added), so a
    /// vertex with nothing acting on it simply does not move.
    #[must_use]
    pub fn solve(&self) -> Vec3 {
        if self.hessian.determinant().abs() <= Real::EPSILON {
            return Vec3::ZERO;
        }
        self.hessian.inverse() * self.force
    }
}

impl Default for VertexSystem {
    fn default() -> Self {
        VertexSystem::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_system_does_not_move() {
        let s = VertexSystem::new();
        assert_eq!(s.solve(), Vec3::ZERO);
    }

    #[test]
    fn pure_inertia_recovers_target_offset() {
        // With only the inertial term, H = (m/h^2) I and f = -(m/h^2)(x - y),
        // so dx = -(x - y) = y - x: one step lands exactly on the target.
        let mut s = VertexSystem::new();
        let x = Vec3::new(1.0, 2.0, 3.0);
        let y = Vec3::new(4.0, 0.0, -1.0);
        s.add_inertia(2.5, 1.0 / 60.0, x, y);
        let dx = s.solve();
        assert!((x + dx - y).length() < 1e-4, "landed at {:?}", x + dx);
    }

    #[test]
    fn inertia_accumulates_diagonal_hessian() {
        let mut s = VertexSystem::new();
        s.add_inertia(1.0, 0.5, Vec3::ZERO, Vec3::ZERO);
        // m / h^2 = 1 / 0.25 = 4 on the diagonal.
        assert!((s.hessian().col(0).x - 4.0).abs() < 1e-6);
        assert!((s.hessian().col(1).y - 4.0).abs() < 1e-6);
        assert!((s.hessian().col(2).z - 4.0).abs() < 1e-6);
    }

    #[test]
    fn spring_contribution_adds_to_force_and_hessian() {
        let mut s = VertexSystem::new();
        s.add_spring(SpringContribution {
            force: Vec3::new(1.0, 0.0, 0.0),
            hessian: Mat3::IDENTITY * 3.0,
        });
        s.add_spring(SpringContribution {
            force: Vec3::new(0.0, 2.0, 0.0),
            hessian: Mat3::IDENTITY * 5.0,
        });
        assert_eq!(s.force(), Vec3::new(1.0, 2.0, 0.0));
        assert!((s.hessian().col(0).x - 8.0).abs() < 1e-6);
    }
}
