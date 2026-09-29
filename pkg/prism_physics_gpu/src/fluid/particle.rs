//! Structure-of-Arrays storage for `FLIP`/`APIC` marker particles.
//!
//! Marker particles carry a world position and a velocity; the `APIC` transfer
//! additionally carries the affine velocity matrix `C` (the local velocity
//! gradient) so angular momentum survives the grid transfer. Columns are
//! index-aligned so the transfers stream over them contiguously, matching the
//! `CPU` reference [`prism_physics_core`](prism_physics_core::fluid::particle).
//!
//! The `GPU` path uploads positions and velocities as `vec4<f32>` (`xyz` plus a
//! zero pad) for natural `std430` alignment; this store keeps the tidy [`Vec3`]
//! form and converts at the buffer boundary.
//!
//! # Provenance
//!
//! The marker-particle representation and the `APIC` affine matrix follow Zhu
//! and Bridson 2005 and Jiang et al. 2015. This module contains no Unreal
//! Engine source or derived code.

use glam::{Mat3, Vec3};

/// A Structure-of-Arrays store of fluid marker particles.
///
/// All columns share the same length; index `p` refers to one particle across
/// every column.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FluidParticles {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine: Vec<Mat3>,
}

impl FluidParticles {
    /// Creates an empty store.
    #[must_use]
    pub const fn new() -> FluidParticles {
        FluidParticles {
            positions: Vec::new(),
            velocities: Vec::new(),
            affine: Vec::new(),
        }
    }

    /// Creates an empty store pre-allocated for `capacity` particles.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> FluidParticles {
        FluidParticles {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            affine: Vec::with_capacity(capacity),
        }
    }

    /// Spawns a marker particle with the given position and velocity (and a
    /// zero affine matrix), returning its index.
    pub fn spawn(&mut self, position: Vec3, velocity: Vec3) -> usize {
        let idx = self.positions.len();
        self.positions.push(position);
        self.velocities.push(velocity);
        self.affine.push(Mat3::ZERO);
        idx
    }

    /// The number of marker particles.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the store holds no particles.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns `true` when every column has the same length.
    #[inline]
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.positions.len() == self.velocities.len() && self.positions.len() == self.affine.len()
    }

    /// The particle positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Mutable access to the particle positions.
    #[inline]
    pub fn positions_mut(&mut self) -> &mut [Vec3] {
        &mut self.positions
    }

    /// The particle velocities.
    #[inline]
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Mutable access to the particle velocities.
    #[inline]
    pub fn velocities_mut(&mut self) -> &mut [Vec3] {
        &mut self.velocities
    }

    /// The per-particle affine (`APIC`) matrices.
    #[inline]
    #[must_use]
    pub fn affine(&self) -> &[Mat3] {
        &self.affine
    }

    /// Mutable access to the affine matrices.
    #[inline]
    pub fn affine_mut(&mut self) -> &mut [Mat3] {
        &mut self.affine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_initialises_state() {
        let mut mp = FluidParticles::new();
        let i = mp.spawn(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(i, 0);
        assert_eq!(mp.len(), 1);
        assert!(!mp.is_empty());
        assert!(mp.is_consistent());
        assert_eq!(mp.positions()[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(mp.velocities()[0], Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(mp.affine()[0], Mat3::ZERO);
    }
}
