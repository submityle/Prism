//! Structure-of-Arrays storage for FLIP/APIC marker particles.
//!
//! Marker particles carry a world position and a velocity; when the APIC
//! transfer is enabled they additionally carry the affine velocity matrix `C`
//! (the local velocity gradient) so that angular momentum is preserved across
//! the grid transfer. Columns are index-aligned so the transfers can stream
//! over them contiguously.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! marker-particle representation and the APIC affine matrix follow Zhu &
//! Bridson 2005 and Jiang et al. 2015, as summarised by Bridson, *Fluid
//! Simulation for Computer Graphics*.

use glam::{Mat3, Vec3};

/// A Structure-of-Arrays store of fluid marker particles.
///
/// All columns share the same length; index `p` refers to one particle across
/// every column.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MarkerParticles {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine: Vec<Mat3>,
}

impl MarkerParticles {
    /// Creates an empty store.
    #[must_use]
    pub const fn new() -> MarkerParticles {
        MarkerParticles {
            positions: Vec::new(),
            velocities: Vec::new(),
            affine: Vec::new(),
        }
    }

    /// Creates an empty store pre-allocated for `capacity` particles.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> MarkerParticles {
        MarkerParticles {
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

    /// The per-particle affine (APIC) matrices.
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
    fn spawn_initializes_state() {
        let mut mp = MarkerParticles::new();
        let i = mp.spawn(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(i, 0);
        assert_eq!(mp.len(), 1);
        assert!(!mp.is_empty());
        assert_eq!(mp.positions()[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(mp.velocities()[0], Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(mp.affine()[0], Mat3::ZERO);
    }

    #[test]
    fn capacity_constructor_is_empty() {
        let mp = MarkerParticles::with_capacity(16);
        assert!(mp.is_empty());
        assert_eq!(mp.len(), 0);
    }
}
