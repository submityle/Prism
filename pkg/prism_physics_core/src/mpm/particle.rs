//! Structure-of-Arrays storage for MPM material points.
//!
//! Each material point carries its position `x`, velocity `v`, affine-momentum
//! matrix `C` (the APIC velocity gradient), deformation gradient `F`, mass,
//! initial volume, and plastic determinant `Jp`. Columns are index-aligned so
//! the transfers can stream over them.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! material-point state (deformation gradient, affine matrix, plastic
//! determinant) follows Stomakhin et al. 2013 and Hu et al. 2018.

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

/// A Structure-of-Arrays store of MPM material points.
///
/// All columns share the same length; index `p` refers to one particle across
/// every column.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialPoints {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine: Vec<Mat3>,
    deformation: Vec<Mat3>,
    masses: Vec<Real>,
    volumes: Vec<Real>,
    plastic_det: Vec<Real>,
}

impl MaterialPoints {
    /// Creates an empty store.
    #[must_use]
    pub const fn new() -> MaterialPoints {
        MaterialPoints {
            positions: Vec::new(),
            velocities: Vec::new(),
            affine: Vec::new(),
            deformation: Vec::new(),
            masses: Vec::new(),
            volumes: Vec::new(),
            plastic_det: Vec::new(),
        }
    }

    /// Creates an empty store pre-allocated for `capacity` particles.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> MaterialPoints {
        MaterialPoints {
            positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            affine: Vec::with_capacity(capacity),
            deformation: Vec::with_capacity(capacity),
            masses: Vec::with_capacity(capacity),
            volumes: Vec::with_capacity(capacity),
            plastic_det: Vec::with_capacity(capacity),
        }
    }

    /// Spawns a material point at rest (`v = 0`, `C = 0`, `F = I`, `Jp = 1`)
    /// with the given `mass` and reference `volume`, returning its index.
    pub fn spawn(&mut self, position: Vec3, velocity: Vec3, mass: Real, volume: Real) -> usize {
        let idx = self.positions.len();
        self.positions.push(position);
        self.velocities.push(velocity);
        self.affine.push(Mat3::ZERO);
        self.deformation.push(Mat3::IDENTITY);
        self.masses.push(mass);
        self.volumes.push(volume);
        self.plastic_det.push(1.0);
        idx
    }

    /// The number of material points.
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

    /// The per-particle deformation gradients.
    #[inline]
    #[must_use]
    pub fn deformation(&self) -> &[Mat3] {
        &self.deformation
    }

    /// Mutable access to the deformation gradients.
    #[inline]
    pub fn deformation_mut(&mut self) -> &mut [Mat3] {
        &mut self.deformation
    }

    /// The particle masses.
    #[inline]
    #[must_use]
    pub fn masses(&self) -> &[Real] {
        &self.masses
    }

    /// The particle reference volumes.
    #[inline]
    #[must_use]
    pub fn volumes(&self) -> &[Real] {
        &self.volumes
    }

    /// The per-particle plastic determinants `Jp`.
    #[inline]
    #[must_use]
    pub fn plastic_det(&self) -> &[Real] {
        &self.plastic_det
    }

    /// Mutable access to the plastic determinants.
    #[inline]
    pub fn plastic_det_mut(&mut self) -> &mut [Real] {
        &mut self.plastic_det
    }

    /// The total mass of all particles.
    #[must_use]
    pub fn total_mass(&self) -> Real {
        self.masses.iter().copied().sum()
    }

    /// The total linear momentum `Σ mᵢ vᵢ` of all particles.
    #[must_use]
    pub fn total_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for i in 0..self.positions.len() {
            p += self.masses[i] * self.velocities[i];
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_initializes_rest_state() {
        let mut mp = MaterialPoints::new();
        let i = mp.spawn(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, 0.5, 0.001);
        assert_eq!(i, 0);
        assert_eq!(mp.len(), 1);
        assert_eq!(mp.deformation()[0], Mat3::IDENTITY);
        assert_eq!(mp.affine()[0], Mat3::ZERO);
        assert_eq!(mp.plastic_det()[0], 1.0);
    }

    #[test]
    fn totals_are_correct() {
        let mut mp = MaterialPoints::new();
        mp.spawn(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 2.0, 1.0);
        mp.spawn(Vec3::ZERO, Vec3::new(0.0, 3.0, 0.0), 4.0, 1.0);
        assert!((mp.total_mass() - 6.0).abs() < 1.0e-6);
        let p = mp.total_momentum();
        assert!((p.x - 2.0).abs() < 1.0e-6);
        assert!((p.y - 12.0).abs() < 1.0e-6);
    }
}
