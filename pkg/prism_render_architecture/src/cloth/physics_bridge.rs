//! Adapters bridging render-side cloth data to the authoritative physics engine.
//!
//! The render pipeline keeps its own compact particle layout
//! ([`ClothParticle`] with a hand-rolled [`Vec3`], chosen so the architecture
//! crate stays allocation-light and GPU-upload friendly), while
//! [`prism_physics_core`] expresses every solver as structure-of-arrays
//! `glam` math. Rather than maintain two copies of each solver, the render
//! cloth modules convert through these helpers and project through the single
//! physics-engine implementation.
//!
//! The conversions are deliberately trivial (component copies); they exist so
//! the delegation sites read clearly and so pinned particles are mapped to a
//! zero inverse mass exactly once, in one place.

use alloc::vec::Vec;

use glam::Vec3 as GlamVec3;

use super::{ClothParticle, Vec3};

/// Converts a render [`Vec3`] into a `glam` vector.
#[inline]
#[must_use]
pub(crate) fn to_glam(v: Vec3) -> GlamVec3 {
    GlamVec3::new(v.x, v.y, v.z)
}

/// Converts a `glam` vector back into a render [`Vec3`].
#[inline]
#[must_use]
pub(crate) fn from_glam(v: GlamVec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Extracts the structure-of-arrays `(positions, inverse_masses)` the physics
/// engine solvers consume from a render particle slice.
///
/// Pinned particles are mapped to a zero inverse mass so the physics projection
/// leaves them fixed, matching the render convention that a pinned particle
/// never moves regardless of its stored `inverse_mass`.
#[must_use]
pub(crate) fn to_soa(particles: &[ClothParticle]) -> (Vec<GlamVec3>, Vec<f32>) {
    let mut positions = Vec::with_capacity(particles.len());
    let mut inverse_masses = Vec::with_capacity(particles.len());
    for particle in particles {
        positions.push(to_glam(particle.position));
        inverse_masses.push(if particle.is_pinned() {
            0.0
        } else {
            particle.inverse_mass
        });
    }
    (positions, inverse_masses)
}

/// Writes solved `glam` positions back into a render particle slice.
///
/// Only `position` is updated; velocity and inverse mass are untouched. The
/// slices are zipped, so a shorter `positions` leaves the trailing particles
/// unchanged (the physics projection only ever touches referenced indices).
pub(crate) fn write_positions_back(particles: &mut [ClothParticle], positions: &[GlamVec3]) {
    for (particle, position) in particles.iter_mut().zip(positions.iter()) {
        particle.position = from_glam(*position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_vectors() {
        let v = Vec3::new(1.5, -2.0, 3.25);
        let back = from_glam(to_glam(v));
        assert_eq!(back, v);
    }

    #[test]
    fn pinned_particles_map_to_zero_inverse_mass() {
        let particles = [
            ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 2.0),
            ClothParticle::pinned(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let (positions, inverse_masses) = to_soa(&particles);
        assert_eq!(positions.len(), 2);
        assert!(inverse_masses[0] > 0.0);
        assert_eq!(inverse_masses[1], 0.0);
    }

    #[test]
    fn write_back_updates_only_position() {
        let mut particles = [ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0)];
        let before_inv = particles[0].inverse_mass;
        write_positions_back(&mut particles, &[GlamVec3::new(5.0, 6.0, 7.0)]);
        assert_eq!(particles[0].position, Vec3::new(5.0, 6.0, 7.0));
        assert_eq!(particles[0].inverse_mass, before_inv);
    }
}
