//! Adapters bridging render-side guide-strand data to the authoritative
//! physics engine.
//!
//! The hair pipeline keeps its own compact [`StrandParticle`] layout (built on
//! the hand-rolled [`Vec3`] so the architecture crate stays allocation-light
//! and GPU-upload friendly), while [`prism_physics_core`] expresses every
//! solver as structure-of-arrays `glam` math. Rather than maintain a second
//! copy of the XPBD constraint arithmetic inside the hair module, the strand
//! solver converts through these helpers and projects through the single,
//! authoritative physics-engine implementation — the same delegation discipline
//! the render-side cloth modules already follow (see `cloth/physics_bridge.rs`).
//!
//! The conversions are deliberately trivial component copies; they exist so the
//! delegation sites read clearly and so pinned particles are mapped to a zero
//! inverse mass exactly once, in one place.

use alloc::vec::Vec;

use glam::Vec3 as GlamVec3;

use super::dynamics::{StrandParticle, Vec3};

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
/// engine solvers consume from a strand particle slice.
///
/// Pinned particles are mapped to a zero inverse mass so the physics projection
/// leaves them fixed, matching the strand convention that a pinned particle
/// never moves regardless of its stored `inverse_mass`.
#[must_use]
fn to_soa(particles: &[StrandParticle]) -> (Vec<GlamVec3>, Vec<f32>) {
    let mut positions = Vec::with_capacity(particles.len());
    let mut inverse_masses = Vec::with_capacity(particles.len());
    for particle in particles {
        positions.push(to_glam(particle.position));
        inverse_masses.push(if particle.is_pinned() {
            0.0
        } else {
            particle.inverse_mass.max(0.0)
        });
    }
    (positions, inverse_masses)
}

/// Writes solved `glam` positions back into a strand particle slice.
///
/// Only `position` is updated; `prev_position` and `inverse_mass` are
/// untouched. The slices are zipped, so a shorter `positions` leaves the
/// trailing particles unchanged (the physics projection only ever touches
/// referenced indices). A pinned particle's position round-trips exactly, so
/// this write never disturbs it.
fn write_positions_back(particles: &mut [StrandParticle], positions: &[GlamVec3]) {
    for (particle, position) in particles.iter_mut().zip(positions.iter()) {
        particle.position = from_glam(*position);
    }
}

/// Projects the compliant edge-length (distance) constraint over every segment
/// of one guide strand by delegating to the authoritative physics-engine XPBD
/// distance step.
///
/// Each segment `i` ties particle `i` to particle `i + 1` at target length
/// `rest_lengths[i]`; a missing entry (short slice) disables that segment.
/// `compliance` is the XPBD compliance and `dt` the substep length, so the
/// engine forms the same `alpha_tilde = compliance / dt^2` the strand solver
/// previously computed by hand. The strand runs a fresh (unwarmed) multiplier
/// per sweep, so each projection is driven with `lambda = 0.0` and its returned
/// multiplier is discarded — the single, shared arithmetic lives in
/// [`prism_physics_core::soft::constraint::project_distance_constraint`].
///
/// The projection is sequential (Gauss-Seidel): each segment sees the position
/// updates of the segments before it, exactly as the previous hand-rolled sweep
/// did. Pinned particles (zero inverse mass) absorb none of the correction.
pub(crate) fn solve_edges(
    particles: &mut [StrandParticle],
    rest_lengths: &[f32],
    compliance: f32,
    dt: f32,
) {
    let count = particles.len();
    if count < 2 {
        return;
    }
    let (mut positions, inverse_masses) = to_soa(particles);
    let mut i = 0;
    while i + 1 < count {
        if let Some(&rest) = rest_lengths.get(i) {
            let _ = prism_physics_core::soft::constraint::project_distance_constraint(
                &mut positions,
                &inverse_masses,
                i,
                i + 1,
                rest,
                compliance,
                0.0,
                dt,
            );
        }
        i += 1;
    }
    write_positions_back(particles, &positions);
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
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::pinned(Vec3::new(1.0, 0.0, 0.0)),
        ];
        let (positions, inverse_masses) = to_soa(&particles);
        assert_eq!(positions.len(), 2);
        assert!(inverse_masses[0] > 0.0);
        assert_eq!(inverse_masses[1], 0.0);
    }

    #[test]
    fn rigid_edge_restores_rest_length_between_free_particles() {
        // Two unit-mass particles pulled to length 2 with rest length 1 and
        // zero compliance snap back to the rest length in one projection.
        let mut particles = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        solve_edges(&mut particles, &[1.0], 0.0, 1.0 / 60.0);
        let length = particles[1].position.sub(particles[0].position).length();
        assert!((length - 1.0).abs() < 1.0e-4, "length was {length}");
    }

    #[test]
    fn pinned_root_absorbs_all_correction() {
        let mut particles = [
            StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        solve_edges(&mut particles, &[1.0], 0.0, 1.0 / 60.0);
        // Root never moves; the free tip is pulled to the rest length.
        assert_eq!(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert!((particles[1].position.x - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn missing_rest_length_disables_segment() {
        let mut particles = [
            StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
            StrandParticle::free(Vec3::new(2.0, 0.0, 0.0)),
        ];
        // Empty rest slice: nothing is projected, positions are unchanged.
        solve_edges(&mut particles, &[], 0.0, 1.0 / 60.0);
        assert_eq!(particles[1].position, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn short_strand_is_inert() {
        let mut single = [StrandParticle::free(Vec3::new(5.0, 0.0, 0.0))];
        solve_edges(&mut single, &[1.0], 0.0, 1.0 / 60.0);
        assert_eq!(single[0].position, Vec3::new(5.0, 0.0, 0.0));
    }
}
