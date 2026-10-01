//! The `CPU` golden twin for the cloth two-way coupling kernel.
//!
//! The authoritative two-way contact lives in [`prism_physics_core`] as
//! `resolve_two_way_coupling` (built on the shared scalar kernel
//! [`prism_physics_core::couple_particle_against_body`]). Rather than copy that
//! mass-weighted split and reaction-impulse arithmetic (and risk it drifting
//! from the engine), this twin *delegates* to it over cloned columns and
//! returns the applied particle positions together with the mutated bodies,
//! which is exactly what the [`GpuClothCoupling`](super::gpu::GpuClothCoupling)
//! kernel reproduces. The parity suite then compares the two within a tight
//! tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted contact split and the Newton reaction impulse are
//! textbook position-based-dynamics / rigid-body contact mechanics. No Unreal
//! Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{resolve_two_way_coupling, CouplingBody};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Resolves two-way particle/rigid contact for every particle against every
/// body and returns the applied particle positions together with the mutated
/// bodies (translated colliders and accumulated reaction impulses).
///
/// This is the golden twin of
/// [`GpuClothCoupling::solve`](super::gpu::GpuClothCoupling::solve): both
/// delegate the arithmetic to `prism_physics_core`'s `resolve_two_way_coupling`,
/// so the result is identical to the engine's own coupling pass.
///
/// A non-positive `dt`, an empty particle or body list, or an `inverse_masses`
/// slice shorter than `positions` all degrade exactly as the delegate does
/// (particles beyond the shorter of the two slices are ignored). The input
/// `bodies` are not mutated; the returned `Vec<CouplingBody>` carries the
/// post-pass state.
#[must_use]
pub fn cpu_cloth_coupling(
    positions: &[Vec3],
    inverse_masses: &[Real],
    bodies: &[CouplingBody],
    dt: Real,
) -> (Vec<Vec3>, Vec<CouplingBody>) {
    let mut out_positions = positions.to_vec();
    let mut out_bodies = bodies.to_vec();
    resolve_two_way_coupling(&mut out_positions, inverse_masses, &mut out_bodies, dt);
    (out_positions, out_bodies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::BodyCollider;

    fn unit_sphere() -> BodyCollider {
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    }

    #[test]
    fn delegates_to_the_engine_pass() {
        let positions = [Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.0, 0.4, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        let (pos, out) = cpu_cloth_coupling(&positions, &inverse_masses, &bodies, 1.0 / 60.0);
        // Both particles were inside and got pushed partway out.
        assert!(pos[0].x > 0.5);
        assert!(pos[1].y > 0.4);
        // The body took a reaction impulse and moved.
        assert_ne!(out[0].reaction_impulse, Vec3::ZERO);
    }

    #[test]
    fn empty_bodies_leave_positions_untouched() {
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let (pos, out) = cpu_cloth_coupling(&positions, &inverse_masses, &[], 1.0 / 60.0);
        assert_eq!(pos, positions.to_vec());
        assert!(out.is_empty());
    }

    #[test]
    fn kinematic_body_holds_still_but_records_reaction() {
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let bodies = [CouplingBody::kinematic(unit_sphere())];
        let (pos, out) = cpu_cloth_coupling(&positions, &inverse_masses, &bodies, 1.0 / 60.0);
        // Kinematic sphere never moves.
        assert_eq!(out[0].collider, unit_sphere());
        // Particle pushed fully to the surface.
        assert!((pos[0].x - 1.0).abs() < 1e-5);
        // Reaction still recorded.
        assert_ne!(out[0].reaction_impulse, Vec3::ZERO);
    }
}
