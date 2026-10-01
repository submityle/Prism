//! The `CPU` golden twin for the cloth continuous-collision (CCD) sweep.
//!
//! The authoritative swept-collision sweep lives in [`prism_physics_core`] as
//! [`resolve_ccd`]: closed-form TOI solvers for sphere / capsule / half-space,
//! surface snap plus skin, restitution normal reflection, and position-level
//! Coulomb friction. Rather than copy that arithmetic (and risk it drifting
//! from the engine), this twin *delegates* to it over cloned columns and
//! returns the applied particle positions together with the mutated
//! velocities, which is exactly what the
//! [`GpuClothCcd`](super::gpu::GpuClothCcd) kernel reproduces. The parity suite
//! then compares the two within a tight tolerance.
//!
//! # Provenance
//!
//! The closed-form swept-primitive TOI solvers are standard analytic
//! continuous-collision geometry, and the tangential-friction projection reuses
//! the Macklin et al. (2014) primitive. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{resolve_ccd, BodyCollider, CcdParams};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Sweeps every free particle's `prev -> curr` motion against every collider
/// and returns the applied particle positions together with the mutated
/// velocities (restitution-reflected on a hit).
///
/// This is the golden twin of
/// [`GpuClothCcd::solve`](super::gpu::GpuClothCcd::solve): both delegate the
/// arithmetic to [`prism_physics_core::resolve_ccd`], so the result is
/// identical to the engine's own CCD pass.
///
/// A disabled [`CcdParams`], an empty collider list, or a pinned particle
/// (`inverse_mass <= 0`) degrades exactly as the delegate does. The input
/// slices are not mutated; the returned `Vec`s carry the post-sweep state. Only
/// indices valid in every read-only input are swept (the delegate takes the
/// shortest of `positions`, `prev_positions`, and `inverse_masses`), and
/// velocities are written only where the velocity column is long enough.
#[must_use]
pub fn cpu_cloth_ccd(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[Real],
    colliders: &[BodyCollider],
    params: CcdParams,
    dt: Real,
    friction: Real,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut out_positions = positions.to_vec();
    let mut out_velocities = velocities.to_vec();
    resolve_ccd(
        &mut out_positions,
        prev_positions,
        &mut out_velocities,
        inverse_masses,
        colliders,
        params,
        dt,
        friction,
    );
    (out_positions, out_velocities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere() -> BodyCollider {
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    }

    #[test]
    fn delegates_to_the_engine_sweep() {
        // A particle sweeps straight through a unit sphere in one step; the CCD
        // pass must catch the tunnelling and snap it back out to the surface.
        let positions = [Vec3::new(2.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-2.0, 0.0, 0.0)];
        let velocities = [Vec3::ZERO];
        let inverse_masses = [1.0];
        let (pos, _vel) = cpu_cloth_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            &[unit_sphere()],
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        // Placed on the entry side of the sphere (+skin), not left at x = 2.
        assert!(pos[0].x < 2.0);
        assert!((pos[0].length() - (1.0 + CcdParams::default().skin)).abs() < 1e-4);
    }

    #[test]
    fn pinned_particle_never_moves() {
        let positions = [Vec3::new(2.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-2.0, 0.0, 0.0)];
        let velocities = [Vec3::ZERO];
        let inverse_masses = [0.0];
        let (pos, _vel) = cpu_cloth_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            &[unit_sphere()],
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert_eq!(pos[0], positions[0]);
    }

    #[test]
    fn empty_colliders_leave_inputs_untouched() {
        let positions = [Vec3::new(2.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-2.0, 0.0, 0.0)];
        let velocities = [Vec3::new(1.0, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let (pos, vel) = cpu_cloth_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            &[],
            CcdParams::default(),
            1.0 / 60.0,
            0.0,
        );
        assert_eq!(pos, positions.to_vec());
        assert_eq!(vel, velocities.to_vec());
    }
}
