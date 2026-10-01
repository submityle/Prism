//! The `CPU` golden twins for the cloth body-collision and backstop kernels.
//!
//! The authoritative projections live in [`prism_physics_core`] as the
//! `resolve_body_collisions_with_friction` and `resolve_backstops` free
//! functions. Rather than copy that analytic sphere/capsule/half-space and
//! Coulomb-friction arithmetic (and risk it drifting from the engine), these
//! twins *delegate* to those functions over cloned columns and return the
//! applied positions, which is exactly what the
//! [`GpuClothBodyCollision`](super::gpu::GpuClothBodyCollision) kernel runs. The
//! parity suite then compares the two applied-position fields within a tight
//! tolerance.
//!
//! Body collision is *per-particle independent* (each particle resolves itself
//! against every collider in slice order), so unlike the batched or global
//! sibling kernels there is no colouring and the device maps one particle to one
//! thread. A single [`cpu_cloth_body_collision`] entry covers both the plain and
//! the friction path: `mu == 0` makes `resolve_body_collisions_with_friction`
//! delegate straight to the plain resolver.
//!
//! # Provenance
//!
//! The analytic body-proxy projections and the one-sided backstop are standard
//! position-based collision techniques; the tangential-friction projection is
//! the one published by Macklin et al. (2014), "Unified Particle Physics for
//! Real-Time Applications". No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{
    resolve_backstops, resolve_body_collisions_with_friction, Backstop, BodyCollider,
};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Projects every free particle out of every body collider, in slice order,
/// rubbing the tangential slide of each contact with position-level Coulomb
/// friction, and returns the applied positions.
///
/// This is the golden twin of
/// [`GpuClothBodyCollision::solve`](super::gpu::GpuClothBodyCollision::solve):
/// both delegate the arithmetic to `prism_physics_core`'s
/// `resolve_body_collisions_with_friction`, so the result is identical to the
/// engine's own body-proxy pass. The one entry covers both the plain and the
/// friction path: `mu <= 0` makes the delegate fall through to the plain
/// resolver (so `prev_positions` is then irrelevant).
///
/// An empty `colliders` slice, or an `inverse_masses` slice whose length differs
/// from `positions`, leaves `positions` unchanged; pinned particles
/// (`inverse_mass <= 0`) never move, and a `prev_positions` slice shorter than
/// `positions` degrades to no tangential slide (hence no friction) for the
/// missing indices, exactly as in the delegate.
#[must_use]
pub fn cpu_cloth_body_collision(
    positions: &[Vec3],
    inverse_masses: &[Real],
    prev_positions: &[Vec3],
    colliders: &[BodyCollider],
    mu: Real,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    resolve_body_collisions_with_friction(&mut out, prev_positions, inverse_masses, colliders, mu);
    out
}

/// Applies each per-particle backstop plane to its matching particle and
/// returns the applied positions.
///
/// This is the golden twin of
/// [`GpuClothBodyCollision::solve_backstops`](super::gpu::GpuClothBodyCollision::solve_backstops):
/// both delegate to `prism_physics_core`'s `resolve_backstops`. The pass runs
/// over the shorter of the position and backstop lengths; pinned particles are
/// skipped, and an empty `backstops` slice or a mismatched `inverse_masses`
/// length leaves `positions` unchanged.
#[must_use]
pub fn cpu_cloth_backstops(
    positions: &[Vec3],
    inverse_masses: &[Real],
    backstops: &[Backstop],
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    resolve_backstops(&mut out, inverse_masses, backstops);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TOL: Real = 1.0e-6;

    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a - b).length() < TOL, "{a:?} != {b:?}");
    }

    #[test]
    fn sphere_pushes_interior_point_to_surface() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn capsule_pushes_point_out_perpendicular_to_axis() {
        let colliders = [BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        }];
        let positions = [Vec3::new(0.0, 0.5, 0.0)];
        let inverse_masses = [1.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn half_space_pushes_infeasible_point_to_plane() {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        }];
        let positions = [Vec3::new(0.3, -0.5, 0.2)];
        let inverse_masses = [1.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], Vec3::new(0.3, 0.0, 0.2));
    }

    #[test]
    fn multiple_colliders_resolve_last_wins() {
        // Two half-spaces: y >= 0 then y >= 1. A point below both is projected
        // by the first, then by the second (last wins): ends on y = 1.
        let colliders = [
            BodyCollider::HalfSpace {
                normal: Vec3::Y,
                offset: 0.0,
            },
            BodyCollider::HalfSpace {
                normal: Vec3::Y,
                offset: 1.0,
            },
        ];
        let positions = [Vec3::new(0.2, -0.5, 0.3)];
        let inverse_masses = [1.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], Vec3::new(0.2, 1.0, 0.3));
    }

    #[test]
    fn friction_cancels_tangential_slide_inside_cone() {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        }];
        let positions = [Vec3::new(0.01, -0.5, 0.0)];
        let prev = [Vec3::new(0.0, -0.5, 0.0)];
        let inverse_masses = [1.0];
        let out = cpu_cloth_body_collision(&positions, &inverse_masses, &prev, &colliders, 1.0);
        assert!(out[0].y.abs() < TOL, "y: {}", out[0].y);
        assert!((out[0].x - prev[0].x).abs() < TOL, "x: {}", out[0].x);
    }

    #[test]
    fn pinned_particle_never_moves() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], positions[0]);
    }

    #[test]
    fn empty_colliders_is_noop() {
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let out = cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &[], 0.5);
        approx_eq(out[0], positions[0]);
    }

    #[test]
    fn mismatched_inverse_mass_is_noop() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let out =
            cpu_cloth_body_collision(&positions, &inverse_masses, &positions, &colliders, 0.0);
        approx_eq(out[0], positions[0]);
    }

    #[test]
    fn backstop_pushes_point_behind_plane_forward() {
        let backstops = [Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.1,
        }];
        let positions = [Vec3::new(0.2, -0.5, 0.3)];
        let inverse_masses = [1.0];
        let out = cpu_cloth_backstops(&positions, &inverse_masses, &backstops);
        approx_eq(out[0], Vec3::new(0.2, -0.1, 0.3));
    }

    #[test]
    fn backstop_skips_pinned_and_short_slice() {
        let backstops = [
            Backstop {
                origin: Vec3::ZERO,
                normal: Vec3::Y,
                distance: 0.0,
            },
            Backstop {
                origin: Vec3::ZERO,
                normal: Vec3::Y,
                distance: 0.0,
            },
        ];
        let positions = vec![
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ];
        let inverse_masses = [0.0, 1.0, 1.0];
        let out = cpu_cloth_backstops(&positions, &inverse_masses, &backstops);
        approx_eq(out[0], Vec3::new(0.0, -0.5, 0.0));
        approx_eq(out[1], Vec3::ZERO);
        approx_eq(out[2], Vec3::new(0.0, -0.5, 0.0));
    }
}
