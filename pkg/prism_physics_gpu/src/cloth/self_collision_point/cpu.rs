//! The `CPU` golden twin for the cloth point (vertex-vertex) self-collision
//! pass.
//!
//! The authoritative parallel-safe pass lives in [`prism_physics_core`] as
//! [`resolve_self_collision_jacobi`] (and its friction variant
//! [`resolve_self_collision_with_friction_jacobi`]): a frozen-snapshot Jacobi
//! reformulation of the Gauss-Seidel point self-collision core that gathers
//! every particle's own half of the separating push from a read-only snapshot
//! and applies the summed per-particle correction once. Rather than copy that
//! arithmetic (and risk it drifting from the engine), this twin *delegates* to
//! it over cloned columns and returns the applied particle positions, which is
//! exactly what the
//! [`GpuClothSelfCollisionPoint`](super::gpu::GpuClothSelfCollisionPoint) kernel
//! reproduces. The parity suite then compares the two within a tight tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted separation is standard position-based dynamics;
//! the tangential-friction projection is the one published by Macklin et al.
//! (2014); the Jacobi own-slot accumulate/apply split is standard parallel
//! position-based dynamics. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{
    resolve_self_collision_jacobi, resolve_self_collision_with_friction_jacobi,
};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Resolves every penetrating cloth-vs-cloth point pair with one parallel-safe
/// (Jacobi) self-collision iteration and returns the applied particle
/// positions.
///
/// This is the golden twin of
/// [`GpuClothSelfCollisionPoint::solve`](super::gpu::GpuClothSelfCollisionPoint::solve):
/// both reproduce the engine's own parallel-safe point self-collision pass. A
/// `friction` that sanitizes to `0` (non-finite, or `<= 0`) delegates to the
/// plain [`resolve_self_collision_jacobi`]; otherwise the friction variant is
/// used, consulting `prev_positions` for each partner's frame-start slide.
///
/// A non-positive `cell_size`/`thickness`, fewer than two particles, a
/// mismatched `inverse_masses` length, or a pinned particle (`inverse_mass <=
/// 0`) degrades exactly as the delegate does. The input slices are not mutated;
/// the returned `Vec` carries the post-pass positions.
#[must_use]
pub fn cpu_cloth_self_collision_point(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
    friction: Real,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    if friction.is_finite() && friction > 0.0 {
        resolve_self_collision_with_friction_jacobi(
            &mut out,
            prev_positions,
            inverse_masses,
            cell_size,
            thickness,
            friction,
        );
    } else {
        resolve_self_collision_jacobi(&mut out, inverse_masses, cell_size, thickness);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penetrating_pair_separates_to_thickness() {
        let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        let out = cpu_cloth_self_collision_point(&positions, &prev, &im, 2.0, 1.0, 0.0);
        let gap = out[0].distance(out[1]);
        assert!((gap - 1.0).abs() < 1e-4, "pair separated to thickness, got {gap}");
    }

    #[test]
    fn pinned_particle_never_moves() {
        let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let prev = positions;
        let im = [0.0, 1.0];
        let out = cpu_cloth_self_collision_point(&positions, &prev, &im, 2.0, 1.0, 0.0);
        assert_eq!(out[0], positions[0], "pinned particle must not move");
    }

    #[test]
    fn disabled_thickness_leaves_inputs_untouched() {
        let positions = [Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        let out = cpu_cloth_self_collision_point(&positions, &prev, &im, 2.0, 0.0, 0.0);
        assert_eq!(out, positions.to_vec());
    }
}
