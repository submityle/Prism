//! The `CPU` golden twin for the cloth continuous self-collision (self-CCD)
//! sweep.
//!
//! The authoritative parallel-safe sweep lives in [`prism_physics_core`] as
//! [`resolve_self_ccd_jacobi`]: a frozen-snapshot Jacobi reformulation of the
//! Gauss-Seidel [`resolve_self_ccd`](prism_physics_core::resolve_self_ccd) core
//! that resolves every tunnelling cloth-vs-cloth pair from a read-only snapshot
//! and applies the summed per-particle correction once. Rather than copy that
//! arithmetic (and risk it drifting from the engine), this twin *delegates* to
//! it over cloned columns and returns the applied particle positions together
//! with the mutated velocities, which is exactly what the
//! [`GpuClothSelfCcd`](super::gpu::GpuClothSelfCcd) kernel reproduces. The
//! parity suite then compares the two within a tight tolerance.
//!
//! # Provenance
//!
//! The closed-form swept-pair time-of-impact resolution is standard analytic
//! continuous-collision geometry; the Jacobi own-slot accumulate/apply split is
//! standard parallel position-based dynamics. No Unreal Engine source or
//! derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{resolve_self_ccd_jacobi, SelfCcdParams};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Resolves every tunnelling cloth-vs-cloth pair with one parallel-safe
/// (Jacobi) continuous self-collision iteration and returns the applied
/// particle positions together with the mutated velocities.
///
/// This is the golden twin of
/// [`GpuClothSelfCcd::solve`](super::gpu::GpuClothSelfCcd::solve): both delegate
/// the arithmetic to [`prism_physics_core::resolve_self_ccd_jacobi`], so the
/// result is identical to the engine's own parallel-safe self-CCD pass.
///
/// A disabled [`SelfCcdParams`], a non-positive `thickness`, fewer than two
/// particles, or a pinned particle (`inverse_mass <= 0`) degrades exactly as
/// the delegate does. The input slices are not mutated; the returned `Vec`s
/// carry the post-sweep state. Only indices valid in every read-only input are
/// swept (the delegate takes the shortest of `positions`, `prev_positions`, and
/// `inverse_masses`), and velocities are written only where the velocity column
/// is long enough.
#[must_use]
pub fn cpu_cloth_self_ccd(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[Real],
    params: SelfCcdParams,
    dt: Real,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut out_positions = positions.to_vec();
    let mut out_velocities = velocities.to_vec();
    resolve_self_ccd_jacobi(
        &mut out_positions,
        prev_positions,
        &mut out_velocities,
        inverse_masses,
        params,
        dt,
    );
    (out_positions, out_velocities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(thickness: Real) -> SelfCcdParams {
        SelfCcdParams::new(0.2, thickness)
    }

    #[test]
    fn delegates_to_the_engine_sweep() {
        // Two particles swap sides in one step; the self-CCD pass must catch the
        // crossing and separate them to the thickness gap.
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [1.0, 1.0];
        let (pos, _vel) = cpu_cloth_self_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            params(0.4),
            1.0,
        );
        let separation = pos[0].distance(pos[1]);
        assert!(
            (separation - 0.4).abs() < 1e-4,
            "pair must be separated to thickness, got {separation}"
        );
    }

    #[test]
    fn pinned_particle_never_moves() {
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let velocities = [Vec3::ZERO, Vec3::ZERO];
        let inverse_masses = [0.0, 1.0];
        let (pos, _vel) = cpu_cloth_self_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            params(0.4),
            1.0,
        );
        assert_eq!(pos[0], positions[0], "pinned particle must not move");
    }

    #[test]
    fn disabled_sweep_leaves_inputs_untouched() {
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev_positions = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let velocities = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let mut p = params(0.4);
        p.enabled = false;
        let (pos, vel) = cpu_cloth_self_ccd(
            &positions,
            &prev_positions,
            &velocities,
            &inverse_masses,
            p,
            1.0,
        );
        assert_eq!(pos, positions.to_vec());
        assert_eq!(vel, velocities.to_vec());
    }
}
