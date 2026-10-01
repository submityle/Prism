//! The `CPU` golden twin for the cloth pressure (closed-mesh volume) kernel.
//!
//! The authoritative compliant-`XPBD` pressure projection lives in
//! [`prism_physics_core`] as `project_pressure`; rather than copy that
//! divergence-theorem volume + gradient arithmetic and risk it drifting,
//! [`cpu_cloth_pressure`] *delegates* every iteration to it and only owns the
//! substep iteration loop, which is exactly what the
//! [`GpuClothPressure`](super::gpu::GpuClothPressure) kernel runs. The parity
//! suite then compares the two applied-position fields within a tight tolerance.
//!
//! Pressure is a single *global* constraint over the whole shell (not a batched
//! set), so there is no colouring: every iteration touches every vertex once,
//! and the accumulated Lagrange multiplier is carried across the iterations of
//! one substep (cleared once here at the start, mirroring the device solve).
//!
//! # Provenance
//!
//! The signed-volume-via-divergence-theorem pressure constraint and its
//! compliant `XPBD` projection are standard, publicly documented position-based
//! dynamics techniques. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::project_pressure;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Runs `iterations` compliant pressure projections over a closed shell on the
/// `CPU`, returning the applied positions.
///
/// This is the golden twin of
/// [`GpuClothPressure::solve`](super::gpu::GpuClothPressure::solve): both start
/// the substep with a zero Lagrange multiplier and carry it across every
/// iteration (standard `XPBD` accumulation), driving the enclosed volume toward
/// `target_volume = overpressure * rest_volume`. Each iteration delegates to
/// [`prism_physics_core`]'s `project_pressure` so the arithmetic is identical to
/// the engine's own `PressureConstraint`.
///
/// Empty triangles, zero iterations, or a non-positive `dt` leave `positions`
/// unchanged; pinned particles (`inverse_mass <= 0`) never move and out-of-range
/// triangles are skipped, exactly as in `project_pressure`.
#[must_use]
pub fn cpu_cloth_pressure(
    positions: &[Vec3],
    inverse_masses: &[Real],
    triangles: &[[u32; 3]],
    target_volume: Real,
    compliance: Real,
    dt: Real,
    iterations: u32,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    if triangles.is_empty() || iterations == 0 || dt <= 0.0 {
        return out;
    }

    let mut lambda = 0.0;
    for _ in 0..iterations {
        project_pressure(
            &mut out,
            inverse_masses,
            triangles,
            target_volume,
            compliance,
            dt,
            &mut lambda,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const EPS: Real = 1.0e-4;

    /// The eight corners of the axis-aligned unit cube `[0,1]^3`.
    fn unit_cube_positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ]
    }

    /// The twelve outward-wound triangles of the unit cube.
    fn unit_cube_triangles() -> Vec<[u32; 3]> {
        vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ]
    }

    /// Independent brute-force signed volume (divergence theorem), re-derived
    /// here so the twin is anchored against first-principles math, not itself.
    fn brute_force_volume(positions: &[Vec3], triangles: &[[u32; 3]]) -> Real {
        let mut sum = 0.0;
        for tri in triangles {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];
            sum += p0.dot(p1.cross(p2));
        }
        sum / 6.0
    }

    #[test]
    fn empty_or_zero_iterations_is_noop() {
        let pos = unit_cube_positions();
        let inv = vec![1.0; 8];
        assert_eq!(cpu_cloth_pressure(&pos, &inv, &[], 2.0, 0.0, 1.0 / 60.0, 8), pos);
        assert_eq!(
            cpu_cloth_pressure(&pos, &inv, &unit_cube_triangles(), 2.0, 0.0, 1.0 / 60.0, 0),
            pos
        );
        assert_eq!(
            cpu_cloth_pressure(&pos, &inv, &unit_cube_triangles(), 2.0, 0.0, 0.0, 8),
            pos
        );
    }

    #[test]
    fn overpressure_inflates_toward_the_target() {
        let pos = unit_cube_positions();
        let tris = unit_cube_triangles();
        let rest = brute_force_volume(&pos, &tris);
        let inv = vec![1.0; 8];
        let out = cpu_cloth_pressure(&pos, &inv, &tris, 2.0 * rest, 0.0, 1.0 / 60.0, 64);
        let inflated = brute_force_volume(&out, &tris);
        assert!(inflated > rest + 0.1, "inflated {inflated} rest {rest}");
        // A rigid (zero-compliance) constraint drives it close to the target.
        assert!(inflated <= 2.0 * rest + EPS, "overshoot {inflated}");
    }

    #[test]
    fn underpressure_deflates_toward_the_target() {
        let pos = unit_cube_positions();
        let tris = unit_cube_triangles();
        let rest = brute_force_volume(&pos, &tris);
        let inv = vec![1.0; 8];
        let out = cpu_cloth_pressure(&pos, &inv, &tris, 0.5 * rest, 0.0, 1.0 / 60.0, 64);
        let deflated = brute_force_volume(&out, &tris);
        assert!(deflated < rest - 0.1, "deflated {deflated} rest {rest}");
    }

    #[test]
    fn pinned_particles_do_not_move() {
        let pos = unit_cube_positions();
        let tris = unit_cube_triangles();
        let rest = brute_force_volume(&pos, &tris);
        let inv = vec![0.0; 8];
        let out = cpu_cloth_pressure(&pos, &inv, &tris, 4.0 * rest, 0.0, 1.0 / 60.0, 8);
        assert_eq!(out, pos);
    }
}
