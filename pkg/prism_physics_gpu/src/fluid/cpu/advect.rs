//! The `CPU` golden twin of the marker-particle advection that closes the full
//! fluid step.
//!
//! After the grid velocity field is made divergence-free and extrapolated into
//! the air, the marker particles are moved through that field so they track the
//! flow. This module integrates each marker with the second-order Runge–Kutta
//! midpoint rule, sampling the *grid* velocity (not the particle's own, possibly
//! noisier velocity) exactly as the reference in
//! [`prism_physics_core`](prism_physics_core::fluid::transfer) does.
//!
//! # Runge–Kutta midpoint
//!
//! For marker position `x0` and step `dt` the update is
//!
//! ```text
//! k1  = sample(x0)
//! mid = x0 + 0.5 * dt * k1
//! k2  = sample(mid)
//! x1  = x0 + dt * k2
//! ```
//!
//! Each marker is independent, so the device kernel is a single dispatch of one
//! thread per marker. The only arithmetic is the two trilinear gathers already
//! shared with the grid-to-particle transfer, so the device reproduces this
//! twin within the same tight tolerance the transfer parity uses.
//!
//! # Provenance
//!
//! Second-order Runge–Kutta advection of markers through a sampled velocity
//! field is a standard semi-Lagrangian technique (Bridson, *Fluid Simulation
//! for Computer Graphics*; Zhu and Bridson 2005). No Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::fields::GoldenGrid;

/// Advects every marker in `positions` through the `grid` velocity field by one
/// [`RK2`](self) midpoint step of size `dt`, writing the new positions in place.
///
/// The sampled field is the normalised grid velocity, so markers follow the
/// divergence-free flow. This does not clamp markers to the fluid interior;
/// domain confinement is a separate stage.
pub fn advect_rk2(grid: &GoldenGrid, positions: &mut [Vec3], dt: f32) {
    for x in positions.iter_mut() {
        let x0 = *x;
        let k1 = grid.sample_velocity(x0);
        let mid = x0 + 0.5 * dt * k1;
        let k2 = grid.sample_velocity(mid);
        *x = x0 + dt * k2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::cpu::p2g::particle_to_grid;
    use crate::fluid::grid::GridDims;
    use crate::fluid::particle::FluidParticles;

    /// Builds a grid whose velocity field is the constant `vel` everywhere the
    /// markers sample, by scattering a dense block of particles carrying `vel`.
    fn constant_field_grid(dims: GridDims, vel: Vec3) -> GoldenGrid {
        let mut grid = GoldenGrid::new(dims);
        let mut src = FluidParticles::new();
        for a in 0..10 {
            for b in 0..10 {
                for c in 0..10 {
                    src.spawn(
                        Vec3::new(
                            0.2 + a as f32 * 0.04,
                            0.2 + b as f32 * 0.04,
                            0.2 + c as f32 * 0.04,
                        ),
                        vel,
                    );
                }
            }
        }
        particle_to_grid(&mut grid, &src);
        grid
    }

    /// In a constant field the midpoint rule moves a marker by exactly `dt * v`.
    #[test]
    fn constant_field_translates_by_dt_times_velocity() {
        let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
        let vel = Vec3::new(0.4, -0.2, 0.1);
        let grid = constant_field_grid(dims, vel);
        let dt = 1.0e-2;
        let start = Vec3::new(0.4, 0.4, 0.4);
        let mut positions = [start];
        advect_rk2(&grid, &mut positions, dt);
        let expected = start + dt * vel;
        assert!(
            (positions[0] - expected).length() < 1.0e-3,
            "got {:?}, want {expected:?}",
            positions[0]
        );
    }

    /// A zero field leaves every marker where it started.
    #[test]
    fn zero_field_leaves_markers_fixed() {
        let dims = GridDims::new(8, 8, 8, 0.1, Vec3::ZERO);
        let grid = GoldenGrid::new(dims);
        let mut positions = [Vec3::new(0.3, 0.3, 0.3), Vec3::new(0.5, 0.4, 0.6)];
        let before = positions;
        advect_rk2(&grid, &mut positions, 1.0e-2);
        assert_eq!(positions, before);
    }

    /// Two half steps land close to one full step in a smooth constant field
    /// (the integrator is consistent).
    #[test]
    fn two_half_steps_track_one_full_step() {
        let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
        let vel = Vec3::new(0.2, 0.15, -0.1);
        let grid = constant_field_grid(dims, vel);
        let dt = 2.0e-2;
        let start = Vec3::new(0.4, 0.4, 0.4);

        let mut one = [start];
        advect_rk2(&grid, &mut one, dt);

        let mut two = [start];
        advect_rk2(&grid, &mut two, 0.5 * dt);
        advect_rk2(&grid, &mut two, 0.5 * dt);

        assert!(
            (one[0] - two[0]).length() < 1.0e-3,
            "{:?} {:?}",
            one[0],
            two[0]
        );
    }
}
