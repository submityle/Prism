//! The `CPU` golden particle-to-grid (`P2G`) transfer.
//!
//! [`particle_to_grid`] clears the accumulators, splats every marker particle's
//! velocity onto the staggered faces with quantised trilinear weights, then
//! normalises momentum by weight. It mirrors the device `p2g_scatter` and
//! `normalize` kernels stage for stage, so both engines produce the same
//! face-velocity field for a given particle set.
//!
//! # Provenance
//!
//! Trilinear `P2G` splatting follows Zhu and Bridson 2005 and Bridson. This
//! module contains no Unreal Engine source or derived code.

use super::fields::GoldenGrid;
use crate::fluid::particle::FluidParticles;

/// Splats particle velocities onto `grid` and normalises the result in place.
///
/// This is the `CPU` reference for the device `P2G` pass chain; the transfer is
/// mass-weighted (momentum divided by weight) on every touched face.
pub fn particle_to_grid(grid: &mut GoldenGrid, particles: &FluidParticles) {
    grid.begin_transfer();
    let positions = particles.positions();
    let velocities = particles.velocities();
    for (position, velocity) in positions.iter().zip(velocities.iter()) {
        grid.scatter_velocity(*position, *velocity);
    }
    grid.normalize_velocity();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::grid::GridDims;
    use glam::Vec3;

    #[test]
    fn recovers_constant_block_velocity() {
        let mut grid = GoldenGrid::new(GridDims::new(10, 10, 10, 0.1, Vec3::ZERO));
        let mut parts = FluidParticles::new();
        let vel = Vec3::new(0.4, -0.2, 0.1);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    parts.spawn(
                        Vec3::new(
                            0.3 + a as f32 * 0.04,
                            0.3 + b as f32 * 0.04,
                            0.3 + c as f32 * 0.04,
                        ),
                        vel,
                    );
                }
            }
        }
        particle_to_grid(&mut grid, &parts);
        let s = grid.sample_velocity(Vec3::new(0.4, 0.4, 0.4));
        assert!((s - vel).length() < 1e-3, "sampled {s:?}");
    }
}
