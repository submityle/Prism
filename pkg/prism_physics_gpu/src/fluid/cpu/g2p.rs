//! The `CPU` golden grid-to-particle (`G2P`) transfer.
//!
//! [`grid_to_particle`] reconstructs each particle's velocity from the grid
//! using the `PIC`/`FLIP` blend: the interpolated grid velocity (`PIC`) is mixed
//! with the old particle velocity plus the interpolated grid *increment*
//! (`FLIP`). It mirrors the device `g2p` kernel arithmetic exactly. With a zero
//! saved field and a zero blend the transfer reduces to pure `PIC`, which is the
//! configuration the real-device transfer parity test uses.
//!
//! # Provenance
//!
//! The `PIC`/`FLIP` blend follows Zhu and Bridson 2005 and Bridson. This module
//! contains no Unreal Engine source or derived code.

use super::fields::GoldenGrid;
use crate::fluid::particle::FluidParticles;

/// Reconstructs particle velocities from `grid` using the `PIC`/`FLIP` blend.
///
/// `blend` is the `FLIP` fraction, clamped to `[0, 1]`: `0` is pure `PIC` and
/// `1` is pure `FLIP`. Call after the pressure projection, having saved the
/// pre-projection field with [`GoldenGrid::save_velocity`] for the increment.
pub fn grid_to_particle(grid: &GoldenGrid, particles: &mut FluidParticles, blend: f32) {
    let blend = blend.clamp(0.0, 1.0);
    let positions = particles.positions().to_vec();
    let old = particles.velocities().to_vec();
    let out = particles.velocities_mut();
    for ((position, old_vel), new_vel) in positions.iter().zip(old.iter()).zip(out.iter_mut()) {
        let v_pic = grid.sample_velocity(*position);
        let v_saved = grid.sample_saved_velocity(*position);
        let v_flip = *old_vel + (v_pic - v_saved);
        *new_vel = blend * v_flip + (1.0 - blend) * v_pic;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::cpu::p2g::particle_to_grid;
    use crate::fluid::grid::GridDims;
    use glam::Vec3;

    #[test]
    fn pure_pic_roundtrip_recovers_constant_field() {
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
                        Vec3::ZERO,
                    );
                }
            }
        }
        // Re-spawn with the constant velocity for the scatter.
        let mut src = FluidParticles::new();
        for p in parts.positions() {
            src.spawn(*p, vel);
        }
        particle_to_grid(&mut grid, &src);
        grid_to_particle(&grid, &mut parts, 0.0);
        for v in parts.velocities() {
            assert!((*v - vel).length() < 1e-3, "got {v:?}");
        }
    }

    #[test]
    fn pure_flip_with_zero_saved_adds_full_increment() {
        // With a zero saved field, v_flip = old + v_pic. A pure-FLIP blend then
        // returns old + v_pic; check the arithmetic on a single particle.
        let mut grid = GoldenGrid::new(GridDims::new(6, 6, 6, 0.1, Vec3::ZERO));
        let mut src = FluidParticles::new();
        let vel = Vec3::new(0.2, 0.0, 0.0);
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    src.spawn(
                        Vec3::new(
                            0.28 + a as f32 * 0.03,
                            0.28 + b as f32 * 0.03,
                            0.28 + c as f32 * 0.03,
                        ),
                        vel,
                    );
                }
            }
        }
        particle_to_grid(&mut grid, &src);
        // Saved stays zero (never saved), so v_flip = old + v_pic.
        let mut probe = FluidParticles::new();
        let old = Vec3::new(1.0, 0.0, 0.0);
        probe.spawn(Vec3::new(0.31, 0.31, 0.31), old);
        let v_pic = grid.sample_velocity(Vec3::new(0.31, 0.31, 0.31));
        grid_to_particle(&grid, &mut probe, 1.0);
        let expected = old + v_pic;
        assert!((probe.velocities()[0] - expected).length() < 1e-6);
    }
}
