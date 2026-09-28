//! Particle-to-grid (P2G) and grid-to-particle (G2P) transfers for the
//! FLIP/APIC fluid solver.
//!
//! P2G splats each marker particle's velocity onto the staggered MAC faces
//! with trilinear weights (optionally with the APIC affine correction) and
//! records which cells contain fluid. G2P reconstructs particle velocities
//! from the projected grid: the PIC/FLIP blend mixes the interpolated velocity
//! (PIC) with the old particle velocity plus the interpolated grid *increment*
//! (FLIP), while the APIC path additionally reconstructs the affine matrix.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! PIC/FLIP blend follows Zhu & Bridson 2005 and Bridson, *Fluid Simulation
//! for Computer Graphics*; the APIC transfer follows Jiang et al. 2015.

use glam::Vec3;

use super::config::{FluidConfig, TransferMode};
use super::mac_grid::MacGrid;
use super::particle::MarkerParticles;
use crate::math::scalar::Real;

/// Splats particle velocities onto the MAC grid and marks fluid cells.
///
/// This clears the grid velocity/weight accumulators first (via
/// [`MacGrid::begin_transfer`]), so any solid-wall classification set with
/// [`MacGrid::set_solid_walls`] is preserved while air cells are reset.
pub fn particle_to_grid(grid: &mut MacGrid, particles: &MarkerParticles, mode: TransferMode) {
    grid.begin_transfer();
    let positions = particles.positions();
    let velocities = particles.velocities();
    let affine = particles.affine();
    for p in 0..particles.len() {
        match mode {
            TransferMode::PicFlip => grid.scatter_velocity(positions[p], velocities[p]),
            TransferMode::Apic => {
                grid.scatter_velocity_affine(positions[p], velocities[p], affine[p]);
            }
        }
        grid.mark_fluid_at(positions[p]);
    }
    grid.normalize_velocity();
}

/// Reconstructs particle velocities (and, for APIC, the affine matrix) from the
/// projected grid using the PIC/FLIP blend configured in `cfg`.
///
/// Call this *after* the pressure projection so the sampled field is
/// divergence-free. It expects the pre-projection field to have been saved with
/// [`MacGrid::save_velocity`] for the FLIP increment.
pub fn grid_to_particle(grid: &MacGrid, particles: &mut MarkerParticles, cfg: &FluidConfig) {
    let positions = particles.positions().to_vec();
    let old_velocities = particles.velocities().to_vec();
    let mut new_velocities = old_velocities.clone();
    let mut new_affine = particles.affine().to_vec();

    match cfg.transfer {
        TransferMode::PicFlip => {
            let blend = cfg.flip_blend.clamp(0.0, 1.0);
            for p in 0..positions.len() {
                let v_pic = grid.sample_velocity(positions[p]);
                let v_saved = grid.sample_saved_velocity(positions[p]);
                let v_flip = old_velocities[p] + (v_pic - v_saved);
                new_velocities[p] = blend * v_flip + (1.0 - blend) * v_pic;
            }
        }
        TransferMode::Apic => {
            for p in 0..positions.len() {
                let (vel, cmat) = grid.sample_velocity_affine(positions[p]);
                new_velocities[p] = vel;
                new_affine[p] = cmat;
            }
        }
    }

    particles.velocities_mut().copy_from_slice(&new_velocities);
    particles.affine_mut().copy_from_slice(&new_affine);
}

/// Advects marker particles through the grid velocity field using a second
/// order Runge–Kutta (midpoint) step, then clamps them to the fluid interior.
///
/// The sampled velocity field is used for advection so that particles follow
/// the divergence-free flow rather than their own (possibly noisier) velocity.
pub fn advect(grid: &MacGrid, particles: &mut MarkerParticles, cfg: &FluidConfig) {
    let dt = cfg.dt;
    let positions = particles.positions().to_vec();
    let mut new_positions = positions.clone();
    for p in 0..positions.len() {
        let x0 = positions[p];
        let k1 = grid.sample_velocity(x0);
        let mid = x0 + 0.5 * dt * k1;
        let k2 = grid.sample_velocity(mid);
        new_positions[p] = x0 + dt * k2;
    }
    particles.positions_mut().copy_from_slice(&new_positions);
    clamp_to_fluid_domain(grid, particles);
}

/// Clamps every particle to stay strictly inside the non-solid interior of the
/// grid, pushing it back by a fraction of a cell from the solid walls.
pub fn clamp_to_fluid_domain(grid: &MacGrid, particles: &mut MarkerParticles) {
    let dx = grid.dx();
    let origin = grid.origin();
    // Detect the solid-wall thickness on the low-x side (walls are symmetric).
    let mut wall = 0usize;
    while wall < grid.nx()
        && grid.cell_type(wall, grid.ny() / 2, grid.nz() / 2) == super::mac_grid::CellType::Solid
    {
        wall += 1;
    }
    let margin = 0.01 * dx;
    let lo = origin + Vec3::splat(wall as Real * dx + margin);
    let hi_x = origin.x + (grid.nx() - wall) as Real * dx - margin;
    let hi_y = origin.y + (grid.ny() - wall) as Real * dx - margin;
    let hi_z = origin.z + (grid.nz() - wall) as Real * dx - margin;
    let positions = particles.positions_mut();
    for x in positions.iter_mut() {
        x.x = x.x.clamp(lo.x, hi_x);
        x.y = x.y.clamp(lo.y, hi_y);
        x.z = x.z.clamp(lo.z, hi_z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat3;

    #[test]
    fn constant_field_roundtrip_pic() {
        let mut grid = MacGrid::new(10, 10, 10, 0.1, Vec3::ZERO);
        let mut parts = MarkerParticles::new();
        let vel = Vec3::new(0.4, -0.2, 0.1);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let x = Vec3::new(
                        0.3 + a as Real * 0.04,
                        0.3 + b as Real * 0.04,
                        0.3 + c as Real * 0.04,
                    );
                    parts.spawn(x, vel);
                }
            }
        }
        let mut cfg = FluidConfig::new(1.0e-2, Vec3::ZERO);
        cfg.transfer = TransferMode::PicFlip;
        cfg.flip_blend = 0.0; // pure PIC
        particle_to_grid(&mut grid, &parts, cfg.transfer);
        grid.save_velocity();
        grid_to_particle(&grid, &mut parts, &cfg);
        for v in parts.velocities() {
            assert!((*v - vel).length() < 1.0e-3, "got {v:?}");
        }
    }

    #[test]
    fn apic_recovers_linear_field() {
        // A linear velocity field v = A x should be captured by APIC's affine
        // matrix so the roundtrip is (nearly) exact.
        let mut grid = MacGrid::new(12, 12, 12, 0.1, Vec3::ZERO);
        let mut parts = MarkerParticles::new();
        let a = Mat3::from_cols(
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        );
        let center = Vec3::splat(0.6);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let x = center
                        + Vec3::new(
                            (i as Real - 2.0) * 0.03,
                            (j as Real - 2.0) * 0.03,
                            (k as Real - 2.0) * 0.03,
                        );
                    let v = a * (x - center);
                    let idx = parts.spawn(x, v);
                    parts.affine_mut()[idx] = a;
                }
            }
        }
        let mut cfg = FluidConfig::new(1.0e-2, Vec3::ZERO);
        cfg.transfer = TransferMode::Apic;
        particle_to_grid(&mut grid, &parts, cfg.transfer);
        grid.save_velocity();
        grid_to_particle(&grid, &mut parts, &cfg);
        // The reconstructed affine matrix should be close to A for interior
        // particles (skip the boundary ring).
        let recon = parts.affine()[parts.len() / 2];
        let diff = recon - a;
        let fro = diff.x_axis.length() + diff.y_axis.length() + diff.z_axis.length();
        assert!(fro < 0.15, "affine mismatch fro={fro}");
    }

    #[test]
    fn advection_moves_with_flow() {
        let mut grid = MacGrid::new(10, 10, 10, 0.1, Vec3::ZERO);
        let mut parts = MarkerParticles::new();
        let vel = Vec3::new(0.5, 0.0, 0.0);
        parts.spawn(Vec3::new(0.5, 0.5, 0.5), vel);
        particle_to_grid(&mut grid, &parts, TransferMode::PicFlip);
        grid.normalize_velocity();
        let cfg = FluidConfig::new(0.05, Vec3::ZERO);
        let before = parts.positions()[0];
        advect(&grid, &mut parts, &cfg);
        let after = parts.positions()[0];
        assert!(after.x > before.x, "particle should move +x");
    }
}
