//! Single-step driver for the FLIP/APIC free-surface fluid solver.
//!
//! One [`FluidSolver::step`] performs the canonical splash-friendly pipeline:
//! splat particle velocities to the MAC grid (P2G), save the grid state for the
//! FLIP increment, apply gravity, enforce solid boundaries, project the field
//! to be divergence-free, extrapolate velocities into the air, gather back to
//! the particles with the PIC/FLIP (or APIC) blend, and advect the markers.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! predictor/projection/advection ordering follows Zhu & Bridson 2005 and
//! Bridson, *Fluid Simulation for Computer Graphics*.

use glam::Vec3;

use super::config::FluidConfig;
use super::mac_grid::MacGrid;
use super::particle::MarkerParticles;
use super::pressure;
use super::transfer;
use crate::math::scalar::Real;

/// Owns the MAC grid, the marker particles, and the step configuration.
#[derive(Clone, Debug)]
pub struct FluidSolver {
    /// The staggered background grid.
    pub grid: MacGrid,
    /// The advected marker particles.
    pub particles: MarkerParticles,
    /// Per-step parameters.
    pub config: FluidConfig,
}

impl FluidSolver {
    /// Creates a solver from a grid, particle set, and configuration.
    #[must_use]
    pub fn new(grid: MacGrid, particles: MarkerParticles, config: FluidConfig) -> FluidSolver {
        FluidSolver {
            grid,
            particles,
            config,
        }
    }

    /// The number of marker particles.
    #[inline]
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Advances the simulation by one time step.
    pub fn step(&mut self) {
        let cfg = self.config;
        // 1. Particle velocities → grid, marking fluid cells.
        transfer::particle_to_grid(&mut self.grid, &self.particles, cfg.transfer);
        // 2. Save the transferred field for the FLIP increment.
        self.grid.save_velocity();
        // 3. Body forces.
        self.grid.add_gravity(cfg.gravity, cfg.dt);
        // 4. No flow through solids.
        self.grid.enforce_solid_faces();
        // 5. Make the field divergence-free.
        pressure::project(&mut self.grid, &cfg);
        // 6. Fill air faces so advection near the surface is stable.
        self.grid.extrapolate_velocity(cfg.extrapolation_iterations);
        // 7. Grid → particles (PIC/FLIP or APIC).
        transfer::grid_to_particle(&self.grid, &mut self.particles, &cfg);
        // 8. Move the markers through the flow.
        transfer::advect(&self.grid, &mut self.particles, &cfg);
    }

    /// Advances the simulation by `steps` time steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// The maximum absolute divergence over fluid cells after the most recent
    /// projection (a residual diagnostic).
    #[must_use]
    pub fn max_divergence(&self) -> Real {
        pressure::max_fluid_divergence(&self.grid)
    }

    /// The total linear momentum `Σ vᵢ` of the marker particles (unit mass).
    #[must_use]
    pub fn total_velocity(&self) -> Vec3 {
        let mut s = Vec3::ZERO;
        for v in self.particles.velocities() {
            s += *v;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::config::TransferMode;

    fn dam_break() -> FluidSolver {
        let mut grid = MacGrid::new(16, 16, 16, 0.1, Vec3::ZERO);
        grid.set_solid_walls(1);
        let mut parts = MarkerParticles::new();
        // A column of water resting a bit above the floor.
        for i in 2..8 {
            for j in 2..12 {
                for k in 2..8 {
                    for s in 0..2 {
                        let jitter = s as Real * 0.05;
                        let x = Vec3::new(
                            (i as Real + 0.25 + jitter) * 0.1,
                            (j as Real + 0.5) * 0.1,
                            (k as Real + 0.5) * 0.1,
                        );
                        parts.spawn(x, Vec3::ZERO);
                    }
                }
            }
        }
        let mut cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));
        cfg.transfer = TransferMode::PicFlip;
        cfg.pressure_iterations = 80;
        FluidSolver::new(grid, parts, cfg)
    }

    #[test]
    fn water_column_conserves_particle_count_and_is_finite() {
        let mut solver = dam_break();
        let n0 = solver.particle_count();
        for _ in 0..12 {
            solver.step();
        }
        assert_eq!(solver.particle_count(), n0);
        for x in solver.particles.positions() {
            assert!(x.is_finite(), "non-finite position {x:?}");
        }
        for v in solver.particles.velocities() {
            assert!(v.is_finite(), "non-finite velocity {v:?}");
        }
    }

    #[test]
    fn projection_keeps_divergence_small() {
        let mut solver = dam_break();
        solver.step();
        assert!(
            solver.max_divergence() < 5.0e-2,
            "div={}",
            solver.max_divergence()
        );
    }

    #[test]
    fn deterministic_repeat() {
        let mut a = dam_break();
        let mut b = dam_break();
        for _ in 0..6 {
            a.step();
            b.step();
        }
        for (pa, pb) in a.particles.positions().iter().zip(b.particles.positions()) {
            assert_eq!(pa, pb);
        }
        for (va, vb) in a
            .particles
            .velocities()
            .iter()
            .zip(b.particles.velocities())
        {
            assert_eq!(va, vb);
        }
    }

    #[test]
    fn gravity_accelerates_downward() {
        let mut solver = dam_break();
        solver.step();
        // After one step under gravity the mean vertical velocity is negative.
        let mut vy = 0.0;
        for v in solver.particles.velocities() {
            vy += v.y;
        }
        vy /= solver.particle_count() as Real;
        assert!(vy < 0.0, "mean vy = {vy}");
    }
}
