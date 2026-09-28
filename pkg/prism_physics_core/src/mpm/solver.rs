//! The single-step MLS-MPM solver and wall boundary conditions.
//!
//! One step performs the canonical MPM cycle: clear the grid, scatter with
//! [`particle_to_grid`], convert momentum to velocity, apply gravity, enforce
//! wall boundary conditions on the grid velocities, gather with
//! [`grid_to_particle`], then clamp particles that stray into the boundary
//! layer back into the domain interior.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The step
//! ordering and collision/boundary handling follow the standard MLS-MPM
//! pipeline (Hu et al. 2018).

use glam::Vec3;

use crate::math::scalar::Real;

use super::config::{BoundaryCondition, MpmConfig, MpmMaterial};
use super::grid::Grid;
use super::particle::MaterialPoints;
use super::transfer::{grid_to_particle, particle_to_grid};

/// A self-contained MLS-MPM simulation: particles, a reusable background grid,
/// the material, and the step configuration.
#[derive(Clone, Debug)]
pub struct MpmSolver {
    /// The material points being simulated.
    pub particles: MaterialPoints,
    /// The reusable background grid (cleared each step).
    pub grid: Grid,
    /// The elastic/plastic material.
    pub material: MpmMaterial,
    /// The step configuration (time step, gravity, boundary behaviour).
    pub config: MpmConfig,
}

impl MpmSolver {
    /// Creates a solver from its parts.
    #[must_use]
    pub fn new(
        particles: MaterialPoints,
        grid: Grid,
        material: MpmMaterial,
        config: MpmConfig,
    ) -> MpmSolver {
        MpmSolver {
            particles,
            grid,
            material,
            config,
        }
    }

    /// Advances the simulation by one time step.
    pub fn step(&mut self) {
        self.grid.clear();
        particle_to_grid(
            &self.particles,
            &mut self.grid,
            &self.config,
            &self.material,
        );
        self.grid.finalize_velocity();
        // Apply gravity to every node carrying mass.
        self.grid
            .add_velocity_to_active(self.config.gravity * self.config.dt);
        apply_grid_boundary(&mut self.grid, &self.config);
        grid_to_particle(
            &mut self.particles,
            &self.grid,
            &self.config,
            &self.material,
        );
        clamp_particles(&mut self.particles, &self.grid, &self.config);
    }

    /// Advances the simulation by `steps` time steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Enforces the wall boundary condition on the grid velocities within
/// `boundary_thickness` nodes of each domain face.
pub fn apply_grid_boundary(grid: &mut Grid, cfg: &MpmConfig) {
    let t = cfg.boundary_thickness as i32;
    let nx = grid.nx() as i32;
    let ny = grid.ny() as i32;
    let nz = grid.nz() as i32;
    for i in 0..grid.nx() {
        for j in 0..grid.ny() {
            for k in 0..grid.nz() {
                if grid.mass_at(i, j, k) <= 0.0 {
                    continue;
                }
                let (ii, jj, kk) = (i as i32, j as i32, k as i32);
                let low = [ii < t, jj < t, kk < t];
                let high = [ii >= nx - t, jj >= ny - t, kk >= nz - t];
                if !(low[0] || low[1] || low[2] || high[0] || high[1] || high[2]) {
                    continue;
                }
                let mut v = grid.velocity_at(i, j, k);
                match cfg.boundary {
                    BoundaryCondition::Sticky => v = Vec3::ZERO,
                    BoundaryCondition::Slip => {
                        if low[0] || high[0] {
                            v.x = 0.0;
                        }
                        if low[1] || high[1] {
                            v.y = 0.0;
                        }
                        if low[2] || high[2] {
                            v.z = 0.0;
                        }
                    }
                    BoundaryCondition::Separate => {
                        if low[0] && v.x < 0.0 {
                            v.x = 0.0;
                        }
                        if high[0] && v.x > 0.0 {
                            v.x = 0.0;
                        }
                        if low[1] && v.y < 0.0 {
                            v.y = 0.0;
                        }
                        if high[1] && v.y > 0.0 {
                            v.y = 0.0;
                        }
                        if low[2] && v.z < 0.0 {
                            v.z = 0.0;
                        }
                        if high[2] && v.z > 0.0 {
                            v.z = 0.0;
                        }
                    }
                }
                grid.set_velocity(i, j, k, v);
            }
        }
    }
}

/// Clamps particle positions so their quadratic stencil stays inside the grid,
/// keeping them at least `boundary_thickness` cells from each face.
pub fn clamp_particles(mp: &mut MaterialPoints, grid: &Grid, cfg: &MpmConfig) {
    let dx = grid.dx();
    let origin = grid.origin();
    // Keep at least two cells of stencil support plus the boundary layer.
    let margin = (cfg.boundary_thickness.max(2)) as Real * dx;
    let lo = origin + Vec3::splat(margin);
    let hi = origin
        + Vec3::new(
            (grid.nx() - 1) as Real * dx,
            (grid.ny() - 1) as Real * dx,
            (grid.nz() - 1) as Real * dx,
        )
        - Vec3::splat(margin);
    for p in mp.positions_mut() {
        p.x = p.x.clamp(lo.x, hi.x);
        p.y = p.y.clamp(lo.y, hi.y);
        p.z = p.z.clamp(lo.z, hi.z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_of_particles() -> MaterialPoints {
        let mut mp = MaterialPoints::new();
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    let x = Vec3::new(
                        0.9 + a as Real * 0.05,
                        0.9 + b as Real * 0.05,
                        0.9 + c as Real * 0.05,
                    );
                    mp.spawn(x, Vec3::ZERO, 1.0, 1.0e-3);
                }
            }
        }
        mp
    }

    #[test]
    fn free_fall_conserves_mass_and_applies_gravity_impulse() {
        let mp = column_of_particles();
        let grid = Grid::new(24, 24, 24, 0.1, Vec3::ZERO);
        let material = MpmMaterial::default();
        let mut cfg = MpmConfig::new(5.0e-4, Vec3::new(0.0, -9.81, 0.0));
        cfg.boundary = BoundaryCondition::Slip;
        let mass0 = mp.total_mass();
        let mut solver = MpmSolver::new(mp, grid, material, cfg);

        let steps = 20;
        solver.advance(steps);

        // Mass is exactly conserved (particle masses never change).
        assert!((solver.particles.total_mass() - mass0).abs() < 1.0e-6);
        // Vertical momentum equals the accumulated gravity impulse.
        let expected_py = mass0 * cfg.gravity.y * cfg.dt * steps as Real;
        let py = solver.particles.total_momentum().y;
        assert!(
            (py - expected_py).abs() < 1.0e-2 * expected_py.abs().max(1.0),
            "py={py} expected={expected_py}"
        );
        // No NaNs anywhere.
        for v in solver.particles.velocities() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn step_is_deterministic() {
        let build = || {
            let mp = column_of_particles();
            let grid = Grid::new(24, 24, 24, 0.1, Vec3::ZERO);
            MpmSolver::new(mp, grid, MpmMaterial::default(), MpmConfig::default())
        };
        let mut a = build();
        let mut b = build();
        a.advance(10);
        b.advance(10);
        for (pa, pb) in a.particles.positions().iter().zip(b.particles.positions()) {
            assert_eq!(pa.to_array(), pb.to_array());
        }
        for (va, vb) in a
            .particles
            .velocities()
            .iter()
            .zip(b.particles.velocities())
        {
            assert_eq!(va.to_array(), vb.to_array());
        }
    }

    #[test]
    fn snow_plasticity_stays_finite_and_stable() {
        let mp = column_of_particles();
        let grid = Grid::new(24, 24, 24, 0.1, Vec3::ZERO);
        let material = MpmMaterial::new(1.4e5, 0.2, 4.0e2);
        let mut cfg = MpmConfig::new(2.0e-4, Vec3::new(0.0, -9.81, 0.0));
        cfg.plastic = true;
        cfg.boundary = BoundaryCondition::Sticky;
        let mut solver = MpmSolver::new(mp, grid, material, cfg);
        solver.advance(50);
        for p in solver.particles.positions() {
            assert!(p.is_finite());
        }
        for jp in solver.particles.plastic_det() {
            assert!(jp.is_finite() && *jp > 0.0);
        }
    }
}
