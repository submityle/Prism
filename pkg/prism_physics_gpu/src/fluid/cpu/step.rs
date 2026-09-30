//! The `CPU` golden twin of a full `FLIP`/`APIC` fluid step.
//!
//! [`fluid_step`] chains the stage twins of this module into the single
//! canonical splash-friendly pipeline the core reference runs in
//! `prism_physics_core::fluid::solver::FluidSolver::step`:
//!
//! 1. `P2G` — splat particle velocities onto the staggered `MAC` grid.
//! 2. save — snapshot the transferred field for the `FLIP` increment.
//! 3. gravity — integrate the constant body force on every face.
//! 4. solids — zero the faces bordering solid cells (no through-flow).
//! 5. project — make the field divergence-free with the red-black `SOR` solve.
//! 6. extrapolate — fill air faces so the surface advection stays stable.
//! 7. `G2P` — gather grid velocities back with the `PIC`/`FLIP` blend.
//! 8. advect — move the markers through the projected flow with `RK2`.
//!
//! Running the identical stage arithmetic as the device orchestrator
//! [`GpuFluidStep`](crate::fluid::gpu::GpuFluidStep) makes this the reference a
//! full-step real-device parity test measures against: because every stage twin
//! is itself already parity-checked against its kernel, the composed step is
//! the natural next anchor.
//!
//! # Provenance
//!
//! The predictor / projection / advection ordering follows Zhu and Bridson 2005
//! and Bridson, *Fluid Simulation for Computer Graphics*. This module contains
//! no Unreal Engine source or derived code.

use super::advect::advect_rk2;
use super::fields::GoldenGrid;
use super::g2p::grid_to_particle;
use super::grid_ops::{add_gravity, enforce_solid_faces};
use super::p2g::particle_to_grid;
use super::pressure::{self, PressureConfig};
use crate::fluid::config::FluidConfig;
use crate::fluid::grid::CellType;
use crate::fluid::particle::FluidParticles;

/// Advances `particles` on `grid` by one full fluid step, in place.
///
/// `cell_types` is the row-major cell classification (length
/// [`GridDims::cell_count`](crate::fluid::grid::GridDims::cell_count)); it drives
/// the solid boundary condition and the free-surface Dirichlet condition of the
/// pressure solve. `cfg` supplies the time step, gravity, blend, and the
/// pressure / extrapolation sweep counts.
///
/// The transfer uses the `PIC`/`FLIP` blend from
/// [`FluidConfig::effective_flip_blend`]; the affine (`APIC`) transfer is a
/// later slice, so [`TransferMode::Apic`](crate::fluid::config::TransferMode)
/// currently falls back to the same blend.
pub fn fluid_step(
    grid: &mut GoldenGrid,
    particles: &mut FluidParticles,
    cell_types: &[CellType],
    cfg: &FluidConfig,
) {
    let dims = grid.dims();

    // 1. Particle velocities -> grid, marking fluid faces.
    particle_to_grid(grid, particles);
    // 2. Save the transferred field for the `FLIP` increment.
    grid.save_velocity();
    // 3. Body force.
    add_gravity(dims, grid.velocity_mut(), cfg.gravity, cfg.dt);
    // 4. No flow through solids.
    enforce_solid_faces(dims, cell_types, grid.velocity_mut());
    // 5. Make the field divergence-free.
    let pcfg = PressureConfig::new(
        cfg.density,
        cfg.dt,
        cfg.over_relaxation,
        cfg.pressure_iterations,
    );
    let _pressure = pressure::project(dims, cell_types, grid.velocity_mut(), pcfg);
    // 6. Fill air faces so surface advection stays stable.
    grid.extrapolate_velocity(cfg.extrapolation_iterations);
    // 7. Grid -> particles with the `PIC`/`FLIP` blend.
    grid_to_particle(grid, particles, cfg.effective_flip_blend());
    // 8. Move the markers through the flow.
    advect_rk2(grid, particles.positions_mut(), cfg.dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::grid::GridDims;
    use glam::Vec3;

    /// Builds a solid-walled grid classification: the one-cell border is solid,
    /// the interior is fluid.
    fn walled_cells(dims: GridDims) -> Vec<CellType> {
        let (nx, ny, nz) = (dims.nx, dims.ny, dims.nz);
        let mut cells = vec![CellType::Fluid; dims.cell_count()];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let border =
                        i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                    if border {
                        cells[dims.cell_idx(i, j, k)] = CellType::Solid;
                    }
                }
            }
        }
        cells
    }

    /// A resting column of markers a few cells above the floor.
    fn water_column(dims: GridDims) -> FluidParticles {
        let mut parts = FluidParticles::new();
        for i in 3..8 {
            for j in 3..10 {
                for k in 3..8 {
                    parts.spawn(
                        Vec3::new(
                            (i as f32 + 0.5) * dims.dx,
                            (j as f32 + 0.5) * dims.dx,
                            (k as f32 + 0.5) * dims.dx,
                        ),
                        Vec3::ZERO,
                    );
                }
            }
        }
        parts
    }

    #[test]
    fn step_preserves_particle_count_and_stays_finite() {
        let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
        let mut grid = GoldenGrid::new(dims);
        let mut parts = water_column(dims);
        let cells = walled_cells(dims);
        let n0 = parts.len();
        let mut cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));
        cfg.pressure_iterations = 40;

        for _ in 0..10 {
            fluid_step(&mut grid, &mut parts, &cells, &cfg);
        }

        assert_eq!(parts.len(), n0);
        for p in parts.positions() {
            assert!(p.is_finite(), "non-finite position {p:?}");
        }
        for v in parts.velocities() {
            assert!(v.is_finite(), "non-finite velocity {v:?}");
        }
    }

    #[test]
    fn gravity_pulls_the_column_down() {
        let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
        let mut grid = GoldenGrid::new(dims);
        let mut parts = water_column(dims);
        let cells = walled_cells(dims);
        let cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));

        let mean_y0 = mean_height(parts.positions());
        for _ in 0..20 {
            fluid_step(&mut grid, &mut parts, &cells, &cfg);
        }
        let mean_y1 = mean_height(parts.positions());

        assert!(
            mean_y1 < mean_y0,
            "expected the column to settle: {mean_y0} -> {mean_y1}"
        );
    }

    fn mean_height(positions: &[Vec3]) -> f32 {
        let mut s = 0.0;
        for p in positions {
            s += p.y;
        }
        s / positions.len() as f32
    }
}
