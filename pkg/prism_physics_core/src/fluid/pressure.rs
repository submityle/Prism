//! Pressure projection (a discrete Poisson solve) that makes the MAC velocity
//! field divergence-free while respecting solid and free-surface boundaries.
//!
//! For each fluid cell we solve `∇²p = (ρ/Δt)·∇·u` with a matrix-free
//! Gauss–Seidel / successive-over-relaxation (SOR) sweep. Solid neighbours are
//! dropped from the stencil (no flow through walls, `u·n = 0`) and air (empty)
//! neighbours impose the free-surface Dirichlet condition `p = 0`. The pressure
//! gradient is then subtracted from the faces between fluid cells to remove the
//! divergent component of the flow.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The MAC
//! pressure-projection scheme, the solid/free-surface boundary handling, and
//! the SOR Poisson solve follow Bridson, *Fluid Simulation for Computer
//! Graphics*, and Foster & Fedkiw 2001.

use super::config::FluidConfig;
use super::mac_grid::{CellType, MacGrid};
use crate::math::scalar::Real;

/// Projects the grid velocity field to be (approximately) divergence-free.
///
/// Runs `cfg.pressure_iterations` SOR sweeps of the pressure Poisson solve,
/// then subtracts the pressure gradient from the fluid faces. Returns the
/// pressure field (row-major over cells) for inspection or reuse.
pub fn project(grid: &mut MacGrid, cfg: &FluidConfig) -> Vec<Real> {
    let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
    let dx = grid.dx();
    let mut pressure = vec![0.0 as Real; nx * ny * nz];
    // RHS scale: p has units so that the velocity update below cancels the
    // divergence exactly. `rhs = (ρ·dx/dt)·rawdiv`, `rawdiv = dx·divergence`.
    let rhs_scale = cfg.density * dx * dx / cfg.dt;
    let omega = cfg.over_relaxation.clamp(1.0, 1.99);

    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    for _ in 0..cfg.pressure_iterations {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if grid.cell_type(i, j, k) != CellType::Fluid {
                        continue;
                    }
                    let mut diag = 0.0;
                    let mut sum = 0.0;
                    // -x
                    if i > 0 {
                        match grid.cell_type(i - 1, j, k) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i - 1, j, k)];
                            }
                        }
                    }
                    // +x
                    if i + 1 < nx {
                        match grid.cell_type(i + 1, j, k) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i + 1, j, k)];
                            }
                        }
                    }
                    // -y
                    if j > 0 {
                        match grid.cell_type(i, j - 1, k) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i, j - 1, k)];
                            }
                        }
                    }
                    // +y
                    if j + 1 < ny {
                        match grid.cell_type(i, j + 1, k) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i, j + 1, k)];
                            }
                        }
                    }
                    // -z
                    if k > 0 {
                        match grid.cell_type(i, j, k - 1) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i, j, k - 1)];
                            }
                        }
                    }
                    // +z
                    if k + 1 < nz {
                        match grid.cell_type(i, j, k + 1) {
                            CellType::Solid => {}
                            CellType::Air => diag += 1.0,
                            CellType::Fluid => {
                                diag += 1.0;
                                sum += pressure[idx(i, j, k + 1)];
                            }
                        }
                    }
                    if diag <= 0.0 {
                        continue;
                    }
                    let rawdiv = grid.divergence(i, j, k) * dx;
                    let target = (sum - rhs_scale * rawdiv) / diag;
                    let cur = pressure[idx(i, j, k)];
                    pressure[idx(i, j, k)] = cur + omega * (target - cur);
                }
            }
        }
    }

    subtract_gradient(grid, &pressure, cfg);
    pressure
}

/// Subtracts the pressure gradient from the faces separating fluid cells,
/// making the field divergence-free. Faces touching solids are left at zero.
fn subtract_gradient(grid: &mut MacGrid, pressure: &[Real], cfg: &FluidConfig) {
    let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
    let dx = grid.dx();
    let scale = cfg.dt / (cfg.density * dx);
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    // u faces between (i-1) and (i).
    for k in 0..nz {
        for j in 0..ny {
            for i in 1..nx {
                let left = grid.cell_type(i - 1, j, k);
                let right = grid.cell_type(i, j, k);
                if left == CellType::Solid || right == CellType::Solid {
                    grid.set_u(i, j, k, 0.0);
                    continue;
                }
                if left == CellType::Fluid || right == CellType::Fluid {
                    let grad = pressure[idx(i, j, k)] - pressure[idx(i - 1, j, k)];
                    let u = grid.u_at(i, j, k) - scale * grad / dx;
                    grid.set_u(i, j, k, u);
                }
            }
        }
    }
    // v faces between (j-1) and (j).
    for k in 0..nz {
        for j in 1..ny {
            for i in 0..nx {
                let down = grid.cell_type(i, j - 1, k);
                let up = grid.cell_type(i, j, k);
                if down == CellType::Solid || up == CellType::Solid {
                    grid.set_v(i, j, k, 0.0);
                    continue;
                }
                if down == CellType::Fluid || up == CellType::Fluid {
                    let grad = pressure[idx(i, j, k)] - pressure[idx(i, j - 1, k)];
                    let v = grid.v_at(i, j, k) - scale * grad / dx;
                    grid.set_v(i, j, k, v);
                }
            }
        }
    }
    // w faces between (k-1) and (k).
    for k in 1..nz {
        for j in 0..ny {
            for i in 0..nx {
                let back = grid.cell_type(i, j, k - 1);
                let front = grid.cell_type(i, j, k);
                if back == CellType::Solid || front == CellType::Solid {
                    grid.set_w(i, j, k, 0.0);
                    continue;
                }
                if back == CellType::Fluid || front == CellType::Fluid {
                    let grad = pressure[idx(i, j, k)] - pressure[idx(i, j, k - 1)];
                    let w = grid.w_at(i, j, k) - scale * grad / dx;
                    grid.set_w(i, j, k, w);
                }
            }
        }
    }
}

/// The maximum absolute divergence over all fluid cells (a residual measure
/// used by tests and diagnostics).
#[must_use]
pub fn max_fluid_divergence(grid: &MacGrid) -> Real {
    let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
    let mut m = 0.0;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if grid.cell_type(i, j, k) == CellType::Fluid {
                    let d = grid.divergence(i, j, k).abs();
                    if d > m {
                        m = d;
                    }
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn projection_reduces_divergence() {
        let mut grid = MacGrid::new(12, 12, 12, 0.1, Vec3::ZERO);
        grid.set_solid_walls(1);
        // Mark an interior block as fluid and give it a divergent field.
        for i in 3..9 {
            for j in 3..9 {
                for k in 3..9 {
                    grid.set_cell_type(i, j, k, CellType::Fluid);
                }
            }
        }
        // Impose an outward (divergent) velocity by setting a radial-ish field.
        for i in 0..=grid.nx() {
            for j in 0..grid.ny() {
                for k in 0..grid.nz() {
                    grid.set_u(i, j, k, (i as Real - 6.0) * 0.05);
                }
            }
        }
        let before = max_fluid_divergence(&grid);
        let mut cfg = FluidConfig::new(1.0e-2, Vec3::ZERO);
        cfg.pressure_iterations = 120;
        cfg.over_relaxation = 1.7;
        grid.enforce_solid_faces();
        project(&mut grid, &cfg);
        let after = max_fluid_divergence(&grid);
        assert!(after < before * 0.1, "before={before} after={after}");
        assert!(after < 1.0e-2, "residual too large: {after}");
    }
}
