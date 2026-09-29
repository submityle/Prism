//! The `CPU` golden pressure-projection twin: a red-black `SOR` Poisson solve
//! that makes the staggered velocity field (approximately) divergence-free
//! while respecting solid and free-surface boundaries.
//!
//! For each fluid cell it solves the discrete Poisson equation
//! `laplacian(p) = (rho / dt) * div(u)` with a matrix-free
//! successive-over-relaxation sweep, then subtracts the pressure gradient from
//! the faces between fluid cells. Solid neighbours drop out of the stencil (no
//! flow through walls) and air neighbours impose the free-surface Dirichlet
//! condition `p = 0`.
//!
//! # Red-black ordering
//!
//! Unlike the sequential reference in
//! [`prism_physics_core`](prism_physics_core::fluid), this twin sweeps the cells
//! in *red-black* order: within one colour a cell's six face-neighbours all
//! have the opposite colour, so no two same-colour cells interact and the whole
//! colour can be relaxed in parallel. Relaxing one colour per compute dispatch
//! reproduces this exact sequential-per-colour order on the device, which is
//! what the real-device parity test checks. The two schemes converge to the
//! same divergence-free field; only the iteration path differs.
//!
//! # Provenance
//!
//! The `MAC` pressure-projection scheme, the solid / free-surface boundary
//! handling, and the red-black `SOR` Poisson solve follow Bridson, *Fluid
//! Simulation for Computer Graphics*, and Foster and Fedkiw 2001. No Unreal
//! Engine source or derived code.

use crate::fluid::grid::{CellType, GridDims};

/// Tunables for one pressure-projection solve.
///
/// These mirror the pressure-relevant fields of the `CPU` reference's
/// `FluidConfig`: the rest density, the time step, the over-relaxation factor,
/// and the sweep count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureConfig {
    /// Rest fluid density (scales the right-hand side of the Poisson solve).
    pub density: f32,
    /// Time step in seconds.
    pub dt: f32,
    /// Successive over-relaxation factor; clamped to `[1, 1.99]` when used.
    pub over_relaxation: f32,
    /// Number of red-black `SOR` iterations (each is a red sweep then a black
    /// sweep).
    pub iterations: u32,
}

impl PressureConfig {
    /// Creates a configuration from raw parameters.
    #[must_use]
    pub fn new(density: f32, dt: f32, over_relaxation: f32, iterations: u32) -> PressureConfig {
        PressureConfig {
            density,
            dt,
            over_relaxation,
            iterations,
        }
    }

    /// The over-relaxation factor clamped to the convergent range `[1, 1.99]`.
    #[must_use]
    pub fn effective_over_relaxation(&self) -> f32 {
        self.over_relaxation.clamp(1.0, 1.99)
    }
}

/// Read-only staggered-face accessors over a concatenated `[u | v | w]` array.
struct Faces<'a> {
    dims: GridDims,
    velocity: &'a [f32],
}

impl Faces<'_> {
    fn u(&self, i: u32, j: u32, k: u32) -> f32 {
        self.velocity[self.dims.u_offset() + self.dims.u_idx(i, j, k)]
    }
    fn v(&self, i: u32, j: u32, k: u32) -> f32 {
        self.velocity[self.dims.v_offset() + self.dims.v_idx(i, j, k)]
    }
    fn w(&self, i: u32, j: u32, k: u32) -> f32 {
        self.velocity[self.dims.w_offset() + self.dims.w_idx(i, j, k)]
    }

    /// The raw (undivided) divergence `du + dv + dw` at cell `(i, j, k)`.
    ///
    /// This is `dx * divergence`; keeping it undivided avoids a divide-then-
    /// multiply round-trip and is the exact quantity the device kernel forms.
    fn raw_divergence(&self, i: u32, j: u32, k: u32) -> f32 {
        let du = self.u(i + 1, j, k) - self.u(i, j, k);
        let dv = self.v(i, j + 1, k) - self.v(i, j, k);
        let dw = self.w(i, j, k + 1) - self.w(i, j, k);
        du + dv + dw
    }
}

/// The result of one relaxation of a single fluid cell.
struct CellUpdate {
    /// The new pressure value after the `SOR` blend.
    value: f32,
    /// Whether the cell had at least one non-solid neighbour (otherwise it is
    /// left untouched).
    active: bool,
}

/// Relaxes one fluid cell against its neighbours' current pressures.
///
/// Mirrors the device kernel arithmetic exactly, including the fixed neighbour
/// order (`-x, +x, -y, +y, -z, +z`) so the accumulation matches face-for-face.
fn relax_cell(
    dims: GridDims,
    cell_types: &[CellType],
    pressure: &[f32],
    faces: &Faces<'_>,
    rhs_scale: f32,
    omega: f32,
    i: u32,
    j: u32,
    k: u32,
) -> CellUpdate {
    let mut diag = 0.0f32;
    let mut sum = 0.0f32;
    let mut accumulate = |ti: u32, tj: u32, tk: u32| match cell_types[dims.cell_idx(ti, tj, tk)] {
        CellType::Solid => {}
        CellType::Air => diag += 1.0,
        CellType::Fluid => {
            diag += 1.0;
            sum += pressure[dims.cell_idx(ti, tj, tk)];
        }
    };
    if i > 0 {
        accumulate(i - 1, j, k);
    }
    if i + 1 < dims.nx {
        accumulate(i + 1, j, k);
    }
    if j > 0 {
        accumulate(i, j - 1, k);
    }
    if j + 1 < dims.ny {
        accumulate(i, j + 1, k);
    }
    if k > 0 {
        accumulate(i, j, k - 1);
    }
    if k + 1 < dims.nz {
        accumulate(i, j, k + 1);
    }
    if diag <= 0.0 {
        return CellUpdate {
            value: pressure[dims.cell_idx(i, j, k)],
            active: false,
        };
    }
    let raw = faces.raw_divergence(i, j, k);
    let target = (sum - rhs_scale * raw) / diag;
    let cur = pressure[dims.cell_idx(i, j, k)];
    CellUpdate {
        value: cur + omega * (target - cur),
        active: true,
    }
}

/// Projects the concatenated `[u | v | w]` velocity field in place, returning
/// the row-major pressure field.
///
/// Runs `cfg.iterations` red-black `SOR` sweeps of the pressure Poisson solve,
/// then subtracts the pressure gradient from the fluid faces.
#[must_use]
pub fn project(
    dims: GridDims,
    cell_types: &[CellType],
    velocity: &mut [f32],
    cfg: PressureConfig,
) -> Vec<f32> {
    let mut pressure = vec![0.0f32; dims.cell_count()];
    let rhs_scale = cfg.density * dims.dx * dims.dx / cfg.dt;
    let omega = cfg.effective_over_relaxation();

    for _ in 0..cfg.iterations {
        for colour in 0..2u32 {
            sweep_colour(
                dims,
                cell_types,
                &mut pressure,
                velocity,
                rhs_scale,
                omega,
                colour,
            );
        }
    }

    subtract_gradient(dims, cell_types, &pressure, velocity, cfg);
    pressure
}

/// Relaxes every fluid cell of one `colour` (`(i + j + k)` parity) once.
fn sweep_colour(
    dims: GridDims,
    cell_types: &[CellType],
    pressure: &mut [f32],
    velocity: &[f32],
    rhs_scale: f32,
    omega: f32,
    colour: u32,
) {
    let faces = Faces { dims, velocity };
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 0..dims.nx {
                if (i + j + k) & 1 != colour {
                    continue;
                }
                if cell_types[dims.cell_idx(i, j, k)] != CellType::Fluid {
                    continue;
                }
                let update = relax_cell(
                    dims, cell_types, pressure, &faces, rhs_scale, omega, i, j, k,
                );
                if update.active {
                    pressure[dims.cell_idx(i, j, k)] = update.value;
                }
            }
        }
    }
}

/// Subtracts the pressure gradient from the faces separating fluid cells,
/// removing the divergent component. Faces touching solids are zeroed.
fn subtract_gradient(
    dims: GridDims,
    cell_types: &[CellType],
    pressure: &[f32],
    velocity: &mut [f32],
    cfg: PressureConfig,
) {
    let scale = cfg.dt / (cfg.density * dims.dx);
    let u_off = dims.u_offset();
    let v_off = dims.v_offset();
    let w_off = dims.w_offset();

    // u faces between (i-1) and (i).
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 1..dims.nx {
                let left = cell_types[dims.cell_idx(i - 1, j, k)];
                let right = cell_types[dims.cell_idx(i, j, k)];
                let face = u_off + dims.u_idx(i, j, k);
                if left == CellType::Solid || right == CellType::Solid {
                    velocity[face] = 0.0;
                } else if left == CellType::Fluid || right == CellType::Fluid {
                    let grad =
                        pressure[dims.cell_idx(i, j, k)] - pressure[dims.cell_idx(i - 1, j, k)];
                    velocity[face] -= scale * grad / dims.dx;
                }
            }
        }
    }
    // v faces between (j-1) and (j).
    for k in 0..dims.nz {
        for j in 1..dims.ny {
            for i in 0..dims.nx {
                let down = cell_types[dims.cell_idx(i, j - 1, k)];
                let up = cell_types[dims.cell_idx(i, j, k)];
                let face = v_off + dims.v_idx(i, j, k);
                if down == CellType::Solid || up == CellType::Solid {
                    velocity[face] = 0.0;
                } else if down == CellType::Fluid || up == CellType::Fluid {
                    let grad =
                        pressure[dims.cell_idx(i, j, k)] - pressure[dims.cell_idx(i, j - 1, k)];
                    velocity[face] -= scale * grad / dims.dx;
                }
            }
        }
    }
    // w faces between (k-1) and (k).
    for k in 1..dims.nz {
        for j in 0..dims.ny {
            for i in 0..dims.nx {
                let back = cell_types[dims.cell_idx(i, j, k - 1)];
                let front = cell_types[dims.cell_idx(i, j, k)];
                let face = w_off + dims.w_idx(i, j, k);
                if back == CellType::Solid || front == CellType::Solid {
                    velocity[face] = 0.0;
                } else if back == CellType::Fluid || front == CellType::Fluid {
                    let grad =
                        pressure[dims.cell_idx(i, j, k)] - pressure[dims.cell_idx(i, j, k - 1)];
                    velocity[face] -= scale * grad / dims.dx;
                }
            }
        }
    }
}

/// The maximum absolute divergence over all fluid cells, a residual measure for
/// tests and diagnostics. The divergence is `raw_divergence / dx`.
#[must_use]
pub fn max_fluid_divergence(dims: GridDims, cell_types: &[CellType], velocity: &[f32]) -> f32 {
    let faces = Faces { dims, velocity };
    let mut m = 0.0f32;
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 0..dims.nx {
                if cell_types[dims.cell_idx(i, j, k)] == CellType::Fluid {
                    let d = (faces.raw_divergence(i, j, k) / dims.dx).abs();
                    m = m.max(d);
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

    /// Builds the cell classification for a 12-cube with a one-thick solid wall
    /// shell and a fluid block in the interior, mirroring the reference scene in
    /// `prism_physics_core`'s pressure test.
    fn walled_fluid_block(dims: GridDims) -> Vec<CellType> {
        let mut cells = vec![CellType::Air; dims.cell_count()];
        for k in 0..dims.nz {
            for j in 0..dims.ny {
                for i in 0..dims.nx {
                    let on_wall = i == 0
                        || j == 0
                        || k == 0
                        || i == dims.nx - 1
                        || j == dims.ny - 1
                        || k == dims.nz - 1;
                    if on_wall {
                        cells[dims.cell_idx(i, j, k)] = CellType::Solid;
                    }
                }
            }
        }
        for k in 3..9 {
            for j in 3..9 {
                for i in 3..9 {
                    cells[dims.cell_idx(i, j, k)] = CellType::Fluid;
                }
            }
        }
        cells
    }

    /// A concatenated `[u | v | w]` field carrying a divergent `u` ramp.
    fn divergent_field(dims: GridDims) -> Vec<f32> {
        let mut velocity = vec![0.0f32; dims.face_total()];
        let u_off = dims.u_offset();
        for k in 0..dims.nz {
            for j in 0..dims.ny {
                for i in 0..=dims.nx {
                    velocity[u_off + dims.u_idx(i, j, k)] = (i as f32 - 6.0) * 0.05;
                }
            }
        }
        velocity
    }

    #[test]
    fn projection_reduces_divergence() {
        let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
        let cells = walled_fluid_block(dims);
        let mut velocity = divergent_field(dims);

        let before = max_fluid_divergence(dims, &cells, &velocity);
        let cfg = PressureConfig::new(1.0, 1.0e-2, 1.7, 120);
        let _pressure = project(dims, &cells, &mut velocity, cfg);
        let after = max_fluid_divergence(dims, &cells, &velocity);

        assert!(after < before * 0.1, "before={before} after={after}");
        assert!(after < 1.0e-2, "residual too large: {after}");
    }

    #[test]
    fn over_relaxation_is_clamped() {
        assert!(
            (PressureConfig::new(1.0, 1.0, 0.5, 1).effective_over_relaxation() - 1.0).abs() < 1e-6
        );
        assert!(
            (PressureConfig::new(1.0, 1.0, 5.0, 1).effective_over_relaxation() - 1.99).abs() < 1e-6
        );
    }

    #[test]
    fn solid_faces_are_zeroed() {
        // A single fluid cell wrapped in solids: every face touches a solid and
        // must be zeroed by the gradient subtraction.
        let dims = GridDims::new(3, 3, 3, 0.1, Vec3::ZERO);
        let mut cells = vec![CellType::Solid; dims.cell_count()];
        cells[dims.cell_idx(1, 1, 1)] = CellType::Fluid;
        let mut velocity = vec![1.0f32; dims.face_total()];
        let cfg = PressureConfig::new(1.0, 1.0e-2, 1.7, 4);
        let _pressure = project(dims, &cells, &mut velocity, cfg);
        // The six faces of the central cell must have been forced to zero.
        let u_off = dims.u_offset();
        let v_off = dims.v_offset();
        let w_off = dims.w_offset();
        assert_eq!(velocity[u_off + dims.u_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[u_off + dims.u_idx(2, 1, 1)], 0.0);
        assert_eq!(velocity[v_off + dims.v_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[v_off + dims.v_idx(1, 2, 1)], 0.0);
        assert_eq!(velocity[w_off + dims.w_idx(1, 1, 1)], 0.0);
        assert_eq!(velocity[w_off + dims.w_idx(1, 1, 2)], 0.0);
    }
}
