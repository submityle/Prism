//! `FLIP`/`APIC` particle-grid fluid transfer, projection, and blend
//! scheduling.
//!
//! For large or film-grade volumes the fluid is stepped on a staggered `MAC`
//! grid rather than by particle constraints: particle velocities are
//! splatted to the grid (`P2G`), an incompressible pressure projection removes
//! divergence, and the corrected grid velocities are gathered back to the
//! particles (`G2P`). `FLIP` gathers the *change* in grid velocity for low
//! numerical dissipation, `PIC` gathers the absolute velocity for stability,
//! and the two are blended; `APIC` additionally carries a per-particle affine
//! velocity field so angular momentum survives the round trip without the
//! `PIC` smoothing or the `FLIP` noise.
//!
//! This module owns the *scheduling and per-cell/per-particle numerics* of that
//! loop — the transfer weights, the blend, the affine reconstruction, the
//! divergence measure, the pressure-solver choice, and the sub-step budget — as
//! pure, deterministic functions. The large-grid `Multigrid` path drives the
//! deterministic CPU reference solve in [`super::pressure_multigrid`]; the
//! small/mid `Jacobi` and `Conjugate-Gradient` paths stay shared GPU services,
//! so here we choose the solver and, when `Multigrid` is chosen, dispatch the
//! projection to that reference solver. Trilinear weights form a
//! partition of unity, the blend and affine transfers reproduce constant and
//! linear fields exactly, and only `sqrt` is used. No `f32` equality tests and
//! no AI/ML.

use super::pressure_multigrid::{self, MultigridConfig, SolveReport};
use super::{Vec3, EPS};

/// Tuning for one `FLIP`/`APIC` fluid domain's step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipParams {
    /// `FLIP`/`PIC` blend in `0..=1`: `0` is pure `PIC` (stable, dissipative),
    /// `1` is pure `FLIP` (energetic, noisier). Production values sit high,
    /// around `0.95`.
    pub flip_blend: f32,
    /// Whether the affine (`APIC`) velocity field is carried through the
    /// transfer.
    pub use_affine: bool,
    /// `CFL` number bounding the sub-step so no particle crosses more than this
    /// fraction of a cell per step (`> 0`).
    pub cfl: f32,
    /// Cell edge length of the `MAC` grid (`> 0`).
    pub dx: f32,
}

/// Trilinear interpolation weights for the eight cell corners.
///
/// Given the fractional position `(fx, fy, fz)` inside a cell (each clamped to
/// `0..=1`), returns the eight corner weights ordered with x fastest, then y,
/// then z. They are non-negative and sum to exactly one (a partition of unity),
/// which is what makes `P2G` conservative and `G2P` unbiased.
#[must_use]
pub fn trilinear_weights(fx: f32, fy: f32, fz: f32) -> [f32; 8] {
    let x = fx.clamp(0.0, 1.0);
    let y = fy.clamp(0.0, 1.0);
    let z = fz.clamp(0.0, 1.0);
    let xs = [1.0 - x, x];
    let ys = [1.0 - y, y];
    let zs = [1.0 - z, z];
    let mut w = [0.0_f32; 8];
    let mut i = 0;
    let mut cz = 0;
    while cz < 2 {
        let mut cy = 0;
        while cy < 2 {
            let mut cx = 0;
            while cx < 2 {
                w[i] = xs[cx] * ys[cy] * zs[cz];
                i += 1;
                cx += 1;
            }
            cy += 1;
        }
        cz += 1;
    }
    w
}

/// Blends the `PIC` and `FLIP` velocity updates for one particle.
///
/// Returns `(1 - alpha) * pic + alpha * flip`, with `alpha` clamped to `0..=1`.
/// At `alpha = 0` it is the stable `PIC` velocity, at `alpha = 1` the
/// low-dissipation `FLIP` velocity, and it varies monotonically between them.
#[must_use]
pub fn blend_flip_pic(pic: Vec3, flip: Vec3, alpha: f32) -> Vec3 {
    let a = alpha.clamp(0.0, 1.0);
    pic.scale(1.0 - a).add(flip.scale(a))
}

/// Reconstructs an `APIC` velocity at a grid node from a particle's base
/// velocity and affine matrix.
///
/// The affine velocity field is `v(x) = v_p + C_p (x - x_p)`, with `C_p` given
/// as its three rows. `offset` is `x - x_p`, the node position relative to the
/// particle. A zero affine matrix collapses to plain `PIC`-style transfer, and
/// a linear velocity field is reproduced exactly — the property that gives
/// `APIC` its angular-momentum conservation.
#[must_use]
pub fn apic_velocity(base: Vec3, affine_rows: [Vec3; 3], offset: Vec3) -> Vec3 {
    Vec3::new(
        base.x + affine_rows[0].dot(offset),
        base.y + affine_rows[1].dot(offset),
        base.z + affine_rows[2].dot(offset),
    )
}

/// The six staggered face velocities bordering one `MAC` cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceVelocities {
    /// Velocity through the `+x` face.
    pub x_pos: f32,
    /// Velocity through the `-x` face.
    pub x_neg: f32,
    /// Velocity through the `+y` face.
    pub y_pos: f32,
    /// Velocity through the `-y` face.
    pub y_neg: f32,
    /// Velocity through the `+z` face.
    pub z_pos: f32,
    /// Velocity through the `-z` face.
    pub z_neg: f32,
}

/// Discrete velocity divergence at a `MAC` cell.
///
/// Returns `((x_pos - x_neg) + (y_pos - y_neg) + (z_pos - z_neg)) / dx`, the
/// right-hand side the pressure projection drives to zero for incompressible
/// flow. A uniform flow (equal opposing faces) is divergence-free; net outflow
/// is positive and net inflow negative. A non-positive `dx` yields `0`.
#[must_use]
pub fn cell_divergence(faces: FaceVelocities, dx: f32) -> f32 {
    if dx <= EPS {
        return 0.0;
    }
    ((faces.x_pos - faces.x_neg) + (faces.y_pos - faces.y_neg) + (faces.z_pos - faces.z_neg)) / dx
}

/// The pressure-projection solver a domain is routed to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PressureSolver {
    /// Damped `Jacobi` relaxation: cheapest per iteration, best for small grids
    /// and tight time budgets.
    Jacobi,
    /// `Conjugate-Gradient`: mid-size grids where `Jacobi` would need too many
    /// iterations.
    ConjugateGradient,
    /// Geometric `Multigrid`: large grids where only a hierarchy converges in a
    /// bounded number of passes.
    Multigrid,
}

/// Cell-count thresholds selecting the pressure solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureSolverThresholds {
    /// At or below this cell count, `Jacobi` is used.
    pub jacobi_max_cells: u32,
    /// Above `jacobi_max_cells` and at or below this, `Conjugate-Gradient` is
    /// used; larger grids use `Multigrid`.
    pub cg_max_cells: u32,
}

/// Chooses a pressure solver for a grid of `cell_count` cells.
///
/// Higher `quality_bias` in `0..=1` shrinks the thresholds so heavier solvers
/// engage sooner. The choice is monotonic in `cell_count` at a fixed bias: a
/// larger grid is never routed to a cheaper solver, so there is no pop as a
/// domain grows.
#[must_use]
pub fn select_pressure_solver(
    cell_count: u32,
    quality_bias: f32,
    thresholds: PressureSolverThresholds,
) -> PressureSolver {
    let bias = quality_bias.clamp(0.0, 1.0);
    // Up to a 50% threshold reduction at full quality bias.
    let shrink = 1.0 - 0.5 * bias;
    let jacobi_max = ((thresholds.jacobi_max_cells as f32) * shrink) as u32;
    let cg_max = ((thresholds.cg_max_cells as f32) * shrink) as u32;
    if cell_count <= jacobi_max {
        PressureSolver::Jacobi
    } else if cell_count <= cg_max.max(jacobi_max) {
        PressureSolver::ConjugateGradient
    } else {
        PressureSolver::Multigrid
    }
}

/// `CFL`-bounded sub-step for a `FLIP` domain.
///
/// Returns the largest `dt` with `max_speed * dt <= cfl * dx`, i.e.
/// `cfl * dx / max_speed`. Slower flows and coarser grids permit larger steps;
/// a still domain (`max_speed` near zero) returns `max_dt` unchanged.
#[must_use]
pub fn flip_cfl_timestep(max_speed: f32, dx: f32, cfl: f32, max_dt: f32) -> f32 {
    let speed = max_speed.max(0.0);
    if speed <= EPS || dx <= EPS {
        return max_dt.max(0.0);
    }
    let limit = cfl.max(0.0) * dx / speed;
    limit.min(max_dt.max(0.0))
}

/// The ordered pass schedule for one `FLIP`/`APIC` step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipPlan {
    /// Whether the affine (`APIC`) transfer is active.
    pub affine: bool,
    /// The clamped `FLIP`/`PIC` blend applied in `G2P`.
    pub blend: f32,
    /// The pressure solver the projection stage will drive.
    pub solver: PressureSolver,
}

/// Builds the step schedule from the parameters and grid size.
///
/// The blend is clamped into `0..=1` and the pressure solver is chosen from the
/// cell count. Fully determined by its inputs, so repeated planning is
/// reproducible.
#[must_use]
pub fn plan_flip(
    params: FlipParams,
    cell_count: u32,
    quality_bias: f32,
    thresholds: PressureSolverThresholds,
) -> FlipPlan {
    FlipPlan {
        affine: params.use_affine,
        blend: params.flip_blend.clamp(0.0, 1.0),
        solver: select_pressure_solver(cell_count, quality_bias, thresholds),
    }
}

/// Builds a `Multigrid` V-cycle schedule from the `FLIP` quality bias.
///
/// Starts from [`MultigridConfig::balanced`] and, as `quality_bias` rises in
/// `0..=1`, adds V-cycles and tightens the residual tolerance so film-grade
/// domains converge further while interactive domains stay cheap. The result
/// always satisfies [`MultigridConfig::is_valid`], so it can be handed to
/// [`super::pressure_multigrid::solve`] directly.
#[must_use]
pub fn multigrid_schedule(quality_bias: f32) -> MultigridConfig {
    let bias = quality_bias.clamp(0.0, 1.0);
    let base = MultigridConfig::balanced();
    // Up to +40 extra V-cycles and a 100x tighter tolerance at full quality.
    let extra_cycles = (40.0 * bias) as u32;
    let tol_scale = 1.0 / (1.0 + 99.0 * bias);
    MultigridConfig {
        max_cycles: base.max_cycles + extra_cycles,
        tolerance: base.tolerance * tol_scale,
        ..base
    }
}

/// Dispatches the pressure projection chosen by [`plan_flip`].
///
/// Only the large-grid [`PressureSolver::Multigrid`] path runs on the CPU here,
/// driving [`super::pressure_multigrid::solve`] over the divergence right-hand
/// side on a vertex-centred `grid_size^3` grid with node spacing
/// `cell_spacing`. The `Jacobi` and `Conjugate-Gradient` paths are the
/// small/mid shared GPU services and return `None`, so a caller can tell a CPU
/// reference solve ran from a dispatched GPU path. An invalid grid size or a
/// right-hand side whose length is not `grid_size^3` also yields `None` rather
/// than panicking.
#[must_use]
pub fn project_pressure(
    plan: FlipPlan,
    divergence_rhs: &[f32],
    grid_size: usize,
    cell_spacing: f32,
    quality_bias: f32,
) -> Option<SolveReport> {
    match plan.solver {
        PressureSolver::Multigrid => {
            if !pressure_multigrid::is_valid_level_size(grid_size)
                || divergence_rhs.len() != grid_size * grid_size * grid_size
            {
                return None;
            }
            Some(pressure_multigrid::solve(
                divergence_rhs,
                grid_size,
                cell_spacing,
                multigrid_schedule(quality_bias),
            ))
        }
        PressureSolver::Jacobi | PressureSolver::ConjugateGradient => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const THRESHOLDS: PressureSolverThresholds = PressureSolverThresholds {
        jacobi_max_cells: 10_000,
        cg_max_cells: 1_000_000,
    };

    #[test]
    fn trilinear_weights_are_a_partition_of_unity() {
        for &(fx, fy, fz) in &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.2, 0.7, 0.9),
        ] {
            let w = trilinear_weights(fx, fy, fz);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < EPS, "weights must sum to one: {sum}");
            assert!(w.iter().all(|&x| x >= 0.0), "weights must be non-negative");
        }
        // At a corner all weight lands on that corner.
        let corner = trilinear_weights(0.0, 0.0, 0.0);
        assert!((corner[0] - 1.0).abs() < EPS);
        // Out-of-range fractions clamp instead of blowing up.
        let clamped = trilinear_weights(2.0, -1.0, 0.5);
        let sum: f32 = clamped.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn blend_interpolates_pic_and_flip() {
        let pic = Vec3::new(1.0, 0.0, 0.0);
        let flip = Vec3::new(3.0, 0.0, 0.0);
        assert_eq!(blend_flip_pic(pic, flip, 0.0), pic);
        assert_eq!(blend_flip_pic(pic, flip, 1.0), flip);
        let mid = blend_flip_pic(pic, flip, 0.5);
        assert!((mid.x - 2.0).abs() < EPS);
        // Monotonic in alpha along the blend axis.
        let mut prev = blend_flip_pic(pic, flip, 0.0).x;
        let mut a = 0.0;
        while a <= 1.0 {
            let x = blend_flip_pic(pic, flip, a).x;
            assert!(x + EPS >= prev);
            prev = x;
            a += 0.1;
        }
        // Out-of-range alpha clamps.
        assert_eq!(blend_flip_pic(pic, flip, 2.0), flip);
    }

    #[test]
    fn apic_reproduces_constant_and_linear_fields() {
        let base = Vec3::new(2.0, -1.0, 0.5);
        // Zero affine -> constant transfer.
        let zero = [Vec3::ZERO; 3];
        assert_eq!(apic_velocity(base, zero, Vec3::new(1.0, 2.0, 3.0)), base);
        // A pure x-gradient in the x-component: v_x = base_x + 1 * offset_x.
        let rows = [Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, Vec3::ZERO];
        let v = apic_velocity(base, rows, Vec3::new(0.5, 9.0, 9.0));
        assert!((v.x - (base.x + 0.5)).abs() < EPS);
        assert!((v.y - base.y).abs() < EPS);
    }

    #[test]
    fn divergence_is_zero_for_uniform_flow_and_signed_otherwise() {
        // Equal opposing faces: incompressible.
        let uniform = FaceVelocities {
            x_pos: 1.0,
            x_neg: 1.0,
            y_pos: -2.0,
            y_neg: -2.0,
            z_pos: 0.5,
            z_neg: 0.5,
        };
        assert!(cell_divergence(uniform, 0.1).abs() < EPS);
        // Net outflow: positive divergence.
        let source = FaceVelocities {
            x_pos: 2.0,
            x_neg: -2.0,
            y_pos: 0.0,
            y_neg: 0.0,
            z_pos: 0.0,
            z_neg: 0.0,
        };
        assert!(cell_divergence(source, 1.0) > 0.0);
        // Degenerate spacing is inert.
        assert_eq!(cell_divergence(source, 0.0), 0.0);
    }

    #[test]
    fn pressure_solver_selection_is_monotonic_in_grid_size() {
        // Small -> Jacobi, mid -> CG, large -> Multigrid.
        assert_eq!(
            select_pressure_solver(500, 0.0, THRESHOLDS),
            PressureSolver::Jacobi
        );
        assert_eq!(
            select_pressure_solver(100_000, 0.0, THRESHOLDS),
            PressureSolver::ConjugateGradient
        );
        assert_eq!(
            select_pressure_solver(5_000_000, 0.0, THRESHOLDS),
            PressureSolver::Multigrid
        );
        // Monotonic: never a cheaper solver as the grid grows.
        let rank = |s: PressureSolver| match s {
            PressureSolver::Jacobi => 0,
            PressureSolver::ConjugateGradient => 1,
            PressureSolver::Multigrid => 2,
        };
        let mut prev = rank(select_pressure_solver(0, 0.3, THRESHOLDS));
        let mut cells = 0u32;
        while cells <= 4_000_000 {
            let r = rank(select_pressure_solver(cells, 0.3, THRESHOLDS));
            assert!(r >= prev, "solver must not get cheaper as cells grow");
            prev = r;
            cells += 100_000;
        }
    }

    #[test]
    fn cfl_timestep_shrinks_with_speed_and_respects_cap() {
        let fast = flip_cfl_timestep(10.0, 0.1, 1.0, 1.0);
        let slow = flip_cfl_timestep(1.0, 0.1, 1.0, 1.0);
        assert!(slow >= fast, "slower flow permits a larger step");
        // Capped by max_dt.
        assert!((flip_cfl_timestep(0.001, 1.0, 1.0, 0.5) - 0.5).abs() < EPS);
        // Still domain returns the cap.
        assert!((flip_cfl_timestep(0.0, 0.1, 1.0, 0.25) - 0.25).abs() < EPS);
    }

    #[test]
    fn plan_flip_is_deterministic_and_clamped() {
        let params = FlipParams {
            flip_blend: 1.5,
            use_affine: true,
            cfl: 1.0,
            dx: 0.1,
        };
        let plan = plan_flip(params, 500, 0.0, THRESHOLDS);
        assert!(plan.affine);
        assert!((plan.blend - 1.0).abs() < EPS);
        assert_eq!(plan.solver, PressureSolver::Jacobi);
        assert_eq!(plan_flip(params, 500, 0.0, THRESHOLDS), plan);
    }

    #[test]
    fn multigrid_schedule_is_valid_and_monotonic() {
        let mut prev = multigrid_schedule(0.0);
        assert!(prev.is_valid());
        let mut bias = 0.1f32;
        while bias <= 1.0 + EPS {
            let cfg = multigrid_schedule(bias);
            assert!(cfg.is_valid(), "schedule must stay valid at bias {bias}");
            assert!(
                cfg.max_cycles >= prev.max_cycles,
                "cycle budget never shrinks"
            );
            assert!(
                cfg.tolerance <= prev.tolerance + EPS,
                "tolerance never loosens with higher quality"
            );
            prev = cfg;
            bias += 0.1;
        }
    }

    #[test]
    fn project_pressure_routes_multigrid_to_a_cpu_solve() {
        let n = 9usize;
        let h = 1.0f32 / (n as f32 - 1.0);
        // Smooth manufactured solution, zero on the boundary.
        let denom = (n - 1) as f32;
        let mut exact = vec![0.0f32; n * n * n];
        for z in 1..n - 1 {
            let zf = z as f32 / denom;
            for y in 1..n - 1 {
                let yf = y as f32 / denom;
                for x in 1..n - 1 {
                    let xf = x as f32 / denom;
                    let v = (xf * (1.0 - xf)) * (yf * (1.0 - yf)) * (zf * (1.0 - zf));
                    exact[(z * n + y) * n + x] = v;
                }
            }
        }
        let rhs = pressure_multigrid::apply_operator(&exact, n, h);
        let params = FlipParams {
            flip_blend: 0.95,
            use_affine: true,
            cfl: 1.0,
            dx: h,
        };
        // A huge cell count forces the Multigrid branch.
        let plan = plan_flip(params, 50_000_000, 1.0, THRESHOLDS);
        assert_eq!(plan.solver, PressureSolver::Multigrid);
        let report =
            project_pressure(plan, &rhs, n, h, 1.0).expect("multigrid must dispatch a CPU solve");
        assert_eq!(report.pressure.len(), n * n * n);
        assert!(report.cycles > 0 && report.cycles <= multigrid_schedule(1.0).max_cycles);
        assert!(report.residual <= multigrid_schedule(1.0).tolerance);
    }

    #[test]
    fn project_pressure_leaves_small_solvers_to_gpu() {
        let n = 9usize;
        let rhs = vec![0.0f32; n * n * n];
        let params = FlipParams {
            flip_blend: 0.5,
            use_affine: false,
            cfl: 1.0,
            dx: 0.1,
        };
        let jacobi = plan_flip(params, 500, 0.0, THRESHOLDS);
        assert_eq!(jacobi.solver, PressureSolver::Jacobi);
        assert!(project_pressure(jacobi, &rhs, n, 0.1, 0.0).is_none());
        let cg = plan_flip(params, 100_000, 0.0, THRESHOLDS);
        assert_eq!(cg.solver, PressureSolver::ConjugateGradient);
        assert!(project_pressure(cg, &rhs, n, 0.1, 0.0).is_none());
    }

    #[test]
    fn project_pressure_rejects_a_bad_grid() {
        let n = 10usize; // not 2^L + 1
        let rhs = vec![1.0f32; n * n * n];
        let params = FlipParams {
            flip_blend: 0.9,
            use_affine: true,
            cfl: 1.0,
            dx: 0.1,
        };
        let plan = plan_flip(params, 50_000_000, 1.0, THRESHOLDS);
        assert_eq!(plan.solver, PressureSolver::Multigrid);
        assert!(project_pressure(plan, &rhs, n, 0.1, 1.0).is_none());
        // A valid grid size but a mismatched RHS length also yields None.
        assert!(project_pressure(plan, &rhs, 9, 0.1, 1.0).is_none());
    }
}
