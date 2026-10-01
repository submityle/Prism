//! Viscous diffusion relaxation solver — step 2 of the grid-fluid pipeline
//! (design §10).
//!
//! The stable-fluids pipeline documented in [`super::fluid`] runs
//! advection → **diffusion** → pressure projection → velocity write-back. The
//! sibling module pins the schedule and the field contracts — the
//! [`FluidStage::Diffuse`] pass reads the post-advection scratch velocity
//! (`FluidField::VelocityScratch`) and writes the primary velocity field
//! (`FluidField::Velocity`) — and it ships a complete pressure solver in
//! [`jacobi_pressure_solve`], but it leaves the viscous-diffusion kernel
//! unimplemented. This module fills exactly that gap, reusing the sibling's
//! [`GridResolution`] grid, its row-major [`GridResolution::linear_index`]
//! layout, its homogeneous-boundary neighbor convention, and its [`Vec3`]
//! math; it never redefines the grid.
//!
//! # The physics
//!
//! Newtonian viscosity diffuses momentum. For a kinematic viscosity `ν` the
//! velocity field obeys the vector heat equation `∂v/∂t = ν·∇²v`. Discretizing
//! the Laplacian `∇²v` with the standard 7-point stencil on a grid of spacing
//! `h`, an **explicit** Euler step is only stable while `ν·dt/h²` stays below
//! `1/6`, which is far too strict for the large steps a cinematic solver takes.
//! Following `Stam`'s stable-fluids method we therefore step diffusion
//! *implicitly*: solve `(I − α·L)·v' = v` for the new field `v'`, where `L` is
//! the discrete Laplacian and
//!
//! ```text
//! α = ν·dt/h²
//! ```
//!
//! is the dimensionless diffusion number ([`diffusion_alpha`]). The implicit
//! system is symmetric and diagonally dominant for any `α ≥ 0`, so it is
//! unconditionally stable and is solved with the very same damped `Jacobi`
//! relaxation the pressure projection uses.
//!
//! # The relaxation
//!
//! Writing the Laplacian stencil out, each interior cell couples to its six
//! axis-aligned face neighbors, so one `Jacobi` sweep is
//!
//! ```text
//! v' = (v + α·Σ₆ neighbors) / (1 + 6·α)
//! ```
//!
//! where `v` is the (constant) pre-diffusion field that forms the right-hand
//! side, the neighbor sum uses the previous iterate, and the sweep is repeated
//! for a fixed iteration count ([`viscous_diffuse`]). Each of the three
//! velocity components relaxes independently; because the stencil weights are
//! identical per axis, the [`Vec3`] arithmetic handles all three at once.
//!
//! # Boundaries
//!
//! Two wall models are supported, matching the stencil the sibling pressure
//! solver already assumes outside the grid:
//!
//! * [`DiffusionBoundary::Fixed`] — a no-slip solid wall. The velocity outside
//!   the grid is pinned to zero (homogeneous Dirichlet), exactly like the
//!   `neighbor_sum` convention behind [`jacobi_pressure_solve`]. The diagonal
//!   stays at the full six faces, so boundary cells are pulled toward the zero
//!   wall velocity.
//! * [`DiffusionBoundary::Free`] — an open / zero-gradient wall. A missing
//!   neighbor mirrors the center cell (homogeneous Neumann), so no momentum
//!   flux crosses the boundary. This is expressed by lowering the diagonal
//!   count to the number of neighbors that actually exist, which both keeps a
//!   uniform field uniform and conserves the field's total momentum at
//!   convergence.
//!
//! # Determinism
//!
//! Determinism matches the sibling particle modules (design §29): the only
//! floating-point primitive beyond ordinary arithmetic is `sqrt`, used solely
//! inside the optional [`diffusion_residual_l2`] convergence probe (through
//! [`Vec3::length_squared`] accumulation). There are no transcendental calls
//! (no `sin` / `cos` / `exp` / `ln` / `pow`), no hashing, and no `RNG`: every
//! sweep is multiply-add plus one reciprocal, so the `CPU` reference is
//! bit-reproducible against a future `GPU` compute kernel.

use alloc::vec;
use alloc::vec::Vec;

use super::fluid::GridResolution;
use super::{Vec3, EPS_LEN_SQ};

/// The number of axis-aligned face neighbors a voxel has in a 3-D grid.
///
/// The implicit diffusion stencil couples each cell to these six faces, so the
/// [`DiffusionBoundary::Fixed`] diagonal is `1 + 6·α`.
const FACE_NEIGHBOR_COUNT: f32 = 6.0;

/// How the diffusion stencil treats the velocity just outside the grid
/// (design §10).
///
/// Both modes read the same in-grid neighbors; they differ only in what an
/// out-of-grid neighbor contributes, which is encoded entirely in the implicit
/// diagonal (see [`diffuse_relax_sweep`]).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiffusionBoundary {
    /// No-slip solid wall: the velocity outside the grid is zero (homogeneous
    /// Dirichlet), matching the `neighbor_sum` convention behind
    /// [`jacobi_pressure_solve`]. The diagonal keeps all six faces, so a
    /// boundary cell is dragged toward the stationary wall.
    Fixed,
    /// Open / zero-gradient wall: an out-of-grid neighbor mirrors the center
    /// cell (homogeneous Neumann), so no momentum crosses the boundary. The
    /// diagonal drops to the count of existing neighbors, which keeps uniform
    /// flow uniform and conserves total momentum at convergence.
    Free,
}

/// The tunable inputs of one viscous-diffusion solve (design §10).
///
/// The diffusion number `α = ν·dt/h²` is derived from `viscosity`, `dt`, and
/// `cell_size` by [`diffusion_alpha`]; `iterations` fixes how many `Jacobi`
/// sweeps [`viscous_diffuse`] runs; `boundary` selects the wall model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionParams {
    /// Kinematic viscosity `ν`. Expected non-negative; `0` disables diffusion.
    pub viscosity: f32,
    /// Simulation time step `dt` for this frame.
    pub dt: f32,
    /// Uniform grid spacing `h` (cell size) used to form `1 / h²`.
    pub cell_size: f32,
    /// Number of `Jacobi` relaxation sweeps to run.
    pub iterations: u32,
    /// Wall model applied outside the grid.
    pub boundary: DiffusionBoundary,
}

/// The outcome of a viscous-diffusion solve (design §10).
#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionResult {
    /// The diffused velocity field, row-major, one [`Vec3`] per voxel.
    pub velocity: Vec<Vec3>,
    /// How many `Jacobi` sweeps actually ran.
    pub iterations_run: u32,
}

/// The dimensionless diffusion number `α = ν·dt/h²` (design §10).
///
/// This is the only place the physical parameters are combined. The `1 / h²`
/// is guarded against a degenerate (zero) cell size exactly as the sibling
/// `cfl_number` guards its division: a non-positive-magnitude `cell_size`
/// yields `0` (no diffusion) rather than a division by zero. Multiply and one
/// reciprocal only.
#[must_use]
pub fn diffusion_alpha(params: DiffusionParams) -> f32 {
    let cell = params.cell_size;
    if cell.abs() > EPS_LEN_SQ {
        (params.viscosity * params.dt) / (cell * cell)
    } else {
        0.0
    }
}

/// Sum of the in-grid face-neighbor velocities around `(x, y, z)`, together
/// with the count of neighbors that actually exist.
///
/// Mirrors the sibling scalar `neighbor_sum`: a neighbor outside the grid is
/// omitted from the sum. Under [`DiffusionBoundary::Fixed`] the omitted term is
/// the zero wall velocity (homogeneous Dirichlet); under
/// [`DiffusionBoundary::Free`] the returned count lets the caller lower the
/// diagonal so the wall carries zero normal gradient (homogeneous Neumann).
/// Multiply-add only.
fn neighbor_velocity_sum(
    field: &[Vec3],
    res: GridResolution,
    x: u32,
    y: u32,
    z: u32,
) -> (Vec3, u32) {
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    if x + 1 < res.nx {
        sum = sum.add(field[res.linear_index(x + 1, y, z) as usize]);
        count += 1;
    }
    if x > 0 {
        sum = sum.add(field[res.linear_index(x - 1, y, z) as usize]);
        count += 1;
    }
    if y + 1 < res.ny {
        sum = sum.add(field[res.linear_index(x, y + 1, z) as usize]);
        count += 1;
    }
    if y > 0 {
        sum = sum.add(field[res.linear_index(x, y - 1, z) as usize]);
        count += 1;
    }
    if z + 1 < res.nz {
        sum = sum.add(field[res.linear_index(x, y, z + 1) as usize]);
        count += 1;
    }
    if z > 0 {
        sum = sum.add(field[res.linear_index(x, y, z - 1) as usize]);
        count += 1;
    }
    (sum, count)
}

/// The implicit diagonal count for a cell with `active` in-grid neighbors under
/// a given wall model.
///
/// [`DiffusionBoundary::Fixed`] always uses the full [`FACE_NEIGHBOR_COUNT`]
/// (the missing neighbors are the zero Dirichlet wall), while
/// [`DiffusionBoundary::Free`] uses only the existing neighbors so the wall is
/// a zero-gradient Neumann boundary.
fn diagonal_count(boundary: DiffusionBoundary, active: u32) -> f32 {
    match boundary {
        DiffusionBoundary::Fixed => FACE_NEIGHBOR_COUNT,
        // Widening `u32` -> `f32` is exact for the 0..=6 neighbor counts.
        DiffusionBoundary::Free => active as f32,
    }
}

/// Performs one damped `Jacobi` relaxation sweep of the implicit diffusion
/// system (design §10).
///
/// `source` is the constant right-hand side (the pre-diffusion velocity, i.e.
/// `FluidField::VelocityScratch`); `current` is the latest iterate whose
/// neighbors are read. Each cell is updated to
/// `(source + α·Σ neighbors) / (1 + diagonal·α)`, where `α` comes from
/// [`diffusion_alpha`] and `diagonal` is six for [`DiffusionBoundary::Fixed`]
/// or the live-neighbor count for [`DiffusionBoundary::Free`]. Because
/// `α ≥ 0` the denominator is at least one, so this never divides by zero. The
/// three velocity components relax together through [`Vec3`] arithmetic.
/// Returns an empty vector when the grid is empty or either slice is too short.
/// Multiply-add plus one reciprocal per cell.
#[must_use]
pub fn diffuse_relax_sweep(
    source: &[Vec3],
    current: &[Vec3],
    res: GridResolution,
    params: DiffusionParams,
) -> Vec<Vec3> {
    let count = res.voxel_count() as usize;
    if count == 0 || source.len() < count || current.len() < count {
        return Vec::new();
    }
    let alpha = diffusion_alpha(params);
    let mut next = vec![Vec3::ZERO; count];
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let (neighbor_sum, active) = neighbor_velocity_sum(current, res, x, y, z);
                let denom = 1.0 + diagonal_count(params.boundary, active) * alpha;
                let numerator = source[idx].add(neighbor_sum.scale(alpha));
                next[idx] = numerator.scale(1.0 / denom);
            }
        }
    }
    next
}

/// Solves the implicit viscous-diffusion system with a fixed number of damped
/// `Jacobi` sweeps (design §10).
///
/// Starts the iterate at `source` (the post-advection velocity field) and
/// applies [`diffuse_relax_sweep`] `params.iterations` times, holding `source`
/// as the constant right-hand side throughout. A zero iteration count returns
/// an unmodified copy of the field, and `ν = 0` (hence `α = 0`) is an exact
/// fixed point that leaves the field unchanged. Returns an empty field when the
/// grid is empty or `source` is too short.
#[must_use]
pub fn viscous_diffuse(
    source: &[Vec3],
    res: GridResolution,
    params: DiffusionParams,
) -> DiffusionResult {
    let count = res.voxel_count() as usize;
    if count == 0 || source.len() < count {
        return DiffusionResult {
            velocity: Vec::new(),
            iterations_run: 0,
        };
    }
    let mut current: Vec<Vec3> = source[..count].to_vec();
    let mut iterations_run = 0;
    while iterations_run < params.iterations {
        current = diffuse_relax_sweep(source, &current, res, params);
        iterations_run += 1;
    }
    DiffusionResult {
        velocity: current,
        iterations_run,
    }
}

/// `L2` residual of an iterate against the implicit diffusion system
/// (design §10).
///
/// Computes `sqrt(mean(‖source − ((1 + diagonal·α)·v − α·Σ neighbors)‖²))`,
/// i.e. the per-cell shortfall of the implicit equation `(I − α·L)·v = source`.
/// A shrinking residual across sweeps is the convergence signal for the
/// `Jacobi` relaxation, mirroring the sibling `pressure_residual_l2`. Returns
/// `0` when the grid is empty or a slice is too short. Multiply-add plus the
/// final `sqrt`.
#[must_use]
pub fn diffusion_residual_l2(
    source: &[Vec3],
    current: &[Vec3],
    res: GridResolution,
    params: DiffusionParams,
) -> f32 {
    let count = res.voxel_count() as usize;
    if count == 0 || source.len() < count || current.len() < count {
        return 0.0;
    }
    let alpha = diffusion_alpha(params);
    let mut sum_sq = 0.0;
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let (neighbor_sum, active) = neighbor_velocity_sum(current, res, x, y, z);
                let diagonal = diagonal_count(params.boundary, active);
                let lhs = current[idx]
                    .scale(1.0 + diagonal * alpha)
                    .sub(neighbor_sum.scale(alpha));
                let residual = source[idx].sub(lhs);
                sum_sq += residual.length_squared();
            }
        }
    }
    (sum_sq / count as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Comparison tolerance for the finite-precision golden checks.
    const F_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < F_EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// Bit-identical check used by the determinism test: compares raw `f32`
    /// bit patterns rather than numeric equality.
    fn bits_equal(a: Vec3, b: Vec3) -> bool {
        a.x.to_bits() == b.x.to_bits()
            && a.y.to_bits() == b.y.to_bits()
            && a.z.to_bits() == b.z.to_bits()
    }

    fn fields_bits_equal(a: &[Vec3], b: &[Vec3]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&p, &q)| bits_equal(p, q))
    }

    fn total_momentum(field: &[Vec3]) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for &v in field {
            sum = sum.add(v);
        }
        sum
    }

    fn params(viscosity: f32, iterations: u32, boundary: DiffusionBoundary) -> DiffusionParams {
        DiffusionParams {
            viscosity,
            dt: 1.0,
            cell_size: 1.0,
            iterations,
            boundary,
        }
    }

    #[test]
    fn alpha_is_viscosity_dt_over_h_squared() {
        let p = DiffusionParams {
            viscosity: 2.0,
            dt: 3.0,
            cell_size: 2.0,
            iterations: 1,
            boundary: DiffusionBoundary::Free,
        };
        // α = ν·dt/h² = 2·3 / 4 = 1.5.
        assert!(approx(diffusion_alpha(p), 1.5));
    }

    #[test]
    fn alpha_guards_degenerate_cell_size() {
        let p = DiffusionParams {
            viscosity: 2.0,
            dt: 3.0,
            cell_size: 0.0,
            iterations: 1,
            boundary: DiffusionBoundary::Free,
        };
        // A zero cell size yields no diffusion instead of a division by zero.
        assert!(approx(diffusion_alpha(p), 0.0));
    }

    #[test]
    fn zero_viscosity_leaves_field_unchanged() {
        let res = GridResolution::uniform(4);
        let count = res.voxel_count() as usize;
        let mut source = vec![Vec3::ZERO; count];
        // An arbitrary, non-uniform field.
        for (i, v) in source.iter_mut().enumerate() {
            let f = i as f32;
            *v = Vec3::new(f, -f, 0.5 * f);
        }
        let out = viscous_diffuse(&source, res, params(0.0, 8, DiffusionBoundary::Free));
        assert_eq!(out.iterations_run, 8);
        // ν = 0 -> α = 0 is an exact fixed point: bit-for-bit identical.
        assert!(fields_bits_equal(&out.velocity, &source));
    }

    #[test]
    fn zero_iterations_returns_source_copy() {
        let res = GridResolution::uniform(3);
        let count = res.voxel_count() as usize;
        let source = vec![Vec3::new(1.0, 2.0, 3.0); count];
        let out = viscous_diffuse(&source, res, params(0.7, 0, DiffusionBoundary::Fixed));
        assert_eq!(out.iterations_run, 0);
        assert!(fields_bits_equal(&out.velocity, &source));
    }

    #[test]
    fn uniform_field_stays_uniform_under_free_boundary() {
        let res = GridResolution::uniform(5);
        let count = res.voxel_count() as usize;
        let fill = Vec3::new(2.0, -1.0, 0.5);
        let source = vec![fill; count];
        let out = viscous_diffuse(&source, res, params(0.9, 20, DiffusionBoundary::Free));
        // Free (Neumann) walls carry zero gradient, so a constant field is a
        // fixed point everywhere, including the boundary cells.
        for &v in &out.velocity {
            assert!(approx_vec(v, fill));
        }
    }

    #[test]
    fn free_boundary_conserves_total_momentum() {
        let res = GridResolution::new(4, 4, 1);
        let count = res.voxel_count() as usize;
        let mut source = vec![Vec3::ZERO; count];
        // A single-cell momentum spike.
        source[res.linear_index(1, 2, 0) as usize] = Vec3::new(6.0, -3.0, 1.5);
        let before = total_momentum(&source);
        // Many sweeps drive the Jacobi iterate to the implicit solution, whose
        // Neumann walls admit no flux, so total momentum is conserved.
        let out = viscous_diffuse(&source, res, params(1.0, 400, DiffusionBoundary::Free));
        let after = total_momentum(&out.velocity);
        assert!(approx_vec(after, before));
    }

    #[test]
    fn diffusion_smooths_a_peak() {
        let res = GridResolution::new(5, 1, 1);
        let count = res.voxel_count() as usize;
        let mut source = vec![Vec3::ZERO; count];
        let peak_idx = res.linear_index(2, 0, 0) as usize;
        let neighbor_idx = res.linear_index(1, 0, 0) as usize;
        source[peak_idx] = Vec3::new(5.0, 0.0, 0.0);
        let out = viscous_diffuse(&source, res, params(1.0, 30, DiffusionBoundary::Free));
        // The peak drops and its neighbor rises: momentum spreads outward.
        assert!(out.velocity[peak_idx].x < source[peak_idx].x);
        assert!(out.velocity[neighbor_idx].x > source[neighbor_idx].x);
        // Still conserved on a Neumann domain.
        assert!(approx(total_momentum(&out.velocity).x, 5.0));
    }

    #[test]
    fn fixed_boundary_drains_toward_wall() {
        let res = GridResolution::uniform(3);
        let count = res.voxel_count() as usize;
        let fill = Vec3::new(1.0, 0.0, 0.0);
        let source = vec![fill; count];
        // One sweep of the Dirichlet model: the center cell (all six neighbors
        // present) is a fixed point, but a corner cell is pulled toward the
        // zero wall.
        let swept = diffuse_relax_sweep(
            &source,
            &source,
            res,
            params(1.0, 1, DiffusionBoundary::Fixed),
        );
        let center = swept[res.linear_index(1, 1, 1) as usize];
        let corner = swept[res.linear_index(0, 0, 0) as usize];
        assert!(approx_vec(center, fill));
        assert!(corner.x < fill.x);
        // The Free model keeps the same corner uniform.
        let swept_free = diffuse_relax_sweep(
            &source,
            &source,
            res,
            params(1.0, 1, DiffusionBoundary::Free),
        );
        let corner_free = swept_free[res.linear_index(0, 0, 0) as usize];
        assert!(approx_vec(corner_free, fill));
    }

    #[test]
    fn single_sweep_matches_hand_computation() {
        // 1-D chain of three cells, α = 0.5, Free (Neumann) walls.
        //   cell 0: 1 neighbor (cell 1)
        //   cell 1: 2 neighbors (cells 0 and 2)
        //   cell 2: 1 neighbor (cell 1)
        // source = [(0,0,0), (1,0,0), (0,0,0)].
        let res = GridResolution::new(3, 1, 1);
        let source = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
        // α = ν·dt/h² = 0.5·1 / 1 = 0.5.
        let swept = diffuse_relax_sweep(
            &source,
            &source,
            res,
            params(0.5, 1, DiffusionBoundary::Free),
        );
        // cell 0: (0 + 0.5·1) / (1 + 1·0.5) = 0.5 / 1.5 = 1/3.
        // cell 1: (1 + 0.5·0) / (1 + 2·0.5) = 1 / 2 = 0.5.
        // cell 2: symmetric to cell 0 = 1/3.
        assert!(approx(swept[0].x, 1.0 / 3.0));
        assert!(approx(swept[1].x, 0.5));
        assert!(approx(swept[2].x, 1.0 / 3.0));
    }

    #[test]
    fn jacobi_residual_decreases_monotonically() {
        let res = GridResolution::uniform(4);
        let count = res.voxel_count() as usize;
        let mut source = vec![Vec3::ZERO; count];
        source[res.linear_index(2, 1, 2) as usize] = Vec3::new(4.0, -2.0, 1.0);
        let p = params(1.0, 1, DiffusionBoundary::Free);
        let mut current: Vec<Vec3> = source.clone();
        let mut prev_residual = diffusion_residual_l2(&source, &current, res, p);
        let initial_residual = prev_residual;
        for _ in 0..40 {
            current = diffuse_relax_sweep(&source, &current, res, p);
            let residual = diffusion_residual_l2(&source, &current, res, p);
            // Non-increasing every sweep (small slack for rounding).
            assert!(residual <= prev_residual + F_EPS);
            prev_residual = residual;
        }
        // And clearly converging overall.
        assert!(prev_residual < initial_residual * 0.5);
        assert!(prev_residual >= 0.0);
    }

    #[test]
    fn solve_is_bit_deterministic() {
        let res = GridResolution::uniform(4);
        let count = res.voxel_count() as usize;
        let mut source = vec![Vec3::ZERO; count];
        for (i, v) in source.iter_mut().enumerate() {
            let f = i as f32;
            *v = Vec3::new(0.25 * f, 1.0 - f, 0.5 * f);
        }
        let p = params(0.8, 16, DiffusionBoundary::Fixed);
        let first = viscous_diffuse(&source, res, p);
        let second = viscous_diffuse(&source, res, p);
        // Identical inputs must give bit-identical outputs.
        assert_eq!(first.iterations_run, second.iterations_run);
        assert!(fields_bits_equal(&first.velocity, &second.velocity));
    }

    #[test]
    fn empty_or_short_inputs_are_handled() {
        let res = GridResolution::uniform(2);
        // Too-short source: empty result, no iterations.
        let short = vec![Vec3::ZERO; 3];
        let out = viscous_diffuse(&short, res, params(1.0, 5, DiffusionBoundary::Free));
        assert!(out.velocity.is_empty());
        assert_eq!(out.iterations_run, 0);
        // A single sweep on short input is empty too.
        let swept =
            diffuse_relax_sweep(&short, &short, res, params(1.0, 1, DiffusionBoundary::Free));
        assert!(swept.is_empty());
        // Residual of short input is zero, not a panic.
        assert!(approx(
            diffusion_residual_l2(&short, &short, res, params(1.0, 1, DiffusionBoundary::Free)),
            0.0
        ));
    }
}
