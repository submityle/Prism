//! Geometric multigrid `V`-cycle pressure-projection solver — the *exact* tier
//! of step 3 (pressure projection) in the grid-fluid pipeline (design §10, and
//! §37 open question 3: "approximate `Jacobi` first, exact multigrid later").
//!
//! [`super::fluid`] ships the approximate tier: [`jacobi_pressure_solve`] relaxes
//! the pressure `Poisson` equation `∇²p = div` with a fixed `Jacobi` budget and
//! leaves a non-zero residual. Flat `Jacobi` kills the *high*-frequency error of
//! the residual quickly but crawls on the *low*-frequency (smooth) error, so
//! driving the residual to tolerance on a cinematic `128³` grid costs hundreds
//! of sweeps. This module fills the gap §37 flags with the standard remedy from
//! `Briggs`' multigrid tutorial: a geometric multigrid `V`-cycle.
//!
//! # The idea
//!
//! Smooth error on a fine grid looks *oscillatory* on a coarser grid, where a
//! `Jacobi` smoother can attack it cheaply. One `V`-cycle therefore:
//!
//! 1. **pre-smooths** the current pressure with a few weighted-`Jacobi` sweeps,
//! 2. forms the residual `r = b − A·p` and **restricts** it to the next coarser
//!    grid with a full-weighting operator (fine → coarse),
//! 3. recurses on the coarse residual equation `A·e = r` for the error `e`,
//! 4. **prolongs** the coarse error back up with trilinear interpolation and
//!    adds it as a **coarse-grid correction** `p ← p + e` (coarse → fine), then
//! 5. **post-smooths** to remove the high-frequency error the interpolation
//!    reintroduces.
//!
//! Steps 1-5 are one *cycle*; the recursion in step 3 visits the coarse level
//! `γ` times (the cycle index of a `μ`-cycle). This module defaults to `γ = 2`,
//! a `W`-cycle, because a pointwise `Jacobi` smoother is weak in three
//! dimensions and the rediscretized coarse operators are not the exact
//! `Galerkin` product; a single coarse visit (`γ = 1`, a plain `V`-cycle) then
//! stops being a contraction past two levels. See [`DEFAULT_COARSE_CYCLES`].
//!
//! The recursion bottoms out on a grid small enough to solve almost exactly,
//! for which this module reuses the sibling [`jacobi_pressure_solve`] directly.
//! Because the exact solution is a fixed point of every stage, the cycle
//! converges to the same field the flat solver would, but in a handful of
//! cycles instead of hundreds of sweeps.
//!
//! # Discretization and reuse
//!
//! The fine-grid operator matches the sibling exactly: the 7-point `Laplacian`
//! `L(p) = (Σ₆ neighbors) − 6·p` on a unit-spaced grid, so a solved field here
//! is bit-compatible with [`pressure_residual_l2`] and [`jacobi_pressure_solve`]
//! and can be handed to [`subtract_pressure_gradient`] unchanged. Each coarser
//! level rediscretizes the same stencil with mesh spacing `h = 2^level`, i.e.
//! the operator carries a per-level scale `1 / h²`; [`project_velocity_field`]
//! wires the whole thing into the velocity write-back contract. The grid,
//! indexing, [`GridResolution`], [`ProjectionPlan`], [`ProjectionMethod`], and
//! [`PressureSolveResult`] types are reused from [`super::fluid`]; nothing is
//! redefined.
//!
//! # Boundaries
//!
//! Two wall models are supported, matching the stencil conventions the sibling
//! modules already assume:
//!
//! * [`PressureBoundary::Dirichlet`] — pressure outside the grid is pinned to
//!   zero, so the diagonal keeps all six faces. This is exactly the convention
//!   behind [`jacobi_pressure_solve`] and [`pressure_residual_l2`], giving an
//!   invertible system with a unique solution.
//! * [`PressureBoundary::Neumann`] — a zero-gradient solid wall: a missing
//!   neighbor mirrors the center cell, so the diagonal drops to the live
//!   neighbor count. This is the physical pressure wall for incompressible
//!   flow; the operator then has a constant null space, which the solver pins
//!   by subtracting the mean each cycle.
//!
//! # Determinism
//!
//! Determinism matches the sibling particle modules (design §29): the only
//! floating-point primitive beyond ordinary arithmetic is `sqrt` (reached only
//! through the residual norm). There are no transcendental calls — no `sin`,
//! `cos`, `exp`, `ln`, or `pow` — no hashing, and no `RNG`. Every sweep,
//! restriction, and prolongation is multiply-add, so the `CPU` reference is
//! bit-reproducible against a future `GPU` compute kernel.

use alloc::vec;
use alloc::vec::Vec;

use super::fluid::{
    jacobi_pressure_solve, subtract_pressure_gradient, GridResolution, PressureSolveResult,
    ProjectionMethod, ProjectionPlan,
};
use super::Vec3;
// `pressure_residual_l2` is reused only by the sibling cross-check unit test; a
// `#[cfg(test)]` import keeps it at module scope (so `use super::*` picks it up)
// without tripping the unused-import lint in the non-test build.
#[cfg(test)]
use super::fluid::pressure_residual_l2;

/// The number of axis-aligned face neighbors a voxel has in a 3-D grid; the
/// full `Dirichlet` diagonal of the 7-point `Laplacian`.
const FACE_NEIGHBOR_COUNT: f32 = 6.0;

/// Default weighted-`Jacobi` damping factor `ω = 2/3`.
///
/// Undamped `Jacobi` (`ω = 1`) is a poor multigrid smoother because it barely
/// touches the highest-frequency error mode; `ω = 2/3` minimizes the smoothing
/// factor of the `Laplacian` and is the textbook choice. Written as a division
/// of small integers so no transcendental or inexact literal is introduced.
const DEFAULT_SMOOTHING_OMEGA: f32 = 2.0 / 3.0;

/// Default number of recursive coarse-level cycles per visit — the cycle index
/// `γ` of a `μ`-cycle: `γ = 1` is a plain `V`-cycle, `γ = 2` is a `W`-cycle.
///
/// A plain `V`-cycle is the textbook ideal, but a pointwise weighted-`Jacobi`
/// smoother is a weak smoother in three dimensions (its smoothing factor only
/// reaches about `5/7`), and the rediscretized coarse operators are not the
/// exact `Galerkin` product. With a single coarse visit the coarse-grid
/// correction under-resolves the smoothest error mode and overshoots it, so the
/// recursive cycle stops being a contraction past two levels. Visiting the
/// coarse level twice (a `W`-cycle) solves the coarse residual equation
/// accurately enough to restore a clean contraction — the standard remedy in
/// `Briggs`' multigrid tutorial for exactly this smoother/operator pairing.
const DEFAULT_COARSE_CYCLES: u32 = 2;

/// How the pressure stencil treats a cell just outside the grid (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PressureBoundary {
    /// Pressure outside the grid is zero (homogeneous `Dirichlet`). The
    /// diagonal keeps all six faces, matching the convention behind
    /// [`jacobi_pressure_solve`]. The system is invertible.
    Dirichlet,
    /// Zero-gradient solid wall (homogeneous `Neumann`): a missing neighbor
    /// mirrors the center cell, so the diagonal drops to the live neighbor
    /// count. Physical for incompressible flow; the operator has a constant
    /// null space that the solver removes by mean subtraction.
    Neumann,
}

/// Tunable inputs of one multigrid solve (design §10, §37).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultigridConfig {
    /// Weighted-`Jacobi` sweeps applied before restriction on each level.
    pub pre_smooth: u32,
    /// Weighted-`Jacobi` sweeps applied after the coarse-grid correction.
    pub post_smooth: u32,
    /// Hard cap on the number of `V`-cycles.
    pub max_cycles: u32,
    /// `L2` residual at or below which the solve stops early.
    pub residual_tolerance: f32,
    /// Weighted-`Jacobi` damping factor used by the smoother.
    pub smoothing_omega: f32,
    /// Grids with every axis at or below this size are solved directly on the
    /// coarsest level instead of recursing further.
    pub coarsest_axis: u32,
    /// Recursive coarse-level cycles per visit — the cycle index `γ`. `γ = 1`
    /// gives a plain `V`-cycle; `γ = 2` gives the `W`-cycle this module defaults
    /// to (see [`DEFAULT_COARSE_CYCLES`]). Clamped to at least one.
    pub coarse_cycles: u32,
    /// Wall model applied outside the grid on every level.
    pub boundary: PressureBoundary,
}

impl MultigridConfig {
    /// A cinematic default: two pre- and two post-smoothing sweeps, up to the
    /// [`ProjectionMethod::Multigrid`] cycle budget, with a `Dirichlet` wall.
    #[must_use]
    pub fn cinematic(residual_tolerance: f32) -> Self {
        Self {
            // Two sweeps each side with a `W`-cycle coarse schedule — a robust
            // `W(2, 2)` cycle (see [`DEFAULT_COARSE_CYCLES`]).
            pre_smooth: 2,
            post_smooth: 2,
            max_cycles: ProjectionMethod::Multigrid.default_iterations(),
            residual_tolerance,
            smoothing_omega: DEFAULT_SMOOTHING_OMEGA,
            // A 2³ (or smaller) block is cheap to solve almost exactly.
            coarsest_axis: 2,
            coarse_cycles: DEFAULT_COARSE_CYCLES,
            boundary: PressureBoundary::Dirichlet,
        }
    }

    /// Builds a plan from a [`ProjectionPlan`], inheriting its tolerance and
    /// cycle cap so the multigrid tier is a drop-in for the sibling solver.
    #[must_use]
    pub fn from_plan(plan: ProjectionPlan, boundary: PressureBoundary) -> Self {
        Self {
            pre_smooth: 2,
            post_smooth: 2,
            max_cycles: plan.max_iterations,
            residual_tolerance: plan.residual_tolerance,
            smoothing_omega: DEFAULT_SMOOTHING_OMEGA,
            coarsest_axis: 2,
            coarse_cycles: DEFAULT_COARSE_CYCLES,
            boundary,
        }
    }
}

/// Coarsens a single axis: halve it (rounding up) unless it is already a single
/// cell, which cannot be coarsened further.
#[must_use]
fn coarse_axis(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        // Round up so an odd axis still produces a strictly smaller grid.
        n.div_ceil(2)
    }
}

/// The next coarser resolution for a full 3-D grid.
#[must_use]
fn coarsen(res: GridResolution) -> GridResolution {
    GridResolution::new(
        coarse_axis(res.nx),
        coarse_axis(res.ny),
        coarse_axis(res.nz),
    )
}

/// Whether `res` is small enough (every axis at or below `coarsest_axis`, or no
/// axis can be coarsened) to be solved directly instead of recursed.
#[must_use]
fn is_coarsest(res: GridResolution, coarsest_axis: u32) -> bool {
    let within = res.nx <= coarsest_axis && res.ny <= coarsest_axis && res.nz <= coarsest_axis;
    let stuck = coarsen(res) == res;
    within || stuck
}

/// A single axis of the inter-grid transfer: up to two coarse indices and their
/// interpolation weights for one fine index (or the reverse gather).
#[derive(Clone, Copy, Debug)]
struct AxisStencil {
    idx: [u32; 2],
    weight: [f32; 2],
    len: usize,
}

/// The cell-centered linear-interpolation weights that map a fine index `i` to
/// its coarse contributors on one axis.
///
/// Fine cell `i` sits inside coarse parent `i / 2`. Its center leans toward the
/// adjacent coarse cell (the left child leans to the parent minus one, the
/// right child to the parent plus one), giving the standard cell-centered
/// `(3/4, 1/4)` split. When the far coarse neighbor falls outside the grid the
/// weight is folded back onto the parent (a mirrored, zero-gradient
/// extrapolation), so a constant coarse field always interpolates to the same
/// constant — the property the coarse-grid correction relies on.
#[must_use]
fn axis_contributors(i: u32, coarse_n: u32) -> AxisStencil {
    if coarse_n == 0 {
        return AxisStencil {
            idx: [0, 0],
            weight: [0.0, 0.0],
            len: 0,
        };
    }
    let parent = i / 2;
    // Clamp the parent into range for the degenerate "axis not coarsened" case.
    let parent = if parent >= coarse_n {
        coarse_n - 1
    } else {
        parent
    };
    // Three-quarters stays on the parent, one-quarter leans to a neighbor.
    let own_weight = 0.75;
    let far_weight = 0.25;
    // Left child (even) leans to parent-1, right child (odd) to parent+1.
    let far_is_lower = i.is_multiple_of(2);
    let far_in_range = if far_is_lower {
        parent > 0
    } else {
        parent + 1 < coarse_n
    };
    if far_in_range {
        let far = if far_is_lower { parent - 1 } else { parent + 1 };
        AxisStencil {
            idx: [parent, far],
            weight: [own_weight, far_weight],
            len: 2,
        }
    } else {
        // Mirror: the far quarter folds back onto the parent.
        AxisStencil {
            idx: [parent, 0],
            weight: [own_weight + far_weight, 0.0],
            len: 1,
        }
    }
}

/// The number of axes that are genuinely halved between `fine` and `coarse`.
///
/// Each coarsened axis contributes a factor of two to the full-weighting
/// normalization, so a constant residual restricts to the same constant.
#[must_use]
fn coarsened_axis_count(fine: GridResolution, coarse: GridResolution) -> u32 {
    let mut k = 0u32;
    if coarse.nx < fine.nx {
        k += 1;
    }
    if coarse.ny < fine.ny {
        k += 1;
    }
    if coarse.nz < fine.nz {
        k += 1;
    }
    k
}

/// `2^k` as an `f32`, computed with an integer-doubling loop so no `pow` is
/// used. `k` is at most three here.
#[must_use]
fn pow2_f32(k: u32) -> f32 {
    let mut value = 1.0f32;
    let mut remaining = k;
    while remaining > 0 {
        value *= 2.0;
        remaining -= 1;
    }
    value
}

/// Trilinear prolongation (coarse → fine): interpolates a coarse field onto the
/// finer grid using the tensor product of the per-axis [`axis_contributors`].
///
/// A constant coarse field maps to the same constant on every fine cell because
/// the per-axis weights sum to one. Multiply-add only.
#[must_use]
fn prolong(coarse: &[f32], coarse_res: GridResolution, fine_res: GridResolution) -> Vec<f32> {
    let fine_count = fine_res.voxel_count() as usize;
    let coarse_count = coarse_res.voxel_count() as usize;
    let mut fine = vec![0.0f32; fine_count];
    if coarse.len() < coarse_count {
        return fine;
    }
    for z in 0..fine_res.nz {
        let sz = axis_contributors(z, coarse_res.nz);
        for y in 0..fine_res.ny {
            let sy = axis_contributors(y, coarse_res.ny);
            for x in 0..fine_res.nx {
                let sx = axis_contributors(x, coarse_res.nx);
                let mut acc = 0.0f32;
                for iz in 0..sz.len {
                    let wz = sz.weight[iz];
                    let cz = sz.idx[iz];
                    for iy in 0..sy.len {
                        let wy = sy.weight[iy];
                        let cy = sy.idx[iy];
                        for ix in 0..sx.len {
                            let cidx = coarse_res.linear_index(sx.idx[ix], cy, cz) as usize;
                            acc += sx.weight[ix] * wy * wz * coarse[cidx];
                        }
                    }
                }
                fine[fine_res.linear_index(x, y, z) as usize] = acc;
            }
        }
    }
    fine
}

/// Full-weighting restriction (fine → coarse): the scaled transpose of
/// [`prolong`], `R = (1 / 2^k)·Pᵀ` for `k` coarsened axes.
///
/// Building it as the exact transpose of the prolongation guarantees the
/// adjoint relationship `⟨R·f, c⟩ = (1 / 2^k)·⟨f, P·c⟩` and preserves constant
/// fields. Multiply-add only.
#[must_use]
fn restrict(fine: &[f32], fine_res: GridResolution, coarse_res: GridResolution) -> Vec<f32> {
    let coarse_count = coarse_res.voxel_count() as usize;
    let fine_count = fine_res.voxel_count() as usize;
    let mut coarse = vec![0.0f32; coarse_count];
    if fine.len() < fine_count {
        return coarse;
    }
    let k = coarsened_axis_count(fine_res, coarse_res);
    let scale = 1.0 / pow2_f32(k);
    for z in 0..fine_res.nz {
        let sz = axis_contributors(z, coarse_res.nz);
        for y in 0..fine_res.ny {
            let sy = axis_contributors(y, coarse_res.ny);
            for x in 0..fine_res.nx {
                let sx = axis_contributors(x, coarse_res.nx);
                let value = fine[fine_res.linear_index(x, y, z) as usize];
                for iz in 0..sz.len {
                    let wz = sz.weight[iz];
                    let cz = sz.idx[iz];
                    for iy in 0..sy.len {
                        let wy = sy.weight[iy];
                        let cy = sy.idx[iy];
                        for ix in 0..sx.len {
                            let cidx = coarse_res.linear_index(sx.idx[ix], cy, cz) as usize;
                            coarse[cidx] += scale * sx.weight[ix] * wy * wz * value;
                        }
                    }
                }
            }
        }
    }
    coarse
}

/// Sum of the in-grid face-neighbor pressures around `(x, y, z)` together with
/// the count of neighbors that actually exist.
///
/// This mirrors the sibling module's private `neighbor_sum` plus a live-neighbor
/// count (which the private helper does not expose): a neighbor outside the grid
/// is dropped from the sum, so under [`PressureBoundary::Dirichlet`] the dropped
/// term is the zero wall pressure, while under [`PressureBoundary::Neumann`] the
/// returned count lets the diagonal shrink to a zero-gradient wall.
#[must_use]
fn neighbor_sum_count(field: &[f32], res: GridResolution, x: u32, y: u32, z: u32) -> (f32, u32) {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    if x + 1 < res.nx {
        sum += field[res.linear_index(x + 1, y, z) as usize];
        count += 1;
    }
    if x > 0 {
        sum += field[res.linear_index(x - 1, y, z) as usize];
        count += 1;
    }
    if y + 1 < res.ny {
        sum += field[res.linear_index(x, y + 1, z) as usize];
        count += 1;
    }
    if y > 0 {
        sum += field[res.linear_index(x, y - 1, z) as usize];
        count += 1;
    }
    if z + 1 < res.nz {
        sum += field[res.linear_index(x, y, z + 1) as usize];
        count += 1;
    }
    if z > 0 {
        sum += field[res.linear_index(x, y, z - 1) as usize];
        count += 1;
    }
    (sum, count)
}

/// The stencil diagonal for a cell under a given wall model.
///
/// [`PressureBoundary::Dirichlet`] always uses the full six faces; the missing
/// neighbors are the zero wall. [`PressureBoundary::Neumann`] uses only the
/// live neighbors so the wall carries zero gradient, guarding against a lone
/// cell with no neighbors by clamping the diagonal to one.
#[must_use]
fn diagonal(boundary: PressureBoundary, live: u32) -> f32 {
    match boundary {
        PressureBoundary::Dirichlet => FACE_NEIGHBOR_COUNT,
        // Widening a 0..=6 count to f32 is exact; a lone cell keeps a unit
        // diagonal so the sweep is a safe no-op rather than a divide by zero.
        PressureBoundary::Neumann => {
            if live == 0 {
                1.0
            } else {
                live as f32
            }
        }
    }
}

/// `L2` residual `‖b − A·p‖` of a level with operator scale `inv_h2`.
///
/// The level operator is `A·p = inv_h2·(Σ live neighbors − diagonal·p)`. With
/// `inv_h2 = 1` and [`PressureBoundary::Dirichlet`] this is identical to the
/// sibling [`pressure_residual_l2`]; a unit test cross-checks that equivalence.
/// One `sqrt`, otherwise multiply-add.
#[must_use]
fn level_residual_l2(
    pressure: &[f32],
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    boundary: PressureBoundary,
) -> f32 {
    let count = res.voxel_count() as usize;
    if count == 0 || pressure.len() < count || rhs.len() < count {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let (nsum, live) = neighbor_sum_count(pressure, res, x, y, z);
                let diag = diagonal(boundary, live);
                let laplacian = inv_h2 * (nsum - diag * pressure[idx]);
                let r = rhs[idx] - laplacian;
                sum_sq += r * r;
            }
        }
    }
    (sum_sq / count as f32).sqrt()
}

/// Runs `sweeps` weighted-`Jacobi` relaxations of `A·p = rhs` in place.
///
/// Solving `inv_h2·(Σ neighbors − diagonal·p) = rhs` for the center cell and
/// damping by `omega` gives
/// `p ← (1 − ω)·p + ω·(Σ neighbors − rhs / inv_h2) / diagonal`. With `ω = 1`,
/// `inv_h2 = 1`, and the full six-face diagonal this is exactly the sibling
/// [`jacobi_pressure_solve`] sweep, so the smoother and the sibling solver share
/// one relaxation. Multiply-add plus one reciprocal per cell.
fn jacobi_smooth(
    pressure: &mut Vec<f32>,
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    omega: f32,
    boundary: PressureBoundary,
    sweeps: u32,
) {
    let count = res.voxel_count() as usize;
    if count == 0 || rhs.len() < count || pressure.len() < count {
        return;
    }
    let inv_scale = 1.0 / inv_h2;
    let mut done = 0u32;
    while done < sweeps {
        let mut next = vec![0.0f32; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    let (nsum, live) = neighbor_sum_count(pressure, res, x, y, z);
                    let diag = diagonal(boundary, live);
                    let relaxed = (nsum - rhs[idx] * inv_scale) / diag;
                    next[idx] = (1.0 - omega) * pressure[idx] + omega * relaxed;
                }
            }
        }
        *pressure = next;
        done += 1;
    }
}

/// Subtracts the mean of a field in place, pinning the constant null space of a
/// pure-`Neumann` operator so the correction stays bounded.
fn remove_mean(field: &mut [f32]) {
    if field.is_empty() {
        return;
    }
    let mut sum = 0.0f32;
    for &v in field.iter() {
        sum += v;
    }
    let mean = sum / field.len() as f32;
    for v in field.iter_mut() {
        *v -= mean;
    }
}

/// Solves the coarsest level nearly exactly.
///
/// For a [`PressureBoundary::Dirichlet`] wall the system is invertible and this
/// reuses the sibling [`jacobi_pressure_solve`] directly: that solver assumes a
/// unit-spaced (`inv_h2 = 1`) operator starting from a zero field, so the level
/// right-hand side is rescaled by `1 / inv_h2` first and the returned field is
/// the coarse error. For a [`PressureBoundary::Neumann`] wall the operator is
/// singular, so a long run of the weighted smoother plus mean removal is used
/// instead.
#[must_use]
fn solve_coarsest(
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    config: MultigridConfig,
) -> Vec<f32> {
    let count = res.voxel_count() as usize;
    match config.boundary {
        PressureBoundary::Dirichlet => {
            // Rescale the level RHS into the unit-spacing convention the sibling
            // solver assumes: A·p = inv_h2·L(p) = rhs ⇔ L(p) = rhs / inv_h2.
            let inv_scale = 1.0 / inv_h2;
            let mut scaled = vec![0.0f32; count];
            for i in 0..count {
                scaled[i] = rhs[i] * inv_scale;
            }
            // A tight tolerance and generous cap make the coarse solve "exact".
            let plan = ProjectionPlan {
                method: ProjectionMethod::Multigrid,
                // The coarsest grid is tiny, so many sweeps are still cheap.
                max_iterations: 200,
                residual_tolerance: config.residual_tolerance * 0.25,
            };
            let result: PressureSolveResult = jacobi_pressure_solve(&scaled, res, plan);
            result.pressure
        }
        PressureBoundary::Neumann => {
            let mut pressure = vec![0.0f32; count];
            // The singular Neumann system is only solved up to a constant, which
            // is pinned by mean removal between smoothing bursts.
            let bursts = 40u32;
            let mut done = 0u32;
            while done < bursts {
                jacobi_smooth(
                    &mut pressure,
                    rhs,
                    res,
                    inv_h2,
                    config.smoothing_omega,
                    config.boundary,
                    5,
                );
                remove_mean(&mut pressure);
                done += 1;
            }
            pressure
        }
    }
}

/// Performs one recursive multigrid cycle on level `res`, updating `pressure` in
/// place toward the solution of `A·pressure = rhs`.
///
/// The coarse level is visited `config.coarse_cycles` times (the cycle index
/// `γ`): `γ = 1` is a plain `V`-cycle and `γ = 2` is the default `W`-cycle. The
/// stage order — pre-smooth, restrict residual, recurse, prolong-and-correct,
/// post-smooth — is identical either way.
fn v_cycle(
    pressure: &mut Vec<f32>,
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    config: MultigridConfig,
) {
    if is_coarsest(res, config.coarsest_axis) {
        *pressure = solve_coarsest(rhs, res, inv_h2, config);
        if matches!(config.boundary, PressureBoundary::Neumann) {
            remove_mean(pressure);
        }
        return;
    }

    // 1. Pre-smooth to damp the high-frequency error on this level.
    jacobi_smooth(
        pressure,
        rhs,
        res,
        inv_h2,
        config.smoothing_omega,
        config.boundary,
        config.pre_smooth,
    );

    // 2. Residual r = b − A·p on this level.
    let count = res.voxel_count() as usize;
    let mut residual = vec![0.0f32; count];
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let (nsum, live) = neighbor_sum_count(pressure, res, x, y, z);
                let diag = diagonal(config.boundary, live);
                let laplacian = inv_h2 * (nsum - diag * pressure[idx]);
                residual[idx] = rhs[idx] - laplacian;
            }
        }
    }

    // 3. Restrict the residual and recurse for the coarse error.
    let coarse_res = coarsen(res);
    let coarse_rhs = restrict(&residual, res, coarse_res);
    // Each coarsening doubles the spacing, so the coarse operator scale is a
    // quarter of this level's (1 / (2h)² = inv_h2 / 4).
    let coarse_inv_h2 = inv_h2 * 0.25;
    let mut coarse_error = vec![0.0f32; coarse_res.voxel_count() as usize];
    // Visit the coarse level `γ` times (a `μ`-cycle). Repeating the recursion
    // (a `W`-cycle at `γ = 2`) solves the coarse residual equation accurately
    // enough that the correction stays a contraction with the weak pointwise
    // smoother; see [`DEFAULT_COARSE_CYCLES`]. Zero would skip the correction,
    // so the index is clamped to at least one.
    let gamma = if config.coarse_cycles == 0 {
        1
    } else {
        config.coarse_cycles
    };
    let mut visit = 0u32;
    while visit < gamma {
        v_cycle(
            &mut coarse_error,
            &coarse_rhs,
            coarse_res,
            coarse_inv_h2,
            config,
        );
        visit += 1;
    }

    // 4. Prolong the coarse error and apply the coarse-grid correction.
    let fine_error = prolong(&coarse_error, coarse_res, res);
    for i in 0..count {
        pressure[i] += fine_error[i];
    }
    if matches!(config.boundary, PressureBoundary::Neumann) {
        remove_mean(pressure);
    }

    // 5. Post-smooth to remove error reintroduced by the interpolation.
    jacobi_smooth(
        pressure,
        rhs,
        res,
        inv_h2,
        config.smoothing_omega,
        config.boundary,
        config.post_smooth,
    );
}

/// Solves the pressure `Poisson` equation `∇²p = divergence` with a geometric
/// multigrid `V`-cycle (design §10, §37).
///
/// The fine-grid operator and boundary convention match [`jacobi_pressure_solve`]
/// for [`PressureBoundary::Dirichlet`], so the returned [`PressureSolveResult`]
/// is a drop-in replacement that reaches the same residual in far fewer cycles.
/// The solve repeats `V`-cycles until the `L2` residual falls to
/// `config.residual_tolerance` or `config.max_cycles` is reached; the reported
/// `iterations_run` counts `V`-cycles. Starts from a zero field.
#[must_use]
pub fn multigrid_pressure_solve(
    divergence: &[f32],
    res: GridResolution,
    config: MultigridConfig,
) -> PressureSolveResult {
    let count = res.voxel_count() as usize;
    if count == 0 || divergence.len() < count {
        return PressureSolveResult {
            pressure: Vec::new(),
            residual: 0.0,
            iterations_run: 0,
        };
    }
    let mut pressure = vec![0.0f32; count];
    // The fine grid is unit-spaced, so its operator scale is one.
    let fine_inv_h2 = 1.0f32;
    let mut residual = level_residual_l2(&pressure, divergence, res, fine_inv_h2, config.boundary);
    let mut cycles = 0u32;
    while cycles < config.max_cycles && residual > config.residual_tolerance {
        v_cycle(&mut pressure, divergence, res, fine_inv_h2, config);
        cycles += 1;
        residual = level_residual_l2(&pressure, divergence, res, fine_inv_h2, config.boundary);
    }
    PressureSolveResult {
        pressure,
        residual,
        iterations_run: cycles,
    }
}

/// Forward-difference divergence of a velocity field, the negative adjoint of
/// the wall-aware gradient in [`project_velocity_field`].
///
/// Composing the two one-sided operators reproduces the compact `Neumann`
/// `Laplacian` `Σ live neighbors − (live count)·p` exactly, so the discrete
/// `Hodge` decomposition is tight: projecting with the solved pressure drives
/// this divergence to the solver residual (up to the unavoidable net-flux
/// constant). An out-of-grid velocity is treated as zero (a closed wall), which
/// supplies the zero-flux high face of that `Neumann` stencil. Multiply-add
/// only.
#[must_use]
pub fn divergence_forward(velocity: &[Vec3], res: GridResolution) -> Vec<f32> {
    let count = res.voxel_count() as usize;
    let mut div = vec![0.0f32; count];
    if velocity.len() < count {
        return div;
    }
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let here = velocity[idx];
                let vx = if x + 1 < res.nx {
                    velocity[res.linear_index(x + 1, y, z) as usize].x
                } else {
                    0.0
                };
                let vy = if y + 1 < res.ny {
                    velocity[res.linear_index(x, y + 1, z) as usize].y
                } else {
                    0.0
                };
                let vz = if z + 1 < res.nz {
                    velocity[res.linear_index(x, y, z + 1) as usize].z
                } else {
                    0.0
                };
                div[idx] = (vx - here.x) + (vy - here.y) + (vz - here.z);
            }
        }
    }
    div
}

/// Wall-aware backward-difference pressure gradient at one cell, the negative
/// adjoint of [`divergence_forward`].
///
/// A solid wall carries zero normal pressure gradient, so the backward
/// difference is dropped on the low face of each axis (where the neighbor would
/// be outside the grid). Composed with the forward-difference
/// [`divergence_forward`] — whose own out-of-grid velocity is zero on the high
/// face — this reproduces the compact `Neumann` `Laplacian`
/// `Σ live neighbors − (live count)·p` exactly on every face, so projecting
/// with a `Neumann` pressure solve drives the forward divergence to the solver
/// residual. Multiply-add only.
#[must_use]
fn gradient_backward(pressure: &[f32], res: GridResolution, x: u32, y: u32, z: u32) -> Vec3 {
    let idx = res.linear_index(x, y, z) as usize;
    let here = pressure[idx];
    let gx = if x > 0 {
        here - pressure[res.linear_index(x - 1, y, z) as usize]
    } else {
        0.0
    };
    let gy = if y > 0 {
        here - pressure[res.linear_index(x, y - 1, z) as usize]
    } else {
        0.0
    };
    let gz = if z > 0 {
        here - pressure[res.linear_index(x, y, z - 1) as usize]
    } else {
        0.0
    };
    Vec3::new(gx, gy, gz)
}

/// Projects a velocity field onto its (near) divergence-free part (design §10).
///
/// This wires the multigrid solve into the velocity write-back contract: it
/// forms the forward-difference [`divergence_forward`], solves the pressure
/// `Poisson` equation with [`multigrid_pressure_solve`], and subtracts the
/// wall-aware backward-difference gradient through the sibling
/// [`subtract_pressure_gradient`].
///
/// Solid walls make this a pure `Neumann` problem (the physical incompressible
/// wall condition), so the solve is forced to [`PressureBoundary::Neumann`]
/// regardless of the caller's wall model. The forward divergence and the
/// wall-aware [`gradient_backward`] compose to exactly the `Neumann`
/// `Laplacian` the solver inverts, so the residual divergence of the result is
/// the solver residual plus one unavoidable constant: the mean divergence (the
/// net wall flux) is removed before the solve because no internal pressure can
/// cancel it, and it reappears as a uniform offset in the projected field.
#[must_use]
pub fn project_velocity_field(
    velocity: &[Vec3],
    res: GridResolution,
    config: MultigridConfig,
) -> Vec<Vec3> {
    let count = res.voxel_count() as usize;
    if velocity.len() < count {
        return Vec::new();
    }
    let mut divergence = divergence_forward(velocity, res);
    // A pure `Neumann` system is only solvable for a mean-free source; the
    // removed constant is the net wall flux, which no internal pressure can
    // cancel, so it survives as a uniform divergence in the projected field.
    remove_mean(&mut divergence);
    let wall_config = MultigridConfig {
        boundary: PressureBoundary::Neumann,
        ..config
    };
    let solve = multigrid_pressure_solve(&divergence, res, wall_config);
    let pressure = solve.pressure;
    let mut projected = vec![Vec3::ZERO; count];
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let grad = gradient_backward(&pressure, res, x, y, z);
                projected[idx] = subtract_pressure_gradient(velocity[idx], grad);
            }
        }
    }
    projected
}

#[cfg(test)]
mod tests {
    use super::*;

    const F_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= F_EPS
    }

    /// Applies the unit-spacing `Dirichlet` Laplacian `Σ₆ neighbors − 6·p` to a
    /// field, matching the sibling convention. Used to manufacture a right-hand
    /// side with a known analytic solution.
    fn apply_dirichlet_laplacian(field: &[f32], res: GridResolution) -> Vec<f32> {
        let count = res.voxel_count() as usize;
        let mut out = vec![0.0f32; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    let (nsum, _live) = neighbor_sum_count(field, res, x, y, z);
                    out[idx] = nsum - 6.0 * field[idx];
                }
            }
        }
        out
    }

    /// A deterministic, transcendental-free "analytic" pressure field: a smooth
    /// low-order polynomial in the voxel coordinates.
    fn polynomial_field(res: GridResolution) -> Vec<f32> {
        let count = res.voxel_count() as usize;
        let mut field = vec![0.0f32; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    let fx = x as f32;
                    let fy = y as f32;
                    let fz = z as f32;
                    // A mixed quadratic/cubic polynomial: smooth, non-separable,
                    // and exactly representable, so it is a fair analytic target.
                    field[idx] = fx * fx - 2.0 * fy + fx * fz + 0.5 * fz * fz;
                }
            }
        }
        field
    }

    fn bits_equal(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn vec3_bits_equal(a: &[Vec3], b: &[Vec3]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(x, y)| {
                x.x.to_bits() == y.x.to_bits()
                    && x.y.to_bits() == y.y.to_bits()
                    && x.z.to_bits() == y.z.to_bits()
            })
    }

    #[test]
    fn level_residual_matches_sibling_on_dirichlet() {
        // The level operator at inv_h2 = 1 with a Dirichlet wall must reproduce
        // the sibling pressure_residual_l2 bit-for-bit in behavior.
        let res = GridResolution::new(6, 5, 4);
        let field = polynomial_field(res);
        let rhs = apply_dirichlet_laplacian(&field, res);
        let mine = level_residual_l2(&field, &rhs, res, 1.0, PressureBoundary::Dirichlet);
        let theirs = pressure_residual_l2(&field, &rhs, res);
        // field solves the system exactly, so both residuals are ~0.
        assert!(mine <= F_EPS);
        assert!(theirs <= F_EPS);
        // And on a non-solution field they agree.
        let mut perturbed = field.clone();
        perturbed[res.linear_index(2, 2, 2) as usize] += 3.0;
        let mine2 = level_residual_l2(&perturbed, &rhs, res, 1.0, PressureBoundary::Dirichlet);
        let theirs2 = pressure_residual_l2(&perturbed, &rhs, res);
        assert!(approx(mine2, theirs2));
    }

    #[test]
    fn prolong_preserves_constant_field() {
        let coarse_res = GridResolution::uniform(4);
        let fine_res = GridResolution::uniform(8);
        let coarse = vec![2.5f32; coarse_res.voxel_count() as usize];
        let fine = prolong(&coarse, coarse_res, fine_res);
        for &v in &fine {
            assert!(approx(v, 2.5));
        }
    }

    #[test]
    fn restrict_preserves_constant_field() {
        let fine_res = GridResolution::uniform(8);
        let coarse_res = coarsen(fine_res);
        let fine = vec![-1.75f32; fine_res.voxel_count() as usize];
        let coarse = restrict(&fine, fine_res, coarse_res);
        for &v in &coarse {
            assert!(approx(v, -1.75));
        }
    }

    #[test]
    fn restriction_is_scaled_adjoint_of_prolongation() {
        // ⟨R·f, c⟩ must equal (1/2^k)·⟨f, P·c⟩ for arbitrary f (fine) and
        // c (coarse). Built as an exact transpose, this holds to fp rounding.
        let fine_res = GridResolution::new(8, 8, 8);
        let coarse_res = coarsen(fine_res);
        let fcount = fine_res.voxel_count() as usize;
        let ccount = coarse_res.voxel_count() as usize;
        // Deterministic, varied test vectors.
        let mut f = vec![0.0f32; fcount];
        for (i, v) in f.iter_mut().enumerate() {
            *v = ((i % 7) as f32) - 3.0;
        }
        let mut c = vec![0.0f32; ccount];
        for (i, v) in c.iter_mut().enumerate() {
            *v = 1.0 + ((i % 5) as f32) * 0.5;
        }
        let rf = restrict(&f, fine_res, coarse_res);
        let pc = prolong(&c, coarse_res, fine_res);
        let lhs: f32 = rf.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        let k = coarsened_axis_count(fine_res, coarse_res);
        let scale = 1.0 / pow2_f32(k);
        let rhs: f32 = scale * f.iter().zip(pc.iter()).map(|(a, b)| a * b).sum::<f32>();
        assert!((lhs - rhs).abs() <= 1.0e-3);
    }

    #[test]
    fn v_cycle_converges_below_tolerance() {
        let res = GridResolution::uniform(16);
        let field = polynomial_field(res);
        let rhs = apply_dirichlet_laplacian(&field, res);
        let config = MultigridConfig {
            residual_tolerance: 1.0e-4,
            max_cycles: 30,
            ..MultigridConfig::cinematic(1.0e-4)
        };
        let result = multigrid_pressure_solve(&rhs, res, config);
        assert!(
            result.residual <= config.residual_tolerance,
            "residual {} did not reach tolerance",
            result.residual
        );
    }

    #[test]
    fn v_cycle_matches_known_analytic_solution() {
        // Manufacture rhs = L(p_known); the Dirichlet system is invertible, so
        // the solver must recover p_known (up to tolerance).
        let res = GridResolution::uniform(16);
        let known = polynomial_field(res);
        let rhs = apply_dirichlet_laplacian(&known, res);
        let config = MultigridConfig {
            residual_tolerance: 1.0e-5,
            max_cycles: 40,
            ..MultigridConfig::cinematic(1.0e-5)
        };
        let result = multigrid_pressure_solve(&rhs, res, config);
        let mut max_err = 0.0f32;
        for (got, want) in result.pressure.iter().zip(known.iter()) {
            let e = (got - want).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err <= 1.0e-2, "max abs error {max_err} too large");
    }

    #[test]
    fn multigrid_beats_jacobi_at_equal_smoother_cost() {
        let res = GridResolution::uniform(16);
        let field = polynomial_field(res);
        let rhs = apply_dirichlet_laplacian(&field, res);

        // Multigrid: a few V-cycles. Count its total fine-grid smoother sweeps
        // so the flat Jacobi comparison gets at least as many relaxations.
        let config = MultigridConfig {
            residual_tolerance: 1.0e-6,
            max_cycles: 4,
            ..MultigridConfig::cinematic(1.0e-6)
        };
        let mg = multigrid_pressure_solve(&rhs, res, config);
        let fine_sweeps_per_cycle = config.pre_smooth + config.post_smooth;
        // Generously credit flat Jacobi with the full per-cycle budget as if it
        // ran only on the fine grid, plus headroom.
        let jacobi_iters = mg.iterations_run * fine_sweeps_per_cycle + 16;

        let plan = ProjectionPlan {
            method: ProjectionMethod::JacobiApprox,
            max_iterations: jacobi_iters,
            residual_tolerance: 1.0e-9,
        };
        let jac = jacobi_pressure_solve(&rhs, res, plan);

        assert!(
            mg.residual < jac.residual * 0.5,
            "multigrid residual {} should beat Jacobi residual {}",
            mg.residual,
            jac.residual
        );
    }

    #[test]
    fn projection_drives_divergence_toward_zero() {
        let res = GridResolution::uniform(16);
        // A deterministic, strongly divergent velocity field.
        let count = res.voxel_count() as usize;
        let mut velocity = vec![Vec3::ZERO; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    let fx = x as f32;
                    let fy = y as f32;
                    let fz = z as f32;
                    velocity[idx] = Vec3::new(fx - fy, 0.5 * fy + fz, fx * 0.25 - fz);
                }
            }
        }
        let before = divergence_forward(&velocity, res);
        let before_norm = level_residual_l2(
            &vec![0.0f32; count],
            &before,
            res,
            1.0,
            PressureBoundary::Dirichlet,
        );

        let config = MultigridConfig {
            residual_tolerance: 1.0e-5,
            max_cycles: 40,
            ..MultigridConfig::cinematic(1.0e-5)
        };
        let projected = project_velocity_field(&velocity, res, config);
        let after = divergence_forward(&projected, res);

        // Mean-square divergence of the projected field.
        let mut after_sq = 0.0f32;
        for &d in &after {
            after_sq += d * d;
        }
        let after_rms = (after_sq / count as f32).sqrt();

        assert!(before_norm > 0.1, "test field should start divergent");
        assert!(
            after_rms < before_norm * 0.05,
            "divergence {after_rms} not driven down from {before_norm}"
        );
    }

    #[test]
    fn neumann_constant_pressure_has_zero_residual() {
        // A constant pressure is in the Neumann null space: its Laplacian is
        // zero everywhere, so the residual against a zero RHS vanishes.
        let res = GridResolution::uniform(8);
        let count = res.voxel_count() as usize;
        let field = vec![3.0f32; count];
        let rhs = vec![0.0f32; count];
        let residual = level_residual_l2(&field, &rhs, res, 1.0, PressureBoundary::Neumann);
        assert!(residual <= F_EPS);
        // The Dirichlet wall instead pulls the constant toward the zero wall, so
        // its residual is non-zero — confirming the two boundaries differ.
        let dir = level_residual_l2(&field, &rhs, res, 1.0, PressureBoundary::Dirichlet);
        assert!(dir > F_EPS);
    }

    #[test]
    fn neumann_solve_converges_on_compatible_rhs() {
        // Manufacture a compatible RHS (zero mean) from the Neumann Laplacian of
        // a known field, then confirm the solver reaches tolerance.
        let res = GridResolution::uniform(8);
        let known = polynomial_field(res);
        let count = res.voxel_count() as usize;
        let mut rhs = vec![0.0f32; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    let (nsum, live) = neighbor_sum_count(&known, res, x, y, z);
                    rhs[idx] = nsum - (live as f32) * known[idx];
                }
            }
        }
        // Project the RHS onto the compatible (zero-mean) subspace.
        remove_mean(&mut rhs);
        let config = MultigridConfig {
            residual_tolerance: 1.0e-3,
            max_cycles: 60,
            boundary: PressureBoundary::Neumann,
            ..MultigridConfig::cinematic(1.0e-3)
        };
        let result = multigrid_pressure_solve(&rhs, res, config);
        assert!(
            result.residual <= config.residual_tolerance,
            "neumann residual {} did not converge",
            result.residual
        );
    }

    #[test]
    fn solve_is_bit_deterministic() {
        let res = GridResolution::uniform(16);
        let field = polynomial_field(res);
        let rhs = apply_dirichlet_laplacian(&field, res);
        let config = MultigridConfig::cinematic(1.0e-5);
        let a = multigrid_pressure_solve(&rhs, res, config);
        let b = multigrid_pressure_solve(&rhs, res, config);
        assert!(bits_equal(&a.pressure, &b.pressure));
        assert_eq!(a.iterations_run, b.iterations_run);
        assert_eq!(a.residual.to_bits(), b.residual.to_bits());
    }

    #[test]
    fn projection_is_bit_deterministic() {
        let res = GridResolution::uniform(8);
        let count = res.voxel_count() as usize;
        let mut velocity = vec![Vec3::ZERO; count];
        for (i, slot) in velocity.iter_mut().enumerate() {
            let f = i as f32;
            *slot = Vec3::new(f * 0.5, 1.0 - f, f * 0.25);
        }
        let config = MultigridConfig::cinematic(1.0e-5);
        let a = project_velocity_field(&velocity, res, config);
        let b = project_velocity_field(&velocity, res, config);
        assert!(vec3_bits_equal(&a, &b));
    }

    #[test]
    fn empty_and_degenerate_grids_are_safe() {
        let empty = GridResolution::new(0, 0, 0);
        let result = multigrid_pressure_solve(&[], empty, MultigridConfig::cinematic(1.0e-5));
        assert!(result.pressure.is_empty());
        assert_eq!(result.iterations_run, 0);
        // A single cell cannot be coarsened; it must still solve safely.
        let one = GridResolution::uniform(1);
        let r1 = multigrid_pressure_solve(&[0.0], one, MultigridConfig::cinematic(1.0e-5));
        assert_eq!(r1.pressure.len(), 1);
    }

    #[test]
    fn coarsen_rounds_up_and_bottoms_out() {
        assert_eq!(coarse_axis(16), 8);
        assert_eq!(coarse_axis(7), 4);
        assert_eq!(coarse_axis(2), 1);
        assert_eq!(coarse_axis(1), 1);
        assert!(is_coarsest(GridResolution::uniform(1), 2));
        assert!(is_coarsest(GridResolution::uniform(2), 2));
        assert!(!is_coarsest(GridResolution::uniform(8), 2));
    }
}
