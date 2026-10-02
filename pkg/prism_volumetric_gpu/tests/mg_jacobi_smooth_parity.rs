//! Real-device parity for the single-level weighted-`Jacobi` smoother twin:
//! [`GpuMgJacobiSmooth`] must reproduce the `CPU` golden single-level
//! `jacobi_smooth` followed by `level_residual_l2` (extracted from
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure))
//! across random fields, several operator scales, sweep counts and grid
//! resolutions, both wall models, and the degenerate grids the reference guards.
//!
//! Those two golden helpers are private to the architecture crate, so this test
//! carries a faithful re-implementation of exactly their arithmetic (the same
//! neighbor order, the same diagonal convention, the same `z`-`y`-`x`
//! accumulation) and asserts the twin matches it. The re-implementation is the
//! reference; it is not imported from the golden module.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernels are portable
//! core-`WGSL`, so they need no optional device feature.
//!
//! # Parity criterion
//!
//! Each sweep is multiply-add plus one reciprocal, summed over the sweep count,
//! with the neighbors accumulated in the identical order the reference uses, so
//! `CPU` and `GPU` evaluate the same algebra. Values are asserted to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a `GPU`
//! fused multiply-add compounded across the sweeps, yet tight enough to fail a
//! wrong port (a swapped neighbor, a missing diagonal term, a dropped boundary
//! case).
//!
//! Provenance: standard `Briggs` multigrid weighted-`Jacobi` smoother and
//! residual norm; no Unreal Engine source or derived code.

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_jacobi_smooth::{
    GpuMgJacobiSmooth, GpuMgJacobiSmoothQuery, GpuMgJacobiSmoothResult,
    PRESSURE_BOUNDARY_DIRICHLET, PRESSURE_BOUNDARY_NEUMANN,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance, a decade above the single-sweep multiply-add
/// rounding so the iterated `GPU` fused multiply-add stays inside it while a
/// genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-difference floor so a near-zero reference value does not blow up the
/// relative test.
const REL_FLOOR: f32 = 1.0e-6;

/// A tiny deterministic linear-congruential generator so the "random" fields
/// are reproducible run to run without pulling in any external math or random
/// crate. The constants are the Numerical Recipes `LCG` multiplier and
/// increment.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]`, built only from integer
    /// arithmetic (no transcendental or inexact helper).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible field of `count` scalars in `[-range, range]`.
    fn field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }
}

/// Faithful re-implementation of the golden private `neighbor_sum_count`: the
/// six axis-aligned face neighbors in the exact `+x`, `−x`, `+y`, `−y`, `+z`,
/// `−z` order, dropping out-of-grid neighbors from the sum and the count.
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

/// Faithful re-implementation of the golden private `diagonal`: Dirichlet keeps
/// all six faces, Neumann uses the live-neighbor count clamped to one.
fn diagonal(boundary: u32, live: u32) -> f32 {
    if boundary == PRESSURE_BOUNDARY_NEUMANN {
        if live == 0 {
            1.0
        } else {
            live as f32
        }
    } else {
        6.0
    }
}

/// Faithful re-implementation of the golden private `level_residual_l2`,
/// accumulating the squared residual in the same `z`-`y`-`x` order.
fn level_residual_l2(
    pressure: &[f32],
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    boundary: u32,
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

/// Faithful re-implementation of the golden private `jacobi_smooth`: `sweeps`
/// weighted-`Jacobi` relaxations with ping-pong double buffering.
fn jacobi_smooth(
    pressure: &mut Vec<f32>,
    rhs: &[f32],
    res: GridResolution,
    inv_h2: f32,
    omega: f32,
    boundary: u32,
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

/// The combined `CPU` reference: smooth `pressure` in place, then measure the
/// residual of the smoothed field. This is the exact pair the twin reproduces.
fn reference(query: &GpuMgJacobiSmoothQuery) -> (Vec<f32>, f32) {
    let mut p = query.pressure.clone();
    jacobi_smooth(
        &mut p,
        &query.rhs,
        query.resolution,
        query.inv_h2,
        query.omega,
        query.boundary,
        query.sweeps,
    );
    let residual = level_residual_l2(
        &p,
        &query.rhs,
        query.resolution,
        query.inv_h2,
        query.boundary,
    );
    (p, residual)
}

/// Asserts two scalars match to within the documented tolerance.
fn close(label: &str, cpu: f32, gpu: f32) {
    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(REL_FLOOR);
    assert!(
        abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
        "{label}: cpu {cpu}, gpu {gpu} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts the `GPU` result matches the `CPU` reference field and residual.
fn assert_parity(label: &str, cpu: &(Vec<f32>, f32), gpu: &GpuMgJacobiSmoothResult) {
    assert_eq!(
        cpu.0.len(),
        gpu.pressure.len(),
        "{label}: pressure length mismatch"
    );
    for (i, (c, g)) in cpu.0.iter().zip(gpu.pressure.iter()).enumerate() {
        close(&format!("{label}: cell {i}"), *c, *g);
    }
    close(&format!("{label}: residual"), cpu.1, gpu.residual);
}

/// Builds a query from explicit parameters.
fn query(
    pressure: Vec<f32>,
    rhs: Vec<f32>,
    resolution: GridResolution,
    inv_h2: f32,
    omega: f32,
    boundary: u32,
    sweeps: u32,
) -> GpuMgJacobiSmoothQuery {
    GpuMgJacobiSmoothQuery {
        pressure,
        rhs,
        resolution,
        inv_h2,
        omega,
        boundary,
        sweeps,
    }
}

/// Runs one `CPU`-vs-`GPU` scenario and asserts parity.
fn check_scenario(
    label: &str,
    engine: &GpuMgJacobiSmooth,
    ctx: &GpuContext,
    q: &GpuMgJacobiSmoothQuery,
) {
    let cpu = reference(q);
    let gpu = engine.smooth(ctx, q);
    assert_parity(label, &cpu, &gpu);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg-jacobi-smooth parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMgJacobiSmooth::new(&ctx);

    // Default weighted-Jacobi damping (2/3), formed by integer division so no
    // transcendental or inexact literal is introduced.
    let omega = 2.0f32 / 3.0f32;

    // A spread of resolutions, operator scales, sweep counts and wall models,
    // each on its own reproducible random pressure and right-hand side.
    let cases = [
        (
            GridResolution::uniform(4),
            1.0f32,
            4u32,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
        (
            GridResolution::uniform(5),
            1.0,
            3,
            PRESSURE_BOUNDARY_NEUMANN,
        ),
        (
            GridResolution::new(6, 3, 2),
            0.25,
            2,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
        (
            GridResolution::new(2, 7, 3),
            4.0,
            2,
            PRESSURE_BOUNDARY_NEUMANN,
        ),
        (
            GridResolution::uniform(8),
            1.0,
            1,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
        (
            GridResolution::new(3, 3, 3),
            0.5,
            4,
            PRESSURE_BOUNDARY_NEUMANN,
        ),
    ];
    for (seed, (res, inv_h2, sweeps, boundary)) in cases.into_iter().enumerate() {
        let count = res.voxel_count() as usize;
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32 * 7 + 1));
        let pressure = rng.field(count, 2.0);
        let rhs = rng.field(count, 1.5);
        let label = format!(
            "random case {seed} ({}x{}x{}, inv_h2={inv_h2}, sweeps={sweeps}, boundary={boundary})",
            res.nx, res.ny, res.nz
        );
        check_scenario(
            &label,
            &engine,
            &ctx,
            &query(pressure, rhs, res, inv_h2, omega, boundary, sweeps),
        );
    }
}

#[test]
fn gpu_matches_cpu_on_both_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgJacobiSmooth::new(&ctx);
    let res = GridResolution::uniform(5);
    let count = res.voxel_count() as usize;
    let mut rng = Lcg::new(0xB0D1);
    let pressure = rng.field(count, 2.0);
    let rhs = rng.field(count, 1.0);
    let omega = 2.0f32 / 3.0f32;

    // Same field, same schedule: each wall model must match its reference, and
    // since the reference differs between them this also proves the boundary
    // code reaches the kernels.
    check_scenario(
        "dirichlet wall",
        &engine,
        &ctx,
        &query(
            pressure.clone(),
            rhs.clone(),
            res,
            1.0,
            omega,
            PRESSURE_BOUNDARY_DIRICHLET,
            3,
        ),
    );
    check_scenario(
        "neumann wall",
        &engine,
        &ctx,
        &query(pressure, rhs, res, 1.0, omega, PRESSURE_BOUNDARY_NEUMANN, 3),
    );
}

#[test]
fn gpu_matches_cpu_across_sweep_counts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgJacobiSmooth::new(&ctx);
    let res = GridResolution::new(4, 4, 3);
    let count = res.voxel_count() as usize;
    let omega = 2.0f32 / 3.0f32;

    // A zero-sweep solve is the identity on the field; the residual is still
    // measured. One, two and four sweeps exercise the ping-pong buffer parity.
    for sweeps in [0u32, 1, 2, 4] {
        let mut rng = Lcg::new(0x1234_u32.wrapping_add(sweeps));
        let pressure = rng.field(count, 2.0);
        let rhs = rng.field(count, 1.0);
        check_scenario(
            &format!("sweeps {sweeps}"),
            &engine,
            &ctx,
            &query(
                pressure,
                rhs,
                res,
                1.0,
                omega,
                PRESSURE_BOUNDARY_DIRICHLET,
                sweeps,
            ),
        );
    }
}

#[test]
fn gpu_matches_cpu_on_single_cell_neumann() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgJacobiSmooth::new(&ctx);
    // A lone cell has no live neighbors; the Neumann diagonal clamps to one so
    // the sweep is a safe no-op-style relaxation rather than a divide by zero.
    let res = GridResolution::uniform(1);
    check_scenario(
        "single-cell neumann",
        &engine,
        &ctx,
        &query(
            vec![0.75],
            vec![0.5],
            res,
            1.0,
            2.0f32 / 3.0f32,
            PRESSURE_BOUNDARY_NEUMANN,
            2,
        ),
    );
}

#[test]
fn gpu_matches_cpu_on_degenerate_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgJacobiSmooth::new(&ctx);
    // An empty grid (a zero extent) short-circuits on the host to the input
    // field unchanged and a zero residual, matching the reference guards.
    let res = GridResolution::new(0, 4, 4);
    let q = query(
        Vec::new(),
        Vec::new(),
        res,
        1.0,
        2.0f32 / 3.0f32,
        PRESSURE_BOUNDARY_DIRICHLET,
        3,
    );
    let cpu = reference(&q);
    let gpu = engine.smooth(&ctx, &q);
    assert_eq!(
        cpu.0, gpu.pressure,
        "degenerate grid should return the input field"
    );
    close("degenerate residual", cpu.1, gpu.residual);
}
