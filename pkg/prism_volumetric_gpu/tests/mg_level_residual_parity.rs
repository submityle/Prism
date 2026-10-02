//! Real-device parity for the single-level `L2` residual twin:
//! [`GpuMgLevelResidual`] must reproduce the `CPU` golden `level_residual_l2`
//! (extracted from
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure))
//! across random fields, several operator scales and grid resolutions (including
//! odd and even axes), both wall models, and the degenerate grids the reference
//! guards.
//!
//! That golden helper and its two private dependencies are private to the
//! architecture crate, so this test carries a faithful re-implementation of
//! exactly their arithmetic (the same neighbor order, the same diagonal
//! convention, the same `z`-`y`-`x` accumulation) and asserts the twin matches
//! it. The re-implementation is the reference; it is not imported from the
//! golden module.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each per-cell residual is multiply-add, with the neighbors accumulated in the
//! identical order the reference uses, so `CPU` and `GPU` evaluate the same
//! algebra. Values are asserted to within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — loose enough to admit a `GPU` fused multiply-add, yet
//! tight enough to fail a wrong port (a swapped neighbor, a missing diagonal
//! term, a dropped boundary case).
//!
//! Provenance: standard `Briggs` multigrid level residual norm; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_level_residual::{
    GpuMgLevelResidual, GpuMgLevelResidualQuery, GpuMgLevelResidualResult,
    PRESSURE_BOUNDARY_DIRICHLET, PRESSURE_BOUNDARY_NEUMANN,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance, a decade above the single-cell multiply-add
/// rounding so the `GPU` fused multiply-add stays inside it while a genuinely
/// wrong port falls outside.
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

    /// A reproducible `f32` in `[-range, range]` biased away from zero by
    /// `offset`, built only from integer arithmetic (no transcendental or
    /// inexact helper) so the fields stay well clear of the stencil's near-zero
    /// cancellation.
    fn next_signed(&mut self, range: f32, offset: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let signed = (unit * 2.0 - 1.0) * range;
        if signed < 0.0 {
            signed - offset
        } else {
            signed + offset
        }
    }

    /// A reproducible field of `count` scalars in `[-range, range]`, each pushed
    /// at least `offset` away from zero.
    fn field(&mut self, count: usize, range: f32, offset: f32) -> Vec<f32> {
        (0..count)
            .map(|_| self.next_signed(range, offset))
            .collect()
    }
}

// MIRROR of multigrid_pressure::neighbor_sum_count (private, exact transcription)
/// The six axis-aligned face neighbors in the exact `+x`, `−x`, `+y`, `−y`,
/// `+z`, `−z` order, dropping out-of-grid neighbors from the sum and the count.
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

// MIRROR of multigrid_pressure::diagonal (private, exact transcription)
/// Dirichlet keeps all six faces; Neumann uses the live-neighbor count clamped
/// to one for a lone cell.
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

// MIRROR of multigrid_pressure::level_residual_l2 (private, exact transcription)
/// Accumulates the squared residual in the same `z`-`y`-`x` order, divides by
/// the voxel count and takes one `sqrt`.
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

/// The `CPU` reference: the exact residual the twin reproduces.
fn reference(query: &GpuMgLevelResidualQuery) -> f32 {
    level_residual_l2(
        &query.pressure,
        &query.rhs,
        query.resolution,
        query.inv_h2,
        query.boundary,
    )
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

/// Builds a query from explicit parameters.
fn query(
    pressure: Vec<f32>,
    rhs: Vec<f32>,
    resolution: GridResolution,
    inv_h2: f32,
    boundary: u32,
) -> GpuMgLevelResidualQuery {
    GpuMgLevelResidualQuery {
        pressure,
        rhs,
        resolution,
        inv_h2,
        boundary,
    }
}

/// Runs one `CPU`-vs-`GPU` scenario and asserts parity.
fn check_scenario(
    label: &str,
    engine: &GpuMgLevelResidual,
    ctx: &GpuContext,
    q: &GpuMgLevelResidualQuery,
) {
    let cpu = reference(q);
    let gpu = engine.residual(ctx, q);
    close(&format!("{label}: residual"), cpu, gpu.residual);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg-level-residual parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMgLevelResidual::new(&ctx);

    // A spread of resolutions (cubic, odd and even axes, strongly anisotropic),
    // several operator scales, and both wall models; each on its own
    // reproducible random pressure and right-hand side kept clear of zero.
    let cases = [
        (
            GridResolution::uniform(4),
            1.0f32,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
        (GridResolution::uniform(5), 1.0, PRESSURE_BOUNDARY_NEUMANN),
        (
            GridResolution::new(6, 3, 2),
            0.25,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
        (GridResolution::new(2, 7, 3), 4.0, PRESSURE_BOUNDARY_NEUMANN),
        (GridResolution::uniform(8), 1.0, PRESSURE_BOUNDARY_DIRICHLET),
        (GridResolution::new(3, 3, 3), 0.5, PRESSURE_BOUNDARY_NEUMANN),
        (
            GridResolution::new(7, 1, 5),
            2.0,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
    ];
    for (seed, (res, inv_h2, boundary)) in cases.into_iter().enumerate() {
        let count = res.voxel_count() as usize;
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32 * 7 + 1));
        let pressure = rng.field(count, 2.0, 0.25);
        let rhs = rng.field(count, 1.5, 0.25);
        let label = format!(
            "random case {seed} ({}x{}x{}, inv_h2={inv_h2}, boundary={boundary})",
            res.nx, res.ny, res.nz
        );
        check_scenario(
            &label,
            &engine,
            &ctx,
            &query(pressure, rhs, res, inv_h2, boundary),
        );
    }
}

#[test]
fn gpu_matches_cpu_on_both_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgLevelResidual::new(&ctx);
    let res = GridResolution::uniform(5);
    let count = res.voxel_count() as usize;
    let mut rng = Lcg::new(0xB0D1);
    let pressure = rng.field(count, 2.0, 0.25);
    let rhs = rng.field(count, 1.0, 0.25);

    // Same field: each wall model must match its reference, and since the
    // reference differs between them this also proves the boundary code reaches
    // the kernel.
    check_scenario(
        "dirichlet wall",
        &engine,
        &ctx,
        &query(
            pressure.clone(),
            rhs.clone(),
            res,
            1.0,
            PRESSURE_BOUNDARY_DIRICHLET,
        ),
    );
    check_scenario(
        "neumann wall",
        &engine,
        &ctx,
        &query(pressure, rhs, res, 1.0, PRESSURE_BOUNDARY_NEUMANN),
    );
}

#[test]
fn gpu_matches_cpu_across_operator_scales() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgLevelResidual::new(&ctx);
    let res = GridResolution::new(4, 4, 3);
    let count = res.voxel_count() as usize;

    // The per-level operator scale 1/h^2 ranges over several coarse levels.
    for (i, inv_h2) in [1.0f32, 0.25, 0.0625, 4.0].into_iter().enumerate() {
        let mut rng = Lcg::new(0x1234_u32.wrapping_add(i as u32));
        let pressure = rng.field(count, 2.0, 0.25);
        let rhs = rng.field(count, 1.0, 0.25);
        check_scenario(
            &format!("inv_h2 {inv_h2}"),
            &engine,
            &ctx,
            &query(pressure, rhs, res, inv_h2, PRESSURE_BOUNDARY_DIRICHLET),
        );
    }
}

#[test]
fn gpu_matches_cpu_on_single_cell_neumann() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgLevelResidual::new(&ctx);
    // A lone cell has no live neighbors; the Neumann diagonal clamps to one so
    // the operator is a safe residual rather than a divide by zero.
    let res = GridResolution::uniform(1);
    check_scenario(
        "single-cell neumann",
        &engine,
        &ctx,
        &query(vec![0.75], vec![0.5], res, 1.0, PRESSURE_BOUNDARY_NEUMANN),
    );
}

#[test]
fn gpu_matches_cpu_on_degenerate_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgLevelResidual::new(&ctx);
    // An empty grid (a zero extent) short-circuits on the host to a zero
    // residual, matching the reference guards.
    let res = GridResolution::new(0, 4, 4);
    let q = query(
        Vec::new(),
        Vec::new(),
        res,
        1.0,
        PRESSURE_BOUNDARY_DIRICHLET,
    );
    let cpu = reference(&q);
    let gpu: GpuMgLevelResidualResult = engine.residual(&ctx, &q);
    close("degenerate residual", cpu, gpu.residual);
}
