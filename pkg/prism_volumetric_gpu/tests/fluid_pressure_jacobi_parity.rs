//! Real-device parity for the pressure-solve twin: [`GpuFluidPressureJacobi`]
//! must reproduce the `CPU` golden
//! [`jacobi_pressure_solve`](prism_render_architecture::particle::fluid::jacobi_pressure_solve)
//! across random divergence fields, several sweep counts and grid resolutions,
//! the degenerate grids the reference guards, plus the gradient-projection
//! cluster
//! ([`central_gradient`](prism_render_architecture::particle::fluid::central_gradient)
//! and
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernels are portable
//! core-`WGSL`, so they need no optional device feature.
//!
//! # Parity criterion
//!
//! Each sweep is multiply-add plus one division, summed over the fixed
//! iteration count, with the neighbors accumulated in the identical order the
//! reference uses, so `CPU` and `GPU` evaluate the same algebra. Values are
//! asserted to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to
//! admit a `GPU` fused multiply-add compounded across the sweeps, yet tight
//! enough to fail a wrong port (a swapped neighbor, a dropped boundary case, a
//! missing divisor). Iteration counts are compared exactly.
//!
//! The reference's adaptive residual stop is pinned to the full iteration
//! budget by giving the plan a non-positive `residual_tolerance`, so the
//! reference runs exactly as many sweeps as the twin's fixed count.
//!
//! Provenance: twins the `CPU` golden pressure solve and gradient projection in
//! `prism_render_architecture::particle::fluid`; no Unreal Engine source or
//! derived code.

use prism_render_architecture::particle::fluid::{
    central_gradient, jacobi_pressure_solve, subtract_pressure_gradient, GridResolution,
    NeighborScalars, ProjectionMethod, ProjectionPlan,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid_pressure_jacobi::{
    GpuFluidPressureJacobi, GpuGradientProjectionQuery, GpuPressureSolveQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance. Chosen a decade above the single-sweep
/// multiply-add rounding so the iterated `GPU` fused multiply-add stays inside
/// it while a genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-difference floor so a near-zero reference value does not blow up
/// the ratio.
const REL_FLOOR: f32 = 1.0e-6;

/// A tiny deterministic linear-congruential generator so the "random" fields
/// are reproducible run to run without needing an external math library. The
/// constants are the Numerical Recipes `LCG` multiplier and increment.
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

    /// A reproducible `f32` in `[-range, range]`, formed with pure integer and
    /// multiply arithmetic (no transcendental method).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible divergence field of `count` scalars in `[-range, range]`.
    fn scalar_field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }
}

/// Returns `true` when two scalars agree to within the documented tolerance.
fn approx(a: f32, b: f32) -> bool {
    let abs_diff = (a - b).abs();
    let rel_diff = abs_diff / a.abs().max(b.abs()).max(REL_FLOOR);
    abs_diff <= ABS_EPS || rel_diff <= REL_EPS
}

/// Asserts two scalar fields match element for element to within tolerance.
fn assert_field_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx(*c, *g),
            "{label}: cell {i} mismatch: cpu {c}, gpu {g}"
        );
    }
}

/// Builds a plan that runs exactly `iterations` sweeps by pinning the adaptive
/// residual stop below zero so the reference never stops early.
fn full_run_plan(iterations: u32) -> ProjectionPlan {
    ProjectionPlan {
        method: ProjectionMethod::JacobiApprox,
        max_iterations: iterations,
        residual_tolerance: -1.0,
    }
}

/// Runs one `CPU`-vs-`GPU` pressure-solve scenario and asserts parity.
fn check_solve(
    label: &str,
    engine: &GpuFluidPressureJacobi,
    ctx: &GpuContext,
    divergence: &[f32],
    res: GridResolution,
    iterations: u32,
) {
    let cpu = jacobi_pressure_solve(divergence, res, full_run_plan(iterations));
    let gpu = engine.solve(
        ctx,
        &GpuPressureSolveQuery {
            divergence: divergence.to_vec(),
            resolution: res,
            iterations,
        },
    );
    assert_eq!(
        cpu.iterations_run, gpu.iterations_run,
        "{label}: iteration count mismatch"
    );
    assert_field_parity(label, &cpu.pressure, &gpu.pressure);
    assert!(
        approx(cpu.residual, gpu.residual),
        "{label}: residual mismatch: cpu {}, gpu {}",
        cpu.residual,
        gpu.residual
    );
}

#[test]
fn gpu_matches_cpu_across_grids_and_iterations() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);

    let grids = [
        GridResolution::new(3, 3, 3),
        GridResolution::new(4, 4, 4),
        GridResolution::new(5, 6, 7),
        GridResolution::new(8, 8, 8),
    ];
    let iteration_counts = [1u32, 2, 5, 10, 20];

    let mut rng = Lcg::new(0xA5F0_1234);
    for res in grids {
        let divergence = rng.scalar_field(res.voxel_count() as usize, 1.5);
        for iters in iteration_counts {
            check_solve(
                &format!("grid {}x{}x{} iters {iters}", res.nx, res.ny, res.nz),
                &engine,
                &ctx,
                &divergence,
                res,
                iters,
            );
        }
    }
}

#[test]
fn gpu_matches_cpu_single_voxel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);
    // A lone voxel has no neighbors: each sweep sets p = -div/6; the GPU must
    // track the reference exactly.
    let res = GridResolution::uniform(1);
    let divergence = vec![0.75f32];
    check_solve("single voxel", &engine, &ctx, &divergence, res, 7);
}

#[test]
fn gpu_matches_cpu_one_dimensional_chain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);
    // A 1-D chain with a single impulse in the middle exercises the +x/-x walls.
    let res = GridResolution::new(5, 1, 1);
    let divergence = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    check_solve("1-D chain", &engine, &ctx, &divergence, res, 12);
}

#[test]
fn gpu_matches_cpu_zero_iterations_seed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);
    let res = GridResolution::uniform(4);
    let mut rng = Lcg::new(0x0BAD_F00D);
    let divergence = rng.scalar_field(res.voxel_count() as usize, 2.0);
    // Zero sweeps: the field stays the all-zero seed and the residual is the
    // reference's pre-loop residual, with iterations_run == 0.
    check_solve("zero iterations", &engine, &ctx, &divergence, res, 0);
}

#[test]
fn gpu_handles_degenerate_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);

    // A zero-extent grid: empty result, zero residual, zero iterations, no
    // dispatch.
    let empty_res = GridResolution::new(0, 4, 4);
    let empty = engine.solve(
        &ctx,
        &GpuPressureSolveQuery {
            divergence: Vec::new(),
            resolution: empty_res,
            iterations: 5,
        },
    );
    assert!(
        empty.pressure.is_empty(),
        "zero-extent grid yields no field"
    );
    assert!(approx(empty.residual, 0.0), "zero-extent residual is zero");
    assert_eq!(empty.iterations_run, 0);

    // A too-short divergence: the guard returns empty rather than reading past
    // it.
    let res = GridResolution::uniform(3);
    let short = GpuPressureSolveQuery {
        divergence: vec![0.0; 3],
        resolution: res,
        iterations: 5,
    };
    let out = engine.solve(&ctx, &short);
    assert!(
        out.pressure.is_empty(),
        "too-short divergence yields no field"
    );
    assert_eq!(out.iterations_run, 0);
}

#[test]
fn gpu_matches_cpu_gradient_projection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);

    let mut rng = Lcg::new(0x1357_9BDF);
    let mut queries = Vec::new();
    for _ in 0..256 {
        let neighbors = NeighborScalars {
            x_plus: rng.next_signed(3.0),
            x_minus: rng.next_signed(3.0),
            y_plus: rng.next_signed(3.0),
            y_minus: rng.next_signed(3.0),
            z_plus: rng.next_signed(3.0),
            z_minus: rng.next_signed(3.0),
        };
        let velocity = Vec3::new(
            rng.next_signed(2.0),
            rng.next_signed(2.0),
            rng.next_signed(2.0),
        );
        // inv_2h biased away from zero so the gradient stays well-scaled.
        let inv_2h = 0.5 + rng.next_signed(0.25);
        queries.push(GpuGradientProjectionQuery {
            neighbors,
            velocity,
            inv_2h,
        });
    }

    let gpu = engine.project(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "projection length mismatch");
    for (i, (q, g)) in queries.iter().zip(gpu.iter()).enumerate() {
        let grad = central_gradient(q.neighbors, q.inv_2h);
        let cpu = subtract_pressure_gradient(q.velocity, grad);
        for (cv, gv, axis) in [(cpu.x, g.x, "x"), (cpu.y, g.y, "y"), (cpu.z, g.z, "z")] {
            assert!(
                approx(cv, gv),
                "projection {i} axis {axis} mismatch: cpu {cv}, gpu {gv}"
            );
        }
    }
}

#[test]
fn gpu_projection_empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidPressureJacobi::new(&ctx);
    let out = engine.project(&ctx, &[]);
    assert!(out.is_empty(), "empty batch yields no projected velocities");
}
