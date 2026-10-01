//! Real-device parity for the viscous-diffusion twin: [`GpuFluidDiffusion`] must
//! reproduce the `CPU` golden
//! [`viscous_diffuse`](prism_render_architecture::particle::fluid_diffusion::viscous_diffuse)
//! across random initial fields, several viscosities, iteration counts and grid
//! resolutions, both wall models, and the degenerate grids the reference guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each sweep is multiply-add plus one reciprocal, summed over the fixed
//! iteration count, with the neighbors accumulated in the identical order the
//! reference uses, so `CPU` and `GPU` evaluate the same algebra. Values are
//! asserted to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to
//! admit a `GPU` fused multiply-add compounded across the sweeps, yet tight
//! enough to fail a wrong port (a swapped neighbor, a missing diagonal term, a
//! dropped boundary case).
//!
//! Provenance: standard `Stam` stable-fluids implicit viscous diffusion; no
//! Unreal Engine source or derived code.

use prism_render_architecture::particle::Vec3;
use prism_render_architecture::particle::fluid::GridResolution;
use prism_render_architecture::particle::fluid_diffusion::{
    DiffusionBoundary, DiffusionParams, viscous_diffuse,
};
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::fluid_diffusion::GpuFluidDiffusion;

/// Absolute parity tolerance. Chosen a decade above the single-sweep
/// multiply-add rounding so the iterated `GPU` fused multiply-add stays inside
/// it while a genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// A tiny deterministic linear-congruential generator so the "random" fields
/// are reproducible run to run without pulling in an external crate. The
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
        // Numerical Recipes constants; wrapping arithmetic keeps it in range.
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]`.
    fn next_signed(&mut self, range: f32) -> f32 {
        // Map the top 24 bits into [0, 1), then into the symmetric range.
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible field of `count` velocities with components in
    /// `[-range, range]`.
    fn field(&mut self, count: usize, range: f32) -> Vec<Vec3> {
        (0..count)
            .map(|_| {
                Vec3::new(
                    self.next_signed(range),
                    self.next_signed(range),
                    self.next_signed(range),
                )
            })
            .collect()
    }
}

/// Asserts the `GPU` field matches the `CPU` golden element for element to
/// within the documented tolerance.
fn assert_parity(label: &str, cpu: &[Vec3], gpu: &[Vec3]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        for (cv, gv, axis) in [(c.x, g.x, "x"), (c.y, g.y, "y"), (c.z, g.z, "z")] {
            let abs_diff = (cv - gv).abs();
            let rel_diff = abs_diff / cv.abs().max(1.0e-6);
            assert!(
                abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
                "{label}: cell {i} axis {axis} mismatch: cpu {cv}, gpu {gv} \
                 (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

/// Builds a parameter set with unit `dt`/`cell_size` so `alpha` equals the
/// viscosity, matching the reference test convention.
fn params(viscosity: f32, iterations: u32, boundary: DiffusionBoundary) -> DiffusionParams {
    DiffusionParams {
        viscosity,
        dt: 1.0,
        cell_size: 1.0,
        iterations,
        boundary,
    }
}

/// Runs one `CPU`-vs-`GPU` scenario and asserts parity.
fn check_scenario(
    label: &str,
    engine: &GpuFluidDiffusion,
    ctx: &GpuContext,
    source: &[Vec3],
    res: GridResolution,
    p: DiffusionParams,
) {
    let cpu = viscous_diffuse(source, res, p);
    let gpu = engine.diffuse(ctx, source, res, p);
    assert_eq!(
        cpu.iterations_run, gpu.iterations_run,
        "{label}: iteration count mismatch"
    );
    assert_parity(label, &cpu.velocity, &gpu.velocity);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fluid-diffusion parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);

    // A spread of resolutions, viscosities, iteration counts and wall models,
    // each on its own reproducible random field.
    let cases = [
        (GridResolution::uniform(4), 0.5, 4u32, DiffusionBoundary::Fixed),
        (GridResolution::uniform(5), 1.0, 10, DiffusionBoundary::Free),
        (GridResolution::new(6, 3, 2), 0.25, 8, DiffusionBoundary::Fixed),
        (GridResolution::new(2, 7, 3), 2.0, 6, DiffusionBoundary::Free),
        (GridResolution::uniform(8), 0.75, 16, DiffusionBoundary::Fixed),
    ];
    for (seed, (res, viscosity, iterations, boundary)) in cases.into_iter().enumerate() {
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32));
        let source = rng.field(res.voxel_count() as usize, 3.0);
        let label = format!(
            "random case {seed} ({}x{}x{}, nu={viscosity}, it={iterations}, {boundary:?})",
            res.nx, res.ny, res.nz
        );
        check_scenario(&label, &engine, &ctx, &source, res, params(viscosity, iterations, boundary));
    }
}

#[test]
fn gpu_matches_cpu_on_both_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    let res = GridResolution::uniform(5);
    let mut rng = Lcg::new(0xB0D1);
    let source = rng.field(res.voxel_count() as usize, 2.0);

    // Same field, same schedule, the two wall models must each match the
    // reference (and, since the reference differs between them, this also
    // proves the boundary tag reaches the kernel).
    check_scenario(
        "fixed wall",
        &engine,
        &ctx,
        &source,
        res,
        params(1.2, 12, DiffusionBoundary::Fixed),
    );
    check_scenario(
        "free wall",
        &engine,
        &ctx,
        &source,
        res,
        params(1.2, 12, DiffusionBoundary::Free),
    );
}

#[test]
fn gpu_matches_cpu_one_dimensional_chain() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    // The hand-checked 1-D chain from the reference unit tests: three cells, a
    // single unit impulse in the middle, Neumann walls.
    let res = GridResolution::new(3, 1, 1);
    let source = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
    check_scenario(
        "1-D chain",
        &engine,
        &ctx,
        &source,
        res,
        params(0.5, 1, DiffusionBoundary::Free),
    );
}

#[test]
fn gpu_matches_cpu_zero_viscosity_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    let res = GridResolution::uniform(4);
    let mut rng = Lcg::new(0x1234_5678);
    let source = rng.field(res.voxel_count() as usize, 5.0);
    // nu = 0 -> alpha = 0 is an exact fixed point; the GPU must leave the field
    // unchanged just as the reference does.
    let p = params(0.0, 8, DiffusionBoundary::Free);
    let gpu = engine.diffuse(&ctx, &source, res, p);
    assert_eq!(gpu.iterations_run, 8);
    assert_parity("zero viscosity", &source, &gpu.velocity);
}

#[test]
fn gpu_matches_cpu_zero_iterations_copy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    let res = GridResolution::uniform(3);
    let source = vec![Vec3::new(1.0, 2.0, 3.0); res.voxel_count() as usize];
    check_scenario(
        "zero iterations",
        &engine,
        &ctx,
        &source,
        res,
        params(0.7, 0, DiffusionBoundary::Fixed),
    );
}

#[test]
fn gpu_matches_cpu_single_voxel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    // A lone voxel has no neighbors: Fixed drains it toward the zero wall while
    // Free leaves it fixed. Both must match the reference.
    let res = GridResolution::uniform(1);
    let source = vec![Vec3::new(2.0, -3.0, 1.5)];
    check_scenario(
        "single voxel fixed",
        &engine,
        &ctx,
        &source,
        res,
        params(1.0, 5, DiffusionBoundary::Fixed),
    );
    check_scenario(
        "single voxel free",
        &engine,
        &ctx,
        &source,
        res,
        params(1.0, 5, DiffusionBoundary::Free),
    );
}

#[test]
fn gpu_handles_degenerate_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    let p = params(1.0, 5, DiffusionBoundary::Free);

    // A zero-extent grid: empty result, zero iterations, no dispatch.
    let empty_res = GridResolution::new(0, 4, 4);
    let empty = engine.diffuse(&ctx, &[], empty_res, p);
    assert!(empty.velocity.is_empty(), "zero-extent grid yields no field");
    assert_eq!(empty.iterations_run, 0);

    // A too-short source: the guard returns empty rather than reading past it.
    let res = GridResolution::uniform(3);
    let short = vec![Vec3::ZERO; 3];
    let out = engine.diffuse(&ctx, &short, res, p);
    assert!(out.velocity.is_empty(), "too-short source yields no field");
    assert_eq!(out.iterations_run, 0);
}

#[test]
fn gpu_matches_cpu_degenerate_cell_size() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidDiffusion::new(&ctx);
    let res = GridResolution::uniform(4);
    let mut rng = Lcg::new(0xFEED_FACE);
    let source = rng.field(res.voxel_count() as usize, 2.0);
    // A zero cell size makes alpha collapse to 0 (no division by zero); the
    // field must stay put, matching the reference guard.
    let p = DiffusionParams {
        viscosity: 1.5,
        dt: 1.0,
        cell_size: 0.0,
        iterations: 6,
        boundary: DiffusionBoundary::Fixed,
    };
    check_scenario("degenerate cell size", &engine, &ctx, &source, res, p);
}
