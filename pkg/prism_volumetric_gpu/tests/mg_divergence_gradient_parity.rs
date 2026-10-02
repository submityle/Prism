//! Real-device parity for the single-level multigrid difference operators:
//! [`GpuMgDivergenceGradient`] must reproduce the `CPU` golden forward-
//! difference divergence
//! [`divergence_forward`](prism_render_architecture::particle::multigrid_pressure::divergence_forward)
//! and the wall-aware backward-difference gradient projection that composes the
//! private golden `gradient_backward` with the public
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient),
//! across random fields, several grid resolutions and the degenerate grids the
//! twin guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernels are portable
//! core-`WGSL`, so they need no optional device feature.
//!
//! # Parity criterion
//!
//! Each operator is a per-voxel gather of subtractions, so `CPU` and `GPU`
//! evaluate the same algebra in the same order. Values are asserted to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a `GPU`
//! fused multiply-add, yet tight enough to fail a wrong port (a swapped
//! neighbor, a dropped boundary case, a flipped difference direction).
//!
//! # Mirrored private golden
//!
//! The golden `gradient_backward` is a private function in
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! so this test transcribes it exactly as the local
//! [`golden_gradient_backward`] (see its provenance comment) and composes it
//! with the public `subtract_pressure_gradient` to form the projection
//! reference.
//!
//! Provenance: twins the `CPU` golden `divergence_forward` and the composition
//! of the private `gradient_backward` with the public
//! `subtract_pressure_gradient` in
//! `prism_render_architecture::particle::multigrid_pressure` and
//! `prism_render_architecture::particle::fluid`; no Unreal Engine source or
//! derived code.

use prism_render_architecture::particle::fluid::{subtract_pressure_gradient, GridResolution};
use prism_render_architecture::particle::multigrid_pressure::divergence_forward;
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mg_divergence_gradient::{
    GpuDivergenceQuery, GpuMgDivergenceGradient, GpuProjectionQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance. Chosen a decade above the single-gather
/// subtraction rounding so a legal `GPU` fused multiply-add stays inside it
/// while a genuinely wrong port falls outside.
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

    /// A reproducible scalar field of `count` entries in `[-range, range]`.
    fn scalar_field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }

    /// A reproducible velocity field of `count` vectors each in `[-range, range]`.
    fn vector_field(&mut self, count: usize, range: f32) -> Vec<Vec3> {
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

/// MIRROR of `multigrid_pressure.rs::gradient_backward` (private, exact transcription).
///
/// Wall-aware backward-difference pressure gradient at one cell: the low-face
/// neighbor outside the grid drops that term (zero normal gradient at a wall).
/// Transcribed verbatim from the golden private function because it is not
/// `pub`; the projection reference is this composed with the public
/// `subtract_pressure_gradient`.
fn golden_gradient_backward(pressure: &[f32], res: GridResolution, x: u32, y: u32, z: u32) -> Vec3 {
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

/// The `CPU` projection reference: the mirrored private `gradient_backward`
/// composed with the public `subtract_pressure_gradient`, evaluated per voxel
/// in the twin's row-major order.
fn golden_projection(velocity: &[Vec3], pressure: &[f32], res: GridResolution) -> Vec<Vec3> {
    let count = res.voxel_count() as usize;
    let mut out = Vec::with_capacity(count);
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let grad = golden_gradient_backward(pressure, res, x, y, z);
                out.push(subtract_pressure_gradient(velocity[idx], grad));
            }
        }
    }
    out
}

/// Returns `true` when two scalars agree to within the documented tolerance.
fn approx(a: f32, b: f32) -> bool {
    let abs_diff = (a - b).abs();
    let rel_diff = abs_diff / a.abs().max(b.abs()).max(REL_FLOOR);
    abs_diff <= ABS_EPS || rel_diff <= REL_EPS
}

/// Asserts two scalar fields match element for element to within tolerance.
fn assert_scalar_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx(*c, *g),
            "{label}: cell {i} mismatch: cpu {c}, gpu {g}"
        );
    }
}

/// Asserts two vector fields match component for component to within tolerance.
fn assert_vector_parity(label: &str, cpu: &[Vec3], gpu: &[Vec3]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx(c.x, g.x) && approx(c.y, g.y) && approx(c.z, g.z),
            "{label}: cell {i} mismatch: cpu ({}, {}, {}), gpu ({}, {}, {})",
            c.x,
            c.y,
            c.z,
            g.x,
            g.y,
            g.z
        );
    }
}

/// Runs one `CPU`-vs-`GPU` divergence scenario and asserts parity.
fn check_divergence(
    label: &str,
    engine: &GpuMgDivergenceGradient,
    ctx: &GpuContext,
    velocity: &[Vec3],
    res: GridResolution,
) {
    let cpu = divergence_forward(velocity, res);
    let gpu = engine.divergence(
        ctx,
        &GpuDivergenceQuery {
            velocity: velocity.to_vec(),
            resolution: res,
        },
    );
    assert_scalar_parity(label, &cpu, &gpu);
}

/// Runs one `CPU`-vs-`GPU` projection scenario and asserts parity.
fn check_projection(
    label: &str,
    engine: &GpuMgDivergenceGradient,
    ctx: &GpuContext,
    velocity: &[Vec3],
    pressure: &[f32],
    res: GridResolution,
) {
    let cpu = golden_projection(velocity, pressure, res);
    let gpu = engine.project(
        ctx,
        &GpuProjectionQuery {
            velocity: velocity.to_vec(),
            pressure: pressure.to_vec(),
            resolution: res,
        },
    );
    assert_vector_parity(label, &cpu, &gpu);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn divergence_matches_golden_across_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg_divergence_gradient divergence parity: no wgpu adapter");
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let mut rng = Lcg::new(0x1234_5678);

    let grids = [
        GridResolution::new(3, 3, 3),
        GridResolution::new(4, 4, 4),
        GridResolution::new(5, 6, 7),
        GridResolution::new(8, 8, 8),
    ];
    for res in grids {
        let count = res.voxel_count() as usize;
        let velocity = rng.vector_field(count, 4.0);
        check_divergence(
            &format!("divergence {}x{}x{}", res.nx, res.ny, res.nz),
            &engine,
            &ctx,
            &velocity,
            res,
        );
    }
}

#[test]
fn projection_matches_golden_across_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let mut rng = Lcg::new(0x0bad_f00d);

    let grids = [
        GridResolution::new(3, 3, 3),
        GridResolution::new(4, 4, 4),
        GridResolution::new(5, 6, 7),
        GridResolution::new(8, 8, 8),
    ];
    for res in grids {
        let count = res.voxel_count() as usize;
        let velocity = rng.vector_field(count, 3.0);
        let pressure = rng.scalar_field(count, 5.0);
        check_projection(
            &format!("projection {}x{}x{}", res.nx, res.ny, res.nz),
            &engine,
            &ctx,
            &velocity,
            &pressure,
            res,
        );
    }
}

#[test]
fn single_voxel_grid_is_fully_walled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let res = GridResolution::new(1, 1, 1);

    // Every face is out of grid: divergence = -(here.x + here.y + here.z) and
    // gradient_backward = 0 so projection leaves the velocity unchanged.
    let velocity = vec![Vec3::new(1.5, -2.25, 0.75)];
    let pressure = vec![3.5];
    check_divergence("single voxel divergence", &engine, &ctx, &velocity, res);
    check_projection(
        "single voxel projection",
        &engine,
        &ctx,
        &velocity,
        &pressure,
        res,
    );
}

#[test]
fn one_dimensional_chain_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let mut rng = Lcg::new(0x5151_2727);

    let res = GridResolution::new(9, 1, 1);
    let count = res.voxel_count() as usize;
    let velocity = rng.vector_field(count, 2.0);
    let pressure = rng.scalar_field(count, 2.0);
    check_divergence("1-D chain divergence", &engine, &ctx, &velocity, res);
    check_projection(
        "1-D chain projection",
        &engine,
        &ctx,
        &velocity,
        &pressure,
        res,
    );
}

#[test]
fn empty_grid_short_circuits_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let res = GridResolution::new(0, 4, 4);

    let div = engine.divergence(
        &ctx,
        &GpuDivergenceQuery {
            velocity: Vec::new(),
            resolution: res,
        },
    );
    assert!(div.is_empty(), "empty grid divergence should be empty");

    let projected = engine.project(
        &ctx,
        &GpuProjectionQuery {
            velocity: Vec::new(),
            pressure: Vec::new(),
            resolution: res,
        },
    );
    assert!(
        projected.is_empty(),
        "empty grid projection should be empty"
    );
}

#[test]
fn too_short_input_short_circuits_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgDivergenceGradient::new(&ctx);
    let res = GridResolution::new(4, 4, 4);

    // One sample short of the 64-voxel grid: the twin returns an empty vector.
    let short_velocity = vec![Vec3::ZERO; 63];
    let short_pressure = vec![0.0f32; 63];

    let div = engine.divergence(
        &ctx,
        &GpuDivergenceQuery {
            velocity: short_velocity.clone(),
            resolution: res,
        },
    );
    assert!(div.is_empty(), "short divergence input should be empty");

    let projected = engine.project(
        &ctx,
        &GpuProjectionQuery {
            velocity: short_velocity,
            pressure: short_pressure,
            resolution: res,
        },
    );
    assert!(
        projected.is_empty(),
        "short projection input should be empty"
    );
}
