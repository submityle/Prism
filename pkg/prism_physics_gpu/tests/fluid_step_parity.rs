//! Real-device parity: the fused `GPU` full fluid step must reproduce the `CPU`
//! golden [`fluid_step`] within a tight tolerance.
//!
//! The step chains a particle-to-grid scatter, body force, solid handling, a
//! red-black `SOR` pressure projection, free-surface extrapolation, a
//! grid-to-particle gather, and an `RK2` advection. Every stage has already
//! been parity-checked against its standalone kernel, so this test anchors the
//! *composition*: the device orchestrator [`GpuFluidStep`] runs all stages over
//! one resident buffer set in a single submission, and its output must track
//! the sequentially executed host twin.
//!
//! The device reassociates the many sums (fused multiply-add, differing
//! rounding) and performs one division per projected / extrapolated face, so
//! parity is checked within a small relative tolerance rather than bit-for-bit.
//! A single step is checked strictly; a short multi-step run then confirms the
//! composition stays in lockstep as small per-step differences accumulate.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the transfer, `MAC` operators, red-black `SOR` projection,
//! free-surface extrapolation, and `RK2` advection are standard published
//! techniques (Bridson; Zhu and Bridson 2005; Foster and Fedkiw 2001). No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    fluid::cpu::{fields::GoldenGrid, step::fluid_step},
    CellType, FluidConfig, FluidParticles, GpuContext, GpuFluidStep, GridDims,
};

/// Builds a solid-walled grid classification: the one-cell border is solid, the
/// interior is fluid.
fn walled_cells(dims: GridDims) -> Vec<CellType> {
    let (nx, ny, nz) = (dims.nx, dims.ny, dims.nz);
    let mut cells = vec![CellType::Fluid; dims.cell_count()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let border =
                    i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                if border {
                    cells[dims.cell_idx(i, j, k)] = CellType::Solid;
                }
            }
        }
    }
    cells
}

/// A resting column of markers a few cells above the floor, seeded with a mild
/// shear so the transfer, projection, and advection all do real work.
fn water_column(dims: GridDims) -> FluidParticles {
    let mut parts = FluidParticles::new();
    for i in 3..8 {
        for j in 3..10 {
            for k in 3..8 {
                let pos = Vec3::new(
                    (i as f32 + 0.5) * dims.dx,
                    (j as f32 + 0.5) * dims.dx,
                    (k as f32 + 0.5) * dims.dx,
                );
                let vel = Vec3::new(0.1 * j as f32, -0.05 * i as f32, 0.02 * k as f32);
                parts.spawn(pos, vel);
            }
        }
    }
    parts
}

/// Largest relative position / velocity difference between two particle sets.
fn max_relative_error(cpu: &FluidParticles, gpu: &FluidParticles) -> f32 {
    let mut max_rel = 0.0f32;
    for (c, g) in cpu.positions().iter().zip(gpu.positions()) {
        let rel = (*c - *g).length() / c.length().max(1.0e-3);
        max_rel = max_rel.max(rel);
    }
    for (c, g) in cpu.velocities().iter().zip(gpu.velocities()) {
        let rel = (*c - *g).length() / c.length().max(1.0e-3);
        max_rel = max_rel.max(rel);
    }
    max_rel
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn gpu_full_step_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU full-step parity: no wgpu adapter on this host");
        return;
    };
    let stepper = GpuFluidStep::new(&ctx);
    let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
    let cells = walled_cells(dims);
    let mut cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));
    cfg.pressure_iterations = 40;

    // --- Single step: strict parity. ---
    let mut cpu_grid = GoldenGrid::new(dims);
    let mut cpu = water_column(dims);
    fluid_step(&mut cpu_grid, &mut cpu, &cells, &cfg);

    let mut gpu = water_column(dims);
    stepper
        .step(&ctx, dims, &cells, &mut gpu, &cfg)
        .expect("gpu full step");

    assert_eq!(cpu.len(), gpu.len(), "particle count must be preserved");
    let single_rel = max_relative_error(&cpu, &gpu);
    assert!(
        single_rel < 5.0e-3,
        "single-step parity exceeded tolerance: max relative error {single_rel}"
    );

    // Confirm the step actually moved the markers (guards against a no-op).
    let mut moved = false;
    let fresh = water_column(dims);
    for (a, b) in gpu.positions().iter().zip(fresh.positions()) {
        if (*a - *b).length() > 1.0e-5 {
            moved = true;
            break;
        }
    }
    assert!(moved, "expected the step to move at least one marker");

    // --- Five steps: parity must survive accumulation. ---
    let mut cpu_grid = GoldenGrid::new(dims);
    let mut cpu = water_column(dims);
    let mut gpu = water_column(dims);
    let steps = 5;
    for _ in 0..steps {
        fluid_step(&mut cpu_grid, &mut cpu, &cells, &cfg);
        stepper
            .step(&ctx, dims, &cells, &mut gpu, &cfg)
            .expect("gpu full step");
    }
    let multi_rel = max_relative_error(&cpu, &gpu);
    assert!(
        multi_rel < 2.0e-2,
        "{steps}-step parity exceeded tolerance: max relative error {multi_rel}"
    );

    eprintln!(
        "gpu full-step parity: single-step max relative error {single_rel}, \
         {steps}-step max relative error {multi_rel}"
    );
}
