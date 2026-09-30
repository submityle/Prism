//! Real-device parity: the fused `GPU` full `APIC` fluid step must reproduce the
//! `CPU` golden [`fluid_step`] (with [`TransferMode::Apic`]) within a tight
//! tolerance, including the per-particle affine matrix `C`.
//!
//! The affine step chains an affine particle-to-grid scatter, body force, solid
//! handling, a red-black `SOR` pressure projection, free-surface extrapolation,
//! an affine grid-to-particle gather that refits `C` by a per-component
//! regularised least-squares solve, and an `RK2` advection. Every shared stage
//! is already parity-checked by the `FLIP` full-step test; this test anchors the
//! affine endpoints: the scatter's `C·(x_face − x_p)` correction and the gather's
//! moment-matrix solve. The device orchestrator [`GpuFluidApicStep`] runs all
//! stages over one resident buffer set in a single submission, and its output —
//! positions, velocities, and affine matrices — must track the sequentially
//! executed host twin.
//!
//! The device reassociates the many sums (fused multiply-add, differing
//! rounding) and inverts one moment matrix per particle per component, so parity
//! is checked within a small relative tolerance rather than bit-for-bit. A
//! single step is checked strictly; a short multi-step run then confirms the
//! composition stays in lockstep as small per-step differences accumulate.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing.
//!
//! Provenance: the affine `P2G`/`G2P` transfer (Jiang et al. 2015), the `MAC`
//! operators, red-black `SOR` projection, free-surface extrapolation, and `RK2`
//! advection are standard published techniques (Bridson; Zhu and Bridson 2005;
//! Foster and Fedkiw 2001). No Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use prism_physics_gpu::{
    fluid::cpu::{fields::GoldenGrid, step::fluid_step},
    CellType, FluidConfig, FluidParticles, GpuContext, GpuFluidApicStep, GridDims, TransferMode,
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

/// A constant affine velocity gradient (a mild in-plane shear / rotation). Both
/// the seed velocity `v = C·(x − centre)` and the affine matrix `C` are derived
/// from this tensor so the affine scatter and gather both carry real signal.
fn shear_tensor() -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, 0.15, 0.0),
        Vec3::new(-0.15, 0.0, 0.0),
        Vec3::ZERO,
    )
}

/// A resting column of markers a few cells above the floor, seeded with the
/// affine velocity field `v = C·(x − centre)` and matching affine matrix `C`, so
/// the transfer, projection, advection, and affine refit all do real work.
fn water_column(dims: GridDims) -> FluidParticles {
    let mut parts = FluidParticles::new();
    let affine = shear_tensor();
    let extent = Vec3::new(
        dims.nx as f32 * dims.dx,
        dims.ny as f32 * dims.dx,
        dims.nz as f32 * dims.dx,
    );
    let centre = dims.origin + 0.5 * extent;
    for i in 3..8 {
        for j in 3..10 {
            for k in 3..8 {
                let pos = dims.origin
                    + Vec3::new(
                        (i as f32 + 0.5) * dims.dx,
                        (j as f32 + 0.5) * dims.dx,
                        (k as f32 + 0.5) * dims.dx,
                    );
                let vel = affine * (pos - centre);
                let idx = parts.spawn(pos, vel);
                parts.affine_mut()[idx] = affine;
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

/// Largest relative Frobenius difference between the two affine-matrix columns.
fn max_affine_error(cpu: &FluidParticles, gpu: &FluidParticles) -> f32 {
    let mut max_rel = 0.0f32;
    for (c, g) in cpu.affine().iter().zip(gpu.affine()) {
        let diff = (*c - *g)
            .to_cols_array()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let norm = c
            .to_cols_array()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1.0e-3);
        max_rel = max_rel.max(diff / norm);
    }
    max_rel
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and measured error must reach the test log"
)]
fn gpu_apic_step_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU APIC full-step parity: no wgpu adapter on this host");
        return;
    };
    let stepper = GpuFluidApicStep::new(&ctx);
    let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
    let cells = walled_cells(dims);
    let mut cfg = FluidConfig::new(5.0e-3, Vec3::new(0.0, -9.81, 0.0));
    cfg.pressure_iterations = 40;
    cfg.transfer = TransferMode::Apic;

    // --- Single step: strict parity across positions, velocities, and C. ---
    let mut cpu_grid = GoldenGrid::new(dims);
    let mut cpu = water_column(dims);
    fluid_step(&mut cpu_grid, &mut cpu, &cells, &cfg);

    let mut gpu = water_column(dims);
    stepper
        .step(&ctx, dims, &cells, &mut gpu, &cfg)
        .expect("gpu APIC step");

    assert_eq!(cpu.len(), gpu.len(), "particle count must be preserved");
    let single_rel = max_relative_error(&cpu, &gpu);
    assert!(
        single_rel < 5.0e-3,
        "single-step position/velocity parity exceeded tolerance: max relative error {single_rel}"
    );
    let single_affine = max_affine_error(&cpu, &gpu);
    assert!(
        single_affine < 5.0e-3,
        "single-step affine parity exceeded tolerance: max relative error {single_affine}"
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
            .expect("gpu APIC step");
    }
    let multi_rel = max_relative_error(&cpu, &gpu);
    assert!(
        multi_rel < 2.0e-2,
        "{steps}-step position/velocity parity exceeded tolerance: max relative error {multi_rel}"
    );
    let multi_affine = max_affine_error(&cpu, &gpu);
    assert!(
        multi_affine < 3.0e-2,
        "{steps}-step affine parity exceeded tolerance: max relative error {multi_affine}"
    );

    eprintln!(
        "gpu APIC full-step parity: single-step pos/vel {single_rel}, affine {single_affine}; \
         {steps}-step pos/vel {multi_rel}, affine {multi_affine}"
    );
}
