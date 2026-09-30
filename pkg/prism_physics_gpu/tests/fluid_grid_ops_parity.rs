//! Real-device parity: the `GPU` per-face grid operators (body-force
//! integration then solid-face enforcement) must reproduce the `CPU` golden
//! twin's field exactly.
//!
//! Both engines only ever add a host-computed constant to a face or store zero,
//! so — unlike the iterative pressure solve — there is no floating-point
//! reassociation to bound: the projected fields must match bit-for-bit. The
//! test asserts exact equality.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! Provenance: explicit body-force integration on a staggered `MAC` grid and
//! the solid no-through-flow boundary are standard `CFD` constructs (Harlow and
//! Welch 1965; Bridson). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    fluid::cpu::grid_ops::{add_gravity, enforce_solid_faces},
    CellType, GpuContext, GpuGridOps, GridDims,
};

/// Builds a 12-cube scene with a one-thick solid wall shell and a fluid block
/// filling the interior, so the enforcement touches both boundary and interior
/// faces.
fn walled_fluid_block(dims: GridDims) -> Vec<CellType> {
    let mut cells = vec![CellType::Air; dims.cell_count()];
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 0..dims.nx {
                let on_wall = i == 0
                    || j == 0
                    || k == 0
                    || i == dims.nx - 1
                    || j == dims.ny - 1
                    || k == dims.nz - 1;
                if on_wall {
                    cells[dims.cell_idx(i, j, k)] = CellType::Solid;
                }
            }
        }
    }
    for k in 3..9 {
        for j in 3..9 {
            for i in 3..9 {
                cells[dims.cell_idx(i, j, k)] = CellType::Fluid;
            }
        }
    }
    cells
}

/// A non-trivial concatenated `[u | v | w]` field so both operators have
/// something to act on.
fn seed_field(dims: GridDims) -> Vec<f32> {
    let mut velocity = vec![0.0f32; dims.face_total()];
    for (idx, face) in velocity.iter_mut().enumerate() {
        *face = 0.1 + (idx % 7) as f32 * 0.05;
    }
    velocity
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_grid_ops_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU grid-ops parity: no wgpu adapter on this host");
        return;
    };
    let ops = GpuGridOps::new(&ctx);
    let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
    let cells = walled_fluid_block(dims);
    let gravity = Vec3::new(0.7, -9.81, 0.3);
    let dt = 1.0e-2;
    let field = seed_field(dims);

    // CPU golden: gravity then solid enforcement, matching the reference step.
    let mut cpu_velocity = field.clone();
    add_gravity(dims, &mut cpu_velocity, gravity, dt);
    enforce_solid_faces(dims, &cells, &mut cpu_velocity);

    // GPU.
    let mut gpu_velocity = field.clone();
    ops.apply(&ctx, dims, &cells, &mut gpu_velocity, gravity, dt)
        .expect("gpu grid ops");

    // The operators are pure adds of a constant and stores of zero, so the
    // fields must agree bit-for-bit.
    for (idx, (c, g)) in cpu_velocity.iter().zip(gpu_velocity.iter()).enumerate() {
        assert_eq!(c, g, "face {idx} differs: cpu {c} vs gpu {g}");
    }

    // Sanity: at least one solid-bordering face was actually zeroed.
    let u_base = dims.u_offset();
    assert_eq!(gpu_velocity[u_base + dims.u_idx(1, 5, 5)], 0.0);
}
