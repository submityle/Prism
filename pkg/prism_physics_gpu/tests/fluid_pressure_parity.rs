//! Real-device parity: the `GPU` red-black `SOR` pressure projection must
//! reproduce the `CPU` golden twin's projected velocity field within a tight
//! floating-point tolerance, and both must drive the fluid-cell divergence
//! down.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! Both engines run the identical red-black sweep order (one colour per
//! dispatch on the device, one colour per in-place sweep on the twin), so the
//! only differences are floating-point reassociation and the fused
//! multiply-add the device may use. Over 60 iterations of an `O(100)` pressure
//! field these accumulate, so the projected velocities are compared within a
//! relative tolerance rather than for exact equality.
//!
//! Provenance: the `MAC` pressure-projection scheme and the red-black `SOR`
//! Poisson solve follow Bridson and Foster and Fedkiw 2001. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    fluid::cpu::pressure::{max_fluid_divergence, project},
    CellType, GpuContext, GpuPressureSolver, GridDims, PressureConfig,
};

/// Maximum allowed relative per-face velocity divergence between the engines.
const TOLERANCE: f32 = 5e-3;

/// Builds the cell classification for a 12-cube with a one-thick solid wall
/// shell and a fluid block filling the interior.
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

/// A concatenated `[u | v | w]` field carrying a divergent `u` ramp plus a
/// mild `v` shear, so the solve exercises more than one axis.
fn divergent_field(dims: GridDims) -> Vec<f32> {
    let mut velocity = vec![0.0f32; dims.face_total()];
    let u_off = dims.u_offset();
    let v_off = dims.v_offset();
    for k in 0..dims.nz {
        for j in 0..dims.ny {
            for i in 0..=dims.nx {
                velocity[u_off + dims.u_idx(i, j, k)] = (i as f32 - 6.0) * 0.05;
            }
        }
    }
    for k in 0..dims.nz {
        for j in 0..=dims.ny {
            for i in 0..dims.nx {
                velocity[v_off + dims.v_idx(i, j, k)] = (j as f32 - 6.0) * 0.02;
            }
        }
    }
    velocity
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_pressure_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU pressure parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuPressureSolver::new(&ctx);
    let dims = GridDims::new(12, 12, 12, 0.1, Vec3::ZERO);
    let cells = walled_fluid_block(dims);
    let cfg = PressureConfig::new(1.0, 1.0e-2, 1.7, 60);

    let field = divergent_field(dims);
    let before = max_fluid_divergence(dims, &cells, &field);

    // CPU golden.
    let mut cpu_velocity = field.clone();
    let _cpu_pressure = project(dims, &cells, &mut cpu_velocity, cfg);
    let cpu_after = max_fluid_divergence(dims, &cells, &cpu_velocity);

    // GPU.
    let mut gpu_velocity = field.clone();
    solver
        .project(&ctx, dims, &cells, &mut gpu_velocity, cfg)
        .expect("gpu pressure projection");
    let gpu_after = max_fluid_divergence(dims, &cells, &gpu_velocity);

    // Both engines must reduce divergence substantially.
    assert!(
        cpu_after < before * 0.1,
        "cpu did not reduce divergence: before={before} after={cpu_after}"
    );
    assert!(
        gpu_after < before * 0.1,
        "gpu did not reduce divergence: before={before} after={gpu_after}"
    );

    // The two projected fields must agree within a relative tolerance.
    for (idx, (c, g)) in cpu_velocity.iter().zip(gpu_velocity.iter()).enumerate() {
        let delta = (c - g).abs();
        let bound = TOLERANCE * (1.0 + c.abs());
        assert!(
            delta <= bound,
            "face {idx} diverged by {delta} (bound {bound}): cpu {c} vs gpu {g}"
        );
    }
}
