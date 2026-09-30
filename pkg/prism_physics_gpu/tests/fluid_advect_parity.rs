//! Real-device parity: the `GPU` marker advection must reproduce the `CPU`
//! golden twin's advected positions within a tight tolerance.
//!
//! Advection integrates each marker with the second-order Runge–Kutta midpoint
//! rule, whose only arithmetic is two trilinear velocity gathers. The device
//! reassociates those sums differently from the host (fused multiply-add,
//! differing rounding), so parity is checked within a small relative tolerance
//! rather than bit-for-bit.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! Provenance: second-order Runge–Kutta semi-Lagrangian advection (Bridson; Zhu
//! and Bridson 2005). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    fluid::cpu::{advect::advect_rk2, fields::GoldenGrid, p2g::particle_to_grid},
    FluidParticles, GpuAdvect, GpuContext, GridDims,
};

/// Builds a grid carrying a smoothly varying velocity field by scattering a
/// dense block of particles whose velocity depends on position, so the two
/// midpoint gathers differ and genuinely exercise the integrator.
fn sheared_field(dims: GridDims) -> GoldenGrid {
    let mut grid = GoldenGrid::new(dims);
    let mut src = FluidParticles::new();
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..16 {
                let x = 0.2 + a as f32 * 0.05;
                let y = 0.2 + b as f32 * 0.05;
                let z = 0.2 + c as f32 * 0.05;
                // A divergent-free-ish shear: vx grows with y, vy with z.
                let vel = Vec3::new(0.3 + 0.5 * y, -0.2 + 0.4 * z, 0.1 * x);
                src.spawn(Vec3::new(x, y, z), vel);
            }
        }
    }
    particle_to_grid(&mut grid, &src);
    grid
}

/// A scatter of probe markers inside the seeded field region.
fn probe_markers() -> Vec<Vec3> {
    let mut v = Vec::new();
    for a in 0..5 {
        for b in 0..5 {
            for c in 0..5 {
                v.push(Vec3::new(
                    0.4 + a as f32 * 0.03,
                    0.4 + b as f32 * 0.03,
                    0.4 + c as f32 * 0.03,
                ));
            }
        }
    }
    v
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_advect_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU advection parity: no wgpu adapter on this host");
        return;
    };
    let advector = GpuAdvect::new(&ctx);
    let dims = GridDims::new(20, 20, 20, 0.05, Vec3::ZERO);
    let grid = sheared_field(dims);
    let dt = 2.0e-2;

    let start = probe_markers();

    // CPU golden.
    let mut cpu = start.clone();
    advect_rk2(&grid, &mut cpu, dt);

    // GPU.
    let mut gpu = start.clone();
    advector
        .advect(&ctx, dims, grid.velocity(), &mut gpu, dt)
        .expect("gpu advect");

    let mut max_rel = 0.0f32;
    let mut moved = false;
    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let rel = (*c - *g).length() / c.length().max(1.0e-4);
        assert!(
            rel < 5.0e-3,
            "marker {idx} differs: cpu {c:?} vs gpu {g:?} (rel {rel})"
        );
        max_rel = max_rel.max(rel);
        if (*c - start[idx]).length() > 1.0e-4 {
            moved = true;
        }
    }
    assert!(
        moved,
        "expected the sheared field to move at least one marker"
    );
    eprintln!("gpu advection parity: max relative error {max_rel}");
}
