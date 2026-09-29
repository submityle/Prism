//! Real-device parity: the `GPU` `FLIP`/`APIC` particle-grid transfer must
//! reproduce the `CPU` golden twin's reconstructed velocities within a tight
//! floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! The transfer stage scatters particle velocities onto the staggered grid with
//! a fixed-point atomic splat, normalises each face by its accumulated weight,
//! then gathers the field back to the particles. Because the momentum and
//! weight accumulators are integer-exact and order-independent, the `CPU` and
//! `GPU` accumulators agree bit-for-bit; only the final per-face division and
//! the trilinear gather are floating point, so the reconstructed velocities are
//! compared within a tolerance rather than for exact equality. The scenes use a
//! pure-`PIC` blend (`0`) over the zero saved field of this transfer-only
//! stage, which is an exact particle-grid-particle round-trip.
//!
//! Provenance: trilinear `P2G`/`G2P` with the `PIC`/`FLIP` blend (Zhu and
//! Bridson 2005; Bridson). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    grid_to_particle, particle_to_grid, FluidParticles, GoldenGrid, GpuContext, GpuFluidSolver,
    GridDims,
};

/// Maximum allowed per-particle velocity divergence between the two engines.
const TOLERANCE: f32 = 1e-3;

/// A cubic block of particles filling the grid interior, each carrying a
/// velocity produced by `velocity_at` evaluated at its position.
fn block(dims: GridDims, span: u32, velocity_at: impl Fn(Vec3) -> Vec3) -> FluidParticles {
    let mut particles = FluidParticles::new();
    let origin = dims.origin;
    let dx = dims.dx;
    // Fill a `span`-per-axis lattice inside the grid, offset off cell centres so
    // the trilinear stencils straddle several faces.
    for a in 0..span {
        for b in 0..span {
            for c in 0..span {
                let position = origin
                    + Vec3::new(
                        (a as f32 + 1.3) * dx,
                        (b as f32 + 1.7) * dx,
                        (c as f32 + 1.5) * dx,
                    );
                particles.spawn(position, velocity_at(position));
            }
        }
    }
    particles
}

/// Runs the `CPU` golden transfer (scatter, normalise, pure-`PIC` gather) on a
/// clone of `particles`, returning the reconstructed velocities.
fn cpu_reference(dims: GridDims, particles: &FluidParticles) -> Vec<Vec3> {
    let mut grid = GoldenGrid::new(dims);
    particle_to_grid(&mut grid, particles);
    let mut out = particles.clone();
    grid_to_particle(&grid, &mut out, 0.0);
    out.velocities().to_vec()
}

/// Asserts every particle velocity in `gpu` is within [`TOLERANCE`] of `cpu`.
fn assert_parity(cpu: &[Vec3], gpu: &[Vec3], scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: particle counts differ");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let delta = (*c - *g).length();
        assert!(
            delta <= TOLERANCE,
            "{scene}: particle {i} diverged by {delta}: cpu {c:?} vs gpu {g:?}"
        );
    }
}

/// Runs one scene through both engines and checks parity.
fn run_scene(
    ctx: &GpuContext,
    solver: &GpuFluidSolver,
    dims: GridDims,
    mut particles: FluidParticles,
    scene: &str,
) {
    let cpu = cpu_reference(dims, &particles);
    solver
        .transfer(ctx, dims, &mut particles, 0.0)
        .expect("gpu transfer");
    assert_parity(&cpu, particles.velocities(), scene);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_fluid_transfer_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU fluid transfer parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuFluidSolver::new(&ctx);
    let dims = GridDims::new(16, 16, 16, 0.1, Vec3::ZERO);

    // Uniform flow: every particle carries the same velocity.
    let uniform = block(dims, 6, |_| Vec3::new(0.4, -0.2, 0.1));
    run_scene(&ctx, &solver, dims, uniform, "uniform_flow");

    // Linear shear: velocity varies with position so the gather straddles a
    // non-constant field and exercises the interpolation weights.
    let shear = block(dims, 6, |p| Vec3::new(0.5 * p.y, 0.3 * p.x, -0.2 * p.z));
    run_scene(&ctx, &solver, dims, shear, "linear_shear");
}
