//! Real-device parity for the `GPU` cloth continuous self-collision (self-CCD)
//! sweep against its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The `CPU` twin ([`cpu_cloth_self_ccd`]) delegates to `prism_physics_core`'s
//! `resolve_self_ccd_jacobi`, while the kernel reimplements the same closed-form
//! swept-pair TOI resolution, thickness separation, and normal restitution
//! impulse in `WGSL` over a host-built candidate set and incidence list. The
//! only divergence is a few `ULP` in `sqrt`/division, so parity is checked
//! within a tight tolerance — and the inputs deliberately avoid exact
//! knife-edge grazes (where `c` is near zero) that would flip a hit/miss branch.
//!
//! Provenance: the closed-form swept-pair TOI resolution is standard analytic
//! continuous-collision geometry; the Jacobi own-slot accumulate/apply split is
//! standard parallel position-based dynamics. No Unreal Engine source or derived
//! code.

use glam::Vec3;
use prism_physics_core::SelfCcdParams;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_self_ccd, GpuClothSelfCcd};

/// Absolute/relative tolerance for position and velocity parity.
const TOL: f32 = 1.0e-4;

/// Frame step used across the suite.
const DT: f32 = 1.0 / 60.0;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth self-ccd parity: no GPU adapter available");
            None
        }
    }
}

/// Asserts two vectors agree within [`TOL`] (combined absolute/relative).
#[track_caller]
fn assert_vec_close(a: Vec3, b: Vec3, what: &str) {
    let scale = a.length().max(b.length()).max(1.0);
    assert!(
        a.distance(b) <= TOL * scale,
        "{what}: cpu={a:?} gpu={b:?} (dist {})",
        a.distance(b)
    );
}

/// Runs both paths over the same inputs and asserts full parity of the applied
/// positions and velocities.
#[track_caller]
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuClothSelfCcd,
    positions: &[Vec3],
    prev_positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[f32],
    params: SelfCcdParams,
    dt: f32,
) {
    let (cpu_pos, cpu_vel) = cpu_cloth_self_ccd(
        positions,
        prev_positions,
        velocities,
        inverse_masses,
        params,
        dt,
    );
    let (gpu_pos, gpu_vel) = kernel.solve(
        ctx,
        positions,
        prev_positions,
        velocities,
        inverse_masses,
        params,
        dt,
    );

    assert_eq!(cpu_pos.len(), gpu_pos.len(), "position count mismatch");
    assert_eq!(cpu_vel.len(), gpu_vel.len(), "velocity count mismatch");
    for (i, (c, g)) in cpu_pos.iter().zip(&gpu_pos).enumerate() {
        assert_vec_close(*c, *g, &format!("position[{i}]"));
    }
    for (i, (c, g)) in cpu_vel.iter().zip(&gpu_vel).enumerate() {
        assert_vec_close(*c, *g, &format!("velocity[{i}]"));
    }
}

/// Enabled, no-bounce settings over a cell size that keeps the test clusters in
/// a shared bucket.
fn params(cell_size: f32, thickness: f32) -> SelfCcdParams {
    SelfCcdParams::new(cell_size, thickness)
}

#[test]
fn tunnelling_crossing_pair_is_caught() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    // Two particles swap sides in one step; TOI is a clean 0.4 (far from a
    // knife-edge grazing where c -> 0).
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::ZERO, Vec3::ZERO];
    let inv_mass = [1.0, 1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.8, 0.4),
        DT,
    );
}

#[test]
fn shared_cell_cluster_resolves_in_order() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    // Three mutually near, mutually crossing particles sharing a cell, so the
    // Jacobi cluster folds pairs (0,1), (0,2), (1,2) per particle.
    let positions = [
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(-0.05, 0.0, 0.0),
        Vec3::new(0.0, 0.05, 0.0),
    ];
    let prev = [
        Vec3::new(-0.05, 0.0, 0.0),
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::new(0.0, -0.05, 0.0),
    ];
    let velocities = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
    let inv_mass = [1.0, 1.0, 1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.4, 0.3),
        DT,
    );
}

#[test]
fn pinned_partner_takes_the_full_correction() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    // Particle 0 is pinned (inv_mass 0); the whole separation and velocity
    // recovery must land on the free partner.
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::ZERO, Vec3::ZERO];
    let inv_mass = [0.0, 1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.8, 0.4),
        DT,
    );
}

#[test]
fn double_pinned_pair_is_a_no_op() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::ZERO, Vec3::ZERO];
    let inv_mass = [0.0, 0.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.8, 0.4),
        DT,
    );
}

#[test]
fn restitution_recovers_normal_velocity() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::new(0.2, 0.1, 0.0), Vec3::new(-0.2, -0.1, 0.0)];
    let inv_mass = [1.0, 1.0];
    for &restitution in &[0.0_f32, 1.0] {
        let mut p = params(0.8, 0.4);
        p.restitution = restitution;
        assert_parity(
            &ctx,
            &kernel,
            &positions,
            &prev,
            &velocities,
            &inv_mass,
            p,
            DT,
        );
    }
}

#[test]
fn zero_dt_disables_velocity_recovery() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::new(0.3, 0.0, 0.0), Vec3::new(-0.3, 0.0, 0.0)];
    let inv_mass = [1.0, 1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.8, 0.4),
        0.0,
    );
}

#[test]
fn disabled_sweep_is_a_no_op() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
    let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
    let velocities = [Vec3::new(0.5, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0)];
    let inv_mass = [1.0, 1.0];
    let mut p = params(0.8, 0.4);
    p.enabled = false;
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        p,
        DT,
    );
}

#[test]
fn single_particle_is_a_no_op() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    let positions = [Vec3::new(0.3, 0.1, -0.2)];
    let prev = [Vec3::new(-0.3, 0.1, -0.2)];
    let velocities = [Vec3::new(0.1, 0.0, 0.0)];
    let inv_mass = [1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.8, 0.4),
        DT,
    );
}

#[test]
fn separated_particles_produce_no_contact() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    // Two particles far apart whose swept boxes never share a cell.
    let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)];
    let prev = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)];
    let velocities = [Vec3::ZERO, Vec3::ZERO];
    let inv_mass = [1.0, 1.0];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        params(0.4, 0.2),
        DT,
    );
}

#[test]
fn mixed_batch_many_particles() {
    let Some(ctx) = headless() else { return };
    let kernel = GpuClothSelfCcd::new(&ctx);
    // A deterministic, trig-free lattice of crossing and non-crossing pairs
    // with scattered pinned particles — exercises the full host broad phase and
    // per-particle CSR reduction.
    let mut positions = Vec::new();
    let mut prev = Vec::new();
    let mut velocities = Vec::new();
    let mut inv_mass = Vec::new();
    for k in 0..96u32 {
        let i = k as f32;
        // Each adjacent pair (2j, 2j+1) swaps across a shared line so it crosses
        // cleanly, while distinct j land in different cells along x.
        let base_x = (k / 2) as f32 * 0.6;
        let z = ((k % 5) as f32) * 0.03 - 0.06;
        if k % 2 == 0 {
            positions.push(Vec3::new(base_x + 0.1, 0.0, z));
            prev.push(Vec3::new(base_x - 0.1, 0.0, z));
        } else {
            positions.push(Vec3::new(base_x - 0.1, 0.0, z));
            prev.push(Vec3::new(base_x + 0.1, 0.0, z));
        }
        velocities.push(Vec3::new(0.05 - (k % 3) as f32 * 0.03, i * 0.001, 0.0));
        // Every eleventh particle is pinned.
        inv_mass.push(if k % 11 == 0 { 0.0 } else { 1.0 });
    }
    let mut p = params(0.3, 0.2);
    p.restitution = 0.4;
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &prev,
        &velocities,
        &inv_mass,
        p,
        DT,
    );
}
