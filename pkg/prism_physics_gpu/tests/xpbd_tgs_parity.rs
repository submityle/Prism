//! Real-device parity: the `GPU` Temporal Gauss-Seidel (`TGS`) distance solver
//! must reproduce the `CPU` golden twin's trajectory within a tight
//! floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! The scenes are a pinned hanging chain (a stress test for colour ordering,
//! since consecutive edges share a particle and must land in different colours)
//! and a soft single-link pendulum (which exercises the `SoftParams`
//! coefficients: a non-rigid `hertz` drives the biased sweep's `mass_scale` and
//! `impulse_scale` on both the host and the device). Because the device runs
//! `GPU` floating-point with fused multiply-add and differing division and
//! square-root rounding, both positions and velocities are compared within a
//! tolerance rather than for exact equality.
//!
//! Provenance: Temporal Gauss-Seidel substepping with soft constraints
//! (`PhysX` 5 / Chaos lineage; Catto, "Soft Constraints", GDC 2011). No Unreal
//! Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    tgs_solve, DistanceConstraint, GpuContext, GpuTgsSolver, ParticleState, TgsConfig,
};

/// Maximum allowed per-particle position divergence between the two engines.
const POSITION_TOLERANCE: f32 = 1e-3;

/// Maximum allowed per-particle velocity divergence between the two engines.
const VELOCITY_TOLERANCE: f32 = 1e-3;

/// A pinned chain of `n` particles hanging along `-Y`, unit spacing. Particle 0
/// is pinned (inverse mass 0); the rest are free.
fn hanging_chain(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let y = -(i as f32);
        let inv_mass = if i == 0 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.0, y, 0.0), inv_mass);
    }
    let constraints = (0..n - 1)
        .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 0.0))
        .collect();
    (state, constraints)
}

/// A single soft link: a pinned anchor and one bob one unit below it. The low
/// stiffness lets the link stretch under gravity, exercising the soft
/// coefficients rather than the rigid Baumgarte limit.
fn soft_pendulum() -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    state.push(Vec3::ZERO, 0.0);
    state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
    let constraints = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
    (state, constraints)
}

/// Asserts every particle in `gpu` is within tolerance of `cpu` for both
/// position and velocity.
fn assert_parity(cpu: &ParticleState, gpu: &ParticleState, scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: particle counts differ");
    for i in 0..cpu.len() {
        let dp = (cpu.positions[i] - gpu.positions[i]).length();
        assert!(
            dp <= POSITION_TOLERANCE,
            "{scene}: particle {i} position diverged by {dp}: cpu {:?} vs gpu {:?}",
            cpu.positions[i],
            gpu.positions[i]
        );
        let dv = (cpu.velocities[i] - gpu.velocities[i]).length();
        assert!(
            dv <= VELOCITY_TOLERANCE,
            "{scene}: particle {i} velocity diverged by {dv}: cpu {:?} vs gpu {:?}",
            cpu.velocities[i],
            gpu.velocities[i]
        );
    }
}

/// Runs `frames` of both engines from the same initial state and checks parity.
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuTgsSolver,
    initial: &ParticleState,
    constraints: &[DistanceConstraint],
    config: &TgsConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        tgs_solve(&mut cpu, constraints, config, dt).expect("cpu tgs solve");
        solver
            .solve(ctx, &mut gpu, constraints, config, dt)
            .expect("gpu tgs solve");
    }
    assert_parity(&cpu, &gpu, scene);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_tgs_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU TGS parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuTgsSolver::new(&ctx);

    // Rigid hanging chain: default config is the stiff limit, so the biased
    // sweep uses a full Baumgarte bias with no impulse decay.
    let rigid = TgsConfig::default();
    let (chain, chain_cons) = hanging_chain(16);
    run_parity(
        &ctx,
        &solver,
        &chain,
        &chain_cons,
        &rigid,
        30,
        "rigid_chain",
    );

    // Soft pendulum: a finite stiffness drives the soft `mass_scale` /
    // `impulse_scale` path through many biased and relaxation sweeps.
    let soft = TgsConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 2, 2, 4.0, 1.0, 0.5);
    let (pendulum, pendulum_cons) = soft_pendulum();
    run_parity(
        &ctx,
        &solver,
        &pendulum,
        &pendulum_cons,
        &soft,
        40,
        "soft_pendulum",
    );
}
