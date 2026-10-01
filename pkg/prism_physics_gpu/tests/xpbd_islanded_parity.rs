//! Real-device parity: the island-aware, sleep-skipping `GPU` `XPBD` stepper
//! must reproduce the `CPU` island-aware twin's trajectory within a tight
//! floating-point tolerance, and must agree with it on which islands sleep.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full masked upload / dispatch / readback path on any machine
//! with a real device.
//!
//! Two scenes are checked:
//! * an **awake** pinned chain with a dwell long enough that nothing sleeps in
//!   the window — this isolates the masked substep path and demands tight
//!   per-frame parity against the `CPU` [`IslandedSolver`], including identical
//!   per-frame [`IslandStep`] summaries; and
//! * a **settling** pair of independent chains run long enough to fall asleep —
//!   this exercises the sleep-skipping path end to end and checks both engines
//!   reach the same sleeping partition and the same frozen rest state.
//!
//! Because the device runs `GPU` floating-point with fused multiply-add and
//! differing division and square-root rounding, positions are compared within a
//! tolerance rather than for exact equality.
//!
//! Provenance: standard island partitioning + velocity-threshold island
//! sleeping layered over substep `XPBD` (Müller et al.). No Unreal Engine source
//! or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    DistanceConstraint, GpuContext, GpuIslandedXpbdSolver, IslandedSolver, ParticleState,
    SleepConfig, XpbdConfig,
};

/// Maximum allowed per-particle divergence while both engines are awake.
const AWAKE_TOLERANCE: f32 = 1e-3;
/// Looser bound once both engines have settled and frozen: a one-frame
/// difference in sleep onset leaves the two frozen at near-identical rest.
const SETTLED_TOLERANCE: f32 = 2e-2;

/// A pinned chain of `n` particles hanging along `-Y`, unit spacing.
fn hanging_chain(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.0, -(i as f32), 0.0), inv_mass);
    }
    let constraints = (0..n - 1)
        .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 0.0))
        .collect();
    (state, constraints)
}

/// Two independent pinned pendulums, far enough apart to form separate islands.
fn two_chains() -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    state.push(Vec3::ZERO, 0.0);
    state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
    state.push(Vec3::new(5.0, 0.0, 0.0), 0.0);
    state.push(Vec3::new(5.0, -1.0, 0.0), 1.0);
    let cons = vec![
        DistanceConstraint::new(0, 1, 1.0, 0.0),
        DistanceConstraint::new(2, 3, 1.0, 0.0),
    ];
    (state, cons)
}

/// Asserts every particle in `gpu` is within `tol` of `cpu` in position.
fn assert_parity(cpu: &ParticleState, gpu: &ParticleState, tol: f32, scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: particle counts differ");
    for i in 0..cpu.len() {
        let delta = (cpu.positions[i] - gpu.positions[i]).length();
        assert!(
            delta <= tol,
            "{scene}: particle {i} diverged by {delta}: cpu {:?} vs gpu {:?}",
            cpu.positions[i],
            gpu.positions[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_islanded_matches_cpu_islanded_while_awake() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU islanded parity (awake): no wgpu adapter on this host");
        return;
    };

    let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 1.0);
    // A dwell long enough that nothing sleeps during the compared window.
    let sleep = SleepConfig::new(0.01, 1_000.0);
    let mut gpu_solver = GpuIslandedXpbdSolver::new(&ctx, sleep);
    let mut cpu_solver = IslandedSolver::new(sleep);

    let (mut gpu_state, cons) = hanging_chain(16);
    let mut cpu_state = gpu_state.clone();
    let dt = 1.0 / 60.0;

    for frame in 0..30 {
        let cpu_step = cpu_solver
            .step(&mut cpu_state, &cons, &config, dt)
            .expect("cpu islanded step");
        let gpu_step = gpu_solver
            .step(&ctx, &mut gpu_state, &cons, &config, dt)
            .expect("gpu islanded step");
        assert_eq!(
            cpu_step, gpu_step,
            "frame {frame}: island summaries diverged"
        );
        assert_eq!(
            gpu_step.asleep_islands, 0,
            "frame {frame}: slept unexpectedly"
        );
        assert_parity(
            &cpu_state,
            &gpu_state,
            AWAKE_TOLERANCE,
            "hanging_chain_awake",
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_islanded_sleeps_and_settles_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU islanded parity (settle): no wgpu adapter on this host");
        return;
    };

    let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 4.0);
    let sleep = SleepConfig::new(0.05, 0.2);
    let mut gpu_solver = GpuIslandedXpbdSolver::new(&ctx, sleep);
    let mut cpu_solver = IslandedSolver::new(sleep);

    let (mut gpu_state, cons) = two_chains();
    let mut cpu_state = gpu_state.clone();
    let dt = 1.0 / 60.0;

    for _ in 0..600 {
        cpu_solver
            .step(&mut cpu_state, &cons, &config, dt)
            .expect("cpu islanded step");
        gpu_solver
            .step(&ctx, &mut gpu_state, &cons, &config, dt)
            .expect("gpu islanded step");
    }

    // Both engines must have put both pendulums fully to sleep.
    // Each chain has a single dynamic bob (its anchor is pinned and never
    // counted asleep), so a fully settled two-chain scene reports two asleep
    // particles — one per chain — on both engines.
    assert_eq!(
        gpu_solver.sleep_state().asleep_count(),
        2,
        "gpu did not sleep both chains"
    );
    assert_eq!(
        cpu_solver.sleep_state().asleep_count(),
        2,
        "cpu did not sleep both chains"
    );

    // Frozen velocities and matching rest positions.
    for p in 0..gpu_state.len() {
        assert!(
            gpu_state.velocities[p].length() < 1e-6,
            "gpu particle {p} velocity not pinned"
        );
    }
    assert_parity(
        &cpu_state,
        &gpu_state,
        SETTLED_TOLERANCE,
        "two_chains_settled",
    );

    // Rest lengths preserved on the device after settling.
    for &(a, b) in &[(0u32, 1u32), (2, 3)] {
        let len = (gpu_state.positions[b as usize] - gpu_state.positions[a as usize]).length();
        assert!(
            (len - 1.0).abs() < 2e-2,
            "link {a}-{b} rest length was {len}"
        );
    }

    // Waking one chain re-engages the solver for only that island.
    gpu_solver.wake(1);
    let step = gpu_solver
        .step(&ctx, &mut gpu_state, &cons, &config, dt)
        .expect("gpu islanded step after wake");
    assert_eq!(
        step.asleep_islands, 1,
        "the undisturbed chain should stay asleep"
    );
    assert_eq!(
        step.solved_constraints, 1,
        "only the woken chain should solve"
    );
}
