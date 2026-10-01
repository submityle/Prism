//! Real-device parity for the implicit-gyroscopic angular update: the `GPU`
//! integrator's [`integrate_with_gyroscopic`](GpuRigidIntegrator::integrate_with_gyroscopic)
//! must reproduce the `CPU` golden twin
//! [`cpu_integrate_gyro`](prism_physics_gpu::cpu_integrate_gyro), frame for
//! frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device such as an Apple `M`-series `GPU`.
//!
//! The scenes stress the implicit Newton solve where it matters: a body spun
//! fast about its intermediate principal axis (the Dzhanibekov / tennis-racket
//! regime where the explicit scheme gains energy), a general asymmetric tumbler
//! under constant torque, and — critically — a body with a locked axis whose
//! zero inertia must drive both backends down the explicit fallback so they
//! stay in parity. Running many substeps per frame over many frames lets any
//! per-iteration divergence in the hand-written `3x3` cofactor solve accumulate,
//! so a shader that only approximately mirrored the `CPU` arithmetic would drift
//! past the tolerance rather than sliding under it.
//!
//! Provenance: implicit (backward-Euler) gyroscopic integration of Euler's
//! rigid-body equations, as in Bullet's `computeGyroscopicImpulseImplicit_Body`
//! and `PhysX`'s gyroscopic forces option. No Unreal Engine source or derived
//! code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_integrate_gyro, GpuContext, GpuRigidIntegrator, GyroscopicConfig, IntegratorConfig,
    RigidBodyState,
};

/// Absolute per-body divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// Asserts every body in `gpu` is within the magnitude-scaled tolerance of
/// `cpu` across position, orientation, and both velocities.
fn assert_parity(cpu: &RigidBodyState, gpu: &RigidBodyState, scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: body counts differ");
    for i in 0..cpu.len() {
        let dp = (cpu.positions[i] - gpu.positions[i]).length();
        let pos_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu.positions[i].length();
        assert!(
            dp <= pos_bound,
            "{scene}: body {i} position diverged by {dp} (bound {pos_bound}): cpu {:?} vs gpu {:?}",
            cpu.positions[i],
            gpu.positions[i]
        );

        // Quaternions q and -q encode the same orientation; compare the smaller
        // of the component-wise distances to each sign.
        let cq = cpu.orientations[i];
        let gq = gpu.orientations[i];
        let same = quat_component_distance(cq, gq);
        let flipped = quat_component_distance(cq, -gq);
        let dq = same.min(flipped);
        assert!(
            dq <= ABS_TOLERANCE + REL_TOLERANCE,
            "{scene}: body {i} orientation diverged by {dq}: cpu {cq:?} vs gpu {gq:?}"
        );

        let dv = (cpu.linear_velocities[i] - gpu.linear_velocities[i]).length();
        let vel_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu.linear_velocities[i].length();
        assert!(
            dv <= vel_bound,
            "{scene}: body {i} linear velocity diverged by {dv} (bound {vel_bound})"
        );

        let dw = (cpu.angular_velocities[i] - gpu.angular_velocities[i]).length();
        let ang_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu.angular_velocities[i].length();
        assert!(
            dw <= ang_bound,
            "{scene}: body {i} angular velocity diverged by {dw} (bound {ang_bound})"
        );
    }
}

/// Euclidean distance between two quaternions' raw `(x, y, z, w)` components.
fn quat_component_distance(a: Quat, b: Quat) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    let dw = a.w - b.w;
    (dx * dx + dy * dy + dz * dz + dw * dw).sqrt()
}

/// A scene exercising the implicit gyroscopic solve from several angles:
///
/// * body `0`: intermediate-axis spin (inertia `1, 2, 4`, a moderate spin about
///   the intermediate `y` axis with a small perturbation) — the Dzhanibekov
///   regime the implicit scheme stabilises, kept moderate here so the per-frame
///   parity check tests kernel arithmetic equivalence rather than the chaotic
///   (Lyapunov-unstable) phase divergence a fast flip would amplify; long-term
///   stability is covered by the `CPU` golden unit tests;
/// * body `1`: a general asymmetric tumbler with an off-axis initial spin;
/// * body `2`: a locked-axis body (inverse inertia `z = 0`, so inertia `z` is
///   infinite / zero inverse) whose zero inertia forces the explicit fallback
///   on both backends;
/// * body `3`: an asymmetric body driven by a constant torque.
fn gyroscopic_scene() -> (RigidBodyState, Vec<Vec3>, Vec<Vec3>) {
    let mut state = RigidBodyState::new();

    // 0: intermediate-axis (tennis-racket) spin, inertia 1, 2, 4.
    state.push(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        Vec3::new(1.0, 0.5, 0.25),
    );
    state.angular_velocities[0] = Vec3::new(0.03, 1.0, 0.03);

    // 1: general asymmetric tumbler, inertia 1.5, 0.8, 0.4.
    state.push(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        Vec3::new(1.0 / 1.5, 1.0 / 0.8, 1.0 / 0.4),
    );
    state.angular_velocities[1] = Vec3::new(0.6, 0.3, -0.4);

    // 2: locked z axis (inverse inertia z = 0) -> explicit fallback on both.
    state.push(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        Vec3::new(1.0, 0.5, 0.0),
    );
    state.angular_velocities[2] = Vec3::new(0.4, 0.5, 0.8);

    // 3: asymmetric body driven by a constant torque, inertia 2, 1, 0.5.
    state.push(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        Vec3::new(0.5, 1.0, 2.0),
    );

    let forces = vec![Vec3::ZERO; state.len()];
    let mut torques = vec![Vec3::ZERO; state.len()];
    torques[3] = Vec3::new(0.0, 0.3, 0.1);
    (state, forces, torques)
}

/// Runs `frames` of both engines from the same initial state under the implicit
/// gyroscopic scheme and checks per-frame parity.
fn run_parity(
    ctx: &GpuContext,
    integrator: &GpuRigidIntegrator,
    initial: &RigidBodyState,
    forces: &[Vec3],
    torques: &[Vec3],
    config: &IntegratorConfig,
    gyro: &GyroscopicConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_integrate_gyro(&mut cpu, forces, torques, config, gyro, dt).expect("cpu integrate");
        integrator
            .integrate_with_gyroscopic(ctx, &mut gpu, forces, torques, config, gyro, dt)
            .expect("gpu integrate");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_implicit_gyroscopic_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU implicit gyroscopic parity: no wgpu adapter on this host");
        return;
    };
    let integrator = GpuRigidIntegrator::new(&ctx);

    // Single Newton iteration (the default) over a long, high-substep run in
    // zero gravity isolates the rotational dynamics so any divergence in the
    // implicit solve accumulates.
    let no_gravity = IntegratorConfig::new(Vec3::ZERO, 16, 0.0, 0.0);
    let gyro1 = GyroscopicConfig::implicit(1);
    let (state, forces, torques) = gyroscopic_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &no_gravity,
        &gyro1,
        240,
        "implicit_gyro_1iter_no_gravity",
    );

    // Multiple Newton iterations exercise the inner solve loop more than once
    // per substep.
    let gyro3 = GyroscopicConfig::implicit(3);
    let (state, forces, torques) = gyroscopic_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &no_gravity,
        &gyro3,
        240,
        "implicit_gyro_3iter_no_gravity",
    );

    // Gravity + damping together drive the linear channel and the per-substep
    // velocity scales alongside the implicit angular solve.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.3, 0.15);
    let (state, forces, torques) = gyroscopic_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &damped,
        &gyro1,
        120,
        "implicit_gyro_damped_gravity",
    );

    // Explicit mode through the same entry point must match the explicit CPU
    // twin as well, confirming the mode switch does not perturb the base path.
    let explicit = GyroscopicConfig::explicit();
    let (state, forces, torques) = gyroscopic_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &no_gravity,
        &explicit,
        200,
        "explicit_through_gyro_entry",
    );

}
