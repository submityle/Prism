//! Real-device parity for the revolute (hinge) angular velocity-motor joint
//! stepper: the `GPU`
//! [`GpuRevoluteMotorJointSolver::solve_joints_revolute_motor`] must reproduce
//! the `CPU` golden twin
//! [`cpu_solve_joints_revolute_motor`](prism_physics_gpu::cpu_solve_joints_revolute_motor),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise all three constraints — the hinge axis alignment, the
//! point-to-point weld, and the about-axis velocity motor — in both regimes of
//! the motor: the rigid (`motor_compliance = 0`) rate-lock that forces the spin
//! onto its commanded angular velocity each sweep, and the soft compliant motor
//! whose `alpha_tilde = motor_compliance / h^2` regularisation applies a finite
//! torque so the hinge approaches the target rate asymptotically. The motor's
//! relative-displacement term reads the pre-substep snapshot
//! (`prev_orientations`), so the soft-motor scene keeps that term continuously
//! active and any `CPU` / `GPU` divergence in the snapshot-relative hinge rate
//! accumulates past the tolerance rather than sliding under it. A free-floating
//! equal-mass pair confirms the motor's internal corrections stay equal and
//! opposite (linear momentum conserved) identically on both engines.
//!
//! Provenance: the point-to-point (ball-socket) and hinge axis-alignment
//! constraints with their substep `XPBD` handling, and the velocity-level motor
//! expressed as a per-substep compliant equality on the relative angular
//! displacement about the hinge axis (Müller et al., "Detailed Rigid Body
//! Simulation with XPBD"; Macklin et al., "XPBD: Position-Based Simulation of
//! Compliant Constrained Dynamics"). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_revolute_motor, GpuContext, GpuRevoluteMotorJointSolver, IntegratorConfig,
    JointSolverConfig, RevoluteMotorJoint, RigidBodyState,
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

/// Pushes a dynamic unit body (unit inverse mass, unit inverse inertia) at
/// `position` with identity orientation.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
}

/// Pushes an immovable body (zero inverse mass and inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A dynamic rotor hinged to a static housing about the world `+Z` axis, their
/// anchors coincident at the origin, driven by a rigid velocity motor onto
/// `target_velocity`.
///
/// Body `a` (index 0) is the static housing at the origin; body `b` (index 1)
/// is the dynamic rotor. A rigid motor forces the relative hinge rate exactly
/// onto the target each substep, so the rotor spins up to the commanded angular
/// velocity and holds it.
fn rigid_rotor_scene(target_velocity: f32) -> (RigidBodyState, Vec<RevoluteMotorJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: housing
    push_dynamic(&mut state, Vec3::ZERO); // 1: rotor

    let joints = vec![RevoluteMotorJoint::rigid(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Z,
        Vec3::Z,
        target_velocity,
    )];
    (state, joints)
}

/// The same housing-and-rotor hinge driven by a soft velocity motor of the
/// given torque stiffness onto `target_velocity`, so the finite-torque motor's
/// snapshot-relative rate term is continuously active while the rotor spins up
/// asymptotically.
fn soft_rotor_scene(
    target_velocity: f32,
    stiffness: f32,
) -> (RigidBodyState, Vec<RevoluteMotorJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: housing
    push_dynamic(&mut state, Vec3::ZERO); // 1: rotor

    let joints = vec![RevoluteMotorJoint::soft(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Z,
        Vec3::Z,
        target_velocity,
        stiffness,
    )];
    (state, joints)
}

/// A free-floating equal-mass pair hinged about the `+Y` axis with both bodies
/// sharing one translational velocity and a rigid velocity motor commanding the
/// hinge rate: every correction is internal, so total linear momentum stays
/// conserved on both engines.
fn free_pair_scene(target_velocity: f32) -> (RigidBodyState, Vec<RevoluteMotorJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1
                                                        // Both bodies share one translational velocity: the weld carries them
                                                        // together with no induced rigid tumbling, so the only rotational mode is
                                                        // the motor's own counter-rotation toward the target rate.
    state.linear_velocities[0] = Vec3::new(0.2, 0.1, 0.0);
    state.linear_velocities[1] = Vec3::new(0.2, 0.1, 0.0);

    let joints = vec![RevoluteMotorJoint::soft(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::Y,
        Vec3::Y,
        target_velocity,
        100.0,
    )];
    (state, joints)
}

/// Runs `frames` of both engines from the same initial state under the same
/// joint set and checks per-frame parity.
#[expect(
    clippy::too_many_arguments,
    reason = "a parity run is parameterised by its full scene"
)]
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuRevoluteMotorJointSolver,
    initial: &RigidBodyState,
    joints: &[RevoluteMotorJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_revolute_motor(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_revolute_motor(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_revolute_motor_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU revolute-motor joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuRevoluteMotorJointSolver::new(&ctx);

    // Zero gravity throughout: the motor itself supplies all the forcing, so the
    // scenes isolate the velocity-motor path from any gravitational droop.
    let still = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
    let still_free = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let two_sweeps = JointSolverConfig::new(2);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid motor spinning a rotor against a static housing onto a positive
    // target rate: the zero-compliance (`d_lambda = -C / w`) rate-lock branch is
    // active every sweep as the rotor holds the commanded angular velocity.
    let (state, joints) = rigid_rotor_scene(3.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        120,
        "rigid_motor_holds_target_rate",
    );

    // Soft compliant motor spinning up asymptotically onto its target rate: the
    // regularised motor and its snapshot-relative rate term are continuously
    // active, so this is the scene that stresses the `prev_orientations` path
    // shared by CPU and GPU. Four substeps (not eight) keep the first-order
    // velocity recovery `omega = 2 * inv_h * delta` well-conditioned: at eight
    // substeps `inv_h` doubles and amplifies the sub-microradian per-substep
    // orientation rounding gap between the two backends past the parity floor,
    // even though the physics is identical. The soft regime (finite-torque
    // asymptotic spin-up, continuous snapshot-relative term) is unchanged.
    let (state, joints) = soft_rotor_scene(2.5, 60.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still_free,
        &four_sweeps,
        180,
        "soft_motor_spins_up_to_target",
    );

    // Free-floating equal-mass pair with an initial velocity and a motor
    // commanding the hinge rate: both bodies are dynamic, so the
    // equal-and-opposite motor corrections must conserve linear momentum
    // identically on both engines across the window.
    let (state, joints) = free_pair_scene(0.8);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still_free,
        &two_sweeps,
        90,
        "free_pair_conserves_momentum",
    );
}
