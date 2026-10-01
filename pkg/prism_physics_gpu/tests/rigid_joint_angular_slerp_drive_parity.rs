//! Real-device parity for the angular `SLERP` drive joint stepper: the `GPU`
//! [`GpuAngularSlerpDriveJointSolver::solve_joints_angular_slerp_drive`] must
//! reproduce the `CPU` golden twin
//! [`cpu_solve_joints_angular_slerp_drive`](prism_physics_gpu::cpu_solve_joints_angular_slerp_drive),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise the single geodesic angular drive across its regimes:
//!
//! * a rigid (zero-compliance) attitude servo pulling a tilted body onto a fixed
//!   world target, where the drive is continuously active until it settles;
//! * a soft angular spring with damping relaxing onto the same target, where the
//!   compliance and the snapshot-based damping term both run;
//! * a torque-capped servo from a large error, where the clamp saturates the
//!   accumulated impulse every sweep;
//! * a rigid servo under gravity, where the bodies also translate freely so the
//!   linear predict/recover path runs alongside the angular drive; and
//! * a free-floating equal-mass pair sharing one translational velocity, where
//!   the drive's equal-and-opposite angular impulses leave linear momentum
//!   untouched on both engines.
//!
//! Provenance: the relative-orientation geodesic measurement and its substep
//! `XPBD` angular correction (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), with the bilateral compliant-and-damped drive and its
//! Macklin-style damping regularisation (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics") and the box-limited torque
//! cap, over the world-space inverse inertia and quaternion kinematics of Baraff
//! & Witkin. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_angular_slerp_drive, AngularSlerpDriveJoint, GpuAngularSlerpDriveJointSolver,
    GpuContext, IntegratorConfig, JointSolverConfig, RigidBodyState,
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
/// `position` with orientation `orientation`.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3, orientation: Quat) {
    state.push(position, orientation, 1.0, Vec3::splat(1.0));
}

/// Pushes an immovable body (zero inverse mass and inertia) at `position` with
/// identity orientation.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A static reference body `a` (index 0) at identity and a dynamic driven body
/// `b` (index 1) starting at `start`, both at the origin. The drive servos `b`
/// toward the world target `target` (expressed as the relative rotation in
/// `a`'s identity frame).
fn servo_pair(
    start: Quat,
    joint: AngularSlerpDriveJoint,
) -> (RigidBodyState, Vec<AngularSlerpDriveJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: static reference
    push_dynamic(&mut state, Vec3::ZERO, start); // 1: driven body
    (state, vec![joint])
}

/// A free-floating equal-mass pair, both dynamic, sharing one translational
/// velocity. Body `b` starts offset from `a`; the drive pulls their relative
/// orientation onto `target_rotation` with equal-and-opposite angular impulses,
/// so linear momentum is untouched and the shared drift is reproduced on both
/// engines.
fn free_pair(
    start_b: Quat,
    joint: AngularSlerpDriveJoint,
) -> (RigidBodyState, Vec<AngularSlerpDriveJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5, 0.0, 0.0), Quat::IDENTITY); // 0
    push_dynamic(&mut state, Vec3::new(0.5, 0.0, 0.0), start_b); // 1
    state.linear_velocities[0] = Vec3::new(0.2, 0.0, 0.1);
    state.linear_velocities[1] = Vec3::new(0.2, 0.0, 0.1);
    (state, vec![joint])
}

/// Runs `frames` of both engines from the same initial state under the same
/// joint set and checks per-frame parity.
#[expect(
    clippy::too_many_arguments,
    reason = "a parity run is parameterised by its full scene"
)]
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuAngularSlerpDriveJointSolver,
    initial: &RigidBodyState,
    joints: &[AngularSlerpDriveJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_angular_slerp_drive(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_angular_slerp_drive(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_angular_slerp_drive_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU angular SLERP drive joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuAngularSlerpDriveJointSolver::new(&ctx);

    let still = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    let four_sweeps = JointSolverConfig::new(4);
    let eight_sweeps = JointSolverConfig::new(8);

    // Rigid attitude servo: a tilted body is pulled onto a fixed world target
    // 0.8 rad about a diagonal axis. The drive is continuously active until it
    // settles, exercising the geodesic error measurement and the rigid update.
    let target = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0).normalize(), 0.8);
    let (state, joints) = servo_pair(
        Quat::IDENTITY,
        AngularSlerpDriveJoint::servo(0, 1, target, 1.0e4),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &eight_sweeps,
        120,
        "rigid_servo_converges",
    );

    // Soft angular spring with damping relaxing onto the same target from a 1.0
    // rad error: the compliance softens the drive and the snapshot-based damping
    // term runs, the mode most sensitive to CPU/GPU rounding in the kinematics.
    let target = Quat::from_axis_angle(Vec3::Z, 0.5);
    let (state, joints) = servo_pair(
        Quat::IDENTITY,
        AngularSlerpDriveJoint::drive(0, 1, target, 30.0, 0.5, 0.0),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &eight_sweeps,
        120,
        "soft_spring_with_damping",
    );

    // Torque-capped servo from a large 1.2 rad error with a tight cap: the clamp
    // saturates the accumulated impulse every sweep, so the box-limited step is
    // exercised on both engines.
    let target = Quat::from_axis_angle(Vec3::X, 1.2);
    let (state, joints) = servo_pair(
        Quat::IDENTITY,
        AngularSlerpDriveJoint::servo(0, 1, target, 0.5),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        90,
        "torque_capped_saturates",
    );

    // Rigid servo under gravity: the driven body also falls freely, so the
    // linear predict/recover path runs alongside the angular drive and must stay
    // in lock-step on both engines.
    let target = Quat::from_axis_angle(Vec3::Y, 0.6);
    let (state, joints) = servo_pair(
        Quat::IDENTITY,
        AngularSlerpDriveJoint::servo(0, 1, target, 1.0e4),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &eight_sweeps,
        90,
        "servo_under_gravity",
    );

    // Free-floating equal-mass pair sharing one translational velocity, driven
    // toward a 0.5 rad relative target: both bodies are dynamic, so the
    // equal-and-opposite angular impulses leave the shared linear momentum
    // untouched, and the shared drift is reproduced identically on both engines.
    let start_b = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 1.0).normalize(), 0.2);
    let (state, joints) = free_pair(
        start_b,
        AngularSlerpDriveJoint::drive(0, 1, Quat::IDENTITY, 20.0, 1.0, 0.0),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        90,
        "free_pair_conserves_momentum",
    );
}
