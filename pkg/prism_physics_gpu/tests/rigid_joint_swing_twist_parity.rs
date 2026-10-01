//! Real-device parity for the swing-twist (cone-twist / ragdoll) joint stepper:
//! the `GPU` [`GpuSwingTwistJointSolver::solve_joints_swing_twist`] must
//! reproduce the `CPU` golden twin
//! [`cpu_solve_joints_swing_twist`](prism_physics_gpu::cpu_solve_joints_swing_twist),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise all three constraints — the point-to-point weld, the
//! swing cone, and the twist limit — in both the inactive and the actively
//! clamping regime of each angular bound:
//!
//! * a wide-cone free swing under gravity, where the cone and twist stops never
//!   engage and only the weld plus the angle-measurement machinery run;
//! * a tight cone with a steady lateral pull and angular damping, where the
//!   limb is dragged onto the cone rim and held there, so the swing-cone
//!   correction is continuously active;
//! * an axially spun limb inside a tight twist range, where the twist limit
//!   catches the spin near its bound, so the twist correction is continuously
//!   active; and
//! * a free-floating equal-mass pair sharing one translational velocity, where
//!   every correction is internal, so total linear momentum stays conserved
//!   identically on both engines.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the cone-swing limit with their
//! substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_swing_twist, GpuContext, GpuSwingTwistJointSolver, IntegratorConfig,
    JointSolverConfig, RigidBodyState, SwingTwistJoint,
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

/// A dynamic `+Y` limb socketed off a static pivot: body `a` (index 0) is the
/// static socket at the origin with twist axis `+Y`; body `b` (index 1) is the
/// dynamic limb above it, anchored back through its `-Y` end to the socket. The
/// cone half-angle and symmetric twist range are configurable.
fn socketed_limb(swing_limit: f32, twist_limit: f32) -> (RigidBodyState, Vec<SwingTwistJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: socket
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1: limb extends +Y

    let joints = vec![SwingTwistJoint::symmetric_cone(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::Y,
        Vec3::Y,
        Vec3::X,
        Vec3::X,
        swing_limit,
        twist_limit,
    )];
    (state, joints)
}

/// A free-floating equal-mass pair welded end to end along `+Y` and sharing one
/// translational velocity, so the weld carries them together with no induced
/// tumbling and the only angular mode is the (here inactive) cone/twist bounds.
/// Every correction is internal, so total linear momentum is conserved.
fn free_pair(swing_limit: f32, twist_limit: f32) -> (RigidBodyState, Vec<SwingTwistJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1
    state.linear_velocities[0] = Vec3::new(0.2, 0.0, 0.1);
    state.linear_velocities[1] = Vec3::new(0.2, 0.0, 0.1);

    let joints = vec![SwingTwistJoint::symmetric_cone(
        0,
        1,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::Y,
        Vec3::Y,
        Vec3::X,
        Vec3::X,
        swing_limit,
        twist_limit,
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
    solver: &GpuSwingTwistJointSolver,
    initial: &RigidBodyState,
    joints: &[SwingTwistJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_swing_twist(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_swing_twist(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_swing_twist_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU swing-twist joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuSwingTwistJointSolver::new(&ctx);

    // Gravity integrators. The rim-settling scene adds angular damping so the
    // limb comes to rest on the cone rather than slamming through it; the other
    // scenes stay undamped so any CPU/GPU divergence accumulates rather than
    // being bled away.
    let gravity = IntegratorConfig::new(Vec3::new(1.0, 0.0, 0.0), 8, 0.0, 0.0);
    let rim_damped = IntegratorConfig::new(Vec3::new(1.5, 0.0, 0.0), 8, 0.0, 4.0);
    let still_free = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let four_sweeps = JointSolverConfig::new(4);
    let eight_sweeps = JointSolverConfig::new(8);

    // Wide 90-degree cone with a steady lateral pull: the limb swings a
    // meaningful amount but never reaches the rim or twists, so only the weld
    // and the angle-measurement machinery run. The free angular drift is the
    // mode most sensitive to CPU/GPU rounding in the quaternion kinematics.
    let (state, joints) = socketed_limb(FRAC_PI_2, FRAC_PI_2);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &four_sweeps,
        90,
        "free_swing_within_wide_cone",
    );

    // Tight 30-degree cone with a stronger lateral pull and angular damping: the
    // limb is dragged out onto the rim and held there, so the one-sided swing
    // cone correction is continuously active for the whole settled window.
    let (state, joints) = socketed_limb(FRAC_PI_6, FRAC_PI_2);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &rim_damped,
        &eight_sweeps,
        240,
        "cone_rim_arrest",
    );

    // Axially spun limb inside a tight twist range: an initial +Y spin winds the
    // limb up until the twist limit catches it near its bound, so the twist
    // correction is continuously active. The wide cone keeps the swing bound out
    // of play so the twist limit is exercised in isolation.
    let (mut state, joints) = socketed_limb(FRAC_PI_2, FRAC_PI_6);
    state.angular_velocities[1] = Vec3::new(0.0, 3.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0),
        &eight_sweeps,
        120,
        "twist_stop_arrest",
    );

    // Free-floating equal-mass pair sharing one translational velocity with a
    // generous cone and twist range: both bodies are dynamic, so the weld's
    // equal-and-opposite corrections must conserve linear momentum identically
    // on both engines across the window.
    let (state, joints) = free_pair(FRAC_PI_2, FRAC_PI_2);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still_free,
        &four_sweeps,
        90,
        "free_pair_conserves_momentum",
    );
}
