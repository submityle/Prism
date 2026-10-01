//! Real-device parity for the universal (Cardan / Hooke) joint stepper: the
//! `GPU` [`GpuUniversalJointSolver::solve_joints_universal`] must reproduce the
//! `CPU` golden twin
//! [`cpu_solve_joints_universal`](prism_physics_gpu::cpu_solve_joints_universal),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise both of the joint's constraints — the point-to-point
//! weld and the perpendicularity (orthogonality) constraint — in both the
//! settling and the continuously-corrected regime of the gimbal:
//!
//! * a gimbal whose two drive axes start `45` degrees from perpendicular with no
//!   gravity, so the two-sided perpendicularity constraint fires hard and keeps
//!   correcting as the axis swings back toward and past the right angle — the
//!   regime most sensitive to `CPU`/`GPU` rounding in the `acos`/cross
//!   kinematics, and the guard against a swing-cone-style free-zone miss;
//! * a dynamic shaft gimbaled off a static mount under gravity, where the weld
//!   holds the pivot and the perpendicularity constraint continuously resists
//!   the gravity-induced torque that tries to tilt the drive axis off the right
//!   angle; and
//! * a free-floating equal-mass pair sharing one translational velocity, where
//!   every correction is internal, so total linear momentum stays conserved
//!   identically on both engines.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! perpendicularity (orthogonality) constraint over two body-fixed axes with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use std::f32::consts::FRAC_PI_4;

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_universal, GpuContext, GpuUniversalJointSolver, IntegratorConfig,
    JointSolverConfig, RigidBodyState, UniversalJoint,
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
/// `position` with the given orientation.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3, orientation: Quat) {
    state.push(position, orientation, 1.0, Vec3::splat(1.0));
}

/// Pushes a static body (zero inverse mass and inverse inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A dynamic shaft gimbaled off a static mount at the origin. The mount's drive
/// axis is `+X` and the shaft's body-local drive axis is `+Z`; the shaft's
/// initial orientation is `tilt`, so a non-identity tilt starts the two drive
/// axes off the right angle and arms the perpendicularity constraint. The
/// anchors coincide at the origin.
fn gimbaled_shaft(tilt: Quat) -> (RigidBodyState, Vec<UniversalJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: mount
    push_dynamic(&mut state, Vec3::new(0.0, -1.0, 0.0), tilt); // 1: shaft hangs -Y

    let joints = vec![UniversalJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::X,
        Vec3::Z,
        0.0,
        0.0,
    )];
    (state, joints)
}

/// A free-floating equal-mass pair welded end to end along `+X`, sharing one
/// translational velocity. Both bodies are dynamic, so the joint's internal
/// corrections are equal and opposite and total linear momentum is conserved.
fn free_pair() -> (RigidBodyState, Vec<UniversalJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO, Quat::IDENTITY); // 0
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY); // 1
    state.linear_velocities[0] = Vec3::new(0.2, 0.1, -0.1);
    state.linear_velocities[1] = Vec3::new(0.2, 0.1, -0.1);

    let joints = vec![UniversalJoint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::X,
        Vec3::Z,
        0.0,
        0.0,
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
    solver: &GpuUniversalJointSolver,
    initial: &RigidBodyState,
    joints: &[UniversalJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_universal(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_universal(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_universal_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU universal joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuUniversalJointSolver::new(&ctx);

    let four_sweeps = JointSolverConfig::new(4);
    let eight_sweeps = JointSolverConfig::new(8);

    // Gimbal whose drive axes start 45 degrees from perpendicular with no
    // gravity: the two-sided perpendicularity constraint fires hard from the
    // first sweep and keeps correcting as the shaft's drive axis swings back
    // toward the right angle. This is the regime most sensitive to CPU/GPU
    // rounding in the acos/cross kinematics and the guard against a free-zone
    // miss.
    let (state, joints) = gimbaled_shaft(Quat::from_axis_angle(Vec3::Y, FRAC_PI_4));
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0),
        &eight_sweeps,
        90,
        "perpendicular_arrest_from_45_degrees",
    );

    // Dynamic shaft gimbaled off a static mount under gravity with angular
    // damping: the weld holds the pivot while the perpendicularity constraint
    // continuously resists the gravity-induced torque that tilts the drive axis
    // off the right angle. The shaft starts slightly tilted so the constraint is
    // active from the outset.
    let (state, joints) = gimbaled_shaft(Quat::from_axis_angle(Vec3::Y, FRAC_PI_4 * 0.5));
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 2.0),
        &eight_sweeps,
        180,
        "gimbaled_shaft_under_gravity",
    );

    // Free-floating equal-mass pair sharing one translational velocity: both
    // bodies are dynamic, so the weld's and perpendicularity constraint's
    // equal-and-opposite corrections must conserve linear momentum identically
    // on both engines across the window.
    let (state, joints) = free_pair();
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0),
        &four_sweeps,
        90,
        "free_pair_conserves_momentum",
    );
}
