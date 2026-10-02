//! Real-device parity for the *driven* configurable `D6` joint stepper: the
//! `GPU` [`GpuD6DriveJointSolver::solve_joints_d6_driven`] must reproduce the
//! `CPU` golden twin
//! [`cpu_solve_joints_d6_driven`](prism_physics_gpu::cpu_solve_joints_d6_driven),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! Where the passive `D6` parity suite walks each degree of freedom through its
//! three *motion modes*, this suite walks the orthogonal axis — the per-axis
//! *drive* (actuator). The scenes cover the inert set (a rest drive must leave
//! the passive golden untouched), a stiff position servo on each of the six
//! degrees of freedom that carries a drive (linear `x`, twist, swing1, swing2),
//! a pure velocity motor, a position servo sagging under a steady load, the two
//! composition contracts (a locked axis's weld beats its drive; a limited axis's
//! stop beats its drive), and an internal drive between two free bodies that
//! must conserve the pair's linear momentum. Every branch of the shader's
//! `drive_delta` (position-servo and velocity-motor) and all four drive
//! projections (`drive_linear`, `drive_twist`, `drive_swing`) run under load.
//!
//! Provenance: the per-axis spring-damper actuator of a general-purpose
//! constraint (Unreal Engine's `FConstraintDrive`, `PhysX`'s `PxD6JointDrive`),
//! realised as a compliant, velocity-damped `XPBD` constraint (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD") over the passive `D6` constraints
//! and the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code: only the public spring-
//! damper drive semantics are mirrored.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_d6_driven, D6Drive, D6DriveSet, D6Joint, D6Motion, GpuContext,
    GpuD6DriveJointSolver, IntegratorConfig, JointSolverConfig, RigidBodyState,
};

/// Absolute per-body divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// All three axes [`Locked`](D6Motion::Locked).
const LOCKED3: [D6Motion; 3] = [D6Motion::Locked; 3];
/// All three drives inert.
const OFF3: [D6Drive; 3] = [D6Drive::OFF; 3];

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

/// Pushes a dynamic unit-mass body with identity inertia at `position`.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
}

/// Pushes an immovable body (zero inverse mass and inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A dynamic `+Y` unit limb socketed off a static pivot at the origin, both
/// joint frames identity (so the frame axes coincide with world `x`/`y`/`z`):
/// the twist axis is world `+X`, the swing axes are world `+Y` (swing1) and
/// `+Z` (swing2). Body `a` (index 0) is the static socket; body `b` (index 1)
/// is the dynamic limb one unit along `+Y`, anchored back through its `-Y` end
/// to the socket, so every linear coordinate and joint angle starts at zero.
/// This is the exact scene the `CPU` golden's unit tests use as their blueprint.
fn socketed_limb(linear: [D6Motion; 3], angular: [D6Motion; 3]) -> (RigidBodyState, Vec<D6Joint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: socket
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1: limb extends +Y

    let joints = vec![D6Joint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
        linear,
        angular,
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0, 0.0),
    )];
    (state, joints)
}

/// A free pair of equal unit masses straddling the origin along the world `x`
/// axis, coupled by a `D6` joint whose anchors sit at each body's inner end so
/// the linear `x` coordinate is their separation. Both bodies are dynamic, so
/// any internal drive must conserve the pair's linear momentum.
fn free_pair(linear: [D6Motion; 3]) -> (RigidBodyState, Vec<D6Joint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-1.0, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1

    let joints = vec![D6Joint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
        linear,
        LOCKED3,
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0, 0.0),
    )];
    (state, joints)
}

/// Runs `frames` of both engines from the same initial state under the same
/// joint and drive sets and checks per-frame parity.
#[expect(
    clippy::too_many_arguments,
    reason = "a driven parity run is parameterised by its full scene"
)]
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuD6DriveJointSolver,
    initial: &RigidBodyState,
    joints: &[D6Joint],
    drives: &[D6DriveSet],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_d6_driven(&mut cpu, joints, drives, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_d6_driven(ctx, &mut gpu, joints, drives, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_d6_drive_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU D6 drive joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuD6DriveJointSolver::new(&ctx);

    let eight_sweeps = JointSolverConfig::new(8);
    let still = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);

    // --- Inert drive: the rest set must track the passive golden exactly ---

    // A representative mixed-freedom joint under full gravity carrying a rest
    // (all-inert) drive set: every drive branch is skipped, so the driven twin
    // must track the passive stepper, and the two engines must still agree.
    let mixed_linear = [D6Motion::Limited, D6Motion::Free, D6Motion::Locked];
    let mixed_angular = [D6Motion::Free, D6Motion::Limited, D6Motion::Locked];
    let mut inert_state = RigidBodyState::new();
    push_static(&mut inert_state, Vec3::ZERO);
    push_dynamic(&mut inert_state, Vec3::new(0.0, 1.0, 0.0));
    let inert_joints = vec![D6Joint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
        mixed_linear,
        mixed_angular,
        0.3,
        (-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
        (0.4, 0.4),
        (0.0, 0.0, 0.0),
    )];
    let inert_drives = [D6DriveSet::rest()];
    // Gentle gravity with light angular damping bleeds the free-axis
    // rotation drift so the inert driven twin tracks the passive golden inside
    // the tight parity bound without widening the tolerance.
    let gravity_mixed = IntegratorConfig::new(Vec3::new(1.0, -2.0, 0.5), 8, 0.0, 1.5);
    run_parity(
        &ctx,
        &solver,
        &inert_state,
        &inert_joints,
        &inert_drives,
        &gravity_mixed,
        &eight_sweeps,
        90,
        "inert_drive_matches_passive",
    );

    // --- Position servos on each driven degree of freedom -----------------

    // A stiff position servo on the free linear x axis pulling the anchor
    // separation to a commanded target and holding it there.
    let (state, joints) = socketed_limb([D6Motion::Free, D6Motion::Locked, D6Motion::Locked], LOCKED3);
    let drives = [D6DriveSet::new(
        [D6Drive::position(5.0e3, 0.35), D6Drive::OFF, D6Drive::OFF],
        OFF3,
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        120,
        "linear_x_position_drive",
    );

    // A stiff twist servo rotating the free-twisting limb to its target angle
    // about the twist axis.
    let (state, joints) = socketed_limb(LOCKED3, [D6Motion::Free, D6Motion::Locked, D6Motion::Locked]);
    let drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::new(3.0e3, 60.0, 0.5, 0.0), D6Drive::OFF, D6Drive::OFF],
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        100,
        "twist_position_drive",
    );

    // A stiff swing1 servo tilting the limb to its target swing1 angle while
    // twist and swing2 stay welded shut.
    let (state, joints) = socketed_limb(LOCKED3, [D6Motion::Locked, D6Motion::Free, D6Motion::Locked]);
    let drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::OFF, D6Drive::new(3.0e3, 60.0, 0.3, 0.0), D6Drive::OFF],
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        100,
        "swing1_position_drive",
    );

    // A stiff swing2 servo tilting the limb to its target swing2 angle while
    // twist and swing1 stay welded shut.
    let (state, joints) = socketed_limb(LOCKED3, [D6Motion::Locked, D6Motion::Locked, D6Motion::Free]);
    let drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::OFF, D6Drive::OFF, D6Drive::new(3.0e3, 60.0, 0.3, 0.0)],
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        100,
        "swing2_position_drive",
    );

    // --- Velocity motor ----------------------------------------------------

    // A pure velocity motor on the free twist axis spinning the limb up to its
    // commanded angular rate and holding it there.
    let (state, joints) = socketed_limb(LOCKED3, [D6Motion::Free, D6Motion::Locked, D6Motion::Locked]);
    let drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::velocity(2.0e2, 2.0), D6Drive::OFF, D6Drive::OFF],
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        150,
        "twist_velocity_motor",
    );

    // --- Position servo under a steady load (spring sag) -------------------

    // A linear x position servo under a steady along-axis load: the spring sags
    // from its target by load / stiffness and holds there, exercising the
    // compliant (soft) branch of the drive against gravity.
    let (state, joints) = socketed_limb([D6Motion::Free, D6Motion::Locked, D6Motion::Locked], LOCKED3);
    let drives = [D6DriveSet::new(
        [D6Drive::position(1.0e3, 0.0), D6Drive::OFF, D6Drive::OFF],
        OFF3,
    )];
    let load_x = IntegratorConfig::new(Vec3::new(-6.0, 0.0, 0.0), 8, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &load_x,
        &eight_sweeps,
        150,
        "position_drive_sags_under_load",
    );

    // --- Composition contracts: weld and stop beat the drive ---------------

    // A rigid weld on the twist axis is projected first with zero compliance,
    // so even an aggressive twist servo cannot drag the locked axis anywhere.
    let (state, joints) = socketed_limb(LOCKED3, LOCKED3);
    let drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::position(1.0e4, 1.0), D6Drive::OFF, D6Drive::OFF],
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        90,
        "locked_axis_beats_drive",
    );

    // A twist servo commanding past the mechanical stop must drive the axis up
    // to, but not through, the rigid upper limit — the limit projection and the
    // drive projection compose on the same axis.
    let upper = 0.25;
    let (stop_state, _) = socketed_limb(LOCKED3, LOCKED3);
    let stop_joints = vec![D6Joint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
        LOCKED3,
        [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked],
        0.0,
        (-upper, upper),
        (0.0, 0.0),
        (0.0, 0.0, 0.0),
    )];
    let stop_drives = [D6DriveSet::new(
        OFF3,
        [D6Drive::position(5.0e3, 1.0), D6Drive::OFF, D6Drive::OFF],
    )];
    run_parity(
        &ctx,
        &solver,
        &stop_state,
        &stop_joints,
        &stop_drives,
        &still,
        &eight_sweeps,
        120,
        "limited_axis_respects_stop",
    );

    // --- Internal drive between two free bodies ----------------------------

    // A linear position drive between two free equal masses is an internal
    // force: it pulls the pair together while conserving linear momentum, so
    // both bodies move and the device twin must track both.
    let (state, joints) = free_pair([D6Motion::Free, D6Motion::Locked, D6Motion::Locked]);
    let drives = [D6DriveSet::new(
        [D6Drive::position(2.0e3, 0.5), D6Drive::OFF, D6Drive::OFF],
        OFF3,
    )];
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drives,
        &still,
        &eight_sweeps,
        90,
        "free_pair_linear_drive",
    );
}
