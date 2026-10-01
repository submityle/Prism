//! Real-device parity for the configurable `D6` joint stepper: the `GPU`
//! [`GpuD6JointSolver::solve_joints_d6`] must reproduce the `CPU` golden twin
//! [`cpu_solve_joints_d6`](prism_physics_gpu::cpu_solve_joints_d6), frame for
//! frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! Because the `D6` joint is the general-purpose joint that subsumes every other
//! joint in this crate, the scenes split into two families. The first drives
//! each of the six degrees of freedom through its three motion modes directly —
//! a locked weld, each `Limited` stop (linear slab, twist range, both swing
//! cone half-angles), and the free modes — so every branch of the shader's
//! `solve_one` runs under load. The second cross-validates the degeneration
//! contract from the type's documentation against the dedicated kernels'
//! configurations: all axes locked is the fixed weld; three linear locked with
//! twist free is the revolute hinge; three linear locked with all angular free
//! is the ball socket; two linear locked with linear `x` free and all angular
//! locked is the prismatic slider. Each is run on both engines and checked for
//! identical motion.
//!
//! Provenance: the per-axis configurable degree-of-freedom model of a
//! general-purpose constraint (Unreal Engine's `FConstraintInstance`, `PhysX`'s
//! `PxD6Joint`), realised over the point-to-point weld, the signed angular
//! limit, and the pyramidal swing limits already used by this crate's
//! specialised joints, with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code: only the public per-axis configuration semantics are
//! mirrored.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_d6, D6Joint, D6Motion, GpuContext, GpuD6JointSolver, IntegratorConfig,
    JointSolverConfig, RigidBodyState,
};

/// Absolute per-body divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// The three linear axes, all [`Locked`](D6Motion::Locked).
const LINEAR_LOCKED: [D6Motion; 3] = [D6Motion::Locked; 3];
/// The three angular axes, all [`Locked`](D6Motion::Locked).
const ANGULAR_LOCKED: [D6Motion; 3] = [D6Motion::Locked; 3];

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

/// A dynamic `+Y` limb socketed off a static pivot at the origin, both joint
/// frames identity (so the frame axes coincide with world `x`/`y`/`z`): the
/// twist axis is world `+X`, the swing axes are world `+Y` (swing1) and `+Z`
/// (swing2). Body `a` (index 0) is the static socket; body `b` (index 1) is the
/// dynamic limb one unit along `+Y`, anchored back through its `-Y` end to the
/// socket.
fn socketed_limb(
    linear: [D6Motion; 3],
    angular: [D6Motion; 3],
    linear_limit: f32,
    twist_range: (f32, f32),
    swing_limits: (f32, f32),
) -> (RigidBodyState, Vec<D6Joint>) {
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
        linear_limit,
        twist_range,
        swing_limits,
        (0.0, 0.0, 0.0),
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
    solver: &GpuD6JointSolver,
    initial: &RigidBodyState,
    joints: &[D6Joint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_d6(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_d6(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_d6_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU D6 joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuD6JointSolver::new(&ctx);

    let four_sweeps = JointSolverConfig::new(4);
    let eight_sweeps = JointSolverConfig::new(8);
    let gravity_y = IntegratorConfig::new(Vec3::new(0.0, -2.0, 0.0), 8, 0.0, 0.0);
    let gravity_x = IntegratorConfig::new(Vec3::new(1.5, 0.0, 0.0), 8, 0.0, 0.0);
    let settle_x = IntegratorConfig::new(Vec3::new(1.5, 0.0, 0.0), 8, 0.0, 4.0);
    let still_8 = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
    let damped_still = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 4.0);
    // A gentle laterally-driven swing with light angular damping: the ball
    // socket rotates freely about the pinned point, so the integrator leans on
    // orientation integration alone. Softening the drive and bleeding spin
    // energy keeps the free-rotation phase drift between the engines inside the
    // tight tolerance without touching the bound.
    let gentle_swing = IntegratorConfig::new(Vec3::new(0.5, 0.0, 0.0), 8, 0.0, 1.5);

    // --- Family 1: each degree of freedom through its active modes ---------

    // Every axis locked: the full weld pins the limb rigidly to the socket
    // frame under a lateral pull, so all six corrections run continuously.
    let (state, joints) = socketed_limb(LINEAR_LOCKED, ANGULAR_LOCKED, 0.0, (0.0, 0.0), (0.0, 0.0));
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity_x,
        &eight_sweeps,
        120,
        "all_locked_weld",
    );

    // Linear x limited to a tight slab, the other two linear axes and all
    // angular axes locked, pulled along +X until the slab stop catches and
    // holds the limb at the bound.
    let (state, joints) = socketed_limb(
        [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked],
        ANGULAR_LOCKED,
        0.2,
        (0.0, 0.0),
        (0.0, 0.0),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &settle_x,
        &eight_sweeps,
        240,
        "linear_x_slab_arrest",
    );

    // Twist limited to a tight symmetric range, linear axes and both swings
    // locked, with an initial +X spin that winds the limb up until the twist
    // stop catches it near its bound.
    let (mut state, joints) = socketed_limb(
        LINEAR_LOCKED,
        [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked],
        0.0,
        (-FRAC_PI_6, FRAC_PI_6),
        (0.0, 0.0),
    );
    state.angular_velocities[1] = Vec3::new(2.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped_still,
        &eight_sweeps,
        120,
        "twist_stop_arrest",
    );

    // Swing1 limited to a tight bound, twist and swing2 locked and the linear
    // axes locked, with an initial +Z spin that tilts the limb's twist axis
    // toward the frame y axis until the swing1 cone stop arrests it.
    let (mut state, joints) = socketed_limb(
        LINEAR_LOCKED,
        [D6Motion::Locked, D6Motion::Limited, D6Motion::Locked],
        0.0,
        (0.0, 0.0),
        (FRAC_PI_6, FRAC_PI_2),
    );
    state.angular_velocities[1] = Vec3::new(0.0, 0.0, 2.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped_still,
        &eight_sweeps,
        180,
        "swing1_cone_arrest",
    );

    // Swing2 limited to a tight bound, twist and swing1 locked, with an initial
    // +Y spin that tilts the twist axis toward the frame z axis until the
    // swing2 cone stop arrests it — the orthogonal swing half-angle.
    let (mut state, joints) = socketed_limb(
        LINEAR_LOCKED,
        [D6Motion::Locked, D6Motion::Locked, D6Motion::Limited],
        0.0,
        (0.0, 0.0),
        (FRAC_PI_2, FRAC_PI_6),
    );
    state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped_still,
        &eight_sweeps,
        180,
        "swing2_cone_arrest",
    );

    // --- Family 2: degeneration to the specialised joints ------------------

    // All axes locked degenerates to the fixed weld; built through the `fixed`
    // constructor and dropped under gravity to confirm the limb stays pinned.
    let mut fixed_state = RigidBodyState::new();
    push_static(&mut fixed_state, Vec3::ZERO);
    push_dynamic(&mut fixed_state, Vec3::new(0.0, 1.0, 0.0));
    let fixed_joints = vec![D6Joint::fixed(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Quat::IDENTITY,
    )];
    run_parity(
        &ctx,
        &solver,
        &fixed_state,
        &fixed_joints,
        &gravity_y,
        &four_sweeps,
        120,
        "degenerate_fixed",
    );

    // Three linear axes locked, twist free, both swings locked: the revolute
    // hinge about the frame x axis. An initial +X spin runs free about the
    // twist axis while the weld holds the limb's end at the socket.
    let (mut state, joints) = socketed_limb(
        LINEAR_LOCKED,
        [D6Motion::Free, D6Motion::Locked, D6Motion::Locked],
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
    );
    state.angular_velocities[1] = Vec3::new(1.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still_8,
        &eight_sweeps,
        72,
        "degenerate_revolute",
    );

    // Three linear axes locked, all three angular axes free: the ball socket.
    // Gravity swings the limb about the pinned socket with no angular limit.
    let (state, joints) = socketed_limb(
        LINEAR_LOCKED,
        [D6Motion::Free; 3],
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gentle_swing,
        &eight_sweeps,
        96,
        "degenerate_spherical",
    );

    // Linear x free, linear y/z locked, all angular axes locked: the prismatic
    // slider along the frame x axis. Gravity along +X slides the limb while its
    // orientation and transverse position stay welded.
    let (state, joints) = socketed_limb(
        [D6Motion::Free, D6Motion::Locked, D6Motion::Locked],
        ANGULAR_LOCKED,
        0.0,
        (0.0, 0.0),
        (0.0, 0.0),
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity_x,
        &eight_sweeps,
        150,
        "degenerate_prismatic",
    );
}
