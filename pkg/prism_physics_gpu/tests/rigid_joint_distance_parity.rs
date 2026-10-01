//! Real-device parity for the distance (limit) joint stepper: the `GPU`
//! [`GpuDistanceJointSolver::solve_joints_distance`] must reproduce the `CPU`
//! golden twin
//! [`cpu_solve_joints_distance`](prism_physics_gpu::cpu_solve_joints_distance),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise every regime of the limit constraint: a rigid rod
//! pendulum (bilateral, `min == max`), a rope arresting a free fall (unilateral
//! upper limit crossing the dead zone into the active zone), a minimum-distance
//! strut pushing a close pair apart (unilateral lower limit), and a
//! free-floating pair stretched past its rod length in zero gravity. Running
//! many substeps per frame over many frames, with both the rigid
//! (`compliance = 0`) and soft (`compliance > 0`) regimes and more than one
//! projection sweep, lets any per-iteration divergence between the hand-written
//! `CPU` and shader arithmetic accumulate, so a kernel that only approximately
//! mirrored the `CPU` would drift past the tolerance rather than sliding under
//! it.
//!
//! Provenance: the point-to-point distance constraint and its substep `XPBD`
//! positional handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), with the one-sided limit / dead-zone treatment standard to distance
//! joints. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_distance, DistanceJoint, GpuContext, GpuDistanceJointSolver, IntegratorConfig,
    JointSolverConfig, RigidBodyState,
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

/// Pushes a dynamic unit body (unit inverse mass, unit diagonal inverse
/// inertia) at `position`.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 1.0, Vec3::ONE);
}

/// Pushes a static body (zero inverse mass and inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A rigid-rod pendulum: a dynamic body held exactly `length` from a static
/// pivot, kicked sideways so it swings. Centre-of-mass anchors keep the
/// geometry simple.
fn rod_pendulum_scene(length: f32, compliance: f32) -> (RigidBodyState, Vec<DistanceJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::new(length, 0.0, 0.0)); // 1: bob
    state.linear_velocities[1] = Vec3::new(0.0, 0.5, 0.0);

    let joints = vec![DistanceJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        length,
        length,
        compliance,
    )];
    (state, joints)
}

/// A rope arresting a free fall: a dynamic body starts coincident with a static
/// anchor and falls under gravity until the `max_length` rope snaps taut. This
/// exercises the dead-zone-to-active-zone transition of the upper limit.
fn rope_fall_scene(max_length: f32) -> (RigidBodyState, Vec<DistanceJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: anchor
    push_dynamic(&mut state, Vec3::new(0.0, -0.1, 0.0)); // 1: weight, slightly below

    let joints = vec![DistanceJoint::rope(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        max_length,
    )];
    (state, joints)
}

/// A minimum-distance strut: two dynamic bodies closer than the lower bound in
/// zero gravity, so the limit pushes them apart. Exercises the lower-limit sign.
fn min_strut_scene(min_distance: f32, compliance: f32) -> (RigidBodyState, Vec<DistanceJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.1, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.1, 0.0, 0.0)); // 1

    let joints = vec![DistanceJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        min_distance,
        min_distance + 1.0,
        compliance,
    )];
    (state, joints)
}

/// A free-floating pair stretched past their rod length in zero gravity: body 1
/// is kicked outward so the rigid rod snaps it back, with momentum conserved.
fn free_pair_scene(length: f32) -> (RigidBodyState, Vec<DistanceJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5 * length, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.5 * length, 0.0, 0.0)); // 1
    state.linear_velocities[1] = Vec3::new(0.4, 0.3, 0.0);

    let joints = vec![DistanceJoint::rigid_rod(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        length,
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
    solver: &GpuDistanceJointSolver,
    initial: &RigidBodyState,
    joints: &[DistanceJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_distance(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_distance(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_distance_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU distance joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuDistanceJointSolver::new(&ctx);

    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid-rod pendulum, single sweep: the bilateral limit holds the length
    // while the bob swings.
    let (state, joints) = rod_pendulum_scene(1.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        180,
        "rod_pendulum_rigid_1sweep",
    );

    // Soft-rod pendulum, four sweeps: exercises the compliance term and the
    // repeated projection loop.
    let (state, joints) = rod_pendulum_scene(1.0, 1.0e-4);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &four_sweeps,
        180,
        "rod_pendulum_soft_4sweep",
    );

    // Rope arresting a free fall, four sweeps with damping: the body falls
    // through the free dead zone, then the upper limit snaps taut. This is the
    // key dead-zone-to-active-zone transition for the one-sided limit.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.1, 0.05);
    let (state, joints) = rope_fall_scene(1.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped,
        &four_sweeps,
        180,
        "rope_fall_damped_4sweep",
    );

    // Minimum-distance strut in zero gravity: a close pair pushed apart by the
    // lower limit, exercising the opposite correction sign.
    let no_gravity = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
    let (state, joints) = min_strut_scene(1.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &no_gravity,
        &four_sweeps,
        150,
        "min_strut_rigid_4sweep",
    );

    // Free-floating pair snapped back by a rigid rod in zero gravity: pure
    // constraint dynamics with both bodies dynamic, stressing the symmetric
    // correction and velocity recovery. Four substeps and one sweep keep the
    // undamped angular-velocity recovery's `2 * inv_h` amplification moderate.
    let no_gravity_4 = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let (state, joints) = free_pair_scene(1.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &no_gravity_4,
        &one_sweep,
        120,
        "free_pair_rigid_1sweep",
    );
}
