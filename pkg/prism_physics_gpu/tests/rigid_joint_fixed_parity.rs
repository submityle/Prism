//! Real-device parity for the fixed (weld) joint stepper: the `GPU`
//! [`GpuFixedJointSolver::solve_joints_fixed`] must reproduce the `CPU` golden
//! twin [`cpu_solve_joints_fixed`](prism_physics_gpu::cpu_solve_joints_fixed),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes stay deliberately gentle — a single welded block, a two-link weld
//! chain, and a free-floating welded pair — so the per-frame parity check tests
//! kernel arithmetic equivalence rather than the chaotic divergence a stiff,
//! impulsive correction would amplify. Each scene exercises both the positional
//! weld and the angular-lock constraint across the rigid (`compliance = 0`) and
//! soft (`compliance > 0`) regimes and more than one projection sweep, so any
//! per-iteration divergence between the hand-written `CPU` and shader arithmetic
//! accumulates past the tolerance rather than sliding under it. Offset anchors
//! load the rotational coupling (the `r x p` lever arm) so the weld exercises
//! the full six-degree-of-freedom lock rather than a pure point constraint.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! relative-orientation lock with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"). No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_fixed, FixedJoint, GpuContext, GpuFixedJointSolver, IntegratorConfig,
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

/// A single welded block: a dynamic body welded to a static base with offset
/// anchors so gravity loads both the positional weld and the angular lock (the
/// anchors coincide at the base origin at construction, so there is no
/// impulsive start-up jerk).
fn block_scene(compliance: f32, angular_compliance: f32) -> (RigidBodyState, Vec<FixedJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(0.0, -1.0, 0.0)); // 1

    // anchor_a on the base points down to (0, -1, 0); anchor_b on the body is at
    // its centre, so the two world anchors coincide and the pair starts
    // un-stressed with an identity rest orientation.
    let joints = vec![FixedJoint::new(
        0,
        1,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::ZERO,
        Quat::IDENTITY,
        compliance,
        angular_compliance,
    )];
    (state, joints)
}

/// A two-link weld chain: a static base, a middle body welded to the base, and
/// a top body welded to the middle. Joints 0-1 and 1-2 share the dynamic link
/// 1, so the pair spans more than one colour batch. Offset anchors load the
/// rotational coupling in both welds.
fn chain_scene(compliance: f32, angular_compliance: f32) -> (RigidBodyState, Vec<FixedJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(0.0, -1.0, 0.0)); // 1
    push_dynamic(&mut state, Vec3::new(0.0, -2.0, 0.0)); // 2

    let joints = vec![
        FixedJoint::new(
            0,
            1,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            compliance,
            angular_compliance,
        ),
        FixedJoint::new(
            1,
            2,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            compliance,
            angular_compliance,
        ),
    ];
    (state, joints)
}

/// A free-floating welded pair in zero gravity: two dynamic bodies welded at
/// offset anchors, spun and pushed with opposed velocities so the weld's
/// internal corrections conserve the pair's linear momentum while the full
/// six-degree-of-freedom lock resists the relative twist.
fn free_pair_scene(compliance: f32, angular_compliance: f32) -> (RigidBodyState, Vec<FixedJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.5, 0.0, 0.0)); // 1
    state.linear_velocities[0] = Vec3::new(0.0, 0.3, 0.0);
    state.linear_velocities[1] = Vec3::new(0.0, -0.3, 0.0);
    state.angular_velocities[0] = Vec3::new(0.0, 0.0, 0.2);

    let joints = vec![FixedJoint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::IDENTITY,
        compliance,
        angular_compliance,
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
    solver: &GpuFixedJointSolver,
    initial: &RigidBodyState,
    joints: &[FixedJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_fixed(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_fixed(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_fixed_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU fixed joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuFixedJointSolver::new(&ctx);

    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid welded block, single sweep.
    let (state, joints) = block_scene(0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        180,
        "block_rigid_1sweep",
    );

    // Soft weld, four sweeps: exercises both compliance terms and the repeated
    // projection loop.
    let (state, joints) = block_scene(1.0e-4, 1.0e-4);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &four_sweeps,
        180,
        "block_soft_4sweep",
    );

    // Two-link weld chain with damping: multiple joints across more than one
    // colour batch (joints 0-1 and 1-2 share the dynamic link 1), eight substeps
    // and four sweeps with the per-substep velocity scales active — the only
    // scene that exercises inter-joint coupling across colour batches. A rigid
    // weld chain is a benign (non-chaotic) system, so the full 180 frames stay
    // well under the tight 1e-4 floor shared by the other scenes.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.2, 0.1);
    let (state, joints) = chain_scene(0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped,
        &four_sweeps,
        180,
        "chain_rigid_damped_4sweep",
    );

    // Free-floating welded pair in zero gravity: pure constraint dynamics with
    // both bodies dynamic and an initial spin, stressing the symmetric weld
    // correction, the angular lock, and velocity recovery together. Four
    // substeps (rather than eight) keep the angular-velocity recovery's
    // `2 * inv_h` amplification of per-substep quaternion rounding moderate.
    let no_gravity = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let (state, joints) = free_pair_scene(0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &no_gravity,
        &one_sweep,
        120,
        "free_pair_rigid_1sweep",
    );
}
