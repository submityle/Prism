//! Real-device parity for the hinge angular-limit joint stepper: the `GPU`
//! [`GpuHingeLimitJointSolver::solve_joints_hinge_limit`] must reproduce the
//! `CPU` golden twin
//! [`cpu_solve_joints_hinge_limit`](prism_physics_gpu::cpu_solve_joints_hinge_limit),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise every one of the three constraints — the point-to-point
//! weld, the axis alignment, and the one-sided angular limit — across both the
//! free interior of the range (where the limit is a no-op dead zone) and against
//! each stop, in the rigid (`compliance = 0`) and soft (`compliance > 0`)
//! regimes with more than one projection sweep, so any per-iteration divergence
//! between the hand-written `CPU` and shader arithmetic accumulates past the
//! tolerance rather than sliding under it. The limit's branchy dead zone — a
//! `return` inside the range, a signed correction outside it — is the novel path
//! here, so the stops are approached from both sides.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the hinge
//! axis-alignment constraint, and the one-sided angular limit with their substep
//! `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with XPBD").
//! No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_hinge_limit, GpuContext, GpuHingeLimitJointSolver, HingeLimitJoint,
    IntegratorConfig, JointSolverConfig, RigidBodyState,
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

/// A dynamic link hinged by its inner end to a static pivot about the world Z
/// axis, extending along +X. The hinge angle is bounded by `[min_angle,
/// max_angle]`; gravity (when enabled) pulls the link to a negative angle. The
/// weld point is coincident and the axes already parallel at construction, so
/// there is no impulsive start-up correction.
fn hinged_link_scene(
    min_angle: f32,
    max_angle: f32,
    compliance: f32,
    angular_compliance: f32,
    limit_compliance: f32,
) -> (RigidBodyState, Vec<HingeLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1: link

    let joints = vec![HingeLimitJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::Z,
        Vec3::Z,
        Vec3::X,
        Vec3::X,
        min_angle,
        max_angle,
        compliance,
        angular_compliance,
        limit_compliance,
    )];
    (state, joints)
}

/// A free-floating hinged pair in zero gravity: two dynamic bodies welded at the
/// midpoint and hinged about the world Y axis, with body 1 given a spin about
/// the hinge axis that drives the angle toward (and past) the upper stop so the
/// one-sided limit fires while linear momentum stays conserved.
fn free_pair_scene(
    min_angle: f32,
    max_angle: f32,
    compliance: f32,
    angular_compliance: f32,
    limit_compliance: f32,
) -> (RigidBodyState, Vec<HingeLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.5, 0.0, 0.0)); // 1
    state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);

    let joints = vec![HingeLimitJoint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::Y,
        Vec3::Y,
        Vec3::X,
        Vec3::X,
        min_angle,
        max_angle,
        compliance,
        angular_compliance,
        limit_compliance,
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
    solver: &GpuHingeLimitJointSolver,
    initial: &RigidBodyState,
    joints: &[HingeLimitJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_hinge_limit(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_hinge_limit(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_hinge_limit_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU hinge-limit joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHingeLimitJointSolver::new(&ctx);

    use std::f32::consts::FRAC_PI_2;
    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    // A lightly damped gravity integrator. Rigid one-sided stops chatter at the
    // boundary under zero damping, and the velocity recovery's `2 * inv_h` gain
    // amplifies the tiny CPU/GPU rounding gap of that chatter past the tight
    // tolerance; the linear/angular damping dissipates the chatter energy while
    // the sustained drive keeps the limit continuously active (never flipping
    // branch), exactly as the distance joint's `rope_fall` scene relies on.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.1, 0.05);
    // A more strongly damped gravity integrator for the compliant soft stop: a
    // regularised limit is a spring, so under light damping the link oscillates
    // about the stop for the whole window and the velocity peaks of that
    // oscillation amplify the CPU/GPU rounding gap; the heavier damping lets it
    // settle onto the stop so the residual velocity — and hence the divergence —
    // stays within the tight tolerance.
    let well_damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.2, 0.1);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Free swing inside a wide range: the link falls under gravity but never
    // leaves the free interior over the window tested, so the limit is a pure
    // dead-zone `return` on every sweep. Rigid, single sweep.
    let (state, joints) = hinged_link_scene(-FRAC_PI_2, FRAC_PI_2, 0.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        40,
        "free_swing_rigid_1sweep",
    );

    // Tight lower stop under gravity: the link descends onto the stop and the
    // limit arrests it, exercising the active lower branch with a rigid stop and
    // repeated projection sweeps.
    let (state, joints) = hinged_link_scene(-0.3, FRAC_PI_2, 0.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped,
        &four_sweeps,
        120,
        "lower_stop_rigid_4sweep",
    );

    // Soft lower stop under gravity: the same descent against a genuinely
    // compliant stop, exercising all three compliance terms and the regularised
    // limit at once; the softer spring keeps the settling velocities low enough
    // for frame-for-frame parity across the full window.
    let (state, joints) = hinged_link_scene(-0.3, FRAC_PI_2, 1.0e-3, 1.0e-3, 1.0e-3);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &well_damped,
        &four_sweeps,
        120,
        "lower_stop_soft_4sweep",
    );

    // Upper stop in zero gravity: a link spun up about the hinge axis drives the
    // angle up against a tight upper bound, exercising the active upper branch.
    let spun = IntegratorConfig::new(Vec3::ZERO, 8, 0.1, 0.05);
    let (mut state, joints) = hinged_link_scene(-FRAC_PI_2, 0.4, 0.0, 0.0, 0.0);
    state.angular_velocities[1] = Vec3::new(0.0, 0.0, 3.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &spun,
        &four_sweeps,
        90,
        "upper_stop_rigid_4sweep",
    );

    // Free-floating counter case: both bodies dynamic in zero gravity, the pair
    // spinning so the one-sided limit fires while linear momentum is conserved.
    // Four substeps keep the angular-velocity recovery's `2 * inv_h`
    // amplification of per-substep quaternion rounding moderate.
    let drive = IntegratorConfig::new(Vec3::ZERO, 4, 0.1, 0.05);
    let (state, joints) = free_pair_scene(-0.2, 0.2, 0.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &drive,
        &one_sweep,
        90,
        "free_pair_upper_1sweep",
    );
}
