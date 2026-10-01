//! Real-device parity for the prismatic travel-limit joint stepper: the `GPU`
//! [`GpuPrismaticLimitJointSolver::solve_joints_prismatic_limit`] must reproduce
//! the `CPU` golden twin
//! [`cpu_solve_joints_prismatic_limit`](prism_physics_gpu::cpu_solve_joints_prismatic_limit),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise every one of the three constraints — the angular lock,
//! the perpendicular point-to-point weld, and the one-sided travel limit —
//! across both the free interior of the range (where the limit is a no-op dead
//! zone) and against each stop, in the rigid (`compliance = 0`) and soft
//! (`compliance > 0`) regimes with more than one projection sweep, so any
//! per-iteration divergence between the hand-written `CPU` and shader arithmetic
//! accumulates past the tolerance rather than sliding under it. The limit's
//! branchy dead zone — a skip inside the range, a signed correction outside it —
//! is the novel path here, so the stops are approached from both sides and at
//! least one scene keeps a rigid (`limit_compliance = 0`) stop continuously
//! active.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis, the relative-orientation lock, and the
//! one-sided along-axis limit with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"). No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_prismatic_limit, GpuContext, GpuPrismaticLimitJointSolver, IntegratorConfig,
    JointSolverConfig, PrismaticLimitJoint, RigidBodyState,
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

/// A dynamic slider hanging from a static pivot, free to slide along the world
/// `+Y` axis through the pivot. The anchors coincide at the origin so the
/// initial slide position is zero; gravity (when enabled) drives the body along
/// the axis until it meets the active travel stop in `[min_distance,
/// max_distance]`.
fn vertical_slider_scene(
    min_distance: f32,
    max_distance: f32,
    compliance: f32,
    angular_compliance: f32,
    limit_compliance: f32,
) -> (RigidBodyState, Vec<PrismaticLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::ZERO); // 1: slider

    // Body `a` is the dynamic slider (index 1), body `b` the static pivot
    // (index 0), so the signed slide position runs negative as the slider
    // descends the +Y axis under gravity — a tight `min_distance` is the lower
    // stop, a tight `max_distance` the upper stop.
    let joints = vec![PrismaticLimitJoint::new(
        1,
        0,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Quat::IDENTITY,
        min_distance,
        max_distance,
        compliance,
        angular_compliance,
        limit_compliance,
    )];
    (state, joints)
}

/// A free-floating slider pair in zero gravity: two dynamic bodies sharing a
/// slide axis, with body 1 given a velocity along `+Y` that drives the signed
/// separation toward (and past) the upper stop so the one-sided limit fires
/// while total linear momentum stays conserved.
fn free_pair_scene(
    min_distance: f32,
    max_distance: f32,
    compliance: f32,
    angular_compliance: f32,
    limit_compliance: f32,
) -> (RigidBodyState, Vec<PrismaticLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(0.0, 0.5, 0.0)); // 1
    state.linear_velocities[1] = Vec3::new(0.0, 2.0, 0.0);

    let joints = vec![PrismaticLimitJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Quat::IDENTITY,
        min_distance,
        max_distance,
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
    solver: &GpuPrismaticLimitJointSolver,
    initial: &RigidBodyState,
    joints: &[PrismaticLimitJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_prismatic_limit(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_prismatic_limit(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_prismatic_limit_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU prismatic-limit joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuPrismaticLimitJointSolver::new(&ctx);

    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    // A lightly damped gravity integrator. Rigid one-sided stops chatter at the
    // boundary under zero damping, and the velocity recovery's `2 * inv_h` gain
    // amplifies the tiny CPU/GPU rounding gap of that chatter past the tight
    // tolerance; the linear/angular damping dissipates the chatter energy while
    // the sustained drive keeps the limit continuously active (never flipping
    // branch).
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.1, 0.05);
    // Gravity along `+Y` (an upward push) for the upper-stop scene.
    let damped_up = IntegratorConfig::new(Vec3::new(0.0, 9.81, 0.0), 8, 0.1, 0.05);
    // A more strongly damped gravity integrator for the compliant soft stop: a
    // regularised limit is a spring, so under light damping the slider
    // oscillates about the stop for the whole window and the velocity peaks of
    // that oscillation amplify the CPU/GPU rounding gap; the heavier damping lets
    // it settle onto the stop so the residual velocity — and hence the
    // divergence — stays within the tight tolerance.
    let well_damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.2, 0.1);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Free slide inside a wide range: the body falls under gravity but never
    // leaves the free interior over the window tested, so the limit is a pure
    // dead-zone skip on every sweep. Rigid, single sweep.
    let (state, joints) = vertical_slider_scene(-5.0, 5.0, 0.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        40,
        "free_inside_range_rigid_1sweep",
    );

    // Tight lower stop under gravity: the body descends onto the stop and the
    // rigid (`limit_compliance = 0`) limit arrests it, exercising the active
    // lower branch with repeated projection sweeps and a continuously active
    // rigid stop.
    let (state, joints) = vertical_slider_scene(-0.3, 5.0, 0.0, 0.0, 0.0);
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
    let (state, joints) = vertical_slider_scene(-0.3, 5.0, 1.0e-3, 1.0e-3, 1.0e-3);
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

    // Tight upper stop under an upward push: gravity along `+Y` drives the signed
    // separation positive into a tight upper bound, exercising the active upper
    // branch with a rigid stop.
    let (state, joints) = vertical_slider_scene(-5.0, 0.3, 0.0, 0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped_up,
        &four_sweeps,
        120,
        "upper_stop_rigid_4sweep",
    );

    // Free-floating counter case: both bodies dynamic in zero gravity, body 1
    // sliding along the axis so the one-sided limit fires while linear momentum
    // is conserved. A single sweep keeps the velocity recovery's `2 * inv_h`
    // amplification of per-substep rounding moderate.
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
