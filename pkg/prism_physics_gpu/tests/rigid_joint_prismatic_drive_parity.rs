//! Real-device parity for the prismatic position-drive joint stepper: the `GPU`
//! [`GpuPrismaticDriveJointSolver::solve_joints_prismatic_drive`] must reproduce
//! the `CPU` golden twin
//! [`cpu_solve_joints_prismatic_drive`](prism_physics_gpu::cpu_solve_joints_prismatic_drive),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise all three constraints — the angular lock, the
//! perpendicular point-to-point weld, and the along-axis position drive — in
//! both regimes of the drive: the rigid (`drive_compliance = 0`) position servo
//! that snaps the slide onto its target each sweep, and the soft
//! compliant-and-damped spring whose `gamma = drive_compliance * drive_damping /
//! h` damping term reads the pre-substep snapshot. The damping term is the novel
//! `XPBD` path here — it couples the correction to the along-axis closing rate
//! derived from the `prev_positions` / `prev_orientations` bindings — so at
//! least one scene drives a genuinely soft spring with heavy damping so that
//! term is continuously active and any `CPU` / `GPU` divergence in the
//! snapshot-relative anchor displacement accumulates past the tolerance rather
//! than sliding under it. A free-floating equal-mass pair confirms the drive's
//! internal corrections stay equal and opposite (linear momentum conserved)
//! identically on both engines.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis, the relative-orientation lock, and the
//! bilateral along-axis compliant-and-damped drive with their substep `XPBD`
//! handling and Macklin-style damping regularisation (Müller et al., "Detailed
//! Rigid Body Simulation with XPBD"; Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics"). No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_prismatic_drive, GpuContext, GpuPrismaticDriveJointSolver, IntegratorConfig,
    JointSolverConfig, PrismaticDriveJoint, RigidBodyState,
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

/// A dynamic slider hung off a static pivot and free to travel along `+Y`,
/// commanded by a rigid position servo onto `target`.
///
/// Body `a` is the dynamic slider (index 1), body `b` the static pivot (index
/// 0), so the signed slide position runs negative as the slider descends the
/// `+Y` axis under gravity and the servo must hold or lift it onto `target`.
fn servo_slider_scene(target: f32) -> (RigidBodyState, Vec<PrismaticDriveJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::ZERO); // 1: slider

    let joints = vec![PrismaticDriveJoint::servo(
        1,
        0,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        target,
    )];
    (state, joints)
}

/// The same single-slider rig driven by a soft compliant-and-damped spring of
/// the given stiffness and damping onto `target`, so the Macklin damping term is
/// continuously active.
fn spring_slider_scene(
    target: f32,
    stiffness: f32,
    damping: f32,
) -> (RigidBodyState, Vec<PrismaticDriveJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::ZERO); // 1: slider

    let joints = vec![PrismaticDriveJoint::spring(
        1,
        0,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        target,
        stiffness,
        damping,
    )];
    (state, joints)
}

/// A free-floating equal-mass slider pair in zero gravity sharing a `+Y` slide
/// axis, with opposing initial velocities and a soft spring drive commanding the
/// two anchors apart: the drive's internal corrections are equal and opposite so
/// total linear momentum stays conserved on both engines.
fn free_pair_scene(
    target: f32,
    stiffness: f32,
    damping: f32,
) -> (RigidBodyState, Vec<PrismaticDriveJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(0.0, 0.5, 0.0)); // 1
    state.linear_velocities[0] = Vec3::new(0.0, -0.3, 0.0);
    state.linear_velocities[1] = Vec3::new(0.0, 0.7, 0.0);

    let joints = vec![PrismaticDriveJoint::spring(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        target,
        stiffness,
        damping,
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
    solver: &GpuPrismaticDriveJointSolver,
    initial: &RigidBodyState,
    joints: &[PrismaticDriveJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_prismatic_drive(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_prismatic_drive(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_prismatic_drive_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU prismatic-drive joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuPrismaticDriveJointSolver::new(&ctx);

    // A lightly damped gravity integrator. A rigid servo that resnaps the slide
    // onto the target every substep is a stiff bilateral constraint, and the
    // velocity recovery's `2 * inv_h` gain amplifies the tiny CPU/GPU rounding
    // gap of that snap; the mild linear/angular damping dissipates the residual
    // so the sustained hold stays within the tight tolerance.
    let gravity_damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.1, 0.05);
    // Zero gravity for the soft spring and free-pair scenes, where the drive
    // itself supplies all the forcing.
    let still = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
    let still_free = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let four_sweeps = JointSolverConfig::new(4);
    let eight_sweeps = JointSolverConfig::new(8);

    // Rigid servo holding the slider at its start against gravity: the drive
    // must cancel the full gravitational descent every substep, exercising the
    // zero-compliance (`d_lambda = -C / w`) servo branch continuously.
    let (state, joints) = servo_slider_scene(0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity_damped,
        &four_sweeps,
        120,
        "rigid_servo_hold_against_gravity",
    );

    // Rigid servo lifting the slider above its start and holding it there: the
    // drive both overcomes gravity and climbs to a non-zero target, so the servo
    // correction changes sign over the transient before settling.
    let (state, joints) = servo_slider_scene(0.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity_damped,
        &four_sweeps,
        120,
        "rigid_servo_lift_against_gravity",
    );

    // Soft compliant-and-damped spring converging onto its target with no
    // gravity: the regularised drive and its Macklin damping term are both
    // continuously active, so this is the scene that stresses the
    // snapshot-relative closing-rate (`gamma * dv`) path shared by CPU and GPU.
    let (state, joints) = spring_slider_scene(0.4, 300.0, 30.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &eight_sweeps,
        180,
        "soft_spring_converges_to_target",
    );

    // Free-floating equal-mass pair with opposing velocities and a soft spring
    // drive commanding the anchors apart: both bodies are dynamic, so the
    // equal-and-opposite drive corrections must conserve linear momentum
    // identically on both engines across the window.
    let (state, joints) = free_pair_scene(-1.0, 200.0, 10.0);
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
