//! Real-device parity for the prismatic (slider) joint stepper: the `GPU`
//! [`GpuPrismaticJointSolver::solve_joints_prismatic`] must reproduce the `CPU`
//! golden twin
//! [`cpu_solve_joints_prismatic`](prism_physics_gpu::cpu_solve_joints_prismatic),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes stay deliberately gentle — a single sliding carriage, a two-link
//! sliding stack, and a free-floating sliding pair — so the per-frame parity
//! check tests kernel arithmetic equivalence rather than the chaotic divergence
//! a stiff, impulsive correction would amplify. Each scene exercises both the
//! perpendicular weld and the angular-lock constraint across the rigid
//! (`compliance = 0`) and soft (`compliance > 0`) regimes and more than one
//! projection sweep, so any per-iteration divergence between the hand-written
//! `CPU` and shader arithmetic accumulates past the tolerance rather than
//! sliding under it.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis and the relative-orientation lock with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_prismatic, GpuContext, GpuPrismaticJointSolver, IntegratorConfig,
    JointSolverConfig, PrismaticJoint, RigidBodyState,
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

/// A single sliding carriage: a dynamic body anchored to a static base and free
/// to slide along the world Y axis, falling under gravity. The anchors coincide
/// and the orientation already matches rest at construction, so there is no
/// impulsive start-up correction.
fn carriage_scene(
    compliance: f32,
    angular_compliance: f32,
) -> (RigidBodyState, Vec<PrismaticJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::ZERO); // 1

    let joints = vec![PrismaticJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Quat::IDENTITY,
        compliance,
        angular_compliance,
    )];
    (state, joints)
}

/// A two-link sliding stack: a static base, a middle carriage sliding along Y
/// relative to the base, and a top carriage sliding along Y relative to the
/// middle. Joints 0-1 and 1-2 share the dynamic link 1, so the pair spans more
/// than one colour batch.
fn stack_scene(compliance: f32, angular_compliance: f32) -> (RigidBodyState, Vec<PrismaticJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::ZERO); // 1
    push_dynamic(&mut state, Vec3::ZERO); // 2

    let joints = vec![
        PrismaticJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            compliance,
            angular_compliance,
        ),
        PrismaticJoint::new(
            1,
            2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            compliance,
            angular_compliance,
        ),
    ];
    (state, joints)
}

/// A free-floating sliding pair in zero gravity: two dynamic bodies joined by a
/// slider along X, pushed apart along their perpendicular (Y) so the weld pulls
/// them back while the pair's linear momentum stays conserved.
fn free_pair_scene(
    compliance: f32,
    angular_compliance: f32,
) -> (RigidBodyState, Vec<PrismaticJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.5, 0.0, 0.0)); // 1
    state.linear_velocities[0] = Vec3::new(0.0, 0.4, 0.0);
    state.linear_velocities[1] = Vec3::new(0.0, -0.4, 0.0);

    let joints = vec![PrismaticJoint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::X,
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
    solver: &GpuPrismaticJointSolver,
    initial: &RigidBodyState,
    joints: &[PrismaticJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_prismatic(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_prismatic(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_prismatic_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU prismatic joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuPrismaticJointSolver::new(&ctx);

    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid sliding carriage, single sweep.
    let (state, joints) = carriage_scene(0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        180,
        "carriage_rigid_1sweep",
    );

    // Soft slider, four sweeps: exercises both compliance terms and the repeated
    // projection loop.
    let (state, joints) = carriage_scene(1.0e-4, 1.0e-4);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &four_sweeps,
        180,
        "carriage_soft_4sweep",
    );

    // Two-link sliding stack with damping: multiple joints across more than one
    // colour batch (joints 0-1 and 1-2 share the dynamic link 1), eight substeps
    // and four sweeps with the per-substep velocity scales active — the only
    // scene that exercises inter-joint coupling across colour batches. We hold
    // the same tight 1e-4 floor as the other scenes; a stack of pure sliders
    // along a single axis is a benign (non-chaotic) system, so the full 180
    // frames stay well under the bound.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.2, 0.1);
    let (state, joints) = stack_scene(0.0, 0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped,
        &four_sweeps,
        180,
        "stack_rigid_damped_4sweep",
    );

    // Free-floating sliding pair in zero gravity: pure constraint dynamics with
    // both bodies dynamic, stressing the symmetric perpendicular correction and
    // velocity recovery. Four substeps (rather than eight) keep the
    // angular-velocity recovery's `2 * inv_h` amplification of per-substep
    // quaternion rounding moderate.
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
