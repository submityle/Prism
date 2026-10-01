//! Real-device parity for the spherical (ball-and-socket) joint stepper: the
//! `GPU` [`GpuSphericalJointSolver::solve_joints_spherical`] must reproduce the
//! `CPU` golden twin
//! [`cpu_solve_joints_spherical`](prism_physics_gpu::cpu_solve_joints_spherical),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes stay deliberately gentle — a single pendulum, a three-link hanging
//! chain, and a free-floating pair spinning about their shared weld — so the
//! per-frame parity check tests kernel arithmetic equivalence rather than the
//! chaotic divergence a stiff, impulsive correction would amplify. Running many
//! substeps per frame over many frames, with both the rigid (`compliance = 0`)
//! and the soft (`compliance > 0`) regimes and more than one projection sweep,
//! lets any per-iteration divergence between the hand-written `CPU` and shader
//! arithmetic accumulate, so a kernel that only approximately mirrored the `CPU`
//! would drift past the tolerance rather than sliding under it.
//!
//! Provenance: the point-to-point (ball-socket) constraint and its substep
//! `XPBD` positional handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_spherical, GpuContext, GpuSphericalJointSolver, IntegratorConfig,
    JointSolverConfig, RigidBodyState, SphericalJoint,
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

/// A single pendulum: a dynamic body welded by its top to a static pivot, with a
/// small sideways kick so it swings. The weld point is coincident at
/// construction, so there is no impulsive start-up correction.
fn pendulum_scene(compliance: f32) -> (RigidBodyState, Vec<SphericalJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::new(0.0, -1.0, 0.0)); // 1: bob
    state.linear_velocities[1] = Vec3::new(0.5, 0.0, 0.0);

    // Pivot anchor at the world origin; bob anchor at its own top, so both map
    // to the origin.
    let joints = vec![SphericalJoint::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        compliance,
    )];
    (state, joints)
}

/// A three-link hanging chain: a static pivot with two dynamic links below it,
/// each welded top-to-bottom at a coincident world point.
fn chain_scene(compliance: f32) -> (RigidBodyState, Vec<SphericalJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: pivot
    push_dynamic(&mut state, Vec3::new(0.0, -1.0, 0.0)); // 1
    push_dynamic(&mut state, Vec3::new(0.0, -2.0, 0.0)); // 2
    state.linear_velocities[2] = Vec3::new(0.3, 0.0, 0.1);

    let joints = vec![
        // pivot (origin) to link 1's top (local +0.5y -> world -0.5y)... keep the
        // weld at the shared point -0.5y.
        SphericalJoint::new(
            0,
            1,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            compliance,
        ),
        // link 1's bottom to link 2's top, coincident at -1.5y.
        SphericalJoint::new(
            1,
            2,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            compliance,
        ),
    ];
    (state, joints)
}

/// A free-floating pair of dynamic bodies welded at their shared midpoint,
/// counter-spinning in zero gravity so the pair rotates about its centre of
/// mass while the linear momentum stays null.
fn free_pair_scene(compliance: f32) -> (RigidBodyState, Vec<SphericalJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::new(-0.5, 0.0, 0.0)); // 0
    push_dynamic(&mut state, Vec3::new(0.5, 0.0, 0.0)); // 1
    state.linear_velocities[0] = Vec3::new(0.0, 0.6, 0.0);
    state.linear_velocities[1] = Vec3::new(0.0, -0.6, 0.0);

    // Both anchors reach the origin midpoint.
    let joints = vec![SphericalJoint::new(
        0,
        1,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        compliance,
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
    solver: &GpuSphericalJointSolver,
    initial: &RigidBodyState,
    joints: &[SphericalJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_spherical(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_spherical(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_spherical_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU spherical joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuSphericalJointSolver::new(&ctx);

    let gravity = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    let one_sweep = JointSolverConfig::new(1);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid pendulum, single sweep.
    let (state, joints) = pendulum_scene(0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &one_sweep,
        180,
        "pendulum_rigid_1sweep",
    );

    // Soft pendulum, four sweeps: exercises the compliance term and the repeated
    // projection loop.
    let (state, joints) = pendulum_scene(1.0e-4);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &gravity,
        &four_sweeps,
        180,
        "pendulum_soft_4sweep",
    );

    // Hanging chain with damping: multiple joints across more than one colour
    // batch (joints 0-1 and 1-2 share the dynamic link 1), with the per-substep
    // velocity scales active.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.2, 0.1);
    let (state, joints) = chain_scene(0.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &damped,
        &four_sweeps,
        150,
        "chain_rigid_damped_4sweep",
    );

    // Free-floating counter-spinning pair in zero gravity: pure constraint
    // dynamics with both bodies dynamic, stressing the symmetric correction and
    // velocity recovery. This undamped rotating system is the most phase-drift
    // sensitive scene, so its horizon is kept moderate — long enough to
    // accumulate any kernel-arithmetic divergence, short enough that the
    // Lyapunov phase separation of two bit-divergent trajectories stays under
    // the tolerance.
    // Four substeps (rather than eight) keep the angular-velocity recovery's
    // `2 * inv_h` amplification of per-substep quaternion rounding moderate; at
    // eight substeps the ~960x gain pushes undamped free-rotation parity to its
    // floating-point floor.
    let no_gravity = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let (state, joints) = free_pair_scene(0.0);
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
