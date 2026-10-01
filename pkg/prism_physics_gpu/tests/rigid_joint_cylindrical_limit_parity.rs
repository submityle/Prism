//! Real-device parity for the cylindrical-limit joint stepper: the `GPU`
//! [`GpuCylindricalLimitJointSolver::solve_joints_cylindrical_limit`] must
//! reproduce the `CPU` golden twin
//! [`cpu_solve_joints_cylindrical_limit`](prism_physics_gpu::cpu_solve_joints_cylindrical_limit),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise all three of the joint's constraints — the axis
//! alignment, the point-on-line weld, and the one-sided along-axis travel limit
//! — across the lower-stop, upper-stop, free-spin, and momentum-conserving
//! regimes:
//!
//! * a dynamic sleeve dropped under a gravity pulling *along* the slide axis
//!   until it slams into the far travel stop, where the limit must engage and
//!   then hold the slide pinned at the bound;
//! * a sleeve launched the other way along the axis with no gravity until it
//!   catches on the near travel stop, exercising the opposite active bound;
//! * a spinning sleeve pushed into a stop with no gravity, where the spin about
//!   the shared axis must stay free while the limit and alignment both correct
//!   (low substep count to keep the single-precision spin integration from
//!   diverging between the two engines); and
//! * a free-floating equal-mass pair pulled apart past the symmetric stop, where
//!   every correction is internal, so total linear momentum stays conserved
//!   identically on both engines.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge, the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, and the one-sided along-axis limit shared
//! with the prismatic limit, with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"; Macklin et al., "XPBD:
//! Position-Based Simulation of Compliant Constrained Dynamics"), over the
//! world-space inverse inertia and quaternion kinematics of Baraff & Witkin. No
//! Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_cylindrical_limit, CylindricalLimitJoint, GpuContext,
    GpuCylindricalLimitJointSolver, IntegratorConfig, JointSolverConfig, RigidBodyState,
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

/// A static rod along `+Y` at the origin and a dynamic sleeve coincident with
/// it, free to slide `half_travel` either way from the start before a rigid
/// stop. The caller supplies a gravity *along* the slide axis so the sleeve
/// drops until the far stop catches it.
fn dropped_sleeve(half_travel: f32) -> (RigidBodyState, Vec<CylindricalLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: rod
    push_dynamic(&mut state, Vec3::ZERO, Quat::IDENTITY); // 1: sleeve

    let joints = vec![CylindricalLimitJoint::symmetric(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Vec3::Y,
        half_travel,
    )];
    (state, joints)
}

/// A static rod along `+Y` and a dynamic sleeve launched along the axis with no
/// gravity, so it slides until the near travel stop catches it.
fn launched_sleeve(half_travel: f32) -> (RigidBodyState, Vec<CylindricalLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: rod
    push_dynamic(&mut state, Vec3::ZERO, Quat::IDENTITY); // 1: sleeve
    state.linear_velocities[1] = Vec3::new(0.0, 2.0, 0.0);

    let joints = vec![CylindricalLimitJoint::symmetric(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Vec3::Y,
        half_travel,
    )];
    (state, joints)
}

/// A dynamic sleeve spinning about the shared `+Y` axis — its own symmetry
/// axis — while a gravity along the axis pushes it into the far travel stop.
/// The spin must stay free (the alignment locks only the two perpendicular
/// rotations, never the spin about the axis) while the limit pins the slide.
fn spinning_against_stop(half_travel: f32) -> (RigidBodyState, Vec<CylindricalLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: rod
    push_dynamic(&mut state, Vec3::ZERO, Quat::IDENTITY); // 1: sleeve
    state.angular_velocities[1] = Vec3::new(0.0, 1.5, 0.0);

    let joints = vec![CylindricalLimitJoint::symmetric(
        0,
        1,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Y,
        Vec3::Y,
        half_travel,
    )];
    (state, joints)
}

/// A free-floating equal-mass pair coupled along a shared `+Y` axis and pulled
/// apart past the symmetric stop while sharing one translational drift. Both
/// bodies are dynamic, so once the limit engages its internal corrections are
/// equal and opposite and total linear momentum is conserved.
fn free_pair(half_travel: f32) -> (RigidBodyState, Vec<CylindricalLimitJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO, Quat::IDENTITY); // 0
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY); // 1
    state.linear_velocities[0] = Vec3::new(0.1, -0.2, -0.1);
    state.linear_velocities[1] = Vec3::new(0.1, 0.3, -0.1);

    let joints = vec![CylindricalLimitJoint::symmetric(
        0,
        1,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::Y,
        Vec3::Y,
        half_travel,
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
    solver: &GpuCylindricalLimitJointSolver,
    initial: &RigidBodyState,
    joints: &[CylindricalLimitJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_cylindrical_limit(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_cylindrical_limit(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_cylindrical_limit_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU cylindrical limit joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCylindricalLimitJointSolver::new(&ctx);

    let eight_sweeps = JointSolverConfig::new(8);

    // Sleeve dropped under a gravity pulling along the slide axis until it slams
    // into the far travel stop: the limit is inert through the free dead zone,
    // then engages and holds the slide pinned at the bound. The slide position
    // must track the bound identically on both engines.
    let (state, joints) = dropped_sleeve(0.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0),
        &eight_sweeps,
        180,
        "sleeve_drops_onto_far_stop",
    );

    // Sleeve launched the other way along the axis with no gravity until the near
    // travel stop catches it: exercises the opposite active bound and the dead
    // zone on the way out.
    let (state, joints) = launched_sleeve(0.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0),
        &eight_sweeps,
        180,
        "sleeve_launches_onto_near_stop",
    );

    // Spinning sleeve pushed into a stop with no spin damping: the sleeve turns
    // about its own axis while the limit pins the slide and the alignment/weld
    // hold the axis parallel. The spin must be preserved identically on both
    // engines — the alignment and limit must never bleed into the free about-axis
    // rotation. A low substep count keeps the single-precision quaternion
    // integration of the free spin from diverging between the two engines.
    let (state, joints) = spinning_against_stop(0.25);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0),
        &JointSolverConfig::new(4),
        90,
        "spin_preserved_against_stop",
    );

    // Free-floating equal-mass pair pulled apart past the symmetric stop while
    // sharing one translational drift: both bodies are dynamic, so once the limit
    // engages its equal-and-opposite corrections must conserve linear momentum
    // identically on both engines across the window.
    let (state, joints) = free_pair(0.3);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0),
        &JointSolverConfig::new(4),
        90,
        "free_pair_conserves_momentum",
    );
}
