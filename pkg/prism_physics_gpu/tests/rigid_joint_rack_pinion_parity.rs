//! Real-device parity for the rack-and-pinion (rotation-to-translation ratio
//! coupling) joint stepper: the `GPU`
//! [`GpuRackPinionJointSolver::solve_joints_rack_pinion`] must reproduce the
//! `CPU` golden twin
//! [`cpu_solve_joints_rack_pinion`](prism_physics_gpu::cpu_solve_joints_rack_pinion),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise the single mixed angular-linear coupling in both regimes:
//! the rigid (`compliance = 0`) rate-lock that forces
//! `(u_a . omega_a) = ratio * (u_b . v_b)` each sweep, and the soft compliant
//! coupling whose `alpha_tilde = compliance / h^2` regularisation applies a
//! finite coupling force so the ratio is satisfied asymptotically. Both a
//! positive ratio and a negative ratio (reversed sense) are checked, along with
//! a static pinion that brakes a sliding rack to the coupled rest. The coupling's
//! incremental-motion terms read the pre-substep snapshots (`prev_orientations`
//! and `prev_positions`), so any `CPU` / `GPU` divergence in the
//! snapshot-relative motion accumulates past the tolerance rather than sliding
//! under it.
//!
//! Provenance: the rotation-to-translation ratio coupling expressed as a
//! per-substep compliant equality on the pinion's angular displacement and the
//! rack's linear displacement (Macklin et al., "XPBD: Position-Based Simulation
//! of Compliant Constrained Dynamics"). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_rack_pinion, GpuContext, GpuRackPinionJointSolver, IntegratorConfig,
    JointSolverConfig, RackPinionJoint, RigidBodyState,
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
/// `position` with identity orientation.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
}

/// Pushes an immovable body (zero inverse mass and inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// A dynamic pinion spinning about `+Z` (body 0) coupled to a dynamic rack
/// sliding along `+X` (body 1). The pinion is spun up by an initial angular
/// velocity; the rigid coupling ties its spin to the rack's slide at `ratio`.
fn dynamic_pair_scene(ratio: f32, spin_a: f32) -> (RigidBodyState, Vec<RackPinionJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0: pinion
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1: rack
    state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);

    let joints = vec![RackPinionJoint::rigid(0, 1, Vec3::Z, Vec3::X, ratio)];
    (state, joints)
}

/// A static pinion shaft (body 0, locked) coupled to a dynamic rack (body 1)
/// with an initial slide velocity: the rigid coupling against a static pinion
/// brakes the rack toward the coupled rest.
fn static_pinion_scene(ratio: f32, slide_b: f32) -> (RigidBodyState, Vec<RackPinionJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: static pinion
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1: dynamic rack
    state.linear_velocities[1] = Vec3::new(slide_b, 0.0, 0.0);

    let joints = vec![RackPinionJoint::rigid(0, 1, Vec3::Z, Vec3::X, ratio)];
    (state, joints)
}

/// Two dynamic bodies coupled by a soft (finite-force) rack-and-pinion, the
/// pinion spun up: the compliant coupling is continuously active as the rack
/// spins up asymptotically, stressing the snapshot-relative terms shared by CPU
/// and GPU.
fn soft_scene(ratio: f32, spin_a: f32, stiffness: f32) -> (RigidBodyState, Vec<RackPinionJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0: pinion
    push_dynamic(&mut state, Vec3::new(0.0, 1.0, 0.0)); // 1: rack
    state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);

    let joints = vec![RackPinionJoint::soft(
        0,
        1,
        Vec3::Z,
        Vec3::X,
        ratio,
        stiffness,
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
    solver: &GpuRackPinionJointSolver,
    initial: &RigidBodyState,
    joints: &[RackPinionJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_rack_pinion(&mut cpu, joints, config, joint_config, dt)
            .expect("cpu solve");
        solver
            .solve_joints_rack_pinion(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_rack_pinion_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU rack-and-pinion joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuRackPinionJointSolver::new(&ctx);

    // Zero gravity throughout: the coupling itself supplies all the forcing, so
    // the scenes isolate the ratio-coupling path. Four substeps (not eight) keep
    // the first-order velocity recovery well-conditioned, so the per-substep
    // rounding gap between the backends stays under the parity floor.
    let still = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let two_sweeps = JointSolverConfig::new(2);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid positive-ratio coupling driving a rack from a spinning pinion: the
    // zero-compliance (`d_lambda = -C / w`) rate-lock branch is active every
    // sweep as the coupling holds `(u_a . omega_a) = ratio * (u_b . v_b)`.
    let (state, joints) = dynamic_pair_scene(2.0, 1.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        80,
        "rigid_positive_ratio",
    );

    // Rigid negative-ratio coupling reversing the slide sense: the gradient's
    // sign path is exercised in the opposite direction.
    let (state, joints) = dynamic_pair_scene(-1.5, 1.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        80,
        "rigid_negative_ratio",
    );

    // Static pinion shaft braking a sliding rack to the coupled rest: one body
    // static (corrections scale by zero), the other dynamic.
    let (state, joints) = static_pinion_scene(2.0, 2.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        90,
        "static_pinion_brakes_rack",
    );

    // Soft compliant coupling asymptotically: the regularised coupling and its
    // snapshot-relative terms are continuously active, so this is the scene that
    // stresses the `prev_orientations` / `prev_positions` path shared by CPU and
    // GPU.
    let (state, joints) = soft_scene(2.0, 2.0, 60.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &two_sweeps,
        90,
        "soft_rack_pinion_couples_gradually",
    );
}
