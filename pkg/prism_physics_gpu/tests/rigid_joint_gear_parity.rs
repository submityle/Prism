//! Real-device parity for the gear (angular ratio coupling) joint stepper: the
//! `GPU` [`GpuGearJointSolver::solve_joints_gear`] must reproduce the `CPU`
//! golden twin [`cpu_solve_joints_gear`](prism_physics_gpu::cpu_solve_joints_gear),
//! frame for frame, within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any machine
//! with a real device such as an Apple `M`-series `GPU`.
//!
//! The scenes exercise the single ratio-coupling constraint in both regimes: the
//! rigid (`compliance = 0`) rate-lock that forces `ratio * omega_a + omega_b = 0`
//! each sweep, and the soft compliant gear whose
//! `alpha_tilde = compliance / h^2` regularisation applies a finite coupling
//! torque so the ratio is satisfied asymptotically. Both a positive ratio
//! (counter-rotating external mesh) and a negative ratio (co-rotating internal
//! gear) are checked, along with a static-housing gear that brakes a spinning
//! body to the coupled rest. The coupling's relative-displacement term reads the
//! pre-substep snapshot (`prev_orientations`), so any `CPU` / `GPU` divergence in
//! the snapshot-relative spin accumulates past the tolerance rather than sliding
//! under it.
//!
//! Provenance: the angular ratio coupling expressed as a per-substep compliant
//! equality on the relative angular displacement about the two gear axes
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_joints_gear, GearJoint, GpuContext, GpuGearJointSolver, IntegratorConfig,
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

/// Pushes a dynamic unit body (unit inverse mass, unit inverse inertia) at
/// `position` with identity orientation.
fn push_dynamic(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
}

/// Pushes an immovable body (zero inverse mass and inertia) at `position`.
fn push_static(state: &mut RigidBodyState, position: Vec3) {
    state.push(position, Quat::IDENTITY, 0.0, Vec3::ZERO);
}

/// Two coaxial dynamic gears about the world `+Z` axis, body 0 spun up by an
/// initial angular velocity, coupled at `ratio`.
fn dynamic_pair_scene(ratio: f32, spin_a: f32) -> (RigidBodyState, Vec<GearJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0: driving gear
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1: driven gear
    state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);

    let joints = vec![GearJoint::rigid(0, 1, Vec3::Z, Vec3::Z, ratio)];
    (state, joints)
}

/// A static gear housing (locked driving shaft) coupled to a dynamic spinning
/// gear: the rigid gear brakes the dynamic body toward the coupled rest.
fn static_housing_scene(ratio: f32, spin_b: f32) -> (RigidBodyState, Vec<GearJoint>) {
    let mut state = RigidBodyState::new();
    push_static(&mut state, Vec3::ZERO); // 0: static housing
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1: dynamic gear
    state.angular_velocities[1] = Vec3::new(0.0, 0.0, spin_b);

    let joints = vec![GearJoint::rigid(0, 1, Vec3::Z, Vec3::Z, ratio)];
    (state, joints)
}

/// Two dynamic gears coupled by a soft (finite-torque) gear, body 0 spun up: the
/// compliant coupling is continuously active as the driven gear spins up
/// asymptotically, stressing the snapshot-relative term shared by CPU and GPU.
fn soft_pair_scene(ratio: f32, spin_a: f32, stiffness: f32) -> (RigidBodyState, Vec<GearJoint>) {
    let mut state = RigidBodyState::new();
    push_dynamic(&mut state, Vec3::ZERO); // 0
    push_dynamic(&mut state, Vec3::new(1.0, 0.0, 0.0)); // 1
    state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);

    let joints = vec![GearJoint::soft(0, 1, Vec3::Z, Vec3::Z, ratio, stiffness)];
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
    solver: &GpuGearJointSolver,
    initial: &RigidBodyState,
    joints: &[GearJoint],
    config: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_solve_joints_gear(&mut cpu, joints, config, joint_config, dt).expect("cpu solve");
        solver
            .solve_joints_gear(ctx, &mut gpu, joints, config, joint_config, dt)
            .expect("gpu solve");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_gear_joint_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU gear joint parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGearJointSolver::new(&ctx);

    // Zero gravity throughout: the gear coupling itself supplies all the forcing,
    // so the scenes isolate the ratio-coupling path. Four substeps (not eight)
    // keep the first-order velocity recovery `omega = 2 * inv_h * delta`
    // well-conditioned, so the per-substep orientation rounding gap between the
    // backends stays under the parity floor.
    let still = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
    let two_sweeps = JointSolverConfig::new(2);
    let four_sweeps = JointSolverConfig::new(4);

    // Rigid positive-ratio gear counter-rotating a driven gear against a driving
    // one: the zero-compliance (`d_lambda = -C / w`) rate-lock branch is active
    // every sweep as the coupling holds `2 * omega_a + omega_b = 0`.
    let (state, joints) = dynamic_pair_scene(2.0, 1.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        120,
        "rigid_positive_ratio_counter_rotation",
    );

    // Rigid negative-ratio gear co-rotating the driven gear (internal gear /
    // belt): the gradient's sign path is exercised in the opposite direction.
    let (state, joints) = dynamic_pair_scene(-1.5, 1.5);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        80,
        "rigid_negative_ratio_co_rotation",
    );

    // Static gear housing braking a spinning dynamic gear to the coupled rest:
    // one body static (corrections scale by zero), the other dynamic.
    let (state, joints) = static_housing_scene(2.0, 3.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &four_sweeps,
        120,
        "static_housing_brakes_gear",
    );

    // Soft compliant gear coupling asymptotically: the regularised coupling and
    // its snapshot-relative term are continuously active, so this is the scene
    // that stresses the `prev_orientations` path shared by CPU and GPU.
    let (state, joints) = soft_pair_scene(2.0, 2.0, 60.0);
    run_parity(
        &ctx,
        &solver,
        &state,
        &joints,
        &still,
        &two_sweeps,
        90,
        "soft_gear_couples_gradually",
    );
}
