//! Real-device parity: the `GPU` 6-DOF rigid-body integrator must reproduce the
//! `CPU` golden twin's trajectory, frame for frame, within a tight
//! floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device such as an Apple `M`-series `GPU`.
//!
//! Each scene stresses a different corner of the integrator: free fall (linear
//! only), an isotropic spinner (rotation with zero gyroscopic coupling), an
//! asymmetric tumbler (the explicit gyroscopic term that makes a wrench flip),
//! and a body driven by a constant torque. Running many frames lets any
//! per-substep divergence accumulate, so a shader that only *approximately*
//! mirrored the `CPU` arithmetic would drift past the tolerance rather than
//! sliding under it.
//!
//! Provenance: Euler's rigid-body equations with explicit gyroscopic coupling
//! and the quaternion kinematic equation (Baraff & Witkin). No Unreal Engine
//! source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_integrate, GpuContext, GpuRigidIntegrator, IntegratorConfig, RigidBodyState,
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

/// A mixed scene: a free-falling body, an isotropic spinner, an asymmetric
/// tumbler, and a body under constant torque — advanced in one dispatch so the
/// parity check covers several bodies at once.
fn mixed_scene() -> (RigidBodyState, Vec<Vec3>, Vec<Vec3>) {
    let mut state = RigidBodyState::new();
    // 0: free fall, no rotation input.
    state.push(Vec3::new(0.0, 10.0, 0.0), Quat::IDENTITY, 1.0, Vec3::ONE);
    // 1: isotropic spinner.
    state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(2.0));
    state.angular_velocities[1] = Vec3::new(0.4, 1.3, -0.7);
    // 2: asymmetric tumbler (inertia 1, 2, 4).
    state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::new(1.0, 0.5, 0.25));
    state.angular_velocities[2] = Vec3::new(1.0, 0.2, 0.1);
    // 3: body driven by a constant torque.
    state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::new(0.5, 0.5, 0.5));

    let forces = vec![Vec3::ZERO; state.len()];
    let mut torques = vec![Vec3::ZERO; state.len()];
    torques[3] = Vec3::new(1.5, 0.0, 0.0);
    (state, forces, torques)
}

/// Runs `frames` of both engines from the same initial state and checks
/// per-frame parity.
fn run_parity(
    ctx: &GpuContext,
    integrator: &GpuRigidIntegrator,
    initial: &RigidBodyState,
    forces: &[Vec3],
    torques: &[Vec3],
    config: &IntegratorConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_integrate(&mut cpu, forces, torques, config, dt).expect("cpu integrate");
        integrator
            .integrate(ctx, &mut gpu, forces, torques, config, dt)
            .expect("gpu integrate");
        assert_parity(&cpu, &gpu, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_rigid_integrator_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU rigid integrator parity: no wgpu adapter on this host");
        return;
    };
    let integrator = GpuRigidIntegrator::new(&ctx);

    // Gravity + rotation, no damping: linear free fall interleaves with the
    // isotropic, tumbling, and torque-driven bodies in a single dispatch.
    let (state, forces, torques) = mixed_scene();
    let config = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &config,
        120,
        "mixed_scene",
    );

    // Zero gravity isolates the rotational dynamics, including the gyroscopic
    // coupling of the asymmetric tumbler, over a long run.
    let no_gravity = IntegratorConfig::new(Vec3::ZERO, 16, 0.0, 0.0);
    let (state, forces, torques) = mixed_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &no_gravity,
        200,
        "mixed_scene_no_gravity",
    );

    // Damping on both channels exercises the per-substep velocity scales.
    let damped = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.4, 0.2);
    let (state, forces, torques) = mixed_scene();
    run_parity(
        &ctx,
        &integrator,
        &state,
        &forces,
        &torques,
        &damped,
        120,
        "mixed_scene_damped",
    );
}
