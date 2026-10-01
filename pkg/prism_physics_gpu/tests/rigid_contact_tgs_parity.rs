//! Real-device parity: the `GPU` TGS-soft rigid-body contact solver must
//! reproduce the `CPU` golden twin's result — post-step positions,
//! orientations, linear and angular velocities, *and* the accumulated normal
//! and tangent impulses carried by warm starting — within a tight
//! floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / substep-dispatch / readback path on
//! any machine with a real device such as an Apple `M`-series `GPU`.
//!
//! Each scene stresses a different corner of the substepped soft solver: a
//! resting box over a static floor (gravity integration, relinearisation of the
//! separation each substep, Baumgarte-free soft penetration recovery, and the
//! warm-start carry that keeps the box from jittering), a head-on elastic pair
//! (the deferred restitution pass restoring the approach speed), an off-centre
//! elastic hit (the angular arm that the per-substep relinearisation rotates),
//! and a three-body stack under gravity (static-aware colouring across more
//! than one batch with load propagating through shared dynamic bodies). Every
//! scene is advanced for several frames so the integrated trajectory — not just
//! a single velocity delta — has room to drift past the tolerance rather than
//! hide under it.
//!
//! Restitution-active scenes use a single contact pair that does **not** share
//! a dynamic body with any other contact, so the deferred restitution pass is
//! order-independent regardless of how the colouring batches the contacts.
//!
//! Provenance: substepped soft-constraint (`TGS`-soft) contact solving with
//! per-substep relinearisation, a soft Baumgarte spring parameterised by
//! frequency and damping ratio, a deferred restitution pass, and warm starting
//! (Catto / `Box2D` TGS-soft). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_contacts_tgs, GpuContext, GpuRigidTgsContactSolver, IntegratorConfig,
    RigidBodyState, RigidContact, TgsContactConfig,
};

/// Absolute per-quantity divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// The smallest allowed difference between two scalar quantities under the
/// magnitude-scaled rule.
fn within(a: f32, b: f32) -> bool {
    (a - b).abs() <= ABS_TOLERANCE + REL_TOLERANCE * a.abs()
}

/// Quaternion divergence under the double-cover: `q` and `-q` are the same
/// orientation, so the error is the smaller of the two component-wise
/// distances.
fn quat_divergence(a: Quat, b: Quat) -> f32 {
    let (sx, sy, sz, sw) = (a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
    let same = (sx * sx + sy * sy + sz * sz + sw * sw).sqrt();
    let (fx, fy, fz, fw) = (a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    let flipped = (fx * fx + fy * fy + fz * fz + fw * fw).sqrt();
    same.min(flipped)
}

/// Asserts every body's post-step pose and velocities and every contact's
/// accumulated impulses in `gpu` are within the magnitude-scaled tolerance of
/// `cpu`.
fn assert_parity(
    cpu_state: &RigidBodyState,
    gpu_state: &RigidBodyState,
    cpu_contacts: &[RigidContact],
    gpu_contacts: &[RigidContact],
    scene: &str,
) {
    assert_eq!(
        cpu_state.len(),
        gpu_state.len(),
        "{scene}: body counts differ"
    );
    assert_eq!(
        cpu_contacts.len(),
        gpu_contacts.len(),
        "{scene}: contact counts differ"
    );

    for i in 0..cpu_state.len() {
        let dp = (cpu_state.positions[i] - gpu_state.positions[i]).length();
        let p_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu_state.positions[i].length();
        assert!(
            dp <= p_bound,
            "{scene}: body {i} position diverged by {dp} (bound {p_bound}): \
             cpu {:?} vs gpu {:?}",
            cpu_state.positions[i],
            gpu_state.positions[i]
        );

        let dq = quat_divergence(cpu_state.orientations[i], gpu_state.orientations[i]);
        assert!(
            dq <= ABS_TOLERANCE + REL_TOLERANCE,
            "{scene}: body {i} orientation diverged by {dq}: cpu {:?} vs gpu {:?}",
            cpu_state.orientations[i],
            gpu_state.orientations[i]
        );

        let dv = (cpu_state.linear_velocities[i] - gpu_state.linear_velocities[i]).length();
        let vel_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu_state.linear_velocities[i].length();
        assert!(
            dv <= vel_bound,
            "{scene}: body {i} linear velocity diverged by {dv} (bound {vel_bound}): \
             cpu {:?} vs gpu {:?}",
            cpu_state.linear_velocities[i],
            gpu_state.linear_velocities[i]
        );

        let dw = (cpu_state.angular_velocities[i] - gpu_state.angular_velocities[i]).length();
        let ang_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu_state.angular_velocities[i].length();
        assert!(
            dw <= ang_bound,
            "{scene}: body {i} angular velocity diverged by {dw} (bound {ang_bound}): \
             cpu {:?} vs gpu {:?}",
            cpu_state.angular_velocities[i],
            gpu_state.angular_velocities[i]
        );
    }

    for (ci, (cc, gc)) in cpu_contacts.iter().zip(gpu_contacts.iter()).enumerate() {
        assert!(
            within(cc.normal_impulse, gc.normal_impulse),
            "{scene}: contact {ci} normal impulse diverged: cpu {} vs gpu {}",
            cc.normal_impulse,
            gc.normal_impulse
        );
        assert!(
            within(cc.tangent_impulse_0, gc.tangent_impulse_0),
            "{scene}: contact {ci} tangent impulse 0 diverged: cpu {} vs gpu {}",
            cc.tangent_impulse_0,
            gc.tangent_impulse_0
        );
        assert!(
            within(cc.tangent_impulse_1, gc.tangent_impulse_1),
            "{scene}: contact {ci} tangent impulse 1 diverged: cpu {} vs gpu {}",
            cc.tangent_impulse_1,
            gc.tangent_impulse_1
        );
    }
}

/// A free body at `position` with the given inverse mass and isotropic inverse
/// inertia, at identity orientation and rest.
fn push_body(state: &mut RigidBodyState, position: Vec3, inv_mass: f32, inv_inertia: Vec3) {
    state.push(position, Quat::IDENTITY, inv_mass, inv_inertia);
}

/// A resting box of unit mass on a static floor, the contact point directly
/// beneath the box's centre so the normal impulse exerts no torque. Body 0 is
/// the static floor, body 1 the box; the normal points up.
fn resting_box_scene() -> (RigidBodyState, Vec<RigidContact>, IntegratorConfig) {
    let mut state = RigidBodyState::new();
    push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO);
    push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::splat(6.0));
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let contact = RigidContact::new(
        1,
        0,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.0, 0.5, 0.0),
        normal,
        0.0,
    )
    .with_friction(0.5);
    let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0);
    (state, vec![contact], integrator)
}

/// Two equal point masses closing head-on along `x` with restitution 1: a
/// single contact (no shared body elsewhere) whose deferred restitution pass
/// must restore the approach speed.
fn head_on_scene() -> (RigidBodyState, Vec<RigidContact>, IntegratorConfig) {
    let mut state = RigidBodyState::new();
    push_body(&mut state, Vec3::new(-0.6, 0.0, 0.0), 1.0, Vec3::ZERO);
    push_body(&mut state, Vec3::new(0.6, 0.0, 0.0), 1.0, Vec3::ZERO);
    state.linear_velocities[0] = Vec3::new(2.0, 0.0, 0.0);
    state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);
    let normal = Vec3::new(-1.0, 0.0, 0.0);
    let contact =
        RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, normal, 0.0).with_restitution(1.0);
    let integrator = IntegratorConfig::new(Vec3::ZERO, 1, 0.0, 0.0);
    (state, vec![contact], integrator)
}

/// A dynamic body struck off its centre against a static wall with restitution
/// 1: the angular arm converts part of the normal impulse into spin, and the
/// per-substep relinearisation rotates that arm.
fn off_centre_scene() -> (RigidBodyState, Vec<RigidContact>, IntegratorConfig) {
    let mut state = RigidBodyState::new();
    push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO);
    push_body(&mut state, Vec3::new(1.0, 0.0, 0.0), 1.0, Vec3::splat(6.0));
    state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);
    let normal = Vec3::new(1.0, 0.0, 0.0);
    let contact = RigidContact::new(1, 0, Vec3::new(-0.5, 0.5, 0.0), Vec3::ZERO, normal, 0.0)
        .with_restitution(1.0);
    let integrator = IntegratorConfig::new(Vec3::ZERO, 1, 0.0, 0.0);
    (state, vec![contact], integrator)
}

/// A floor and two stacked boxes under gravity (restitution 0): two contacts
/// that share dynamic bodies, forcing the static-aware colouring into more than
/// one batch with the load propagating up the stack over many frames.
fn stack_scene() -> (RigidBodyState, Vec<RigidContact>, IntegratorConfig) {
    let mut state = RigidBodyState::new();
    push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO);
    push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::splat(6.0));
    push_body(&mut state, Vec3::new(0.0, 1.5, 0.0), 1.0, Vec3::splat(6.0));
    let up = Vec3::new(0.0, 1.0, 0.0);
    let contacts = vec![
        RigidContact::new(
            1,
            0,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            up,
            0.0,
        )
        .with_friction(0.5),
        RigidContact::new(
            2,
            1,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            up,
            0.0,
        )
        .with_friction(0.5),
    ];
    let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0);
    (state, contacts, integrator)
}

/// Runs `frames` of both engines from the same initial state and contacts,
/// carrying poses, velocities, and accumulated impulses forward so warm
/// starting and the integrated trajectory are exercised, and checks parity
/// after each frame.
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuRigidTgsContactSolver,
    initial_state: &RigidBodyState,
    initial_contacts: &[RigidContact],
    integrator: &IntegratorConfig,
    tgs: &TgsContactConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu_state = initial_state.clone();
    let mut gpu_state = initial_state.clone();
    let mut cpu_contacts = initial_contacts.to_vec();
    let mut gpu_contacts = initial_contacts.to_vec();
    for frame in 0..frames {
        cpu_solve_contacts_tgs(&mut cpu_state, &mut cpu_contacts, integrator, tgs, dt)
            .expect("cpu solve");
        solver
            .solve_contacts_tgs(ctx, &mut gpu_state, &mut gpu_contacts, integrator, tgs, dt)
            .expect("gpu solve");
        let labelled = format!("{scene} (frame {frame})");
        assert_parity(
            &cpu_state,
            &gpu_state,
            &cpu_contacts,
            &gpu_contacts,
            &labelled,
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_rigid_tgs_contact_solver_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU rigid TGS contact parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuRigidTgsContactSolver::new(&ctx);
    let tgs = TgsContactConfig::DEFAULT;

    let (state, contacts, integrator) = resting_box_scene();
    run_parity(
        &ctx, &solver, &state, &contacts, &integrator, &tgs, 120, "resting_box",
    );

    let (state, contacts, integrator) = head_on_scene();
    run_parity(
        &ctx, &solver, &state, &contacts, &integrator, &tgs, 5, "head_on",
    );

    let (state, contacts, integrator) = off_centre_scene();
    run_parity(
        &ctx, &solver, &state, &contacts, &integrator, &tgs, 5, "off_centre",
    );

    let (state, contacts, integrator) = stack_scene();
    run_parity(
        &ctx, &solver, &state, &contacts, &integrator, &tgs, 120, "stack",
    );

    // The rigid Baumgarte limit (zero contact frequency) exercises a different
    // branch of the soft-parameter host precompute; the resting box must track
    // the CPU twin there too.
    let rigid_limit = TgsContactConfig::new(0.0, 10.0, 0.005, 0.5, 1);
    let (state, contacts, integrator) = resting_box_scene();
    run_parity(
        &ctx,
        &solver,
        &state,
        &contacts,
        &integrator,
        &rigid_limit,
        120,
        "resting_box_rigid_limit",
    );
}
