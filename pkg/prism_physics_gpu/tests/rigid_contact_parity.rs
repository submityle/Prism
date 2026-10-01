//! Real-device parity: the `GPU` 6-DOF rigid-body contact solver must reproduce
//! the `CPU` golden twin's result — post-solve velocities *and* the accumulated
//! normal and tangent impulses — within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full colour / upload / dispatch / readback path on any
//! machine with a real device such as an Apple `M`-series `GPU`.
//!
//! Each scene stresses a different corner of the solver: a head-on pair
//! (pure normal impulse, linear-momentum exchange), an off-centre hit (the
//! angular arm that induces spin), a sliding box held by friction (the 2D
//! Coulomb cone), and a resting stack over a static floor (Baumgarte
//! penetration recovery, warm starting, and static-aware colouring across more
//! than one batch). Every scene is advanced for several frames so the
//! accumulated impulse carried by warm starting — and any per-iteration
//! divergence — has room to drift past the tolerance rather than hide under it.
//!
//! Provenance: velocity-level sequential-impulse contact solving with a boxed
//! Coulomb friction cone, Baumgarte stabilisation, restitution, and warm
//! starting (Catto, "Iterative Dynamics with Temporal Coherence", 2005; `Box2D` /
//! Bullet). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_solve_contacts, ContactSolverConfig, GpuContext, GpuRigidContactSolver, RigidBodyState,
    RigidContact,
};

/// Absolute per-quantity divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// Asserts every body's post-solve velocities and every contact's accumulated
/// impulses in `gpu` are within the magnitude-scaled tolerance of `cpu`.
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
        let dn = (cc.normal_impulse - gc.normal_impulse).abs();
        let n_bound = ABS_TOLERANCE + REL_TOLERANCE * cc.normal_impulse.abs();
        assert!(
            dn <= n_bound,
            "{scene}: contact {ci} normal impulse diverged by {dn} (bound {n_bound}): \
             cpu {} vs gpu {}",
            cc.normal_impulse,
            gc.normal_impulse
        );

        let dt0 = (cc.tangent_impulse_0 - gc.tangent_impulse_0).abs();
        let t0_bound = ABS_TOLERANCE + REL_TOLERANCE * cc.tangent_impulse_0.abs();
        assert!(
            dt0 <= t0_bound,
            "{scene}: contact {ci} tangent impulse 0 diverged by {dt0} (bound {t0_bound})"
        );

        let dt1 = (cc.tangent_impulse_1 - gc.tangent_impulse_1).abs();
        let t1_bound = ABS_TOLERANCE + REL_TOLERANCE * cc.tangent_impulse_1.abs();
        assert!(
            dt1 <= t1_bound,
            "{scene}: contact {ci} tangent impulse 1 diverged by {dt1} (bound {t1_bound})"
        );
    }
}

/// A unit sphere's diagonal inverse inertia: `2/5 m r^2` with `m = r = 1`,
/// inverted, is `2.5` on each principal axis.
fn unit_sphere_inv_inertia() -> Vec3 {
    Vec3::splat(2.5)
}

/// Two unit point masses closing head-on along `x`: a single contact whose
/// normal impulse must swap their linear momentum, with no spin.
fn head_on_scene() -> (RigidBodyState, Vec<RigidContact>) {
    let mut state = RigidBodyState::new();
    state.push(
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::IDENTITY,
        1.0,
        unit_sphere_inv_inertia(),
    );
    state.push(
        Vec3::new(0.5, 0.0, 0.0),
        Quat::IDENTITY,
        1.0,
        unit_sphere_inv_inertia(),
    );
    state.linear_velocities[0] = Vec3::new(2.0, 0.0, 0.0);
    state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);

    let contact = RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.02)
        .with_restitution(0.5);
    (state, vec![contact])
}

/// An off-centre strike: the contact anchor sits above the centre of mass, so
/// the normal impulse must induce angular velocity, not just translation.
fn off_centre_scene() -> (RigidBodyState, Vec<RigidContact>) {
    let mut state = RigidBodyState::new();
    state.push(
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::IDENTITY,
        1.0,
        unit_sphere_inv_inertia(),
    );
    state.push(
        Vec3::new(0.5, 0.0, 0.0),
        Quat::IDENTITY,
        1.0,
        unit_sphere_inv_inertia(),
    );
    state.linear_velocities[0] = Vec3::new(3.0, 0.0, 0.0);

    let contact = RigidContact::new(
        0,
        1,
        Vec3::new(0.5, 0.4, 0.0),
        Vec3::new(-0.5, 0.4, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        0.03,
    )
    .with_friction(0.4)
    .with_restitution(0.2);
    (state, vec![contact])
}

/// A box sliding tangentially over a static floor: the normal impulse arrests
/// the downward approach while friction opposes the lateral slide, clamped to
/// the Coulomb cone.
fn sliding_scene() -> (RigidBodyState, Vec<RigidContact>) {
    let mut state = RigidBodyState::new();
    // Static floor: zero inverse mass (translation frozen) and zero inverse
    // inertia (rotation locked).
    state.push(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, 0.0, Vec3::ZERO);
    // Dynamic box resting on the floor, sliding along x and sinking along y.
    state.push(
        Vec3::new(0.0, 1.0, 0.0),
        Quat::IDENTITY,
        1.0,
        Vec3::splat(6.0),
    );
    state.linear_velocities[1] = Vec3::new(1.5, -1.0, 0.0);

    let contact = RigidContact::new(
        0,
        1,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        0.04,
    )
    .with_friction(0.6)
    .with_restitution(0.0);
    (state, vec![contact])
}

/// A three-body stack resting on a static floor: four contacts (floor-box,
/// box-box twice up the stack) that share dynamic bodies, forcing the
/// static-aware colouring into more than one batch, with meaningful warm-start
/// carry across frames.
fn stack_scene() -> (RigidBodyState, Vec<RigidContact>) {
    let mut state = RigidBodyState::new();
    // 0: static floor.
    state.push(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, 0.0, Vec3::ZERO);
    // 1, 2, 3: dynamic boxes stacked with slight interpenetration, all
    // approaching downward under a prior gravity kick.
    state.push(
        Vec3::new(0.0, 1.0, 0.0),
        Quat::IDENTITY,
        1.0,
        Vec3::splat(6.0),
    );
    state.push(
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
        1.0,
        Vec3::splat(6.0),
    );
    state.push(
        Vec3::new(0.0, 3.0, 0.0),
        Quat::IDENTITY,
        1.0,
        Vec3::splat(6.0),
    );
    for i in 1..4 {
        state.linear_velocities[i] = Vec3::new(0.0, -1.0, 0.0);
    }

    let up = Vec3::new(0.0, 1.0, 0.0);
    let contacts = vec![
        // floor <-> box 1
        RigidContact::new(
            0,
            1,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            up,
            0.02,
        )
        .with_friction(0.5),
        // box 1 <-> box 2
        RigidContact::new(
            1,
            2,
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            up,
            0.02,
        )
        .with_friction(0.5),
        // box 2 <-> box 3
        RigidContact::new(
            2,
            3,
            Vec3::new(0.0, 2.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            up,
            0.02,
        )
        .with_friction(0.5),
        // a second floor <-> box 1 anchor off to the side (wider support base)
        RigidContact::new(
            0,
            1,
            Vec3::new(0.3, 0.5, 0.0),
            Vec3::new(0.3, -0.5, 0.0),
            up,
            0.02,
        )
        .with_friction(0.5),
    ];
    (state, contacts)
}

/// Runs `frames` of both engines from the same initial state and contacts,
/// carrying velocities and accumulated impulses forward so warm starting is
/// exercised, and checks parity after each frame.
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuRigidContactSolver,
    initial_state: &RigidBodyState,
    initial_contacts: &[RigidContact],
    config: &ContactSolverConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu_state = initial_state.clone();
    let mut gpu_state = initial_state.clone();
    let mut cpu_contacts = initial_contacts.to_vec();
    let mut gpu_contacts = initial_contacts.to_vec();
    for _ in 0..frames {
        cpu_solve_contacts(&mut cpu_state, &mut cpu_contacts, config, dt).expect("cpu solve");
        solver
            .solve_contacts(ctx, &mut gpu_state, &mut gpu_contacts, config, dt)
            .expect("gpu solve");
        assert_parity(&cpu_state, &gpu_state, &cpu_contacts, &gpu_contacts, scene);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_rigid_contact_solver_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU rigid contact parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuRigidContactSolver::new(&ctx);
    let config = ContactSolverConfig::new(8, 0.2, 0.005, 0.5);

    let (state, contacts) = head_on_scene();
    run_parity(&ctx, &solver, &state, &contacts, &config, 20, "head_on");

    let (state, contacts) = off_centre_scene();
    run_parity(&ctx, &solver, &state, &contacts, &config, 20, "off_centre");

    let (state, contacts) = sliding_scene();
    run_parity(&ctx, &solver, &state, &contacts, &config, 40, "sliding");

    let (state, contacts) = stack_scene();
    run_parity(&ctx, &solver, &state, &contacts, &config, 40, "stack");

    // A high iteration count stresses the per-sweep loop and the warm-start
    // carry on the coupled stack.
    let many_iters = ContactSolverConfig::new(24, 0.2, 0.005, 0.5);
    let (state, contacts) = stack_scene();
    run_parity(
        &ctx,
        &solver,
        &state,
        &contacts,
        &config,
        1,
        "stack_single_frame",
    );
    run_parity(
        &ctx,
        &solver,
        &state,
        &contacts,
        &many_iters,
        30,
        "stack_many_iters",
    );
}
