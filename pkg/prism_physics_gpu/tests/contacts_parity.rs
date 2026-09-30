//! Real-device parity: the `GPU` one-sided contact solver must reproduce the
//! `CPU` golden twin's trajectory within a tight floating-point tolerance.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / readback path on any machine with a
//! real device.
//!
//! The scenes deliberately avoid the `length ≈ rest` boundary where the
//! `c >= 0` branch could flip between the two engines under low-bit
//! floating-point noise: every pair starts either clearly penetrating (so both
//! engines project it) or clearly separated (so both skip it). A pinned
//! particle checks that the immovable half of a contact takes none of the
//! correction. Adjacent contacts along a line share particles, so the colouring
//! must split them across colours — a direct stress test of the shared
//! partitioner and the one-colour-per-dispatch schedule.
//!
//! A frictional scene additionally exercises the positional Coulomb friction
//! path: free particles slide tangentially while friction (reading the substep
//! `prev_positions` snapshot) damps them, so the parity check covers both the
//! static and dynamic regimes on top of the normal solve.
//!
//! A bouncing scene exercises the velocity-level restitution pass: gently
//! overlapping head-on pairs rebound at a range of coefficients, so the parity
//! check covers the restitution kernel (which reads the post-prediction
//! `vel_pre` snapshot) across both its active impulse and its separated-pair
//! activity-gate skip.
//!
//! Provenance: substep `XPBD` with the one-sided contact constraint, the
//! positional Coulomb friction of Müller et al. 2020, and the substep
//! velocity-level restitution of that same 2020 work. No Unreal Engine source or
//! derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_resolve_contacts, ContactConstraint, GpuContactSolver, GpuContext, ParticleState,
    XpbdConfig,
};

/// Absolute per-particle divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so a fixed absolute bound is wrong for the large velocities a rigid contact
/// briefly produces; the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// A line of `n` unit-diameter particles packed at 0.85 spacing so every
/// adjacent pair gently overlaps (rest 1.0 vs actual 0.85). Particle 0 is
/// pinned. Adjacent contacts share a particle, forcing a two-colour split. The
/// gentle overlap keeps the free relaxation stable rather than explosive.
fn overlapping_line(n: u32) -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.85 * i as f32, 0.0, 0.0), inv_mass);
    }
    let contacts = (0..n - 1)
        .map(|i| ContactConstraint::new(i, i + 1, 1.0, 0.0))
        .collect();
    (state, contacts)
}

/// A gently overlapping line pinned at *both* ends, so the run resolves against
/// gravity while staying confined: the free particles sag under gravity but the
/// contacts keep the line bounded instead of flinging it away, giving a stable
/// moving-state parity check. Adjacent contacts still share particles.
fn confined_line(n: u32) -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 || i == n - 1 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.85 * i as f32, 0.0, 0.0), inv_mass);
    }
    let contacts = (0..n - 1)
        .map(|i| ContactConstraint::new(i, i + 1, 1.0, 1.0e-6))
        .collect();
    (state, contacts)
}

/// A mix of one clearly penetrating pair and one clearly separated pair, plus a
/// compliant (soft) penetrating pair, to exercise the skip branch and a
/// non-zero compliance simultaneously.
fn mixed_pairs() -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    // Pair (0, 1): penetrating, both free.
    state.push(Vec3::new(0.0, 0.0, 0.0), 1.0);
    state.push(Vec3::new(0.5, 0.0, 0.0), 1.0);
    // Pair (2, 3): clearly separated (3.0 apart, rest 1.0): must be skipped.
    state.push(Vec3::new(10.0, 0.0, 0.0), 1.0);
    state.push(Vec3::new(13.0, 0.0, 0.0), 1.0);
    // Pair (4, 5): penetrating, particle 4 pinned, soft compliance.
    state.push(Vec3::new(-5.0, 0.0, 0.0), 0.0);
    state.push(Vec3::new(-4.4, 0.0, 0.0), 1.0);
    let contacts = vec![
        ContactConstraint::new(0, 1, 1.0, 0.0),
        ContactConstraint::new(2, 3, 1.0, 0.0),
        ContactConstraint::new(4, 5, 1.0, 1.0e-6),
    ];
    (state, contacts)
}

/// A gently overlapping line (particle 0 pinned) whose free particles each carry
/// a tangential (+z) initial velocity, with Coulomb friction on every contact.
/// The friction correction reads `prev_positions` and applies a mass-weighted
/// tangential push, so this scene is the direct parity probe for the friction
/// path on top of the normal solve; adjacent contacts share particles, keeping
/// the two-colour split under test. A mix of static- and dynamic-regime drifts
/// arises naturally as the velocities decay frame to frame.
fn frictional_slide(n: u32) -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.9 * i as f32, 0.0, 0.0), inv_mass);
        if i != 0 {
            state.velocities[i as usize] = Vec3::new(0.0, 0.0, 3.0);
        }
    }
    let contacts = (0..n - 1)
        .map(|i| ContactConstraint::new(i, i + 1, 1.0, 0.0).with_friction(0.6, 0.4))
        .collect();
    (state, contacts)
}

/// Three independent head-on bouncers: each is a pinned particle and a movable
/// particle gently overlapping it (rest 1.0 vs 0.95 spacing) and approaching at
/// a modest -x speed, with a different restitution coefficient per pair. The
/// overlap drives the position solve while the restitution pass (reading the
/// post-prediction `vel_pre` snapshot) corrects the rebound speed to the
/// coefficient's target; the pairs then coast apart, so the run stays bounded
/// and non-chaotic while exercising both the active restitution impulse and the
/// separated-pair activity-gate skip. The bouncers are spaced far apart so the
/// contacts stay disjoint — a clean isolation of the restitution kernel.
fn bouncing_pairs() -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    for &px in &[0.0f32, 10.0, 20.0] {
        state.push(Vec3::new(px, 0.0, 0.0), 0.0);
        state.push(Vec3::new(px + 0.95, 0.0, 0.0), 1.0);
        let movable = state.len() - 1;
        state.velocities[movable] = Vec3::new(-0.5, 0.0, 0.0);
    }
    let contacts = vec![
        ContactConstraint::new(0, 1, 1.0, 0.0).with_restitution(0.5),
        ContactConstraint::new(2, 3, 1.0, 0.0).with_restitution(0.7),
        ContactConstraint::new(4, 5, 1.0, 0.0).with_restitution(0.9),
    ];
    (state, contacts)
}

/// Asserts every particle in `gpu` is within the magnitude-scaled tolerance of
/// `cpu` in both position and velocity.
fn assert_parity(cpu: &ParticleState, gpu: &ParticleState, scene: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{scene}: particle counts differ");
    for i in 0..cpu.len() {
        let dp = (cpu.positions[i] - gpu.positions[i]).length();
        let pos_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu.positions[i].length();
        assert!(
            dp <= pos_bound,
            "{scene}: particle {i} position diverged by {dp} (bound {pos_bound}): cpu {:?} vs gpu {:?}",
            cpu.positions[i],
            gpu.positions[i]
        );
        let dv = (cpu.velocities[i] - gpu.velocities[i]).length();
        let vel_bound = ABS_TOLERANCE + REL_TOLERANCE * cpu.velocities[i].length();
        assert!(
            dv <= vel_bound,
            "{scene}: particle {i} velocity diverged by {dv} (bound {vel_bound}): cpu {:?} vs gpu {:?}",
            cpu.velocities[i],
            gpu.velocities[i]
        );
    }
}

/// Runs `frames` of both engines from the same initial state and checks parity.
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuContactSolver,
    initial: &ParticleState,
    contacts: &[ContactConstraint],
    config: &XpbdConfig,
    frames: u32,
    scene: &str,
) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    for _ in 0..frames {
        cpu_resolve_contacts(&mut cpu, contacts, config, dt).expect("cpu resolve");
        solver
            .resolve(ctx, &mut gpu, contacts, config, dt)
            .expect("gpu resolve");
    }
    assert_parity(&cpu, &gpu, scene);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_contacts_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU contact parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuContactSolver::new(&ctx);

    // No gravity: isolate the contact projection from free-fall so the parity
    // is a clean test of the one-sided solve itself.
    let no_gravity = XpbdConfig::new(Vec3::ZERO, 4, 8, 0.0);
    let (line, line_contacts) = overlapping_line(16);
    run_parity(
        &ctx,
        &solver,
        &line,
        &line_contacts,
        &no_gravity,
        30,
        "overlapping_line_no_gravity",
    );

    // With gravity and damping: contacts resolve against a moving state, so the
    // skip branch and the correction interleave with integration every substep.
    // The line is pinned at both ends so it stays confined instead of diverging
    // chaotically, keeping the parity comparison meaningful over many frames.
    let gravity = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 0.5);
    let (confined, confined_contacts) = confined_line(16);
    run_parity(
        &ctx,
        &solver,
        &confined,
        &confined_contacts,
        &gravity,
        30,
        "confined_line_gravity",
    );

    let (mixed, mixed_contacts) = mixed_pairs();
    run_parity(
        &ctx,
        &solver,
        &mixed,
        &mixed_contacts,
        &gravity,
        30,
        "mixed_pairs",
    );

    // Frictional sliding line: isolate friction from gravity so the parity is a
    // clean test of the tangential correction, then run long enough for the
    // velocities to decay through both the dynamic and static regimes.
    let (slide, slide_contacts) = frictional_slide(12);
    run_parity(
        &ctx,
        &solver,
        &slide,
        &slide_contacts,
        &no_gravity,
        40,
        "frictional_slide_no_gravity",
    );

    // Bouncing pairs: independent head-on bouncers with a range of restitution
    // coefficients. No gravity isolates the restitution kernel's velocity
    // correction; each pair rebounds once and coasts apart, so the run is
    // bounded and non-chaotic while covering both the active impulse and the
    // activity-gate skip once the pair separates.
    let (bounce, bounce_contacts) = bouncing_pairs();
    run_parity(
        &ctx,
        &solver,
        &bounce,
        &bounce_contacts,
        &no_gravity,
        40,
        "bouncing_pairs_no_gravity",
    );
}
