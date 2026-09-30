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
//! Provenance: substep `XPBD` with the one-sided contact constraint (Müller et
//! al.). No Unreal Engine source or derived code.

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
}
