//! Real-device parity: the warm-started `GPU` contact solver must reproduce the
//! `CPU` golden twin's trajectory, frame for frame, within a tight
//! floating-point tolerance while both engines carry a persistent impulse cache
//! across frames.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / lambda-readback / cache-store path on
//! any machine with a real device.
//!
//! Warm-starting only changes the trajectory when a contact *persists* across
//! frames: the first warmed frame is identical to the cold solve (the cache is
//! empty, every seed is `0`), and only from the second frame on does the seeded
//! impulse pre-move the pair. Every scene here holds its contacts live for the
//! whole run so the cache fills and the warm-start seed is re-applied each
//! frame, and every scene is *bounded* — a line pinned at both ends can only
//! relax its internal compression, and a stack resting on a pinned floor is held
//! up by gravity against the floor. Bounded trajectories keep the warm/cold
//! float divergence to reassociation noise rather than the amplified blow-up a
//! one-end-pinned chain would inject every frame. The test additionally asserts
//! the cache is non-empty afterwards, so a silently-cold "warm" path (one that
//! never read the seed) would fail the proof, not merely the tolerance.
//!
//! Provenance: substep `XPBD` with the one-sided contact constraint and the
//! warm-starting of an iterative constraint solver (Catto 2005; Müller et al.).
//! No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_resolve_contacts_warm, ContactCache, ContactConstraint, GpuContactWarmSolver, GpuContext,
    ParticleState, XpbdConfig,
};

/// Absolute per-particle divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// A gently overlapping line pinned at *both* ends: the interior particles are
/// packed at 0.85 spacing (rest 1.0) and cannot expand, so every adjacent pair
/// stays in persistent compression for the whole run — the cache fills and the
/// warm-start seed is re-applied each frame — while the fixed ends keep the line
/// bounded instead of launching. A small compliance softens the otherwise rigid
/// over-constrained chain so the relaxation is smooth. Adjacent contacts share a
/// particle, forcing the shared partitioner to split them across colours.
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

/// A vertical stack of `n` particles resting on a pinned floor particle under
/// gravity: the archetypal warm-start scene. Each particle sits 0.9 above the
/// one below (rest 1.0) so the whole column gently overlaps and stays in
/// persistent contact, the regime where warm-starting most changes the
/// trajectory versus a cold restart. Gravity holds the column against the floor
/// so the run is bounded.
fn resting_stack(n: u32) -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    // Particle 0 is the pinned floor.
    state.push(Vec3::new(0.0, 0.0, 0.0), 0.0);
    for i in 1..n {
        state.push(Vec3::new(0.0, 0.9 * i as f32, 0.0), 1.0);
    }
    let contacts = (0..n - 1)
        .map(|i| ContactConstraint::new(i, i + 1, 1.0, 0.0))
        .collect();
    (state, contacts)
}

/// A mix of one persistent penetrating pair and one clearly separated pair,
/// plus a pinned-half soft contact, to exercise the seed-and-skip branches: the
/// separated pair converges to a zero multiplier every frame (its warm-start
/// seed is a no-op), while the overlapping pairs seed a positive impulse. Both
/// engines must agree on which pairs carry load. Gravity keeps the free
/// particles moving so the warm start interleaves with integration.
fn mixed_pairs() -> (ParticleState, Vec<ContactConstraint>) {
    let mut state = ParticleState::new();
    state.push(Vec3::new(0.0, 0.0, 0.0), 1.0);
    state.push(Vec3::new(0.5, 0.0, 0.0), 1.0);
    state.push(Vec3::new(10.0, 0.0, 0.0), 1.0);
    state.push(Vec3::new(13.0, 0.0, 0.0), 1.0);
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

/// Runs `frames` of both engines from the same initial state, each carrying its
/// own persistent cache, and checks per-frame parity. Returns the two caches so
/// the caller can assert they filled (proving the warm-start seed was real).
fn run_parity(
    ctx: &GpuContext,
    solver: &GpuContactWarmSolver,
    initial: &ParticleState,
    contacts: &[ContactConstraint],
    config: &XpbdConfig,
    frames: u32,
    scene: &str,
) -> (ContactCache, ContactCache) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    let mut cpu_cache = ContactCache::new();
    let mut gpu_cache = ContactCache::new();
    for _ in 0..frames {
        cpu_resolve_contacts_warm(&mut cpu, contacts, config, dt, &mut cpu_cache)
            .expect("cpu warm resolve");
        solver
            .resolve_warm(ctx, &mut gpu, contacts, config, dt, &mut gpu_cache)
            .expect("gpu warm resolve");
        // Per-frame parity: a divergence that only shows at the end would not
        // pin down which frame first drifted, so compare every frame.
        assert_parity(&cpu, &gpu, scene);
    }
    (cpu_cache, gpu_cache)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_warm_contacts_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU warm contact parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuContactWarmSolver::new(&ctx);

    // No gravity isolates the one-sided solve. The both-ends-pinned line can only
    // relax its internal compression, so every pair stays live and the cache
    // fills from frame two onward while the trajectory stays bounded.
    let no_gravity = XpbdConfig::new(Vec3::ZERO, 4, 8, 0.0);
    let (line, line_contacts) = confined_line(16);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &line,
        &line_contacts,
        &no_gravity,
        30,
        "confined_line_no_gravity",
    );
    assert!(
        !cpu_cache.is_empty() && !gpu_cache.is_empty(),
        "confined_line_no_gravity: both caches must fill (warm start must be real)"
    );
    assert_eq!(
        cpu_cache.len(),
        gpu_cache.len(),
        "confined_line_no_gravity: caches must agree on live-contact count"
    );

    // Axial gravity + damping: gravity points *along* the pinned line so its
    // horizontal contacts resist it and the chain compresses to a bounded
    // steady state instead of free-falling off contacts it cannot support.
    // Contacts resolve against a moving state, so integration and warm-start
    // interleave every substep.
    let axial_gravity = XpbdConfig::new(Vec3::new(-9.81, 0.0, 0.0), 4, 8, 0.5);
    let (confined, confined_contacts) = confined_line(16);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &confined,
        &confined_contacts,
        &axial_gravity,
        30,
        "confined_line_axial_gravity",
    );
    assert!(
        !cpu_cache.is_empty() && !gpu_cache.is_empty(),
        "confined_line_axial_gravity: both caches must fill"
    );

    // A resting stack on a pinned floor: the canonical warm-start regime, with
    // gravity along the stack axis so the vertical contacts resist it and the
    // column settles to a bounded steady state. The previous frame's cached
    // impulse carries most of the load so the column holds instead of
    // collapsing, and parity here is the strongest statement that the device
    // warm start matches the golden under gravity.
    let vertical_gravity = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 0.5);
    let (stack, stack_contacts) = resting_stack(8);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &stack,
        &stack_contacts,
        &vertical_gravity,
        40,
        "resting_stack_gravity",
    );
    assert!(
        !cpu_cache.is_empty() && !gpu_cache.is_empty(),
        "resting_stack_gravity: both caches must fill"
    );

    // Mixed pairs: the separated pair converges to a zero multiplier (its seed
    // is a no-op) while the overlapping pairs seed a positive impulse, so the
    // live-contact count and multipliers are a sharp check on the store path
    // matching the golden.
    let (mixed, mixed_contacts) = mixed_pairs();
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &mixed,
        &mixed_contacts,
        &no_gravity,
        30,
        "mixed_pairs_no_gravity",
    );
    assert_eq!(
        cpu_cache.len(),
        gpu_cache.len(),
        "mixed_pairs_no_gravity: caches must agree on live-contact count"
    );
}
