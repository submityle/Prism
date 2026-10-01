//! Real-device parity: the warm-started `GPU` `XPBD` distance solver must
//! reproduce the `CPU` golden twin's trajectory, frame for frame, within a tight
//! floating-point tolerance while both engines carry a persistent multiplier
//! cache across frames.
//!
//! On a headless host with no `wgpu` adapter the test skips (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full upload / dispatch / lambda-readback / cache-store path on
//! any machine with a real device.
//!
//! Warm-starting only changes the trajectory once a constraint *persists* across
//! frames: the first warmed frame is identical to the cold solve (the cache is
//! empty, every seed is `0`), and only from the second frame on does the seeded
//! multiplier pre-move the pair. Every scene here holds its constraints live for
//! the whole run so the cache fills and the warm-start seed is re-applied each
//! frame. Unlike the one-sided contact, a distance constraint is two-sided, so
//! both a compressed pair (negative multiplier) and a stretched pair (positive
//! multiplier) exercise the signed cache. Every scene is *bounded* — the line is
//! pinned at both ends, so it can only relax its internal stretch or compression
//! and never launches, which keeps the warm/cold float divergence to
//! reassociation noise rather than an amplified blow-up. The test additionally
//! asserts the cache is non-empty afterwards, so a silently-cold "warm" path
//! (one that never read the seed) would fail the proof, not merely the
//! tolerance.
//!
//! Provenance: substep `XPBD` with the canonical stretch constraint and the
//! warm-starting of an iterative constraint solver (Müller et al.). No Unreal
//! Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_solve_warm, DistanceCache, DistanceConstraint, GpuContext, GpuXpbdWarmSolver,
    ParticleState, XpbdConfig,
};

/// Absolute per-particle divergence floor between the two engines.
const ABS_TOLERANCE: f32 = 1e-3;

/// Relative divergence bound, scaled by the reference magnitude. `GPU`
/// floating-point reassociation (fused multiply-add, differing division and
/// square-root rounding) perturbs each result in proportion to its magnitude,
/// so the allowed error is `ABS_TOLERANCE + REL_TOLERANCE * |ref|`.
const REL_TOLERANCE: f32 = 1e-4;

/// A compressed line pinned at *both* ends: the interior particles are packed at
/// 0.85 spacing (rest 1.0) so every adjacent pair stays in persistent
/// compression for the whole run — the cache fills with signed (here, negative)
/// multipliers and the warm-start seed is re-applied each frame — while the
/// fixed ends keep the line bounded instead of expanding away. A small
/// compliance softens the otherwise rigid over-constrained chain so the
/// relaxation is smooth. Adjacent constraints share a particle, forcing the
/// shared partitioner to split them across colours.
fn confined_line(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 || i == n - 1 { 0.0 } else { 1.0 };
        state.push(Vec3::new(0.85 * i as f32, 0.0, 0.0), inv_mass);
    }
    let constraints = (0..n - 1)
        .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 1.0e-6))
        .collect();
    (state, constraints)
}

/// A stretched line pinned at *both* ends: the interior particles are packed at
/// 1.15 spacing (rest 1.0) so every adjacent pair stays in persistent *tension*
/// for the whole run, caching positive multipliers. The two-sided distance
/// constraint pulls the pair together, and the fixed ends bound the motion.
fn stretched_line(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
    let mut state = ParticleState::new();
    for i in 0..n {
        let inv_mass = if i == 0 || i == n - 1 { 0.0 } else { 1.0 };
        state.push(Vec3::new(1.15 * i as f32, 0.0, 0.0), inv_mass);
    }
    let constraints = (0..n - 1)
        .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 1.0e-6))
        .collect();
    (state, constraints)
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
    solver: &GpuXpbdWarmSolver,
    initial: &ParticleState,
    constraints: &[DistanceConstraint],
    config: &XpbdConfig,
    frames: u32,
    scene: &str,
) -> (DistanceCache, DistanceCache) {
    let dt = 1.0 / 60.0;
    let mut cpu = initial.clone();
    let mut gpu = initial.clone();
    let mut cpu_cache = DistanceCache::new();
    let mut gpu_cache = DistanceCache::new();
    for _ in 0..frames {
        cpu_solve_warm(&mut cpu, constraints, config, dt, &mut cpu_cache).expect("cpu warm solve");
        solver
            .solve_warm(ctx, &mut gpu, constraints, config, dt, &mut gpu_cache)
            .expect("gpu warm solve");
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
fn gpu_warm_xpbd_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU warm XPBD parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuXpbdWarmSolver::new(&ctx);

    // No gravity isolates the stretch solve. The both-ends-pinned compressed
    // line can only relax its internal compression, so every pair stays live and
    // the cache fills with negative multipliers from frame two onward while the
    // trajectory stays bounded.
    let no_gravity = XpbdConfig::new(Vec3::ZERO, 4, 8, 0.0);
    let (line, line_cons) = confined_line(16);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &line,
        &line_cons,
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
        "confined_line_no_gravity: caches must agree on live-constraint count"
    );

    // A stretched line (positive multipliers) under no gravity exercises the
    // opposite sign of the signed cache from the compressed line above.
    let (stretched, stretched_cons) = stretched_line(16);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &stretched,
        &stretched_cons,
        &no_gravity,
        30,
        "stretched_line_no_gravity",
    );
    assert!(
        !cpu_cache.is_empty() && !gpu_cache.is_empty(),
        "stretched_line_no_gravity: both caches must fill"
    );

    // Axial gravity + damping: gravity points *along* the pinned line so its
    // horizontal constraints resist it and the chain settles to a bounded steady
    // state instead of free-falling. Constraints resolve against a moving state,
    // so integration and warm-start interleave every substep.
    let axial_gravity = XpbdConfig::new(Vec3::new(-9.81, 0.0, 0.0), 4, 8, 0.5);
    let (confined, confined_cons) = confined_line(16);
    let (cpu_cache, gpu_cache) = run_parity(
        &ctx,
        &solver,
        &confined,
        &confined_cons,
        &axial_gravity,
        40,
        "confined_line_axial_gravity",
    );
    assert!(
        !cpu_cache.is_empty() && !gpu_cache.is_empty(),
        "confined_line_axial_gravity: both caches must fill"
    );
}
