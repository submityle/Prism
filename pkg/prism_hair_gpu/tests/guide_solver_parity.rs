//! Real-device parity for the guide-strand XPBD solver twin: [`GpuGuideSolver`]
//! must reproduce the `CPU` golden
//! [`simulate_guides`](prism_render_architecture::hair::dynamics::simulate_guides)
//! for a batch of independent guide strands, covering the gravity integration,
//! the compliant edge-length constraint, the discrete-Laplacian bending term,
//! the global goal-pose pull, the one-sided long-range attachment (tether), the
//! analytic sphere/capsule collision push-out, the pinned-root invariant, the
//! per-strand `has_rest` / `has_goal` slice gating, the multi-strand truncation
//! walk, and the no-op guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The solve is closed-form arithmetic (the reference restricts itself to
//! `sqrt`, `min`, `max`, `clamp`, `dot`), so the `CPU` and `GPU` evaluate the
//! same expressions and diverge only through legal fused-multiply-add
//! contraction. Because the solve is *iterated* (substeps × iterations of
//! Gauss-Seidel sweeps), that per-sweep divergence compounds, so parity is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a swapped constraint, a missing clamp, a wrong
//! epsilon or gravity scale), loose enough to admit the compounded contraction.
//! Test strands are built from straight lines, kinks and polynomial curves
//! (never `sin`/`cos`), and each case additionally asserts a non-trivial
//! displacement so a no-op kernel could not pass.
//!
//! Provenance: standard XPBD strand solver plus analytic push-out; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::guide_solver::GpuGuideSolver;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::{
    simulate_guides, StrandParticle, Vec3, XpbdParams,
};

/// Asserts a single component matches within the documented iterative tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Runs `simulate_guides` on a clone of `particles` and asserts the `GPU`
/// result matches it per component.
fn assert_parity(
    ctx: &GpuContext,
    solver: &GpuGuideSolver,
    particles: &[StrandParticle],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    goal_positions: &[Vec3],
    colliders: &[Collider],
    params: XpbdParams,
) -> Vec<StrandParticle> {
    let mut cpu = particles.to_vec();
    simulate_guides(
        &mut cpu,
        strand_lengths,
        rest_lengths,
        goal_positions,
        colliders,
        params,
    );
    let gpu = solver.eval(
        ctx,
        particles,
        strand_lengths,
        rest_lengths,
        goal_positions,
        colliders,
        params,
    );
    assert_eq!(gpu.len(), particles.len(), "one particle per input");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.position.x, c.position.x, &format!("particle {i} pos.x"));
        assert_close(g.position.y, c.position.y, &format!("particle {i} pos.y"));
        assert_close(g.position.z, c.position.z, &format!("particle {i} pos.z"));
        assert_close(
            g.prev_position.x,
            c.prev_position.x,
            &format!("particle {i} prev.x"),
        );
        assert_close(
            g.prev_position.y,
            c.prev_position.y,
            &format!("particle {i} prev.y"),
        );
        assert_close(
            g.prev_position.z,
            c.prev_position.z,
            &format!("particle {i} prev.z"),
        );
    }
    gpu
}

/// Moderate substeps/iterations: enough to exercise the iterated solve while
/// keeping the compounded fused-multiply-add divergence inside tolerance.
fn base_params() -> XpbdParams {
    XpbdParams {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        substeps: 2,
        iterations: 4,
        edge_compliance: 0.0,
        local_stiffness: 0.2,
        global_stiffness: 0.1,
        lra_stiffness: 0.5,
        damping: 0.05,
    }
}

/// Builds a vertical strand of `n` points spaced `spacing` apart with a pinned
/// root, plus the matching per-particle rest lengths and goal positions.
fn vertical_strand(n: usize, spacing: f32) -> (Vec<StrandParticle>, Vec<f32>, Vec<Vec3>) {
    let mut particles = Vec::with_capacity(n);
    let mut rest = Vec::with_capacity(n);
    let mut goals = Vec::with_capacity(n);
    for i in 0..n {
        let y = -(i as f32) * spacing;
        let p = Vec3::new(0.0, y, 0.0);
        if i == 0 {
            particles.push(StrandParticle::pinned(p));
        } else {
            particles.push(StrandParticle::free(p));
        }
        rest.push(spacing);
        goals.push(p);
    }
    (particles, rest, goals)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_strand_gravity_and_constraints_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    let (particles, rest, goals) = vertical_strand(8, 0.1);
    let strand_lengths = [particles.len()];
    let params = base_params();

    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        params,
    );

    // The pinned root must not move; a free tip must have been displaced by
    // gravity, so a no-op kernel could not pass.
    assert_eq!(gpu[0].position, particles[0].position, "root stays pinned");
    let tip_moved = (gpu[7].position.y - particles[7].position.y).abs();
    assert!(
        tip_moved > 1e-5,
        "free tip should fall under gravity: {tip_moved}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_kinked_strand_bending_and_lra_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    // A sharply kinked poly-line so the bending (local) constraint is active,
    // with a tip flung well past its tether radius so LRA pulls it back.
    let mut particles = vec![
        StrandParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        StrandParticle::free(Vec3::new(0.1, -0.02, 0.0)),
        StrandParticle::free(Vec3::new(0.05, -0.15, 0.03)),
        StrandParticle::free(Vec3::new(0.25, -0.1, -0.05)),
        StrandParticle::free(Vec3::new(2.0, -0.2, 0.0)),
    ];
    // Give the free particles a small implicit velocity via prev_position.
    particles[4].prev_position = Vec3::new(1.9, -0.18, 0.0);
    let rest = vec![0.12, 0.12, 0.12, 0.12, 0.0];
    let goals: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let strand_lengths = [particles.len()];

    let mut params = base_params();
    params.local_stiffness = 0.8;
    params.lra_stiffness = 1.0;
    params.global_stiffness = 0.0;

    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        params,
    );
    // The far tip is tethered to a cumulative rest length of 0.48 from the
    // root; LRA must have pulled it much closer than its initial ~2.0.
    let tip_radius = (gpu[4].position.x * gpu[4].position.x
        + gpu[4].position.y * gpu[4].position.y
        + gpu[4].position.z * gpu[4].position.z)
        .sqrt();
    assert!(tip_radius < 1.5, "LRA should reel the tip in: {tip_radius}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sphere_and_capsule_collision_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    // A strand that hangs straight down through a sphere and a capsule so the
    // per-substep push-out projects several particles.
    let (particles, rest, goals) = vertical_strand(9, 0.1);
    let strand_lengths = [particles.len()];
    let colliders = [
        Collider::Sphere {
            center: Vec3::new(0.0, -0.35, 0.0),
            radius: 0.18,
        },
        Collider::Capsule {
            a: Vec3::new(-0.2, -0.7, 0.0),
            b: Vec3::new(0.2, -0.7, 0.0),
            radius: 0.12,
        },
    ];

    let with_colliders = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &colliders,
        base_params(),
    );
    // The same strand solved without colliders must differ from the collided
    // result, proving the push-out branch is genuinely exercised (an on-axis
    // strand is pushed along Y, so the effect shows in the delta, not a
    // sideways offset).
    let without = solver.eval(
        &ctx,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        base_params(),
    );
    let max_delta = with_colliders
        .iter()
        .zip(without.iter())
        .map(|(a, b)| (a.position.y - b.position.y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-4,
        "colliders should change the solved pose: {max_delta}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_with_truncation_and_gating_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    // Three strands laid end to end in one flat pool, then a fourth length that
    // deliberately runs past the pool so the walk truncates.
    let (mut s0, mut r0, mut g0) = vertical_strand(4, 0.1);
    let (s1, r1, g1) = vertical_strand(5, 0.08);
    let (s2, r2, g2) = vertical_strand(3, 0.12);
    s0.extend(s1);
    s0.extend(s2);
    r0.extend(r1);
    r0.extend(r2);
    g0.extend(g1);
    g0.extend(g2);
    let particles = s0;
    let rest = r0;
    let goals = g0;
    // Last length (99) exceeds the remaining pool, so the reference truncates.
    let strand_lengths = [4usize, 5, 3, 99];

    let params = base_params();
    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        params,
    );
    // A free particle in the second strand must have moved, proving multiple
    // strands are advanced (not just the first).
    let moved = (gpu[6].position.y - particles[6].position.y).abs();
    assert!(moved > 1e-6, "second strand should be simulated: {moved}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_missing_goal_slice_disables_global_pull_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    // An empty goal slice must disable the global constraint (has_goal = 0) yet
    // still parity-match the reference, which reads `goal_positions.get` as None.
    let (particles, rest, _goals) = vertical_strand(6, 0.1);
    let strand_lengths = [particles.len()];
    let mut params = base_params();
    params.global_stiffness = 1.0;

    assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &[],
        &[],
        params,
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noop_guards_return_pool_unchanged_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping guide-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuGuideSolver::new(&ctx);

    let (particles, rest, goals) = vertical_strand(5, 0.1);
    let strand_lengths = [particles.len()];

    // Zero substeps → no-op.
    let mut zero_sub = base_params();
    zero_sub.substeps = 0;
    let got = solver.eval(
        &ctx,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        zero_sub,
    );
    assert_eq!(
        got, particles,
        "zero substeps must leave the pool unchanged"
    );

    // Non-positive dt → no-op.
    let mut zero_dt = base_params();
    zero_dt.dt = 0.0;
    let got = solver.eval(
        &ctx,
        &particles,
        &strand_lengths,
        &rest,
        &goals,
        &[],
        zero_dt,
    );
    assert_eq!(
        got, particles,
        "non-positive dt must leave the pool unchanged"
    );

    // Empty pool → empty result, no dispatch.
    let got = solver.eval(&ctx, &[], &[], &[], &[], &[], base_params());
    assert!(got.is_empty(), "empty pool must return empty");
}
