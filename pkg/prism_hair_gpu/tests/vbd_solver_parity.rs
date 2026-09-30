//! Real-device parity for the Vertex Block Descent (VBD) strand solver twin:
//! [`GpuVbdSolver`] must reproduce the `CPU` golden
//! [`simulate_strand_vbd`](prism_render_architecture::hair::solver::simulate_strand_vbd)
//! run on each strand slice of a flat pool, covering the inertial-target
//! prediction, the stretch (edge) spring, the bending term, the PSD Hessian
//! projection under compression, the per-vertex 3x3 Newton solve, the analytic
//! sphere/capsule collision push-out, the pinned-root invariant, the per-strand
//! `has_rest` slice gating, the multi-strand truncation walk, and the no-op
//! guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The solve is closed-form arithmetic (`sqrt`, `min`, `max`, `clamp`, `dot`
//! plus a 3x3 cofactor inverse), so the `CPU` and `GPU` evaluate the same
//! expressions and diverge only through legal fused-multiply-add contraction.
//! Because the solve is *iterated* (substeps × iterations of Gauss-Seidel
//! Newton sweeps), that per-sweep divergence compounds, so parity is asserted
//! to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a
//! genuinely wrong port (a swapped spring, a missing PSD clamp, a wrong epsilon
//! or gravity scale), loose enough to admit the compounded contraction. Test
//! strands are built from straight lines and kinks (never `sin`/`cos`), and
//! each case additionally asserts a non-trivial displacement so a no-op kernel
//! could not pass.
//!
//! Provenance: standard VBD strand solver plus analytic push-out; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::vbd_solver::GpuVbdSolver;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::solver::{simulate_strand_vbd, VbdParams};

/// Asserts a single component matches within the documented iterative tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// The `CPU` batch reference: slice the flat pool into per-strand ranges exactly
/// as [`GpuVbdSolver::eval`] does (truncating a length that runs past the pool)
/// and run [`simulate_strand_vbd`] on each strand with its own rest slice.
fn batch_vbd_reference(
    particles: &[StrandParticle],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    colliders: &[Collider],
    params: VbdParams,
) -> Vec<StrandParticle> {
    let mut pool = particles.to_vec();
    let mut offset = 0usize;
    for &length in strand_lengths {
        let Some(end) = offset.checked_add(length) else {
            break;
        };
        if end > pool.len() {
            break;
        }
        // Mirror the twin's `has_rest` gating: an out-of-range slice disables
        // the springs for the whole strand (empty rest slice).
        let rest_slice: &[f32] = rest_lengths.get(offset..end).unwrap_or(&[]);
        // Take an owned copy of the rest slice so the mutable strand borrow does
        // not alias the shared `rest_lengths`.
        let rest_owned = rest_slice.to_vec();
        simulate_strand_vbd(&mut pool[offset..end], &rest_owned, colliders, params);
        offset = end;
    }
    pool
}

/// Runs the batch reference on a clone of `particles` and asserts the `GPU`
/// result matches it per component.
fn assert_parity(
    ctx: &GpuContext,
    solver: &GpuVbdSolver,
    particles: &[StrandParticle],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    colliders: &[Collider],
    params: VbdParams,
) -> Vec<StrandParticle> {
    let cpu = batch_vbd_reference(particles, strand_lengths, rest_lengths, colliders, params);
    let gpu = solver.eval(
        ctx,
        particles,
        strand_lengths,
        rest_lengths,
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

/// Moderate substeps/iterations and absolute VBD stiffnesses: enough to
/// exercise the iterated Newton solve while keeping the compounded
/// fused-multiply-add divergence inside tolerance.
fn base_params() -> VbdParams {
    VbdParams {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        substeps: 2,
        iterations: 6,
        stretch_stiffness: 500.0,
        bending_stiffness: 30.0,
        damping: 0.05,
    }
}

/// Builds a vertical strand of `n` points spaced `spacing` apart with a pinned
/// root, an interior point kinked sideways so the stretch springs are genuinely
/// loaded, plus the matching per-particle rest lengths (`rest[i]` is the rest
/// length of the segment leaving point `i`).
fn kinked_strand(n: usize, spacing: f32) -> (Vec<StrandParticle>, Vec<f32>) {
    let mut particles = Vec::with_capacity(n);
    let mut rest = Vec::with_capacity(n);
    for i in 0..n {
        let y = -(i as f32) * spacing;
        // Kink one interior point sideways so segments are stretched/compressed
        // and both the tangential and outer-product Hessian branches are hit.
        let x = if i == n / 2 { spacing * 0.6 } else { 0.0 };
        let pos = Vec3::new(x, y, 0.0);
        let p = if i == 0 {
            StrandParticle::pinned(pos)
        } else {
            StrandParticle::free(pos)
        };
        particles.push(p);
        rest.push(spacing);
    }
    (particles, rest)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_strand_stretch_and_bending_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vbd-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuVbdSolver::new(&ctx);

    let (particles, rest) = kinked_strand(7, 0.1);
    let strand_lengths = [particles.len()];

    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &[],
        base_params(),
    );
    // A free interior point must have moved from its kinked start, proving the
    // solve ran (gravity + spring relaxation), not a no-op.
    let moved = (gpu[3].position.x - particles[3].position.x).abs()
        + (gpu[3].position.y - particles[3].position.y).abs();
    assert!(
        moved > 1e-5,
        "the strand should be solved, not static: {moved}"
    );
    // The pinned root must be held exactly in place.
    assert_close(gpu[0].position.x, particles[0].position.x, "root pos.x");
    assert_close(gpu[0].position.y, particles[0].position.y, "root pos.y");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_bending_only_without_rest_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vbd-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuVbdSolver::new(&ctx);

    // An empty rest slice disables the stretch springs (has_rest = 0) yet the
    // bending term and inertia still run; parity must hold against the
    // reference, which reads `rest_lengths.get(i)` as None.
    let (particles, _rest) = kinked_strand(6, 0.1);
    let strand_lengths = [particles.len()];
    let mut params = base_params();
    params.bending_stiffness = 80.0;

    let gpu = assert_parity(&ctx, &solver, &particles, &strand_lengths, &[], &[], params);
    let moved = (gpu[3].position.x - particles[3].position.x).abs()
        + (gpu[3].position.y - particles[3].position.y).abs();
    assert!(
        moved > 1e-5,
        "bending + gravity should move the strand: {moved}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_collision_push_out_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vbd-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuVbdSolver::new(&ctx);

    let (particles, rest) = kinked_strand(8, 0.1);
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
        &colliders,
        base_params(),
    );
    // The same strand solved without colliders must differ, proving the
    // push-out branch is genuinely exercised.
    let without = solver.eval(&ctx, &particles, &strand_lengths, &rest, &[], base_params());
    let max_delta = with_colliders
        .iter()
        .zip(without.iter())
        .map(|(a, b)| {
            (a.position.x - b.position.x).abs()
                + (a.position.y - b.position.y).abs()
                + (a.position.z - b.position.z).abs()
        })
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
fn gpu_multi_strand_with_truncation_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vbd-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuVbdSolver::new(&ctx);

    // Three strands laid end to end in one flat pool, then a fourth length that
    // deliberately runs past the pool so the walk truncates.
    let (mut s0, mut r0) = kinked_strand(4, 0.1);
    let (s1, r1) = kinked_strand(5, 0.08);
    let (s2, r2) = kinked_strand(3, 0.12);
    s0.extend(s1);
    s0.extend(s2);
    r0.extend(r1);
    r0.extend(r2);
    let particles = s0;
    let rest = r0;
    // Last length (99) exceeds the remaining pool, so the reference truncates.
    let strand_lengths = [4usize, 5, 3, 99];

    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        &[],
        base_params(),
    );
    // A free particle in the second strand must have moved, proving multiple
    // strands are advanced (not just the first).
    let moved = (gpu[6].position.x - particles[6].position.x).abs()
        + (gpu[6].position.y - particles[6].position.y).abs();
    assert!(moved > 1e-5, "second strand should be simulated: {moved}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noop_guards_return_pool_unchanged_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vbd-solver parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuVbdSolver::new(&ctx);

    let (particles, rest) = kinked_strand(5, 0.1);
    let strand_lengths = [particles.len()];

    // Zero substeps → no-op.
    let mut zero_sub = base_params();
    zero_sub.substeps = 0;
    let got = solver.eval(&ctx, &particles, &strand_lengths, &rest, &[], zero_sub);
    assert_eq!(
        got, particles,
        "zero substeps must leave the pool unchanged"
    );

    // Non-positive dt → no-op.
    let mut zero_dt = base_params();
    zero_dt.dt = 0.0;
    let got = solver.eval(&ctx, &particles, &strand_lengths, &rest, &[], zero_dt);
    assert_eq!(
        got, particles,
        "non-positive dt must leave the pool unchanged"
    );

    // Empty pool → empty result, no dispatch.
    let got = solver.eval(&ctx, &[], &[], &[], &[], base_params());
    assert!(got.is_empty(), "empty pool must return empty");
}
