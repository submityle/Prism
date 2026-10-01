//! Real-device parity for the Dynamic Follow-The-Leader strand-solver twin:
//! [`GpuHairDftl`] must reproduce the `CPU` golden
//! [`simulate_guides_dftl`](prism_render_architecture::hair::dftl::simulate_guides_dftl)
//! for a batch of independent guide strands, covering the semi-implicit gravity
//! prediction, the root-to-tip `FTL` length snap, Müller's follower velocity
//! correction, the pinned-root invariant, the per-strand `has_rest` slice
//! gating, the collapsed-segment safe-axis fallback, the on-device parameter
//! sanitation, the multi-strand truncation walk, and the no-op guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The solve is closed-form arithmetic (the reference restricts itself to
//! `sqrt`, `dot`, `min`, `max`, `clamp`), so the `CPU` and `GPU` evaluate the
//! same expressions and diverge only through legal fused-multiply-add
//! contraction. Because the solve is *iterated* (over `substeps`), that
//! per-step divergence compounds, so parity is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped phase, a missing clamp, a wrong epsilon or gravity
//! scale), loose enough to admit the compounded contraction. Test strands are
//! laid out horizontally (a vertical strand under vertical gravity has no net
//! `FTL` displacement and could not catch a broken length pass), and each case
//! additionally asserts a non-trivial displacement so a no-op kernel could not
//! pass.
//!
//! Provenance: standard `DFTL` inextensible-strand integrator; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::dftl::GpuHairDftl;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dftl::{simulate_guides_dftl, DftlParams, FtlParticle, Vec3};

/// Asserts a single component matches within the documented iterative tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Runs `simulate_guides_dftl` on a clone of `particles` and asserts the `GPU`
/// result matches it per position and velocity component.
fn assert_parity(
    ctx: &GpuContext,
    solver: &GpuHairDftl,
    particles: &[FtlParticle],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    params: DftlParams,
) -> Vec<FtlParticle> {
    let mut cpu = particles.to_vec();
    simulate_guides_dftl(&mut cpu, strand_lengths, rest_lengths, params);
    let gpu = solver.eval(ctx, particles, strand_lengths, rest_lengths, params);
    assert_eq!(gpu.len(), particles.len(), "one particle per input");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.position.x, c.position.x, &format!("particle {i} pos.x"));
        assert_close(g.position.y, c.position.y, &format!("particle {i} pos.y"));
        assert_close(g.position.z, c.position.z, &format!("particle {i} pos.z"));
        assert_close(g.velocity.x, c.velocity.x, &format!("particle {i} vel.x"));
        assert_close(g.velocity.y, c.velocity.y, &format!("particle {i} vel.y"));
        assert_close(g.velocity.z, c.velocity.z, &format!("particle {i} vel.z"));
        assert!(
            (g.inverse_mass - c.inverse_mass).abs() < 1e-6,
            "particle {i} inverse_mass carried through"
        );
    }
    gpu
}

/// A substepped gravity parameter set exercising the iterated solve.
fn base_params() -> DftlParams {
    DftlParams {
        dt: 1.0 / 60.0,
        substeps: 4,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.05,
        correction: 0.9,
    }
}

/// Builds a *horizontal* strand of `n` points spaced `spacing` apart along `+X`
/// with a pinned root, plus the matching per-particle rest lengths. A
/// horizontal layout under vertical gravity produces a genuine `FTL` length
/// snap (a vertical strand would stay collinear with gravity and barely move).
fn horizontal_strand(n: usize, spacing: f32) -> (Vec<FtlParticle>, Vec<f32>) {
    let mut particles = Vec::with_capacity(n);
    let mut rest = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f32) * spacing;
        let p = Vec3::new(x, 0.0, 0.0);
        if i == 0 {
            particles.push(FtlParticle::pinned(p));
        } else {
            particles.push(FtlParticle::free(p));
        }
        // Entry `i` is the length of the segment leaving particle `i`; the last
        // particle has no outgoing segment and its entry is ignored.
        rest.push(spacing);
    }
    (particles, rest)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_horizontal_strand_swings_under_gravity_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dftl parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairDftl::new(&ctx);

    let (particles, rest) = horizontal_strand(8, 0.1);
    let strand_lengths = [particles.len()];

    let gpu = assert_parity(
        &ctx,
        &solver,
        &particles,
        &strand_lengths,
        &rest,
        base_params(),
    );
    // The root stayed pinned and a free tip swung down under gravity, proving
    // the integration ran rather than returning the pool unchanged.
    assert_eq!(gpu[0].position, particles[0].position, "root stays pinned");
    let dropped = particles[7].position.y - gpu[7].position.y;
    assert!(dropped > 1e-4, "tip should fall under gravity: {dropped}");
    // The outgoing segment kept (close to) its rest length, proving the FTL
    // length snap fired.
    let seg = gpu[1].position.sub(gpu[0].position).length();
    assert!((seg - 0.1).abs() < 1e-3, "segment length preserved: {seg}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_strand_with_truncation_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dftl parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairDftl::new(&ctx);

    // Three strands laid end to end in one flat pool, then a fourth length that
    // deliberately runs past the pool so the walk truncates.
    let (mut s0, mut r0) = horizontal_strand(4, 0.1);
    let (s1, r1) = horizontal_strand(5, 0.08);
    let (s2, r2) = horizontal_strand(3, 0.12);
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
        base_params(),
    );
    // A free particle in the second strand (index 4 is its pinned root, 5 is a
    // free follower) must have moved, proving multiple strands are advanced.
    let moved = (gpu[6].position.y - particles[6].position.y).abs();
    assert!(moved > 1e-6, "second strand should be simulated: {moved}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_non_finite_params_are_sanitized_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dftl parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairDftl::new(&ctx);

    // Poisoned parameters: non-finite dt, zero substeps, part-non-finite
    // gravity, non-finite damping and an out-of-range correction. The host
    // uploads them raw; the device must sanitize them to the same stable domain
    // the CPU `DftlParams::sanitized` uses, so the two still parity-match. A
    // non-zero initial velocity ensures there is motion to compare.
    let particles = vec![
        FtlParticle {
            position: Vec3::new(0.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            inverse_mass: 0.0,
        },
        FtlParticle {
            position: Vec3::new(0.1, 0.0, 0.0),
            velocity: Vec3::new(0.0, -0.5, 0.2),
            inverse_mass: 1.0,
        },
        FtlParticle {
            position: Vec3::new(0.2, 0.0, 0.0),
            velocity: Vec3::new(0.1, -0.3, 0.0),
            inverse_mass: 1.0,
        },
    ];
    let rest = [0.1, 0.1, 0.1];
    let strand_lengths = [particles.len()];
    let params = DftlParams {
        dt: f32::NAN,
        substeps: 0,
        gravity: Vec3::new(f32::INFINITY, f32::NAN, 0.0),
        damping: f32::NAN,
        correction: 5.0,
    };

    let gpu = assert_parity(&ctx, &solver, &particles, &strand_lengths, &rest, params);
    for (i, p) in gpu.iter().enumerate() {
        assert!(p.position.is_finite(), "particle {i} position finite");
        assert!(p.velocity.is_finite(), "particle {i} velocity finite");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_length_segment_falls_back_to_safe_axis_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dftl parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairDftl::new(&ctx);

    // Parent and child coincide, so the FTL direction is undefined and must
    // fall back to the +X safe axis instead of producing NaN. No gravity keeps
    // the collapse the only driver so the fallback is unambiguous.
    let particles = vec![
        FtlParticle::pinned(Vec3::ZERO),
        FtlParticle::free(Vec3::ZERO),
    ];
    let rest = [2.0, 0.0];
    let strand_lengths = [particles.len()];
    let params = DftlParams {
        dt: 1.0 / 60.0,
        substeps: 1,
        gravity: Vec3::ZERO,
        damping: 0.0,
        correction: 0.9,
    };

    let gpu = assert_parity(&ctx, &solver, &particles, &strand_lengths, &rest, params);
    // The child was snapped onto the safe axis at the rest distance.
    assert!(
        (gpu[1].position.x - 2.0).abs() < 1e-4,
        "child snapped to +X safe axis: {}",
        gpu[1].position.x
    );
    assert!(
        gpu[1].position.is_finite(),
        "no NaN from zero-length segment"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noop_guards_return_pool_unchanged_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping dftl parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairDftl::new(&ctx);

    // Empty pool → empty result, no dispatch.
    let got = solver.eval(&ctx, &[], &[], &[], base_params());
    assert!(got.is_empty(), "empty pool must return empty");

    // A sub-two-particle strand has no segment to constrain and must come back
    // exactly as uploaded.
    let single = vec![FtlParticle::free(Vec3::new(1.0, 2.0, 3.0))];
    let got = solver.eval(&ctx, &single, &[1usize], &[], base_params());
    assert_eq!(got, single, "single-particle strand is a no-op");

    // A strand_lengths list whose only entry overruns the pool fits no strand,
    // so no descriptor is built and the pool returns unchanged.
    let (particles, rest) = horizontal_strand(3, 0.1);
    let got = solver.eval(&ctx, &particles, &[99usize], &rest, base_params());
    assert_eq!(got, particles, "overrunning length fits no strand");
}
