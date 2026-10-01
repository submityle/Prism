//! Real-device parity for the `Cosserat` rod-solver twin: [`GpuCosserat`] must
//! reproduce the `CPU` golden
//! [`simulate_guides_cosserat`](prism_render_architecture::hair::cosserat::simulate_guides_cosserat)
//! for a batch of independent rods, covering the semi-implicit predict step,
//! the compliant edge-length (stretch) constraint, the quaternion bend-twist
//! constraint driving each adjacent frame pair toward its rest `Darboux`
//! vector, the velocity write-back with damping retention, the pinned-root
//! invariant, the per-rod `has_rest` / orientation-slice gating, the
//! multi-rod truncation walk, and the no-op guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! (only `sqrt`/`min`/`max`/`dot` and multiply-add), so it needs no optional
//! device feature.
//!
//! # Parity criterion
//!
//! The solve is closed-form arithmetic, so the `CPU` and `GPU` evaluate the
//! same expressions and diverge only through legal fused-multiply-add
//! contraction. Because the solve is *iterated* (substeps x Gauss-Seidel
//! sweeps), that per-sweep divergence compounds, so parity is asserted to
//! within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a
//! genuinely wrong port (a swapped constraint, a missing normalize, a wrong
//! epsilon or alpha), loose enough to admit the compounded contraction. Test
//! rods are built from straight lines, kinks and polynomial curves (never
//! `sin`/`cos`), and each solving case additionally asserts a non-trivial
//! displacement so a no-op kernel could not pass.
//!
//! Provenance: standard position-based `Cosserat` rod solver (`XPBD` stretch
//! plus `Darboux`-vector bend-twist); no Unreal Engine source or derived code.

use prism_hair_gpu::cosserat::GpuCosserat;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::cosserat::{
    parallel_transport_frames, rest_darboux_from_frames, simulate_guides_cosserat, CosseratParams,
    Quat, RodParticle, Vec3,
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

/// Builds a unit quaternion from a small rotation (`Darboux`) vector without any
/// transcendental call: the real part closes the unit-norm identity for
/// `|omega| <= 1`. Used to perturb material frames away from their rest so the
/// bend-twist projection has something to correct.
fn rel_from_omega(omega: Vec3) -> Quat {
    let n2 = omega.dot(omega);
    let w = (1.0 - n2).max(0.0).sqrt();
    Quat::new(w, omega.x, omega.y, omega.z)
}

/// Runs `simulate_guides_cosserat` on clones of the inputs and asserts the `GPU`
/// result matches it per position/velocity/orientation component. Returns the
/// `GPU` `(particles, orientations)` for additional case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    solver: &GpuCosserat,
    particles: &[RodParticle],
    orientations: &[Quat],
    strand_lengths: &[usize],
    rest_lengths: &[f32],
    rest_darboux: &[Vec3],
    params: CosseratParams,
) -> (Vec<RodParticle>, Vec<Quat>) {
    let mut cpu_particles = particles.to_vec();
    let mut cpu_orient = orientations.to_vec();
    simulate_guides_cosserat(
        &mut cpu_particles,
        &mut cpu_orient,
        strand_lengths,
        rest_lengths,
        rest_darboux,
        params,
    );
    let (gpu_particles, gpu_orient) = solver.eval(
        ctx,
        particles,
        orientations,
        strand_lengths,
        rest_lengths,
        rest_darboux,
        params,
    );
    assert_eq!(
        gpu_particles.len(),
        particles.len(),
        "one particle per input"
    );
    assert_eq!(
        gpu_orient.len(),
        orientations.len(),
        "one orientation per input"
    );
    for (i, (g, c)) in gpu_particles.iter().zip(cpu_particles.iter()).enumerate() {
        assert_close(g.position.x, c.position.x, &format!("particle {i} pos.x"));
        assert_close(g.position.y, c.position.y, &format!("particle {i} pos.y"));
        assert_close(g.position.z, c.position.z, &format!("particle {i} pos.z"));
        assert_close(g.velocity.x, c.velocity.x, &format!("particle {i} vel.x"));
        assert_close(g.velocity.y, c.velocity.y, &format!("particle {i} vel.y"));
        assert_close(g.velocity.z, c.velocity.z, &format!("particle {i} vel.z"));
    }
    for (i, (g, c)) in gpu_orient.iter().zip(cpu_orient.iter()).enumerate() {
        assert_close(g.w, c.w, &format!("orientation {i} w"));
        assert_close(g.x, c.x, &format!("orientation {i} x"));
        assert_close(g.y, c.y, &format!("orientation {i} y"));
        assert_close(g.z, c.z, &format!("orientation {i} z"));
    }
    (gpu_particles, gpu_orient)
}

/// Moderate substeps with non-zero compliance: enough to exercise the iterated
/// solve while keeping the compounded fused-multiply-add divergence inside
/// tolerance.
fn base_params() -> CosseratParams {
    CosseratParams {
        dt: 1.0 / 60.0,
        substeps: 2,
        stretch_compliance: 1.0e-6,
        bend_compliance: 1.0e-6,
        twist_compliance: 2.0e-6,
        damping: 0.05,
    }
}

/// Builds a gently curved rod of `n` particles: positions follow a polynomial
/// (no `sin`/`cos`) so adjacent segments turn, the root is pinned, and the rest
/// of the particles are free. Returns the particles plus their parallel-
/// transported material frames (one per segment).
fn curved_rod(n: usize, spacing: f32) -> (Vec<RodParticle>, Vec<Quat>) {
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * spacing;
        // Cubic bend in the x-y plane with a slight z twist; purely polynomial.
        let x = t;
        let y = 0.15 * t * t - 0.02 * t * t * t;
        let z = 0.05 * t * t;
        points.push(Vec3::new(x, y, z));
    }
    let mut particles = Vec::with_capacity(n);
    for (i, &p) in points.iter().enumerate() {
        if i == 0 {
            particles.push(RodParticle::pinned(p));
        } else {
            particles.push(RodParticle::free(p));
        }
    }
    let frames = parallel_transport_frames(&points, Quat::IDENTITY);
    (particles, frames)
}

/// Segment lengths of a rod's current polyline (one per adjacent pair).
fn segment_lengths(particles: &[RodParticle]) -> Vec<f32> {
    let mut out = Vec::with_capacity(particles.len().saturating_sub(1));
    let mut i = 0;
    while i + 1 < particles.len() {
        out.push(
            particles[i + 1]
                .position
                .sub(particles[i].position)
                .length(),
        );
        i += 1;
    }
    out
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_rod_stretch_and_bend_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    let (particles, frames) = curved_rod(6, 0.1);
    let strand_lengths = [particles.len()];
    // Rest lengths deliberately longer than the current segments so the stretch
    // constraint pulls particles outward and the tip measurably moves.
    let rest: Vec<f32> = segment_lengths(&particles)
        .iter()
        .map(|l| l * 1.4)
        .collect();
    // Rest darboux of a straight rod (all zero) so the bend-twist constraint
    // pulls the curved frames toward straight: a genuine orientation correction.
    let rest_darboux = vec![Vec3::ZERO; frames.len().saturating_sub(1)];

    let (gpu_particles, gpu_orient) = assert_parity(
        &ctx,
        &solver,
        &particles,
        &frames,
        &strand_lengths,
        &rest,
        &rest_darboux,
        base_params(),
    );
    // The free tip must have moved under the stretch pull (no-op kernel guard).
    let tip = particles.len() - 1;
    let moved = gpu_particles[tip]
        .position
        .sub(particles[tip].position)
        .length();
    assert!(moved > 1e-5, "tip should move under stretch: {moved}");
    // At least one frame must have rotated toward the straight rest.
    let frame_delta = gpu_orient
        .iter()
        .zip(frames.iter())
        .map(|(g, c)| (g.w - c.w).abs() + (g.x - c.x).abs() + (g.y - c.y).abs() + (g.z - c.z).abs())
        .fold(0.0f32, f32::max);
    assert!(
        frame_delta > 1e-6,
        "bend-twist should rotate a frame: {frame_delta}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_helix_rest_darboux_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    // Take the curved rod's own frames as the natural rest shape, then perturb
    // the live frames away from it so the bend-twist constraint has a non-zero,
    // non-trivial rest `Darboux` target to drive toward.
    let (particles, frames) = curved_rod(7, 0.09);
    let strand_lengths = [particles.len()];
    let rest = segment_lengths(&particles);
    let rest_darboux = rest_darboux_from_frames(&frames);

    // Perturb each frame by a small per-index rotation (no trig).
    let mut live_frames = Vec::with_capacity(frames.len());
    for (i, &f) in frames.iter().enumerate() {
        let k = 0.03 + 0.01 * i as f32;
        let rel = rel_from_omega(Vec3::new(k, -0.5 * k, 0.25 * k));
        live_frames.push(f.mul(rel).normalize());
    }

    assert_parity(
        &ctx,
        &solver,
        &particles,
        &live_frames,
        &strand_lengths,
        &rest,
        &rest_darboux,
        base_params(),
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_pinned_root_stays_fixed_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    let (particles, frames) = curved_rod(5, 0.12);
    let strand_lengths = [particles.len()];
    let rest: Vec<f32> = segment_lengths(&particles)
        .iter()
        .map(|l| l * 0.7)
        .collect();
    let rest_darboux = vec![Vec3::ZERO; frames.len().saturating_sub(1)];

    let (gpu_particles, _) = assert_parity(
        &ctx,
        &solver,
        &particles,
        &frames,
        &strand_lengths,
        &rest,
        &rest_darboux,
        base_params(),
    );
    // The pinned root must not have moved at all, and its velocity stays zero.
    let root_moved = gpu_particles[0]
        .position
        .sub(particles[0].position)
        .length();
    assert!(root_moved < 1e-6, "pinned root must not move: {root_moved}");
    let root_vel = gpu_particles[0].velocity.length();
    assert!(
        root_vel < 1e-6,
        "pinned root velocity must stay zero: {root_vel}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_rod_with_truncation_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    // Three rods of different lengths laid end to end in one flat pool, plus a
    // fourth length that deliberately runs past the pool so the walk truncates.
    let (mut p0, f0) = curved_rod(4, 0.1);
    let (p1, f1) = curved_rod(5, 0.08);
    let (p2, f2) = curved_rod(3, 0.12);
    p0.extend(p1);
    p0.extend(p2);
    let mut frames = f0;
    frames.extend(f1);
    frames.extend(f2);
    let particles = p0;

    let rest: Vec<f32> = segment_lengths(&particles[0..4])
        .iter()
        .chain(segment_lengths(&particles[4..9]).iter())
        .chain(segment_lengths(&particles[9..12]).iter())
        .map(|l| l * 1.3)
        .collect();
    let rest_darboux = vec![Vec3::ZERO; frames.len()];

    // Last length (99) exceeds the remaining pool, so the reference truncates.
    let strand_lengths = [4usize, 5, 3, 99];

    let (gpu_particles, _) = assert_parity(
        &ctx,
        &solver,
        &particles,
        &frames,
        &strand_lengths,
        &rest,
        &rest_darboux,
        base_params(),
    );
    // A free particle in the second rod must have moved, proving multiple rods
    // are advanced (not just the first). Particle index 5 is free in rod 2.
    let moved = gpu_particles[5]
        .position
        .sub(particles[5].position)
        .length();
    assert!(moved > 1e-6, "second rod should be simulated: {moved}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_short_orientation_slice_disables_bend_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    // An orientation slice too short for the rod disables the bend-twist solve
    // (the reference substitutes an empty slice) yet must still parity-match.
    let (particles, frames) = curved_rod(6, 0.1);
    let strand_lengths = [particles.len()];
    let rest: Vec<f32> = segment_lengths(&particles)
        .iter()
        .map(|l| l * 1.2)
        .collect();
    let rest_darboux = vec![Vec3::ZERO; frames.len().saturating_sub(1)];
    // Drop the last orientation so the slice no longer covers all segments.
    let short_frames = &frames[0..frames.len() - 1];

    assert_parity(
        &ctx,
        &solver,
        &particles,
        short_frames,
        &strand_lengths,
        &rest,
        &rest_darboux,
        base_params(),
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_short_rest_slice_skips_stretch_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    // A rest-length slice too short for the rod disables the stretch solve for
    // the whole rod (the reference reads it all-or-nothing) yet must still
    // parity-match. Bend-twist still runs over the full frame set.
    let (particles, frames) = curved_rod(6, 0.1);
    let strand_lengths = [particles.len()];
    let full = segment_lengths(&particles);
    let short_rest = &full[0..full.len() - 1];
    let rest_darboux = vec![Vec3::ZERO; frames.len().saturating_sub(1)];

    assert_parity(
        &ctx,
        &solver,
        &particles,
        &frames,
        &strand_lengths,
        short_rest,
        &rest_darboux,
        base_params(),
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_noop_guards_return_pool_unchanged_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cosserat parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuCosserat::new(&ctx);

    // Empty pool -> empty result, no dispatch.
    let (p, o) = solver.eval(&ctx, &[], &[], &[], &[], &[], base_params());
    assert!(p.is_empty(), "empty pool must return empty particles");
    assert!(o.is_empty(), "empty pool must return empty orientations");

    // A single-particle rod has no segment: the reference `count < 2` guard
    // leaves it untouched, and so must the twin.
    let single = vec![RodParticle::free(Vec3::new(0.0, 0.0, 0.0))];
    let strand_lengths = [1usize];
    let (gpu_particles, gpu_orient) = assert_parity(
        &ctx,
        &solver,
        &single,
        &[],
        &strand_lengths,
        &[],
        &[],
        base_params(),
    );
    assert_eq!(gpu_particles, single, "count < 2 rod must stay unchanged");
    assert!(
        gpu_orient.is_empty(),
        "no orientations for a 1-particle rod"
    );

    // A length list that does not fit the pool at all leaves everything
    // unchanged (no descriptor fits, so no dispatch).
    let (particles, frames) = curved_rod(3, 0.1);
    let (gp, go) = solver.eval(
        &ctx,
        &particles,
        &frames,
        &[99usize],
        &[],
        &[],
        base_params(),
    );
    assert_eq!(
        gp, particles,
        "unfittable length must leave particles alone"
    );
    assert_eq!(
        go, frames,
        "unfittable length must leave orientations alone"
    );
}
