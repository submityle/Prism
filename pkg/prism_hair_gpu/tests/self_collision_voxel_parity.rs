//! Real-device parity for the voxel-density strand self-collision twin:
//! [`GpuHairSelfCollisionVoxel`] must reproduce the `CPU` golden
//! [`resolve_self_collision`](prism_render_architecture::hair::self_collision_voxel::resolve_self_collision)
//! for a batch of particles against a shared [`VoxelGrid`], including the
//! one-sided density-gradient repulsion, the two-sided volume-preservation
//! restore, the hair-hair friction velocity blend, and the non-finite,
//! empty-array and degenerate-grid/params guards the reference enforces.
//!
//! The density field itself is built host-side with the golden
//! [`splat_density`](prism_render_architecture::hair::self_collision_voxel::splat_density)
//! (phase 1, a many-writers-one-cell scatter-add), so these tests exercise the
//! embarrassingly parallel phase-2 gather-and-resolve the kernel actually ports.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The resolve contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form trilinear gather, central-difference gradient and friction
//! blend. They diverge only through legal fused-multiply-add contraction, but
//! the gradient divides by a cell separation and the push direction is a
//! renormalised gradient, so a small error upstream is amplified: parity is
//! asserted per component to within `abs_diff < 5e-3` or `rel_diff < 5e-3` —
//! tight enough to fail a genuinely wrong port (a dropped volume term, a wrong
//! push sign, a missing friction blend), loose enough to admit the division and
//! renormalisation chain. Non-finite reference components are matched by NaN-ness
//! rather than value, and the physical cases assert a non-trivial change so a
//! no-op kernel could not pass.
//!
//! Provenance: standard voxel-density hair self-collision (AMD `TressFX` /
//! `Petrovic` et al. 2005 volumetric hair) plus `wgpu` compute dispatch; no
//! Unreal Engine source or derived code.

use prism_hair_gpu::self_collision_voxel::GpuHairSelfCollisionVoxel;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::self_collision_voxel::{
    resolve_self_collision, HairPoint, SelfCollisionParams, Vec3, VoxelGrid,
};

/// Asserts a single component matches within the documented tolerance, treating
/// a non-finite reference value as a NaN/infinity match rather than a numeric
/// comparison (the golden leaves non-finite positions untouched).
fn assert_close(got: f32, expected: f32, label: &str) {
    if expected.is_nan() {
        assert!(got.is_nan(), "{label}: expected NaN, got {got}");
        return;
    }
    if expected.is_infinite() {
        assert!(
            got.is_infinite() && got.signum() == expected.signum(),
            "{label}: expected {expected}, got {got}"
        );
        return;
    }
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 5e-3 || rel_diff < 5e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Runs the golden resolve on a clone of `points` and asserts the `GPU` result
/// matches it per component (position and velocity), with the input mass carried
/// through unchanged.
fn assert_parity(
    ctx: &GpuContext,
    points: &[HairPoint],
    grid: VoxelGrid,
    params: SelfCollisionParams,
) -> Vec<HairPoint> {
    let solver = GpuHairSelfCollisionVoxel::new(ctx);
    let gpu = solver.eval(ctx, points, grid, params);

    let mut cpu = points.to_vec();
    resolve_self_collision(&mut cpu, grid, params);

    assert_eq!(gpu.len(), cpu.len(), "one output per particle");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.position.x, c.position.x, &format!("particle {i} pos.x"));
        assert_close(g.position.y, c.position.y, &format!("particle {i} pos.y"));
        assert_close(g.position.z, c.position.z, &format!("particle {i} pos.z"));
        assert_close(g.velocity.x, c.velocity.x, &format!("particle {i} vel.x"));
        assert_close(g.velocity.y, c.velocity.y, &format!("particle {i} vel.y"));
        assert_close(g.velocity.z, c.velocity.z, &format!("particle {i} vel.z"));
        let md = (g.mass - c.mass).abs();
        assert!(
            md < 1e-6,
            "particle {i} mass carried through: {g:?} vs {c:?}"
        );
    }
    gpu
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_repulsion_pushes_overcrowded_apart() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };

    // Two particles in a 1D row of cells: the density piles up between them, so
    // each is pushed down the gradient away from the other. Pure repulsion
    // (rest_density 0 so all density is excess), no volume, no friction.
    let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
    let points = [
        HairPoint::at(Vec3::new(1.5, 0.5, 0.5)),
        HairPoint::at(Vec3::new(2.0, 0.5, 0.5)),
    ];
    let params = SelfCollisionParams::new(0.0, 0.1, 0.0, 0.0);

    let before = points[1].position.sub(points[0].position).length();
    let gpu = assert_parity(&ctx, &points, grid, params);
    let after = gpu[1].position.sub(gpu[0].position).length();
    // The overcrowded pair must actually be pushed apart, so a no-op kernel
    // cannot pass.
    assert!(after > before, "after={after} before={before}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_volume_term_pulls_sparse_together() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };

    // A pair in an under-dense field (rest_density far above the local density)
    // is pulled up-gradient toward the denser hair, not pushed. No repulsion so
    // the two-sided volume term is isolated.
    let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
    let points = [
        HairPoint::at(Vec3::new(1.3, 0.5, 0.5)),
        HairPoint::at(Vec3::new(2.4, 0.5, 0.5)),
    ];
    let params = SelfCollisionParams::new(10.0, 0.0, 0.0, 0.05);

    let before = points[1].position.sub(points[0].position).length();
    let gpu = assert_parity(&ctx, &points, grid, params);
    let after = gpu[1].position.sub(gpu[0].position).length();
    assert!(after < before, "after={after} before={before}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_friction_converges_velocities() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };

    // Two neighbouring particles with opposite velocities: with friction 1 each
    // snaps toward the local mass-weighted mean, so they converge. A huge
    // rest_density suppresses any positional push so the velocity blend is
    // isolated.
    let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [4, 1, 1]);
    let points = [
        HairPoint::new(Vec3::new(1.6, 0.5, 0.5), Vec3::new(2.0, 0.0, 0.0)),
        HairPoint::new(Vec3::new(1.9, 0.5, 0.5), Vec3::new(0.0, 0.0, 0.0)),
    ];
    let params = SelfCollisionParams::new(1.0e9, 0.0, 1.0, 0.0);

    let before = points[0].velocity.sub(points[1].velocity).length();
    let gpu = assert_parity(&ctx, &points, grid, params);
    let after = gpu[0].velocity.sub(gpu[1].velocity).length();
    assert!(after < before, "after={after} before={before}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mixed_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };

    // A fully 3-D cluster with varied masses and velocities, all three forces
    // active at once, to exercise the general gather/gradient/friction path with
    // non-trivial cross terms.
    let grid = VoxelGrid::new(Vec3::new(-1.0, -1.0, -1.0), 0.75, [5, 5, 5]);
    let points = [
        HairPoint {
            position: Vec3::new(0.3, 0.2, 0.1),
            velocity: Vec3::new(1.0, -0.5, 0.2),
            mass: 1.5,
        },
        HairPoint {
            position: Vec3::new(0.5, 0.35, 0.2),
            velocity: Vec3::new(-0.8, 0.3, 0.0),
            mass: 0.75,
        },
        HairPoint {
            position: Vec3::new(0.42, 0.55, 0.3),
            velocity: Vec3::new(0.1, 0.1, -1.2),
            mass: 2.0,
        },
        HairPoint {
            position: Vec3::new(0.9, 0.1, 0.6),
            velocity: Vec3::new(0.0, 0.4, 0.4),
            mass: 1.0,
        },
    ];
    let params = SelfCollisionParams::new(1.0, 0.08, 0.35, 0.05);

    let gpu = assert_parity(&ctx, &points, grid, params);
    // At least one particle must have moved or had its velocity blended, so a
    // no-op kernel could not pass this case either.
    let moved = gpu.iter().zip(points.iter()).any(|(g, p)| {
        g.position.sub(p.position).length() > 1e-4 || g.velocity.sub(p.velocity).length() > 1e-4
    });
    assert!(moved, "the active cluster must change under the resolve");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_nan_and_inf_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };

    // A NaN-position particle is left exactly as-is (never advected or scrubbed);
    // a finite-position particle with an infinite incoming velocity has that
    // velocity scrubbed to zero before the friction blend. A healthy neighbour
    // gives the finite particle a real local density to interact with.
    let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [3, 3, 3]);
    let points = [
        HairPoint::at(Vec3::new(f32::NAN, 0.0, 0.0)),
        HairPoint::new(Vec3::new(1.5, 1.5, 1.5), Vec3::new(f32::INFINITY, 0.0, 0.0)),
        HairPoint::new(Vec3::new(1.7, 1.5, 1.5), Vec3::new(0.5, 0.0, 0.0)),
    ];
    let params = SelfCollisionParams::default();

    let gpu = assert_parity(&ctx, &points, grid, params);
    // The NaN-position particle is still NaN (never repaired).
    assert!(gpu[0].position.x.is_nan(), "NaN position preserved");
    // The finite particles stay finite throughout.
    assert!(gpu[1].position.is_finite() && gpu[1].velocity.is_finite());
    assert!(gpu[2].position.is_finite() && gpu[2].velocity.is_finite());
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_and_degenerate_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel self-collision parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuHairSelfCollisionVoxel::new(&ctx);

    // Empty array yields an empty output vector.
    let empty: [HairPoint; 0] = [];
    let grid = VoxelGrid::new(Vec3::ZERO, 1.0, [2, 2, 2]);
    let gpu_empty = solver.eval(&ctx, &empty, grid, SelfCollisionParams::default());
    assert!(gpu_empty.is_empty(), "empty batch yields no output");

    // A degenerate grid (NaN origin, zero cell size, zero dims) and clamped
    // params are repaired by the shared `sanitized()` on both sides, so the twin
    // still matches the golden and never panics.
    let degenerate_grid = VoxelGrid::new(Vec3::new(f32::NAN, 0.0, 0.0), 0.0, [0, 0, 0]);
    let degenerate_params = SelfCollisionParams::new(f32::NAN, -5.0, 10.0, f32::INFINITY);
    let points = [
        HairPoint::at(Vec3::new(0.2, 0.2, 0.2)),
        HairPoint::at(Vec3::new(0.4, 0.3, 0.1)),
    ];
    let gpu = assert_parity(&ctx, &points, degenerate_grid, degenerate_params);
    for (i, g) in gpu.iter().enumerate() {
        assert!(
            g.position.is_finite(),
            "degenerate case leaves particle {i} finite, got {:?}",
            g.position
        );
    }
}
