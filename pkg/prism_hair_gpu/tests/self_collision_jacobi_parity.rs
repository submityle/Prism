//! Real-device parity for the Jacobi strand self-collision twin:
//! [`GpuSelfCollisionJacobi`] must reproduce the `CPU` golden
//! [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections)
//! for a batch of particles against a shared [`UniformGrid`], including the
//! inverse-mass-weighted push split, the pinned-partner and coincident-pair
//! guards, and the degenerate-parameter and empty-array no-ops the reference
//! guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The accumulation is closed-form geometry (the reference restricts itself to
//! `sqrt`, `min`, `max`, `dot`) summed in ascending neighbor order on both
//! sides, so the `CPU` and `GPU` evaluate the same expression and diverge only
//! through legal fused-multiply-add contraction. Parity is asserted per
//! component to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a dropped inverse-mass weight, a wrong push
//! sign, a missing coincident guard), loose enough to admit the fma
//! contraction. Overlapping cases also assert a non-zero correction so a no-op
//! kernel could not pass. Particle clusters are chosen so no pair sits exactly
//! on the `min_sep` boundary, so `CPU` and `GPU` never split on a compare.
//!
//! Provenance: standard uniform-grid self-collision push plus Jacobi
//! accumulation; no Unreal Engine source or derived code.

use prism_hair_gpu::self_collision_jacobi::GpuSelfCollisionJacobi;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::self_collision::SelfCollisionParams;
use prism_render_architecture::hair::self_collision_grid::UniformGrid;
use prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections;

/// A radius/stiffness/cell-size triple with a unit collision diameter.
fn params() -> SelfCollisionParams {
    SelfCollisionParams {
        particle_radius: 0.5,
        stiffness: 1.0,
        cell_size: 1.0,
    }
}

/// Asserts a single component matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts every `GPU` correction equals the `CPU` golden per component.
fn assert_parity(particles: &[StrandParticle], p: SelfCollisionParams, gpu: &[Vec3]) {
    let grid = UniformGrid::build(particles, p.cell_size);
    let mut cpu: Vec<Vec3> = Vec::new();
    accumulate_jacobi_corrections(particles, &grid, p, &mut cpu);
    assert_eq!(gpu.len(), cpu.len(), "one correction per particle");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g.x, c.x, &format!("particle {i} x"));
        assert_close(g.y, c.y, &format!("particle {i} y"));
        assert_close(g.z, c.z, &format!("particle {i} z"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_overlapping_pair_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision jacobi parity: no wgpu adapter on this host");
        return;
    };

    // Two equal-mass free particles overlapping along +x by min_sep - 0.1, plus
    // a pinned particle overlapping a free one so the free partner takes the
    // whole push. No pair sits on the min_sep boundary.
    let particles = [
        StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
        StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
        StrandParticle::pinned(Vec3::new(0.6, 0.7, 0.0)),
        StrandParticle::free(Vec3::new(0.7, 0.7, 0.0)),
    ];
    let p = params();
    let grid = UniformGrid::build(&particles, p.cell_size);
    let solver = GpuSelfCollisionJacobi::new(&ctx);
    let gpu = solver.eval(&ctx, &particles, &grid, p);

    assert_parity(&particles, p, &gpu);
    // The overlapping pair must actually be pushed apart along +/-x, so a no-op
    // kernel (all-zero corrections) cannot pass.
    assert!(gpu[0].x < -1e-3, "particle 0 pushed -x, got {}", gpu[0].x);
    assert!(gpu[1].x > 1e-3, "particle 1 pushed +x, got {}", gpu[1].x);
    // The pinned partner absorbs no correction.
    assert!(
        gpu[2].x.abs() < 1e-6 && gpu[2].y.abs() < 1e-6 && gpu[2].z.abs() < 1e-6,
        "pinned particle stays put, got {:?}",
        (gpu[2].x, gpu[2].y, gpu[2].z)
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_symmetric_cluster_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision jacobi parity: no wgpu adapter on this host");
        return;
    };

    // A 2x2 symmetric cluster whose members all mutually overlap, exercising the
    // multi-neighbor accumulation with a mix of soft stiffness.
    let particles = [
        StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
        StrandParticle::free(Vec3::new(0.15, 0.0, 0.0)),
        StrandParticle::free(Vec3::new(0.0, 0.15, 0.0)),
        StrandParticle::free(Vec3::new(0.15, 0.15, 0.0)),
    ];
    let p = SelfCollisionParams {
        stiffness: 0.6,
        ..params()
    };
    let grid = UniformGrid::build(&particles, p.cell_size);
    let solver = GpuSelfCollisionJacobi::new(&ctx);
    let gpu = solver.eval(&ctx, &particles, &grid, p);

    assert_parity(&particles, p, &gpu);
    // Every member overlaps its neighbors, so each correction is non-trivial.
    for (i, c) in gpu.iter().enumerate() {
        let mag = (c.x * c.x + c.y * c.y + c.z * c.z).sqrt();
        assert!(mag > 1e-3, "particle {i} should be pushed, mag {mag}");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_and_empty_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping self-collision jacobi parity: no wgpu adapter on this host");
        return;
    };
    let solver = GpuSelfCollisionJacobi::new(&ctx);

    // Empty array yields an empty correction vector.
    let empty: [StrandParticle; 0] = [];
    let grid_empty = UniformGrid::build(&empty, params().cell_size);
    let gpu_empty = solver.eval(&ctx, &empty, &grid_empty, params());
    assert!(gpu_empty.is_empty(), "empty batch yields no corrections");

    // Degenerate parameters (non-positive radius / stiffness / cell size, or a
    // non-finite cell size) yield all-zero corrections, matching the golden's
    // cleared-out early return.
    let overlapping = [
        StrandParticle::free(Vec3::new(0.0, 0.0, 0.0)),
        StrandParticle::free(Vec3::new(0.1, 0.0, 0.0)),
    ];
    for degenerate in [
        SelfCollisionParams {
            particle_radius: 0.0,
            ..params()
        },
        SelfCollisionParams {
            stiffness: 0.0,
            ..params()
        },
        SelfCollisionParams {
            cell_size: 0.0,
            ..params()
        },
        SelfCollisionParams {
            cell_size: f32::NAN,
            ..params()
        },
    ] {
        // Build the grid with a safe cell size; the eval guard fires on params.
        let grid = UniformGrid::build(&overlapping, params().cell_size);
        let gpu = solver.eval(&ctx, &overlapping, &grid, degenerate);
        assert_eq!(gpu.len(), overlapping.len(), "one correction per particle");
        for (i, c) in gpu.iter().enumerate() {
            assert!(
                c.x.abs() < 1e-12 && c.y.abs() < 1e-12 && c.z.abs() < 1e-12,
                "degenerate params leave particle {i} uncorrected, got {:?}",
                (c.x, c.y, c.z)
            );
        }
        // The `CPU` golden agrees value-for-value on the degenerate case.
        assert_parity(&overlapping, degenerate, &gpu);
    }
}
