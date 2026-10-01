//! Real-device parity for the in-place strand body-collision resolve twin:
//! [`GpuStrandCollisionResolve`] must reproduce the `CPU` golden
//! [`resolve_strand_collisions`](prism_render_architecture::hair::collision::resolve_strand_collisions)
//! for a batch of guide particles folded through a shared collider array,
//! including the pinned skip, the last-push-wins ordering, and the degenerate
//! arguments the reference guards.
//!
//! # Parity criterion
//!
//! The per-collider push-out contains no transcendental call, so the `CPU` and
//! `GPU` evaluate the same closed-form geometry and diverge only through legal
//! fused-multiply-add contraction in the sphere `normalize`. Each resolved
//! component is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` —
//! tight enough to fail a genuinely wrong port (a swapped branch, a missing
//! clamp, a dropped collider, a lost pinned skip), loose enough to admit fma
//! contraction. Several cases also assert the physical result (a pinned particle
//! never moves, an interior point lands on a surface, the last collider wins) so
//! a degenerate all-constant kernel could not pass.
//!
//! The suite drives a pinned particle (never moved), a single sphere pushing an
//! interior point to its surface, an overlapping multi-collider stack
//! (last-push-wins), the capsule branch, a mixed pinned/free batch, the empty
//! collider no-op, the empty batch, and a 200-particle batch that crosses the
//! 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: standard analytic sphere/capsule collider push-out plus an
//! in-order fold; no Unreal Engine source or derived code.

use prism_hair_gpu::strand_collision_resolve::{
    reference_resolve_strand_collisions, GpuStrandCollisionResolve,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the resolve tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// A free particle (unit inverse mass) at `(x, y, z)`.
fn free_at(x: f32, y: f32, z: f32) -> StrandParticle {
    StrandParticle::free(Vec3::new(x, y, z))
}

/// A pinned particle at `(x, y, z)`.
fn pinned_at(x: f32, y: f32, z: f32) -> StrandParticle {
    StrandParticle::pinned(Vec3::new(x, y, z))
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, particles: &[StrandParticle], colliders: &[Collider]) -> Vec<[f32; 3]> {
    GpuStrandCollisionResolve::new(ctx).eval(ctx, particles, colliders)
}

/// Asserts a whole batch matches the golden within tolerance.
fn assert_batch_close(got: &[[f32; 3]], particles: &[StrandParticle], colliders: &[Collider]) {
    let want = reference_resolve_strand_collisions(particles, colliders);
    assert_eq!(
        got.len(),
        want.len(),
        "one resolved position per particle (got {}, want {})",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for axis in 0..3 {
            assert!(
                close(g[axis], w[axis]),
                "particle {i} axis {axis}: device {} must match golden {}",
                g[axis],
                w[axis]
            );
        }
    }
}

/// Distance from `[x, y, z]` to `c`.
fn distance(p: [f32; 3], c: Vec3) -> f32 {
    let dx = p[0] - c.x;
    let dy = p[1] - c.y;
    let dz = p[2] - c.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[test]
fn pinned_particle_never_moves() {
    let Some(ctx) = context_or_skip("pinned_particle_never_moves") else {
        return;
    };
    // A pinned particle sitting deep inside a sphere must be returned unchanged.
    let particles = [pinned_at(0.0, 0.0, 0.0)];
    let colliders = [Collider::Sphere {
        center: Vec3::ZERO,
        radius: 5.0,
    }];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
    assert!(
        close(got[0][0], 0.0) && close(got[0][1], 0.0) && close(got[0][2], 0.0),
        "pinned particle stays at the origin"
    );
}

#[test]
fn sphere_pushes_interior_free_particle_to_surface() {
    let Some(ctx) = context_or_skip("sphere_pushes_interior_free_particle_to_surface") else {
        return;
    };
    let center = Vec3::new(1.0, 2.0, 3.0);
    let radius = 2.0;
    let particles = [free_at(1.0, 3.0, 3.0)]; // one unit up from the center
    let colliders = [Collider::Sphere { center, radius }];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
    // Interior point must land exactly on the sphere surface.
    assert!(
        close(distance(got[0], center), radius),
        "resolved point lies on the sphere surface (dist {}, radius {radius})",
        distance(got[0], center)
    );
}

#[test]
fn overlapping_colliders_last_push_wins() {
    let Some(ctx) = context_or_skip("overlapping_colliders_last_push_wins") else {
        return;
    };
    // Two overlapping spheres: the particle is pushed out of the first, then the
    // second is applied to that result. The last collider wins for the particle,
    // which the golden fold encodes.
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(0.5, 0.0, 0.0);
    let particles = [free_at(0.1, 0.1, 0.0)];
    let colliders = [
        Collider::Sphere {
            center: a,
            radius: 1.0,
        },
        Collider::Sphere {
            center: b,
            radius: 1.0,
        },
    ];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
    // After the full fold the point must be on the surface of the *last* sphere.
    assert!(
        close(distance(got[0], b), 1.0),
        "resolved point lies on the last sphere's surface (dist {})",
        distance(got[0], b)
    );
}

#[test]
fn capsule_branch_matches_golden() {
    let Some(ctx) = context_or_skip("capsule_branch_matches_golden") else {
        return;
    };
    let colliders = [Collider::Capsule {
        a: Vec3::new(-1.0, 0.0, 0.0),
        b: Vec3::new(1.0, 0.0, 0.0),
        radius: 1.0,
    }];
    // Points near the segment (pushed out), past the ends (clamped to a/b), and
    // clear of the capsule (untouched).
    let particles = [
        free_at(0.0, 0.2, 0.0),  // interior, mid-segment
        free_at(2.0, 0.1, 0.0),  // past end b, within radius of b
        free_at(0.0, 5.0, 0.0),  // well clear, untouched
        free_at(-3.0, 0.0, 0.0), // past end a, beyond radius, untouched
    ];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
}

#[test]
fn mixed_pinned_and_free_batch_matches_golden() {
    let Some(ctx) = context_or_skip("mixed_pinned_and_free_batch_matches_golden") else {
        return;
    };
    let colliders = [
        Collider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.5,
        },
        Collider::Capsule {
            a: Vec3::new(0.0, -2.0, 0.0),
            b: Vec3::new(0.0, 2.0, 0.0),
            radius: 0.75,
        },
    ];
    let particles = [
        pinned_at(0.1, 0.1, 0.1), // pinned, deep inside both: must not move
        free_at(0.2, 0.0, 0.0),   // free, inside: resolved by the fold
        free_at(3.0, 3.0, 3.0),   // free, clear: untouched
        pinned_at(0.0, 0.5, 0.0), // pinned again
        free_at(0.0, 0.0, 0.3),   // free, inside
    ];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
    // The pinned entries are returned verbatim.
    assert!(
        close(got[0][0], 0.1) && close(got[0][1], 0.1) && close(got[0][2], 0.1),
        "first pinned particle unchanged"
    );
    assert!(
        close(got[3][1], 0.5),
        "second pinned particle unchanged on Y"
    );
}

#[test]
fn empty_collider_set_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_collider_set_is_a_no_op") else {
        return;
    };
    let particles = [free_at(1.0, 2.0, 3.0), pinned_at(-1.0, 0.0, 4.0)];
    let colliders: [Collider; 0] = [];
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
    // No colliders: every particle keeps its original position.
    assert!(
        close(got[0][0], 1.0) && close(got[0][1], 2.0) && close(got[0][2], 3.0),
        "free particle unchanged with no colliders"
    );
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let colliders = [Collider::Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    }];
    let got = run(&ctx, &[], &colliders);
    assert!(
        got.is_empty(),
        "an empty batch yields no resolved positions"
    );
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 particles > 3 full 64-wide workgroups: every particle index must map to
    // its own resolve independent of the dispatch tiling. Every third particle is
    // pinned to exercise the skip across the boundary.
    let colliders = [
        Collider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 2.0,
        },
        Collider::Capsule {
            a: Vec3::new(-1.0, -1.0, 0.0),
            b: Vec3::new(1.0, 1.0, 0.0),
            radius: 1.0,
        },
    ];
    let mut particles = Vec::with_capacity(200);
    for i in 0..200u32 {
        // Deterministic spread with no transcendental calls (those are a
        // disallowed method here): a few affine ramps modulo small periods place
        // particles both inside and clear of the colliders.
        let fi = i as f32;
        let x = (((i % 13) as f32) - 6.0) * 0.5;
        let y = (((i % 7) as f32) - 3.0) * 0.7;
        let z = (((i % 5) as f32) - 2.0) * 0.6 + fi * 0.001;
        if i % 3 == 0 {
            particles.push(pinned_at(x, y, z));
        } else {
            particles.push(free_at(x, y, z));
        }
    }
    let got = run(&ctx, &particles, &colliders);
    assert_batch_close(&got, &particles, &colliders);
}
