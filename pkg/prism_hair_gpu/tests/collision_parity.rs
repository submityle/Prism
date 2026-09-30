//! Real-device parity for the strand collider-projection twin:
//! [`GpuColliderProjector`] must reproduce the `CPU` golden
//! [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
//! for every query across sphere and capsule sweeps, including out-of-range and
//! degenerate arguments the reference guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The projection contains no transcendental call, so the `CPU` and `GPU`
//! evaluate the same closed-form geometry and diverge only through legal
//! fused-multiply-add contraction. Each component is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped branch, a missing clamp, a sign error), loose enough
//! to admit fma contraction. The sweeps also assert the physical result (an
//! interior point lands on the collider surface, an exterior point is
//! untouched) so a degenerate all-constant kernel could not pass.
//!
//! Provenance: standard analytic sphere/capsule collider push-out; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::{query_for, CollisionQuery, GpuColliderProjector, GpuContext};
use prism_render_architecture::hair::collision::Collider;
use prism_render_architecture::hair::dynamics::Vec3;

/// Asserts `gpu` matches the `CPU` golden [`Collider::push_out`] for every
/// query to within the documented fma tolerance. `colliders` and `points` are
/// the reference inputs, aligned with `queries`/`gpu` by index.
fn assert_parity(colliders: &[Collider], points: &[Vec3], gpu: &[[f32; 3]]) {
    assert_eq!(gpu.len(), colliders.len(), "one pushed point per query");
    assert_eq!(points.len(), colliders.len(), "one point per collider");
    for i in 0..colliders.len() {
        let expected = colliders[i].push_out(points[i]);
        let got = gpu[i];
        for (axis, (g, e)) in [
            (got[0], expected.x),
            (got[1], expected.y),
            (got[2], expected.z),
        ]
        .into_iter()
        .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-4 || rel_diff < 1e-3,
                "collision mismatch for query {i} axis {axis}: gpu {g}, cpu {e} (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

/// Distance from `p` to `c`.
fn distance(p: [f32; 3], c: Vec3) -> f32 {
    let dx = p[0] - c.x;
    let dy = p[1] - c.y;
    let dz = p[2] - c.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_sphere_projection_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping collision parity: no wgpu adapter on this host");
        return;
    };
    let projector = GpuColliderProjector::new(&ctx);

    let center = Vec3::new(0.5, -0.25, 1.0);
    let radius = 2.0;
    let sphere = Collider::Sphere { center, radius };

    // Sweep interior and exterior points along a diagonal ray from the center.
    let mut colliders = Vec::new();
    let mut points = Vec::new();
    let mut queries: Vec<CollisionQuery> = Vec::new();
    for k in 0..=30 {
        let s = -0.5 + (k as f32) * 0.2;
        let point = Vec3::new(center.x + s, center.y + s * 0.5, center.z - s * 0.75);
        colliders.push(sphere);
        points.push(point);
        queries.push(query_for(sphere, point));
    }
    // Coincident-with-center point: exercises the +Y escape fallback.
    colliders.push(sphere);
    points.push(center);
    queries.push(query_for(sphere, center));

    let gpu = projector.eval(&ctx, &queries);
    assert_parity(&colliders, &points, &gpu);

    // Physical shape: an interior point is pushed exactly onto the surface; an
    // exterior point keeps its distance. Check the coincident case escaped.
    let coincident = *gpu.last().expect("non-empty sweep");
    assert!(
        (distance(coincident, center) - radius).abs() < 1e-4,
        "coincident point must be pushed to the sphere surface"
    );
    assert!(
        (coincident[1] - (center.y + radius)).abs() < 1e-4,
        "coincident point must escape along +Y"
    );
}

#[test]
fn gpu_capsule_projection_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let projector = GpuColliderProjector::new(&ctx);

    let a = Vec3::new(-1.0, 0.0, 0.0);
    let b = Vec3::new(1.0, 0.0, 0.0);
    let radius = 0.75;
    let capsule = Collider::Capsule { a, b, radius };

    let mut colliders = Vec::new();
    let mut points = Vec::new();
    let mut queries: Vec<CollisionQuery> = Vec::new();
    // Points around and along the capsule, including beyond both end caps and a
    // zero-length degenerate capsule.
    for k in 0..=24 {
        let x = -2.0 + (k as f32) * (4.0 / 24.0);
        let y = 0.1 + (k as f32) * 0.03;
        let point = Vec3::new(x, y, 0.2);
        colliders.push(capsule);
        points.push(point);
        queries.push(query_for(capsule, point));
    }
    let degenerate = Collider::Capsule {
        a: Vec3::ZERO,
        b: Vec3::ZERO,
        radius: 1.0,
    };
    let dp = Vec3::new(0.0, 0.3, 0.0);
    colliders.push(degenerate);
    points.push(dp);
    queries.push(query_for(degenerate, dp));

    let gpu = projector.eval(&ctx, &queries);
    assert_parity(&colliders, &points, &gpu);

    // The zero-length capsule behaves like a sphere at the origin: the point is
    // pushed onto the unit sphere surface.
    let last = *gpu.last().expect("non-empty sweep");
    assert!(
        (distance(last, Vec3::ZERO) - 1.0).abs() < 1e-4,
        "degenerate capsule must behave like a unit sphere"
    );
}

#[test]
fn gpu_inert_and_out_of_range_colliders_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let projector = GpuColliderProjector::new(&ctx);

    let point = Vec3::new(0.3, 0.4, 0.5);
    let colliders = vec![
        // Non-positive radius sphere: inert, point untouched.
        Collider::Sphere {
            center: Vec3::ZERO,
            radius: 0.0,
        },
        Collider::Sphere {
            center: Vec3::new(1.0, 1.0, 1.0),
            radius: -1.0,
        },
        // Non-positive radius capsule: inert.
        Collider::Capsule {
            a: Vec3::new(-1.0, 0.0, 0.0),
            b: Vec3::new(1.0, 0.0, 0.0),
            radius: 0.0,
        },
        // Exterior point far outside a small sphere: untouched.
        Collider::Sphere {
            center: Vec3::new(-5.0, -5.0, -5.0),
            radius: 0.5,
        },
    ];
    let points = vec![point; colliders.len()];
    let queries: Vec<CollisionQuery> = colliders.iter().map(|c| query_for(*c, point)).collect();

    let gpu = projector.eval(&ctx, &queries);
    assert_parity(&colliders, &points, &gpu);

    // Every collider here is inert or exterior, so each result equals the input.
    for g in &gpu {
        assert!(
            (g[0] - point.x).abs() < 1e-4
                && (g[1] - point.y).abs() < 1e-4
                && (g[2] - point.z).abs() < 1e-4,
            "inert/exterior collider must leave the point unchanged, got {g:?}"
        );
    }
}

#[test]
fn empty_queries_yields_no_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let projector = GpuColliderProjector::new(&ctx);
    let out = projector.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
