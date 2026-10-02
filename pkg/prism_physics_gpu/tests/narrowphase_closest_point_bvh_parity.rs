//! Real-device parity for the closest-point distance scene query: the
//! `GPU`-driven [`GpuSceneClosestPoint`] must agree with the `CPU`
//! [`closest_point`] brute golden and the [`closest_point_bvh`] branch-and-bound
//! query over the same one-against-many static scene.
//!
//! The `CPU` `BVH` query is pinned to the brute golden in its own unit suite, so
//! matching both here transitively pins the device result to the brute golden.
//! The cases mirror the `CPU` unit scenes: an outside query reporting a surface
//! point, distance, and outward normal; a convex rounding radius that pushes the
//! surface out and shortens the distance; an interior query reporting zero
//! distance and the inside flag; three targets where the middle one is nearest;
//! and a large grid-plus-wall scene queried from several points so the `CPU`
//! `BVH` prune and the device brute sweep must still land on the identical
//! nearest target and surface geometry.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: closest-point distance via a Gilbert-Johnson-Keerthi distance
//! walk (Gilbert, Johnson, and Keerthi, 1988) with Ericson's Voronoi
//! sub-distance (2005). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    closest_point, closest_point_bvh, ClosestPointHit, ConvexHull, ConvexPose, GpuContext,
    GpuSceneClosestPoint, RoundedConvex, SceneClosestPoint,
};

/// Tolerance on the surface distance, point, and normal: the only inexact steps
/// are the `GJK` support argmax, the normalise reciprocal, and the witness blend.
const TOL: f32 = 1e-4;

/// An axis-aligned box hull with the given half-extents.
fn box_hull(he: Vec3) -> ConvexHull {
    ConvexHull::from_box(he)
}

/// A stationary target pose centred at `center`.
fn at(center: Vec3) -> ConvexPose {
    ConvexPose::new(center, Quat::IDENTITY)
}

/// Asserts two closest-point hits agree on target index, inside flag, and on
/// distance, point, and normal within tolerance. The point and normal are only
/// meaningful for an outside hit, so they are checked only when both agree the
/// query is outside.
fn assert_hit_matches(want: ClosestPointHit, got: ClosestPointHit) {
    assert_eq!(want.target, got.target, "nearest target index differs");
    assert_eq!(want.inside, got.inside, "inside flag differs");
    let dd = (want.distance - got.distance).abs();
    assert!(dd <= TOL, "distance diverged by {dd}");
    if !want.inside && !got.inside {
        let dp = (want.point - got.point).length();
        let dn = (want.normal - got.normal).length();
        assert!(dp <= TOL, "surface point diverged by {dp}");
        assert!(dn <= TOL, "surface normal diverged by {dn}");
    }
}

/// Runs the `CPU` brute query, the `CPU` `BVH` query, and the device query over
/// the same static scene and query point, asserting all three agree.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuSceneClosestPoint,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    radii: &[f32],
    position: Vec3,
) {
    let query = SceneClosestPoint::new(position);
    let targets: Vec<RoundedConvex> = (0..hulls.len())
        .map(|i| RoundedConvex::still(&hulls[i], poses[i], radii[i]))
        .collect();
    let brute = closest_point(&targets, &query);
    let bvh = closest_point_bvh(&targets, &query);
    assert_eq!(brute, bvh, "CPU BVH must equal CPU brute at {position:?}");
    let device = gpu.closest(ctx, hulls, poses, radii, &query);
    match (brute, device) {
        (None, None) => {}
        (Some(want), Some(got)) => assert_hit_matches(want, got),
        (want, got) => panic!("hit presence differs at {position:?}: cpu {want:?} gpu {got:?}"),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_outside_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let hulls = [box_hull(Vec3::splat(1.0))];
    let poses = [at(Vec3::ZERO)];
    let radii = [0.0_f32];
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, Vec3::new(3.0, 0.0, 0.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_rounding_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let hulls = [box_hull(Vec3::splat(1.0))];
    let poses = [at(Vec3::ZERO)];
    let radii = [0.5_f32];
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, Vec3::new(3.0, 0.0, 0.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_inside_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let hulls = [box_hull(Vec3::splat(1.0))];
    let poses = [at(Vec3::ZERO)];
    let radii = [0.0_f32];
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, Vec3::new(0.2, -0.1, 0.3));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_nearest_of_many_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let hulls = [box_hull(Vec3::splat(1.0)), box_hull(Vec3::splat(1.0)), box_hull(Vec3::splat(1.0))];
    let poses = [at(Vec3::new(0.0, 0.0, 0.0)), at(Vec3::new(8.0, 0.0, 0.0)), at(Vec3::new(16.0, 0.0, 0.0))];
    let radii = [0.0_f32, 0.0, 0.0];
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, Vec3::new(9.0, 0.0, 0.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_empty_scene_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let hulls: [ConvexHull; 0] = [];
    let poses: [ConvexPose; 0] = [];
    let radii: [f32; 0] = [];
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, Vec3::ZERO);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn closest_point_large_scene_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneClosestPoint::new(&ctx);
    let mut hulls = Vec::new();
    let mut poses = Vec::new();
    let mut radii = Vec::new();
    for gx in 0..5 {
        for gz in 0..5 {
            hulls.push(box_hull(Vec3::splat(1.0)));
            poses.push(at(Vec3::new(4.0 * (gx as f32), 0.0, 4.0 * (gz as f32))));
            radii.push(if (gx + gz) % 3 == 0 { 0.3_f32 } else { 0.0 });
        }
    }
    for k in 0..10 {
        hulls.push(box_hull(Vec3::splat(1.0)));
        poses.push(at(Vec3::new(2.0 * (k as f32), 80.0, 0.0)));
        radii.push(0.0);
    }
    for q in [
        Vec3::new(5.0, 1.0, 5.0),
        Vec3::new(-3.0, 0.0, 10.0),
        Vec3::new(9.5, 0.2, 2.1),
        Vec3::new(16.0, 2.0, 16.0),
        Vec3::new(7.0, 0.0, 7.0),
    ] {
        run_parity(&ctx, &gpu, &hulls, &poses, &radii, q);
    }
}
