//! Real-device parity for the broad-phase-accelerated collide-shape query: the
//! `GPU`-driven [`GpuBvhCollideShape::collide`] and the resident-tree
//! [`GpuBvhCollideShape::collide_resident`] must agree with the `CPU`
//! [`collide_shape_bvh`] golden over the same one-against-many static scene.
//!
//! The `CPU` `BVH` query is itself proven exactly equivalent to the brute-force
//! walk in its own unit suite, so matching it here transitively pins both `GPU`
//! paths to the brute-force golden. Each case mirrors a `CPU` unit scene: a
//! mixed scene where the query overlaps some targets and misses others, a scene
//! where nothing overlaps, and a large scene whose far wall forces the gather to
//! prune rather than fall back to the whole batch.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern),
//! `LBVH` per Karras 2012, stackless traversal per Hapala 2011,
//! convex-versus-convex manifold via Gilbert-Johnson-Keerthi 1988 /
//! expanding-polytope (van den Bergen 2001) with Sutherland-Hodgman clipping. No
//! Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    collide_shape_bvh, CollideShapeHit, ConvexHull, ConvexPose, GpuBvhCollideShape, GpuContext,
    GpuLbvh, GpuResidentLbvh,
};

/// Tolerance on manifold point positions and depths and on the shared normal,
/// matching the per-pair manifold suite: the only inexact steps are the support
/// argmax, the normalise reciprocal, and the clip divisions.
const TOL: f32 = 1e-4;

/// A unit box core, reused across bodies.
fn unit_box() -> ConvexHull {
    ConvexHull::from_box(Vec3::splat(0.5))
}

/// A pose at `center` with no rotation.
fn at(center: Vec3) -> ConvexPose {
    ConvexPose::new(center, Quat::IDENTITY)
}

/// Asserts two collide-shape hits agree on target index, manifold identity, and
/// the shared normal, point count, point positions, and depths within tolerance.
fn assert_hit_matches(want: &CollideShapeHit, got: &CollideShapeHit) {
    assert_eq!(want.target, got.target, "overlapped target index differs");
    let wm = &want.manifold;
    let gm = &got.manifold;
    assert_eq!(wm.a, gm.a, "manifold body a differs");
    assert_eq!(wm.b, gm.b, "manifold body b differs");
    assert_eq!(wm.count, gm.count, "manifold point count differs");
    let dn = (wm.normal - gm.normal).length();
    assert!(dn <= TOL, "normal diverged by {dn}");
    for i in 0..(wm.count as usize) {
        let dp = (wm.points[i].position - gm.points[i].position).length();
        let dd = (wm.points[i].depth - gm.points[i].depth).abs();
        assert!(dp <= TOL, "point {i} position diverged by {dp}");
        assert!(dd <= TOL, "point {i} depth diverged by {dd}");
    }
}

/// Asserts a `GPU` hit list matches the `CPU` golden slot for slot.
fn assert_lists_match(want: &[CollideShapeHit], got: &[CollideShapeHit]) {
    assert_eq!(want.len(), got.len(), "overlapped hit count differs");
    for (w, g) in want.iter().zip(got) {
        assert_hit_matches(w, g);
    }
}

/// Runs the `CPU` `BVH` query, the per-call `GPU` query, and the resident-tree
/// `GPU` query over the same static scene, asserting all three agree.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuBvhCollideShape,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    margin: f32,
) {
    let cpu = collide_shape_bvh(hulls, poses, margin);

    let per_call = gpu.collide(ctx, hulls, poses, margin);
    assert_lists_match(&cpu, &per_call);

    // Build the resident tree once from the static targets' bounds, in target
    // order (margin 0: the per-query margin is applied to the moving query box
    // at traversal time, not baked into the static tree), then reuse it.
    let boxes = GpuBvhCollideShape::scene_boxes(&hulls[1..], &poses[1..], 0.0);
    let resident: GpuResidentLbvh = GpuLbvh::new(ctx).build_resident(ctx, &boxes);
    let resident_hits = gpu.collide_resident(ctx, &resident, hulls, poses, margin);
    assert_lists_match(&cpu, &resident_hits);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_mixed_scene_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-shape parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhCollideShape::new(&ctx);

    // A query box overlapping two of four targets; the other two are far.
    let hulls = [unit_box(), unit_box(), unit_box(), unit_box(), unit_box()];
    let poses = [
        at(Vec3::new(0.0, 0.0, 0.0)),
        at(Vec3::new(0.6, 0.03, 0.0)),  // overlaps
        at(Vec3::new(-0.6, 0.0, 0.05)), // overlaps
        at(Vec3::new(40.0, 0.0, 0.0)),  // far
        at(Vec3::new(0.0, 40.0, 0.0)),  // far
    ];
    run_parity(&ctx, &gpu, &hulls, &poses, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_disjoint_scene_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-shape parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhCollideShape::new(&ctx);

    // Nothing overlaps the query, so both engines must agree on an empty result.
    let hulls = [unit_box(), unit_box(), unit_box()];
    let poses = [
        at(Vec3::ZERO),
        at(Vec3::new(10.0, 0.0, 0.0)),
        at(Vec3::new(0.0, -10.0, 0.0)),
    ];
    run_parity(&ctx, &gpu, &hulls, &poses, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_large_scene_prunes_and_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-shape parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhCollideShape::new(&ctx);

    // One query box at the origin; a dense cluster of overlapping boxes plus a
    // far wall the gather must prune. Distinct tiny offsets avoid support ties.
    let mut hulls = vec![unit_box()];
    let mut poses = vec![at(Vec3::ZERO)];
    for k in 0..5 {
        let d = 0.3 + 0.02 * (k as f32);
        hulls.push(unit_box());
        poses.push(at(Vec3::new(d, 0.01 * (k as f32), 0.0)));
    }
    for k in 0..40 {
        hulls.push(unit_box());
        poses.push(at(Vec3::new(5.0 + 1.5 * (k as f32), 80.0, 0.0)));
    }
    run_parity(&ctx, &gpu, &hulls, &poses, 0.0);
}
