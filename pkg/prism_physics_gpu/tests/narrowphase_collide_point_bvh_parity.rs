//! Real-device parity for the broad-phase-accelerated collide-point query: the
//! `GPU`-driven [`GpuSceneCollidePoint`] per-call and resident-tree paths must
//! agree with the `CPU` [`collide_point_bvh`] golden over the same
//! one-against-many static scene.
//!
//! The `CPU` `BVH` collide-point query is itself the proven façade over the
//! shape-cast path whose brute-force equivalence is pinned in its own unit
//! suite, so matching it here transitively pins both `GPU` paths to the
//! brute-force golden. The cases mirror the `CPU` unit scenes: a single
//! containment among disjoint boxes, a clean miss, a point in the shared overlap
//! of two boxes, and a large scene whose far wall forces the gather to prune.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern),
//! `LBVH` per Karras 2012, stackless traversal per Hapala 2011, conservative
//! advancement per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk
//! (van den Bergen, 2004). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    collide_point_bvh, CollidePointHit, ConvexHull, ConvexPose, GpuContext, GpuLbvh,
    GpuSceneCollidePoint, RoundedConvex, ScenePoint,
};

/// A unit box core, reused across targets.
fn unit_box() -> ConvexHull {
    ConvexHull::from_box(Vec3::splat(0.5))
}

/// A stationary unit box centred at `center`.
fn at(center: Vec3) -> ConvexPose {
    ConvexPose::new(center, Quat::IDENTITY)
}

/// Asserts a `GPU` containing-target list matches the `CPU` golden slot for
/// slot. Collide-point hits are a bare target index, so equality is exact.
fn assert_lists_match(want: &[CollidePointHit], got: &[CollidePointHit]) {
    assert_eq!(want.len(), got.len(), "containing-target count differs");
    for (w, g) in want.iter().copied().zip(got.iter().copied()) {
        assert_eq!(w, g, "containing target differs");
    }
}

/// Runs the `CPU` `BVH` collide-point query, the per-call `GPU` query, and the
/// resident-tree `GPU` query over the same static scene, asserting all three
/// agree.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuSceneCollidePoint,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    radii: &[f32],
    point: &ScenePoint,
) {
    let targets: Vec<RoundedConvex> = (0..hulls.len())
        .map(|i| RoundedConvex::still(&hulls[i], poses[i], radii[i]))
        .collect();

    let cpu = collide_point_bvh(&targets, point);

    let gpu_hits = gpu.collide(ctx, hulls, poses, radii, point);
    assert_lists_match(&cpu, &gpu_hits);

    let boxes = GpuSceneCollidePoint::scene_boxes(hulls, poses, radii);
    let resident = GpuLbvh::new(ctx).build_resident(ctx, &boxes);
    let gpu_hits_res = gpu.collide_resident(ctx, &resident, hulls, poses, radii, point);
    assert_lists_match(&cpu, &gpu_hits_res);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_point_single_containment_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneCollidePoint::new(&ctx);
    let hulls = [unit_box(), unit_box(), unit_box()];
    let poses = [
        at(Vec3::new(0.0, 0.0, 0.0)),
        at(Vec3::new(3.0, 0.0, 0.0)),
        at(Vec3::new(6.0, 0.0, 0.0)),
    ];
    let radii = [0.0_f32, 0.0, 0.0];
    let point = ScenePoint::new(Vec3::new(3.1, 0.05, -0.1));
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &point);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_point_clean_miss_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneCollidePoint::new(&ctx);
    let hulls = [unit_box(), unit_box()];
    let poses = [at(Vec3::new(0.0, 0.0, 0.0)), at(Vec3::new(10.0, 0.0, 0.0))];
    let radii = [0.0_f32, 0.0];
    let point = ScenePoint::new(Vec3::new(5.0, 5.0, 5.0));
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &point);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_point_shared_overlap_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneCollidePoint::new(&ctx);
    // Two boxes overlapping near the origin plus a far one; the point sits in
    // the shared overlap of the first two.
    let hulls = [unit_box(), unit_box(), unit_box()];
    let poses = [
        at(Vec3::new(0.0, 0.0, 0.0)),
        at(Vec3::new(0.4, 0.0, 0.0)),
        at(Vec3::new(20.0, 0.0, 0.0)),
    ];
    let radii = [0.0_f32, 0.0, 0.0];
    let point = ScenePoint::new(Vec3::new(0.2, 0.0, 0.0));
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &point);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn collide_point_large_scene_prunes_and_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU collide-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneCollidePoint::new(&ctx);

    // One box containing the point at the origin, plus a far wall in +y the
    // gather must prune rather than fall back to the whole batch.
    let mut hulls = Vec::new();
    let mut poses = Vec::new();
    let mut radii = Vec::new();
    hulls.push(unit_box());
    poses.push(at(Vec3::new(0.0, 0.0, 0.0)));
    radii.push(0.0_f32);
    for k in 0..40 {
        hulls.push(unit_box());
        poses.push(at(Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        radii.push(0.0_f32);
    }
    let point = ScenePoint::new(Vec3::new(0.1, -0.1, 0.2));
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &point);
}
