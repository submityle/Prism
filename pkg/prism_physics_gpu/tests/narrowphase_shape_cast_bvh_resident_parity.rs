//! Real-device parity for the resident-tree `GPU` shape cast: the persistent
//! broad-phase query path [`GpuBvhShapeCast::cast_resident`] /
//! [`GpuBvhShapeCast::cast_all_resident`] must agree with the `CPU`
//! [`cast_shape_bvh`] / [`cast_shape_all_bvh`] golden over the same
//! one-against-many static scene.
//!
//! This differs from the per-cast `BVH` parity suite in that the acceleration
//! structure is built once with [`GpuLbvh::build_resident`] from
//! [`GpuBvhShapeCast::scene_boxes`] and then reused across the earliest-hit and
//! full-list casts, exercising the on-device resident tree instead of a tree
//! rebuilt per cast. The `CPU` `BVH` cast is itself proven exactly equivalent
//! to the brute-force cast in its own unit suite, so matching it here
//! transitively pins the resident `GPU` path to the brute-force golden.
//!
//! Every target is static (the resident tree is only valid for an unchanging
//! scene); only the cast shape moves. The scenes are asymmetric so no two gaps
//! close at once and no support argmax is a near-tie, keeping `GPU` and `CPU`
//! float rounding from flipping the chosen target, the time, or the hit flag. A
//! `miss`-only scene exercises the clean-miss agreement, and a large-scene case
//! exercises the gather actually pruning rather than falling back to the whole
//! batch.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: persistent broad-phase query (Jolt / `PhysX` / `Chaos` pattern),
//! `LBVH` per Karras 2012, stackless traversal per Hapala 2011, conservative
//! advancement per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk
//! (van den Bergen, 2004). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cast_shape_all_bvh, cast_shape_bvh, BodyMotion, ConvexHull, ConvexPose, GpuBvhShapeCast,
    GpuContext, GpuLbvh, GpuResidentLbvh, RoundedConvex, ShapeCastHit,
};

/// Tolerance on the impact time, point, and normal, matching the per-pair and
/// per-cast `BVH` suites: the only inexact steps are the support argmax, the
/// normalise reciprocals, and the advance division.
const TOL: f32 = 1e-4;

/// Asserts two shape-cast hits agree on target index and on the time, point,
/// and normal within tolerance.
fn assert_hit_matches(want: ShapeCastHit, got: ShapeCastHit) {
    assert_eq!(want.target, got.target, "struck target index differs");
    let dt = (want.toi.time - got.toi.time).abs();
    let dp = (want.toi.point - got.toi.point).length();
    let dn = (want.toi.normal - got.toi.normal).length();
    assert!(dt <= TOL, "time diverged by {dt}");
    assert!(dp <= TOL, "point diverged by {dp}");
    assert!(dn <= TOL, "normal diverged by {dn}");
}

/// Builds the resident `BVH` over the static targets (bodies `1..`) once, then
/// runs the `CPU` `BVH` cast and the resident `GPU` cast over the same scene,
/// asserting both the earliest hit and the full ordered hit list agree.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuBvhShapeCast,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    radii: &[f32],
    dt: f32,
    target_sep: f32,
) {
    let shape = RoundedConvex::new(&hulls[0], poses[0], motions[0], radii[0]);
    let targets: Vec<RoundedConvex> = (1..hulls.len())
        .map(|i| RoundedConvex::new(&hulls[i], poses[i], motions[i], radii[i]))
        .collect();

    let cpu_best = cast_shape_bvh(&shape, &targets, dt, target_sep);
    let cpu_all = cast_shape_all_bvh(&shape, &targets, dt, target_sep);

    // Build the resident tree once from the static targets' bounds, in target
    // order, then reuse it across both resident casts.
    let boxes = GpuBvhShapeCast::scene_boxes(&hulls[1..], &poses[1..], &radii[1..]);
    let resident: GpuResidentLbvh = GpuLbvh::new(ctx).build_resident(ctx, &boxes);

    let gpu_best = gpu.cast_resident(ctx, &resident, hulls, poses, motions, radii, dt, target_sep);
    let gpu_all =
        gpu.cast_all_resident(ctx, &resident, hulls, poses, motions, radii, dt, target_sep);

    match (cpu_best, gpu_best) {
        (None, None) => {}
        (Some(w), Some(g)) => assert_hit_matches(w, g),
        (w, g) => panic!(
            "earliest-hit mismatch: cpu {:?} vs gpu {:?}",
            w.is_some(),
            g.is_some()
        ),
    }
    assert_eq!(cpu_all.len(), gpu_all.len(), "ordered hit count differs");
    for (w, g) in cpu_all.into_iter().zip(gpu_all) {
        assert_hit_matches(w, g);
    }
}

/// A unit box core, reused across targets.
fn unit_box() -> ConvexHull {
    ConvexHull::from_box(Vec3::splat(0.5))
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn resident_cast_nearest_blocker_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhShapeCast::new(&ctx);

    // A sphere core (point core + radius 0.5) sweeps +x past two static boxes.
    // Slight y offsets keep the gaps from closing along two axes at once.
    let hulls = [ConvexHull::from_point(), unit_box(), unit_box()];
    let poses = [
        ConvexPose::new(Vec3::new(0.0, 0.03, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(8.0, 0.05, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
        BodyMotion::still(),
    ];
    let radii = [0.5_f32, 0.0, 0.0];
    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, 1.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn resident_cast_rounded_mixed_targets_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhShapeCast::new(&ctx);

    // A capsule core (segment core along y + radius 0.3) sweeps +x against a
    // rounded sphere target, a bevelled box, and an off-path box it misses.
    let hulls = [
        ConvexHull::from_segment(Vec3::Y, 0.5),
        ConvexHull::from_point(),
        unit_box(),
        unit_box(),
    ];
    let poses = [
        ConvexPose::new(Vec3::new(0.0, 0.04, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(6.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(3.0, 0.06, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(5.0, 30.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(8.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
        BodyMotion::still(),
        BodyMotion::still(),
    ];
    let radii = [0.3_f32, 0.4, 0.1, 0.0];
    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, 1.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn resident_cast_all_targets_off_path_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhShapeCast::new(&ctx);

    // Nothing lies on the sweep line, so both engines must agree on a clean miss.
    let hulls = [ConvexHull::from_point(), unit_box(), unit_box()];
    let poses = [
        ConvexPose::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(4.0, 25.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(8.0, -25.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
        BodyMotion::still(),
    ];
    let radii = [0.5_f32, 0.0, 0.0];
    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, 1.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn resident_cast_large_scene_prunes_and_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping resident GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvhShapeCast::new(&ctx);

    // A sphere sweeps +x down a corridor of on-path blockers while a wall of
    // off-path boxes forces the resident gather to actually prune. The nearest
    // on-path blocker must win on both engines, and the full ordered on-path
    // list must match.
    let mut hulls = vec![ConvexHull::from_point()];
    let mut poses = vec![ConvexPose::new(Vec3::new(0.0, 0.02, 0.0), Quat::IDENTITY)];
    let mut motions = vec![BodyMotion::new(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO)];
    let mut radii = vec![0.5_f32];

    // On-path blockers at increasing x, tiny distinct y offsets to avoid ties.
    for k in 0..6 {
        let x = 3.0 + (k as f32) * 2.0;
        let y = 0.01 * (k as f32 + 1.0);
        hulls.push(unit_box());
        poses.push(ConvexPose::new(Vec3::new(x, y, 0.0), Quat::IDENTITY));
        motions.push(BodyMotion::still());
        radii.push(0.0);
    }
    // Off-path wall far in +y the swept query box never reaches.
    for k in 0..24 {
        let x = 1.0 + (k as f32) * 1.5;
        hulls.push(unit_box());
        poses.push(ConvexPose::new(Vec3::new(x, 60.0, 0.0), Quat::IDENTITY));
        motions.push(BodyMotion::still());
        radii.push(0.0);
    }

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, 1.0, 0.0);
}
