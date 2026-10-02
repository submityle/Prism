//! Real-device parity for the broad-phase-accelerated box cast: the
//! `GPU`-driven [`GpuSceneBoxCast`] per-call and resident-tree paths must agree
//! with the `CPU` [`box_cast_bvh`] / [`box_cast_all_bvh`] golden over the same
//! one-against-many static scene.
//!
//! The `CPU` `BVH` box cast is itself the proven façade over the shape-cast path
//! whose brute-force equivalence is pinned in its own unit suite, so matching it
//! here transitively pins both `GPU` box paths to the brute-force golden. The
//! cases mirror the `CPU` unit scenes: a nearest hit down a corridor, an
//! orientation that presents a corner, a clean miss, a max-distance cutoff, a
//! single-leaf resident scene (the degenerate-tree regression), and a large
//! scene whose off-path wall forces the gather to prune.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: broad-phase ray/sweep gather (Jolt / `PhysX` / `Chaos` pattern),
//! `LBVH` per Karras 2012, stackless traversal per Hapala 2011, conservative
//! advancement per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk
//! (van den Bergen, 2004). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    box_cast_all_bvh, box_cast_bvh, BoxCastHit, ConvexHull, ConvexPose, GpuContext, GpuLbvh,
    GpuSceneBoxCast, RoundedConvex, SceneBoxCast,
};

/// Tolerance on the entry distance, point, and normal, matching the shape-cast
/// suites: the only inexact steps are the support argmax, the normalise
/// reciprocal, and the advance division.
const TOL: f32 = 1e-4;

/// A unit box core, reused across targets.
fn unit_box() -> ConvexHull {
    ConvexHull::from_box(Vec3::splat(0.5))
}

/// A stationary unit box centred at `center`.
fn at(center: Vec3) -> ConvexPose {
    ConvexPose::new(center, Quat::IDENTITY)
}

/// Asserts two box-cast hits agree on target index and on distance, point, and
/// normal within tolerance.
fn assert_hit_matches(want: BoxCastHit, got: BoxCastHit) {
    assert_eq!(want.target, got.target, "struck target index differs");
    let dd = (want.distance - got.distance).abs();
    let dp = (want.point - got.point).length();
    let dn = (want.normal - got.normal).length();
    assert!(dd <= TOL, "distance diverged by {dd}");
    assert!(dp <= TOL, "point diverged by {dp}");
    assert!(dn <= TOL, "normal diverged by {dn}");
}

/// Runs the `CPU` `BVH` box cast, the per-call `GPU` cast, and the
/// resident-tree `GPU` cast over the same static scene, asserting all three
/// agree on both the nearest hit and the full ordered list.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuSceneBoxCast,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    radii: &[f32],
    cast: &SceneBoxCast,
) {
    let targets: Vec<RoundedConvex> = (0..hulls.len())
        .map(|i| RoundedConvex::still(&hulls[i], poses[i], radii[i]))
        .collect();

    let cpu_best = box_cast_bvh(&targets, cast);
    let cpu_all = box_cast_all_bvh(&targets, cast);

    let gpu_best = gpu.cast(ctx, hulls, poses, radii, cast);
    let gpu_all = gpu.cast_all(ctx, hulls, poses, radii, cast);

    let boxes = GpuSceneBoxCast::scene_boxes(hulls, poses, radii);
    let resident = GpuLbvh::new(ctx).build_resident(ctx, &boxes);
    let gpu_best_res = gpu.cast_resident(ctx, &resident, hulls, poses, radii, cast);
    let gpu_all_res = gpu.cast_all_resident(ctx, &resident, hulls, poses, radii, cast);

    for got in [gpu_best, gpu_best_res] {
        match (cpu_best, got) {
            (None, None) => {}
            (Some(w), Some(g)) => assert_hit_matches(w, g),
            (w, g) => panic!(
                "nearest-hit mismatch: cpu {:?} vs gpu {:?}",
                w.is_some(),
                g.is_some()
            ),
        }
    }

    for got_all in [gpu_all, gpu_all_res] {
        assert_eq!(cpu_all.len(), got_all.len(), "ordered hit count differs");
        for (w, g) in cpu_all.iter().copied().zip(got_all) {
            assert_hit_matches(w, g);
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_nearest_hit_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);
    let hulls = [unit_box(), unit_box(), unit_box()];
    let poses = [
        at(Vec3::new(3.0, 0.0, 0.0)),
        at(Vec3::new(6.0, 0.03, 0.0)),
        at(Vec3::new(9.0, -0.02, 0.0)),
    ];
    let radii = [0.0_f32, 0.0, 0.0];
    let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_oriented_corner_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);
    let hulls = [unit_box()];
    let poses = [at(Vec3::new(3.0, 0.0, 0.0))];
    let radii = [0.0_f32];
    for angle in [0.0_f32, core::f32::consts::FRAC_PI_4, core::f32::consts::FRAC_PI_6] {
        let cast = SceneBoxCast::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::splat(0.5),
            Quat::from_rotation_z(angle),
            20.0,
        );
        run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_clean_miss_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);
    let hulls = [unit_box(), unit_box()];
    let poses = [at(Vec3::new(0.0, 20.0, 0.0)), at(Vec3::new(0.0, -20.0, 0.0))];
    let radii = [0.0_f32, 0.0];
    let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_max_distance_cutoff_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);
    // Near face at x = 9.5; a unit box meets it at centre travel 9.0, beyond a
    // max distance of 5: both engines miss.
    let hulls = [unit_box()];
    let poses = [at(Vec3::new(10.0, 0.0, 0.0))];
    let radii = [0.0_f32];
    let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 5.0);
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_single_leaf_resident_hit_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);
    // A single target builds a degenerate single-leaf resident tree with no
    // internal nodes; the gather must still return that leaf as a candidate.
    let hulls = [unit_box()];
    let poses = [at(Vec3::new(4.0, 0.0, 0.0))];
    let radii = [0.0_f32];
    let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn box_cast_large_scene_prunes_and_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSceneBoxCast::new(&ctx);

    let mut hulls = Vec::new();
    let mut poses = Vec::new();
    let mut radii = Vec::new();
    for k in 0..6 {
        hulls.push(unit_box());
        poses.push(at(Vec3::new(3.0 + 2.0 * (k as f32), 0.01 * (k as f32), 0.0)));
        radii.push(0.0_f32);
    }
    for k in 0..30 {
        hulls.push(unit_box());
        poses.push(at(Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        radii.push(0.0_f32);
    }
    let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 60.0);
    run_parity(&ctx, &gpu, &hulls, &poses, &radii, &cast);
}
