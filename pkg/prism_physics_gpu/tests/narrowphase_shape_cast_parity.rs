//! Real-device parity for the Jolt-style shape cast: the CPU `cast_shape` /
//! `cast_shape_all` collectors must agree with the `GPU` convex-convex
//! conservative-advancement kernel driven over the same one-against-many scene.
//!
//! `cast_shape` is a thin CPU collector over the per-pair time of impact, and
//! that per-pair kernel already has its own slot-for-slot `GPU` parity suite.
//! This suite closes the loop at the query level: it drives the `GPU` kernel
//! across the cast shape paired with every target, reduces the readback with
//! the exact earliest-hit and ordering rules the collector uses, and asserts
//! the `GPU`-derived hit matches the CPU golden target index, impact time,
//! contact point, and contact normal.
//!
//! The scenes are asymmetric so no two gaps close at once and no support argmax
//! is a near-tie, keeping the `GPU` and CPU float rounding from flipping the
//! chosen target, the time, or the hit flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: conservative advancement (Mirtich, 2000; van den Bergen, 2004)
//! over a Gilbert-Johnson-Keerthi distance walk. No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cast_shape, cast_shape_all, BodyMotion, ConvexConvexSweepPair, ConvexHull, ConvexPose,
    GpuContext, GpuConvexConvexToiNarrowphase, RoundedConvex, ShapeCastHit, Toi,
};

/// Tolerance on the impact time, point, and normal, matching the per-pair
/// suite: the only inexact steps are the support argmax, the normalise
/// reciprocals, and the advance division.
const TOL: f32 = 1e-4;

/// Reduces a per-pair `GPU` readback (slot `k` is the cast shape versus target
/// `k`) to the earliest hit using the same rule as `cast_shape`: strictly
/// earlier time replaces, so an exact time tie keeps the lower target index.
fn gpu_earliest(results: &[Option<Toi>]) -> Option<ShapeCastHit> {
    let mut best: Option<ShapeCastHit> = None;
    for (k, slot) in results.iter().enumerate() {
        let Some(toi) = *slot else {
            continue;
        };
        if best.is_none_or(|current| toi.time < current.toi.time) {
            best = Some(ShapeCastHit {
                target: k as u32,
                toi,
            });
        }
    }
    best
}

/// Orders a per-pair `GPU` readback into every hit by increasing time, ties
/// broken by ascending target index, matching `cast_shape_all`.
fn gpu_all(results: &[Option<Toi>]) -> Vec<ShapeCastHit> {
    let mut hits: Vec<ShapeCastHit> = results
        .iter()
        .enumerate()
        .filter_map(|(k, slot)| {
            slot.map(|toi| ShapeCastHit {
                target: k as u32,
                toi,
            })
        })
        .collect();
    hits.sort_by(|lhs, rhs| {
        lhs.toi
            .time
            .partial_cmp(&rhs.toi.time)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    hits
}

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

/// Builds the body-indexed `GPU` tables for a cast: body 0 is the shape, bodies
/// `1..=n` are the targets, and pair `k` sweeps the shape (as `A`) against
/// target `k` (as `B`) so the resulting normal points target toward shape, the
/// same convention `cast_shape` reports.
fn gpu_pairs(target_count: usize) -> Vec<ConvexConvexSweepPair> {
    (0..target_count)
        .map(|k| ConvexConvexSweepPair::new(0, (k + 1) as u32))
        .collect()
}

/// Runs the CPU collectors and the `GPU`-driven reduction over one scene and
/// asserts both the earliest hit and the full ordered hit list agree.
#[expect(
    clippy::too_many_arguments,
    reason = "the harness mirrors the kernel's body-indexed input: hulls, poses,               motions, radii, and the two step scalars, plus the context and kernel"
)]
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuConvexConvexToiNarrowphase,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    radii: &[f32],
    dt: f32,
    target_sep: f32,
) {
    // CPU view: shape is body 0, targets are bodies 1.. in order.
    let shape = RoundedConvex::new(&hulls[0], poses[0], motions[0], radii[0]);
    let targets: Vec<RoundedConvex> = (1..hulls.len())
        .map(|i| RoundedConvex::new(&hulls[i], poses[i], motions[i], radii[i]))
        .collect();
    let cpu_best = cast_shape(&shape, &targets, dt, target_sep);
    let cpu_all = cast_shape_all(&shape, &targets, dt, target_sep);

    // GPU view: the same bodies fed to the per-pair kernel, reduced the same way.
    let pairs = gpu_pairs(targets.len());
    let results = gpu.query(ctx, hulls, poses, motions, radii, &pairs, dt, target_sep);
    let gpu_best = gpu_earliest(&results);
    let gpu_hits = gpu_all(&results);

    match (cpu_best, gpu_best) {
        (None, None) => {}
        (Some(w), Some(g)) => assert_hit_matches(w, g),
        (w, g) => panic!(
            "earliest-hit mismatch: cpu {:?} vs gpu {:?}",
            w.is_some(),
            g.is_some()
        ),
    }
    assert_eq!(cpu_all.len(), gpu_hits.len(), "ordered hit count differs");
    for (w, g) in cpu_all.into_iter().zip(gpu_hits) {
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
fn shape_cast_nearest_blocker_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

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
fn shape_cast_rounded_mixed_targets_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

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
    // Capsule radius 0.3, sphere radius 0.4, bevelled box radius 0.1, off-path 0.
    let radii = [0.3_f32, 0.4, 0.1, 0.0];
    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, 1.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn shape_cast_all_targets_off_path_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU shape-cast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

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
