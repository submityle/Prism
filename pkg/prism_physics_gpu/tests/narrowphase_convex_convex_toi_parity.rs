//! Real-device parity: the `GPU` convex-versus-convex conservative-advancement
//! kernel must reproduce the `CPU` golden twin's time of impact slot for slot.
//!
//! Each test hands a batch of swept `(hull, hull)` couples to both
//! [`GpuConvexConvexToiNarrowphase::query`] and [`cpu_convex_convex_toi_rounded`] and
//! compares the two outputs index by index. The hit decision (a reported
//! impact versus a `None` slot) must match exactly; when both report an
//! impact, the impact time, the contact point, and the contact normal must all
//! match to within a tight tolerance, since the kernel runs the advance loop
//! operation-for-operation with the reference.
//!
//! Every scene is deliberately asymmetric: offsets, extents, and spins are
//! chosen so no gap closes along two directions at once and no support argmax
//! is a near-tie. That keeps the `GPU`'s and `CPU`'s float rounding from
//! flipping the chosen normal, the converged time, or the hit flag, which is
//! the only way the two could legitimately diverge.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: conservative advancement (Mirtich, 2000; van den Bergen, 2004)
//! over a Gilbert-Johnson-Keerthi distance walk. No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_convex_convex_toi_rounded, BodyMotion, ConvexConvexSweepPair, ConvexHull, ConvexPose,
    GpuContext, GpuConvexConvexToiNarrowphase, Toi,
};

/// Tolerance on the impact time, point, and normal; the only inexact steps are
/// the support argmax, the normalise reciprocals, and the advance division.
const TOL: f32 = 1e-4;

/// Compares one `GPU` time-of-impact slot to the `CPU` twin's, allowing only
/// the tight float tolerance.
#[expect(
    clippy::print_stderr,
    reason = "a mismatch must reach the test log to pinpoint the diverging slot"
)]
fn assert_slot_matches(index: usize, want: Option<Toi>, got: Option<Toi>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            let dt = (w.time - g.time).abs();
            let dp = (w.point - g.point).length();
            let dn = (w.normal - g.normal).length();
            if dt > TOL || dp > TOL || dn > TOL {
                eprintln!("slot {index}: cpu {w:?} vs gpu {g:?}");
            }
            assert!(dt <= TOL, "slot {index}: time diverged by {dt}");
            assert!(dp <= TOL, "slot {index}: point diverged by {dp}");
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
        }
        (w, g) => panic!(
            "slot {index}: hit mismatch: cpu {:?} vs gpu {:?}",
            w.is_some(),
            g.is_some()
        ),
    }
}

/// Runs both engines over the same swept scene and asserts slot-for-slot parity.
#[expect(
    clippy::too_many_arguments,
    reason = "the parity harness mirrors the kernel's full body-indexed input: \
              hulls, poses, motions, radii, pairs, and the two step scalars"
)]
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuConvexConvexToiNarrowphase,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    motions: &[BodyMotion],
    radii: &[f32],
    pairs: &[ConvexConvexSweepPair],
    dt: f32,
    target: f32,
) {
    let want = cpu_convex_convex_toi_rounded(hulls, poses, motions, radii, pairs, dt, target);
    let got = gpu.query(ctx, hulls, poses, motions, radii, pairs, dt, target);
    assert_eq!(want.len(), got.len(), "slot count differs");
    for (i, (w, g)) in want.into_iter().zip(got).enumerate() {
        assert_slot_matches(i, w, g);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_linear_sweeps_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    // Six bodies feeding four couples that each exercise a distinct outcome,
    // all sharing one substep (dt = 2.0, touching target). Offsets are nudged
    // off the axes so no gap closes along two directions at once.
    let hulls = [
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.5)),
    ];
    let poses = [
        ConvexPose::new(Vec3::new(-3.0, 0.07, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(0.0, 20.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(-5.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(5.0, 0.11, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(-2.0, 0.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(2.5, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::new(Vec3::new(-1.5, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
        // Pair (3, 4): nine apart closing at 1.7, so impact near t = 4 lands
        // well beyond dt = 2.0 and must be reported as a miss.
        BodyMotion::new(Vec3::new(0.8, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::new(Vec3::new(-0.9, 0.0, 0.0), Vec3::ZERO),
        // Pair (5, 1): already apart and moving farther apart; a clean miss.
        BodyMotion::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO),
    ];
    let pairs = [
        ConvexConvexSweepPair::new(0, 1),
        ConvexConvexSweepPair::new(0, 2),
        ConvexConvexSweepPair::new(3, 4),
        ConvexConvexSweepPair::new(5, 1),
    ];
    // Every body is a sharp box: zero convex radius across the batch.
    let radii = [0.0_f32; 6];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 2.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_rotating_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    // A long thin paddle spinning about z while a smaller box drifts in on +x,
    // nudged off the axis so the swept contact normal is unambiguous.
    let hulls = [
        ConvexHull::from_box(Vec3::new(1.0, 0.2, 0.3)),
        ConvexHull::from_box(Vec3::splat(0.4)),
    ];
    let poses = [
        ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
        ConvexPose::new(Vec3::new(3.0, 0.05, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 3.5)),
        BodyMotion::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO),
    ];
    let pairs = [ConvexConvexSweepPair::new(0, 1)];
    let radii = [0.0_f32; 2];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 2.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_speculative_target_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    // The same linear approach reported against a non-zero speculative margin,
    // so the impact fires early at the target separation rather than at touch.
    let hulls = [
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_box(Vec3::splat(0.6)),
    ];
    let poses = [
        ConvexPose::new(Vec3::new(-3.0, 0.09, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::ZERO),
    ];
    let pairs = [ConvexConvexSweepPair::new(0, 1)];
    let radii = [0.0_f32; 2];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 3.0, 0.25);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    let hulls = [ConvexHull::from_box(Vec3::splat(0.5))];
    let poses = [ConvexPose::new(Vec3::ZERO, Quat::IDENTITY)];
    let motions = [BodyMotion::still()];
    let pairs: [ConvexConvexSweepPair; 0] = [];
    let radii = [0.0_f32; 1];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 1.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_rounded_spheres_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    // Two rounded spheres (point cores plus a convex radius) closing head-on,
    // and a third drifting far above as a clean miss. The convex radius lives in
    // the hull header, so this exercises the device's rounded shape cast end to
    // end against the inflated-core CPU golden.
    let hulls = [
        ConvexHull::from_point(),
        ConvexHull::from_point(),
        ConvexHull::from_point(),
    ];
    let radii = [0.5_f32, 0.6, 0.4];
    let poses = [
        ConvexPose::new(Vec3::new(-3.0, 0.05, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(0.0, 25.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::new(Vec3::new(-1.5, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
    ];
    let pairs = [
        ConvexConvexSweepPair::new(0, 1),
        ConvexConvexSweepPair::new(0, 2),
    ];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 3.0, 0.0);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_toi_rounded_mixed_shapes_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex TOI parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexToiNarrowphase::new(&ctx);

    // A rounded sphere sweeping into a static sharp box, plus a rounded capsule
    // (segment core) spinning as it drifts onto a static rounded box: a mix of
    // point, segment, and polyhedral cores with and without a convex radius,
    // all off the axes so no normal is a near-tie.
    let hulls = [
        ConvexHull::from_point(),
        ConvexHull::from_box(Vec3::splat(0.5)),
        ConvexHull::from_segment(Vec3::Y, 0.6),
        ConvexHull::from_box(Vec3::new(0.4, 0.5, 0.6)),
    ];
    let radii = [0.5_f32, 0.0, 0.2, 0.1];
    let poses = [
        ConvexPose::new(Vec3::new(-3.0, 0.07, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
        ConvexPose::new(Vec3::new(-2.5, 0.09, 0.0), Quat::IDENTITY),
        ConvexPose::new(Vec3::new(2.5, 0.0, 0.0), Quat::IDENTITY),
    ];
    let motions = [
        BodyMotion::new(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO),
        BodyMotion::still(),
        BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.5)),
        BodyMotion::still(),
    ];
    let pairs = [
        ConvexConvexSweepPair::new(0, 1),
        ConvexConvexSweepPair::new(2, 3),
    ];

    run_parity(&ctx, &gpu, &hulls, &poses, &motions, &radii, &pairs, 2.0, 0.0);
}
