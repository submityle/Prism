//! Real-device parity: the `GPU` two-point capsule-capsule manifold kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of capsule couples to both
//! [`GpuCapsuleCapsuleManifoldNarrowphase::query`] and
//! [`cpu_capsule_capsule_manifold`] and compares the two outputs index by
//! index. The validity decision (a reported manifold versus a `None` slot) and
//! the live point count must match exactly; when both report a manifold, the
//! shared normal must match to within a tight tolerance and each `CPU` point
//! must pair with a `GPU` point (position and depth) to within that tolerance.
//! Points are matched as a multiset because the two overlap-span corners can be
//! emitted in either order under float rounding, though the algorithm is
//! otherwise operation-for-operation identical. Every scene sits far from the
//! near-parallel test, the overlap-collapse threshold, and the grazing contact
//! boundary, so the tolerance can never flip a validity flag, a point count, or
//! the single-versus-two-point decision.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook capsule-versus-capsule closest-segment collision plus
//! axis-overlap span clipping. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_capsule_manifold, Capsule, CapsuleCapsulePair, ContactManifold,
    GpuCapsuleCapsuleManifoldNarrowphase, GpuContext,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the square roots and reciprocals in the closest-segment search and the
/// overlap-span projection.
const TOL: f32 = 1e-4;

/// A capsule from two endpoints and a radius.
fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
    Capsule::new(p0, p1, radius)
}

/// Asserts every live `CPU` point pairs with a distinct live `GPU` point within
/// [`TOL`] (position and depth), matched greedily as a multiset.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing point on a parity failure"
)]
fn assert_points_match(index: usize, want: &ContactManifold, got: &ContactManifold) {
    let count = want.count as usize;
    let mut used = [false; 4];
    for wp in want.points.iter().take(count) {
        let mut matched = false;
        for (j, gp) in got.points.iter().take(count).enumerate() {
            if used[j] {
                continue;
            }
            let dp = (wp.position - gp.position).length();
            let dd = (wp.depth - gp.depth).abs();
            if dp <= TOL && dd <= TOL {
                used[j] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            eprintln!(
                "slot {index}: no GPU point matched CPU point {wp:?}\n cpu {want:?}\n gpu {got:?}"
            );
        }
        assert!(matched, "slot {index}: unmatched CPU point {wp:?}");
    }
}

/// Compares one `GPU` manifold slot to the `CPU` twin's, allowing only the tight
/// float tolerance.
fn assert_slot_matches(index: usize, want: Option<ContactManifold>, got: Option<ContactManifold>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: first capsule index differs");
            assert_eq!(w.b, g.b, "slot {index}: second capsule index differs");
            assert_eq!(w.count, g.count, "slot {index}: point count differs");
            let dn = (w.normal - g.normal).length();
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            assert_points_match(index, &w, &g);
        }
        (w, g) => panic!(
            "slot {index}: validity mismatch: cpu {:?} vs gpu {:?}",
            w.is_some(),
            g.is_some()
        ),
    }
}

/// Runs both engines over the same scene and asserts slot-for-slot parity.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuCapsuleCapsuleManifoldNarrowphase,
    capsules: &[Capsule],
    pairs: &[CapsuleCapsulePair],
) {
    let want = cpu_capsule_capsule_manifold(capsules, pairs);
    let got = gpu.query(ctx, capsules, pairs);
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
fn gpu_capsule_capsule_manifold_resting_and_partial_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-capsule manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleCapsuleManifoldNarrowphase::new(&ctx);

    let capsules = [
        // Slot 0: two fully overlapping +x capsules offset 0.8 in y: a shared
        // +y normal with two live corners at x = 0 and x = 2.
        cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
        cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
        // Slot 1: b starts halfway along a: the overlap span x in [1, 2] yields
        // two corners at those ends.
        cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
        cap(Vec3::new(1.0, 0.8, 0.0), Vec3::new(3.0, 0.8, 0.0), 0.5),
        // Slot 2: perpendicular axes are not parallel, so the honest single
        // closest-point contact is reported (normal +z, depth 0.7).
        cap(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5),
        cap(Vec3::new(0.0, -1.0, 0.3), Vec3::new(0.0, 1.0, 0.3), 0.5),
    ];
    let pairs = [
        CapsuleCapsulePair::new(0, 1),
        CapsuleCapsulePair::new(2, 3),
        CapsuleCapsulePair::new(4, 5),
    ];
    run_parity(&ctx, &gpu, &capsules, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_capsule_manifold_sphere_and_separated_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-capsule manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleCapsuleManifoldNarrowphase::new(&ctx);

    let capsules = [
        // Slot 0: a zero-length capsule is a sphere, so a single contact is
        // reported regardless of the neighbour's orientation (normal +y).
        cap(Vec3::ZERO, Vec3::ZERO, 0.6),
        cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
        // Slot 1: parallel but farther apart than the radius sum: a None slot on
        // both paths.
        cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
        cap(Vec3::new(0.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.5),
    ];
    let pairs = [CapsuleCapsulePair::new(0, 1), CapsuleCapsulePair::new(2, 3)];
    run_parity(&ctx, &gpu, &capsules, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_capsule_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-capsule manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleCapsuleManifoldNarrowphase::new(&ctx);

    let capsules = [cap(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    run_parity(&ctx, &gpu, &capsules, &[]);
}
