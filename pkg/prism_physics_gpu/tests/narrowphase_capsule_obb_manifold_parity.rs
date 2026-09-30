//! Real-device parity: the `GPU` two-point capsule-OBB manifold kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(capsule, box)` couples to both
//! [`GpuCapsuleObbManifoldNarrowphase::query`] and [`cpu_capsule_obb_manifold`]
//! and compares the two outputs index by index. The validity decision (a
//! reported manifold versus a `None` slot) and the live point count must match
//! exactly; when both report a manifold, the shared normal must match to within
//! a tight tolerance and each `CPU` point must pair with a `GPU` point (position
//! and depth) to within that tolerance. Points are matched as a multiset because
//! the two clip-boundary corners can be emitted in either order under float
//! rounding, though the algorithm is otherwise operation-for-operation
//! identical. Every scene sits far from the grazing overlap boundary and from a
//! reference-axis tie, so the tolerance can never flip a validity flag, a point
//! count, or the chosen face.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
//! collision plus Liang-Barsky segment-rectangle clipping. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_obb_manifold, Capsule, CapsuleObbPair, ContactManifold,
    GpuCapsuleObbManifoldNarrowphase, GpuContext, Obb,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the square roots and reciprocals in the closest-feature search, the
/// outside-face normalise, and the edge-crossing solves.
const TOL: f32 = 1e-4;

/// A unit cube centred at the origin, axis-aligned.
fn unit_box() -> Obb {
    Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::splat(1.0))
}

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
            assert_eq!(w.a, g.a, "slot {index}: capsule index differs");
            assert_eq!(w.b, g.b, "slot {index}: box index differs");
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
    gpu: &GpuCapsuleObbManifoldNarrowphase,
    capsules: &[Capsule],
    boxes: &[Obb],
    pairs: &[CapsuleObbPair],
) {
    let want = cpu_capsule_obb_manifold(capsules, boxes, pairs);
    let got = gpu.query(ctx, capsules, boxes, pairs);
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
fn gpu_capsule_obb_manifold_resting_and_poking_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleObbManifoldNarrowphase::new(&ctx);

    let boxes = [unit_box()];
    let capsules = [
        // Lies flat over the +y face: two clipped corners at x = -1 and x = +1.
        cap(Vec3::new(-2.0, 1.4, 0.0), Vec3::new(2.0, 1.4, 0.0), 0.5),
        // Pokes horizontally into the +x face along y: two corners on that face.
        cap(Vec3::new(1.3, -0.6, 0.0), Vec3::new(1.3, 0.6, 0.0), 0.5),
        // Threads straight through the box along x at y = 0.2: the earliest
        // closest feature lands on the -x entry face, so both paths honestly
        // report a single point (parity still compares CPU vs GPU exactly).
        cap(Vec3::new(-3.0, 0.2, 0.0), Vec3::new(3.0, 0.2, 0.0), 0.25),
    ];
    let pairs = [
        CapsuleObbPair::new(0, 0),
        CapsuleObbPair::new(1, 0),
        CapsuleObbPair::new(2, 0),
    ];
    run_parity(&ctx, &gpu, &capsules, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_obb_manifold_clear_and_single_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleObbManifoldNarrowphase::new(&ctx);

    let boxes = [unit_box()];
    let capsules = [
        // Clears the box entirely: a None slot on both paths.
        cap(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.5),
        // One end buried over the +y face, the other lifted clear: the clip keeps
        // fewer than two live corners, so both collapse to the single deepest
        // contact.
        cap(Vec3::new(0.0, 1.4, -0.5), Vec3::new(0.0, 3.0, 0.5), 0.5),
    ];
    let pairs = [CapsuleObbPair::new(0, 0), CapsuleObbPair::new(1, 0)];
    run_parity(&ctx, &gpu, &capsules, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_obb_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleObbManifoldNarrowphase::new(&ctx);

    let capsules = [cap(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    let boxes = [unit_box()];
    run_parity(&ctx, &gpu, &capsules, &boxes, &[]);
}
