//! Real-device parity: the `GPU` multi-point capsule-triangle manifold kernel
//! must reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(capsule, triangle)` couples to both
//! [`GpuCapsuleTriangleManifoldNarrowphase::query`] and
//! [`cpu_capsule_triangle_manifold`] and compares the two outputs index by
//! index. The validity decision (a reported manifold versus a `None` slot) and
//! the live point count must match exactly; when both report a manifold, the
//! shared normal must match to within a tight tolerance and each `CPU` point
//! must pair with a `GPU` point (position and depth) to within that tolerance.
//! Points are matched as a multiset because the two clipped corners can order
//! either way under float rounding, though the algorithm is otherwise
//! operation-for-operation identical. Every scene sits far from the grazing
//! `overlap ~= 0` boundary, so the tolerance can never flip a validity flag or a
//! point count.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: closest-point-on-triangle from Christer Ericson, *Real-Time
//! Collision Detection* (2004); textbook Liang-Barsky segment clip. No Unreal
//! Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_triangle_manifold, Capsule, CapsuleTrianglePair, ContactManifold, GpuContext,
    GpuCapsuleTriangleManifoldNarrowphase, Triangle,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the square root in the distance, the barycentric reciprocals in the
/// closest-point helper, and the reciprocals in the clip crossings.
const TOL: f32 = 1e-4;

/// A large triangle in the `z = 0` plane that contains any unit footprint around
/// the origin, matching the `CPU` golden tests.
fn big_floor() -> Triangle {
    Triangle::new(
        Vec3::new(-10.0, -10.0, 0.0),
        Vec3::new(10.0, -10.0, 0.0),
        Vec3::new(0.0, 10.0, 0.0),
    )
}

/// Builds a capsule from its two axis endpoints and swept radius.
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
            assert_eq!(w.b, g.b, "slot {index}: triangle index differs");
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
    gpu: &GpuCapsuleTriangleManifoldNarrowphase,
    capsules: &[Capsule],
    triangles: &[Triangle],
    pairs: &[CapsuleTrianglePair],
) {
    let want = cpu_capsule_triangle_manifold(capsules, triangles, pairs);
    let got = gpu.query(ctx, capsules, triangles, pairs);
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
fn gpu_capsule_triangle_manifold_face_contacts_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleTriangleManifoldNarrowphase::new(&ctx);

    let capsules = [
        // Lying flat over the floor triangle: both clipped ends penetrate, so a
        // two-point manifold.
        cap(Vec3::new(-2.0, 0.0, 0.4), Vec3::new(2.0, 0.0, 0.4), 0.5),
        // Standing on the plane normal: pierces at one point, collapses to one.
        cap(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 0.5),
        // Hovering well clear of the floor: a clean gap.
        cap(Vec3::new(-2.0, 0.0, 5.0), Vec3::new(2.0, 0.0, 5.0), 0.5),
    ];
    let triangles = [big_floor()];
    let pairs = [
        CapsuleTrianglePair::new(0, 0), // flat: two points
        CapsuleTrianglePair::new(1, 0), // vertical pierce: one point
        CapsuleTrianglePair::new(2, 0), // clear gap
    ];
    run_parity(&ctx, &gpu, &capsules, &triangles, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_triangle_manifold_tilted_and_diagonal_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleTriangleManifoldNarrowphase::new(&ctx);

    let capsules = [
        // One end hugs the face, the other rides far above it: single-point
        // fallback.
        cap(Vec3::new(0.0, 0.0, 0.4), Vec3::new(0.0, 0.0, 3.0), 0.5),
        // A diagonal capsule flat over the face: two clipped corners.
        cap(Vec3::new(-3.0, -2.0, 0.3), Vec3::new(3.0, 2.0, 0.3), 0.5),
    ];
    let triangles = [big_floor()];
    let pairs = [
        CapsuleTrianglePair::new(0, 0), // tilted one-end: one point
        CapsuleTrianglePair::new(1, 0), // diagonal flat: two points
    ];
    run_parity(&ctx, &gpu, &capsules, &triangles, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_triangle_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleTriangleManifoldNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let capsules = [cap(Vec3::new(-2.0, 0.0, 0.4), Vec3::new(2.0, 0.0, 0.4), 0.5)];
    let triangles = [big_floor()];
    let pairs: [CapsuleTrianglePair; 0] = [];
    run_parity(&ctx, &gpu, &capsules, &triangles, &pairs);
}
