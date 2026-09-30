//! Real-device parity: the `GPU` OBB-halfspace incident-face manifold kernel
//! must reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(box, plane)` couples to both
//! [`GpuObbHalfspaceManifoldNarrowphase::query`] and
//! [`cpu_obb_halfspace_manifold`] and compares the two outputs index by index.
//! The validity decision (a reported manifold versus a `None` slot) and the live
//! point count must match exactly; when both report a manifold, the shared
//! normal must match to within a tight tolerance and each `CPU` point must pair
//! with a `GPU` point (position and depth) to within that tolerance. Points are
//! matched as a multiset because the incident face's corners can be emitted in
//! either order under float rounding, though the algorithm is otherwise
//! operation-for-operation identical. Every scene sits far from the grazing
//! overlap boundary and from an incident-axis tie, so the tolerance can never
//! flip a validity flag, a point count, or the chosen face.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook oriented-bounding-box-versus-halfspace incident-face
//! clipping (the standard box-on-plane resting manifold). No Unreal Engine
//! source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_halfspace_manifold, ContactManifold, GpuContext, GpuObbHalfspaceManifoldNarrowphase,
    Obb, ObbPlanePair, Plane,
};

/// Tolerance on the normal, positions, and depths. This kernel carries no square
/// root or reciprocal, so the only gap is float reassociation of the same
/// products and sums.
const TOL: f32 = 1e-4;

/// The ground plane `y = 0` with an upward outward normal.
fn ground() -> Plane {
    Plane::new(Vec3::Y, 0.0)
}

/// A unit-axis box at `center` with unit half extents.
fn axis_box(center: Vec3) -> Obb {
    Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)
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
            assert_eq!(w.a, g.a, "slot {index}: box index differs");
            assert_eq!(w.b, g.b, "slot {index}: plane index differs");
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
    gpu: &GpuObbHalfspaceManifoldNarrowphase,
    boxes: &[Obb],
    planes: &[Plane],
    pairs: &[ObbPlanePair],
) {
    let want = cpu_obb_halfspace_manifold(boxes, planes, pairs);
    let got = gpu.query(ctx, boxes, planes, pairs);
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
fn gpu_obb_halfspace_manifold_flush_and_edge_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceManifoldNarrowphase::new(&ctx);

    let boxes = [
        // Flush rest: bottom face dips 0.1 below the ground, all four corners
        // cross the surface (count 4).
        axis_box(Vec3::new(0.0, 0.9, 0.0)),
        // Edge landing: tilted 12 degrees about x so two corners of the incident
        // face dip below the ground (count 2), well clear of the 45-degree axis
        // switch.
        Obb::from_quat(
            Vec3::new(0.0, 0.85, 0.0),
            Quat::from_rotation_x(12.0_f32.to_radians()),
            Vec3::ONE,
        ),
    ];
    let planes = [ground()];
    let pairs = [ObbPlanePair::new(0, 0), ObbPlanePair::new(1, 0)];
    run_parity(&ctx, &gpu, &boxes, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_halfspace_manifold_corner_and_floating_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceManifoldNarrowphase::new(&ctx);

    let boxes = [
        // Corner poke: tilted about two axes so only one vertex dips below the
        // surface (count 1).
        Obb::from_quat(
            Vec3::new(0.0, 1.6, 0.0),
            Quat::from_rotation_x(35.0_f32.to_radians())
                * Quat::from_rotation_z(30.0_f32.to_radians()),
            Vec3::ONE,
        ),
        // Floating: the whole box rides well above the ground (None).
        axis_box(Vec3::new(0.0, 3.0, 0.0)),
    ];
    let planes = [ground()];
    let pairs = [ObbPlanePair::new(0, 0), ObbPlanePair::new(1, 0)];
    run_parity(&ctx, &gpu, &boxes, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_halfspace_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceManifoldNarrowphase::new(&ctx);

    let boxes = [axis_box(Vec3::new(0.0, 0.9, 0.0))];
    let planes = [ground()];
    run_parity(&ctx, &gpu, &boxes, &planes, &[]);
}
