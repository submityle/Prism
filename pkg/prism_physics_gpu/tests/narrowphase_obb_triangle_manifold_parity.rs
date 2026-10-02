//! Real-device parity: the `GPU` multi-point OBB-triangle manifold kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(box, triangle)` couples to both
//! [`GpuObbTriangleManifoldNarrowphase::query`] and
//! [`cpu_obb_triangle_manifold`] and compares the two outputs index by index.
//! The validity decision (a reported manifold versus a `None` slot) and the
//! live point count must match exactly; when both report a manifold, the shared
//! normal must match to within a tight tolerance and each `CPU` point must pair
//! with a `GPU` point (position and depth) to within that tolerance. Points are
//! matched as a multiset because the four-point reduction can order equal-area
//! corners either way under float rounding, though the algorithm is otherwise
//! operation-for-operation identical. Every scene sits far from the grazing
//! `overlap ~= 0` boundary, so the tolerance can never flip a validity flag or a
//! point count.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook reference/incident face-clipping contact manifold. No
//! Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_triangle_manifold, ContactManifold, GpuContext, GpuObbTriangleManifoldNarrowphase, Obb,
    ObbTrianglePair, Triangle,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the reciprocal square roots in the axis normalise and the barycentric
/// reciprocals in the closest-point helper.
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

/// Builds an axis-aligned box at `center` with half extents `he`.
fn axis_aligned(center: Vec3, he: Vec3) -> Obb {
    Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
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
    gpu: &GpuObbTriangleManifoldNarrowphase,
    boxes: &[Obb],
    triangles: &[Triangle],
    pairs: &[ObbTrianglePair],
) {
    let want = cpu_obb_triangle_manifold(boxes, triangles, pairs);
    let got = gpu.query(ctx, boxes, triangles, pairs);
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
fn gpu_obb_triangle_manifold_box_face_contacts_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbTriangleManifoldNarrowphase::new(&ctx);

    // Box-face reference contacts: a box resting flush on the floor triangle
    // (four coplanar corners), a small triangle poking a box face (the whole
    // triangle survives the clip as three points), and a clean gap.
    let boxes = [
        // Bottom face sits 0.1 below the floor plane: a full four-corner contact.
        axis_aligned(Vec3::new(0.0, 0.0, 0.9), Vec3::splat(1.0)),
        // A larger box whose +x face a small triangle pokes.
        axis_aligned(Vec3::ZERO, Vec3::splat(2.0)),
        // Far above the floor: a clean separating axis exists.
        axis_aligned(Vec3::new(0.0, 0.0, 20.0), Vec3::splat(1.0)),
    ];
    let triangles = [
        big_floor(),
        // A small triangle fully inside the box's y/z slab, penetrating +x.
        Triangle::new(
            Vec3::new(1.5, -0.5, -0.5),
            Vec3::new(1.5, 0.5, -0.5),
            Vec3::new(1.5, 0.0, 0.5),
        ),
    ];
    let pairs = [
        ObbTrianglePair::new(0, 0), // flush four-corner face contact
        ObbTrianglePair::new(1, 1), // small triangle poke, three points
        ObbTrianglePair::new(2, 0), // clear gap
    ];
    run_parity(&ctx, &gpu, &boxes, &triangles, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_triangle_manifold_rotated_and_corner_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbTriangleManifoldNarrowphase::new(&ctx);

    // Tilted and tumbled boxes dipping into the floor triangle, so the
    // triangle-face reference clip (incident box quad against the triangle edge
    // planes) and the edge/corner representative-point branch both run alongside
    // a clean gap.
    let tilt = Quat::from_rotation_x(core::f32::consts::FRAC_PI_4);
    let tumble = Quat::from_euler(
        glam::EulerRot::XYZ,
        core::f32::consts::FRAC_PI_4,
        core::f32::consts::FRAC_PI_4,
        0.0,
    );
    let boxes = [
        // Tilted about x: a bottom edge dips below the floor plane.
        Obb::from_quat(Vec3::new(0.0, 0.0, 0.6), tilt, Vec3::ONE),
        // Tumbled about two axes: a corner dips below the floor plane.
        Obb::from_quat(Vec3::new(0.0, 0.0, 0.5), tumble, Vec3::ONE),
        // Far above the floor: a clean separating axis exists.
        Obb::from_quat(Vec3::new(0.0, 0.0, 25.0), tumble, Vec3::ONE),
    ];
    let triangles = [big_floor()];
    let pairs = [
        ObbTrianglePair::new(0, 0), // tilted edge into the triangle face
        ObbTrianglePair::new(1, 0), // tumbled corner into the triangle face
        ObbTrianglePair::new(2, 0), // clear gap
    ];
    run_parity(&ctx, &gpu, &boxes, &triangles, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_triangle_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-triangle manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbTriangleManifoldNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let boxes = [axis_aligned(Vec3::ZERO, Vec3::ONE)];
    let triangles = [big_floor()];
    let pairs: [ObbTrianglePair; 0] = [];
    run_parity(&ctx, &gpu, &boxes, &triangles, &pairs);
}
