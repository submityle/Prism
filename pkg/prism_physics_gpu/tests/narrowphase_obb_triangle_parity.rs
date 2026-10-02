//! Real-device parity: the `GPU` OBB-versus-triangle narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate box-versus-triangle couples to both
//! [`GpuObbTriangleNarrowphase::query`] and [`cpu_obb_triangle_narrowphase`] and
//! compares the two outputs index by index. The validity decision (a reported
//! contact versus a `None` slot) must match exactly; when both report a contact,
//! the box and triangle indices must match exactly and the normal, depth, and
//! point within a tight tolerance, since the only inexact steps are the
//! reciprocal square roots in the thirteen-axis normalisation and the handful of
//! barycentric reciprocals in the closest-point clamps. The scenes are built
//! with clear overlaps and clear gaps (never a grazing `overlap ~= 0`
//! boundary), so the tiny floating-point tolerance can never flip a validity
//! flag. Face, edge, corner, and clearly separated cases are exercised, on both
//! axis-aligned boxes and a box rotated 45 degrees about z, so the separating-
//! axis search and the support-vertex plus closest-point cascade both run on
//! each path.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
//! Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
//! separating-axis minimum-translation manifold and the closest-point-on-
//! triangle Voronoi cascade are Christer Ericson, *Real-Time Collision
//! Detection* (2004), sections 5.2.9 and 5.1.5. No Unreal Engine source or
//! derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_triangle_narrowphase, Contact, GpuContext, GpuObbTriangleNarrowphase, Obb,
    ObbTrianglePair, Triangle,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// reciprocal square roots in the axis normalisation and the barycentric
/// reciprocals, both well under this bound over the test scene scales.
const TOL: f32 = 1e-4;

/// Compares one `GPU` contact slot to the `CPU` twin's, allowing only the
/// reciprocal square-root and reciprocal tolerance.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(index: usize, want: Option<Contact>, got: Option<Contact>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: box index must match");
            assert_eq!(w.b, g.b, "slot {index}: triangle index must match");
            assert!(
                (w.normal - g.normal).length() < TOL,
                "slot {index}: normal {:?} vs {:?}",
                w.normal,
                g.normal
            );
            assert!(
                (w.depth - g.depth).abs() < TOL,
                "slot {index}: depth {} vs {}",
                w.depth,
                g.depth
            );
            assert!(
                (w.point - g.point).length() < TOL,
                "slot {index}: point {:?} vs {:?}",
                w.point,
                g.point
            );
        }
        (want, got) => {
            eprintln!("slot {index}: validity mismatch: cpu {want:?} vs gpu {got:?}");
            panic!("slot {index}: one path reported a contact and the other did not");
        }
    }
}

/// Runs both paths over the same inputs and checks every slot agrees.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn check(boxes: &[Obb], triangles: &[Triangle], pairs: &[ObbTrianglePair]) {
    let cpu = cpu_obb_triangle_narrowphase(boxes, triangles, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU obb-triangle parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuObbTriangleNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, boxes, triangles, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per couple like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
}

/// A unit-axis box at `center` with the given half extents.
fn axis_box(center: Vec3, he: Vec3) -> Obb {
    Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
}

/// The canonical unit triangle in the z = 0 plane with a +z face normal.
fn unit_triangle() -> Triangle {
    Triangle::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    )
}

#[test]
fn axis_aligned_cases_agree_slot_for_slot() {
    // One axis-aligned triangle hit on the interior face (shallow +z push), over
    // the hypotenuse edge (in-plane edge normal), past vertex B (corner), and
    // clearly missed above. Every case is far from the grazing boundary, so the
    // validity flag is unambiguous.
    let tri = unit_triangle();
    let boxes = [
        axis_box(Vec3::new(0.25, 0.25, 0.4), Vec3::splat(0.5)), // face overlap (+z normal)
        axis_box(Vec3::new(0.7, 0.7, 0.0), Vec3::splat(0.3)),   // hypotenuse edge overlap
        axis_box(Vec3::new(1.1, -0.1, 0.0), Vec3::splat(0.3)),  // corner overlap at vertex B
        axis_box(Vec3::new(0.25, 0.25, 3.0), Vec3::splat(0.5)), // clear gap above
    ];
    let triangles = [tri];
    let pairs = [
        ObbTrianglePair::new(0, 0),
        ObbTrianglePair::new(1, 0),
        ObbTrianglePair::new(2, 0),
        ObbTrianglePair::new(3, 0),
    ];
    check(&boxes, &triangles, &pairs);
}

#[test]
fn rotated_box_cases_agree_slot_for_slot() {
    // A box rotated 45 degrees about z, lowered onto the triangle so a slanted
    // edge digs into the face, exercising the edge-edge axes on both paths; and
    // the same orientation lifted clear above the triangle for the separated
    // case. Both are far from the grazing boundary.
    let tri = unit_triangle();
    let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);
    let boxes = [
        Obb::from_quat(Vec3::new(0.3, 0.3, 0.3), rot, Vec3::splat(0.5)), // rotated overlap
        Obb::from_quat(Vec3::new(0.3, 0.3, 4.0), rot, Vec3::splat(0.5)), // clear gap above
    ];
    let triangles = [tri];
    let pairs = [ObbTrianglePair::new(0, 0), ObbTrianglePair::new(1, 0)];
    check(&boxes, &triangles, &pairs);
}

#[test]
fn tilted_triangle_face_agrees_slot_for_slot() {
    // A triangle lifted off the origin and tilted so none of its vertices share
    // an axis, with an axis-aligned box pushed into its interior face along the
    // normal and another clearly clear of it, so the separating-axis search runs
    // away from any axis-aligned shortcut on both paths.
    let tri = Triangle::new(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let centroid = (tri.a + tri.b + tri.c) / 3.0;
    let normal = tri.raw_normal().normalize();
    let hit = centroid + normal * 0.3;
    let miss = centroid + normal * 6.0;
    let boxes = [
        axis_box(hit, Vec3::splat(0.5)),  // interior-face overlap on the tilted triangle
        axis_box(miss, Vec3::splat(0.5)), // clear gap along the normal
    ];
    let triangles = [tri];
    let pairs = [ObbTrianglePair::new(0, 0), ObbTrianglePair::new(1, 0)];
    check(&boxes, &triangles, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let boxes = [axis_box(Vec3::ZERO, Vec3::splat(1.0))];
    let triangles = [unit_triangle()];
    check(&boxes, &triangles, &[]);
}
