//! Real-device parity: the `GPU` capsule-versus-triangle narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate capsule-versus-triangle pairs to both
//! [`GpuCapsuleTriangleNarrowphase::query`] and
//! [`cpu_capsule_triangle_narrowphase`] and compares the two outputs index by
//! index. The validity decision (a reported contact versus a `None` slot) must
//! match exactly; when both report a contact, the capsule and triangle indices
//! must match exactly and the normal, depth, and point within a tight tolerance,
//! since the only inexact steps are the square root and the handful of
//! segment/barycentric reciprocals. The scenes are built with clear overlaps,
//! clear gaps, and unambiguous pierces (never a grazing `dist ~= rc` boundary),
//! so the tiny floating-point tolerance can never flip a validity flag. Parallel
//! face contact, edge contact, vertex contact, a piercing axis, an axis lying on
//! the face, and clear separation are exercised, on both an axis-aligned triangle
//! and a tilted one, so the closest-feature cascade, the pierce test, and the
//! face-normal fallback all run on each path.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! clamped segment-segment routine is Ericson section 5.1.9; the pierce test is
//! textbook. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_triangle_narrowphase, Capsule, CapsuleTrianglePair, Contact, GpuContext,
    GpuCapsuleTriangleNarrowphase, Triangle,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root and the segment/barycentric reciprocals, both well under this
/// bound over the test scene scales.
const TOL: f32 = 1e-4;

/// Compares one `GPU` contact slot to the `CPU` twin's, allowing only the
/// square-root and reciprocal tolerance.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(index: usize, want: Option<Contact>, got: Option<Contact>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: capsule index must match");
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
fn check(capsules: &[Capsule], triangles: &[Triangle], pairs: &[CapsuleTrianglePair]) {
    let cpu = cpu_capsule_triangle_narrowphase(capsules, triangles, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-triangle parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuCapsuleTriangleNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, capsules, triangles, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per pair like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
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
    // One axis-aligned triangle hit on the interior face (parallel axis), over an
    // edge, past a vertex, pierced through the face, with the axis lying on the
    // face (the degenerate fallback), and clearly missed above. Every case is far
    // from the touching boundary, so the validity flag is unambiguous.
    let tri = unit_triangle();
    let capsules = [
        // face overlap, axis parallel to +z face.
        Capsule::new(Vec3::new(0.2, 0.3, 0.3), Vec3::new(0.4, 0.3, 0.3), 0.5),
        // edge AB overlap (y = 0 edge), axis below it.
        Capsule::new(Vec3::new(0.3, -0.3, 0.0), Vec3::new(0.7, -0.3, 0.0), 0.5),
        // vertex A overlap: nearest endpoint off (0,0,0).
        Capsule::new(Vec3::new(-0.3, -0.4, 0.0), Vec3::new(-0.9, -1.0, 0.0), 0.6),
        // piercing axis crossing the face from +z to -z through the interior.
        Capsule::new(Vec3::new(0.25, 0.25, 0.5), Vec3::new(0.25, 0.25, -0.3), 0.4),
        // axis lying in the face plane over the interior: degenerate fallback.
        Capsule::new(Vec3::new(0.2, 0.3, 0.0), Vec3::new(0.4, 0.3, 0.0), 0.5),
        // clear gap above the face.
        Capsule::new(Vec3::new(0.2, 0.3, 2.0), Vec3::new(0.4, 0.3, 2.0), 0.5),
    ];
    let triangles = [tri];
    let pairs = [
        CapsuleTrianglePair::new(0, 0),
        CapsuleTrianglePair::new(1, 0),
        CapsuleTrianglePair::new(2, 0),
        CapsuleTrianglePair::new(3, 0),
        CapsuleTrianglePair::new(4, 0),
        CapsuleTrianglePair::new(5, 0),
    ];
    check(&capsules, &triangles, &pairs);
}

#[test]
fn tilted_triangle_cases_agree_slot_for_slot() {
    // A triangle lifted off the origin and tilted so none of its vertices share
    // an axis, hit over its interior and clearly missed, so the closest-feature
    // cascade and the face recombination run away from any axis-aligned shortcut
    // on both paths.
    let tri = Triangle::new(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 2.0),
    );
    let centroid = (tri.a + tri.b + tri.c) / 3.0;
    let normal = tri.raw_normal().normalize();
    // Build a short capsule tilted across the face, lifted 0.4 along the normal,
    // so its axis clearly overlaps the interior with the swept radius. The axis is
    // deliberately *not* parallel to the face: one endpoint dips closer to the
    // plane than the other, so a single closest feature wins outright and the
    // contact point is unique on both paths (a face-parallel axis would make every
    // point on the segment equidistant, leaving the closest point ambiguous).
    let tangent = (tri.b - tri.a).normalize();
    let hit_centre = centroid + normal * 0.4;
    let hit = Capsule::new(
        hit_centre - tangent * 0.1 - normal * 0.15,
        hit_centre + tangent * 0.1 + normal * 0.15,
        0.6,
    );
    let miss_centre = centroid + normal * 5.0;
    let miss = Capsule::new(
        miss_centre - tangent * 0.1 - normal * 0.15,
        miss_centre + tangent * 0.1 + normal * 0.15,
        0.6,
    );
    let capsules = [hit, miss];
    let triangles = [tri];
    let pairs = [
        CapsuleTrianglePair::new(0, 0),
        CapsuleTrianglePair::new(1, 0),
    ];
    check(&capsules, &triangles, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    let triangles = [unit_triangle()];
    check(&capsules, &triangles, &[]);
}
