//! Real-device parity: the `GPU` sphere-versus-triangle narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate sphere-versus-triangle pairs to both
//! [`GpuSphereTriangleNarrowphase::query`] and [`cpu_sphere_triangle_narrowphase`]
//! and compares the two outputs index by index. The validity decision (a
//! reported contact versus a `None` slot) must match exactly; when both report a
//! contact, the sphere and triangle indices must match exactly and the normal,
//! depth, and point within a tight tolerance, since the only inexact steps are
//! the square root and the handful of barycentric reciprocals in the closest-
//! point clamps. The scenes are built with clear overlaps and clear gaps (never
//! a grazing `dist ~= r` boundary), so the tiny floating-point tolerance can
//! never flip a validity flag. Face, edge, vertex, centre-on-face, and clearly
//! separated cases are exercised, on both an axis-aligned triangle and a tilted
//! one, so the closest-point cascade and the face-normal fallback both run on
//! each path.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! sphere manifold is textbook. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_sphere_triangle_narrowphase, Contact, GpuContext, GpuSphereTriangleNarrowphase, Particle,
    SphereTrianglePair, Triangle,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root and the barycentric reciprocals, both well under this bound over
/// the test scene scales.
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
            assert_eq!(w.a, g.a, "slot {index}: sphere index must match");
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
fn check(spheres: &[Particle], triangles: &[Triangle], pairs: &[SphereTrianglePair]) {
    let cpu = cpu_sphere_triangle_narrowphase(spheres, triangles, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU sphere-triangle parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuSphereTriangleNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, spheres, triangles, pairs);

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
    // One axis-aligned triangle hit on the interior face, over an edge, past a
    // vertex, with the centre exactly on the face (the degenerate fallback), and
    // clearly missed above. Every case is far from the touching boundary, so the
    // validity flag is unambiguous.
    let tri = unit_triangle();
    let spheres = [
        Particle::new(Vec3::new(0.25, 0.25, 0.4), 0.5),  // face overlap (+z normal)
        Particle::new(Vec3::new(0.5, -0.3, 0.0), 0.5),   // edge AB overlap
        Particle::new(Vec3::new(-0.3, -0.4, 0.0), 0.6),  // vertex A overlap
        Particle::new(Vec3::new(0.25, 0.25, 0.0), 0.5),  // centre on face: fallback
        Particle::new(Vec3::new(0.25, 0.25, 2.0), 0.5),  // clear gap above
    ];
    let triangles = [tri];
    let pairs = [
        SphereTrianglePair::new(0, 0),
        SphereTrianglePair::new(1, 0),
        SphereTrianglePair::new(2, 0),
        SphereTrianglePair::new(3, 0),
        SphereTrianglePair::new(4, 0),
    ];
    check(&spheres, &triangles, &pairs);
}

#[test]
fn tilted_triangle_cases_agree_slot_for_slot() {
    // A triangle lifted off the origin and tilted so none of its vertices share
    // an axis, hit over its interior and clearly missed, so the closest-point
    // cascade and the face recombination run away from any axis-aligned
    // shortcut on both paths.
    let tri = Triangle::new(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.0, 0.0, 2.0),
    );
    // Centroid of the triangle, offset along the (normalised) face normal by 0.4
    // so the sphere of radius 0.6 clearly overlaps the interior face.
    let centroid = (tri.a + tri.b + tri.c) / 3.0;
    let normal = tri.raw_normal().normalize();
    let hit = centroid + normal * 0.4;
    let miss = centroid + normal * 5.0;
    let spheres = [
        Particle::new(hit, 0.6),  // interior-face overlap on the tilted triangle
        Particle::new(miss, 0.6), // clear gap along the normal
    ];
    let triangles = [tri];
    let pairs = [SphereTrianglePair::new(0, 0), SphereTrianglePair::new(1, 0)];
    check(&spheres, &triangles, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let spheres = [Particle::new(Vec3::ZERO, 1.0)];
    let triangles = [unit_triangle()];
    check(&spheres, &triangles, &[]);
}
