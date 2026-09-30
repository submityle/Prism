//! Real-device parity: the `GPU` sphere-versus-OBB narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate sphere-versus-box pairs to both
//! [`GpuObbNarrowphase::query`] and [`cpu_obb_narrowphase`] and compares the two
//! outputs index by index. The validity decision (a reported contact versus a
//! `None` slot) must match exactly; when both report a contact, the sphere and
//! box indices must match exactly and the normal, depth, and point within a
//! tight tolerance, since the only inexact steps are the square root and the
//! reciprocal in the outside-face normalisation. The scenes are built with clear
//! overlaps and clear gaps (never a grazing `dist ~= r` boundary), so the tiny
//! floating-point tolerance can never flip a validity flag. Both axis-aligned
//! and rotated boxes are exercised.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook sphere-versus-oriented-bounding-box collision manifold.
//! No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_narrowphase, Contact, GpuContext, GpuObbNarrowphase, Obb, Particle, SphereObbPair,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root and the reciprocal in the outside-face normalisation, both well
/// under this bound over the test scene scales.
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
            assert_eq!(w.b, g.b, "slot {index}: box index must match");
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
fn check(spheres: &[Particle], boxes: &[Obb], pairs: &[SphereObbPair]) {
    let cpu = cpu_obb_narrowphase(spheres, boxes, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU sphere-OBB parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuObbNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, spheres, boxes, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per pair like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
}

#[test]
fn axis_aligned_cases_agree_slot_for_slot() {
    // A unit box and a wide box at the origin, hit on a face, an edge, a corner,
    // from inside, and clearly missed. Every case is far from the touching
    // boundary, so the validity flag is unambiguous.
    let spheres = [
        Particle::new(Vec3::new(1.5, 0.0, 0.0), 0.7), // face overlap
        Particle::new(Vec3::new(1.5, 1.5, 0.0), 0.8), // edge overlap
        Particle::new(Vec3::new(1.4, 1.4, 1.4), 0.8), // corner overlap
        Particle::new(Vec3::new(0.5, 0.0, 0.0), 0.3), // centre inside wide box
        Particle::new(Vec3::new(6.0, 0.0, 0.0), 1.0), // clear gap
    ];
    let boxes = [
        Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE),
        Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::splat(2.0)),
    ];
    let pairs = [
        SphereObbPair::new(0, 0),
        SphereObbPair::new(1, 0),
        SphereObbPair::new(2, 0),
        SphereObbPair::new(3, 1),
        SphereObbPair::new(4, 0),
    ];
    check(&spheres, &boxes, &pairs);
}

#[test]
fn rotated_boxes_agree_slot_for_slot() {
    // Boxes rotated about z and about a tilted axis, each hit by a clear overlap
    // and a clear miss, so the axis recombination is exercised on both paths.
    let box_z = Obb::from_quat(
        Vec3::new(0.0, 0.0, 0.0),
        Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        Vec3::ONE,
    );
    let box_tilt = Obb::from_quat(
        Vec3::new(5.0, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.6),
        Vec3::new(1.5, 0.5, 1.0),
    );
    let boxes = [box_z, box_tilt];
    let spheres = [
        Particle::new(Vec3::new(1.6, 0.0, 0.0), 0.7), // clips the z-rotated edge
        Particle::new(Vec3::new(0.0, 3.0, 0.0), 0.5), // clear miss on box_z
        Particle::new(Vec3::new(6.7341, 0.1659, -0.7586), 0.7), // overlaps the tilted box face
        Particle::new(Vec3::new(5.0, 0.0, 6.0), 0.5), // clear miss on box_tilt
    ];
    let pairs = [
        SphereObbPair::new(0, 0),
        SphereObbPair::new(1, 0),
        SphereObbPair::new(2, 1),
        SphereObbPair::new(3, 1),
    ];
    check(&spheres, &boxes, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let spheres = [Particle::new(Vec3::ZERO, 1.0)];
    let boxes = [Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)];
    check(&spheres, &boxes, &[]);
}
