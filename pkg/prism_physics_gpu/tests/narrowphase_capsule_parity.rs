//! Real-device parity: the `GPU` sphere-capsule narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate sphere-capsule pairs to both
//! [`GpuCapsuleNarrowphase::query`] and [`cpu_capsule_narrowphase`] and compares
//! the two outputs index by index. The validity decision (a reported contact
//! versus a `None` slot) must match exactly; when both report a contact, the
//! sphere and capsule indices must match exactly and the normal, depth, and
//! point within a tight tolerance, since the only inexact steps are the square
//! root and the reciprocal in the normalisation. The scenes are built with clear
//! overlaps and clear gaps (never a grazing `dist ~= rs + rc` boundary), so the
//! tiny floating-point tolerance can never flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook sphere-capsule closest-feature collision manifold. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_narrowphase, Capsule, Contact, GpuCapsuleNarrowphase, GpuContext, Particle,
    SphereCapsulePair,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root and the reciprocal in the normalisation, both well under this
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
            assert_eq!(w.a, g.a, "slot {index}: sphere index must match");
            assert_eq!(w.b, g.b, "slot {index}: capsule index must match");
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
fn check(spheres: &[Particle], capsules: &[Capsule], pairs: &[SphereCapsulePair]) {
    let cpu = cpu_capsule_narrowphase(spheres, capsules, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU sphere-capsule parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuCapsuleNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, spheres, capsules, pairs);

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
fn hand_built_cases_agree_slot_for_slot() {
    // A deliberate mix: a clear flank overlap, a clear end-cap overlap, a clear
    // gap, a degenerate zero-length capsule overlap, and a sphere sitting on the
    // axis (fallback normal). Every case is far from the touching boundary, so
    // the validity flag is unambiguous.
    let spheres = [
        Particle::new(Vec3::new(1.0, 0.7, 0.0), 0.5), // flank overlap of capsule 0
        Particle::new(Vec3::new(2.6, 0.0, 0.0), 0.5), // end-cap overlap of capsule 0
        Particle::new(Vec3::new(1.0, 8.0, 0.0), 0.5), // clear gap from capsule 0
        Particle::new(Vec3::new(3.0, 0.0, 0.0), 0.5), // degenerate capsule 1 overlap
        Particle::new(Vec3::new(1.0, 0.0, 0.0), 0.5), // on the axis of capsule 0
    ];
    let capsules = [
        Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
        Capsule::new(Vec3::new(3.2, 0.0, 0.0), Vec3::new(3.2, 0.0, 0.0), 0.5),
    ];
    let pairs = [
        SphereCapsulePair::new(0, 0), // flank overlap
        SphereCapsulePair::new(1, 0), // end-cap overlap
        SphereCapsulePair::new(2, 0), // clear gap, no contact
        SphereCapsulePair::new(3, 1), // degenerate capsule overlap
        SphereCapsulePair::new(4, 0), // coincident with axis, fallback normal
    ];
    check(&spheres, &capsules, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let spheres = [Particle::new(Vec3::ZERO, 1.0)];
    let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    check(&spheres, &capsules, &[]);
}

#[test]
fn oblique_capsules_agree_slot_for_slot() {
    // A capsule skewed through space so the closest-point projection exercises a
    // non-trivial `t`, plus a clear miss, to confirm the projection and the
    // manifold agree between paths on off-axis geometry.
    let capsules = [
        Capsule::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(3.0, 2.0, 1.0), 0.4),
        Capsule::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 5.0, 4.0), 0.3),
    ];
    let spheres = [
        // Near the interior of capsule 0's segment, clearly overlapping.
        Particle::new(Vec3::new(1.0, 0.5, 0.6), 0.6),
        // Clearly clear of capsule 1.
        Particle::new(Vec3::new(5.0, 5.0, 2.0), 0.5),
        // Beyond capsule 0's far endpoint, clamped to the cap.
        Particle::new(Vec3::new(3.4, 2.3, 1.15), 0.5),
    ];
    let pairs = [
        SphereCapsulePair::new(0, 0),
        SphereCapsulePair::new(1, 1),
        SphereCapsulePair::new(2, 0),
    ];
    check(&spheres, &capsules, &pairs);
}
