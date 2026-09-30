//! Real-device parity: the `GPU` capsule-capsule narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate capsule-capsule pairs to both
//! [`GpuCapsuleCapsuleNarrowphase::query`] and [`cpu_capsule_capsule_narrowphase`]
//! and compares the two outputs index by index. The validity decision (a
//! reported contact versus a `None` slot) must match exactly; when both report a
//! contact, the capsule indices must match exactly and the normal, depth, and
//! point within a tight tolerance, since the only inexact steps are the square
//! root and the reciprocal in the normalisation. The scenes are built with clear
//! overlaps and clear gaps (never a grazing `dist ~= ra + rb` boundary), so the
//! tiny floating-point tolerance can never flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook capsule-capsule closest-feature collision manifold. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_capsule_narrowphase, Capsule, CapsuleCapsulePair, Contact,
    GpuCapsuleCapsuleNarrowphase, GpuContext,
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
            assert_eq!(w.a, g.a, "slot {index}: capsule a must match");
            assert_eq!(w.b, g.b, "slot {index}: capsule b must match");
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
fn check(capsules: &[Capsule], pairs: &[CapsuleCapsulePair]) {
    let cpu = cpu_capsule_capsule_narrowphase(capsules, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-capsule parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuCapsuleCapsuleNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, capsules, pairs);

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
    // A deliberate mix: a clear parallel-flank overlap, a clear crossing-axis
    // overlap, a clear gap, coincident segments (fallback normal), and two
    // degenerate zero-length capsules (sphere-sphere collapse). Every case is
    // far from the touching boundary, so the validity flag is unambiguous.
    let capsules = [
        Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5), // 0
        Capsule::new(Vec3::new(0.0, 0.6, 0.0), Vec3::new(2.0, 0.6, 0.0), 0.5), // 1: flank of 0
        Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5), // 2
        Capsule::new(Vec3::new(0.0, -1.0, 0.6), Vec3::new(0.0, 1.0, 0.6), 0.5), // 3: crosses 2
        Capsule::new(Vec3::new(0.0, 20.0, 0.0), Vec3::new(2.0, 20.0, 0.0), 0.5), // 4: far away
        Capsule::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(5.0, 5.0, 5.0), 0.5), // 5: point
        Capsule::new(Vec3::new(5.6, 5.0, 5.0), Vec3::new(5.6, 5.0, 5.0), 0.5), // 6: point near 5
    ];
    let pairs = [
        CapsuleCapsulePair::new(0, 1), // parallel flank overlap
        CapsuleCapsulePair::new(2, 3), // crossing-axis overlap
        CapsuleCapsulePair::new(0, 4), // clear gap, no contact
        CapsuleCapsulePair::new(0, 2), // overlapping, general skew-ish
        CapsuleCapsulePair::new(5, 6), // degenerate point-point overlap
    ];
    check(&capsules, &pairs);
}

#[test]
fn oblique_capsules_agree_slot_for_slot() {
    // Genuinely skew segments so the closest-point solver exercises a
    // non-trivial (s, t), plus a clear miss, to confirm the projection and the
    // manifold agree between paths on off-axis geometry.
    let capsules = [
        Capsule::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), 0.4), // 0
        Capsule::new(Vec3::new(-1.0, 1.0, 0.5), Vec3::new(1.0, -1.0, 0.5), 0.4), // 1: skew over 0
        Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0), 0.5),   // 2: vertical
        Capsule::new(Vec3::new(0.7, 0.0, 1.0), Vec3::new(2.7, 0.0, 1.0), 0.5),   // 3: touches 2 mid
        Capsule::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(2.0, 3.0, 0.5), 0.3),   // 4: far
    ];
    let pairs = [
        CapsuleCapsulePair::new(0, 1), // skew overlap
        CapsuleCapsulePair::new(2, 3), // T-junction overlap
        CapsuleCapsulePair::new(0, 4), // clear gap
    ];
    check(&capsules, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    check(&capsules, &[]);
}
