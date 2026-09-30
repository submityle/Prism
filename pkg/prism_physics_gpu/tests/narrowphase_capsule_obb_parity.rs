//! Real-device parity: the `GPU` capsule-versus-OBB narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact slot for slot.
//!
//! Each test hands a batch of candidate `(capsule, box)` couples to both
//! [`GpuCapsuleObbNarrowphase::query`] and [`cpu_capsule_obb_narrowphase`] and
//! compares the two outputs index by index. The validity decision (a reported
//! contact versus a `None` slot) must match exactly; when both report a contact,
//! the capsule and box indices must match exactly and the normal, depth, and
//! point within a tight tolerance, since the only inexact steps are the square
//! roots and reciprocals in the closest-feature search and the outside-face
//! normalisation. The scenes are built with clear penetrations and clear gaps
//! (never a grazing `dist ~= rc` boundary) and avoid near-tie closest-feature
//! configurations, so the tiny float tolerance can never flip a validity flag
//! and the tie-break can never fork the segment parameter. Both axis-aligned and
//! rotated boxes are exercised.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
//! collision. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_capsule_obb_narrowphase, Capsule, CapsuleObbPair, Contact, GpuCapsuleObbNarrowphase,
    GpuContext, Obb,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square roots and reciprocals in the closest-feature search and the
/// outside-face normalisation, all well under this bound over the test scales.
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
            assert_eq!(w.b, g.b, "slot {index}: box index must match");
            let dn = (w.normal - g.normal).length();
            if dn > TOL {
                eprintln!("slot {index} normal diverged by {dn}\n cpu {w:?}\n gpu {g:?}");
            }
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            let dd = (w.depth - g.depth).abs();
            if dd > TOL {
                eprintln!("slot {index} depth diverged by {dd}\n cpu {w:?}\n gpu {g:?}");
            }
            assert!(dd <= TOL, "slot {index}: depth diverged by {dd}");
            let dp = (w.point - g.point).length();
            if dp > TOL {
                eprintln!("slot {index} point diverged by {dp}\n cpu {w:?}\n gpu {g:?}");
            }
            assert!(dp <= TOL, "slot {index}: point diverged by {dp}");
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
fn check(capsules: &[Capsule], boxes: &[Obb], pairs: &[CapsuleObbPair]) {
    let cpu = cpu_capsule_obb_narrowphase(capsules, boxes, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-OBB parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuCapsuleObbNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, capsules, boxes, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per pair like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
}

/// An axis-aligned unit-half-extent box centred at the origin.
fn unit_box() -> Obb {
    Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::splat(1.0))
}

#[test]
fn axis_aligned_cases_agree_slot_for_slot() {
    // A capsule resting flat on the top face (outside branch), a vertical
    // capsule poking a side face, a capsule threading straight through the box
    // (inside branch), and a capsule floating clear. Every case sits far from
    // the grazing boundary and from a closest-feature tie.
    let boxes = [unit_box()];
    let capsules = [
        Capsule::new(Vec3::new(-2.0, 1.4, 0.0), Vec3::new(2.0, 1.4, 0.0), 0.5), // top face
        Capsule::new(Vec3::new(1.3, 0.0, 0.0), Vec3::new(1.3, 3.0, 0.0), 0.5),  // +x face poke
        Capsule::new(Vec3::new(-3.0, 0.2, 0.0), Vec3::new(3.0, 0.2, 0.0), 0.25), // through inside
        Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.5), // clear gap
    ];
    let pairs = [
        CapsuleObbPair::new(0, 0),
        CapsuleObbPair::new(1, 0),
        CapsuleObbPair::new(2, 0),
        CapsuleObbPair::new(3, 0),
    ];
    check(&capsules, &boxes, &pairs);
}

#[test]
fn tilted_box_and_degenerate_capsule_agree_slot_for_slot() {
    // A box rotated 45 degrees about z, with a capsule sitting above along the
    // rotated up axis, exercises the local-frame projection; a zero-length
    // capsule exercises the sphere collapse of the closest-feature search.
    let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let tilted = Obb::from_quat(Vec3::ZERO, rot, Vec3::splat(1.0));
    let up = rot * Vec3::Y;
    let along = rot * Vec3::X;
    let centre = up * 1.4;
    let boxes = [tilted, unit_box()];
    let capsules = [
        Capsule::new(centre - along * 2.0, centre + along * 2.0, 0.5), // tilted top face
        Capsule::new(Vec3::new(0.0, 1.4, 0.0), Vec3::new(0.0, 1.4, 0.0), 0.5), // degenerate sphere
    ];
    let pairs = [CapsuleObbPair::new(0, 0), CapsuleObbPair::new(1, 1)];
    check(&capsules, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-OBB parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuCapsuleObbNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    let boxes = [unit_box()];
    let pairs: [CapsuleObbPair; 0] = [];
    let cpu = cpu_capsule_obb_narrowphase(&capsules, &boxes, &pairs);
    let gpu = narrow.query(&ctx, &capsules, &boxes, &pairs);
    assert_eq!(cpu.len(), 0, "empty batch: cpu must be empty");
    assert_eq!(gpu.len(), 0, "empty batch: gpu must be empty");
}
