//! Real-device parity: the `GPU` OBB-OBB separating-axis narrow-phase kernel
//! must reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(box, box)` couples to both
//! [`GpuObbObbNarrowphase::query`] and [`cpu_obb_obb_narrowphase`] and compares
//! the two outputs index by index. The validity decision (a reported contact
//! versus a `None` slot) must match exactly; when both report a contact, the two
//! box indices must match exactly and the normal, depth, and point to within a
//! tight tolerance. The only inexact step on this path is the reciprocal square
//! root in the edge-axis `normalize`, so the tolerance stays small. Every scene
//! sits far from the grazing `overlap ~= 0` boundary, so the tolerance can never
//! flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook separating-axis oriented-bounding-box collision
//! manifold. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_obb_narrowphase, Contact, GpuContext, GpuObbObbNarrowphase, Obb, ObbObbPair,
};

/// Tolerance on the normal, depth, and point; the only inexact step on this
/// path is the reciprocal square root in the edge-axis normalise.
const TOL: f32 = 1e-4;

/// Compares one `GPU` contact slot to the `CPU` twin's, allowing only the tight
/// float tolerance.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(index: usize, want: Option<Contact>, got: Option<Contact>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: first box index differs");
            assert_eq!(w.b, g.b, "slot {index}: second box index differs");
            let dn = (w.normal - g.normal).length();
            let dd = (w.depth - g.depth).abs();
            let dp = (w.point - g.point).length();
            if dn > TOL || dd > TOL || dp > TOL {
                eprintln!(
                    "slot {index} diverged: normal {dn}, depth {dd}, point {dp}\n cpu {w:?}\n gpu {g:?}"
                );
            }
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            assert!(dd <= TOL, "slot {index}: depth diverged by {dd}");
            assert!(dp <= TOL, "slot {index}: point diverged by {dp}");
        }
        (w, g) => panic!("slot {index}: validity mismatch: cpu {w:?} vs gpu {g:?}"),
    }
}

/// Runs both engines over the same scene and asserts slot-for-slot parity.
fn run_parity(ctx: &GpuContext, gpu: &GpuObbObbNarrowphase, boxes: &[Obb], pairs: &[ObbObbPair]) {
    let want = cpu_obb_obb_narrowphase(boxes, pairs);
    let got = gpu.query(ctx, boxes, pairs);
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
fn gpu_obb_obb_axis_aligned_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbNarrowphase::new(&ctx);

    // Axis-aligned boxes mixing a clear face overlap and a clear gap so both the
    // penetrating and separated branches run. Every couple sits well away from
    // the grazing overlap == 0 boundary.
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let boxes = [
        Obb::new(Vec3::ZERO, axes, Vec3::ONE),
        // Overlaps the first along +X by 0.5.
        Obb::new(Vec3::new(1.5, 0.0, 0.0), axes, Vec3::ONE),
        // Far away: a clean separating axis exists.
        Obb::new(Vec3::new(10.0, 0.0, 0.0), axes, Vec3::ONE),
        // Fully contained inside a large box centred at the origin.
        Obb::new(Vec3::ZERO, axes, Vec3::splat(4.0)),
    ];
    let pairs = [
        ObbObbPair::new(0, 1), // face overlap
        ObbObbPair::new(0, 2), // clear gap
        ObbObbPair::new(0, 3), // deep containment
    ];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_obb_rotated_edge_edge_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbNarrowphase::new(&ctx);

    // Rotated boxes exercise the edge-edge cross axes and the normalise path.
    // The first pair penetrates through an edge-edge configuration; the second
    // is a rotated pair with a clear separating gap.
    let flat = [Vec3::X, Vec3::Y, Vec3::Z];
    let spin_y = Quat::from_rotation_y(core::f32::consts::FRAC_PI_4);
    let spin_xyz = Quat::from_euler(
        glam::EulerRot::XYZ,
        core::f32::consts::FRAC_PI_6,
        core::f32::consts::FRAC_PI_4,
        core::f32::consts::FRAC_PI_3,
    );
    let boxes = [
        Obb::new(Vec3::ZERO, flat, Vec3::ONE),
        // Yaw-rotated box overlapping the first: the deepest axis is an
        // edge-edge cross, not a face normal.
        Obb::from_quat(Vec3::new(1.4, 0.0, 0.0), spin_y, Vec3::ONE),
        // Tumbled box overlapping the first from another direction.
        Obb::from_quat(Vec3::new(0.0, 1.3, 0.6), spin_xyz, Vec3::new(1.0, 0.5, 1.2)),
        // Tumbled box parked far away: clearly separated.
        Obb::from_quat(Vec3::new(0.0, 12.0, 0.0), spin_xyz, Vec3::ONE),
    ];
    let pairs = [
        ObbObbPair::new(0, 1), // edge-edge penetration
        ObbObbPair::new(0, 2), // tumbled overlap
        ObbObbPair::new(0, 3), // clear gap
        ObbObbPair::new(2, 3), // clear gap between two tumbled boxes
    ];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_obb_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let boxes = [Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)];
    let pairs: [ObbObbPair; 0] = [];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}
