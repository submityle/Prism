//! Real-device parity: the `GPU` OBB-halfspace narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(box, plane)` couples to both
//! [`GpuObbHalfspaceNarrowphase::query`] and [`cpu_obb_halfspace_narrowphase`]
//! and compares the two outputs index by index. The validity decision (a
//! reported contact versus a `None` slot) must match exactly; when both report a
//! contact, the box and plane indices must match exactly and the normal, depth,
//! and point to within a tight tolerance. This path carries no square root or
//! reciprocal, so the only perturbation is fused-multiply-add reassociation in
//! the dot products and the support projection, well under the tolerance. The
//! scenes use clear penetrations and clear gaps (never a grazing `s_min ~= 0`
//! boundary), so the tolerance can never flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook OBB-versus-halfspace (support-function) collision
//! manifold. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_halfspace_narrowphase, Contact, GpuContext, GpuObbHalfspaceNarrowphase, Obb,
    ObbPlanePair, Plane,
};

/// Tolerance on the normal, depth, and point; the only inexact step on this
/// path is fused-multiply-add reassociation in the axis dot products.
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
            assert_eq!(w.a, g.a, "slot {index}: box index differs");
            assert_eq!(w.b, g.b, "slot {index}: plane index differs");
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
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuObbHalfspaceNarrowphase,
    boxes: &[Obb],
    planes: &[Plane],
    pairs: &[ObbPlanePair],
) {
    let want = cpu_obb_halfspace_narrowphase(boxes, planes, pairs);
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
fn gpu_obb_halfspace_axis_aligned_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceNarrowphase::new(&ctx);

    // Axis-aligned boxes against the ground and a slanted wall, mixing clear
    // penetrations and clear gaps so both branches run. All scenes sit far from
    // the grazing s_min == 0 boundary.
    let ground = Plane::new(Vec3::Y, 0.0);
    let inv = 1.0 / (2.0f32).sqrt();
    let wall = Plane::new(Vec3::new(inv, 0.0, inv), -1.0);
    let planes = [ground, wall];
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let boxes = [
        Obb::new(Vec3::new(0.0, -0.5, 0.0), axes, Vec3::ONE), // sunk into ground
        Obb::new(Vec3::new(0.0, 5.0, 0.0), axes, Vec3::ONE),  // clear of ground
        Obb::new(Vec3::new(-2.0, 2.0, -2.0), axes, Vec3::ONE), // pushes into wall
    ];
    let pairs = [
        ObbPlanePair::new(0, 0),
        ObbPlanePair::new(1, 0),
        ObbPlanePair::new(2, 1),
        ObbPlanePair::new(2, 0),
    ];
    run_parity(&ctx, &gpu, &boxes, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_halfspace_rotated_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceNarrowphase::new(&ctx);

    // Rotated boxes exercise the support projection and per-axis sign choice on
    // non-axis-aligned frames. Each dips clearly below the ground.
    let ground = Plane::new(Vec3::Y, 0.0);
    let rot_z = Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);
    let rot_xyz = Quat::from_euler(
        glam::EulerRot::XYZ,
        core::f32::consts::FRAC_PI_6,
        core::f32::consts::FRAC_PI_4,
        core::f32::consts::FRAC_PI_3,
    );
    let boxes = [
        Obb::from_quat(Vec3::new(0.0, 0.5, 0.0), rot_z, Vec3::ONE),
        Obb::from_quat(Vec3::new(3.0, -0.5, 1.0), rot_xyz, Vec3::new(1.0, 0.5, 1.5)),
        Obb::from_quat(Vec3::new(-3.0, 8.0, 0.0), rot_xyz, Vec3::ONE), // clear
    ];
    let planes = [ground];
    let pairs = [
        ObbPlanePair::new(0, 0),
        ObbPlanePair::new(1, 0),
        ObbPlanePair::new(2, 0),
    ];
    run_parity(&ctx, &gpu, &boxes, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_halfspace_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHalfspaceNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let boxes = [Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)];
    let planes = [Plane::new(Vec3::Y, 0.0)];
    let pairs: [ObbPlanePair; 0] = [];
    run_parity(&ctx, &gpu, &boxes, &planes, &pairs);
}
