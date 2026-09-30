//! Real-device parity: the `GPU` capsule-halfspace narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(capsule, plane)` couples to both
//! [`GpuCapsuleHalfspaceNarrowphase::query`] and
//! [`cpu_capsule_halfspace_manifold`] and compares the two outputs index by
//! index. The point count (0, 1, or 2) must match exactly; when both report a
//! manifold, the capsule and plane indices must match exactly and the normal and
//! each point's position and depth to within a tight tolerance. Points are
//! emitted in axis order (`p0` then `p1`) on both paths, so the slots line up
//! directly. This path carries no square root or reciprocal, so the only
//! perturbation is fused-multiply-add reassociation in the endpoint dot
//! products, well under the tolerance. The scenes use clear penetrations and
//! clear gaps (never a grazing `s == rc` boundary), so the tolerance can never
//! flip a point count.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook capsule-versus-halfspace (affine support) collision
//! manifold. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_capsule_halfspace_manifold, Capsule, CapsulePlanePair, ContactManifold,
    GpuCapsuleHalfspaceNarrowphase, GpuContext, Plane,
};

/// Tolerance on the normal, positions, and depths; the only inexact step on this
/// path is fused-multiply-add reassociation in the endpoint dot products.
const TOL: f32 = 1e-4;

/// Compares one `GPU` manifold slot to the `CPU` twin's, allowing only the tight
/// float tolerance. Points are compared index by index because both paths emit
/// them in axis order.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(index: usize, want: Option<ContactManifold>, got: Option<ContactManifold>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: capsule index differs");
            assert_eq!(w.b, g.b, "slot {index}: plane index differs");
            assert_eq!(w.count, g.count, "slot {index}: point count differs");
            let dn = (w.normal - g.normal).length();
            if dn > TOL {
                eprintln!("slot {index} normal diverged by {dn}\n cpu {w:?}\n gpu {g:?}");
            }
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            for k in 0..(w.count as usize) {
                let dp = (w.points[k].position - g.points[k].position).length();
                let dd = (w.points[k].depth - g.points[k].depth).abs();
                if dp > TOL || dd > TOL {
                    eprintln!(
                        "slot {index} point {k} diverged: pos {dp}, depth {dd}\n cpu {w:?}\n gpu {g:?}"
                    );
                }
                assert!(
                    dp <= TOL,
                    "slot {index} point {k}: position diverged by {dp}"
                );
                assert!(dd <= TOL, "slot {index} point {k}: depth diverged by {dd}");
            }
        }
        (w, g) => panic!("slot {index}: validity mismatch: cpu {w:?} vs gpu {g:?}"),
    }
}

/// Runs both engines over the same scene and asserts slot-for-slot parity.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuCapsuleHalfspaceNarrowphase,
    capsules: &[Capsule],
    planes: &[Plane],
    pairs: &[CapsulePlanePair],
) {
    let want = cpu_capsule_halfspace_manifold(capsules, planes, pairs);
    let got = gpu.query(ctx, capsules, planes, pairs);
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
fn gpu_capsule_halfspace_axis_aligned_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleHalfspaceNarrowphase::new(&ctx);

    // A flat capsule resting on the ground (two points), a vertical capsule with
    // only its lower end buried (one point), and a capsule floating clear
    // (none). All scenes sit far from the grazing s == rc boundary.
    let ground = Plane::new(Vec3::Y, 0.0);
    let capsules = [
        Capsule::new(Vec3::new(-1.0, 0.3, 0.0), Vec3::new(1.0, 0.3, 0.0), 0.5), // flat -> two
        Capsule::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, 3.0, 0.0), 0.5),  // vertical -> one
        Capsule::new(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0), 0.5), // clear -> none
    ];
    let planes = [ground];
    let pairs = [
        CapsulePlanePair::new(0, 0),
        CapsulePlanePair::new(1, 0),
        CapsulePlanePair::new(2, 0),
    ];
    run_parity(&ctx, &gpu, &capsules, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_halfspace_slanted_and_tilted_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleHalfspaceNarrowphase::new(&ctx);

    // A slanted plane with a general normal exercises the full dot products; a
    // tilted capsule with one buried end exercises the single-point branch; a
    // degenerate zero-length capsule exercises the sphere collapse.
    let inv = 1.0 / (2.0f32).sqrt();
    let slanted = Plane::new(Vec3::new(inv, inv, 0.0), 0.0);
    let ground = Plane::new(Vec3::Y, 0.0);
    let capsules = [
        Capsule::new(Vec3::new(-0.2, -0.2, -1.0), Vec3::new(-0.2, -0.2, 1.0), 0.5), // both ends
        Capsule::new(Vec3::new(0.0, -0.4, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.5),     // one end
        Capsule::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, 0.2, 0.0), 0.5),      // degenerate
    ];
    let planes = [slanted, ground];
    let pairs = [
        CapsulePlanePair::new(0, 0),
        CapsulePlanePair::new(1, 1),
        CapsulePlanePair::new(2, 1),
    ];
    run_parity(&ctx, &gpu, &capsules, &planes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_capsule_halfspace_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU capsule-halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuCapsuleHalfspaceNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
    let planes = [Plane::new(Vec3::Y, 0.0)];
    let pairs: [CapsulePlanePair; 0] = [];
    run_parity(&ctx, &gpu, &capsules, &planes, &pairs);
}
