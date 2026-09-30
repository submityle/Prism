//! Real-device parity: the `GPU` multi-point OBB-OBB manifold kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(box, box)` couples to both
//! [`GpuObbObbManifoldNarrowphase::query`] and [`cpu_obb_obb_manifold`] and
//! compares the two outputs index by index. The validity decision (a reported
//! manifold versus a `None` slot) and the live point count must match exactly;
//! when both report a manifold, the shared normal must match to within a tight
//! tolerance and each `CPU` point must pair with a `GPU` point (position and
//! depth) to within that tolerance. Points are matched as a multiset because the
//! four-point reduction can order equal-area corners either way under float
//! rounding, though the algorithm is otherwise operation-for-operation identical.
//! Every scene sits far from the grazing `overlap ~= 0` boundary, so the
//! tolerance can never flip a validity flag or a point count.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook reference/incident face-clipping contact manifold. No
//! Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_obb_obb_manifold, ContactManifold, GpuContext, GpuObbObbManifoldNarrowphase, Obb,
    ObbObbPair,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the reciprocal square roots in the edge-axis normalise and segment solver.
const TOL: f32 = 1e-4;

/// Asserts every live `CPU` point pairs with a distinct live `GPU` point within
/// [`TOL`] (position and depth), matched greedily as a multiset.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing point on a parity failure"
)]
fn assert_points_match(index: usize, want: &ContactManifold, got: &ContactManifold) {
    let count = want.count as usize;
    let mut used = [false; 4];
    for wp in want.points.iter().take(count) {
        let mut matched = false;
        for (j, gp) in got.points.iter().take(count).enumerate() {
            if used[j] {
                continue;
            }
            let dp = (wp.position - gp.position).length();
            let dd = (wp.depth - gp.depth).abs();
            if dp <= TOL && dd <= TOL {
                used[j] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            eprintln!(
                "slot {index}: no GPU point matched CPU point {wp:?}\n cpu {want:?}\n gpu {got:?}"
            );
        }
        assert!(matched, "slot {index}: unmatched CPU point {wp:?}");
    }
}

/// Compares one `GPU` manifold slot to the `CPU` twin's, allowing only the tight
/// float tolerance.
fn assert_slot_matches(index: usize, want: Option<ContactManifold>, got: Option<ContactManifold>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: first box index differs");
            assert_eq!(w.b, g.b, "slot {index}: second box index differs");
            assert_eq!(w.count, g.count, "slot {index}: point count differs");
            let dn = (w.normal - g.normal).length();
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            assert_points_match(index, &w, &g);
        }
        (w, g) => panic!(
            "slot {index}: validity mismatch: cpu {:?} vs gpu {:?}",
            w.is_some(),
            g.is_some()
        ),
    }
}

/// Runs both engines over the same scene and asserts slot-for-slot parity.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuObbObbManifoldNarrowphase,
    boxes: &[Obb],
    pairs: &[ObbObbPair],
) {
    let want = cpu_obb_obb_manifold(boxes, pairs);
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
fn gpu_obb_obb_manifold_flat_stacks_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbManifoldNarrowphase::new(&ctx);

    // Axis-aligned face contacts: a flush flat stack (four coplanar points), an
    // offset stack whose incident face overhangs the reference (the clip trims
    // the overlap column), and a clean gap.
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let boxes = [
        Obb::new(Vec3::ZERO, axes, Vec3::ONE),
        // Rests flush on the first, overlapping 0.5 in Y over the full square.
        Obb::new(Vec3::new(0.0, 1.5, 0.0), axes, Vec3::ONE),
        // Rests on the first but shifted in X so its face overhangs.
        Obb::new(Vec3::new(0.5, 1.5, 0.0), axes, Vec3::ONE),
        // Far away: a clean separating axis exists.
        Obb::new(Vec3::new(20.0, 0.0, 0.0), axes, Vec3::ONE),
    ];
    let pairs = [
        ObbObbPair::new(0, 1), // flush flat stack
        ObbObbPair::new(0, 2), // offset stack, clipped column
        ObbObbPair::new(0, 3), // clear gap
    ];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_obb_manifold_rotated_and_edge_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbManifoldNarrowphase::new(&ctx);

    // Rotated face contact (an octagon overlap the reduction trims to four) and
    // an edge-edge contact (a single point), so both manifold branches run.
    let flat = [Vec3::X, Vec3::Y, Vec3::Z];
    let yaw = Quat::from_rotation_y(core::f32::consts::FRAC_PI_4);
    let tumble = Quat::from_euler(
        glam::EulerRot::XYZ,
        core::f32::consts::FRAC_PI_4,
        0.0,
        core::f32::consts::FRAC_PI_4,
    );
    let boxes = [
        Obb::new(Vec3::ZERO, flat, Vec3::ONE),
        // Yawed box resting flat: incident diamond clipped to an octagon.
        Obb::from_quat(Vec3::new(0.0, 1.5, 0.0), yaw, Vec3::ONE),
        // Tumbled box in an edge-edge configuration with the first.
        Obb::from_quat(Vec3::new(1.6, 1.6, 0.0), tumble, Vec3::ONE),
        // Far away: a clean separating axis exists.
        Obb::new(Vec3::new(0.0, 30.0, 0.0), flat, Vec3::ONE),
    ];
    let pairs = [
        ObbObbPair::new(0, 1), // rotated face overlap, reduced to four
        ObbObbPair::new(0, 2), // edge-edge single point
        ObbObbPair::new(0, 3), // clear gap
    ];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_obb_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU OBB-OBB manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbObbManifoldNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let boxes = [Obb::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], Vec3::ONE)];
    let pairs: [ObbObbPair; 0] = [];
    run_parity(&ctx, &gpu, &boxes, &pairs);
}
