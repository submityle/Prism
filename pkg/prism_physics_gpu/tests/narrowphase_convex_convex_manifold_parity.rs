//! Real-device parity: the `GPU` convex-versus-convex manifold kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(hull, hull)` couples to both
//! [`GpuConvexConvexManifoldNarrowphase::query`] and
//! [`cpu_convex_convex_manifold`] and compares the two outputs index by index.
//! The validity decision (a reported manifold versus a `None` slot) and the
//! live point count must match exactly; when both report a manifold, the shared
//! normal must match to within a tight tolerance and each `CPU` point must pair
//! with a `GPU` point (position and depth) to within that tolerance. Points are
//! matched as a multiset because the four-point reduction can order equal
//! survivors either way under float rounding, though the algorithm is otherwise
//! operation-for-operation identical.
//!
//! Every scene is deliberately asymmetric: the offsets and rotations are
//! irrational-looking so no two faces are a near-tie for `most_aligned_face`
//! and no four survivors are an exact area tie for the reduction. That keeps the
//! `GPU`'s and `CPU`'s float rounding from flipping a face pick, a point count,
//! or a validity flag, which is the only way the two could legitimately diverge.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: Gilbert-Johnson-Keerthi distance, the expanding-polytope
//! algorithm, and a textbook reference/incident face-clipping contact manifold.
//! No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_convex_convex_manifold, ContactManifold, ConvexConvexPair, ConvexHull, ConvexPose,
    GpuContext, GpuConvexConvexManifoldNarrowphase,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the support argmax, the normalise reciprocals, and the barycentric divisions.
const TOL: f32 = 1e-4;

/// Builds a convex hull from `verts` and triangular face loops given by vertex
/// index, reversing any loop whose raw winding would face inward so every face
/// normal points outward regardless of the caller's winding.
fn oriented_hull(verts: Vec<Vec3>, loops: Vec<Vec<u32>>) -> ConvexHull {
    let mut centroid = Vec3::ZERO;
    for v in &verts {
        centroid += *v;
    }
    centroid /= verts.len() as f32;
    let fixed: Vec<Vec<u32>> = loops
        .into_iter()
        .map(|l| {
            let v0 = verts[l[0] as usize];
            let v1 = verts[l[1] as usize];
            let v2 = verts[l[2] as usize];
            let raw = (v1 - v0).cross(v2 - v0);
            if raw.dot(v0 - centroid) >= 0.0 {
                l
            } else {
                l.into_iter().rev().collect()
            }
        })
        .collect();
    ConvexHull::new(verts, fixed)
}

/// A regular tetrahedron inscribed in the cube `[-s, s]^3`, with its four
/// triangular faces oriented outward.
fn tetrahedron(s: f32) -> ConvexHull {
    let verts = vec![
        Vec3::new(s, s, s),
        Vec3::new(s, -s, -s),
        Vec3::new(-s, s, -s),
        Vec3::new(-s, -s, s),
    ];
    let loops = vec![
        vec![0, 1, 2],
        vec![0, 1, 3],
        vec![0, 2, 3],
        vec![1, 2, 3],
    ];
    oriented_hull(verts, loops)
}

/// A regular octahedron with vertices at `+-s` along each axis and its eight
/// triangular faces oriented outward.
fn octahedron(s: f32) -> ConvexHull {
    let verts = vec![
        Vec3::new(s, 0.0, 0.0),
        Vec3::new(-s, 0.0, 0.0),
        Vec3::new(0.0, s, 0.0),
        Vec3::new(0.0, -s, 0.0),
        Vec3::new(0.0, 0.0, s),
        Vec3::new(0.0, 0.0, -s),
    ];
    let loops = vec![
        vec![0, 2, 4],
        vec![0, 4, 3],
        vec![0, 3, 5],
        vec![0, 5, 2],
        vec![1, 4, 2],
        vec![1, 3, 4],
        vec![1, 5, 3],
        vec![1, 2, 5],
    ];
    oriented_hull(verts, loops)
}

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
            assert_eq!(w.a, g.a, "slot {index}: first body index differs");
            assert_eq!(w.b, g.b, "slot {index}: second body index differs");
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
    gpu: &GpuConvexConvexManifoldNarrowphase,
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    pairs: &[ConvexConvexPair],
) {
    let want = cpu_convex_convex_manifold(hulls, poses, pairs);
    let got = gpu.query(ctx, hulls, poses, pairs);
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
fn gpu_convex_convex_manifold_box_stacks_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexManifoldNarrowphase::new(&ctx);

    // Boxes exercised through the ConvexHull path: an asymmetric offset face
    // stack (the clip trims the overhanging columns to a deterministic quad) and
    // a clean gap. The offset is irrational-looking so no clip edge is a tie.
    let hulls = [
        ConvexHull::from_box(Vec3::ONE),
        ConvexHull::from_box(Vec3::ONE),
        ConvexHull::from_box(Vec3::ONE),
    ];
    let poses = [
        ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
        // Rests on body 0, shifted asymmetrically in x and z so the overlap is
        // a trimmed rectangle, overlapping 0.5 in y.
        ConvexPose::new(Vec3::new(0.37, 1.5, -0.21), Quat::IDENTITY),
        // Far away: a clean separating axis exists.
        ConvexPose::new(Vec3::new(25.0, 0.0, 0.0), Quat::IDENTITY),
    ];
    let pairs = [
        ConvexConvexPair::new(0, 1), // offset face stack, clipped quad
        ConvexConvexPair::new(0, 2), // clear gap
    ];
    run_parity(&ctx, &gpu, &hulls, &poses, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_manifold_tetra_and_octa_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexManifoldNarrowphase::new(&ctx);

    // Non-box hulls so the full GJK/EPA path runs on genuinely different
    // geometry: a tetrahedron and an octahedron each tumbled by distinct angles
    // and driven vertex-first into a box's top face. The odd rotations keep
    // every face alignment distinct so no face pick is a near-tie.
    let tumble_a = Quat::from_euler(glam::EulerRot::XYZ, 0.31, 0.52, -0.17);
    let tumble_b = Quat::from_euler(glam::EulerRot::ZYX, -0.23, 0.44, 0.19);
    let hulls = [
        ConvexHull::from_box(Vec3::new(1.5, 1.0, 1.5)),
        tetrahedron(1.2),
        octahedron(1.1),
        ConvexHull::from_box(Vec3::ONE),
    ];
    let poses = [
        ConvexPose::new(Vec3::ZERO, Quat::IDENTITY),
        // Tetra tumbled and lowered so a feature penetrates the box top.
        ConvexPose::new(Vec3::new(0.28, 1.55, 0.19), tumble_a),
        // Octa tumbled and lowered into the same top face, offset aside.
        ConvexPose::new(Vec3::new(-0.33, 1.42, 0.24), tumble_b),
        // Far away: a clean separating axis exists.
        ConvexPose::new(Vec3::new(0.0, 40.0, 0.0), Quat::IDENTITY),
    ];
    let pairs = [
        ConvexConvexPair::new(0, 1), // box vs tumbled tetrahedron
        ConvexConvexPair::new(0, 2), // box vs tumbled octahedron
        ConvexConvexPair::new(1, 2), // tetrahedron vs octahedron
        ConvexConvexPair::new(0, 3), // clear gap
    ];
    run_parity(&ctx, &gpu, &hulls, &poses, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_convex_convex_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU convex-convex manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConvexConvexManifoldNarrowphase::new(&ctx);

    // An empty couple batch must return an empty vector without touching the
    // device, matching the twin.
    let hulls = [ConvexHull::from_box(Vec3::ONE)];
    let poses = [ConvexPose::new(Vec3::ZERO, Quat::IDENTITY)];
    let pairs: [ConvexConvexPair; 0] = [];
    run_parity(&ctx, &gpu, &hulls, &poses, &pairs);
}
