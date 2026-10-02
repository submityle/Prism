//! Real-device parity: the `GPU` multi-point box-versus-heightfield manifold
//! kernel must reproduce the `CPU` golden twin's contact manifolds slot for
//! slot.
//!
//! Each test hands a batch of `(box, heightfield)` pairs to both
//! [`GpuObbHeightfieldManifoldNarrowphase::query`] and
//! [`cpu_obb_heightfield_manifold`] and compares the two outputs index by
//! index. The validity decision (a reported manifold versus a `None` slot) and
//! the live point count must match exactly; when both report a manifold, the
//! shared normal must match to within a tight tolerance and each `CPU` point
//! must pair with a `GPU` point (position and depth) to within that tolerance.
//! Points are matched as a multiset because the clipped corners can order either
//! way under float rounding, though the algorithm is otherwise
//! operation-for-operation identical. Every scene sits far from the grazing
//! `overlap ~= 0` boundary, so the tolerance can never flip a validity flag or a
//! point count.
//!
//! The incline scene uses a single uniform slope (every cell triangle coplanar)
//! rather than a symmetric ridge: on a ridge the two faces have near-equal peak
//! depth, so the strictly-greater deepest tie-break can fall to opposite
//! triangles on `CPU` and `GPU` under float rounding and flip the reference
//! normal. A uniform slope shares one normal across every triangle, so whichever
//! triangle wins the tie the reported normal is identical.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
//! Akenine-Moller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
//! reference-face clip, the Sutherland-Hodgman clip, and the four-point manifold
//! reduction are Christer Ericson, *Real-Time Collision Detection* (2004); the
//! heightfield cell triangulation is textbook. No Unreal Engine source or
//! derived code.

use prism_physics_gpu::{
    cpu_obb_heightfield_manifold, ContactManifold, GpuContext,
    GpuObbHeightfieldManifoldNarrowphase, Heightfield, HeightfieldObbPair, Obb,
};

/// Tolerance on the normal, positions, and depths; the only inexact steps are
/// the axis normalise, the floor in the cell lookup, and the reciprocals in the
/// clip crossings.
const TOL: f32 = 1e-4;

/// A flat `rows x cols` field at `y = 0` with unit spacing, origin at the
/// world origin, matching the `CPU` golden tests.
fn flat_field(rows: u32, cols: u32) -> Heightfield {
    Heightfield::new(
        rows,
        cols,
        1.0,
        glam::Vec3::ZERO,
        vec![0.0; (rows * cols) as usize],
    )
}

/// An axis-aligned box at `center` with half extents `he`, matching the `CPU`
/// golden tests.
fn axis_box(center: glam::Vec3, he: glam::Vec3) -> Obb {
    Obb::new(
        center,
        [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z],
        he,
    )
}

/// Asserts every live `CPU` point pairs with a distinct live `GPU` point
/// within [`TOL`] (position and depth), matched greedily as a multiset.
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

/// Compares one `GPU` manifold slot to the `CPU` twin's, allowing only the
/// tight float tolerance.
fn assert_slot_matches(index: usize, want: Option<ContactManifold>, got: Option<ContactManifold>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: box index differs");
            assert_eq!(w.b, g.b, "slot {index}: heightfield index differs");
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
    gpu: &GpuObbHeightfieldManifoldNarrowphase,
    boxes: &[Obb],
    fields: &[Heightfield],
    pairs: &[HeightfieldObbPair],
) {
    let want = cpu_obb_heightfield_manifold(boxes, fields, pairs);
    let got = gpu.query(ctx, boxes, fields, pairs);
    assert_eq!(want.len(), got.len(), "slot count differs");
    for (index, (w, g)) in want.into_iter().zip(got).enumerate() {
        assert_slot_matches(index, w, g);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_heightfield_manifold_flat_corners_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-heightfield manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHeightfieldManifoldNarrowphase::new(&ctx);

    // A unit box centred at (1, 0.4, 1) over a flat 3x3 field penetrates by 0.1
    // with a +Y normal and a stable four-corner manifold.
    let fields = [flat_field(3, 3)];
    let boxes = [axis_box(
        glam::Vec3::new(1.0, 0.4, 1.0),
        glam::Vec3::splat(0.5),
    )];
    let pairs = [HeightfieldObbPair::new(0, 0)];
    // Guard the scene is meaningful: a full four-corner manifold, not a trivial
    // None or single point, so parity actually exercises the pooling path.
    let golden = cpu_obb_heightfield_manifold(&boxes, &fields, &pairs);
    assert!(
        golden[0].as_ref().is_some_and(|m| m.count == 4),
        "flat scene must yield a four-corner contact, got {golden:?}"
    );
    run_parity(&ctx, &gpu, &boxes, &fields, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_heightfield_manifold_misses_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-heightfield manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHeightfieldManifoldNarrowphase::new(&ctx);

    let fields = [flat_field(3, 3)];
    let boxes = [
        // Footprint wholly beyond the grid: no candidate cells, no contact.
        axis_box(glam::Vec3::new(50.0, 0.4, 50.0), glam::Vec3::splat(0.5)),
        // Hovering well above the ground inside the footprint: cells visited but
        // no triangle penetrates.
        axis_box(glam::Vec3::new(1.0, 5.0, 1.0), glam::Vec3::splat(0.5)),
    ];
    let pairs = [
        HeightfieldObbPair::new(0, 0),
        HeightfieldObbPair::new(1, 0),
    ];
    run_parity(&ctx, &gpu, &boxes, &fields, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_heightfield_manifold_incline_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-heightfield manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHeightfieldManifoldNarrowphase::new(&ctx);

    // A uniform incline rising +0.2 per unit z: every cell triangle is coplanar,
    // so a box resting on the slope reports one tilted normal regardless of which
    // sub-manifold wins the deepest tie-break. This exercises the sloped-terrain
    // pooling and reduction without the razor-thin symmetric-ridge tie that float
    // rounding would otherwise break differently on each engine.
    let heights = vec![
        0.0, 0.0, 0.0, // row 0 (z = 0)
        0.2, 0.2, 0.2, // row 1 (z = 1)
        0.4, 0.4, 0.4, // row 2 (z = 2)
    ];
    let fields = [Heightfield::new(3, 3, 1.0, glam::Vec3::ZERO, heights)];
    let boxes = [axis_box(
        glam::Vec3::new(1.0, 0.6, 1.0),
        glam::Vec3::splat(0.5),
    )];
    let pairs = [HeightfieldObbPair::new(0, 0)];
    // Guard the scene is meaningful: the box must actually penetrate the slope
    // with a multi-point manifold, so parity is not a trivial None match.
    let golden = cpu_obb_heightfield_manifold(&boxes, &fields, &pairs);
    assert!(
        golden[0].as_ref().is_some_and(|m| m.count >= 2),
        "incline scene must yield a multi-point contact, got {golden:?}"
    );
    run_parity(&ctx, &gpu, &boxes, &fields, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_heightfield_manifold_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-heightfield manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHeightfieldManifoldNarrowphase::new(&ctx);

    // One hitting pair and one missing pair keep their input slots.
    let fields = [flat_field(3, 3)];
    let boxes = [
        axis_box(glam::Vec3::new(1.0, 0.4, 1.0), glam::Vec3::splat(0.5)),
        axis_box(glam::Vec3::new(50.0, 0.4, 50.0), glam::Vec3::splat(0.5)),
    ];
    let pairs = [
        HeightfieldObbPair::new(0, 0),
        HeightfieldObbPair::new(1, 0),
    ];
    run_parity(&ctx, &gpu, &boxes, &fields, &pairs);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_obb_heightfield_manifold_empty_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU box-heightfield manifold parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuObbHeightfieldManifoldNarrowphase::new(&ctx);

    // An empty pair batch must return an empty vector without touching the
    // device, matching the twin.
    let fields = [flat_field(3, 3)];
    let boxes = [axis_box(
        glam::Vec3::new(1.0, 0.4, 1.0),
        glam::Vec3::splat(0.5),
    )];
    let pairs: [HeightfieldObbPair; 0] = [];
    run_parity(&ctx, &gpu, &boxes, &fields, &pairs);
}
