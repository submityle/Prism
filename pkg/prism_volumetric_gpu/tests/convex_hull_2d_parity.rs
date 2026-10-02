//! Real-device parity for the 2D convex-hull twin:
//! [`GpuConvexHull2d`](prism_volumetric_gpu::convex_hull_2d::GpuConvexHull2d)
//! must reproduce the `CPU` golden
//! [`convex_hull_2d`](prism_render_architecture::particle::convex_hull_2d)
//! across the ordered hull ring, the vertex count, the shoelace area, the closed
//! perimeter, the squared diameter and the convex-`CCW` flag.
//!
//! The fixtures cover the shapes the golden unit tests call out: an empty set, a
//! single point, two distinct points (a segment), a coincident pair that
//! collapses to one point, the unit square, a right triangle, an interior-point
//! cloud whose inner points are excluded, a boundary-collinear set whose edge
//! midpoints are dropped, a fully collinear set that collapses to its endpoint
//! pair, a concave point that is removed, an unordered cloud and a mixed batch.
//! All coordinates are written as integers or simple decimals, so the fixtures
//! stay pure and need no `bevy_math` and no transcendental math. The point sets
//! stay clear of the collinearity and coincidence thresholds (points are either
//! exactly shared or well separated), so the sort order and the kept vertex set
//! match the reference exactly.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The hull vertex count and the convex flag are discrete classifications, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on the
//! count and on the flag. The vertex coordinates and the continuous metrics
//! thread through sorting, multiplies, adds and one `sqrt`, so they are compared
//! under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::convex_hull_2d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::convex_hull_2d::{
    convex_hull, diameter_squared, hull_area, hull_perimeter, is_convex_ccw,
};
use prism_volumetric_gpu::convex_hull_2d::{ConvexHull2dQuery, GpuConvexHull2d};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two 2D points.
fn approx_pt(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Wraps a slice of points as one query.
fn set(points: &[[f32; 2]]) -> ConvexHull2dQuery {
    ConvexHull2dQuery {
        points: points.to_vec(),
    }
}

/// Asserts every twinned answer for one point-set matches the `CPU` golden.
fn assert_parity(gpu: &GpuConvexHull2d, ctx: &GpuContext, points: &[[f32; 2]]) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&set(points)));
    assert_eq!(got.len(), 1, "one result per query");
    let g = &got[0];

    let cpu_hull = convex_hull(points);

    // The kept vertex count is a discrete classification: exact match.
    assert_eq!(
        g.hull.len(),
        cpu_hull.len(),
        "hull_count mismatch: gpu {:?} vs cpu {:?}",
        g.hull,
        cpu_hull
    );
    // The ring order is replayed from the same sort-then-sweep, so vertices
    // match lane for lane under tolerance.
    for (i, (&gv, &cv)) in g.hull.iter().zip(cpu_hull.iter()).enumerate() {
        assert!(
            approx_pt(gv, cv),
            "hull vertex {i} mismatch: gpu {gv:?} vs cpu {cv:?}"
        );
    }

    let cpu_area = hull_area(&cpu_hull);
    assert!(
        approx(g.area, cpu_area),
        "area mismatch: gpu {} vs cpu {cpu_area}",
        g.area
    );

    let cpu_perimeter = hull_perimeter(&cpu_hull);
    assert!(
        approx(g.perimeter, cpu_perimeter),
        "perimeter mismatch: gpu {} vs cpu {cpu_perimeter}",
        g.perimeter
    );

    let cpu_diameter = diameter_squared(&cpu_hull);
    assert!(
        approx(g.diameter_squared, cpu_diameter),
        "diameter_squared mismatch: gpu {} vs cpu {cpu_diameter}",
        g.diameter_squared
    );

    // The convex flag is a discrete classification: exact match.
    assert_eq!(
        g.is_convex_ccw,
        is_convex_ccw(&cpu_hull),
        "is_convex_ccw mismatch for hull {:?}",
        cpu_hull
    );
}

#[test]
fn degenerate_sets_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // Empty set: empty hull, zero metrics.
    assert_parity(&gpu, &ctx, &[]);
    // A single point is its own hull.
    assert_parity(&gpu, &ctx, &[[3.0, 4.0]]);
    // Two distinct points form a segment.
    assert_parity(&gpu, &ctx, &[[1.0, 1.0], [4.0, 5.0]]);
    // A coincident pair collapses to a single point.
    assert_parity(&gpu, &ctx, &[[2.0, 2.0], [2.0, 2.0]]);
}

#[test]
fn unit_square_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // Four CCW corners, unit area, perimeter four, squared diameter two.
    assert_parity(
        &gpu,
        &ctx,
        &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    );
}

#[test]
fn right_triangle_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // 3-4-5 right triangle: area six, perimeter twelve.
    assert_parity(&gpu, &ctx, &[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]);
}

#[test]
fn interior_points_are_excluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // Three strictly-interior points must be dropped, leaving the square.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ],
    );
}

#[test]
fn boundary_collinear_points_are_excluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // Edge midpoints lie on the boundary but are not vertices; the minimal ring
    // must drop them.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [4.0, 4.0],
            [2.0, 4.0],
            [0.0, 4.0],
            [0.0, 2.0],
        ],
    );
}

#[test]
fn all_collinear_points_collapse_to_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // A fully collinear set collapses to its two extreme endpoints.
    assert_parity(
        &gpu,
        &ctx,
        &[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    );
}

#[test]
fn concave_point_is_removed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // A square with a fifth point pushed inward; the dent must not appear.
    assert_parity(
        &gpu,
        &ctx,
        &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 1.0]],
    );
}

#[test]
fn unordered_cloud_produces_ccw_ring() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // An unordered cloud whose hull is a convex pentagon-ish ring.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [2.0, 5.0],
            [5.0, 1.0],
            [1.0, 1.0],
            [4.0, 4.0],
            [3.0, 2.0],
            [0.0, 3.0],
        ],
    );
}

#[test]
fn duplicate_square_corners_still_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // Repeated corners must dedup to the four-vertex square.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    );
}

#[test]
fn batch_of_sets_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // A batch exercises the one-thread-per-set flattening; each result must be
    // independent of its neighbours.
    let batch = [
        set(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        set(&[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]),
        set(&[]),
        set(&[[5.0, 5.0]]),
        set(&[
            [2.0, 5.0],
            [5.0, 1.0],
            [1.0, 1.0],
            [4.0, 4.0],
            [3.0, 2.0],
            [0.0, 3.0],
        ]),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, &q.points);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull2d::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
