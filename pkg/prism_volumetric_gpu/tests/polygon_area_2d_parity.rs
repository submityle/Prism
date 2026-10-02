//! Real-device parity for the 2D polygon-metrics twin:
//! [`GpuPolygonArea2d`](prism_volumetric_gpu::polygon_area_2d::GpuPolygonArea2d)
//! must reproduce the `CPU` golden
//! [`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d)
//! across the signed and absolute shoelace area, the closed perimeter, the
//! area-weighted centroid, the `CCW` winding flag, the convexity flag, and the
//! axis-aligned bounding box.
//!
//! The fixtures cover the shapes the golden unit tests call out: a `CCW` and a
//! `CW` unit square, a right triangle, a `3-4-5` triangle, a scaled square, a
//! degenerate collinear ring, a two-vertex ring (perimeter counts both
//! directions), a single vertex, an empty ring, a concave arrowhead dart, a
//! convex ring with a collinear edge vertex, and an off-axis ring exercising the
//! bounding box. All coordinates are written as integers or simple decimals, so
//! the fixtures stay pure and need no `bevy_math` and no transcendental math,
//! and every signed area is exactly zero or well clear of [`CMP_EPS`] so the
//! `CPU` and `GPU` take the same classification branch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The winding and convexity flags are discrete classifications, so `CPU` and
//! `GPU` must agree exactly: the comparison is an exact `==` on each `u32` flag.
//! The area, perimeter, centroid and bounding box thread through multiplies,
//! adds, a `sqrt` and one guarded division, so they are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::polygon_area_2d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::polygon_area_2d::{
    area, bounding_box, centroid, is_convex, perimeter, signed_area, winding_is_ccw,
};
use prism_volumetric_gpu::polygon_area_2d::GpuPolygonArea2d;
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
fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Asserts every twinned metric for every ring matches the `CPU` golden, in one
/// batched dispatch.
fn assert_parity(gpu: &GpuPolygonArea2d, ctx: &GpuContext, rings: &[&[[f32; 2]]]) {
    let got = gpu.evaluate(ctx, rings);
    assert_eq!(got.len(), rings.len(), "one result per polygon");

    for (ring, g) in rings.iter().zip(got.iter()) {
        let cpu_signed = signed_area(ring);
        assert!(
            approx(g.signed_area, cpu_signed),
            "signed_area mismatch: gpu {} vs cpu {cpu_signed} for {ring:?}",
            g.signed_area
        );

        let cpu_area = area(ring);
        assert!(
            approx(g.area, cpu_area),
            "area mismatch: gpu {} vs cpu {cpu_area} for {ring:?}",
            g.area
        );

        let cpu_perim = perimeter(ring);
        assert!(
            approx(g.perimeter, cpu_perim),
            "perimeter mismatch: gpu {} vs cpu {cpu_perim} for {ring:?}",
            g.perimeter
        );

        let cpu_centroid = centroid(ring);
        assert!(
            approx2(g.centroid, cpu_centroid),
            "centroid mismatch: gpu {:?} vs cpu {cpu_centroid:?} for {ring:?}",
            g.centroid
        );

        let (cpu_min, cpu_max) = bounding_box(ring);
        assert!(
            approx2(g.bbox_min, cpu_min) && approx2(g.bbox_max, cpu_max),
            "bbox mismatch: gpu ({:?},{:?}) vs cpu ({cpu_min:?},{cpu_max:?}) for {ring:?}",
            g.bbox_min,
            g.bbox_max
        );

        let cpu_ccw = u32::from(winding_is_ccw(ring));
        assert_eq!(
            g.winding_ccw, cpu_ccw,
            "winding mismatch: gpu {} vs cpu {cpu_ccw} for {ring:?}",
            g.winding_ccw
        );

        let cpu_convex = u32::from(is_convex(ring));
        assert_eq!(
            g.is_convex, cpu_convex,
            "convexity mismatch: gpu {} vs cpu {cpu_convex} for {ring:?}",
            g.is_convex
        );
    }
}

/// A `CCW` unit square.
const CCW_SQUARE: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// A `CW` unit square (reversed winding).
const CW_SQUARE: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];

#[test]
fn squares_and_triangles_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolygonArea2d::new(&ctx);

    let tri: [[f32; 2]; 3] = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
    let tri345: [[f32; 2]; 3] = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]];
    let scaled: [[f32; 2]; 4] = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]];

    let rings: [&[[f32; 2]]; 5] = [&CCW_SQUARE, &CW_SQUARE, &tri, &tri345, &scaled];
    assert_parity(&gpu, &ctx, &rings);
}

#[test]
fn convex_and_concave_rings_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolygonArea2d::new(&ctx);

    // A convex pentagon-ish ring with a collinear vertex on the bottom edge.
    let collinear_edge: [[f32; 2]; 5] =
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    // An arrowhead / dart with one reflex vertex (concave).
    let dart: [[f32; 2]; 4] = [[0.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.5, 1.0]];
    // An off-axis ring exercising the bounding box.
    let bbox_ring: [[f32; 2]; 3] = [[-1.0, 2.0], [3.0, -4.0], [0.0, 5.0]];

    let rings: [&[[f32; 2]]; 3] = [&collinear_edge, &dart, &bbox_ring];
    assert_parity(&gpu, &ctx, &rings);
}

#[test]
fn degenerate_rings_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPolygonArea2d::new(&ctx);

    // Collinear 3-vertex ring: zero area, falls back to the vertex mean.
    let line: [[f32; 2]; 3] = [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]];
    // Two vertices: perimeter counts out and back; centroid is the midpoint.
    let segment: [[f32; 2]; 2] = [[0.0, 0.0], [3.0, 4.0]];
    // One vertex: centroid is itself, bounding box is the degenerate point.
    let point: [[f32; 2]; 1] = [[3.0, -7.0]];
    // Empty ring: origin centroid and origin-pair bounding box.
    let empty: [[f32; 2]; 0] = [];

    let rings: [&[[f32; 2]]; 4] = [&line, &segment, &point, &empty];
    assert_parity(&gpu, &ctx, &rings);
}
