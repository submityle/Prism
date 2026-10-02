//! Real-device parity for the simple-polygon scanline-fill twin:
//! [`GpuScanlinePolygonFill`](prism_volumetric_gpu::scanline_polygon_fill::GpuScanlinePolygonFill)
//! must reproduce the `CPU` golden
//! [`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill)
//! across the full ordered span list and the summed interior pixel count:
//! [`scanline_fill`](prism_render_architecture::particle::scanline_polygon_fill::scanline_fill)
//! yields one
//! [`Span`](prism_render_architecture::particle::scanline_polygon_fill::Span)
//! per even-odd crossing pair, and
//! [`filled_pixel_count`](prism_render_architecture::particle::scanline_polygon_fill::filled_pixel_count)
//! sums every span's width.
//!
//! The fixtures mirror the shapes the golden unit tests call out: a `10x10`
//! rectangle, a right triangle, a `U`-shaped concave ring (two disjoint spans
//! per row in its arms), a diamond, a plus / cross, a rightward chevron whose
//! tip is a monotone pass-through vertex, a centred rectangle, a rectangle with
//! a collinear edge vertex, a translated rectangle, a multi-polygon batch filled
//! in one dispatch, and the degenerate rings (empty, single vertex, two
//! vertices, a collinear-horizontal line and a collinear-vertical line) that all
//! enclose nothing. Every vertex is an integer and every ring edge has a dyadic
//! slope (`0`, `+/-1`, `+/-2`, `+/-4`), so each per-edge crossing lands on a
//! `.0` or `.5` value well clear of a half-integer pixel boundary; a legal fused
//! multiply-add perturbation therefore cannot flip a floored integer, and the
//! fixtures need no `bevy_math` and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned answer is an integer: the span count is a `u32` tally, each
//! [`GpuSpan`](prism_volumetric_gpu::scanline_polygon_fill::GpuSpan) `y` /
//! `x_start` / `x_end` is a pixel-snapped `i32`, and the pixel count is a `u32`
//! sum of integer widths. `CPU` and `GPU` must therefore agree exactly: the
//! comparison is an exact `==` on the span count, on every span's three
//! coordinates in order, and on the total pixel count. No continuous quantity is
//! ever compared, so no tolerance is used.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::scanline_polygon_fill`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::scanline_polygon_fill::{
    filled_pixel_count, scanline_fill, Vec2,
};
use prism_volumetric_gpu::scanline_polygon_fill::GpuScanlinePolygonFill;
use prism_volumetric_gpu::GpuContext;

/// Converts an open ring of `[x, y]` pairs into the golden
/// [`Vec2`](prism_render_architecture::particle::scanline_polygon_fill::Vec2)
/// slice the reference consumes.
fn to_vec2(points: &[[f32; 2]]) -> Vec<Vec2> {
    points.iter().map(|&p| Vec2::new(p[0], p[1])).collect()
}

/// Asserts the twinned span list and pixel count match the `CPU` golden exactly
/// for one ring, dispatched on its own.
fn assert_parity(gpu: &GpuScanlinePolygonFill, ctx: &GpuContext, points: &[[f32; 2]]) {
    let got = gpu.evaluate(ctx, &[points]);
    assert_eq!(got.len(), 1, "one result per polygon");
    let g = &got[0];

    let ring = to_vec2(points);
    let cpu_spans = scanline_fill(&ring);
    let cpu_pixels = filled_pixel_count(&ring);

    assert_eq!(
        g.span_count as usize,
        cpu_spans.len(),
        "span count mismatch for {points:?}",
    );
    assert_eq!(
        g.spans.len(),
        cpu_spans.len(),
        "returned span lane count mismatch for {points:?}",
    );
    for (i, (gpu_span, cpu_span)) in g.spans.iter().zip(cpu_spans.iter()).enumerate() {
        assert_eq!(
            gpu_span.y, cpu_span.y,
            "span {i} y mismatch for {points:?}: gpu {} vs cpu {}",
            gpu_span.y, cpu_span.y,
        );
        assert_eq!(
            gpu_span.x_start, cpu_span.x_start,
            "span {i} x_start mismatch for {points:?}: gpu {} vs cpu {}",
            gpu_span.x_start, cpu_span.x_start,
        );
        assert_eq!(
            gpu_span.x_end, cpu_span.x_end,
            "span {i} x_end mismatch for {points:?}: gpu {} vs cpu {}",
            gpu_span.x_end, cpu_span.x_end,
        );
    }
    assert_eq!(
        g.filled_pixel_count as usize, cpu_pixels,
        "filled pixel count mismatch for {points:?}: gpu {} vs cpu {cpu_pixels}",
        g.filled_pixel_count,
    );
}

/// A `10x10` axis-aligned rectangle: ten full-width rows.
const RECT_10: [[f32; 2]; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
/// A right triangle, slope `-1` hypotenuse.
const TRI: [[f32; 2]; 3] = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
/// A `U`-shaped concave ring: two disjoint spans per row in its arms.
const U_SHAPE: [[f32; 2]; 8] = [
    [0.0, 0.0],
    [4.0, 0.0],
    [4.0, 4.0],
    [3.0, 4.0],
    [3.0, 1.0],
    [1.0, 1.0],
    [1.0, 4.0],
    [0.0, 4.0],
];
/// A diamond centred on the origin, all edges slope `+/-1`.
const DIAMOND: [[f32; 2]; 4] = [[0.0, -4.0], [4.0, 0.0], [0.0, 4.0], [-4.0, 0.0]];

#[test]
fn convex_rings_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanlinePolygonFill::new(&ctx);

    // A rectangle whose bottom and top edges carry an extra collinear vertex.
    let split_rect: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [5.0, 10.0],
        [0.0, 10.0],
    ];
    // A rectangle centred on the origin, exercising negative rows and columns.
    let centred_rect: [[f32; 2]; 4] = [[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]];
    // The rectangle translated by (5, 3): spans shift uniformly.
    let moved_rect: [[f32; 2]; 4] = [[5.0, 3.0], [15.0, 3.0], [15.0, 13.0], [5.0, 13.0]];

    assert_parity(&gpu, &ctx, &RECT_10);
    assert_parity(&gpu, &ctx, &TRI);
    assert_parity(&gpu, &ctx, &DIAMOND);
    assert_parity(&gpu, &ctx, &split_rect);
    assert_parity(&gpu, &ctx, &centred_rect);
    assert_parity(&gpu, &ctx, &moved_rect);
}

#[test]
fn concave_rings_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanlinePolygonFill::new(&ctx);

    // A plus / cross: four crossings in the wide middle band, one in the arms.
    let plus: [[f32; 2]; 12] = [
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [3.0, 2.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [1.0, 3.0],
        [1.0, 2.0],
        [0.0, 2.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    // A rightward chevron: the tip (4, 1) is a monotone pass-through vertex.
    let chevron: [[f32; 2]; 4] = [[0.0, 0.0], [4.0, 1.0], [0.0, 2.0], [1.0, 1.0]];

    assert_parity(&gpu, &ctx, &U_SHAPE);
    assert_parity(&gpu, &ctx, &plus);
    assert_parity(&gpu, &ctx, &chevron);
}

#[test]
fn degenerate_rings_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanlinePolygonFill::new(&ctx);

    // Fewer than three vertices: no edges, no spans.
    let empty: [[f32; 2]; 0] = [];
    let single: [[f32; 2]; 1] = [[1.0, 1.0]];
    let segment: [[f32; 2]; 2] = [[0.0, 0.0], [5.0, 5.0]];
    // All vertices share a y: every edge is horizontal and skipped.
    let collinear_horizontal: [[f32; 2]; 3] = [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]];
    // A zero-width ring: crossings coincide, so every run is empty.
    let collinear_vertical: [[f32; 2]; 3] = [[2.0, 0.0], [2.0, 10.0], [2.0, 5.0]];

    assert_parity(&gpu, &ctx, &empty);
    assert_parity(&gpu, &ctx, &single);
    assert_parity(&gpu, &ctx, &segment);
    assert_parity(&gpu, &ctx, &collinear_horizontal);
    assert_parity(&gpu, &ctx, &collinear_vertical);
}

#[test]
fn batch_is_filled_in_one_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanlinePolygonFill::new(&ctx);

    // One dispatch over a mix of convex, concave and degenerate rings; each
    // result must match its own golden span list and pixel count in order.
    let moved_rect: [[f32; 2]; 4] = [[5.0, 3.0], [15.0, 3.0], [15.0, 13.0], [5.0, 13.0]];
    let collinear_vertical: [[f32; 2]; 3] = [[2.0, 0.0], [2.0, 10.0], [2.0, 5.0]];
    let batch: [&[[f32; 2]]; 5] = [&RECT_10, &TRI, &U_SHAPE, &moved_rect, &collinear_vertical];

    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per polygon");
    for (points, g) in batch.iter().zip(got.iter()) {
        let ring = to_vec2(points);
        let cpu_spans = scanline_fill(&ring);
        let cpu_pixels = filled_pixel_count(&ring);
        assert_eq!(
            g.span_count as usize,
            cpu_spans.len(),
            "batch span count mismatch for {points:?}",
        );
        for (i, (gpu_span, cpu_span)) in g.spans.iter().zip(cpu_spans.iter()).enumerate() {
            assert_eq!(
                gpu_span.y, cpu_span.y,
                "batch span {i} y mismatch for {points:?}"
            );
            assert_eq!(
                gpu_span.x_start, cpu_span.x_start,
                "batch span {i} x_start mismatch for {points:?}",
            );
            assert_eq!(
                gpu_span.x_end, cpu_span.x_end,
                "batch span {i} x_end mismatch for {points:?}",
            );
        }
        assert_eq!(
            g.filled_pixel_count as usize, cpu_pixels,
            "batch filled pixel count mismatch for {points:?}",
        );
    }
}

#[test]
fn empty_batch_dispatches_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScanlinePolygonFill::new(&ctx);

    let batch: [&[[f32; 2]]; 0] = [];
    assert!(
        gpu.evaluate(&ctx, &batch).is_empty(),
        "an empty batch issues no dispatch and returns no results",
    );
}
