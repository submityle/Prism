//! Real-device parity for the conservative triangle-raster twin:
//! [`GpuConservativeRaster`](prism_volumetric_gpu::conservative_raster::GpuConservativeRaster)
//! must reproduce the `CPU` golden
//! [`conservative_raster`](prism_render_architecture::particle::conservative_raster)
//! across the three directed edge values at a pixel center and the conservative
//! coverage classification, for both `CCW` and `CW` input winding and for the
//! degenerate (zero-area) short-circuit.
//!
//! The fixtures cover the shapes the golden unit tests call out and the task
//! requires: a deep-interior pixel (covered with a large margin on every edge),
//! a far-exterior pixel (uncovered with a large margin), a dilation-band pixel
//! whose center is outside the strict triangle yet comfortably inside the
//! dilated one (exercising the `half_pixel` outward push), a clockwise triangle
//! that must classify identically to its counter-clockwise twin after the
//! winding swap, and a degenerate collinear triangle that covers no pixel. Every
//! probed pixel center is kept well away from each dilated edge zero-line, so the
//! coverage classification is unambiguous and robust to floating-point noise.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `inside` flag is a discrete classification, so `CPU` and `GPU` must agree
//! exactly: the comparison is an exact `==` on the coverage `bool`. The three
//! edge values thread through multiplies and adds, so they are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::conservative_raster`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::conservative_raster::{
    dilate_edges, edges_from_triangle, is_degenerate, pixel_covered,
};
use prism_volumetric_gpu::conservative_raster::{
    ConservativeRasterQuery, ConservativeRasterResult, GpuConservativeRaster,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous edge values.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous edge values.
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

/// Builds a pixel query at integer coordinates `(x, y)`.
fn q(x: i32, y: i32) -> ConservativeRasterQuery {
    ConservativeRasterQuery { x, y }
}

/// Computes the `CPU` golden answer for one pixel query against `triangle`
/// dilated by `half_pixel`: the three dilated edge values at the pixel center
/// and the coverage flag (forced `false` for a degenerate triangle, mirroring
/// the empty coverage of `rasterize_coverage`).
fn cpu_answer(
    triangle: [[f32; 2]; 3],
    half_pixel: f32,
    query: ConservativeRasterQuery,
) -> ConservativeRasterResult {
    let edges = dilate_edges(&edges_from_triangle(triangle), half_pixel);
    let inside = !is_degenerate(triangle) && pixel_covered(&edges, query.x, query.y);
    let cx = query.x as f32 + 0.5;
    let cy = query.y as f32 + 0.5;
    ConservativeRasterResult {
        inside,
        edges: [
            edges[0].eval(cx, cy),
            edges[1].eval(cx, cy),
            edges[2].eval(cx, cy),
        ],
    }
}

/// Dispatches `queries` on the `GPU` and asserts every answer matches the `CPU`
/// golden: an exact `==` on the `inside` flag and a tolerant compare on each
/// edge value.
fn assert_parity(
    gpu: &GpuConservativeRaster,
    ctx: &GpuContext,
    triangle: [[f32; 2]; 3],
    half_pixel: f32,
    queries: &[ConservativeRasterQuery],
) {
    let got = gpu.evaluate(ctx, triangle, half_pixel, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (g, &query) in got.iter().zip(queries) {
        let cpu = cpu_answer(triangle, half_pixel, query);
        assert_eq!(
            g.inside, cpu.inside,
            "inside mismatch at {query:?}: gpu {} vs cpu {}",
            g.inside, cpu.inside
        );
        for (lane, (&gv, &cv)) in g.edges.iter().zip(&cpu.edges).enumerate() {
            assert!(
                approx(gv, cv),
                "edge[{lane}] mismatch at {query:?}: gpu {gv} vs cpu {cv}"
            );
        }
    }
}

/// A counter-clockwise right triangle with integer corners, well clear of the
/// probed pixels used below.
const CCW_TRI: [[f32; 2]; 3] = [[2.0, 2.0], [10.0, 2.0], [2.0, 10.0]];
/// The same triangle wound clockwise (last two corners swapped); the twin must
/// normalize it to the identical coverage.
const CW_TRI: [[f32; 2]; 3] = [[2.0, 2.0], [2.0, 10.0], [10.0, 2.0]];

#[test]
fn interior_pixel_is_covered_with_margin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // Pixel (4, 4) center (4.5, 4.5) sits deep inside the triangle: every
    // dilated edge value is far from its zero-line.
    assert_parity(&gpu, &ctx, CCW_TRI, 0.5, &[q(4, 4)]);
}

#[test]
fn exterior_pixel_is_uncovered_with_margin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // Pixel (20, 20) is far outside the hypotenuse; its diagonal edge value is
    // strongly negative, so the pixel is uncovered with a large margin.
    assert_parity(&gpu, &ctx, CCW_TRI, 0.5, &[q(20, 20)]);
}

#[test]
fn dilation_band_pixel_flips_to_covered() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // With a 2-pixel dilation the left edge pushes out from center x = 2 to
    // x = 0, so pixel (0, 4) center (0.5, 4.5) lies in the dilation band: strictly
    // outside the un-dilated triangle but inside the dilated one, with every
    // dilated edge value well clear of zero.
    assert_parity(&gpu, &ctx, CCW_TRI, 2.0, &[q(0, 4)]);
}

#[test]
fn clockwise_triangle_matches_counter_clockwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // The clockwise input is normalized to CCW by the winding swap, so the same
    // interior and exterior pixels classify identically to the CCW twin.
    assert_parity(&gpu, &ctx, CW_TRI, 0.5, &[q(4, 4), q(20, 20)]);
}

#[test]
fn degenerate_triangle_covers_no_pixel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // Collinear corners give a zero-area triangle: every pixel reports uncovered,
    // matching the empty coverage of the reference short-circuit.
    let line = [[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]];
    assert_parity(&gpu, &ctx, line, 0.5, &[q(1, 1), q(4, 4), q(2, 6)]);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours and match the per-query golden.
    let batch = [q(4, 4), q(3, 3), q(5, 3), q(20, 20), q(0, 4), q(2, 8)];
    assert_parity(&gpu, &ctx, CCW_TRI, 2.0, &batch);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConservativeRaster::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, CCW_TRI, 0.5, &[]).is_empty());
}
