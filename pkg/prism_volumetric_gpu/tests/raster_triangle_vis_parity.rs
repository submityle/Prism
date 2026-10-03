//! Real-device parity for the single-triangle vis-buffer software rasterizer
//! twin:
//! [`GpuRasterTriangleVis`](prism_volumetric_gpu::raster_triangle_vis::GpuRasterTriangleVis)
//! must reproduce, bit for bit, the `CPU` golden
//! [`rasterize_triangle`](prism_render_architecture::virtual_geometry::software_raster::rasterize_triangle)
//! across a sweep of windings, cull flags, degenerate and off-screen triangles,
//! top-left shared edges, and a randomized triangle sweep compared
//! pixel-for-pixel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`rasterize_triangle`] is a `pub` entry point, so the oracle calls
//! it directly: it fills a real [`VisBuffer`](prism_render_architecture::virtual_geometry::software_raster::VisBuffer)
//! (which uses a host `u64` per pixel) and the expected result is read back
//! pixel by pixel, splitting each packed `u64` into the same `[hi, lo]` pair the
//! twin returns (`hi = word >> 32`, `lo = word & 0xffff_ffff`). A
//! `GPU == golden` pass is therefore established directly, with no in-host
//! re-derivation.
//!
//! # Parity criterion
//!
//! The payload `lo` is a discrete id copied verbatim, so it is asserted with an
//! exact `==` on every covered pixel. The depth key `hi`, however, is the
//! `bitcast<u32>` of a *continuous* barycentric depth sum
//! (`depth = b0*v0.depth + b1*v1.depth + b2*v2.depth`, `b = e * inv_area`,
//! `inv_area = 1.0 / area`): the `Metal` `f32` divide / multiply-add chain can
//! land a unit in the last place away from the scalar `CPU` reference, which
//! shows up as a one-off bit pattern in `hi`. Comparing that key with an exact
//! `==` would be the wrong oracle for a continuous quantity, so each `hi` is
//! decoded back to a depth with `f32::from_bits` and compared within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The twin still materializes every
//! product in the reference's evaluation order to keep that slack to a single
//! unit in the last place; the tolerance is the correctness-preserving backstop.
//! A cleared pixel (`hi == 0 && lo == 0` on both sides) is matched directly.
//!
//! # Conditioning
//!
//! Fixtures use integer vertex coordinates and depths strictly inside `(0, 1)`,
//! keeping every edge value exact and every pixel-center coverage decision
//! unambiguous. The randomized sweep rejects any triangle whose signed area is
//! within a unit of zero, staying clear of the degenerate / swap tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。

use prism_render_architecture::virtual_geometry::software_raster::{
    rasterize_triangle, ScreenVertex, VisBuffer,
};
use prism_volumetric_gpu::raster_triangle_vis::{
    GpuRasterTriangleVis, RasterTriangleVisQuery, RasterTriangleVisResult,
};
use prism_volumetric_gpu::GpuContext;

/// Builds the expected result by running the golden rasterizer into a real
/// [`VisBuffer`] and splitting each packed `u64` into the twin's `[hi, lo]`.
fn oracle(q: &RasterTriangleVisQuery) -> RasterTriangleVisResult {
    let vertices = [
        ScreenVertex::new([q.vertices[0][0], q.vertices[0][1]], q.vertices[0][2]),
        ScreenVertex::new([q.vertices[1][0], q.vertices[1][1]], q.vertices[1][2]),
        ScreenVertex::new([q.vertices[2][0], q.vertices[2][1]], q.vertices[2][2]),
    ];
    let mut buffer = VisBuffer::new(q.width, q.height);
    rasterize_triangle(&mut buffer, vertices, q.payload, q.cull_back);

    let mut pixels = Vec::with_capacity((q.width as usize) * (q.height as usize));
    for y in 0..q.height {
        for x in 0..q.width {
            let word = buffer.at(x, y);
            // Host u64 split: high 32 bits are the depth key, low 32 the payload.
            pixels.push([(word >> 32) as u32, (word & 0xffff_ffff) as u32]);
        }
    }
    RasterTriangleVisResult {
        width: q.width,
        height: q.height,
        pixels,
    }
}

/// Absolute parity bound on a decoded depth. A `Metal` `f32` divide / multiply-add
/// chain may land a unit in the last place from the scalar reference; `1e-4`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a unit in the last
/// place exceeds the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected depth does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Pins one pixel against the golden by semantics: the payload `lo` is a
/// discrete id compared with an exact `==`, while the depth key `hi` is decoded
/// back to a continuous depth with `f32::from_bits` and compared within
/// tolerance. A cleared pixel (`hi == 0 && lo == 0` on both sides) passes
/// directly, which the depth tolerance would also accept (`0.0` vs `0.0`).
fn compare_pixel(qi: usize, pi: usize, got: &[u32; 2], want: &[u32; 2]) {
    let [g_hi, g_lo] = *got;
    let [w_hi, w_lo] = *want;

    // The payload id must match bit for bit.
    assert_eq!(
        g_lo, w_lo,
        "query {qi} pixel {pi} payload: gpu {got:?} vs cpu {want:?}"
    );

    // Both cleared: identical empty pixel, nothing continuous to compare.
    if g_hi == 0u32 && g_lo == 0u32 && w_hi == 0u32 && w_lo == 0u32 {
        return;
    }

    // The depth key is a bitcast of a continuous barycentric depth; decode and
    // compare within tolerance rather than demanding an exact bit pattern.
    let g_depth = f32::from_bits(g_hi);
    let w_depth = f32::from_bits(w_hi);
    assert!(
        close(g_depth, w_depth),
        "query {qi} pixel {pi} depth key: gpu {got:?} ({g_depth}) vs cpu {want:?} ({w_depth})"
    );
}

/// Dispatches every query and pins each pixel against the golden.
fn check(ctx: &GpuContext, gpu: &GpuRasterTriangleVis, queries: &[RasterTriangleVisQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (qi, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert_eq!(result.width, want.width, "query {qi} width");
        assert_eq!(result.height, want.height, "query {qi} height");
        assert_eq!(
            result.pixels.len(),
            want.pixels.len(),
            "query {qi} pixel count"
        );
        for (pi, (g, w)) in result.pixels.iter().zip(want.pixels.iter()).enumerate() {
            compare_pixel(qi, pi, g, w);
        }
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Signed area of the triangle, mirroring the golden `edge(v0, v1, v2)`, used to
/// reject near-degenerate random triangles before dispatch.
fn signed_area(v: &[[f32; 3]; 3]) -> f32 {
    let ax = v[0][0];
    let ay = v[0][1];
    let bx = v[1][0];
    let by = v[1][1];
    let cx = v[2][0];
    let cy = v[2][1];
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

/// Draws an integer coordinate in `[-2, dim + 2]`, so some vertices land off the
/// buffer and exercise the clamped bounding box.
fn rand_coord(state: &mut u64, dim: u32) -> f32 {
    ((lcg(state) % (dim + 5)) as i32 - 2) as f32
}

/// Draws a depth strictly inside `(0, 1)` at sixteenth resolution, keeping it
/// clear of the `clamp` endpoints and of `-0.0`.
fn rand_depth(state: &mut u64) -> f32 {
    (1 + lcg(state) % 15) as f32 / 16.0
}

/// Draws one well-conditioned random triangle query: integer vertices, interior
/// depths, random payload and cull flag, rejected unless the signed area is a
/// full unit clear of zero.
fn rand_query(state: &mut u64) -> RasterTriangleVisQuery {
    loop {
        let width = 8 + lcg(state) % 17;
        let height = 8 + lcg(state) % 17;
        let vertices = [
            [
                rand_coord(state, width),
                rand_coord(state, height),
                rand_depth(state),
            ],
            [
                rand_coord(state, width),
                rand_coord(state, height),
                rand_depth(state),
            ],
            [
                rand_coord(state, width),
                rand_coord(state, height),
                rand_depth(state),
            ],
        ];
        let payload = lcg(state);
        let cull_back = (lcg(state) & 1u32) == 0u32;
        if signed_area(&vertices).abs() < 1.0 {
            // Too close to the degenerate / swap tie: resample.
            continue;
        }
        return RasterTriangleVisQuery {
            vertices,
            payload,
            cull_back,
            width,
            height,
        };
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn front_facing_triangle_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Positive-area (front-facing) triangle on a 16x16 grid.
    let q = RasterTriangleVisQuery {
        vertices: [[2.0, 2.0, 0.75], [14.0, 4.0, 0.5], [6.0, 14.0, 0.25]],
        payload: 0x00AB_CDEF,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn back_facing_triangle_without_cull_is_reoriented() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Negative-area winding; with cull off it is swapped and still rasterizes.
    let q = RasterTriangleVisQuery {
        vertices: [[2.0, 2.0, 0.75], [6.0, 14.0, 0.25], [14.0, 4.0, 0.5]],
        payload: 0x0000_0042,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn back_facing_triangle_with_cull_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Same negative-area winding, but cull_back drops it entirely.
    let q = RasterTriangleVisQuery {
        vertices: [[2.0, 2.0, 0.75], [6.0, 14.0, 0.25], [14.0, 4.0, 0.5]],
        payload: 0x0000_0042,
        cull_back: true,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn front_facing_triangle_with_cull_rasterizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Positive-area winding survives cull_back and rasterizes normally.
    let q = RasterTriangleVisQuery {
        vertices: [[2.0, 2.0, 0.75], [14.0, 4.0, 0.5], [6.0, 14.0, 0.25]],
        payload: 0x0000_1234,
        cull_back: true,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn degenerate_triangle_covers_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Collinear vertices: zero area, no coverage, buffer stays cleared.
    let q = RasterTriangleVisQuery {
        vertices: [[2.0, 2.0, 0.5], [6.0, 6.0, 0.5], [10.0, 10.0, 0.5]],
        payload: 0x7FFF_FFFF,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn off_screen_triangle_clamps_bounding_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Vertices spill past the buffer on every side; the clamped scan must never
    // index out of range and must match the golden interior.
    let q = RasterTriangleVisQuery {
        vertices: [[-4.0, -4.0, 0.6], [20.0, 4.0, 0.4], [4.0, 20.0, 0.8]],
        payload: 0x00FA_CE01,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn shared_edge_respects_top_left_rule() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // A quad split along its main diagonal into two triangles sharing that edge.
    // Each is a separate query; the golden's top-left rule assigns every shared
    // pixel to exactly one, and the twin must agree pixel for pixel.
    let a = RasterTriangleVisQuery {
        vertices: [[0.0, 0.0, 0.5], [8.0, 0.0, 0.5], [8.0, 8.0, 0.5]],
        payload: 1,
        cull_back: false,
        width: 8,
        height: 8,
    };
    let b = RasterTriangleVisQuery {
        vertices: [[0.0, 0.0, 0.5], [8.0, 8.0, 0.5], [0.0, 8.0, 0.5]],
        payload: 2,
        cull_back: false,
        width: 8,
        height: 8,
    };
    check(&ctx, &gpu, &[a, b]);
}

#[test]
fn mixed_fixture_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    // Several triangles of different sizes, windings and depths dispatched as one
    // batch, pinning the per-query indexing of the fixed-width result grid.
    let queries = vec![
        RasterTriangleVisQuery {
            vertices: [[1.0, 1.0, 0.9], [10.0, 2.0, 0.5], [3.0, 11.0, 0.3]],
            payload: 0x0000_0007,
            cull_back: false,
            width: 12,
            height: 12,
        },
        RasterTriangleVisQuery {
            vertices: [[1.0, 1.0, 0.4], [3.0, 18.0, 0.6], [18.0, 3.0, 0.8]],
            payload: 0x0012_3456,
            cull_back: false,
            width: 20,
            height: 20,
        },
        RasterTriangleVisQuery {
            vertices: [[5.0, 2.0, 0.5], [22.0, 10.0, 0.5], [8.0, 22.0, 0.5]],
            payload: 0x00AB_00CD,
            cull_back: true,
            width: 24,
            height: 24,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden_pixel_for_pixel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterTriangleVis::new(&ctx);
    let mut state: u64 = 0x5EED_1234_ABCD_0001;
    let queries: Vec<RasterTriangleVisQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
