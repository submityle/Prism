//! Real-device parity for the cluster vis-buffer software rasterizer twin:
//! [`GpuRasterClusterVis`](prism_volumetric_gpu::raster_cluster_vis::GpuRasterClusterVis)
//! must reproduce the `CPU` golden
//! [`rasterize_cluster`](prism_render_architecture::virtual_geometry::software_raster::rasterize_cluster)
//! across single- and multi-triangle clusters, shared top-left edges, cull
//! flags, degenerate and off-screen triangles, out-of-range index skips, and a
//! randomized cluster sweep compared pixel-for-pixel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`rasterize_cluster`] is a `pub` entry point, so the oracle calls
//! it directly: it fills a real [`VisBuffer`](prism_render_architecture::virtual_geometry::software_raster::VisBuffer)
//! (which uses a host `u64` per pixel) and the expected result is read back
//! pixel by pixel, splitting each packed `u64` into the same `[hi, lo]` pair the
//! twin returns (`hi = word >> 32`, `lo = word & 0xffff_ffff`). A
//! `GPU == golden` pass is therefore established directly, with no in-host
//! re-derivation of the raster logic.
//!
//! # Parity criterion
//!
//! The payload `lo` is a discrete `(cluster_id, triangle_id)` id copied
//! verbatim, so it is asserted with an exact `==` on every covered pixel. The
//! depth key `hi`, however, is the `bitcast<u32>` of a *continuous* barycentric
//! depth sum (`depth = b0*v0.depth + b1*v1.depth + b2*v2.depth`,
//! `b = e * inv_area`, `inv_area = 1.0 / area`): the `Metal` `f32` divide /
//! multiply-add chain can land a unit in the last place away from the scalar
//! `CPU` reference, which shows up as a one-off bit pattern in `hi`. Comparing
//! that key with an exact `==` would be the wrong oracle for a continuous
//! quantity, so each `hi` is decoded back to a depth with `f32::from_bits` and
//! compared within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The twin still
//! materializes every product in the reference's evaluation order to keep that
//! slack to a single unit in the last place; the tolerance is the
//! correctness-preserving backstop. A cleared pixel (`hi == 0 && lo == 0` on
//! both sides) is matched directly.
//!
//! # Conditioning
//!
//! Fixtures use integer vertex coordinates and depths strictly inside `(0, 1)`,
//! keeping every edge value exact and every pixel-center coverage decision
//! unambiguous. The randomized sweep gives every triangle a single flat,
//! well-separated depth, so overlapping triangles never tie within a unit in
//! the last place and the nearest-depth winner (hence its `lo` payload) is
//! deterministic; it also rejects any triangle whose signed area is within a
//! unit of zero, staying clear of the degenerate / swap tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。

use prism_render_architecture::virtual_geometry::software_raster::{
    rasterize_cluster, ScreenVertex, VisBuffer,
};
use prism_volumetric_gpu::raster_cluster_vis::{
    GpuRasterClusterVis, RasterClusterVisQuery, RasterClusterVisResult,
};
use prism_volumetric_gpu::GpuContext;

/// Builds the expected result by running the golden cluster rasterizer into a
/// real [`VisBuffer`] and splitting each packed `u64` into the twin's
/// `[hi, lo]`.
fn oracle(q: &RasterClusterVisQuery) -> RasterClusterVisResult {
    let vertices: Vec<ScreenVertex> = q
        .vertices
        .iter()
        .map(|v| ScreenVertex::new([v[0], v[1]], v[2]))
        .collect();
    let mut buffer = VisBuffer::new(q.width, q.height);
    rasterize_cluster(
        &mut buffer,
        &vertices,
        &q.triangles,
        q.cluster_id,
        q.cull_back,
    );

    let pixels: Vec<[u32; 2]> = (0..q.height)
        .flat_map(|y| (0..q.width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let word = buffer.at(x, y);
            // Host u64 split: high 32 bits are the depth key, low 32 the payload.
            [(word >> 32) as u32, (word & 0xffff_ffff) as u32]
        })
        .collect();
    RasterClusterVisResult {
        width: q.width,
        height: q.height,
        pixels,
    }
}

/// Absolute parity bound on a decoded depth. A `Metal` `f32` divide /
/// multiply-add chain may land a unit in the last place from the scalar
/// reference; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a unit in the
/// last place exceeds the absolute floor.
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
fn check(ctx: &GpuContext, gpu: &GpuRasterClusterVis, queries: &[RasterClusterVisQuery]) {
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

/// Signed area of a triangle, mirroring the golden `edge(v0, v1, v2)`, used to
/// reject near-degenerate random triangles before dispatch.
fn signed_area(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Draws an integer coordinate in `[-2, dim + 2]`, so some vertices land off the
/// buffer and exercise the clamped bounding box.
fn rand_coord(state: &mut u64, dim: u32) -> f32 {
    ((lcg(state) % (dim + 5)) as i32 - 2) as f32
}

/// Draws one well-conditioned random cluster query: two to five triangles, each
/// owning three integer vertices, every triangle carrying a single flat,
/// well-separated depth so overlapping winners never tie. Triangles whose
/// signed area is within a unit of zero are resampled.
fn rand_query(state: &mut u64) -> RasterClusterVisQuery {
    let width = 8 + lcg(state) % 9;
    let height = 8 + lcg(state) % 9;
    let tri_count = 2 + lcg(state) % 4;
    let cluster_id = lcg(state) % 0x0010_0000;
    let cull_back = (lcg(state) & 1u32) == 0u32;

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity((tri_count as usize) * 3);
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(tri_count as usize);
    for t in 0..tri_count {
        // Flat, well-separated depth per triangle keeps the nearest-depth
        // winner (and its payload) unambiguous across overlaps.
        let depth = (t + 1) as f32 / (tri_count + 1) as f32;
        let tri = loop {
            let a = [rand_coord(state, width), rand_coord(state, height), depth];
            let b = [rand_coord(state, width), rand_coord(state, height), depth];
            let c = [rand_coord(state, width), rand_coord(state, height), depth];
            if signed_area(&a, &b, &c).abs() >= 1.0 {
                break [a, b, c];
            }
        };
        let base = t * 3;
        vertices.extend_from_slice(&tri);
        triangles.push([base, base + 1, base + 2]);
    }

    RasterClusterVisQuery {
        vertices,
        triangles,
        cluster_id,
        cull_back,
        width,
        height,
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // No queries: the host short-circuits and never dispatches.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch must return no results");
}

#[test]
fn single_triangle_cluster_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // One front-facing triangle, one cluster: the baseline single-triangle path.
    let q = RasterClusterVisQuery {
        vertices: vec![[2.0, 2.0, 0.75], [14.0, 4.0, 0.5], [6.0, 14.0, 0.25]],
        triangles: vec![[0, 1, 2]],
        cluster_id: 0x0001_2345,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn overlapping_triangles_keep_nearest_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // Three overlapping triangles with distinct, well-separated flat depths: the
    // nearest (largest reversed-Z) must win every shared pixel regardless of the
    // compositing order.
    let q = RasterClusterVisQuery {
        vertices: vec![
            [1.0, 1.0, 0.25],
            [15.0, 1.0, 0.25],
            [1.0, 15.0, 0.25],
            [15.0, 15.0, 0.5],
            [1.0, 15.0, 0.5],
            [15.0, 1.0, 0.5],
            [2.0, 2.0, 0.75],
            [14.0, 8.0, 0.75],
            [4.0, 14.0, 0.75],
        ],
        triangles: vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        cluster_id: 0x0000_00AB,
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
    let gpu = GpuRasterClusterVis::new(&ctx);
    // A quad split along its main diagonal into two triangles of one cluster
    // that share that edge at equal depth. The golden's top-left rule assigns
    // every shared pixel to exactly one triangle id, and the twin must agree.
    let q = RasterClusterVisQuery {
        vertices: vec![
            [0.0, 0.0, 0.5],
            [8.0, 0.0, 0.5],
            [8.0, 8.0, 0.5],
            [0.0, 8.0, 0.5],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
        cluster_id: 0x0000_0007,
        cull_back: false,
        width: 8,
        height: 8,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn cull_back_drops_reversed_winding() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // The same cluster rasterized with cull off (negative-area triangle is
    // reoriented and drawn) and with cull on (it is dropped entirely).
    let vertices = vec![[2.0, 2.0, 0.75], [6.0, 14.0, 0.25], [14.0, 4.0, 0.5]];
    let triangles = vec![[0, 1, 2]];
    let without_cull = RasterClusterVisQuery {
        vertices: vertices.clone(),
        triangles: triangles.clone(),
        cluster_id: 0x0000_0042,
        cull_back: false,
        width: 16,
        height: 16,
    };
    let with_cull = RasterClusterVisQuery {
        vertices,
        triangles,
        cluster_id: 0x0000_0042,
        cull_back: true,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[without_cull, with_cull]);
}

#[test]
fn degenerate_triangle_covers_nothing() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // A collinear (zero-area) triangle alongside a real one: only the real
    // triangle covers pixels, the degenerate one is inert.
    let q = RasterClusterVisQuery {
        vertices: vec![
            [2.0, 2.0, 0.5],
            [6.0, 6.0, 0.5],
            [10.0, 10.0, 0.5],
            [2.0, 10.0, 0.6],
            [12.0, 4.0, 0.6],
            [12.0, 12.0, 0.6],
        ],
        triangles: vec![[0, 1, 2], [3, 4, 5]],
        cluster_id: 0x0000_0099,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn out_of_range_index_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // The second triple indexes past the vertex list and must be skipped, while
    // the first still rasterizes normally.
    let q = RasterClusterVisQuery {
        vertices: vec![[1.0, 1.0, 0.6], [13.0, 3.0, 0.4], [4.0, 13.0, 0.8]],
        triangles: vec![[0, 1, 2], [0, 1, 9]],
        cluster_id: 0x0000_0021,
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
    let gpu = GpuRasterClusterVis::new(&ctx);
    // Vertices spill past the buffer on every side; the clamped scan must never
    // index out of range and must match the golden interior.
    let q = RasterClusterVisQuery {
        vertices: vec![[-4.0, -4.0, 0.6], [20.0, 4.0, 0.4], [4.0, 20.0, 0.8]],
        triangles: vec![[0, 1, 2]],
        cluster_id: 0x0000_0303,
        cull_back: false,
        width: 16,
        height: 16,
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_fixture_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    // Several clusters of different sizes dispatched as one batch, pinning the
    // per-query indexing of the fixed-width result grid.
    let queries = vec![
        RasterClusterVisQuery {
            vertices: vec![[1.0, 1.0, 0.9], [10.0, 2.0, 0.5], [3.0, 11.0, 0.3]],
            triangles: vec![[0, 1, 2]],
            cluster_id: 0x0000_0007,
            cull_back: false,
            width: 12,
            height: 12,
        },
        RasterClusterVisQuery {
            vertices: vec![
                [1.0, 1.0, 0.4],
                [3.0, 15.0, 0.4],
                [15.0, 3.0, 0.4],
                [5.0, 5.0, 0.7],
                [14.0, 9.0, 0.7],
                [8.0, 14.0, 0.7],
            ],
            triangles: vec![[0, 1, 2], [3, 4, 5]],
            cluster_id: 0x0001_2345,
            cull_back: false,
            width: 16,
            height: 16,
        },
        RasterClusterVisQuery {
            vertices: vec![[5.0, 2.0, 0.5], [14.0, 10.0, 0.5], [8.0, 14.0, 0.5]],
            triangles: vec![[0, 1, 2]],
            cluster_id: 0x0000_00CD,
            cull_back: true,
            width: 16,
            height: 16,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden_pixel_for_pixel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterClusterVis::new(&ctx);
    let mut state: u64 = 0x5EED_C0DE_1234_0001;
    let queries: Vec<RasterClusterVisQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
