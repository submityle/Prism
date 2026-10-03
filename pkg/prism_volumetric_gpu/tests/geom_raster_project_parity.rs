//! Real-device parity for the software-raster vertex-projection twin:
//! [`GpuGeomRasterProject`](prism_volumetric_gpu::geom_raster_project::GpuGeomRasterProject)
//! must reproduce the numeric core of the `CPU` golden
//! [`software_raster`](prism_render_architecture::virtual_geometry::software_raster)
//! — the column-major clip-space projection
//! [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
//! and the reversed-Z depth key
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
//! — across a normal projection, a perspective-divide fixture, a near-plane
//! cull, in-range and out-of-range depth clamps, a mixed batch and a randomized
//! sweep compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values come straight from the public golden
//! [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
//! and
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth),
//! so a `GPU == golden` pass is direct evidence the ported kernel projects the
//! same way the reference does.
//!
//! # Parity criterion
//!
//! The validity flag and the encoded-depth bit pattern are discrete and
//! asserted bit-exact (`==`); the screen position and depth thread through a
//! matrix-vector product and a guarded perspective divide, so a `GPU` divide
//! may land a few units in the last place from the scalar reference and are
//! asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor
//! `1e-6`).
//!
//! # Conditioning
//!
//! The near-plane cull (`clip.w ~ 0`) is a discontinuity that flips `valid`.
//! Fixtures and the randomized sweep keep every sample's `clip.w` well clear of
//! zero (rejecting any draw with `|clip.w| < 0.25`), so `CPU` and `GPU` cannot
//! straddle it. Both signs of `clip.w` are admitted, exercising the culled and
//! projectable branches.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。

use prism_render_architecture::virtual_geometry::software_raster::{encode_depth, project_vertex};
use prism_volumetric_gpu::geom_raster_project::{
    GeomRasterProjectQuery, GeomRasterProjectResult, GpuGeomRasterProject,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous field. A `GPU` divide may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the golden result for one query straight from the public
/// [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
/// and
/// [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth).
///
/// `project_vertex` returning `None` maps to `valid == 0` with zeroed screen
/// fields; the encoded depth is always taken from the query's independent depth
/// so it stays defined even for a culled vertex.
fn oracle(q: &GeomRasterProjectQuery) -> GeomRasterProjectResult {
    let encoded_depth = encode_depth(q.depth);
    match project_vertex(&q.clip_from_world, q.world_pos, q.viewport) {
        Some(sv) => GeomRasterProjectResult {
            valid: 1,
            screen_pos: sv.pos,
            depth_ndc: sv.depth,
            encoded_depth,
        },
        None => GeomRasterProjectResult {
            valid: 0,
            screen_pos: [0.0, 0.0],
            depth_ndc: 0.0,
            encoded_depth,
        },
    }
}

/// Pins one `GPU` sample against the golden: discrete fields bit-exact, and the
/// continuous projection fields within tolerance when the vertex is
/// projectable.
fn check_sample(idx: usize, got: &GeomRasterProjectResult, want: &GeomRasterProjectResult) {
    assert_eq!(
        got.valid, want.valid,
        "sample {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    assert_eq!(
        got.encoded_depth, want.encoded_depth,
        "sample {idx} encoded_depth: gpu {} vs cpu {}",
        got.encoded_depth, want.encoded_depth
    );
    if want.valid == 1 {
        assert!(
            close(got.screen_pos[0], want.screen_pos[0]),
            "sample {idx} screen_x: gpu {} vs cpu {}",
            got.screen_pos[0],
            want.screen_pos[0]
        );
        assert!(
            close(got.screen_pos[1], want.screen_pos[1]),
            "sample {idx} screen_y: gpu {} vs cpu {}",
            got.screen_pos[1],
            want.screen_pos[1]
        );
        assert!(
            close(got.depth_ndc, want.depth_ndc),
            "sample {idx} depth_ndc: gpu {} vs cpu {}",
            got.depth_ndc,
            want.depth_ndc
        );
    }
}

/// Dispatches `queries` and pins every `GPU` sample against the golden oracle.
fn check(ctx: &GpuContext, gpu: &GpuGeomRasterProject, queries: &[GeomRasterProjectQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query is expected");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, g, &oracle(q));
    }
}

/// A viewport used by the fixtures; a common 1080p render target.
const VIEWPORT: [f32; 2] = [1920.0, 1080.0];

/// An identity-like orthographic-ish matrix with `clip.w = 1` for every vertex:
/// column-major, so the last column's `w` row is `1`.
fn ortho_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A simple perspective-style matrix whose `clip.w` equals the world `z`, so
/// the perspective divide is non-trivial for `z > 0`.
fn perspective_matrix() -> [[f32; 4]; 4] {
    [
        [1.5, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0],
        [0.0, 0.0, 0.25, 0.0],
    ]
}

/// The deterministic fixture battery exercising both branches and the depth
/// clamp.
fn fixture_queries() -> Vec<GeomRasterProjectQuery> {
    vec![
        // Orthographic projection, origin vertex, mid-range depth.
        GeomRasterProjectQuery::new(ortho_matrix(), [0.0, 0.0, 0.0], VIEWPORT, 0.5),
        // Orthographic projection, off-center vertex.
        GeomRasterProjectQuery::new(ortho_matrix(), [0.3, -0.4, 0.2], VIEWPORT, 0.75),
        // Perspective projection with w = z = 2 (projectable).
        GeomRasterProjectQuery::new(perspective_matrix(), [0.4, 0.6, 2.0], VIEWPORT, 0.9),
        // Perspective projection with w = z <= 0 (near-plane culled).
        GeomRasterProjectQuery::new(perspective_matrix(), [0.1, 0.2, -1.0], VIEWPORT, 0.3),
        // Depth below 0 clamps to 0 before encoding.
        GeomRasterProjectQuery::new(ortho_matrix(), [0.1, 0.1, 0.1], VIEWPORT, -0.5),
        // Depth above 1 clamps to 1 before encoding.
        GeomRasterProjectQuery::new(ortho_matrix(), [-0.2, 0.2, 0.3], VIEWPORT, 1.5),
    ]
}

/// A 64-bit linear-congruential generator (`SplitMix`/`PCG`-style multiplier)
/// for host-side fixtures; no transcendental and no float equality.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a value in `[0, span)` with milli-resolution from the generator.
fn draw(state: &mut u64, span: f32) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0 * span
}

/// Draws a signed value in `(-span, span)` with milli-resolution.
fn draw_signed(state: &mut u64, span: f32) -> f32 {
    draw(state, 2.0 * span) - span
}

/// Computes the golden `clip.w` for a candidate matrix and world position so the
/// sweep can reject samples near the cull boundary. Mirrors the column-major
/// product's `w` row exactly.
fn clip_w(m: &[[f32; 4]; 4], world: [f32; 3]) -> f32 {
    m[0][3] * world[0] + m[1][3] * world[1] + m[2][3] * world[2] + m[3][3]
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    // An empty batch never dispatches (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn orthographic_projection_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    let q = GeomRasterProjectQuery::new(ortho_matrix(), [0.3, -0.4, 0.2], VIEWPORT, 0.5);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "orthographic vertex must be projectable");
    check_sample(0, &got[0], &want);
}

#[test]
fn perspective_projection_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    let q = GeomRasterProjectQuery::new(perspective_matrix(), [0.4, 0.6, 2.0], VIEWPORT, 0.9);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "w = z = 2 must be projectable");
    check_sample(0, &got[0], &want);
}

#[test]
fn near_plane_cull_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    let q = GeomRasterProjectQuery::new(perspective_matrix(), [0.1, 0.2, -1.0], VIEWPORT, 0.3);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.valid, 0, "w = z = -1 must be culled");
    check_sample(0, &got[0], &want);
}

#[test]
fn depth_clamp_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    // Below 0 and above 1 both clamp before the bit pattern is taken.
    let low = GeomRasterProjectQuery::new(ortho_matrix(), [0.0, 0.0, 0.0], VIEWPORT, -0.5);
    let high = GeomRasterProjectQuery::new(ortho_matrix(), [0.0, 0.0, 0.0], VIEWPORT, 1.5);
    assert_eq!(oracle(&low).encoded_depth, encode_depth(0.0));
    assert_eq!(oracle(&high).encoded_depth, encode_depth(1.0));
    check(&ctx, &gpu, &[low, high]);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    // Every fixture dispatched together exercises per-thread indexing and the
    // contiguous output slots.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGeomRasterProject::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    let mut accepted = 0u32;
    let mut tries = 0u32;
    // Many well-conditioned random samples (several workgroups' worth) pin the
    // projection across a wide span of matrices and vertices, with rejection
    // sampling keeping every sample's clip.w clear of the cull boundary so the
    // CPU and GPU cannot straddle it.
    while accepted < 512 && tries < 40_000 {
        tries += 1;
        let mut m = [[0.0_f32; 4]; 4];
        let mut col = 0;
        while col < 4 {
            let mut row = 0;
            while row < 4 {
                m[col][row] = draw_signed(&mut state, 2.0);
                row += 1;
            }
            col += 1;
        }
        // Keep the translation w (last column, w row) near 1 so clip.w stays
        // clear of zero for typical vertices.
        m[3][3] = 1.0 + draw(&mut state, 1.0);
        let world = [
            draw_signed(&mut state, 3.0),
            draw_signed(&mut state, 3.0),
            draw_signed(&mut state, 3.0),
        ];
        if clip_w(&m, world).abs() < 0.25 {
            continue;
        }
        let depth = draw(&mut state, 1.4) - 0.2;
        queries.push(GeomRasterProjectQuery::new(m, world, VIEWPORT, depth));
        accepted += 1;
    }
    assert!(
        accepted >= 512,
        "expected at least 512 random samples, got {accepted}"
    );
    check(&ctx, &gpu, &queries);
}
