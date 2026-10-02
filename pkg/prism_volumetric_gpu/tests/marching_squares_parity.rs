//! Real-device parity for the Marching Squares per-cell twin:
//! [`GpuMarchingSquares`](prism_volumetric_gpu::marching_squares::GpuMarchingSquares)
//! must reproduce the `CPU` golden
//! [`marching_squares`](prism_render_architecture::particle::marching_squares)
//! per-cell path: the
//! [`case_index`](prism_render_architecture::particle::marching_squares::case_index)
//! classification code, the number of segments the cell contributes, those
//! segments' endpoint coordinates (built from the reference edge table and
//! [`lerp_param`](prism_render_architecture::particle::marching_squares::lerp_param)),
//! and each produced segment's
//! [`Segment::length`](prism_render_architecture::particle::marching_squares::Segment::length).
//!
//! Each cell query is checked against a single-cell `2 × 2` invocation of the
//! reference
//! [`extract_contours`](prism_render_architecture::particle::marching_squares::extract_contours),
//! which is exactly the per-cell primitive the twin reproduces — the growable
//! whole-grid concatenation is deliberately not twinned. The single cell is laid
//! out row-major as `[v0, v1, v3, v2]` (row `0` then row `1`), so the reference
//! segments are produced at grid origin and translated by the query `base` to
//! compare against the device, since the reference `edge_point` is a pure
//! translation in the cell's bottom-left corner.
//!
//! The fixtures cover every case family that yields a segment — the four
//! single-corner cases (`1`, `2`, `4`, `8`), the edge-split cases (`3`, `6`), a
//! three-corner case (`7`), both diagonal saddles (`5` and `10`) with the cell
//! center resolved inside and outside — and the two empty cases (`0` and `15`).
//! A dedicated fixture drives the `lerp_param` guarded-division degeneracy: a
//! crossed edge whose two corner values differ by less than `EPS`, forcing the
//! midpoint `0.5` fallback. All corner values sit a clear interval away from the
//! `== iso` classification boundary (rejection sampling by construction), so the
//! case codes are unambiguous; coordinates are written as integers or simple
//! rationals, so the fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The case code and the segment count are discrete classifications, so `CPU`
//! and `GPU` must agree exactly (`==`). The crossing coordinates and the segment
//! lengths thread through subtracts, one guarded division, adds and a `sqrt`, so
//! they are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::marching_squares`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::marching_squares::{case_index, extract_contours};
use prism_volumetric_gpu::marching_squares::{GpuMarchingSquares, MarchingSquaresQuery};
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

/// Tolerant comparison of two 2D endpoints.
fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Builds a cell query from the four corner values, the cell base coordinate,
/// and the iso threshold.
fn query(corners: [f32; 4], base: [f32; 2], iso: f32) -> MarchingSquaresQuery {
    MarchingSquaresQuery { corners, base, iso }
}

/// Row-major `2 × 2` single cell from corner values named in the
/// `[bottom-left, bottom-right, top-right, top-left]` order used by
/// [`case_index`]. Row-major layout stores row `0` (`v0`, `v1`) then row `1`
/// (`v3`, `v2`).
fn cell2(corners: [f32; 4]) -> [f32; 4] {
    [corners[0], corners[1], corners[3], corners[2]]
}

/// Asserts the per-cell twin matches the `CPU` golden for one query.
fn assert_parity(gpu: &GpuMarchingSquares, ctx: &GpuContext, q: &MarchingSquaresQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let [v0, v1, v2, v3] = q.corners;
    let cpu_case = u32::from(case_index(v0, v1, v2, v3, q.iso));
    assert_eq!(
        g.case_index, cpu_case,
        "case_index mismatch for corners {:?} iso {}",
        q.corners, q.iso
    );

    // The reference per-cell primitive is a single-cell 2x2 extraction at grid
    // origin; translate its endpoints by the query base, since edge_point is a
    // pure translation in the cell's bottom-left corner.
    let field = cell2(q.corners);
    let cpu_segs = extract_contours(&field, 2, 2, q.iso);
    assert_eq!(
        g.seg_count as usize,
        cpu_segs.len(),
        "seg_count mismatch for corners {:?} iso {}",
        q.corners,
        q.iso
    );

    let bx = q.base[0];
    let by = q.base[1];
    for (i, cpu) in cpu_segs.iter().enumerate() {
        let want_a = [cpu.a.x + bx, cpu.a.y + by];
        let want_b = [cpu.b.x + bx, cpu.b.y + by];
        let gpu_a = g.segments[i][0];
        let gpu_b = g.segments[i][1];
        assert!(
            approx2(gpu_a, want_a) && approx2(gpu_b, want_b),
            "segment {i} endpoint mismatch: gpu [{gpu_a:?}, {gpu_b:?}] vs cpu [{want_a:?}, {want_b:?}]"
        );
        // Segment length (the module's only sqrt) is base-invariant.
        let want_len = cpu.length();
        assert!(
            approx(g.lengths[i], want_len),
            "segment {i} length mismatch: gpu {} vs cpu {want_len}",
            g.lengths[i]
        );
    }
}

#[test]
fn empty_cases_emit_no_segments() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 0: all corners below iso. Case 15: all corners inside.
    let all_below = query([0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.5);
    let all_above = query([3.0, 3.0, 3.0, 3.0], [0.0, 0.0], 0.5);
    assert_parity(&gpu, &ctx, &all_below);
    assert_parity(&gpu, &ctx, &all_above);
}

#[test]
fn single_corner_cases_one_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 1: bottom-left inside -> left/bottom segment.
    assert_parity(&gpu, &ctx, &query([2.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.5));
    // Case 2: bottom-right inside -> bottom/right segment.
    assert_parity(&gpu, &ctx, &query([0.0, 2.0, 0.0, 0.0], [0.0, 0.0], 0.5));
    // Case 4: top-right inside -> right/top segment.
    assert_parity(&gpu, &ctx, &query([0.0, 0.0, 2.0, 0.0], [0.0, 0.0], 0.5));
    // Case 8: top-left inside -> top/left segment.
    assert_parity(&gpu, &ctx, &query([0.0, 0.0, 0.0, 2.0], [0.0, 0.0], 0.5));
}

#[test]
fn edge_split_cases_one_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 3: bottom row inside -> horizontal left/right split.
    assert_parity(&gpu, &ctx, &query([2.0, 2.0, 0.0, 0.0], [0.0, 0.0], 0.5));
    // Case 6: right column inside -> vertical bottom/top split.
    assert_parity(&gpu, &ctx, &query([0.0, 2.0, 2.0, 0.0], [0.0, 0.0], 0.5));
}

#[test]
fn three_corner_case_isolates_missing_corner() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 7: v0, v1, v2 inside, v3 outside -> top/left segment isolating the
    // top-left corner.
    assert_parity(&gpu, &ctx, &query([2.0, 2.0, 2.0, 0.0], [0.0, 0.0], 0.5));
}

#[test]
fn saddle_case_5_resolves_both_ways() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 5: bottom-left and top-right inside. Center average 1.0 >= 0.5 keeps
    // the center inside -> two segments connecting bottom/right and top/left.
    let inside = query([2.0, 0.0, 2.0, 0.0], [0.0, 0.0], 0.5);
    assert_parity(&gpu, &ctx, &inside);
    // Case 5 again but center average -0.5 < 0.5 so the center is outside ->
    // the complementary pairing left/bottom and right/top.
    let outside = query([1.0, -2.0, 1.0, -2.0], [0.0, 0.0], 0.5);
    assert_parity(&gpu, &ctx, &outside);
}

#[test]
fn saddle_case_10_resolves_both_ways() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Case 10: bottom-right and top-left inside. Center average 1.0 >= 0.5.
    let inside = query([0.0, 2.0, 0.0, 2.0], [0.0, 0.0], 0.5);
    assert_parity(&gpu, &ctx, &inside);
    // Case 10 with center average -0.5 < 0.5.
    let outside = query([-2.0, 1.0, -2.0, 1.0], [0.0, 0.0], 0.5);
    assert_parity(&gpu, &ctx, &outside);
}

#[test]
fn nonzero_base_translates_segments() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // A single-corner cell placed at base (3, 2): every endpoint is translated.
    assert_parity(&gpu, &ctx, &query([2.0, 0.0, 0.0, 0.0], [3.0, 2.0], 0.5));
    // A saddle cell at a non-origin base exercises two translated segments.
    assert_parity(&gpu, &ctx, &query([2.0, 0.0, 2.0, 0.0], [5.0, 7.0], 0.5));
}

#[test]
fn negative_iso_and_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // v0 = -1 >= -2 inside; the rest -3 < -2 outside -> case 1.
    assert_parity(
        &gpu,
        &ctx,
        &query([-1.0, -3.0, -3.0, -3.0], [0.0, 0.0], -2.0),
    );
}

#[test]
fn lerp_param_guarded_division_degeneracy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // Bottom edge corners straddle iso = 0 with a tiny gap: v0 = 4e-7 (inside),
    // v1 = -4e-7 (outside) differ by 8e-7 < EPS = 1e-6, so the bottom crossing
    // falls back to the midpoint 0.5 instead of dividing by ~zero. The other
    // corners sit a clear interval below iso, so the classification (case 1) is
    // unambiguous.
    let q = query([4.0e-7, -4.0e-7, -1.0, -1.0], [0.0, 0.0], 0.0);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn batch_of_cells_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // A mixed batch exercises the one-thread-per-cell flattening; each result
    // must be independent of its neighbours.
    let batch = [
        query([0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.5),
        query([2.0, 0.0, 0.0, 0.0], [1.0, 0.0], 0.5),
        query([2.0, 0.0, 2.0, 0.0], [2.0, 3.0], 0.5),
        query([0.0, 2.0, 2.0, 0.0], [0.0, 4.0], 0.5),
        query([3.0, 3.0, 3.0, 3.0], [1.0, 1.0], 0.5),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingSquares::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
