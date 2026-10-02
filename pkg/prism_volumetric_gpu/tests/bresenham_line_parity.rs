//! Real-device parity for the integer `Bresenham` line-rasterization twin:
//! [`GpuBresenhamLine`](prism_volumetric_gpu::bresenham_line::GpuBresenhamLine)
//! must reproduce the `CPU` golden
//! [`rasterize`](prism_render_architecture::particle::bresenham_line::rasterize)
//! and
//! [`step_count`](prism_render_architecture::particle::bresenham_line::step_count)
//! across an empty batch, horizontal and vertical runs, both pure diagonals,
//! shallow and steep slopes in every quadrant, a degenerate single-point
//! segment, a forward/reverse mirror pair, a long line near the restricted-box
//! bound, and a mixed batch, compared cell for cell.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The walk is pure integer arithmetic with no floating point, so `CPU` and
//! `GPU` agree bit for bit. The comparison asserts an *exact* `==` on the
//! emitted cell count and on every `(x, y)` cell, in order. The twin is honest
//! about scope: `WGSL` has no `i64`, so only the coordinate-restricted subset
//! (`|coord| <= MAX_COORD`) is twinnable, and every fixture lives inside that
//! box.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bresenham_line`；
//! classic integer `Bresenham` line rasterization; 纯整数、无需外部数学库；no
//! third-party engine source or derived code.

use prism_render_architecture::particle::bresenham_line::{rasterize, step_count};
use prism_volumetric_gpu::bresenham_line::{
    GpuBresenhamLine, GpuBresenhamQuery, GpuBresenhamResult, MAX_COORD,
};
use prism_volumetric_gpu::GpuContext;

/// Builds one rasterization query from two integer endpoints.
fn query(x0: i32, y0: i32, x1: i32, y1: i32) -> GpuBresenhamQuery {
    GpuBresenhamQuery { x0, y0, x1, y1 }
}

/// Runs the `GPU` dispatch and asserts strict cell-for-cell parity against the
/// `CPU` golden: the emitted `step_count` matches
/// [`step_count`](prism_render_architecture::particle::bresenham_line::step_count)
/// exactly and the cell list matches
/// [`rasterize`](prism_render_architecture::particle::bresenham_line::rasterize)
/// exactly, in order. Returns the `GPU` results for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBresenhamLine,
    queries: &[GpuBresenhamQuery],
) -> Vec<GpuBresenhamResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want_count = step_count(q.x0, q.y0, q.x1, q.y1);
        let want_cells = rasterize(q.x0, q.y0, q.x1, q.y1);
        assert_eq!(
            g.step_count as usize, want_count,
            "lane {lane}: step_count gpu {} vs cpu {want_count}",
            g.step_count
        );
        assert_eq!(
            g.cells.len(),
            want_count,
            "lane {lane}: cell count must equal step_count"
        );
        assert_eq!(
            g.cells, want_cells,
            "lane {lane}: cell walk mismatch for endpoints ({}, {}) -> ({}, {})",
            q.x0, q.y0, q.x1, q.y1
        );
    }
    got
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bresenham_line parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    let out = gpu.eval(&ctx, &[]);
    assert!(
        out.is_empty(),
        "empty query batch must return an empty vector"
    );
}

#[test]
fn horizontal_and_vertical_runs_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    let queries = [
        // Horizontal, both directions.
        query(0, 3, 6, 3),
        query(6, 3, 0, 3),
        // Vertical, both directions.
        query(-2, 0, -2, 5),
        query(-2, 5, -2, 0),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].cells.first(), Some(&(0, 3)), "horizontal start");
    assert_eq!(got[0].cells.last(), Some(&(6, 3)), "horizontal end");
    assert_eq!(got[2].cells.len(), 6, "vertical run has six cells");
}

#[test]
fn pure_diagonals_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    let queries = [
        // The two principal diagonals through the origin.
        query(0, 0, 5, 5),
        query(0, 0, 5, -5),
        query(0, 0, -5, 5),
        query(0, 0, -5, -5),
    ];
    let got = check(&ctx, &gpu, &queries);
    for g in &got {
        // A pure diagonal advances both axes by one every step.
        assert_eq!(g.step_count, 6, "a 5-span diagonal has six cells");
    }
}

#[test]
fn shallow_and_steep_slopes_each_quadrant_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    let queries = [
        // Shallow (x-dominant) in each quadrant.
        query(0, 0, 11, 4),
        query(0, 0, -11, 4),
        query(0, 0, -11, -4),
        query(0, 0, 11, -4),
        // Steep (y-dominant) in each quadrant.
        query(0, 0, 4, 11),
        query(0, 0, -4, 11),
        query(0, 0, -4, -11),
        query(0, 0, 4, -11),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_point_yields_single_cell() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    let queries = [query(4, -7, 4, -7), query(0, 0, 0, 0)];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].step_count, 1, "coincident endpoints yield one cell");
    assert_eq!(got[0].cells, vec![(4, -7)], "the single cell is the point");
}

#[test]
fn forward_and_reverse_are_exact_mirrors() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    // A generic non-diagonal segment and its reverse.
    let queries = [query(-5, -2, 12, 3), query(12, 3, -5, -2)];
    let got = check(&ctx, &gpu, &queries);
    let mut reversed = got[1].cells.clone();
    reversed.reverse();
    assert_eq!(
        got[0].cells, reversed,
        "a segment and its reverse must produce mirror-image cell lists"
    );
}

#[test]
fn long_line_near_box_bound_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    // A long, shallow line spanning most of the restricted box, exercising the
    // pure-i32 error walk over a span where the i64 golden would otherwise be
    // needed. The x-span dominates, so the cell count is the x-span plus one.
    let q = query(-MAX_COORD, -17, MAX_COORD, 23);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(
        got[0].step_count,
        (2 * MAX_COORD + 1) as u32,
        "x-dominant span sets the cell count"
    );
    assert_eq!(got[0].cells.first(), Some(&(-MAX_COORD, -17)), "start cell");
    assert_eq!(got[0].cells.last(), Some(&(MAX_COORD, 23)), "end cell");
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBresenhamLine::new(&ctx);
    // A single dispatch mixing short, long, axis-aligned, diagonal and
    // off-origin segments across all quadrants, so threads with very different
    // walk lengths run side by side.
    let queries = [
        query(0, 0, 1, 0),
        query(0, 0, 0, 1),
        query(0, 0, 1, 1),
        query(0, 0, -1, -1),
        query(100, -50, 109, -46),
        query(-30, 20, 7, -14),
        query(13, 13, -21, 2),
        query(-1000, -3, 1000, 7),
    ];
    check(&ctx, &gpu, &queries);
}
