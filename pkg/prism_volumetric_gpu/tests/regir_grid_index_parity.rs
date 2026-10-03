//! Real-device parity for the `ReGIR` grid-index geometry twin:
//! [`GpuRegirGridIndex`](prism_volumetric_gpu::regir_grid_index::GpuRegirGridIndex)
//! must reproduce the stateless index geometry of the `CPU` golden
//! [`RegirConfig`](prism_render_architecture::lighting::regir::RegirConfig) — the
//! position-to-index map (with the reference's out-of-grid rejection), the
//! index-to-coordinate decode, and the coordinate-to-center map — across typical
//! in-grid lookups, per-axis boundary and out-of-grid cases, a
//! `linear`-to-`coords` round trip, degenerate `cell_size`, and a randomized
//! sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference methods are public, so they are called directly as the oracle:
//! [`cell_index_of`](prism_render_architecture::lighting::regir::RegirConfig::cell_index_of)
//! for the position-to-index direction (its [`Option`] mapping to the
//! `in_grid` flag plus the linear index),
//! [`coords_of`](prism_render_architecture::lighting::regir::RegirConfig::coords_of)
//! for the index-to-coordinate decode, and
//! [`cell_center`](prism_render_architecture::lighting::regir::RegirConfig::cell_center)
//! for the coordinate-to-center map. A passing `GPU == oracle` run is direct
//! evidence the kernel computes the same cell geometry.
//!
//! # Parity criterion
//!
//! The `in_grid` flag, the linear index and the decoded coordinates are built
//! from ordered comparisons and integer `%`/`/`, so for fixtures clear of a cell
//! boundary they agree exactly and are asserted with `==`. The continuous cell
//! center threads through a multiply and an add, so it is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The randomized sweep keeps every axis's `local = (pos - grid_min) /
//! cell_size` a safe fraction of a cell away from an integer, so a `GPU` divide
//! and a `CPU` divide truncate to the same cell and the discrete answers stay
//! bit-identical. Degenerate `cell_size` fixtures (zero and `NaN`) are asserted
//! only on the `in_grid` rejection, since a `NaN` `cell_size` propagates a `NaN`
//! center on both sides that no tolerance can compare.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::regir`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::regir::RegirConfig;
use prism_volumetric_gpu::regir_grid_index::{
    GpuRegirGridIndex, RegirGridIndexQuery, RegirGridIndexResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a cell-center coordinate. A `GPU` multiply-add may
/// land a few units in the last place from the scalar reference; `1e-4` admits
/// that legal slack while still failing a wrong port.
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

/// Builds a reference config from a query's grid fields. `reservoirs_per_cell`
/// is irrelevant to the index geometry under test and is fixed at `1`.
fn config_of(q: &RegirGridIndexQuery) -> RegirConfig {
    RegirConfig {
        grid_min: q.grid_min,
        cell_size: q.cell_size,
        dims: q.dims,
        reservoirs_per_cell: 1,
    }
}

/// Computes the reference answer for one query by calling the golden directly:
/// the position-to-index [`Option`] becomes the `in_grid` flag plus the linear
/// index, and the two other directions decode `linear` and map `coords`.
fn oracle(q: &RegirGridIndexQuery) -> RegirGridIndexResult {
    let cfg = config_of(q);
    let (linear, in_grid) = match cfg.cell_index_of(q.pos) {
        Some(l) => (l as u32, true),
        None => (0u32, false),
    };
    let coords = cfg.coords_of(q.linear as usize);
    let center = cfg.cell_center(q.coords);
    RegirGridIndexResult {
        linear,
        in_grid,
        coords,
        center,
    }
}

/// Pins one `GPU` result against the oracle: the flag, the linear index and the
/// decoded coordinates exactly, the center within tolerance.
fn check_result(idx: usize, got: &RegirGridIndexResult, want: &RegirGridIndexResult) {
    assert_eq!(
        got.in_grid, want.in_grid,
        "query {idx} in_grid: gpu {} vs cpu {}",
        got.in_grid, want.in_grid
    );
    assert_eq!(
        got.linear, want.linear,
        "query {idx} linear: gpu {} vs cpu {}",
        got.linear, want.linear
    );
    assert_eq!(
        got.coords, want.coords,
        "query {idx} coords: gpu {:?} vs cpu {:?}",
        got.coords, want.coords
    );
    for axis in 0..3 {
        assert!(
            close(got.center[axis], want.center[axis]),
            "query {idx} center[{axis}]: gpu {} vs cpu {}",
            got.center[axis],
            want.center[axis]
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRegirGridIndex, queries: &[RegirGridIndexQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// The deterministic grid fixture: a non-aligned origin, a sub-unit cell size,
/// and an asymmetric extent so each axis's decode and bound differ.
const GRID_MIN: [f32; 3] = [-4.0, 2.0, -1.5];
/// Cell edge length for the fixture grid.
const CELL_SIZE: f32 = 0.5;
/// Cell extent of the fixture grid; `cell_count` is `8 * 6 * 10 = 480`.
const DIMS: [u32; 3] = [8, 6, 10];

/// Total cells in the fixture grid.
fn cell_count() -> u32 {
    DIMS[0] * DIMS[1] * DIMS[2]
}

/// Builds a query over the fixture grid with the three direction inputs filled.
fn query(pos: [f32; 3], linear: u32, coords: [u32; 3]) -> RegirGridIndexQuery {
    RegirGridIndexQuery::new(GRID_MIN, CELL_SIZE, DIMS, pos, linear, coords)
}

/// Returns a world position landing in cell `cell` along `axis` at a safe
/// fractional offset `frac` of a cell (kept clear of `0` and `1` so a `GPU` and
/// a `CPU` divide truncate to the same cell).
fn axis_pos(axis: usize, cell: u32, frac: f32) -> f32 {
    GRID_MIN[axis] + (cell as f32 + frac) * CELL_SIZE
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping regir_grid_index parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn typical_in_grid_pos_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // A position safely inside cell (3, 2, 5): local fractions all near 0.5.
    let pos = [
        axis_pos(0, 3, 0.5),
        axis_pos(1, 2, 0.5),
        axis_pos(2, 5, 0.5),
    ];
    check(&ctx, &gpu, &[query(pos, 123, [3, 2, 5])]);
}

#[test]
fn last_cell_boundary_is_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // One position in the last cell of each axis (`dims - 1`), still inside.
    let pos = [
        axis_pos(0, DIMS[0] - 1, 0.5),
        axis_pos(1, DIMS[1] - 1, 0.5),
        axis_pos(2, DIMS[2] - 1, 0.5),
    ];
    let got = gpu.evaluate(&ctx, &[query(pos, 0, [0, 0, 0])]);
    assert_eq!(got.len(), 1);
    assert!(got[0].in_grid, "a position in the last cell must be inside");
    check_result(0, &got[0], &oracle(&query(pos, 0, [0, 0, 0])));
}

#[test]
fn past_high_edge_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // One axis pushed past its high edge (`local >= dims`): outside the grid.
    let queries = [
        query(
            [
                axis_pos(0, DIMS[0], 0.3),
                axis_pos(1, 2, 0.5),
                axis_pos(2, 5, 0.5),
            ],
            0,
            [0, 0, 0],
        ),
        query(
            [
                axis_pos(0, 3, 0.5),
                axis_pos(1, DIMS[1], 0.3),
                axis_pos(2, 5, 0.5),
            ],
            0,
            [0, 0, 0],
        ),
        query(
            [
                axis_pos(0, 3, 0.5),
                axis_pos(1, 2, 0.5),
                axis_pos(2, DIMS[2], 0.3),
            ],
            0,
            [0, 0, 0],
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, result) in got.iter().enumerate() {
        assert!(
            !result.in_grid,
            "query {idx} past the high edge must be outside"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn below_grid_min_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // Each axis in turn placed below `grid_min` (`local < 0`): outside the grid.
    let queries = [
        query(
            [GRID_MIN[0] - 1.0, axis_pos(1, 2, 0.5), axis_pos(2, 5, 0.5)],
            0,
            [0, 0, 0],
        ),
        query(
            [axis_pos(0, 3, 0.5), GRID_MIN[1] - 1.0, axis_pos(2, 5, 0.5)],
            0,
            [0, 0, 0],
        ),
        query(
            [axis_pos(0, 3, 0.5), axis_pos(1, 2, 0.5), GRID_MIN[2] - 1.0],
            0,
            [0, 0, 0],
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, result) in got.iter().enumerate() {
        assert!(
            !result.in_grid,
            "query {idx} below grid_min must be outside"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn linear_to_coords_round_trip_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // Every linear index decodes to the coordinates the golden produces; the
    // same coordinates re-encode back to the linear index (checked in-host).
    let mut queries = Vec::new();
    let n = cell_count();
    let mut lin = 0u32;
    while lin < n {
        let pos = [
            axis_pos(0, 1, 0.5),
            axis_pos(1, 1, 0.5),
            axis_pos(2, 1, 0.5),
        ];
        queries.push(query(pos, lin, [0, 0, 0]));
        lin += 7;
    }
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let cfg = config_of(q);
        let want = cfg.coords_of(q.linear as usize);
        assert_eq!(result.coords, want, "query {idx} decoded coords mismatch");
        // Re-encode: the decoded coordinates must map back to the same index.
        assert_eq!(
            cfg.linear_index(result.coords) as u32,
            q.linear,
            "query {idx} round trip mismatch"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn cell_center_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // A spread of coordinates (including the extremes) exercising the center map.
    let coords = [[0, 0, 0], [7, 5, 9], [3, 2, 5], [1, 4, 8], [6, 0, 2]];
    let queries: Vec<RegirGridIndexQuery> = coords
        .iter()
        .map(|&c| {
            query(
                [
                    axis_pos(0, 1, 0.5),
                    axis_pos(1, 1, 0.5),
                    axis_pos(2, 1, 0.5),
                ],
                0,
                c,
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn zero_cell_size_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // A non-positive cell_size rejects every position; `in_grid` must be false
    // and the linear index zero, matching the golden `None`.
    let q = RegirGridIndexQuery::new(GRID_MIN, 0.0, DIMS, [0.0, 3.0, 0.0], 10, [2, 1, 3]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert!(!got[0].in_grid, "a zero cell_size must reject the position");
    assert_eq!(
        got[0].linear, 0,
        "a rejected position has linear index zero"
    );
    // The center uses cell_size == 0, so it collapses to grid_min on both sides.
    let want = oracle(&q);
    for axis in 0..3 {
        assert!(
            close(got[0].center[axis], want.center[axis]),
            "center[{axis}] gpu {} vs cpu {}",
            got[0].center[axis],
            want.center[axis]
        );
    }
    // The coordinate decode is independent of cell_size.
    assert_eq!(got[0].coords, want.coords);
}

#[test]
fn nan_cell_size_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    // A NaN cell_size rejects every position via the ordered `!(cs > 0.0)`
    // guard. The center propagates NaN on both sides, so only the rejection and
    // the cell_size-independent decode are asserted.
    let q = RegirGridIndexQuery::new(GRID_MIN, f32::NAN, DIMS, [0.0, 3.0, 0.0], 10, [2, 1, 3]);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert!(!got[0].in_grid, "a NaN cell_size must reject the position");
    assert_eq!(
        got[0].linear, 0,
        "a rejected position has linear index zero"
    );
    let want = oracle(&q);
    assert_eq!(
        got[0].coords, want.coords,
        "coord decode is cell_size-independent"
    );
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRegirGridIndex::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let n = cell_count();
    let mut queries = Vec::new();
    // Several workgroups' worth of queries: each picks an in-grid position at a
    // safe fractional offset, an independent linear index to decode, and
    // independent coordinates to map to a center.
    for _ in 0..300 {
        let mut pos = [0.0f32; 3];
        for axis in 0..3 {
            let cell = lcg(&mut state) % DIMS[axis];
            // Fraction in [0.15, 0.85], clear of a cell boundary.
            let frac = 0.15 + (lcg(&mut state) % 701) as f32 / 1000.0;
            pos[axis] = axis_pos(axis, cell, frac);
        }
        let linear = lcg(&mut state) % n;
        let coords = [
            lcg(&mut state) % DIMS[0],
            lcg(&mut state) % DIMS[1],
            lcg(&mut state) % DIMS[2],
        ];
        queries.push(query(pos, linear, coords));
    }
    check(&ctx, &gpu, &queries);
}
