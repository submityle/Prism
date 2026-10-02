//! Real-device parity for the statistics debug-overlay twin:
//! [`GpuStatsOverlay`](prism_volumetric_gpu::stats_overlay::GpuStatsOverlay)
//! must reproduce the `CPU` golden
//! [`stats_overlay`](prism_render_architecture::particle::stats_overlay)
//! numeric surface across an empty batch, every per-op fixture, a mixed tagged
//! batch and a large pseudo-random batch compared lane for lane.
//!
//! The twin reproduces the module's pure numeric pieces one thread per query:
//! the
//! [`histogram_bucket`](prism_render_architecture::particle::stats_overlay::histogram_bucket)
//! divide-and-`floor`, the saturating `u32` grid products
//! [`OverlayLayout::cell_count`](prism_render_architecture::particle::stats_overlay::OverlayLayout::cell_count),
//! [`OverlayLayout::grid_width_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_width_px)
//! and
//! [`OverlayLayout::grid_height_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_height_px),
//! the cell-index reverse lookups
//! [`OverlayLayout::row_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::row_index)
//! and
//! [`OverlayLayout::col_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::col_index),
//! and the color table of
//! [`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba).
//! The variable-length
//! [`build_rows`](prism_render_architecture::particle::stats_overlay::build_rows)
//! builder, the `u64`
//! [`OverlayLayout::vertex_bytes`](prism_render_architecture::particle::stats_overlay::OverlayLayout::vertex_bytes)
//! budget and the `&'static str`
//! [`OverlayStatField::label`](prism_render_architecture::particle::stats_overlay::OverlayStatField::label)
//! lookup stay host-side and are not twinned.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer answers (`histogram_bucket`, `cell_count`, `grid_width_px`,
//! `grid_height_px`, `row_index`, `col_index`) are exact integer work, so they
//! are compared bit for bit. Only the `RGBA` color channels are `f32`, and they
//! are the exact literals `0.0`/`1.0`; they are compared with `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` to honor the `f32`-never-`==` rule. The histogram
//! fixtures stay clear of exact bucket edges (where a `floor` could flip across
//! `CPU`/`GPU` `ULP`) by rejection sampling, except for the explicit degenerate
//! cases.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`；
//! 无需外部数学库，无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::stats_overlay::{
    histogram_bucket, OverlayLayout, OverlayRow, OverlayStatField,
};
use prism_volumetric_gpu::stats_overlay::{GpuStatsOverlay, StatsOverlayQuery, StatsOverlayResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the `f32` color lanes. The channels are exact
/// `0.0`/`1.0` literals, but the rule forbids `f32` equality, so a tolerance is
/// used even though the values are exact.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether two `RGBA` colors agree channel-wise within tolerance.
fn approx_rgba(a: [f32; 4], b: [f32; 4]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3])
}

/// Builds an [`OverlayRow`] that classifies to the given `severity_code`
/// (`0` nominal, `1` warn, `2` critical), so the golden
/// [`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba)
/// drives the expected color.
fn severity_row(severity_code: u32) -> OverlayRow {
    match severity_code {
        2 => OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 100,
            warn_threshold: 10,
            critical_threshold: 50,
        },
        1 => OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 100,
            warn_threshold: 10,
            critical_threshold: 0,
        },
        _ => OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 0,
            warn_threshold: 0,
            critical_threshold: 0,
        },
    }
}

/// Evaluates the `CPU` golden for one query, returning the expected
/// [`StatsOverlayResult`].
fn expected(query: &StatsOverlayQuery) -> StatsOverlayResult {
    match query {
        StatsOverlayQuery::HistogramBucket {
            value,
            min,
            max,
            bucket_count,
        } => StatsOverlayResult::Int(histogram_bucket(*value, *min, *max, *bucket_count)),
        StatsOverlayQuery::CellCount { columns, rows } => {
            let layout = OverlayLayout::new(*columns, *rows, 1, 1);
            StatsOverlayResult::Int(layout.cell_count())
        }
        StatsOverlayQuery::GridWidth {
            columns,
            cell_width_px,
        } => {
            let layout = OverlayLayout::new(*columns, 1, *cell_width_px, 1);
            StatsOverlayResult::Int(layout.grid_width_px())
        }
        StatsOverlayQuery::GridHeight {
            rows,
            cell_height_px,
        } => {
            let layout = OverlayLayout::new(1, *rows, 1, *cell_height_px);
            StatsOverlayResult::Int(layout.grid_height_px())
        }
        StatsOverlayQuery::RowIndex { cell, columns } => {
            let layout = OverlayLayout::new(*columns, 1, 1, 1);
            StatsOverlayResult::Int(layout.row_index(*cell))
        }
        StatsOverlayQuery::ColIndex { cell, columns } => {
            let layout = OverlayLayout::new(*columns, 1, 1, 1);
            StatsOverlayResult::Int(layout.col_index(*cell))
        }
        StatsOverlayQuery::SeverityRgba { severity_code } => {
            StatsOverlayResult::Rgba(severity_row(*severity_code).severity_rgba())
        }
    }
}

/// Asserts the `GPU` result matches the `CPU` golden for one lane: integers bit
/// for bit, color channels within tolerance.
fn compare(lane: usize, query: &StatsOverlayQuery, got: &StatsOverlayResult) {
    let want = expected(query);
    match (got, &want) {
        (StatsOverlayResult::Int(g), StatsOverlayResult::Int(w)) => {
            assert_eq!(*g, *w, "lane {lane}: int gpu {g} vs cpu {w}");
        }
        (StatsOverlayResult::Rgba(g), StatsOverlayResult::Rgba(w)) => {
            assert!(
                approx_rgba(*g, *w),
                "lane {lane}: rgba gpu {g:?} vs cpu {w:?}"
            );
        }
        (g, w) => panic!("lane {lane}: result kind mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches a single query and asserts it matches the golden.
fn check_one(ctx: &GpuContext, gpu: &GpuStatsOverlay, query: StatsOverlayQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one result per query");
    compare(0, &query, &got[0]);
}

// ---------------------------------------------------------------------------
// Per-op fixtures.
// ---------------------------------------------------------------------------

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn histogram_bucket_interior_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    // Interior samples well away from bucket edges.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 2.5,
            min: 0.0,
            max: 10.0,
            bucket_count: 10,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 9.9,
            min: 0.0,
            max: 10.0,
            bucket_count: 10,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 0.5,
            min: 0.0,
            max: 10.0,
            bucket_count: 10,
        },
    );
}

#[test]
fn histogram_bucket_clamps_and_degenerates_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    // Below min, at min, above max, at max.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: -5.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 8,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 0.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 8,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 11.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 8,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 10.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 8,
        },
    );
    // Degenerate range (min == max and min > max) and degenerate bucket counts.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 5.0,
            min: 10.0,
            max: 10.0,
            bucket_count: 8,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 5.0,
            min: 10.0,
            max: 0.0,
            bucket_count: 8,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 5.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 0,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::HistogramBucket {
            value: 5.0,
            min: 0.0,
            max: 10.0,
            bucket_count: 1,
        },
    );
}

#[test]
fn cell_count_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::CellCount {
            columns: 3,
            rows: 2,
        },
    );
    // Clamped-to-one inputs still match.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::CellCount {
            columns: 1,
            rows: 1,
        },
    );
}

#[test]
fn cell_count_saturates_on_huge_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    // The product overflows u32 and must saturate to u32::MAX, matching the
    // reference `saturating_mul`.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::CellCount {
            columns: u32::MAX,
            rows: u32::MAX,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::CellCount {
            columns: 100_000,
            rows: 100_000,
        },
    );
}

#[test]
fn grid_dimensions_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::GridWidth {
            columns: 3,
            cell_width_px: 10,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::GridHeight {
            rows: 2,
            cell_height_px: 20,
        },
    );
    // Saturating products.
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::GridWidth {
            columns: u32::MAX,
            cell_width_px: u32::MAX,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::GridHeight {
            rows: u32::MAX,
            cell_height_px: 7,
        },
    );
}

#[test]
fn cell_index_reverse_lookup_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::RowIndex {
            cell: 6,
            columns: 4,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::ColIndex {
            cell: 6,
            columns: 4,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::RowIndex {
            cell: 0,
            columns: 4,
        },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::ColIndex {
            cell: 0,
            columns: 4,
        },
    );
}

#[test]
fn severity_rgba_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::SeverityRgba { severity_code: 0 },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::SeverityRgba { severity_code: 1 },
    );
    check_one(
        &ctx,
        &gpu,
        StatsOverlayQuery::SeverityRgba { severity_code: 2 },
    );
}

// ---------------------------------------------------------------------------
// Mixed batch.
// ---------------------------------------------------------------------------

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    let queries = vec![
        StatsOverlayQuery::HistogramBucket {
            value: 3.3,
            min: 0.0,
            max: 10.0,
            bucket_count: 16,
        },
        StatsOverlayQuery::CellCount {
            columns: 5,
            rows: 4,
        },
        StatsOverlayQuery::GridWidth {
            columns: 6,
            cell_width_px: 12,
        },
        StatsOverlayQuery::GridHeight {
            rows: 3,
            cell_height_px: 18,
        },
        StatsOverlayQuery::RowIndex {
            cell: 11,
            columns: 4,
        },
        StatsOverlayQuery::ColIndex {
            cell: 11,
            columns: 4,
        },
        StatsOverlayQuery::SeverityRgba { severity_code: 0 },
        StatsOverlayQuery::SeverityRgba { severity_code: 1 },
        StatsOverlayQuery::SeverityRgba { severity_code: 2 },
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random batch.
// ---------------------------------------------------------------------------

/// A tiny host-side `u64` linear congruential generator, so the fixtures use no
/// `f32` transcendental method. The multiplier and increment are the well-known
/// `PCG` / `Knuth` constants.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A float in `[0, 1)` with 24 bits of entropy.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A histogram sample value in `(min, max)` whose scaled position stays at
    /// least `0.02` of a bucket away from any integer edge, so the `floor`
    /// never flips across `CPU`/`GPU` `ULP`.
    fn histogram_value(&mut self, min: f32, max: f32, bucket_count: u32) -> f32 {
        loop {
            let t = self.unit();
            let scaled = t * (bucket_count as f32);
            let frac = scaled - (scaled as u32) as f32;
            if frac > 0.02 && frac < 0.98 {
                return min + t * (max - min);
            }
        }
    }
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStatsOverlay::new(&ctx);
    let mut rng = Lcg::new(0x5ea7_1d3b_u64);
    let mut queries: Vec<StatsOverlayQuery> = Vec::new();
    for _ in 0..200 {
        let op = rng.next_u32() % 7;
        let query = match op {
            0 => {
                let bucket_count = 2 + rng.next_u32() % 30;
                StatsOverlayQuery::HistogramBucket {
                    value: rng.histogram_value(0.0, 10.0, bucket_count),
                    min: 0.0,
                    max: 10.0,
                    bucket_count,
                }
            }
            1 => StatsOverlayQuery::CellCount {
                // Clamped-equivalent inputs (at least one), small enough to
                // avoid overflow.
                columns: 1 + rng.next_u32() % 64,
                rows: 1 + rng.next_u32() % 64,
            },
            2 => StatsOverlayQuery::GridWidth {
                columns: 1 + rng.next_u32() % 64,
                cell_width_px: rng.next_u32() % 64,
            },
            3 => StatsOverlayQuery::GridHeight {
                rows: 1 + rng.next_u32() % 64,
                cell_height_px: rng.next_u32() % 64,
            },
            4 => {
                let columns = 1 + rng.next_u32() % 16;
                StatsOverlayQuery::RowIndex {
                    cell: rng.next_u32() % 256,
                    columns,
                }
            }
            5 => {
                let columns = 1 + rng.next_u32() % 16;
                StatsOverlayQuery::ColIndex {
                    cell: rng.next_u32() % 256,
                    columns,
                }
            }
            _ => StatsOverlayQuery::SeverityRgba {
                severity_code: rng.next_u32() % 3,
            },
        };
        queries.push(query);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}
