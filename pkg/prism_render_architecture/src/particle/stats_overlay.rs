//! `CPU`-verifiable layout contract for the particle statistics debug overlay
//! (design §9 pipeline stats, §13 culling counters, §30 in-engine `HUD`).
//!
//! Production `GPU`-driven `VFX` engines (`Unreal` `Niagara`, `Unity`
//! `VFX Graph`, `Frostbite`'s `FX` stack) render a small on-screen overlay that
//! echoes the per-frame simulation counters read back from the device: how many
//! particles are alive, how many spawned or died this frame, how deep the pool
//! peaked, and whether any budget overflowed. The counters themselves arrive
//! through the latent `GPU`→`CPU` readback path; this module owns the pure,
//! device-free half that turns a reduced counter record into a drawable overlay
//! *layout* — the rows, their severity colors, the grid geometry, and the
//! vertex byte budget a quad batch needs.
//!
//! Everything here is deterministic integer/`f32` arithmetic:
//!
//! 1. [`OverlayStatField`] — the seven counters the overlay can display, with a
//!    stable [`OverlayStatField::ALL`] table and a human [`OverlayStatField::label`].
//! 2. [`Severity`] and [`OverlayRow`] — one displayed counter plus its warn and
//!    critical thresholds, folded into a [`Severity`] and a normalized `RGBA`
//!    tint.
//! 3. [`OverlayLayout`] — the clamped grid of cells the overlay draws into, with
//!    cell-index reverse lookups and the quad vertex byte budget derived from
//!    the shared `std430` strides.
//! 4. [`histogram_bucket`] — the linear bucket assignment a spark-line/histogram
//!    widget uses, with every degenerate range folded to bucket zero.
//! 5. [`build_rows`] — the shortest-slice-aligned reference builder that zips
//!    values, fields and thresholds into [`OverlayRow`]s.
//!
//! No transcendental math is used anywhere: bucketing is a single divide and a
//! [`f32::floor`], and no `f32` equality is ever tested (see [`CMP_EPS`]).

use crate::particle::gpu_layout::{VEC2_STRIDE, VEC4_STRIDE};
use alloc::vec::Vec;

/// Absolute tolerance for comparing normalized `f32` color channels.
///
/// The overlay never tests `f32` values for exact equality; callers and tests
/// compare channels with `(a - b).abs() < CMP_EPS` instead.
pub const CMP_EPS: f32 = 1e-6;

/// One statistic the debug overlay can display for the current frame.
///
/// This mirrors the reduced `GPU` readback record conceptually but is declared
/// independently so the overlay layer never depends on the readback module: the
/// overlay only needs the *identity* and *label* of each counter, not its
/// `std430` byte offset. New fields must be appended so [`OverlayStatField::ALL`]
/// and any persisted overlay preset stay stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OverlayStatField {
    /// Particles still alive at the end of the frame's simulation.
    LiveCount,
    /// Particles spawned this frame by the rate/burst emission passes.
    SpawnedThisFrame,
    /// Particles killed this frame by age-out, events, or collision.
    KilledThisFrame,
    /// High-water mark of [`OverlayStatField::LiveCount`] since the last reset.
    PeakLiveCount,
    /// Simulation time in fixed-point milliseconds (microseconds folded into a
    /// scaled integer so the overlay never stores an `f32` timing).
    SimMillisScaled,
    /// Times a per-frame budget (pool capacity, spawn budget) overflowed.
    OverflowCount,
    /// Emitters that produced at least one live particle this frame.
    ActiveEmitters,
}

impl OverlayStatField {
    /// Every overlay field in stable display order.
    pub const ALL: [OverlayStatField; 7] = [
        OverlayStatField::LiveCount,
        OverlayStatField::SpawnedThisFrame,
        OverlayStatField::KilledThisFrame,
        OverlayStatField::PeakLiveCount,
        OverlayStatField::SimMillisScaled,
        OverlayStatField::OverflowCount,
        OverlayStatField::ActiveEmitters,
    ];

    /// Short human-readable label drawn next to the counter value.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            OverlayStatField::LiveCount => "Live",
            OverlayStatField::SpawnedThisFrame => "Spawned",
            OverlayStatField::KilledThisFrame => "Killed",
            OverlayStatField::PeakLiveCount => "Peak",
            OverlayStatField::SimMillisScaled => "Sim ms",
            OverlayStatField::OverflowCount => "Overflow",
            OverlayStatField::ActiveEmitters => "Emitters",
        }
    }
}

/// How urgently a counter should be highlighted on the overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Within budget: nothing to flag.
    Nominal,
    /// At or above the warn threshold but below critical.
    Warn,
    /// At or above the critical threshold.
    Critical,
}

/// One displayed counter and the thresholds that decide its highlight color.
///
/// A threshold of `0` disables that level, so a counter that should never turn
/// red (an informational gauge) simply carries `critical_threshold == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverlayRow {
    /// Which statistic this row shows.
    pub field: OverlayStatField,
    /// The counter value read back for the current frame.
    pub value: u64,
    /// Value at or above which the row is [`Severity::Warn`] (`0` disables).
    pub warn_threshold: u64,
    /// Value at or above which the row is [`Severity::Critical`] (`0` disables).
    pub critical_threshold: u64,
}

impl OverlayRow {
    /// Classifies the row: critical wins over warn, and a zero threshold
    /// disables its level so it can never fire.
    #[must_use]
    pub fn severity(&self) -> Severity {
        if self.critical_threshold > 0 && self.value >= self.critical_threshold {
            Severity::Critical
        } else if self.warn_threshold > 0 && self.value >= self.warn_threshold {
            Severity::Warn
        } else {
            Severity::Nominal
        }
    }

    /// Normalized `RGBA` tint for the row: green nominal, yellow warn, red
    /// critical, each fully opaque.
    #[must_use]
    pub fn severity_rgba(&self) -> [f32; 4] {
        match self.severity() {
            Severity::Nominal => [0.0, 1.0, 0.0, 1.0],
            Severity::Warn => [1.0, 1.0, 0.0, 1.0],
            Severity::Critical => [1.0, 0.0, 0.0, 1.0],
        }
    }
}

/// A clamped grid of equal cells the overlay draws its rows and widgets into.
///
/// Column and row counts are clamped to at least one so the grid is always
/// usable and no divide-by-zero can occur in the index reverse lookups. Pixel
/// dimensions are stored verbatim; all products saturate rather than wrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverlayLayout {
    columns: u32,
    rows: u32,
    cell_width_px: u32,
    cell_height_px: u32,
}

impl OverlayLayout {
    /// Builds a layout, clamping `columns` and `rows` up to one.
    #[must_use]
    pub fn new(columns: u32, rows: u32, cell_width_px: u32, cell_height_px: u32) -> Self {
        Self {
            columns: columns.max(1),
            rows: rows.max(1),
            cell_width_px,
            cell_height_px,
        }
    }

    /// The clamped column count (always at least one).
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.columns
    }

    /// The clamped row count (always at least one).
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// The width of a single cell in pixels.
    #[must_use]
    pub fn cell_width_px(&self) -> u32 {
        self.cell_width_px
    }

    /// The height of a single cell in pixels.
    #[must_use]
    pub fn cell_height_px(&self) -> u32 {
        self.cell_height_px
    }

    /// Total number of cells (`columns * rows`), saturating on overflow.
    #[must_use]
    pub fn cell_count(&self) -> u32 {
        self.columns.saturating_mul(self.rows)
    }

    /// Total grid width in pixels (`columns * cell_width_px`), saturating.
    #[must_use]
    pub fn grid_width_px(&self) -> u32 {
        self.columns.saturating_mul(self.cell_width_px)
    }

    /// Total grid height in pixels (`rows * cell_height_px`), saturating.
    #[must_use]
    pub fn grid_height_px(&self) -> u32 {
        self.rows.saturating_mul(self.cell_height_px)
    }

    /// Row index of a linear cell index (`cell / columns`).
    ///
    /// `columns` is always at least one, so this never divides by zero.
    #[must_use]
    pub fn row_index(&self, cell: u32) -> u32 {
        cell / self.columns
    }

    /// Column index of a linear cell index (`cell % columns`).
    ///
    /// `columns` is always at least one, so this never divides by zero.
    #[must_use]
    pub fn col_index(&self, cell: u32) -> u32 {
        cell % self.columns
    }

    /// Total vertex bytes for a quad batch that draws every cell.
    ///
    /// Each cell is one quad = four vertices, and each vertex packs
    /// `pos` (`vec2`) + `uv` (`vec2`) + `color` (`vec4`) using the shared
    /// `std430` strides. Every product saturates, so a degenerate grid can never
    /// wrap to a small allocation.
    #[must_use]
    pub fn vertex_bytes(&self) -> u64 {
        const VERTEX_STRIDE: u64 =
            (VEC2_STRIDE as u64) + (VEC2_STRIDE as u64) + (VEC4_STRIDE as u64);
        const VERTS_PER_CELL: u64 = 4;
        u64::from(self.cell_count())
            .saturating_mul(VERTS_PER_CELL)
            .saturating_mul(VERTEX_STRIDE)
    }
}

/// Assigns `value` to a linear histogram bucket in `[0, bucket_count - 1]`.
///
/// The mapping is `floor((value - min) / (max - min) * bucket_count)`, clamped
/// into range: values at or below `min` land in bucket zero and values at or
/// above `max` land in the last bucket. Any degenerate configuration — an empty
/// range (`min >= max`) or `bucket_count == 0` — collapses to bucket zero. No
/// transcendental math is used; only a divide and [`f32::floor`].
#[must_use]
pub fn histogram_bucket(value: f32, min: f32, max: f32, bucket_count: u32) -> u32 {
    if bucket_count == 0 || min >= max {
        return 0;
    }
    let last = bucket_count - 1;
    if value <= min {
        return 0;
    }
    if value >= max {
        return last;
    }
    let normalized = (value - min) / (max - min);
    // `normalized` is in the open interval (0, 1) here, so the scaled value is in
    // (0, bucket_count) and its floor is a non-negative integer no larger than
    // `last`; the float→int cast is saturating in Rust and cannot be UB, and the
    // explicit `.min(last)` guards the rare case where rounding pushes the
    // product up to `bucket_count`.
    let scaled = normalized * (bucket_count as f32);
    let bucket = scaled.floor() as u32;
    bucket.min(last)
}

/// Builds overlay rows from parallel slices, aligned to the shortest length.
///
/// `values[i]`, `fields[i]`, `warn[i]` and `critical[i]` become one
/// [`OverlayRow`]; iteration stops at the shortest slice so mismatched inputs
/// never index out of bounds and no defaults are invented.
#[must_use]
pub fn build_rows(
    values: &[u64],
    fields: &[OverlayStatField],
    warn: &[u64],
    critical: &[u64],
) -> Vec<OverlayRow> {
    fields
        .iter()
        .copied()
        .zip(values.iter().copied())
        .zip(warn.iter().copied())
        .zip(critical.iter().copied())
        .map(
            |(((field, value), warn_threshold), critical_threshold)| OverlayRow {
                field,
                value,
                warn_threshold,
                critical_threshold,
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_eq(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < CMP_EPS)
    }

    #[test]
    fn all_table_matches_labels_and_length() {
        assert_eq!(OverlayStatField::ALL.len(), 7);
        assert_eq!(OverlayStatField::LiveCount.label(), "Live");
        assert_eq!(OverlayStatField::ActiveEmitters.label(), "Emitters");
        // Every field has a non-empty label.
        assert!(OverlayStatField::ALL.iter().all(|f| !f.label().is_empty()));
    }

    #[test]
    fn severity_promotes_critical_over_warn() {
        let row = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 100,
            warn_threshold: 50,
            critical_threshold: 90,
        };
        assert_eq!(row.severity(), Severity::Critical);
    }

    #[test]
    fn severity_value_exactly_at_warn_threshold_warns() {
        let row = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 50,
            warn_threshold: 50,
            critical_threshold: 100,
        };
        assert_eq!(row.severity(), Severity::Warn);
    }

    #[test]
    fn severity_value_exactly_at_critical_threshold_is_critical() {
        let row = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 100,
            warn_threshold: 50,
            critical_threshold: 100,
        };
        assert_eq!(row.severity(), Severity::Critical);
    }

    #[test]
    fn severity_zero_threshold_disables_level() {
        // Critical disabled: a huge value can only reach warn.
        let row = OverlayRow {
            field: OverlayStatField::OverflowCount,
            value: u64::MAX,
            warn_threshold: 1,
            critical_threshold: 0,
        };
        assert_eq!(row.severity(), Severity::Warn);

        // Both disabled: always nominal.
        let row = OverlayRow {
            field: OverlayStatField::OverflowCount,
            value: u64::MAX,
            warn_threshold: 0,
            critical_threshold: 0,
        };
        assert_eq!(row.severity(), Severity::Nominal);
    }

    #[test]
    fn severity_below_all_thresholds_is_nominal() {
        let row = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 10,
            warn_threshold: 50,
            critical_threshold: 90,
        };
        assert_eq!(row.severity(), Severity::Nominal);
    }

    #[test]
    fn severity_rgba_uses_epsilon_comparison() {
        let nominal = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 0,
            warn_threshold: 10,
            critical_threshold: 20,
        };
        assert!(rgba_eq(nominal.severity_rgba(), [0.0, 1.0, 0.0, 1.0]));

        let warn = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 10,
            warn_threshold: 10,
            critical_threshold: 20,
        };
        assert!(rgba_eq(warn.severity_rgba(), [1.0, 1.0, 0.0, 1.0]));

        let critical = OverlayRow {
            field: OverlayStatField::LiveCount,
            value: 20,
            warn_threshold: 10,
            critical_threshold: 20,
        };
        assert!(rgba_eq(critical.severity_rgba(), [1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn layout_clamps_columns_and_rows_to_one() {
        let layout = OverlayLayout::new(0, 0, 8, 12);
        assert_eq!(layout.columns(), 1);
        assert_eq!(layout.rows(), 1);
        assert_eq!(layout.cell_count(), 1);
    }

    #[test]
    fn layout_grid_dimensions_and_cell_count() {
        let layout = OverlayLayout::new(3, 2, 10, 20);
        assert_eq!(layout.cell_count(), 6);
        assert_eq!(layout.grid_width_px(), 30);
        assert_eq!(layout.grid_height_px(), 40);
    }

    #[test]
    fn layout_cell_index_reverse_lookup() {
        let layout = OverlayLayout::new(4, 3, 8, 8);
        // Cell 6 in a 4-wide grid is row 1, column 2.
        assert_eq!(layout.row_index(6), 1);
        assert_eq!(layout.col_index(6), 2);
        // Cell 0 is the origin.
        assert_eq!(layout.row_index(0), 0);
        assert_eq!(layout.col_index(0), 0);
    }

    #[test]
    fn layout_vertex_bytes_are_expected_multiple() {
        let layout = OverlayLayout::new(2, 2, 8, 8);
        // 4 cells * 4 verts * (8 + 8 + 16) bytes = 4 * 4 * 32 = 512.
        assert_eq!(layout.vertex_bytes(), 512);
    }

    #[test]
    fn layout_vertex_bytes_saturate_on_huge_grid() {
        let layout = OverlayLayout::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        // The cell-count product saturates to u32::MAX; widening to u64 and
        // multiplying by 4 verts * 32 bytes stays well inside u64 and must not
        // wrap (u32::MAX * 128 = 549_755_813_760).
        let expected = u64::from(u32::MAX).saturating_mul(4).saturating_mul(32);
        assert_eq!(layout.vertex_bytes(), expected);
        assert_eq!(layout.vertex_bytes(), 549_755_813_760);
    }

    #[test]
    fn histogram_bucket_interior_is_linear() {
        // [0, 10) into 10 buckets: value 2.5 -> bucket 2, value 9.9 -> bucket 9.
        assert_eq!(histogram_bucket(2.5, 0.0, 10.0, 10), 2);
        assert_eq!(histogram_bucket(9.9, 0.0, 10.0, 10), 9);
        assert_eq!(histogram_bucket(0.5, 0.0, 10.0, 10), 0);
    }

    #[test]
    fn histogram_bucket_below_min_is_zero() {
        assert_eq!(histogram_bucket(-5.0, 0.0, 10.0, 8), 0);
        assert_eq!(histogram_bucket(0.0, 0.0, 10.0, 8), 0);
    }

    #[test]
    fn histogram_bucket_above_max_is_last() {
        assert_eq!(histogram_bucket(11.0, 0.0, 10.0, 8), 7);
        assert_eq!(histogram_bucket(10.0, 0.0, 10.0, 8), 7);
    }

    #[test]
    fn histogram_bucket_degenerate_range_is_zero() {
        // min == max and min > max both collapse to bucket zero.
        assert_eq!(histogram_bucket(5.0, 10.0, 10.0, 8), 0);
        assert_eq!(histogram_bucket(5.0, 10.0, 0.0, 8), 0);
    }

    #[test]
    fn histogram_bucket_zero_buckets_is_zero() {
        assert_eq!(histogram_bucket(5.0, 0.0, 10.0, 0), 0);
    }

    #[test]
    fn histogram_bucket_single_bucket_clamps_to_zero() {
        assert_eq!(histogram_bucket(5.0, 0.0, 10.0, 1), 0);
    }

    #[test]
    fn build_rows_aligns_to_shortest_slice() {
        let values = [1_u64, 2, 3, 4];
        let fields = [
            OverlayStatField::LiveCount,
            OverlayStatField::SpawnedThisFrame,
        ];
        let warn = [10_u64, 20, 30];
        let critical = [100_u64, 200, 300, 400, 500];
        let rows = build_rows(&values, &fields, &warn, &critical);
        // Shortest slice is `fields` with length 2.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].field, OverlayStatField::LiveCount);
        assert_eq!(rows[0].value, 1);
        assert_eq!(rows[0].warn_threshold, 10);
        assert_eq!(rows[0].critical_threshold, 100);
        assert_eq!(rows[1].field, OverlayStatField::SpawnedThisFrame);
        assert_eq!(rows[1].value, 2);
    }

    #[test]
    fn build_rows_empty_input_is_empty() {
        let rows = build_rows(&[], &[], &[], &[]);
        assert!(rows.is_empty());
    }
}
