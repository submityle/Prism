//! GPU-resident column upload / dirty-block census (design §15, §16.6).
//!
//! [`GpuResidentColumn`] is a CPU-side upload planner, not a mapped GPU buffer.
//! It tracks caller-reported dirty blocks since the last consumed plan, not
//! since GPU acknowledgement. Allocation, execution and synchronization are
//! caller-owned. Initial allocation, growth and explicit invalidation copy the
//! whole live extent; otherwise only coalesced dirty spans are planned.
//!
//! This module reads one or more columns and reports, per column, how much the
//! next plan would copy and how well the dirty set coalesces, plus
//! a registry-wide roll-up. It uses only the columns' *immutable* accessors —
//! it never calls `take_upload` / `take_upload_plan`, so running it from a
//! diagnostics system does not consume or disturb pending uploads.
//!
//! # What it surfaces
//!
//! * **pending upload size** —
//!   [`pending_upload_bytes`](ColumnUploadEntry::pending_upload_bytes) is the
//!   byte count the next successful plan will copy, including full replacement;
//!   [`total_pending_upload_bytes`](GpuUploadReport::total_pending_upload_bytes)
//!   is a saturating total for the supplied planners, not a measured GPU cost.
//! * **upload amplification** —
//!   [`upload_amplification_permille`](ColumnUploadEntry::upload_amplification_permille)
//!   is pending bytes over live bytes: `1000` means the entire column is being
//!   re-sent (a dirty-everything frame that defeats incremental upload).
//! * **dirty-set fragmentation** —
//!   [`blocks_per_span_permille`](ColumnUploadEntry::blocks_per_span_permille)
//!   contrasts dirty blocks with the coalesced
//!   [`dirty_span_count`](ColumnUploadEntry::dirty_span_count): low contiguity
//!   means many small scattered copies instead of a few big ones.
//! * **capacity pressure** —
//!   [`needs_reallocation`](ColumnUploadEntry::needs_reallocation) /
//!   [`columns_needing_reallocation`](GpuUploadReport::columns_needing_reallocation)
//!   flag growth or explicit invalidation. Empty invalidations request release
//!   without copying bytes. Estimates do not validate device limits or whether
//!   the planner's checked capacity/version arithmetic can produce a plan.
//!
//! # Determinism (design §14)
//!
//! [`GpuUploadReport::from_columns`] sorts entries by [`ComponentId`] and every
//! tie-break resolves to the lowest id, so a given set of columns yields a
//! byte-identical report regardless of the order they were supplied in.

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::gpu_resident::GpuResidentColumn;

/// One GPU-resident column's incremental-upload accounting for the current
/// frame (design §15 / §16.6).
///
/// All figures are a read-only snapshot of the column's *pending* state: the
/// raw dirty blocks, bytes the next successful plan would copy, live extent,
/// and capacity recorded by CPU planning (not acknowledged GPU capacity).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColumnUploadEntry {
    /// The component whose GPU-resident column this describes.
    pub component: ComponentId,
    /// Live logical element count of the column.
    pub element_count: usize,
    /// Number of blocks covering the live extent.
    pub block_count: usize,
    /// Blocks currently marked dirty (the raw, uncoalesced dirty count).
    pub dirty_block_count: usize,
    /// Coalesced raw dirty spans. Replacement may instead use one full span;
    /// zero-byte spans need not issue a GPU copy.
    pub dirty_span_count: usize,
    /// Bytes the next successful plan copies: full live extent on replacement,
    /// otherwise coalesced dirty bytes. Zero for an empty release or zero stride.
    pub pending_upload_bytes: usize,
    /// Live byte extent of the column (`element_count * element_stride`).
    pub live_bytes: usize,
    /// Block capacity recorded by CPU planning, not GPU acknowledgement.
    pub gpu_capacity_blocks: usize,
    /// Byte capacity recorded by CPU planning, not GPU acknowledgement.
    pub gpu_capacity_bytes: usize,
    /// Whether growth or invalidation requires replacement/full upload, or an
    /// empty release. This is not a device allocation success guarantee.
    pub needs_reallocation: bool,
    /// The column's upload version (bumped once per non-empty upload taken).
    pub upload_version: u64,
    /// Identity recorded by the last allocation plan, not GPU acknowledgement.
    pub buffer_generation: u64,
}

impl ColumnUploadEntry {
    /// Account one GPU-resident column into an entry, without disturbing its
    /// pending-upload state.
    pub fn from_column(component: ComponentId, column: &GpuResidentColumn) -> Self {
        let spans = column.coalesced_dirty_spans();
        let dirty_span_count = spans.len();
        let live_bytes = column.len() * column.element_stride();
        let needs_reallocation = column.needs_reallocation();
        let pending_upload_bytes = if needs_reallocation {
            live_bytes
        } else {
            spans.map(|d| d.byte_len).sum()
        };
        Self {
            component,
            element_count: column.len(),
            block_count: column.block_count(),
            dirty_block_count: column.dirty_block_count(),
            dirty_span_count,
            pending_upload_bytes,
            live_bytes,
            gpu_capacity_blocks: column.gpu_capacity_blocks(),
            gpu_capacity_bytes: column.gpu_capacity_bytes(),
            needs_reallocation,
            upload_version: column.upload_version(),
            buffer_generation: column.buffer_generation(),
        }
    }

    /// Whether no raw dirty blocks are pending. A clean column may still need
    /// replacement/full upload or an empty release; see [`Self::has_pending_plan`].
    #[inline]
    pub const fn is_clean(&self) -> bool {
        self.dirty_block_count == 0
    }

    /// Whether a successful next plan would contain spans or replacement/release.
    /// Includes zero-stride spans and zero-byte releases, not only byte copies.
    #[inline]
    pub const fn has_pending_plan(&self) -> bool {
        self.needs_reallocation || !self.is_clean()
    }

    /// Whether every live block is dirty — a worst-case frame where the
    /// incremental scheme degenerates into a full re-upload.
    #[inline]
    pub const fn is_fully_dirty(&self) -> bool {
        self.block_count != 0 && self.dirty_block_count == self.block_count
    }

    /// Fraction of blocks that are dirty, in per-mille (`1000` = all dirty).
    /// Returns `0` for an empty column.
    #[inline]
    pub const fn dirty_permille(&self) -> u64 {
        permille(self.dirty_block_count as u128, self.block_count as u128)
    }

    /// Pending upload bytes over live bytes, in per-mille (`1000` = the entire
    /// column is being re-sent). The key signal that incremental upload is
    /// paying off: small is good. Returns `0` for an empty column.
    #[inline]
    pub const fn upload_amplification_permille(&self) -> u64 {
        permille(self.pending_upload_bytes as u128, self.live_bytes as u128)
    }

    /// Dirty blocks per coalesced span, in per-mille (`1000` = one block per
    /// span = maximally scattered; higher = more contiguous, fewer copies).
    /// Returns `0` when nothing is dirty.
    #[inline]
    pub const fn blocks_per_span_permille(&self) -> u64 {
        permille(
            self.dirty_block_count as u128,
            self.dirty_span_count as u128,
        )
    }

    /// GPU buffer blocks allocated beyond the live extent (grow-only slack a
    /// persistent buffer keeps to amortise reallocations). `0` once the live
    /// extent meets or exceeds capacity (see
    /// [`needs_reallocation`](Self::needs_reallocation)).
    #[inline]
    pub const fn capacity_slack_blocks(&self) -> usize {
        self.gpu_capacity_blocks.saturating_sub(self.block_count)
    }

    /// Live blocks over GPU capacity blocks, in per-mille (`1000` = buffer
    /// exactly full; `> 1000` means the live extent has outgrown it). Returns
    /// `0` before any GPU capacity has been established.
    #[inline]
    pub const fn capacity_utilization_permille(&self) -> u64 {
        permille(self.block_count as u128, self.gpu_capacity_blocks as u128)
    }
}

// Widen before scaling; saturate only the returned ratio. The fallback also
// handles wide aggregate totals without overflowing numerator * 1000.
const fn permille(numerator: u128, denominator: u128) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let value = if let Some(scaled) = numerator.checked_mul(1000) {
        scaled / denominator
    } else {
        let whole = numerator / denominator;
        if whole > u64::MAX as u128 / 1000 {
            return u64::MAX;
        }
        let base = numerator % denominator;
        let mut remainder = 0;
        let mut fraction = 0;
        let mut bit = 10;
        // Long multiplication/division of base * 1000, keeping every
        // intermediate remainder below denominator. Neither addition wraps.
        while bit != 0 {
            bit -= 1;
            fraction *= 2;
            if remainder >= denominator - remainder {
                remainder -= denominator - remainder;
                fraction += 1;
            } else {
                remainder *= 2;
            }
            if (1000 >> bit) & 1 != 0 {
                if remainder >= denominator - base {
                    remainder -= denominator - base;
                    fraction += 1;
                } else {
                    remainder += base;
                }
            }
        }
        whole * 1000 + fraction
    };
    if value > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}

/// Registry-wide GPU-resident upload census (design §15 / §16.6).
///
/// Built from an explicit list of columns via
/// [`from_columns`](Self::from_columns). The entries are sorted by
/// [`ComponentId`]; the roll-up accessors aggregate the whole supplied set.
/// `usize` totals saturate at `usize::MAX`; aggregate ratios use unsaturated
/// wide totals so overflowing a display total does not distort the ratio.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GpuUploadReport {
    entries: Vec<ColumnUploadEntry>,
}

impl GpuUploadReport {
    /// Account a set of GPU-resident columns, keyed by component.
    ///
    /// The caller supplies the columns to inspect — the kernel's
    /// [`GpuResidentColumns`](crate::gpu_resident::GpuResidentColumns) registry
    /// hands them out by [`ComponentId`] — so this report stays agnostic to how
    /// the registry is iterated. Entries are sorted ascending by component id
    /// for a deterministic report.
    pub fn from_columns(columns: &[(ComponentId, &GpuResidentColumn)]) -> Self {
        let mut entries: Vec<ColumnUploadEntry> = columns
            .iter()
            .map(|(id, column)| ColumnUploadEntry::from_column(*id, column))
            .collect();
        entries.sort_unstable_by_key(|e| e.component);
        Self { entries }
    }

    /// The per-column entries, ascending by component id.
    #[inline]
    pub fn entries(&self) -> &[ColumnUploadEntry] {
        &self.entries
    }

    /// Number of GPU-resident columns in the census.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.entries.len()
    }

    /// Whether no columns were supplied.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes the next successful plans copy, saturating at `usize::MAX`.
    #[inline]
    pub fn total_pending_upload_bytes(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |sum, e| sum.saturating_add(e.pending_upload_bytes))
    }

    /// Total live byte extent, saturating at `usize::MAX`.
    #[inline]
    pub fn total_live_bytes(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |sum, e| sum.saturating_add(e.live_bytes))
    }

    /// Total dirty blocks, saturating at `usize::MAX`.
    #[inline]
    pub fn total_dirty_blocks(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |sum, e| sum.saturating_add(e.dirty_block_count))
    }

    /// Total live blocks, saturating at `usize::MAX`.
    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.entries
            .iter()
            .fold(0usize, |sum, e| sum.saturating_add(e.block_count))
    }

    /// Number of columns with at least one dirty block this frame.
    #[inline]
    pub fn dirty_column_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.is_clean()).count()
    }

    /// Number of columns with no raw dirty blocks. They may still need
    /// replacement/full upload or release.
    #[inline]
    pub fn clean_column_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_clean()).count()
    }

    /// Number of columns with spans or replacement/release pending, including
    /// zero-byte work. A successful plan from each would be nonempty.
    #[inline]
    pub fn pending_plan_column_count(&self) -> usize {
        self.entries.iter().filter(|e| e.has_pending_plan()).count()
    }

    /// Number of columns requiring replacement or release on their next plan.
    #[inline]
    pub fn columns_needing_reallocation(&self) -> usize {
        self.entries.iter().filter(|e| e.needs_reallocation).count()
    }

    /// Aggregate dirty fraction over the whole set, in per-mille. Returns `0`
    /// when there are no live blocks.
    #[inline]
    pub fn aggregate_dirty_permille(&self) -> u64 {
        permille(
            self.entries
                .iter()
                .map(|e| e.dirty_block_count as u128)
                .sum(),
            self.entries.iter().map(|e| e.block_count as u128).sum(),
        )
    }

    /// Aggregate upload amplification over the whole set, in per-mille: total
    /// pending bytes over total live bytes (`1000` = re-sending everything).
    /// Returns `0` when there are no live bytes.
    #[inline]
    pub fn aggregate_upload_amplification_permille(&self) -> u64 {
        permille(
            self.entries
                .iter()
                .map(|e| e.pending_upload_bytes as u128)
                .sum(),
            self.entries.iter().map(|e| e.live_bytes as u128).sum(),
        )
    }

    /// The column with the largest pending upload (ties resolve to the lowest
    /// component id), or `None` when the census is empty.
    pub fn hottest_upload(&self) -> Option<ColumnUploadEntry> {
        // Entries are already ascending by id, so a strictly-greater test keeps
        // the lowest id on a tie.
        self.entries.iter().copied().reduce(|best, e| {
            if e.pending_upload_bytes > best.pending_upload_bytes {
                e
            } else {
                best
            }
        })
    }

    /// The entry for `component`, or `None` if it is not in the census.
    #[inline]
    pub fn column(&self, component: ComponentId) -> Option<ColumnUploadEntry> {
        self.entries
            .binary_search_by(|e| e.component.cmp(&component))
            .ok()
            .map(|i| self.entries[i])
    }

    /// Whether `component` is in the census.
    #[inline]
    pub fn contains(&self, component: ComponentId) -> bool {
        self.entries
            .binary_search_by(|e| e.component.cmp(&component))
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(i: u32) -> ComponentId {
        ComponentId::new(i)
    }

    /// A column with capacity established (buffer_generation == 1) and the
    /// given elements, then clean. `stride` bytes/element, `block_len`
    /// elements/block.
    fn resident(stride: usize, block_len: usize, elements: usize) -> GpuResidentColumn {
        let mut c = GpuResidentColumn::new(stride, block_len);
        c.ensure_len(elements);
        // Establish GPU capacity and clear dirties (full initial upload).
        let _ = c.take_upload_plan();
        c
    }

    #[test]
    fn empty_report_rolls_up_to_zero() {
        let report = GpuUploadReport::from_columns(&[]);
        assert!(report.is_empty());
        assert_eq!(report.column_count(), 0);
        assert_eq!(report.total_pending_upload_bytes(), 0);
        assert_eq!(report.total_live_bytes(), 0);
        assert_eq!(report.total_dirty_blocks(), 0);
        assert_eq!(report.total_blocks(), 0);
        assert_eq!(report.dirty_column_count(), 0);
        assert_eq!(report.clean_column_count(), 0);
        assert_eq!(report.columns_needing_reallocation(), 0);
        assert_eq!(report.pending_plan_column_count(), 0);
        assert_eq!(report.aggregate_dirty_permille(), 0);
        assert_eq!(report.aggregate_upload_amplification_permille(), 0);
        assert_eq!(report.hottest_upload(), None);
        assert_eq!(report.column(cid(0)), None);
        assert!(!report.contains(cid(0)));
    }

    #[test]
    fn single_dirty_column_accounts_pending_upload() {
        // stride 16, block_len 4 => block 64 bytes. 10 elements => 3 blocks,
        // live 160 bytes.
        let mut c = resident(16, 4, 10);
        c.mark_dirty(0); // block 0
        c.mark_dirty(5); // block 1 (adjacent to 0 => one coalesced span)
        let report = GpuUploadReport::from_columns(&[(cid(7), &c)]);

        let e = report.column(cid(7)).unwrap();
        assert_eq!(e.element_count, 10);
        assert_eq!(e.block_count, 3);
        assert_eq!(e.dirty_block_count, 2);
        assert_eq!(e.dirty_span_count, 1);
        // blocks 0..=1 => [0,128), clamped to live 160 => 128.
        assert_eq!(e.pending_upload_bytes, 128);
        assert_eq!(e.live_bytes, 160);
        assert!(!e.is_clean());
        assert!(!e.is_fully_dirty());
        assert_eq!(e.dirty_permille(), 666); // 2/3
        assert_eq!(e.upload_amplification_permille(), 800); // 128/160
        assert_eq!(e.blocks_per_span_permille(), 2000); // 2 blocks / 1 span
        assert_eq!(e.buffer_generation, 1);
        // capacity grew 0 -> 4 blocks (doubling to cover 3).
        assert_eq!(e.gpu_capacity_blocks, 4);
        assert!(!e.needs_reallocation);
        assert_eq!(e.capacity_slack_blocks(), 1);
        assert_eq!(e.capacity_utilization_permille(), 750); // 3/4

        assert_eq!(report.total_pending_upload_bytes(), 128);
        assert_eq!(report.dirty_column_count(), 1);
        assert_eq!(report.hottest_upload(), Some(e));
    }

    #[test]
    fn clean_column_has_nothing_pending() {
        let c = resident(8, 2, 4); // 2 blocks, clean after initial upload
        let report = GpuUploadReport::from_columns(&[(cid(3), &c)]);
        let e = report.column(cid(3)).unwrap();
        assert!(e.is_clean());
        assert_eq!(e.dirty_block_count, 0);
        assert_eq!(e.dirty_span_count, 0);
        assert_eq!(e.pending_upload_bytes, 0);
        assert_eq!(e.dirty_permille(), 0);
        assert_eq!(e.blocks_per_span_permille(), 0);
        assert_eq!(report.clean_column_count(), 1);
        assert_eq!(report.dirty_column_count(), 0);
        assert!(!e.has_pending_plan());
        assert_eq!(report.pending_plan_column_count(), 0);
    }

    #[test]
    fn outgrown_column_needs_reallocation() {
        let mut c = resident(4, 1, 2); // cap grows to 2 blocks
        assert!(!c.needs_reallocation());
        c.ensure_len(5); // 5 blocks > capacity 2
        c.mark_dirty(4);
        let report = GpuUploadReport::from_columns(&[(cid(1), &c)]);
        let e = report.column(cid(1)).unwrap();
        assert!(e.needs_reallocation);
        assert_eq!(e.block_count, 5);
        assert_eq!(e.gpu_capacity_blocks, 2);
        assert_eq!(e.capacity_slack_blocks(), 0);
        // utilization > 1000 since live extent outgrew the buffer: 5/2.
        assert_eq!(e.capacity_utilization_permille(), 2500);
        assert_eq!(report.columns_needing_reallocation(), 1);
        // The replacement must copy the complete live column, not just block 4.
        assert_eq!(e.pending_upload_bytes, 20);
        assert_eq!(e.upload_amplification_permille(), 1000);
    }

    #[test]
    fn clean_initial_growth_and_invalidation_predict_full_upload_without_consuming() {
        let mut column = GpuResidentColumn::new(4, 2);
        column.ensure_len(3);
        for phase in 0..3 {
            if phase == 1 {
                column.ensure_len(9);
            } else if phase == 2 {
                column.force_reallocation();
            }
            assert!(column.is_clean());
            let before = (column.upload_version(), column.buffer_generation());
            let report = GpuUploadReport::from_columns(&[(cid(1), &column)]);
            assert_eq!(report, GpuUploadReport::from_columns(&[(cid(1), &column)]));
            let entry = report.column(cid(1)).unwrap();
            assert!(entry.is_clean());
            assert!(entry.has_pending_plan());
            assert_eq!(report.pending_plan_column_count(), 1);
            assert_eq!(entry.pending_upload_bytes, column.len() * 4);
            assert_eq!(entry.upload_amplification_permille(), 1000);
            assert_eq!(
                before,
                (column.upload_version(), column.buffer_generation())
            );
            let plan = column.take_upload_plan();
            assert!(plan.reallocate.is_some());
            assert_eq!(plan.spans.len(), 1);
            assert_eq!(entry.pending_upload_bytes, plan.spans[0].byte_len);
            assert!(!ColumnUploadEntry::from_column(cid(1), &column).has_pending_plan());
        }
    }

    #[test]
    fn empty_release_is_pending_even_without_bytes_or_dirty_blocks() {
        let mut column = resident(8, 2, 7);
        column.mark_dirty(6);
        column.truncate(0);
        column.force_reallocation();
        let version = column.upload_version();
        let generation = column.buffer_generation();
        let report = GpuUploadReport::from_columns(&[(cid(1), &column)]);
        let entry = report.column(cid(1)).unwrap();
        assert!(entry.is_clean());
        assert!(entry.has_pending_plan());
        assert_eq!(report.pending_plan_column_count(), 1);
        assert_eq!(report.total_pending_upload_bytes(), 0);
        assert_eq!(entry.dirty_permille(), 0);
        assert_eq!(entry.upload_amplification_permille(), 0);
        let plan = column.take_upload_plan();
        let release = plan.reallocate.unwrap();
        assert_eq!(release.capacity_blocks, 0);
        assert_eq!(release.capacity_bytes, 0);
        assert_eq!(release.buffer_generation, generation + 1);
        assert!(plan.spans.is_empty());
        assert_eq!(column.upload_version(), version);
        assert!(!ColumnUploadEntry::from_column(cid(1), &column).has_pending_plan());
    }

    #[test]
    fn zero_stride_growth_dirty_spans_and_release_remain_visible() {
        let mut column = GpuResidentColumn::new(0, 4);
        column.ensure_len(5);
        for phase in 0..3 {
            if phase == 1 {
                column.mark_dirty(4);
            } else if phase == 2 {
                column.truncate(0);
                column.force_reallocation();
            }
            let entry = ColumnUploadEntry::from_column(cid(1), &column);
            assert!(entry.has_pending_plan());
            assert_eq!(entry.pending_upload_bytes, 0);
            assert_eq!(entry.live_bytes, 0);
            assert_eq!(entry.upload_amplification_permille(), 0);
            let plan = column.take_upload_plan();
            assert!(!plan.is_empty());
            assert!(plan.spans.iter().all(|span| span.byte_len == 0));
            assert_eq!(plan.spans.is_empty(), phase == 2);
            assert!(!ColumnUploadEntry::from_column(cid(1), &column).has_pending_plan());
        }
    }

    #[test]
    fn census_does_not_consume_incremental_spans_or_versions() {
        let mut column = resident(3, 2, 9);
        column.mark_dirty(0);
        column.mark_dirty(8);
        let spans = column.coalesced_dirty_blocks();
        let version = column.upload_version();
        let generation = column.buffer_generation();
        let entry = ColumnUploadEntry::from_column(cid(1), &column);
        assert_eq!(entry.pending_upload_bytes, 9);
        assert_eq!(entry.dirty_span_count, 2);
        assert_eq!(column.coalesced_dirty_blocks(), spans);
        assert_eq!(column.upload_version(), version);
        assert_eq!(column.buffer_generation(), generation);
        let plan = column.take_upload_plan();
        assert_eq!(plan.reallocate, None);
        assert_eq!(plan.spans, spans);
        assert_eq!(column.upload_version(), version + 1);
    }

    #[test]
    fn huge_logical_columns_saturate_display_totals_without_distorting_ratios() {
        // Planners allocate only dirty intervals, not the logical byte extent.
        let rows = usize::MAX / 2;
        let mut dirty = resident(1, 1, rows);
        let clean = resident(1, 1, rows);
        dirty.mark_range(0, rows);
        let report =
            GpuUploadReport::from_columns(&[(cid(1), &dirty), (cid(2), &clean), (cid(3), &clean)]);
        assert_eq!(report.total_live_bytes(), usize::MAX);
        assert_eq!(report.total_blocks(), usize::MAX);
        assert_eq!(report.total_dirty_blocks(), rows);
        assert_eq!(report.total_pending_upload_bytes(), rows);
        assert_eq!(report.aggregate_dirty_permille(), 333);
        assert_eq!(report.aggregate_upload_amplification_permille(), 333);
        let entry = report.column(cid(1)).unwrap();
        assert_eq!(entry.dirty_permille(), 1000);
        assert_eq!(entry.upload_amplification_permille(), 1000);
        assert_eq!(
            entry.blocks_per_span_permille(),
            u64::try_from(rows as u128 * 1000).unwrap_or(u64::MAX)
        );
    }

    #[test]
    fn wide_permille_handles_fractional_overflow_and_saturation_boundaries() {
        assert_eq!(permille(u128::MAX, 0), 0);
        assert_eq!(permille(u128::MAX, u128::MAX), 1000);
        assert_eq!(permille(u128::MAX - 1, u128::MAX), 999);
        assert_eq!(permille(u128::MAX / 3, u128::MAX), 333);
        assert_eq!(permille(u128::MAX, u128::MAX / 3), 3000);
        assert_eq!(permille(u128::MAX, 1), u64::MAX);
        assert_eq!(permille(u64::MAX as u128, 1000), u64::MAX);
        assert_eq!(permille(u64::MAX as u128 + 1, 1000), u64::MAX);
        for denominator in 1..=257u128 {
            for numerator in [0, 1, denominator / 2, denominator, denominator * 3 - 1] {
                let scale = u128::MAX / (denominator * 3);
                assert_eq!(
                    permille(numerator * scale, denominator * scale),
                    (numerator * 1000 / denominator) as u64
                );
            }
        }
    }

    #[test]
    fn fully_dirty_column_is_flagged() {
        let mut c = resident(4, 2, 4); // 2 blocks
        c.mark_range(0, 4); // dirty every element => every block
        let report = GpuUploadReport::from_columns(&[(cid(0), &c)]);
        let e = report.column(cid(0)).unwrap();
        assert!(e.is_fully_dirty());
        assert_eq!(e.dirty_permille(), 1000);
        // whole live extent re-sent.
        assert_eq!(e.pending_upload_bytes, e.live_bytes);
        assert_eq!(e.upload_amplification_permille(), 1000);
    }

    #[test]
    fn multi_column_rollup_and_hottest() {
        // Supplied out of id order to exercise the sort.
        let mut big = resident(16, 4, 10); // see single-dirty test: 128 pending
        big.mark_dirty(0);
        big.mark_dirty(5);
        let clean = resident(8, 2, 4); // nothing pending

        let mut small = resident(8, 2, 2); // 1 block, 16 bytes
        small.mark_dirty(0); // whole single block dirty => 16 bytes
        let report =
            GpuUploadReport::from_columns(&[(cid(9), &big), (cid(2), &clean), (cid(5), &small)]);

        // Entries sorted ascending by id.
        let ids: Vec<_> = report.entries().iter().map(|e| e.component).collect();
        assert_eq!(ids, alloc::vec![cid(2), cid(5), cid(9)]);

        assert_eq!(report.column_count(), 3);
        assert_eq!(report.total_pending_upload_bytes(), 128 + 16);
        assert_eq!(report.dirty_column_count(), 2);
        assert_eq!(report.clean_column_count(), 1);
        // big is the hottest upload.
        assert_eq!(report.hottest_upload().unwrap().component, cid(9));
        assert!(report.contains(cid(5)));
        assert!(!report.contains(cid(4)));
    }

    #[test]
    fn report_is_order_independent() {
        let mut a = resident(8, 2, 4);
        a.mark_dirty(0);
        let b = resident(8, 2, 2);
        let forward = GpuUploadReport::from_columns(&[(cid(1), &a), (cid(2), &b)]);
        let reverse = GpuUploadReport::from_columns(&[(cid(2), &b), (cid(1), &a)]);
        assert_eq!(forward, reverse);
    }
}
