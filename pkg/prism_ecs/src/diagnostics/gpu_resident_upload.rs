//! GPU-resident column upload / dirty-block census (design §15, §16.6).
//!
//! A GPU-resident component column
//! ([`GpuResidentColumn`](crate::gpu_resident::GpuResidentColumn)) keeps its
//! element data persistently mapped on the GPU and tracks which fixed-size
//! blocks changed since the last acknowledged upload, so each frame only the
//! dirty blocks are re-sent (the Horizon / Insomniac incremental-upload shape,
//! design §15 脏块增量上传). The whole value of that scheme is that upload cost
//! tracks *change volume*, not column size — the same headline as the rest of
//! the kernel (design §1 成本 ∝ 变化量).
//!
//! This module reads one or more columns and reports, per column, how much the
//! next incremental upload will cost and how well the dirty set coalesces, plus
//! a registry-wide roll-up. It uses only the columns' *immutable* accessors —
//! it never calls `take_upload` / `take_upload_plan`, so running it from a
//! diagnostics system does not consume or disturb pending uploads.
//!
//! # What it surfaces
//!
//! * **pending upload size** —
//!   [`pending_upload_bytes`](ColumnUploadEntry::pending_upload_bytes) is the
//!   minimal (coalesced) byte count the next incremental upload will copy;
//!   [`total_pending_upload_bytes`](GpuUploadReport::total_pending_upload_bytes)
//!   is the frame's whole GPU-upload budget.
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
//!   flag columns whose live extent has outgrown their GPU buffer, forcing a
//!   full re-upload next frame.
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
/// dirty blocks that the next incremental upload would copy, the live extent,
/// and the renderer's acknowledged GPU buffer capacity.
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
    /// Coalesced upload spans — adjacent dirty blocks merged — i.e. the number
    /// of separate GPU copy operations the next incremental upload issues.
    pub dirty_span_count: usize,
    /// Minimal (coalesced) bytes the next incremental upload copies, clamped to
    /// the live extent.
    pub pending_upload_bytes: usize,
    /// Live byte extent of the column (`element_count * element_stride`).
    pub live_bytes: usize,
    /// Blocks the renderer currently has allocated on the GPU for this column.
    pub gpu_capacity_blocks: usize,
    /// Bytes the renderer currently has allocated on the GPU for this column.
    pub gpu_capacity_bytes: usize,
    /// Whether the live extent has outgrown the GPU buffer, so the next upload
    /// plan must reallocate and re-upload the whole live extent.
    pub needs_reallocation: bool,
    /// The column's upload version (bumped once per non-empty upload taken).
    pub upload_version: u64,
    /// Identity of the column's current GPU buffer allocation.
    pub buffer_generation: u64,
}

impl ColumnUploadEntry {
    /// Account one GPU-resident column into an entry, without disturbing its
    /// pending-upload state.
    pub fn from_column(component: ComponentId, column: &GpuResidentColumn) -> Self {
        let pending_upload_bytes = column
            .coalesced_dirty_blocks()
            .iter()
            .map(|d| d.byte_len)
            .sum();
        Self {
            component,
            element_count: column.len(),
            block_count: column.block_count(),
            dirty_block_count: column.dirty_block_count(),
            dirty_span_count: column.coalesced_dirty_blocks().len(),
            pending_upload_bytes,
            live_bytes: column.len() * column.element_stride(),
            gpu_capacity_blocks: column.gpu_capacity_blocks(),
            gpu_capacity_bytes: column.gpu_capacity_bytes(),
            needs_reallocation: column.needs_reallocation(),
            upload_version: column.upload_version(),
            buffer_generation: column.buffer_generation(),
        }
    }

    /// Whether this column has no pending dirty blocks (nothing to upload).
    #[inline]
    pub const fn is_clean(&self) -> bool {
        self.dirty_block_count == 0
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
        if self.block_count == 0 {
            return 0;
        }
        self.dirty_block_count as u64 * 1000 / self.block_count as u64
    }

    /// Pending upload bytes over live bytes, in per-mille (`1000` = the entire
    /// column is being re-sent). The key signal that incremental upload is
    /// paying off: small is good. Returns `0` for an empty column.
    #[inline]
    pub const fn upload_amplification_permille(&self) -> u64 {
        if self.live_bytes == 0 {
            return 0;
        }
        self.pending_upload_bytes as u64 * 1000 / self.live_bytes as u64
    }

    /// Dirty blocks per coalesced span, in per-mille (`1000` = one block per
    /// span = maximally scattered; higher = more contiguous, fewer copies).
    /// Returns `0` when nothing is dirty.
    #[inline]
    pub const fn blocks_per_span_permille(&self) -> u64 {
        if self.dirty_span_count == 0 {
            return 0;
        }
        self.dirty_block_count as u64 * 1000 / self.dirty_span_count as u64
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
        if self.gpu_capacity_blocks == 0 {
            return 0;
        }
        self.block_count as u64 * 1000 / self.gpu_capacity_blocks as u64
    }
}

/// Registry-wide GPU-resident upload census (design §15 / §16.6).
///
/// Built from an explicit list of columns via
/// [`from_columns`](Self::from_columns). The entries are sorted by
/// [`ComponentId`]; the roll-up accessors aggregate the whole supplied set.
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

    /// Total minimal (coalesced) bytes the next incremental upload copies
    /// across every column — the frame's whole GPU-upload budget.
    #[inline]
    pub fn total_pending_upload_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.pending_upload_bytes).sum()
    }

    /// Total live byte extent across every column.
    #[inline]
    pub fn total_live_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.live_bytes).sum()
    }

    /// Total dirty blocks pending across every column.
    #[inline]
    pub fn total_dirty_blocks(&self) -> usize {
        self.entries.iter().map(|e| e.dirty_block_count).sum()
    }

    /// Total live blocks across every column.
    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.entries.iter().map(|e| e.block_count).sum()
    }

    /// Number of columns with at least one dirty block this frame.
    #[inline]
    pub fn dirty_column_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.is_clean()).count()
    }

    /// Number of columns with nothing pending (zero-cost this frame).
    #[inline]
    pub fn clean_column_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_clean()).count()
    }

    /// Number of columns whose live extent has outgrown their GPU buffer and
    /// will force a full re-upload on their next upload plan.
    #[inline]
    pub fn columns_needing_reallocation(&self) -> usize {
        self.entries.iter().filter(|e| e.needs_reallocation).count()
    }

    /// Aggregate dirty fraction over the whole set, in per-mille. Returns `0`
    /// when there are no live blocks.
    #[inline]
    pub fn aggregate_dirty_permille(&self) -> u64 {
        let blocks = self.total_blocks();
        if blocks == 0 {
            return 0;
        }
        self.total_dirty_blocks() as u64 * 1000 / blocks as u64
    }

    /// Aggregate upload amplification over the whole set, in per-mille: total
    /// pending bytes over total live bytes (`1000` = re-sending everything).
    /// Returns `0` when there are no live bytes.
    #[inline]
    pub fn aggregate_upload_amplification_permille(&self) -> u64 {
        let live = self.total_live_bytes();
        if live == 0 {
            return 0;
        }
        self.total_pending_upload_bytes() as u64 * 1000 / live as u64
    }

    /// The column with the largest pending upload (ties resolve to the lowest
    /// component id), or `None` when the census is empty.
    pub fn hottest_upload(&self) -> Option<ColumnUploadEntry> {
        // Entries are already ascending by id, so a strictly-greater test keeps
        // the lowest id on a tie.
        self.entries
            .iter()
            .copied()
            .reduce(|best, e| {
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
        let report = GpuUploadReport::from_columns(&[
            (cid(9), &big),
            (cid(2), &clean),
            (cid(5), &small),
        ]);

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
