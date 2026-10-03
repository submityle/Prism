//! GPU-resident columns + dirty-block incremental upload (design §15).
//!
//! The engine can nominate certain component columns as *GPU-resident*: their
//! bytes live in a persistently-mapped GPU buffer that the renderer reads
//! directly. To keep the CPU→GPU bandwidth minimal, we never re-upload the
//! whole column. Instead each column is partitioned into fixed-size **blocks**,
//! and we track exactly which blocks changed since the last acknowledged
//! upload. Each frame the renderer calls [`GpuResidentColumn::take_upload`] to
//! obtain a minimal, coalesced list of contiguous byte spans to copy.
//!
//! This module is the CPU-side bookkeeping half of that scheme. The actual
//! buffer mapping and `copy` commands live in the render crates; here we only
//! compute *which* bytes are dirty. The unit tests therefore verify that the
//! produced [`DirtyBlock`] spans are correct and minimal.
//!
//! See [`GpuResidentColumns`] for the per-[`ComponentId`](crate::component::ComponentId)
//! registry.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;

/// One contiguous span of a GPU-resident column that needs to be uploaded.
///
/// A span always starts at a block boundary (`block_index * block_len *
/// stride`) but, after coalescing, may cover several adjacent blocks. The
/// `byte_len` is clamped so a trailing partial block never reports bytes past
/// the live `len * stride` extent of the column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DirtyBlock {
    /// Index of the first block covered by this span.
    pub block_index: usize,
    /// Byte offset of the span from the start of the column's buffer.
    pub byte_offset: usize,
    /// Number of bytes to upload, clamped to the live extent of the column.
    pub byte_len: usize,
}

/// A persistently-mapped column of fixed-stride elements, partitioned into
/// fixed-size blocks, that tracks which blocks changed since the last
/// acknowledged upload.
///
/// Elements are logically contiguous; block `b` covers element indices
/// `[b * block_len, (b + 1) * block_len)`. Marking any element in a block
/// dirties the whole block, which is the unit of upload granularity.
pub struct GpuResidentColumn {
    /// Size of one element in bytes.
    element_stride_bytes: usize,
    /// Number of elements per block (>= 1).
    block_len_elements: usize,
    /// Current logical element count.
    len: usize,
    /// Per-block dirty bitmap; `dirty[b]` is true iff block `b` changed since
    /// the last [`take_upload`](Self::take_upload). Length tracks
    /// [`block_count`](Self::block_count).
    dirty: Vec<bool>,
    /// Monotonically increasing counter bumped each time a non-empty upload is
    /// taken.
    upload_version: u64,
}

impl GpuResidentColumn {
    /// Create an empty GPU-resident column.
    ///
    /// # Contract
    /// `block_len_elements` must be `>= 1`; this method panics otherwise. A
    /// zero element stride is permitted (e.g. zero-sized components), in which
    /// case every reported span has `byte_len == 0`.
    pub fn new(element_stride_bytes: usize, block_len_elements: usize) -> Self {
        assert!(
            block_len_elements >= 1,
            "block_len_elements must be >= 1, got {block_len_elements}"
        );
        Self {
            element_stride_bytes,
            block_len_elements,
            len: 0,
            dirty: Vec::new(),
            upload_version: 0,
        }
    }

    /// The current logical element count of the column.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the column currently holds zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of one element in bytes.
    #[inline]
    pub fn element_stride(&self) -> usize {
        self.element_stride_bytes
    }

    /// Number of elements per block.
    #[inline]
    pub fn block_len(&self) -> usize {
        self.block_len_elements
    }

    /// Number of blocks needed to cover the current length (rounded up).
    #[inline]
    pub fn block_count(&self) -> usize {
        self.len.div_ceil(self.block_len_elements)
    }

    /// Byte length of a single block in the column's buffer.
    #[inline]
    fn block_byte_len(&self) -> usize {
        self.block_len_elements * self.element_stride_bytes
    }

    /// Total live byte extent of the column (`len * stride`).
    #[inline]
    fn live_byte_len(&self) -> usize {
        self.len * self.element_stride_bytes
    }

    /// Resize the per-block dirty bitmap to match the current
    /// [`block_count`](Self::block_count). Newly added blocks start clean.
    #[inline]
    fn resize_bitmap(&mut self) {
        let needed = self.block_count();
        if self.dirty.len() < needed {
            self.dirty.resize(needed, false);
        } else if self.dirty.len() > needed {
            self.dirty.truncate(needed);
        }
    }

    /// Grow the logical length to at least `element_count` elements.
    ///
    /// Shrinking is not performed: if `element_count` is less than the current
    /// length this is a no-op. Any blocks newly brought into existence start
    /// clean — callers must [`mark_dirty`](Self::mark_dirty) the elements they
    /// actually wrote.
    pub fn ensure_len(&mut self, element_count: usize) {
        if element_count > self.len {
            self.len = element_count;
            self.resize_bitmap();
        }
    }

    /// Mark the block containing `element_index` as dirty, growing the logical
    /// length to include that element if necessary.
    pub fn mark_dirty(&mut self, element_index: usize) {
        self.ensure_len(element_index + 1);
        let block = element_index / self.block_len_elements;
        // `block < block_count()` holds because `ensure_len` grew the bitmap.
        self.dirty[block] = true;
    }

    /// Mark every block spanned by the half-open element range `[start,
    /// start + count)` as dirty, growing the logical length as needed.
    ///
    /// A `count` of zero marks nothing.
    pub fn mark_range(&mut self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        let end = start + count; // exclusive
        self.ensure_len(end);
        let first_block = start / self.block_len_elements;
        let last_block = (end - 1) / self.block_len_elements;
        for block in first_block..=last_block {
            self.dirty[block] = true;
        }
    }

    /// Number of blocks currently marked dirty.
    pub fn dirty_block_count(&self) -> usize {
        self.dirty.iter().filter(|&&d| d).count()
    }

    /// Whether no block is currently dirty.
    pub fn is_clean(&self) -> bool {
        !self.dirty.iter().any(|&d| d)
    }

    /// Byte length of the block at `block_index`, clamped so the final partial
    /// block never reports bytes past the live extent of the column.
    #[inline]
    fn clamped_block_byte_len(&self, block_index: usize) -> usize {
        let start = block_index * self.block_byte_len();
        let unclamped_end = start + self.block_byte_len();
        let clamped_end = unclamped_end.min(self.live_byte_len());
        clamped_end.saturating_sub(start)
    }

    /// Every dirty block as its own span, in ascending block order.
    ///
    /// Each entry's `byte_len` is clamped so a trailing partial block stops at
    /// `len * stride`.
    pub fn dirty_blocks(&self) -> Vec<DirtyBlock> {
        let mut out = Vec::with_capacity(self.dirty_block_count());
        for (block_index, &is_dirty) in self.dirty.iter().enumerate() {
            if is_dirty {
                out.push(DirtyBlock {
                    block_index,
                    byte_offset: block_index * self.block_byte_len(),
                    byte_len: self.clamped_block_byte_len(block_index),
                });
            }
        }
        out
    }

    /// Minimal set of contiguous upload spans: adjacent dirty blocks are merged
    /// into a single [`DirtyBlock`], in ascending order.
    ///
    /// This is the key "minimal upload" output — e.g. dirty blocks `0, 1, 3`
    /// coalesce into spans covering blocks `[0, 1]` and `[3]`.
    pub fn coalesced_dirty_blocks(&self) -> Vec<DirtyBlock> {
        let mut out: Vec<DirtyBlock> = Vec::new();
        let mut run_start: Option<usize> = None;
        let block_byte_len = self.block_byte_len();

        // Iterate one past the end so a trailing run is always flushed.
        for block_index in 0..=self.dirty.len() {
            let is_dirty = block_index < self.dirty.len() && self.dirty[block_index];
            match (run_start, is_dirty) {
                (None, true) => run_start = Some(block_index),
                (Some(start), false) => {
                    // Flush the run [start, block_index).
                    let byte_offset = start * block_byte_len;
                    // The last block in the run may be partial; clamp its end.
                    let last = block_index - 1;
                    let end = (last * block_byte_len + block_byte_len).min(self.live_byte_len());
                    out.push(DirtyBlock {
                        block_index: start,
                        byte_offset,
                        byte_len: end.saturating_sub(byte_offset),
                    });
                    run_start = None;
                }
                _ => {}
            }
        }
        out
    }

    /// Take the current minimal upload: returns
    /// [`coalesced_dirty_blocks`](Self::coalesced_dirty_blocks), then clears all
    /// dirty bits and increments [`upload_version`](Self::upload_version).
    ///
    /// When the column is already clean this returns an empty `Vec` and does
    /// **not** bump the version.
    pub fn take_upload(&mut self) -> Vec<DirtyBlock> {
        if self.is_clean() {
            return Vec::new();
        }
        let spans = self.coalesced_dirty_blocks();
        for d in &mut self.dirty {
            *d = false;
        }
        self.upload_version += 1;
        spans
    }

    /// The current upload version, bumped once per non-empty
    /// [`take_upload`](Self::take_upload).
    #[inline]
    pub fn upload_version(&self) -> u64 {
        self.upload_version
    }
}

/// Registry mapping [`ComponentId`] to its [`GpuResidentColumn`], so the engine
/// can nominate individual component columns as GPU-resident and route dirties
/// by component.
#[derive(Default)]
pub struct GpuResidentColumns {
    columns: HashMap<ComponentId, GpuResidentColumn>,
}

impl GpuResidentColumns {
    /// Create an empty registry with no resident columns.
    pub fn new() -> Self {
        Self {
            columns: HashMap::default(),
        }
    }

    /// Register (or replace) the component `id` as a GPU-resident column with
    /// the given element stride and block length.
    ///
    /// Panics if `block_len_elements` is zero, matching
    /// [`GpuResidentColumn::new`].
    pub fn register(
        &mut self,
        id: ComponentId,
        element_stride_bytes: usize,
        block_len_elements: usize,
    ) {
        self.columns
            .insert(id, GpuResidentColumn::new(element_stride_bytes, block_len_elements));
    }

    /// Borrow the resident column for `id`, if any.
    pub fn get(&self, id: ComponentId) -> Option<&GpuResidentColumn> {
        self.columns.get(&id)
    }

    /// Mutably borrow the resident column for `id`, if any.
    pub fn get_mut(&mut self, id: ComponentId) -> Option<&mut GpuResidentColumn> {
        self.columns.get_mut(&id)
    }

    /// Whether `id` is currently registered as a GPU-resident column.
    pub fn is_resident(&self, id: ComponentId) -> bool {
        self.columns.contains_key(&id)
    }

    /// Mark the block containing `element_index` dirty on the column for `id`.
    ///
    /// This is a no-op when `id` is not a resident column, so callers can route
    /// every component write through here unconditionally.
    pub fn mark_dirty(&mut self, id: ComponentId, element_index: usize) {
        if let Some(col) = self.columns.get_mut(&id) {
            col.mark_dirty(element_index);
        }
    }

    /// Number of resident columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether no columns are registered as GPU-resident.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty_and_clean() {
        let col = GpuResidentColumn::new(16, 4);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
        assert_eq!(col.element_stride(), 16);
        assert_eq!(col.block_len(), 4);
        assert_eq!(col.block_count(), 0);
        assert!(col.is_clean());
        assert_eq!(col.dirty_block_count(), 0);
        assert_eq!(col.upload_version(), 0);
        assert!(col.dirty_blocks().is_empty());
        assert!(col.coalesced_dirty_blocks().is_empty());
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_block_len() {
        let _ = GpuResidentColumn::new(16, 0);
    }

    #[test]
    fn mark_single_element_dirties_one_block() {
        let mut col = GpuResidentColumn::new(16, 4);
        col.mark_dirty(5); // element 5 -> block 1
        assert_eq!(col.len(), 6);
        assert_eq!(col.dirty_block_count(), 1);
        let blocks = col.dirty_blocks();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_index, 1);
        assert_eq!(blocks[0].byte_offset, 4 * 16);
    }

    #[test]
    fn mark_range_spanning_multiple_blocks() {
        let mut col = GpuResidentColumn::new(8, 4);
        // elements 2..=9 -> blocks 0,1,2
        col.mark_range(2, 8);
        assert_eq!(col.len(), 10);
        assert_eq!(col.dirty_block_count(), 3);
        let blocks = col.dirty_blocks();
        let indices: Vec<usize> = blocks.iter().map(|b| b.block_index).collect();
        assert_eq!(indices, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn mark_range_zero_count_is_noop() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.ensure_len(10);
        col.mark_range(3, 0);
        assert!(col.is_clean());
    }

    #[test]
    fn dedup_same_block_marked_twice() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_dirty(0);
        col.mark_dirty(1);
        col.mark_dirty(3); // all in block 0
        assert_eq!(col.dirty_block_count(), 1);
        assert_eq!(col.dirty_blocks().len(), 1);
    }

    #[test]
    fn coalescing_merges_adjacent_blocks() {
        let mut col = GpuResidentColumn::new(8, 4);
        // Make room for 4 blocks (16 elements), all clean.
        col.ensure_len(16);
        // Dirty blocks 0, 1, 3.
        col.mark_dirty(0);
        col.mark_dirty(4);
        col.mark_dirty(12);
        assert_eq!(col.dirty_block_count(), 3);

        let spans = col.coalesced_dirty_blocks();
        assert_eq!(spans.len(), 2);

        // First span covers blocks [0, 1] -> offset 0, len 2*4*8 = 64.
        assert_eq!(spans[0].block_index, 0);
        assert_eq!(spans[0].byte_offset, 0);
        assert_eq!(spans[0].byte_len, 2 * 4 * 8);

        // Second span covers block [3] -> offset 3*4*8 = 96, len 4*8 = 32.
        assert_eq!(spans[1].block_index, 3);
        assert_eq!(spans[1].byte_offset, 3 * 4 * 8);
        assert_eq!(spans[1].byte_len, 4 * 8);
    }

    #[test]
    fn partial_last_block_byte_len_is_clamped() {
        let mut col = GpuResidentColumn::new(10, 4);
        // 6 elements -> block 0 full (4 elems), block 1 partial (2 elems).
        col.mark_range(0, 6);
        assert_eq!(col.len(), 6);
        assert_eq!(col.block_count(), 2);

        let blocks = col.dirty_blocks();
        assert_eq!(blocks.len(), 2);
        // Block 0 full: 4 * 10 = 40 bytes.
        assert_eq!(blocks[0].byte_len, 40);
        // Block 1 partial: only 2 live elements -> 2 * 10 = 20 bytes, not 40.
        assert_eq!(blocks[1].byte_offset, 40);
        assert_eq!(blocks[1].byte_len, 20);

        // Coalesced into one span [0..2) clamped to live extent 6*10 = 60.
        let spans = col.coalesced_dirty_blocks();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].byte_offset, 0);
        assert_eq!(spans[0].byte_len, 60);
    }

    #[test]
    fn take_upload_clears_and_bumps_version() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_dirty(0);
        col.mark_dirty(5);
        assert_eq!(col.upload_version(), 0);

        let spans = col.take_upload();
        assert!(!spans.is_empty());
        assert!(col.is_clean());
        assert_eq!(col.dirty_block_count(), 0);
        assert_eq!(col.upload_version(), 1);
    }

    #[test]
    fn take_upload_on_clean_is_empty_and_no_version_bump() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.ensure_len(8); // clean blocks
        assert_eq!(col.upload_version(), 0);
        let spans = col.take_upload();
        assert!(spans.is_empty());
        assert_eq!(col.upload_version(), 0);

        // After a real upload, a second take on the now-clean column stays at 1.
        col.mark_dirty(0);
        let _ = col.take_upload();
        assert_eq!(col.upload_version(), 1);
        let again = col.take_upload();
        assert!(again.is_empty());
        assert_eq!(col.upload_version(), 1);
    }

    #[test]
    fn ensure_len_only_grows() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.ensure_len(10);
        assert_eq!(col.len(), 10);
        col.ensure_len(3); // no shrink
        assert_eq!(col.len(), 10);
    }

    #[test]
    fn registry_routes_mark_dirty_to_right_column() {
        let a = ComponentId::new(0);
        let b = ComponentId::new(1);
        let unregistered = ComponentId::new(99);

        let mut reg = GpuResidentColumns::new();
        assert!(reg.is_empty());
        reg.register(a, 16, 4);
        reg.register(b, 8, 2);
        assert_eq!(reg.len(), 2);
        assert!(reg.is_resident(a));
        assert!(reg.is_resident(b));
        assert!(!reg.is_resident(unregistered));

        reg.mark_dirty(a, 5); // block 1 of column a
        // No-op for unregistered id.
        reg.mark_dirty(unregistered, 0);

        let col_a = reg.get(a).unwrap();
        assert_eq!(col_a.dirty_block_count(), 1);
        assert_eq!(col_a.dirty_blocks()[0].block_index, 1);

        let col_b = reg.get(b).unwrap();
        assert!(col_b.is_clean());

        // get_mut path.
        reg.get_mut(b).unwrap().mark_range(0, 3);
        assert_eq!(reg.get(b).unwrap().dirty_block_count(), 2);

        assert!(reg.get(unregistered).is_none());
    }

    #[test]
    fn default_matches_new() {
        let reg = GpuResidentColumns::default();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn zero_stride_reports_zero_byte_len() {
        let mut col = GpuResidentColumn::new(0, 4);
        col.mark_dirty(2);
        let blocks = col.dirty_blocks();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].byte_offset, 0);
        assert_eq!(blocks[0].byte_len, 0);
    }
}
