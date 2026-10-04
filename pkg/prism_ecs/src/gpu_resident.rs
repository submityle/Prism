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
    /// Number of blocks the renderer currently has allocated on the GPU (its
    /// acknowledged buffer capacity, in blocks). Grows via
    /// [`take_upload_plan`](Self::take_upload_plan) when the live block count
    /// exceeds it, and is only reset to refit the live extent when the caller
    /// explicitly requests it with
    /// [`force_reallocation`](Self::force_reallocation).
    gpu_capacity_blocks: usize,
    /// Monotonically increasing identity of the current GPU buffer allocation.
    /// Bumped whenever a reallocation is produced, signalling the renderer to
    /// drop its old buffer and treat the next upload as a full re-upload.
    buffer_generation: u64,
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
            gpu_capacity_blocks: 0,
            buffer_generation: 0,
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

    /// Blocks the renderer currently has allocated for this column on the GPU
    /// (its acknowledged buffer capacity). Zero until the first
    /// [`take_upload_plan`](Self::take_upload_plan) that produces a
    /// [`GpuReallocation`].
    #[inline]
    pub fn gpu_capacity_blocks(&self) -> usize {
        self.gpu_capacity_blocks
    }

    /// Byte capacity the renderer currently has allocated on the GPU
    /// (`gpu_capacity_blocks * block_byte_len`).
    #[inline]
    pub fn gpu_capacity_bytes(&self) -> usize {
        self.gpu_capacity_blocks * self.block_byte_len()
    }

    /// Identity of the current GPU buffer allocation. Bumped each time a
    /// reallocation is produced, so the renderer can detect that its buffer
    /// handle is stale and must be recreated.
    #[inline]
    pub fn buffer_generation(&self) -> u64 {
        self.buffer_generation
    }

    /// Whether the live block count currently exceeds the acknowledged GPU
    /// capacity, so the next [`take_upload_plan`](Self::take_upload_plan) will
    /// reallocate the GPU buffer and re-upload the whole live extent.
    #[inline]
    pub fn needs_reallocation(&self) -> bool {
        self.block_count() > self.gpu_capacity_blocks
    }

    /// Grow-only capacity (in blocks) that covers `required_blocks`, doubling
    /// the current capacity to amortise reallocations the way persistent
    /// GPU-driven buffers do (Horizon / Insomniac style, design §15). Always
    /// returns at least `required_blocks`.
    #[inline]
    fn grown_capacity(&self, required_blocks: usize) -> usize {
        if required_blocks == 0 {
            return 0;
        }
        let mut cap = self.gpu_capacity_blocks.max(1);
        while cap < required_blocks {
            cap *= 2;
        }
        cap
    }

    /// Shrink the logical length to `element_count`, dropping trailing elements
    /// (e.g. after entities are removed from the tail). Growing is not
    /// performed here — use [`ensure_len`](Self::ensure_len). The dirty bits of
    /// dropped blocks are discarded, so stale tail bytes already on the GPU are
    /// simply no longer part of the live extent and are never re-uploaded.
    ///
    /// The GPU buffer capacity is intentionally left at its high-water mark (a
    /// persistent GPU buffer is rarely shrunk); call
    /// [`force_reallocation`](Self::force_reallocation) to actually reclaim it.
    pub fn truncate(&mut self, element_count: usize) {
        if element_count < self.len {
            self.len = element_count;
            self.resize_bitmap();
        }
    }

    /// Force the next [`take_upload_plan`](Self::take_upload_plan) to reallocate
    /// the GPU buffer to fit the current live extent and re-upload everything,
    /// even if the existing capacity would otherwise suffice.
    ///
    /// Use to reclaim GPU memory after a large [`truncate`](Self::truncate), or
    /// to recover after a lost device invalidated the buffer.
    #[inline]
    pub fn force_reallocation(&mut self) {
        self.gpu_capacity_blocks = 0;
    }

    /// The full GPU upload plan for this frame: an optional buffer
    /// (re)allocation plus the minimal byte spans to copy, clearing the dirty
    /// state (design §15).
    ///
    /// Two cases:
    /// * **Capacity sufficient** (`block_count <= gpu_capacity_blocks`): behaves
    ///   like [`take_upload`](Self::take_upload) — coalesced dirty spans, no
    ///   reallocation, buffer generation unchanged.
    /// * **Capacity exceeded** (the column grew, or
    ///   [`force_reallocation`](Self::force_reallocation) was requested): grows
    ///   `gpu_capacity_blocks` to cover the live extent, bumps
    ///   [`buffer_generation`](Self::buffer_generation), and returns a single
    ///   span covering the entire live extent — because a freshly allocated GPU
    ///   buffer holds no valid bytes and must be fully re-uploaded.
    ///
    /// In both non-empty cases [`upload_version`](Self::upload_version) is
    /// bumped and all dirty bits are cleared.
    pub fn take_upload_plan(&mut self) -> GpuUpload {
        let required = self.block_count();
        if required > self.gpu_capacity_blocks {
            // Reallocation path: the new (larger) buffer is empty, so the whole
            // live extent must be uploaded regardless of which blocks were dirty.
            let new_capacity = self.grown_capacity(required);
            self.gpu_capacity_blocks = new_capacity;
            self.buffer_generation += 1;

            let spans = if self.len == 0 {
                Vec::new()
            } else {
                alloc::vec![DirtyBlock {
                    block_index: 0,
                    byte_offset: 0,
                    byte_len: self.live_byte_len(),
                }]
            };
            for d in &mut self.dirty {
                *d = false;
            }
            if !spans.is_empty() {
                self.upload_version += 1;
            }
            return GpuUpload {
                reallocate: Some(GpuReallocation {
                    capacity_blocks: new_capacity,
                    capacity_bytes: new_capacity * self.block_byte_len(),
                    buffer_generation: self.buffer_generation,
                }),
                spans,
                buffer_generation: self.buffer_generation,
            };
        }

        // Incremental path: existing capacity is sufficient.
        let spans = self.take_upload();
        GpuUpload {
            reallocate: None,
            spans,
            buffer_generation: self.buffer_generation,
        }
    }
}

/// A GPU buffer (re)allocation request emitted by
/// [`GpuResidentColumn::take_upload_plan`] when the live extent outgrows the
/// renderer's current buffer (design §15).
///
/// The renderer must (re)create its GPU buffer at `capacity_bytes`, record the
/// new `buffer_generation`, and treat the accompanying upload spans as a full
/// re-upload of the live extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuReallocation {
    /// New buffer capacity, in blocks.
    pub capacity_blocks: usize,
    /// New buffer capacity, in bytes (`capacity_blocks * block_byte_len`).
    pub capacity_bytes: usize,
    /// The buffer generation this allocation establishes.
    pub buffer_generation: u64,
}

/// The complete GPU upload plan for one column for one frame (design §15):
/// an optional buffer [`GpuReallocation`] plus the minimal byte `spans` to copy.
///
/// Produced by [`GpuResidentColumn::take_upload_plan`]. When `reallocate` is
/// `Some`, the renderer first (re)allocates the buffer, then copies `spans`
/// (which cover the whole live extent). When it is `None`, `spans` are the
/// incremental coalesced dirty spans against the existing buffer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GpuUpload {
    /// Set when the GPU buffer must be (re)allocated before copying `spans`.
    pub reallocate: Option<GpuReallocation>,
    /// Minimal, coalesced byte spans to copy into the GPU buffer this frame.
    pub spans: Vec<DirtyBlock>,
    /// The buffer generation these spans apply to.
    pub buffer_generation: u64,
}

impl GpuUpload {
    /// Whether this plan requests neither a reallocation nor any copy.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.reallocate.is_none() && self.spans.is_empty()
    }

    /// Whether this plan reallocates the GPU buffer this frame.
    #[inline]
    pub fn is_reallocation(&self) -> bool {
        self.reallocate.is_some()
    }

    /// Total number of bytes this plan copies into the GPU buffer.
    #[inline]
    pub fn total_upload_bytes(&self) -> usize {
        self.spans.iter().map(|d| d.byte_len).sum()
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
        self.columns.insert(
            id,
            GpuResidentColumn::new(element_stride_bytes, block_len_elements),
        );
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

    #[test]
    fn fresh_column_has_no_gpu_capacity() {
        let col = GpuResidentColumn::new(8, 4);
        assert_eq!(col.gpu_capacity_blocks(), 0);
        assert_eq!(col.gpu_capacity_bytes(), 0);
        assert_eq!(col.buffer_generation(), 0);
        assert!(!col.needs_reallocation());
    }

    #[test]
    fn first_growth_triggers_full_reupload_reallocation() {
        let mut col = GpuResidentColumn::new(8, 4);
        // 10 elements -> ceil(10/4) = 3 blocks, capacity starts at 0.
        col.mark_range(0, 10);
        assert_eq!(col.block_count(), 3);
        assert!(col.needs_reallocation());

        let plan = col.take_upload_plan();
        assert!(plan.is_reallocation());
        let realloc = plan.reallocate.expect("reallocation expected");
        // grown_capacity(3): 1 -> 2 -> 4 (geometric doubling, >= required).
        assert_eq!(realloc.capacity_blocks, 4);
        assert_eq!(realloc.capacity_bytes, 4 * 4 * 8);
        assert_eq!(realloc.buffer_generation, 1);
        assert_eq!(plan.buffer_generation, 1);

        // A freshly allocated buffer is empty: the whole live extent uploads as
        // a single span regardless of which blocks were dirtied.
        assert_eq!(plan.spans.len(), 1);
        assert_eq!(plan.spans[0].block_index, 0);
        assert_eq!(plan.spans[0].byte_offset, 0);
        assert_eq!(plan.spans[0].byte_len, 10 * 8);
        assert_eq!(plan.total_upload_bytes(), 10 * 8);

        // State after the plan: capacity recorded, generation bumped, clean.
        assert_eq!(col.gpu_capacity_blocks(), 4);
        assert_eq!(col.gpu_capacity_bytes(), 4 * 4 * 8);
        assert_eq!(col.buffer_generation(), 1);
        assert_eq!(col.upload_version(), 1);
        assert!(col.is_clean());
        assert!(!col.needs_reallocation());
    }

    #[test]
    fn sufficient_capacity_takes_incremental_spans() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_range(0, 10);
        // Reallocate once so capacity (4 blocks) comfortably exceeds the 3 live
        // blocks.
        let _ = col.take_upload_plan();
        assert_eq!(col.buffer_generation(), 1);

        // Dirty one element inside the existing extent (block 1).
        col.mark_dirty(5);
        assert!(!col.needs_reallocation());

        let plan = col.take_upload_plan();
        assert!(!plan.is_reallocation());
        assert!(plan.reallocate.is_none());
        // Only block 1 is re-uploaded, buffer generation unchanged.
        assert_eq!(plan.spans.len(), 1);
        assert_eq!(plan.spans[0].block_index, 1);
        // Block 1 begins after one full block of 4 elements * 8 bytes.
        assert_eq!(plan.spans[0].byte_offset, 32);
        assert_eq!(plan.spans[0].byte_len, 4 * 8);
        assert_eq!(plan.buffer_generation, 1);
        assert_eq!(col.buffer_generation(), 1);
        assert_eq!(col.upload_version(), 2);

        // A follow-up plan with nothing dirty is a true no-op.
        let empty = col.take_upload_plan();
        assert!(empty.is_empty());
        assert_eq!(col.buffer_generation(), 1);
        assert_eq!(col.upload_version(), 2);
    }

    #[test]
    fn zero_length_plan_is_empty_without_reallocation() {
        let mut col = GpuResidentColumn::new(8, 4);
        let plan = col.take_upload_plan();
        assert!(plan.is_empty());
        assert!(!plan.is_reallocation());
        assert_eq!(plan.total_upload_bytes(), 0);
        assert_eq!(col.gpu_capacity_blocks(), 0);
        assert_eq!(col.buffer_generation(), 0);
        assert_eq!(col.upload_version(), 0);
    }

    #[test]
    fn truncate_drops_tail_blocks_and_their_dirty_bits() {
        let mut col = GpuResidentColumn::new(8, 4);
        // 12 elements -> 3 blocks, all dirtied.
        col.mark_range(0, 12);
        assert_eq!(col.block_count(), 3);
        assert_eq!(col.dirty_block_count(), 3);

        // Shrink to 5 elements -> 2 blocks; block 2's dirty bit is discarded.
        col.truncate(5);
        assert_eq!(col.len(), 5);
        assert_eq!(col.block_count(), 2);
        assert_eq!(col.dirty_block_count(), 2);

        // Live extent shrinks: the trailing partial block clamps to 5*8 = 40.
        let spans = col.coalesced_dirty_blocks();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].byte_offset, 0);
        assert_eq!(spans[0].byte_len, 5 * 8);

        // truncate never grows.
        col.truncate(99);
        assert_eq!(col.len(), 5);

        // GPU capacity (high-water mark) is untouched by truncate alone.
        assert_eq!(col.gpu_capacity_blocks(), 0);
    }

    #[test]
    fn force_reallocation_forces_full_reupload_next_plan() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_range(0, 8); // 2 blocks
        let first = col.take_upload_plan();
        assert!(first.is_reallocation());
        assert_eq!(col.gpu_capacity_blocks(), 2);
        assert_eq!(col.buffer_generation(), 1);

        // Reclaim/recover: force the next plan to reallocate even though the
        // existing capacity would suffice.
        col.force_reallocation();
        assert_eq!(col.gpu_capacity_blocks(), 0);
        assert!(col.needs_reallocation());

        let plan = col.take_upload_plan();
        assert!(plan.is_reallocation());
        assert_eq!(col.buffer_generation(), 2);
        assert_eq!(col.gpu_capacity_blocks(), 2);
        // Full live extent re-uploaded against the new buffer.
        assert_eq!(plan.spans.len(), 1);
        assert_eq!(plan.total_upload_bytes(), 8 * 8);
    }

    #[test]
    fn gpu_upload_default_is_empty() {
        let plan = GpuUpload::default();
        assert!(plan.is_empty());
        assert!(!plan.is_reallocation());
        assert_eq!(plan.total_upload_bytes(), 0);
        assert_eq!(plan.buffer_generation, 0);
    }

    #[test]
    fn gpu_upload_total_bytes_sums_all_spans() {
        let plan = GpuUpload {
            reallocate: None,
            spans: alloc::vec![
                DirtyBlock {
                    block_index: 0,
                    byte_offset: 0,
                    byte_len: 32,
                },
                DirtyBlock {
                    block_index: 3,
                    byte_offset: 96,
                    byte_len: 16,
                },
            ],
            buffer_generation: 7,
        };
        assert!(!plan.is_empty());
        assert!(!plan.is_reallocation());
        assert_eq!(plan.total_upload_bytes(), 48);
    }
}
