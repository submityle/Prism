//! GPU-resident columns + dirty-block incremental upload (design §15).
//!
//! This module plans updates for fixed-stride columns in renderer-owned GPU
//! buffers; it neither owns nor persistently maps those buffers. Fixed-size
//! **blocks** track caller-reported writes since the last consumed plan.
//! Incremental uploads coalesce dirty blocks; initial allocation, growth and
//! explicit invalidation require a full live-extent upload.
//!
//! This module is the CPU-side bookkeeping half of that scheme. The actual
//! allocation and upload commands belong to the caller; here we only compute
//! *which* bytes to copy. Taking a plan clears its dirties immediately, NOT on
//! GPU acknowledgement. Callers must execute every consumed plan in order, or
//! call [`GpuResidentColumn::force_reallocation`] before retrying a failed or
//! discarded plan. Buffer retirement and synchronization remain caller-owned.
//! This standalone tracker does not subscribe to ECS journals or detect world
//! restores; its owner must report writes and invalidate it on restore/reset.
//! The unit tests therefore verify that the
//! produced [`DirtyBlock`] spans are correct and minimal.
//!
//! See [`GpuResidentColumns`] for the per-[`ComponentId`](crate::component::ComponentId)
//! registry.

use alloc::collections::BTreeMap;
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

/// CPU bookkeeping for fixed-stride elements partitioned into blocks, tracking
/// caller-reported writes since the last consumed upload. No GPU completion or
/// acknowledgement is observed by this type.
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
    /// Disjoint, non-adjacent half-open dirty block intervals. Clean capacity
    /// owns no entries, and a dense dirty range owns just one entry. Planning
    /// and clearing traverse only these intervals, never the clean blocks.
    dirty: BTreeMap<usize, usize>,
    /// Sum of the dirty interval lengths, maintained on insertion/truncation.
    dirty_count: usize,
    /// Monotonically increasing counter bumped each time a non-empty upload is
    /// taken.
    upload_version: u64,
    /// Capacity recorded by the last produced allocation plan, in blocks.
    /// This is not an acknowledgement of actual GPU allocation. Grows via
    /// [`take_upload_plan`](Self::take_upload_plan) when the live block count
    /// exceeds it, and is only reset to refit the live extent when the caller
    /// explicitly requests it with
    /// [`force_reallocation`](Self::force_reallocation).
    gpu_capacity_blocks: usize,
    /// Explicit invalidation must survive even when the live extent is empty.
    reallocation_pending: bool,
    /// Monotonically increasing identity of the current GPU buffer allocation.
    /// Bumped whenever a reallocation is produced, signalling the renderer to
    /// retire its old buffer and treat the next upload as a full re-upload.
    buffer_generation: u64,
}

impl GpuResidentColumn {
    /// Create an empty GPU-resident column.
    ///
    /// # Contract
    /// `block_len_elements` must be `>= 1`; this method panics otherwise. A
    /// zero element stride is permitted (e.g. zero-sized components), in which
    /// case every reported span has `byte_len == 0`.
    /// Panics if the byte length of one block exceeds usize.
    pub fn new(element_stride_bytes: usize, block_len_elements: usize) -> Self {
        assert!(
            block_len_elements >= 1,
            "block_len_elements must be >= 1, got {block_len_elements}"
        );
        block_len_elements
            .checked_mul(element_stride_bytes)
            .expect("GPU resident block byte length exceeds usize");
        Self {
            element_stride_bytes,
            block_len_elements,
            len: 0,
            dirty: BTreeMap::new(),
            dirty_count: 0,
            upload_version: 0,
            gpu_capacity_blocks: 0,
            reallocation_pending: false,
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

    /// Union a nonempty block interval into the index. Inserting a contained
    /// interval does not allocate; adjacent intervals coalesce immediately.
    fn mark_blocks(&mut self, mut start: usize, mut end: usize) {
        if let Some((&previous, &previous_end)) = self.dirty.range(..=start).next_back() {
            if previous_end >= end {
                return;
            }
            if previous_end >= start {
                self.dirty.remove(&previous);
                self.dirty_count -= previous_end - previous;
                start = previous;
            }
        }
        while let Some((&next, &next_end)) = self.dirty.range(start..=end).next() {
            self.dirty.remove(&next);
            self.dirty_count -= next_end - next;
            end = end.max(next_end);
        }
        self.dirty.insert(start, end);
        self.dirty_count += end - start;
    }

    fn clear_dirty(&mut self) {
        self.dirty.clear();
        self.dirty_count = 0;
    }

    /// Grow the logical length to at least `element_count` elements.
    ///
    /// Shrinking is not performed: if `element_count` is less than the current
    /// length this is a no-op. Any blocks newly brought into existence start
    /// clean — callers must [`mark_dirty`](Self::mark_dirty) the elements they
    /// actually wrote. This is O(1), independent of the number of clean blocks.
    ///
    /// # Panics
    /// Panics without changing the column if its live byte extent exceeds usize.
    pub fn ensure_len(&mut self, element_count: usize) {
        if element_count > self.len {
            element_count
                .checked_mul(self.element_stride_bytes)
                .expect("GPU resident live byte length exceeds usize");
            self.len = element_count;
        }
    }

    /// Mark the block containing `element_index` as dirty, growing the logical
    /// length to include that element if necessary.
    pub fn mark_dirty(&mut self, element_index: usize) {
        self.ensure_len(
            element_index
                .checked_add(1)
                .expect("element index exceeds usize"),
        );
        let block = element_index / self.block_len_elements;
        self.mark_blocks(block, block + 1);
    }

    /// Mark every block spanned by the half-open element range `[start,
    /// start + count)` as dirty, growing the logical length as needed.
    ///
    /// A `count` of zero marks nothing.
    pub fn mark_range(&mut self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        let end = start
            .checked_add(count)
            .expect("element range exceeds usize");
        self.ensure_len(end);
        let first_block = start / self.block_len_elements;
        self.mark_blocks(first_block, end.div_ceil(self.block_len_elements));
    }

    /// Number of blocks currently marked dirty.
    pub fn dirty_block_count(&self) -> usize {
        self.dirty_count
    }

    /// Whether no block is currently dirty.
    pub fn is_clean(&self) -> bool {
        self.dirty.is_empty()
    }

    /// Byte length of the block at `block_index`, clamped so the final partial
    /// block never reports bytes past the live extent of the column.
    #[inline]
    fn clamped_block_byte_len(&self, block_index: usize) -> usize {
        let start = block_index * self.block_byte_len();
        self.live_byte_len()
            .saturating_sub(start)
            .min(self.block_byte_len())
    }

    /// Every dirty block as its own span, in ascending block order.
    ///
    /// Each entry's `byte_len` is clamped so a trailing partial block stops at
    /// `len * stride`.
    pub fn dirty_blocks(&self) -> Vec<DirtyBlock> {
        let mut out = Vec::with_capacity(self.dirty_block_count());
        for (&start, &end) in &self.dirty {
            for block_index in start..end {
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
        self.coalesced_dirty_spans().collect()
    }

    /// Borrow the coalesced dirty spans without allocating or consuming them.
    /// Iteration visits only dirty intervals, not clean blocks.
    pub fn coalesced_dirty_spans(&self) -> impl ExactSizeIterator<Item = DirtyBlock> + '_ {
        self.dirty.iter().map(|(&start, &end)| {
            let byte_offset = start * self.block_byte_len();
            // Avoid computing the rounded-up end: a partial final block
            // can extend beyond usize even when the live extent fits.
            let last = end - 1;
            let byte_end = last * self.block_byte_len() + self.clamped_block_byte_len(last);
            DirtyBlock {
                block_index: start,
                byte_offset,
                byte_len: byte_end - byte_offset,
            }
        })
    }

    /// Take the current minimal upload: returns
    /// [`coalesced_dirty_blocks`](Self::coalesced_dirty_blocks), then clears all
    /// dirty bits and increments [`upload_version`](Self::upload_version).
    ///
    /// When the column is already clean this returns an empty `Vec` and does
    /// **not** bump the version.
    ///
    /// # Panics
    /// Panics without changing state if the upload version is exhausted.
    pub fn take_upload(&mut self) -> Vec<DirtyBlock> {
        if self.is_clean() {
            return Vec::new();
        }
        let next_version = self
            .upload_version
            .checked_add(1)
            .expect("GPU upload version exhausted");
        let spans = self.coalesced_dirty_blocks();
        self.clear_dirty();
        self.upload_version = next_version;
        spans
    }

    /// The current upload version, bumped once per non-empty
    /// upload taken by [`take_upload`](Self::take_upload) or
    /// [`take_upload_plan`](Self::take_upload_plan). Non-empty means at least
    /// one span, including zero-byte spans for zero-stride elements.
    #[inline]
    pub fn upload_version(&self) -> u64 {
        self.upload_version
    }

    /// Block capacity recorded by planning, not GPU acknowledgement. Zero
    /// after invalidation or release, and until the first
    /// [`take_upload_plan`](Self::take_upload_plan) that produces a
    /// [`GpuReallocation`].
    #[inline]
    pub fn gpu_capacity_blocks(&self) -> usize {
        self.gpu_capacity_blocks
    }

    /// Byte capacity recorded by planning, not GPU acknowledgement
    /// (`gpu_capacity_blocks * block_byte_len`).
    #[inline]
    pub fn gpu_capacity_bytes(&self) -> usize {
        self.gpu_capacity_blocks * self.block_byte_len()
    }

    /// Identity of the current GPU buffer allocation. Bumped each time a
    /// reallocation is produced, so the renderer can detect that its buffer
    /// handle is stale and must be replaced (or released for zero capacity).
    #[inline]
    pub fn buffer_generation(&self) -> u64 {
        self.buffer_generation
    }

    /// Whether the next plan must replace or release the buffer, because the
    /// live block count exceeds planned capacity or invalidation is pending.
    #[inline]
    pub fn needs_reallocation(&self) -> bool {
        self.reallocation_pending || self.block_count() > self.gpu_capacity_blocks
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
        self.gpu_capacity_blocks.max(
            required_blocks
                .checked_next_power_of_two()
                .expect("GPU resident block capacity exceeds usize"),
        )
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
            let needed = self.block_count();
            let removed = self.dirty.split_off(&needed);
            for (start, end) in removed {
                self.dirty_count -= end - start;
            }
            if let Some((&start, &end)) = self.dirty.last_key_value() {
                if end > needed {
                    self.dirty.insert(start, needed);
                    self.dirty_count -= end - needed;
                }
            }
        }
    }

    /// Force the next [`take_upload_plan`](Self::take_upload_plan) to reallocate
    /// the GPU buffer to fit the current live extent and re-upload everything,
    /// even if the existing capacity would otherwise suffice.
    ///
    /// Use to reclaim GPU memory after a large [`truncate`](Self::truncate), or
    /// to recover after a lost device invalidated the buffer.
    /// For an empty column the next plan explicitly releases the old buffer:
    /// zero capacity, no spans, and a new generation. Repeated requests before
    /// taking a plan coalesce. This does not itself release GPU resources.
    #[inline]
    pub fn force_reallocation(&mut self) {
        self.gpu_capacity_blocks = 0;
        self.reallocation_pending = true;
    }

    /// The full GPU upload plan for this frame: an optional buffer
    /// (re)allocation plus the minimal byte spans to copy, clearing the dirty
    /// state (design §15).
    ///
    /// Two cases:
    /// * **Capacity sufficient, no pending invalidation**: behaves
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
    /// An explicit invalidation of an empty column instead emits a zero-capacity
    /// release with a new generation, without bumping the upload version.
    /// Taking a plan is destructive bookkeeping, not GPU acknowledgement.
    ///
    /// # Panics
    /// Panics without changing state if capacity arithmetic, upload version or
    /// buffer generation overflows. A failed plan leaves invalidation pending.
    pub fn take_upload_plan(&mut self) -> GpuUpload {
        let required = self.block_count();
        if self.needs_reallocation() {
            // A replacement buffer has no valid bytes, so initialize the whole
            // live extent. Empty forced plans instead release the old binding.
            let new_capacity = self.grown_capacity(required);
            let capacity_bytes = new_capacity
                .checked_mul(self.block_byte_len())
                .expect("GPU resident buffer byte capacity exceeds usize");
            let next_generation = self
                .buffer_generation
                .checked_add(1)
                .expect("GPU buffer generation exhausted");
            let next_version = if self.len == 0 {
                self.upload_version
            } else {
                self.upload_version
                    .checked_add(1)
                    .expect("GPU upload version exhausted")
            };

            let spans = if self.len == 0 {
                Vec::new()
            } else {
                alloc::vec![DirtyBlock {
                    block_index: 0,
                    byte_offset: 0,
                    byte_len: self.live_byte_len(),
                }]
            };
            self.gpu_capacity_blocks = new_capacity;
            self.buffer_generation = next_generation;
            self.upload_version = next_version;
            self.reallocation_pending = false;
            self.clear_dirty();
            return GpuUpload {
                reallocate: Some(GpuReallocation {
                    capacity_blocks: new_capacity,
                    capacity_bytes,
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
/// [`GpuResidentColumn::take_upload_plan`] on growth or explicit invalidation
/// (design §15), including an explicit zero-capacity release.
///
/// For positive `capacity_bytes`, create a replacement buffer and fully upload
/// the live extent. For zero bytes, release the current binding without creating
/// a zero-byte buffer or issuing zero-byte copies. Both cases establish the new
/// generation; retire the old allocation only once its users/GPU work finish.
/// Zero-stride columns may have positive logical block capacity but zero bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuReallocation {
    /// New buffer capacity, in blocks.
    pub capacity_blocks: usize,
    /// New byte capacity (`capacity_blocks * block_byte_len`); zero means release.
    pub capacity_bytes: usize,
    /// The buffer generation this allocation establishes.
    pub buffer_generation: u64,
}

/// The complete GPU upload plan for one column for one frame (design §15):
/// an optional buffer [`GpuReallocation`] plus the minimal byte `spans` to copy.
///
/// Produced by [`GpuResidentColumn::take_upload_plan`]. When `reallocate` is
/// `Some`, the renderer replaces or releases the buffer as specified by
/// [`GpuReallocation`], then copies any nonzero spans (the whole live extent).
/// When it is `None`, `spans` are the
/// incremental coalesced dirty spans against the existing buffer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct GpuUpload {
    /// Set when the buffer must be replaced or released before copying `spans`.
    pub reallocate: Option<GpuReallocation>,
    /// Minimal, coalesced byte spans to copy into the GPU buffer this frame.
    pub spans: Vec<DirtyBlock>,
    /// The buffer generation these spans apply to.
    pub buffer_generation: u64,
}

impl GpuUpload {
    /// Whether this plan requests neither replacement/release nor any copy.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.reallocate.is_none() && self.spans.is_empty()
    }

    /// Whether this plan replaces or releases the GPU buffer this frame.
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

        // The borrowed diagnostic view has exact remaining length and does not
        // allocate a second span list or consume the column's dirty state.
        let mut borrowed = col.coalesced_dirty_spans();
        assert_eq!(borrowed.len(), 2);
        assert_eq!(borrowed.next(), Some(spans[0]));
        assert_eq!(borrowed.len(), 1);
        assert_eq!(borrowed.next(), Some(spans[1]));
        assert_eq!(borrowed.len(), 0);
        assert_eq!(borrowed.next(), None);
        assert_eq!(col.dirty_block_count(), 3);

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
    fn empty_forced_plan_releases_once_then_regrowth_initializes_every_live_byte() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_range(0, 17);
        let first = col.take_upload_plan();
        assert_eq!(first.reallocate.unwrap().capacity_blocks, 8);
        col.truncate(0);
        // Truncation alone retains the high-water allocation.
        assert!(col.take_upload_plan().is_empty());
        assert_eq!(col.gpu_capacity_blocks(), 8);
        col.force_reallocation();
        col.force_reallocation();
        assert!(col.needs_reallocation());
        let release = col.take_upload_plan();
        assert_eq!(
            release.reallocate,
            Some(GpuReallocation {
                capacity_blocks: 0,
                capacity_bytes: 0,
                buffer_generation: 2,
            })
        );
        assert_eq!(release.buffer_generation, 2);
        assert!(release.spans.is_empty());
        assert!(!release.is_empty());
        assert!(release.is_reallocation());
        assert_eq!(release.total_upload_bytes(), 0);
        assert_eq!(col.upload_version(), 1);
        assert_eq!(col.gpu_capacity_bytes(), 0);
        assert!(!col.needs_reallocation());
        assert!(col.take_upload_plan().is_empty());
        assert_eq!(col.buffer_generation(), 2);

        // Even clean growth needs full initialization after the release.
        col.ensure_len(5);
        let regrow = col.take_upload_plan();
        assert_eq!(regrow.reallocate.unwrap().capacity_blocks, 2);
        assert_eq!(regrow.total_upload_bytes(), 40);
        assert_eq!(col.buffer_generation(), 3);
        assert_eq!(col.upload_version(), 2);
    }

    #[test]
    fn force_on_never_allocated_empty_column_is_an_explicit_release() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.force_reallocation();
        let release = col.take_upload_plan();
        assert_eq!(release.reallocate.unwrap().capacity_bytes, 0);
        assert_eq!(release.buffer_generation, 1);
        assert!(release.spans.is_empty());
        assert_eq!(col.upload_version(), 0);
        assert!(col.take_upload_plan().is_empty());
    }

    #[test]
    fn zero_stride_plans_never_require_a_gpu_allocation_or_copy() {
        let mut col = GpuResidentColumn::new(0, 4);
        col.mark_range(0, 9);
        let plan = col.take_upload_plan();
        let replacement = plan.reallocate.unwrap();
        assert_eq!(replacement.capacity_blocks, 4);
        assert_eq!(replacement.capacity_bytes, 0);
        assert_eq!(plan.total_upload_bytes(), 0);
        assert!(plan.spans.iter().all(|span| span.byte_len == 0));
        assert_eq!(col.upload_version(), 1);
        col.truncate(0);
        col.force_reallocation();
        let release = col.take_upload_plan();
        assert_eq!(release.reallocate.unwrap().capacity_blocks, 0);
        assert!(release.spans.is_empty());
        assert_eq!(col.upload_version(), 1);
        assert_eq!(col.buffer_generation(), 2);
    }

    // Deliberately dense, independent oracle: production interval boundaries
    // are never used to calculate expected blocks or coalesced spans.
    fn bitmap_spans(
        bits: &[bool],
        len: usize,
        block_len: usize,
        stride: usize,
        coalesce: bool,
    ) -> Vec<DirtyBlock> {
        let mut spans = Vec::new();
        let mut block = 0;
        while block < bits.len() {
            if !bits[block] {
                block += 1;
                continue;
            }
            let start = block;
            block += 1;
            if coalesce {
                while block < bits.len() && bits[block] {
                    block += 1;
                }
            }
            let byte_offset = start * block_len * stride;
            spans.push(DirtyBlock {
                block_index: start,
                byte_offset,
                byte_len: (block * block_len).min(len) * stride - byte_offset,
            });
        }
        spans
    }

    #[test]
    fn mixed_operations_match_bitmap_oracle() {
        for block_len in [1, 3, 4, 17] {
            for stride in [0, 3, 8] {
                let mut col = GpuResidentColumn::new(stride, block_len);
                let mut bits = Vec::new();
                let mut len = 0usize;
                let mut capacity = 0usize;
                let mut pending = false;
                let mut generation = 0u64;
                let mut version = 0u64;
                let mut seed = 0x51ad_f00d_cafe_1234u64;
                for step in 0..4096 {
                    let mut next = || {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 32) as usize
                    };
                    let op = next() % 9;
                    let start = next() % 193;
                    let count = next() % 49;
                    match op {
                        0 => {
                            col.ensure_len(start);
                            len = len.max(start);
                            bits.resize(len.div_ceil(block_len), false);
                        }
                        1 | 2 => {
                            col.mark_dirty(start);
                            len = len.max(start + 1);
                            bits.resize(len.div_ceil(block_len), false);
                            bits[start / block_len] = true;
                        }
                        3 => {
                            col.mark_range(start, count);
                            if count != 0 {
                                len = len.max(start + count);
                                bits.resize(len.div_ceil(block_len), false);
                                bits[start / block_len..(start + count).div_ceil(block_len)]
                                    .fill(true);
                            }
                        }
                        4 => {
                            col.truncate(start);
                            len = len.min(start);
                            bits.truncate(len.div_ceil(block_len));
                        }
                        5 => {
                            let spans = bitmap_spans(&bits, len, block_len, stride, true);
                            version += u64::from(!spans.is_empty());
                            assert_eq!(col.take_upload(), spans);
                            bits.fill(false);
                        }
                        6 | 7 => {
                            let replace = pending || bits.len() > capacity;
                            let (reallocate, spans) = if replace {
                                capacity = if bits.is_empty() {
                                    0
                                } else {
                                    bits.len().next_power_of_two()
                                };
                                pending = false;
                                generation += 1;
                                let spans = if len == 0 {
                                    Vec::new()
                                } else {
                                    alloc::vec![DirtyBlock {
                                        block_index: 0,
                                        byte_offset: 0,
                                        byte_len: len * stride
                                    }]
                                };
                                (
                                    Some(GpuReallocation {
                                        capacity_blocks: capacity,
                                        capacity_bytes: capacity * block_len * stride,
                                        buffer_generation: generation,
                                    }),
                                    spans,
                                )
                            } else {
                                (None, bitmap_spans(&bits, len, block_len, stride, true))
                            };
                            version += u64::from(!spans.is_empty());
                            assert_eq!(
                                col.take_upload_plan(),
                                GpuUpload {
                                    reallocate,
                                    spans,
                                    buffer_generation: generation
                                }
                            );
                            bits.fill(false);
                        }
                        8 => {
                            col.force_reallocation();
                            capacity = 0;
                            pending = true;
                        }
                        _ => unreachable!(),
                    }
                    assert_eq!(
                        col.len(),
                        len,
                        "step {step}, block {block_len}, stride {stride}"
                    );
                    assert_eq!(col.block_count(), bits.len());
                    assert_eq!(
                        col.dirty_block_count(),
                        bits.iter().filter(|&&bit| bit).count()
                    );
                    assert_eq!(col.is_clean(), !bits.iter().any(|&bit| bit));
                    assert_eq!(
                        col.dirty_blocks(),
                        bitmap_spans(&bits, len, block_len, stride, false)
                    );
                    assert_eq!(
                        col.coalesced_dirty_blocks(),
                        bitmap_spans(&bits, len, block_len, stride, true)
                    );
                    assert_eq!(col.gpu_capacity_blocks(), capacity);
                    assert_eq!(col.gpu_capacity_bytes(), capacity * block_len * stride);
                    assert_eq!(col.needs_reallocation(), pending || bits.len() > capacity);
                    assert_eq!(col.buffer_generation(), generation);
                    assert_eq!(col.upload_version(), version);
                    let mut previous_end = None;
                    for (&start, &end) in &col.dirty {
                        assert!(start < end && end <= bits.len());
                        assert!(previous_end.is_none_or(|previous| previous < start));
                        previous_end = Some(end);
                    }
                }
            }
        }
    }

    #[test]
    fn huge_clean_extent_sparse_and_dense_frontiers_do_not_materialize_clean_blocks() {
        // Enormous even on 32-bit platforms; a dense per-block structure cannot
        // pass this test within a normal unit-test memory budget.
        let extent = usize::MAX / 16;
        let mut col = GpuResidentColumn::new(1, 1);
        col.ensure_len(extent);
        assert!(col.dirty.is_empty());
        assert!(col.take_upload().is_empty());
        col.mark_dirty(1);
        col.mark_dirty(extent - 1);
        assert_eq!(col.dirty.len(), 2);
        assert_eq!(col.dirty_block_count(), 2);
        assert_eq!(col.take_upload().len(), 2);
        col.mark_range(0, extent);
        assert_eq!(col.dirty.len(), 1);
        assert_eq!(col.dirty_block_count(), extent);
        assert_eq!(
            col.coalesced_dirty_blocks(),
            alloc::vec![DirtyBlock {
                block_index: 0,
                byte_offset: 0,
                byte_len: extent,
            }]
        );
        col.truncate(extent / 2);
        col.ensure_len(extent);
        assert_eq!(col.dirty_block_count(), extent / 2);
        assert_eq!(col.take_upload()[0].byte_len, extent / 2);
        assert!(col.dirty.is_empty());
        col.truncate(0);
        col.ensure_len(extent);
        assert!(col.is_clean());
    }

    #[test]
    fn truncate_regrow_keeps_partial_dirty_block_but_not_removed_tail() {
        let mut col = GpuResidentColumn::new(3, 4);
        col.mark_range(0, 16);
        col.truncate(5);
        col.ensure_len(16);
        assert_eq!(col.dirty_block_count(), 2);
        assert_eq!(
            col.take_upload(),
            alloc::vec![DirtyBlock {
                block_index: 0,
                byte_offset: 0,
                byte_len: 24,
            }]
        );
        col.mark_dirty(15);
        col.truncate(4);
        col.ensure_len(16);
        assert!(col.is_clean());
    }

    // Include private pending state and the complete interval index, not just
    // observable spans: a failed operation must remain exactly retryable.
    fn assert_panics_unchanged(
        col: &mut GpuResidentColumn,
        action: impl FnOnce(&mut GpuResidentColumn),
    ) {
        let snapshot = |col: &GpuResidentColumn| {
            (
                col.len,
                col.dirty.clone(),
                col.dirty_count,
                col.upload_version,
                col.gpu_capacity_blocks,
                col.reallocation_pending,
                col.buffer_generation,
            )
        };
        let before = snapshot(col);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(col))).is_err());
        assert_eq!(snapshot(col), before);
    }

    #[test]
    fn usize_overflow_panics_without_side_effects() {
        assert!(std::panic::catch_unwind(|| GpuResidentColumn::new(usize::MAX, 2)).is_err());
        let mut col = GpuResidentColumn::new(2, 4);
        col.mark_dirty(1);
        assert_panics_unchanged(&mut col, |col| col.ensure_len(usize::MAX));
        assert_panics_unchanged(&mut col, |col| col.mark_dirty(usize::MAX));
        assert_panics_unchanged(&mut col, |col| col.mark_dirty(usize::MAX / 2));
        assert_panics_unchanged(&mut col, |col| col.mark_range(usize::MAX, 1));
        assert_panics_unchanged(&mut col, |col| col.mark_range(usize::MAX / 2, 1));

        let mut rounded_blocks = GpuResidentColumn::new(1, 1);
        rounded_blocks.mark_dirty(usize::MAX - 1);
        assert_eq!(rounded_blocks.dirty_blocks()[0].byte_len, 1);
        assert_panics_unchanged(&mut rounded_blocks, |col| {
            col.take_upload_plan();
        });
        // Live bytes fit, but power-of-two capacity times stride does not.
        let mut rounded_bytes = GpuResidentColumn::new(3, 1);
        rounded_bytes.mark_range(0, usize::MAX / 3);
        assert_panics_unchanged(&mut rounded_bytes, |col| {
            col.take_upload_plan();
        });

        // Partial block's rounded-up byte end would overflow, its live end fits.
        let mut partial = GpuResidentColumn::new(1, 4);
        partial.mark_dirty(usize::MAX - 1);
        let spans = partial.coalesced_dirty_blocks();
        assert_eq!(spans[0].byte_offset, usize::MAX - 3);
        assert_eq!(spans[0].byte_len, 3);
    }

    #[test]
    fn exhausted_versions_and_generations_preserve_all_state() {
        let mut incremental = GpuResidentColumn::new(8, 4);
        incremental.ensure_len(8);
        incremental.take_upload_plan();
        incremental.mark_dirty(1);
        incremental.upload_version = u64::MAX;
        assert_panics_unchanged(&mut incremental, |col| {
            col.take_upload();
        });
        assert_panics_unchanged(&mut incremental, |col| {
            col.take_upload_plan();
        });

        let mut full = GpuResidentColumn::new(8, 4);
        full.mark_dirty(1);
        full.upload_version = u64::MAX;
        assert_panics_unchanged(&mut full, |col| {
            col.take_upload_plan();
        });
        full.force_reallocation();
        assert_panics_unchanged(&mut full, |col| {
            col.take_upload_plan();
        });
        full.upload_version = 0;
        full.buffer_generation = u64::MAX;
        assert_panics_unchanged(&mut full, |col| {
            col.take_upload_plan();
        });
        full.force_reallocation();
        assert_panics_unchanged(&mut full, |col| {
            col.take_upload_plan();
        });
        full.truncate(0);
        assert_panics_unchanged(&mut full, |col| {
            col.take_upload_plan();
        });

        // Releasing and clean no-ops do not consume an upload version; an
        // incremental upload does not consume an allocation generation.
        full.buffer_generation = 7;
        full.upload_version = u64::MAX;
        assert_eq!(
            full.take_upload_plan().reallocate.unwrap().capacity_bytes,
            0
        );
        assert_eq!(full.buffer_generation(), 8);
        assert_eq!(full.upload_version(), u64::MAX);
        full.buffer_generation = u64::MAX;
        assert!(full.take_upload_plan().is_empty());
        incremental.upload_version = 0;
        incremental.buffer_generation = u64::MAX;
        assert!(!incremental.take_upload_plan().spans.is_empty());
        assert_eq!(incremental.buffer_generation(), u64::MAX);
    }

    #[test]
    fn last_version_and_generation_are_usable_without_wraparound() {
        let mut col = GpuResidentColumn::new(8, 4);
        col.mark_dirty(1);
        col.upload_version = u64::MAX - 1;
        col.buffer_generation = u64::MAX - 1;
        let plan = col.take_upload_plan();
        assert_eq!(plan.buffer_generation, u64::MAX);
        assert_eq!(col.upload_version(), u64::MAX);
        assert!(col.take_upload_plan().is_empty());
        assert!(col.take_upload().is_empty());
        col.mark_dirty(1);
        assert_panics_unchanged(&mut col, |col| {
            col.take_upload_plan();
        });

        // Releasing may also establish the last generation and must not
        // consume an upload version, even if that version is exhausted.
        col.truncate(0);
        col.buffer_generation = u64::MAX - 1;
        col.force_reallocation();
        assert_eq!(col.take_upload_plan().buffer_generation, u64::MAX);
        assert_eq!(col.upload_version(), u64::MAX);
        col.force_reallocation();
        assert_panics_unchanged(&mut col, |col| {
            col.take_upload_plan();
        });
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
