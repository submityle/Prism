//! A [`Table`] is the columnar storage for a single archetype: one
//! [`Column`] per component type plus a parallel `entities` vector. Every
//! column has exactly the same length as `entities`; row `r` of every column
//! and `entities[r]` all belong to the same entity.
//!
//! M0 uses one contiguous table per archetype. The 16 KiB fixed-size chunking
//! and per-chunk change-versions described in the design doc (§5.3) are an M2
//! refinement layered on top of this representation; the row/column model and
//! its invariants are unchanged by that work.

use alloc::vec::Vec;
use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::change::{ComponentTicks, Tick};
use crate::collections::HashMap;
use crate::component::{Component, ComponentId, DropFn};
use crate::entity::Entity;
use crate::storage::arena::ChunkArena;
use crate::storage::blob_vec::BlobVec;
use crate::storage::chunk::{
    rows_per_chunk, ChunkBounds, ChunkChange, ChunkMeta, ChunkVersions, ChunkVisitStats,
};

static NEXT_CHUNK_INCARNATION: AtomicU64 = AtomicU64::new(1);

#[inline]
fn next_chunk_incarnation() -> u64 {
    NEXT_CHUNK_INCARNATION.fetch_add(1, Ordering::Relaxed)
}

/// A single type-erased component column within a [`Table`].
///
/// Alongside the component bytes in `data`, every row carries two change-
/// detection ticks (design §10): `added_ticks[row]` records when the value was
/// first inserted and `changed_ticks[row]` when it was last written. All three
/// vectors stay in lockstep with the owning table's `entities` vector.
///
/// The ticks live behind [`UnsafeCell`] so the `&mut T` / `Mut<T>` query fetch
/// can stamp the changed tick through a shared `&Column` while iterating,
/// mirroring the interior-mutability discipline already used for the component
/// bytes themselves.
pub struct Column {
    data: BlobVec,
    added_ticks: Vec<UnsafeCell<Tick>>,
    changed_ticks: Vec<UnsafeCell<Tick>>,
    /// Coarse per-chunk changed-versions (design §7/§10). Each cell upper-bounds
    /// the newest `changed_ticks` entry in its row window, letting a query skip
    /// an entire unchanged window without touching per-row ticks. Kept in
    /// lockstep with the row count by the same push/replace/swap-remove paths.
    chunk: ChunkVersions,
}

// SAFETY: `UnsafeCell<Tick>` makes `Column` `!Sync` by default. The tick cells
// are mutated only under the same unique-access discipline that governs the
// component bytes in `BlobVec` (itself `unsafe impl Sync`): a writer stamping a
// row holds unique access to that row, and no reader observes a cell mid-write.
// Sharing a `&Column` across threads is therefore sound.
unsafe impl Sync for Column {}

impl Column {
    /// Create a column whose value bytes are a **borrowed slice of the owning
    /// table's chunk blocks** (design §5.3 DOTS chunk): the table allocates one
    /// block per chunk holding all columns and attaches each column's slice via
    /// [`Column::attach_block`].
    fn new(layout: Layout, drop: Option<DropFn>, rows_per_chunk: usize) -> Self {
        Self {
            data: BlobVec::new_borrowed(layout, drop, rows_per_chunk),
            added_ticks: Vec::new(),
            changed_ticks: Vec::new(),
            chunk: ChunkVersions::new(rows_per_chunk),
        }
    }

    /// Attach this column's change-version cell and value slice of a
    /// newly-allocated chunk block (design §5.3: the version is embedded in the
    /// block header next to the data).
    ///
    /// # Safety
    /// `cell` must point at an initialized, aligned `ChunkChange` and `slice` at the aligned slice
    /// base, both inside a live chunk block (see [`BlobVec::attach_block`]).
    unsafe fn attach_chunk(&mut self, cell: *mut ChunkChange, slice: *mut u8) {
        // SAFETY: forwarded contract for the header cell.
        unsafe { self.chunk.attach_cell(cell) };
        // SAFETY: forwarded contract for the value slice.
        unsafe { self.data.attach_block(slice) };
    }

    /// Detach this column's value slice and version cell of the trailing chunk
    /// block (the block is freed by the table).
    fn detach_chunk(&mut self) {
        self.data.detach_block();
        self.chunk.detach_cell();
    }

    /// Number of elements (equals the owning table's row count).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the column is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Rows this column can hold before its backing allocation must grow.
    ///
    /// The minimum of the value buffer and the two per-row tick vectors, so it
    /// is a sound "no reallocation up to here" bound. Zero-sized components
    /// report [`usize::MAX`] from the value buffer and are bounded by the tick
    /// vectors instead.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data
            .capacity()
            .min(self.added_ticks.capacity())
            .min(self.changed_ticks.capacity())
    }

    /// Reserve room for `additional` more rows in the value buffer, both
    /// per-row tick vectors, and the chunk-version windows (design §17 预热).
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
        self.added_ticks.reserve(additional);
        self.changed_ticks.reserve(additional);
        self.chunk.reserve_rows(additional);
    }

    /// Raw pointer to the component value at `row`.
    ///
    /// # Safety
    /// `row < len()`, and the pointer must only be accessed as the component
    /// type this column stores.
    #[inline]
    pub unsafe fn get_ptr(&self, row: usize) -> *mut u8 {
        // SAFETY: forwarded contract — `row < len()`.
        unsafe { self.data.get_ptr(row) }
    }

    /// Typed reference to the value at `row`.
    ///
    /// # Safety
    /// `row < len()` and `T` must be the exact component type stored here.
    #[inline]
    pub unsafe fn get<T>(&self, row: usize) -> &T {
        // SAFETY: forwarded contract; pointer is valid and aligned for `T`.
        unsafe { &*self.get_ptr(row).cast::<T>() }
    }

    /// Typed mutable reference to the value at `row`.
    ///
    /// # Safety
    /// `row < len()`, `T` must be the exact component type stored here, and the
    /// caller must hold unique access to this value.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut<T>(&self, row: usize) -> &mut T {
        // SAFETY: forwarded contract; caller guarantees unique access, so
        // forming a `&mut` from the interior pointer does not alias.
        unsafe { &mut *self.get_ptr(row).cast::<T>() }
    }

    /// Borrow the whole column as a typed slice `&[T]`.
    ///
    /// A column's values live in fixed-size blocks (design §5.3 physical
    /// chunks), so the whole column is a single contiguous run **only while it
    /// fits in one block**. Columns that span several blocks must be accessed
    /// block by block via [`Column::as_slice_range`] (or iterated with
    /// `simd_iter` / `par_chunks`); this method panics on a multi-block column.
    ///
    /// # Safety
    /// `T` must be the exact component type stored here, and the caller must
    /// hold shared access to the column's rows for the slice's lifetime.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        // SAFETY: forwarded type contract; delegates to `BlobVec::as_slice`.
        unsafe { self.data.as_slice::<T>() }
    }

    /// Borrow the whole column as a typed mutable slice `&mut [T]`.
    ///
    /// Note this stamps no change ticks: callers that mutate through the slice
    /// must bump the relevant ticks themselves (e.g. via
    /// [`Column::set_changed_tick`]) or treat the write as tick-neutral.
    ///
    /// # Safety
    /// `T` must be the exact component type stored here, and the caller must
    /// hold unique access to the column's rows for the slice's lifetime.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice<T>(&self) -> &mut [T] {
        // SAFETY: forwarded type + unique-access contract; delegates to
        // `BlobVec::as_mut_slice`.
        unsafe { self.data.as_mut_slice::<T>() }
    }

    /// Borrow the half-open row range `[start, end)` as a typed slice `&[T]`.
    ///
    /// Forwards to [`BlobVec::as_slice_range`], which builds the slice directly
    /// from the offset element base so that disjoint row ranges of one column
    /// can be lent to different threads (the chunk-parallel slice path, design
    /// §8.3) without ever forming an overlapping whole-column borrow.
    ///
    /// # Safety
    /// `T` must be the exact component type stored here; `start <= end <=
    /// len()`; and the caller must hold shared access to rows `[start, end)`
    /// for the slice's lifetime.
    #[inline]
    pub unsafe fn as_slice_range<T>(&self, start: usize, end: usize) -> &[T] {
        // SAFETY: forwarded type + range + access contract.
        unsafe { self.data.as_slice_range::<T>(start, end) }
    }

    /// Borrow the half-open row range `[start, end)` as a typed mutable slice
    /// `&mut [T]`.
    ///
    /// Like [`Column::as_mut_slice`] this stamps no change ticks; the
    /// chunk-parallel slice path stamps per-row ticks for the covered range
    /// before handing out the slice.
    ///
    /// # Safety
    /// `T` must be the exact component type stored here; `start <= end <=
    /// len()`; and the caller must hold unique access to rows `[start, end)`
    /// for the slice's lifetime (sibling ranges may be borrowed concurrently).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_mut_slice_range<T>(&self, start: usize, end: usize) -> &mut [T] {
        // SAFETY: forwarded type + range + unique-access contract.
        unsafe { self.data.as_mut_slice_range::<T>(start, end) }
    }

    /// Append a value by moving `size` bytes from `value`, stamping both its
    /// added and changed ticks with `change_tick` (a brand-new value).
    ///
    /// # Safety
    /// `value` points to a valid, initialized value of this column's type;
    /// ownership transfers into the column.
    #[inline]
    pub unsafe fn push(&mut self, value: *const u8, change_tick: Tick) {
        // SAFETY: forwarded contract.
        unsafe { self.data.push(value) }
        self.added_ticks.push(UnsafeCell::new(change_tick));
        self.changed_ticks.push(UnsafeCell::new(change_tick));
        self.chunk.on_push(self.data.len(), change_tick);
    }

    /// Append a value by moving its bytes from `value`, stamping its added and
    /// changed ticks *independently* with `added` and `changed`.
    ///
    /// Unlike [`Column::push`] (which stamps a brand-new value with a single
    /// tick for both), this preserves the exact original tick pair of a value
    /// being re-materialized, so a world [`restore`](crate::World::restore)
    /// reproduces change-detection state bit-for-bit (design §14).
    ///
    /// # Safety
    /// `value` points to a valid, initialized value of this column's type;
    /// ownership transfers into the column.
    #[inline]
    pub unsafe fn push_with_ticks(&mut self, value: *const u8, added: Tick, changed: Tick) {
        // SAFETY: forwarded contract.
        unsafe { self.data.push(value) }
        self.added_ticks.push(UnsafeCell::new(added));
        self.changed_ticks.push(UnsafeCell::new(changed));
        // Coarse per-chunk version upper-bounds the newest changed tick.
        self.chunk.on_push(self.data.len(), changed);
    }

    /// Overwrite the value at `row`, dropping the previous one (last-wins), and
    /// advance its changed tick to `change_tick`. The added tick is preserved:
    /// the value has existed since its original insertion, it was merely
    /// written again.
    ///
    /// # Safety
    /// `row < len()` and `value` points to a valid value of this column's
    /// type whose ownership transfers into the column.
    #[inline]
    pub unsafe fn replace(&mut self, row: usize, value: *const u8, change_tick: Tick) {
        // SAFETY: forwarded contract.
        unsafe { self.data.replace(row, value) }
        *self.changed_ticks[row].get_mut() = change_tick;
        self.chunk.bump_row(row, change_tick);
    }

    /// The tick at which the value at `row` was first added.
    ///
    /// # Panics
    /// Panics if `row >= len()`.
    #[inline]
    pub fn added_tick(&self, row: usize) -> Tick {
        // SAFETY: shared read of the cell; any writer holds unique access to
        // this row per the column's access discipline, so no `&mut` aliases.
        unsafe { *self.added_ticks[row].get() }
    }

    /// The tick at which the value at `row` was last changed.
    ///
    /// # Panics
    /// Panics if `row >= len()`.
    #[inline]
    pub fn changed_tick(&self, row: usize) -> Tick {
        // SAFETY: shared read of the cell; see [`Column::added_tick`].
        unsafe { *self.changed_ticks[row].get() }
    }

    /// The added/changed [`ComponentTicks`] pair for the value at `row`.
    ///
    /// # Panics
    /// Panics if `row >= len()`.
    #[inline]
    pub fn component_ticks(&self, row: usize) -> ComponentTicks {
        ComponentTicks {
            added: self.added_tick(row),
            changed: self.changed_tick(row),
        }
    }

    /// Byte size of one element of this column's component type.
    #[inline]
    pub fn item_size(&self) -> usize {
        self.data.item_layout().size()
    }

    /// Raw pointer to the changed-tick cell at `row`, for interior-mutable
    /// stamping through a shared `&Column` (the `&mut T` / `Mut<T>` fetch).
    ///
    /// # Safety
    /// `row < len()` and the caller must hold unique access to this row (the
    /// same discipline as [`Column::get_mut`]).
    #[inline]
    pub unsafe fn changed_tick_ptr(&self, row: usize) -> *mut Tick {
        self.changed_ticks[row].get()
    }

    /// Raw pointer to the per-chunk changed-version cell owning `row`, for
    /// interior-mutable stamping through a shared `&Column` (the coarse half of
    /// the `&mut T` / `Mut<T>` fetch's write; see [`ChunkVersions`]).
    ///
    /// # Safety
    /// `row < len()` and the caller must hold unique access to this row (the
    /// same discipline as [`Column::changed_tick_ptr`]). Only shared atomic
    /// methods may be used through the pointer; never replace the cell.
    #[inline]
    pub unsafe fn chunk_changed_ptr(&self, row: usize) -> *mut ChunkChange {
        // SAFETY: forwarded contract (`row < len`, unique access to the row).
        unsafe { self.chunk.chunk_changed_ptr(row) }
    }

    /// Representative endpoint of chunk `chunk`'s changed-tick interval.
    /// Use [`Self::chunk_may_have_changed`] for filtering and
    /// [`Self::chunk_revision`] for cache invalidation, not tick ordering.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn chunk_version(&self, chunk: usize) -> Tick {
        self.chunk.version(chunk)
    }

    /// Whether this chunk can contain a row changed within the observer window.
    #[inline]
    pub fn chunk_may_have_changed(&self, chunk: usize, last: Tick, now: Tick) -> bool {
        self.chunk.may_have_changed(chunk, last, now)
    }

    /// Visit matching chunks through the column's hierarchical interval index.
    /// Clean subtrees are skipped without inspecting their individual chunks.
    pub fn visit_changed_chunks(
        &self,
        last: Tick,
        now: Tick,
        visit: impl FnMut(usize),
    ) -> ChunkVisitStats {
        self.chunk.visit_changed(last, now, visit)
    }

    /// First possibly changed chunk in `[start, end)`, using the hierarchical
    /// interval index to skip cold gaps. The range is clamped to live chunks.
    #[inline]
    pub fn next_changed_chunk(
        &self,
        start: usize,
        end: usize,
        last: Tick,
        now: Tick,
    ) -> Option<usize> {
        self.chunk.next_changed(start, end, last, now).0
    }

    /// Write revision independent of replayed or wrapping ticks.
    #[inline]
    pub fn chunk_revision(&self, chunk: usize) -> u64 {
        self.chunk.revision(chunk)
    }

    /// Number of logical chunk windows this column currently spans.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunk.chunk_count()
    }

    /// Rows per logical chunk window (constant for the column's lifetime).
    #[inline]
    pub fn rows_per_chunk(&self) -> usize {
        self.chunk.rows_per_chunk()
    }

    /// Stamp the changed tick of the value at `row` (used by structural writes
    /// that already hold `&mut Column`, e.g. [`crate::world::World::get_mut`]).
    ///
    /// # Panics
    /// Panics if `row >= len()`.
    #[inline]
    pub fn set_changed_tick(&mut self, row: usize, change_tick: Tick) {
        *self.changed_ticks[row].get_mut() = change_tick;
        self.chunk.bump_row(row, change_tick);
    }

    /// Clamp every stored tick against `this_run` so none can wrap around and
    /// masquerade as recent (see [`Tick::check_tick`]). Run periodically.
    pub fn check_change_ticks(&mut self, this_run: Tick) {
        for cell in &mut self.added_ticks {
            cell.get_mut().check_tick(this_run);
        }
        for cell in &mut self.changed_ticks {
            cell.get_mut().check_tick(this_run);
        }
        let rpc = self.rows_per_chunk();
        for chunk in 0..self.chunk_count() {
            let start = chunk * rpc;
            let end = ((chunk + 1) * rpc).min(self.len());
            self.chunk.rebuild(
                chunk,
                self.changed_ticks[start..end].iter().map(|cell| {
                    // SAFETY: the exclusive column borrow prevents live row handles.
                    unsafe { *cell.get() }
                }),
                this_run,
            );
        }
    }
}

/// Columnar storage for one archetype.
///
/// Values live in **physical chunk blocks** (design §5.3): one allocation per
/// row window holds *every* column's slice for that window (a DOTS-style chunk),
/// plus the per-chunk metadata sidecar. The [`Column`]s are borrowed views into
/// those blocks — they never call the global allocator; [`Table::allocate`] and
/// [`Table::swap_remove_row`] grow and shrink the blocks and attach/detach each
/// column's slice.
pub struct Table {
    entities: Vec<Entity>,
    columns: HashMap<ComponentId, Column>,
    /// Shared logical chunk window size for every column (design §7). Stored on
    /// the table so chunk-index/row-window math is identical across columns.
    rows_per_chunk: usize,
    /// The owned physical chunk blocks, in row-window order. Each block's header
    /// embeds the per-column `change_version` cells and the chunk's
    /// [`ChunkMeta`] (design §5.3), so there is no separate metadata allocation.
    /// Declared *after* `columns` so the borrowed column views drop before the
    /// blocks they point into.
    chunks: Vec<ChunkBlock>,
    /// Per column: `(component id, byte offset of its change-version cell, byte
    /// offset of its value slice)` inside a chunk block. Computed once (the
    /// column set is fixed per archetype).
    column_slots: Vec<(ComponentId, usize, usize)>,
    /// Byte offset of the chunk's [`ChunkMeta`] inside a chunk block.
    meta_offset: usize,
    /// Owns (and recycles) the chunk block allocations. With the `vm` feature
    /// this is a page-aligned (and where available huge-page-backed) arena;
    /// otherwise it is a thin global-allocator wrapper.
    arena: ChunkArena,
}

/// One physical chunk: a single allocation holding every column's slice for one
/// row window (design §5.3 DOTS chunk). The allocation is owned by the table's
/// [`ChunkArena`](crate::storage::arena::ChunkArena), which frees it.
struct ChunkBlock {
    ptr: NonNull<u8>,
    incarnation: u64,
}

// SAFETY: a chunk block only holds component data for `Component: Send + Sync`
// types (or dynamically-registered data the caller promises is `Send + Sync`)
// plus the chunk's change-version cells; it is aliased and mutated under the
// same unique-per-row discipline the columns that view it already uphold (see
// the `Sync` impl on `Column`). The raw pointer is an owning allocation, not
// shared mutable state on its own.
unsafe impl Send for ChunkBlock {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for ChunkBlock {}

/// One chunk window's row range plus its embedded [`ChunkMeta`], as yielded by
/// [`Table::iter_chunks`] (design §5.3 / §13 / §15).
#[derive(Clone, Copy)]
pub struct ChunkInfo<'a> {
    /// Stable identity of this physical block allocation.
    ///
    /// This is distinct from [`ChunkMeta::structural_epoch`]: an allocator may
    /// recycle the same address and a restored table may reuse the same epoch,
    /// but a newly-created block must still invalidate resident GPU mirrors.
    pub incarnation: u64,
    /// Absolute row of the window's first live entity.
    pub first_row: usize,
    /// Number of live rows in the window (`<= rows_per_chunk`).
    pub rows: usize,
    /// The window's metadata (bounds / visible / lod / batch_key / cell).
    pub meta: &'a ChunkMeta,
}

/// One contiguous span of a table column that must be re-uploaded to a
/// GPU-resident **mirror of that column** (design §15).
///
/// The mirror is a flat `row * element_size` buffer; each chunk's slice is a
/// separate block, so an upload is a small list of `(buffer_offset, src, len)`
/// copies rather than one whole-column copy. Only chunks whose per-chunk
/// `change_version` falls in the `(last_upload, now]` window are emitted, so the
/// cost is proportional to the changed chunks, not the column size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColumnUploadSpan {
    /// Byte offset of the span in the column's linear mirror
    /// (`first_row_in_chunk * element_size`).
    pub buffer_offset: usize,
    /// Host address of the span's first byte (the chunk block's slice start).
    pub src: *const u8,
    /// Bytes to copy for this span (`rows_in_chunk * element_size`).
    pub len: usize,
}

impl Table {
    /// Create an empty table with a column per `(id, layout, drop)` descriptor.
    ///
    /// The logical chunk window size (rows per chunk, design §5.3/§7) is derived
    /// from the combined per-row byte size of all columns so one chunk's worth
    /// of a row is roughly [`TARGET_CHUNK_BYTES`](crate::storage::TARGET_CHUNK_BYTES).
    /// The *same* window size is handed to every column, so chunk index `c` maps
    /// to the identical row range `[c * rpc, (c + 1) * rpc)` across all columns —
    /// the invariant the dirty-chunk accessor relies on to slice a whole table
    /// row-window at once.
    pub fn new(columns: impl IntoIterator<Item = (ComponentId, Layout, Option<DropFn>)>) -> Self {
        let descriptors: Vec<(ComponentId, Layout, Option<DropFn>)> = columns.into_iter().collect();
        let bytes_per_row: usize = descriptors.iter().map(|(_, layout, _)| layout.size()).sum();
        let rpc = rows_per_chunk(bytes_per_row);
        let mut map = HashMap::default();
        // One chunk block holds: [ per-column change-version cells | ChunkMeta |
        // column value slices ] (design §5.3: a chunk embeds its change_version
        // and metadata). Each column slice is aligned to its own element
        // alignment; the whole block is cache-line aligned.
        let ncols = descriptors.len();
        let block_align = descriptors
            .iter()
            .map(|(_, layout, _)| layout.align())
            .max()
            .unwrap_or(1)
            .max(64);
        let tick_size = size_of::<ChunkChange>();
        let versions_bytes = ncols * tick_size;
        let meta_offset = versions_bytes.next_multiple_of(align_of::<ChunkMeta>());
        let values_offset = (meta_offset + size_of::<ChunkMeta>()).next_multiple_of(block_align);
        let mut rel = 0usize;
        let mut slots = Vec::with_capacity(ncols);
        for (i, (id, layout, _)) in descriptors.iter().enumerate() {
            rel = rel.next_multiple_of(layout.align());
            slots.push((*id, i * tick_size, values_offset + rel));
            rel += layout.size() * rpc;
        }
        let block_layout = Layout::from_size_align(values_offset + rel, block_align)
            .expect("invalid chunk block layout");
        for (id, layout, drop) in descriptors {
            // Columns are borrowed views into the table's chunk blocks.
            map.insert(id, Column::new(layout, drop, rpc));
        }
        Self {
            entities: Vec::new(),
            columns: map,
            rows_per_chunk: rpc,
            chunks: Vec::new(),
            column_slots: slots,
            meta_offset,
            arena: ChunkArena::new(block_layout),
        }
    }

    /// Rows per logical chunk window, shared by every column (design §7).
    #[inline]
    pub fn rows_per_chunk(&self) -> usize {
        self.rows_per_chunk
    }

    /// Number of logical chunk windows spanning the current rows
    /// (`ceil(len / rows_per_chunk)`), i.e. the valid chunk-index range for
    /// [`Table::column`]-level chunk-version queries.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.entities.len().div_ceil(self.rows_per_chunk)
    }

    /// Number of rows (live entities) in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Rows this table can hold before any of its allocations must grow.
    ///
    /// The minimum capacity across the entity column and every component column,
    /// so it is a sound bound on "push this many rows without reallocating".
    #[inline]
    pub fn capacity(&self) -> usize {
        self.columns
            .values()
            .map(Column::capacity)
            .fold(self.entities.capacity(), usize::min)
    }

    /// Reserve room for `additional` more rows in every column and the entity
    /// column (design §17 原型预热).
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.entities.reserve(additional);
        // Chunk blocks are the table's own allocations; grow them up front so
        // the borrowed columns already have their slices attached.
        self.grow_chunks_to(self.entities.len() + additional);
        for column in self.columns.values_mut() {
            column.reserve(additional);
        }
    }

    /// Allocate chunk blocks until the table can hold `len` rows, attaching each
    /// column's slice of every new block (design §5.3 DOTS chunk).
    fn grow_chunks_to(&mut self, len: usize) {
        let needed = len.div_ceil(self.rows_per_chunk);
        while self.chunks.len() < needed {
            let block = ChunkBlock {
                ptr: self.arena.alloc(),
                incarnation: next_chunk_incarnation(),
            };
            // Embed a default `ChunkMeta` in the block's header.
            // SAFETY: `meta_offset` is laid out inside `block_layout` and aligned
            // for `ChunkMeta`; the slot is not yet initialised.
            unsafe {
                core::ptr::write(
                    block.ptr.as_ptr().add(self.meta_offset).cast::<ChunkMeta>(),
                    ChunkMeta::default(),
                );
            }
            for (id, cell_offset, slice_offset) in &self.column_slots {
                if let Some(column) = self.columns.get_mut(id) {
                    // SAFETY: both offsets were laid out inside `block_layout`,
                    // the cell is aligned for `ChunkChange` and the slice base for the
                    // column's element type; the block is kept alive in `chunks`.
                    unsafe {
                        let cell = block.ptr.as_ptr().add(*cell_offset).cast::<ChunkChange>();
                        core::ptr::write(cell, ChunkChange::new());
                        let base = block.ptr.as_ptr().add(*slice_offset);
                        column.attach_chunk(cell, base);
                    }
                }
            }
            self.chunks.push(block);
        }
    }

    /// Detach and free trailing chunk blocks until the table holds exactly
    /// `ceil(len / rows_per_chunk)` of them.
    fn shrink_chunks_to(&mut self, len: usize) {
        let needed = len.div_ceil(self.rows_per_chunk);
        while self.chunks.len() > needed {
            // Detach the columns' views first, then drop (free) the block.
            for (id, _, _) in &self.column_slots {
                if let Some(column) = self.columns.get_mut(id) {
                    column.detach_chunk();
                }
            }
            let block = self
                .chunks
                .pop()
                .expect("a block was just checked to exist");
            // SAFETY: the block came from `self.arena` and every column has been
            // detached, so no live reference into it remains.
            unsafe { self.arena.free(block.ptr) };
        }
    }

    /// Per-chunk metadata for every chunk, in chunk order (design §5.3
    /// `ChunkMeta`, embedded in each block's header). Index `c` covers rows
    /// `[c * rows_per_chunk, (c + 1) * rows_per_chunk)`.
    #[inline]
    pub fn chunk_meta_iter(&self) -> impl Iterator<Item = &ChunkMeta> + '_ {
        let offset = self.meta_offset;
        self.chunks.iter().map(move |block| {
            // SAFETY: the block's header embeds a `ChunkMeta` at `offset`.
            unsafe { &*block.ptr.as_ptr().add(offset).cast::<ChunkMeta>() }
        })
    }

    /// Iterate the table's chunks as `(row range, metadata)` pairs — the shape a
    /// GPU-facing consumer wants (design §5.3 / §13 / §15): each item carries the
    /// window's first row and row count plus its embedded [`ChunkMeta`], so a
    /// culling pass can accept or reject a whole window, and an uploader can
    /// address that window's column slices, without touching per-entity data.
    #[inline]
    pub fn iter_chunks(&self) -> impl Iterator<Item = ChunkInfo<'_>> + '_ {
        let rows_per_chunk = self.rows_per_chunk;
        let total = self.entities.len();
        let meta_offset = self.meta_offset;
        self.chunks.iter().enumerate().map(move |(chunk, block)| {
            let first_row = chunk * rows_per_chunk;
            // SAFETY: the block's header embeds a `ChunkMeta` at `meta_offset`.
            let meta = unsafe { &*block.ptr.as_ptr().add(meta_offset).cast::<ChunkMeta>() };
            ChunkInfo {
                incarnation: block.incarnation,
                first_row,
                rows: total.saturating_sub(first_row).min(rows_per_chunk),
                meta,
            }
        })
    }

    /// Append the **dirty upload spans** of table column `id` to `out`, for a
    /// GPU-resident mirror of that column (design §15).
    ///
    /// A chunk contributes a span exactly when its embedded `change_version`
    /// falls in the `(last_upload, now]` change-detection window (the same
    /// predicate the query-side dirty-chunk accessor uses), so a frame that
    /// touched a few chunks re-uploads a few chunk-sized copies instead of the
    /// whole column. Returns the number of spans appended. Columns not present
    /// in this archetype (or zero-sized ones) append nothing.
    ///
    /// The span's `buffer_offset` addresses the *linear* mirror while `src`
    /// addresses the chunk block, so the caller's copy is
    /// `dst[buffer_offset .. buffer_offset + len] = src[..len]`.
    pub fn column_upload_spans(
        &self,
        id: ComponentId,
        last_upload: Tick,
        now: Tick,
        out: &mut Vec<ColumnUploadSpan>,
    ) -> usize {
        let Some(column) = self.columns.get(&id) else {
            return 0;
        };
        let stride = column.item_size();
        if stride == 0 {
            return 0; // a marker component owns no bytes to upload
        }
        let Some(&(_, _, slice_offset)) = self
            .column_slots
            .iter()
            .find(|(component, _, _)| *component == id)
        else {
            return 0;
        };
        let before = out.len();
        let rows_per_chunk = self.rows_per_chunk;
        let total = self.entities.len();
        column.visit_changed_chunks(last_upload, now, |chunk| {
            let block = &self.chunks[chunk];
            let first_row = chunk * rows_per_chunk;
            let rows = total.saturating_sub(first_row).min(rows_per_chunk);
            if rows == 0 {
                return;
            }
            out.push(ColumnUploadSpan {
                buffer_offset: first_row * stride,
                // SAFETY: the block layout places this column's slice at
                // `slice_offset`, inside the block's allocation; the caller only
                // reads `len` bytes from it.
                src: unsafe { block.ptr.as_ptr().add(slice_offset).cast_const() },
                len: rows * stride,
            });
        });
        out.len() - before
    }

    /// Recompute the per-chunk `bounds` of table column `id` from its values
    /// (design §5.3 `ChunkMeta` AABB; feeds §13 culling / §15 GPU upload).
    ///
    /// `point` projects a component value to a world-space point; each chunk's
    /// bounds is the union of its rows' points. A window is recomputed **only
    /// when it is stale** — its cached `bounds_epoch` no longer matches the
    /// chunk's structural epoch (a row moved in/out), the source column differs,
    /// or the column's independent write revision differs (including writes in
    /// the same tick). An untouched frame costs one staleness test per chunk.
    /// Pass `force` when the projection changes, including captured parameters;
    /// a closure's semantics cannot be inferred from its type or address.
    /// Returns the number of windows recomputed.
    ///
    /// # Safety
    /// If `id` is present, its column must contain initialized values of `T`.
    /// A matching size or alignment alone does not establish type identity.
    /// Prefer the checked `World::recompute_chunk_bounds` entry point.
    pub unsafe fn recompute_chunk_bounds<T: Component>(
        &mut self,
        id: ComponentId,
        _now: Tick,
        point: &mut impl FnMut(&T) -> [f32; 3],
        force: bool,
    ) -> usize {
        let Some(column) = self.columns.get(&id) else {
            return 0; // this archetype has no such column
        };
        debug_assert_eq!(
            column.item_size(),
            size_of::<T>(),
            "the component type must match the column",
        );
        let rows_per_chunk = self.rows_per_chunk;
        let total = self.entities.len();
        let meta_offset = self.meta_offset;
        let mut recomputed = 0usize;
        for (chunk, block) in self.chunks.iter().enumerate() {
            // SAFETY: the block header embeds a `ChunkMeta` at `meta_offset`;
            // `&mut self` makes this the only access to it.
            let meta = unsafe { &mut *block.ptr.as_ptr().add(meta_offset).cast::<ChunkMeta>() };
            let version = column.chunk_version(chunk);
            let revision = column.chunk_revision(chunk);
            let stale = force
                || meta.bounds_source() != Some(id)
                || meta.bounds_epoch() != meta.structural_epoch()
                || revision != meta.bounds_revision();
            if !stale {
                continue;
            }
            let first_row = chunk * rows_per_chunk;
            let rows = total.saturating_sub(first_row).min(rows_per_chunk);
            if rows == 0 {
                continue;
            }
            let mut bounds: Option<ChunkBounds> = None;
            for row in first_row..first_row + rows {
                // SAFETY: `row < total == column.len()` and the column stores `T`
                // (the caller's contract), so this reads a live value of T.
                let p = point(unsafe { column.get::<T>(row) });
                match &mut bounds {
                    Some(b) => b.union(&ChunkBounds::new(p, p)),
                    None => bounds = Some(ChunkBounds::new(p, p)),
                }
            }
            let epoch = meta.structural_epoch();
            meta.set_column_bounds(bounds, epoch, version, revision, id);
            recomputed += 1;
        }
        recomputed
    }

    /// The metadata of chunk `chunk`, or `None` if out of range.
    #[inline]
    pub fn chunk_meta_checked(&self, chunk: usize) -> Option<&ChunkMeta> {
        let block = self.chunks.get(chunk)?;
        // SAFETY: the block's header embeds a `ChunkMeta` at `meta_offset`.
        Some(unsafe { &*block.ptr.as_ptr().add(self.meta_offset).cast::<ChunkMeta>() })
    }

    /// The metadata of chunk `chunk`.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn chunk_meta(&self, chunk: usize) -> &ChunkMeta {
        self.chunk_meta_checked(chunk)
            .expect("chunk index out of range")
    }

    /// Mutable access to chunk `chunk`'s metadata, for consumers that cache
    /// derived fields ([`ChunkMeta::bounds`], LOD, batch key, cell) there.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn chunk_meta_mut(&mut self, chunk: usize) -> &mut ChunkMeta {
        let offset = self.meta_offset;
        let block = self
            .chunks
            .get_mut(chunk)
            .expect("chunk index out of range");
        // SAFETY: `&mut self` makes this the sole access to the block, whose
        // header embeds a `ChunkMeta` at `offset`.
        unsafe { &mut *block.ptr.as_ptr().add(offset).cast::<ChunkMeta>() }
    }

    /// Bump the structural epoch of the chunk owning `row`, if it is still live.
    /// Called on every row-membership change.
    #[inline]
    fn bump_chunk_meta(&mut self, row: usize) {
        let chunk = row / self.rows_per_chunk;
        if chunk < self.chunks.len() {
            // SAFETY: `chunk` is in range and the header embeds a `ChunkMeta`.
            unsafe {
                (*self
                    .chunks
                    .get_unchecked_mut(chunk)
                    .ptr
                    .as_ptr()
                    .add(self.meta_offset)
                    .cast::<ChunkMeta>())
                .bump_structural();
            }
        }
    }

    /// Cull this table's chunk windows against `frustum`, writing each window's
    /// [`ChunkMeta::visible`] bit (design §13 / §15).
    ///
    /// A window that has computed [`bounds`](ChunkMeta::bounds) is culled exactly
    /// when its AABB lies fully outside the frustum (a conservative test: a box
    /// that merely touches a plane stays visible). A window whose bounds have not
    /// been produced yet is left **visible** — a render pass must never silently
    /// drop work it has not classified. Pair this with
    /// [`recompute_chunk_bounds`](Table::recompute_chunk_bounds) so the bounds are
    /// current, then feed the table to
    /// [`GpuBatchBuilder::build_from_table`](crate::gpu_batch::GpuBatchBuilder::build_from_table).
    pub fn cull_chunks(
        &mut self,
        frustum: &crate::culling::Frustum,
    ) -> crate::culling::ChunkCullStats {
        let meta_offset = self.meta_offset;
        let mut stats = crate::culling::ChunkCullStats::default();
        for block in self.chunks.iter_mut() {
            // SAFETY: the block header embeds a `ChunkMeta` at `meta_offset`, and
            // `&mut self` makes this the only access to it.
            let meta = unsafe { &mut *block.ptr.as_ptr().add(meta_offset).cast::<ChunkMeta>() };
            stats.tested += 1;
            let visible = match meta.bounds {
                Some(bounds) => frustum.intersects_aabb(bounds.min, bounds.max),
                // Unclassified window: keep it (conservative).
                None => true,
            };
            meta.visible = visible;
            if visible {
                stats.visible += 1;
            } else {
                stats.culled += 1;
            }
        }
        stats
    }

    /// Whether the table has no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The entities stored, indexed by row.
    #[inline]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// The entity at `row`, if any.
    #[inline]
    pub fn entity_at(&self, row: usize) -> Option<Entity> {
        self.entities.get(row).copied()
    }

    /// Whether this table has a column for component `id`.
    #[inline]
    pub fn has_column(&self, id: ComponentId) -> bool {
        self.columns.contains_key(&id)
    }

    /// Immutable access to the column for `id`.
    #[inline]
    pub fn column(&self, id: ComponentId) -> Option<&Column> {
        self.columns.get(&id)
    }

    /// Mutable access to the column for `id`.
    #[inline]
    pub fn column_mut(&mut self, id: ComponentId) -> Option<&mut Column> {
        self.columns.get_mut(&id)
    }

    /// Begin a new row for `entity`, returning its row index.
    ///
    /// After calling this the caller **must** append exactly one value to every
    /// column (via [`Column::push`] / [`Table::move_row_into`]) so the table
    /// invariant (all columns the same length as `entities`) is restored before
    /// any read. This is an internal primitive used by the world's structural
    /// change paths.
    #[inline]
    pub fn allocate(&mut self, entity: Entity) -> usize {
        let row = self.entities.len();
        self.entities.push(entity);
        // Provide the borrowed columns' slice for this row's window *before* the
        // caller fills them.
        self.grow_chunks_to(self.entities.len());
        // A brand-new row joins its window: bump the window's structural epoch
        // (design §5.3 `ChunkMeta`, embedded in the chunk header).
        self.bump_chunk_meta(row);
        row
    }

    /// Mutable access to the column `id`, intended to be paired with
    /// [`Table::allocate`] to fill a freshly-allocated row.
    #[inline]
    pub fn column_for_fill(&mut self, id: ComponentId) -> &mut Column {
        self.columns
            .get_mut(&id)
            .expect("archetype column must exist for fill")
    }

    /// Clamp every column's stored change-detection ticks against `this_run`
    /// (see [`Column::check_change_ticks`]). Run periodically from the world.
    pub fn check_change_ticks(&mut self, this_run: Tick) {
        for col in self.columns.values_mut() {
            col.check_change_ticks(this_run);
        }
    }

    /// Move every column value shared between `src` (at `src_row`) and `self`
    /// into a freshly-[`allocate`](Self::allocate)d row of `self`.
    ///
    /// For components present in `self` but absent from `src` (newly added),
    /// nothing is written here — the caller fills those via
    /// [`Table::column_for_fill`]. Components present in `src` but absent from
    /// `self` (removed) are dropped by the subsequent
    /// [`Table::swap_remove_row`] on `src`.
    ///
    /// # Safety
    /// `src_row < src.len()`, and the two tables must agree on the layout of
    /// every shared column (guaranteed when both derive from the same component
    /// registry).
    pub unsafe fn move_shared_columns_from(&mut self, src: &mut Table, src_row: usize) {
        for (id, dst_col) in self.columns.iter_mut() {
            if let Some(src_col) = src.columns.get(id) {
                // SAFETY: shared layouts (same registry) and `src_row` in-bounds
                // per the caller's contract; `push_from_column` copies one value
                // (and its ticks) out and leaves the source slot for
                // `swap_remove_row` to forget.
                unsafe { dst_col.push_from_column(src_col, src_row) };
            }
        }
    }

    /// Remove `row` from every column and the entity list by swapping the last
    /// row into its place. Dropping is applied to every column value at `row`
    /// that was **not** already moved out.
    ///
    /// Returns the entity that was relocated from the last row into `row`, if
    /// the removed row was not already the last one. The caller uses this to
    /// patch that entity's recorded location.
    ///
    /// # Safety
    /// `row < len()`. If any column value at `row` was previously moved out via
    /// [`move_shared_columns_from`], that column must be passed in
    /// `already_moved` so it is swapped without a double drop.
    pub unsafe fn swap_remove_row(
        &mut self,
        row: usize,
        already_moved: &[ComponentId],
    ) -> Option<Entity> {
        debug_assert!(row < self.entities.len());
        for (id, col) in self.columns.iter_mut() {
            if already_moved.contains(id) {
                // Value already relocated to another table; drop the hole by
                // moving the last element over it without running drop glue.
                // SAFETY: `row < len`; copy-out into a throwaway stack slot of
                // the right size is avoided by using a no-drop swap: we emulate
                // it by copying the last element over `row`.
                unsafe { col.data_swap_remove_forget(row) };
            } else {
                // SAFETY: `row < len`; value still owned here, drop it.
                unsafe { col.data_swap_remove_and_drop(row) };
            }
        }
        let last = self.entities.len() - 1;
        let moved = if row != last {
            Some(self.entities[last])
        } else {
            None
        };
        self.entities.swap_remove(row);
        // Row membership changed: the destination window lost its old row (and,
        // unless `row` was the last row, gained the former last row), while the
        // last row's window lost a row. Bump each distinct affected window once
        // and drop any window that no longer holds rows (design §5.3 `ChunkMeta`).
        let after = self.entities.len();
        // Drop any chunk block whose last row just left (its columns' views are
        // detached, then the block is freed).
        self.shrink_chunks_to(after);
        self.bump_chunk_meta(row);
        if row != last && row / self.rows_per_chunk != last / self.rows_per_chunk {
            self.bump_chunk_meta(last);
        }
        moved
    }
}

// Internal column plumbing kept here so the `BlobVec` surface stays minimal and
// the table is the only place that drives structural moves.
impl Column {
    /// Copy the value at `src_row` of `src` into a fresh slot of this column,
    /// carrying its change-detection ticks along unchanged (a relocation
    /// preserves the value's identity, so it is neither re-added nor changed).
    ///
    /// # Safety
    /// Same contract as [`BlobVec::push_from`]: identical layouts and
    /// `src_row < src.len()`.
    unsafe fn push_from_column(&mut self, src: &Column, src_row: usize) {
        // SAFETY: forwarded contract from `Table::move_shared_columns_from`.
        unsafe { self.data.push_from(&src.data, src_row) }
        self.added_ticks
            .push(UnsafeCell::new(src.added_tick(src_row)));
        let moved_changed = src.changed_tick(src_row);
        self.changed_ticks.push(UnsafeCell::new(moved_changed));
        // A relocation preserves the value's identity, so its changed tick
        // carries over unchanged; fold it into the destination chunk so the
        // upper-bound invariant holds for the grown window.
        self.chunk.on_push(self.data.len(), moved_changed);
    }

    /// Swap-remove and drop the value at `row`, keeping the tick vectors in
    /// lockstep.
    ///
    /// # Safety
    /// `row < len()`; the value at `row` is still owned by this column.
    unsafe fn data_swap_remove_and_drop(&mut self, row: usize) {
        let last = self.changed_ticks.len() - 1;
        let moved_changed = self.changed_tick(last);
        // SAFETY: forwarded contract from `Table::swap_remove_row`.
        unsafe { self.data.swap_remove_and_drop(row) }
        self.added_ticks.swap_remove(row);
        self.changed_ticks.swap_remove(row);
        self.chunk
            .on_swap_remove(row, self.changed_ticks.len(), moved_changed);
    }

    /// Swap-remove the value at `row` without dropping it, keeping the tick
    /// vectors in lockstep.
    ///
    /// # Safety
    /// `row < len()`; the value at `row` must already have been moved out, so
    /// forgetting it here avoids a double drop.
    unsafe fn data_swap_remove_forget(&mut self, row: usize) {
        let last = self.changed_ticks.len() - 1;
        let moved_changed = self.changed_tick(last);
        // SAFETY: forwarded contract; the value at `row` was already moved out,
        // so swapping the last element over it (without drop) leaves one owner.
        unsafe { self.data.swap_remove_forget(row) }
        self.added_ticks.swap_remove(row);
        self.changed_ticks.swap_remove(row);
        self.chunk
            .on_swap_remove(row, self.changed_ticks.len(), moved_changed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::storage::ChunkBounds;
    use crate::world::World;

    /// 2048 B/row -> `TARGET_CHUNK_BYTES / 2048 == 8` rows per chunk window, so
    /// a handful of spawns crosses a window boundary without a huge allocation.
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy)]
    struct Wide([u8; 2048]);
    impl Component for Wide {}

    #[test]
    fn chunk_meta_tracks_windows_and_structural_epochs() {
        let mut world = World::new();
        let arch = world.prewarm::<Wide>(0);
        assert_eq!(
            world
                .archetypes()
                .get(arch)
                .unwrap()
                .table()
                .rows_per_chunk(),
            8,
        );

        // No rows yet -> no windows, hence no metadata.
        assert!(world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .chunk_meta_iter()
            .next()
            .is_none());

        // Fill exactly one window; every spawn changes that window's row set.
        let mut first_window = Vec::new();
        for _ in 0..8 {
            first_window.push(world.spawn(Wide([3; 2048])));
        }
        {
            let t = world.archetypes().get(arch).unwrap().table();
            assert_eq!(t.chunk_count(), 1);
            assert_eq!(t.chunk_meta_iter().count(), 1);
            assert_eq!(t.chunk_meta(0).structural_epoch(), 8);
            assert!(t.chunk_meta(0).visible);
            assert!(t.chunk_meta(0).bounds.is_none());
        }

        // Spill one row into a second window; only the new window's epoch starts.
        let spill = world.spawn(Wide([4; 2048]));
        {
            let t = world.archetypes().get(arch).unwrap().table();
            assert_eq!(t.chunk_count(), 2);
            assert_eq!(t.chunk_meta_iter().count(), 2);
            assert_eq!(
                t.chunk_meta(0).structural_epoch(),
                8,
                "first window untouched"
            );
            assert_eq!(t.chunk_meta(1).structural_epoch(), 1);
        }

        // A consumer caches derived metadata; the kernel stores it verbatim.
        {
            let m = world
                .chunk_meta_mut(arch, 1)
                .expect("chunk 1 exists after the spill");
            m.bounds = Some(ChunkBounds::new([0.0; 3], [1.0; 3]));
            m.lod = 2;
            m.batch_key = 0xDEAD_BEEF;
            m.cell = 7;
        }

        // Despawning the lone row of window 1 drops that window entirely; the
        // surviving window's row set did not change, so its epoch holds.
        assert!(world.despawn(spill));
        {
            let t = world.archetypes().get(arch).unwrap().table();
            assert_eq!(t.chunk_count(), 1);
            assert_eq!(t.chunk_meta_iter().count(), 1);
            assert_eq!(t.chunk_meta(0).structural_epoch(), 8);
        }

        // Removing a non-last row swap-moves the window's last row into its
        // place: the row set changed, so the epoch advances and any cached
        // `bounds` must be treated as stale by consumers.
        assert!(world.despawn(first_window[0]));
        {
            let t = world.archetypes().get(arch).unwrap().table();
            assert_eq!(t.chunk_count(), 1);
            assert_eq!(t.chunk_meta(0).structural_epoch(), 9);
        }
    }

    /// A `Big` row is 512 B, so a `Big`-only table has
    /// `TARGET_CHUNK_BYTES / 512 == 32` rows per chunk — a handful of entities
    /// crosses several block boundaries.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Big(u64, [u8; 504]);
    impl Component for Big {}

    /// A sparse-free marker used to force archetype moves (insert/remove).
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Tag(u8);
    impl Component for Tag {}

    /// Differential test across many chunk windows: random spawns, despawns and
    /// component inserts/removes must leave every live entity's `Big` value equal
    /// to a reference map, however rows shuffle across block boundaries and
    /// archetype moves.
    #[test]
    fn world_structural_churn_matches_a_reference() {
        use crate::collections::HashMap;

        let mut world = World::new();
        let mut reference: HashMap<Entity, u64> = HashMap::default();
        let mut live: Vec<Entity> = Vec::new();
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng >> 33
        };
        let mut tag: u64 = 0;

        for step in 0..4_000u64 {
            match next() % 4 {
                // Spawn a fresh entity holding the tag we remember.
                0 | 1 => {
                    tag += 1;
                    let e = world.spawn(Big(tag, [0; 504]));
                    reference.insert(e, tag);
                    live.push(e);
                }
                // Despawn a random live entity (swap-removes a row, possibly
                // across blocks).
                2 if !live.is_empty() => {
                    let i = (next() as usize) % live.len();
                    let e = live.swap_remove(i);
                    assert!(world.despawn(e), "live entity must despawn");
                    assert!(reference.remove(&e).is_some());
                }
                // Force an archetype move by toggling `Tag`.
                3 if !live.is_empty() => {
                    let i = (next() as usize) % live.len();
                    let e = live[i];
                    if world.get::<Tag>(e).is_some() {
                        assert!(world.remove::<Tag>(e));
                    } else {
                        assert!(world.insert(e, Tag(1)));
                    }
                }
                _ => {}
            }

            assert_eq!(reference.len(), live.len());
            if step % 37 == 0 {
                for (&e, &expect) in &reference {
                    assert_eq!(
                        world.get::<Big>(e).map(|b| b.0),
                        Some(expect),
                        "entity {e:?} diverged at step {step}",
                    );
                }
            }
        }
    }

    /// The DOTS-layout invariant (design §5.3): one chunk is **one** allocation
    /// holding every column's slice, so the table's block count equals its chunk
    /// count — not chunk count × column count.
    #[test]
    fn one_block_per_chunk_holds_every_column() {
        let mut world = World::new();
        let arch = world.prewarm::<(Wide, Big)>(0);
        // Wide 2048 B + Big 512 B = 2560 B/row -> floor(16 KiB / 2560) = 6 rows.
        let rows_per_chunk = world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .rows_per_chunk();
        assert_eq!(rows_per_chunk, 6);
        assert!(world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .chunks
            .is_empty());

        // Cross three chunk boundaries.
        for _ in 0..(rows_per_chunk * 3) {
            world.spawn((Wide([1; 2048]), Big(7, [0; 504])));
        }
        let table = world.archetypes().get(arch).unwrap().table();
        assert_eq!(table.chunk_count(), 3);
        assert_eq!(
            table.chunks.len(),
            3,
            "one allocation per chunk, not one per column",
        );
    }

    /// The chunk's `change_version` cell and its `ChunkMeta` are **embedded in
    /// the chunk block header** (design §5.3): both live inside the block's
    /// allocation, not in a separate Vec.
    #[test]
    fn chunk_headers_embed_versions_and_meta() {
        let mut world = World::new();
        let arch = world.prewarm::<Wide>(0);
        for _ in 0..8 {
            world.spawn(Wide([0; 2048]));
        }
        let table = world.archetypes().get(arch).unwrap().table();
        assert_eq!(table.chunk_count(), 1);
        assert_eq!(table.chunks.len(), 1);

        let block = &table.chunks[0];
        let lo = block.ptr.as_ptr() as usize;
        let hi = lo + table.arena.block_size();

        let column = table.columns.values().next().unwrap();
        // SAFETY: row 0 is live (8 rows) so the chunk cell exists.
        let cell = unsafe { column.chunk.chunk_changed_ptr(0) } as usize;
        assert!(
            (lo..hi).contains(&cell),
            "the change-version cell must live inside the chunk block",
        );
        let meta = table.chunk_meta(0) as *const ChunkMeta as usize;
        assert!(
            (lo..hi).contains(&meta),
            "ChunkMeta must live inside the chunk block",
        );
        // They occupy distinct header slots.
        assert_ne!(cell, meta);
    }

    /// The GPU-resident upload feed (design §15): only chunks whose per-chunk
    /// `change_version` falls in the window produce a span, and each span
    /// addresses that chunk's slice of the column in the linear mirror.
    #[test]
    fn column_upload_spans_track_dirty_chunks() {
        let mut world = World::new();
        let arch = world.prewarm::<Big>(0);
        // Big is 512 B -> 32 rows per chunk; spawn exactly three windows.
        let mut entities = Vec::new();
        for i in 0..96u64 {
            entities.push(world.spawn(Big(i, [0; 504])));
        }
        let id = world
            .components()
            .id_of::<Big>()
            .expect("Big is registered");

        // Nothing has been uploaded, so every window is dirty.
        let now = world.change_tick();
        let mut spans = Vec::new();
        let n = world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .column_upload_spans(id, Tick::ZERO, now, &mut spans);
        assert_eq!(n, 3);
        assert_eq!(spans.len(), 3);
        for (chunk, span) in spans.iter().enumerate() {
            assert_eq!(span.buffer_offset, chunk * 32 * 512);
            assert_eq!(span.len, 32 * 512);
            assert!(!span.src.is_null());
        }

        // Open a fresh observer window, touch one row in the middle window, then
        // close it: only that window's change_version lands in (last, now].
        let last = world.change_tick();
        world.increment_change_tick();
        world.get_mut::<Big>(entities[40]).unwrap().0 = 999;
        let now = world.change_tick();
        spans.clear();
        let n = world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .column_upload_spans(id, last, now, &mut spans);
        assert_eq!(n, 1, "only the touched window re-uploads");
        assert_eq!(spans[0].buffer_offset, 32 * 512);
        assert_eq!(spans[0].len, 32 * 512);

        // A following window with no writes in it emits nothing.
        let before_quiet = now;
        world.increment_change_tick();
        spans.clear();
        let quiet = world.change_tick();
        let n = world
            .archetypes()
            .get(arch)
            .unwrap()
            .table()
            .column_upload_spans(id, before_quiet, quiet, &mut spans);
        assert_eq!(n, 0, "no writes means no upload spans");
    }

    /// A 512-byte component carrying a world-space point, so a chunk holds 32
    /// rows and a small test spans several windows.
    #[allow(dead_code)] // the pad only fixes the row size
    #[derive(Clone, Copy)]
    struct Pt([f32; 3], [u8; 500]);
    impl Component for Pt {}

    #[test]
    fn bounds_cache_distinguishes_source_columns_with_equal_revisions() {
        struct OtherPoint([f32; 3]);
        impl Component for OtherPoint {}

        let mut world = World::new();
        let entity = world.spawn((Pt([1.0, 2.0, 3.0], [0; 500]), OtherPoint([9.0; 3])));
        let arch = world.entities().location(entity).unwrap().archetype_id;
        let first = world.components().id_of::<Pt>().unwrap();
        let second = world.components().id_of::<OtherPoint>().unwrap();
        let table = world.archetypes().get(arch).unwrap().table();
        assert_eq!(
            table.column(first).unwrap().chunk_revision(0),
            table.column(second).unwrap().chunk_revision(0),
            "different columns may legitimately have the same revision",
        );
        let now = world.change_tick();
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            1
        );
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            0
        );
        assert_eq!(
            world.recompute_chunk_bounds::<OtherPoint>(arch, now, &mut |p| p.0, false),
            1,
        );
        let meta = world.chunk_meta(arch, 0).unwrap();
        assert_eq!(meta.bounds, Some(ChunkBounds::new([9.0; 3], [9.0; 3])));
        assert_eq!(meta.bounds_source(), Some(second));
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            1
        );
        assert_eq!(
            world.chunk_meta(arch, 0).unwrap().bounds_source(),
            Some(first)
        );
        assert_eq!(
            world.chunk_meta(arch, 0).unwrap().bounds,
            Some(ChunkBounds::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])),
        );
    }

    #[test]
    fn bounds_cache_observes_same_tick_writes_with_historical_change_stamp() {
        let mut world = World::new();
        let entity = world.spawn(Pt([1.0; 3], [0; 500]));
        let arch = world.entities().location(entity).unwrap().archetype_id;
        let now = world.change_tick();
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            1
        );
        for coordinate in [5.0, 10.0] {
            let mut value = world.get_mut::<Pt>(entity).unwrap();
            value.0 = [coordinate; 3];
            value.set_last_changed(Tick::new(0));
            assert_eq!(world.change_tick(), now);
            assert_eq!(
                world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
                1
            );
            assert_eq!(
                world.chunk_meta(arch, 0).unwrap().bounds,
                Some(ChunkBounds::new([coordinate; 3], [coordinate; 3])),
            );
            assert_eq!(
                world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
                0
            );
        }
    }

    #[test]
    fn bounds_cache_invalidates_manual_bounds_and_supports_projection_changes() {
        let mut world = World::new();
        let entity = world.spawn(Pt([2.0; 3], [0; 500]));
        let arch = world.entities().location(entity).unwrap().archetype_id;
        let now = world.change_tick();
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            1
        );
        let meta = world.chunk_meta_mut(arch, 0).unwrap();
        let epoch = meta.structural_epoch();
        meta.set_bounds(Some(ChunkBounds::new([-1.0; 3], [-1.0; 3])), epoch, now);
        assert_eq!(meta.bounds_source(), None);
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0, false),
            1
        );
        // The source has not changed, but the caller changes the projection.
        // `force` is the explicit invalidation contract for captured state.
        let offset = 8.0;
        assert_eq!(
            world.recompute_chunk_bounds::<Pt>(arch, now, &mut |p| p.0.map(|x| x + offset), true),
            1,
        );
        assert_eq!(
            world.chunk_meta(arch, 0).unwrap().bounds,
            Some(ChunkBounds::new([10.0; 3], [10.0; 3])),
        );
    }

    #[test]
    fn recompute_chunk_bounds_produces_and_caches_aabbs() {
        let mut world = World::new();
        let arch = world.prewarm::<Pt>(0);
        let mut entities = Vec::new();
        for i in 0..96 {
            // x ascends, y descends, z stays 0.
            entities.push(world.spawn(Pt([i as f32, -(i as f32), 0.0], [0; 500])));
        }
        let mut point = |p: &Pt| p.0;

        // The first pass computes all three windows.
        let now = world.change_tick();
        let n = world.recompute_chunk_bounds::<Pt>(arch, now, &mut point, false);
        assert_eq!(n, 3);
        {
            let table = world.archetypes().get(arch).unwrap().table();
            assert_eq!(
                table.chunk_meta(0).bounds,
                Some(ChunkBounds::new([0.0, -31.0, 0.0], [31.0, 0.0, 0.0])),
            );
            assert_eq!(
                table.chunk_meta(2).bounds,
                Some(ChunkBounds::new([64.0, -95.0, 0.0], [95.0, -64.0, 0.0])),
            );
        }

        // Nothing moved: every window's cache is still valid.
        let n = world.recompute_chunk_bounds::<Pt>(arch, now, &mut point, false);
        assert_eq!(n, 0, "clean windows are skipped");

        // Move one row in the middle window: only that window is stale.
        world.increment_change_tick();
        world.get_mut::<Pt>(entities[40]).unwrap().0 = [1000.0, 0.0, 0.0];
        let now = world.change_tick();
        let n = world.recompute_chunk_bounds::<Pt>(arch, now, &mut point, false);
        assert_eq!(n, 1, "only the window whose value moved is recomputed");
        assert_eq!(
            world
                .archetypes()
                .get(arch)
                .unwrap()
                .table()
                .chunk_meta(1)
                .bounds,
            Some(ChunkBounds::new([32.0, -63.0, 0.0], [1000.0, 0.0, 0.0])),
        );

        // A structural change (a row entering a window) invalidates that window
        // via its structural epoch, even though no value changed.
        world.increment_change_tick();
        world.spawn(Pt([-5.0, 0.0, 0.0], [0; 500]));
        let now = world.change_tick();
        let n = world.recompute_chunk_bounds::<Pt>(arch, now, &mut point, false);
        assert_eq!(n, 1, "the window that gained a row is recomputed");
        assert_eq!(
            world
                .archetypes()
                .get(arch)
                .unwrap()
                .table()
                .chunk_meta(3)
                .bounds,
            Some(ChunkBounds::new([-5.0, 0.0, 0.0], [-5.0, 0.0, 0.0])),
        );
    }
}
