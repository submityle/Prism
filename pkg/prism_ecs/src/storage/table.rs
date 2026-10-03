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

use crate::change::{ComponentTicks, Tick};
use crate::collections::HashMap;
use crate::component::{ComponentId, DropFn};
use crate::entity::Entity;
use crate::storage::blob_vec::BlobVec;
use crate::storage::chunk::{rows_per_chunk, ChunkVersions};

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
    fn new(layout: Layout, drop: Option<DropFn>, rows_per_chunk: usize) -> Self {
        Self {
            data: BlobVec::new(layout, drop),
            added_ticks: Vec::new(),
            changed_ticks: Vec::new(),
            chunk: ChunkVersions::new(rows_per_chunk),
        }
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
    /// same discipline as [`Column::changed_tick_ptr`]). The pointer must only
    /// be written with `this_run`, which upholds the chunk upper-bound invariant.
    #[inline]
    pub unsafe fn chunk_changed_ptr(&self, row: usize) -> *mut Tick {
        // SAFETY: forwarded contract (`row < len`, unique access to the row).
        unsafe { self.chunk.chunk_changed_ptr(row) }
    }

    /// The changed-version of chunk `chunk` within this column.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn chunk_version(&self, chunk: usize) -> Tick {
        self.chunk.version(chunk)
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
        self.chunk.check_ticks(this_run);
    }
}

/// Columnar storage for one archetype.
pub struct Table {
    entities: Vec<Entity>,
    columns: HashMap<ComponentId, Column>,
    /// Shared logical chunk window size for every column (design §7). Stored on
    /// the table so chunk-index/row-window math is identical across columns.
    rows_per_chunk: usize,
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
        for (id, layout, drop) in descriptors {
            map.insert(id, Column::new(layout, drop, rpc));
        }
        Self {
            entities: Vec::new(),
            columns: map,
            rows_per_chunk: rpc,
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
