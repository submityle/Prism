//! Logical chunk change-versions: the coarse half of the double-layer change
//! detection described in the design doc (§7 脏块访问器, §10 chunk 版本号).
//!
//! # What a "chunk" is here
//!
//! The design's end state is a physical 16 KiB chunk ([`§5.3`]). M2 does not yet
//! re-layout storage into physical chunks: each archetype is still ONE
//! contiguous [`Table`](crate::storage::Table). Instead this module layers a
//! *logical* chunk model over that contiguous storage — a chunk is simply a
//! fixed-size window of `rows_per_chunk` consecutive rows — and tracks one
//! change-version [`Tick`] per window. Every column of a table is sliced into
//! the same windows (same `rows_per_chunk`), so chunk index `c` always maps to
//! the same row range `[c * rpc, (c + 1) * rpc)` across every column.
//!
//! # Core invariant
//!
//! For every chunk `c`:
//!
//! ```text
//! version(c) is at least as recent as the newest changed-tick of any row
//!            currently stored in chunk c's row window
//! ```
//!
//! ("At least as recent" is measured in the wrapping sense of
//! [`Tick::is_newer_than`].) This makes the chunk version a correct *upper
//! bound*: if any row in the chunk was changed within a system's
//! `(last_run, this_run]` window, the chunk version also falls in (or after)
//! that window, so a chunk-level test never produces a false negative. It may
//! over-approximate (flag a chunk whose only recently-changed row has since been
//! removed), which the design explicitly permits — the coarse layer yields a
//! *superset* of changed rows and the per-row ticks remain the fine filter.
//!
//! # Interior mutability
//!
//! A chunk version cell is shared by every row in the chunk, so the `&mut T` /
//! [`Mut`](crate::change::Mut) fetch cannot thread a `&mut Tick` to it: two
//! `Mut`s into the same chunk (e.g. `iter_mut().collect()`) would alias the one
//! cell, which is undefined behaviour. The cells therefore live behind
//! [`UnsafeCell`] and are bumped through raw `*mut Tick` writes, exactly as the
//! per-row ticks on [`Column`](crate::storage::Column) are. Single-threaded
//! iteration writes `this_run` to the shared cell sequentially (idempotent); a
//! future parallel iterator splits *disjoint* chunks across threads, so each
//! thread writes a distinct cell.

use alloc::vec::Vec;
use core::cell::UnsafeCell;

use crate::change::Tick;

/// Target physical chunk size the logical windows approximate (design §5.3).
///
/// The row count per chunk is chosen so one chunk's worth of a single column is
/// roughly this many bytes, matching the DOTS-style 16 KiB chunk the physical
/// layout will eventually use.
pub const TARGET_CHUNK_BYTES: usize = 16 * 1024;

/// Hard cap on rows per chunk, so a zero-sized (marker) component — which has no
/// natural byte-derived limit — still partitions into finite windows.
const MAX_ROWS_PER_CHUNK: usize = 1 << 16;

/// Rows per logical chunk for a column whose element is `bytes_per_row` bytes.
///
/// Zero-sized rows (marker components, or a table of only ZSTs) fall back to the
/// [`MAX_ROWS_PER_CHUNK`] cap; otherwise the count is `TARGET_CHUNK_BYTES /
/// bytes_per_row`, clamped to at least one row and at most the cap.
#[inline]
pub fn rows_per_chunk(bytes_per_row: usize) -> usize {
    match TARGET_CHUNK_BYTES.checked_div(bytes_per_row) {
        // A zero-sized row (division by zero) has no byte-derived limit.
        None => MAX_ROWS_PER_CHUNK,
        Some(rows) => rows.clamp(1, MAX_ROWS_PER_CHUNK),
    }
}

/// Per-chunk change-versions for one column, partitioning the column's rows into
/// fixed `rows_per_chunk` windows.
///
/// The vector grows and shrinks in lockstep with the column's row count: it
/// always holds exactly `ceil(len / rows_per_chunk)` cells.
pub struct ChunkVersions {
    rows_per_chunk: usize,
    /// One changed-version cell per chunk window. `changed[c]` upper-bounds the
    /// newest changed-tick among rows `[c * rpc, (c + 1) * rpc)`.
    changed: Vec<UnsafeCell<Tick>>,
}

// SAFETY: the only non-`Sync` field is the `UnsafeCell<Tick>` version vector.
// Those cells are mutated under the same unique-per-row discipline that governs
// the per-row ticks on `Column` (itself `unsafe impl Sync`): a writer stamping a
// row holds unique access to that row and writes `this_run` through a raw
// pointer, and no reader observes a cell mid-write. Distinct chunks are distinct
// cells, so a future parallel split over disjoint chunks never races.
unsafe impl Sync for ChunkVersions {}

impl ChunkVersions {
    /// A fresh, empty version set partitioning rows into `rows_per_chunk`
    /// windows.
    ///
    /// # Panics
    /// Panics if `rows_per_chunk` is zero.
    #[inline]
    pub fn new(rows_per_chunk: usize) -> Self {
        assert!(rows_per_chunk > 0, "rows_per_chunk must be non-zero");
        Self {
            rows_per_chunk,
            changed: Vec::new(),
        }
    }

    /// The fixed row-window size.
    #[inline]
    pub fn rows_per_chunk(&self) -> usize {
        self.rows_per_chunk
    }

    /// Number of chunk windows currently tracked (`ceil(len / rpc)`).
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.changed.len()
    }

    /// The chunk index owning `row`.
    #[inline]
    pub fn chunk_of(&self, row: usize) -> usize {
        row / self.rows_per_chunk
    }

    /// The number of chunk windows needed to hold `len` rows.
    #[inline]
    fn chunks_for_len(&self, len: usize) -> usize {
        len.div_ceil(self.rows_per_chunk)
    }

    /// The changed-version of chunk `chunk`.
    ///
    /// # Panics
    /// Panics if `chunk >= chunk_count()`.
    #[inline]
    pub fn version(&self, chunk: usize) -> Tick {
        // SAFETY: shared read of the cell; any writer holds unique access to the
        // chunk's rows and writes through the raw pointer, so no `&mut` aliases
        // this read (mirrors `Column::changed_tick`).
        unsafe { *self.changed[chunk].get() }
    }

    /// Raw pointer to the changed-version cell owning `row`, for interior-mutable
    /// stamping through a shared `&self` (the `&mut T` / `Mut<T>` fetch).
    ///
    /// # Safety
    /// `row` must be `< len` of the owning column, and the caller must hold
    /// unique access to `row` (the same discipline as
    /// [`Column::changed_tick_ptr`](crate::storage::Column::changed_tick_ptr)).
    /// The returned pointer must only be *written* (never used to form a shared
    /// `&Tick` that outlives a concurrent write) and only with `this_run`, which
    /// is the newest tick and so preserves the upper-bound invariant.
    #[inline]
    pub unsafe fn chunk_changed_ptr(&self, row: usize) -> *mut Tick {
        self.changed[self.chunk_of(row)].get()
    }

    /// Fold `tick` into the version of the chunk owning `row`, keeping
    /// whichever is more recent (never lowering the stored upper bound). Used by
    /// the structural write paths that already hold `&mut Column`
    /// (overwrite-in-place, structural tick stamping).
    ///
    /// # Panics
    /// Panics if `row` is not covered by an existing chunk window.
    #[inline]
    pub fn bump_row(&mut self, row: usize, tick: Tick) {
        let chunk = self.chunk_of(row);
        self.bump_chunk(chunk, tick);
    }

    /// Fold `tick` into chunk `chunk`'s version, keeping whichever is more
    /// recent (never lowering the stored upper bound).
    #[inline]
    fn bump_chunk(&mut self, chunk: usize, tick: Tick) {
        let cell = self.changed[chunk].get_mut();
        // A newer write wins. Raw comparison is exact while `this_run` has not
        // wrapped past the stored value; `check_ticks` keeps every stored tick
        // within one `MAX_CHANGE_AGE` window so the version can only ever
        // over-approximate (a correct superset), never skip a changed chunk.
        if tick.get() > cell.get() {
            *cell = tick;
        }
    }

    /// Account for a freshly appended row (the column just grew to `new_len`),
    /// whose value carries `tick` as its changed-tick.
    ///
    /// Grows the version vector to cover the new row and folds `tick` into the
    /// (possibly new) last chunk so the upper-bound invariant holds.
    pub fn on_push(&mut self, new_len: usize, tick: Tick) {
        debug_assert!(new_len >= 1);
        let needed = self.chunks_for_len(new_len);
        while self.changed.len() < needed {
            self.changed.push(UnsafeCell::new(Tick::ZERO));
        }
        let last_row = new_len - 1;
        let chunk = self.chunk_of(last_row);
        self.bump_chunk(chunk, tick);
    }

    /// Account for a swap-remove: the row at `removed_row` was removed and (if it
    /// was not already the last row) the former last row — carrying `moved_tick`
    /// — was swapped into its place. `new_len` is the row count *after* removal.
    ///
    /// Folds `moved_tick` into the destination chunk (the relocated row may be
    /// newer than that chunk's current bound) and then shrinks the version
    /// vector to `ceil(new_len / rpc)`.
    pub fn on_swap_remove(&mut self, removed_row: usize, new_len: usize, moved_tick: Tick) {
        // A row actually moved into `removed_row` only when it was not the last
        // row; after removal that position is in-range exactly when
        // `removed_row < new_len`.
        if removed_row < new_len {
            let chunk = self.chunk_of(removed_row);
            self.bump_chunk(chunk, moved_tick);
        }
        let needed = self.chunks_for_len(new_len);
        self.changed.truncate(needed);
    }

    /// Clamp every chunk version against `this_run` so none can wrap around and
    /// masquerade as recent (mirrors [`Column::check_change_ticks`]).
    pub fn check_ticks(&mut self, this_run: Tick) {
        for cell in &mut self.changed {
            cell.get_mut().check_tick(this_run);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_per_chunk_scales_with_byte_size() {
        // 16 KiB / 16 B = 1024 rows.
        assert_eq!(rows_per_chunk(16), TARGET_CHUNK_BYTES / 16);
        // Huge rows clamp to at least one row per chunk.
        assert_eq!(rows_per_chunk(TARGET_CHUNK_BYTES * 2), 1);
        // Zero-sized rows fall back to the hard cap.
        assert_eq!(rows_per_chunk(0), MAX_ROWS_PER_CHUNK);
    }

    #[test]
    fn push_grows_chunks_and_bounds_versions() {
        let rpc = 4;
        let mut cv = ChunkVersions::new(rpc);
        assert_eq!(cv.chunk_count(), 0);

        // Fill the first chunk; each push bumps chunk 0 to a newer tick.
        for row in 0..rpc {
            cv.on_push(row + 1, Tick::new((row + 1) as u32));
        }
        assert_eq!(cv.chunk_count(), 1);
        assert_eq!(cv.version(0), Tick::new(rpc as u32));

        // One more row spills into a second chunk.
        cv.on_push(rpc + 1, Tick::new(99));
        assert_eq!(cv.chunk_count(), 2);
        assert_eq!(cv.chunk_of(rpc), 1);
        assert_eq!(cv.version(1), Tick::new(99));
        // The first chunk's bound is unchanged.
        assert_eq!(cv.version(0), Tick::new(rpc as u32));
    }

    #[test]
    fn bump_never_lowers_the_bound() {
        let mut cv = ChunkVersions::new(4);
        cv.on_push(1, Tick::new(10));
        // A stale (older) tick must not lower the recorded upper bound.
        cv.on_push(2, Tick::new(3));
        assert_eq!(cv.version(0), Tick::new(10));
    }

    #[test]
    fn swap_remove_folds_moved_tick_and_shrinks() {
        let rpc = 4;
        let mut cv = ChunkVersions::new(rpc);
        // Two full chunks: rows 0..8.
        for row in 0..(2 * rpc) {
            cv.on_push(row + 1, Tick::new(1));
        }
        assert_eq!(cv.chunk_count(), 2);

        // Remove row 0; the former last row (tick 42) swaps into chunk 0.
        cv.on_swap_remove(0, 2 * rpc - 1, Tick::new(42));
        assert_eq!(cv.version(0), Tick::new(42));
        // 7 rows still need 2 chunks.
        assert_eq!(cv.chunk_count(), 2);

        // Drain the second chunk entirely (rows 7,6,5,4 removed as last rows).
        for len in (rpc..(2 * rpc - 1)).rev() {
            cv.on_swap_remove(len, len, Tick::new(0));
        }
        assert_eq!(cv.chunk_count(), 1);
    }

    #[test]
    fn check_ticks_clamps_versions() {
        let mut cv = ChunkVersions::new(4);
        cv.on_push(1, Tick::new(1));
        let now = Tick::new(Tick::MAX_CHANGE_AGE + 100);
        cv.check_ticks(now);
        assert_eq!(cv.version(0).age_since(now), Tick::MAX_CHANGE_AGE);
    }
}
