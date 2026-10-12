//! [`QuerySliceIter`]: sequential, chunk-aligned **column-slice** iteration
//! (design §7 "SIMD 迭代", §8.3).
//!
//! This is the serial sibling of
//! [`par_chunks`](crate::query::QueryState::par_chunks): instead of dispatching
//! each row range to a worker it hands them back one at a time, so a single
//! thread can run the `simd` bulk kernels
//! ([`add_assign`](crate::query::add_assign) / [`axpy_assign`](crate::query::axpy_assign)
//! / [`scale_assign`](crate::query::scale_assign) / [`sum`](crate::query::sum))
//! over contiguous memory without the per-row `Mut` wrapper:
//!
//! ```ignore
//! for (positions, velocities) in q.simd_iter::<1024>(&world) {
//!     prism_ecs::query::axpy_assign(positions, velocities, dt);
//! }
//! ```
//!
//! The iterator does **not** pack values into a SIMD register type: the slice
//! *is* the vector input, and the kernels pick the fastest backend for the
//! target (`core::arch` intrinsics with a scalar golden path). Chunk size is the
//! const parameter, so it is free at runtime.
//!
//! # Why it is slice-shaped
//!
//! Per-row filters cannot be expressed over a slice, so — exactly like
//! `par_chunks` — the filter must be an [`ArchetypalFilter`] (its verdict is
//! decided once per archetype) and every data term must be
//! [`ColumnSliceData`] (table-backed or `Entity`). Both are checked once, up
//! front, with a clear panic; there is no silent fallback to a slow path.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::query::slice::{ArchetypalFilter, ColumnSliceData};
use crate::world::World;

/// A sequential iterator over the chunk-aligned column slices of a query.
///
/// Created by [`QueryState::simd_iter`](crate::query::QueryState::simd_iter)
/// (shared view, read-only terms) or
/// [`QueryState::simd_iter_mut`](crate::query::QueryState::simd_iter_mut)
/// (exclusive view, `&mut T` terms). Yields `D::Slice<'w>` per slice of at most
/// `K` rows, never crossing a physical chunk boundary (so at most
/// `min(K, rows_per_chunk)` rows); the last slice may be shorter.
pub struct QuerySliceIter<'w, 's, D: ColumnSliceData, F: ArchetypalFilter> {
    /// The world under iteration; valid for `'w`. Only read through shared
    /// derefs — `&mut [T]` slices are formed by the column helpers, exactly as
    /// in the parallel driver.
    world: *mut World,
    /// Resolved component ids for the data terms, borrowed from the state.
    data_state: &'s D::State,
    /// Archetypes this query matches, computed once at creation.
    archetypes: Vec<ArchetypeId>,
    /// Index of the archetype being walked (`archetypes[arch_cursor - 1]`).
    arch_cursor: usize,
    /// Next row of the current archetype to hand out.
    row: usize,
    /// Row count of the current archetype.
    row_len: usize,
    /// Rows per slice (`K`).
    chunk: usize,
    /// End of the querying observer's change-detection window.
    this_run: Tick,
    _marker: PhantomData<(&'w World, fn() -> (D, F))>,
}

impl<'w, 's, D: ColumnSliceData, F: ArchetypalFilter> QuerySliceIter<'w, 's, D, F> {
    /// Build a slice iterator over `archetypes` of `world`.
    ///
    /// # Safety
    /// - `world` must be valid for reads for `'w`, and — when any reachable term
    ///   is a `&mut`/`Option<&mut>` term — uniquely borrowed for `'w`, so the
    ///   `&mut [T]` slices never alias.
    /// - every id in `archetypes` must name an archetype that satisfies
    ///   `D::matches` **and** `F::matches` for the respective states.
    pub(crate) unsafe fn new(
        world: *mut World,
        data_state: &'s D::State,
        archetypes: Vec<ArchetypeId>,
        chunk: usize,
        this_run: Tick,
    ) -> Self {
        Self {
            world,
            data_state,
            archetypes,
            arch_cursor: 0,
            row: 0,
            row_len: 0,
            chunk: chunk.max(1),
            this_run,
            _marker: PhantomData,
        }
    }
}

impl<'w, 's, D: ColumnSliceData, F: ArchetypalFilter> Iterator
    for QuerySliceIter<'w, 's, D, F>
{
    type Item = D::Slice<'w>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.row_len {
                let start = self.row;
                let arch_id = self.archetypes[self.arch_cursor - 1];
                // SAFETY: `self.world` is valid and borrowed for `'w`; only a
                // shared `&'w World` is formed here, and `arch_id` came from this
                // same world's matched list.
                let world: &'w World = unsafe { &*self.world };
                let archetype = world
                    .archetypes()
                    .get(arch_id)
                    .expect("matched archetype id must resolve");
                // Clamp the slice to the current physical chunk: a column block
                // holds exactly `rows_per_chunk` values, so a slice spanning two
                // blocks would be non-contiguous (design §5.3).
                let rows_per_chunk = archetype.table().rows_per_chunk().max(1);
                let chunk_end = (start / rows_per_chunk + 1) * rows_per_chunk;
                let end = (start + self.chunk).min(self.row_len).min(chunk_end);
                self.row = end;
                // SAFETY: `start <= end <= archetype.len()`; the caller holds the
                // access these terms require over those rows for `'w` (read-only
                // entry point) or exclusively (mutable entry point), and the
                // ranges are disjoint by construction, so the slice is unique;
                // `[start, end)` lies inside one block, so the column slice is
                // contiguous.
                return Some(unsafe {
                    D::column_slice(self.data_state, archetype, start, end, self.this_run)
                });
            }

            if self.arch_cursor >= self.archetypes.len() {
                return None;
            }
            let arch_id = self.archetypes[self.arch_cursor];
            self.arch_cursor += 1;
            // SAFETY: as above; only a shared borrow is formed to read the row
            // count of the next matched archetype.
            let world: &'w World = unsafe { &*self.world };
            let archetype = world
                .archetypes()
                .get(arch_id)
                .expect("matched archetype id must resolve");
            self.row = 0;
            self.row_len = archetype.len();
        }
    }
}
