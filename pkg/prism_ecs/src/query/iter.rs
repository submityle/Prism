//! [`QueryIter`]: the row-by-row iterator that walks every matched archetype of
//! a query and yields its typed [`QueryData::Item`] per entity row.
//!
//! # Soundness model
//!
//! `QueryIter` holds a `*mut World` plus a `PhantomData<&'w mut World>`. It is
//! only reachable through [`QueryState::iter`](crate::query::QueryState::iter)
//! / [`QueryState::iter_mut`](crate::query::QueryState::iter_mut), which borrow
//! the world for `'w` — read-only iteration takes `&'w World` (and requires
//! [`ReadOnlyQueryData`](crate::query::ReadOnlyQueryData), so only shared
//! `&T`/`Option<&T>`/[`Entity`] terms are reachable), while mutable iteration
//! takes `&'w mut World`. Either way nothing else can mutate the world while
//! iteration is in flight. Within that borrow the iterator reads component
//! storage through shared derefs of the world pointer and the raw-pointer
//! interior-mutability helpers on [`Column`](crate::storage::Column) — the exact
//! pattern the rest of the kernel uses.
//!
//! Each `(archetype, row)` pair is visited exactly once per iteration (the row
//! cursor only advances, archetypes are never revisited), so a `&mut T` term
//! never produces two aliasing references. Internally-conflicting queries
//! (`&mut A` with `&A`, `&mut A` twice) are rejected up front when the
//! [`QueryState`](crate::query::QueryState) builds its [`Access`](crate::query::Access).

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::archetype::{Archetype, ArchetypeId};
use crate::change::Tick;
use crate::entity::Entity;
use crate::query::fetch::QueryData;
use crate::query::filter::QueryFilter;
use crate::world::World;

/// A borrowing iterator over the rows matched by a `Query<D, F>`.
///
/// Yields one [`D::Item`](QueryData::Item) per matched entity, in archetype
/// order then row order. The item borrows the world for the iterator's `'w`
/// lifetime (tied to the world borrow the query was created from); the resolved
/// data-term state is borrowed from the owning [`QueryState`] for `'s`, so the
/// state stays reusable across iterations.
pub struct QueryIter<'w, 's, D: QueryData, F: QueryFilter = ()> {
    /// The world under iteration. Valid for `'w`; only ever read through shared
    /// derefs, and only written through `&mut` fetches when iteration began
    /// from a `&'w mut World`.
    world: *mut World,
    /// Resolved component ids for the data terms, borrowed from the owning
    /// [`QueryState`] so it can be reused after iteration ends.
    data_state: &'s D::State,
    /// Resolved component ids for the filter terms, borrowed from the owning
    /// [`QueryState`]. Drives per-row [`Added`](crate::query::Added) /
    /// [`Changed`](crate::query::Changed) acceptance.
    filter_state: &'s F::State,
    /// Ids of the archetypes this query matches, computed once at creation.
    archetypes: Arc<Vec<ArchetypeId>>,
    /// Index into `archetypes` of the archetype currently being walked.
    arch_cursor: usize,
    /// Next row to yield within the current archetype.
    row: usize,
    /// Number of rows in the current archetype.
    row_len: usize,
    /// Chunk-window size of the current archetype (rows per physical block).
    rows_per_chunk: usize,
    /// End row (exclusive) of the chunk currently being walked.
    chunk_end: usize,
    /// The current archetype's resolved fetch cursor (`None` before the first
    /// archetype is entered, or for a chunk rejected by the coarse filter).
    current_fetch: Option<D::Fetch<'w>>,
    /// The current archetype's resolved filter fetch cursor (parallel to
    /// `current_fetch`). `None` before the first archetype is entered.
    current_filter_fetch: Option<F::Fetch<'w>>,
    /// The current archetype's entity column, used to supply the row entity.
    current_entities: &'w [Entity],
    /// Start of the querying observer's change-detection window (exclusive).
    last_run: Tick,
    /// End of the querying observer's change-detection window (the current
    /// world/system tick).
    this_run: Tick,
    _marker: PhantomData<(&'w mut World, fn() -> F)>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> QueryIter<'w, 's, D, F> {
    /// Build an iterator over `archetypes` of `world`.
    ///
    /// # Safety
    /// - `world` must be valid for reads for `'w`. If any reachable data term is
    ///   a `&mut`/`Option<&mut>` term, `world` must additionally be uniquely
    ///   borrowed for `'w` (the caller holds `&'w mut World`), so no other
    ///   access occurs during iteration.
    /// - every id in `archetypes` must name an archetype that satisfies
    ///   `D::matches` **and** `F::matches` for the respective states (so
    ///   `D::init_fetch` / `F::init_fetch` are sound).
    pub(crate) unsafe fn new(
        world: *mut World,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        archetypes: Arc<Vec<ArchetypeId>>,
        last_run: Tick,
        this_run: Tick,
    ) -> Self {
        Self {
            world,
            data_state,
            filter_state,
            archetypes,
            arch_cursor: 0,
            row: 0,
            row_len: 0,
            rows_per_chunk: 1,
            chunk_end: 0,
            current_fetch: None,
            current_filter_fetch: None,
            current_entities: &[],
            last_run,
            this_run,
            _marker: PhantomData,
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> Iterator for QueryIter<'w, 's, D, F> {
    type Item = D::Item<'w>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_with_entity().map(|(_, item)| item)
    }

    /// Batched `fold` fast path (design §17.1 性能工程: 收敛串行 vs `par(1)`)。
    ///
    /// The default [`Iterator::fold`] funnels every row through `next()`, which
    /// pays the per-row `Option`/return-value protocol and reloads the fetch
    /// cursors from `self` on each call. `for` loops keep using `next` (and so
    /// see no change), but the batch consumers — `for_each`, `sum`, `count`, and
    /// explicit `fold` — go through here instead. This override keeps the exact
    /// same per-row `Added`/`Changed`/sparse gates while walking each archetype
    /// as a tight inner loop with the entity column and both fetch cursors
    /// hoisted into locals, so it runs at roughly the parallel driver's
    /// single-worker batch speed. Row order and the visited set are identical to
    /// [`next`](Iterator::next).
    #[inline]
    fn fold<B, G>(mut self, init: B, mut f: G) -> B
    where
        G: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        loop {
            if self.row < self.chunk_end {
                // Hoist the chunk-local cursors out of the inner loop; a table
                // term's fetch carries the window's value base, so the inner
                // loop is pure offset arithmetic (design §5.3).
                let entities = self.current_entities;
                // SAFETY: `row < chunk_end` implies a chunk is entered, so both
                // cursors are initialised; `chunk_end <= row_len ==
                // entities.len()`.
                let (filter_fetch, fetch) = unsafe {
                    (
                        self.current_filter_fetch.unwrap_unchecked(),
                        self.current_fetch.unwrap_unchecked(),
                    )
                };
                while self.row < self.chunk_end {
                    let row = self.row;
                    self.row += 1;
                    // SAFETY: `row < chunk_end <= row_len == entities.len()` (the loop
                    // condition), so the index is in bounds.
                    let entity = unsafe { *entities.get_unchecked(row) };
                    // SAFETY: `filter_fetch` was built for this archetype and
                    // `row` indexes a live row of it; `F::Fetch` is `Copy`.
                    if !unsafe { F::filter_fetch(filter_fetch, entity, row) } {
                        continue;
                    }
                    // SAFETY: same row/archetype invariants; resolves sparse
                    // membership for sparse-backed terms; `D::Fetch` is `Copy`.
                    if !unsafe { D::filter_fetch(fetch, entity, row) } {
                        continue;
                    }
                    // SAFETY: `row < row_len == archetype.len()`, `fetch` was
                    // built for this archetype, and the row cursor only
                    // advances, so this `(archetype, row)` is fetched exactly
                    // once — any `&mut` term yields a unique reference valid for
                    // `'w`.
                    acc = f(acc, unsafe { D::fetch(fetch, entity, row) });
                }
            }
            // More chunks left in this archetype: resolve the next window's base
            // once and continue. (Advancing per chunk, not per row, is what makes
            // chunked storage cheap.)
            if self.chunk_end < self.row_len {
                self.enter_chunk(self.chunk_end / self.rows_per_chunk);
                continue;
            }
            if !self.enter_next_archetype() {
                return acc;
            }
        }
    }

    /// Upper bound on the rows left to yield.
    ///
    /// The exact count is unknowable without applying the per-row
    /// `Added`/`Changed`/sparse gates, but the sum of the remaining archetypes'
    /// row counts is a sound upper bound (it is exact when those gates admit
    /// every row). Exposing it lets `collect`/`Vec` size its buffer once.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // SAFETY: `self.world` is valid for `'w` (the constructor's contract),
        // and this only forms a shared `&World` to read archetype lengths.
        let world: &World = unsafe { &*self.world };
        let mut upper = self.row_len.saturating_sub(self.row);
        for &id in &self.archetypes[self.arch_cursor..] {
            if let Some(archetype) = world.archetypes().get(id) {
                upper += archetype.len();
            }
        }
        (0, Some(upper))
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> QueryIter<'w, 's, D, F> {
    /// The next matched row together with **its entity**.
    ///
    /// [`Iterator::next`] is exactly this with the entity dropped; the pairing
    /// form exists so drivers that need identities (e.g. combination iteration,
    /// which re-fetches by entity) can reuse this traversal instead of
    /// duplicating the per-row gate.
    #[inline(always)]
    pub(crate) fn next_with_entity(&mut self) -> Option<(Entity, D::Item<'w>)> {
        loop {
            if self.row < self.chunk_end {
                let row = self.row;
                self.row += 1;
                // SAFETY: an accepted chunk initialises both fetch cursors;
                // a rejected chunk sets `row = chunk_end`. Whenever a row remains
                // (`row < chunk_end <= row_len`) the cursors are initialised; and
                // `row < chunk_end <= archetype.len() == current_entities.len()`.
                // Taking the row entity unchecked, and the cursors without the
                // `Option` discriminant test, keeps the per-row path as tight as
                // the batch driver's (both `Fetch` types are `Copy`). This is the
                // hottest loop in the kernel, so the branches that the type
                // system cannot remove for us are removed here instead.
                let (entity, filter_fetch, fetch) = unsafe {
                    (
                        *self.current_entities.get_unchecked(row),
                        self.current_filter_fetch.unwrap_unchecked(),
                        self.current_fetch.unwrap_unchecked(),
                    )
                };
                // SAFETY: `filter_fetch` was built for this archetype and
                // `row` indexes a live row of it; `F::Fetch` is `Copy`.
                if !unsafe { F::filter_fetch(filter_fetch, entity, row) } {
                    // Row rejected by `Added`/`Changed`/`With`/`Without`; the
                    // cursor already advanced, so just skip it.
                    continue;
                }
                // SAFETY: same row/archetype invariants as the filter gate above.
                // A sparse-backed required data term (design §6) resolves its
                // per-entity membership here; table-backed terms always pass.
                // `D::Fetch` is `Copy`, so this read does not disturb the cursor.
                if !unsafe { D::filter_fetch(fetch, entity, row) } {
                    // Entity lacks a required sparse component; skip the row.
                    continue;
                }
                // SAFETY: `row < self.row_len == archetype.len()`, `fetch` was
                // built for this archetype, and the row cursor only advances —
                // so this `(archetype, row)` is fetched exactly once. Any `&mut`
                // term therefore yields a unique reference, valid for `'w`
                // because the world is exclusively borrowed for `'w`.
                return Some((entity, unsafe { D::fetch(fetch, entity, row) }));
            }

            // Current chunk exhausted: step to the next chunk of this archetype
            // if it has more rows, else advance to the next archetype. Resolving
            // the chunk once here (rather than per row in `fetch`) is what keeps
            // chunked storage cheap.
            if self.chunk_end < self.row_len {
                self.enter_chunk(self.chunk_end / self.rows_per_chunk);
                continue;
            }
            if !self.enter_next_archetype() {
                return None;
            }
        }
    }

    /// Advance to the next non-empty matched archetype, resetting the chunk and
    /// row cursors and rebuilding the entity column and both fetch cursors.
    ///
    /// Returns `false` when no archetype remains (the iterator is then
    /// exhausted). Shared by [`next_with_entity`](Self::next_with_entity) and the
    /// batched [`fold`](Iterator::fold) fast path so the per-archetype entry
    /// logic — the safety contracts in particular — lives in exactly one place.
    #[inline(always)]
    fn enter_next_archetype(&mut self) -> bool {
        loop {
            if self.arch_cursor >= self.archetypes.len() {
                return false;
            }
            let arch_id = self.archetypes[self.arch_cursor];
            self.arch_cursor += 1;

            // SAFETY: `self.world` is valid and borrowed for `'w`; we only form
            // a shared `&'w World` to read archetype storage. Each matched
            // archetype id was produced from this same world.
            let world: &'w World = unsafe { &*self.world };
            let archetype: &'w Archetype = world
                .archetypes()
                .get(arch_id)
                .expect("matched archetype id must resolve");
            let len = archetype.len();
            if len == 0 {
                // Nothing to fetch; a zero-row archetype has no chunk.
                continue;
            }
            self.row_len = len;
            self.current_entities = archetype.table().entities();
            self.rows_per_chunk = archetype.table().rows_per_chunk().max(1);
            // SAFETY: this archetype satisfies F::matches. Resolve the filter
            // once per archetype; its metadata cursor is independent of chunks.
            self.current_filter_fetch = Some(unsafe {
                F::init_fetch(
                    self.filter_state,
                    archetype,
                    world.sparse_sets(),
                    self.last_run,
                    self.this_run,
                )
            });
            self.enter_chunk(0);
            return true;
        }
    }

    /// Enter the first candidate at or after `chunk`, skipping clean subtrees
    /// through the filter's index. Exhaustion advances to the archetype's end.
    ///
    /// Table-backed data terms cache the window's value base and
    /// changed-version cell here (design §5.3), so the per-row
    /// [`QueryData::fetch`] is plain offset arithmetic.
    #[inline(always)]
    fn enter_chunk(&mut self, chunk: usize) {
        let filter_fetch = self
            .current_filter_fetch
            .expect("the archetype filter is initialised before entering chunks");
        let Some(chunk) = F::next_candidate_chunk(
            filter_fetch,
            chunk,
            self.row_len.div_ceil(self.rows_per_chunk),
        ) else {
            self.row = self.row_len;
            self.chunk_end = self.row_len;
            self.current_fetch = None;
            return;
        };
        let first = chunk * self.rows_per_chunk;
        self.row = first;
        self.chunk_end = first + self.rows_per_chunk.min(self.row_len - first);

        // SAFETY: `self.world` is valid and borrowed for `'w`; only a shared
        // `&'w World` is formed. `arch_cursor - 1` is the archetype just entered.
        let world: &'w World = unsafe { &*self.world };
        let arch_id = self.archetypes[self.arch_cursor - 1];
        let archetype: &'w Archetype = world
            .archetypes()
            .get(arch_id)
            .expect("matched archetype id must resolve");
        let sparse_sets = world.sparse_sets();
        // SAFETY: `arch_id` came from the matched list, so the archetype
        // satisfies `D::matches` for `data_state` — the contract of
        // `init_fetch_for_chunk`.
        self.current_fetch = Some(unsafe {
            D::init_fetch_for_chunk(
                self.data_state,
                archetype,
                sparse_sets,
                self.last_run,
                self.this_run,
                chunk,
            )
        });
    }
}
