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
    archetypes: Vec<ArchetypeId>,
    /// Index into `archetypes` of the archetype currently being walked.
    arch_cursor: usize,
    /// Next row to yield within the current archetype.
    row: usize,
    /// Number of rows in the current archetype.
    row_len: usize,
    /// The current archetype's resolved fetch cursor (`None` before the first
    /// archetype is entered, or for a zero-row archetype that is skipped).
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
        archetypes: Vec<ArchetypeId>,
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

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.row_len {
                let row = self.row;
                self.row += 1;
                let entity = self.current_entities[row];
                let filter_fetch = self
                    .current_filter_fetch
                    .expect("current_filter_fetch is Some whenever row_len > 0");
                // SAFETY: `row < self.row_len == archetype.len()` and
                // `filter_fetch` was built for this archetype. `F::Fetch` is
                // `Copy`, so passing it by value does not disturb the cursor.
                if !unsafe { F::filter_fetch(filter_fetch, entity, row) } {
                    // Row rejected by `Added`/`Changed`/`With`/`Without`; the
                    // cursor already advanced, so just skip it.
                    continue;
                }
                let fetch = self
                    .current_fetch
                    .expect("current_fetch is Some whenever row_len > 0");
                // SAFETY: `row < self.row_len == archetype.len()`, `fetch` was
                // built for this archetype, and the row cursor only advances —
                // so this `(archetype, row)` is fetched exactly once. Any `&mut`
                // term therefore yields a unique reference, valid for `'w`
                // because the world is exclusively borrowed for `'w`.
                return Some(unsafe { D::fetch(fetch, entity, row) });
            }

            if self.arch_cursor >= self.archetypes.len() {
                return None;
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

            self.row = 0;
            self.row_len = archetype.len();
            self.current_entities = archetype.table().entities();
            // SAFETY: `arch_id` came from the matched list, so the archetype
            // satisfies `D::matches` for `data_state` — the contract of
            // `init_fetch`.
            self.current_fetch =
                Some(unsafe { D::init_fetch(self.data_state, archetype, self.last_run, self.this_run) });
            // SAFETY: the same `arch_id` also satisfies `F::matches` for
            // `filter_state` (guaranteed by `matched_archetypes`), the contract
            // of `F::init_fetch`.
            self.current_filter_fetch = Some(unsafe {
                F::init_fetch(self.filter_state, archetype, self.last_run, self.this_run)
            });
        }
    }
}
