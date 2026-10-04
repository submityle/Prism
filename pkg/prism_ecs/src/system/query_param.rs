//! [`Query`]: the [`SystemParam`] that gives a system typed, filtered iteration
//! over the entities of its world (design §7–§8).
//!
//! A `Query<D, F>` is the system-facing wrapper around a
//! [`QueryState`](crate::query::QueryState). The state (which component ids the
//! query names and the access it needs) is resolved once in
//! [`init_state`](SystemParam::init_state) and cached; each run the param
//! rebuilds a lightweight `Query` that pairs that cached state with the live
//! [`UnsafeWorldCell`] so the body can iterate.
//!
//! Unlike [`World::query`](crate::world::World::query) — which borrows the world
//! directly — a `Query` reaches the world only through the cell and the query
//! driver's raw iterator ([`QueryState::iter_from_ptr`]). This is what lets
//! several disjoint params of one system (or several compatible systems) touch
//! the same world at once under the kernel's access-conflict discipline.

use crate::query::{Access, QueryData, QueryFilter, QueryIter, QueryState, ReadOnlyQueryData};
#[cfg(feature = "multi_thread")]
use crate::query::{ArchetypalFilter, ColumnSliceData};
use crate::system::param::SystemParam;
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::World;

/// Typed, filtered iteration over a world's entities, usable as a system param.
///
/// - `D: `[`QueryData`] is the per-row data (`&T`, `&mut T`,
///   [`Entity`](crate::entity::Entity), `Option<&T>`, tuples).
/// - `F: `[`QueryFilter`] narrows which archetypes are visited
///   ([`With`](crate::query::With) / [`Without`](crate::query::Without) / `Or` /
///   tuples) without yielding data.
///
/// `'w` is the world borrow; `'s` is the borrow of the cached
/// [`QueryState`](crate::query::QueryState).
pub struct Query<'w, 's, D: QueryData, F: QueryFilter = ()> {
    world: UnsafeWorldCell<'w>,
    state: &'s QueryState<D, F>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> Query<'w, 's, D, F> {
    /// Iterate the matched rows with shared access. Available only when every
    /// data term is read-only (`D: `[`ReadOnlyQueryData`]), so multiple shared
    /// iterations may coexist.
    #[inline]
    pub fn iter(&self) -> QueryIter<'_, '_, D, F>
    where
        D: ReadOnlyQueryData,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` fetch is ever formed, and this system's declared read access
        // (upheld by the scheduler's conflict analysis) guarantees nothing
        // writes the columns we read for the duration of the iteration.
        unsafe {
            self.state.iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Iterate the matched rows with exclusive access, permitting `&mut T` data
    /// terms. Takes `&mut self` so the Rust borrow checker forbids holding two
    /// mutable iterators of one query at once.
    #[inline]
    pub fn iter_mut(&mut self) -> QueryIter<'_, '_, D, F> {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live iterator of this query, and the system's declared write access
        // (upheld by the scheduler's conflict analysis) guarantees nothing else
        // touches the columns we write for the duration of the iteration.
        unsafe {
            self.state.iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Visit every matched row with shared access.
    #[inline]
    pub fn for_each(&self, mut f: impl FnMut(D::Item<'_>))
    where
        D: ReadOnlyQueryData,
    {
        for item in self.iter() {
            f(item);
        }
    }

    /// Visit every matched row with exclusive access, permitting `&mut T`.
    #[inline]
    pub fn for_each_mut(&mut self, mut f: impl FnMut(D::Item<'_>)) {
        for item in self.iter_mut() {
            f(item);
        }
    }

    /// The number of entities this query matches.
    #[inline]
    pub fn count(&self) -> usize
    where
        D: ReadOnlyQueryData,
    {
        self.iter().count()
    }

    /// Whether this query matches no entities.
    #[inline]
    pub fn is_empty(&self) -> bool
    where
        D: ReadOnlyQueryData,
    {
        self.iter().next().is_none()
    }

    /// Visit every matched row in parallel with shared access, dispatching
    /// disjoint row batches onto `pool`'s work-stealing threads (design §7
    /// par_iter / §8.3 system-internal chunk parallelism).
    ///
    /// Available only when every data term is read-only (`D: `[`ReadOnlyQueryData`]),
    /// so several parallel passes may coexist. Rows are partitioned into batches
    /// of at most `batch_size` (clamped to `>= 1`); the call returns only after
    /// every batch has completed. `func` runs concurrently and must be
    /// `Send + Sync`.
    #[cfg(feature = "multi_thread")]
    #[inline]
    pub fn par_for_each<Func>(&self, pool: &prism_tasks::TaskPool, batch_size: usize, func: Func)
    where
        D: ReadOnlyQueryData,
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` term is ever formed, and this system's declared read access
        // (upheld by the scheduler's conflict analysis) guarantees nothing
        // writes the columns we read for the duration of the dispatch.
        unsafe {
            self.state.par_for_each_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
                pool,
                batch_size,
                &func,
            );
        }
    }

    /// Visit every matched row in parallel with exclusive access, permitting
    /// `&mut T` data terms (design §7 par_iter / §8.3 chunk 子作业).
    ///
    /// Takes `&mut self`, so the borrow checker forbids a second concurrent
    /// dispatch of the same query. Rows are partitioned into disjoint batches of
    /// at most `batch_size` (clamped to `>= 1`); because the batches never
    /// overlap, the `&mut` references different threads form never alias the
    /// same column slot. The call returns only after every batch has completed.
    #[cfg(feature = "multi_thread")]
    #[inline]
    pub fn par_for_each_mut<Func>(
        &mut self,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live dispatch of this query, and the system's declared write access
        // (upheld by the scheduler's conflict analysis) guarantees nothing else
        // touches the columns we write; disjoint batches keep each `&mut` term
        // unique across threads for the duration of the dispatch.
        unsafe {
            self.state.par_for_each_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
                pool,
                batch_size,
                &func,
            );
        }
    }

    /// Visit every matched row in **chunk-aligned typed column slices** in
    /// parallel with shared access (design §8.3 `jobs.par_chunks` — the
    /// `IJobChunk`/SIMD-skinning form), calling `func` once per chunk-aligned
    /// batch with a read-only slice tuple.
    ///
    /// Available only when every data term is read-only
    /// (`D: `[`ReadOnlyQueryData`]) and sliceable (`D: `[`ColumnSliceData`]),
    /// with an archetype-uniform filter (`F: `[`ArchetypalFilter`]); the two
    /// `assert_*` checks run once up front. Each batch covers a whole number of
    /// 16KiB chunks, so a kernel can run a single SIMD pass over the slice. The
    /// call returns only after every batch has completed.
    #[cfg(feature = "multi_thread")]
    #[inline]
    pub fn par_chunks<Func>(&self, pool: &prism_tasks::TaskPool, batch_size: usize, func: Func)
    where
        D: ReadOnlyQueryData + ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut [T]` slice is ever formed, and this system's declared read
        // access (upheld by the scheduler's conflict analysis) guarantees
        // nothing writes the columns we read for the duration of the dispatch.
        unsafe {
            self.state.par_chunks_from_ptr(
                self.world.as_ptr(),
                self.world.this_run(),
                pool,
                batch_size,
                &func,
            );
        }
    }

    /// Visit every matched row in **chunk-aligned typed column slices** in
    /// parallel with exclusive access, permitting `&mut [T]` slice terms
    /// (design §8.3 `jobs.par_chunks`, e.g. SIMD skinning writing a `&mut
    /// [Mat4]` column).
    ///
    /// Takes `&mut self`, so the borrow checker forbids a second concurrent
    /// dispatch of the same query. Batches are chunk-aligned and disjoint, so
    /// the `&mut [T]` slices different threads form never overlap; each `&mut T`
    /// term stamps the change ticks of its whole handed-out range up front. The
    /// call returns only after every batch has completed.
    #[cfg(feature = "multi_thread")]
    #[inline]
    pub fn par_chunks_mut<Func>(
        &mut self,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        D: ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live dispatch of this query, and the system's declared write access
        // (upheld by the scheduler's conflict analysis) guarantees nothing else
        // touches the columns we write; chunk-aligned disjoint batches keep each
        // `&mut [T]` slice unique across threads for the dispatch.
        unsafe {
            self.state.par_chunks_from_ptr(
                self.world.as_ptr(),
                self.world.this_run(),
                pool,
                batch_size,
                &func,
            );
        }
    }
}

impl<'w, 's, D: ReadOnlyQueryData, F: QueryFilter> IntoIterator for &Query<'w, 's, D, F> {
    type Item = D::Item<'w>;
    type IntoIter = QueryIter<'w, 's, D, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: identical to `Query::iter` — read-only data over a live world
        // cell under the scheduler's conflict discipline.
        unsafe {
            self.state.iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }
}

impl<'w, 's, D: QueryData, F: QueryFilter> IntoIterator for &mut Query<'w, 's, D, F> {
    type Item = D::Item<'w>;
    type IntoIter = QueryIter<'w, 's, D, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: identical to `Query::iter_mut` — exclusive access via `&mut`
        // over a live world cell under the scheduler's conflict discipline.
        unsafe {
            self.state.iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }
}

// SAFETY: the query declares into the shared per-system `Access` exactly the
// component reads/writes recorded in its `QueryState` (replayed through the
// panicking adders, so any overlap with an earlier param is rejected), and it
// only ever materialises references consistent with that declared access.
unsafe impl<D, F> SystemParam for Query<'_, '_, D, F>
where
    D: QueryData + 'static,
    F: QueryFilter + 'static,
    D::State: 'static,
    F::State: 'static,
{
    type State = QueryState<D, F>;
    type Item<'w, 's> = Query<'w, 's, D, F>;

    #[inline]
    fn init_state(world: &mut World) -> Self::State {
        QueryState::new(world.components_mut())
    }

    #[inline]
    fn update_access(state: &Self::State, access: &mut Access) {
        let query_access = state.access();
        for &id in query_access.reads() {
            access.add_read(id);
        }
        for &id in query_access.writes() {
            access.add_write(id);
        }
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        world: UnsafeWorldCell<'w>,
    ) -> Query<'w, 's, D, F> {
        Query { world, state }
    }
}
