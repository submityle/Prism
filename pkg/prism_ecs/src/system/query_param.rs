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

use alloc::boxed::Box;
use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::entity::Entity;
use crate::query::{
    Access, QueryData, QueryEntityError, QueryFilter, QueryIter, QueryManyIter, QuerySingleError,
    QueryState, ReadOnlyQueryData,
};
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
    // The owner is borrowed for 's, but D/F's marker lifetimes need not
    // outlive 's. Materialize a typed reference only while borrowing Query.
    state: NonNull<QueryState<D, F>>,
    _state_borrow: PhantomData<&'s ()>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> Query<'w, 's, D, F> {
    #[inline]
    fn state(&self) -> &QueryState<D, F> {
        // SAFETY: get_param pairs this pointer with the borrow of its owning
        // QueryParamState for 's. Borrowing self keeps that owner alive; this
        // shared reference exposes only synchronized query-cache mutation.
        unsafe { self.state.as_ref() }
    }

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
            self.state().iter_from_ptr(
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
            self.state().iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Fetch this query's terms for a single `entity` with shared access, or
    /// report why it does not resolve. The random-access dual of
    /// [`iter`](Self::iter); available only when every data term is read-only
    /// (`D: `[`ReadOnlyQueryData`]).
    ///
    /// # Errors
    /// See [`QueryEntityError`].
    #[inline]
    pub fn get(&self, entity: Entity) -> Result<D::Item<'_>, QueryEntityError>
    where
        D: ReadOnlyQueryData,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` fetch is formed, and this system's declared read access
        // guarantees nothing writes the columns we read.
        unsafe {
            self.state().get_from_ptr(
                self.world.as_ptr(),
                entity,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Fetch this query's terms for a single `entity` with exclusive access,
    /// permitting `&mut T`. Takes `&mut self` so the borrow checker forbids
    /// holding two mutable fetches of one query at once.
    ///
    /// # Errors
    /// See [`QueryEntityError`].
    #[inline]
    pub fn get_mut(&mut self, entity: Entity) -> Result<D::Item<'_>, QueryEntityError> {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live fetch of this query, and the system's declared write access
        // guarantees nothing else touches the columns we write.
        unsafe {
            self.state().get_from_ptr(
                self.world.as_ptr(),
                entity,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Whether `entity` is matched by this query (live and satisfying every
    /// data/filter term this frame).
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool
    where
        D: ReadOnlyQueryData,
    {
        self.get(entity).is_ok()
    }

    /// Fetch the single matching row with shared access, or report why the
    /// query does not have exactly one match this frame.
    ///
    /// The system-param face of [`QueryState::single`](crate::query::QueryState::single):
    /// it stops at the first row in the single-match case and at the second in
    /// the multi-match case.
    ///
    /// # Errors
    /// - [`QuerySingleError::NoEntities`] if nothing matched this frame.
    /// - [`QuerySingleError::MultipleEntities`] if two or more entities matched.
    #[inline]
    pub fn single(&self) -> Result<D::Item<'_>, QuerySingleError>
    where
        D: ReadOnlyQueryData,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` fetch is formed, and this system's declared read access
        // guarantees nothing writes the columns we read.
        unsafe {
            self.state().single_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Fetch the single matching row with exclusive access, permitting
    /// `&mut T`. Takes `&mut self` so the borrow checker forbids holding the
    /// single fetch alongside any other fetch of this query.
    ///
    /// # Errors
    /// See [`single`](Self::single).
    #[inline]
    pub fn single_mut(&mut self) -> Result<D::Item<'_>, QuerySingleError> {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live fetch of this query, and the system's declared write access
        // guarantees nothing else touches the columns we write.
        unsafe {
            self.state().single_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Fetch the query terms for a fixed list of entities, in list order. The
    /// first entity that fails short-circuits with its [`QueryEntityError`].
    ///
    /// # Errors
    /// See [`QueryEntityError`]; duplicate ids are permitted (shared fetches).
    #[inline]
    pub fn get_many<const N: usize>(
        &self,
        entities: [Entity; N],
    ) -> Result<[D::Item<'_>; N], QueryEntityError>
    where
        D: ReadOnlyQueryData,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` fetch is formed, and this system's declared read access
        // guarantees nothing writes the columns we read.
        unsafe {
            self.state().get_many_from_ptr(
                self.world.as_ptr(),
                entities,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Fetch the query terms for a fixed list of entities with exclusive
    /// access, permitting `&mut T`.
    ///
    /// # Errors
    /// - [`QueryEntityError::AliasedMutability`] if `entities` repeats an id.
    /// - The other [`QueryEntityError`] variants otherwise.
    #[inline]
    pub fn get_many_mut<const N: usize>(
        &mut self,
        entities: [Entity; N],
    ) -> Result<[D::Item<'_>; N], QueryEntityError> {
        // SAFETY: `self.world` is live for `'w`; `&mut self` makes this the sole
        // live fetch of this query and the duplicate guard inside rejects
        // aliasing ids, while the system's declared write access guarantees
        // nothing else touches the columns we write.
        unsafe {
            self.state().get_many_mut_from_ptr(
                self.world.as_ptr(),
                entities,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Iterate a caller-provided entity list, yielding one
    /// `Result<D::Item, QueryEntityError>` per entity in list order. Use
    /// [`QueryManyIter::matched`] to skip the non-matching entries.
    #[inline]
    pub fn iter_many<I>(&self, entities: I) -> QueryManyIter<'_, '_, D, F, I::IntoIter>
    where
        D: ReadOnlyQueryData,
        I: IntoIterator<Item = Entity>,
    {
        // SAFETY: `self.world` is live for `'w`; `D: ReadOnlyQueryData` means no
        // `&mut` fetch is formed, and this system's declared read access
        // guarantees nothing writes the columns we read for the iteration.
        unsafe {
            self.state().iter_many_from_ptr(
                self.world.as_ptr(),
                entities,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// The chunk windows this query touches that changed within its own
    /// observation window `(last_run, this_run]` (design §7 脏块访问器 / §10).
    ///
    /// The system-param face of
    /// [`QueryState::dirty_chunks`](crate::query::QueryState::dirty_chunks): it
    /// uses this run's change window, so a system can drive a GPU upload, a
    /// spatial-index sync, or a shadow invalidation from the *coarse* layer
    /// (one chunk version per window) and refine per row only where needed.
    /// The result is a correct superset of the rows a per-row `Changed<T>` scan
    /// would visit, never a miss.
    #[inline]
    pub fn dirty_chunks(&self) -> alloc::vec::Vec<crate::query::DirtyChunk> {
        // SAFETY: `self.world` is live for `'w`; `dirty_chunks` only reads
        // archetype/chunk metadata relevant to this query's declared access,
        // which the scheduler has already made conflict-free.
        let world: &World = unsafe { &*self.world.as_ptr() };
        self.state()
            .dirty_chunks(world, self.world.last_run(), self.world.this_run())
    }

    /// Iterate a caller-provided entity list with exclusive access, permitting
    /// `&mut T`. A repeated id yields
    /// [`QueryEntityError::AliasedMutability`] rather than a second `&mut` into
    /// the same row.
    #[inline]
    pub fn iter_many_mut<I>(&mut self, entities: I) -> QueryManyIter<'_, '_, D, F, I::IntoIter>
    where
        I: IntoIterator<Item = Entity>,
    {
        // SAFETY: `&mut self` makes this the sole live borrow of the query, and
        // the system's declared write access guarantees nothing else touches the
        // columns we write; the iterator's duplicate guard keeps each row
        // fetched at most once.
        unsafe {
            self.state().iter_many_mut_from_ptr(
                self.world.as_ptr(),
                entities,
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }

    /// Visit every matched row with shared access.
    #[inline]
    pub fn for_each(&self, f: impl FnMut(D::Item<'_>))
    where
        D: ReadOnlyQueryData,
    {
        // Routes through `QueryIter::fold`, the batched fast path (design §17.1).
        self.iter().for_each(f);
    }

    /// Visit every matched row with exclusive access, permitting `&mut T`.
    #[inline]
    pub fn for_each_mut(&mut self, f: impl FnMut(D::Item<'_>)) {
        self.iter_mut().for_each(f);
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
            self.state().par_for_each_from_ptr(
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
            self.state().par_for_each_from_ptr(
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
            self.state().par_chunks_from_ptr(
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
            self.state().par_chunks_from_ptr(
                self.world.as_ptr(),
                self.world.this_run(),
                pool,
                batch_size,
                &func,
            );
        }
    }
}

impl<'i, 'w, 's, D: ReadOnlyQueryData, F: QueryFilter> IntoIterator for &'i Query<'w, 's, D, F> {
    type Item = D::Item<'w>;
    type IntoIter = QueryIter<'w, 'i, D, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: identical to `Query::iter` — read-only data over a live world
        // cell under the scheduler's conflict discipline.
        unsafe {
            self.state().iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }
}

/// The `&mut Query` iterator is tied to the *mutable borrow of the param* `'i`,
/// not to the world lifetime `'w`: otherwise `(&mut q).into_iter()` could be
/// called twice and two aliasing `&mut` iterators over one world would coexist
/// in safe code. This is a compile-fail regression guard for exactly that:
///
/// ```compile_fail
/// use prism_ecs::prelude::*;
///
/// struct Marker;
/// impl Component for Marker {}
///
/// fn two_aliasing_iterators<'w, 's>(mut q: Query<'w, 's, &'static mut Marker>) {
///     let mut first = (&mut q).into_iter();
///     let mut second = (&mut q).into_iter(); // ERROR: `q` is already mutably borrowed
///     let _ = (first.next(), second.next());
/// }
/// ```
impl<'i, 'w, 's, D: QueryData, F: QueryFilter> IntoIterator for &'i mut Query<'w, 's, D, F> {
    type Item = D::Item<'i>;
    type IntoIter = QueryIter<'i, 'i, D, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: identical to `Query::iter_mut` — exclusive access via `&mut`
        // over a live world cell under the scheduler's conflict discipline.
        unsafe {
            self.state().iter_from_ptr(
                self.world.as_ptr(),
                self.world.last_run(),
                self.world.this_run(),
            )
        }
    }
}

/// Owned query-parameter cache, with the query's marker lifetimes erased.
///
/// `QueryState<D, F>` stores only `D::State`, `F::State`, and kernel metadata;
/// `D` and `F` themselves occur only in a phantom marker. Requiring the stored
/// states to be `'static` is sufficient for ownership. Erasing the marker types
/// here lets nested parameters spell `Query<&mut T>` without requiring the
/// reference in that spelling to be `'static`.
///
/// The allocation keeps its original destructor. Only `Query::get_param` casts
/// it back, under the requirement that a parameter is fetched with its own
/// initialized state. Access replay needs no cast and remains safe even if
/// caches from different query types are passed to `update_access`.
pub struct QueryParamState {
    query: NonNull<()>,
    drop_query: unsafe fn(NonNull<()>),
    access: Access,
}

impl QueryParamState {
    fn new<D: QueryData, F: QueryFilter>(world: &mut World) -> Self
    where
        D::State: 'static,
        F::State: 'static,
    {
        unsafe fn drop_query<D: QueryData, F: QueryFilter>(query: NonNull<()>) {
            // SAFETY: this function is paired with the allocation of exactly
            // `QueryState<D, F>` below and called exactly once by `Drop`.
            unsafe { drop(Box::from_raw(query.cast::<QueryState<D, F>>().as_ptr())) };
        }

        let query = Box::new(QueryState::<D, F>::new(world));
        let access = query.access().clone();
        Self {
            query: NonNull::from(Box::leak(query)).cast(),
            drop_query: drop_query::<D, F>,
            access,
        }
    }
}

// SAFETY: the erased allocation contains a `QueryState<D, F>`, whose actual
// fields are Send + Sync and 'static (the term states and synchronized match
// cache). Neither D nor F is stored. The pointer is owned, never exposed by a
// safe API, and is only dereferenced while borrowing this cache.
unsafe impl Send for QueryParamState {}
// SAFETY: same ownership and synchronized-field reasoning as the Send impl.
unsafe impl Sync for QueryParamState {}

impl Drop for QueryParamState {
    fn drop(&mut self) {
        // SAFETY: `query` is the unique owned allocation paired with this
        // destructor at construction; borrowing prevents dropping a live query.
        unsafe { (self.drop_query)(self.query) };
    }
}

// SAFETY: the query declares into the shared per-system `Access` exactly the
// component reads/writes recorded in its `QueryState` (replayed through the
// panicking adders, so any overlap with an earlier param is rejected), and it
// only ever materialises references consistent with that declared access.
unsafe impl<D, F> SystemParam for Query<'_, '_, D, F>
where
    D: QueryData,
    F: QueryFilter,
    D::State: 'static,
    F::State: 'static,
{
    type State = QueryParamState;
    type Item<'w, 's> = Query<'w, 's, D, F>;

    #[inline]
    fn init_state(world: &mut World) -> Self::State {
        QueryParamState::new::<D, F>(world)
    }

    #[inline]
    fn update_access(state: &Self::State, access: &mut Access) {
        let query_access = &state.access;
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
        // The caller supplies the state initialized for this parameter, so
        // this is exactly QueryState<D, F>. Its owner is borrowed for 's.
        Query {
            world,
            state: state.query.cast(),
            _state_borrow: PhantomData,
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod cache_tests {
    use super::*;
    use alloc::rc::Rc;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use crate::archetype::Archetype;
    use crate::change::Tick;
    use crate::component::Components;
    use crate::storage::SparseSets;

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct DropProbe;

    impl Drop for DropProbe {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    // A deliberately non-Send/non-Sync, non-static descriptor. Only its
    // associated State is stored in the cache; no Rc or borrowed value is.
    struct Tracked<'marker, const FAIL_FILTER: bool>(PhantomData<&'marker Rc<()>>);

    // SAFETY: this unit-valued term matches every row and accesses no storage.
    unsafe impl<const FAIL: bool> QueryData for Tracked<'_, FAIL> {
        type Item<'w> = ();
        type State = DropProbe;
        type Fetch<'w> = ();

        fn init_state(_: &mut Components) -> Self::State {
            DropProbe
        }

        fn matches(_: &Self::State, _: &Archetype) -> bool {
            true
        }

        fn update_access(_: &Self::State, _: &mut Access) {}

        unsafe fn init_fetch<'w>(
            _: &Self::State,
            _: &'w Archetype,
            _: &'w SparseSets,
            _: Tick,
            _: Tick,
        ) -> Self::Fetch<'w> {
        }

        unsafe fn fetch<'w>(_: Self::Fetch<'w>, _: Entity, _: usize) -> Self::Item<'w> {}
    }

    // SAFETY: the unit-valued term never creates a mutable reference.
    unsafe impl<const FAIL: bool> ReadOnlyQueryData for Tracked<'_, FAIL> {}

    // SAFETY: the filter passes every row without reading ticks or values.
    unsafe impl<const FAIL: bool> QueryFilter for Tracked<'_, FAIL> {
        type State = DropProbe;
        type Fetch<'w> = ();

        fn init_state(_: &mut Components) -> Self::State {
            assert!(!FAIL, "test filter initialization failure");
            DropProbe
        }

        fn matches(_: &Self::State, _: &Archetype) -> bool {
            true
        }

        fn update_access(_: &Self::State, _: &mut Access) {}

        unsafe fn init_fetch<'w>(
            _: &Self::State,
            _: &'w Archetype,
            _: &'w SparseSets,
            _: Tick,
            _: Tick,
        ) -> Self::Fetch<'w> {
        }

        unsafe fn filter_fetch<'w>(_: Self::Fetch<'w>, _: Entity, _: usize) -> bool {
            true
        }
    }

    #[test]
    fn erased_cache_owns_static_states_and_drops_them_once_even_after_unwinding() {
        fn cache<'marker>(_: &'marker Rc<()>, world: &mut World) -> QueryParamState {
            type Term<'a> = Tracked<'a, false>;
            let mut state = QueryParamState::new::<Term<'marker>, Term<'marker>>(world);
            let cell = UnsafeWorldCell::new_mutable(world);
            // SAFETY: this state belongs to exactly this query type and world;
            // it declares no access and no other world borrows are live.
            let query =
                unsafe { Query::<Term<'marker>, Term<'marker>>::get_param(&mut state, cell) };
            assert_eq!(query.count(), 1);
            state
        }

        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<QueryParamState>();

        let mut world = World::new();
        world.spawn(());
        let marker = Rc::new(());
        let state = cache(&marker, &mut world);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(marker);
        drop(world);
        // Moving the owner to another thread and outliving both the marker
        // and world is sound: it owns only static Send + Sync cached fields.
        std::thread::spawn(move || drop(state)).join().unwrap();
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);

        let mut world = World::new();
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            QueryParamState::new::<Tracked<'_, false>, Tracked<'_, true>>(&mut world)
        }));
        assert!(failure.is_err());
        // Data state initialized before the failing filter is still dropped;
        // no partially constructed erased allocation is leaked.
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }
}
