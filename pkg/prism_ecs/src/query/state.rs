//! [`QueryState`]: the world-static, reusable resolution of a query's data and
//! filter terms.
//!
//! Building a state registers every component the query names, computes the
//! read/write [`Access`] set (panicking on an internal aliasing conflict), and
//! can be reused across frames. The matched-archetype list is maintained
//! *incrementally*: each refresh only tests archetypes created since the last
//! refresh and appends the matches to a cached list, so steady-state iteration
//! never rescans the whole archetype table (design §7, M2 query cache).

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::change_index::{ChangeCursor, ChangeTracker};
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::query::access::Access;
use crate::query::combinations::QueryCombinationsIter;
use crate::query::fetch::{QueryData, ReadOnlyQueryData};
use crate::query::filter::QueryFilter;
use crate::query::iter::QueryIter;
use crate::query::many::QueryManyIter;
use crate::query::matched::{MatchedArchetypes, SpinLock};
use crate::query::slice::{ArchetypalFilter, ColumnSliceData};
use crate::query::slice_iter::QuerySliceIter;
use crate::world::{World, WorldId};

/// Why [`QueryState::get`] / [`QueryState::get_mut`] could not resolve an
/// entity's query terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryEntityError {
    /// The entity is not live: it was never spawned, has been despawned, or the
    /// handle's generation is stale.
    NoSuchEntity(Entity),
    /// The entity is live but does not match this query — its component set
    /// lacks a required term or carries an excluded one, or its per-row change
    /// state fails an [`Added`](crate::query::Added) / [`Changed`](crate::query::Changed)
    /// filter this frame.
    QueryDoesNotMatch(Entity),
    /// The same entity was requested mutably more than once in one call (for
    /// example a duplicate in [`QueryState::get_many_mut`]), which would create
    /// two aliasing `&mut` references to one row. Rejected before any fetch.
    AliasedMutability(Entity),
}

impl core::fmt::Display for QueryEntityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoSuchEntity(entity) => {
                write!(
                    f,
                    "entity {entity:?} does not exist (never allocated or despawned)"
                )
            }
            Self::QueryDoesNotMatch(entity) => {
                write!(f, "entity {entity:?} does not match the query")
            }
            Self::AliasedMutability(entity) => {
                write!(f, "entity {entity:?} was requested mutably more than once")
            }
        }
    }
}

impl core::error::Error for QueryEntityError {}

/// Why [`QueryState::single`] / [`QueryState::single_mut`] could not produce a
/// unique item (design §7 cardinality-constrained access).
///
/// Carries the query's data-term type name (`core::any::type_name::<D>()`) so a
/// diagnostic can name the offending query, mirroring Bevy's `QuerySingleError`
/// shape without pulling in a name-caching dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuerySingleError {
    /// Nothing matched the query this frame.
    NoEntities(&'static str),
    /// More than one entity matched the query.
    MultipleEntities(&'static str),
}

impl core::fmt::Display for QuerySingleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEntities(name) => write!(f, "no entities fit the query {name}"),
            Self::MultipleEntities(name) => write!(f, "multiple entities fit the query {name}"),
        }
    }
}

impl core::error::Error for QuerySingleError {}

/// The resolved, reusable state of a `Query<D, F>`.
pub struct QueryState<D: QueryData, F: QueryFilter = ()> {
    /// Identity of the world this state was built from. Every safe entry point
    /// verifies the world it is handed carries this id before dereferencing any
    /// column, so a state can never reinterpret a different world's storage.
    world_id: WorldId,
    data_state: D::State,
    filter_state: F::State,
    access: Access,
    /// Incremental archetype-match cache (design §7), behind a [`SpinLock`] so
    /// the `&self` iteration entry points can refresh it lazily while a shared
    /// `&QueryState` is used from several threads without a data race. The
    /// resolved term states are `Send + Sync` by the [`QueryData`]/
    /// [`QueryFilter`] bounds, so the whole state is honestly `Send + Sync`.
    matched: SpinLock<MatchedArchetypes>,
    /// Sparse change sources retained for the coarse dirty-window path. The
    /// tracker keeps only logs while at least one query/observer is alive; the
    /// per-query cursors make collection proportional to new batches.
    dirty_tracker: Option<Arc<ChangeTracker>>,
    dirty_sources: Vec<ComponentId>,
    dirty_cursors: SpinLock<Vec<ChangeCursor>>,
    /// Last explicit dirty-window request. The change index is a forward-only
    /// cursor, so overlapping or rewound requests must use the live fallback.
    dirty_last_window: SpinLock<Option<(Tick, Tick)>>,
    /// The tick at which this query began retaining sparse change-index
    /// history. Windows that begin before this boundary need the conservative
    /// live-set fallback; later windows are fully covered by the index.
    dirty_activation_tick: Tick,
    _marker: PhantomData<fn() -> (D, F)>,
}

impl<D: QueryData, F: QueryFilter> QueryState<D, F> {
    /// Build the state against `world`, registering every named component type,
    /// computing the access set, and recording the world's [`WorldId`] so the
    /// state can never be used against a different world.
    ///
    /// # Panics
    /// Panics if the query's own terms conflict (e.g. `&mut A` with `&A`, or
    /// `&mut A` twice) — see [`Access`].
    pub fn new(world: &mut World) -> Self {
        let world_id = world.id();
        let dirty_activation_tick = world.change_tick();
        let data_state = D::init_state(world.components_mut());
        let filter_state = F::init_state(world.components_mut());
        let mut access = Access::new();
        D::update_access(&data_state, &mut access);
        F::update_access(&filter_state, &mut access);
        let mut dirty_sources = access.reads().to_vec();
        dirty_sources.extend_from_slice(access.writes());
        F::dirty_sources(&filter_state, &mut dirty_sources);
        dirty_sources.sort_unstable();
        dirty_sources.dedup();
        dirty_sources.retain(|id| {
            world
                .components()
                .info(*id)
                .is_some_and(|info| info.storage() == crate::component::StorageType::SparseSet)
        });
        let dirty_tracker = if dirty_sources.is_empty() {
            None
        } else {
            Some(Arc::clone(&world.change_tracker))
        };
        let dirty_cursors = SpinLock::new(
            dirty_sources
                .iter()
                .map(|id| {
                    dirty_tracker
                        .as_ref()
                        .expect("sparse dirty sources have a tracker")
                        .activate(*id)
                })
                .collect(),
        );
        Self {
            world_id,
            data_state,
            filter_state,
            access,
            matched: SpinLock::new(MatchedArchetypes::new(world.archetype_generation())),
            dirty_tracker,
            dirty_sources,
            dirty_cursors,
            dirty_last_window: SpinLock::new(None),
            dirty_activation_tick,
            _marker: PhantomData,
        }
    }

    /// Assert that `world` is the world this state was built from.
    ///
    /// Every safe entry point runs this before touching the world: a mismatch
    /// would let `D::matches`/`D::fetch` reinterpret another world's dense
    /// component ids as this query's types, which is undefined behaviour.
    /// Panicking on misuse is the only sound option for a safe API.
    #[inline]
    pub(crate) fn validate_world(&self, world: &World) {
        assert_eq!(
            self.world_id,
            world.id(),
            "QueryState was built for one World but used against another; \
             build a fresh state with World::query on the target world"
        );
    }

    /// The component read/write access this query requires.
    #[inline]
    pub fn access(&self) -> &Access {
        &self.access
    }

    /// Resolved filter state for query-internal incremental consumers.
    #[inline]
    pub(crate) fn filter_state(&self) -> &F::State {
        &self.filter_state
    }

    /// Collect sparse component writes/removals recorded since the caller's
    /// last cursor window. The query owns one cursor per sparse source, so the
    /// tracker can discard history as soon as no observer/query retains it.
    pub(crate) fn collect_sparse_changes(
        &self,
        last_run: Tick,
        this_run: Tick,
        out: &mut Vec<Entity>,
    ) {
        let Some(tracker) = &self.dirty_tracker else {
            return;
        };
        let mut cursors = self.dirty_cursors.lock();
        for (index, &component) in self.dirty_sources.iter().enumerate() {
            tracker.collect(component, &mut cursors[index], last_run, this_run, out);
        }
    }

    /// Whether the exact sparse change index may be missing writes because the
    /// requested window predates activation or overlaps a previous request.
    /// The comparison uses the world's current tick rather than `this_run`:
    /// callers may inspect an older window whose upper bound is before query
    /// activation, and that window still needs the conservative fallback.
    #[inline]
    pub(crate) fn sparse_fallback_needed(
        &self,
        world: &World,
        last_run: Tick,
        _this_run: Tick,
    ) -> bool {
        let now = world.change_tick();
        if self.dirty_activation_tick.is_newer_than(last_run, now) {
            return true;
        }
        self.dirty_last_window
            .lock()
            .is_some_and(|(_, previous_this)| previous_this.is_newer_than(last_run, now))
    }

    #[inline]
    pub(crate) fn remember_sparse_window(&self, last_run: Tick, this_run: Tick) {
        *self.dirty_last_window.lock() = Some((last_run, this_run));
    }

    /// Whether `entity` is matched by this query (live and satisfying every
    /// data/filter term in `world` this frame).
    ///
    /// The `QueryState` face of [`Query::contains`](crate::system::Query::contains);
    /// it never forms an item, so it is cheap for a single id.
    #[inline]
    pub fn contains(&self, world: &World, entity: Entity) -> bool
    where
        D: ReadOnlyQueryData,
    {
        self.get(world, entity).is_ok()
    }

    /// Number of rows this query matches in `world` this frame.
    ///
    /// Walks the matched rows (per-row `Added`/`Changed`/sparse gates included),
    /// so the cost is `O(matched rows)`; it exists so callers need not spell out
    /// `state.iter(world).count()`.
    #[inline]
    pub fn count(&self, world: &World) -> usize
    where
        D: ReadOnlyQueryData,
    {
        self.iter(world).count()
    }

    /// Whether this query matches no rows in `world` this frame.
    ///
    /// Stops at the first matched row, so it is cheaper than
    /// [`count`](Self::count) `== 0` when a match is likely early.
    #[inline]
    pub fn is_empty(&self, world: &World) -> bool
    where
        D: ReadOnlyQueryData,
    {
        self.iter(world).next().is_none()
    }

    /// Visit every matched row over a shared view of `world`.
    #[inline]
    pub fn for_each(&self, world: &World, f: impl FnMut(D::Item<'_>))
    where
        D: ReadOnlyQueryData,
    {
        // Routes through `QueryIter::fold`, the batched fast path (design §17.1).
        self.iter(world).for_each(f);
    }

    /// Visit every matched row over an exclusive view of `world`, permitting
    /// `&mut` data terms (design §7 / §17.1).
    ///
    /// The mutable counterpart of [`for_each`](Self::for_each); routes through
    /// the batched [`QueryIter::fold`](crate::query::QueryIter) fast path, so it
    /// runs at roughly the parallel driver's single-worker batch speed rather
    /// than paying the per-row `next()` protocol. Each `&mut T` fetch stamps the
    /// change ticks of its row exactly as [`iter_mut`](Self::iter_mut) does.
    #[inline]
    pub fn for_each_mut(&self, world: &mut World, f: impl FnMut(D::Item<'_>)) {
        self.iter_mut(world).for_each(f);
    }

    /// The ids of archetypes this query matches.
    ///
    /// Brings the incremental cache up to date under its spin lock (testing only
    /// archetypes created since the previous refresh, and rebuilding from
    /// scratch when the world's structural generation changed) and returns a
    /// copy of the matched-id list, which the iterator drivers take ownership
    /// of. The copy is `O(matched)`, while the full `D::matches`/`F::matches`
    /// scan it replaces was `O(total archetypes)`.
    pub(crate) fn matched_archetypes(&self, world: &World) -> Vec<ArchetypeId> {
        // Snapshot the scan cursor under the lock (applying any structural
        // generation reset), then release it *before* calling `D::matches` /
        // `F::matches`: those are user-implementable, so holding the cache lock
        // across them risks a re-entrant deadlock, and a panic mid-scan would
        // otherwise leave a half-updated cache that duplicates ids on retry.
        let generation = world.archetype_generation();
        let checked = {
            let mut cache = self.matched.lock();
            // A wholesale archetype-registry replacement (snapshot restore)
            // bumps the structural generation; the cached ids then name freed
            // or renumbered archetypes, so drop them and re-scan from scratch.
            if cache.generation != generation {
                cache.ids.clear();
                cache.checked = 0;
                cache.generation = generation;
            }
            cache.checked
        };

        let total = world.archetypes().len();
        if checked >= total {
            return self.matched.lock().ids.clone();
        }

        // Resolve the newly-created archetypes with the lock released. Relies on
        // the archetype invariants (immutable component sets, never removed
        // within a generation) so an id tested once never needs retesting.
        let mut appended: Vec<ArchetypeId> = Vec::new();
        for archetype in world.archetypes().iter().skip(checked) {
            if D::matches(&self.data_state, archetype) && F::matches(&self.filter_state, archetype)
            {
                appended.push(archetype.id());
            }
        }

        // Commit atomically. If another thread already advanced the cursor, keep
        // its (identical, deterministic) result and drop ours.
        let mut cache = self.matched.lock();
        if cache.generation == generation && cache.checked == checked {
            cache.ids.extend(appended);
            cache.checked = total;
        }
        cache.ids.clone()
    }

    /// Iterate the rows matched by this query over a shared view of `world`.
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so a shared borrow is sufficient and several read-only iterations
    /// may coexist.
    #[inline]
    pub fn iter<'w, 's>(&'s self, world: &'w World) -> QueryIter<'w, 's, D, F>
    where
        D: ReadOnlyQueryData,
    {
        self.iter_with_ticks(world, world.last_change_tick(), world.change_tick())
    }

    /// Iterate a read-only query using this consumer's own change window.
    ///
    /// Independent consumers such as render extraction retain their last
    /// successful tick without changing the world's one-shot query baseline.
    /// The window is `(last_run, this_run]`, with the usual wrapping tick rules.
    #[inline]
    pub fn iter_with_ticks<'w, 's>(
        &'s self,
        world: &'w World,
        last_run: Tick,
        this_run: Tick,
    ) -> QueryIter<'w, 's, D, F>
    where
        D: ReadOnlyQueryData,
    {
        self.validate_world(world);
        let archetypes = self.matched_archetypes(world);
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut` fetch is ever formed and
        // the shared `&'w World` borrow is sufficient; `world_ptr` stays valid
        // for `'w`. Every id in `archetypes` came from `matched_archetypes`, so
        // it satisfies `D::matches`/`F::matches` for the respective states.
        unsafe {
            QueryIter::new(
                world_ptr,
                &self.data_state,
                &self.filter_state,
                archetypes,
                last_run,
                this_run,
            )
        }
    }

    /// Iterate the matched rows in **chunk-aligned typed column slices** over a
    /// shared view of `world` (design §7 SIMD 迭代).
    ///
    /// The serial sibling of [`par_chunks`](Self::par_chunks): each yielded
    /// slice covers at most `K` rows of one archetype and never crosses a
    /// physical chunk boundary (so it is at most `min(K, rows_per_chunk)` rows;
    /// the last slice of a chunk may be shorter), which is exactly the
    /// contiguous input the `simd` bulk kernels take. Every data term must be
    /// [`ColumnSliceData`] and the filter must be an [`ArchetypalFilter`]; both
    /// are checked once, up front, with a clear panic rather than silently
    /// falling back.
    #[inline]
    pub fn simd_iter<'w, 's, const K: usize>(
        &'s self,
        world: &'w World,
    ) -> QuerySliceIter<'w, 's, D, F>
    where
        D: ColumnSliceData + ReadOnlyQueryData,
        F: ArchetypalFilter,
    {
        <D as ColumnSliceData>::assert_sliceable();
        <F as ArchetypalFilter>::assert_archetypal();
        self.validate_world(world);
        let this_run = world.change_tick();
        let archetypes = self.matched_archetypes(world);
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut [T]` slice is ever formed
        // and the shared `&'w World` borrow is sufficient; every id in
        // `archetypes` came from `matched_archetypes`.
        unsafe { QuerySliceIter::new(world_ptr, &self.data_state, archetypes, K, this_run) }
    }

    /// Iterate the matched rows in chunk-aligned typed column slices over an
    /// **exclusive** view of `world`, permitting `&mut [T]` terms (design §7/§8.3).
    ///
    /// The serial counterpart of [`par_chunks_mut`](Self::par_chunks_mut). A
    /// `&mut T` term stamps the change ticks of the whole handed-out range up
    /// front (a bare slice loses the per-row write information), and the
    /// caller's exclusive borrow makes every slice unique.
    #[inline]
    pub fn simd_iter_mut<'w, 's, const K: usize>(
        &'s self,
        world: &'w mut World,
    ) -> QuerySliceIter<'w, 's, D, F>
    where
        D: ColumnSliceData,
        F: ArchetypalFilter,
    {
        <D as ColumnSliceData>::assert_sliceable();
        <F as ArchetypalFilter>::assert_archetypal();
        self.validate_world(world);
        let this_run = world.change_tick();
        let archetypes = self.matched_archetypes(world);
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed for `'w`, so the raw pointer is
        // the sole route to the world during iteration; the slices are disjoint
        // by construction, so each `&mut [T]` is unique.
        unsafe { QuerySliceIter::new(world_ptr, &self.data_state, archetypes, K, this_run) }
    }

    /// Iterate every unordered `K`-subset of the rows this query matches
    /// (design §7; the read-only half of Bevy's `iter_combinations`).
    ///
    /// Matching the rows costs one pass (the ordinary iterator's traversal, so
    /// the per-row gate is shared); each yielded combination then re-fetches its
    /// `K` entities at `O(1)`. Combinations come out in a deterministic
    /// lexicographic order over the matched rows.
    ///
    /// Restricted to [`ReadOnlyQueryData`] because a yield hands out `K` items:
    /// with `&mut T` terms one entity could appear in two live combinations and
    /// alias itself. `K = 2` is the broad-phase pair case; `K = 1` degenerates to
    /// ordinary iteration; `K = 0` or `K` greater than the match count yields
    /// nothing.
    #[inline]
    pub fn iter_combinations<'w, 's, const K: usize>(
        &'s self,
        world: &'w World,
    ) -> QueryCombinationsIter<'w, 's, D, F, K>
    where
        D: ReadOnlyQueryData,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        // Collect the matched rows by driving the ordinary iterator, so the
        // per-row gate lives in exactly one place.
        let mut rows = Vec::new();
        let mut iter = self.iter(world);
        while let Some((entity, _item)) = iter.next_with_entity() {
            rows.push(entity);
        }
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so only shared fetches are formed and
        // the shared `&'w World` borrow is sufficient; every collected id matched
        // this state for the same window.
        unsafe { QueryCombinationsIter::new(self, world_ptr, last_run, this_run, rows) }
    }

    /// Iterate the rows matched by this query over an exclusive view of `world`,
    /// permitting `&mut T` terms.
    #[inline]
    pub fn iter_mut<'w, 's>(&'s self, world: &'w mut World) -> QueryIter<'w, 's, D, F> {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let archetypes = self.matched_archetypes(world);
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed for `'w`, so the raw pointer
        // is the sole route to the world during iteration; `&mut` fetches are
        // therefore unique. Every id in `archetypes` came from
        // `matched_archetypes`, so it satisfies `D::matches`/`F::matches`.
        unsafe {
            QueryIter::new(
                world_ptr,
                &self.data_state,
                &self.filter_state,
                archetypes,
                last_run,
                this_run,
            )
        }
    }

    /// Fetch the query terms for a single `entity` over a shared view of
    /// `world`, or report why it does not resolve (design §7 random access).
    ///
    /// This is the random-access dual of [`iter`](Self::iter): instead of
    /// walking every matched row it jumps straight to `entity`'s storage
    /// location, re-checks the archetype against the query's data/filter terms,
    /// and applies the same per-row gate (`Added`/`Changed`/`With`/`Without`
    /// plus any sparse-presence requirement) that iteration uses. Cost is
    /// independent of the number of matched archetypes.
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so a shared borrow is sufficient and several concurrent reads may
    /// coexist. Use [`get_mut`](Self::get_mut) for `&mut T` terms.
    ///
    /// # Errors
    /// - [`QueryEntityError::NoSuchEntity`] if `entity` is not live.
    /// - [`QueryEntityError::QueryDoesNotMatch`] if `entity` is live but its
    ///   component set (or per-row change state) does not satisfy the query.
    #[inline]
    pub fn get<'w, 's>(
        &'s self,
        world: &'w World,
        entity: Entity,
    ) -> Result<D::Item<'w>, QueryEntityError>
    where
        D: ReadOnlyQueryData,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut` fetch is ever formed and
        // the shared `&'w World` borrow is sufficient; `world_ptr` stays valid
        // for `'w`.
        unsafe { self.get_from_ptr(world_ptr, entity, last_run, this_run) }
    }

    /// Fetch the query terms for a single `entity` over an exclusive view of
    /// `world`, permitting `&mut T` terms.
    ///
    /// The mutable dual of [`get`](Self::get); see it for the lookup contract.
    ///
    /// # Errors
    /// See [`get`](Self::get).
    #[inline]
    pub fn get_mut<'w, 's>(
        &'s self,
        world: &'w mut World,
        entity: Entity,
    ) -> Result<D::Item<'w>, QueryEntityError> {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed for `'w`, so the raw pointer
        // is the sole route to the world during the fetch; a `&mut` term
        // therefore yields a unique reference valid for `'w`.
        unsafe { self.get_from_ptr(world_ptr, entity, last_run, this_run) }
    }

    /// Fetch the single matching row over a shared view of `world`, or report
    /// why the query does not have exactly one match (design §7
    /// cardinality-constrained access).
    ///
    /// This is the cardinality-constrained companion of [`get`](Self::get):
    /// instead of naming an entity it asserts "exactly one entity matches". It
    /// walks the matched archetypes lazily and stops as soon as a second row is
    /// observed, so the single-match case never scans past the first row and
    /// the multi-match case never scans past the second.
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so a shared borrow is sufficient and several concurrent reads
    /// may coexist.
    ///
    /// # Errors
    /// - [`QuerySingleError::NoEntities`] if nothing matched this frame.
    /// - [`QuerySingleError::MultipleEntities`] if two or more entities matched
    ///   (including when a per-row [`Added`](crate::query::Added) /
    ///   [`Changed`](crate::query::Changed) gate admits more than one).
    #[inline]
    pub fn single<'w, 's>(&'s self, world: &'w World) -> Result<D::Item<'w>, QuerySingleError>
    where
        D: ReadOnlyQueryData,
    {
        Self::single_from_iter(self.iter(world))
    }

    /// Fetch the single matching row over an exclusive view of `world`,
    /// permitting `&mut T` terms.
    ///
    /// The mutable counterpart of [`single`](Self::single); see it for the
    /// cardinality contract.
    ///
    /// # Errors
    /// See [`single`](Self::single).
    #[inline]
    pub fn single_mut<'w, 's>(
        &'s self,
        world: &'w mut World,
    ) -> Result<D::Item<'w>, QuerySingleError> {
        Self::single_from_iter(self.iter_mut(world))
    }

    /// Take the first item of `iter` and reject if a second is present.
    ///
    /// Factored out so the shared and exclusive entry points (and the `Query`
    /// param's raw-pointer path) share one definition of the cardinality check.
    /// The matched rows are visited disjointly, so holding the first item while
    /// probing for a second never aliases a `&mut T` term.
    #[inline]
    fn single_from_iter<'w, 's>(
        mut iter: QueryIter<'w, 's, D, F>,
    ) -> Result<D::Item<'w>, QuerySingleError> {
        let first = iter.next();
        let extra = iter.next().is_some();
        match (first, extra) {
            (Some(item), false) => Ok(item),
            (None, _) => Err(QuerySingleError::NoEntities(core::any::type_name::<D>())),
            (Some(_), _) => Err(QuerySingleError::MultipleEntities(
                core::any::type_name::<D>(),
            )),
        }
    }

    /// Fetch the single matching row from a raw `*mut World`.
    ///
    /// Shared engine of the `Query` system parameter's `single` / `single_mut`,
    /// which only ever hold a `*mut World`.
    ///
    /// # Safety
    /// Identical to [`iter_from_ptr`](Self::iter_from_ptr): `world` must be
    /// valid for `'w` and the caller must guarantee that no other live borrow
    /// aliases the columns this query's `D`/`F` terms touch.
    pub(crate) unsafe fn single_from_ptr<'w, 's>(
        &'s self,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
    ) -> Result<D::Item<'w>, QuerySingleError> {
        // SAFETY: the caller upholds the `iter_from_ptr` contract.
        let iter = unsafe { self.iter_from_ptr(world, last_run, this_run) };
        Self::single_from_iter(iter)
    }

    /// Fetch the query terms for a fixed list of `entities`, in list order
    /// (design §7 batched random access).
    ///
    /// The batched dual of [`get`](Self::get): every entity is resolved exactly
    /// as a single `get` would resolve it, and the first entity that fails (not
    /// live, or not matching) short-circuits with its [`QueryEntityError`]; the
    /// entities after it are not touched. The returned array preserves the
    /// input order, so it pairs with parallel index arrays (ids / weights).
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so a shared borrow is sufficient.
    ///
    /// # Errors
    /// See [`QueryEntityError`]; duplicate ids are permitted here (shared refs).
    #[inline]
    pub fn get_many<const N: usize>(
        &self,
        world: &World,
        entities: [Entity; N],
    ) -> Result<[D::Item<'_>; N], QueryEntityError>
    where
        D: ReadOnlyQueryData,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so only shared fetches are formed and
        // the shared `&World` borrow is sufficient; `world_ptr` stays valid.
        unsafe { self.get_many_from_ptr(world_ptr, entities, last_run, this_run) }
    }

    /// Fetch the query terms for a fixed list of `entities` with exclusive
    /// access, permitting `&mut T` terms (design §7 batched random access).
    ///
    /// The mutable batched dual of [`get`](Self::get): the array is built from
    /// per-entity fetches valid for `'w`, which only stays sound if no entity is
    /// named twice. A duplicate is therefore rejected up front with
    /// [`QueryEntityError::AliasedMutability`] before any `&mut` is created;
    /// otherwise the returned array preserves the input order.
    ///
    /// # Errors
    /// - [`QueryEntityError::AliasedMutability`] if `entities` repeats an id.
    /// - The other [`QueryEntityError`] variants for a dead or non-matching
    ///   entity (the first one in list order wins).
    #[inline]
    pub fn get_many_mut<const N: usize>(
        &self,
        world: &mut World,
        entities: [Entity; N],
    ) -> Result<[D::Item<'_>; N], QueryEntityError> {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed, so the raw pointer is the
        // sole route to it; the duplicate check inside rejects aliasing ids.
        unsafe { self.get_many_mut_from_ptr(world_ptr, entities, last_run, this_run) }
    }

    /// Random-access fetch of a fixed entity list from a raw `*mut World`.
    ///
    /// # Safety
    /// Identical to [`get_from_ptr`](Self::get_from_ptr): `world` must be valid
    /// for `'w` and the caller must guarantee no other live borrow aliases the
    /// columns this query's `D`/`F` terms touch.
    pub(crate) unsafe fn get_many_from_ptr<'w, 's, const N: usize>(
        &'s self,
        world: *mut World,
        entities: [Entity; N],
        last_run: Tick,
        this_run: Tick,
    ) -> Result<[D::Item<'w>; N], QueryEntityError> {
        let mut slots: [Option<D::Item<'w>>; N] = core::array::from_fn(|_| None);
        for (slot, entity) in slots.iter_mut().zip(entities) {
            // SAFETY: the caller upholds the `get_from_ptr` contract.
            let item = unsafe { self.get_from_ptr(world, entity, last_run, this_run) }?;
            *slot = Some(item);
        }
        Ok(slots.map(|slot| slot.expect("every slot is filled when no error short-circuits")))
    }

    /// Random-access mutable fetch of a fixed entity list from a raw
    /// `*mut World`, rejecting duplicate ids before any `&mut` is created.
    ///
    /// # Safety
    /// Identical to [`get_from_ptr`](Self::get_from_ptr); the duplicate guard
    /// inside enforces the intra-call part of the mutable aliasing contract.
    pub(crate) unsafe fn get_many_mut_from_ptr<'w, 's, const N: usize>(
        &'s self,
        world: *mut World,
        entities: [Entity; N],
        last_run: Tick,
        this_run: Tick,
    ) -> Result<[D::Item<'w>; N], QueryEntityError> {
        // Reject duplicates before forming any `&mut`: naming one entity twice
        // would mint two aliasing mutable references.
        for i in 0..N {
            for j in 0..i {
                if entities[i] == entities[j] {
                    return Err(QueryEntityError::AliasedMutability(entities[i]));
                }
            }
        }
        // SAFETY: the caller upholds the `get_from_ptr` contract; the duplicate
        // guard above means each entity is fetched at most once.
        unsafe { self.get_many_from_ptr(world, entities, last_run, this_run) }
    }

    /// Iterate the query terms for a caller-provided entity list, in list order
    /// (design §7 batched random access).
    ///
    /// Unlike [`iter`](Self::iter), which walks every matched archetype, this
    /// walks exactly the supplied entities and yields one
    /// `Result<D::Item, QueryEntityError>` per entry, so a caller that already
    /// has an id list (children, a selection set, a snapshot delta) can resolve
    /// it without an archetype scan. Errors are surfaced rather than skipped;
    /// use [`QueryManyIter::matched`] to skip the non-matching entries.
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so a shared borrow is sufficient and duplicates are harmless.
    #[inline]
    pub fn iter_many<'w, 's, I>(
        &'s self,
        world: &'w World,
        entities: I,
    ) -> QueryManyIter<'w, 's, D, F, I::IntoIter>
    where
        D: ReadOnlyQueryData,
        I: IntoIterator<Item = Entity>,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so only shared fetches are formed and
        // the shared `&'w World` borrow is sufficient; `world_ptr` stays valid.
        unsafe { self.iter_many_from_ptr(world_ptr, entities, last_run, this_run) }
    }

    /// Build a many-entity iterator from a raw `*mut World`.
    ///
    /// # Safety
    /// Identical to [`get_from_ptr`](Self::get_from_ptr): `world` must be valid
    /// for `'w` and the caller must guarantee no other live borrow aliases the
    /// columns this query's `D`/`F` terms touch for the whole iteration.
    pub(crate) unsafe fn iter_many_from_ptr<'w, 's, I>(
        &'s self,
        world: *mut World,
        entities: I,
        last_run: Tick,
        this_run: Tick,
    ) -> QueryManyIter<'w, 's, D, F, I::IntoIter>
    where
        I: IntoIterator<Item = Entity>,
    {
        // SAFETY: the caller upholds the `QueryManyIter::new` contract.
        unsafe { QueryManyIter::new(self, world, last_run, this_run, entities.into_iter()) }
    }

    /// Iterate a caller-provided entity list with exclusive access, permitting
    /// `&mut T` (design §7 batched random access).
    ///
    /// The mutable dual of [`iter_many`](Self::iter_many). Because the items may
    /// be `&mut`, a repeated id is **rejected**: the second occurrence yields
    /// [`QueryEntityError::AliasedMutability`] instead of a second `&mut` into
    /// the same row. Every other id is resolved exactly as
    /// [`get_mut`](Self::get_mut) would resolve it.
    #[inline]
    pub fn iter_many_mut<'w, 's, I>(
        &'s self,
        world: &'w mut World,
        entities: I,
    ) -> QueryManyIter<'w, 's, D, F, I::IntoIter>
    where
        I: IntoIterator<Item = Entity>,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed for `'w`, so the raw pointer is
        // the sole route to it; the iterator's duplicate guard keeps each row
        // fetched at most once, so its `&mut` items never alias.
        unsafe { self.iter_many_mut_from_ptr(world_ptr, entities, last_run, this_run) }
    }

    /// Build a mutable many-entity iterator from a raw `*mut World`.
    ///
    /// # Safety
    /// Identical to [`iter_many_from_ptr`](Self::iter_many_from_ptr), and the
    /// caller must hold *exclusive* access to `world` for `'w`.
    pub(crate) unsafe fn iter_many_mut_from_ptr<'w, 's, I>(
        &'s self,
        world: *mut World,
        entities: I,
        last_run: Tick,
        this_run: Tick,
    ) -> QueryManyIter<'w, 's, D, F, I::IntoIter>
    where
        I: IntoIterator<Item = Entity>,
    {
        // SAFETY: the caller upholds the `QueryManyIter::new_dedup` contract.
        unsafe { QueryManyIter::new_dedup(self, world, last_run, this_run, entities.into_iter()) }
    }

    /// Random-access fetch of a single `entity` from a raw `*mut World`.
    ///
    /// Shared engine of [`get`](Self::get) / [`get_mut`](Self::get_mut) and the
    /// entry point used by the scheduler's `Query` system parameter, which only
    /// ever holds a `*mut World` (never a `&mut World`).
    ///
    /// # Safety
    /// `world` must point to a live [`World`] valid for `'w` **and must be the
    /// very world this state was built for** (`self.world_id`) — the resolved
    /// component/archetype ids are only meaningful there, so a different world
    /// would be type-confused. The caller must also guarantee that no other live
    /// borrow aliases the component columns this query's `D`/`F` terms touch —
    /// upheld by the per-system access set plus the scheduler's conflict
    /// analysis, or by an exclusive `&mut World`.
    pub(crate) unsafe fn get_from_ptr<'w, 's>(
        &'s self,
        world: *mut World,
        entity: Entity,
        last_run: Tick,
        this_run: Tick,
    ) -> Result<D::Item<'w>, QueryEntityError> {
        // SAFETY: the caller guarantees `world` is live for `'w`; forming a
        // shared `&World` is all that is needed to resolve the entity location
        // and read archetype storage.
        let world_ref: &'w World = unsafe { &*world };
        let location = world_ref
            .entities()
            .location(entity)
            .filter(|loc| !loc.is_empty())
            .ok_or(QueryEntityError::NoSuchEntity(entity))?;
        let archetype = world_ref
            .archetypes()
            .get(location.archetype_id)
            .ok_or(QueryEntityError::NoSuchEntity(entity))?;
        if !(D::matches(&self.data_state, archetype) && F::matches(&self.filter_state, archetype)) {
            return Err(QueryEntityError::QueryDoesNotMatch(entity));
        }
        let sparse_sets = world_ref.sparse_sets();
        let row = location.row as usize;
        let chunk = row / archetype.table().rows_per_chunk().max(1);
        debug_assert!(
            row < archetype.len(),
            "entity location row must index a live row of its archetype"
        );
        // SAFETY: `archetype` satisfies `D::matches` (checked above), the
        // contract of `init_fetch_for_chunk`. Resolving the specific chunk lets a
        // table term cache the window's value base (design §5.3).
        let fetch = unsafe {
            D::init_fetch_for_chunk(
                &self.data_state,
                archetype,
                sparse_sets,
                last_run,
                this_run,
                chunk,
            )
        };
        // SAFETY: the same archetype also satisfies `F::matches`, the contract
        // of `F::init_fetch`.
        let filter_fetch = unsafe {
            F::init_fetch(
                &self.filter_state,
                archetype,
                sparse_sets,
                last_run,
                this_run,
            )
        };
        // Per-row gate mirroring `QueryIter::next`: a rejected filter row or a
        // missing sparse data term means the entity does not match this frame.
        // SAFETY: `row` indexes a live row of this archetype and both fetches
        // were built for it; `F::Fetch`/`D::Fetch` are `Copy`.
        if !unsafe { F::filter_fetch(filter_fetch, entity, row) } {
            return Err(QueryEntityError::QueryDoesNotMatch(entity));
        }
        // SAFETY: as above; a sparse-backed required data term resolves its
        // per-entity membership here.
        if !unsafe { D::filter_fetch(fetch, entity, row) } {
            return Err(QueryEntityError::QueryDoesNotMatch(entity));
        }
        // SAFETY: this `(archetype, row)` is fetched exactly once for this call,
        // so a `&mut` term yields a unique reference valid for `'w` (the world
        // is exclusively borrowed for `'w` in the `&mut` entry points).
        Ok(unsafe { D::fetch(fetch, entity, row) })
    }

    /// Iterate the rows matched by this query from a raw `*mut World`.
    ///
    /// This is the entry point used by the scheduler's `Query` system
    /// parameter, which only ever holds a `*mut World` (never a `&mut World`),
    /// so that disjoint systems and params can run against the same world under
    /// the kernel's access-conflict discipline (design doc §8.2).
    ///
    /// # Safety
    /// `world` must point to a live [`World`] that stays valid for `'w` **and
    /// must be the very world this state was built for** (`self.world_id`). The
    /// caller must also guarantee that no other live borrow aliases the
    /// component columns this query's `D`/`F` terms touch (upheld by the
    /// per-system access set plus the scheduler's conflict analysis), and that
    /// no structural modification (`World::restore`) races the iteration.
    #[inline]
    pub(crate) unsafe fn iter_from_ptr<'w, 's>(
        &'s self,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
    ) -> QueryIter<'w, 's, D, F> {
        // SAFETY: the caller guarantees `world` is live for `'w`. Forming a
        // shared `&World` (never `&mut`) matches the kernel discipline and is
        // all that is needed to enumerate matched archetypes.
        let world_ref: &World = unsafe { &*world };
        let archetypes = self.matched_archetypes(world_ref);
        // SAFETY: every id in `archetypes` came from `matched_archetypes`, so it
        // satisfies `D::matches`/`F::matches`; the caller upholds non-aliasing
        // of the fetched columns for `'w`.
        unsafe {
            QueryIter::new(
                world,
                &self.data_state,
                &self.filter_state,
                archetypes,
                last_run,
                this_run,
            )
        }
    }

    /// Dispatch `func` over every matched row in parallel from a raw
    /// `*mut World` (design §7 par_iter / §8.3).
    ///
    /// This is the entry point used by the scheduler's
    /// [`Query`](crate::system::query_param::Query) system parameter and
    /// [`JobGraph`](crate::system::job_graph::JobGraph), which only ever hold a
    /// `*mut World` (never a `&mut World`). The mutable-vs-shared distinction is
    /// enforced by the caller: a read-only entry forms the pointer from a shared
    /// borrow and bounds `D: ReadOnlyQueryData`, while a mutable entry forms it
    /// from an exclusive borrow. Rows are partitioned into disjoint batches of
    /// at most `batch_size` (clamped to `>= 1`) and dispatched onto `pool`; the
    /// call returns only after every batch completes.
    ///
    /// # Safety
    /// `world` must point to a live [`World`] that stays valid for the whole
    /// call, and the caller must guarantee that no other live borrow aliases the
    /// component columns this query's `D`/`F` terms touch for the duration of
    /// the dispatch (upheld by the per-system access set plus the scheduler's
    /// conflict analysis, or by an exclusive `&mut World`).
    #[cfg(feature = "multi_thread")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn par_for_each_from_ptr<Func>(
        &self,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: &Func,
    ) where
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        // SAFETY: the caller guarantees `world` is live for the call; forming a
        // shared `&World` (never `&mut`) is all that is needed to enumerate the
        // matched archetypes.
        let world_ref: &World = unsafe { &*world };
        let archetypes = self.matched_archetypes(world_ref);
        // SAFETY: every id in `archetypes` came from `matched_archetypes`, so it
        // satisfies `D::matches`/`F::matches`; the caller upholds non-aliasing
        // of the fetched columns for the whole (scope-joined) dispatch.
        unsafe {
            crate::query::par::par_for_each_raw::<D, F, Func>(
                world,
                &self.data_state,
                &self.filter_state,
                &archetypes,
                last_run,
                this_run,
                pool,
                batch_size,
                func,
            );
        }
    }

    /// Visit every matched row in parallel over a shared view of `world`
    /// (design §7 par_iter), calling `func` once per row.
    ///
    /// Rows are partitioned into disjoint batches of at most `batch_size`
    /// (clamped to `>= 1`) and dispatched onto `pool`'s work-stealing threads;
    /// the call returns only after every batch has completed. Only read-only
    /// data terms are reachable (the `D: `[`ReadOnlyQueryData`] bound), so the
    /// shared borrow is sufficient and several parallel passes may coexist.
    #[cfg(feature = "multi_thread")]
    pub fn par_for_each<Func>(
        &self,
        world: &World,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        D: ReadOnlyQueryData,
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut` term is ever formed and
        // the shared `&World` borrow keeps the world live for the whole
        // (scope-joined) call; several read-only passes may coexist.
        unsafe {
            self.par_for_each_from_ptr(world_ptr, last_run, this_run, pool, batch_size, &func);
        }
    }

    /// Visit every matched row in parallel over an exclusive view of `world`,
    /// permitting `&mut T` terms (design §7 par_iter / §8.3 chunk 子作业).
    ///
    /// Rows are partitioned into disjoint batches of at most `batch_size`
    /// (clamped to `>= 1`) and dispatched onto `pool`; because the batches never
    /// overlap, the `&mut` references different threads form never alias the
    /// same column slot. The call returns only after every batch has completed.
    #[cfg(feature = "multi_thread")]
    pub fn par_for_each_mut<Func>(
        &self,
        world: &mut World,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        self.validate_world(world);
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed, so the raw pointer is the
        // sole route to the world for the whole dispatch; disjoint batches make
        // each `&mut` term unique across threads.
        unsafe {
            self.par_for_each_from_ptr(world_ptr, last_run, this_run, pool, batch_size, &func);
        }
    }

    /// Raw-pointer slice-parallel driver (design §8.3 `jobs.par_chunks`): hand
    /// each task one aligned typed column slice ([`ColumnSliceData::Slice`])
    /// covering a chunk-aligned run of rows, instead of one item per row.
    ///
    /// The matched archetypes are resolved here and forwarded as the already
    /// exact set — the [`ArchetypalFilter`] bound guarantees `F`'s per-row
    /// verdict equals its per-archetype verdict, so no per-row filter runs.
    ///
    /// # Safety
    /// `world` must point to a live [`World`] that stays valid for the whole
    /// call, and the caller must guarantee that no other live borrow aliases the
    /// component columns this query's `D` terms touch for the duration of the
    /// dispatch (upheld by the per-system access set plus the scheduler's
    /// conflict analysis, or by an exclusive `&mut World`).
    #[cfg(feature = "multi_thread")]
    pub(crate) unsafe fn par_chunks_from_ptr<Func>(
        &self,
        world: *mut World,
        this_run: Tick,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: &Func,
    ) where
        D: ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        // SAFETY: the caller guarantees `world` is live for the call; forming a
        // shared `&World` (never `&mut`) is all that is needed to enumerate the
        // matched archetypes.
        let world_ref: &World = unsafe { &*world };
        let archetypes = self.matched_archetypes(world_ref);
        // SAFETY: every id in `archetypes` came from `matched_archetypes`, so it
        // satisfies `D::matches`/`F::matches`; the caller upholds non-aliasing of
        // the fetched columns for the whole (scope-joined) dispatch.
        unsafe {
            crate::query::par::par_chunks_raw::<D, Func>(
                world,
                &self.data_state,
                &archetypes,
                this_run,
                pool,
                batch_size,
                func,
            );
        }
    }

    /// Visit every matched row in chunk-aligned typed slices in parallel over a
    /// shared view of `world` (design §8.3 `jobs.par_chunks`), calling `func`
    /// once per chunk-aligned batch with a read-only slice tuple.
    ///
    /// Only read-only data terms are reachable (the `D: `[`ReadOnlyQueryData`]
    /// bound), so the shared borrow suffices and several parallel passes may
    /// coexist. The `D: `[`ColumnSliceData`] and `F: `[`ArchetypalFilter`]
    /// bounds are checked once up front via their `assert_*` hooks (rejecting
    /// sparse/shared data and per-row / sparse filters with a clear panic).
    #[cfg(feature = "multi_thread")]
    pub fn par_chunks<Func>(
        &self,
        world: &World,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        D: ColumnSliceData + ReadOnlyQueryData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        <D as ColumnSliceData>::assert_sliceable();
        <F as ArchetypalFilter>::assert_archetypal();
        self.validate_world(world);
        let this_run = world.change_tick();
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut [T]` slice is ever formed
        // and the shared `&World` borrow keeps the world live for the whole
        // (scope-joined) call; several read-only passes may coexist.
        unsafe {
            self.par_chunks_from_ptr(world_ptr, this_run, pool, batch_size, &func);
        }
    }

    /// Visit every matched row in chunk-aligned typed slices in parallel over an
    /// exclusive view of `world`, permitting `&mut [T]` slice terms (design
    /// §8.3 `jobs.par_chunks`, the SIMD-skinning headline).
    ///
    /// Because batches are chunk-aligned and disjoint, the `&mut [T]` slices
    /// different threads form never overlap, and each `&mut T` term stamps the
    /// change ticks of its whole handed-out range up front (a bare slice loses
    /// the per-row write information). The call returns only after every batch
    /// has completed.
    #[cfg(feature = "multi_thread")]
    pub fn par_chunks_mut<Func>(
        &self,
        world: &mut World,
        pool: &prism_tasks::TaskPool,
        batch_size: usize,
        func: Func,
    ) where
        D: ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        <D as ColumnSliceData>::assert_sliceable();
        <F as ArchetypalFilter>::assert_archetypal();
        self.validate_world(world);
        let this_run = world.change_tick();
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed, so the raw pointer is the sole
        // route to the world for the whole dispatch; chunk-aligned disjoint
        // batches make each `&mut [T]` slice unique across threads.
        unsafe {
            self.par_chunks_from_ptr(world_ptr, this_run, pool, batch_size, &func);
        }
    }
}

impl<D: QueryData, F: QueryFilter> Drop for QueryState<D, F> {
    fn drop(&mut self) {
        if let Some(tracker) = &self.dirty_tracker {
            for &component in &self.dirty_sources {
                tracker.deactivate(component);
            }
        }
    }
}
