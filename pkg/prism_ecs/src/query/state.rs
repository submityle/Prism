//! [`QueryState`]: the world-static, reusable resolution of a query's data and
//! filter terms.
//!
//! Building a state registers every component the query names, computes the
//! read/write [`Access`] set (panicking on an internal aliasing conflict), and
//! can be reused across frames. The matched-archetype list is maintained
//! *incrementally*: each refresh only tests archetypes created since the last
//! refresh and appends the matches to a cached list, so steady-state iteration
//! never rescans the whole archetype table (design §7, M2 query cache).

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::marker::PhantomData;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::component::Components;
use crate::query::access::Access;
use crate::query::fetch::{QueryData, ReadOnlyQueryData};
use crate::query::filter::QueryFilter;
use crate::query::iter::QueryIter;
use crate::world::World;

/// Incrementally-maintained set of archetypes a query matches.
///
/// Archetype component sets are immutable and archetypes are never removed from
/// a [`World`], so a match decision for a given [`ArchetypeId`] is permanent:
/// once an archetype has been tested it never needs retesting. The cache
/// records how many archetypes have been examined (`checked`) and the ids that
/// matched (`ids`), so a refresh only has to test archetypes created since the
/// previous refresh.
struct MatchedArchetypes {
    /// Ids of every archetype matched so far, in ascending id order.
    ids: Vec<ArchetypeId>,
    /// Number of the world's archetypes already tested. Archetypes with index
    /// `< checked` have a settled (permanent) match decision.
    checked: usize,
}

/// The resolved, reusable state of a `Query<D, F>`.
pub struct QueryState<D: QueryData, F: QueryFilter = ()> {
    data_state: D::State,
    filter_state: F::State,
    access: Access,
    /// Incremental archetype-match cache (design §7). Behind an [`UnsafeCell`]
    /// so the `&self` iteration entry points can refresh it lazily; see the
    /// [`Sync`] impl for why single-cell mutation through `&self` is sound.
    matched: UnsafeCell<MatchedArchetypes>,
    _marker: PhantomData<fn() -> (D, F)>,
}

// SAFETY: the only non-`Sync` field is the `UnsafeCell` match cache. A
// `QueryState` instance is owned by exactly one system (its cached param state)
// or by one `World::query` call site, and the scheduler never runs a given
// system on two threads at once, so a given instance's cache cell is only ever
// touched by a single thread at a time. No `&QueryState` is ever shared between
// threads while its cache is mutated. The `data_state`/`filter_state` bounds
// keep the promise honest: the resolved term states must themselves be `Sync`.
unsafe impl<D: QueryData, F: QueryFilter> Sync for QueryState<D, F>
where
    D::State: Sync,
    F::State: Sync,
{
}

impl<D: QueryData, F: QueryFilter> QueryState<D, F> {
    /// Build the state against `components`, registering every named component
    /// type and computing the access set.
    ///
    /// # Panics
    /// Panics if the query's own terms conflict (e.g. `&mut A` with `&A`, or
    /// `&mut A` twice) — see [`Access`].
    pub fn new(components: &mut Components) -> Self {
        let data_state = D::init_state(components);
        let filter_state = F::init_state(components);
        let mut access = Access::new();
        D::update_access(&data_state, &mut access);
        F::update_access(&filter_state, &mut access);
        Self {
            data_state,
            filter_state,
            access,
            matched: UnsafeCell::new(MatchedArchetypes {
                ids: Vec::new(),
                checked: 0,
            }),
            _marker: PhantomData,
        }
    }

    /// The component read/write access this query requires.
    #[inline]
    pub fn access(&self) -> &Access {
        &self.access
    }

    /// Bring the match cache up to date with `world`, testing only archetypes
    /// created since the previous refresh.
    ///
    /// Relies on the archetype invariants (immutable component sets, never
    /// removed) so that an id tested once never needs retesting.
    fn refresh_matched(&self, world: &World) {
        // SAFETY: a `QueryState` instance is single-threaded (see the `Sync`
        // impl), so this is the only live borrow of the cache cell; the `&mut`
        // does not alias and is dropped before this method returns.
        let cache = unsafe { &mut *self.matched.get() };
        let total = world.archetypes().len();
        if cache.checked >= total {
            return;
        }
        for archetype in world.archetypes().iter().skip(cache.checked) {
            if D::matches(&self.data_state, archetype)
                && F::matches(&self.filter_state, archetype)
            {
                cache.ids.push(archetype.id());
            }
        }
        cache.checked = total;
    }

    /// The ids of archetypes this query matches.
    ///
    /// Refreshes the incremental cache (testing only newly-created archetypes)
    /// and returns a copy of the matched-id list, which the iterator drivers
    /// take ownership of. The copy is `O(matched)`, while the full
    /// `D::matches`/`F::matches` scan it replaces was `O(total archetypes)`.
    pub(crate) fn matched_archetypes(&self, world: &World) -> Vec<ArchetypeId> {
        self.refresh_matched(world);
        // SAFETY: single-threaded per instance (see the `Sync` impl); this
        // shared borrow of the cache does not alias any live `&mut`.
        let cache = unsafe { &*self.matched.get() };
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
        let last_run = world.last_change_tick();
        let this_run = world.change_tick();
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

    /// Iterate the rows matched by this query over an exclusive view of `world`,
    /// permitting `&mut T` terms.
    #[inline]
    pub fn iter_mut<'w, 's>(&'s self, world: &'w mut World) -> QueryIter<'w, 's, D, F> {
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

    /// Iterate the rows matched by this query from a raw `*mut World`.
    ///
    /// This is the entry point used by the scheduler's `Query` system
    /// parameter, which only ever holds a `*mut World` (never a `&mut World`),
    /// so that disjoint systems and params can run against the same world under
    /// the kernel's access-conflict discipline (design doc §8.2).
    ///
    /// # Safety
    /// `world` must point to a live [`World`] that stays valid for `'w`, and the
    /// caller must guarantee that no other live borrow aliases the component
    /// columns this query's `D`/`F` terms touch. This is upheld by the
    /// per-system access set plus the scheduler's conflict analysis.
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
}
