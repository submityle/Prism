//! [`QueryState`]: the world-static, reusable resolution of a query's data and
//! filter terms.
//!
//! Building a state registers every component the query names, computes the
//! read/write [`Access`] set (panicking on an internal aliasing conflict), and
//! can be reused across frames. For M0 the matched-archetype list is recomputed
//! on each iteration by scanning all archetypes; the incremental match cache
//! that avoids the full scan is an M2 refinement (design §7).

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::component::Components;
use crate::query::access::Access;
use crate::query::fetch::{QueryData, ReadOnlyQueryData};
use crate::query::filter::QueryFilter;
use crate::query::iter::QueryIter;
use crate::world::World;

/// The resolved, reusable state of a `Query<D, F>`.
pub struct QueryState<D: QueryData, F: QueryFilter = ()> {
    data_state: D::State,
    filter_state: F::State,
    access: Access,
    _marker: PhantomData<fn() -> (D, F)>,
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
            _marker: PhantomData,
        }
    }

    /// The component read/write access this query requires.
    #[inline]
    pub fn access(&self) -> &Access {
        &self.access
    }

    /// Compute the ids of archetypes this query matches, by scanning all of
    /// `world`'s archetypes (M0 behaviour; cached incrementally in M2).
    pub(crate) fn matched_archetypes(&self, world: &World) -> Vec<ArchetypeId> {
        let mut out = Vec::new();
        for archetype in world.archetypes().iter() {
            if D::matches(&self.data_state, archetype)
                && F::matches(&self.filter_state, archetype)
            {
                out.push(archetype.id());
            }
        }
        out
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
