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
        let archetypes = self.matched_archetypes(world);
        let world_ptr = (world as *const World).cast_mut();
        // SAFETY: `D: ReadOnlyQueryData`, so no `&mut` fetch is ever formed and
        // the shared `&'w World` borrow is sufficient; `world_ptr` stays valid
        // for `'w`. Every id in `archetypes` came from `matched_archetypes`, so
        // it satisfies `D::matches` for `self.data_state`.
        unsafe { QueryIter::new(world_ptr, &self.data_state, archetypes) }
    }

    /// Iterate the rows matched by this query over an exclusive view of `world`,
    /// permitting `&mut T` terms.
    #[inline]
    pub fn iter_mut<'w, 's>(&'s self, world: &'w mut World) -> QueryIter<'w, 's, D, F> {
        let archetypes = self.matched_archetypes(world);
        let world_ptr = world as *mut World;
        // SAFETY: `world` is exclusively borrowed for `'w`, so the raw pointer
        // is the sole route to the world during iteration; `&mut` fetches are
        // therefore unique. Every id in `archetypes` came from
        // `matched_archetypes`, so it satisfies `D::matches`.
        unsafe { QueryIter::new(world_ptr, &self.data_state, archetypes) }
    }
}
