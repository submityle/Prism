//! [`QueryManyIter`]: iterate a caller-provided entity list against a query,
//! resolving each id independently (design §7 batched random access).
//!
//! Whereas [`QueryIter`](crate::query::QueryIter) walks every matched archetype,
//! a `QueryManyIter` is driven by an id list the caller already holds — a
//! hierarchy's children, a selection set, a replication delta, a replay
//! command. Each id is resolved exactly as
//! [`QueryState::get`](crate::query::QueryState::get) would resolve it and
//! surfaced as a `Result`, so the caller can either unwrap or use the
//! [`matched`](QueryManyIter::matched) adapter to skip non-matching ids.
//!
//! # Soundness model
//!
//! The iterator holds the query state (for the resolved term state) and a
//! `*mut World` valid for `'w`. It is reachable only through
//! [`QueryState::iter_many`](crate::query::QueryState::iter_many), which takes
//! `&'w World` and requires
//! [`ReadOnlyQueryData`](crate::query::ReadOnlyQueryData) — so only shared
//! fetches are formed and a duplicate id within the list simply yields the same
//! shared reference twice. The mutable batched path
//! ([`QueryState::get_many_mut`](crate::query::QueryState::get_many_mut)) is
//! separate and rejects duplicates up front rather than aliasing.

use core::marker::PhantomData;

use crate::change::Tick;
use crate::collections::HashMap;
use crate::entity::Entity;
use crate::query::fetch::{QueryData, ReadOnlyQueryData};
use crate::query::filter::QueryFilter;
use crate::query::state::{QueryEntityError, QueryState};
use crate::world::World;

/// A borrowing iterator over the query items of a caller-provided entity list,
/// in list order (design §7 batched random access).
///
/// Yields exactly one [`Result`] per input entity: `Ok(item)` when the entity
/// is live and satisfies the query this frame, `Err(QueryEntityError)`
/// otherwise. Use [`matched`](Self::matched) to skip the failing entries.
///
/// Created by [`QueryState::iter_many`](crate::query::QueryState::iter_many) /
/// [`QueryState::iter_many_mut`](crate::query::QueryState::iter_many_mut) and by
/// the [`Query`](crate::system::Query) system parameter's `iter_many` /
/// `iter_many_mut`. The mutable form rejects a repeated id with
/// [`QueryEntityError::AliasedMutability`] (two `&mut` items for one row would
/// alias); the read-only form lets a repeat yield twice.
pub struct QueryManyIter<'w, 's, D: QueryData, F: QueryFilter, I> {
    /// Resolved query state, borrowed so it stays reusable across iterations.
    state: &'s QueryState<D, F>,
    /// The world under iteration; valid for `'w` and only ever read through
    /// shared derefs (the read-only bound keeps a `&mut` fetch out of reach).
    world: *mut World,
    /// Start/end of the change-detection window, captured once at creation.
    last_run: Tick,
    this_run: Tick,
    /// The caller's entity list.
    entities: I,
    /// Entities already yielded, consulted only when [`dedup`](Self::dedup) is
    /// set.
    seen: HashMap<Entity, ()>,
    /// Whether a repeated entity is rejected rather than yielded again.
    ///
    /// Set by the *mutable* constructor: two `&mut` items for one row would
    /// alias, so a repeat becomes `Err(AliasedMutability)`. The read-only
    /// constructor leaves it clear, where a repeat is harmless (shared refs).
    dedup: bool,
    _marker: PhantomData<(&'w World, fn() -> (D, F))>,
}

impl<'w, 's, D, F, I> QueryManyIter<'w, 's, D, F, I>
where
    D: QueryData,
    F: QueryFilter,
    I: Iterator<Item = Entity>,
{
    /// Build a many-entity iterator.
    ///
    /// # Safety
    /// - `world` must be valid for reads for `'w`, and the caller must
    ///   guarantee that no other live borrow aliases the columns this query's
    ///   `D`/`F` terms touch for the duration of the iteration — upheld by the
    ///   read-only bound on the public entry points plus the scheduler's
    ///   access-conflict discipline.
    /// - `state` must have been built for the same `world`.
    pub(crate) unsafe fn new(
        state: &'s QueryState<D, F>,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
        entities: I,
    ) -> Self {
        Self::build(state, world, last_run, this_run, entities, false)
    }

    /// Build a many-entity iterator that rejects repeated ids.
    ///
    /// # Safety
    /// Same contract as [`new`](Self::new); additionally the caller must hold
    /// *exclusive* access to `world` for `'w` (the mutable entry point), so the
    /// `&mut` items this yields are unique.
    pub(crate) unsafe fn new_dedup(
        state: &'s QueryState<D, F>,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
        entities: I,
    ) -> Self {
        Self::build(state, world, last_run, this_run, entities, true)
    }

    #[inline]
    fn build(
        state: &'s QueryState<D, F>,
        world: *mut World,
        last_run: Tick,
        this_run: Tick,
        entities: I,
        dedup: bool,
    ) -> Self {
        Self {
            state,
            world,
            last_run,
            this_run,
            entities,
            seen: HashMap::new(),
            dedup,
            _marker: PhantomData,
        }
    }

    /// Skip the entities that do not match (dead, or failing a data/filter
    /// term), yielding only the resolved items.
    ///
    /// Mirrors Bevy's `QueryManyIter::matched`.
    #[inline]
    pub fn matched(self) -> QueryManyMatchedIter<'w, 's, D, F, I>
    where
        D: ReadOnlyQueryData,
    {
        QueryManyMatchedIter(self)
    }
}

impl<'w, 's, D, F, I> Iterator for QueryManyIter<'w, 's, D, F, I>
where
    D: QueryData,
    F: QueryFilter,
    I: Iterator<Item = Entity>,
{
    type Item = Result<D::Item<'w>, QueryEntityError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let entity = self.entities.next()?;
        // The mutable constructor sets `dedup`, so a repeated id is refused
        // before any fetch — without that guard two items could alias one row.
        if self.dedup && self.seen.insert(entity, ()).is_some() {
            return Some(Err(QueryEntityError::AliasedMutability(entity)));
        }
        // SAFETY: the entry points guarantee `world` is valid for `'w` and that
        // this state belongs to it; the read-only constructor only ever forms
        // shared fetches, while the mutable one has the duplicate guard above,
        // so no two live items can alias one `(archetype, row)`.
        Some(unsafe {
            self.state
                .get_from_ptr(self.world, entity, self.last_run, self.this_run)
        })
    }

    /// Every input entity yields exactly one item (an `Err` counts), so the
    /// list's own hint is exact.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entities.size_hint()
    }
}

/// Adapter that yields only the matching items of a [`QueryManyIter`], created
/// by [`QueryManyIter::matched`].
pub struct QueryManyMatchedIter<'w, 's, D: QueryData, F: QueryFilter, I>(
    QueryManyIter<'w, 's, D, F, I>,
);

impl<'w, 's, D, F, I> Iterator for QueryManyMatchedIter<'w, 's, D, F, I>
where
    D: ReadOnlyQueryData,
    F: QueryFilter,
    I: Iterator<Item = Entity>,
{
    type Item = D::Item<'w>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.0.next()? {
                Ok(item) => return Some(item),
                Err(_) => continue,
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.0.size_hint();
        (0, upper)
    }
}
