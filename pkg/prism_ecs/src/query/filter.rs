//! [`QueryFilter`]: predicates that narrow which rows a query visits without
//! yielding any data.
//!
//! Filters come in two flavours (design §7, §10):
//!
//! - **Archetype filters** evaluated once per archetype — an archetype that
//!   fails is skipped wholesale, so they are free:
//!   - `()` — the empty filter, matches everything.
//!   - [`With<T>`] — the archetype must have component `T`.
//!   - [`Without<T>`] — the archetype must not have component `T`.
//! - **Change-detection filters** evaluated per row against the querying
//!   system's observer window `(last_run, this_run]` (design §10):
//!   - [`Added<T>`] — `T` was *added* to the entity within the window.
//!   - [`Changed<T>`] — `T` was *written* (or added) within the window.
//! - **Combinators**:
//!   - [`Or<(F0, F1, ...)>`] — at least one inner filter matches.
//!   - tuples of filters — every element matches (logical AND).
//!
//! Each filter exposes a world-static `State` (resolved [`ComponentId`]s), a
//! per-archetype `Fetch` cursor (resolved column + ticks, or a cached archetype
//! match flag), and a leaf [`QueryFilter::filter_fetch`] that decides one row.
//! Archetype filters encode their verdict directly in the `Fetch` so that
//! [`Or`] composes correctly: an `Or` term must still visit an archetype when
//! *some* branch matches it, deferring the real decision to `filter_fetch`.

use core::marker::PhantomData;

use crate::archetype::Archetype;
use crate::change::Tick;
use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::access::Access;
use crate::storage::{Column, ComponentSparseSet, SparseSets};

/// A predicate that narrows the rows a query visits (design §7, §10).
///
/// # Safety
/// This trait is `unsafe` to implement because its methods gate which
/// archetypes and rows a query touches. A filter accesses change-tick metadata
/// but never component values and never forms a `&mut`, so an incorrect
/// implementation cannot by itself cause memory unsafety; the trait is `unsafe`
/// to keep the whole query surface consistent and auditable. Implementors must
/// still register every component whose ticks they inspect via
/// [`update_access`](QueryFilter::update_access) so the scheduler sees the read.
pub unsafe trait QueryFilter {
    /// World-static resolved state (component ids), computed once.
    type State: Send + Sync;

    /// Per-archetype resolved cursor. Must be `Copy` so the iterator can read it
    /// per row without consuming it.
    type Fetch<'w>: Copy;

    /// Resolve this filter's [`State`](Self::State), registering any named
    /// component types.
    fn init_state(components: &mut Components) -> Self::State;

    /// Whether `archetype` can be visited at all by this filter. Returning
    /// `true` does not assert any row matches — the per-row decision is made by
    /// [`filter_fetch`](QueryFilter::filter_fetch). Returning `false` skips the
    /// archetype wholesale.
    fn matches(state: &Self::State, archetype: &Archetype) -> bool;

    /// Record any component tick reads this filter performs into `access`
    /// (via [`Access::add_filter_read`]). Archetype-only filters record nothing.
    fn update_access(state: &Self::State, access: &mut Access);

    /// Resolve the per-archetype fetch cursor for the observer window
    /// `(last_run, this_run]`.
    ///
    /// `sparse_sets` is the world's out-of-band registry (design §6), consulted
    /// by sparse-backed membership (`With`/`Without`) and change-detection
    /// (`Added`/`Changed`) filters, whose presence is per-entity rather than
    /// encoded in the archetype component set.
    ///
    /// # Safety
    /// `archetype` must satisfy [`QueryFilter::matches`] for `state`.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w>;

    /// Decide whether the row passes this filter.
    ///
    /// # Safety
    /// `row` must be `< archetype.len()` for the archetype `fetch` was built
    /// for.
    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool;
}

/// Per-archetype cursor for the membership filters [`With`]/[`Without`]
/// (design §6).
///
/// A table-backed component resolves its archetype-wide verdict once (encoded
/// in [`Table`](MembershipFetch::Table)); a sparse-backed component carries its
/// [`ComponentSparseSet`] so membership is resolved per entity.
#[derive(Clone, Copy)]
pub enum MembershipFetch<'w> {
    /// Table-backed verdict resolved once for the whole archetype.
    Table(bool),
    /// Sparse-backed source; `None` when the component's set was never
    /// allocated. Membership is resolved per entity by `filter_fetch`.
    Sparse(Option<&'w ComponentSparseSet>),
}

/// Per-archetype cursor for the change-detection filters [`Added`]/[`Changed`]
/// (design §6, §10), carrying the observer window `(last_run, this_run]`.
#[derive(Clone, Copy)]
pub enum ChangeFilterFetch<'w> {
    /// Table-backed column plus the observer window, or `None` when the
    /// archetype lacks the column (only reachable through [`Or`]).
    Table(Option<(&'w Column, Tick, Tick)>),
    /// Sparse-backed source plus the observer window; the source is `None` when
    /// the component's set was never allocated. The tick is resolved per entity.
    Sparse(Option<&'w ComponentSparseSet>, Tick, Tick),
}

/// Matches archetypes that **have** component `T` (without reading it).
pub struct With<T>(PhantomData<fn() -> T>);

/// Matches archetypes that **lack** component `T`.
pub struct Without<T>(PhantomData<fn() -> T>);

/// Matches entities to which component `T` was **added** within the querying
/// system's observer window `(last_run, this_run]` (design §10).
///
/// Fires for exactly one system run after the component appears on the entity
/// (via `spawn`/`insert`), then stops — unless the component is removed and
/// re-added.
pub struct Added<T>(PhantomData<fn() -> T>);

/// Matches entities whose component `T` was **written** (or first added) within
/// the querying system's observer window `(last_run, this_run]` (design §10).
///
/// A write is recorded by any mutable access through
/// [`Mut<T>`](crate::change::Mut) (a `&mut T` query term or
/// [`World::get_mut`](crate::world::World::get_mut)); an equal-value guard can
/// suppress it via [`Mut::bypass_change_detection`](crate::change::Mut::bypass_change_detection).
pub struct Changed<T>(PhantomData<fn() -> T>);

/// Matches archetypes/rows where **at least one** inner filter matches
/// (logical OR).
///
/// `F` is a tuple of filters, e.g. `Or<(With<A>, Changed<B>)>`.
pub struct Or<F>(PhantomData<fn() -> F>);

// SAFETY: `With<T>` reads no component data and forms no reference into it. Its
// verdict is a membership test: a table component is resolved once per archetype
// in the `Copy` fetch; a sparse component carries its `ComponentSparseSet` and
// is resolved per entity in `filter_fetch`, so `Or` can still compose it.
unsafe impl<T: Component> QueryFilter for With<T> {
    type State = ComponentId;
    type Fetch<'w> = MembershipFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            // A sparse component is routed out of archetypes, so its presence is
            // per-entity: admit the archetype and resolve membership per row.
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // A shared component splits archetypes by value (design §6); its
            // presence is the archetype-level binding.
            StorageType::Shared => archetype.shared_binding(*state).is_some(),
        }
    }

    fn update_access(_state: &Self::State, _access: &mut Access) {}

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => MembershipFetch::Table(archetype.contains(*state)),
            StorageType::SparseSet => MembershipFetch::Sparse(sparse_sets.get(*state)),
            // Shared presence is archetype-wide: resolve the verdict once.
            StorageType::Shared => {
                MembershipFetch::Table(archetype.shared_binding(*state).is_some())
            }
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> bool {
        match fetch {
            MembershipFetch::Table(present) => present,
            MembershipFetch::Sparse(set) => set.is_some_and(|s| s.contains(entity)),
        }
    }
}

// SAFETY: `Without<T>` reads no component data and forms no reference into it.
// Its verdict is `!membership`: resolved once per archetype for a table
// component, or per entity through the carried `ComponentSparseSet` for a sparse
// component. The `Copy` fetch lets `Or` compose it.
unsafe impl<T: Component> QueryFilter for Without<T> {
    type State = ComponentId;
    type Fetch<'w> = MembershipFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            // A sparse component can be present or absent on entities sharing one
            // archetype, so admit it and resolve the negation per row.
            StorageType::SparseSet => true,
            StorageType::Table => !archetype.contains(*state),
            // A shared component splits archetypes by value (design §6); absence
            // is the archetype-level lack of a binding.
            StorageType::Shared => archetype.shared_binding(*state).is_none(),
        }
    }

    fn update_access(_state: &Self::State, _access: &mut Access) {}

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => MembershipFetch::Table(!archetype.contains(*state)),
            StorageType::SparseSet => MembershipFetch::Sparse(sparse_sets.get(*state)),
            // Shared absence is archetype-wide: resolve the verdict once.
            StorageType::Shared => {
                MembershipFetch::Table(archetype.shared_binding(*state).is_none())
            }
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> bool {
        match fetch {
            MembershipFetch::Table(absent) => absent,
            MembershipFetch::Sparse(set) => !set.is_some_and(|s| s.contains(entity)),
        }
    }
}

// SAFETY: `Added<T>` reads only `T`'s added tick (registered as a filter read)
// and forms no reference into component data. A table-backed `Fetch` is `None`
// when the column is absent (only reachable via `Or`); a sparse-backed `Fetch`
// carries the component's `ComponentSparseSet` so the added tick is resolved per
// entity (`None` set, or an entity the set lacks, never matches).
unsafe impl<T: Component> QueryFilter for Added<T> {
    type State = ComponentId;
    type Fetch<'w> = ChangeFilterFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        assert!(
            T::STORAGE != StorageType::Shared,
            "a shared component (design §6) is immutable and archetype-wide, so it is never `added` or `changed` per entity; change-detection filters do not support shared storage"
        );
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // `init_state` already rejected a shared component for this filter.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_filter_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => ChangeFilterFetch::Table(
                archetype
                    .table()
                    .column(*state)
                    .map(|col| (col, last_run, this_run)),
            ),
            StorageType::SparseSet => {
                ChangeFilterFetch::Sparse(sparse_sets.get(*state), last_run, this_run)
            }
            // `init_state` already rejected a shared component for this filter.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
        match fetch {
            ChangeFilterFetch::Table(Some((col, last_run, this_run))) => {
                col.added_tick(row).is_newer_than(last_run, this_run)
            }
            ChangeFilterFetch::Table(None) => false,
            ChangeFilterFetch::Sparse(set, last_run, this_run) => set
                .and_then(|s| s.added_tick(entity))
                .is_some_and(|tick| tick.is_newer_than(last_run, this_run)),
        }
    }
}

// SAFETY: `Changed<T>` reads only `T`'s changed tick (registered as a filter
// read) and forms no reference into component data. A table-backed `Fetch` is
// `None` when the column is absent (only reachable via `Or`); a sparse-backed
// `Fetch` carries the component's `ComponentSparseSet` so the changed tick is
// resolved per entity (`None` set, or an entity the set lacks, never matches).
unsafe impl<T: Component> QueryFilter for Changed<T> {
    type State = ComponentId;
    type Fetch<'w> = ChangeFilterFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        assert!(
            T::STORAGE != StorageType::Shared,
            "a shared component (design §6) is immutable and archetype-wide, so it is never `added` or `changed` per entity; change-detection filters do not support shared storage"
        );
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // `init_state` already rejected a shared component for this filter.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_filter_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => ChangeFilterFetch::Table(
                archetype
                    .table()
                    .column(*state)
                    .map(|col| (col, last_run, this_run)),
            ),
            StorageType::SparseSet => {
                ChangeFilterFetch::Sparse(sparse_sets.get(*state), last_run, this_run)
            }
            // `init_state` already rejected a shared component for this filter.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
        match fetch {
            ChangeFilterFetch::Table(Some((col, last_run, this_run))) => {
                col.changed_tick(row).is_newer_than(last_run, this_run)
            }
            ChangeFilterFetch::Table(None) => false,
            ChangeFilterFetch::Sparse(set, last_run, this_run) => set
                .and_then(|s| s.changed_tick(entity))
                .is_some_and(|tick| tick.is_newer_than(last_run, this_run)),
        }
    }
}

// SAFETY: the empty filter matches everything, reads nothing, and its row
// verdict is unconditionally `true`.
unsafe impl QueryFilter for () {
    type State = ();
    type Fetch<'w> = ();

    fn init_state(_components: &mut Components) -> Self::State {}

    fn matches(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn update_access(_state: &Self::State, _access: &mut Access) {}

    unsafe fn init_fetch<'w>(
        _state: &Self::State,
        _archetype: &'w Archetype,
        _sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
    }

    unsafe fn filter_fetch<'w>(_fetch: Self::Fetch<'w>, _entity: Entity, _row: usize) -> bool {
        true
    }
}

macro_rules! impl_filter_tuple {
    ($($F:ident),+) => {
        // SAFETY: each element is a `QueryFilter`; the AND combinator only
        // narrows the matched set, registers the union of the elements' filter
        // reads, and composes their per-row verdicts conjunctively.
        #[allow(non_snake_case)]
        unsafe impl<$($F: QueryFilter),+> QueryFilter for ($($F,)+) {
            type State = ($($F::State,)+);
            type Fetch<'w> = ($($F::Fetch<'w>,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($F::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($F,)+) = state;
                $($F::matches($F, archetype))&&+
            }

            fn update_access(state: &Self::State, access: &mut Access) {
                let ($($F,)+) = state;
                $($F::update_access($F, access);)+
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: &'w Archetype,
                sparse_sets: &'w SparseSets,
                last_run: Tick,
                this_run: Tick,
            ) -> Self::Fetch<'w> {
                let ($($F,)+) = state;
                // SAFETY: forwarded — `matches` held for every element.
                unsafe { ($($F::init_fetch($F, archetype, sparse_sets, last_run, this_run),)+) }
            }

            unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
                let ($($F,)+) = fetch;
                // SAFETY: forwarded — same row for every element.
                $((unsafe { $F::filter_fetch($F, entity, row) }))&&+
            }
        }

        // SAFETY: `Or` over the same element set differs only in the combinator
        // (OR). `matches` admits the archetype when *any* branch admits it, and
        // the real per-row decision is deferred to `filter_fetch`, which ORs the
        // branches. Each branch's `Fetch` independently encodes its own
        // archetype match (e.g. `With` -> bool, `Added` -> `Option`), so a
        // branch that does not apply to this archetype simply contributes
        // `false` instead of forcing the whole `Or` to skip it.
        #[allow(non_snake_case)]
        unsafe impl<$($F: QueryFilter),+> QueryFilter for Or<($($F,)+)> {
            type State = ($($F::State,)+);
            type Fetch<'w> = ($($F::Fetch<'w>,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($F::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($F,)+) = state;
                $($F::matches($F, archetype))||+
            }

            fn update_access(state: &Self::State, access: &mut Access) {
                let ($($F,)+) = state;
                $($F::update_access($F, access);)+
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: &'w Archetype,
                sparse_sets: &'w SparseSets,
                last_run: Tick,
                this_run: Tick,
            ) -> Self::Fetch<'w> {
                let ($($F,)+) = state;
                // SAFETY: forwarded. A branch whose component is absent from the
                // archetype builds a "no-match" fetch (e.g. `None`) rather than
                // being unsound, because each leaf `init_fetch` tolerates a
                // missing column for the `Or` case.
                unsafe { ($($F::init_fetch($F, archetype, sparse_sets, last_run, this_run),)+) }
            }

            unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
                let ($($F,)+) = fetch;
                // SAFETY: forwarded — same row for every element.
                $((unsafe { $F::filter_fetch($F, entity, row) }))||+
            }
        }
    };
}

impl_filter_tuple!(A);
impl_filter_tuple!(A, B);
impl_filter_tuple!(A, B, C);
impl_filter_tuple!(A, B, C, D);
impl_filter_tuple!(A, B, C, D, E);
impl_filter_tuple!(A, B, C, D, E, F);
impl_filter_tuple!(A, B, C, D, E, F, G);
impl_filter_tuple!(A, B, C, D, E, F, G, H);
