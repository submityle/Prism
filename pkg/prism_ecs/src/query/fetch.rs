//! [`QueryData`]: the per-term read/write access of a query and the typed items
//! it yields.
//!
//! A `QueryData` describes *what data* a query reads out of each matched
//! archetype row. The supported terms are:
//!
//! - `&T` — shared access to component `T`.
//! - `&mut T` — exclusive access to component `T`.
//! - [`Entity`] — the entity id of the current row (no component access).
//! - `Option<&T>` / `Option<&mut T>` — the component if the archetype has it,
//!   else `None` (the term never excludes an archetype).
//! - [`Ref<T>`](crate::change::Ref) — shared access plus per-value change
//!   detection (`is_added`/`is_changed`), the read-only companion of the
//!   change-detecting [`Mut<T>`](crate::change::Mut) yielded by `&mut T`.
//! - tuples of the above, up to 12 elements.
//!
//! Each term exposes three cooperating pieces: a world-static `State` (resolved
//! [`ComponentId`]s), a per-archetype `Fetch` cursor (resolved column
//! references), and the leaf [`QueryData::fetch`] that reads one row.

use crate::archetype::Archetype;
use crate::change::{Mut, Ref, Tick};
use crate::component::{Component, ComponentId, Components};
use crate::entity::Entity;
use crate::query::access::Access;
use crate::storage::Column;

/// A query term describing the typed data read from each matched row.
///
/// # Safety
/// Implementors must uphold the following so that [`QueryIter`](crate::query::QueryIter)
/// can hand out borrows soundly:
///
/// - [`QueryData::update_access`] must register **every** component the term
///   reads via [`Access::add_read`] and every component it writes via
///   [`Access::add_write`]. The query driver relies on this to reject aliasing
///   `&mut` borrows before iteration begins.
/// - [`QueryData::matches`] must return `true` only for archetypes from which
///   [`QueryData::init_fetch`] can build a valid fetch, and
///   [`QueryData::fetch`] reads only components covered by `update_access`.
/// - For a `&mut`/`Option<&mut>` term, [`QueryData::fetch`] forms a unique
///   `&mut` into the row; the driver guarantees each `(archetype, row)` is
///   fetched at most once per iteration, so these borrows never alias.
pub unsafe trait QueryData {
    /// The item yielded for one row, borrowing the world for `'w`.
    type Item<'w>;
    /// World-static resolved state (component ids), computed once.
    type State: Send + Sync;
    /// Per-archetype resolved cursor (column references). Must be `Copy` so the
    /// iterator can read it per row without consuming it.
    type Fetch<'w>: Copy;

    /// Resolve this term's [`State`](Self::State), registering any component
    /// types it names.
    fn init_state(components: &mut Components) -> Self::State;

    /// Whether `archetype` can satisfy this term (has the required columns).
    fn matches(state: &Self::State, archetype: &Archetype) -> bool;

    /// Record this term's component reads/writes into `access`.
    fn update_access(state: &Self::State, access: &mut Access);

    /// Resolve the per-archetype fetch cursor.
    ///
    /// `last_run`/`this_run` are the querying system's observer window; terms
    /// that report change detection (`&mut T`, `Option<&mut T>`, [`Ref<T>`])
    /// thread them into their [`Item`](QueryData::Item), while plain reads
    /// ignore them.
    ///
    /// # Safety
    /// `archetype` must satisfy [`QueryData::matches`] for `state`.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w>;

    /// Read the item at `row` of the archetype the `fetch` was built for.
    ///
    /// # Safety
    /// `row` must be `< archetype.len()` for the archetype `fetch` was built
    /// for, and for any `&mut` term the driver must not fetch the same
    /// `(archetype, row)` more than once while a prior item is still alive.
    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w>;
}

// --- &T ---------------------------------------------------------------------

// SAFETY: `update_access` registers the single read; `matches` requires the
// column to exist so `init_fetch` always finds it; `fetch` reads only that
// column as `T`, the exact type registered for the id.
unsafe impl<T: Component> QueryData for &T {
    type Item<'w> = &'w T;
    type State = ComponentId;
    type Fetch<'w> = &'w Column;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.contains(*state)
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        archetype
            .table()
            .column(*state)
            .expect("matches() guaranteed the column exists")
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, _entity: Entity, row: usize) -> Self::Item<'w> {
        // SAFETY: `row < len` per the caller's contract, the column stores `T`,
        // and shared `&T` access cannot alias a mutable borrow (enforced by the
        // access conflict check). The lifetime is tied to `'w` — the world
        // borrow held for the whole iteration.
        unsafe { &*fetch.get_ptr(row).cast::<T>() }
    }
}

// --- &mut T -----------------------------------------------------------------

// SAFETY: `update_access` registers the single write (so no other term may read
// or write the same component); `matches`/`init_fetch` mirror `&T`; `fetch`
// forms a `&mut T` that is unique because the driver yields each row once.
unsafe impl<T: Component> QueryData for &mut T {
    type Item<'w> = Mut<'w, T>;
    type State = ComponentId;
    type Fetch<'w> = (&'w Column, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.contains(*state)
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_write(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let column = archetype
            .table()
            .column(*state)
            .expect("matches() guaranteed the column exists");
        (column, last_run, this_run)
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, _entity: Entity, row: usize) -> Self::Item<'w> {
        let (col, last_run, this_run) = fetch;
        let added = col.added_tick(row);
        // SAFETY: `row < len`; the column stores `T`; the write is exclusive
        // (access check rejects any other borrow of this component) and the
        // driver fetches each `(archetype, row)` at most once, so this `&mut T`
        // is unique for `'w`. The value bytes and the changed-tick cell live in
        // separate allocations, so the two `&mut` below never alias.
        let value = unsafe { &mut *col.get_ptr(row).cast::<T>() };
        // SAFETY: as above — unique access to this row's changed-tick cell.
        let changed = unsafe { &mut *col.changed_tick_ptr(row) };
        Mut::new(value, changed, added, last_run, this_run)
    }
}

// --- Entity -----------------------------------------------------------------

// SAFETY: yields a `Copy` id, touches no component storage, registers no access.
unsafe impl QueryData for Entity {
    type Item<'w> = Entity;
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
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
    }

    unsafe fn fetch<'w>(_fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> Self::Item<'w> {
        entity
    }
}

// --- Option<&T> / Option<&mut T> -------------------------------------------

// SAFETY: optional terms never exclude an archetype; access is registered
// unconditionally (the component may be read/written where present); `fetch`
// yields `None` when the column is absent and otherwise behaves like `&T`.
unsafe impl<T: Component> QueryData for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type State = ComponentId;
    type Fetch<'w> = Option<&'w Column>;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        archetype.table().column(*state)
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, _entity: Entity, row: usize) -> Self::Item<'w> {
        // SAFETY: when `Some`, `row < len` and the column stores `T`; shared
        // access cannot alias a mutable borrow.
        fetch.map(|col| unsafe { &*col.get_ptr(row).cast::<T>() })
    }
}

// SAFETY: as `Option<&T>` but exclusive; `update_access` registers a write so
// no other term may touch the component, and `fetch` forms a unique `&mut`.
unsafe impl<T: Component> QueryData for Option<&mut T> {
    type Item<'w> = Option<Mut<'w, T>>;
    type State = ComponentId;
    type Fetch<'w> = (Option<&'w Column>, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_write(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        (archetype.table().column(*state), last_run, this_run)
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, _entity: Entity, row: usize) -> Self::Item<'w> {
        let (column, last_run, this_run) = fetch;
        column.map(|col| {
            let added = col.added_tick(row);
            // SAFETY: when `Some`, `row < len`, the column stores `T`, the write
            // is exclusive, and each row is fetched once — so the `&mut T` is
            // unique. Value bytes and the changed-tick cell are distinct
            // allocations, so the two `&mut` below do not alias.
            let value = unsafe { &mut *col.get_ptr(row).cast::<T>() };
            // SAFETY: as above — unique access to this row's changed-tick cell.
            let changed = unsafe { &mut *col.changed_tick_ptr(row) };
            Mut::new(value, changed, added, last_run, this_run)
        })
    }
}

// --- Ref<T> -----------------------------------------------------------------

// SAFETY: `Ref<T>` reads a single component immutably (registering only a read)
// and additionally reads that component's change ticks; `matches` requires the
// column to exist so `init_fetch` always finds it; `fetch` forms a `&T` plus
// `Copy` tick snapshots, never a `&mut` into storage.
unsafe impl<T: Component> QueryData for Ref<'_, T> {
    type Item<'w> = Ref<'w, T>;
    type State = ComponentId;
    type Fetch<'w> = (&'w Column, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.contains(*state)
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let column = archetype
            .table()
            .column(*state)
            .expect("matches() guaranteed the column exists");
        (column, last_run, this_run)
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, _entity: Entity, row: usize) -> Self::Item<'w> {
        let (col, last_run, this_run) = fetch;
        // SAFETY: `row < len`; the column stores `T`; shared `&T` access cannot
        // alias a mutable borrow (the access check rejects a conflicting write).
        let value = unsafe { &*col.get_ptr(row).cast::<T>() };
        Ref::new(value, col.added_tick(row), col.changed_tick(row), last_run, this_run)
    }
}

// --- Tuples -----------------------------------------------------------------

macro_rules! impl_query_data_tuple {
    ($($T:ident),+) => {
        // SAFETY: each element is a `QueryData` upholding the trait contract;
        // the tuple registers the union of their accesses, matches only when
        // all elements match, and fetches each element at the same row — so the
        // per-element soundness arguments compose.
        #[allow(non_snake_case)]
        unsafe impl<$($T: QueryData),+> QueryData for ($($T,)+) {
            type Item<'w> = ($($T::Item<'w>,)+);
            type State = ($($T::State,)+);
            type Fetch<'w> = ($($T::Fetch<'w>,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($T::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($T,)+) = state;
                $($T::matches($T, archetype))&&+
            }

            fn update_access(state: &Self::State, access: &mut Access) {
                let ($($T,)+) = state;
                $($T::update_access($T, access);)+
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: &'w Archetype,
                last_run: Tick,
                this_run: Tick,
            ) -> Self::Fetch<'w> {
                let ($($T,)+) = state;
                // SAFETY: forwarded — `matches` held for every element.
                unsafe { ($($T::init_fetch($T, archetype, last_run, this_run),)+) }
            }

            unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
                let ($($T,)+) = fetch;
                // SAFETY: forwarded — same row for every element; distinct
                // components so the `&mut` terms do not alias each other.
                unsafe { ($($T::fetch($T, entity, row),)+) }
            }
        }
    };
}

impl_query_data_tuple!(A);
impl_query_data_tuple!(A, B);
impl_query_data_tuple!(A, B, C);
impl_query_data_tuple!(A, B, C, D);
impl_query_data_tuple!(A, B, C, D, E);
impl_query_data_tuple!(A, B, C, D, E, F);
impl_query_data_tuple!(A, B, C, D, E, F, G);
impl_query_data_tuple!(A, B, C, D, E, F, G, H);
impl_query_data_tuple!(A, B, C, D, E, F, G, H, I);
impl_query_data_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_query_data_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_query_data_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

// --- Read-only marker -------------------------------------------------------

/// Marker for [`QueryData`] terms that borrow component data only *immutably*.
///
/// # Safety
/// Implementors must register no write access in [`QueryData::update_access`]
/// and must never form a `&mut` into component storage in [`QueryData::fetch`].
/// This lets [`QueryState::iter`](crate::query::QueryState::iter) hand out items
/// from a shared `&World` borrow soundly.
pub unsafe trait ReadOnlyQueryData: QueryData {}

// SAFETY: `&T` reads a single component immutably and registers only a read.
unsafe impl<T: Component> ReadOnlyQueryData for &T {}

// SAFETY: `Entity` touches no component storage and registers no access.
unsafe impl ReadOnlyQueryData for Entity {}

// SAFETY: `Option<&T>` reads a single component immutably when present.
unsafe impl<T: Component> ReadOnlyQueryData for Option<&T> {}

// SAFETY: `Ref<T>` reads a single component immutably (plus its ticks); it
// forms no `&mut` into storage and registers only a read.
unsafe impl<T: Component> ReadOnlyQueryData for Ref<'_, T> {}

macro_rules! impl_read_only_query_data_tuple {
    ($($T:ident),+) => {
        // SAFETY: a tuple of read-only terms performs only immutable reads, so
        // the composed term is itself read-only.
        unsafe impl<$($T: ReadOnlyQueryData),+> ReadOnlyQueryData for ($($T,)+) {}
    };
}

impl_read_only_query_data_tuple!(A);
impl_read_only_query_data_tuple!(A, B);
impl_read_only_query_data_tuple!(A, B, C);
impl_read_only_query_data_tuple!(A, B, C, D);
impl_read_only_query_data_tuple!(A, B, C, D, E);
impl_read_only_query_data_tuple!(A, B, C, D, E, F);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G, H);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G, H, I);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_read_only_query_data_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
