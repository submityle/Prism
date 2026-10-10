//! [`QueryData`]: the per-term read/write access of a query and the typed items
//! it yields.
//!
//! A `QueryData` describes *what data* a query reads out of each matched
//! archetype row. The supported terms are:
//!
//! - `&T` — shared access to component `T`.
//! - `&mut T` — exclusive access to component `T`.
//! - [`Entity`] — the entity id of the current row (no component access).
//! - `Option<&T>` / `Option<&mut T>` — the component if the entity has it,
//!   else `None` (the term never excludes an archetype).
//! - [`Ref<T>`](crate::change::Ref) — shared access plus per-value change
//!   detection (`is_added`/`is_changed`), the read-only companion of the
//!   change-detecting [`Mut<T>`](crate::change::Mut) yielded by `&mut T`.
//! - [`Has<T>`] — a `bool` reporting whether the entity has `T`, without
//!   excluding any archetype and without reading the value (registers no
//!   access).
//! - [`AnyOf<(..)>`] — matches an archetype when *at least one* element is
//!   present, yielding each element as an `Option` (the `||` dual of a
//!   tuple's `&&`).
//! - tuples of the above, up to 12 elements.
//!
//! Each term exposes four cooperating pieces: a world-static `State` (resolved
//! [`ComponentId`]s), a per-archetype `Fetch` cursor (resolved storage
//! references), a per-row [`QueryData::filter_fetch`] presence gate, and the
//! leaf [`QueryData::fetch`] that reads one row.
//!
//! # Table vs SparseSet (design §6)
//!
//! A term abstracts over the two storage states its component may use:
//!
//! - A **table** component lives in the archetype's columnar table, so its
//!   presence is a property of the *archetype* — [`QueryData::matches`] decides
//!   it once per archetype and [`QueryData::filter_fetch`] always passes.
//! - A **sparse** component ([`StorageType::SparseSet`]) lives out of band in
//!   the world's [`SparseSets`] registry keyed by [`Entity`], so toggling it
//!   never moves the entity between archetypes. Its presence is therefore
//!   *per-entity, not per-archetype*: [`QueryData::matches`] must admit **every**
//!   archetype and [`QueryData::filter_fetch`] resolves membership per row.
//!
//! [`StorageFetch`] carries whichever cursor the term resolved for the current
//! archetype so a single `Fetch` type covers both paths.

use core::marker::PhantomData;

use crate::archetype::Archetype;
use crate::change::{Mut, Ref, Tick};
use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::access::Access;
use alloc::sync::Arc;

use crate::storage::{Column, ComponentSparseSet, SharedValue, SparseSets};

/// Resolved per-archetype storage cursor for one component term, abstracting
/// over table-backed and sparse-backed storage (design §6).
#[derive(Clone, Copy)]
pub enum StorageFetch<'w> {
    /// Table-backed column. `None` only for an `Option<&T>`/`Option<&mut T>`
    /// term in an archetype that lacks the column; required terms always carry
    /// `Some` because [`QueryData::matches`] guaranteed the column exists.
    Table(Option<&'w Column>),
    /// Sparse-backed set. `None` if the sparse component has never been written
    /// (its set was never allocated); otherwise membership is still resolved
    /// per entity by [`QueryData::filter_fetch`].
    Sparse(Option<&'w ComponentSparseSet>),
    /// Shared-backed binding (design §6 SharedComponent): the archetype-level
    /// canonical [`Arc`] every entity here shares, or `None` for an
    /// `Option<&T>` term in an archetype that lacks the binding. A shared
    /// component is immutable and archetype-wide, so there is no per-row
    /// cursor — [`QueryData::fetch`] downcasts the same handle for every row.
    Shared(Option<&'w Arc<dyn SharedValue>>),
}

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
/// - [`QueryData::filter_fetch`] must return `false` for any row where
///   [`QueryData::fetch`] could not produce a valid item (e.g. a sparse
///   component the entity does not have), so a required term never fetches an
///   absent value.
/// - For a `&mut`/`Option<&mut>` term, [`QueryData::fetch`] forms a unique
///   `&mut` into the row; the driver guarantees each `(archetype, row)` is
///   fetched at most once per iteration, so these borrows never alias.
pub unsafe trait QueryData {
    /// The item yielded for one row, borrowing the world for `'w`.
    type Item<'w>;
    /// World-static resolved state (component ids), computed once.
    type State: Send + Sync;
    /// Per-archetype resolved cursor (storage references). Must be `Copy` so the
    /// iterator can read it per row without consuming it.
    type Fetch<'w>: Copy;

    /// Resolve this term's [`State`](Self::State), registering any component
    /// types it names.
    fn init_state(components: &mut Components) -> Self::State;

    /// Whether `archetype` can satisfy this term. A table term requires the
    /// column; a sparse term admits every archetype (membership is resolved per
    /// row by [`filter_fetch`](QueryData::filter_fetch)); an optional term
    /// admits every archetype.
    fn matches(state: &Self::State, archetype: &Archetype) -> bool;

    /// Record this term's component reads/writes into `access`.
    fn update_access(state: &Self::State, access: &mut Access);

    /// Resolve the per-archetype fetch cursor.
    ///
    /// `sparse_sets` is the world's out-of-band registry (design §6), consulted
    /// by sparse-backed terms. `last_run`/`this_run` are the querying system's
    /// observer window; terms that report change detection (`&mut T`,
    /// `Option<&mut T>`, [`Ref<T>`]) thread them into their
    /// [`Item`](QueryData::Item), while plain reads ignore them.
    ///
    /// # Safety
    /// `archetype` must satisfy [`QueryData::matches`] for `state`.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w>;

    /// Per-row presence gate, evaluated **before** [`fetch`](QueryData::fetch).
    ///
    /// Table-backed terms always pass (matching the archetype already proves the
    /// column is present), so the default returns `true`. SparseSet-backed
    /// required terms (design §6) override this to resolve per-entity
    /// membership, because a sparse component's presence is not encoded in the
    /// archetype component set. Optional terms keep the default (`true`) and
    /// surface absence as `None` from `fetch` instead.
    ///
    /// # Safety
    /// `row` must be `< archetype.len()` for the archetype `fetch` was built
    /// for.
    unsafe fn filter_fetch<'w>(_fetch: Self::Fetch<'w>, _entity: Entity, _row: usize) -> bool {
        true
    }

    /// Read the item at `row` of the archetype the `fetch` was built for.
    ///
    /// # Safety
    /// `row` must be `< archetype.len()` for the archetype `fetch` was built
    /// for, [`filter_fetch`](QueryData::filter_fetch) must have admitted this
    /// row, and for any `&mut` term the driver must not fetch the same
    /// `(archetype, row)` more than once while a prior item is still alive.
    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w>;
}

// --- &T ---------------------------------------------------------------------

// SAFETY: `update_access` registers the single read; `matches` admits only
// archetypes/rows `filter_fetch` can satisfy (table column present, or sparse
// membership checked per row); `fetch` reads only that component as `T`, the
// exact type registered for the id.
unsafe impl<T: Component> QueryData for &T {
    type Item<'w> = &'w T;
    type State = ComponentId;
    type Fetch<'w> = StorageFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            // Sparse components live out of band, so every archetype may hold
            // entities that have one; membership is resolved per row below.
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // A shared component splits archetypes by value (design §6); its
            // presence is the per-archetype binding, resolved once here.
            StorageType::Shared => archetype.shared_binding(*state).is_some(),
        }
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => StorageFetch::Table(Some(
                archetype
                    .table()
                    .column(*state)
                    .expect("matches() guaranteed the column exists"),
            )),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            StorageType::Shared => StorageFetch::Shared(Some(
                archetype
                    .shared_arc(*state)
                    .expect("matches() guaranteed the shared binding exists"),
            )),
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> bool {
        match fetch {
            // Table term already matched the whole archetype.
            StorageFetch::Table(_) => true,
            // Sparse term is present only where the entity has a dense row.
            StorageFetch::Sparse(set) => set.is_some_and(|s| s.contains(entity)),
            // Shared presence is archetype-wide (`matches` already decided it).
            StorageFetch::Shared(_) => true,
        }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        match fetch {
            // SAFETY: `row < len` per the caller's contract, the column stores
            // `T`, and shared `&T` access cannot alias a mutable borrow
            // (enforced by the access conflict check). The lifetime is tied to
            // `'w` — the world borrow held for the whole iteration.
            StorageFetch::Table(Some(col)) => unsafe { &*col.get_ptr(row).cast::<T>() },
            // SAFETY: `filter_fetch` admitted this row only when the set holds
            // `entity`, so `get` is `Some`; the set stores `T`, and shared `&T`
            // access cannot alias a mutable borrow.
            StorageFetch::Sparse(Some(set)) => unsafe { set.get::<T>(entity) }
                .expect("filter_fetch gate guaranteed the sparse component is present"),
            // The shared value is interned once and immutable; downcast the
            // archetype-wide handle to the registered type `T`.
            StorageFetch::Shared(Some(arc)) => arc
                .as_any()
                .downcast_ref::<T>()
                .expect("shared binding stores the registered component type"),
            StorageFetch::Table(None) | StorageFetch::Sparse(None) | StorageFetch::Shared(None) => {
                unreachable!("required `&T` fetched a row without the component")
            }
        }
    }
}

// --- &mut T -----------------------------------------------------------------

// SAFETY: `update_access` registers the single write (so no other term may read
// or write the same component); `matches`/`init_fetch`/`filter_fetch` mirror
// `&T`; `fetch` forms a `&mut T` that is unique because the driver yields each
// row once.
unsafe impl<T: Component> QueryData for &mut T {
    type Item<'w> = Mut<'w, T>;
    type State = ComponentId;
    type Fetch<'w> = (StorageFetch<'w>, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        assert!(
            T::STORAGE != StorageType::Shared,
            "a shared component (design §6) is immutable and archetype-wide; it supports only `&T`/`Option<&T>`/`With`/`Without`, not mutable or change-detecting access"
        );
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // `init_state` already rejected a shared component for this term.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_write(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let storage = match T::STORAGE {
            StorageType::Table => StorageFetch::Table(Some(
                archetype
                    .table()
                    .column(*state)
                    .expect("matches() guaranteed the column exists"),
            )),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            // `init_state` already rejected a shared component for this term.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        };
        (storage, last_run, this_run)
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> bool {
        match fetch.0 {
            StorageFetch::Table(_) => true,
            StorageFetch::Sparse(set) => set.is_some_and(|s| s.contains(entity)),
            // `init_state` already rejected a shared component for this term.
            StorageFetch::Shared(_) => unreachable!("shared storage rejected at init_state"),
        }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        let (storage, last_run, this_run) = fetch;
        match storage {
            StorageFetch::Table(Some(col)) => {
                let added = col.added_tick(row);
                // SAFETY: `row < len`; the column stores `T`; the write is
                // exclusive (access check rejects any other borrow) and the
                // driver fetches each `(archetype, row)` at most once, so this
                // `&mut T` is unique for `'w`. The value bytes and the
                // changed-tick cell live in separate allocations, so the two
                // `&mut` below never alias.
                let value = unsafe { &mut *col.get_ptr(row).cast::<T>() };
                // SAFETY: as above — unique access to this row's changed-tick
                // cell.
                let changed = unsafe { &mut *col.changed_tick_ptr(row) };
                // SAFETY: `row < len`; the raw chunk-version pointer is only
                // written (with `this_run`) by `Mut`, never turned into an
                // aliasing `&mut`.
                let chunk_changed = unsafe { col.chunk_changed_ptr(row) };
                Mut::new(
                    value,
                    changed,
                    Some(chunk_changed),
                    added,
                    last_run,
                    this_run,
                )
            }
            StorageFetch::Sparse(Some(set)) => {
                // SAFETY: `filter_fetch` admitted this row only when the set
                // holds `entity`, so these lookups are `Some`.
                let ptr = unsafe { set.get_ptr(entity) }
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                // SAFETY: the set stores `T`; the write is exclusive (access
                // check) and each entity is fetched once, so this `&mut T` is
                // unique for `'w`. The value bytes and the changed-tick cell are
                // distinct allocations, so the two `&mut` below never alias.
                let value = unsafe { &mut *ptr.cast::<T>() };
                // SAFETY: unique per-row access to the changed-tick cell.
                let changed_ptr = unsafe { set.changed_tick_ptr(entity) }
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                // SAFETY: as above — unique access to this entity's tick cell.
                let changed = unsafe { &mut *changed_ptr };
                let added = set
                    .added_tick(entity)
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                // A sparse set has no coarse chunk-version layer (design §6), so
                // `Mut` carries `None` and skips the chunk-version write.
                Mut::new(value, changed, None, added, last_run, this_run)
            }
            StorageFetch::Table(None) | StorageFetch::Sparse(None) => {
                unreachable!("required `&mut T` fetched a row without the component")
            }
            // `init_state` already rejected a shared component for this term.
            StorageFetch::Shared(_) => unreachable!("shared storage rejected at init_state"),
        }
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
        _sparse_sets: &'w SparseSets,
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
// yields `None` when the component is absent and otherwise behaves like `&T`.
// `filter_fetch` keeps the default (`true`): an optional term never gates a row.
unsafe impl<T: Component> QueryData for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type State = ComponentId;
    type Fetch<'w> = StorageFetch<'w>;

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
        sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => StorageFetch::Table(archetype.table().column(*state)),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            StorageType::Shared => StorageFetch::Shared(archetype.shared_arc(*state)),
        }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        match fetch {
            // SAFETY: when `Some`, `row < len` and the column stores `T`; shared
            // access cannot alias a mutable borrow.
            StorageFetch::Table(opt) => opt.map(|col| unsafe { &*col.get_ptr(row).cast::<T>() }),
            // SAFETY: the set stores `T`; `get` validates membership and returns
            // `None` for an absent entity. Shared access cannot alias a mutable
            // borrow.
            StorageFetch::Sparse(opt) => opt.and_then(|set| unsafe { set.get::<T>(entity) }),
            // Absence surfaces as `None`; a present binding downcasts the
            // immutable archetype-wide handle to `T` (design §6).
            StorageFetch::Shared(opt) => opt.map(|arc| {
                arc.as_any()
                    .downcast_ref::<T>()
                    .expect("shared binding stores the registered component type")
            }),
        }
    }
}

// SAFETY: as `Option<&T>` but exclusive; `update_access` registers a write so
// no other term may touch the component, and `fetch` forms a unique `&mut`.
unsafe impl<T: Component> QueryData for Option<&mut T> {
    type Item<'w> = Option<Mut<'w, T>>;
    type State = ComponentId;
    type Fetch<'w> = (StorageFetch<'w>, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        assert!(
            T::STORAGE != StorageType::Shared,
            "a shared component (design §6) is immutable and archetype-wide; it supports only `&T`/`Option<&T>`/`With`/`Without`, not mutable or change-detecting access"
        );
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
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let storage = match T::STORAGE {
            StorageType::Table => StorageFetch::Table(archetype.table().column(*state)),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            // `init_state` already rejected a shared component for this term.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        };
        (storage, last_run, this_run)
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        let (storage, last_run, this_run) = fetch;
        match storage {
            StorageFetch::Table(opt) => opt.map(|col| {
                let added = col.added_tick(row);
                // SAFETY: when `Some`, `row < len`, the column stores `T`, the
                // write is exclusive, and each row is fetched once — so the
                // `&mut T` is unique. Value bytes and the changed-tick cell are
                // distinct allocations, so the two `&mut` below do not alias.
                let value = unsafe { &mut *col.get_ptr(row).cast::<T>() };
                // SAFETY: as above — unique access to this row's changed-tick
                // cell.
                let changed = unsafe { &mut *col.changed_tick_ptr(row) };
                // SAFETY: `row < len`; the raw chunk-version pointer is only
                // written (with `this_run`) by `Mut`, never turned into an
                // aliasing `&mut`.
                let chunk_changed = unsafe { col.chunk_changed_ptr(row) };
                Mut::new(
                    value,
                    changed,
                    Some(chunk_changed),
                    added,
                    last_run,
                    this_run,
                )
            }),
            StorageFetch::Sparse(opt) => opt.and_then(|set| {
                // SAFETY: `get_ptr` validates membership and returns `None` for
                // an absent entity, so the `?` surfaces absence as `None`.
                let ptr = unsafe { set.get_ptr(entity) }?;
                // SAFETY: the set stores `T`; the write is exclusive (access
                // check) and each entity is fetched once, so this `&mut T` is
                // unique. Value bytes and the changed-tick cell are distinct
                // allocations, so the two `&mut` below do not alias.
                let value = unsafe { &mut *ptr.cast::<T>() };
                // SAFETY: unique per-row access to the changed-tick cell; the
                // entity is present (checked above).
                let changed_ptr = unsafe { set.changed_tick_ptr(entity) }
                    .expect("entity present: get_ptr returned Some");
                // SAFETY: as above — unique access to this entity's tick cell.
                let changed = unsafe { &mut *changed_ptr };
                let added = set
                    .added_tick(entity)
                    .expect("entity present: get_ptr returned Some");
                // A sparse set has no coarse chunk-version layer (design §6).
                Some(Mut::new(value, changed, None, added, last_run, this_run))
            }),
            // `init_state` already rejected a shared component for this term.
            StorageFetch::Shared(_) => unreachable!("shared storage rejected at init_state"),
        }
    }
}

// --- Ref<T> -----------------------------------------------------------------

// SAFETY: `Ref<T>` reads a single component immutably (registering only a read)
// and additionally reads that component's change ticks; `matches`/`filter_fetch`
// mirror `&T`; `fetch` forms a `&T` plus `Copy` tick snapshots, never a `&mut`
// into storage.
unsafe impl<T: Component> QueryData for Ref<'_, T> {
    type Item<'w> = Ref<'w, T>;
    type State = ComponentId;
    type Fetch<'w> = (StorageFetch<'w>, Tick, Tick);

    fn init_state(components: &mut Components) -> Self::State {
        assert!(
            T::STORAGE != StorageType::Shared,
            "a shared component (design §6) is immutable and archetype-wide; it supports only `&T`/`Option<&T>`/`With`/`Without`, not mutable or change-detecting access"
        );
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        match T::STORAGE {
            StorageType::SparseSet => true,
            StorageType::Table => archetype.contains(*state),
            // `init_state` already rejected a shared component for this term.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        }
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_read(*state);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let storage = match T::STORAGE {
            StorageType::Table => StorageFetch::Table(Some(
                archetype
                    .table()
                    .column(*state)
                    .expect("matches() guaranteed the column exists"),
            )),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            // `init_state` already rejected a shared component for this term.
            StorageType::Shared => unreachable!("shared storage rejected at init_state"),
        };
        (storage, last_run, this_run)
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> bool {
        match fetch.0 {
            StorageFetch::Table(_) => true,
            StorageFetch::Sparse(set) => set.is_some_and(|s| s.contains(entity)),
            // `init_state` already rejected a shared component for this term.
            StorageFetch::Shared(_) => unreachable!("shared storage rejected at init_state"),
        }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        let (storage, last_run, this_run) = fetch;
        match storage {
            StorageFetch::Table(Some(col)) => {
                // SAFETY: `row < len`; the column stores `T`; shared `&T` access
                // cannot alias a mutable borrow (the access check rejects a
                // conflicting write).
                let value = unsafe { &*col.get_ptr(row).cast::<T>() };
                Ref::new(
                    value,
                    col.added_tick(row),
                    col.changed_tick(row),
                    last_run,
                    this_run,
                )
            }
            StorageFetch::Sparse(Some(set)) => {
                // SAFETY: `filter_fetch` admitted this row only when the set
                // holds `entity`, so these lookups are `Some`.
                let ptr = unsafe { set.get_ptr(entity) }
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                // SAFETY: the set stores `T`; shared `&T` access cannot alias a
                // mutable borrow.
                let value = unsafe { &*ptr.cast::<T>() };
                let added = set
                    .added_tick(entity)
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                let changed = set
                    .changed_tick(entity)
                    .expect("filter_fetch gate guaranteed the sparse component is present");
                Ref::new(value, added, changed, last_run, this_run)
            }
            StorageFetch::Table(None) | StorageFetch::Sparse(None) => {
                unreachable!("required `Ref<T>` fetched a row without the component")
            }
            // `init_state` already rejected a shared component for this term.
            StorageFetch::Shared(_) => unreachable!("shared storage rejected at init_state"),
        }
    }
}

// --- Has<T> -----------------------------------------------------------------

/// A [`QueryData`] term yielding `bool`: whether the current row's entity has
/// component `T`, **without** excluding any archetype and **without** reading
/// the component's value.
///
/// `Has<T>` is the presence-probe companion of `Option<&T>`: where
/// `Option<&T>` borrows the value when present, `Has<T>` reports only presence.
/// Because it inspects presence metadata (an archetype's column set, or a
/// sparse set's per-entity membership) and never dereferences the stored value,
/// it registers **no** component access — so it never conflicts with a mutable
/// term on the same component, and `Query<(&mut T, Has<T>)>` is sound. It works
/// uniformly across all three storage states (design §6): table, sparse, and
/// shared.
///
/// ```
/// use prism_ecs::prelude::*;
///
/// #[derive(Debug)]
/// struct Shield(u32);
/// impl Component for Shield {}
/// #[derive(Debug)]
/// struct Unit;
/// impl Component for Unit {}
///
/// let mut world = World::new();
/// world.spawn((Unit, Shield(50)));
/// world.spawn(Unit); // no Shield
///
/// let state = world.query::<(Has<Shield>, &Unit)>();
/// let mut shielded = 0usize;
/// for (has_shield, _unit) in state.iter(&world) {
///     if has_shield {
///         shielded += 1;
///     }
/// }
/// assert_eq!(shielded, 1);
/// ```
pub struct Has<T>(PhantomData<fn() -> T>);

// SAFETY: `Has<T>` registers no access and forms no reference into component
// storage — it reads only presence metadata (archetype column membership for a
// table/shared component, or the sparse set's per-entity membership bit). It
// admits every archetype (`matches` is always `true`) and never gates a row
// (`filter_fetch` keeps the default `true`), yielding `bool` for every row.
unsafe impl<T: Component> QueryData for Has<T> {
    type Item<'w> = bool;
    type State = ComponentId;
    type Fetch<'w> = StorageFetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn update_access(_state: &Self::State, _access: &mut Access) {
        // Presence probing is not a data read: register nothing so `Has<T>`
        // never conflicts with another term's `&mut T`.
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
        match T::STORAGE {
            StorageType::Table => StorageFetch::Table(archetype.table().column(*state)),
            StorageType::SparseSet => StorageFetch::Sparse(sparse_sets.get(*state)),
            StorageType::Shared => StorageFetch::Shared(archetype.shared_arc(*state)),
        }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, _row: usize) -> Self::Item<'w> {
        match fetch {
            // Table/shared presence is archetype-wide: the cursor is `Some`
            // exactly when the archetype carries the column/binding.
            StorageFetch::Table(opt) => opt.is_some(),
            StorageFetch::Shared(opt) => opt.is_some(),
            // Sparse presence is per entity: the set may exist yet not hold this
            // row's entity, and may be absent entirely before first insertion.
            StorageFetch::Sparse(opt) => opt.is_some_and(|set| set.contains(entity)),
        }
    }
}

// --- Tuples -----------------------------------------------------------------

macro_rules! impl_query_data_tuple {
    ($($T:ident),+) => {
        // SAFETY: each element is a `QueryData` upholding the trait contract;
        // the tuple registers the union of their accesses, matches only when
        // all elements match, gates a row only when every element admits it,
        // and fetches each element at the same row — so the per-element
        // soundness arguments compose.
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
                sparse_sets: &'w SparseSets,
                last_run: Tick,
                this_run: Tick,
            ) -> Self::Fetch<'w> {
                let ($($T,)+) = state;
                // SAFETY: forwarded — `matches` held for every element.
                unsafe { ($($T::init_fetch($T, archetype, sparse_sets, last_run, this_run),)+) }
            }

            unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
                let ($($T,)+) = fetch;
                // SAFETY: forwarded — same row for every element. A tuple admits
                // a row only when every element's per-row gate admits it, so a
                // sparse element the entity lacks excludes the whole row.
                $((unsafe { $T::filter_fetch($T, entity, row) }))&&+
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

// --- AnyOf<(..)> ------------------------------------------------------------

/// A [`QueryData`] term matching an archetype when **at least one** of its
/// element terms is present, yielding each element as an `Option`.
///
/// Where a plain tuple `(A, B)` requires *every* element and excludes an
/// archetype that lacks any of them, `AnyOf<(A, B)>` includes an archetype that
/// carries *A or B (or both)* and yields `(Option<A::Item>, Option<B::Item>)`,
/// with `None` for the elements this row happens to lack. A row is visited only
/// when at least one element is actually present for it (so you never iterate
/// an all-`None` row), which makes `AnyOf` the natural term for "components that
/// play the same role but are stored under distinct types" (design §6/§7).
///
/// Each element is a full [`QueryData`] term, so `&T`, `&mut T`, and `Ref<T>`
/// all compose. `AnyOf` registers the **union** of its elements' accesses — the
/// same conservative rule as `Option<&T>` — so it still conflicts with another
/// term that writes a component it reads, and a self-conflicting set (such as
/// `AnyOf<(&mut T, &T)>`) is rejected by the access-conflict check.
///
/// ```
/// use prism_ecs::prelude::*;
///
/// #[derive(Debug)]
/// struct Melee(u32);
/// impl Component for Melee {}
/// #[derive(Debug)]
/// struct Ranged(u32);
/// impl Component for Ranged {}
///
/// let mut world = World::new();
/// world.spawn(Melee(10));           // only melee
/// world.spawn(Ranged(7));           // only ranged
/// world.spawn((Melee(3), Ranged(5))); // both
///
/// let state = world.query::<AnyOf<(&Melee, &Ranged)>>();
/// let mut armed = 0usize;
/// for (melee, ranged) in state.iter(&world) {
///     // At least one is always `Some` — all-`None` rows are never visited.
///     assert!(melee.is_some() || ranged.is_some());
///     armed += 1;
/// }
/// assert_eq!(armed, 3);
/// ```
pub struct AnyOf<T>(PhantomData<fn() -> T>);

macro_rules! impl_any_of {
    ($($T:ident),+) => {
        // SAFETY: each element is a `QueryData` upholding the trait contract.
        // `update_access` registers the union of element accesses (so a present
        // element is always covered); `matches` admits an archetype when *any*
        // element matches, and `init_fetch` builds a per-element fetch only for
        // the elements that matched (storing `None` otherwise), so no element's
        // `init_fetch` runs on an archetype it does not match. `filter_fetch`
        // admits a row only when some element is present for it, and `fetch`
        // re-checks each element's per-row gate before fetching — so every
        // `Some` item is backed by a real present component at this exact row,
        // and the per-element soundness arguments compose.
        #[allow(non_snake_case)]
        unsafe impl<$($T: QueryData),+> QueryData for AnyOf<($($T,)+)> {
            type Item<'w> = ($(Option<$T::Item<'w>>,)+);
            type State = ($($T::State,)+);
            type Fetch<'w> = ($(Option<$T::Fetch<'w>>,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($T::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($T,)+) = state;
                // OR, unlike the tuple's AND: one present element suffices.
                $($T::matches($T, archetype))||+
            }

            fn update_access(state: &Self::State, access: &mut Access) {
                let ($($T,)+) = state;
                $($T::update_access($T, access);)+
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: &'w Archetype,
                sparse_sets: &'w SparseSets,
                last_run: Tick,
                this_run: Tick,
            ) -> Self::Fetch<'w> {
                let ($($T,)+) = state;
                // SAFETY: an element's `init_fetch` runs only when its `matches`
                // held for this archetype — the per-element contract.
                unsafe {
                    ($(
                        if $T::matches($T, archetype) {
                            Some($T::init_fetch($T, archetype, sparse_sets, last_run, this_run))
                        } else {
                            None
                        },
                    )+)
                }
            }

            unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
                let ($($T,)+) = fetch;
                // SAFETY: forwarded per element at the same row; a `None`
                // element (archetype did not match it) contributes `false`.
                unsafe {
                    $(
                        (match $T {
                            Some(f) => $T::filter_fetch(f, entity, row),
                            None => false,
                        })
                    )||+
                }
            }

            unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
                let ($($T,)+) = fetch;
                // SAFETY: forwarded per element at the same row; distinct
                // components mean the `&mut` elements never alias each other,
                // and the per-row gate is re-checked so a `Some` item is always
                // backed by a present component at this row.
                unsafe {
                    ($(
                        match $T {
                            Some(f) if $T::filter_fetch(f, entity, row) => {
                                Some($T::fetch(f, entity, row))
                            }
                            _ => None,
                        },
                    )+)
                }
            }
        }
    };
}

impl_any_of!(A);
impl_any_of!(A, B);
impl_any_of!(A, B, C);
impl_any_of!(A, B, C, D);
impl_any_of!(A, B, C, D, E);
impl_any_of!(A, B, C, D, E, F);
impl_any_of!(A, B, C, D, E, F, G);
impl_any_of!(A, B, C, D, E, F, G, H);
impl_any_of!(A, B, C, D, E, F, G, H, I);
impl_any_of!(A, B, C, D, E, F, G, H, I, J);
impl_any_of!(A, B, C, D, E, F, G, H, I, J, K);
impl_any_of!(A, B, C, D, E, F, G, H, I, J, K, L);

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

// SAFETY: `Has<T>` probes presence only — it registers no access and forms
// no reference into component storage, so it is trivially read-only.
unsafe impl<T: Component> ReadOnlyQueryData for Has<T> {}

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

macro_rules! impl_any_of_read_only {
    ($($T:ident),+) => {
        // SAFETY: `AnyOf` only ever fetches its elements, so when every element
        // is read-only the composed term performs only immutable reads and
        // registers no write access.
        unsafe impl<$($T: ReadOnlyQueryData),+> ReadOnlyQueryData for AnyOf<($($T,)+)> {}
    };
}

impl_any_of_read_only!(A);
impl_any_of_read_only!(A, B);
impl_any_of_read_only!(A, B, C);
impl_any_of_read_only!(A, B, C, D);
impl_any_of_read_only!(A, B, C, D, E);
impl_any_of_read_only!(A, B, C, D, E, F);
impl_any_of_read_only!(A, B, C, D, E, F, G);
impl_any_of_read_only!(A, B, C, D, E, F, G, H);
impl_any_of_read_only!(A, B, C, D, E, F, G, H, I);
impl_any_of_read_only!(A, B, C, D, E, F, G, H, I, J);
impl_any_of_read_only!(A, B, C, D, E, F, G, H, I, J, K);
impl_any_of_read_only!(A, B, C, D, E, F, G, H, I, J, K, L);
