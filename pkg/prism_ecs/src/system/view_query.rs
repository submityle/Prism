//! The "entity being processed right now" system parameter:
//! [`CurrentView`] + [`ViewQuery`].
//!
//! A recording system — the renderer's per-view work is the motivating user —
//! usually wants exactly one entity: *the one this pass is currently recording
//! for*. The entity is published by the outer loop (the render graph sets it
//! before dispatching a view's systems) as the [`CurrentView`] resource, and
//! [`ViewQuery`] is the parameter that resolves a normal query fetch against
//! that entity — the analogue of `query.get(CurrentView)` without every system
//! having to spell out the lookup and its failure handling.
//!
//! It is a direct port of `bevy_render`'s `CurrentView` / `ViewQuery` /
//! `ViewQueryState` (`crates/bevy_render/src/renderer/render_context.rs`), which
//! `prism_render_scene` uses at ~154 sites, so the Prism spelling deliberately
//! matches: [`ViewQuery::entity`] and [`ViewQuery::into_inner`] are the same two
//! accessors with the same signatures.
//!
//! # Why this lives in `prism_ecs` (and not in the render app)
//!
//! Two hard constraints, both verified against the tree:
//!
//! 1. [`SystemParam`] is an `unsafe trait`, and `prism_render_app` is
//!    `#![forbid(unsafe_code)]` — the `unsafe impl` cannot be written there.
//! 2. The one engine that resolves a single row from an [`UnsafeWorldCell`] is
//!    [`QueryState::get_from_ptr`], which is `pub(crate)`. A downstream crate
//!    cannot reach it at all.
//!
//! So the parameter lives here, next to the other param impls, and the render
//! app re-exports it (`prism_render_app::view_query`). The concept stays
//! ECS-generic in this crate: "the entity currently being processed", which is
//! exactly what a render graph needs per view.
//!
//! # Failure behaviour: no per-param skip, so absence is `None`
//!
//! Bevy's `get_param` returns `Result<_, SystemParamValidationError>` and a
//! failed param *skips the whole system* for that run. **`prism_ecs` has no such
//! mechanism**: [`SystemParam::get_param`] returns the item directly, and there
//! is no `SystemParamValidationError` (nor any equivalent) in the crate — the
//! only run-level gate is a `run_if` condition on the schedule
//! (`resource_exists`, `run_once`, `in_state`, …).
//!
//! Rather than panic (the one thing the port must not do) or silently read a
//! different entity, this module implements the parameter as
//! `Option<ViewQuery<..>>`: the same shape the crate already uses for
//! `Option<Res<T>>` / `Option<ResMut<T>>`. The lookup yields
//!
//! * `None` when [`CurrentView`] is absent, and
//! * `None` when the named entity is not live, or its component set (or its
//!   per-row `Added` / `Changed` / `With` / `Without` state) does not satisfy
//!   the query,
//!
// so a ported system observes exactly the fact Bevy would encode by skipping,
// and says so at the call site:
//!
//! ```no_run
//! use prism_ecs::prelude::*;
//! use prism_ecs::system::ViewQuery;
//!
//! #[derive(Debug)]
//! struct Depth(i32);
//! impl Component for Depth {}
//!
//! fn record(view: Option<ViewQuery<&mut Depth>>) {
//!     // The Bevy `ViewQuery`-skip, spelled out: absent resource or a
//!     // non-matching current view means this system does nothing this run.
//!     let Some(mut view) = view else { return };
//!     let _entity = view.entity();
//!     view.into_inner().0 += 1;
//! }
//! ```
//!
//! A system that wants the raw entity without a query takes
//! `Res<CurrentView>` (or `Option<Res<CurrentView>>`) directly.
//!
//! [`SystemParam`]: super::SystemParam
//! [`SystemParam::get_param`]: super::SystemParam::get_param
//! [`QueryState`]: crate::query::QueryState

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::entity::Entity;
use crate::query::{Access, QueryData, QueryFilter, QueryState};
use crate::resource::{Resource, ResourceId};
use crate::system::world_cell::UnsafeWorldCell;
use crate::system::SystemParam;
use crate::world::World;

/// The entity whose view the renderer is currently recording.
///
/// Published by whoever drives the per-view loop (the render graph sets it
/// before dispatching a view's systems) and consumed by [`ViewQuery`]. It is a
/// resource rather than a query because there is exactly one "current" view at
/// a time per recording pass, and it changes between dispatches.
///
/// Derefs to the wrapped [`Entity`], mirroring Bevy's `CurrentView(pub Entity)`
/// so a ported `*current_view` / `current_view.entity()` keeps compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentView(pub Entity);

impl Resource for CurrentView {}

impl CurrentView {
    /// The entity this pass is recording.
    #[inline]
    pub const fn entity(self) -> Entity {
        self.0
    }
}

impl Deref for CurrentView {
    type Target = Entity;
    #[inline]
    fn deref(&self) -> &Entity {
        &self.0
    }
}

impl DerefMut for CurrentView {
    #[inline]
    fn deref_mut(&mut self) -> &mut Entity {
        &mut self.0
    }
}

/// A resolved fetch for the [`CurrentView`] entity's components.
///
/// Built by the `Option<ViewQuery<..>>` parameter; see the module docs for why
/// the parameter is the `Option` wrapper. `entity()` and `into_inner()` have
/// Bevy's exact signatures, so the accessors of a ported system do not change.
///
/// The `'s` and `F` parameters are carried only to keep the type's shape
/// identical to the source (`ViewQuery<'w, 's, D, F>`): in `prism_ecs` a
/// [`QueryData::Item`] borrows the world for one lifetime, so the item does not
/// borrow the state. The marker is a `fn() -> F` so the type stays `Send`/`Sync`
/// for any filter (the same reasoning as [`QueryState`]'s own marker).
pub struct ViewQuery<'w, 's, D: QueryData, F: QueryFilter = ()> {
    entity: Entity,
    item: D::Item<'w>,
    _marker: PhantomData<(&'s (), fn() -> F)>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> ViewQuery<'w, 's, D, F> {
    /// The entity this fetch resolved, i.e. the [`CurrentView`] at the time the
    /// system was dispatched.
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// The fetched query terms — the same value
    /// `Query::get(current_view.entity())` would have produced.
    #[inline]
    pub fn into_inner(self) -> D::Item<'w> {
        self.item
    }
}

/// Per-system state of the `Option<ViewQuery<..>>` parameter: the resolved
/// [`CurrentView`] resource id plus the query state resolved in `init_state`.
///
/// Public because it is the associated [`SystemParam::State`] type (a private
/// type in a public trait impl would not compile); its fields stay private and
/// only [`SystemParam::init_state`] ever constructs it.
pub struct ViewQueryState<D: QueryData, F: QueryFilter> {
    /// Id of the `CurrentView` resource, minted once in `init_state`.
    view_resource: ResourceId,
    /// The resolved query, built against the same world.
    query_state: QueryState<D, F>,
}

impl<D: QueryData, F: QueryFilter> ViewQueryState<D, F> {
    /// The query state this parameter resolves against — for tests that assert
    /// the declared access, and for diagnostics.
    #[inline]
    pub fn query_state(&self) -> &QueryState<D, F> {
        &self.query_state
    }
}

// SAFETY: `update_access` declares a shared read of the `CurrentView` resource
// (the `&CurrentView` formed in `get_param` is the only borrow of it this param
// makes) and replays the query's own term reads/writes exactly as the `Query`
// parameter does, so the scheduler's conflict analysis sees every column this
// param touches. `get_param` forms only a `&World` from the cell and reaches
// through `Resources::get_ptr` / `QueryState::get_from_ptr`, yielding a shared
// resource borrow plus at most one item per row — no aliasing arises beyond what
// the declared access already excludes. `D::matches` / `F::matches` are
// re-checked for the named entity, so a non-matching entity yields `None`
// instead of a wrongly-typed fetch.
unsafe impl<D, F> SystemParam for Option<ViewQuery<'_, '_, D, F>>
where
    D: QueryData + 'static,
    F: QueryFilter + 'static,
    D::State: 'static,
    F::State: 'static,
{
    type State = ViewQueryState<D, F>;
    type Item<'w, 's> = Option<ViewQuery<'w, 's, D, F>>;

    #[inline]
    fn init_state(world: &mut World) -> Self::State {
        // Registering (not inserting) is what `Res`/`ResMut` do: the id is
        // needed for the access declaration, and absence is a run-time fact.
        ViewQueryState {
            view_resource: world.resources_mut().register::<CurrentView>(),
            query_state: QueryState::new(world),
        }
    }

    #[inline]
    fn update_access(state: &Self::State, access: &mut Access) {
        access.add_resource_read(state.view_resource);
        // Same replay the `Query` param performs: the shared access set must
        // name every component this fetch can touch.
        let query_access = state.query_state.access();
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
        cell: UnsafeWorldCell<'w>,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the caller guarantees `cell` is live for `'w`; a shared
        // `&World` is all this param needs.
        let world: &'w World = unsafe { cell.world() };
        // SAFETY: `view_resource` is the id minted for `CurrentView` in
        // `init_state` against this very world, and `update_access` declared a
        // shared read of it, so no `&mut CurrentView` is live. The value lives
        // in the resource store, which outlives `'w`.
        // An absent resource (or a pointer to the wrong type, impossible by
        // construction) means the current view is simply not set for this run.
        let ptr = unsafe { world.resources().get_ptr::<CurrentView>(state.view_resource) }?;
        // SAFETY: `ptr` points at the live `CurrentView`; the declared read
        // means nothing forms an exclusive borrow of it.
        let entity = unsafe { (*ptr).0 };
        // SAFETY: same `&World`/raw-pointer derivation as `Query::get`: the
        // state was built for this world, and the declared access covers every
        // column the fetch can touch. A non-matching entity is an `Err`, which
        // becomes `None` below rather than a fetch of some other entity.
        let item = unsafe {
            state.query_state.get_from_ptr(
                cell.as_ptr(),
                entity,
                cell.last_run(),
                cell.this_run(),
            )
        };
        // The analogue of Bevy's `SystemParamValidationError::skipped`: the
        // entity is stale, despawned, or filtered out, so this run has no view.
        let item = item.ok()?;
        Some(ViewQuery {
            entity,
            item,
            _marker: PhantomData,
        })
    }
}
