//! Per-view work: running a view's systems with that view as the "current" one.
//!
//! Bevy's render graph runs each view's node systems through a driver that
//! publishes the view's entity as `CurrentView` and hands them a `RenderContext`
//! scoped to that view; systems then read the view's components with `ViewQuery`
//! instead of querying for it themselves. Prism's equivalent is these two
//! functions: [`with_current_view`] scopes a closure to one view's entity, and
//! [`for_each_view`] walks the frame's materialized view entities, calling back
//! once per view with `CurrentView` set — and clearing it again afterwards, so no
//! system can observe a view from a previous iteration.
//!
//! The views come from [`RenderViews`], whose entries each own a render entity
//! carrying [`RenderView`]; the driver never invents a view.

use prism_ecs::prelude::Entity;
use prism_ecs::world::World;

use crate::view_query::CurrentView;
use crate::views::{RenderView, RenderViews};

/// Runs `f` with [`CurrentView`] naming `entity`, restoring the previous value.
///
/// Outside any per-view scope the resource is absent, and it is absent again
/// afterwards: a system that runs between views (or after the loop) sees no
/// current view at all, rather than a stale one. An existing `CurrentView` is
/// put back, so these scopes nest.
pub fn with_current_view<R>(
    world: &mut World,
    entity: Entity,
    f: impl FnOnce(&mut World) -> R,
) -> R {
    let previous = world.insert_resource(CurrentView(entity));
    let result = f(world);
    match previous {
        Some(previous) => {
            world.insert_resource(previous);
        }
        None => {
            world.remove_resource::<CurrentView>();
        }
    }
    result
}

/// Calls `f` once per materialized view, with [`CurrentView`] set to it.
///
/// The view list is snapshotted before the loop, so work that registers or
/// removes a view does not invalidate the iteration; a view added during the
/// walk is picked up by the next frame's walk, which is the same rule the
/// registry's enabled flags follow.
pub fn for_each_view(world: &mut World, mut f: impl FnMut(&mut World, Entity, RenderView)) {
    let views: Vec<(Entity, RenderView)> = world
        .get_resource::<RenderViews>()
        .map(|views| views.entities().collect())
        .unwrap_or_default();
    for (entity, view) in views {
        with_current_view(world, entity, |world| f(world, entity, view));
    }
}
