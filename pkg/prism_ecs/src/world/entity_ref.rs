//! [`EntityRef`]: a lightweight, borrowed handle to a single live entity.
//!
//! `EntityRef` is the `bevy_ecs`-style ergonomic entry point for reading one
//! entity's components without threading the [`Entity`] id through every call.
//! It is produced by [`World::get_entity`](crate::world::World::get_entity) and
//! borrows the [`World`] immutably, so any number of handles can coexist with
//! other shared reads.
//!
//! The handle is deliberately read-only: it delegates to the world's own
//! [`get`](crate::world::World::get) / [`get_ref`](crate::world::World::get_ref)
//! accessors (including change detection), and holds no state of its own beyond
//! the borrow and the entity id.

use crate::change::Ref;
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// A borrowed, read-only view of one live entity within a [`World`].
///
/// Obtained from [`World::get_entity`](crate::world::World::get_entity). The
/// borrow keeps the world immutably locked, so an `EntityRef` cannot outlive
/// the structural change that would invalidate it.
#[derive(Clone, Copy)]
pub struct EntityRef<'w> {
    world: &'w World,
    entity: Entity,
}

impl<'w> EntityRef<'w> {
    /// Wrap a known-live entity. Crate-internal: callers must have already
    /// confirmed the entity is alive (as [`World::get_entity`] does).
    #[inline]
    pub(crate) fn new(world: &'w World, entity: Entity) -> Self {
        Self { world, entity }
    }

    /// The entity this handle refers to.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Borrow component `T` of this entity, or `None` if it lacks `T`.
    #[inline]
    pub fn get<T: Component>(&self) -> Option<&'w T> {
        self.world.get::<T>(self.entity)
    }

    /// Change-detecting shared borrow of component `T`, or `None` if this
    /// entity lacks `T`. See [`World::get_ref`](crate::world::World::get_ref).
    #[inline]
    pub fn get_ref<T: Component>(&self) -> Option<Ref<'w, T>> {
        self.world.get_ref::<T>(self.entity)
    }

    /// Whether this entity currently has a component of type `T`.
    #[inline]
    pub fn contains<T: Component>(&self) -> bool {
        self.world.get::<T>(self.entity).is_some()
    }
}
