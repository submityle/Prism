//! Deferred structural changes: the [`CommandQueue`] and the ergonomic
//! [`Commands`] / [`EntityCommands`] builders layered on top of it (design §9).
//!
//! Immediate structural operations on a [`World`] (spawn / insert / remove /
//! despawn) require `&mut World`, which a running system does not hold while it
//! is iterating queries. [`Commands`] record those same operations as closures
//! in a [`CommandQueue`] and hand back [`Entity`] handles *immediately* (via the
//! lock-free [`Entities::reserve_entity`] reservation path), so a system can
//! queue spawns and edits without taking an exclusive world borrow. The queue
//! is drained and applied later at a synchronization point by
//! [`CommandQueue::apply`].
//!
//! This M0 queue is single-threaded and preserves insertion order. The parallel
//! per-thread buffers with deterministic merge-sort replay described in design
//! §9 are an M2 refinement that layers on top of this type without changing its
//! public surface.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::bundle::Bundle;
use crate::entity::{Entities, Entity};
use crate::world::World;

/// A boxed, type-erased structural mutation applied to a [`World`].
type Command = Box<dyn FnOnce(&mut World) + Send + Sync>;

/// An ordered buffer of deferred structural changes.
///
/// Record commands through the [`Commands`] handle returned by
/// [`CommandQueue::commands`], then apply them all with [`CommandQueue::apply`].
#[derive(Default)]
pub struct CommandQueue {
    commands: Vec<Command>,
}

impl CommandQueue {
    /// Create an empty queue.
    #[inline]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Number of queued-but-unapplied commands.
    #[inline]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the queue currently holds no commands.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Borrow a [`Commands`] builder that records into this queue, reserving new
    /// entity handles from `entities`.
    ///
    /// `entities` must be the allocator of the same [`World`] the queue is later
    /// [applied](CommandQueue::apply) to, so reserved handles resolve correctly.
    #[inline]
    pub fn commands<'w, 's>(&'s mut self, entities: &'w Entities) -> Commands<'w, 's> {
        Commands {
            queue: self,
            entities,
        }
    }

    /// Apply every queued command to `world` in recorded order, emptying the
    /// queue.
    ///
    /// Reserved entity handles are materialised first (via
    /// [`World::flush_reserved`]) so deferred spawns land on live, unplaced
    /// slots.
    pub fn apply(&mut self, world: &mut World) {
        world.flush_reserved();
        for command in self.commands.drain(..) {
            command(world);
        }
    }
}

/// An ergonomic builder that records deferred structural changes into a
/// [`CommandQueue`], handing back entity handles immediately.
///
/// Mirrors the shape of `bevy_ecs`'s `Commands` to keep migration a
/// change-the-import exercise (design §18).
pub struct Commands<'w, 's> {
    queue: &'s mut CommandQueue,
    entities: &'w Entities,
}

impl<'w, 's> Commands<'w, 's> {
    /// Reserve a fresh entity and queue a spawn of `bundle` onto it.
    ///
    /// The returned [`Entity`] is valid immediately (for later commands and for
    /// the caller to store), but its components do not exist until the queue is
    /// [applied](CommandQueue::apply).
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.reserve_entity();
        self.queue.commands.push(Box::new(move |world: &mut World| {
            world.spawn_at(entity, bundle);
        }));
        entity
    }

    /// Reserve a fresh entity with no components (an empty bundle spawn).
    #[inline]
    pub fn spawn_empty(&mut self) -> Entity {
        self.spawn(())
    }

    /// Borrow a per-entity command builder for queuing edits to `entity`.
    #[inline]
    pub fn entity(&mut self, entity: Entity) -> EntityCommands<'_, 'w, 's> {
        EntityCommands {
            entity,
            commands: self,
        }
    }
}

/// A per-entity command builder returned by [`Commands::entity`].
///
/// Chains deferred edits (`insert` / `remove` / `despawn`) targeting one
/// [`Entity`]; each call records a command and returns `&mut Self` for chaining.
pub struct EntityCommands<'a, 'w, 's> {
    entity: Entity,
    commands: &'a mut Commands<'w, 's>,
}

impl EntityCommands<'_, '_, '_> {
    /// The entity these commands target.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Queue inserting `bundle` onto the entity (overwriting existing
    /// components last-wins; see [`World::insert`]).
    pub fn insert<B: Bundle>(&mut self, bundle: B) -> &mut Self {
        let entity = self.entity;
        self.commands
            .queue
            .commands
            .push(Box::new(move |world: &mut World| {
                world.insert(entity, bundle);
            }));
        self
    }

    /// Queue removing the components named by bundle type `B` from the entity
    /// (see [`World::remove`]).
    pub fn remove<B: Bundle>(&mut self) -> &mut Self {
        let entity = self.entity;
        self.commands
            .queue
            .commands
            .push(Box::new(move |world: &mut World| {
                world.remove::<B>(entity);
            }));
        self
    }

    /// Queue despawning the entity (see [`World::despawn`]).
    pub fn despawn(&mut self) {
        let entity = self.entity;
        self.commands
            .queue
            .commands
            .push(Box::new(move |world: &mut World| {
                world.despawn(entity);
            }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Debug, PartialEq)]
    struct Position(i32, i32);
    impl Component for Position {}
    #[derive(Debug, PartialEq)]
    struct Velocity(i32, i32);
    impl Component for Velocity {}

    #[test]
    fn deferred_spawn_insert_remove_despawn() {
        let mut world = World::new();
        let mut queue = CommandQueue::new();

        let (a, b) = {
            let mut commands = queue.commands(world.entities());
            let a = commands.spawn((Position(1, 2), Velocity(3, 4)));
            let b = commands.spawn(Position(5, 6));
            // Entity ids are valid immediately, but not yet materialised.
            commands.entity(b).insert(Velocity(7, 8));
            (a, b)
        };

        // Nothing applied yet.
        assert_eq!(world.entity_count(), 0);
        assert!(!world.contains(a));

        queue.apply(&mut world);

        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.get::<Position>(a), Some(&Position(1, 2)));
        assert_eq!(world.get::<Velocity>(a), Some(&Velocity(3, 4)));
        assert_eq!(world.get::<Position>(b), Some(&Position(5, 6)));
        assert_eq!(world.get::<Velocity>(b), Some(&Velocity(7, 8)));
        assert!(queue.is_empty());

        // A second batch: remove a component and despawn an entity.
        {
            let mut commands = queue.commands(world.entities());
            commands.entity(a).remove::<Velocity>();
            commands.entity(b).despawn();
        }
        queue.apply(&mut world);

        assert_eq!(world.get::<Velocity>(a), None);
        assert_eq!(world.get::<Position>(a), Some(&Position(1, 2)));
        assert!(!world.contains(b));
        assert_eq!(world.entity_count(), 1);
    }
}
