//! The single-threaded, insertion-ordered [`CommandQueue`] and its ergonomic
//! [`Commands`] / [`EntityCommands`] builders (design §9, M0 layer).

use alloc::vec::Vec;

use super::Command;
use crate::bundle::Bundle;
use crate::entity::{Entities, Entity};
use crate::world::World;

/// An ordered buffer of deferred structural changes.
///
/// Record commands through the [`Commands`] handle returned by
/// [`CommandQueue::commands`], then apply them all with [`CommandQueue::apply`].
#[derive(Default)]
pub struct CommandQueue {
    commands: Vec<Command>,
    reserved: Vec<Entity>,
}

impl CommandQueue {
    /// Create an empty queue.
    #[inline]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            reserved: Vec::new(),
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
        super::coalescing::replay(world, self.commands.drain(..));
        self.reserved.clear();
    }

    /// Discard unapplied commands and release this queue's unplaced spawns.
    ///
    /// Already applied changes remain. Reservations owned by other queues are
    /// preserved, even when this flush materialises their allocator slots.
    /// Use the same world that supplied the queue's entity handles.
    pub fn discard(&mut self, world: &mut World) {
        world.flush_reserved();
        for entity in self.reserved.drain(..) {
            if world
                .entities()
                .location(entity)
                .is_some_and(|location| location.is_empty())
            {
                world.despawn(entity);
            }
        }
        self.commands.clear();
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
    /// Queue arbitrary world code. This is a replay barrier: every preceding
    /// structural batch is committed before `command` observes the world.
    pub fn queue(&mut self, command: impl FnOnce(&mut World) + Send + Sync + 'static) {
        self.queue.commands.push(Command::opaque(command));
    }

    /// Reserve a fresh entity and queue a spawn of `bundle` onto it.
    ///
    /// The returned [`Entity`] is valid immediately (for later commands and for
    /// the caller to store), but its components do not exist until the queue is
    /// [applied](CommandQueue::apply).
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.reserve_entity();
        self.queue.reserved.push(entity);
        self.queue
            .commands
            .push(Command::opaque(move |world: &mut World| {
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

impl<'a, 'w, 's> EntityCommands<'a, 'w, 's> {
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
            .push(Command::insert(entity, bundle));
        self
    }

    /// Queue removing the components named by bundle type `B` from the entity
    /// (see [`World::remove`]).
    pub fn remove<B: Bundle>(&mut self) -> &mut Self {
        let entity = self.entity;
        self.commands
            .queue
            .commands
            .push(Command::remove::<B>(entity));
        self
    }

    /// Queue despawning the entity (see [`World::despawn`]).
    pub fn despawn(&mut self) {
        let entity = self.entity;
        self.commands
            .queue
            .commands
            .push(Command::opaque(move |world: &mut World| {
                world.despawn(entity);
            }));
    }

    /// Queue attaching an existing `child` to this entity with
    /// [`ChildOf`](crate::hierarchy::ChildOf) (design §23.6).
    ///
    /// `ChildOf` is exclusive, so a child that already had a parent is moved
    /// rather than double-parented (design §23.3).
    pub fn add_child(&mut self, child: Entity) -> &mut Self {
        let parent = self.entity;
        self.commands
            .queue
            .commands
            .push(Command::opaque(move |world: &mut World| {
                world.register_child_of();
                world.add_relation::<crate::hierarchy::ChildOf>(child, parent);
            }));
        self
    }

    /// Queue spawning this entity's children through a nested builder (design
    /// §23.6 作用域命令).
    ///
    /// Everything `f` builds is spawned *and* attached to this entity with
    /// [`ChildOf`](crate::hierarchy::ChildOf) inside one deferred scope. The
    /// queue replays a child's spawn before its relation add, and replays the
    /// whole scope in the order the builder was written, so applying the queue
    /// yields exactly the tree the code describes.
    ///
    /// ```ignore
    /// commands.entity(root).with_children(|children| {
    ///     children.spawn(Torso);
    ///     children.spawn(Head);
    /// });
    /// ```
    pub fn with_children(&mut self, f: impl FnOnce(&mut ChildBuilder<'_, 'w, 's>)) -> &mut Self {
        let parent = self.entity;
        let mut builder = ChildBuilder {
            parent,
            commands: self.commands,
        };
        f(&mut builder);
        self
    }
}

/// The scoped builder handed to
/// [`EntityCommands::with_children`] (design §23.6 作用域命令).
///
/// Every entity it creates is already tied to the scope's parent, so callers
/// cannot forget the relation and cannot attach a child to the wrong parent.
/// The returned [`Entity`] handles are valid immediately for further commands
/// and for storing, exactly like [`Commands::spawn`].
pub struct ChildBuilder<'a, 'w, 's> {
    parent: Entity,
    commands: &'a mut Commands<'w, 's>,
}

impl<'w, 's> ChildBuilder<'_, 'w, 's> {
    /// The parent every entity in this scope is attached to.
    #[inline]
    pub fn parent(&self) -> Entity {
        self.parent
    }

    /// Reserve a child entity, queue spawning `bundle` on it, and queue
    /// attaching it to this scope's parent.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        // `Commands::spawn` queues the spawn; the relation add is queued after
        // it, so apply order (spawn, then link) is guaranteed by construction.
        let child = self.commands.spawn(bundle);
        let parent = self.parent;
        self.commands
            .queue
            .commands
            .push(Command::opaque(move |world: &mut World| {
                world.register_child_of();
                world.add_relation::<crate::hierarchy::ChildOf>(child, parent);
            }));
        child
    }

    /// Reserve an empty child and queue attaching it to this scope's parent.
    #[inline]
    pub fn spawn_empty(&mut self) -> Entity {
        self.spawn(())
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
