//! A registry of synchronisation bindings plus whole-frame sync passes.
//!
//! [`EcsBridge`] owns a collection of type-erased [`SyncBinding`]s and drives
//! them together: [`EcsBridge::pull_all`] refreshes every signal from its
//! component, and [`EcsBridge::push_all`] writes every two-way signal back.
//!
//! # Feedback-loop protection
//!
//! Both directions are guarded by equality checks, so an `ECS -> signal -> ECS`
//! round trip settles instead of oscillating:
//!
//! * **Pull** uses [`Signal::set_if_changed`](prism_ui_reactive::Signal::set_if_changed):
//!   a component value already matching the signal produces no signal update.
//! * **Push** compares the projected field to the signal value *before*
//!   mutating, so a component already equal to the signal is left untouched and
//!   its change tick is not advanced.
//!
//! Consequently, after a `push_all` the next `pull_all` sees no net change (the
//! value just written equals the signal), and after a `pull_all` the next
//! `push_all` finds the component already equal to the signal.

use bevy_ecs::component::Mutable;
use bevy_ecs::prelude::{Component, Entity, World};
use prism_ui_reactive::Signal;

use crate::binding::FieldBinding;
use crate::entity_binding::{EntityBinding, SyncBinding};

/// A registry of entity/field bindings with frame-level sync passes.
#[derive(Default)]
pub struct EcsBridge {
    bindings: Vec<Box<dyn SyncBinding>>,
}

impl EcsBridge {
    /// Creates an empty bridge.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Registers a one-way binding from `entity`'s component field into
    /// `signal`.
    pub fn bind<C, T>(
        &mut self,
        entity: Entity,
        signal: Signal<T>,
        reader: impl Fn(&C) -> T + 'static,
    ) where
        C: Component<Mutability = Mutable>,
        T: Clone + PartialEq + 'static,
    {
        let binding = FieldBinding::read_only(signal, reader);
        self.bindings
            .push(Box::new(EntityBinding::new(entity, binding)));
    }

    /// Registers a two-way binding between `entity`'s component field and
    /// `signal`.
    pub fn bind_two_way<C, T>(
        &mut self,
        entity: Entity,
        signal: Signal<T>,
        reader: impl Fn(&C) -> T + 'static,
        writer: impl Fn(&mut C, &T) + 'static,
    ) where
        C: Component<Mutability = Mutable>,
        T: Clone + PartialEq + 'static,
    {
        let binding = FieldBinding::read_write(signal, reader, writer);
        self.bindings
            .push(Box::new(EntityBinding::new(entity, binding)));
    }

    /// Number of registered bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether no bindings are registered.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Pulls every binding, returning how many signals changed.
    pub fn pull_all(&mut self, world: &World) -> usize {
        let mut changed = 0;
        for binding in &mut self.bindings {
            if binding.pull(world) {
                changed += 1;
            }
        }
        changed
    }

    /// Pushes every binding, returning how many components were mutated.
    pub fn push_all(&self, world: &mut World) -> usize {
        let mut written = 0;
        for binding in &self.bindings {
            if binding.push(world) {
                written += 1;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Position {
        x: i32,
        y: i32,
    }

    #[test]
    fn new_bridge_is_empty() {
        let bridge = EcsBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.len(), 0);
    }

    #[test]
    fn bind_registers_binding() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1, y: 2 }).id();
        let rt = Runtime::new();
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, rt.signal(0), |p| p.x);
        assert_eq!(bridge.len(), 1);
        assert!(!bridge.is_empty());
    }

    #[test]
    fn pull_all_counts_changed_bindings() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 10, y: 20 }).id();
        let rt = Runtime::new();
        let x = rt.signal(0i32);
        let y = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, x.clone(), |p| p.x);
        bridge.bind::<Position, i32>(entity, y.clone(), |p| p.y);

        assert_eq!(bridge.pull_all(&world), 2);
        assert_eq!(x.get_untracked(), 10);
        assert_eq!(y.get_untracked(), 20);
        // Nothing changed on the component, so a second pull reports zero.
        assert_eq!(bridge.pull_all(&world), 0);
    }

    #[test]
    fn push_all_counts_written_bindings() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 0, y: 0 }).id();
        let rt = Runtime::new();
        let x = rt.signal(3i32);
        let y = rt.signal(4i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Position, i32>(entity, x, |p| p.x, |p, v| p.x = *v);
        bridge.bind_two_way::<Position, i32>(entity, y, |p| p.y, |p, v| p.y = *v);

        assert_eq!(bridge.push_all(&mut world), 2);
        let pos = world.get::<Position>(entity).unwrap();
        assert_eq!((pos.x, pos.y), (3, 4));
        // Values already match now, so a second push writes nothing.
        assert_eq!(bridge.push_all(&mut world), 0);
    }

    #[test]
    fn read_only_bindings_do_not_push() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 1, y: 2 }).id();
        let rt = Runtime::new();
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(entity, rt.signal(99), |p| p.x);

        assert_eq!(bridge.push_all(&mut world), 0);
        assert_eq!(world.get::<Position>(entity).unwrap().x, 1);
    }

    #[test]
    fn multiple_entities_sync_independently() {
        let mut world = World::new();
        let a = world.spawn(Position { x: 1, y: 0 }).id();
        let b = world.spawn(Position { x: 2, y: 0 }).id();
        let rt = Runtime::new();
        let sa = rt.signal(0i32);
        let sb = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind::<Position, i32>(a, sa.clone(), |p| p.x);
        bridge.bind::<Position, i32>(b, sb.clone(), |p| p.x);

        assert_eq!(bridge.pull_all(&world), 2);
        assert_eq!(sa.get_untracked(), 1);
        assert_eq!(sb.get_untracked(), 2);
    }

    #[test]
    fn two_way_round_trip_does_not_oscillate() {
        let mut world = World::new();
        let entity = world.spawn(Position { x: 0, y: 0 }).id();
        let rt = Runtime::new();
        let x = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Position, i32>(entity, x.clone(), |p| p.x, |p, v| p.x = *v);

        // ECS -> signal.
        world.get_mut::<Position>(entity).unwrap().x = 5;
        assert_eq!(bridge.pull_all(&world), 1);
        assert_eq!(x.get_untracked(), 5);
        // signal already equals component: push is a no-op, no oscillation.
        assert_eq!(bridge.push_all(&mut world), 0);

        // signal -> ECS.
        x.set(9);
        assert_eq!(bridge.push_all(&mut world), 1);
        assert_eq!(world.get::<Position>(entity).unwrap().x, 9);
        // component now equals signal: pull reports no net change.
        assert_eq!(bridge.pull_all(&world), 0);
        assert_eq!(x.get_untracked(), 9);
    }
}
