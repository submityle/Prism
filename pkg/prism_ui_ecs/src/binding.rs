//! Field-level projection binding between a single ECS component field and a
//! reactive [`Signal`].
//!
//! A [`FieldBinding`] projects *one field* of a component `C` to and from a
//! `Signal<T>`. Reading uses the ECS change-detection tick as the transport: a
//! [`FieldBinding::pull`] only re-reads the component when it actually changed
//! since the binding last observed it. Writing uses an equality guard: a
//! [`FieldBinding::push`] only mutates the component when the projected field
//! differs from the signal's value, so it never sets a spurious change tick.

use bevy_ecs::change_detection::Tick;
use bevy_ecs::component::Mutable;
use bevy_ecs::prelude::{Component, DetectChanges, Entity, World};
use prism_ui_reactive::Signal;

/// A projection binding tying one field of component `C` to a `Signal<T>`.
///
/// The `reader` extracts the field value from the component; the optional
/// `writer` applies a signal value back onto the component. `last_tick` records
/// the component change tick observed by the most recent successful
/// [`pull`](FieldBinding::pull), so later pulls can skip unchanged components.
pub struct FieldBinding<C: Component, T: Clone + PartialEq + 'static> {
    signal: Signal<T>,
    reader: Box<dyn Fn(&C) -> T>,
    writer: Option<Box<dyn Fn(&mut C, &T)>>,
    last_tick: Option<Tick>,
}

impl<C: Component, T: Clone + PartialEq + 'static> FieldBinding<C, T> {
    /// Creates a one-way binding that only ever reads from the component into
    /// the signal.
    pub fn read_only(signal: Signal<T>, reader: impl Fn(&C) -> T + 'static) -> Self {
        Self {
            signal,
            reader: Box::new(reader),
            writer: None,
            last_tick: None,
        }
    }

    /// Creates a two-way binding that can both read from and write back to the
    /// component.
    pub fn read_write(
        signal: Signal<T>,
        reader: impl Fn(&C) -> T + 'static,
        writer: impl Fn(&mut C, &T) + 'static,
    ) -> Self {
        Self {
            signal,
            reader: Box::new(reader),
            writer: Some(Box::new(writer)),
            last_tick: None,
        }
    }

    /// The signal this binding drives.
    pub fn signal(&self) -> &Signal<T> {
        &self.signal
    }

    /// Whether this binding can write back to the component.
    pub fn is_two_way(&self) -> bool {
        self.writer.is_some()
    }

    /// The component change tick observed by the last successful pull, if any.
    pub fn last_tick(&self) -> Option<Tick> {
        self.last_tick
    }

    /// Reads the component field into the signal when the component changed
    /// since the last pull.
    ///
    /// Returns `true` only when the signal's value actually changed. Returns
    /// `false` when the entity or component is missing, when the component has
    /// not changed since the last observed tick, or when the projected value is
    /// already equal to the signal's current value.
    pub fn pull(&mut self, world: &World, entity: Entity) -> bool {
        let Ok(entity_ref) = world.get_entity(entity) else {
            return false;
        };
        let Some(component) = entity_ref.get_ref::<C>() else {
            return false;
        };

        let should_read = match self.last_tick {
            None => true,
            Some(tick) => component.is_changed_after(tick),
        };
        if !should_read {
            return false;
        }

        let value = (self.reader)(&component);
        self.last_tick = Some(component.last_changed());
        self.signal.set_if_changed(value)
    }

    /// Writes the signal's value back onto the component field when they differ.
    ///
    /// Returns `true` only when the component was actually mutated. Returns
    /// `false` for read-only bindings, when the entity or component is missing,
    /// or when the projected field already equals the signal's value (the
    /// equality guard that prevents feedback oscillation).
    pub fn push(&self, world: &mut World, entity: Entity) -> bool
    where
        C: Component<Mutability = Mutable>,
    {
        let Some(writer) = self.writer.as_ref() else {
            return false;
        };
        let value = self.signal.get_untracked();
        let Some(mut component) = world.get_mut::<C>(entity) else {
            return false;
        };

        // Read-only comparison first: `Mut` only sets a change tick on
        // `deref_mut`, so comparing through the shared `Deref` keeps the write
        // conditional and avoids spurious change detection.
        let current = (self.reader)(&component);
        if current == value {
            return false;
        }
        writer(&mut component, &value);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::DetectChangesMut;
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Counter {
        value: i32,
        label: u32,
    }

    fn spawn(world: &mut World, value: i32, label: u32) -> Entity {
        world.spawn(Counter { value, label }).id()
    }

    #[test]
    fn read_only_pull_propagates_initial_value() {
        let mut world = World::new();
        let entity = spawn(&mut world, 7, 0);
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let mut binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);

        assert!(binding.pull(&world, entity));
        assert_eq!(signal.get_untracked(), 7);
        assert!(binding.last_tick().is_some());
    }

    #[test]
    fn pull_is_skipped_when_component_unchanged() {
        let mut world = World::new();
        let entity = spawn(&mut world, 3, 0);
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let mut binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);

        assert!(binding.pull(&world, entity));
        // No component change: the tick guard short-circuits the second pull.
        assert!(!binding.pull(&world, entity));
        assert_eq!(signal.get_untracked(), 3);
    }

    #[test]
    fn pull_detects_subsequent_change() {
        let mut world = World::new();
        let entity = spawn(&mut world, 1, 0);
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let mut binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);

        assert!(binding.pull(&world, entity));
        // Advance the world change tick to simulate a new frame, then mutate.
        world.increment_change_tick();
        world.get_mut::<Counter>(entity).unwrap().value = 42;
        assert!(binding.pull(&world, entity));
        assert_eq!(signal.get_untracked(), 42);
    }

    #[test]
    fn pull_returns_false_when_value_equal_after_change() {
        let mut world = World::new();
        let entity = spawn(&mut world, 5, 0);
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let mut binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);
        assert!(binding.pull(&world, entity));

        // Advance to a new frame and mark the component changed, but keep the
        // projected field identical: the tick guard lets the read happen, yet
        // `set_if_changed` reports no change because the value is equal.
        world.increment_change_tick();
        world.get_mut::<Counter>(entity).unwrap().set_changed();
        assert!(!binding.pull(&world, entity));
        assert_eq!(signal.get_untracked(), 5);
    }

    #[test]
    fn read_only_push_is_noop() {
        let mut world = World::new();
        let entity = spawn(&mut world, 9, 0);
        let rt = Runtime::new();
        let signal = rt.signal(100i32);
        let binding = FieldBinding::<Counter, i32>::read_only(signal, |c| c.value);

        assert!(!binding.push(&mut world, entity));
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 9);
    }

    #[test]
    fn read_write_push_writes_when_differing() {
        let mut world = World::new();
        let entity = spawn(&mut world, 0, 0);
        let rt = Runtime::new();
        let signal = rt.signal(55i32);
        let binding =
            FieldBinding::<Counter, i32>::read_write(signal, |c| c.value, |c, v| c.value = *v);

        assert!(binding.push(&mut world, entity));
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 55);
    }

    #[test]
    fn push_equality_guard_prevents_write() {
        let mut world = World::new();
        let entity = spawn(&mut world, 55, 0);
        let rt = Runtime::new();
        let signal = rt.signal(55i32);
        let binding =
            FieldBinding::<Counter, i32>::read_write(signal, |c| c.value, |c, v| c.value = *v);

        // Signal already equals the field; the guard suppresses the write.
        assert!(!binding.push(&mut world, entity));
    }

    #[test]
    fn missing_entity_is_safe() {
        let mut world = World::new();
        let entity = spawn(&mut world, 1, 0);
        world.despawn(entity);
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let mut binding =
            FieldBinding::<Counter, i32>::read_write(signal, |c| c.value, |c, v| c.value = *v);

        assert!(!binding.pull(&world, entity));
        assert!(!binding.push(&mut world, entity));
    }

    #[test]
    fn independent_fields_project_separately() {
        let mut world = World::new();
        let entity = spawn(&mut world, 3, 9);
        let rt = Runtime::new();
        let value_sig = rt.signal(0i32);
        let label_sig = rt.signal(0u32);
        let mut value_binding =
            FieldBinding::<Counter, i32>::read_only(value_sig.clone(), |c| c.value);
        let mut label_binding =
            FieldBinding::<Counter, u32>::read_only(label_sig.clone(), |c| c.label);

        assert!(value_binding.pull(&world, entity));
        assert!(label_binding.pull(&world, entity));
        assert_eq!(value_sig.get_untracked(), 3);
        assert_eq!(label_sig.get_untracked(), 9);
    }
}
