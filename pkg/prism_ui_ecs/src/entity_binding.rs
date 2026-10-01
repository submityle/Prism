//! Attaching a [`FieldBinding`] to a concrete [`Entity`] and erasing its types.
//!
//! [`FieldBinding`] is generic over the component and field types, which makes
//! it awkward to store heterogeneous bindings together. [`EntityBinding`] pairs
//! a binding with the entity it targets and implements the object-safe
//! [`SyncBinding`] trait, so a registry can hold `Box<dyn SyncBinding>` values
//! of differing component/field types in one collection.

use bevy_ecs::component::Mutable;
use bevy_ecs::prelude::{Component, Entity, World};

use crate::binding::FieldBinding;

/// A type-erased, entity-attached synchronisation binding.
///
/// Both directions forward to a concrete [`FieldBinding`] against a fixed
/// entity. See [`EntityBinding`] for the single implementor.
pub trait SyncBinding {
    /// Reads the bound component field into its signal, returning whether the
    /// signal changed. See [`FieldBinding::pull`].
    fn pull(&mut self, world: &World) -> bool;

    /// Writes the signal value back onto the bound component field, returning
    /// whether the component was mutated. See [`FieldBinding::push`].
    fn push(&self, world: &mut World) -> bool;
}

/// A [`FieldBinding`] bound to a specific [`Entity`].
pub struct EntityBinding<C: Component, T: Clone + PartialEq + 'static> {
    entity: Entity,
    inner: FieldBinding<C, T>,
}

impl<C: Component, T: Clone + PartialEq + 'static> EntityBinding<C, T> {
    /// Attaches `inner` to `entity`.
    pub fn new(entity: Entity, inner: FieldBinding<C, T>) -> Self {
        Self { entity, inner }
    }

    /// The entity this binding targets.
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// The wrapped field binding.
    pub fn binding(&self) -> &FieldBinding<C, T> {
        &self.inner
    }
}

impl<C, T> SyncBinding for EntityBinding<C, T>
where
    C: Component<Mutability = Mutable>,
    T: Clone + PartialEq + 'static,
{
    fn pull(&mut self, world: &World) -> bool {
        self.inner.pull(world, self.entity)
    }

    fn push(&self, world: &mut World) -> bool {
        self.inner.push(world, self.entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Counter {
        value: i32,
    }

    #[test]
    fn forwards_pull_to_inner_binding() {
        let mut world = World::new();
        let entity = world.spawn(Counter { value: 11 }).id();
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);
        let mut entity_binding = EntityBinding::new(entity, binding);

        assert_eq!(entity_binding.entity(), entity);
        assert!(entity_binding.pull(&world));
        assert_eq!(signal.get_untracked(), 11);
    }

    #[test]
    fn forwards_push_to_inner_binding() {
        let mut world = World::new();
        let entity = world.spawn(Counter { value: 0 }).id();
        let rt = Runtime::new();
        let signal = rt.signal(77i32);
        let binding =
            FieldBinding::<Counter, i32>::read_write(signal, |c| c.value, |c, v| c.value = *v);
        let entity_binding = EntityBinding::new(entity, binding);

        assert!(entity_binding.push(&mut world));
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 77);
    }

    #[test]
    fn stored_as_trait_object() {
        let mut world = World::new();
        let entity = world.spawn(Counter { value: 5 }).id();
        let rt = Runtime::new();
        let signal = rt.signal(0i32);
        let binding = FieldBinding::<Counter, i32>::read_only(signal.clone(), |c| c.value);
        let mut erased: Box<dyn SyncBinding> = Box::new(EntityBinding::new(entity, binding));

        assert!(erased.pull(&world));
        assert_eq!(signal.get_untracked(), 5);
    }
}
