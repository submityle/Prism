//! Prefabs and `IsA` inheritance (design §16.3).
//!
//! A **prefab** is nothing more than an ordinary entity used as a *template*:
//! it holds the default components an instance should start from. An
//! **instance** is spawned with a built-in [`IsA`] relation pointing at its
//! prefab, and it *inherits* the prefab's components by resolution rather than
//! by copying — reading a component on the instance falls through the `IsA`
//! chain to the first ancestor that provides it. Writing a component on the
//! instance (`insert`) **overrides** the inherited value for that instance
//! only.
//!
//! This is the flecs-v4 `IsA` model (design §2, §16.3): resolve-through
//! inheritance with no clone/v-table. It composes with the relation system
//! (§11): `IsA` is a transitive relation, so prefab-of-prefab chains resolve
//! through the transitive closure, nearest ancestor first, and the walk is
//! cycle-safe. Scene serialization of prefabs + per-field overrides is handled
//! by `prism_reflect` (design §16.3, §24.3); this module provides the kernel
//! inheritance primitives those layers build on.
//!
//! # Example
//!
//! ```ignore
//! let goblin = world.spawn((Health(30), Speed(5)));          // prefab template
//! let scout  = world.spawn_instance_of(goblin);              // instance
//! world.insert(scout, Speed(9));                             // per-field override
//! assert_eq!(world.get_inherited::<Health>(scout).unwrap().0, 30); // inherited
//! assert_eq!(world.get_inherited::<Speed>(scout).unwrap().0, 9);   // overridden
//! ```

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::relation::RelationKind;
use crate::world::World;

/// Built-in prefab-inheritance relation (design §16.3).
///
/// `instance --IsA--> prefab` declares that `instance` inherits `prefab`'s
/// components. It is registered as a **transitive** relation on first use, so
/// `a IsA b` and `b IsA c` imply `a` inherits from both `b` and `c` (nearest
/// first). This is a zero-sized marker [`Component`] used purely as a relation
/// kind, exactly like the user-defined relations in [`crate::relation`].
pub struct IsA;

impl Component for IsA {}

impl World {
    /// Ensure [`IsA`] is registered as a transitive relation kind.
    ///
    /// Registration of the *kind* is advisory metadata (the transitive closure
    /// helpers work regardless), but recording it keeps query planning and
    /// cleanup semantics correct. Registering only when absent avoids clobbering
    /// a kind the caller configured deliberately.
    fn ensure_isa_registered(&mut self) {
        if self.components().id_of::<IsA>().is_none() {
            self.register_relation::<IsA>(RelationKind::new().with_transitive(true));
        }
    }

    /// Spawn a fresh entity that inherits from `prefab` via an [`IsA`] edge
    /// (design §16.3).
    ///
    /// The new instance starts with **no components of its own**; every
    /// component read resolves through the prefab chain until overridden with
    /// [`insert`](World::insert). `prefab` can itself be an instance of another
    /// prefab, forming an inheritance chain.
    pub fn spawn_instance_of(&mut self, prefab: Entity) -> Entity {
        self.ensure_isa_registered();
        let instance = self.spawn(());
        self.add_relation::<IsA>(instance, prefab);
        instance
    }

    /// Alias for [`spawn_instance_of`](World::spawn_instance_of) matching the
    /// flecs vocabulary.
    #[inline]
    pub fn instantiate(&mut self, prefab: Entity) -> Entity {
        self.spawn_instance_of(prefab)
    }

    /// Make `instance` inherit from `prefab` by adding an [`IsA`] edge, without
    /// spawning a new entity. Useful to turn an existing entity into an
    /// instance, or to add a second inheritance source.
    #[inline]
    pub fn add_prefab(&mut self, instance: Entity, prefab: Entity) {
        self.ensure_isa_registered();
        self.add_relation::<IsA>(instance, prefab);
    }

    /// Resolve component `T` for `entity` through the [`IsA`] chain
    /// (design §16.3).
    ///
    /// Returns the entity's **own** `T` if present (an override), otherwise the
    /// `T` of the nearest ancestor that provides one, following the transitive
    /// `IsA` closure (nearest prefab first). Returns `None` if neither the
    /// entity nor any ancestor has `T`. The walk is cycle-safe.
    pub fn get_inherited<T: Component>(&self, entity: Entity) -> Option<&T> {
        if let Some(own) = self.get::<T>(entity) {
            return Some(own);
        }
        // `query_transitive` returns an owned, cycle-safe, nearest-first chain,
        // so there is no borrow conflict with the immutable `get` below.
        for ancestor in self.query_transitive::<IsA>(entity) {
            if let Some(found) = self.get::<T>(ancestor) {
                return Some(found);
            }
        }
        None
    }

    /// Whether `entity` has `T` either directly or by [`IsA`] inheritance.
    #[inline]
    pub fn has_inherited<T: Component>(&self, entity: Entity) -> bool {
        if self.has::<T>(entity) {
            return true;
        }
        self.query_transitive::<IsA>(entity)
            .into_iter()
            .any(|ancestor| self.has::<T>(ancestor))
    }

    /// The entity that actually provides `entity`'s inherited `T`: `entity`
    /// itself if it overrides `T`, otherwise the nearest `IsA` ancestor that
    /// supplies it, or `None` if nobody does.
    pub fn inherited_source<T: Component>(&self, entity: Entity) -> Option<Entity> {
        if self.has::<T>(entity) {
            return Some(entity);
        }
        self.query_transitive::<IsA>(entity)
            .into_iter()
            .find(|&ancestor| self.has::<T>(ancestor))
    }

    /// The direct prefab(s) `entity` is an instance of (its immediate [`IsA`]
    /// targets, in insertion order). Empty if `entity` is not an instance.
    #[inline]
    pub fn prefabs_of(&self, entity: Entity) -> &[Entity] {
        self.relation_targets::<IsA>(entity)
    }

    /// Whether `instance` inherits from `prefab`, directly or transitively.
    pub fn is_instance_of(&self, instance: Entity, prefab: Entity) -> bool {
        self.has_relation::<IsA>(instance, prefab)
            || self.query_transitive::<IsA>(instance).contains(&prefab)
    }

    /// The full `IsA` ancestry of `entity` (its transitive prefab chain,
    /// nearest first, excluding `entity`). Cycle-safe.
    #[inline]
    pub fn prefab_chain(&self, entity: Entity) -> Vec<Entity> {
        self.query_transitive::<IsA>(entity)
    }
}

#[cfg(test)]
mod tests {
    use crate::component::Component;
    use crate::world::World;

    /// Test component: a hit-point value.
    struct Health(u32);
    impl Component for Health {}

    /// Test component: a movement speed value.
    struct Speed(u32);
    impl Component for Speed {}

    /// Test component present only on a deep ancestor.
    struct Armor(u32);
    impl Component for Armor {}

    #[test]
    fn instance_inherits_prefab_components() {
        let mut world = World::new();
        let prefab = world.spawn((Health(100), Speed(5)));
        let instance = world.spawn_instance_of(prefab);

        // The instance owns nothing yet...
        assert!(!world.has::<Health>(instance));
        assert!(world.get::<Health>(instance).is_none());
        // ...but resolves both components through IsA.
        assert_eq!(world.get_inherited::<Health>(instance).unwrap().0, 100);
        assert_eq!(world.get_inherited::<Speed>(instance).unwrap().0, 5);
        assert!(world.has_inherited::<Health>(instance));
        assert_eq!(world.inherited_source::<Health>(instance), Some(prefab));
        assert!(world.is_instance_of(instance, prefab));
    }

    #[test]
    fn override_shadows_inherited_value() {
        let mut world = World::new();
        let prefab = world.spawn((Health(100), Speed(5)));
        let instance = world.spawn_instance_of(prefab);

        world.insert(instance, Speed(9));

        // Overridden on the instance, untouched on the prefab.
        assert_eq!(world.get_inherited::<Speed>(instance).unwrap().0, 9);
        assert_eq!(world.inherited_source::<Speed>(instance), Some(instance));
        assert_eq!(world.get::<Speed>(prefab).unwrap().0, 5);
        // Non-overridden component still inherits.
        assert_eq!(world.get_inherited::<Health>(instance).unwrap().0, 100);
        assert_eq!(world.inherited_source::<Health>(instance), Some(prefab));
    }

    #[test]
    fn prefab_of_prefab_resolves_nearest_first() {
        let mut world = World::new();
        // base → mid → leaf, with an override at the middle layer.
        let base = world.spawn((Health(100), Speed(5), Armor(3)));
        let mid = world.spawn_instance_of(base);
        world.insert(mid, Health(80)); // mid overrides Health
        let leaf = world.spawn_instance_of(mid);

        // Nearest ancestor (mid) wins for the overridden component.
        assert_eq!(world.get_inherited::<Health>(leaf).unwrap().0, 80);
        assert_eq!(world.inherited_source::<Health>(leaf), Some(mid));
        // Deeper ancestor (base) supplies the rest.
        assert_eq!(world.get_inherited::<Speed>(leaf).unwrap().0, 5);
        assert_eq!(world.get_inherited::<Armor>(leaf).unwrap().0, 3);
        assert_eq!(world.inherited_source::<Armor>(leaf), Some(base));

        assert!(world.is_instance_of(leaf, base));
        assert_eq!(world.prefab_chain(leaf), alloc::vec![mid, base]);
    }

    #[test]
    fn missing_component_resolves_to_none() {
        let mut world = World::new();
        let prefab = world.spawn(Health(100));
        let instance = world.spawn_instance_of(prefab);
        assert!(world.get_inherited::<Armor>(instance).is_none());
        assert!(!world.has_inherited::<Armor>(instance));
        assert_eq!(world.inherited_source::<Armor>(instance), None);
    }

    #[test]
    fn cyclic_isa_chain_is_safe() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(Armor(7));
        // Deliberately cyclic: a IsA b and b IsA a.
        world.add_prefab(a, b);
        world.add_prefab(b, a);
        // Must terminate and still find the component on the reachable ancestor.
        assert_eq!(world.get_inherited::<Armor>(a).unwrap().0, 7);
        // And return None for a component nobody in the cycle has.
        assert!(world.get_inherited::<Health>(a).is_none());
    }

    #[test]
    fn non_instance_entity_has_empty_prefab_set() {
        let mut world = World::new();
        let lone = world.spawn(Health(1));
        assert!(world.prefabs_of(lone).is_empty());
        assert_eq!(world.get_inherited::<Health>(lone).unwrap().0, 1);
    }
}
