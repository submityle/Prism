//! [`World`] integration for the flecs-style relation system (design §11).
//!
//! The relation *data structures* (the registry, the bidirectional index,
//! cascade planning) live in [`crate::relation`]; this module adds the
//! type-driven [`World`] API that routes marker [`Component`] types through
//! that registry. A relation kind `R` is just a zero-sized (or small) component
//! type whose [`ComponentId`] keys both the kind metadata and the edge index,
//! so relations never need a parallel type registry.
//!
//! All edges stored here are **non-fragmenting** (design §23.3): they live in
//! the bypass index rather than in the archetype graph, which keeps
//! high-cardinality relations (equipment, targeting, ownership) cheap to add
//! and remove without archetype churn.

use alloc::vec::Vec;

use super::World;
use crate::component::Component;
use crate::entity::Entity;
use crate::relation::{RelationKind, RelationTarget, Relations};

impl World {
    /// Shared access to the relation registry (design §11).
    #[inline]
    pub fn relations(&self) -> &Relations {
        &self.relations
    }

    /// Mutable access to the relation registry.
    ///
    /// Prefer the typed helpers ([`World::add_relation`] and friends) for
    /// everyday use; this escape hatch is for bulk or dynamic edits.
    #[inline]
    pub fn relations_mut(&mut self) -> &mut Relations {
        &mut self.relations
    }

    /// Register the relation-kind metadata for marker component `R` (design
    /// §11, §23.2/§23.3), registering `R` as a component if needed. Returns the
    /// relation's [`ComponentId`](crate::component::ComponentId).
    ///
    /// Re-registering overwrites the previous [`RelationKind`]. A relation that
    /// is never registered behaves as the default kind (non-exclusive,
    /// non-transitive, [`CleanupPolicy::Remove`](crate::relation::CleanupPolicy)).
    pub fn register_relation<R: Component>(&mut self, kind: RelationKind) {
        let id = self.components.register::<R>();
        self.relations.register(id, kind);
    }

    /// Add the edge `source --R--> target` (design §11).
    ///
    /// `R` is registered as a component on demand. If `R` was registered as an
    /// [`exclusive`](RelationKind::exclusive) relation, any previous target of
    /// `source` is evicted and returned.
    pub fn add_relation<R: Component>(&mut self, source: Entity, target: Entity) -> Option<Entity> {
        let id = self.components.register::<R>();
        self.relations.add(id, source, target)
    }

    /// Remove the edge `source --R--> target`. Returns whether it existed.
    ///
    /// A relation type that was never registered holds no edges, so this
    /// returns `false` without allocating.
    pub fn remove_relation<R: Component>(&mut self, source: Entity, target: Entity) -> bool {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.remove(id, source, target),
            None => false,
        }
    }

    /// The targets `source` points at under relation `R`, in insertion order.
    ///
    /// Returns an empty slice if `R` is unregistered or `source` holds no such
    /// edges.
    pub fn relation_targets<R: Component>(&self, source: Entity) -> &[Entity] {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.index().targets(id, source),
            None => &[],
        }
    }

    /// The sources that point at `target` under relation `R`, in insertion
    /// order. Powers reverse / "who references me" queries without scanning.
    pub fn relation_sources<R: Component>(&self, target: Entity) -> &[Entity] {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.index().sources(id, target),
            None => &[],
        }
    }

    /// Whether the edge `source --R--> target` currently exists.
    pub fn has_relation<R: Component>(&self, source: Entity, target: Entity) -> bool {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.index().targets(id, source).contains(&target),
            None => false,
        }
    }

    /// Query the pair `(R, target)` (design §11): pass
    /// [`RelationTarget::Wildcard`] for `(R, *)` to enumerate every
    /// `(source, target)` edge of kind `R`, or
    /// [`RelationTarget::Entity`] for a concrete target.
    ///
    /// Returns `(source, target)` tuples. Unregistered relations yield an empty
    /// vector.
    pub fn query_pair<R: Component>(&self, target: RelationTarget) -> Vec<(Entity, Entity)> {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.index().query_pair(id, target),
            None => Vec::new(),
        }
    }

    /// The transitive closure of `source` under relation `R` (design §11): all
    /// entities reachable by following `R` edges, excluding `source` itself.
    /// The walk is cycle-safe.
    ///
    /// This computes the closure over the index regardless of whether `R` was
    /// registered as [`transitive`](RelationKind::transitive); the kind flag is
    /// advisory metadata for query planning, not a gate on this helper.
    pub fn query_transitive<R: Component>(&self, source: Entity) -> Vec<Entity> {
        match self.components.id_of::<R>() {
            Some(id) => self.relations.index().transitive_targets(id, source),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::component::Component;
    use crate::relation::{CleanupPolicy, RelationKind, RelationTarget};
    use crate::world::World;

    /// `ChildOf` — exclusive hierarchy relation whose target deletion cascades.
    struct ChildOf;
    impl Component for ChildOf {}

    /// `EquippedBy` — non-exclusive; equipment survives if the wielder dies.
    struct EquippedBy;
    impl Component for EquippedBy {}

    /// `LocatedIn` — transitive containment relation.
    struct LocatedIn;
    impl Component for LocatedIn {}

    #[test]
    fn add_and_query_targets() {
        let mut world = World::new();
        let parent = world.spawn(());
        let child = world.spawn(());
        assert!(world.add_relation::<ChildOf>(child, parent).is_none());
        assert_eq!(world.relation_targets::<ChildOf>(child), &[parent]);
        assert_eq!(world.relation_sources::<ChildOf>(parent), &[child]);
        assert!(world.has_relation::<ChildOf>(child, parent));
    }

    #[test]
    fn unregistered_relation_is_empty() {
        let world = World::new();
        let a = crate::entity::Entity::from_bits(((1u64) << 32) | 1).unwrap();
        assert!(world.relation_targets::<ChildOf>(a).is_empty());
        assert!(world.query_pair::<ChildOf>(RelationTarget::Wildcard).is_empty());
        assert!(world.query_transitive::<ChildOf>(a).is_empty());
    }

    #[test]
    fn exclusive_relation_evicts_previous_target() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        let child = world.spawn(());
        let p1 = world.spawn(());
        let p2 = world.spawn(());
        assert!(world.add_relation::<ChildOf>(child, p1).is_none());
        assert_eq!(world.add_relation::<ChildOf>(child, p2), Some(p1));
        assert_eq!(world.relation_targets::<ChildOf>(child), &[p2]);
    }

    #[test]
    fn remove_relation_returns_existence() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<EquippedBy>(a, b);
        assert!(world.remove_relation::<EquippedBy>(a, b));
        assert!(!world.remove_relation::<EquippedBy>(a, b));
        assert!(world.relation_targets::<EquippedBy>(a).is_empty());
    }

    #[test]
    fn query_pair_wildcard_and_concrete() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let hub = world.spawn(());
        world.add_relation::<EquippedBy>(a, hub);
        world.add_relation::<EquippedBy>(b, hub);
        let mut all = world.query_pair::<EquippedBy>(RelationTarget::Wildcard);
        all.sort();
        let mut expected = alloc::vec![(a, hub), (b, hub)];
        expected.sort();
        assert_eq!(all, expected);
        let concrete = world.query_pair::<EquippedBy>(RelationTarget::Entity(hub));
        assert_eq!(concrete.len(), 2);
    }

    #[test]
    fn transitive_closure_follows_chain() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<LocatedIn>(a, b);
        world.add_relation::<LocatedIn>(b, c);
        let mut closure = world.query_transitive::<LocatedIn>(a);
        closure.sort();
        let mut expected = alloc::vec![b, c];
        expected.sort();
        assert_eq!(closure, expected);
    }

    #[test]
    fn despawn_removes_entitys_edges_from_index() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<EquippedBy>(a, b);
        world.add_relation::<EquippedBy>(b, a);
        world.despawn(a);
        // `a` must leave no dangling entries in either direction.
        assert!(world.relation_targets::<EquippedBy>(a).is_empty());
        assert!(world.relation_sources::<EquippedBy>(a).is_empty());
        assert!(world.relation_targets::<EquippedBy>(b).is_empty());
    }

    #[test]
    fn despawn_cascades_delete_policy_to_holders() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        let parent = world.spawn(());
        let child = world.spawn(());
        let grandchild = world.spawn(());
        world.add_relation::<ChildOf>(child, parent);
        world.add_relation::<ChildOf>(grandchild, child);

        world.despawn(parent);

        // Deleting the parent recursively deletes child and grandchild.
        assert!(!world.contains(child));
        assert!(!world.contains(grandchild));
        assert!(!world.contains(parent));
    }

    #[test]
    fn despawn_remove_policy_keeps_holder_alive() {
        let mut world = World::new();
        // Default policy is `Remove`: the sword survives, only the edge drops.
        let hero = world.spawn(());
        let sword = world.spawn(());
        world.add_relation::<EquippedBy>(sword, hero);
        world.despawn(hero);
        assert!(world.contains(sword));
        assert!(world.relation_targets::<EquippedBy>(sword).is_empty());
    }
}
