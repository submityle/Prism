//! [`World`] integration for the flecs-style relation system (design §11).
//!
//! The relation *data structures* (the registry, the bidirectional index,
//! cascade planning) live in [`crate::relation`]; this module adds the
//! type-driven [`World`] API that routes marker [`Component`] types through
//! that registry. A relation kind `R` is just a zero-sized (or small) component
//! type whose [`ComponentId`] keys both the kind metadata and the edge index,
//! so relations never need a parallel type registry.
//!
//! Every edge is indexed for reverse lookup and cascading. Fragmenting pairs
//! additionally participate in archetype identity; non-fragmenting edges do
//! not relocate component data (design §11 / §23.3).

use alloc::vec::Vec;

use super::World;
use crate::archetype::RelationPair;
use crate::component::{Component, ComponentId};
use crate::entity::Entity;
use crate::relation::{Relation, RelationKind, RelationTarget, Relations};

/// A staged bulk edit: forgetting the guard leaves the live world unchanged.
/// Publishing relocates bytes only; no component hooks or destructors run,
/// including when the guard is dropped during unwinding.
#[must_use = "relation edits publish when the guard is dropped"]
struct RelationsMut<'w> {
    world: &'w mut World,
    staged: Relations,
}

impl core::ops::Deref for RelationsMut<'_> {
    type Target = Relations;

    fn deref(&self) -> &Relations {
        &self.staged
    }
}

impl core::ops::DerefMut for RelationsMut<'_> {
    fn deref_mut(&mut self) -> &mut Relations {
        &mut self.staged
    }
}

impl Drop for RelationsMut<'_> {
    fn drop(&mut self) {
        let mut staged = core::mem::take(&mut self.staged);
        // Raw edits must not smuggle dead/reserved endpoints into a live
        // world's index. Removing invalid edges is deterministic and invokes
        // no cleanup policies (there is no live target to delete).
        let invalid: Vec<_> = staged
            .index()
            .iter_edges()
            .filter(|&(_, s, t)| {
                !self.world.relation_endpoint_is_live(s) || !self.world.relation_endpoint_is_live(t)
            })
            .collect();
        for (r, s, t) in invalid {
            staged.remove(r, s, t);
        }
        let exclusive: Vec<_> = staged
            .iter_kinds()
            .filter_map(|(id, kind)| kind.exclusive.then_some(id))
            .collect();
        for id in exclusive {
            staged.normalize_exclusive(id);
        }
        // Include old holders as well, so deleting all edges or turning
        // fragmenting off merges entities back into their base archetypes.
        let mut sources: Vec<_> = self
            .world
            .relations
            .index()
            .iter_edges()
            .chain(staged.index().iter_edges())
            .map(|(_, s, _)| s)
            .collect();
        sources.sort_unstable();
        sources.dedup();
        for source in sources {
            self.world
                .move_relation_pairs(source, &staged.fragmenting_pairs(source));
        }
        self.world.relations = staged;
    }
}

impl World {
    /// Shared access to the relation registry (design §11).
    #[inline]
    pub fn relations(&self) -> &Relations {
        &self.relations
    }

    /// Stage bulk or dynamic edits to the relation registry.
    ///
    /// Prefer the typed helpers ([`World::add_relation`] and friends) for
    /// everyday use. Edits publish when the returned guard is dropped, also
    /// during unwinding; forgetting it discards the edits. Publication repairs
    /// fragmenting archetype identities and exclusivity (last target wins),
    /// and discards edges whose endpoints are dead or not yet placed.
    #[inline]
    pub fn relations_mut(&mut self) -> impl core::ops::DerefMut<Target = Relations> + '_ {
        let staged = self.relations.clone();
        RelationsMut {
            world: self,
            staged,
        }
    }

    /// Register the relation-kind metadata for marker component `R` (design
    /// §11, §23.2/§23.3), registering `R` as a component if needed.
    ///
    /// Re-registering overwrites the previous [`RelationKind`]. A relation that
    /// is never registered behaves as the default kind (non-exclusive,
    /// non-transitive, [`CleanupPolicy::Remove`](crate::relation::CleanupPolicy)).
    pub fn register_relation<R: Component>(&mut self, kind: RelationKind) {
        let id = self.components.register::<R>();
        if self.relations.kind(id) == Some(&kind) {
            return;
        }
        self.relations_mut().register(id, kind);
    }

    /// Register a statically-typed [`Relation`] kind using the metadata pinned
    /// by its [`Relation::KIND`] associated constant (design §11, §23.2/§23.3).
    ///
    /// This is the type-driven companion to [`World::register_relation`]: where
    /// the latter takes an explicit [`RelationKind`], this reads the kind from
    /// the type itself, so a `#[derive(Relation)]` type registers with a single
    /// turbofish and no restated metadata. Re-registering overwrites the
    /// previous metadata, matching [`World::register_relation`].
    pub fn register_relation_type<R: Relation>(&mut self) {
        self.register_relation::<R>(R::KIND);
    }

    /// Add the edge `source --R--> target` (design §11).
    ///
    /// `R` is registered as a component on demand. If `R` was registered as an
    /// [`exclusive`](RelationKind::exclusive) relation, any previous target of
    /// `source` is evicted and returned.
    ///
    /// # Panics
    /// Both endpoints must be live and placed in this world. Reserved but
    /// unmaterialized handles and stale generations are rejected before edits.
    pub fn add_relation<R: Component>(&mut self, source: Entity, target: Entity) -> Option<Entity> {
        assert!(
            self.relation_endpoint_is_live(source),
            "relation source must be live and placed"
        );
        assert!(
            self.relation_endpoint_is_live(target),
            "relation target must be live and placed"
        );
        let id = self.components.register::<R>();
        self.add_relation_by_id_unchecked(id, source, target)
    }

    /// Insert a relation edge without validating endpoint liveness.
    ///
    /// This is intentionally crate-private and is reserved for the prefab
    /// low-level escape hatch, which must be able to preserve malformed
    /// ancestry for diagnostics. Ordinary relation callers must use
    /// [`World::add_relation`], whose live-endpoint checks protect the relation
    /// registry invariant.
    pub(crate) fn add_relation_by_id_unchecked(
        &mut self,
        id: ComponentId,
        source: Entity,
        target: Entity,
    ) -> Option<Entity> {
        if self.relations.kind(id).is_some_and(|kind| kind.fragmenting) {
            let mut pairs = self.relations.fragmenting_pairs(source);
            if self.relations.is_exclusive(id) {
                pairs.retain(|&(r, _)| r != id);
            }
            pairs.push((id, target));
            self.move_relation_pairs(source, &pairs);
        }
        self.relations.add(id, source, target)
    }

    /// Remove the edge `source --R--> target`. Returns whether it existed.
    ///
    /// A relation type that was never registered holds no edges, so this
    /// returns `false` without allocating.
    pub fn remove_relation<R: Component>(&mut self, source: Entity, target: Entity) -> bool {
        match self.components.id_of::<R>() {
            Some(id) => self.remove_relation_by_id(id, source, target),
            None => false,
        }
    }

    fn relation_endpoint_is_live(&self, entity: Entity) -> bool {
        self.entities
            .location(entity)
            .is_some_and(|loc| !loc.is_empty())
    }

    /// Shared edge removal used by typed APIs and cascade cleanup. Removing
    /// a pair is not removal of its marker component and runs no value hooks.
    pub(crate) fn remove_relation_by_id(
        &mut self,
        relation: ComponentId,
        source: Entity,
        target: Entity,
    ) -> bool {
        if !self
            .relations
            .index()
            .targets(relation, source)
            .contains(&target)
        {
            return false;
        }
        if self
            .relations
            .kind(relation)
            .is_some_and(|kind| kind.fragmenting)
        {
            let mut pairs = self.relations.fragmenting_pairs(source);
            pairs.retain(|&pair| pair != (relation, target));
            self.move_relation_pairs(source, &pairs);
        }
        self.relations.remove(relation, source, target)
    }

    /// Relocate all table columns without cloning, dropping, changing ticks,
    /// touching sparse order or changing shared-value reference counts.
    fn move_relation_pairs(&mut self, entity: Entity, pairs: &[RelationPair]) {
        let Some(loc) = self.entities.location(entity).filter(|loc| !loc.is_empty()) else {
            return;
        };
        let source = self
            .archetypes
            .get(loc.archetype_id)
            .expect("placed entity");
        let current = source.components().clone();
        let bindings = source.shared_bindings().to_vec();
        let dst_id =
            self.archetypes
                .get_or_create_with_pairs(&current, &bindings, pairs, &self.components);
        if dst_id == loc.archetype_id {
            return;
        }
        let dst = self
            .archetypes
            .get_mut(dst_id)
            .expect("destination archetype");
        // Complete allocation before transferring ownership. The following
        // move/forget sequence has no user callbacks or fallible allocation.
        dst.reserve(1);
        let dst_row = dst.table_mut().allocate(entity);
        let (src, dst) = self.archetypes.get_pair_mut(loc.archetype_id, dst_id);
        // SAFETY: identical component sets and ABI, reserved empty destination
        // row, initialized source. finish_move forgets every transferred value
        // and patches both entity locations before any user code can run.
        unsafe {
            dst.table_mut()
                .move_shared_columns_from(src.table_mut(), loc.row as usize);
        }
        self.finish_move(
            entity,
            loc,
            loc.archetype_id,
            current.ids(),
            dst_id,
            dst_row,
        );
    }

    /// Rebuild derived pair identities after snapshot entity materialization.
    /// The restored relation registry is authoritative; this operation is
    /// idempotent and preserves all component values and exact ticks.
    pub(crate) fn rebuild_fragmenting_archetypes(&mut self) {
        let entities: Vec<_> = self
            .archetypes
            .iter()
            .flat_map(|arch| arch.table().entities().iter().copied())
            .collect();
        for entity in entities {
            self.move_relation_pairs(entity, &self.relations.fragmenting_pairs(entity));
        }
    }

    /// Apply the preflighted cascade before component teardown. Re-plan after
    /// deleting holders: their lifecycle hooks may attach fresh edges to this
    /// still-live target. No dangling pair may survive the eventual free.
    pub(crate) fn cleanup_relations_for_despawn(&mut self, entity: Entity) {
        while !self.relations.index().is_empty() {
            let plan = self.relations.plan_cascade(entity);
            assert!(
                plan.panics.is_empty(),
                "despawn of a relation target violated a `CleanupPolicy::Panic` policy",
            );
            if plan.removals.is_empty() {
                break;
            }
            for edge in plan.removals {
                self.remove_relation_by_id(edge.relation, edge.source, edge.target);
            }
            for holder in plan.deletions {
                self.despawn(holder);
            }
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
#[path = "fragmenting_tests.rs"]
mod fragmenting_tests;

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
        assert!(world
            .query_pair::<ChildOf>(RelationTarget::Wildcard)
            .is_empty());
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
