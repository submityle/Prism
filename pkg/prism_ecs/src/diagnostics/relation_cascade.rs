//! Relation cleanup-policy and cascade blast-radius diagnostics
//! (design §23.2 / §16.6).
//!
//! [`relation_graph`](super::relation_graph) already summarises the *shape* of
//! the relation system — per-kind flags and edge fan-out — and
//! [`relation_cycles`](super::relation_cycles) /
//! [`relation_topology`](super::relation_topology) describe cycles and chain
//! depth. None of them answer the operational §23.2 question an editor or a
//! streaming layer actually asks before destroying an entity: *if this entity
//! is despawned now, how far does the destruction propagate?*
//!
//! This module joins the per-kind [`CleanupPolicy`] metadata (design §23.2)
//! with the pure, side-effect-free cascade planner
//! ([`Relations::plan_cascade`](crate::relation::Relations::plan_cascade)) to
//! produce, for every entity that participates in a relation edge, a
//! *blast-radius* record: how many holders are recursively deleted, how many
//! edges are unlinked, and how many edges trip a [`CleanupPolicy::Panic`]
//! guard. It is the diagnostic lens over the cascade-delete mechanism the
//! kernel uses for hierarchy teardown (`ChildOf` 父死子亡) and for cleaning up
//! cross-cell relations when a World Partition cell unloads (design §13.1).
//!
//! # Scope
//! Cascade planning walks the non-fragmenting bypass
//! [`RelationIndex`](crate::relation::RelationIndex) (design §23.3), so blast
//! radius is measured over the explicit edges stored there. A relation kind
//! registered with no edges still contributes to the policy census; an
//! unregistered kind that nonetheless holds edges is planned under the default
//! [`CleanupPolicy::Remove`] rule, mirroring
//! [`Relations::plan_cascade`](crate::relation::Relations::plan_cascade).
//! Capture is read-only and never mutates simulation state.

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::relation::CleanupPolicy;
use crate::world::World;

/// Cleanup-policy record for one registered relation kind (design §23.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RelationPolicyEntry {
    /// The relation marker component identifying this kind.
    pub relation: ComponentId,
    /// Whether the relation fragments the archetype graph (design §11).
    pub fragmenting: bool,
    /// Whether the relation is transitive (design §11).
    pub transitive: bool,
    /// Whether the relation is exclusive — at most one target per source
    /// (design §23.3).
    pub exclusive: bool,
    /// Policy when the relation *kind* is removed from a holder (design §23.2).
    pub on_delete: CleanupPolicy,
    /// Policy when the *target* a holder points at is destroyed — the policy
    /// that drives cascade-delete planning (design §23.2).
    pub on_delete_target: CleanupPolicy,
}

impl RelationPolicyEntry {
    /// Whether either cleanup slot is a [`CleanupPolicy::Panic`] guard, i.e.
    /// this kind treats a dangling deletion as a bug to be caught.
    #[inline]
    pub fn has_panic_guard(&self) -> bool {
        matches!(self.on_delete, CleanupPolicy::Panic)
            || matches!(self.on_delete_target, CleanupPolicy::Panic)
    }

    /// Whether destroying a target of this kind cascades into deleting the
    /// holder ([`CleanupPolicy::Delete`] on the target slot).
    #[inline]
    pub fn cascades_on_target_delete(&self) -> bool {
        matches!(self.on_delete_target, CleanupPolicy::Delete)
    }
}

/// Blast-radius record for one entity that participates in a relation edge:
/// the outcome of despawning it, as planned by
/// [`Relations::plan_cascade`](crate::relation::Relations::plan_cascade).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CascadeBlastEntry {
    /// The entity whose despawn was planned (the cascade root).
    pub root: Entity,
    /// Holder entities recursively deleted because they point at a deleted
    /// target under [`CleanupPolicy::Delete`]. Does *not* include `root`.
    pub cascade_deletions: usize,
    /// Edges unlinked from the index while their holders survive (the
    /// [`CleanupPolicy::Remove`] outcome plus the outgoing edges of every
    /// deleted entity).
    pub edge_removals: usize,
    /// Edges that trip a [`CleanupPolicy::Panic`] guard during the walk.
    pub policy_panics: usize,
}

impl CascadeBlastEntry {
    /// Total entities destroyed by this despawn: the root itself plus every
    /// cascaded holder.
    #[inline]
    pub fn total_destroyed(&self) -> usize {
        1 + self.cascade_deletions
    }

    /// Whether despawning this entity propagates to at least one other entity.
    #[inline]
    pub fn cascades(&self) -> bool {
        self.cascade_deletions > 0
    }
}

/// Read-only cleanup-policy and cascade blast-radius census over the relation
/// system (design §23.2 / §16.6).
#[derive(Clone, Debug, Default)]
pub struct RelationCascadeReport {
    /// Per-kind policy records, sorted ascending by [`ComponentId`].
    policies: Vec<RelationPolicyEntry>,
    /// Per-participant blast-radius records, sorted ascending by
    /// [`Entity::to_bits`].
    blasts: Vec<CascadeBlastEntry>,
}

impl RelationCascadeReport {
    /// Capture a cascade census from `world`. Read-only; never mutates
    /// simulation state.
    ///
    /// The policy list is built from every registered relation kind. The blast
    /// list is built by collecting the distinct entities that appear as a
    /// source or target of any edge in the bypass index and planning a cascade
    /// despawn from each. Both lists are sorted deterministically so the report
    /// is stable regardless of hash-map iteration order (design §14).
    pub fn from_world(world: &World) -> Self {
        let relations = world.relations();

        let mut policies: Vec<RelationPolicyEntry> = relations
            .iter_kinds()
            .map(|(relation, kind)| RelationPolicyEntry {
                relation,
                fragmenting: kind.fragmenting,
                transitive: kind.transitive,
                exclusive: kind.exclusive,
                on_delete: kind.on_delete,
                on_delete_target: kind.on_delete_target,
            })
            .collect();
        policies.sort_unstable_by_key(|entry| entry.relation.index());

        // Collect the distinct entities touched by any edge (either endpoint),
        // deduplicated via a sort over the stable bit layout.
        let index = relations.index();
        let mut participants: Vec<Entity> = Vec::new();
        for (_relation, source, target) in index.iter_edges() {
            participants.push(source);
            participants.push(target);
        }
        participants.sort_unstable_by_key(|entity| entity.to_bits());
        participants.dedup();

        let blasts: Vec<CascadeBlastEntry> = participants
            .into_iter()
            .map(|root| {
                let plan = relations.plan_cascade(root);
                CascadeBlastEntry {
                    root,
                    cascade_deletions: plan.deletions.len(),
                    edge_removals: plan.removals.len(),
                    policy_panics: plan.panics.len(),
                }
            })
            .collect();

        Self { policies, blasts }
    }

    /// The per-kind policy records, sorted ascending by [`ComponentId`].
    #[inline]
    pub fn policies(&self) -> &[RelationPolicyEntry] {
        &self.policies
    }

    /// The per-participant blast-radius records, sorted ascending by
    /// [`Entity::to_bits`].
    #[inline]
    pub fn blasts(&self) -> &[CascadeBlastEntry] {
        &self.blasts
    }

    /// Whether the report holds neither policies nor blast records.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty() && self.blasts.is_empty()
    }

    /// Number of registered relation kinds.
    #[inline]
    pub fn kind_count(&self) -> usize {
        self.policies.len()
    }

    /// Number of distinct entities participating in at least one edge.
    #[inline]
    pub fn participant_count(&self) -> usize {
        self.blasts.len()
    }

    /// Number of relation kinds that fragment the archetype graph.
    #[inline]
    pub fn fragmenting_kind_count(&self) -> usize {
        self.policies.iter().filter(|p| p.fragmenting).count()
    }

    /// Number of exclusive relation kinds.
    #[inline]
    pub fn exclusive_kind_count(&self) -> usize {
        self.policies.iter().filter(|p| p.exclusive).count()
    }

    /// Number of transitive relation kinds.
    #[inline]
    pub fn transitive_kind_count(&self) -> usize {
        self.policies.iter().filter(|p| p.transitive).count()
    }

    /// Number of kinds that cascade-delete the holder when a target dies
    /// ([`CleanupPolicy::Delete`] on the target slot).
    #[inline]
    pub fn cascade_delete_kind_count(&self) -> usize {
        self.policies
            .iter()
            .filter(|p| p.cascades_on_target_delete())
            .count()
    }

    /// Number of kinds with a [`CleanupPolicy::Panic`] guard on either slot.
    #[inline]
    pub fn panic_policy_kind_count(&self) -> usize {
        self.policies.iter().filter(|p| p.has_panic_guard()).count()
    }

    /// Number of participants whose despawn cascades into at least one other
    /// entity.
    #[inline]
    pub fn participants_with_cascade(&self) -> usize {
        self.blasts.iter().filter(|b| b.cascades()).count()
    }

    /// The entity with the largest cascade blast radius (most cascaded
    /// deletions). Ties resolve to the lowest [`Entity::to_bits`], so the
    /// result is deterministic. Returns `None` when no entity participates.
    pub fn worst_blast(&self) -> Option<Entity> {
        let mut worst: Option<&CascadeBlastEntry> = None;
        for entry in &self.blasts {
            match worst {
                // `blasts` is sorted ascending by `to_bits`, so a strict `>`
                // keeps the first (lowest-bits) entry on ties.
                Some(current) if entry.cascade_deletions <= current.cascade_deletions => {}
                _ => worst = Some(entry),
            }
        }
        worst.map(|entry| entry.root)
    }

    /// Look up the blast record for a specific entity via binary search over
    /// the [`Entity::to_bits`]-sorted list.
    pub fn blast(&self, entity: Entity) -> Option<&CascadeBlastEntry> {
        let key = entity.to_bits();
        self.blasts
            .binary_search_by_key(&key, |entry| entry.root.to_bits())
            .ok()
            .map(|idx| &self.blasts[idx])
    }

    /// Look up the policy record for a specific relation kind via binary search
    /// over the [`ComponentId`]-sorted list.
    pub fn policy(&self, relation: ComponentId) -> Option<&RelationPolicyEntry> {
        let key = relation.index();
        self.policies
            .binary_search_by_key(&key, |entry| entry.relation.index())
            .ok()
            .map(|idx| &self.policies[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::world::World;

    /// `ChildOf` — exclusive hierarchy relation; deleting a parent deletes its
    /// children (cascade on target delete).
    #[derive(PartialEq)]
    struct ChildOf;
    impl Component for ChildOf {}

    /// `LocatedIn` — transitive containment; deleting the container only
    /// unlinks the holder (Remove policy).
    #[derive(PartialEq)]
    struct LocatedIn;
    impl Component for LocatedIn {}

    /// `GuardedBy` — a relation whose dangling deletion is treated as a bug.
    #[derive(PartialEq)]
    struct GuardedBy;
    impl Component for GuardedBy {}

    fn child_of_id(world: &World) -> ComponentId {
        world.components().id_of::<ChildOf>().unwrap()
    }

    #[test]
    fn empty_world_has_no_policies_or_blasts() {
        let world = World::new();
        let report = RelationCascadeReport::from_world(&world);
        assert!(report.is_empty());
        assert_eq!(report.kind_count(), 0);
        assert_eq!(report.participant_count(), 0);
        assert_eq!(report.worst_blast(), None);
    }

    #[test]
    fn registered_kind_with_no_edges_still_appears_in_policies() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.kind_count(), 1);
        assert_eq!(report.participant_count(), 0);

        let id = child_of_id(&world);
        let policy = report.policy(id).unwrap();
        assert!(policy.exclusive);
        assert!(policy.cascades_on_target_delete());
        assert!(!policy.has_panic_guard());
        assert_eq!(report.cascade_delete_kind_count(), 1);
        assert_eq!(report.exclusive_kind_count(), 1);
    }

    #[test]
    fn delete_policy_cascades_from_parent_to_children() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        let parent = world.spawn(());
        let child_a = world.spawn(());
        let child_b = world.spawn(());
        world.add_relation::<ChildOf>(child_a, parent);
        world.add_relation::<ChildOf>(child_b, parent);

        let report = RelationCascadeReport::from_world(&world);
        // parent + 2 children participate.
        assert_eq!(report.participant_count(), 3);

        // Despawning the parent cascades into both children.
        let blast = report.blast(parent).unwrap();
        assert_eq!(blast.cascade_deletions, 2);
        assert_eq!(blast.total_destroyed(), 3);
        assert!(blast.cascades());

        // The parent is the worst blast; a leaf child cascades to nothing.
        assert_eq!(report.worst_blast(), Some(parent));
        let child_blast = report.blast(child_a).unwrap();
        assert_eq!(child_blast.cascade_deletions, 0);
        assert_eq!(report.participants_with_cascade(), 1);
    }

    #[test]
    fn remove_policy_does_not_cascade() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(
            RelationKind::new()
                .with_transitive(true)
                .with_on_delete_target(CleanupPolicy::Remove),
        );
        let room = world.spawn(());
        let occupant = world.spawn(());
        world.add_relation::<LocatedIn>(occupant, room);

        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.transitive_kind_count(), 1);
        assert_eq!(report.cascade_delete_kind_count(), 0);
        // Deleting the room removes the edge but deletes no holder.
        let blast = report.blast(room).unwrap();
        assert_eq!(blast.cascade_deletions, 0);
        assert_eq!(blast.edge_removals, 1);
        assert_eq!(report.participants_with_cascade(), 0);
    }

    #[test]
    fn panic_policy_is_counted_and_recorded() {
        let mut world = World::new();
        world.register_relation::<GuardedBy>(
            RelationKind::new().with_on_delete_target(CleanupPolicy::Panic),
        );
        let guard = world.spawn(());
        let holder = world.spawn(());
        world.add_relation::<GuardedBy>(holder, guard);

        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.panic_policy_kind_count(), 1);
        // Deleting the guarded target trips the panic guard for the incoming
        // edge from `holder`.
        let blast = report.blast(guard).unwrap();
        assert_eq!(blast.policy_panics, 1);
        assert_eq!(blast.cascade_deletions, 0);
    }

    #[test]
    fn chained_delete_cascades_transitively() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        // c --ChildOf--> b --ChildOf--> a: deleting `a` must reach both.
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<ChildOf>(b, a);
        world.add_relation::<ChildOf>(c, b);

        let report = RelationCascadeReport::from_world(&world);
        let blast = report.blast(a).unwrap();
        assert_eq!(blast.cascade_deletions, 2);
        assert_eq!(blast.total_destroyed(), 3);
        assert_eq!(report.worst_blast(), Some(a));
    }

    #[test]
    fn flag_counts_tally_across_multiple_kinds() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_fragmenting(true)
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        world.register_relation::<GuardedBy>(
            RelationKind::new().with_on_delete(CleanupPolicy::Panic),
        );

        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.kind_count(), 3);
        assert_eq!(report.fragmenting_kind_count(), 1);
        assert_eq!(report.exclusive_kind_count(), 1);
        assert_eq!(report.transitive_kind_count(), 1);
        assert_eq!(report.cascade_delete_kind_count(), 1);
        assert_eq!(report.panic_policy_kind_count(), 1);
        // Policies are sorted ascending by component id.
        let ids: Vec<u32> = report.policies().iter().map(|p| p.relation.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn worst_blast_breaks_ties_on_lowest_bits() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new().with_on_delete_target(CleanupPolicy::Delete),
        );
        // Two independent parents each with exactly one child => tie at
        // cascade_deletions == 1. The lowest-bits root must win.
        let p1 = world.spawn(());
        let c1 = world.spawn(());
        let p2 = world.spawn(());
        let c2 = world.spawn(());
        world.add_relation::<ChildOf>(c1, p1);
        world.add_relation::<ChildOf>(c2, p2);

        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.participants_with_cascade(), 2);
        let winner = report.worst_blast().unwrap();
        let lowest = if p1.to_bits() <= p2.to_bits() { p1 } else { p2 };
        assert_eq!(winner, lowest);
    }

    #[test]
    fn participants_are_deduplicated_and_sorted() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new().with_on_delete_target(CleanupPolicy::Delete),
        );
        let parent = world.spawn(());
        let child_a = world.spawn(());
        let child_b = world.spawn(());
        // `parent` is the target of two edges but must appear once.
        world.add_relation::<ChildOf>(child_a, parent);
        world.add_relation::<ChildOf>(child_b, parent);

        let report = RelationCascadeReport::from_world(&world);
        assert_eq!(report.participant_count(), 3);
        let bits: Vec<u64> = report.blasts().iter().map(|b| b.root.to_bits()).collect();
        let mut sorted = bits.clone();
        sorted.sort_unstable();
        assert_eq!(bits, sorted);
    }
}
