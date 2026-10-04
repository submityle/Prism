//! Relation-graph diagnostics (design §16.6).
//!
//! Design §16.6 lists the kernel's read-only diagnostic surfaces as "原型/chunk
//! 占用、system 耗时、变更量、关系图谱" (archetype/chunk occupancy, system
//! timing, change volume, and the *relation graph*). The first three are
//! already covered by [`inspector`](super::inspector),
//! [`profiler`](super::profiler), and
//! [`change_volume`](super::change_volume)/[`step_inspector`](super::step_inspector);
//! this module supplies the last one.
//!
//! A [`RelationGraphReport`] is a point-in-time, read-only summary of the
//! flecs-style relation system (design §11): every registered relation *kind*
//! with its [`RelationKind`] flags, plus the live edge topology (edge count and
//! distinct source / target fan-out) drawn from the bypass
//! [`RelationIndex`](crate::relation::RelationIndex). It lets an editor
//! inspector or a devtools panel render the relation graph without reaching
//! into kernel internals.
//!
//! # Scope
//! Only the non-fragmenting bypass index carries explicit edges (design §23.3),
//! so the per-edge figures describe that index. A relation kind that is
//! registered but holds no edges still appears (with zero counts); a relation
//! that holds edges but was never registered appears with the default
//! [`RelationKind`] flags, mirroring the "unregistered behaves as default kind"
//! rule in [`Relations`](crate::relation::Relations). Capture is read-only and
//! never mutates simulation state.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::relation::{CleanupPolicy, RelationKind};
use crate::world::World;

/// Per-relation edge aggregation accumulated while walking the index.
#[derive(Default)]
struct EdgeAgg {
    edge_count: usize,
    sources: HashMap<Entity, ()>,
    targets: HashMap<Entity, ()>,
}

/// Read-only summary of a single relation kind (design §11 / §16.6).
///
/// Combines the registered [`RelationKind`] metadata with the live edge
/// topology for that relation. Flags fall back to the [`RelationKind`] default
/// when the relation holds edges but was never explicitly registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationKindReport {
    /// The marker [`ComponentId`] that names this relation kind.
    pub relation: ComponentId,
    /// Whether this relation kind was explicitly registered (vs. inferred from
    /// edges and reported with default flags).
    pub registered: bool,
    /// Directed edges of this relation currently held in the index.
    pub edge_count: usize,
    /// Distinct source entities that hold at least one edge of this relation.
    pub distinct_sources: usize,
    /// Distinct target entities pointed at by at least one edge of this
    /// relation.
    pub distinct_targets: usize,
    /// [`RelationKind::fragmenting`].
    pub fragmenting: bool,
    /// [`RelationKind::transitive`].
    pub transitive: bool,
    /// [`RelationKind::exclusive`].
    pub exclusive: bool,
    /// [`RelationKind::on_delete`].
    pub on_delete: CleanupPolicy,
    /// [`RelationKind::on_delete_target`].
    pub on_delete_target: CleanupPolicy,
}

impl RelationKindReport {
    /// Build a per-kind report from the (optional) registered metadata and the
    /// (optional) edge aggregation. At least one of the two is present for any
    /// relation that appears in a [`RelationGraphReport`].
    fn build(relation: ComponentId, kind: Option<&RelationKind>, agg: Option<&EdgeAgg>) -> Self {
        let meta = kind.copied().unwrap_or_default();
        let (edge_count, distinct_sources, distinct_targets) = match agg {
            Some(a) => (a.edge_count, a.sources.len(), a.targets.len()),
            None => (0, 0, 0),
        };
        Self {
            relation,
            registered: kind.is_some(),
            edge_count,
            distinct_sources,
            distinct_targets,
            fragmenting: meta.fragmenting,
            transitive: meta.transitive,
            exclusive: meta.exclusive,
            on_delete: meta.on_delete,
            on_delete_target: meta.on_delete_target,
        }
    }
}

/// Whole-world relation-graph summary (design §16.6).
///
/// Produced by [`RelationGraphReport::capture`]. [`per_kind`](Self::per_kind)
/// is sorted by [`ComponentId`] index so the report is deterministic regardless
/// of the index's internal hash order.
#[derive(Debug, Clone)]
pub struct RelationGraphReport {
    /// Number of explicitly registered relation kinds (design §11).
    pub kind_count: usize,
    /// Total directed edges across every relation in the bypass index.
    pub total_edges: usize,
    /// One entry per relation that is either registered or holds edges, sorted
    /// by relation [`ComponentId`] index.
    pub per_kind: Vec<RelationKindReport>,
}

impl RelationGraphReport {
    /// Capture the relation graph of `world` (design §16.6).
    ///
    /// Walks the registered relation kinds and the live edge index once each,
    /// aggregating per-relation edge counts and distinct source / target
    /// fan-out. Read-only: it borrows the world immutably and mutates nothing.
    pub fn capture(world: &World) -> Self {
        let relations = world.relations();

        // Aggregate the live edge topology by relation.
        let mut agg: HashMap<ComponentId, EdgeAgg> = HashMap::new();
        let mut total_edges = 0usize;
        for (relation, source, target) in relations.index().iter_edges() {
            total_edges += 1;
            let entry = agg.entry(relation).or_default();
            entry.edge_count += 1;
            entry.sources.insert(source, ());
            entry.targets.insert(target, ());
        }

        // Union of registered kinds and relations that carry edges. Registered
        // kinds are reported first (with their flags); relations that only
        // appear in the index are reported with default flags.
        let mut per_kind: Vec<RelationKindReport> = Vec::new();
        let mut seen: HashMap<ComponentId, ()> = HashMap::new();
        for (relation, kind) in relations.iter_kinds() {
            seen.insert(relation, ());
            per_kind.push(RelationKindReport::build(
                relation,
                Some(kind),
                agg.get(&relation),
            ));
        }
        for (&relation, a) in agg.iter() {
            if !seen.contains_key(&relation) {
                per_kind.push(RelationKindReport::build(relation, None, Some(a)));
            }
        }

        // Deterministic ordering for stable devtools rendering and tests.
        per_kind.sort_by_key(|r| r.relation.index());

        Self {
            kind_count: relations.kind_count(),
            total_edges,
            per_kind,
        }
    }

    /// Whether the graph has neither registered kinds nor edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.per_kind.is_empty()
    }

    /// Number of distinct relations reported (registered or edge-bearing).
    #[inline]
    pub fn relation_count(&self) -> usize {
        self.per_kind.len()
    }

    /// The per-kind report for `relation`, if present.
    pub fn kind(&self, relation: ComponentId) -> Option<&RelationKindReport> {
        self.per_kind.iter().find(|r| r.relation == relation)
    }

    /// The relation carrying the most edges, if any.
    ///
    /// Ties are broken by the lower [`ComponentId`] index (the first entry in
    /// the sorted [`per_kind`](Self::per_kind) list), so the result is
    /// deterministic.
    pub fn hottest_kind(&self) -> Option<&RelationKindReport> {
        // `per_kind` is sorted ascending by relation id, so walking it and
        // keeping the first strict maximum yields the lowest-id winner on ties.
        let mut best: Option<&RelationKindReport> = None;
        for report in &self.per_kind {
            match best {
                Some(current) if current.edge_count >= report.edge_count => {}
                _ => best = Some(report),
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::{CleanupPolicy, RelationKind};
    use crate::world::World;

    /// `ChildOf` — exclusive hierarchy relation with cascade-delete targets.
    struct ChildOf;
    impl Component for ChildOf {}

    /// `LocatedIn` — transitive containment relation.
    struct LocatedIn;
    impl Component for LocatedIn {}

    /// `EquippedBy` — never explicitly registered; exercises the default-kind
    /// fallback for edge-bearing but unregistered relations.
    struct EquippedBy;
    impl Component for EquippedBy {}

    #[test]
    fn empty_world_report_is_empty() {
        let world = World::new();
        let report = RelationGraphReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.kind_count, 0);
        assert_eq!(report.total_edges, 0);
        assert_eq!(report.relation_count(), 0);
        assert!(report.hottest_kind().is_none());
    }

    #[test]
    fn registered_kind_without_edges_appears_with_zero_counts() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(
            RelationKind::new()
                .with_exclusive(true)
                .with_on_delete_target(CleanupPolicy::Delete),
        );
        let child_of = world.components().id_of::<ChildOf>().unwrap();

        let report = RelationGraphReport::capture(&world);
        assert!(!report.is_empty());
        assert_eq!(report.kind_count, 1);
        assert_eq!(report.total_edges, 0);
        assert_eq!(report.relation_count(), 1);

        let kind = report.kind(child_of).expect("ChildOf reported");
        assert!(kind.registered);
        assert!(kind.exclusive);
        assert!(!kind.transitive);
        assert_eq!(kind.on_delete_target, CleanupPolicy::Delete);
        assert_eq!(kind.edge_count, 0);
        assert_eq!(kind.distinct_sources, 0);
        assert_eq!(kind.distinct_targets, 0);
    }

    #[test]
    fn edges_aggregate_per_kind_with_flags() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));

        let p1 = world.spawn(());
        let p2 = world.spawn(());
        let c1 = world.spawn(());
        let c2 = world.spawn(());
        let c3 = world.spawn(());
        // ChildOf: three edges, three distinct sources, two distinct targets.
        world.add_relation::<ChildOf>(c1, p1);
        world.add_relation::<ChildOf>(c2, p1);
        world.add_relation::<ChildOf>(c3, p2);

        let room = world.spawn(());
        let a = world.spawn(());
        let b = world.spawn(());
        // LocatedIn: two edges, two distinct sources, one distinct target.
        world.add_relation::<LocatedIn>(a, room);
        world.add_relation::<LocatedIn>(b, room);

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let located_in = world.components().id_of::<LocatedIn>().unwrap();

        let report = RelationGraphReport::capture(&world);
        assert_eq!(report.kind_count, 2);
        assert_eq!(report.total_edges, 5);
        assert_eq!(report.relation_count(), 2);

        let child = report.kind(child_of).unwrap();
        assert_eq!(child.edge_count, 3);
        assert_eq!(child.distinct_sources, 3);
        assert_eq!(child.distinct_targets, 2);
        assert!(child.exclusive);

        let located = report.kind(located_in).unwrap();
        assert_eq!(located.edge_count, 2);
        assert_eq!(located.distinct_sources, 2);
        assert_eq!(located.distinct_targets, 1);
        assert!(located.transitive);

        // ChildOf carries the most edges.
        let hottest = report.hottest_kind().unwrap();
        assert_eq!(hottest.relation, child_of);
        assert_eq!(hottest.edge_count, 3);

        // `per_kind` is deterministically sorted by relation id.
        let ids: Vec<u32> = report.per_kind.iter().map(|r| r.relation.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn unregistered_relation_with_edges_uses_default_flags() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        // No register_relation call: EquippedBy behaves as the default kind.
        world.add_relation::<EquippedBy>(a, b);

        let equipped = world.components().id_of::<EquippedBy>().unwrap();
        let report = RelationGraphReport::capture(&world);

        // Zero registered kinds, but one edge-bearing relation is reported.
        assert_eq!(report.kind_count, 0);
        assert_eq!(report.total_edges, 1);
        assert_eq!(report.relation_count(), 1);

        let kind = report.kind(equipped).unwrap();
        assert!(!kind.registered);
        assert!(!kind.fragmenting);
        assert!(!kind.transitive);
        assert!(!kind.exclusive);
        assert_eq!(kind.on_delete, CleanupPolicy::Remove);
        assert_eq!(kind.on_delete_target, CleanupPolicy::Remove);
        assert_eq!(kind.edge_count, 1);
        assert_eq!(kind.distinct_sources, 1);
        assert_eq!(kind.distinct_targets, 1);
    }
}
