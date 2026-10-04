//! Per-entity relation connectivity census (design §11 / §16.6).
//!
//! The other relation diagnostics in this module look at the graph *per
//! relation kind*: [`relation_graph`](super::relation_graph) counts edges and
//! endpoints of each kind, [`relation_topology`](super::relation_topology)
//! measures each kind's chain depth and fan extremes, and
//! [`relation_cascade`](super::relation_cascade) walks each kind's cleanup
//! policy. None of them answer the orthogonal question *"which entities are the
//! relation hubs?"* — the entities whose combined participation across **all**
//! relation kinds makes them expensive.
//!
//! That cross-kind, per-entity view matters because the relation index is a
//! bidirectional store (design §11, "关系索引缓存"): a source with a large
//! aggregate out-degree drives wildcard-query (`query_pair(r, *)`) spread and
//! forward-bucket size, while a target with a large aggregate in-degree drives
//! the reverse index and the cascade blast when it is despawned (design §23.2).
//! A single entity can be a hub in several kinds at once, and only an
//! entity-centric roll-up surfaces that.
//!
//! For every entity that participates in at least one edge this report records
//! its total outgoing and incoming edge counts, how many distinct relation
//! kinds it sources and sinks, and the union breadth of kinds it touches. It
//! also classifies each entity's structural role — pure source, pure sink, or
//! relay (both directions) — which is the raw, policy-agnostic shape underneath
//! the roots/sinks that [`relation_topology`](super::relation_topology) reports
//! per kind.
//!
//! Capture reads only the public relation index
//! ([`RelationIndex`](crate::relation::RelationIndex)) and never mutates
//! simulation state. Entities are listed in a deterministic order (ascending
//! [`Entity::to_bits`]) so the report is reproducible (design §14).

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::world::World;

/// Relation connectivity roll-up for one participating entity across every
/// relation kind (design §11).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RelationDegreeEntry {
    /// The participating entity.
    pub entity: Entity,
    /// Total outgoing edges summed over every relation kind (the entity as a
    /// relation *source*).
    pub outgoing: usize,
    /// Total incoming edges summed over every relation kind (the entity as a
    /// relation *target*).
    pub incoming: usize,
    /// Distinct relation kinds in which the entity holds at least one outgoing
    /// edge.
    pub out_kinds: usize,
    /// Distinct relation kinds in which the entity holds at least one incoming
    /// edge.
    pub in_kinds: usize,
    /// Distinct relation kinds the entity touches in *either* direction (the
    /// union of the kinds behind [`out_kinds`](Self::out_kinds) and
    /// [`in_kinds`](Self::in_kinds)).
    pub kinds: usize,
}

impl RelationDegreeEntry {
    /// Total edges incident on the entity: `outgoing + incoming` (saturating).
    /// This is the aggregate relation-index weight the entity carries.
    #[inline]
    pub fn total_degree(&self) -> usize {
        self.outgoing.saturating_add(self.incoming)
    }

    /// Whether the entity is a pure *source*: it holds outgoing edges but no
    /// incoming ones (a root in every kind it participates in).
    #[inline]
    pub fn is_pure_source(&self) -> bool {
        self.outgoing > 0 && self.incoming == 0
    }

    /// Whether the entity is a pure *sink*: it holds incoming edges but no
    /// outgoing ones (a leaf in every kind it participates in).
    #[inline]
    pub fn is_pure_sink(&self) -> bool {
        self.incoming > 0 && self.outgoing == 0
    }

    /// Whether the entity is a *relay*: it holds both outgoing and incoming
    /// edges, so it sits in the interior of at least one chain.
    #[inline]
    pub fn is_relay(&self) -> bool {
        self.outgoing > 0 && self.incoming > 0
    }
}

/// Read-only per-entity relation connectivity census over the whole relation
/// index (design §11 / §16.6).
#[derive(Clone, Debug, Default)]
pub struct RelationDegreeReport {
    /// Per-entity roll-ups, sorted ascending by [`Entity::to_bits`].
    entries: Vec<RelationDegreeEntry>,
}

impl RelationDegreeReport {
    /// Capture a relation-degree census from `world`. Read-only; every entity
    /// that participates in at least one relation edge (as source or target)
    /// gets exactly one entry. Entries are sorted deterministically by
    /// [`Entity::to_bits`] (design §14).
    pub fn from_world(world: &World) -> Self {
        let index = world.relations().index();

        // Distinct participants: every entity appearing as a source or target.
        let mut participants: Vec<Entity> = Vec::new();
        for (_relation, source, target) in index.iter_edges() {
            participants.push(source);
            participants.push(target);
        }
        participants.sort_unstable_by_key(|entity| entity.to_bits());
        participants.dedup();

        let mut entries: Vec<RelationDegreeEntry> = Vec::with_capacity(participants.len());
        for &entity in &participants {
            let out_edges = index.outgoing_edges(entity);
            let in_edges = index.incoming_edges(entity);

            let outgoing = out_edges.len();
            let incoming = in_edges.len();
            let out_kinds = distinct_relation_count(out_edges.iter().map(|&(relation, _)| relation));
            let in_kinds = distinct_relation_count(in_edges.iter().map(|&(relation, _)| relation));

            // Union breadth of kinds touched in either direction.
            let mut union: Vec<ComponentId> =
                Vec::with_capacity(out_edges.len() + in_edges.len());
            union.extend(out_edges.iter().map(|&(relation, _)| relation));
            union.extend(in_edges.iter().map(|&(relation, _)| relation));
            union.sort_unstable();
            union.dedup();
            let kinds = union.len();

            entries.push(RelationDegreeEntry {
                entity,
                outgoing,
                incoming,
                out_kinds,
                in_kinds,
                kinds,
            });
        }

        Self { entries }
    }

    /// The per-entity roll-ups, sorted ascending by [`Entity::to_bits`].
    #[inline]
    pub fn entries(&self) -> &[RelationDegreeEntry] {
        &self.entries
    }

    /// Whether no entity participates in any relation edge.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entities participating in at least one relation edge.
    #[inline]
    pub fn participant_count(&self) -> usize {
        self.entries.len()
    }

    /// Total outgoing edges summed over every participant. Equals the relation
    /// index edge count, since every edge is one entity's outgoing edge.
    pub fn total_outgoing(&self) -> usize {
        self.entries.iter().map(|entry| entry.outgoing).sum()
    }

    /// Total incoming edges summed over every participant. Equals the relation
    /// index edge count, since every edge is one entity's incoming edge.
    pub fn total_incoming(&self) -> usize {
        self.entries.iter().map(|entry| entry.incoming).sum()
    }

    /// Number of pure-source entities (outgoing only).
    pub fn pure_source_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_pure_source())
            .count()
    }

    /// Number of pure-sink entities (incoming only).
    pub fn pure_sink_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.is_pure_sink())
            .count()
    }

    /// Number of relay entities (both outgoing and incoming edges).
    pub fn relay_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.is_relay()).count()
    }

    /// The widest kind-union breadth across all participants, i.e. the most
    /// distinct relation kinds any single entity touches. Zero on an empty
    /// report.
    pub fn max_kind_breadth(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.kinds)
            .max()
            .unwrap_or(0)
    }

    /// The entity with the largest aggregate out-degree. Ties resolve to the
    /// lowest [`Entity::to_bits`]. `None` when the report is empty.
    pub fn busiest_source(&self) -> Option<&RelationDegreeEntry> {
        self.pick_max(|entry| entry.outgoing)
    }

    /// The entity with the largest aggregate in-degree. Ties resolve to the
    /// lowest [`Entity::to_bits`]. `None` when the report is empty.
    pub fn busiest_sink(&self) -> Option<&RelationDegreeEntry> {
        self.pick_max(|entry| entry.incoming)
    }

    /// The entity with the largest total degree (outgoing + incoming). Ties
    /// resolve to the lowest [`Entity::to_bits`]. `None` when the report is
    /// empty.
    pub fn busiest(&self) -> Option<&RelationDegreeEntry> {
        self.pick_max(|entry| entry.total_degree())
    }

    /// Look up a participant's entry by entity. `O(log n)` binary search over
    /// the [`Entity::to_bits`]-sorted entries.
    pub fn entry(&self, entity: Entity) -> Option<&RelationDegreeEntry> {
        self.entries
            .binary_search_by_key(&entity.to_bits(), |candidate| candidate.entity.to_bits())
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Pick the entry maximising `key`, breaking ties on the lowest
    /// [`Entity::to_bits`]. Because entries are sorted ascending by bits and we
    /// keep the first strict maximum, the tie-break is the lowest-bits entity.
    fn pick_max(
        &self,
        key: impl Fn(&RelationDegreeEntry) -> usize,
    ) -> Option<&RelationDegreeEntry> {
        let mut best: Option<&RelationDegreeEntry> = None;
        let mut best_key = 0usize;
        for entry in &self.entries {
            let value = key(entry);
            if best.is_none() || value > best_key {
                best = Some(entry);
                best_key = value;
            }
        }
        best
    }
}

/// Count distinct [`ComponentId`]s in an iterator by collecting, sorting, and
/// de-duplicating. Deterministic and allocation-local.
fn distinct_relation_count(relations: impl Iterator<Item = ComponentId>) -> usize {
    let mut ids: Vec<ComponentId> = relations.collect();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::world::World;

    /// `ChildOf` — a hierarchy relation.
    #[derive(PartialEq)]
    struct ChildOf;
    impl Component for ChildOf {}

    /// `EquippedBy` — a second relation kind for multi-kind tests.
    #[derive(PartialEq)]
    struct EquippedBy;
    impl Component for EquippedBy {}

    fn entry_of<'r>(
        report: &'r RelationDegreeReport,
        entity: Entity,
    ) -> &'r RelationDegreeEntry {
        report.entry(entity).expect("participant must have an entry")
    }

    #[test]
    fn empty_world_has_no_entries() {
        let world = World::new();
        let report = RelationDegreeReport::from_world(&world);
        assert!(report.is_empty());
        assert_eq!(report.participant_count(), 0);
        assert_eq!(report.busiest(), None);
        assert_eq!(report.busiest_source(), None);
        assert_eq!(report.busiest_sink(), None);
        assert_eq!(report.max_kind_breadth(), 0);
    }

    #[test]
    fn single_edge_lists_both_endpoints_with_direction() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<ChildOf>(a, b);

        let report = RelationDegreeReport::from_world(&world);
        assert_eq!(report.participant_count(), 2);

        let ea = entry_of(&report, a);
        assert_eq!(ea.outgoing, 1);
        assert_eq!(ea.incoming, 0);
        assert_eq!(ea.out_kinds, 1);
        assert_eq!(ea.in_kinds, 0);
        assert_eq!(ea.kinds, 1);
        assert!(ea.is_pure_source());

        let eb = entry_of(&report, b);
        assert_eq!(eb.outgoing, 0);
        assert_eq!(eb.incoming, 1);
        assert!(eb.is_pure_sink());
    }

    #[test]
    fn relay_entity_has_both_directions() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        // a -> b -> c: b is the relay.
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, c);

        let report = RelationDegreeReport::from_world(&world);
        let eb = entry_of(&report, b);
        assert_eq!(eb.outgoing, 1);
        assert_eq!(eb.incoming, 1);
        assert!(eb.is_relay());
        assert!(!eb.is_pure_source());
        assert!(!eb.is_pure_sink());
        assert_eq!(eb.total_degree(), 2);

        assert_eq!(report.pure_source_count(), 1); // a
        assert_eq!(report.pure_sink_count(), 1); // c
        assert_eq!(report.relay_count(), 1); // b
    }

    #[test]
    fn multi_kind_source_counts_distinct_kinds() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        world.register_relation::<EquippedBy>(RelationKind::new());
        // hub sources two different relation kinds.
        let hub = world.spawn(());
        let child = world.spawn(());
        let item = world.spawn(());
        world.add_relation::<ChildOf>(hub, child);
        world.add_relation::<EquippedBy>(hub, item);

        let report = RelationDegreeReport::from_world(&world);
        let hub_entry = entry_of(&report, hub);
        assert_eq!(hub_entry.outgoing, 2);
        assert_eq!(hub_entry.out_kinds, 2);
        assert_eq!(hub_entry.kinds, 2);
        assert_eq!(report.max_kind_breadth(), 2);
    }

    #[test]
    fn busiest_source_identifies_max_outgoing() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        let hub = world.spawn(());
        let x = world.spawn(());
        let y = world.spawn(());
        let z = world.spawn(());
        world.add_relation::<ChildOf>(hub, x);
        world.add_relation::<ChildOf>(hub, y);
        world.add_relation::<ChildOf>(hub, z);

        let report = RelationDegreeReport::from_world(&world);
        let busiest = report.busiest_source().unwrap();
        assert_eq!(busiest.entity, hub);
        assert_eq!(busiest.outgoing, 3);
        // hub is also the overall busiest here.
        assert_eq!(report.busiest().unwrap().entity, hub);
    }

    #[test]
    fn busiest_sink_identifies_max_incoming() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        let a = world.spawn(());
        let b = world.spawn(());
        let shrine = world.spawn(());
        world.add_relation::<ChildOf>(a, shrine);
        world.add_relation::<ChildOf>(b, shrine);

        let report = RelationDegreeReport::from_world(&world);
        let sink = report.busiest_sink().unwrap();
        assert_eq!(sink.entity, shrine);
        assert_eq!(sink.incoming, 2);
    }

    #[test]
    fn ties_resolve_to_lowest_bits() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        // Two independent single edges => two sources each with outgoing == 1.
        let s1 = world.spawn(());
        let t1 = world.spawn(());
        let s2 = world.spawn(());
        let t2 = world.spawn(());
        world.add_relation::<ChildOf>(s1, t1);
        world.add_relation::<ChildOf>(s2, t2);

        let report = RelationDegreeReport::from_world(&world);
        let lowest = if s1.to_bits() <= s2.to_bits() { s1 } else { s2 };
        assert_eq!(report.busiest_source().unwrap().entity, lowest);
    }

    #[test]
    fn totals_equal_edge_count() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        world.register_relation::<EquippedBy>(RelationKind::new());
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, c);
        world.add_relation::<EquippedBy>(a, c);

        let report = RelationDegreeReport::from_world(&world);
        let edge_count = world.relations().index().edge_count();
        assert_eq!(report.total_outgoing(), edge_count);
        assert_eq!(report.total_incoming(), edge_count);
        assert_eq!(report.total_outgoing(), report.total_incoming());
    }

    #[test]
    fn entries_sorted_by_bits_and_lookup_round_trips() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new());
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, c);

        let report = RelationDegreeReport::from_world(&world);
        let bits: Vec<u64> = report.entries().iter().map(|e| e.entity.to_bits()).collect();
        let mut sorted = bits.clone();
        sorted.sort_unstable();
        assert_eq!(bits, sorted);
        for entry in report.entries() {
            assert_eq!(report.entry(entry.entity), Some(entry));
        }
    }
}
