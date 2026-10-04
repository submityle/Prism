//! Relation topology diagnostics (design §16.6 / §11 / §22 risk #5).
//!
//! The relation-graph summary ([`relation_graph`](super::relation_graph))
//! answers "how many edges does each relation kind hold, over how many distinct
//! sources and targets", and the cycle detector
//! ([`relation_cycles`](super::relation_cycles)) answers "is any relation's
//! directed graph cyclic". This module answers the remaining *shape* questions
//! an editor hierarchy panel needs about each relation's directed graph:
//!
//! * **Depth** — the longest directed chain. For `ChildOf` this is the deepest
//!   hierarchy branch; for a transitive containment relation it bounds the
//!   worst-case closure walk (design §11).
//! * **Fan-out / fan-in extremes** — the widest source (most outgoing edges)
//!   and the most-referenced target (most incoming edges), with the owning
//!   entity so a devtools panel can jump straight to the hotspot.
//! * **Roots and sinks** — source-roots (outgoing edges, no incoming) and sinks
//!   (incoming edges, no outgoing). A healthy hierarchy has few roots and many
//!   sinks; a flood of roots is the archetype-explosion-adjacent "orphan
//!   subtree" smell (design §22 risk #5).
//!
//! All figures are stated in graph edge-direction terms (`source --rel-->
//! target`), independent of whatever gameplay meaning a relation attaches to
//! that direction, so the module stays correct for `ChildOf` (child points at
//! parent), ownership, targeting, and any other edge convention.
//!
//! Capture walks the non-fragmenting [`RelationIndex`](crate::relation::RelationIndex)
//! once per relation, is read-only, and is deterministic: relations are sorted
//! by [`ComponentId`] index, extreme-entity ties resolve to the lowest
//! [`Entity`], and the longest-path computation uses an order-independent
//! Kahn/topological dynamic program.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::relation::Relations;
use crate::world::World;

/// Per-relation degree accumulator built from the edge list.
#[derive(Default)]
struct TopoAcc {
    /// `source -> targets`; the directed adjacency for this relation.
    out: HashMap<Entity, Vec<Entity>>,
    /// Out-degree (distinct targets) per source.
    outdeg: HashMap<Entity, usize>,
    /// In-degree (distinct sources) per target.
    indeg: HashMap<Entity, usize>,
    /// Every entity touched by an edge, used as the node set.
    nodes: HashMap<Entity, ()>,
    /// Total directed edges.
    edge_count: usize,
}

impl TopoAcc {
    fn add_edge(&mut self, source: Entity, target: Entity) {
        self.out.entry(source).or_default().push(target);
        *self.outdeg.entry(source).or_insert(0) += 1;
        *self.indeg.entry(target).or_insert(0) += 1;
        self.nodes.insert(source, ());
        self.nodes.insert(target, ());
        self.edge_count += 1;
    }

    /// Node set as a deterministically sorted vector (ascending [`Entity`]).
    fn sorted_nodes(&self) -> Vec<Entity> {
        let mut nodes: Vec<Entity> = self.nodes.keys().copied().collect();
        nodes.sort_unstable();
        nodes
    }
}

/// Directed-graph shape facts for a single relation kind (design §16.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationTopologyEntry {
    /// The relation marker component this entry describes.
    pub relation: ComponentId,
    /// Whether the relation kind was explicitly registered (vs. inferred from
    /// its edges).
    pub registered: bool,
    /// Total directed edges held for this relation.
    pub edge_count: usize,
    /// Distinct source entities (out-degree ≥ 1).
    pub distinct_sources: usize,
    /// Distinct target entities (in-degree ≥ 1).
    pub distinct_targets: usize,
    /// Source-roots: nodes with outgoing edges but no incoming edge.
    pub root_count: usize,
    /// Sinks: nodes with incoming edges but no outgoing edge.
    pub sink_count: usize,
    /// Largest out-degree held by any single source.
    pub max_out_degree: usize,
    /// The lowest-id source achieving [`max_out_degree`](Self::max_out_degree),
    /// or `None` when there are no edges.
    pub max_out_entity: Option<Entity>,
    /// Largest in-degree held by any single target.
    pub max_in_degree: usize,
    /// The lowest-id target achieving [`max_in_degree`](Self::max_in_degree),
    /// or `None` when there are no edges.
    pub max_in_entity: Option<Entity>,
    /// Length, in edges, of the longest directed chain. When
    /// [`has_cycle`](Self::has_cycle) is set this is the longest chain through
    /// the acyclic portion only (a lower bound), since an unbounded cycle has
    /// no finite depth.
    pub max_depth: usize,
    /// Whether the relation's directed graph contains a cycle (so
    /// [`max_depth`](Self::max_depth) is a lower bound rather than exact).
    pub has_cycle: bool,
}

impl RelationTopologyEntry {
    /// Mean out-degree across sources (`edge_count / distinct_sources`); `0.0`
    /// when the relation holds no edges.
    #[inline]
    pub fn avg_out_degree(&self) -> f32 {
        if self.distinct_sources == 0 {
            0.0
        } else {
            self.edge_count as f32 / self.distinct_sources as f32
        }
    }

    /// Whether the directed graph is acyclic (depth is exact).
    #[inline]
    pub fn is_acyclic(&self) -> bool {
        !self.has_cycle
    }

    /// Whether the relation is "flat" — no chain longer than a single edge
    /// (`max_depth <= 1`), e.g. a one-level parent/child star rather than a
    /// deep tree.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.max_depth <= 1
    }

    /// Whether the relation holds no edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// Build the topology entry for one relation from its edge accumulator.
    fn build(relation: ComponentId, registered: bool, acc: &TopoAcc) -> Self {
        let nodes = acc.sorted_nodes();

        let mut root_count = 0;
        let mut sink_count = 0;
        let mut max_out_degree = 0;
        let mut max_out_entity = None;
        let mut max_in_degree = 0;
        let mut max_in_entity = None;

        // `nodes` is ascending, so the first entity to reach each maximum keeps
        // it (strict `>`), yielding the lowest-id tie-break deterministically.
        for &e in &nodes {
            let out = acc.outdeg.get(&e).copied().unwrap_or(0);
            let inc = acc.indeg.get(&e).copied().unwrap_or(0);
            if out > 0 && inc == 0 {
                root_count += 1;
            }
            if inc > 0 && out == 0 {
                sink_count += 1;
            }
            if out > max_out_degree {
                max_out_degree = out;
                max_out_entity = Some(e);
            }
            if inc > max_in_degree {
                max_in_degree = inc;
                max_in_entity = Some(e);
            }
        }

        let (max_depth, has_cycle) = longest_path(acc, &nodes);

        Self {
            relation,
            registered,
            edge_count: acc.edge_count,
            distinct_sources: acc.outdeg.len(),
            distinct_targets: acc.indeg.len(),
            root_count,
            sink_count,
            max_out_degree,
            max_out_entity,
            max_in_degree,
            max_in_entity,
            max_depth,
            has_cycle,
        }
    }
}

/// Longest directed path (in edges) via a Kahn topological dynamic program.
///
/// Returns `(max_depth, has_cycle)`. The DP value `dist[v] = max(dist[u] + 1)`
/// over resolved predecessors `u` is independent of the queue draining order
/// for any valid topological order, so the depth is deterministic without
/// sorting the work queue. Nodes left unprocessed indicate a cycle; their chain
/// contribution is excluded, making `max_depth` a lower bound in that case.
fn longest_path(acc: &TopoAcc, nodes: &[Entity]) -> (usize, bool) {
    if nodes.is_empty() {
        return (0, false);
    }

    // Working in-degree we decrement as edges are "removed".
    let mut remaining: HashMap<Entity, usize> =
        HashMap::with_capacity(nodes.len());
    for &n in nodes {
        remaining.insert(n, acc.indeg.get(&n).copied().unwrap_or(0));
    }

    let mut dist: HashMap<Entity, usize> = HashMap::with_capacity(nodes.len());
    let mut queue: Vec<Entity> = Vec::new();
    // Seed with roots (in-degree 0). `nodes` is sorted, so the queue starts
    // ordered; this does not affect the DP result, only aids reproducibility.
    for &n in nodes {
        if remaining.get(&n).copied().unwrap_or(0) == 0 {
            queue.push(n);
            dist.insert(n, 0);
        }
    }

    let mut head = 0;
    let mut processed = 0;
    let mut max_depth = 0;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        processed += 1;
        let du = dist.get(&u).copied().unwrap_or(0);
        if let Some(targets) = acc.out.get(&u) {
            for &v in targets {
                let nd = du + 1;
                let entry = dist.entry(v).or_insert(0);
                if nd > *entry {
                    *entry = nd;
                }
                if nd > max_depth {
                    max_depth = nd;
                }
                let rem = remaining.get_mut(&v).expect("target is a known node");
                *rem -= 1;
                if *rem == 0 {
                    queue.push(v);
                }
            }
        }
    }

    let has_cycle = processed < nodes.len();
    (max_depth, has_cycle)
}

/// Whole-world relation topology summary (design §16.6).
///
/// Produced by [`RelationTopologyReport::capture`]. [`entries`](Self::entries)
/// holds one [`RelationTopologyEntry`] per relation that is registered or holds
/// edges, sorted by relation [`ComponentId`] index for determinism.
#[derive(Debug, Clone, Default)]
pub struct RelationTopologyReport {
    /// One entry per relation (registered or edge-bearing), sorted by relation
    /// [`ComponentId`] index.
    pub entries: Vec<RelationTopologyEntry>,
    /// Number of explicitly registered relation kinds (design §11).
    pub kind_count: usize,
    /// Total directed edges across every relation.
    pub total_edges: usize,
}

impl RelationTopologyReport {
    /// Capture the relation topology of `world` (design §16.6).
    ///
    /// Walks the registered relation kinds and the live edge index once each,
    /// grouping edges per relation and folding each group into degree, root /
    /// sink, and longest-chain figures. Read-only and deterministic.
    pub fn capture(world: &World) -> Self {
        Self::from_relations(world.relations())
    }

    /// Build the report directly from a [`Relations`] registry, so callers that
    /// already hold one need not route through [`World`].
    pub fn from_relations(relations: &Relations) -> Self {
        // Group every edge by its relation component.
        let mut by_relation: HashMap<ComponentId, TopoAcc> = HashMap::new();
        for (relation, source, target) in relations.index().iter_edges() {
            by_relation
                .entry(relation)
                .or_default()
                .add_edge(source, target);
        }

        // Registered kinds appear even with zero edges; mark which relations
        // were explicitly registered.
        let mut registered: HashMap<ComponentId, ()> = HashMap::new();
        for (relation, _kind) in relations.iter_kinds() {
            registered.insert(relation, ());
            by_relation.entry(relation).or_default();
        }

        let empty = TopoAcc::default();
        let mut relations_seen: Vec<ComponentId> = by_relation.keys().copied().collect();
        relations_seen.sort_unstable_by_key(|c| c.index());

        let mut entries = Vec::with_capacity(relations_seen.len());
        let mut total_edges = 0;
        for relation in relations_seen {
            let acc = by_relation.get(&relation).unwrap_or(&empty);
            total_edges += acc.edge_count;
            entries.push(RelationTopologyEntry::build(
                relation,
                registered.contains_key(&relation),
                acc,
            ));
        }

        Self {
            entries,
            kind_count: relations.kind_count(),
            total_edges,
        }
    }

    /// Whether no relation is registered or carries edges.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of relations described (registered or edge-bearing).
    #[inline]
    pub fn relation_count(&self) -> usize {
        self.entries.len()
    }

    /// The entry for `relation`, if present.
    pub fn entry(&self, relation: ComponentId) -> Option<&RelationTopologyEntry> {
        self.entries.iter().find(|e| e.relation == relation)
    }

    /// The relation with the deepest directed chain
    /// ([`max_depth`](RelationTopologyEntry::max_depth)); ties resolve to the
    /// lowest relation [`ComponentId`] index. `None` when there are no entries.
    pub fn deepest(&self) -> Option<&RelationTopologyEntry> {
        self.entries.iter().max_by(|a, b| {
            a.max_depth
                .cmp(&b.max_depth)
                .then_with(|| b.relation.index().cmp(&a.relation.index()))
        })
    }

    /// The relation with the widest single source
    /// ([`max_out_degree`](RelationTopologyEntry::max_out_degree)); ties resolve
    /// to the lowest relation [`ComponentId`] index. `None` when there are no
    /// entries.
    pub fn widest(&self) -> Option<&RelationTopologyEntry> {
        self.entries.iter().max_by(|a, b| {
            a.max_out_degree
                .cmp(&b.max_out_degree)
                .then_with(|| b.relation.index().cmp(&a.relation.index()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::world::World;

    /// `ChildOf` — exclusive single-parent hierarchy (child points at parent).
    struct ChildOf;
    impl Component for ChildOf {}

    /// `Targets` — plain, high-cardinality relation.
    struct Targets;
    impl Component for Targets {}

    #[test]
    fn empty_world_has_no_topology() {
        let world = World::new();
        let report = RelationTopologyReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.relation_count(), 0);
        assert_eq!(report.total_edges, 0);
        assert!(report.deepest().is_none());
        assert!(report.widest().is_none());
    }

    #[test]
    fn registered_relation_with_no_edges_appears_empty() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let report = RelationTopologyReport::capture(&world);

        let entry = report.entry(child_of).expect("registered kind is listed");
        assert!(entry.registered);
        assert!(entry.is_empty());
        assert_eq!(entry.edge_count, 0);
        assert_eq!(entry.max_depth, 0);
        assert!(entry.max_out_entity.is_none());
        assert!(!entry.has_cycle);
        assert_eq!(report.kind_count, 1);
    }

    #[test]
    fn chain_depth_roots_and_sinks() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        // a -> b -> c : a two-edge chain.
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, c);

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let entry = *RelationTopologyReport::capture(&world)
            .entry(child_of)
            .unwrap();

        assert_eq!(entry.edge_count, 2);
        assert_eq!(entry.max_depth, 2, "longest chain a->b->c is two edges");
        assert!(entry.is_acyclic());
        assert!(!entry.is_flat());
        // a has an outgoing edge and no incoming → the one source-root.
        assert_eq!(entry.root_count, 1);
        // c has an incoming edge and no outgoing → the one sink.
        assert_eq!(entry.sink_count, 1);
        assert_eq!(entry.distinct_sources, 2, "a and b are sources");
        assert_eq!(entry.distinct_targets, 2, "b and c are targets");
    }

    #[test]
    fn star_fan_in_picks_hub_and_stays_flat() {
        let mut world = World::new();
        // Four children all point at one parent: a flat, in-heavy star.
        let parent = world.spawn(());
        let c0 = world.spawn(());
        let c1 = world.spawn(());
        let c2 = world.spawn(());
        world.add_relation::<ChildOf>(c0, parent);
        world.add_relation::<ChildOf>(c1, parent);
        world.add_relation::<ChildOf>(c2, parent);

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let report = RelationTopologyReport::capture(&world);
        let entry = *report.entry(child_of).unwrap();

        assert_eq!(entry.max_depth, 1, "no chain longer than one edge");
        assert!(entry.is_flat());
        assert_eq!(entry.max_in_degree, 3, "the parent is pointed at thrice");
        assert_eq!(entry.max_in_entity, Some(parent));
        assert_eq!(entry.max_out_degree, 1, "each child points once");
        assert_eq!(entry.root_count, 3, "the three children are source-roots");
        assert_eq!(entry.sink_count, 1, "the parent is the only sink");
        // ChildOf is this world's only relation, so it is both deepest & widest.
        assert_eq!(report.deepest().unwrap().relation, child_of);
        assert_eq!(report.widest().unwrap().relation, child_of);
    }

    #[test]
    fn cycle_sets_flag_and_bounds_depth() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        // a -> b -> c -> a : a pure three-node cycle (no in-degree-0 node).
        world.add_relation::<Targets>(a, b);
        world.add_relation::<Targets>(b, c);
        world.add_relation::<Targets>(c, a);

        let targets = world.components().id_of::<Targets>().unwrap();
        let entry = *RelationTopologyReport::capture(&world)
            .entry(targets)
            .unwrap();

        assert!(entry.has_cycle);
        assert!(!entry.is_acyclic());
        assert_eq!(entry.edge_count, 3);
        // Every node is on the loop, so none seeds the topological DP.
        assert_eq!(entry.max_depth, 0, "a pure cycle contributes no finite chain");
        assert_eq!(entry.root_count, 0);
        assert_eq!(entry.sink_count, 0);
    }

    #[test]
    fn widest_and_deepest_select_across_relations() {
        let mut world = World::new();
        // ChildOf: a deep but narrow chain h0->h1->h2->h3 (depth 3, fan-out 1).
        let h: Vec<_> = (0..4).map(|_| world.spawn(())).collect();
        for pair in h.windows(2) {
            world.add_relation::<ChildOf>(pair[0], pair[1]);
        }
        // Targets: a flat but wide star hub->{t0,t1,t2,t3,t4} (depth 1, fan-out 5).
        let hub = world.spawn(());
        for _ in 0..5 {
            let t = world.spawn(());
            world.add_relation::<Targets>(hub, t);
        }

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let targets = world.components().id_of::<Targets>().unwrap();
        let report = RelationTopologyReport::capture(&world);

        assert_eq!(report.relation_count(), 2);
        assert_eq!(report.total_edges, 8, "3 chain edges + 5 star edges");
        assert_eq!(report.deepest().unwrap().relation, child_of, "chain is deepest");
        assert_eq!(report.widest().unwrap().relation, targets, "star is widest");
        assert_eq!(report.deepest().unwrap().max_depth, 3);
        assert_eq!(report.widest().unwrap().max_out_degree, 5);
    }
}
