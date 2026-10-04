//! Relation cycle detection (design §16.6 / §11 / §23.2).
//!
//! Hierarchical and transitive relations are expected to be *acyclic*: a cycle
//! in `ChildOf` makes [`despawn_recursive`](crate::world) loop forever, a cycle
//! in a transitive relation makes its closure (design §11) diverge, and a cycle
//! in any relation with a cascade [`CleanupPolicy::Delete`](crate::relation::CleanupPolicy)
//! target rule (design §23.2) turns a single delete into an unbounded cascade.
//! This module walks the non-fragmenting [`RelationIndex`](crate::relation::RelationIndex)
//! per relation kind and reports, for each relation that carries at least one
//! edge, whether its directed graph contains a cycle — plus one concrete
//! example cycle for the editor / log to surface.
//!
//! A cycle is only a *bug* for relations that model a hierarchy or ordering;
//! mutual relations (e.g. `Likes`) may legitimately cycle. The report therefore
//! records the relation's `transitive` / `exclusive` flags so callers can tell
//! an expected cycle from a suspicious one (see
//! [`RelationCycleEntry::is_suspicious`]).
//!
//! Capture is read-only and allocates its own working maps; it never mutates
//! the world. Output is deterministic: relations are sorted by
//! [`ComponentId`] index, adjacency is explored in [`Entity`] order, and the
//! reported example cycle is the first one found under that fixed order.

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::world::World;

/// DFS visit color for three-color cycle detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    /// On the current DFS stack (a back-edge to a gray node is a cycle).
    Gray,
    /// Fully explored; cannot be part of a new cycle through this root.
    Black,
}

/// Per-relation cycle finding for one relation kind (design §16.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationCycleEntry {
    /// The relation marker component this entry describes.
    pub relation: ComponentId,
    /// Whether the relation kind was registered with transitive semantics
    /// (design §11); a cycle here corrupts the transitive closure.
    pub transitive: bool,
    /// Whether the relation kind was registered as exclusive (design §23.3);
    /// exclusive relations model single-parent hierarchies that must be
    /// acyclic.
    pub exclusive: bool,
    /// Number of directed edges scanned for this relation.
    pub edge_count: usize,
    /// Whether a directed cycle was found.
    pub has_cycle: bool,
    /// One concrete cycle, as the ordered list of entities visited around the
    /// loop (the edge from the last back to the first closes it). Empty when
    /// [`has_cycle`](Self::has_cycle) is `false`. A single-element vector is a
    /// self-loop (`e --rel--> e`).
    pub example_cycle: Vec<Entity>,
}

impl RelationCycleEntry {
    /// Whether this cycle is likely a bug: a cycle in a transitive or exclusive
    /// relation breaks the hierarchy / closure invariants (design §11 / §23.3),
    /// whereas a cycle in a plain symmetric relation may be intentional.
    #[inline]
    pub fn is_suspicious(&self) -> bool {
        self.has_cycle && (self.transitive || self.exclusive)
    }

    /// The number of entities on the reported example cycle (0 when acyclic).
    #[inline]
    pub fn cycle_len(&self) -> usize {
        self.example_cycle.len()
    }
}

/// A per-relation cycle report over every edge-bearing relation (design §16.6).
///
/// Produced by [`RelationCycleReport::capture`]. [`per_relation`](Self::per_relation)
/// is sorted by relation [`ComponentId`] index for determinism and contains
/// only relations that carry at least one edge (an edge-free relation cannot
/// cycle).
#[derive(Debug, Clone, Default)]
pub struct RelationCycleReport {
    /// One entry per edge-bearing relation, sorted by relation id.
    pub per_relation: Vec<RelationCycleEntry>,
}

impl RelationCycleReport {
    /// Scan every relation in the world's [`RelationIndex`](crate::relation::RelationIndex)
    /// for directed cycles.
    ///
    /// Groups edges by relation, builds a deterministic adjacency list, and runs
    /// an iterative three-color depth-first search per relation. Read-only.
    pub fn capture(world: &World) -> Self {
        let relations = world.relations();
        let index = relations.index();

        // Group outgoing adjacency by relation, in a stable structure.
        let mut adjacency: HashMap<ComponentId, HashMap<Entity, Vec<Entity>>> = HashMap::new();
        let mut edge_counts: HashMap<ComponentId, usize> = HashMap::new();
        for (relation, source, target) in index.iter_edges() {
            adjacency
                .entry(relation)
                .or_default()
                .entry(source)
                .or_default()
                .push(target);
            *edge_counts.entry(relation).or_insert(0) += 1;
        }

        let mut per_relation = Vec::with_capacity(adjacency.len());
        for (relation, adj) in adjacency {
            let edge_count = edge_counts.get(&relation).copied().unwrap_or(0);
            let example_cycle = find_cycle(&adj);
            let kind = relations.kind(relation);
            per_relation.push(RelationCycleEntry {
                relation,
                transitive: kind.is_some_and(|k| k.transitive),
                exclusive: kind.is_some_and(|k| k.exclusive),
                edge_count,
                has_cycle: !example_cycle.is_empty(),
                example_cycle,
            });
        }

        // Deterministic order independent of hash iteration.
        per_relation.sort_by_key(|e| e.relation.index());

        Self { per_relation }
    }

    /// Whether no edge-bearing relation was found (nothing to check).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.per_relation.is_empty()
    }

    /// Number of edge-bearing relations scanned.
    #[inline]
    pub fn relation_count(&self) -> usize {
        self.per_relation.len()
    }

    /// Whether any scanned relation contains a cycle.
    #[inline]
    pub fn has_any_cycle(&self) -> bool {
        self.per_relation.iter().any(|e| e.has_cycle)
    }

    /// Whether any scanned relation has a *suspicious* cycle — a cycle in a
    /// transitive or exclusive relation, which breaks a hierarchy / closure
    /// invariant (design §11 / §23.3).
    #[inline]
    pub fn has_suspicious_cycle(&self) -> bool {
        self.per_relation.iter().any(|e| e.is_suspicious())
    }

    /// The entry for `relation`, if it carried edges.
    pub fn entry(&self, relation: ComponentId) -> Option<&RelationCycleEntry> {
        self.per_relation.iter().find(|e| e.relation == relation)
    }

    /// Iterate the relations that contain a cycle.
    pub fn cycles(&self) -> impl Iterator<Item = &RelationCycleEntry> + '_ {
        self.per_relation.iter().filter(|e| e.has_cycle)
    }
}

/// Find one directed cycle in `adj`, returned as the ordered entities around
/// the loop, or an empty vector if the graph is acyclic.
///
/// Iterative three-color DFS so deep hierarchies cannot overflow the stack. The
/// exploration order is fixed (roots and neighbors visited in [`Entity`] order),
/// so the returned cycle is deterministic for a given graph.
fn find_cycle(adj: &HashMap<Entity, Vec<Entity>>) -> Vec<Entity> {
    // Deterministic, sorted adjacency and root ordering.
    let mut roots: Vec<Entity> = adj.keys().copied().collect();
    roots.sort_unstable();
    let mut sorted_adj: HashMap<Entity, Vec<Entity>> = HashMap::with_capacity(adj.len());
    for (&node, targets) in adj {
        let mut t = targets.clone();
        t.sort_unstable();
        sorted_adj.insert(node, t);
    }

    let empty: &[Entity] = &[];
    let neighbors = |node: Entity| -> &[Entity] {
        sorted_adj.get(&node).map(Vec::as_slice).unwrap_or(empty)
    };

    let mut color: HashMap<Entity, Color> = HashMap::new();

    for &root in &roots {
        if color.contains_key(&root) {
            continue;
        }
        // Explicit DFS stack of (node, next-neighbor-index); `path` mirrors the
        // gray nodes on the stack so a back edge can be reconstructed.
        let mut stack: Vec<(Entity, usize)> = Vec::new();
        let mut path: Vec<Entity> = Vec::new();

        color.insert(root, Color::Gray);
        stack.push((root, 0));
        path.push(root);

        while let Some(&(node, idx)) = stack.last() {
            let outs = neighbors(node);
            if idx < outs.len() {
                stack.last_mut().unwrap().1 += 1;
                let next = outs[idx];
                match color.get(&next).copied() {
                    Some(Color::Gray) => {
                        // Back edge: the cycle is the path suffix from `next`.
                        let pos = path.iter().position(|&e| e == next).unwrap();
                        return path[pos..].to_vec();
                    }
                    Some(Color::Black) => {}
                    None => {
                        color.insert(next, Color::Gray);
                        stack.push((next, 0));
                        path.push(next);
                    }
                }
            } else {
                color.insert(node, Color::Black);
                stack.pop();
                path.pop();
            }
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::world::World;

    /// `ChildOf` — exclusive single-parent hierarchy; a cycle is a bug.
    struct ChildOf;
    impl Component for ChildOf {}

    /// `LocatedIn` — transitive containment; a cycle corrupts the closure.
    struct LocatedIn;
    impl Component for LocatedIn {}

    /// `Likes` — plain, possibly-mutual relation; a cycle is legitimate.
    struct Likes;
    impl Component for Likes {}

    #[test]
    fn empty_world_has_nothing_to_check() {
        let world = World::new();
        let report = RelationCycleReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.relation_count(), 0);
        assert!(!report.has_any_cycle());
        assert!(!report.has_suspicious_cycle());
    }

    #[test]
    fn acyclic_hierarchy_reports_no_cycle() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        let root = world.spawn(());
        let a = world.spawn(());
        let b = world.spawn(());
        // a -> root, b -> root : a tree, acyclic.
        world.add_relation::<ChildOf>(a, root);
        world.add_relation::<ChildOf>(b, root);

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let report = RelationCycleReport::capture(&world);

        let entry = report.entry(child_of).expect("ChildOf scanned");
        assert_eq!(entry.edge_count, 2);
        assert!(!entry.has_cycle);
        assert!(entry.example_cycle.is_empty());
        assert!(!entry.is_suspicious());
        assert!(!report.has_any_cycle());
    }

    #[test]
    fn cycle_in_exclusive_relation_is_detected_and_suspicious() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        // a -> b -> c -> a : a three-node cycle.
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, c);
        world.add_relation::<ChildOf>(c, a);

        let child_of = world.components().id_of::<ChildOf>().unwrap();
        let report = RelationCycleReport::capture(&world);

        assert!(report.has_any_cycle());
        assert!(report.has_suspicious_cycle());

        let entry = report.entry(child_of).unwrap();
        assert!(entry.has_cycle);
        assert!(entry.exclusive);
        assert!(entry.is_suspicious());
        assert_eq!(entry.cycle_len(), 3, "all three nodes on the loop");
        // The reported cycle must be a genuine closed walk: each node links to
        // the next, and the last links back to the first.
        let cyc = &entry.example_cycle;
        for pair in cyc.windows(2) {
            assert!(world.has_relation::<ChildOf>(pair[0], pair[1]));
        }
        assert!(world.has_relation::<ChildOf>(cyc[cyc.len() - 1], cyc[0]));
    }

    #[test]
    fn self_loop_is_a_single_node_cycle() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        let e = world.spawn(());
        world.add_relation::<LocatedIn>(e, e);

        let located = world.components().id_of::<LocatedIn>().unwrap();
        let report = RelationCycleReport::capture(&world);
        let entry = report.entry(located).unwrap();
        assert!(entry.has_cycle);
        assert!(entry.transitive);
        assert!(entry.is_suspicious());
        assert_eq!(entry.example_cycle, alloc::vec![e]);
    }

    #[test]
    fn cycle_in_plain_relation_is_not_suspicious() {
        let mut world = World::new();
        // Likes is never registered: default kind (non-transitive, non-exclusive).
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<Likes>(a, b);
        world.add_relation::<Likes>(b, a);

        let likes = world.components().id_of::<Likes>().unwrap();
        let report = RelationCycleReport::capture(&world);
        let entry = report.entry(likes).unwrap();
        assert!(entry.has_cycle, "mutual Likes forms a 2-cycle");
        assert!(!entry.transitive && !entry.exclusive);
        assert!(!entry.is_suspicious(), "a mutual plain relation may cycle");
        assert!(report.has_any_cycle());
        assert!(!report.has_suspicious_cycle());
    }

    #[test]
    fn per_relation_is_sorted_by_relation_id() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<LocatedIn>(a, b);

        let report = RelationCycleReport::capture(&world);
        assert_eq!(report.relation_count(), 2);
        let ids: Vec<u32> = report.per_relation.iter().map(|e| e.relation.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }
}
