//! Transitive-closure amplification diagnostics (design §11 / §16.6).
//!
//! Design §11 lists *transitive* relations (e.g. `LocatedIn`: `a` in `b` and
//! `b` in `c` imply `a` in `c`) and calls out "传递关系闭包、关系索引缓存" as a
//! performance concern: a transitive query expands a source's *direct* edges
//! into its full reachable set, and the gap between the two is exactly what a
//! closure cache would save.
//!
//! [`relation_topology`](super::relation_topology) already reports the *longest
//! directed chain* (`max_depth`) per relation, which bounds the worst-case
//! walk length. But chain depth is not closure *size*: a wide, shallow DAG has
//! tiny depth yet a large reachable set, and the ratio of closure size to
//! direct edges — the **closure amplification** — is what quantifies the cost
//! of recomputing transitive queries without a cache. This module supplies
//! that lens, restricted to the relation kinds actually registered as
//! [`transitive`](crate::relation::RelationKind::transitive).
//!
//! For every transitive kind it aggregates, over the sources that hold edges,
//! the total direct out-edges, the total transitive-closure size
//! ([`transitive_targets`](crate::relation::RelationIndex::transitive_targets)),
//! and the single worst source. Capture is read-only and never mutates
//! simulation state; the walk is cycle-safe (design §11).

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::world::World;

/// Closure-amplification aggregate for one transitive relation kind
/// (design §11).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TransitiveClosureEntry {
    /// The transitive relation marker component.
    pub relation: ComponentId,
    /// Distinct source entities that hold at least one outgoing edge.
    pub source_count: usize,
    /// Total *direct* out-edges summed over every source.
    pub direct_edges: usize,
    /// Total transitive-closure size summed over every source — the number of
    /// `(source, reachable)` pairs a transitive query would enumerate.
    pub closure_total: usize,
    /// Largest single-source closure size.
    pub max_closure: usize,
    /// The source whose closure is [`max_closure`](Self::max_closure); ties
    /// resolve to the lowest [`Entity::to_bits`]. `None` when the kind holds no
    /// edges.
    pub worst_source: Option<Entity>,
}

impl TransitiveClosureEntry {
    /// Reachable-but-not-direct pairs: `closure_total - direct_edges`
    /// (saturating). This is the extra reachability a closure cache would
    /// serve beyond the stored edges.
    #[inline]
    pub fn indirect_total(&self) -> usize {
        self.closure_total.saturating_sub(self.direct_edges)
    }

    /// Closure amplification in per-mille: `closure_total * 1000 /
    /// direct_edges`. Returns `0` when there are no direct edges. A value of
    /// `1000` means closure equals direct edges (no amplification); larger
    /// values mean transitive reach exceeds the stored edges.
    #[inline]
    pub fn amplification_permille(&self) -> u64 {
        if self.direct_edges == 0 {
            0
        } else {
            self.closure_total as u64 * 1000 / self.direct_edges as u64
        }
    }

    /// Whether the transitive closure reaches beyond the direct edges.
    #[inline]
    pub fn is_amplified(&self) -> bool {
        self.closure_total > self.direct_edges
    }
}

/// Read-only transitive-closure amplification census over every registered
/// transitive relation kind (design §11 / §16.6).
#[derive(Clone, Debug, Default)]
pub struct RelationClosureReport {
    /// Per-kind aggregates, sorted ascending by [`ComponentId`].
    entries: Vec<TransitiveClosureEntry>,
}

impl RelationClosureReport {
    /// Capture a closure census from `world`. Read-only; only relation kinds
    /// registered as [`transitive`](crate::relation::RelationKind::transitive)
    /// are analysed. Entries are sorted deterministically by [`ComponentId`]
    /// (design §14).
    pub fn from_world(world: &World) -> Self {
        let relations = world.relations();
        let index = relations.index();

        let mut entries: Vec<TransitiveClosureEntry> = Vec::new();
        for (relation, kind) in relations.iter_kinds() {
            if !kind.transitive {
                continue;
            }

            // Distinct sources holding an outgoing edge of this relation.
            let mut sources: Vec<Entity> = index
                .iter_edges()
                .filter(|(r, _, _)| *r == relation)
                .map(|(_, source, _)| source)
                .collect();
            sources.sort_unstable_by_key(|entity| entity.to_bits());
            sources.dedup();

            let mut direct_edges = 0usize;
            let mut closure_total = 0usize;
            let mut max_closure = 0usize;
            let mut worst_source: Option<Entity> = None;
            for &source in &sources {
                let direct = index.targets(relation, source).len();
                let closure = index.transitive_targets(relation, source).len();
                direct_edges += direct;
                closure_total += closure;
                // `sources` is sorted ascending, so a strict `>` keeps the
                // lowest-bits source on ties.
                if closure > max_closure {
                    max_closure = closure;
                    worst_source = Some(source);
                }
            }

            entries.push(TransitiveClosureEntry {
                relation,
                source_count: sources.len(),
                direct_edges,
                closure_total,
                max_closure,
                worst_source,
            });
        }
        entries.sort_unstable_by_key(|entry| entry.relation.index());

        Self { entries }
    }

    /// The per-kind aggregates, sorted ascending by [`ComponentId`].
    #[inline]
    pub fn entries(&self) -> &[TransitiveClosureEntry] {
        &self.entries
    }

    /// Whether no transitive relation kind is registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of registered transitive relation kinds.
    #[inline]
    pub fn transitive_kind_count(&self) -> usize {
        self.entries.len()
    }

    /// Total direct out-edges across every transitive kind.
    #[inline]
    pub fn total_direct_edges(&self) -> usize {
        self.entries.iter().map(|e| e.direct_edges).sum()
    }

    /// Total transitive-closure size across every transitive kind.
    #[inline]
    pub fn total_closure(&self) -> usize {
        self.entries.iter().map(|e| e.closure_total).sum()
    }

    /// Look up the aggregate for a relation kind via binary search over the
    /// [`ComponentId`]-sorted list.
    pub fn entry(&self, relation: ComponentId) -> Option<&TransitiveClosureEntry> {
        let key = relation.index();
        self.entries
            .binary_search_by_key(&key, |entry| entry.relation.index())
            .ok()
            .map(|idx| &self.entries[idx])
    }

    /// The transitive kind with the largest closure amplification. Ties resolve
    /// to the lowest [`ComponentId`], so the result is deterministic. Returns
    /// `None` when no transitive kind is registered.
    pub fn most_amplified(&self) -> Option<&TransitiveClosureEntry> {
        let mut best: Option<&TransitiveClosureEntry> = None;
        for entry in &self.entries {
            match best {
                // `entries` is sorted ascending by id, so a strict `>` keeps
                // the lowest-id entry on ties.
                Some(current)
                    if entry.amplification_permille() <= current.amplification_permille() => {}
                _ => best = Some(entry),
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::relation::RelationKind;
    use crate::world::World;

    /// `LocatedIn` — transitive containment relation.
    #[derive(PartialEq)]
    struct LocatedIn;
    impl Component for LocatedIn {}

    /// `PartOf` — a second transitive relation for multi-kind tests.
    #[derive(PartialEq)]
    struct PartOf;
    impl Component for PartOf {}

    /// `ChildOf` — a non-transitive relation that must be ignored.
    #[derive(PartialEq)]
    struct ChildOf;
    impl Component for ChildOf {}

    fn id_of<R: Component>(world: &World) -> ComponentId {
        world.components().id_of::<R>().unwrap()
    }

    #[test]
    fn empty_world_has_no_entries() {
        let world = World::new();
        let report = RelationClosureReport::from_world(&world);
        assert!(report.is_empty());
        assert_eq!(report.transitive_kind_count(), 0);
        assert_eq!(report.most_amplified(), None);
    }

    #[test]
    fn non_transitive_relation_is_ignored() {
        let mut world = World::new();
        world.register_relation::<ChildOf>(RelationKind::new().with_exclusive(true));
        let a = world.spawn(());
        let b = world.spawn(());
        world.add_relation::<ChildOf>(a, b);

        let report = RelationClosureReport::from_world(&world);
        assert!(report.is_empty());
    }

    #[test]
    fn registered_transitive_kind_without_edges_appears_with_zeros() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        let report = RelationClosureReport::from_world(&world);
        assert_eq!(report.transitive_kind_count(), 1);
        let entry = report.entry(id_of::<LocatedIn>(&world)).unwrap();
        assert_eq!(entry.source_count, 0);
        assert_eq!(entry.direct_edges, 0);
        assert_eq!(entry.closure_total, 0);
        assert_eq!(entry.max_closure, 0);
        assert_eq!(entry.worst_source, None);
        assert_eq!(entry.amplification_permille(), 0);
        assert!(!entry.is_amplified());
    }

    #[test]
    fn chain_closure_expands_beyond_direct_edges() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        // a -> b -> c: edges a->b, b->c.
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<LocatedIn>(a, b);
        world.add_relation::<LocatedIn>(b, c);

        let report = RelationClosureReport::from_world(&world);
        let entry = report.entry(id_of::<LocatedIn>(&world)).unwrap();
        // Sources are a and b (c has no outgoing edge).
        assert_eq!(entry.source_count, 2);
        assert_eq!(entry.direct_edges, 2);
        // closure(a) = {b, c} = 2, closure(b) = {c} = 1 => total 3.
        assert_eq!(entry.closure_total, 3);
        assert_eq!(entry.max_closure, 2);
        assert_eq!(entry.worst_source, Some(a));
        assert_eq!(entry.indirect_total(), 1);
        // 3 * 1000 / 2 = 1500.
        assert_eq!(entry.amplification_permille(), 1500);
        assert!(entry.is_amplified());
    }

    #[test]
    fn flat_relation_has_no_amplification() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        // A one-level star: hub -> {x, y, z}. Closure == direct edges.
        let hub = world.spawn(());
        let x = world.spawn(());
        let y = world.spawn(());
        let z = world.spawn(());
        world.add_relation::<LocatedIn>(hub, x);
        world.add_relation::<LocatedIn>(hub, y);
        world.add_relation::<LocatedIn>(hub, z);

        let report = RelationClosureReport::from_world(&world);
        let entry = report.entry(id_of::<LocatedIn>(&world)).unwrap();
        assert_eq!(entry.direct_edges, 3);
        assert_eq!(entry.closure_total, 3);
        assert_eq!(entry.amplification_permille(), 1000);
        assert!(!entry.is_amplified());
        assert_eq!(entry.indirect_total(), 0);
    }

    #[test]
    fn cyclic_closure_terminates_and_is_finite() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        // a -> b -> a cycle plus a -> c.
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<LocatedIn>(a, b);
        world.add_relation::<LocatedIn>(b, a);
        world.add_relation::<LocatedIn>(a, c);

        let report = RelationClosureReport::from_world(&world);
        let entry = report.entry(id_of::<LocatedIn>(&world)).unwrap();
        // closure(a) excludes a itself: {b, c} = 2; closure(b) = {a, c} = 2.
        assert_eq!(entry.source_count, 2);
        assert_eq!(entry.max_closure, 2);
        assert!(entry.closure_total >= entry.direct_edges);
    }

    #[test]
    fn multiple_transitive_kinds_sorted_and_most_amplified() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        world.register_relation::<PartOf>(RelationKind::new().with_transitive(true));
        // LocatedIn: deep chain a->b->c->d (amplified).
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let d = world.spawn(());
        world.add_relation::<LocatedIn>(a, b);
        world.add_relation::<LocatedIn>(b, c);
        world.add_relation::<LocatedIn>(c, d);
        // PartOf: a single flat edge (no amplification).
        let p = world.spawn(());
        let q = world.spawn(());
        world.add_relation::<PartOf>(p, q);

        let report = RelationClosureReport::from_world(&world);
        assert_eq!(report.transitive_kind_count(), 2);
        let ids: Vec<u32> = report.entries().iter().map(|e| e.relation.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);

        let worst = report.most_amplified().unwrap();
        assert_eq!(worst.relation, id_of::<LocatedIn>(&world));
        assert!(worst.amplification_permille() > 1000);
    }

    #[test]
    fn worst_source_breaks_ties_on_lowest_bits() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        // Two independent single edges => each source closure == 1 (tie).
        let s1 = world.spawn(());
        let t1 = world.spawn(());
        let s2 = world.spawn(());
        let t2 = world.spawn(());
        world.add_relation::<LocatedIn>(s1, t1);
        world.add_relation::<LocatedIn>(s2, t2);

        let report = RelationClosureReport::from_world(&world);
        let entry = report.entry(id_of::<LocatedIn>(&world)).unwrap();
        assert_eq!(entry.max_closure, 1);
        let lowest = if s1.to_bits() <= s2.to_bits() { s1 } else { s2 };
        assert_eq!(entry.worst_source, Some(lowest));
    }

    #[test]
    fn report_totals_sum_across_kinds() {
        let mut world = World::new();
        world.register_relation::<LocatedIn>(RelationKind::new().with_transitive(true));
        world.register_relation::<PartOf>(RelationKind::new().with_transitive(true));
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        world.add_relation::<LocatedIn>(a, b);
        world.add_relation::<PartOf>(b, c);

        let report = RelationClosureReport::from_world(&world);
        assert_eq!(report.total_direct_edges(), 2);
        assert_eq!(report.total_closure(), 2);
    }
}
