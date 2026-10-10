//! M6 statistics tests: counts are real (static frames visit nothing, a
//! localized edit visits exactly its subtree), depth levels equal the longest
//! chain, and the parallel grain equals the dirty-root count.

use crate::hierarchy::Hierarchy;
use crate::stats::{depth_levels, PropagationStats};
use crate::{Transform, TransformGraph};

/// Build root(0) -> child(1) -> grandchild(2), plus root(3) -> child(4).
fn two_tree_graph() -> TransformGraph {
    let mut g = TransformGraph::new();
    let r0 = g.spawn_root(Transform::IDENTITY);
    let c1 = g.spawn_child(r0, Transform::from_xyz(1.0, 0.0, 0.0));
    let _g2 = g.spawn_child(c1, Transform::from_xyz(1.0, 0.0, 0.0));
    let r3 = g.spawn_root(Transform::from_xyz(5.0, 0.0, 0.0));
    let _c4 = g.spawn_child(r3, Transform::from_xyz(1.0, 0.0, 0.0));
    g
}

#[test]
fn first_pass_visits_every_spawned_node() {
    let mut g = two_tree_graph();
    let s = g.propagate_incremental_stats();
    assert_eq!(s.nodes_total, 5);
    assert_eq!(s.nodes_visited, 5, "every freshly spawned node is dirty");
    assert_eq!(s.dirty_skipped, 0);
    // Minimal dirty roots are the two forest roots.
    assert_eq!(s.dirty_roots, 2);
    assert_eq!(s.parallel_chunks, 2);
    // Longest chain root->child->grandchild has three levels.
    assert_eq!(s.levels, 3);
    assert_eq!(s.elapsed_nanos, 0);
}

#[test]
fn static_frame_visits_nothing() {
    let mut g = two_tree_graph();
    g.propagate_incremental_stats();
    let s = g.propagate_incremental_stats();
    assert_eq!(s.nodes_total, 5);
    assert_eq!(s.nodes_visited, 0);
    assert_eq!(s.dirty_skipped, 5);
    assert_eq!(s.dirty_roots, 0);
    assert_eq!(s.parallel_chunks, 0);
    assert_eq!(
        s.levels, 3,
        "topology still reports its depth on a static frame"
    );
}

#[test]
fn localized_edit_visits_only_its_subtree() {
    let mut g = two_tree_graph();
    g.propagate_incremental_stats();

    // Edit node 1 (the middle of the first chain): subtree {1, 2} recomputes.
    let node1 = crate::hierarchy::NodeId::new(1);
    g.set_local(node1, Transform::from_xyz(1.0, 9.0, 0.0));
    let s = g.propagate_incremental_stats();
    assert_eq!(s.nodes_visited, 2);
    assert_eq!(s.dirty_skipped, 3);
    assert_eq!(s.dirty_roots, 1);
    assert_eq!(s.parallel_chunks, 1);
}

#[test]
fn depth_levels_counts_the_longest_chain() {
    // Empty forest -> 0 levels.
    assert_eq!(depth_levels(&Hierarchy::new()), 0);

    // A single root is one level.
    let mut h = Hierarchy::new();
    let r = h.spawn_root();
    assert_eq!(depth_levels(&h), 1);

    // Extend to a depth-3 chain.
    let c = h.spawn_child(r);
    h.spawn_child(c);
    assert_eq!(depth_levels(&h), 3);
}

#[test]
fn with_elapsed_nanos_fills_the_timing_hook() {
    let s = PropagationStats::default().with_elapsed_nanos(12_345);
    assert_eq!(s.elapsed_nanos, 12_345);
}
