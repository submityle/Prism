//! M2 tests: incremental dirty-subtree propagation equals the full pass,
//! static scenes recompute nothing, and localized edits touch only the right
//! subtree.

use crate::dirty::DirtyStats;
use crate::hierarchy::NodeId;
use crate::{Transform, TransformGraph};
use prism_math::{Affine3, Quat, Vec3, vec3};

// ---- helpers ---------------------------------------------------------------

fn v_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

fn affine_approx(a: Affine3, b: Affine3, eps: f32) -> bool {
    v_approx(a.matrix3.x_axis, b.matrix3.x_axis, eps)
        && v_approx(a.matrix3.y_axis, b.matrix3.y_axis, eps)
        && v_approx(a.matrix3.z_axis, b.matrix3.z_axis, eps)
        && v_approx(a.translation, b.translation, eps)
}

/// Assert that an incrementally-propagated graph matches a full pass on an
/// independent clone of the same locals/hierarchy, node for node.
fn assert_matches_full_pass(g: &TransformGraph) {
    let mut reference = g.clone();
    reference.propagate();
    for i in 0..g.len() {
        let node = NodeId::new(i as u32);
        assert!(
            affine_approx(g.global(node).affine(), reference.global(node).affine(), 1e-4),
            "node {i} incremental global disagrees with full pass",
        );
    }
}

/// Tiny deterministic PRNG (xorshift64*) so the differential test is
/// reproducible without pulling in a dev-dependency.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Build a deterministic multi-level tree and return its node ids. Shape:
/// one root, three mid nodes, each with two or three leaves (12 nodes total).
fn build_sample_tree() -> (TransformGraph, Vec<NodeId>) {
    let mut g = TransformGraph::new();
    let mut ids = Vec::new();

    let root = g.spawn_root(Transform::from_xyz(1.0, 2.0, 3.0));
    ids.push(root);

    let m0 = g.spawn_child(root, Transform::from_xyz(1.0, 0.0, 0.0));
    let m1 = g.spawn_child(root, Transform::from_xyz(0.0, 1.0, 0.0));
    let m2 = g.spawn_child(root, Transform::from_xyz(0.0, 0.0, 1.0));
    ids.extend([m0, m1, m2]);

    ids.push(g.spawn_child(m0, Transform::from_xyz(0.5, 0.0, 0.0)));
    ids.push(g.spawn_child(m0, Transform::from_xyz(0.0, 0.5, 0.0)));
    ids.push(g.spawn_child(m1, Transform::from_xyz(0.0, 0.0, 0.5)));
    ids.push(g.spawn_child(m1, Transform::from_xyz(0.25, 0.25, 0.0)));
    ids.push(g.spawn_child(m2, Transform::from_xyz(0.0, 0.25, 0.25)));
    ids.push(g.spawn_child(m2, Transform::from_xyz(0.25, 0.0, 0.25)));
    ids.push(g.spawn_child(m2, Transform::from_xyz(0.1, 0.1, 0.1)));

    (g, ids)
}

/// Count the nodes in `node`'s subtree (inclusive).
fn subtree_size(g: &TransformGraph, node: NodeId) -> usize {
    let mut count = 0;
    let mut stack = alloc_vec(node);
    while let Some(n) = stack.pop() {
        count += 1;
        for &c in g.hierarchy().children(n) {
            stack.push(c);
        }
    }
    count
}

fn alloc_vec(seed: NodeId) -> Vec<NodeId> {
    let mut v = Vec::new();
    v.push(seed);
    v
}

// ---- static scene: zero recomputation --------------------------------------

#[test]
fn static_scene_recomputes_nothing() {
    let (mut g, _ids) = build_sample_tree();

    // First incremental pass computes every freshly-spawned node once.
    let first = g.propagate_incremental();
    assert_eq!(first.recomputed, g.len(), "first pass must compute every node");

    // Nothing changed since: a static scene must cost zero world-matrix work.
    let second = g.propagate_incremental();
    assert_eq!(second, DirtyStats { recomputed: 0, dirty_roots: 0 });

    // Repeat to be sure it stays at zero.
    let third = g.propagate_incremental();
    assert_eq!(third.recomputed, 0);
    assert_eq!(third.dirty_roots, 0);
}

// ---- single-leaf edit: only that leaf recomputes ---------------------------

#[test]
fn single_leaf_edit_recomputes_only_that_leaf() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    // Pick a leaf (last spawned node is a leaf under m2).
    let leaf = *ids.last().unwrap();
    assert!(g.hierarchy().children(leaf).is_empty(), "test node must be a leaf");

    let before_sibling = g.global(ids[1]).affine(); // an untouched mid node

    g.set_local(leaf, Transform::from_xyz(9.0, 9.0, 9.0));
    let stats = g.propagate_incremental();

    assert_eq!(stats.dirty_roots, 1);
    assert_eq!(stats.recomputed, 1, "a leaf's subtree is just the leaf");

    // Untouched subtree's cached global is unchanged.
    assert!(affine_approx(g.global(ids[1]).affine(), before_sibling, 0.0));
    // And the result matches a full pass.
    assert_matches_full_pass(&g);
}

// ---- mid-tree edit: only that subtree recomputes ---------------------------

#[test]
fn mid_tree_edit_recomputes_only_that_subtree() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    // ids[3] == m2, which has three leaf children -> subtree size 4.
    let mid = ids[3];
    let expected = subtree_size(&g, mid);
    assert_eq!(expected, 4);

    // Capture a global in a *different* subtree; it must not move.
    let other = ids[1]; // m0
    let other_before = g.global(other).affine();
    let other_leaf_before = g.global(ids[4]).affine(); // a leaf under m0

    g.set_local(mid, Transform::from_xyz(0.0, 0.0, 5.0).with_rotation(Quat::from_rotation_y(0.3)));
    let stats = g.propagate_incremental();

    assert_eq!(stats.dirty_roots, 1);
    assert_eq!(stats.recomputed, expected, "mid edit recomputes its subtree only");

    assert!(affine_approx(g.global(other).affine(), other_before, 0.0));
    assert!(affine_approx(g.global(ids[4]).affine(), other_leaf_before, 0.0));
    assert_matches_full_pass(&g);
}

// ---- two disjoint edits collapse to two dirty roots ------------------------

#[test]
fn disjoint_edits_make_two_dirty_roots() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    let a = ids[1]; // m0 subtree size 3
    let b = ids[3]; // m2 subtree size 4
    let sa = subtree_size(&g, a);
    let sb = subtree_size(&g, b);

    g.set_local(a, Transform::from_xyz(2.0, 0.0, 0.0));
    g.set_local(b, Transform::from_xyz(0.0, 0.0, 2.0));
    let stats = g.propagate_incremental();

    assert_eq!(stats.dirty_roots, 2);
    assert_eq!(stats.recomputed, sa + sb);
    assert_matches_full_pass(&g);
}

// ---- nested edits collapse to a single dirty root --------------------------

#[test]
fn nested_edits_collapse_to_ancestor_root() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    let ancestor = ids[0]; // root -> whole forest
    let descendant = *ids.last().unwrap();

    g.set_local(ancestor, Transform::from_xyz(5.0, 5.0, 5.0));
    g.set_local(descendant, Transform::from_xyz(1.0, 1.0, 1.0));
    let stats = g.propagate_incremental();

    // The descendant is covered by the ancestor's sweep: a single dirty root
    // that recomputes every node.
    assert_eq!(stats.dirty_roots, 1);
    assert_eq!(stats.recomputed, g.len());
    assert_matches_full_pass(&g);
}

// ---- reparent marks the moved subtree dirty --------------------------------

#[test]
fn reparent_marks_moved_subtree_and_yields_correct_world() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    let moved = ids[3]; // m2 with three leaves
    let new_parent = ids[1]; // m0
    let moved_subtree = subtree_size(&g, moved);

    g.reparent(moved, Some(new_parent)).unwrap();
    let stats = g.propagate_incremental();

    assert_eq!(g.hierarchy().parent(moved), Some(new_parent));
    // reparent marks the moved node; its whole subtree is recomputed.
    assert_eq!(stats.dirty_roots, 1);
    assert_eq!(stats.recomputed, moved_subtree);
    // World poses are correct versus a full pass.
    assert_matches_full_pass(&g);
}

#[test]
fn reparent_to_root_recomputes_moved_subtree() {
    let (mut g, ids) = build_sample_tree();
    g.propagate_incremental();

    let moved = ids[1]; // m0
    let moved_subtree = subtree_size(&g, moved);

    g.reparent(moved, None).unwrap();
    let stats = g.propagate_incremental();

    assert_eq!(g.hierarchy().parent(moved), None);
    assert_eq!(stats.dirty_roots, 1);
    assert_eq!(stats.recomputed, moved_subtree);
    assert_matches_full_pass(&g);
}

// ---- differential: incremental == full over random edit sequences ----------

#[test]
fn incremental_matches_full_over_random_edits() {
    // Several independent trials, each a long randomized edit sequence with a
    // distinct, deterministic seed.
    for trial in 0..16u64 {
        let (mut g, ids) = build_sample_tree();
        let mut rng = Rng::new(0xC0FFEE_1234_5678 ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // Prime both the incremental graph and the invariant that globals are
        // valid before incremental passes.
        g.propagate_incremental();

        for _ in 0..40 {
            // A batch of random local edits.
            let edits = 1 + rng.below(4);
            for _ in 0..edits {
                let node = ids[rng.below(ids.len())];
                // Compose three axis rotations so the quaternion is always
                // well-formed (no degenerate zero-length axis).
                let rotation = Quat::from_rotation_z(rng.range(-3.0, 3.0))
                    * Quat::from_rotation_y(rng.range(-3.0, 3.0))
                    * Quat::from_rotation_x(rng.range(-3.0, 3.0));
                let t = Transform {
                    translation: vec3(
                        rng.range(-5.0, 5.0),
                        rng.range(-5.0, 5.0),
                        rng.range(-5.0, 5.0),
                    ),
                    rotation,
                    scale: vec3(
                        rng.range(0.5, 2.0),
                        rng.range(0.5, 2.0),
                        rng.range(0.5, 2.0),
                    ),
                };
                g.set_local(node, t);
            }

            // Occasionally reparent a node to a legal target (no cycle).
            if rng.below(3) == 0 {
                let child = ids[rng.below(ids.len())];
                // Try a handful of candidate parents until one is legal.
                for _ in 0..4 {
                    let target = if rng.below(5) == 0 {
                        None
                    } else {
                        Some(ids[rng.below(ids.len())])
                    };
                    if g.reparent(child, target).is_ok() {
                        break;
                    }
                }
            }

            g.propagate_incremental();
            assert_matches_full_pass(&g);
        }
    }
}
