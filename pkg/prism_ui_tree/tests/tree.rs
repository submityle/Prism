//! Integration tests for the retained tree and keyed reconciler.

use prism_ui_tree::reconcile::{diff_keyed, DiffOp};
use prism_ui_tree::{Arena, Tree};

#[test]
fn arena_insert_get_remove_and_generations() {
    let mut arena: Arena<i32> = Arena::new();
    let a = arena.insert(10);
    let b = arena.insert(20);
    assert_eq!(arena.len(), 2);
    assert_eq!(arena.get(a), Some(&10));
    assert_eq!(arena.get(b), Some(&20));

    assert_eq!(arena.remove(a), Some(10));
    assert_eq!(arena.len(), 1);
    assert_eq!(arena.get(a), None);
    assert!(!arena.contains(a));

    // Reusing the freed slot yields a new generation; the stale handle stays
    // invalid.
    let c = arena.insert(30);
    assert_eq!(c.index(), a.index());
    assert_ne!(c.generation(), a.generation());
    assert_eq!(arena.get(c), Some(&30));
    assert_eq!(arena.get(a), None);
}

#[test]
fn diff_pure_append_only_creates() {
    let diff = diff_keyed(&[1, 2, 3], &[1, 2, 3, 4, 5]);
    assert_eq!(diff.create_count(), 2);
    assert_eq!(diff.move_count(), 0);
    assert!(diff.removals.is_empty());
    assert!(matches!(diff.ops[0], DiffOp::Keep { old_index: 0 }));
    assert!(matches!(diff.ops[3], DiffOp::Create { new_index: 3 }));
}

#[test]
fn diff_removal_reports_old_indices() {
    let diff = diff_keyed(&[1, 2, 3, 4], &[1, 3]);
    assert_eq!(diff.removals, vec![1, 3]);
    assert_eq!(diff.create_count(), 0);
    assert_eq!(diff.move_count(), 0);
}

#[test]
fn diff_minimises_moves_with_lis() {
    // Moving the first element to the end should move exactly one node, not
    // shuffle the whole list.
    let diff = diff_keyed(&[1, 2, 3, 4], &[2, 3, 4, 1]);
    assert_eq!(diff.move_count(), 1);
    assert_eq!(diff.create_count(), 0);
    assert!(diff.removals.is_empty());
    // Key 1 (old_index 0) is the one that moves.
    assert!(matches!(diff.ops[3], DiffOp::Move { old_index: 0 }));
}

#[test]
fn diff_full_reverse_moves_all_but_lis() {
    let diff = diff_keyed(&[1, 2, 3, 4], &[4, 3, 2, 1]);
    // A reversed list has an LIS of length 1, so three of four must move.
    assert_eq!(diff.move_count(), 3);
}

#[test]
fn reconcile_reuses_nodes_by_key() {
    let mut tree: Tree<u32, u32> = Tree::new();
    let root = tree.create(None, 0);

    tree.reconcile_children(root, &[1, 2, 3], |k| *k * 10);
    let before: Vec<_> = tree.children(root).to_vec();
    assert_eq!(before.len(), 3);
    // Node identities for the keys we keep.
    let id1 = before[0];
    let id2 = before[1];

    let diff = tree.reconcile_children(root, &[2, 1, 4], |k| *k * 10);
    let after: Vec<_> = tree.children(root).to_vec();
    assert_eq!(after.len(), 3);
    // Key 3 removed, key 4 created.
    assert_eq!(diff.create_count(), 1);
    assert_eq!(diff.removals.len(), 1);
    // Reused nodes keep their original NodeId.
    assert_eq!(after[0], id2);
    assert_eq!(after[1], id1);
    // Payload of reused node is untouched (still 10 for key 1).
    assert_eq!(tree.get(id1), Some(&10));
}

#[test]
fn reconcile_removes_subtrees_of_dropped_keys() {
    let mut tree: Tree<u32, u32> = Tree::new();
    let root = tree.create(None, 0);
    tree.reconcile_children(root, &[1, 2], |k| *k);

    // Give child with key 2 a grandchild.
    let child2 = tree.children(root)[1];
    let grandchild = tree.create(Some(99), 999);
    tree.append_child(child2, grandchild);
    assert_eq!(tree.len(), 4); // root + 2 children + grandchild

    // Drop key 2 — its subtree (child + grandchild) must go.
    tree.reconcile_children(root, &[1], |k| *k);
    assert_eq!(tree.children(root).len(), 1);
    assert!(!tree.contains(child2));
    assert!(!tree.contains(grandchild));
    assert_eq!(tree.len(), 2); // root + child 1
}

#[test]
fn walk_visits_pre_order_with_depth() {
    let mut tree: Tree<u32, &'static str> = Tree::new();
    let root = tree.create(None, "root");
    tree.reconcile_children(root, &[1, 2], |_| "child");
    let c1 = tree.children(root)[0];
    tree.reconcile_children(c1, &[10], |_| "grandchild");

    let mut visited = Vec::new();
    tree.walk(root, |id, depth| visited.push((id, depth)));
    assert_eq!(visited.len(), 4);
    assert_eq!(visited[0].1, 0); // root at depth 0
    assert_eq!(visited[1].1, 1); // first child at depth 1
    assert_eq!(visited[2].1, 2); // grandchild at depth 2
}

#[test]
fn detach_keeps_node_but_unlinks_parent() {
    let mut tree: Tree<u32, u32> = Tree::new();
    let root = tree.create(None, 0);
    let child = tree.create(Some(1), 1);
    tree.append_child(root, child);
    assert_eq!(tree.parent(child), Some(root));

    tree.detach(child);
    assert_eq!(tree.parent(child), None);
    assert!(tree.contains(child));
    assert_eq!(tree.children(root).len(), 0);
}
