//! §24.3 structured-concurrency and cancellation tests.
//!
//! Anti-vacuous contract: the deterministic scope-tree core
//! ([`ScopeTree`](crate::ScopeTree)) is checked against an *independent* serial
//! oracle (recursive pre-order descent, skipping already-terminal nodes) for
//! both the cancelled set and its traversal order. The
//! [`StructuredScope`](crate::StructuredScope) façade is then exercised on a
//! real multi-threaded [`TaskPool`](crate::TaskPool) to prove the join barrier,
//! cooperative-cancellation skip accounting, parent-to-child token cascade, and
//! born-cancelled behaviour.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::{CancelToken, NodeId, NodeState, ScopeTree, TaskPool};

// ----------------------------------------------------------------------------
// Independent serial oracle over a mirrored tree.
// ----------------------------------------------------------------------------

/// A mirror of the tree built with the same spawn calls, used to derive the
/// expected cancellation set/order without consulting [`ScopeTree`].
#[derive(Clone)]
struct Mirror {
    parent: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    state: Vec<NodeState>,
}

impl Mirror {
    fn new() -> Self {
        Self {
            parent: vec![None],
            children: vec![Vec::new()],
            state: vec![NodeState::Running],
        }
    }

    fn spawn_child(&mut self, parent: usize) -> usize {
        let born_cancelled = self.state[parent] == NodeState::Cancelled;
        let id = self.state.len();
        self.parent.push(Some(parent));
        self.children.push(Vec::new());
        self.state.push(if born_cancelled {
            NodeState::Cancelled
        } else {
            NodeState::Pending
        });
        self.children[parent].push(id);
        id
    }

    fn set_joined(&mut self, id: usize) {
        if self.state[id] != NodeState::Cancelled {
            self.state[id] = NodeState::Joined;
        }
    }

    /// Expected pre-order list of nodes that `cancel_subtree(root)` should turn
    /// from non-terminal to `Cancelled`, applied to this mirror as a side
    /// effect so repeated cancels behave like the real tree.
    fn cancel_subtree(&mut self, root: usize) -> Vec<usize> {
        let mut order = Vec::new();
        self.preorder(root, &mut order);
        let mut newly = Vec::new();
        for id in order {
            if self.state[id] != NodeState::Joined && self.state[id] != NodeState::Cancelled {
                self.state[id] = NodeState::Cancelled;
                newly.push(id);
            }
        }
        newly
    }

    fn preorder(&self, id: usize, out: &mut Vec<usize>) {
        out.push(id);
        for &child in &self.children[id] {
            self.preorder(child, out);
        }
    }

    fn pending_descendants(&self, id: usize) -> usize {
        let mut order = Vec::new();
        self.preorder(id, &mut order);
        order
            .into_iter()
            .filter(|&n| {
                n != id && !matches!(self.state[n], NodeState::Joined | NodeState::Cancelled)
            })
            .count()
    }
}

// ----------------------------------------------------------------------------
// Deterministic scope-tree core (oracle-checked).
// ----------------------------------------------------------------------------

#[test]
fn spawn_shapes_match_the_mirror() {
    let mut tree = ScopeTree::new();
    let mut mirror = Mirror::new();

    // Build: root -> {a, b}; a -> {a0, a1}; b -> {b0}.
    let a = tree.spawn_child(ScopeTree::ROOT);
    assert_eq!(a.index(), mirror.spawn_child(0));
    let b = tree.spawn_child(ScopeTree::ROOT);
    assert_eq!(b.index(), mirror.spawn_child(0));
    let a0 = tree.spawn_child(a);
    assert_eq!(a0.index(), mirror.spawn_child(a.index()));
    let a1 = tree.spawn_child(a);
    assert_eq!(a1.index(), mirror.spawn_child(a.index()));
    let b0 = tree.spawn_child(b);
    assert_eq!(b0.index(), mirror.spawn_child(b.index()));

    assert_eq!(tree.len(), 6);
    assert_eq!(tree.depth(a0), 2);
    assert_eq!(tree.depth(b), 1);
    assert_eq!(tree.pending_descendants(ScopeTree::ROOT), 5);
    assert_eq!(
        tree.pending_descendants(ScopeTree::ROOT),
        mirror.pending_descendants(0)
    );
    assert_eq!(tree.pending_descendants(a), 2);
}

#[test]
fn cancel_subtree_matches_oracle_order_and_set() {
    let mut tree = ScopeTree::new();
    let mut mirror = Mirror::new();
    let mut ids = vec![ScopeTree::ROOT];
    // Build a wider tree deterministically.
    for parent_ix in 0..4 {
        let parent = ids[parent_ix];
        for _ in 0..3 {
            let child = tree.spawn_child(parent);
            mirror.spawn_child(parent.index());
            ids.push(child);
        }
    }
    // Finish a couple of nodes so the cooperative skip is exercised.
    tree.set_joined(ids[2]);
    mirror.set_joined(ids[2].index());
    tree.set_joined(ids[5]);
    mirror.set_joined(ids[5].index());

    let got = tree.cancel_subtree(ScopeTree::ROOT);
    let want: Vec<NodeId> = mirror.cancel_subtree(0).into_iter().map(NodeId).collect();
    assert_eq!(got, want, "cancelled set + pre-order traversal");

    // Joined nodes stayed joined; everything else is cancelled now.
    assert_eq!(tree.state(ids[2]), NodeState::Joined);
    assert_eq!(tree.state(ids[5]), NodeState::Joined);
    assert!(tree.is_cancelled(ids[1]));

    // Re-cancelling is idempotent: nothing newly transitions.
    assert!(tree.cancel_subtree(ScopeTree::ROOT).is_empty());
}

#[test]
fn child_spawned_after_cancel_is_born_cancelled() {
    let mut tree = ScopeTree::new();
    let parent = tree.spawn_child(ScopeTree::ROOT);
    tree.cancel_subtree(parent);
    assert!(tree.is_cancelled(parent));
    // A child spawned under an already-cancelled parent is born cancelled.
    let late = tree.spawn_child(parent);
    assert_eq!(tree.state(late), NodeState::Cancelled);
}

#[test]
fn set_running_and_joined_respect_terminal_states() {
    let mut tree = ScopeTree::new();
    let n = tree.spawn_child(ScopeTree::ROOT);
    assert_eq!(tree.set_running(n), NodeState::Running);
    assert_eq!(tree.set_joined(n), NodeState::Joined);
    // A joined node is not reopened by set_running.
    assert_eq!(tree.set_running(n), NodeState::Joined);

    let c = tree.spawn_child(ScopeTree::ROOT);
    tree.cancel_subtree(c);
    // A cancelled node stays cancelled even if we try to join it.
    assert_eq!(tree.set_joined(c), NodeState::Cancelled);
}

#[test]
fn cancelling_one_branch_leaves_siblings_untouched() {
    let mut tree = ScopeTree::new();
    let a = tree.spawn_child(ScopeTree::ROOT);
    let b = tree.spawn_child(ScopeTree::ROOT);
    let a0 = tree.spawn_child(a);
    let b0 = tree.spawn_child(b);
    let cancelled = tree.cancel_subtree(a);
    assert_eq!(cancelled, vec![a, a0]);
    assert!(tree.is_cancelled(a));
    assert!(tree.is_cancelled(a0));
    assert!(!tree.is_cancelled(b));
    assert!(!tree.is_cancelled(b0));
}

// ----------------------------------------------------------------------------
// Façade over a real TaskPool.
// ----------------------------------------------------------------------------

#[test]
fn structured_scope_joins_every_task_and_counts_them() {
    let pool = TaskPool::with_threads(4);
    let root = CancelToken::new();
    let data = Arc::new(Mutex::new(vec![0u32; 64]));
    let outcome = {
        let data = Arc::clone(&data);
        pool.structured_scope(&root, |s| {
            for i in 0..64 {
                let data = Arc::clone(&data);
                s.spawn(move |token| {
                    assert!(!token.is_cancelled());
                    data.lock().unwrap()[i] = i as u32 + 1;
                });
            }
            "done"
        })
    };
    assert_eq!(outcome.value, "done");
    assert_eq!(outcome.spawned, 64);
    assert_eq!(outcome.completed, 64);
    assert_eq!(outcome.skipped, 0);
    assert!(!outcome.cancelled);
    let data = data.lock().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, i as u32 + 1);
    }
}

#[test]
fn pre_cancelled_scope_skips_every_task() {
    let pool = TaskPool::with_threads(4);
    let root = CancelToken::new();
    root.cancel(); // ancestor already cancelled: the child scope is born cancelled.
    let ran = Arc::new(AtomicUsize::new(0));
    let outcome = {
        let ran = Arc::clone(&ran);
        pool.structured_scope(&root, |s| {
            assert!(s.is_cancelled());
            for _ in 0..32 {
                let ran = Arc::clone(&ran);
                s.spawn(move |_| {
                    ran.fetch_add(1, Ordering::Relaxed);
                });
            }
        })
    };
    assert_eq!(outcome.spawned, 32);
    assert_eq!(outcome.completed, 0);
    assert_eq!(outcome.skipped, 32);
    assert!(outcome.cancelled);
    assert_eq!(ran.load(Ordering::Relaxed), 0);
}

#[test]
fn cancelling_the_scope_cascades_to_child_tokens() {
    let pool = TaskPool::with_threads(2);
    let root = CancelToken::new();
    pool.structured_scope(&root, |s| {
        let child = s.child_token();
        let grandchild = child.child();
        assert!(!child.is_cancelled());
        assert!(!grandchild.is_cancelled());
        s.cancel();
        // Cancelling the scope trips its whole derived subtree.
        assert!(s.is_cancelled());
        assert!(child.is_cancelled());
        assert!(grandchild.is_cancelled());
    });
}

#[test]
fn parent_token_cancellation_reaches_the_scope() {
    let pool = TaskPool::with_threads(2);
    let parent = CancelToken::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let ran_body = Arc::clone(&ran);
    let outcome = pool.structured_scope(&parent, |s| {
        // Cancelling the *parent* must reach the scope's own token.
        parent.cancel();
        assert!(s.is_cancelled());
        // The scope token is already tripped, so the task is skipped entirely
        // (cooperative cancellation: it never starts) rather than run and bail.
        s.spawn(move |_token| {
            ran_body.fetch_add(1, Ordering::Relaxed);
        });
    });
    assert!(outcome.cancelled);
    assert_eq!(outcome.spawned, 1);
    assert_eq!(outcome.completed, 0);
    assert_eq!(outcome.skipped, 1);
    assert_eq!(ran.load(Ordering::Relaxed), 0);
}

#[test]
fn single_threaded_fallback_runs_structured_scope_inline() {
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    let root = CancelToken::new();
    let sum = Arc::new(AtomicUsize::new(0));
    let outcome = {
        let sum = Arc::clone(&sum);
        pool.structured_scope(&root, |s| {
            for i in 1..=10 {
                let sum = Arc::clone(&sum);
                s.spawn(move |_| {
                    sum.fetch_add(i, Ordering::Relaxed);
                });
            }
            42
        })
    };
    assert_eq!(outcome.value, 42);
    assert_eq!(outcome.completed, 10);
    assert_eq!(sum.load(Ordering::Relaxed), 55);
}
