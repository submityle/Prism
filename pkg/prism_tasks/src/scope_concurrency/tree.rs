//! Deterministic, `no_std`-friendly core for the structured-concurrency task
//! tree and cancellation propagation (design §24.3).
//!
//! This module owns the *decision* half of §24.3 and holds no threads, no
//! clock, and no allocation beyond the node arena. The shape of the scope tree
//! (who is a child of whom) and the exact set of nodes a cancellation
//! propagates to is a pure function of the recorded structure, so it is
//! directly unit-testable against a serial oracle. The thread-pool façade that
//! actually spawns and joins the borrowed tasks lives in the parent
//! [`scope_concurrency`](crate::scope_concurrency) module.
//!
//! # Model
//! A [`ScopeTree`] is an arena of [`Node`]s rooted at [`ScopeTree::ROOT`]. Each
//! node models one structured scope or one spawned task:
//!
//! - [`ScopeTree::spawn_child`] appends a `Pending` child under a parent,
//!   returning its stable [`NodeId`]; children keep spawn order.
//! - [`ScopeTree::set_running`] / [`ScopeTree::set_joined`] record a task's
//!   lifecycle as the façade runs and joins it.
//! - [`ScopeTree::cancel_subtree`] cooperatively cancels a node and every
//!   descendant that has not already finished, returning the newly cancelled
//!   nodes in deterministic pre-order (parent before children, children in
//!   spawn order).
//!
//! Cancellation is **cooperative**: a node that has already reached
//! [`NodeState::Joined`] is left untouched (its work is done and its state is
//! consistent), mirroring the real [`CancelToken`](crate::CancelToken) contract
//! where a finished task simply never observes the trip.

use alloc::vec::Vec;

/// Stable identifier of a node in a [`ScopeTree`]. The value is the node's
/// index in the arena; [`ScopeTree::ROOT`] is always `NodeId(0)`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub usize);

impl NodeId {
    /// The raw arena index backing this id.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Lifecycle state of a [`Node`] in the scope tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum NodeState {
    /// Spawned but not yet started on an executor.
    Pending,
    /// Currently executing on an executor.
    Running,
    /// Finished normally and joined by its parent scope.
    Joined,
    /// Cancelled before finishing; its subtree is being torn down.
    Cancelled,
}

impl NodeState {
    /// Whether this state is terminal (the node will not transition again on
    /// its own): [`NodeState::Joined`] or [`NodeState::Cancelled`].
    #[must_use]
    #[inline]
    pub const fn is_terminal(self) -> bool {
        matches!(self, NodeState::Joined | NodeState::Cancelled)
    }
}

/// One node of a [`ScopeTree`]: a structured scope or a spawned task.
#[derive(Clone, Debug)]
pub struct Node {
    /// Parent scope, or `None` for [`ScopeTree::ROOT`].
    parent: Option<NodeId>,
    /// Children in spawn order.
    children: Vec<NodeId>,
    /// Current lifecycle state.
    state: NodeState,
}

impl Node {
    /// This node's parent, or `None` for the root.
    #[must_use]
    #[inline]
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// This node's children, in spawn order.
    #[must_use]
    #[inline]
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    /// This node's current [`NodeState`].
    #[must_use]
    #[inline]
    pub fn state(&self) -> NodeState {
        self.state
    }
}

/// An arena-backed structured-concurrency task tree (design §24.3).
///
/// Build it by spawning children under existing nodes, drive node lifecycles
/// with [`ScopeTree::set_running`] / [`ScopeTree::set_joined`], and tear a
/// subtree down with [`ScopeTree::cancel_subtree`]. Every query is a pure
/// function of the recorded structure.
#[derive(Clone, Debug)]
pub struct ScopeTree {
    /// Arena of nodes; index `0` is the root.
    nodes: Vec<Node>,
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ScopeTree {
    /// The always-present root scope.
    pub const ROOT: NodeId = NodeId(0);

    /// Create a tree holding only the root scope (`Running`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: alloc::vec![Node {
                parent: None,
                children: Vec::new(),
                state: NodeState::Running,
            }],
        }
    }

    /// Number of nodes in the tree, including the root.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree holds only its root (no spawned children anywhere).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Borrow a node by id.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    /// Spawn a new `Pending` child under `parent`, returning its id.
    ///
    /// If `parent` is already [`NodeState::Cancelled`], the child is born
    /// `Cancelled` too, closing the register/cancel race exactly as
    /// [`CancelToken::child`](crate::CancelToken::child) does.
    ///
    /// # Panics
    /// Panics if `parent` is out of range for this tree.
    pub fn spawn_child(&mut self, parent: NodeId) -> NodeId {
        let born_cancelled = self.nodes[parent.index()].state == NodeState::Cancelled;
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            parent: Some(parent),
            children: Vec::new(),
            state: if born_cancelled {
                NodeState::Cancelled
            } else {
                NodeState::Pending
            },
        });
        self.nodes[parent.index()].children.push(id);
        id
    }

    /// Mark `id` as `Running`, unless it is already terminal (a cancelled or
    /// joined node keeps its terminal state). Returns the resulting state.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    pub fn set_running(&mut self, id: NodeId) -> NodeState {
        if !self.nodes[id.index()].state.is_terminal() {
            self.nodes[id.index()].state = NodeState::Running;
        }
        self.nodes[id.index()].state
    }

    /// Mark `id` as `Joined`, unless it is already [`NodeState::Cancelled`]
    /// (a cancelled node stays cancelled). Returns the resulting state.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    pub fn set_joined(&mut self, id: NodeId) -> NodeState {
        if self.nodes[id.index()].state != NodeState::Cancelled {
            self.nodes[id.index()].state = NodeState::Joined;
        }
        self.nodes[id.index()].state
    }

    /// Whether `id` is currently [`NodeState::Cancelled`].
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    #[inline]
    pub fn is_cancelled(&self, id: NodeId) -> bool {
        self.nodes[id.index()].state == NodeState::Cancelled
    }

    /// The state of `id`.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    #[inline]
    pub fn state(&self, id: NodeId) -> NodeState {
        self.nodes[id.index()].state
    }

    /// Depth of `id` below the root (`ROOT` has depth `0`).
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    pub fn depth(&self, id: NodeId) -> usize {
        let mut depth = 0;
        let mut cursor = self.nodes[id.index()].parent;
        while let Some(parent) = cursor {
            depth += 1;
            cursor = self.nodes[parent.index()].parent;
        }
        depth
    }

    /// Number of not-yet-terminal descendants of `id` (its outstanding
    /// subtree, excluding `id` itself). This is what a scope must still join.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    pub fn pending_descendants(&self, id: NodeId) -> usize {
        let mut count = 0;
        self.visit_preorder(id, &mut |node, visited| {
            if visited != id && !node.state.is_terminal() {
                count += 1;
            }
        });
        count
    }

    /// Collect the subtree rooted at `id` in deterministic pre-order (the node
    /// itself first, then each child's subtree in spawn order).
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    #[must_use]
    pub fn subtree_preorder(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.visit_preorder(id, &mut |_, visited| out.push(visited));
        out
    }

    /// Cooperatively cancel the subtree rooted at `id`: mark `id` and every
    /// descendant [`NodeState::Cancelled`], skipping any node already
    /// [`NodeState::Joined`] (finished work stays consistent) or already
    /// [`NodeState::Cancelled`] (idempotent).
    ///
    /// Returns the nodes whose state actually transitioned to `Cancelled` this
    /// call, in deterministic pre-order. Re-cancelling returns an empty list.
    ///
    /// # Panics
    /// Panics if `id` is out of range for this tree.
    pub fn cancel_subtree(&mut self, id: NodeId) -> Vec<NodeId> {
        let order = self.subtree_preorder(id);
        let mut newly = Vec::new();
        for node in order {
            let state = &mut self.nodes[node.index()].state;
            if *state != NodeState::Joined && *state != NodeState::Cancelled {
                *state = NodeState::Cancelled;
                newly.push(node);
            }
        }
        newly
    }

    /// Shared pre-order walk used by the public queries. The visitor receives
    /// each node and its id, parent before children, children in spawn order.
    fn visit_preorder(&self, root: NodeId, visit: &mut impl FnMut(&Node, NodeId)) {
        // Explicit stack keeps this iterative (no recursion depth limit) and
        // deterministic. Children are pushed in reverse so they pop in spawn
        // order.
        let mut stack = alloc::vec![root];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id.index()];
            visit(node, id);
            for &child in node.children.iter().rev() {
                stack.push(child);
            }
        }
    }
}
