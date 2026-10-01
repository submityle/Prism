//! The retained [`Tree`]: a generational arena of [`Node`]s with keyed-child
//! reconciliation.

use alloc::vec::Vec;

use crate::arena::{Arena, NodeId};
use crate::node::Node;
use crate::reconcile::{diff_keyed, Diff, DiffOp};

/// A retained tree of nodes keyed by `K` carrying payload `T`.
///
/// The tree never diffs payloads itself; it manages identity and structure so
/// higher layers can patch component fields in place. [`reconcile_children`]
/// turns a declarative "these are the children now" statement into the minimal
/// create/move/remove operations against the live tree.
///
/// [`reconcile_children`]: Tree::reconcile_children
#[derive(Debug)]
pub struct Tree<K, T> {
    arena: Arena<Node<K, T>>,
}

impl<K, T> Default for Tree<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, T> Tree<K, T> {
    /// Creates an empty tree.
    #[inline]
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
        }
    }

    /// Number of live nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// Whether the tree has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Creates a detached node (no parent) and returns its id.
    pub fn create(&mut self, key: Option<K>, value: T) -> NodeId {
        self.arena.insert(Node::new(key, value))
    }

    /// Returns `true` if `id` is still live.
    pub fn contains(&self, id: NodeId) -> bool {
        self.arena.contains(id)
    }

    /// Borrows a node.
    pub fn node(&self, id: NodeId) -> Option<&Node<K, T>> {
        self.arena.get(id)
    }

    /// Borrows a node's payload.
    pub fn get(&self, id: NodeId) -> Option<&T> {
        self.arena.get(id).map(Node::value)
    }

    /// Mutably borrows a node's payload.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut T> {
        self.arena.get_mut(id).map(Node::value_mut)
    }

    /// The parent of `id`, if attached.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.arena.get(id).and_then(Node::parent)
    }

    /// The children of `id`, in order. Returns an empty slice for a missing
    /// node.
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        match self.arena.get(id) {
            Some(node) => node.children(),
            None => &[],
        }
    }

    /// Appends `child` to the end of `parent`'s child list, detaching it from a
    /// previous parent first. Returns `false` if either id is missing.
    pub fn append_child(&mut self, parent: NodeId, child: NodeId) -> bool {
        if !self.arena.contains(parent) || !self.arena.contains(child) {
            return false;
        }
        self.detach(child);
        if let Some(node) = self.arena.get_mut(child) {
            node.parent = Some(parent);
        }
        if let Some(node) = self.arena.get_mut(parent) {
            node.children.push(child);
        }
        true
    }

    /// Removes `child` from its parent's child list without destroying it.
    /// The node becomes detached (a root).
    pub fn detach(&mut self, child: NodeId) {
        let parent = self.arena.get(child).and_then(Node::parent);
        if let Some(parent) = parent
            && let Some(parent_node) = self.arena.get_mut(parent)
        {
            parent_node.children.retain(|&c| c != child);
        }
        if let Some(node) = self.arena.get_mut(child) {
            node.parent = None;
        }
    }

    /// Removes `id` and its entire subtree, returning how many nodes were
    /// removed.
    pub fn remove_subtree(&mut self, id: NodeId) -> usize {
        if !self.arena.contains(id) {
            return 0;
        }
        self.detach(id);
        self.remove_recursive(id)
    }

    fn remove_recursive(&mut self, id: NodeId) -> usize {
        // Take the children out first so we can recurse without aliasing.
        let children = match self.arena.get(id) {
            Some(node) => node.children.clone(),
            None => return 0,
        };
        let mut removed = 0;
        for child in children {
            removed += self.remove_recursive(child);
        }
        if self.arena.remove(id).is_some() {
            removed += 1;
        }
        removed
    }

    /// Depth-first pre-order traversal starting at `root`, invoking `visit`
    /// with `(id, depth)`.
    pub fn walk<F: FnMut(NodeId, usize)>(&self, root: NodeId, mut visit: F) {
        self.walk_inner(root, 0, &mut visit);
    }

    fn walk_inner<F: FnMut(NodeId, usize)>(&self, id: NodeId, depth: usize, visit: &mut F) {
        if self.arena.get(id).is_none() {
            return;
        }
        visit(id, depth);
        let children = self.children(id).to_vec();
        for child in children {
            self.walk_inner(child, depth + 1, visit);
        }
    }
}

impl<K: Ord + Clone, T> Tree<K, T> {
    /// Reconciles `parent`'s children against `new_keys`, reusing existing
    /// children whose key is unchanged, creating missing ones via `create`,
    /// removing dropped ones, and reordering with the minimum number of moves.
    ///
    /// Every current child of `parent` must carry a key; a keyless child is
    /// treated as removable. Returns the [`Diff`] that was applied, which
    /// exposes move/create counts for performance assertions.
    pub fn reconcile_children<F>(&mut self, parent: NodeId, new_keys: &[K], mut create: F) -> Diff
    where
        F: FnMut(&K) -> T,
    {
        let old_children: Vec<NodeId> = self.children(parent).to_vec();
        let old_keys: Vec<K> = old_children
            .iter()
            .filter_map(|&c| self.arena.get(c).and_then(|n| n.key.clone()))
            .collect();

        // Keyless children can't be matched; fall back to a wholesale rebuild
        // for `parent` to keep behaviour well-defined.
        if old_keys.len() != old_children.len() {
            for &child in &old_children {
                self.remove_subtree(child);
            }
            let diff = diff_keyed::<K>(&[], new_keys);
            self.apply_diff(parent, &[], new_keys, &diff, &mut create);
            return diff;
        }

        let diff = diff_keyed(&old_keys, new_keys);
        self.apply_diff(parent, &old_children, new_keys, &diff, &mut create);
        diff
    }

    fn apply_diff<F>(
        &mut self,
        parent: NodeId,
        old_children: &[NodeId],
        new_keys: &[K],
        diff: &Diff,
        create: &mut F,
    ) where
        F: FnMut(&K) -> T,
    {
        // Build the new ordered child list.
        let mut new_children: Vec<NodeId> = Vec::with_capacity(diff.ops.len());
        for op in &diff.ops {
            match *op {
                DiffOp::Keep { old_index } | DiffOp::Move { old_index } => {
                    new_children.push(old_children[old_index]);
                }
                DiffOp::Create { new_index } => {
                    let key = new_keys[new_index].clone();
                    let value = create(&key);
                    let id = self.create(Some(key), value);
                    if let Some(node) = self.arena.get_mut(id) {
                        node.parent = Some(parent);
                    }
                    new_children.push(id);
                }
            }
        }

        // Remove dropped nodes and their subtrees.
        for &old_index in &diff.removals {
            self.remove_subtree(old_children[old_index]);
        }

        // Commit the new child order.
        if let Some(node) = self.arena.get_mut(parent) {
            node.children = new_children;
        }
    }
}
