//! Index-based scene hierarchy: the self-contained computational core that M1
//! propagation runs on.
//!
//! This module intentionally knows nothing about an ECS. It stores nodes in a
//! dense arena keyed by [`NodeId`], where every node has at most one parent and
//! an ordered list of children. The ECS-relation wiring (deriving this graph
//! from `prism_ecs` `ChildOf` relations) is deferred to **M6**; keeping the
//! algebra and traversal here lets the propagation logic be developed, tested,
//! and benchmarked without an ECS dependency.
//!
//! The two guarantees callers rely on are:
//! - **Acyclicity**: [`Hierarchy::set_parent`] rejects edges that would form a
//!   cycle, and [`Hierarchy::from_parents`] / [`Hierarchy::validate`] reject
//!   cyclic input. A valid hierarchy is always a forest.
//! - **Stable parent-before-child order**: [`Hierarchy::compute_order`] returns
//!   every node with each parent strictly before its children, breaking ties by
//!   ascending [`NodeId`] so the order is deterministic across runs.

use alloc::vec::Vec;

/// Stable identifier for a node inside a [`Hierarchy`].
///
/// Ids are dense indices assigned in creation order; the first spawned node is
/// `NodeId(0)`. Use [`NodeId::index`] to index parallel component arrays such
/// as the local/world transform buffers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u32);

impl NodeId {
    /// Build an id from a raw dense index. Primarily for [`Hierarchy::from_parents`];
    /// ids are only meaningful against the hierarchy they index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The backing array index for this id.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Build an id from a backing array index.
    #[inline]
    const fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Error returned by fallible [`Hierarchy`] operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HierarchyError {
    /// A referenced [`NodeId`] is out of bounds for this hierarchy.
    InvalidNode,
    /// A parent index referenced during construction does not exist.
    InvalidParent,
    /// The requested edge would introduce a cycle (a node cannot be its own
    /// ancestor), so the hierarchy would no longer be a forest.
    Cycle,
    /// A component buffer length did not match the number of nodes.
    LengthMismatch,
}

/// A dense forest of nodes: each node has an optional parent and an ordered
/// list of children.
///
/// See the [module docs](self) for the invariants this type upholds.
#[derive(Clone, Debug, Default)]
pub struct Hierarchy {
    parents: Vec<Option<NodeId>>,
    children: Vec<Vec<NodeId>>,
}

impl Hierarchy {
    /// Create an empty hierarchy.
    #[inline]
    pub const fn new() -> Self {
        Self { parents: Vec::new(), children: Vec::new() }
    }

    /// Number of nodes in the hierarchy.
    #[inline]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Whether the hierarchy contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// Whether `node` is a valid id for this hierarchy.
    #[inline]
    pub fn contains(&self, node: NodeId) -> bool {
        node.index() < self.len()
    }

    /// Spawn a parentless root node and return its id.
    pub fn spawn_root(&mut self) -> NodeId {
        let id = NodeId::from_index(self.parents.len());
        self.parents.push(None);
        self.children.push(Vec::new());
        id
    }

    /// Spawn a child of `parent` and return its id. The child is appended to
    /// `parent`'s child list, preserving sibling insertion order.
    ///
    /// # Panics
    /// Panics if `parent` is not a valid id for this hierarchy. (Spawning under
    /// an existing node can never create a cycle, so this operation is
    /// otherwise infallible.)
    pub fn spawn_child(&mut self, parent: NodeId) -> NodeId {
        assert!(self.contains(parent), "spawn_child: parent id is out of bounds");
        let id = NodeId::from_index(self.parents.len());
        self.parents.push(Some(parent));
        self.children.push(Vec::new());
        self.children[parent.index()].push(id);
        id
    }

    /// The parent of `node`, or `None` if it is a root.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.parents[node.index()]
    }

    /// The ordered children of `node`.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn children(&self, node: NodeId) -> &[NodeId] {
        &self.children[node.index()]
    }

    /// Re-parent `child` under `new_parent` (or make it a root with `None`),
    /// detaching it from its previous parent first.
    ///
    /// This is a topology-only edit: it never touches transform data. The
    /// caller is responsible for recomputing world transforms afterwards (see
    /// [`crate::propagation::propagate`]).
    ///
    /// # Errors
    /// - [`HierarchyError::InvalidNode`] if `child` or `new_parent` is out of
    ///   bounds.
    /// - [`HierarchyError::Cycle`] if `new_parent` is `child` itself or a
    ///   descendant of `child`.
    pub fn set_parent(
        &mut self,
        child: NodeId,
        new_parent: Option<NodeId>,
    ) -> Result<(), HierarchyError> {
        if !self.contains(child) {
            return Err(HierarchyError::InvalidNode);
        }
        if let Some(parent) = new_parent {
            if !self.contains(parent) {
                return Err(HierarchyError::InvalidNode);
            }
            if parent == child || self.is_descendant_of(parent, child) {
                return Err(HierarchyError::Cycle);
            }
        }

        if let Some(old) = self.parents[child.index()] {
            let siblings = &mut self.children[old.index()];
            if let Some(pos) = siblings.iter().position(|&c| c == child) {
                siblings.remove(pos);
            }
        }

        self.parents[child.index()] = new_parent;
        if let Some(parent) = new_parent {
            self.children[parent.index()].push(child);
        }
        Ok(())
    }

    /// Build a hierarchy from a flat parent table, where `parents[i]` is the
    /// parent of node `i` (or `None` for a root). Children are ordered by
    /// ascending child id.
    ///
    /// # Errors
    /// - [`HierarchyError::InvalidParent`] if any entry references a
    ///   non-existent index.
    /// - [`HierarchyError::Cycle`] if the table is not a forest.
    pub fn from_parents(parents: &[Option<NodeId>]) -> Result<Self, HierarchyError> {
        let n = parents.len();
        let mut hierarchy = Self {
            parents: Vec::with_capacity(n),
            children: Vec::with_capacity(n),
        };
        for _ in 0..n {
            hierarchy.parents.push(None);
            hierarchy.children.push(Vec::new());
        }
        for (i, slot) in parents.iter().enumerate() {
            if let Some(parent) = *slot {
                if parent.index() >= n {
                    return Err(HierarchyError::InvalidParent);
                }
                hierarchy.parents[i] = Some(parent);
                hierarchy.children[parent.index()].push(NodeId::from_index(i));
            }
        }
        hierarchy.validate()?;
        Ok(hierarchy)
    }

    /// Validate that the hierarchy is acyclic (a forest).
    ///
    /// # Errors
    /// Returns [`HierarchyError::Cycle`] if any node participates in a cycle.
    #[inline]
    pub fn validate(&self) -> Result<(), HierarchyError> {
        self.compute_order().map(drop)
    }

    /// Compute a stable parent-before-child traversal order over every node.
    ///
    /// Roots are emitted first in ascending id order; thereafter each node's
    /// children are emitted in their stored sibling order. Because every node
    /// is appended only after its parent has already been emitted, the result
    /// satisfies `order.position(parent) < order.position(child)` for every
    /// edge, which is exactly what [`crate::propagation::propagate`] requires.
    ///
    /// # Errors
    /// Returns [`HierarchyError::Cycle`] if the graph is not a forest (a cycle
    /// leaves some nodes unreachable from any root, so fewer than `len` nodes
    /// are emitted).
    pub fn compute_order(&self) -> Result<Vec<NodeId>, HierarchyError> {
        let n = self.len();
        let mut order: Vec<NodeId> = Vec::with_capacity(n);
        for i in 0..n {
            if self.parents[i].is_none() {
                order.push(NodeId::from_index(i));
            }
        }
        let mut head = 0;
        while head < order.len() {
            let node = order[head];
            head += 1;
            for &child in &self.children[node.index()] {
                order.push(child);
            }
        }
        if order.len() != n {
            return Err(HierarchyError::Cycle);
        }
        Ok(order)
    }

    /// Whether `node` is a (strict) descendant of `ancestor`, walking parent
    /// links upward. Returns `false` when `node == ancestor`.
    fn is_descendant_of(&self, node: NodeId, ancestor: NodeId) -> bool {
        let mut cursor = self.parents[node.index()];
        while let Some(parent) = cursor {
            if parent == ancestor {
                return true;
            }
            cursor = self.parents[parent.index()];
        }
        false
    }
}
