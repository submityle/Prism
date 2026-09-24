//! Internal node representation for the dynamic bounding-volume hierarchy.

use crate::bounding::Aabb;

/// Sentinel "null" index used for missing parent/child links and the free-list
/// terminator.
pub(in crate::bvh) const NULL: u32 = u32::MAX;

/// A single node in the tree's node pool.
///
/// Nodes are stored contiguously in a [`Vec`] and referenced by index. A node
/// is either a *leaf* (both children are [`NULL`]) or an *internal* node (both
/// children are set). Free slots form a singly linked list through
/// [`Node::parent_or_next`] and are marked by a height of `-1`.
#[derive(Clone, Copy, Debug)]
pub(in crate::bvh) struct Node {
    /// Fat bounding box of this node. For a leaf this is the stored fat box;
    /// for an internal node it encloses both children.
    pub aabb: Aabb,
    /// Parent index when the node is live, or the next free slot when the node
    /// is on the free list.
    pub parent_or_next: u32,
    /// First child index, or [`NULL`] for a leaf.
    pub child1: u32,
    /// Second child index, or [`NULL`] for a leaf.
    pub child2: u32,
    /// User payload associated with a leaf; unused for internal nodes.
    pub data: u64,
    /// Node height: `0` for a leaf, `1 + max(child heights)` for an internal
    /// node, and `-1` for a free slot.
    pub height: i32,
    /// Generation counter for the slot, bumped on free to invalidate stale
    /// [`ProxyId`] handles.
    ///
    /// [`ProxyId`]: crate::proxy::ProxyId
    pub generation: u32,
}

impl Node {
    /// Returns `true` if this node is a leaf (has no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.child1 == NULL
    }
}
