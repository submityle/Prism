//! The per-node record stored in a [`Tree`](crate::Tree).

use alloc::vec::Vec;

use crate::arena::NodeId;

/// A single retained node: an optional reconciliation key, parent/child links
/// and a user payload `T`.
#[derive(Debug, Clone)]
pub struct Node<K, T> {
    pub(crate) key: Option<K>,
    pub(crate) parent: Option<NodeId>,
    pub(crate) children: Vec<NodeId>,
    pub(crate) value: T,
}

impl<K, T> Node<K, T> {
    pub(crate) fn new(key: Option<K>, value: T) -> Self {
        Self {
            key,
            parent: None,
            children: Vec::new(),
            value,
        }
    }

    /// The node's reconciliation key, if it has one.
    #[inline]
    pub fn key(&self) -> Option<&K> {
        self.key.as_ref()
    }

    /// The node's parent, if it is attached.
    #[inline]
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// The node's children, in order.
    #[inline]
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    /// The payload.
    #[inline]
    pub fn value(&self) -> &T {
        &self.value
    }

    /// The payload, mutably.
    #[inline]
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
