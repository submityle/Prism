//! Overlap and ray-cast queries over a [`DynamicBvh`].
//!
//! These are inherent methods on [`DynamicBvh`] implemented in a separate file
//! to keep the tree-maintenance logic and the read-only traversals apart.

use alloc::vec;
use alloc::vec::Vec;

use crate::bounding::{Aabb, Ray};

use super::node::NULL;
use super::tree::DynamicBvh;

impl DynamicBvh {
    /// Visits the payload of every leaf whose fat box overlaps `aabb`.
    ///
    /// The traversal prunes subtrees whose enclosing box does not intersect
    /// `aabb`, so the cost is output-sensitive rather than linear in leaf
    /// count.
    pub fn query_aabb(&self, aabb: Aabb, visit: &mut impl FnMut(u64)) {
        if self.root == NULL {
            return;
        }
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if !node.aabb.intersects(&aabb) {
                continue;
            }
            if node.is_leaf() {
                visit(node.data);
            } else {
                stack.push(node.child1);
                stack.push(node.child2);
            }
        }
    }

    /// Collects the payloads of all leaves whose fat box overlaps `aabb`.
    pub fn query_aabb_collect(&self, aabb: Aabb) -> Vec<u64> {
        let mut out = Vec::new();
        self.query_aabb(aabb, &mut |data| out.push(data));
        out
    }

    /// Visits every leaf whose fat box is hit by `ray`, passing the payload and
    /// the leaf's fat box.
    ///
    /// This is a broad-phase ray query: it reports candidate leaves (those the
    /// ray's box test does not prune), not exact geometric hits.
    pub fn ray_cast(&self, ray: &Ray, visit: &mut impl FnMut(u64, Aabb)) {
        if self.root == NULL {
            return;
        }
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if node.aabb.ray_hit(ray).is_none() {
                continue;
            }
            if node.is_leaf() {
                visit(node.data, node.aabb);
            } else {
                stack.push(node.child1);
                stack.push(node.child2);
            }
        }
    }
}
