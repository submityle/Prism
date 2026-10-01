//! Frustum-culling queries over a [`DynamicBvh`].
//!
//! This visits the leaves whose fat boxes survive a view-frustum cull, the
//! acceleration-structure counterpart of per-object culling. It is kept in its
//! own file alongside the other read-only traversals.

use alloc::vec;
use alloc::vec::Vec;

use crate::bounding::Frustum;

use super::node::NULL;
use super::tree::DynamicBvh;

impl DynamicBvh {
    /// Visits the payload of every leaf whose fat box is not culled by
    /// `frustum`.
    ///
    /// Internal nodes are tested first: if an internal node's enclosing box is
    /// fully outside the frustum, its whole subtree is pruned, so the cost is
    /// output sensitive. Because the test is the conservative
    /// [`Frustum::intersects_aabb`], a visited leaf is a visibility candidate
    /// that a precise per-object test can still reject.
    pub fn query_frustum(&self, frustum: &Frustum, visit: &mut impl FnMut(u64)) {
        if self.root == NULL {
            return;
        }
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if !frustum.intersects_aabb(&node.aabb) {
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

    /// Collects the payloads of all leaves whose fat box survives `frustum`.
    pub fn query_frustum_collect(&self, frustum: &Frustum) -> Vec<u64> {
        let mut out = Vec::new();
        self.query_frustum(frustum, &mut |data| out.push(data));
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::bounding::{Aabb, Frustum, Plane};
    use crate::bvh::DynamicBvh;
    use glam::Vec3;

    /// Three unit boxes spaced along `+X` at x = 0, 5, 10 (payloads 0, 1, 2).
    fn spaced_tree() -> DynamicBvh {
        let mut bvh = DynamicBvh::new();
        for (i, x) in [0.0_f32, 5.0, 10.0].into_iter().enumerate() {
            let b = Aabb::new(Vec3::new(x, -0.5, -0.5), Vec3::new(x + 1.0, 0.5, 0.5));
            bvh.insert(b, i as u64);
        }
        bvh
    }

    /// Axis-aligned box frustum spanning `[min, max]` on x and all of y/z.
    fn slab_frustum(min_x: f32, max_x: f32) -> Frustum {
        let big = 1.0e6;
        Frustum::new([
            Plane::new(Vec3::X, -min_x),     // x >= min_x
            Plane::new(Vec3::NEG_X, max_x),  // x <= max_x
            Plane::new(Vec3::Y, big),
            Plane::new(Vec3::NEG_Y, big),
            Plane::new(Vec3::Z, big),
            Plane::new(Vec3::NEG_Z, big),
        ])
    }

    #[test]
    fn frustum_selects_boxes_in_slab() {
        let bvh = spaced_tree();
        // Keep only the x in [-1, 2] slab -> just box 0 (x = 0..1).
        let mut hits = bvh.query_frustum_collect(&slab_frustum(-1.0, 2.0));
        hits.sort_unstable();
        assert_eq!(hits, [0]);
    }

    #[test]
    fn frustum_spanning_two_boxes() {
        let bvh = spaced_tree();
        // Slab x in [-1, 7] covers boxes 0 and 1 but not box 2 at x = 10.
        let mut hits = bvh.query_frustum_collect(&slab_frustum(-1.0, 7.0));
        hits.sort_unstable();
        assert_eq!(hits, [0, 1]);
    }

    #[test]
    fn frustum_rejecting_all() {
        let bvh = spaced_tree();
        // Slab well past every box.
        assert!(bvh.query_frustum_collect(&slab_frustum(50.0, 60.0)).is_empty());
    }

    #[test]
    fn empty_tree_yields_nothing() {
        let bvh = DynamicBvh::new();
        assert!(bvh.query_frustum_collect(&slab_frustum(-1.0, 1.0)).is_empty());
    }
}
