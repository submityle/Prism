//! Overlap and ray-cast queries over a [`DynamicBvh`].
//!
//! These are inherent methods on [`DynamicBvh`] implemented in a separate file
//! to keep the tree-maintenance logic and the read-only traversals apart.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

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

    /// Visits candidate leaves whose fat box is hit by `ray`, in order of
    /// increasing ray entry distance (front-to-back).
    ///
    /// For each hit leaf the visitor receives the payload, the leaf's fat box,
    /// and the entry parameter `t`. Returning `false` stops the traversal
    /// immediately, which lets a narrow phase terminate as soon as it confirms
    /// an exact hit that no later candidate box can beat.
    ///
    /// Unlike [`DynamicBvh::ray_cast`], which reports candidates in arbitrary
    /// traversal order, this descends the nearer subtree first using a
    /// distance-ordered frontier. Because an internal node encloses its
    /// children, a child's box entry distance is never smaller than its
    /// parent's, so the first leaf popped is the globally nearest leaf box and
    /// a nearest-hit search is output sensitive.
    pub fn ray_cast_ordered(&self, ray: &Ray, visit: &mut impl FnMut(u64, Aabb, f32) -> bool) {
        if self.root == NULL {
            return;
        }
        let mut frontier = BinaryHeap::new();
        if let Some(t) = self.nodes[self.root as usize].aabb.ray_hit(ray) {
            frontier.push(RayCandidate { t, node: self.root });
        }
        while let Some(RayCandidate { t, node }) = frontier.pop() {
            let node = self.nodes[node as usize];
            if node.is_leaf() {
                if !visit(node.data, node.aabb, t) {
                    return;
                }
            } else {
                for child in [node.child1, node.child2] {
                    if let Some(ct) = self.nodes[child as usize].aabb.ray_hit(ray) {
                        frontier.push(RayCandidate { t: ct, node: child });
                    }
                }
            }
        }
    }

    /// Returns the payload, fat box, and ray entry distance of the leaf whose
    /// fat box is hit nearest along `ray`, or [`None`] when no leaf box is hit.
    ///
    /// This is a broad-phase result: it reports the nearest candidate leaf box,
    /// not an exact geometric hit against the stored geometry. A narrow phase
    /// should refine the reported leaf. Internally this stops at the first leaf
    /// yielded by [`DynamicBvh::ray_cast_ordered`], so it does not enumerate
    /// farther candidates.
    pub fn ray_cast_nearest(&self, ray: &Ray) -> Option<(u64, Aabb, f32)> {
        let mut hit = None;
        self.ray_cast_ordered(ray, &mut |data, aabb, t| {
            hit = Some((data, aabb, t));
            false
        });
        hit
    }
}

/// A pending node in a distance-ordered ray traversal, keyed by the ray entry
/// parameter `t` of its fat box.
///
/// [`BinaryHeap`] is a max-heap, so [`Ord`] is reversed on `t` to make the
/// frontier pop the nearest node first. Entry distances produced by
/// [`Aabb::ray_hit`] are finite and non-negative, so [`f32::total_cmp`] yields
/// a consistent total order here.
#[derive(Clone, Copy)]
struct RayCandidate {
    /// Ray entry parameter `t` of this node's fat box.
    t: f32,
    /// Index of the node within the tree's pool.
    node: u32,
}

impl PartialEq for RayCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.t.total_cmp(&other.t) == Ordering::Equal
    }
}

impl Eq for RayCandidate {}

impl Ord for RayCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the max-heap yields the smallest `t` (nearest) first.
        other.t.total_cmp(&self.t)
    }
}

impl PartialOrd for RayCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::bounding::{Aabb, Ray};
    use crate::bvh::DynamicBvh;
    use glam::Vec3;

    /// Builds a tree with three unit boxes spaced along `+X` at x = 0, 5, 10,
    /// tagged with payloads 0, 1, 2.
    fn spaced_tree() -> DynamicBvh {
        let mut bvh = DynamicBvh::new();
        for (i, x) in [0.0_f32, 5.0, 10.0].into_iter().enumerate() {
            let b = Aabb::new(Vec3::new(x, -0.5, -0.5), Vec3::new(x + 1.0, 0.5, 0.5));
            bvh.insert(b, i as u64);
        }
        bvh
    }

    #[test]
    fn ordered_visits_front_to_back() {
        let bvh = spaced_tree();
        let ray = Ray::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::X);
        let mut order = Vec::new();
        let mut last_t = f32::NEG_INFINITY;
        bvh.ray_cast_ordered(&ray, &mut |data, _aabb, t| {
            assert!(t >= last_t, "entry distances must be non-decreasing");
            last_t = t;
            order.push(data);
            true
        });
        assert_eq!(order, [0, 1, 2]);
    }

    #[test]
    fn ordered_reverse_ray_visits_back_to_front() {
        let bvh = spaced_tree();
        // Fire from +X back toward the origin: nearest box is payload 2.
        let ray = Ray::new(Vec3::new(12.0, 0.0, 0.0), -Vec3::X);
        let mut order = Vec::new();
        bvh.ray_cast_ordered(&ray, &mut |data, _aabb, _t| {
            order.push(data);
            true
        });
        assert_eq!(order, [2, 1, 0]);
    }

    #[test]
    fn nearest_returns_closest_leaf_only() {
        let bvh = spaced_tree();
        let ray = Ray::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::X);
        let (data, aabb, t) = bvh.ray_cast_nearest(&ray).expect("ray hits a leaf");
        assert_eq!(data, 0);
        // Entry is at the near x face of the first fat box (x = -0.1), one unit
        // ahead of the origin minus the 0.1 fattening margin.
        assert!((t - 0.9).abs() < 1e-5, "unexpected entry distance {t}");
        assert!(aabb.contains_point(Vec3::new(0.5, 0.0, 0.0)));
    }

    #[test]
    fn nearest_is_none_on_miss() {
        let bvh = spaced_tree();
        // Parallel ray offset well above every box.
        let ray = Ray::new(Vec3::new(-1.0, 10.0, 0.0), Vec3::X);
        assert!(bvh.ray_cast_nearest(&ray).is_none());
    }

    #[test]
    fn tmax_prunes_far_leaves() {
        let bvh = spaced_tree();
        // Limit the ray so only the first box is within range.
        let ray = Ray::with_tmax(Vec3::new(-1.0, 0.0, 0.0), Vec3::X, 3.0);
        let mut order = Vec::new();
        bvh.ray_cast_ordered(&ray, &mut |data, _aabb, _t| {
            order.push(data);
            true
        });
        assert_eq!(order, [0]);
    }

    #[test]
    fn empty_tree_yields_nothing() {
        let bvh = DynamicBvh::new();
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert!(bvh.ray_cast_nearest(&ray).is_none());
        let mut count = 0;
        bvh.ray_cast_ordered(&ray, &mut |_d, _a, _t| {
            count += 1;
            true
        });
        assert_eq!(count, 0);
    }
}
