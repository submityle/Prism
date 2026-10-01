//! Nearest-leaf point queries over a [`DynamicBvh`].
//!
//! This complements the ray traversals in `query` with a best-first search for
//! the leaf whose fat box is closest to a query point. It lives in its own file
//! so the point-distance frontier logic stays separate from the overlap and
//! ray-cast code.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;

use glam::Vec3;

use crate::bounding::Aabb;

use super::node::NULL;
use super::tree::DynamicBvh;

impl DynamicBvh {
    /// Returns the payload, fat box, and squared box distance of the leaf whose
    /// fat box is nearest to `p`, or [`None`] when the tree is empty.
    ///
    /// This is a broad-phase result: the reported distance is to the leaf's fat
    /// box, not to the exact geometry stored under the payload, and a point
    /// inside a fat box yields a squared distance of `0.0`. A narrow phase
    /// should refine the reported leaf.
    ///
    /// The search is best-first over a frontier ordered by each node's lower
    /// bound (the squared distance from `p` to the node's enclosing fat box).
    /// Because an internal node encloses its children, a child's box distance
    /// is never smaller than its parent's, so the lower bound is monotone along
    /// any root-to-leaf path. The first leaf popped is therefore the globally
    /// nearest leaf box, and the traversal is output sensitive rather than
    /// linear in the leaf count.
    pub fn nearest_leaf_to_point(&self, p: Vec3) -> Option<(u64, Aabb, f32)> {
        if self.root == NULL {
            return None;
        }
        let mut frontier = BinaryHeap::new();
        let root_d2 = self.nodes[self.root as usize]
            .aabb
            .distance_squared_to_point(p);
        frontier.push(PointCandidate { d2: root_d2, node: self.root });
        while let Some(PointCandidate { d2, node }) = frontier.pop() {
            let node = self.nodes[node as usize];
            if node.is_leaf() {
                // The frontier yields nodes by non-decreasing lower bound, so
                // the first leaf popped already has the smallest box distance.
                return Some((node.data, node.aabb, d2));
            }
            for child in [node.child1, node.child2] {
                let cd2 = self.nodes[child as usize]
                    .aabb
                    .distance_squared_to_point(p);
                frontier.push(PointCandidate { d2: cd2, node: child });
            }
        }
        None
    }

    /// Returns up to `k` leaves ordered by increasing squared fat-box distance
    /// from `p`, nearest first. Fewer than `k` entries are returned when the
    /// tree has fewer leaves, and an empty vector when `k == 0` or the tree is
    /// empty.
    ///
    /// Each entry is `(payload, fat_box, squared_box_distance)`, matching
    /// [`DynamicBvh::nearest_leaf_to_point`]. The best-first frontier yields
    /// leaves in non-decreasing lower-bound order, so collecting the first `k`
    /// popped leaves is exact; the traversal stops as soon as `k` leaves are
    /// found and never enumerates farther subtrees.
    pub fn nearest_k_leaves_to_point(&self, p: Vec3, k: usize) -> Vec<(u64, Aabb, f32)> {
        let mut out = Vec::new();
        if k == 0 || self.root == NULL {
            return out;
        }
        let mut frontier = BinaryHeap::new();
        let root_d2 = self.nodes[self.root as usize]
            .aabb
            .distance_squared_to_point(p);
        frontier.push(PointCandidate { d2: root_d2, node: self.root });
        while let Some(PointCandidate { d2, node }) = frontier.pop() {
            let node = self.nodes[node as usize];
            if node.is_leaf() {
                out.push((node.data, node.aabb, d2));
                if out.len() == k {
                    break;
                }
            } else {
                for child in [node.child1, node.child2] {
                    let cd2 = self.nodes[child as usize]
                        .aabb
                        .distance_squared_to_point(p);
                    frontier.push(PointCandidate { d2: cd2, node: child });
                }
            }
        }
        out
    }

    /// Returns the nearest leaf and its exact squared geometric distance using
    /// branch-and-bound refinement.
    ///
    /// The frontier yields nodes by non-decreasing fat-box lower bound, so once
    /// a node's box distance exceeds the best confirmed geometric distance the
    /// search can terminate: every remaining node is at least as far. For each
    /// popped leaf, `refine` is invoked with the payload and fat box and must
    /// return the exact squared distance from the query to the stored geometry,
    /// or [`None`] to reject the leaf. The returned pair is the payload and
    /// squared geometric distance of the globally nearest accepted leaf, or
    /// [`None`] when the tree is empty or every leaf was rejected.
    pub fn nearest_leaf_refined<F>(&self, p: Vec3, mut refine: F) -> Option<(u64, f32)>
    where
        F: FnMut(u64, Aabb) -> Option<f32>,
    {
        if self.root == NULL {
            return None;
        }
        let mut frontier = BinaryHeap::new();
        let root_d2 = self.nodes[self.root as usize]
            .aabb
            .distance_squared_to_point(p);
        frontier.push(PointCandidate { d2: root_d2, node: self.root });
        let mut best: Option<(u64, f32)> = None;
        while let Some(PointCandidate { d2, node }) = frontier.pop() {
            // Box lower bound already beyond the best hit: nothing nearer left.
            if best.is_some_and(|(_, b)| d2 >= b) {
                break;
            }
            let node = self.nodes[node as usize];
            if node.is_leaf() {
                if let Some(actual) = refine(node.data, node.aabb)
                    && best.is_none_or(|(_, b)| actual < b)
                {
                    best = Some((node.data, actual));
                }
            } else {
                for child in [node.child1, node.child2] {
                    let cd2 = self.nodes[child as usize]
                        .aabb
                        .distance_squared_to_point(p);
                    frontier.push(PointCandidate { d2: cd2, node: child });
                }
            }
        }
        best
    }
}

/// A pending node in a nearest-point traversal, keyed by the squared distance
/// `d2` from the query point to the node's fat box.
///
/// [`BinaryHeap`] is a max-heap, so [`Ord`] is reversed on `d2` to make the
/// frontier pop the nearest node first. Squared box distances are finite and
/// non-negative, so [`f32::total_cmp`] yields a consistent total order here.
#[derive(Clone, Copy)]
struct PointCandidate {
    /// Squared distance from the query point to this node's fat box.
    d2: f32,
    /// Index of the node within the tree's pool.
    node: u32,
}

impl PartialEq for PointCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.d2.total_cmp(&other.d2) == Ordering::Equal
    }
}

impl Eq for PointCandidate {}

impl Ord for PointCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the max-heap yields the smallest `d2` (nearest) first.
        other.d2.total_cmp(&self.d2)
    }
}

impl PartialOrd for PointCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use crate::bounding::Aabb;
    use crate::bvh::DynamicBvh;
    use glam::Vec3;

    /// Builds a tree with three unit boxes spaced along `+X` at x = 0, 5, 10,
    /// tagged with payloads 0, 1, 2. Each stored fat box is fattened by the
    /// tree's 0.1 margin, so the near x face of box 0 sits at x = -0.1.
    fn spaced_tree() -> DynamicBvh {
        let mut bvh = DynamicBvh::new();
        for (i, x) in [0.0_f32, 5.0, 10.0].into_iter().enumerate() {
            let b = Aabb::new(Vec3::new(x, -0.5, -0.5), Vec3::new(x + 1.0, 0.5, 0.5));
            bvh.insert(b, i as u64);
        }
        bvh
    }

    #[test]
    fn nearest_picks_closest_box() {
        let bvh = spaced_tree();
        let (data, aabb, d2) = bvh
            .nearest_leaf_to_point(Vec3::new(-2.0, 0.0, 0.0))
            .expect("non-empty tree");
        assert_eq!(data, 0);
        // Near x face of the first fat box is at x = -0.1, so the squared
        // distance from x = -2.0 is (2.0 - 0.1)^2 = 1.9^2 = 3.61.
        assert!((d2 - 3.61).abs() < 1e-5, "unexpected squared distance {d2}");
        assert!(aabb.contains_point(Vec3::new(0.5, 0.0, 0.0)));
    }

    #[test]
    fn nearest_from_far_side_picks_last_box() {
        let bvh = spaced_tree();
        let (data, _aabb, d2) = bvh
            .nearest_leaf_to_point(Vec3::new(13.0, 0.0, 0.0))
            .expect("non-empty tree");
        assert_eq!(data, 2);
        // Far x face of box 2 is at x = 11.1, squared distance (13 - 11.1)^2.
        assert!((d2 - 3.61).abs() < 1e-5, "unexpected {d2}");
    }

    #[test]
    fn point_inside_box_is_zero_distance() {
        let bvh = spaced_tree();
        let (data, _aabb, d2) = bvh
            .nearest_leaf_to_point(Vec3::new(5.5, 0.0, 0.0))
            .expect("non-empty tree");
        assert_eq!(data, 1);
        assert!((d2 - 0.0).abs() < 1e-6, "point inside box should be 0, got {d2}");
    }

    #[test]
    fn empty_tree_returns_none() {
        let bvh = DynamicBvh::new();
        assert!(bvh.nearest_leaf_to_point(Vec3::ZERO).is_none());
    }

    #[test]
    fn single_leaf_is_returned() {
        let mut bvh = DynamicBvh::new();
        bvh.insert(Aabb::new(Vec3::ZERO, Vec3::ONE), 7);
        let (data, _aabb, _d2) = bvh
            .nearest_leaf_to_point(Vec3::new(10.0, 10.0, 10.0))
            .expect("one leaf");
        assert_eq!(data, 7);
    }

    #[test]
    fn k_nearest_orders_by_distance() {
        let bvh = spaced_tree();
        let got = bvh.nearest_k_leaves_to_point(Vec3::new(-2.0, 0.0, 0.0), 2);
        assert_eq!(got[0].0, 0);
        assert_eq!(got[1].0, 1);
        // Distances must be non-decreasing.
        assert!(got[0].2 <= got[1].2);
    }

    #[test]
    fn k_nearest_clamps_to_leaf_count() {
        let bvh = spaced_tree();
        let got = bvh.nearest_k_leaves_to_point(Vec3::ZERO, 10);
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn k_nearest_zero_or_empty_is_empty() {
        let bvh = spaced_tree();
        assert!(bvh.nearest_k_leaves_to_point(Vec3::ZERO, 0).is_empty());
        let empty = DynamicBvh::new();
        assert!(empty.nearest_k_leaves_to_point(Vec3::ZERO, 3).is_empty());
    }
}
