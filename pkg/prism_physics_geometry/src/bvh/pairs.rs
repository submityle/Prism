//! All-pairs overlap query over a [`DynamicBvh`].
//!
//! This finds every unordered pair of leaves whose fat boxes overlap, the
//! standard broad-phase step that feeds a narrow phase with collision
//! candidates. It lives in its own file to keep the pair-descent logic separate
//! from the single-target overlap, ray, and nearest-point traversals.

use alloc::vec;
use alloc::vec::Vec;

use super::node::NULL;
use super::tree::DynamicBvh;

impl DynamicBvh {
    /// Visits every unordered pair of leaf payloads whose fat boxes overlap.
    ///
    /// Each overlapping pair is reported exactly once; a leaf is never paired
    /// with itself. The traversal exploits the tree structure: two distinct
    /// leaves share a unique lowest common ancestor, at which they fall into
    /// opposite child subtrees, so the sibling-cross descent reaches each pair
    /// on a single path. Subtrees whose enclosing boxes do not overlap are
    /// pruned, so the cost scales with the number of overlapping candidates
    /// rather than the square of the leaf count.
    pub fn query_self_pairs(&self, visit: &mut impl FnMut(u64, u64)) {
        if self.root == NULL {
            return;
        }
        if self.nodes[self.root as usize].is_leaf() {
            return;
        }

        // First expand every internal node into a cross pair of its two
        // children; this seeds the sibling-vs-sibling comparisons.
        let mut singles = vec![self.root];
        let mut crosses: Vec<(u32, u32)> = Vec::new();
        while let Some(index) = singles.pop() {
            let node = self.nodes[index as usize];
            if node.is_leaf() {
                continue;
            }
            crosses.push((node.child1, node.child2));
            singles.push(node.child1);
            singles.push(node.child2);
        }

        // Resolve each cross pair, splitting the internal side(s) until both
        // ends are leaves. Non-overlapping subtree pairs are pruned.
        while let Some((a, b)) = crosses.pop() {
            let na = self.nodes[a as usize];
            let nb = self.nodes[b as usize];
            if !na.aabb.intersects(&nb.aabb) {
                continue;
            }
            match (na.is_leaf(), nb.is_leaf()) {
                (true, true) => visit(na.data, nb.data),
                (false, true) => {
                    crosses.push((na.child1, b));
                    crosses.push((na.child2, b));
                }
                (true, false) => {
                    crosses.push((a, nb.child1));
                    crosses.push((a, nb.child2));
                }
                (false, false) => {
                    crosses.push((na.child1, nb.child1));
                    crosses.push((na.child1, nb.child2));
                    crosses.push((na.child2, nb.child1));
                    crosses.push((na.child2, nb.child2));
                }
            }
        }
    }

    /// Collects every unordered pair of leaf payloads whose fat boxes overlap.
    pub fn query_self_pairs_collect(&self) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        self.query_self_pairs(&mut |a, b| out.push((a, b)));
        out
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::bounding::Aabb;
    use crate::bvh::DynamicBvh;
    use glam::Vec3;

    /// Normalizes a pair so `(a, b)` and `(b, a)` compare equal, for order-
    /// independent assertions.
    fn norm(mut v: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
        for p in &mut v {
            if p.0 > p.1 {
                *p = (p.1, p.0);
            }
        }
        v.sort_unstable();
        v
    }

    #[test]
    fn disjoint_boxes_have_no_pairs() {
        let mut bvh = DynamicBvh::new();
        for (i, x) in [0.0_f32, 5.0, 10.0].into_iter().enumerate() {
            let b = Aabb::new(Vec3::new(x, -0.5, -0.5), Vec3::new(x + 1.0, 0.5, 0.5));
            bvh.insert(b, i as u64);
        }
        assert!(bvh.query_self_pairs_collect().is_empty());
    }

    #[test]
    fn overlapping_pair_reported_once() {
        let mut bvh = DynamicBvh::new();
        bvh.insert(Aabb::new(Vec3::ZERO, Vec3::ONE), 0);
        // Overlaps box 0.
        bvh.insert(Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5)), 1);
        // Far away, overlaps nothing.
        bvh.insert(Aabb::new(Vec3::splat(10.0), Vec3::splat(11.0)), 2);
        let pairs = norm(bvh.query_self_pairs_collect());
        assert_eq!(pairs, [(0, 1)]);
    }

    #[test]
    fn mutual_cluster_reports_all_pairs() {
        let mut bvh = DynamicBvh::new();
        // Three boxes all overlapping around the origin.
        bvh.insert(Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5)), 0);
        bvh.insert(Aabb::new(Vec3::splat(-0.4), Vec3::splat(0.6)), 1);
        bvh.insert(Aabb::new(Vec3::splat(-0.3), Vec3::splat(0.7)), 2);
        let pairs = norm(bvh.query_self_pairs_collect());
        assert_eq!(pairs, [(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn single_leaf_and_empty_have_no_pairs() {
        let empty = DynamicBvh::new();
        assert!(empty.query_self_pairs_collect().is_empty());
        let mut one = DynamicBvh::new();
        one.insert(Aabb::new(Vec3::ZERO, Vec3::ONE), 0);
        assert!(one.query_self_pairs_collect().is_empty());
    }
}
