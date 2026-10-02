//! Incremental refit of a linear `BVH`: recomputing node bounds while keeping
//! the tree topology fixed.
//!
//! A full [`cpu_build_lbvh`](crate::bvh::cpu_build_lbvh) is the right tool when
//! primitives move far enough that their Morton order changes, but it pays for a
//! Morton pass, a stable sort, and a radix-tree build every frame. When
//! primitives move only a little between frames their sorted order rarely
//! changes, so that topology work is wasted. An incremental refit keeps the
//! existing leaf ordering and node links and only re-derives bounds, which is
//! the standard real-time acceleration-structure update: the tree slowly loses
//! quality as primitives drift, so callers periodically rebuild from scratch,
//! but between rebuilds a refit is far cheaper.
//!
//! [`cpu_refit_lbvh`] is the golden twin of the device refit in
//! [`GpuBvhRefit`](crate::bvh::GpuBvhRefit). It reuses the tree's existing
//! `sorted_indices`, `sorted_codes`, and node links verbatim, gathers each
//! leaf's new box through the unchanged sorted payload, and climbs the same
//! parent links the full build does to union child boxes bottom-up. Because the
//! climb is identical to the one [`cpu_build_lbvh`] runs, refitting a tree with
//! the very boxes it was built from reproduces that build's bounds exactly.
//!
//! # Provenance
//!
//! Bottom-up bounds refit over the linear `BVH` of Karras, "Maximizing
//! Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (High
//! Performance Graphics 2012). No Unreal Engine source or derived code.

use crate::bvh::config::Aabb;
use crate::bvh::cpu::{climb_bounds, Lbvh};

/// Recomputes a tree's bounds for `new_boxes` without changing its topology.
///
/// The returned [`Lbvh`] shares `tree`'s leaf ordering and node links; only
/// its `leaf_aabb` and `internal_aabb` are recomputed from `new_boxes`. Each
/// leaf's box is gathered through the unchanged `sorted_indices` payload, and
/// internal-node boxes are the exact componentwise union of their descendant
/// leaves, matching the full build's bottom-up climb.
///
/// `new_boxes` is indexed in the original primitive order, exactly as the
/// slice passed to [`cpu_build_lbvh`](crate::bvh::cpu_build_lbvh).
///
/// # Panics
///
/// Panics if `new_boxes.len()` differs from the tree's leaf count, since the
/// refit cannot keep the topology meaningful for a different primitive set.
#[must_use]
pub fn cpu_refit_lbvh(tree: &Lbvh, new_boxes: &[Aabb]) -> Lbvh {
    assert_eq!(
        new_boxes.len(),
        tree.num_leaves,
        "refit box count must match the tree's leaf count"
    );

    // Gather each leaf's new box through the unchanged sorted payload: leaf `i`
    // is primitive `sorted_indices[i]`, exactly as the build assigned it.
    let leaf_aabb: Vec<Aabb> = tree
        .sorted_indices
        .iter()
        .map(|&i| new_boxes[i as usize])
        .collect();

    // Trivial trees (zero or one leaf) have no internal nodes, so there is
    // nothing to climb; the leaf box alone describes the tree.
    let internal_aabb = if tree.num_internal == 0 {
        Vec::new()
    } else {
        climb_bounds(
            tree.num_internal,
            &tree.left,
            &tree.right,
            &tree.parent,
            &leaf_aabb,
        )
    };

    Lbvh {
        num_leaves: tree.num_leaves,
        num_internal: tree.num_internal,
        root: tree.root,
        sorted_indices: tree.sorted_indices.clone(),
        sorted_codes: tree.sorted_codes.clone(),
        left: tree.left.clone(),
        right: tree.right.clone(),
        parent: tree.parent.clone(),
        internal_aabb,
        leaf_aabb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::cpu::cpu_build_lbvh;
    use glam::Vec3;

    fn boxes_grid(n: usize, scale: f32) -> Vec<Aabb> {
        (0..n)
            .map(|i| {
                let c = Vec3::new(i as f32 * 2.0, (i % 3) as f32, (i % 5) as f32) * scale;
                Aabb::new(c - Vec3::splat(0.4), c + Vec3::splat(0.4))
            })
            .collect()
    }

    /// Collects the original-order primitive indices of every leaf under an
    /// encoded node id, descending the left/right links of internal nodes.
    fn descendant_prims(tree: &Lbvh, encoded: u32, out: &mut Vec<u32>) {
        if tree.is_leaf(encoded) {
            let leaf = encoded as usize - tree.num_internal;
            out.push(tree.sorted_indices[leaf]);
        } else {
            let i = encoded as usize;
            descendant_prims(tree, tree.left[i], out);
            descendant_prims(tree, tree.right[i], out);
        }
    }

    fn brute_union(boxes: &[Aabb], prims: &[u32]) -> Aabb {
        let mut acc = boxes[prims[0] as usize];
        for &p in &prims[1..] {
            acc = acc.union(&boxes[p as usize]);
        }
        acc
    }

    #[test]
    fn refit_with_same_boxes_reproduces_build_bounds() {
        let boxes = boxes_grid(17, 1.0);
        let built = cpu_build_lbvh(&boxes);
        let refit = cpu_refit_lbvh(&built, &boxes);
        assert_eq!(refit.internal_aabb, built.internal_aabb);
        assert_eq!(refit.leaf_aabb, built.leaf_aabb);
        assert_eq!(refit.parent, built.parent);
        assert_eq!(refit.left, built.left);
        assert_eq!(refit.right, built.right);
    }

    #[test]
    fn refit_bounds_are_exact_descendant_unions() {
        let built = cpu_build_lbvh(&boxes_grid(23, 1.0));
        let moved: Vec<Aabb> = boxes_grid(23, 1.0)
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let d = Vec3::new(0.05 * ((i % 11) as f32 - 5.0), 0.03, -0.02 * i as f32);
                Aabb::new(b.min + d, b.max + d)
            })
            .collect();
        let refit = cpu_refit_lbvh(&built, &moved);
        for node in 0..refit.num_internal {
            let mut prims = Vec::new();
            descendant_prims(&refit, node as u32, &mut prims);
            let expected = brute_union(&moved, &prims);
            assert_eq!(refit.internal_aabb[node], expected, "node {node}");
        }
    }

    #[test]
    fn refit_single_leaf_keeps_the_lone_box() {
        let built = cpu_build_lbvh(&boxes_grid(1, 1.0));
        let moved = vec![Aabb::new(Vec3::splat(9.0), Vec3::splat(10.0))];
        let refit = cpu_refit_lbvh(&built, &moved);
        assert_eq!(refit.num_internal, 0);
        assert_eq!(refit.leaf_aabb, moved);
        assert!(refit.internal_aabb.is_empty());
    }

    #[test]
    fn refit_empty_tree_is_empty() {
        let built = cpu_build_lbvh(&[]);
        let refit = cpu_refit_lbvh(&built, &[]);
        assert_eq!(refit.num_leaves, 0);
        assert!(refit.leaf_aabb.is_empty());
        assert!(refit.internal_aabb.is_empty());
    }

    #[test]
    #[should_panic(expected = "refit box count")]
    fn refit_rejects_mismatched_box_count() {
        let built = cpu_build_lbvh(&boxes_grid(4, 1.0));
        let _ = cpu_refit_lbvh(&built, &boxes_grid(5, 1.0));
    }
}
