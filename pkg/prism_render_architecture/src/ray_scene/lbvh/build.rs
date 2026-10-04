//! Top-level parallel linear-`BVH` build: `Morton` sort plus Karras radix tree,
//! flattened into the shared depth-first [`LinearBvhNode`] layout.
//!
//! This is the realtime, rebuild-every-frame alternative to the binned-`SAH`
//! builder in [`crate::ray_scene::bvh`]. The `SAH` builder produces a
//! higher-quality tree but is inherently serial; the linear builder trades a
//! little traversal quality for an `O(n)`-work, fully parallel construction
//! that a `GPU` kernel runs in a handful of dispatches (quantize, radix sort,
//! radix-tree emit, bottom-up refit). The output is the *same* flattened node
//! array the existing traversal and `GPU` buffer upload already consume, so the
//! two builders are drop-in interchangeable.

use alloc::vec;
use alloc::vec::Vec;

use crate::ray_scene::bvh::{Aabb, Bvh, LinearBvhNode, Triangle};

use super::morton::MortonQuantizer;
use super::radix_sort::{self, MortonEntry};
use super::radix_tree::{self, Child, RadixTree};

/// A parallel linear-`BVH` builder (Karras 2012).
///
/// Stateless; the entry point is [`LinearBvh::build`]. Kept as a type rather
/// than a free function so the `GPU` twin can hang device handles off it later
/// without changing call sites.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearBvh;

impl LinearBvh {
    /// Builds a linear `BVH` over `triangles`.
    ///
    /// Empty input yields an empty hierarchy; a single triangle yields a single
    /// leaf. The returned [`Bvh`] uses the identical node and primitive-table
    /// conventions as [`Bvh::build`], so [`crate::ray_scene::traversal`] and the
    /// `GPU` upload path treat it exactly like a `SAH`-built hierarchy.
    #[must_use]
    pub fn build(triangles: &[Triangle]) -> Bvh {
        let count = triangles.len();
        if count == 0 {
            return Bvh::from_linear(Vec::new(), Vec::new());
        }
        if count == 1 {
            let tri = triangles[0];
            let node = LinearBvhNode {
                bounds: tri.bounds(),
                first_primitive: 0,
                second_child: 0,
                primitive_count: 1,
                axis: 0,
            };
            return Bvh::from_linear(vec![node], vec![tri]);
        }

        // 1. Centroid bounds drive the quantization grid.
        let mut centroid_bounds = Aabb::empty();
        for tri in triangles {
            centroid_bounds = centroid_bounds.enclose(tri.centroid());
        }
        let quantizer = MortonQuantizer::new(centroid_bounds.min, centroid_bounds.max);

        // 2. Quantize each centroid to a Morton key.
        let mut entries: Vec<MortonEntry> = triangles
            .iter()
            .enumerate()
            .map(|(index, tri)| MortonEntry {
                key: quantizer.key(tri.centroid()),
                primitive: index as u32,
            })
            .collect();

        // 3. Sort by key (stable; ties broken later by sorted position).
        radix_sort::radix_sort(&mut entries);

        // 4. Reordered primitive table and per-leaf bounds in sorted order.
        let order: Vec<u32> = entries.iter().map(|e| e.primitive).collect();
        let keys: Vec<u32> = entries.iter().map(|e| e.key).collect();
        let primitives: Vec<Triangle> = order
            .iter()
            .map(|&i| triangles[i as usize])
            .collect();
        let leaf_bounds: Vec<Aabb> = primitives.iter().map(Triangle::bounds).collect();

        // 5. Build the binary radix tree and fit interior bounds bottom-up.
        let tree = radix_tree::build(&keys);
        let mut internal_bounds = vec![Aabb::empty(); tree.internal.len()];
        fit_bounds(Child::Internal(0), &tree, &leaf_bounds, &mut internal_bounds);

        // 6. Flatten the tree into the depth-first linear layout.
        let mut nodes = Vec::with_capacity(2 * count - 1);
        flatten(
            Child::Internal(0),
            &tree,
            &leaf_bounds,
            &internal_bounds,
            &mut nodes,
        );

        Bvh::from_linear(nodes, primitives)
    }
}

/// Post-order pass that fills `internal_bounds[i]` with the union of internal
/// node `i`'s descendant leaf bounds, returning the bounds of `child`.
fn fit_bounds(
    child: Child,
    tree: &RadixTree,
    leaf_bounds: &[Aabb],
    internal_bounds: &mut [Aabb],
) -> Aabb {
    match child {
        Child::Leaf(pos) => leaf_bounds[pos as usize],
        Child::Internal(i) => {
            let node = tree.internal[i as usize];
            let left = fit_bounds(node.left, tree, leaf_bounds, internal_bounds);
            let right = fit_bounds(node.right, tree, leaf_bounds, internal_bounds);
            let bounds = left.union(&right);
            internal_bounds[i as usize] = bounds;
            bounds
        }
    }
}

/// Pre-order pass that appends `child`'s subtree to `nodes` in the depth-first
/// layout: an interior node's first child sits immediately after it and its
/// second child is referenced by [`LinearBvhNode::second_child`].
fn flatten(
    child: Child,
    tree: &RadixTree,
    leaf_bounds: &[Aabb],
    internal_bounds: &[Aabb],
    nodes: &mut Vec<LinearBvhNode>,
) {
    match child {
        Child::Leaf(pos) => {
            nodes.push(LinearBvhNode {
                bounds: leaf_bounds[pos as usize],
                first_primitive: pos,
                second_child: 0,
                primitive_count: 1,
                axis: 0,
            });
        }
        Child::Internal(i) => {
            let bounds = internal_bounds[i as usize];
            let index = nodes.len();
            // Placeholder; `second_child` is patched once the first subtree is
            // laid out so the second child's real index is known.
            nodes.push(LinearBvhNode {
                bounds,
                first_primitive: 0,
                second_child: 0,
                primitive_count: 0,
                axis: bounds.max_extent_axis() as u8,
            });
            let node = tree.internal[i as usize];
            flatten(node.left, tree, leaf_bounds, internal_bounds, nodes);
            let second_child = nodes.len() as u32;
            flatten(node.right, tree, leaf_bounds, internal_bounds, nodes);
            nodes[index].second_child = second_child;
        }
    }
}
