//! Sequential golden twin for the `GPU` batched `AABB`-versus-`BVH` overlap
//! query.
//!
//! Where [`cpu_bvh_pairs`](super::query::cpu_bvh_pairs) answers "which leaves of
//! one tree overlap each other", this query answers "which leaves of a tree
//! overlap each of a batch of *external* boxes". It is the gather step a shape
//! collider runs against a triangle-mesh `LBVH`: broad-phase the mesh with the
//! shape's world `AABB`, collect the candidate triangles, then hand them to the
//! per-triangle narrow phase. Costing work proportional to genuine overlap (not
//! the full primitive count) is exactly why a hierarchy query, rather than the
//! uniform grid in [`crate::broadphase`], is the gather `AAA` engines build over
//! a mesh `BVH`.
//!
//! # Determinism
//!
//! For query `q` the result is the set of original primitive indices (recovered
//! through [`Lbvh::sorted_indices`]) whose leaf box overlaps query box `q`,
//! boundaries inclusive. The tree is walked from the root with an explicit stack
//! (never recursion: a degenerate Karras tree is `O(n)` deep), pruning any
//! subtree whose box misses the query box, exactly as the device kernel does.
//! The device appends hits in traversal order under a per-query atomic, so this
//! twin's stack order and the kernel's append order differ; callers compare the
//! two per query after sorting, which is then bit-for-bit.
//!
//! # Provenance
//!
//! Per-query descent with box pruning over the linear `BVH` of Karras,
//! "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//! (High Performance Graphics 2012); the stackless device counterpart follows
//! Hapala et al., "Efficient Stack-less BVH Traversal for Ray Tracing" (2011).
//! No Unreal Engine source or derived code.

use super::config::Aabb;
use super::cpu::Lbvh;

/// An error from a batched `AABB`-versus-`BVH` overlap query.
///
/// Kept separate from [`BvhQueryError`](super::query::BvhQueryError) so the
/// pair broad phase and this gather query stay semantically independent even
/// though both can overflow a fixed per-slot output buffer on device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlapQueryError {
    /// Some query overlapped more leaves than `capacity_per_query` slots, which
    /// on device would have silently dropped output past that query's region.
    CapacityExceeded {
        /// The per-query output capacity, in primitive indices, that was
        /// exceeded.
        capacity_per_query: u32,
        /// The first query index whose overlap count exceeded the capacity.
        query: u32,
    },
}

impl core::fmt::Display for OverlapQueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            OverlapQueryError::CapacityExceeded {
                capacity_per_query,
                query,
            } => write!(
                f,
                "query {query} overlapped more leaves than the per-query capacity of {capacity_per_query}"
            ),
        }
    }
}

impl core::error::Error for OverlapQueryError {}

/// Whether two boxes overlap, treating shared boundaries as touching.
///
/// A closed test (`<=`) is used on both sides, matching the device kernel and
/// the pair query exactly, so coincident faces count as overlapping.
#[must_use]
fn overlap(a: &Aabb, b: &Aabb) -> bool {
    a.min.x <= b.max.x
        && b.min.x <= a.max.x
        && a.min.y <= b.max.y
        && b.min.y <= a.max.y
        && a.min.z <= b.max.z
        && b.min.z <= a.max.z
}

/// The box of an encoded node id within `lbvh`.
///
/// Internal ids index [`Lbvh::internal_aabb`]; leaf ids (at or above
/// [`Lbvh::num_internal`]) index [`Lbvh::leaf_aabb`] after removing the offset.
#[must_use]
fn node_box(lbvh: &Lbvh, encoded: u32) -> Aabb {
    let id = encoded as usize;
    if id < lbvh.num_internal {
        lbvh.internal_aabb[id]
    } else {
        lbvh.leaf_aabb[id - lbvh.num_internal]
    }
}

/// Gathers, for every box in `queries`, the original indices of the primitives
/// in `lbvh` whose leaf box overlaps it; the golden twin of the `GPU` query.
///
/// The returned outer vector has one entry per query, in query order; each
/// inner vector lists the overlapping primitives' original indices in the
/// hierarchy-traversal order the stack produces. An empty tree yields an empty
/// list for every query, and an empty `queries` slice yields an empty outer
/// vector.
///
/// `capacity_per_query` mirrors the fixed per-query region of the device output
/// buffer: a query that overlaps more leaves than that reports overflow rather
/// than silently truncating, matching the kernel's buffer contract.
///
/// # Errors
///
/// Returns [`OverlapQueryError::CapacityExceeded`] naming the first query whose
/// overlap count exceeds `capacity_per_query`.
pub fn cpu_bvh_aabb_overlap(
    lbvh: &Lbvh,
    queries: &[Aabb],
    capacity_per_query: u32,
) -> Result<Vec<Vec<u32>>, OverlapQueryError> {
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(queries.len());

    // An empty tree overlaps nothing, so every query is empty; a non-empty tree
    // always has a valid encoded root (leaf 0 for a single-leaf tree).
    if lbvh.num_leaves == 0 {
        out.resize_with(queries.len(), Vec::new);
        return Ok(out);
    }

    let mut stack: Vec<u32> = Vec::new();
    for (query_index, query) in queries.iter().enumerate() {
        let mut hits: Vec<u32> = Vec::new();
        stack.clear();
        stack.push(lbvh.root);
        while let Some(node) = stack.pop() {
            if !overlap(&node_box(lbvh, node), query) {
                continue;
            }
            if lbvh.is_leaf(node) {
                let leaf_slot = node as usize - lbvh.num_internal;
                hits.push(lbvh.sorted_indices[leaf_slot]);
            } else {
                stack.push(lbvh.left[node as usize]);
                stack.push(lbvh.right[node as usize]);
            }
        }
        if hits.len() as u32 > capacity_per_query {
            return Err(OverlapQueryError::CapacityExceeded {
                capacity_per_query,
                query: u32::try_from(query_index).unwrap_or(u32::MAX),
            });
        }
        out.push(hits);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{cpu_bvh_aabb_overlap, OverlapQueryError};
    use crate::bvh::config::Aabb;
    use crate::bvh::cpu::cpu_build_lbvh;
    use glam::Vec3;

    /// A unit box centred on `(x, y, z)`.
    fn box_at(x: f32, y: f32, z: f32) -> Aabb {
        let c = Vec3::new(x, y, z);
        let h = Vec3::splat(0.5);
        Aabb::new(c - h, c + h)
    }

    /// Sorts a query's overlap list so the set can be compared independent of
    /// traversal order.
    fn sorted(mut v: Vec<u32>) -> Vec<u32> {
        v.sort_unstable();
        v
    }

    /// Independent `O(n*m)` reference: the original indices of every leaf box
    /// that overlaps the query box. This never touches the tree, so agreement
    /// with [`cpu_bvh_aabb_overlap`] proves the traversal loses no hit and
    /// invents none.
    fn brute_force(boxes: &[Aabb], query: &Aabb) -> Vec<u32> {
        let overlap = |a: &Aabb, b: &Aabb| {
            a.min.x <= b.max.x
                && b.min.x <= a.max.x
                && a.min.y <= b.max.y
                && b.min.y <= a.max.y
                && a.min.z <= b.max.z
                && b.min.z <= a.max.z
        };
        (0..boxes.len() as u32)
            .filter(|&i| overlap(&boxes[i as usize], query))
            .collect()
    }

    #[test]
    fn empty_tree_yields_empty_lists() {
        let lbvh = cpu_build_lbvh(&[]);
        let queries = [box_at(0.0, 0.0, 0.0), box_at(5.0, 0.0, 0.0)];
        let got = cpu_bvh_aabb_overlap(&lbvh, &queries, 8).expect("no overflow");
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(Vec::is_empty));
    }

    #[test]
    fn no_queries_yields_empty_outer() {
        let lbvh = cpu_build_lbvh(&[box_at(0.0, 0.0, 0.0)]);
        let got = cpu_bvh_aabb_overlap(&lbvh, &[], 8).expect("no overflow");
        assert!(got.is_empty());
    }

    #[test]
    fn single_leaf_hit_and_miss() {
        let lbvh = cpu_build_lbvh(&[box_at(0.0, 0.0, 0.0)]);
        let queries = [box_at(0.0, 0.0, 0.0), box_at(10.0, 0.0, 0.0)];
        let got = cpu_bvh_aabb_overlap(&lbvh, &queries, 8).expect("no overflow");
        assert_eq!(sorted(got[0].clone()), vec![0]);
        assert!(got[1].is_empty());
    }

    #[test]
    fn shared_boundary_counts_as_overlap() {
        // Two unit boxes exactly touching along x: the query sits flush against
        // leaf 0's +x face, so the closed test must report both as overlapping.
        let lbvh = cpu_build_lbvh(&[box_at(0.0, 0.0, 0.0), box_at(5.0, 0.0, 0.0)]);
        // Query spanning x in [0.5, 1.5]: left face coincides with leaf 0's +x.
        let query = Aabb::new(Vec3::new(0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5));
        let got = cpu_bvh_aabb_overlap(&lbvh, &[query], 8).expect("no overflow");
        assert_eq!(sorted(got[0].clone()), vec![0]);
    }

    #[test]
    fn query_covering_several_leaves_gathers_all() {
        let boxes = [
            box_at(0.0, 0.0, 0.0),
            box_at(1.0, 0.0, 0.0),
            box_at(2.0, 0.0, 0.0),
            box_at(50.0, 0.0, 0.0),
        ];
        let lbvh = cpu_build_lbvh(&boxes);
        // Big query covering the first three clustered boxes but not the far one.
        let query = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));
        let got = cpu_bvh_aabb_overlap(&lbvh, &[query], 16).expect("no overflow");
        assert_eq!(sorted(got[0].clone()), vec![0, 1, 2]);
    }

    #[test]
    fn recovers_original_indices_through_sort() {
        // Spread boxes so the Morton sort permutes leaf slots; the query must
        // still report original input indices, not sorted slots.
        let boxes = [
            box_at(9.0, 9.0, 9.0),
            box_at(-9.0, -9.0, -9.0),
            box_at(0.0, 0.0, 0.0),
        ];
        let lbvh = cpu_build_lbvh(&boxes);
        let got = cpu_bvh_aabb_overlap(&lbvh, &[box_at(0.0, 0.0, 0.0)], 8).expect("no overflow");
        // Only the box at the origin (original index 2) overlaps.
        assert_eq!(sorted(got[0].clone()), vec![2]);
    }

    #[test]
    fn matches_brute_force_over_a_grid() {
        // A 4x4x1 grid of unit boxes and a handful of queries of varied size;
        // the hierarchy query must agree with the independent O(n*m) scan on
        // every query.
        let mut boxes = Vec::new();
        for ix in 0..4 {
            for iz in 0..4 {
                boxes.push(box_at(ix as f32 * 1.5, 0.0, iz as f32 * 1.5));
            }
        }
        let lbvh = cpu_build_lbvh(&boxes);
        let queries = [
            box_at(0.0, 0.0, 0.0),
            Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(3.5, 0.5, 3.5)),
            box_at(100.0, 0.0, 0.0),
            Aabb::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0)),
        ];
        let got = cpu_bvh_aabb_overlap(&lbvh, &queries, 64).expect("no overflow");
        for (q, query) in queries.iter().enumerate() {
            assert_eq!(
                sorted(got[q].clone()),
                sorted(brute_force(&boxes, query)),
                "query {q} disagrees with brute force"
            );
        }
    }

    #[test]
    fn overflow_is_reported_with_the_offending_query() {
        let boxes = [
            box_at(0.0, 0.0, 0.0),
            box_at(1.0, 0.0, 0.0),
            box_at(2.0, 0.0, 0.0),
        ];
        let lbvh = cpu_build_lbvh(&boxes);
        let big = Aabb::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        // Capacity 2 but the big query overlaps all three leaves.
        let err = cpu_bvh_aabb_overlap(&lbvh, &[box_at(0.0, 0.0, 0.0), big], 2).unwrap_err();
        assert_eq!(
            err,
            OverlapQueryError::CapacityExceeded {
                capacity_per_query: 2,
                query: 1,
            }
        );
    }

    #[test]
    fn capacity_boundary_is_inclusive() {
        // Exactly `capacity_per_query` hits must succeed; one more must fail.
        let boxes = [box_at(0.0, 0.0, 0.0), box_at(1.0, 0.0, 0.0)];
        let lbvh = cpu_build_lbvh(&boxes);
        let both = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 1.0, 1.0));
        assert!(cpu_bvh_aabb_overlap(&lbvh, &[both], 2).is_ok());
        assert!(cpu_bvh_aabb_overlap(&lbvh, &[both], 1).is_err());
    }
}
