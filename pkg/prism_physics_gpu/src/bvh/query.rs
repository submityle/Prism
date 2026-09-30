//! Sequential golden twin for the `GPU` `BVH` overlap-pair broad phase.
//!
//! Once the [`Lbvh`] is built, [`cpu_bvh_pairs`] enumerates every unordered pair
//! of leaf primitives whose boxes overlap by traversing the hierarchy, exactly
//! as the device kernel does: one query per leaf, descending the tree and
//! pruning any subtree whose box misses the query box. Unlike the Teschner
//! uniform-grid broad phase in [`crate::broadphase`], which degrades when
//! primitive sizes differ wildly, a hierarchy query costs work proportional to
//! genuine overlap regardless of the size distribution, which is why it is the
//! broad phase `AAA` engines build over an `LBVH`.
//!
//! # Determinism and de-duplication
//!
//! Every unordered overlapping pair is emitted exactly once: the query for leaf
//! slot `s` only emits an overlapping leaf slot strictly greater than `s`, so
//! the smaller slot owns the pair. Output is a [`CandidatePair`], whose
//! canonical `a < b` form makes the emission order irrelevant, so the device's
//! atomic-append order and this twin's traversal order compare equal after both
//! sides are sorted.
//!
//! # Provenance
//!
//! Per-primitive descent with box pruning over the linear `BVH` of Karras,
//! "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//! (High Performance Graphics 2012); the stackless device counterpart follows
//! Hapala et al., "Efficient Stack-less BVH Traversal for Ray Tracing" (2011).
//! No Unreal Engine source or derived code.

use crate::broadphase::CandidatePair;

use super::config::Aabb;
use super::cpu::Lbvh;

/// An error from a `BVH` overlap-pair query.
///
/// Kept local to the query rather than reusing the broad-phase error so the two
/// broad phases stay semantically independent even though both can overflow a
/// fixed output buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvhQueryError {
    /// The query found more overlapping pairs than `capacity` slots, which on
    /// device would have silently dropped output past the buffer end.
    PairCapacityExceeded {
        /// The output-buffer capacity, in pairs, that was exceeded.
        capacity: u32,
    },
}

impl core::fmt::Display for BvhQueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            BvhQueryError::PairCapacityExceeded { capacity } => {
                write!(f, "overlap pairs exceeded capacity of {capacity}")
            }
        }
    }
}

impl core::error::Error for BvhQueryError {}

/// Whether two boxes overlap, treating shared boundaries as touching.
///
/// The half-open-versus-closed choice must match the device kernel exactly; a
/// closed test (`<=`) is used on both sides so coincident faces count as an
/// overlapping pair.
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

/// Enumerates every overlapping primitive pair in `lbvh`, the golden twin of the
/// `GPU` query.
///
/// For each leaf slot the tree is walked from the root with an explicit stack
/// (never recursion, since a degenerate Karras tree can be `O(n)` deep), and any
/// subtree whose box misses the query box is pruned. A leaf slot strictly
/// greater than the querying slot and whose box overlaps is emitted as a
/// [`CandidatePair`] of the two primitives' original indices.
///
/// # Errors
///
/// Returns [`BvhQueryError::PairCapacityExceeded`] when more than `capacity`
/// pairs are found, mirroring the device buffer's fixed size.
pub fn cpu_bvh_pairs(lbvh: &Lbvh, capacity: u32) -> Result<Vec<CandidatePair>, BvhQueryError> {
    let mut pairs = Vec::new();
    // Fewer than two leaves can produce no pair.
    if lbvh.num_leaves < 2 {
        return Ok(pairs);
    }

    let mut stack: Vec<u32> = Vec::new();
    for query_slot in 0..lbvh.num_leaves {
        let query = lbvh.leaf_aabb[query_slot];
        stack.clear();
        stack.push(lbvh.root);
        while let Some(node) = stack.pop() {
            if !overlap(&node_box(lbvh, node), &query) {
                continue;
            }
            if lbvh.is_leaf(node) {
                let other_slot = node as usize - lbvh.num_internal;
                if other_slot > query_slot {
                    pairs.push(CandidatePair::new(
                        lbvh.sorted_indices[query_slot],
                        lbvh.sorted_indices[other_slot],
                    ));
                }
            } else {
                stack.push(lbvh.left[node as usize]);
                stack.push(lbvh.right[node as usize]);
            }
        }
    }

    if pairs.len() as u32 > capacity {
        return Err(BvhQueryError::PairCapacityExceeded { capacity });
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::{cpu_bvh_pairs, BvhQueryError};
    use crate::broadphase::CandidatePair;
    use crate::bvh::config::Aabb;
    use crate::bvh::cpu::cpu_build_lbvh;
    use glam::Vec3;

    /// A unit box centred on `(x, y, z)`.
    fn box_at(x: f32, y: f32, z: f32) -> Aabb {
        let c = Vec3::new(x, y, z);
        let h = Vec3::splat(0.5);
        Aabb::new(c - h, c + h)
    }

    /// Independent `O(n^2)` reference: every unordered pair of input boxes that
    /// overlap, canonicalised. This does not touch the tree, so agreement with
    /// [`cpu_bvh_pairs`] proves the hierarchy traversal loses no pair and
    /// invents none.
    fn brute_force(boxes: &[Aabb]) -> Vec<CandidatePair> {
        let overlap = |a: &Aabb, b: &Aabb| {
            a.min.x <= b.max.x
                && b.min.x <= a.max.x
                && a.min.y <= b.max.y
                && b.min.y <= a.max.y
                && a.min.z <= b.max.z
                && b.min.z <= a.max.z
        };
        let mut pairs = Vec::new();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                if overlap(&boxes[i], &boxes[j]) {
                    pairs.push(CandidatePair::new(i as u32, j as u32));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    /// Sorts a pair list into the canonical order used for comparison.
    fn sorted(mut pairs: Vec<CandidatePair>) -> Vec<CandidatePair> {
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn empty_and_single_have_no_pairs() {
        assert!(cpu_bvh_pairs(&cpu_build_lbvh(&[]), 16)
            .expect("ok")
            .is_empty());
        let one = cpu_build_lbvh(&[box_at(0.0, 0.0, 0.0)]);
        assert!(cpu_bvh_pairs(&one, 16).expect("ok").is_empty());
    }

    #[test]
    fn two_overlapping_boxes_are_one_pair() {
        let boxes = [box_at(0.0, 0.0, 0.0), box_at(0.5, 0.0, 0.0)];
        let tree = cpu_build_lbvh(&boxes);
        let got = sorted(cpu_bvh_pairs(&tree, 16).expect("ok"));
        assert_eq!(got, brute_force(&boxes));
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn two_disjoint_boxes_are_no_pair() {
        let boxes = [box_at(0.0, 0.0, 0.0), box_at(10.0, 0.0, 0.0)];
        let tree = cpu_build_lbvh(&boxes);
        assert!(cpu_bvh_pairs(&tree, 16).expect("ok").is_empty());
    }

    #[test]
    fn touching_faces_count_as_overlap() {
        // Unit boxes one apart in x share the plane x = 0.5.
        let boxes = [box_at(0.0, 0.0, 0.0), box_at(1.0, 0.0, 0.0)];
        let tree = cpu_build_lbvh(&boxes);
        assert_eq!(cpu_bvh_pairs(&tree, 16).expect("ok").len(), 1);
    }

    #[test]
    fn dense_cluster_matches_brute_force() {
        // A 3x3x3 lattice of unit boxes half a unit apart: heavy overlap.
        let mut boxes = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    boxes.push(box_at(x as f32 * 0.5, y as f32 * 0.5, z as f32 * 0.5));
                }
            }
        }
        let tree = cpu_build_lbvh(&boxes);
        let got = sorted(cpu_bvh_pairs(&tree, 4096).expect("ok"));
        assert_eq!(got, brute_force(&boxes));
    }

    #[test]
    fn random_scene_matches_brute_force() {
        // Deterministic pseudo-random boxes; xorshift keeps the test dependency
        // free.
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let coord =
            |lo: f32, span: f32, r: u64| lo + ((r >> 40) as f32 / (1u64 << 24) as f32) * span;
        let mut boxes = Vec::new();
        for _ in 0..200 {
            let c = Vec3::new(
                coord(-5.0, 10.0, next()),
                coord(-5.0, 10.0, next()),
                coord(-5.0, 10.0, next()),
            );
            let h = Vec3::splat(0.6);
            boxes.push(Aabb::new(c - h, c + h));
        }
        let tree = cpu_build_lbvh(&boxes);
        let got = sorted(cpu_bvh_pairs(&tree, 1 << 20).expect("ok"));
        assert_eq!(got, brute_force(&boxes));
    }

    #[test]
    fn duplicate_positions_match_brute_force() {
        // Many coincident boxes stress equal Morton codes and full overlap.
        let boxes = vec![box_at(2.0, 2.0, 2.0); 12];
        let tree = cpu_build_lbvh(&boxes);
        let got = sorted(cpu_bvh_pairs(&tree, 4096).expect("ok"));
        assert_eq!(got, brute_force(&boxes));
        // 12 identical boxes: C(12, 2) = 66 pairs.
        assert_eq!(got.len(), 66);
    }

    #[test]
    fn capacity_overflow_is_reported() {
        let boxes = vec![box_at(0.0, 0.0, 0.0); 5];
        let tree = cpu_build_lbvh(&boxes);
        // C(5, 2) = 10 pairs; a capacity of 9 must overflow.
        assert_eq!(
            cpu_bvh_pairs(&tree, 9),
            Err(BvhQueryError::PairCapacityExceeded { capacity: 9 })
        );
        assert!(cpu_bvh_pairs(&tree, 10).is_ok());
    }
}
