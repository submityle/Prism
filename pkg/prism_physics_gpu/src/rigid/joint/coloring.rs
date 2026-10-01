//! Static-aware greedy colouring of the rigid-body joint graph.
//!
//! Projecting joint constraints in parallel is only race-free when the joints
//! in a batch write disjoint bodies. [`JointColouring`] partitions a joint set
//! into *batches* (graph colours) such that no two joints in the same batch
//! share a **movable** body, so every batch is one parallel dispatch in which
//! each movable body's transform is written by at most one invocation. Solving
//! the batches in sequence (a barrier between them) reproduces a Gauss-Seidel
//! sweep over the joints, which is what lets the `CPU` golden and the `GPU`
//! kernel — both walking the identical batch order over the identical reordered
//! joint list — agree.
//!
//! # Why static bodies do not conflict
//!
//! An immovable body (zero inverse mass *and* zero inverse inertia) is never
//! written by the solver: the positional and angular corrections it would
//! receive are scaled by its zero inverse mass and inertia. Any number of
//! joints in the same batch may therefore share the *same* static body — the
//! fixed world anchor every pendulum in a rack hangs from, say — without racing.
//! Treating static bodies as non-conflicting keeps a scene of many bodies all
//! pinned to one static frame from collapsing into one serial batch per joint;
//! only the dynamic bodies drive the colour count.
//!
//! The colouring is a deterministic first-fit (greedy) assignment: joints are
//! visited in input order and each takes the lowest batch not already used by a
//! joint sharing one of its movable bodies. This mirrors the contact-graph
//! colouring in [`RigidContactColouring`](super::super::RigidContactColouring);
//! the two are kept as separate concepts because a joint set and a contact set
//! are coloured and solved independently.
//!
//! Provenance: textbook greedy (first-fit) graph colouring, specialised to skip
//! the never-written static bodies of a rigid joint graph. No Unreal Engine
//! source or derived code.

use super::super::config::RigidError;
use super::spherical::SphericalJoint;

/// The maximum number of batches the first-fit packing supports.
///
/// Each body's used-batch set is a single `u64` bitmask, so the ceiling is
/// `64`. Physical articulated scenes never approach this for the dynamic bodies
/// that actually drive the count, so the limit only rejects pathological inputs.
pub const MAX_JOINT_BATCHES: u32 = 64;

/// A batch-partitioned view of a rigid joint set.
///
/// [`order`](JointColouring::order) lists the original joint indices grouped by
/// batch; [`ranges`](JointColouring::ranges) gives the `[start, end)` half-open
/// slice of `order` occupied by each batch. Solving batch `b` means visiting
/// `order[ranges[b].0 .. ranges[b].1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointColouring {
    /// Original joint indices, grouped contiguously by batch.
    order: Vec<u32>,
    /// Half-open `[start, end)` ranges into `order`, one per batch.
    ranges: Vec<(u32, u32)>,
}

impl JointColouring {
    /// Colours `joints` so that same-batch joints share no movable body.
    ///
    /// `movable` holds one flag per body (`true` when the body has a non-zero
    /// inverse mass or inverse inertia, i.e. the solver may write it); its
    /// length bounds the valid body indices. A joint's two body indices are
    /// range-checked, and only its movable endpoints participate in the conflict
    /// test, so joints sharing only a static body may land in the same batch.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InconsistentState`] if a joint references a body
    /// `>= movable.len()`, or [`RigidError::TooManyJointBatches`] if the graph
    /// needs more than [`MAX_JOINT_BATCHES`] batches.
    pub fn build(
        joints: &[SphericalJoint],
        movable: &[bool],
    ) -> Result<JointColouring, RigidError> {
        let body_count = movable.len();
        // Per-body bitmask of batches already taken by an incident joint.
        let mut body_mask = vec![0u64; body_count];
        let mut batch_of = vec![0u32; joints.len()];
        let mut batch_count = 0u32;

        for (ji, joint) in joints.iter().enumerate() {
            let (a, b) = (joint.body_a as usize, joint.body_b as usize);
            if a >= body_count || b >= body_count {
                return Err(RigidError::InconsistentState {
                    reason: "joint references a body outside the state",
                });
            }
            // Only movable endpoints constrain the batch choice; a static body
            // is never written, so it is invisible to the conflict test.
            let mut taken = 0u64;
            if movable[a] {
                taken |= body_mask[a];
            }
            if movable[b] {
                taken |= body_mask[b];
            }
            let batch = lowest_free_bit(taken);
            if batch >= MAX_JOINT_BATCHES {
                return Err(RigidError::TooManyJointBatches {
                    batches: batch,
                    maximum: MAX_JOINT_BATCHES,
                });
            }
            let bit = 1u64 << batch;
            if movable[a] {
                body_mask[a] |= bit;
            }
            if movable[b] {
                body_mask[b] |= bit;
            }
            batch_of[ji] = batch;
            batch_count = batch_count.max(batch + 1);
        }

        Ok(JointColouring::group_by_batch(&batch_of, batch_count))
    }

    /// Groups joint indices contiguously by their assigned batch via a stable
    /// counting sort, preserving input order within a batch so the reordered
    /// list is deterministic.
    fn group_by_batch(batch_of: &[u32], batch_count: u32) -> JointColouring {
        let mut counts = vec![0u32; batch_count as usize];
        for &c in batch_of {
            counts[c as usize] += 1;
        }
        let mut ranges = Vec::with_capacity(batch_count as usize);
        let mut cursor = 0u32;
        for &count in &counts {
            ranges.push((cursor, cursor + count));
            cursor += count;
        }
        let mut fill = ranges.iter().map(|&(start, _)| start).collect::<Vec<_>>();
        let mut order = vec![0u32; batch_of.len()];
        for (ji, &c) in batch_of.iter().enumerate() {
            let slot = &mut fill[c as usize];
            order[*slot as usize] = ji as u32;
            *slot += 1;
        }
        JointColouring { order, ranges }
    }

    /// Number of batches used.
    #[must_use]
    pub fn batch_count(&self) -> u32 {
        self.ranges.len() as u32
    }

    /// The reordered joint indices, grouped by batch.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The half-open `[start, end)` ranges into [`order`](Self::order).
    #[must_use]
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    /// Reorders `items` into batch-grouped order, ready for upload.
    #[must_use]
    pub fn reorder<T: Copy>(&self, items: &[T]) -> Vec<T> {
        self.order.iter().map(|&i| items[i as usize]).collect()
    }
}

/// Returns the index of the lowest zero bit in `mask`, or `64` when full.
fn lowest_free_bit(mask: u64) -> u32 {
    (!mask).trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// A minimal joint coupling bodies `a` and `b` (anchors irrelevant to
    /// colouring).
    fn joint(a: u32, b: u32) -> SphericalJoint {
        SphericalJoint::new(a, b, Vec3::ZERO, Vec3::ZERO, 0.0)
    }

    #[test]
    fn lowest_free_bit_finds_first_gap() {
        assert_eq!(lowest_free_bit(0b0), 0);
        assert_eq!(lowest_free_bit(0b1), 1);
        assert_eq!(lowest_free_bit(0b1011), 2);
        assert_eq!(lowest_free_bit(u64::MAX), 64);
    }

    #[test]
    fn disjoint_dynamic_joints_share_one_batch() {
        let js = [joint(0, 1), joint(2, 3)];
        let movable = [true, true, true, true];
        let col = JointColouring::build(&js, &movable).unwrap();
        assert_eq!(col.batch_count(), 1);
        assert_eq!(col.ranges(), &[(0, 2)]);
    }

    #[test]
    fn chain_sharing_a_dynamic_body_needs_two_batches() {
        let js = [joint(0, 1), joint(1, 2)];
        let movable = [true, true, true];
        let col = JointColouring::build(&js, &movable).unwrap();
        assert_eq!(col.batch_count(), 2);
    }

    #[test]
    fn joints_sharing_only_a_static_body_share_one_batch() {
        // Body 0 is the static frame; bodies 1, 2, 3 hang from it. Every joint
        // shares body 0, but because it is static they all pack into one batch.
        let js = [joint(1, 0), joint(2, 0), joint(3, 0)];
        let movable = [false, true, true, true];
        let col = JointColouring::build(&js, &movable).unwrap();
        assert_eq!(col.batch_count(), 1);
        assert_eq!(col.ranges(), &[(0, 3)]);
    }

    #[test]
    fn same_batch_joints_never_share_a_movable_body() {
        let js = [
            joint(0, 1),
            joint(0, 2),
            joint(1, 2),
            joint(2, 3),
            joint(1, 3),
        ];
        let movable = [true, true, true, true];
        let col = JointColouring::build(&js, &movable).unwrap();
        for &(start, end) in col.ranges() {
            let mut seen = std::collections::HashSet::new();
            for &ji in &col.order()[start as usize..end as usize] {
                let j = js[ji as usize];
                assert!(seen.insert(j.body_a), "batch repeats movable body");
                assert!(seen.insert(j.body_b), "batch repeats movable body");
            }
        }
    }

    #[test]
    fn reorder_matches_order_permutation() {
        let js = [joint(0, 1), joint(1, 2), joint(3, 4)];
        let movable = [true, true, true, true, true];
        let col = JointColouring::build(&js, &movable).unwrap();
        let reordered = col.reorder(&js);
        for (k, &ji) in col.order().iter().enumerate() {
            assert_eq!(reordered[k], js[ji as usize]);
        }
    }

    #[test]
    fn out_of_range_body_is_rejected() {
        let js = [joint(0, 5)];
        let movable = [true, true];
        let err = JointColouring::build(&js, &movable).unwrap_err();
        assert!(matches!(err, RigidError::InconsistentState { .. }));
    }
}
