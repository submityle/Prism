//! Grouping of contact and joint constraints into simulation islands.
//!
//! An *island* is a connected component of dynamic bodies linked by contacts or
//! joints. Because two islands never share a dynamic body, their constraint
//! solves are independent: solving them one after another (or, later, in
//! parallel) is numerically identical to a single global Gauss-Seidel pass.
//! Static and kinematic bodies act as *separators* — they are read-only during
//! the solve, so a static body (such as the ground plane) may be referenced by
//! many islands without merging them.
//!
//! [`SolveIslands`] builds this partition once per sub-step and exposes, per
//! island, the indices of the contact constraints and the active joints that
//! belong to it. The heavy per-body columns are left untouched; only small
//! index lists are produced.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! thin grouping layer over the crate's own union-find
//! ([`IslandBuilder`](crate::island::IslandBuilder)).

use crate::island::{index_buckets::IndexBuckets, IslandBuilder};
use crate::joint::ragdoll::RagdollConstraint;
use crate::joint::Joint;
use crate::solver::xpbd::contact_constraint::ContactConstraint;
use crate::state::view::BodySolverView;

/// A per-sub-step partition of the active constraints into islands.
///
/// Each island is a compact bucket holding the indices of the contact
/// constraints and active joints that connect its dynamic bodies. Buckets are
/// created in order of first appearance while scanning contacts then joints, so
/// the partition is deterministic.
#[derive(Clone, Debug, Default)]
pub struct SolveIslands {
    contact_indices: IndexBuckets,
    joint_indices: IndexBuckets,
    ragdoll_indices: IndexBuckets,
    /// Sorted, de-duplicated dynamic body slots per island. Used by the sleep
    /// bookkeeping to test and update every dynamic member of an island.
    member_slots: IndexBuckets,
}

impl SolveIslands {
    /// Builds the island partition for the current sub-step.
    ///
    /// `constraints` is the slice of contact constraints produced by
    /// [`ContactConstraint::build`](crate::solver::xpbd::contact_constraint::ContactConstraint::build);
    /// `joints` is the collected slice of active joints (in storage order).
    /// Only dynamic-dynamic connections merge bodies; a constraint touching a
    /// single dynamic body is filed under that body's island.
    #[must_use]
    pub fn build(
        view: &BodySolverView<'_>,
        constraints: &[ContactConstraint],
        joints: &[&Joint],
    ) -> SolveIslands {
        Self::build_with_ragdolls(view, constraints, joints, &[])
    }

    /// Builds the island partition including live ragdoll Swing-Twist links.
    #[must_use]
    pub fn build_with_ragdolls(
        view: &BodySolverView<'_>,
        constraints: &[ContactConstraint],
        joints: &[&Joint],
        ragdolls: &[RagdollConstraint],
    ) -> SolveIslands {
        let slot_count = view.slot_count();
        let mut builder = IslandBuilder::new(slot_count);

        // Merge dynamic bodies that share a contact.
        for constraint in constraints {
            if view.is_dynamic(constraint.slot_a) && view.is_dynamic(constraint.slot_b) {
                builder.union(constraint.slot_a, constraint.slot_b);
            }
        }
        // Merge dynamic bodies that share a joint.
        for joint in joints {
            let (slot_a, slot_b) = joint_slots(joint);
            if slot_a < slot_count
                && slot_b < slot_count
                && view.is_dynamic(slot_a)
                && view.is_dynamic(slot_b)
            {
                builder.union(slot_a, slot_b);
            }
        }
        // Ragdoll links are solver-owned Swing-Twist constraints and must
        // participate in the same connectivity graph as ordinary joints.
        for ragdoll in ragdolls {
            let slot_a = ragdoll.body_a.index() as usize;
            let slot_b = ragdoll.body_b.index() as usize;
            if slot_a < slot_count
                && slot_b < slot_count
                && view.is_dynamic(slot_a)
                && view.is_dynamic(slot_b)
            {
                builder.union(slot_a, slot_b);
            }
        }

        // Address roots directly: materializing an IslandSet here used to
        // allocate member lists for every body, including free/static bodies
        // that never participate in this solve. Dense buckets retain the
        // original first-contact, then joint, then ragdoll appearance order.
        let mut bucket_of = vec![usize::MAX; slot_count];
        let mut counts: Vec<[usize; 4]> = Vec::new();
        let mut count_constraint = |root: usize, kind: usize| {
            if bucket_of[root] == usize::MAX {
                bucket_of[root] = counts.len();
                counts.push([0; 4]);
            }
            counts[bucket_of[root]][kind] += 1;
        };
        // Establish ordering and counts in one scan, without per-constraint
        // temporary storage.
        for constraint in constraints {
            let Some(dynamic_slot) = dynamic_slot_of(view, constraint.slot_a, constraint.slot_b)
            else {
                continue;
            };
            count_constraint(builder.find(dynamic_slot), 0);
        }
        for joint in joints {
            let (slot_a, slot_b) = joint_slots(joint);
            if slot_a >= slot_count || slot_b >= slot_count {
                continue;
            }
            let Some(dynamic_slot) = dynamic_slot_of(view, slot_a, slot_b) else {
                continue;
            };
            count_constraint(builder.find(dynamic_slot), 1);
        }
        for ragdoll in ragdolls {
            let slot_a = ragdoll.body_a.index() as usize;
            let slot_b = ragdoll.body_b.index() as usize;
            if slot_a >= slot_count || slot_b >= slot_count {
                continue;
            }
            let Some(dynamic_slot) = dynamic_slot_of(view, slot_a, slot_b) else {
                continue;
            };
            count_constraint(builder.find(dynamic_slot), 2);
        }
        // Resolve each live dynamic slot once. Ascending slot traversal gives
        // sorted, unique members without sorting duplicated contact endpoints.
        let mut body_bucket = vec![usize::MAX; slot_count];
        for (slot, bucket) in body_bucket.iter_mut().enumerate() {
            if view.is_dynamic(slot) {
                *bucket = bucket_of[builder.find(slot)];
                if *bucket != usize::MAX {
                    counts[*bucket][3] += 1;
                }
            }
        }
        let bucket_for_pair = |a: usize, b: usize| {
            let first = body_bucket.get(a).copied().unwrap_or(usize::MAX);
            if first != usize::MAX {
                first
            } else {
                body_bucket.get(b).copied().unwrap_or(usize::MAX)
            }
        };
        // Counting and filling use identical stable scans. The four packed
        // lists each own just offsets + indices, independent of island count.
        let mut contact_indices = IndexBuckets::with_counts(counts.iter().map(|c| c[0]));
        let mut joint_indices = IndexBuckets::with_counts(counts.iter().map(|c| c[1]));
        let mut ragdoll_indices = IndexBuckets::with_counts(counts.iter().map(|c| c[2]));
        let mut member_slots = IndexBuckets::with_counts(counts.iter().map(|c| c[3]));
        counts.fill([0; 4]);
        for (index, constraint) in constraints.iter().enumerate() {
            let bucket = bucket_for_pair(constraint.slot_a, constraint.slot_b);
            if bucket != usize::MAX {
                contact_indices.set(bucket, counts[bucket][0], index);
                counts[bucket][0] += 1;
            }
        }
        for (index, joint) in joints.iter().enumerate() {
            let (a, b) = joint_slots(joint);
            if a < slot_count && b < slot_count {
                let bucket = bucket_for_pair(a, b);
                if bucket != usize::MAX {
                    joint_indices.set(bucket, counts[bucket][1], index);
                    counts[bucket][1] += 1;
                }
            }
        }
        for (index, ragdoll) in ragdolls.iter().enumerate() {
            let a = ragdoll.body_a.index() as usize;
            let b = ragdoll.body_b.index() as usize;
            if a < slot_count && b < slot_count {
                let bucket = bucket_for_pair(a, b);
                if bucket != usize::MAX {
                    ragdoll_indices.set(bucket, counts[bucket][2], index);
                    counts[bucket][2] += 1;
                }
            }
        }
        for (slot, bucket) in body_bucket.into_iter().enumerate() {
            if bucket != usize::MAX {
                member_slots.set(bucket, counts[bucket][3], slot);
                counts[bucket][3] += 1;
            }
        }
        SolveIslands {
            contact_indices,
            joint_indices,
            ragdoll_indices,
            member_slots,
        }
    }

    /// Returns the number of islands that own at least one constraint or joint.
    #[must_use]
    pub fn island_count(&self) -> usize {
        self.contact_indices.len()
    }

    /// Returns `true` when no island owns any constraint or joint.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contact_indices.is_empty()
    }

    /// Returns the contact-constraint indices belonging to `island`.
    ///
    /// Returns an empty slice when `island` is out of range.
    #[must_use]
    pub fn contacts(&self, island: usize) -> &[usize] {
        self.contact_indices.get(island)
    }

    /// Returns the active-joint indices belonging to `island`.
    ///
    /// Returns an empty slice when `island` is out of range.
    #[must_use]
    pub fn joints(&self, island: usize) -> &[usize] {
        self.joint_indices.get(island)
    }

    /// Returns the ragdoll Swing-Twist constraint indices belonging to `island`.
    #[must_use]
    pub fn ragdolls(&self, island: usize) -> &[usize] {
        self.ragdoll_indices.get(island)
    }

    /// Returns the sorted dynamic body slots belonging to `island`.
    ///
    /// Returns an empty slice when `island` is out of range. Static and
    /// kinematic bodies are never members because they are read-only during the
    /// solve and cannot sleep.
    #[must_use]
    pub fn members(&self, island: usize) -> &[usize] {
        self.member_slots.get(island)
    }
}

/// Extracts the two body slot indices a joint connects.
fn joint_slots(joint: &Joint) -> (usize, usize) {
    (
        joint.anchor_a.body.index() as usize,
        joint.anchor_b.body.index() as usize,
    )
}

/// Returns whichever of `slot_a`/`slot_b` is a dynamic body, preferring
/// `slot_a`. Returns `None` when neither is dynamic.
fn dynamic_slot_of(view: &BodySolverView<'_>, slot_a: usize, slot_b: usize) -> Option<usize> {
    if view.is_dynamic(slot_a) {
        Some(slot_a)
    } else if view.is_dynamic(slot_b) {
        Some(slot_b)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collide::contact::{ContactManifold, ContactPoint};
    use crate::solver::xpbd::contact_constraint::ContactConstraint;
    use crate::state::body::{BodyDesc, BodyKind};
    use crate::state::handle::BodyHandle;
    use crate::state::storage::BodyStorage;
    use glam::Vec3;

    fn storage_with_kinds(kinds: &[BodyKind]) -> BodyStorage {
        let mut storage = BodyStorage::new();
        for kind in kinds {
            let desc = match kind {
                BodyKind::Dynamic => BodyDesc::dynamic_at(Vec3::ZERO),
                BodyKind::Static => BodyDesc::static_at(Vec3::ZERO),
                BodyKind::Kinematic => {
                    let mut d = BodyDesc::dynamic_at(Vec3::ZERO);
                    d.kind = BodyKind::Kinematic;
                    d
                }
            };
            storage.insert(desc);
        }
        storage
    }

    fn manifold(a: BodyHandle, b: BodyHandle) -> ContactManifold {
        let mut m = ContactManifold::new(a, b, Vec3::Y);
        m.push(ContactPoint::new(Vec3::ZERO, Vec3::ZERO, 0.01));
        m
    }

    /// Builds real constraints from `pairs` of slot indices, then partitions.
    fn partition(kinds: &[BodyKind], pairs: &[(usize, usize)]) -> SolveIslands {
        let mut storage = storage_with_kinds(kinds);
        let handles: Vec<BodyHandle> = (0..kinds.len())
            .map(|slot| storage.handle_at_slot(slot).expect("live slot"))
            .collect();
        let manifolds: Vec<ContactManifold> = pairs
            .iter()
            .map(|&(a, b)| manifold(handles[a], handles[b]))
            .collect();
        let view = storage.solver_view_mut();
        let constraints = ContactConstraint::build(&view, &manifolds);
        SolveIslands::build(&view, &constraints, &[])
    }

    #[test]
    fn two_dynamic_chains_form_two_islands() {
        let islands = partition(&[BodyKind::Dynamic; 4], &[(0, 1), (2, 3)]);
        assert_eq!(islands.island_count(), 2);
        assert_eq!(islands.contacts(0), &[0]);
        assert_eq!(islands.contacts(1), &[1]);
    }

    #[test]
    fn shared_static_body_does_not_merge_islands() {
        let islands = partition(
            &[BodyKind::Static, BodyKind::Dynamic, BodyKind::Dynamic],
            &[(1, 0), (2, 0)],
        );
        assert_eq!(islands.island_count(), 2);
    }

    #[test]
    fn transitive_contacts_merge_into_one_island() {
        let islands = partition(&[BodyKind::Dynamic; 3], &[(0, 1), (1, 2)]);
        assert_eq!(islands.island_count(), 1);
        assert_eq!(islands.contacts(0), &[0, 1]);
        // All three dynamic bodies are members of the single island, listed once
        // each in sorted order.
        assert_eq!(islands.members(0), &[0, 1, 2]);
    }

    #[test]
    fn members_exclude_static_bodies() {
        // Two dynamic bodies each resting on a shared static ground form two
        // islands; the static slot (0) is never a member of either.
        let islands = partition(
            &[BodyKind::Static, BodyKind::Dynamic, BodyKind::Dynamic],
            &[(1, 0), (2, 0)],
        );
        assert_eq!(islands.island_count(), 2);
        assert_eq!(islands.members(0), &[1]);
        assert_eq!(islands.members(1), &[2]);
        assert_eq!(islands.members(5), &[] as &[usize]);
    }

    #[test]
    fn no_constraints_yields_no_islands() {
        let islands = partition(&[BodyKind::Dynamic; 2], &[]);
        assert!(islands.is_empty());
        assert_eq!(islands.island_count(), 0);
        assert_eq!(islands.contacts(0), &[] as &[usize]);
    }

    #[test]
    fn ragdoll_link_merges_dynamic_bodies_and_is_indexed() {
        let mut storage = storage_with_kinds(&[BodyKind::Dynamic, BodyKind::Dynamic]);
        let a = storage.handle_at_slot(0).expect("body a");
        let b = storage.handle_at_slot(1).expect("body b");
        let joint = crate::joint::SwingTwistJoint::try_new(
            crate::joint::SwingTwistConfig::symmetric(Vec3::X, Vec3::X, Vec3::Y, Vec3::Y, 0.5, 0.5),
        )
        .expect("valid swing-twist");
        let ragdolls = vec![RagdollConstraint {
            body_a: a,
            body_b: b,
            local_point_a: Vec3::ZERO,
            local_point_b: Vec3::ZERO,
            joint,
        }];
        let view = storage.solver_view_mut();
        let islands = SolveIslands::build_with_ragdolls(&view, &[], &[], &ragdolls);
        assert_eq!(islands.island_count(), 1);
        assert_eq!(islands.ragdolls(0), &[0]);
        assert_eq!(islands.members(0), &[0, 1]);
    }
}
