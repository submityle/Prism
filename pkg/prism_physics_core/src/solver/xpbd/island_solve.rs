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

use crate::island::{IslandBuilder, IslandId};
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
    contact_indices: Vec<Vec<usize>>,
    joint_indices: Vec<Vec<usize>>,
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

        let set = builder.build();
        // Compact map from raw IslandId to a dense bucket index. Only islands
        // that actually own a constraint or joint get a bucket.
        let mut bucket_of: Vec<Option<usize>> = vec![None; set.island_count()];
        let mut contact_indices: Vec<Vec<usize>> = Vec::new();
        let mut joint_indices: Vec<Vec<usize>> = Vec::new();

        let mut bucket_for = |island: IslandId,
                              contacts: &mut Vec<Vec<usize>>,
                              jts: &mut Vec<Vec<usize>>|
         -> usize {
            let raw = island.0 as usize;
            match bucket_of[raw] {
                Some(b) => b,
                None => {
                    let b = contacts.len();
                    bucket_of[raw] = Some(b);
                    contacts.push(Vec::new());
                    jts.push(Vec::new());
                    b
                }
            }
        };

        for (index, constraint) in constraints.iter().enumerate() {
            let Some(dynamic_slot) = dynamic_slot_of(view, constraint.slot_a, constraint.slot_b)
            else {
                continue;
            };
            let bucket = bucket_for(
                set.island_of(dynamic_slot),
                &mut contact_indices,
                &mut joint_indices,
            );
            contact_indices[bucket].push(index);
        }

        for (index, joint) in joints.iter().enumerate() {
            let (slot_a, slot_b) = joint_slots(joint);
            if slot_a >= slot_count || slot_b >= slot_count {
                continue;
            }
            let Some(dynamic_slot) = dynamic_slot_of(view, slot_a, slot_b) else {
                continue;
            };
            let bucket = bucket_for(
                set.island_of(dynamic_slot),
                &mut contact_indices,
                &mut joint_indices,
            );
            joint_indices[bucket].push(index);
        }

        SolveIslands {
            contact_indices,
            joint_indices,
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
        self.contact_indices.get(island).map_or(&[], Vec::as_slice)
    }

    /// Returns the active-joint indices belonging to `island`.
    ///
    /// Returns an empty slice when `island` is out of range.
    #[must_use]
    pub fn joints(&self, island: usize) -> &[usize] {
        self.joint_indices.get(island).map_or(&[], Vec::as_slice)
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
    }

    #[test]
    fn no_constraints_yields_no_islands() {
        let islands = partition(&[BodyKind::Dynamic; 2], &[]);
        assert!(islands.is_empty());
        assert_eq!(islands.island_count(), 0);
        assert_eq!(islands.contacts(0), &[] as &[usize]);
    }
}
