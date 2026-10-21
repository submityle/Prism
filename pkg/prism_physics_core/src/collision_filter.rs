//! World-owned rigid collision exclusions with linear retained storage.
//!
//! A body belongs to at most one explicit group. Within a group both bodies
//! must opt into self collision; other groups and unregistered bodies are
//! unaffected. Group membership is independent of the surviving joint graph.

use alloc::collections::{BTreeMap, BTreeSet};

use crate::state::{handle::BodyHandle, storage::BodyStorage};

/// Opaque world-local identity of one installed exclusion group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollisionExclusionGroupId(u64);

/// An exclusion group could not be installed. Installation is atomic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionExclusionError {
    /// At least one member is required.
    EmptyGroup,
    /// A member does not refer to a live body in this world.
    InvalidBody(BodyHandle),
    /// The same body appears twice in the request.
    DuplicateBody(BodyHandle),
    /// A live body already belongs to an exclusion group.
    AlreadyGrouped(BodyHandle),
    /// A merged assembly proxy cannot represent its members' group policy.
    MergedBody(BodyHandle),
    /// The world-local group identity allocator is exhausted.
    GroupIdExhausted,
}

/// One body's persistent group policy, keyed by its full generational handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionExclusionMembership {
    /// World-local group shared by all original members.
    pub group: CollisionExclusionGroupId,
    /// Both members must opt in for a same-group contact.
    pub self_collision: bool,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    body: BodyHandle,
    membership: CollisionExclusionMembership,
}

/// One entry per registered body, never an expanded list of excluded pairs.
///
/// Queries cost O(log n); installing n members costs O(n log n). Entries are
/// indexed by slot but validate the generation, so slot reuse never inherits
/// a removed body's policy. No per-group allocation survives its last member.
#[derive(Clone, Debug, Default)]
pub struct CollisionExclusions {
    members: BTreeMap<u32, Entry>,
    next_group: u64,
}

impl CollisionExclusions {
    /// Number of retained membership records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    #[must_use]
    pub fn membership(&self, body: BodyHandle) -> Option<CollisionExclusionMembership> {
        self.members
            .get(&body.index())
            .filter(|entry| entry.body == body)
            .map(|entry| entry.membership)
    }

    /// Whether the policy permits a pair. A body never collides with itself.
    #[must_use]
    pub fn allows_pair(&self, a: BodyHandle, b: BodyHandle) -> bool {
        self.allows_leaf_pair(a, b, None, None)
    }

    /// Exact narrow-phase/CCD gate. Body-level membership may be a conservative
    /// OR of several attachment policies; explicit leaf flags refine it only
    /// within the same group. They never suppress external-body collisions.
    #[must_use]
    pub fn allows_leaf_pair(
        &self,
        a: BodyHandle,
        b: BodyHandle,
        a_self_collision: Option<bool>,
        b_self_collision: Option<bool>,
    ) -> bool {
        if a == b {
            return false;
        }
        if self.members.is_empty() {
            return true;
        }
        match (self.membership(a), self.membership(b)) {
            (Some(a), Some(b)) if a.group == b.group => {
                a_self_collision.unwrap_or(a.self_collision)
                    && b_self_collision.unwrap_or(b.self_collision)
            }
            _ => true,
        }
    }

    /// Checks only whether another group ID can be issued. This never mutates
    /// allocator or membership state, so handoff can preflight before creating
    /// shapes/bodies and install fresh, unique handles without a late failure.
    pub(crate) fn validate_group_capacity(&self) -> Result<(), CollisionExclusionError> {
        self.next_group
            .checked_add(1)
            .map(|_| ())
            .ok_or(CollisionExclusionError::GroupIdExhausted)
    }

    pub(crate) fn install(
        &mut self,
        bodies: &BodyStorage,
        members: &[(BodyHandle, bool)],
    ) -> Result<CollisionExclusionGroupId, CollisionExclusionError> {
        if members.is_empty() {
            return Err(CollisionExclusionError::EmptyGroup);
        }
        self.validate_group_capacity()?;
        let mut seen = BTreeSet::new();
        for &(body, _) in members {
            if !bodies.contains(body) {
                return Err(CollisionExclusionError::InvalidBody(body));
            }
            if !seen.insert(body.index()) {
                return Err(CollisionExclusionError::DuplicateBody(body));
            }
            if self.membership(body).is_some() {
                return Err(CollisionExclusionError::AlreadyGrouped(body));
            }
        }
        let group = CollisionExclusionGroupId(self.next_group);
        for &(body, self_collision) in members {
            self.members.insert(
                body.index(),
                Entry {
                    body,
                    membership: CollisionExclusionMembership {
                        group,
                        self_collision,
                    },
                },
            );
        }
        self.next_group += 1;
        Ok(group)
    }

    pub(crate) fn remove(&mut self, body: BodyHandle) {
        if self.membership(body).is_some() {
            self.members.remove(&body.index());
        }
    }

    // Public BodyStorage allows low-level removals bypassing PhysicsWorld.
    // Reclaim their records at a fixed-step boundary; queries are already
    // generation-safe even before this cleanup runs.
    pub(crate) fn retain_live(&mut self, bodies: &BodyStorage) {
        self.members.retain(|_, entry| bodies.contains(entry.body));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn capacity_preflight_is_read_only_and_exhaustion_is_atomic() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::X));
        let mut policy = CollisionExclusions {
            next_group: u64::MAX - 1,
            ..CollisionExclusions::default()
        };
        let before = format!("{policy:?}");
        for _ in 0..2 {
            assert_eq!(policy.validate_group_capacity(), Ok(()));
            assert_eq!(format!("{policy:?}"), before);
        }
        policy.install(&bodies, &[(a, false)]).unwrap();
        let exhausted = format!("{policy:?}");
        assert_eq!(
            policy.validate_group_capacity(),
            Err(CollisionExclusionError::GroupIdExhausted)
        );
        assert_eq!(
            policy.install(&bodies, &[(b, false)]),
            Err(CollisionExclusionError::GroupIdExhausted)
        );
        assert_eq!(format!("{policy:?}"), exhausted);
        assert_eq!(policy.len(), 1);
        assert!(policy.membership(b).is_none());
    }
}
