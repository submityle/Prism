//! World-owned collision attachments for reduced-coordinate bodies.

use std::collections::BTreeMap;

use super::ArticulationHandle;
use crate::collider::{ColliderHandle, PhysicsMaterial};
use crate::math::transform::Isometry;

/// A collision attachment identity. Identities are never recycled in a world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArticulationColliderId(u64);

/// Collision geometry attached to a link, or to the articulation base.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationColliderDesc {
    /// Generational identity of the owning articulation.
    pub articulation: ArticulationHandle,
    /// Link index; `None` attaches to the base (including a fixed base).
    pub link: Option<usize>,
    /// Shared, potentially compound, narrow-phase geometry.
    pub shape: ColliderHandle,
    /// Collider pose relative to the link/base frame.
    pub local_pose: Isometry,
    /// Contact friction and restitution.
    pub material: PhysicsMaterial,
    /// Opt into self collision. Both attachments must opt in; multiple
    /// colliders attached to the very same body never collide with each other.
    pub self_collision: bool,
}

impl ArticulationColliderDesc {
    /// Attaches geometry at the body origin, with default material and no
    /// self collision. Registration validates the owner, link and shape.
    #[must_use]
    pub fn new(
        articulation: ArticulationHandle,
        link: Option<usize>,
        shape: ColliderHandle,
    ) -> Self {
        Self {
            articulation,
            link,
            shape,
            local_pose: Isometry::IDENTITY,
            material: PhysicsMaterial::DEFAULT,
            self_collision: false,
        }
    }
}

/// A rejected collision attachment leaves the world's registry unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArticulationColliderError {
    /// The articulation handle is stale or its tree is degraded.
    InvalidArticulation,
    /// The requested link does not exist.
    InvalidLink,
    /// The shared shape handle is stale.
    InvalidShape,
    /// The pose is non-finite or its quaternion is not normalized.
    InvalidPose,
    /// Friction is negative/non-finite or restitution is outside `[0, 1]`.
    InvalidMaterial,
    /// All non-recycled identities have been used.
    IdentityExhausted,
}

/// Deterministically ordered attachments, included in full-world snapshots.
#[derive(Clone, Debug, Default)]
pub struct ArticulationColliderSet {
    next_id: u64,
    entries: BTreeMap<ArticulationColliderId, ArticulationColliderDesc>,
}

impl ArticulationColliderSet {
    /// Number of live attachments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry contains no attachments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up an immutable attachment. Replace an attachment by explicitly
    /// removing and registering it, so all authoring changes are validated.
    #[must_use]
    pub fn get(&self, id: ArticulationColliderId) -> Option<&ArticulationColliderDesc> {
        self.entries.get(&id)
    }

    /// Iterates in stable identity order.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (ArticulationColliderId, &ArticulationColliderDesc)> {
        self.entries.iter().map(|(id, desc)| (*id, desc))
    }

    /// Removes one attachment. The identity cannot address a future entry.
    pub fn remove(&mut self, id: ArticulationColliderId) -> bool {
        self.entries.remove(&id).is_some()
    }

    pub(crate) fn insert(
        &mut self,
        desc: ArticulationColliderDesc,
    ) -> Result<ArticulationColliderId, ArticulationColliderError> {
        let next = self
            .next_id
            .checked_add(1)
            .ok_or(ArticulationColliderError::IdentityExhausted)?;
        let id = ArticulationColliderId(self.next_id);
        self.entries.insert(id, desc);
        self.next_id = next;
        Ok(id)
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&ArticulationColliderDesc) -> bool) {
        self.entries.retain(|_, desc| keep(desc));
    }
}
