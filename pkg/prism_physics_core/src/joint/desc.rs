//! The value type used to create a joint.
//!
//! [`JointDesc`] bundles the two [`JointAnchor`]s with a [`JointKind`] and an
//! enabled flag, and offers convenience constructors for each joint family so
//! callers rarely have to spell out the anchors and kind separately.
//!
//! # Provenance
//!
//! Original builder type. It contains **no Unreal Engine source or derived
//! code**.

use crate::joint::anchor::JointAnchor;
use crate::joint::kind::{
    DistanceJoint, FixedJoint, JointKind, PrismaticJoint, RevoluteJoint, SphericalJoint,
};
use crate::state::handle::BodyHandle;
use glam::Vec3;

/// A description of a joint to be inserted into a
/// [`JointStorage`](crate::joint::JointStorage).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JointDesc {
    /// Attachment on the first body.
    pub anchor_a: JointAnchor,
    /// Attachment on the second body.
    pub anchor_b: JointAnchor,
    /// The constraint family and its parameters.
    pub kind: JointKind,
    /// Whether the joint is solved. Disabled joints are skipped entirely.
    pub enabled: bool,
}

impl JointDesc {
    /// Creates a joint description from explicit anchors and kind.
    #[must_use]
    pub const fn new(anchor_a: JointAnchor, anchor_b: JointAnchor, kind: JointKind) -> JointDesc {
        JointDesc {
            anchor_a,
            anchor_b,
            kind,
            enabled: true,
        }
    }

    /// Enables or disables the joint and returns the modified description.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> JointDesc {
        self.enabled = enabled;
        self
    }

    /// Convenience: a rigid weld between two bodies at the given anchors.
    #[must_use]
    pub const fn fixed(anchor_a: JointAnchor, anchor_b: JointAnchor) -> JointDesc {
        JointDesc::new(
            anchor_a,
            anchor_b,
            JointKind::Fixed(FixedJoint { compliance: 0.0 }),
        )
    }

    /// Convenience: a length-limited link between two body-local points.
    #[must_use]
    pub const fn distance(
        body_a: BodyHandle,
        local_a: Vec3,
        body_b: BodyHandle,
        local_b: Vec3,
        joint: DistanceJoint,
    ) -> JointDesc {
        JointDesc::new(
            JointAnchor::at_point(body_a, local_a),
            JointAnchor::at_point(body_b, local_b),
            JointKind::Distance(joint),
        )
    }

    /// Convenience: a ball-and-socket joint at the given anchors.
    #[must_use]
    pub const fn spherical(anchor_a: JointAnchor, anchor_b: JointAnchor) -> JointDesc {
        JointDesc::new(
            anchor_a,
            anchor_b,
            JointKind::Spherical(SphericalJoint { compliance: 0.0 }),
        )
    }

    /// Convenience: a hinge joint at the given anchors.
    #[must_use]
    pub const fn revolute(
        anchor_a: JointAnchor,
        anchor_b: JointAnchor,
        joint: RevoluteJoint,
    ) -> JointDesc {
        JointDesc::new(anchor_a, anchor_b, JointKind::Revolute(joint))
    }

    /// Convenience: a slider joint at the given anchors.
    #[must_use]
    pub const fn prismatic(
        anchor_a: JointAnchor,
        anchor_b: JointAnchor,
        joint: PrismaticJoint,
    ) -> JointDesc {
        JointDesc::new(anchor_a, anchor_b, JointKind::Prismatic(joint))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_enabled() {
        let d = JointDesc::fixed(
            JointAnchor::at_center(BodyHandle::INVALID),
            JointAnchor::at_center(BodyHandle::INVALID),
        );
        assert!(d.enabled);
    }

    #[test]
    fn distance_helper_sets_points() {
        let d = JointDesc::distance(
            BodyHandle::INVALID,
            Vec3::X,
            BodyHandle::INVALID,
            Vec3::NEG_X,
            DistanceJoint::rigid(2.0),
        );
        assert_eq!(d.anchor_a.local_point, Vec3::X);
        assert_eq!(d.anchor_b.local_point, Vec3::NEG_X);
        assert!(matches!(d.kind, JointKind::Distance(_)));
    }
}
