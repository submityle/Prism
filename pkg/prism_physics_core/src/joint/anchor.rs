//! Attachment frames that bind a joint to a pair of bodies.
//!
//! A joint connects two bodies at a chosen local point and reference frame on
//! each body. [`JointAnchor`] stores that binding: the body handle, the
//! attachment point expressed in the body's local frame (an offset from the
//! centre of mass), and a local reference orientation used by joints that also
//! constrain rotation (hinge, prismatic, fixed).
//!
//! # Provenance
//!
//! This is an original data type built from standard rigid-body-joint concepts.
//! It contains **no Unreal Engine source or derived code**.

use crate::state::handle::BodyHandle;
use glam::{Quat, Vec3};

/// The attachment of one side of a joint to a single body.
///
/// `local_point` and `local_frame` are expressed in the body's local frame, so
/// they follow the body rigidly as it moves. The world-space anchor point is
/// `position + orientation * local_point`; the world-space reference frame is
/// `orientation * local_frame`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JointAnchor {
    /// The body this side of the joint attaches to.
    pub body: BodyHandle,
    /// Attachment point in the body's local frame (offset from the centre of
    /// mass).
    pub local_point: Vec3,
    /// Reference frame orientation in the body's local frame. Used by joints
    /// that constrain relative rotation (hinge axis, prismatic axis, fixed
    /// frame alignment).
    pub local_frame: Quat,
}

impl JointAnchor {
    /// Creates an anchor with an explicit local point and reference frame.
    #[must_use]
    pub const fn new(body: BodyHandle, local_point: Vec3, local_frame: Quat) -> JointAnchor {
        JointAnchor {
            body,
            local_point,
            local_frame,
        }
    }

    /// Creates an anchor at `local_point` with an identity reference frame.
    #[must_use]
    pub const fn at_point(body: BodyHandle, local_point: Vec3) -> JointAnchor {
        JointAnchor {
            body,
            local_point,
            local_frame: Quat::IDENTITY,
        }
    }

    /// Creates an anchor at the body's centre of mass with an identity frame.
    #[must_use]
    pub const fn at_center(body: BodyHandle) -> JointAnchor {
        JointAnchor::at_point(body, Vec3::ZERO)
    }

    /// Sets the reference frame and returns the modified anchor.
    #[must_use]
    pub const fn with_frame(mut self, local_frame: Quat) -> JointAnchor {
        self.local_frame = local_frame;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_center_is_origin_identity() {
        let a = JointAnchor::at_center(BodyHandle::INVALID);
        assert_eq!(a.local_point, Vec3::ZERO);
        assert_eq!(a.local_frame, Quat::IDENTITY);
    }

    #[test]
    fn with_frame_overrides_orientation() {
        let q = Quat::from_rotation_y(0.5);
        let a = JointAnchor::at_point(BodyHandle::INVALID, Vec3::X).with_frame(q);
        assert_eq!(a.local_point, Vec3::X);
        assert_eq!(a.local_frame, q);
    }
}
