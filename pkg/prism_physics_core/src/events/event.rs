//! Physics event value types.
//!
//! Events describe changes in the contact/overlap relationship between pairs of
//! bodies across simulation frames. They are produced by
//! [`ContactEventTracker`](crate::events::ContactEventTracker) and consumed by
//! observers or drained from a queue.
//!
//! Two families of events exist:
//!
//! - **Collision** events ([`PhysicsEvent::CollisionStarted`] /
//!   [`PhysicsEvent::CollisionEnded`]) describe solid, force-resolving contacts
//!   between two non-sensor bodies.
//! - **Trigger** events ([`PhysicsEvent::TriggerEntered`] /
//!   [`PhysicsEvent::TriggerExited`]) describe overlaps where at least one body
//!   is a sensor (trigger volume). Sensors never receive a physical response.
//!
//! # Provenance
//!
//! Contact enter/exit event semantics and the sensor/trigger distinction are
//! standard, publicly documented physics-engine concepts. This file contains no
//! Unreal Engine source or derived code.

use crate::state::handle::BodyHandle;

/// An unordered pair of body handles used as a stable contact key.
///
/// The pair is normalized so that the handle with the smaller slot index (and,
/// on a tie, the smaller generation) is stored first. This makes `(a, b)` and
/// `(b, a)` compare and hash equal, so a contact is tracked once regardless of
/// the order the narrow phase reports it in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContactPair {
    first: BodyHandle,
    second: BodyHandle,
}

impl ContactPair {
    /// Creates a normalized pair from two body handles.
    #[must_use]
    pub fn new(a: BodyHandle, b: BodyHandle) -> ContactPair {
        let a_key = (a.index(), a.generation());
        let b_key = (b.index(), b.generation());
        if a_key <= b_key {
            ContactPair {
                first: a,
                second: b,
            }
        } else {
            ContactPair {
                first: b,
                second: a,
            }
        }
    }

    /// Returns the first (lower-keyed) body of the pair.
    #[must_use]
    pub fn first(&self) -> BodyHandle {
        self.first
    }

    /// Returns the second (higher-keyed) body of the pair.
    #[must_use]
    pub fn second(&self) -> BodyHandle {
        self.second
    }

    /// Returns the other body of the pair given one of its members, or `None`
    /// if `body` is not part of the pair.
    #[must_use]
    pub fn other(&self, body: BodyHandle) -> Option<BodyHandle> {
        if body == self.first {
            Some(self.second)
        } else if body == self.second {
            Some(self.first)
        } else {
            None
        }
    }
}

/// A physics event describing a change in a contact or trigger relationship.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhysicsEvent {
    /// Two solid (non-sensor) bodies began touching this frame.
    CollisionStarted(ContactPair),
    /// Two solid (non-sensor) bodies stopped touching this frame.
    CollisionEnded(ContactPair),
    /// A body entered a sensor's volume this frame.
    TriggerEntered {
        /// The sensor (trigger-volume) body.
        sensor: BodyHandle,
        /// The other body that entered the sensor's volume.
        other: BodyHandle,
    },
    /// A body left a sensor's volume this frame.
    TriggerExited {
        /// The sensor (trigger-volume) body.
        sensor: BodyHandle,
        /// The other body that left the sensor's volume.
        other: BodyHandle,
    },
}

impl PhysicsEvent {
    /// Returns `true` if this event is a collision start or end.
    #[must_use]
    pub fn is_collision(&self) -> bool {
        matches!(
            self,
            PhysicsEvent::CollisionStarted(_) | PhysicsEvent::CollisionEnded(_)
        )
    }

    /// Returns `true` if this event is a trigger enter or exit.
    #[must_use]
    pub fn is_trigger(&self) -> bool {
        matches!(
            self,
            PhysicsEvent::TriggerEntered { .. } | PhysicsEvent::TriggerExited { .. }
        )
    }

    /// Returns `true` if this event marks the beginning of a relationship
    /// (a collision start or a trigger enter).
    #[must_use]
    pub fn is_start(&self) -> bool {
        matches!(
            self,
            PhysicsEvent::CollisionStarted(_) | PhysicsEvent::TriggerEntered { .. }
        )
    }
}
