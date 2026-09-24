//! Frame-to-frame contact/trigger event tracking.
//!
//! [`ContactEventTracker`] remembers which body pairs were in contact on the
//! previous frame and diffs that against the current frame's manifolds to emit
//! [`PhysicsEvent`]s. Solid pairs (no sensor) produce collision start/end
//! events; pairs where at least one body is a sensor produce trigger
//! enter/exit events.
//!
//! The tracker keeps no borrow on the world: it consumes a slice of manifolds
//! and a sensor classifier, so it can be embedded in a
//! [`PhysicsWorld`](crate::world::PhysicsWorld) as a plain data field or driven
//! standalone.
//!
//! # Provenance
//!
//! Diffing successive contact sets to derive enter/exit events is a standard,
//! publicly documented technique. This file contains no Unreal Engine source or
//! derived code.

use crate::collide::ContactManifold;
use crate::events::event::{ContactPair, PhysicsEvent};
use crate::state::handle::BodyHandle;
use std::collections::{HashMap, HashSet};

/// Which body of an overlapping pair is the sensor, and which is the other.
///
/// When both bodies of a pair are sensors, the pair's lower-keyed body (see
/// [`ContactPair`]) is recorded as the `sensor`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct TriggerRoles {
    /// The sensor (trigger-volume) body.
    sensor: BodyHandle,
    /// The other body overlapping the sensor.
    other: BodyHandle,
}

/// Tracks contact and trigger relationships across frames to emit events.
///
/// Call [`ContactEventTracker::record`] once per frame with the current
/// manifolds; it returns the events that fired this frame and updates its
/// internal previous-frame state. Use [`ContactEventTracker::reset`] to forget
/// all history (for example after teleporting bodies).
#[derive(Clone, Debug, Default)]
pub struct ContactEventTracker {
    prev_solid: HashSet<ContactPair>,
    prev_triggers: HashMap<ContactPair, TriggerRoles>,
}

impl ContactEventTracker {
    /// Creates an empty tracker with no remembered contacts.
    #[must_use]
    pub fn new() -> ContactEventTracker {
        ContactEventTracker::default()
    }

    /// Forgets all previous-frame contact state.
    ///
    /// The next [`ContactEventTracker::record`] call will treat every current
    /// overlap as freshly started.
    pub fn reset(&mut self) {
        self.prev_solid.clear();
        self.prev_triggers.clear();
    }

    /// Returns the number of solid contacts currently being tracked.
    #[must_use]
    pub fn active_collision_count(&self) -> usize {
        self.prev_solid.len()
    }

    /// Returns the number of trigger overlaps currently being tracked.
    #[must_use]
    pub fn active_trigger_count(&self) -> usize {
        self.prev_triggers.len()
    }

    /// Diffs the current frame's manifolds against the previous frame and
    /// returns the resulting events, then adopts the current frame as the new
    /// baseline.
    ///
    /// `is_sensor` classifies a body handle as a sensor (trigger volume). A
    /// pair is treated as a trigger overlap when either body is a sensor, and
    /// as a solid collision otherwise.
    ///
    /// Events are returned grouped: collision ends, trigger exits, collision
    /// starts, then trigger enters. Within each group the order is
    /// unspecified (it follows hash iteration order).
    #[must_use]
    pub fn record(
        &mut self,
        manifolds: &[ContactManifold],
        is_sensor: impl Fn(BodyHandle) -> bool,
    ) -> Vec<PhysicsEvent> {
        let mut cur_solid: HashSet<ContactPair> = HashSet::new();
        let mut cur_triggers: HashMap<ContactPair, TriggerRoles> = HashMap::new();

        for manifold in manifolds {
            let a = manifold.body_a;
            let b = manifold.body_b;
            let a_sensor = is_sensor(a);
            let b_sensor = is_sensor(b);
            let pair = ContactPair::new(a, b);

            if a_sensor || b_sensor {
                // Determine sensor/other. If both are sensors, the pair's
                // lower-keyed body is treated as the sensor for a stable role.
                let roles = if a_sensor && b_sensor {
                    TriggerRoles {
                        sensor: pair.first(),
                        other: pair.second(),
                    }
                } else if a_sensor {
                    TriggerRoles {
                        sensor: a,
                        other: b,
                    }
                } else {
                    TriggerRoles {
                        sensor: b,
                        other: a,
                    }
                };
                cur_triggers.insert(pair, roles);
            } else {
                cur_solid.insert(pair);
            }
        }

        let mut events = Vec::new();

        // Collision ends: pairs present last frame but gone this frame.
        for pair in self.prev_solid.difference(&cur_solid) {
            events.push(PhysicsEvent::CollisionEnded(*pair));
        }
        // Trigger exits: pairs present last frame but gone this frame.
        for (pair, roles) in &self.prev_triggers {
            if !cur_triggers.contains_key(pair) {
                events.push(PhysicsEvent::TriggerExited {
                    sensor: roles.sensor,
                    other: roles.other,
                });
            }
        }
        // Collision starts: pairs new this frame.
        for pair in cur_solid.difference(&self.prev_solid) {
            events.push(PhysicsEvent::CollisionStarted(*pair));
        }
        // Trigger enters: pairs new this frame.
        for (pair, roles) in &cur_triggers {
            if !self.prev_triggers.contains_key(pair) {
                events.push(PhysicsEvent::TriggerEntered {
                    sensor: roles.sensor,
                    other: roles.other,
                });
            }
        }

        self.prev_solid = cur_solid;
        self.prev_triggers = cur_triggers;
        events
    }
}
