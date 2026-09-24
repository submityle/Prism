//! The top-level physics world container.
//!
//! [`PhysicsWorld`] bundles the body storage, the shared collider registry, and
//! the world configuration. It is deliberately backend-agnostic: solvers and
//! backends operate on it but are not owned by it.

use crate::collider::ShapeRegistry;
use crate::config::WorldConfig;
use crate::events::{ContactEventTracker, PhysicsEvent};
use crate::joint::{JointDesc, JointHandle, JointStorage};
use crate::pipeline::detect_contacts;
use crate::state::body::BodyDesc;
use crate::state::handle::BodyHandle;
use crate::state::storage::BodyStorage;
use glam::Vec3;

/// A physics world: all body state, shared shapes, and configuration.
#[derive(Clone, Debug, Default)]
pub struct PhysicsWorld {
    /// Structure-of-Arrays storage for all bodies.
    pub bodies: BodyStorage,
    /// Registry of shared collider shapes.
    pub shapes: ShapeRegistry,
    /// Storage for all joints binding pairs of bodies.
    pub joints: JointStorage,
    /// Global simulation configuration.
    pub config: WorldConfig,
    /// Frame-to-frame contact/trigger event tracker. Updated by
    /// [`PhysicsWorld::drain_contact_events`].
    pub contact_events: ContactEventTracker,
}

impl PhysicsWorld {
    /// Creates a world with the given configuration and empty storage.
    #[must_use]
    pub fn new(config: WorldConfig) -> PhysicsWorld {
        PhysicsWorld {
            bodies: BodyStorage::new(),
            shapes: ShapeRegistry::new(),
            joints: JointStorage::new(),
            config,
            contact_events: ContactEventTracker::new(),
        }
    }

    /// Creates a world with the given gravity and default configuration
    /// otherwise.
    #[must_use]
    pub fn with_gravity(gravity: Vec3) -> PhysicsWorld {
        PhysicsWorld::new(WorldConfig::with_gravity(gravity))
    }

    /// Spawns a body described by `desc`, returning its handle.
    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        self.bodies.insert(desc)
    }

    /// Spawns a joint described by `desc`, returning its handle.
    pub fn spawn_joint(&mut self, desc: JointDesc) -> JointHandle {
        self.joints.insert(desc)
    }

    /// Detects the current contacts and diffs them against the previous call to
    /// produce collision and trigger events.
    ///
    /// This should be called once per rendered/simulated frame, after the
    /// solver has advanced the world, so that events reflect the settled poses.
    /// Pairs where at least one body is a sensor
    /// ([`BodyStorage::is_sensor`](crate::state::storage::BodyStorage::is_sensor))
    /// yield trigger enter/exit events; all other pairs yield collision
    /// start/end events. The returned events can be forwarded to an
    /// [`ObserverRegistry`](crate::events::ObserverRegistry).
    #[must_use]
    pub fn drain_contact_events(&mut self) -> Vec<PhysicsEvent> {
        let manifolds = detect_contacts(self);
        let bodies = &self.bodies;
        self.contact_events.record(&manifolds, |handle| {
            bodies.is_sensor(handle).unwrap_or(false)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_adds_body() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::Y));
        assert_eq!(world.bodies.len(), 1);
        assert_eq!(world.bodies.position(h), Some(Vec3::Y));
        assert_eq!(world.config.gravity.y, -9.81);
    }
}
