//! The top-level physics world container.
//!
//! [`PhysicsWorld`] bundles the body storage, the shared collider registry, and
//! the world configuration. It is deliberately backend-agnostic: solvers and
//! backends operate on it but are not owned by it.

use crate::collider::ShapeRegistry;
use crate::config::WorldConfig;
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
    /// Global simulation configuration.
    pub config: WorldConfig,
}

impl PhysicsWorld {
    /// Creates a world with the given configuration and empty storage.
    #[must_use]
    pub fn new(config: WorldConfig) -> PhysicsWorld {
        PhysicsWorld {
            bodies: BodyStorage::new(),
            shapes: ShapeRegistry::new(),
            config,
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
