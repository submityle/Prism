//! The top-level physics world container.
//!
//! [`PhysicsWorld`] bundles the body storage, the shared collider registry, and
//! the world configuration. It is deliberately backend-agnostic: solvers and
//! backends operate on it but are not owned by it.

use crate::collider::ShapeRegistry;
use crate::config::WorldConfig;
use crate::events::{ContactEventTracker, PhysicsEvent};
use crate::joint::{JointDesc, JointHandle, JointStorage};
use crate::math::scalar::Real;
use crate::pipeline::detect_contacts;
use crate::soft::rigid_coupling::{
    couple_cloth_to_rigid, ClothRigidCouplingConfig, CouplingReport,
};
use crate::soft::solver::{SoftContacts, SoftSolver};
use crate::soft::{SoftBody, SoftBodyHandle, SoftBodyStorage};
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
    /// Opt-in configuration for the cloth-rigid two-way coupling stage.
    /// Defaults to disabled so rigid-only worlds stay bit-identical.
    pub cloth_coupling: ClothRigidCouplingConfig,
    /// World-owned deformable state. Entries are advanced by
    /// [`crate::solver::XpbdSolver`] in deterministic slot order.
    pub soft_bodies: SoftBodyStorage,
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
            cloth_coupling: ClothRigidCouplingConfig::disabled(),
            soft_bodies: SoftBodyStorage::new(),
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

    /// Inserts a project-independent soft body into world-owned storage.
    ///
    /// The body may represent cloth, rope, hair, a volumetric body, or any
    /// other particle/constraint authoring. Its state is then advanced by the
    /// world solver at the same fixed substep boundary as rigid bodies.
    pub fn spawn_soft_body(&mut self, body: SoftBody) -> SoftBodyHandle {
        self.soft_bodies.insert(body)
    }

    /// Removes a world-owned soft body and invalidates its handle.
    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) -> bool {
        self.soft_bodies.remove(handle)
    }

    /// Spawns a joint described by `desc`, returning its handle.
    pub fn spawn_joint(&mut self, desc: JointDesc) -> JointHandle {
        self.joints.insert(desc)
    }

    /// Runs one substep of the opt-in cloth↔rigid two-way coupling stage.
    ///
    /// `positions` is a soft body's particle position column (corrected in
    /// place) and `inverse_masses` its index-aligned inverse-mass column; `dt`
    /// is the substep. The soft particles and the rigid proxies they rest
    /// against exchange a mass-weighted push, and each rigid body receives the
    /// linear reaction impulse the particles exerted on it. When
    /// [`cloth_coupling`](Self::cloth_coupling) is disabled (the default) this
    /// is a no-op and the rigid-only path is untouched, so existing goldens stay
    /// bit-identical.
    ///
    /// Returns a [`CouplingReport`] summarising the pass (participating proxy
    /// count, bodies written back, and the total applied impulse).
    pub fn resolve_cloth_coupling(
        &mut self,
        positions: &mut [Vec3],
        inverse_masses: &[Real],
        dt: Real,
    ) -> CouplingReport {
        couple_cloth_to_rigid(self, positions, inverse_masses, dt)
    }

    /// Runs the existing two-way coupling driver against one world-owned soft
    /// body. The body is temporarily detached from storage only to satisfy the
    /// borrow boundary between particle state and rigid-world writeback; its
    /// handle and deterministic slot are unchanged.
    pub fn resolve_soft_body_coupling(
        &mut self,
        handle: SoftBodyHandle,
        dt: Real,
    ) -> Option<CouplingReport> {
        let mut body = self.soft_bodies.take(handle)?;
        let (positions, inverse_masses) = body.particles.positions_and_inverse_masses_mut();
        let report = self.resolve_cloth_coupling(positions, inverse_masses, dt);
        self.soft_bodies.put(handle, body);
        Some(report)
    }

    /// Advances one fixed substep for every world-owned soft body.
    ///
    /// The order is fixed: soft prediction and internal projection, proxy
    /// gather/two-way coupling/reaction writeback when enabled, then particle
    /// velocity recovery. Empty storage is a no-op. With coupling disabled the
    /// rigid-only path is not entered and existing rigid worlds remain
    /// bit-identical.
    pub(crate) fn step_soft_bodies_substep(&mut self, h: Real) {
        if self.soft_bodies.is_empty() {
            return;
        }
        let handles: Vec<SoftBodyHandle> = self.soft_bodies.handles().collect();
        let solver = SoftSolver::new();
        for handle in handles {
            let Some(mut body) = self.soft_bodies.take(handle) else {
                continue;
            };
            solver.step_substep_before_velocity_recovery(
                &mut body.particles,
                &mut body.constraints,
                &body.config,
                &SoftContacts::EMPTY,
                h,
            );
            self.soft_bodies.put(handle, body);

            if self.cloth_coupling.is_enabled() {
                let _ = self.resolve_soft_body_coupling(handle, h);
            }

            let Some(mut body) = self.soft_bodies.take(handle) else {
                continue;
            };
            solver.recover_substep_velocities(&mut body.particles, h);
            self.soft_bodies.put(handle, body);
        }
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
