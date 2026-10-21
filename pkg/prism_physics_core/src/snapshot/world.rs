//! Complete in-memory world snapshots for rollback and deterministic replay.
//!
//! [`StateSnapshot`] is intentionally a small render transport. It cannot be
//! used for rollback because it omits solver queues, soft-body state, topology
//! revisions, fields, aerodynamic history, articulation state, and assembly
//! bookkeeping. This module captures the complete [`PhysicsWorld`] value at a
//! fixed-step boundary instead.

use std::collections::VecDeque;

use crate::snapshot::StateSnapshot;
use crate::world::PhysicsWorld;

/// A complete clone of a physics world at one fixed-step boundary.
///
/// The snapshot owns every world-level state field, including pending force
/// queues, topology and degradation events, aerodynamic history,
/// reduced-coordinate articulation state, and merge bookkeeping. Restoring a
/// snapshot therefore replaces the destination world atomically from the
/// caller's perspective.
#[derive(Clone, Debug)]
pub struct FullStateSnapshot {
    frame: u64,
    world: PhysicsWorld,
    render: StateSnapshot,
}

impl FullStateSnapshot {
    /// Captures the complete world and its render projection at `frame`.
    #[must_use]
    pub fn capture(frame: u64, world: &PhysicsWorld) -> Self {
        Self {
            frame,
            world: world.clone(),
            render: StateSnapshot::capture(world),
        }
    }

    /// Returns the fixed-step frame number assigned by the caller.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Returns the immutable render projection captured alongside the world.
    #[must_use]
    pub fn render(&self) -> &StateSnapshot {
        &self.render
    }

    /// Replaces `world` with the complete captured state.
    pub fn restore_into(&self, world: &mut PhysicsWorld) {
        *world = self.world.clone();
    }

    /// Returns a clone of the captured world for callers that need an
    /// isolated branch of the simulation.
    #[must_use]
    pub fn fork_world(&self) -> PhysicsWorld {
        self.world.clone()
    }

    /// Fingerprints all captured world state (excluding the frame label).
    /// See [`super::world_hash`] for schema stability and NaN restrictions.
    pub fn state_hash(
        &self,
    ) -> Result<super::world_hash::WorldHash, super::world_hash::WorldHashError> {
        super::world_hash::hash_world(&self.world)
    }
}

/// A bounded, frame-addressed ring of complete world snapshots.
///
/// The ring drops the oldest snapshot when it reaches `capacity`. Capacity is
/// required to be non-zero so every successful insertion remains recoverable
/// through [`SnapshotRing::get`].
#[derive(Clone, Debug)]
pub struct SnapshotRing {
    capacity: usize,
    snapshots: VecDeque<FullStateSnapshot>,
}

impl SnapshotRing {
    /// Creates an empty ring with the requested bounded capacity.
    ///
    /// A zero capacity is rejected because accepting it would silently lose a
    /// snapshot immediately after capture.
    pub fn new(capacity: usize) -> Result<Self, SnapshotRingError> {
        if capacity == 0 {
            return Err(SnapshotRingError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            snapshots: VecDeque::with_capacity(capacity),
        })
    }

    /// Returns the configured maximum number of snapshots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of snapshots currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` when no snapshots are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Captures and appends the current world at `frame`.
    /// Capturing an earlier frame starts a new branch and discards that frame
    /// and every later snapshot from the previous branch.
    pub fn capture(&mut self, frame: u64, world: &PhysicsWorld) {
        self.push(FullStateSnapshot::capture(frame, world));
    }

    /// Appends an already captured snapshot, applying the same bounded policy.
    pub fn push(&mut self, snapshot: FullStateSnapshot) {
        while self
            .snapshots
            .back()
            .is_some_and(|last| last.frame() >= snapshot.frame())
        {
            self.snapshots.pop_back();
        }
        if self.snapshots.len() == self.capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Finds a retained snapshot by its fixed-step frame number.
    #[must_use]
    pub fn get(&self, frame: u64) -> Option<&FullStateSnapshot> {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.frame() == frame)
    }

    /// Returns the newest retained snapshot.
    #[must_use]
    pub fn latest(&self) -> Option<&FullStateSnapshot> {
        self.snapshots.back()
    }

    /// Restores the retained snapshot for `frame`, returning whether it was
    /// found. The destination world is unchanged when the frame is absent.
    pub fn restore(&self, frame: u64, world: &mut PhysicsWorld) -> bool {
        let Some(snapshot) = self.get(frame) else {
            return false;
        };
        snapshot.restore_into(world);
        true
    }
}

/// Errors returned while constructing a snapshot ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotRingError {
    /// A ring must retain at least one snapshot.
    ZeroCapacity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::{AeroBodyConfig, WindField};
    use crate::dynamics::field::{Field, FieldId, FieldRegistry};
    use crate::soft::build::ClothGrid;
    use crate::state::body::BodyDesc;
    use glam::Vec3;

    #[test]
    fn full_snapshot_restores_body_and_world_owned_state() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let body = world.spawn(BodyDesc::dynamic_at(Vec3::new(1.0, 2.0, 3.0)));
        world.set_aero_field(WindField::new(Vec3::new(4.0, 0.0, 0.0)));
        world
            .fields
            .insert(Field::constant(FieldId(1), Vec3::new(0.0, 3.0, 0.0)));
        world
            .external_forces
            .accumulate_force(body, Vec3::X)
            .unwrap();
        world
            .register_aero_body(body, AeroBodyConfig::default())
            .unwrap();
        let cloth = ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default();
        let soft = world.spawn_soft_body(cloth.body);
        let snapshot = FullStateSnapshot::capture(42, &world);

        world.bodies.set_position(body, Vec3::new(99.0, 99.0, 99.0));
        world.unregister_aero_body(body);
        world.remove_soft_body(soft);
        world.fields = FieldRegistry::new();
        world.external_forces.clear();

        snapshot.restore_into(&mut world);
        assert_eq!(snapshot.frame(), 42);
        assert_eq!(world.bodies.position(body), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(world.aero_bodies.len(), 1);
        assert_eq!(world.soft_bodies.len(), 1);
        assert_eq!(world.fields.fields().len(), 1);
        assert_eq!(world.external_forces.pending(body).unwrap().force, Vec3::X);
        assert!(world.soft_bodies.get(soft).is_some());
        assert_eq!(
            snapshot.render().pose(body).unwrap().position,
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn ring_is_bounded_and_restores_only_retained_frames() {
        let mut ring = SnapshotRing::new(2).unwrap();
        let mut world = PhysicsWorld::default();
        let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        ring.capture(1, &world);
        world.bodies.set_position(body, Vec3::X);
        ring.capture(2, &world);
        world.bodies.set_position(body, Vec3::Y);
        ring.capture(3, &world);

        assert_eq!(ring.len(), 2);
        assert!(ring.get(1).is_none());
        assert!(ring.restore(2, &mut world));
        assert_eq!(world.bodies.position(body), Some(Vec3::X));
        assert!(!ring.restore(1, &mut world));
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(matches!(
            SnapshotRing::new(0),
            Err(SnapshotRingError::ZeroCapacity)
        ));
    }

    #[test]
    fn recapturing_a_frame_discards_the_old_future() {
        let mut ring = SnapshotRing::new(4).unwrap();
        let mut world = PhysicsWorld::default();
        let body = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        for frame in 1..=3 {
            ring.capture(frame, &world);
        }
        world.bodies.set_position(body, Vec3::Z);
        ring.capture(2, &world);
        assert_eq!(ring.len(), 2);
        assert!(ring.get(3).is_none());
        assert!(ring.restore(2, &mut world));
        assert_eq!(world.bodies.position(body), Some(Vec3::Z));
        assert_eq!(ring.latest().unwrap().frame(), 2);
    }

    #[test]
    fn fork_is_isolated_and_resimulation_matches() {
        use crate::snapshot::hash::hash_state;
        use crate::solver::{Solver, XpbdSolver};

        let mut world = PhysicsWorld::with_gravity(Vec3::NEG_Y);
        let body = world.spawn(BodyDesc::dynamic_at(Vec3::Y));
        world
            .external_forces
            .accumulate_force(body, Vec3::X)
            .unwrap();
        let snapshot = FullStateSnapshot::capture(0, &world);
        let mut branch = snapshot.fork_world();
        let mut first = XpbdSolver::default();
        let mut second = XpbdSolver::default();
        for _ in 0..8 {
            first.step(&mut world, 1.0 / 60.0, 1);
            second.step(&mut branch, 1.0 / 60.0, 1);
        }
        assert_eq!(
            hash_state(&StateSnapshot::capture(&world)),
            hash_state(&StateSnapshot::capture(&branch))
        );
        assert_eq!(snapshot.fork_world().bodies.position(body), Some(Vec3::Y));
    }
}
