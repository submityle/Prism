//! Interpolatable simulation-state snapshots and their transport buffers.
//!
//! M2.5 decouples the physics tick rate from the render frame rate. To do that
//! without visual jitter, the renderer never reads live solver state directly;
//! instead the pipeline publishes an immutable [`pose::BodyPose`]-per-body
//! [`StateSnapshot`] after every fixed physics step, and the renderer
//! interpolates between the two most recently published snapshots.
//!
//! - [`pose`] defines the minimal render pose ([`pose::BodyPose`]) and the
//!   [`pose::lerp_pose`] interpolation used to produce a smooth in-between pose.
//! - [`StateSnapshot`] is a slot-indexed, immutable capture of every live
//!   body's pose and velocity, captured from a [`PhysicsWorld`].
//! - [`buffer::TripleBuffer`] is the lock-free-ready three-slot transport that
//!   hands snapshots from the physics writer to the render reader.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! snapshot/interpolation and triple-buffer patterns are standard, publicly
//! documented real-time-simulation techniques implemented from scratch here.

pub mod buffer;
pub mod pose;

use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};
use pose::BodyPose;

/// An immutable, slot-indexed capture of every live body's render pose and
/// velocity at a single physics step boundary.
///
/// A snapshot is pure data: it borrows nothing from the [`PhysicsWorld`] it was
/// captured from and is cheap to clone, publish through a
/// [`buffer::TripleBuffer`], serialize, or hash. Bodies are addressed by their
/// storage slot; the parallel [`StateSnapshot::generation`] column lets a
/// [`BodyHandle`] be validated so a stale handle to a recycled slot is rejected.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateSnapshot {
    /// Whether each slot held a live body when the snapshot was captured.
    occupied: Vec<bool>,
    /// Generation counter of the body that occupied each slot.
    generations: Vec<u32>,
    /// Render pose (position + orientation) of the body in each slot.
    poses: Vec<BodyPose>,
    /// Linear velocity of the body in each slot (used for hashing/extrapolation).
    linear_velocities: Vec<Vec3>,
    /// Angular velocity of the body in each slot (used for hashing).
    angular_velocities: Vec<Vec3>,
}

impl StateSnapshot {
    /// Creates an empty snapshot with no slots.
    #[must_use]
    pub fn new() -> StateSnapshot {
        StateSnapshot::default()
    }

    /// Captures the current pose and velocity of every live body in `world`.
    ///
    /// The snapshot is sized to `world`'s slot count so that a body's slot index
    /// is preserved across captures, which is what lets two successive snapshots
    /// be interpolated slot-by-slot.
    #[must_use]
    pub fn capture(world: &PhysicsWorld) -> StateSnapshot {
        let n = world.bodies.slot_count();
        let mut snap = StateSnapshot {
            occupied: vec![false; n],
            generations: vec![0; n],
            poses: vec![BodyPose::IDENTITY; n],
            linear_velocities: vec![Vec3::ZERO; n],
            angular_velocities: vec![Vec3::ZERO; n],
        };
        for slot in 0..n {
            let Some(handle) = world.bodies.handle_at_slot(slot) else {
                continue;
            };
            snap.occupied[slot] = true;
            snap.generations[slot] = handle.generation();
            let position = world.bodies.position(handle).unwrap_or(Vec3::ZERO);
            let orientation = world.bodies.orientation(handle).unwrap_or(Quat::IDENTITY);
            snap.poses[slot] = BodyPose::new(position, orientation);
            snap.linear_velocities[slot] =
                world.bodies.linear_velocity(handle).unwrap_or(Vec3::ZERO);
            snap.angular_velocities[slot] =
                world.bodies.angular_velocity(handle).unwrap_or(Vec3::ZERO);
        }
        snap
    }

    /// Re-captures `world` into `self` in place, reusing the existing
    /// allocations when the slot count is unchanged.
    ///
    /// This is the allocation-free path used by the fixed-step pipeline's
    /// triple buffer so that steady-state stepping does not allocate.
    pub fn recapture(&mut self, world: &PhysicsWorld) {
        let n = world.bodies.slot_count();
        self.occupied.clear();
        self.generations.clear();
        self.poses.clear();
        self.linear_velocities.clear();
        self.angular_velocities.clear();
        self.occupied.resize(n, false);
        self.generations.resize(n, 0);
        self.poses.resize(n, BodyPose::IDENTITY);
        self.linear_velocities.resize(n, Vec3::ZERO);
        self.angular_velocities.resize(n, Vec3::ZERO);
        for slot in 0..n {
            let Some(handle) = world.bodies.handle_at_slot(slot) else {
                continue;
            };
            self.occupied[slot] = true;
            self.generations[slot] = handle.generation();
            let position = world.bodies.position(handle).unwrap_or(Vec3::ZERO);
            let orientation = world.bodies.orientation(handle).unwrap_or(Quat::IDENTITY);
            self.poses[slot] = BodyPose::new(position, orientation);
            self.linear_velocities[slot] =
                world.bodies.linear_velocity(handle).unwrap_or(Vec3::ZERO);
            self.angular_velocities[slot] =
                world.bodies.angular_velocity(handle).unwrap_or(Vec3::ZERO);
        }
    }

    /// Returns the number of slots this snapshot spans (live and freed).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.occupied.len()
    }

    /// Returns `true` if the snapshot spans no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.occupied.is_empty()
    }

    /// Returns the captured pose for `handle`, or `None` if the handle does not
    /// match a live body in this snapshot.
    #[must_use]
    pub fn pose(&self, handle: BodyHandle) -> Option<BodyPose> {
        let slot = handle.index() as usize;
        (slot < self.occupied.len()
            && self.occupied[slot]
            && self.generations[slot] == handle.generation())
        .then(|| self.poses[slot])
    }

    /// Returns the captured linear velocity for `handle`, or `None`.
    #[must_use]
    pub fn linear_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        let slot = handle.index() as usize;
        (slot < self.occupied.len()
            && self.occupied[slot]
            && self.generations[slot] == handle.generation())
        .then(|| self.linear_velocities[slot])
    }

    /// Returns `true` if `slot` held a live body when captured.
    #[must_use]
    pub fn slot_occupied(&self, slot: usize) -> bool {
        slot < self.occupied.len() && self.occupied[slot]
    }

    /// Returns the raw pose stored at `slot` regardless of occupancy.
    ///
    /// This is the slot-addressed accessor used by interpolation and hashing,
    /// which iterate by slot index; prefer [`StateSnapshot::pose`] when you hold
    /// a [`BodyHandle`].
    #[must_use]
    pub fn pose_at_slot(&self, slot: usize) -> Option<BodyPose> {
        (slot < self.poses.len()).then(|| self.poses[slot])
    }

    /// Returns the generation recorded for `slot`, or `None` if out of range.
    #[must_use]
    pub fn generation_at_slot(&self, slot: usize) -> Option<u32> {
        (slot < self.generations.len()).then(|| self.generations[slot])
    }

    /// Returns the linear velocity stored at `slot` regardless of occupancy.
    #[must_use]
    pub fn linear_velocity_at_slot(&self, slot: usize) -> Vec3 {
        self.linear_velocities
            .get(slot)
            .copied()
            .unwrap_or(Vec3::ZERO)
    }

    /// Returns the angular velocity stored at `slot` regardless of occupancy.
    #[must_use]
    pub fn angular_velocity_at_slot(&self, slot: usize) -> Vec3 {
        self.angular_velocities
            .get(slot)
            .copied()
            .unwrap_or(Vec3::ZERO)
    }

    /// Interpolates `self` (the earlier snapshot) toward `target` (the later
    /// snapshot) by `alpha` for the body referenced by `handle`.
    ///
    /// Returns `None` if the body is not live in *both* snapshots (so a body
    /// that spawned or despawned between the two captures is not smeared). This
    /// is the operation the renderer calls each frame to obtain a smooth,
    /// jitter-free pose between two physics steps.
    #[must_use]
    pub fn interpolate(
        &self,
        target: &StateSnapshot,
        handle: BodyHandle,
        alpha: f32,
    ) -> Option<BodyPose> {
        let a = self.pose(handle)?;
        let b = target.pose(handle)?;
        Some(pose::lerp_pose(a, b, alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;

    #[test]
    fn capture_records_live_body_pose() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::new(1.0, 2.0, 3.0)));
        let snap = StateSnapshot::capture(&world);
        let pose = snap.pose(h).expect("live body should be captured");
        assert_eq!(pose.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn stale_handle_is_rejected_by_snapshot() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        world.bodies.remove(h);
        let snap = StateSnapshot::capture(&world);
        assert!(snap.pose(h).is_none());
    }

    #[test]
    fn interpolate_returns_lerp_of_two_captures() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let prev = StateSnapshot::capture(&world);
        world.bodies.set_position(h, Vec3::new(4.0, 0.0, 0.0));
        let curr = StateSnapshot::capture(&world);
        let mid = prev.interpolate(&curr, h, 0.5).expect("body live in both");
        assert_eq!(mid.position, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn interpolate_skips_body_not_live_in_both() {
        let mut world = PhysicsWorld::default();
        let a = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let prev = StateSnapshot::capture(&world);
        let b = world.spawn(BodyDesc::dynamic_at(Vec3::ONE));
        let curr = StateSnapshot::capture(&world);
        // Body b did not exist in prev, so it cannot be interpolated.
        assert!(prev.interpolate(&curr, b, 0.5).is_none());
        // Body a exists in both.
        assert!(prev.interpolate(&curr, a, 0.5).is_some());
    }

    #[test]
    fn recapture_reuses_and_matches_fresh_capture() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::new(5.0, 6.0, 7.0)));
        let mut snap = StateSnapshot::new();
        snap.recapture(&world);
        assert_eq!(snap, StateSnapshot::capture(&world));
        assert_eq!(snap.pose(h).unwrap().position, Vec3::new(5.0, 6.0, 7.0));
    }
}
