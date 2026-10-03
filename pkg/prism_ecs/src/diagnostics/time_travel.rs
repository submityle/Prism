//! Time-travel debugging facade (design §16.5 / §16.6).
//!
//! [`TimeTravel`] wraps the existing [`SnapshotRing`] into a frame-indexed
//! record/seek API suitable for an editor "time travel" panel, deterministic
//! rollback-network debugging, and undo stacks (design §16.6 对接
//! `prism_ui_timetravel`). It is intentionally a thin, allocation-faithful
//! facade: all heavy lifting (structural snapshot capture and byte-equivalent
//! restore) lives in [`World::snapshot`] / [`World::restore`].
//!
//! This module is `std`-gated because world snapshots require `std`.
//!
//! # Registration
//! [`World::snapshot`] panics if any resident component lacks clone glue, so
//! every snapshotted component type must first be registered with
//! [`World::register_snapshot_component`](crate::world::World::register_snapshot_component).
//! [`TimeTravel::record`] inherits that contract.

use alloc::vec::Vec;

use crate::world::snapshot::{SnapshotRing, WorldSnapshot};
use crate::world::World;

/// A bounded, frame-indexed history of world snapshots.
///
/// Records are keyed by an application-defined frame number (typically the
/// deterministic simulation tick). The ring retains the most recent `capacity`
/// distinct frames; recording beyond that evicts the oldest.
pub struct TimeTravel {
    ring: SnapshotRing,
}

impl TimeTravel {
    /// Create a history retaining up to `capacity` distinct frames.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: SnapshotRing::new(capacity),
        }
    }

    /// Capture `world` and store it under `frame`.
    ///
    /// Re-recording an already-stored frame overwrites it in place without
    /// disturbing eviction order (idempotent re-confirmation).
    ///
    /// # Panics
    /// Panics if any resident component lacks clone glue; see the module-level
    /// registration note.
    pub fn record(&mut self, frame: u64, world: &World) {
        self.ring.push(frame, world.snapshot());
    }

    /// Restore `world` to the recorded state of `frame`, returning `true` on
    /// success or `false` if that frame is no longer retained.
    ///
    /// Restoring does not fire component hooks or observers and leaves
    /// resources untouched (see [`World::restore`]).
    pub fn seek(&self, frame: u64, world: &mut World) -> bool {
        match self.ring.get(frame) {
            Some(snapshot) => {
                world.restore(snapshot);
                true
            }
            None => false,
        }
    }

    /// Number of retained frames.
    #[inline]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether no frames are retained.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Maximum number of distinct frames the history can retain.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Whether `frame` is still retained.
    #[inline]
    pub fn contains(&self, frame: u64) -> bool {
        self.ring.contains(frame)
    }

    /// The retained frame numbers, oldest first.
    pub fn frames(&self) -> Vec<u64> {
        self.ring.frames().collect()
    }

    /// The newest retained frame number, if any.
    pub fn latest_frame(&self) -> Option<u64> {
        self.ring.latest().map(|(frame, _)| frame)
    }

    /// The oldest retained frame number, if any (the next to be evicted).
    pub fn oldest_frame(&self) -> Option<u64> {
        self.ring.oldest().map(|(frame, _)| frame)
    }

    /// Borrow the snapshot recorded for `frame`, if retained.
    pub fn snapshot(&self, frame: u64) -> Option<&WorldSnapshot> {
        self.ring.get(frame)
    }

    /// Drop every retained frame.
    pub fn clear(&mut self) {
        self.ring.clear();
    }

    /// Borrow the underlying snapshot ring for advanced inspection.
    #[inline]
    pub fn ring(&self) -> &SnapshotRing {
        &self.ring
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    #[derive(Debug, Clone, PartialEq)]
    struct Position(f32, f32);
    impl Component for Position {}

    fn world_with(x: f32) -> (World, crate::entity::Entity) {
        let mut world = World::new();
        world.register_snapshot_component::<Position>();
        let e = world.spawn(Position(x, 0.0));
        (world, e)
    }

    #[test]
    fn record_and_seek_restores_values() {
        let (mut world, e) = world_with(1.0);
        let mut history = TimeTravel::new(8);
        history.record(1, &world);

        // Advance the world's state.
        world.get_mut::<Position>(e).unwrap().0 = 99.0;
        history.record(2, &world);
        assert_eq!(world.get::<Position>(e), Some(&Position(99.0, 0.0)));

        // Travel back to frame 1.
        assert!(history.seek(1, &mut world));
        assert_eq!(world.get::<Position>(e), Some(&Position(1.0, 0.0)));

        // And forward again to frame 2.
        assert!(history.seek(2, &mut world));
        assert_eq!(world.get::<Position>(e), Some(&Position(99.0, 0.0)));
    }

    #[test]
    fn seek_missing_frame_is_false() {
        let (world, _e) = world_with(1.0);
        let mut history = TimeTravel::new(4);
        history.record(10, &world);
        let mut target = World::new();
        target.register_snapshot_component::<Position>();
        assert!(!history.seek(999, &mut target));
    }

    #[test]
    fn capacity_evicts_oldest() {
        let (world, _e) = world_with(1.0);
        let mut history = TimeTravel::new(2);
        assert!(history.is_empty());
        history.record(1, &world);
        history.record(2, &world);
        history.record(3, &world);
        assert_eq!(history.len(), 2);
        assert_eq!(history.capacity(), 2);
        assert!(!history.contains(1));
        assert!(history.contains(2));
        assert!(history.contains(3));
        assert_eq!(history.oldest_frame(), Some(2));
        assert_eq!(history.latest_frame(), Some(3));
        assert_eq!(history.frames(), alloc::vec![2, 3]);
    }

    #[test]
    fn clear_drops_history() {
        let (world, _e) = world_with(1.0);
        let mut history = TimeTravel::new(4);
        history.record(1, &world);
        assert!(!history.is_empty());
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.latest_frame(), None);
        assert!(history.snapshot(1).is_none());
        assert_eq!(history.ring().len(), 0);
    }
}
