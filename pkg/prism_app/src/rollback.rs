//! Frame rollback history: bounded world snapshots for rollback networking and
//! time travel (design §14, §15, §22 M5).
//!
//! The hard part of rollback — a cheap, deterministic, owned copy of a world
//! and its restore — already lives in [`prism_ecs`], as
//! [`WorldSnapshot`](prism_ecs::world::snapshot::WorldSnapshot),
//! [`SnapshotRing`](prism_ecs::world::snapshot::SnapshotRing), and
//! [`World::snapshot`]/[`World::restore`]. This module is the thin *app-level
//! orchestration* on top of it: a bounded per-frame history that captures the
//! world at successive frame numbers, hands each captured frame back for an
//! input log, and restores the world to any frame still retained.
//!
//! The rollback-loop shape (design §14, GGPO / Quantum form) is:
//!
//! 1. [`capture`](SnapshotHistory::capture) the confirmed world at frame `f` and
//!    attach the frame's inputs to that number.
//! 2. A late input for an earlier frame arrives.
//! 3. [`rollback`](SnapshotHistory::rollback) to that frame and re-simulate the
//!    frames forward with the corrected input.
//!
//! # Scope, exactly as `prism_ecs` defines it
//!
//! A snapshot captures entity / component / change-tick state and **not**
//! [`Resources`](prism_ecs::resource::Resource), and it is registry-bound:
//! register each component's clone glue with
//! [`World::register_snapshot_component`] first. [`SnapshotHistory::capture`]
//! panics on an unregistered resident component;
//! [`try_capture`](SnapshotHistory::try_capture) reports the offenders instead.
//! [`state_hash`](SnapshotHistory::state_hash) folds only the component values
//! whose types were registered as *hashable*
//! ([`World::register_snapshot_component_hashable`]); a component with clone glue
//! but no value hashing contributes structure but not its value to the hash.

use alloc::vec::Vec;

use prism_ecs::component::ComponentId;
use prism_ecs::world::snapshot::SnapshotRing;
use prism_ecs::world::World;

/// A bounded, frame-numbered history of [`World`] snapshots.
///
/// Frames are numbered `0, 1, 2, ...` in capture order. Once `capacity` frames
/// are retained the oldest is evicted, so rollback memory is bounded regardless
/// of session length; a rollback to an already-evicted frame is reported as
/// `false` rather than silently doing nothing.
pub struct SnapshotHistory {
    ring: SnapshotRing,
    next_frame: u64,
}

impl SnapshotHistory {
    /// A history retaining the most recent `capacity` frames.
    ///
    /// # Panics
    /// Panics if `capacity` is zero (the history must retain at least one
    /// frame), matching [`SnapshotRing::new`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: SnapshotRing::new(capacity),
            next_frame: 0,
        }
    }

    /// The maximum number of frames retained at once.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// The number of frames currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Whether no frame is retained yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// The frame number the next [`capture`](Self::capture) will record
    /// (equivalently, the number of frames ever captured).
    #[must_use]
    pub fn next_frame(&self) -> u64 {
        self.next_frame
    }

    /// The most recently captured frame, or `None` before the first capture.
    #[must_use]
    pub fn last_frame(&self) -> Option<u64> {
        self.next_frame.checked_sub(1)
    }

    /// The oldest retained frame, or `None` before the first capture.
    ///
    /// This is the ring's own oldest entry, so it stays correct after a
    /// [`seek`](Self::seek) rewind: seeking moves the *capture cursor* back
    /// without evicting anything, after which the retained window no longer ends
    /// at `next_frame - 1` and cannot be derived from the cursor.
    #[must_use]
    pub fn oldest_frame(&self) -> Option<u64> {
        self.ring.oldest().map(|(frame, _)| frame)
    }

    /// Whether `frame` is still retained (not yet evicted).
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        self.ring.contains(frame)
    }

    /// Capture `world` at the next frame number and return that number.
    ///
    /// # Panics
    /// Panics if a resident component has no snapshot clone glue; use
    /// [`try_capture`](Self::try_capture) to report that instead of panicking.
    pub fn capture(&mut self, world: &World) -> u64 {
        let frame = self.next_frame;
        self.ring.push(frame, world.snapshot());
        self.next_frame += 1;
        frame
    }

    /// Fallible [`capture`](Self::capture): on an unregistered component the
    /// snapshot is refused and the offending component ids are returned,
    /// leaving the history unchanged (the frame counter does not advance).
    pub fn try_capture(&mut self, world: &World) -> Result<u64, Vec<ComponentId>> {
        let frame = self.next_frame;
        let snapshot = world.try_snapshot()?;
        self.ring.push(frame, snapshot);
        self.next_frame += 1;
        Ok(frame)
    }

    /// Restore `world` to `frame` if it is still retained.
    ///
    /// Returns `true` when the restore ran and `false` when `frame` has been
    /// evicted (or was never captured), in which case `world` is untouched.
    pub fn rollback(&self, world: &mut World, frame: u64) -> bool {
        match self.ring.get(frame) {
            Some(snapshot) => {
                world.restore(snapshot);
                true
            }
            None => false,
        }
    }

    /// Move the frame cursor so the next [`capture`](Self::capture) records
    /// `frame`, overwriting that frame's retained snapshot (used when
    /// re-simulating forward after a [`rollback`](Self::rollback)). Retained
    /// snapshots are untouched, and seeking past the last frame is the caller's
    /// responsibility (it would leave a gap the ring never fills).
    pub fn seek(&mut self, frame: u64) {
        self.next_frame = frame;
    }

    /// Discard every retained frame strictly newer than `frame` and rebase the
    /// capture cursor to `frame`, so the next
    /// [`capture`](Self::capture) records `frame` again.
    ///
    /// A rollback abandons the future branch. Truncating is what stops those
    /// frames from staying addressable through [`contains`](Self::contains) /
    /// [`state_hash`](Self::state_hash) / a later [`rollback`](Self::rollback)
    /// and thereby resurrecting state a correction was supposed to replace
    /// (design §14/§15). `frame` itself is kept: its pre-input snapshot is what a
    /// re-simulation overwrites.
    pub fn truncate_after(&mut self, frame: u64) {
        self.ring.truncate_after(frame);
        self.next_frame = frame;
    }

    /// The deterministic state hash of a retained frame, for desync comparison
    /// (design §14, 逐帧状态哈希去同步). `None` once the frame is evicted.
    #[must_use]
    pub fn state_hash(&self, frame: u64) -> Option<u64> {
        self.ring.get(frame).map(|snapshot| snapshot.state_hash())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::sub_app::SubApp;
    use prism_ecs::component::Component;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Counter(u32);
    impl Component for Counter {}

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Marker;
    impl Component for Marker {}

    fn world() -> World {
        let mut w = World::new();
        w.register_snapshot_component_hashable::<Counter>();
        w.register_snapshot_component::<Marker>();
        w
    }

    #[test]
    fn capture_then_rollback_restores_component_state() {
        let mut w = world();
        let e = w.spawn((Counter(1), Marker));
        let mut history = SnapshotHistory::new(4);

        let f0 = history.capture(&w);
        assert_eq!(f0, 0);
        assert_eq!(history.len(), 1);
        assert_eq!(history.last_frame(), Some(0));

        w.get_mut::<Counter>(e).unwrap().0 = 42;
        assert!(history.rollback(&mut w, f0));
        assert_eq!(w.get::<Counter>(e).unwrap().0, 1);
    }

    #[test]
    fn ring_evicts_the_oldest_beyond_capacity() {
        let mut w = world();
        let _ = w.spawn((Counter(0), Marker));
        let mut history = SnapshotHistory::new(2);

        history.capture(&w);
        history.capture(&w);
        history.capture(&w);
        assert_eq!(history.len(), 2);
        assert!(!history.contains(0));
        assert!(history.contains(1));
        assert!(history.contains(2));

        // Rolling back to an evicted frame is reported, not silently ignored.
        assert!(!history.rollback(&mut w, 0));
        assert!(history.rollback(&mut w, 2));
    }

    /// `oldest_frame` survives a rewind: seeking the cursor back does not evict
    /// anything, so the oldest retained frame stays the ring's front (regression:
    /// it was derived from `next_frame - len` and returned `None` after a rewind,
    /// which made peer reconciliation silently report agreement).
    #[test]
    fn oldest_frame_is_correct_after_a_rewind() {
        let w = world();
        let mut history = SnapshotHistory::new(8);
        for _ in 0..10 {
            history.capture(&w);
        }
        assert_eq!(history.next_frame(), 10);
        assert_eq!(history.oldest_frame(), Some(2), "capacity 8 retains 2..=9");

        // Rewind the capture cursor, as a rollback + re-simulate does.
        history.seek(2);
        assert_eq!(history.oldest_frame(), Some(2), "no frame was evicted");
        assert!(
            history.contains(9),
            "frame 9 is still retained after the rewind"
        );
        assert!(history.state_hash(9).is_some());
    }

    /// `truncate_after` drops the abandoned future and rebases the cursor, so the
    /// next capture records the restore point again (not `old_max + 1`).
    #[test]
    fn truncate_after_drops_the_future_and_rebases() {
        let w = world();
        let mut history = SnapshotHistory::new(8);
        for _ in 0..5 {
            history.capture(&w);
        }
        assert_eq!(history.next_frame(), 5);

        history.truncate_after(2);
        assert_eq!(history.next_frame(), 2);
        assert!(history.contains(2), "the restore point is kept");
        assert!(
            !history.contains(3) && !history.contains(4),
            "future dropped"
        );
        assert_eq!(history.oldest_frame(), Some(0));

        assert_eq!(history.capture(&w), 2, "the next capture is frame 2 again");
        assert_eq!(history.next_frame(), 3);
    }

    #[test]
    fn try_capture_reports_unregistered_components() {
        let mut w = World::new(); // no snapshot glue registered
        let _ = w.spawn((Counter(1), Marker));
        let mut history = SnapshotHistory::new(2);
        let missing = history.try_capture(&w).unwrap_err();
        assert!(!missing.is_empty());
        // The history is unchanged on refusal.
        assert!(history.is_empty());
        assert_eq!(history.next_frame(), 0);
    }

    #[test]
    fn state_hash_is_stable_and_detects_change() {
        let mut w = world();
        let e = w.spawn((Counter(10), Marker));
        let mut history = SnapshotHistory::new(4);
        let f0 = history.capture(&w);
        let h0 = history.state_hash(f0).unwrap();

        w.get_mut::<Counter>(e).unwrap().0 = 11;
        let f1 = history.capture(&w);
        let h1 = history.state_hash(f1).unwrap();
        assert_ne!(h0, h1, "distinct states hash differently");

        history.rollback(&mut w, f0);
        assert_eq!(history.state_hash(f0), Some(h0));
        assert_eq!(w.get::<Counter>(e).unwrap().0, 10);
        // Recomputing f0's hash is stable (deterministic fold).
        assert_eq!(history.state_hash(f0), Some(h0));
    }

    #[test]
    fn rollback_to_an_uncaptured_frame_is_a_noop() {
        let mut w = world();
        let e = w.spawn((Counter(7), Marker));
        let history = SnapshotHistory::new(2);
        assert!(!history.rollback(&mut w, 99));
        assert_eq!(w.get::<Counter>(e).unwrap().0, 7);
    }

    #[test]
    fn sub_app_capture_and_rollback() {
        let mut sub = SubApp::new();
        sub.world.register_snapshot_component_hashable::<Counter>();
        sub.world.register_snapshot_component::<Marker>();
        let e = sub.world.spawn((Counter(1), Marker));
        sub.init_snapshot_history(4);
        let frame = sub.capture_snapshot().expect("history installed");
        sub.world.get_mut::<Counter>(e).unwrap().0 = 5;
        assert!(sub.rollback_to(frame));
        assert_eq!(sub.world.get::<Counter>(e).unwrap().0, 1);
    }

    #[test]
    fn capture_and_rollback_without_history_report_absence() {
        let mut sub = SubApp::new();
        assert!(sub.capture_snapshot().is_none());
        assert!(!sub.rollback_to(0));
    }

    #[test]
    fn app_snapshot_rollback_roundtrip() {
        let mut app = App::new();
        app.world_mut()
            .register_snapshot_component_hashable::<Counter>();
        app.world_mut().register_snapshot_component::<Marker>();
        let e = app.world_mut().spawn((Counter(2), Marker));
        app.init_snapshot_history(4);
        let frame = app.capture_snapshot().expect("history installed");
        app.world_mut().get_mut::<Counter>(e).unwrap().0 = 9;
        assert_eq!(app.world().get::<Counter>(e).unwrap().0, 9);
        assert!(app.rollback_to(frame));
        assert_eq!(app.world().get::<Counter>(e).unwrap().0, 2);
    }
}
