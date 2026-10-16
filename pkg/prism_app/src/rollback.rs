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
//! A snapshot captures entity / component / change-tick / relation state and
//! explicitly registered snapshot resources, and it is registry-bound:
//! register each component's clone glue with
//! [`World::register_snapshot_component`] first. [`SnapshotHistory::capture`]
//! panics on an unregistered resident component;
//! [`try_capture`](SnapshotHistory::try_capture) reports the offenders instead.
//! [`state_hash`](SnapshotHistory::state_hash) folds only the component values
//! whose types were registered as *hashable*
//! ([`World::register_snapshot_component_hashable`]); a component with clone glue
//! but no value hashing contributes structure but not its value to the hash.

use alloc::sync::Arc;
use alloc::vec::Vec;

use prism_ecs::component::ComponentId;
use prism_ecs::resource::Resource;
use prism_ecs::world::snapshot::SnapshotAppendUndo;
use prism_ecs::world::snapshot::SnapshotRing;
use prism_ecs::world::snapshot::WorldSnapshot;
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
    cursor_identity: Arc<()>,
}

/// A reversible, monotonically appended history frame. Keep this only until
/// the operation following capture (for example a network send) succeeds.
pub struct SnapshotCaptureUndo {
    frame: u64,
    ring: SnapshotAppendUndo,
    cursor_identity: Arc<()>,
}

impl Resource for SnapshotHistory {}

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
            cursor_identity: Arc::new(()),
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

    /// The frame number the next [`capture`](Self::capture) will record.
    /// Frame keys do not wrap: `u64::MAX` denotes an exhausted cursor.
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
    /// Also panics before capture if the frame cursor is exhausted. Destruction
    /// of displaced values may panic after both the ring and cursor commit.
    pub fn capture(&mut self, world: &World) -> u64 {
        self.checked_next_frame();
        self.record_snapshot(world.snapshot())
    }

    /// Fallible [`capture`](Self::capture): on an unregistered component the
    /// snapshot is refused and the offending component ids are returned,
    /// leaving the history unchanged (the frame counter does not advance).
    pub fn try_capture(&mut self, world: &World) -> Result<u64, Vec<ComponentId>> {
        self.checked_next_frame();
        let snapshot = world.try_snapshot()?;
        Ok(self.record_snapshot(snapshot))
    }

    /// Capture a strictly increasing frame with an eviction-preserving undo
    /// token. The caller must not have sought into retained history. Exhausted
    /// or non-increasing frame keys panic without changing the history.
    pub fn try_capture_reversible(
        &mut self,
        world: &World,
    ) -> Result<(u64, SnapshotCaptureUndo), Vec<ComponentId>> {
        let next = self.checked_next_frame();
        let frame = self.next_frame;
        let snapshot = world.try_snapshot()?;
        let ring = self.ring.append_reversible(frame, snapshot);
        self.next_frame = next;
        Ok((
            frame,
            SnapshotCaptureUndo {
                frame,
                ring,
                cursor_identity: self.cursor_identity.clone(),
            },
        ))
    }

    /// Refuse a stale token, or restore the preceding frame window and cursor.
    /// Restoring the live World is the transaction owner's responsibility.
    /// A successful undo publishes both before destroying displaced values;
    /// destructor panics do not roll back external side effects.
    pub fn undo_capture(&mut self, undo: SnapshotCaptureUndo) -> bool {
        if undo.frame.checked_add(1) != Some(self.next_frame)
            || !Arc::ptr_eq(&self.cursor_identity, &undo.cursor_identity)
        {
            return false;
        }
        let frame = undo.frame;
        let Some(retired) = self.ring.undo_append_deferred(undo.ring) else {
            return false;
        };
        self.next_frame = frame;
        drop(retired);
        true
    }

    /// Record an already captured snapshot at the cursor. Used to publish a
    /// successful replay without executing gameplay a second time.
    pub fn record_snapshot(&mut self, snapshot: WorldSnapshot) -> u64 {
        let next = self.checked_next_frame();
        let frame = self.next_frame;
        let retired = self.ring.push_deferred(frame, snapshot);
        self.next_frame = next;
        drop(retired);
        frame
    }

    fn checked_next_frame(&self) -> u64 {
        self.next_frame
            .checked_add(1)
            .expect("snapshot history frame cursor exhausted")
    }

    /// Copy the retained prefix into a separate bounded branch. The original
    /// remains intact until the caller publishes the completed replacement.
    #[must_use]
    pub fn branch_through(&self, frame: u64) -> Self {
        let next = frame
            .checked_add(1)
            .expect("snapshot history frame cursor exhausted");
        let mut branch = Self::new(self.capacity());
        for key in self.ring.frames().take_while(|&key| key <= frame) {
            branch.seek(key);
            branch.record_snapshot(self.ring.get(key).expect("retained prefix key"));
        }
        branch.seek(next);
        branch
    }

    /// Restore `world` to `frame` if it is still retained.
    ///
    /// Returns `true` when the restore ran and `false` when `frame` has been
    /// evicted (or was never captured), in which case `world` is untouched.
    pub fn rollback(&self, world: &mut World, frame: u64) -> bool {
        match self.ring.get(frame) {
            Some(snapshot) => {
                world.restore(&snapshot);
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
        // Even seeking away and back invalidates an outstanding undo token.
        self.cursor_identity = Arc::new(());
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
        // Ring truncation detaches the entire tail before invoking destructors.
        // Publish its associated cursor first so destructor panics stay coherent.
        self.seek(frame);
        self.ring.truncate_after(frame);
    }

    /// The deterministic state hash of a retained frame, for desync comparison
    /// (design §14, 逐帧状态哈希去同步). `None` once the frame is evicted.
    #[must_use]
    pub fn state_hash(&self, frame: u64) -> Option<u64> {
        self.ring.get(frame).map(|snapshot| snapshot.state_hash())
    }

    /// Reconstruct an owned snapshot for a retained frame. This is the
    /// equality-grade inspection API for rollback tests and diagnostics;
    /// [`state_hash`](Self::state_hash) remains only a bounded diagnostic.
    #[must_use]
    pub fn snapshot(&self, frame: u64) -> Option<WorldSnapshot> {
        self.ring.get(frame)
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
    fn reversible_capture_restores_the_evicted_window_and_rejects_foreign_tokens() {
        let mut w = world();
        let entity = w.spawn(Counter(0));
        let mut history = SnapshotHistory::new(2);
        for value in 0..3 {
            w.get_mut::<Counter>(entity).unwrap().0 = value;
            history.capture(&w);
        }
        let old_hashes = [history.state_hash(1), history.state_hash(2)];
        let (_, undo) = history.try_capture_reversible(&w).unwrap();
        assert_eq!(history.oldest_frame(), Some(2));
        assert!(history.undo_capture(undo));
        assert_eq!(history.next_frame(), 3);
        assert_eq!(history.oldest_frame(), Some(1));
        assert_eq!([history.state_hash(1), history.state_hash(2)], old_hashes);

        let mut other = SnapshotHistory::new(2);
        other.capture(&w);
        let (_, foreign) = other.try_capture_reversible(&w).unwrap();
        let mut target = SnapshotHistory::new(2);
        target.capture(&w);
        target.capture(&w);
        assert!(!target.undo_capture(foreign));
        assert_eq!(target.next_frame(), 2);
        assert_eq!(target.len(), 2);
    }

    #[test]
    fn independent_branch_keeps_the_original_prefix_and_cursor() {
        let mut w = world();
        let entity = w.spawn(Counter(0));
        let mut history = SnapshotHistory::new(4);
        for value in 0..6 {
            w.get_mut::<Counter>(entity).unwrap().0 = value;
            history.capture(&w);
        }
        let mut branch = history.branch_through(3);
        assert_eq!(branch.oldest_frame(), Some(2));
        assert_eq!(branch.next_frame(), 4);
        assert_eq!(branch.state_hash(3), history.state_hash(3));
        w.get_mut::<Counter>(entity).unwrap().0 = 99;
        branch.capture(&w);
        assert_ne!(branch.state_hash(4), history.state_hash(4));
        assert_eq!(history.next_frame(), 6);
        assert!(history.contains(5));
        assert!(!branch.contains(5));
    }

    #[test]
    fn reversible_windows_restore_exact_states_at_capacities_one_and_four() {
        for capacity in [1, 4] {
            let mut w = world();
            w.register_snapshot_component_equatable::<Counter>();
            let entity = w.spawn(Counter(0));
            let mut history = SnapshotHistory::new(capacity);
            for frame in 0..12 {
                w.get_mut::<Counter>(entity).unwrap().0 = frame;
                history.capture(&w);
            }
            let before: Vec<_> = history
                .ring
                .frames()
                .map(|key| (key, history.snapshot(key).unwrap()))
                .collect();
            w.get_mut::<Counter>(entity).unwrap().0 = 99;
            let (_, undo) = history.try_capture_reversible(&w).unwrap();
            assert_eq!(history.len(), capacity);
            // Tokens follow the ring through a move, not its stack address.
            let mut moved = history;
            assert!(moved.undo_capture(undo));
            assert_eq!(moved.next_frame(), 12);
            assert_eq!(moved.len(), capacity);
            for (key, snapshot) in before {
                assert!(moved.snapshot(key).unwrap().structurally_eq(&snapshot));
            }
            assert_eq!(
                w.get::<Counter>(entity).unwrap().0,
                99,
                "undo does not roll back the live World or external effects"
            );
        }
    }

    #[test]
    fn exhausted_cursors_refuse_all_capture_paths_before_writing() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let w = world();
        for capacity in [1, 4] {
            let mut history = SnapshotHistory::new(capacity);
            history.seek(u64::MAX - 1);
            assert_eq!(history.capture(&w), u64::MAX - 1);
            let before = history.state_hash(u64::MAX - 1);
            for path in 0..4 {
                assert!(catch_unwind(AssertUnwindSafe(|| {
                    match path {
                        0 => {
                            history.capture(&w);
                        }
                        1 => {
                            let _ = history.try_capture(&w);
                        }
                        2 => {
                            let _ = history.try_capture_reversible(&w);
                        }
                        _ => {
                            history.record_snapshot(w.snapshot());
                        }
                    }
                }))
                .is_err());
                assert_eq!(history.next_frame(), u64::MAX);
                assert_eq!(history.len(), 1);
                assert!(!history.contains(u64::MAX));
                assert_eq!(history.state_hash(u64::MAX - 1), before);
            }
            assert!(catch_unwind(AssertUnwindSafe(|| history.branch_through(u64::MAX))).is_err());
        }
    }

    #[test]
    fn seek_round_trips_and_later_mutations_invalidate_capture_tokens() {
        let w = world();
        for mutation in 0..4 {
            let mut history = SnapshotHistory::new(4);
            let (_, undo) = history.try_capture_reversible(&w).unwrap();
            match mutation {
                0 => {
                    history.seek(9);
                    history.seek(1);
                }
                1 => {
                    history.seek(1);
                }
                2 => {
                    history.capture(&w);
                    history.truncate_after(0);
                    history.seek(1);
                }
                _ => {
                    let (_, newer) = history.try_capture_reversible(&w).unwrap();
                    assert!(history.undo_capture(newer));
                }
            }
            assert!(!history.undo_capture(undo));
            assert_eq!(history.next_frame(), 1);
            assert!(history.contains(0));
        }
    }

    #[derive(Clone)]
    struct DropBomb {
        value: u64,
        armed: Arc<core::sync::atomic::AtomicU64>,
    }

    impl Resource for DropBomb {}

    impl PartialEq for DropBomb {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Drop for DropBomb {
        fn drop(&mut self) {
            use core::sync::atomic::Ordering::SeqCst;
            if self
                .armed
                .compare_exchange(self.value, u64::MAX, SeqCst, SeqCst)
                .is_ok()
            {
                panic!("injected snapshot destructor panic");
            }
        }
    }

    fn bomb_world() -> (World, Arc<core::sync::atomic::AtomicU64>) {
        let armed = Arc::new(core::sync::atomic::AtomicU64::new(u64::MAX));
        let mut w = world();
        w.register_snapshot_resource_equatable::<DropBomb>();
        w.insert_resource(DropBomb {
            value: 0,
            armed: armed.clone(),
        });
        (w, armed)
    }

    #[test]
    fn destructor_panics_leave_capture_and_undo_cursors_committed() {
        use core::sync::atomic::Ordering::SeqCst;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        for capacity in [1, 4] {
            for undoing in [false, true] {
                let (mut w, armed) = bomb_world();
                let mut history = SnapshotHistory::new(capacity);
                for frame in 0..capacity {
                    w.resource_mut::<DropBomb>().value = frame as u64;
                    history.capture(&w);
                }
                w.resource_mut::<DropBomb>().value = 99;
                if undoing {
                    let (_, undo) = history.try_capture_reversible(&w).unwrap();
                    armed.store(99, SeqCst);
                    assert!(catch_unwind(AssertUnwindSafe(|| history.undo_capture(undo))).is_err());
                    assert_eq!(history.next_frame(), capacity as u64);
                    assert_eq!(history.oldest_frame(), Some(0));
                    assert!(!history.contains(capacity as u64));
                } else {
                    armed.store(0, SeqCst);
                    assert!(catch_unwind(AssertUnwindSafe(|| history.capture(&w))).is_err());
                    assert_eq!(history.next_frame(), capacity as u64 + 1);
                    assert_eq!(history.oldest_frame(), Some(1));
                    assert!(history.contains(capacity as u64));
                }
                assert_eq!(history.len(), capacity);
                let restored = history.oldest_frame().unwrap();
                assert!(history.rollback(&mut w, restored));
                let expected = if restored == capacity as u64 {
                    99
                } else {
                    restored
                };
                assert_eq!(w.resource::<DropBomb>().value, expected);
            }
        }
    }

    #[test]
    fn refused_capture_keeps_the_window_cursor_and_outstanding_token() {
        struct Unregistered;
        impl Component for Unregistered {}
        for capacity in [1, 4] {
            let mut w = world();
            let mut history = SnapshotHistory::new(capacity);
            for _ in 0..capacity {
                history.capture(&w);
            }
            let (_, undo) = history.try_capture_reversible(&w).unwrap();
            let oldest = history.oldest_frame();
            let entity = w.spawn(Unregistered);
            assert!(history.try_capture(&w).is_err());
            assert!(history.try_capture_reversible(&w).is_err());
            assert_eq!(history.next_frame(), capacity as u64 + 1);
            assert_eq!(history.oldest_frame(), oldest);
            assert_eq!(history.len(), capacity);
            assert!(history.undo_capture(undo));
            assert_eq!(history.next_frame(), capacity as u64);
            assert!(w.get::<Unregistered>(entity).is_some());
        }
    }

    #[test]
    fn input_destructor_panics_before_commit_and_preserves_existing_token() {
        use core::sync::atomic::Ordering::SeqCst;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let (mut w, armed) = bomb_world();
        let mut history = SnapshotHistory::new(4);
        history.capture(&w);
        w.resource_mut::<DropBomb>().value = 1;
        let (_, undo) = history.try_capture_reversible(&w).unwrap();
        w.resource_mut::<DropBomb>().value = 99;
        armed.store(99, SeqCst);
        assert!(catch_unwind(AssertUnwindSafe(|| history.try_capture_reversible(&w))).is_err());
        assert_eq!(history.next_frame(), 2);
        assert_eq!(history.len(), 2);
        assert!(history.undo_capture(undo));
        assert_eq!(history.next_frame(), 1);
    }

    #[test]
    fn replacement_destructor_panic_still_invalidates_the_ring_token() {
        use core::sync::atomic::Ordering::SeqCst;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let (mut w, armed) = bomb_world();
        let mut history = SnapshotHistory::new(4);
        history.capture(&w);
        w.resource_mut::<DropBomb>().value = 1;
        let (_, undo) = history.try_capture_reversible(&w).unwrap();
        w.resource_mut::<DropBomb>().value = 99;
        armed.store(1, SeqCst);
        assert!(catch_unwind(AssertUnwindSafe(|| history.ring.push(1, w.snapshot()))).is_err());
        assert!(!history.undo_capture(undo));
        assert_eq!(history.next_frame(), 2);
        assert!(history.rollback(&mut w, 1));
        assert_eq!(w.resource::<DropBomb>().value, 99);
    }

    #[test]
    fn truncate_detaches_the_entire_tail_before_destructors_run() {
        use core::sync::atomic::Ordering::SeqCst;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let (mut w, armed) = bomb_world();
        let mut history = SnapshotHistory::new(4);
        for frame in 0..5 {
            w.resource_mut::<DropBomb>().value = frame;
            history.capture(&w);
        }
        armed.store(4, SeqCst);
        assert!(catch_unwind(AssertUnwindSafe(|| history.truncate_after(1))).is_err());
        assert_eq!(history.next_frame(), 1);
        assert_eq!(history.len(), 1);
        assert_eq!(history.ring.frames().collect::<Vec<_>>(), [1]);
        assert!(history.rollback(&mut w, 1));
        assert_eq!(w.resource::<DropBomb>().value, 1);
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
