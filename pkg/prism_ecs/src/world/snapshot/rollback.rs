//! A fixed-capacity ring of frame snapshots that bounds rollback memory
//! (design §14: 回滚内存靠 chunk 增量快照控制 / 回滚到任意已确认帧并重放输入).
//!
//! Rollback networking (Quantum / GGPO form) keeps a sliding window of recently
//! confirmed authoritative frames: when a late input arrives for frame `f`, the
//! simulation restores the snapshot it stored for `f` and re-simulates forward.
//! A [`SnapshotRing`] caps that window to a fixed number of frames, evicting the
//! oldest when full, so history memory is bounded regardless of session length.
//! Pair it with [`SnapshotDelta`](super::SnapshotDelta) to shrink per-frame cost
//! to the changed cells.

use alloc::collections::VecDeque;

use super::WorldSnapshot;

/// A bounded ring buffer mapping a frame number to its [`WorldSnapshot`],
/// evicting the oldest frame once `capacity` is reached (design §14).
pub struct SnapshotRing {
    /// The maximum number of frames retained at once.
    capacity: usize,
    /// Retained `(frame, snapshot)` pairs, oldest at the front.
    frames: VecDeque<(u64, WorldSnapshot)>,
}

impl SnapshotRing {
    /// Create a ring that retains at most `capacity` frames.
    ///
    /// # Panics
    /// Panics if `capacity` is zero (a ring must retain at least one frame).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SnapshotRing capacity must be non-zero");
        Self {
            capacity,
            frames: VecDeque::with_capacity(capacity),
        }
    }

    /// The maximum number of frames retained at once.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of frames currently retained.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frames are currently retained.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Store `snapshot` for `frame`, evicting the oldest frame if the ring is
    /// full. Re-storing an already-present frame overwrites its snapshot in
    /// place (idempotent re-confirmation) without touching eviction order.
    pub fn push(&mut self, frame: u64, snapshot: WorldSnapshot) {
        if let Some(slot) = self.frames.iter_mut().find(|(f, _)| *f == frame) {
            slot.1 = snapshot;
            return;
        }
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back((frame, snapshot));
    }

    /// Borrow the snapshot stored for `frame`, if still retained.
    pub fn get(&self, frame: u64) -> Option<&WorldSnapshot> {
        self.frames
            .iter()
            .find(|(f, _)| *f == frame)
            .map(|(_, s)| s)
    }

    /// Whether `frame` is still retained.
    #[inline]
    pub fn contains(&self, frame: u64) -> bool {
        self.frames.iter().any(|(f, _)| *f == frame)
    }

    /// The newest retained `(frame, snapshot)`, if any.
    pub fn latest(&self) -> Option<(u64, &WorldSnapshot)> {
        self.frames.back().map(|(f, s)| (*f, s))
    }

    /// The oldest retained `(frame, snapshot)`, if any (the next to be evicted).
    pub fn oldest(&self) -> Option<(u64, &WorldSnapshot)> {
        self.frames.front().map(|(f, s)| (*f, s))
    }

    /// The retained frame numbers, oldest first.
    pub fn frames(&self) -> impl Iterator<Item = u64> + '_ {
        self.frames.iter().map(|(f, _)| *f)
    }

    /// Drop every retained frame.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}
