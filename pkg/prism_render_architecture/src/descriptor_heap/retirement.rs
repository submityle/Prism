//! Deferred slot retirement keyed on frame epochs.
//!
//! A descriptor slot cannot be recycled the instant its handle is retired: the
//! GPU may still be reading that index for frames already in flight. This queue
//! records each retired slot together with the frame epoch at which it was
//! retired, then releases it only once enough frames have elapsed that no
//! in-flight submission can still reference it.
//!
//! The release rule is `current_epoch >= retire_epoch + frames_in_flight`. With
//! `frames_in_flight` equal to the number of frames the GPU may buffer, a slot
//! retired on frame `N` is reclaimable no earlier than frame
//! `N + frames_in_flight`, by which point every submission that could have used
//! it has retired on the GPU timeline. The queue is `GPU`-independent: it holds
//! only slot indices and epochs, and the caller drives it from whatever frame
//! pacing the backend reports.
//!
//! Ordering is deterministic. Entries are appended in retirement order and
//! reclaimed in that same order, so a fixed retire/reclaim schedule always
//! frees slots in a reproducible sequence.

use alloc::vec::Vec;

/// One retired slot awaiting reclamation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PendingSlot {
    /// Slot index within the owning segment.
    index: u32,
    /// Frame epoch at which the slot was retired.
    retire_epoch: u64,
}

/// FIFO queue of retired slots pending epoch-gated reclamation.
///
/// One queue backs one descriptor segment. It never reclaims eagerly; the owner
/// calls [`reclaim`](Self::reclaim) each frame with the current epoch and the
/// GPU's frame-in-flight depth and returns the freed indices to its allocator.
#[derive(Clone, Debug, Default)]
pub struct RetirementQueue {
    pending: Vec<PendingSlot>,
}

impl RetirementQueue {
    /// Creates an empty retirement queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Records `index` as retired at `frame_epoch`.
    ///
    /// The caller is expected to have already invalidated the handle; this only
    /// schedules the physical slot for later reclamation.
    pub fn retire(&mut self, index: u32, frame_epoch: u64) {
        self.pending.push(PendingSlot {
            index,
            retire_epoch: frame_epoch,
        });
    }

    /// Number of slots awaiting reclamation.
    #[must_use]
    pub fn pending_count(&self) -> u32 {
        self.pending.len() as u32
    }

    /// Whether no slots are pending reclamation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Releases every slot whose retirement is old enough to be safe.
    ///
    /// A slot is released when `current_epoch >= retire_epoch + frames_in_flight`.
    /// Released indices are returned in retirement order; still-pending slots are
    /// retained in their original order. A `frames_in_flight` of `0` reclaims any
    /// slot retired at or before `current_epoch`.
    #[must_use]
    pub fn reclaim(&mut self, current_epoch: u64, frames_in_flight: u32) -> Vec<u32> {
        let hold = u64::from(frames_in_flight);
        let mut released = Vec::new();
        let mut retained = Vec::with_capacity(self.pending.len());
        for slot in self.pending.drain(..) {
            let ready = current_epoch >= slot.retire_epoch.saturating_add(hold);
            if ready {
                released.push(slot.index);
            } else {
                retained.push(slot);
            }
        }
        self.pending = retained;
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_queue_reclaims_nothing() {
        let mut queue = RetirementQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.pending_count(), 0);
        assert!(queue.reclaim(100, 2).is_empty());
    }

    #[test]
    fn slot_is_held_until_frames_in_flight_elapse() {
        let mut queue = RetirementQueue::new();
        queue.retire(4, 5);
        assert_eq!(queue.pending_count(), 1);
        // Retired at epoch 5, three frames in flight => not ready before 8.
        assert!(queue.reclaim(6, 3).is_empty());
        assert!(queue.reclaim(7, 3).is_empty());
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.reclaim(8, 3), alloc::vec![4]);
        assert!(queue.is_empty());
    }

    #[test]
    fn zero_frames_in_flight_reclaims_at_same_epoch() {
        let mut queue = RetirementQueue::new();
        queue.retire(2, 10);
        assert_eq!(queue.reclaim(10, 0), alloc::vec![2]);
    }

    #[test]
    fn reclaim_preserves_retirement_order() {
        let mut queue = RetirementQueue::new();
        queue.retire(7, 1);
        queue.retire(3, 1);
        queue.retire(9, 1);
        assert_eq!(queue.reclaim(3, 2), alloc::vec![7, 3, 9]);
    }

    #[test]
    fn partial_reclaim_retains_newer_entries_in_order() {
        let mut queue = RetirementQueue::new();
        queue.retire(1, 1);
        queue.retire(2, 4);
        queue.retire(3, 2);
        // frames_in_flight 2: ready when current >= retire + 2.
        // current 3 => entry@1 ready (3>=3), entry@4 not (3<6), entry@2 ready (3>=4? no).
        let released = queue.reclaim(3, 2);
        assert_eq!(released, alloc::vec![1]);
        assert_eq!(queue.pending_count(), 2);
        // Advance far enough to drain the rest, still in original order.
        assert_eq!(queue.reclaim(100, 2), alloc::vec![2, 3]);
    }

    #[test]
    fn saturating_hold_does_not_overflow() {
        let mut queue = RetirementQueue::new();
        queue.retire(0, 10);
        // A huge hold saturates rather than overflowing, so the slot stays
        // pending instead of wrapping to an early release.
        assert!(queue.reclaim(5, u32::MAX).is_empty());
        assert_eq!(queue.pending_count(), 1);
        // At the saturated boundary the slot finally releases.
        assert_eq!(queue.reclaim(u64::MAX, u32::MAX), alloc::vec![0]);
    }
}
