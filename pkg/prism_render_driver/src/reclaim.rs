//! Deferred resource reclamation keyed by frames-in-flight.
//!
//! A GPU runs behind the CPU: when the CPU records frame *N* the GPU may still
//! be executing frame *N − k*. Destroying a buffer/texture the moment the CPU
//! drops it would free memory the GPU is about to read, so every explicit API
//! defers destruction until the frame that last used the resource has
//! finished. [`DeferredDeleter`] implements that policy: callers *enqueue* a
//! to-be-destroyed item tagged with the current frame index, and once a frame
//! is known complete they *collect* everything enqueued at or before it.
//!
//! The deleter is a monotonically-advancing timeline, not a fixed ring: the
//! CPU calls [`DeferredDeleter::begin_frame`] to stamp new work with a frame
//! number and [`DeferredDeleter::collect_completed`] when the GPU signals a
//! frame's fence, which yields exactly the items safe to destroy now. This
//! generalizes the common "N-deep ring" (the backend simply collects
//! `current - frames_in_flight`) while also supporting out-of-order or stalled
//! completion. Pure, `no_std`, no `unsafe`.

use alloc::vec::Vec;

/// A monotonic frame counter. Wraps `u64`, which at 1000 FPS lasts ~584
/// million years, so overflow is not a practical concern.
pub type FrameIndex = u64;

/// An item queued for destruction, tagged with the frame it became garbage in.
struct Pending<T> {
    /// The frame after which this item is safe to destroy.
    retire_after: FrameIndex,
    /// The payload (a native handle, id, or allocation to free).
    item: T,
}

/// Defers destruction of `T` until the frame that produced the garbage has
/// finished executing on the GPU.
///
/// Typical use: a backend holds one `DeferredDeleter<NativeResource>`, calls
/// [`Self::begin_frame`] at the top of each CPU frame, [`Self::enqueue`] for
/// every resource dropped during that frame, and [`Self::collect_completed`]
/// with the highest GPU-completed frame index to get back the handles it may
/// now actually destroy.
pub struct DeferredDeleter<T> {
    pending: Vec<Pending<T>>,
    current_frame: FrameIndex,
}

impl<T> DeferredDeleter<T> {
    /// Creates an empty deleter positioned at frame 0.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
            current_frame: 0,
        }
    }

    /// The frame index new [`Self::enqueue`] calls are currently stamped with.
    #[must_use]
    pub const fn current_frame(&self) -> FrameIndex {
        self.current_frame
    }

    /// The number of items still awaiting destruction.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Whether nothing is awaiting destruction.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Advances to a new CPU frame. Subsequent [`Self::enqueue`] calls tag
    /// items with this frame. The index must not move backward; a smaller or
    /// equal value is clamped to stay monotonic so reclamation can never be
    /// tricked into freeing live resources early.
    pub fn begin_frame(&mut self, frame: FrameIndex) {
        if frame > self.current_frame {
            self.current_frame = frame;
        }
    }

    /// Queues `item` for destruction once the current frame completes on the
    /// GPU.
    pub fn enqueue(&mut self, item: T) {
        self.pending.push(Pending {
            retire_after: self.current_frame,
            item,
        });
    }

    /// Queues `item` for destruction once the explicitly given frame
    /// completes. Useful when the last-use frame differs from the current
    /// frame (e.g. a resource read by an in-flight async-compute submission).
    pub fn enqueue_for_frame(&mut self, item: T, retire_after: FrameIndex) {
        self.pending.push(Pending { item, retire_after });
    }

    /// Removes and returns every item whose tagged frame is `<= completed_frame`
    /// — i.e. everything the GPU is now guaranteed to be done with. Items
    /// stamped with later frames stay queued. Order of the returned items is
    /// unspecified.
    pub fn collect_completed(&mut self, completed_frame: FrameIndex) -> Vec<T> {
        let mut ready = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].retire_after <= completed_frame {
                // swap_remove is O(1); ordering of reclamation does not matter.
                let p = self.pending.swap_remove(i);
                ready.push(p.item);
            } else {
                i += 1;
            }
        }
        ready
    }

    /// Removes and returns *every* pending item regardless of frame, for device
    /// teardown where the GPU is fully idle and all resources may be destroyed.
    pub fn drain_all(&mut self) -> Vec<T> {
        core::mem::take(&mut self.pending)
            .into_iter()
            .map(|p| p.item)
            .collect()
    }
}

impl<T> Default for DeferredDeleter<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defers_until_frame_completes() {
        let mut d: DeferredDeleter<u32> = DeferredDeleter::new();
        d.begin_frame(5);
        d.enqueue(100);
        d.enqueue(101);
        // GPU has only finished frame 4: nothing is safe yet.
        assert!(d.collect_completed(4).is_empty());
        assert_eq!(d.pending_len(), 2);
        // Frame 5 done: both are reclaimable.
        let mut got = d.collect_completed(5);
        got.sort_unstable();
        assert_eq!(got, alloc::vec![100, 101]);
        assert!(d.is_empty());
    }

    #[test]
    fn n_deep_ring_pattern() {
        // Emulate a 2-frame-in-flight ring: collect current-2 each frame.
        let mut d: DeferredDeleter<u64> = DeferredDeleter::new();
        let frames_in_flight = 2;
        for f in 0..6u64 {
            d.begin_frame(f);
            d.enqueue(f * 10);
            if f >= frames_in_flight {
                let ready = d.collect_completed(f - frames_in_flight);
                assert_eq!(ready, alloc::vec![(f - frames_in_flight) * 10]);
            }
        }
        // Two frames remain outstanding at the end.
        assert_eq!(d.pending_len(), 2);
    }

    #[test]
    fn begin_frame_is_monotonic() {
        let mut d: DeferredDeleter<u32> = DeferredDeleter::new();
        d.begin_frame(10);
        d.begin_frame(3); // ignored
        assert_eq!(d.current_frame(), 10);
        d.enqueue(1);
        // Must not be freed by completing the stale frame 3.
        assert!(d.collect_completed(9).is_empty());
        assert_eq!(d.collect_completed(10), alloc::vec![1]);
    }

    #[test]
    fn enqueue_for_specific_frame() {
        let mut d: DeferredDeleter<u32> = DeferredDeleter::new();
        d.begin_frame(1);
        d.enqueue_for_frame(7, 3);
        assert!(d.collect_completed(2).is_empty());
        assert_eq!(d.collect_completed(3), alloc::vec![7]);
    }

    #[test]
    fn drain_all_on_teardown() {
        let mut d: DeferredDeleter<u32> = DeferredDeleter::new();
        d.begin_frame(100);
        d.enqueue(1);
        d.enqueue(2);
        let mut all = d.drain_all();
        all.sort_unstable();
        assert_eq!(all, alloc::vec![1, 2]);
        assert!(d.is_empty());
    }
}
