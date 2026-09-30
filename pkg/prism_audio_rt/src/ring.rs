//! Bounded, lock-free single-payload rings used to move control and telemetry
//! data across the audio-callback / task-thread boundary.
//!
//! The rings are thin, real-time-safe wrappers around
//! [`crossbeam_queue::ArrayQueue`]: capacity is fixed at construction, so
//! [`RingProducer::push`] and [`RingConsumer::pop`] never allocate, never lock,
//! and never block. When the ring is full, `push` hands the payload back to the
//! caller instead of growing or spinning, which keeps the audio thread bounded.
//!
//! Two directions are modelled with the same primitive:
//!
//! - The *command ring* carries [`crate::command::AudioCommand`] from any task
//!   thread (multi-producer) to the single audio thread (single consumer).
//! - The *telemetry ring* carries [`crate::telemetry::TelemetryFrame`] from the
//!   single audio thread (single producer) back to an observer thread (single
//!   consumer).
//!
//! `ArrayQueue` is itself multi-producer / multi-consumer, so both patterns are
//! safe supersets of what we expose here.

use alloc::sync::Arc;

use crossbeam_queue::ArrayQueue;

/// The sending half of a bounded ring. Cloning yields another producer that
/// shares the same backing queue, which is how the command ring supports many
/// task-thread producers feeding one audio-thread consumer.
#[derive(Debug)]
pub struct RingProducer<T> {
    /// Shared backing queue; construction pre-allocates all slots.
    queue: Arc<ArrayQueue<T>>,
}

impl<T> Clone for RingProducer<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
        }
    }
}

impl<T> RingProducer<T> {
    /// Enqueues `item`.
    ///
    /// Returns `Ok(())` on success. When the ring is full the item is handed
    /// back as `Err(item)` so the caller can decide whether to retry, coalesce,
    /// or drop it; the call never allocates, locks, or blocks.
    ///
    /// # Errors
    ///
    /// Returns `Err(item)` when the ring has no free slot.
    #[inline]
    pub fn push(&self, item: T) -> Result<(), T> {
        self.queue.push(item)
    }

    /// Total number of slots reserved at construction.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Number of items currently queued (a racy snapshot under concurrency).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the ring currently holds no items (a racy snapshot).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Whether the ring currently has no free slot (a racy snapshot).
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }
}

/// The receiving half of a bounded ring. Cloning shares the same backing queue.
#[derive(Debug)]
pub struct RingConsumer<T> {
    /// Shared backing queue; construction pre-allocates all slots.
    queue: Arc<ArrayQueue<T>>,
}

impl<T> Clone for RingConsumer<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            queue: Arc::clone(&self.queue),
        }
    }
}

impl<T> RingConsumer<T> {
    /// Dequeues the oldest item, or returns `None` when the ring is empty.
    ///
    /// The call never allocates, locks, or blocks, so it is safe to drain from
    /// the audio-callback thread.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }

    /// Total number of slots reserved at construction.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Number of items currently queued (a racy snapshot under concurrency).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the ring currently holds no items (a racy snapshot).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Creates a bounded ring with `capacity` pre-allocated slots and returns the
/// paired producer and consumer.
///
/// `capacity` is clamped up to `1` so the returned ring always has at least one
/// usable slot (`ArrayQueue` panics on a zero capacity).
#[must_use]
pub fn ring<T>(capacity: usize) -> (RingProducer<T>, RingConsumer<T>) {
    let queue = Arc::new(ArrayQueue::new(capacity.max(1)));
    (
        RingProducer {
            queue: Arc::clone(&queue),
        },
        RingConsumer { queue },
    )
}

#[cfg(test)]
mod tests {
    use super::ring;

    #[test]
    fn push_pop_round_trips_in_fifo_order() {
        let (tx, rx) = ring::<u32>(4);
        assert_eq!(tx.capacity(), 4);
        assert!(tx.is_empty());
        for value in 0..4 {
            assert_eq!(tx.push(value), Ok(()));
        }
        assert!(tx.is_full());
        assert_eq!(tx.push(99), Err(99));
        for value in 0..4 {
            assert_eq!(rx.pop(), Some(value));
        }
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let (tx, rx) = ring::<u8>(0);
        assert_eq!(tx.capacity(), 1);
        assert_eq!(tx.push(7), Ok(()));
        assert_eq!(tx.push(8), Err(8));
        assert_eq!(rx.pop(), Some(7));
    }

    #[test]
    fn cloned_producers_share_backing_storage() {
        let (tx, rx) = ring::<u16>(8);
        let tx2 = tx.clone();
        assert_eq!(tx.push(1), Ok(()));
        assert_eq!(tx2.push(2), Ok(()));
        assert_eq!(rx.pop(), Some(1));
        assert_eq!(rx.pop(), Some(2));
    }
}
