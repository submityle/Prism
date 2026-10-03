//! Bounded lock-free single-producer / single-consumer (`SPSC`) ring buffer.
//!
//! Exactly one thread may own the [`SpscProducer`] and exactly one (other, or
//! the same) thread may own the [`SpscConsumer`]. Under that discipline the
//! queue is wait-free for both endpoints: a push or pop is a bounded,
//! loop-free sequence of a relaxed load, an acquire load, a slot write/read and
//! a release store. This is the diagnostic thread-local buffer form from design
//! doc §11 / §13.
//!
//! The two cursors live on separate cache lines (see
//! [`CachePadded`](super::CachePadded)) so the producer's `tail` stores never
//! invalidate the consumer's `head` line, and vice versa.

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::CachePadded;

/// Shared ring storage. Indices are *monotonic* counters (never wrapped); the
/// physical slot is `index % capacity`. The distance `tail - head` is the
/// number of occupied slots and is always in `0..=capacity`.
struct Inner<T> {
    /// Backing slots, each independently initialised/!initialised. Access is
    /// disjoint by construction: the producer only ever writes the slot it is
    /// about to publish, the consumer only ever reads a slot the producer has
    /// already published.
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Number of slots; also the modulus for mapping a cursor to a slot.
    capacity: usize,
    /// Consumer cursor: the index of the next element to pop. Only the consumer
    /// stores to it; the producer only loads it (to detect "full").
    head: CachePadded<AtomicUsize>,
    /// Producer cursor: the index of the next slot to push into. Only the
    /// producer stores to it; the consumer only loads it (to detect "empty").
    tail: CachePadded<AtomicUsize>,
}

// SAFETY: `Inner` hands out disjoint access to its slots (producer writes the
// publish slot, consumer reads already-published slots), synchronised by the
// acquire/release ordering on `head`/`tail`. Sending it across threads is
// therefore sound whenever the stored elements may themselves cross threads,
// i.e. when `T: Send`.
#[expect(
    unsafe_code,
    reason = "the ring synchronises disjoint slot access via its atomic cursors"
)]
// SAFETY: the ring synchronises disjoint slot access via its atomic cursors
unsafe impl<T: Send> Send for Inner<T> {}
// SAFETY: see the `Send` impl; shared access is only ever the producer and
// consumer touching disjoint slots under acquire/release synchronisation.
#[expect(
    unsafe_code,
    reason = "the ring synchronises disjoint slot access via its atomic cursors"
)]
// SAFETY: the ring synchronises disjoint slot access via its atomic cursors
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    #[inline]
    fn slot(&self, index: usize) -> &UnsafeCell<MaybeUninit<T>> {
        // `index` is a monotonic cursor; map it to a physical slot.
        &self.slots[index % self.capacity]
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Drop every element still queued (those in `head..tail`). After a
        // normal drop both endpoints are gone, so these loads need no
        // synchronisation beyond `&mut self` exclusivity.
        let head = *self.head.get_mut();
        let tail = *self.tail.get_mut();
        for index in head..tail {
            let slot = self.slots[index % self.capacity].get();
            #[expect(
                unsafe_code,
                reason = "slots in head..tail were published and are initialised"
            )]
            // SAFETY: every index in `head..tail` names a slot the producer
            // published and the consumer never popped, so it holds an
            // initialised `T` that we now own exclusively via `&mut self`.
            unsafe {
                (*slot).assume_init_drop();
            }
        }
    }
}

/// The producing endpoint of an [`SpscQueue`]. Not clonable: there is only ever
/// one producer.
pub struct SpscProducer<T> {
    inner: Arc<Inner<T>>,
}

/// The consuming endpoint of an [`SpscQueue`]. Not clonable: there is only ever
/// one consumer.
pub struct SpscConsumer<T> {
    inner: Arc<Inner<T>>,
}

// SAFETY: `SpscProducer`/`SpscConsumer` are thin owners of the shared `Inner`,
// which is already `Send + Sync` for `T: Send`. Each endpoint is meant to be
// held (and used) by a single thread, so only `Send` is required to move it
// onto a worker thread.
#[expect(
    unsafe_code,
    reason = "endpoints are single-threaded owners of a Send/Sync ring"
)]
// SAFETY: endpoints are single-threaded owners of a Send/Sync ring
unsafe impl<T: Send> Send for SpscProducer<T> {}
// SAFETY: see above.
#[expect(
    unsafe_code,
    reason = "endpoints are single-threaded owners of a Send/Sync ring"
)]
// SAFETY: endpoints are single-threaded owners of a Send/Sync ring
unsafe impl<T: Send> Send for SpscConsumer<T> {}

/// Namespace handle for constructing a bounded `SPSC` ring. Call
/// [`SpscQueue::with_capacity`] to obtain a connected
/// ([`SpscProducer`], [`SpscConsumer`]) pair.
#[derive(Debug)]
pub struct SpscQueue<T> {
    _never: core::marker::PhantomData<T>,
}

impl<T> SpscQueue<T> {
    /// Creates a bounded ring that can hold up to `capacity` elements and
    /// returns its connected producer/consumer endpoints.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> (SpscProducer<T>, SpscConsumer<T>) {
        assert!(capacity > 0, "SPSC capacity must be non-zero");
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || UnsafeCell::new(MaybeUninit::uninit()));
        let inner = Arc::new(Inner {
            slots: slots.into_boxed_slice(),
            capacity,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
        });
        (
            SpscProducer {
                inner: Arc::clone(&inner),
            },
            SpscConsumer { inner },
        )
    }
}

impl<T> SpscProducer<T> {
    /// Attempts to push `value`. Returns `Err(value)` (handing the element
    /// back) if the ring is full.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let inner = &*self.inner;
        // Only this thread writes `tail`, so a relaxed load reads our own value.
        let tail = inner.tail.load(Ordering::Relaxed);
        // Acquire-load `head` to observe the consumer's latest progress.
        let head = inner.head.load(Ordering::Acquire);
        if tail - head == inner.capacity {
            return Err(value);
        }
        let slot = inner.slot(tail).get();
        #[expect(
            unsafe_code,
            reason = "the producer uniquely owns the publish slot until it stores tail"
        )]
        // SAFETY: `tail - head < capacity` means slot `tail % capacity` is free
        // (the consumer has already popped any previous occupant and will not
        // touch it until we publish by storing `tail + 1`). We hold the sole
        // producer, so no other writer races us here.
        unsafe {
            (*slot).write(value);
        }
        // Release-store publishes the write to the consumer.
        inner.tail.store(tail + 1, Ordering::Release);
        Ok(())
    }

    /// Returns `true` if the ring currently has no free slots. This is a
    /// momentary observation in a concurrent setting.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        let inner = &*self.inner;
        let tail = inner.tail.load(Ordering::Relaxed);
        let head = inner.head.load(Ordering::Acquire);
        tail - head == inner.capacity
    }

    /// The fixed capacity of the ring.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

impl<T> SpscConsumer<T> {
    /// Attempts to pop the oldest element. Returns `None` if the ring is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let inner = &*self.inner;
        // Only this thread writes `head`, so a relaxed load reads our own value.
        let head = inner.head.load(Ordering::Relaxed);
        // Acquire-load `tail` to observe the producer's latest publication.
        let tail = inner.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let slot = inner.slot(head).get();
        #[expect(
            unsafe_code,
            reason = "slot head..tail was published by the producer and is initialised"
        )]
        // SAFETY: `head != tail` means slot `head % capacity` holds an element
        // the producer published (its release-store on `tail` happens-before
        // our acquire-load). We are the sole consumer, so we read it exactly
        // once before advancing `head`.
        let value = unsafe { (*slot).assume_init_read() };
        // Release-store frees the slot for the producer to reuse.
        inner.head.store(head + 1, Ordering::Release);
        Some(value)
    }

    /// Returns `true` if the ring currently has no elements. This is a
    /// momentary observation in a concurrent setting.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let inner = &*self.inner;
        let head = inner.head.load(Ordering::Relaxed);
        let tail = inner.tail.load(Ordering::Acquire);
        head == tail
    }

    /// A momentary count of queued elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        let inner = &*self.inner;
        let tail = inner.tail.load(Ordering::Acquire);
        let head = inner.head.load(Ordering::Relaxed);
        tail - head
    }

    /// The fixed capacity of the ring.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

impl<T> fmt::Debug for SpscProducer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpscProducer")
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

impl<T> fmt::Debug for SpscConsumer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpscConsumer")
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}
