//! Bounded lock-free multi-producer / multi-consumer (`MPMC`) queue.
//!
//! This is Dmitry Vyukov's classic bounded `MPMC` queue: a power-of-two array
//! of cells, each carrying a `sequence` number, plus two monotonic cursors
//! (`enqueue_pos`, `dequeue_pos`). A producer or consumer claims a slot with a
//! single `compare_exchange` on the relevant cursor and then uses the cell's
//! sequence number as a per-slot handshake, so producers and consumers never
//! block each other and the queue is lock-free (an operation only retries when
//! some *other* operation made progress).
//!
//! It is the cross-thread message / work-queue form from design doc §11 /
//! §24.2. [`MpmcQueue`] is a cheaply clonable handle (an [`Arc`] inside), so it
//! can be shared with any number of producer and consumer threads.

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::CachePadded;

/// One ring cell: a published-ness handshake (`sequence`) plus the payload.
struct Cell<T> {
    /// Handshake counter. Invariant mapping to a cursor `pos`:
    /// `sequence == pos`      → the cell is empty and ready for the producer at
    ///                          `pos`; `sequence == pos + 1` → the cell is full
    /// and ready for the consumer at `pos`.
    sequence: AtomicUsize,
    /// Payload storage, initialised exactly while the cell is "full".
    value: UnsafeCell<MaybeUninit<T>>,
}

struct Inner<T> {
    /// Power-of-two number of cells.
    buffer: Box<[Cell<T>]>,
    /// `capacity - 1`; maps a monotonic cursor to a physical cell via `& mask`.
    mask: usize,
    /// Next ticket a producer will claim.
    enqueue_pos: CachePadded<AtomicUsize>,
    /// Next ticket a consumer will claim.
    dequeue_pos: CachePadded<AtomicUsize>,
}

// SAFETY: concurrent access to each cell's payload is gated by its `sequence`
// handshake (a producer only writes after observing `sequence == pos`, a
// consumer only reads after observing `sequence == pos + 1`), with
// acquire/release ordering carrying the data. The structure is therefore safe
// to share across threads whenever the payload may cross threads (`T: Send`).
#[expect(
    unsafe_code,
    reason = "per-cell sequence handshake serialises access to each payload"
)]
// SAFETY: per-cell sequence handshake serialises access to each payload
unsafe impl<T: Send> Send for Inner<T> {}
// SAFETY: see the `Send` impl.
#[expect(
    unsafe_code,
    reason = "per-cell sequence handshake serialises access to each payload"
)]
// SAFETY: per-cell sequence handshake serialises access to each payload
unsafe impl<T: Send> Sync for Inner<T> {}

/// A bounded lock-free `MPMC` queue handle. Clone it to share the same queue
/// with more producer/consumer threads.
pub struct MpmcQueue<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Clone for MpmcQueue<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> MpmcQueue<T> {
    /// Creates a queue whose capacity is `capacity` rounded up to the next
    /// power of two (and at least 2).
    ///
    /// The power-of-two capacity lets the cursor→cell mapping be a mask instead
    /// of a modulo, and keeps the sequence-number arithmetic monotone.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(2).next_power_of_two();
        let mut buffer = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffer.push(Cell {
                // Cell `i` is initially empty and ready for the producer whose
                // ticket is `i`.
                sequence: AtomicUsize::new(i),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }
        Self {
            inner: Arc::new(Inner {
                buffer: buffer.into_boxed_slice(),
                mask: capacity - 1,
                enqueue_pos: CachePadded::new(AtomicUsize::new(0)),
                dequeue_pos: CachePadded::new(AtomicUsize::new(0)),
            }),
        }
    }

    /// The fixed capacity (a power of two).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.mask + 1
    }

    /// Attempts to enqueue `value`. Returns `Err(value)` if the queue is full.
    pub fn push(&self, value: T) -> Result<(), T> {
        let inner = &*self.inner;
        let mut pos = inner.enqueue_pos.load(Ordering::Relaxed);
        loop {
            let cell = &inner.buffer[pos & inner.mask];
            let seq = cell.sequence.load(Ordering::Acquire);
            // Signed difference between the cell's handshake and our ticket.
            let diff = seq.wrapping_sub(pos) as isize;
            if diff == 0 {
                // Cell is empty and expects exactly our ticket: try to claim it.
                match inner.enqueue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let slot = cell.value.get();
                        #[expect(
                            unsafe_code,
                            reason = "winning the CAS grants exclusive ownership of this cell slot"
                        )]
                        // SAFETY: winning the `compare_exchange` makes us the
                        // unique owner of this cell until we publish; the cell
                        // was empty (`seq == pos`), so writing initialises it.
                        unsafe {
                            (*slot).write(value);
                        }
                        // Publish: a consumer at ticket `pos` now sees
                        // `sequence == pos + 1`.
                        cell.sequence.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // The cell is still full from a previous lap: the queue is full.
                return Err(value);
            } else {
                // Another producer advanced the cursor; reload and retry.
                pos = inner.enqueue_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Attempts to dequeue the oldest element. Returns `None` if the queue is
    /// empty.
    pub fn pop(&self) -> Option<T> {
        let inner = &*self.inner;
        let mut pos = inner.dequeue_pos.load(Ordering::Relaxed);
        loop {
            let cell = &inner.buffer[pos & inner.mask];
            let seq = cell.sequence.load(Ordering::Acquire);
            // A full cell ready for consumer ticket `pos` has `sequence == pos + 1`.
            let diff = seq.wrapping_sub(pos.wrapping_add(1)) as isize;
            if diff == 0 {
                match inner.dequeue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let slot = cell.value.get();
                        #[expect(
                            unsafe_code,
                            reason = "winning the CAS grants exclusive ownership of this cell slot"
                        )]
                        // SAFETY: winning the `compare_exchange` makes us the
                        // unique reader of this cell; its `sequence == pos + 1`
                        // means the producer already initialised and published
                        // the payload, so reading it out is sound and happens
                        // exactly once.
                        let value = unsafe { (*slot).assume_init_read() };
                        // Mark the cell empty for the producer one lap ahead
                        // (`pos + mask + 1 == pos + capacity`).
                        cell.sequence.store(
                            pos.wrapping_add(inner.mask).wrapping_add(1),
                            Ordering::Release,
                        );
                        return Some(value);
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // Cell not yet published for this ticket: the queue is empty.
                return None;
            } else {
                pos = inner.dequeue_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Returns `true` if the queue currently appears empty. Momentary in a
    /// concurrent setting.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let inner = &*self.inner;
        let enq = inner.enqueue_pos.load(Ordering::Acquire);
        let deq = inner.dequeue_pos.load(Ordering::Acquire);
        enq == deq
    }

    /// A momentary estimate of the number of queued elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        let inner = &*self.inner;
        let enq = inner.enqueue_pos.load(Ordering::Acquire);
        let deq = inner.dequeue_pos.load(Ordering::Acquire);
        enq.saturating_sub(deq)
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Drain any elements left between the two cursors, dropping their
        // payloads. With no live handles remaining, `&mut self` gives us
        // exclusive access.
        let mut pos = *self.dequeue_pos.get_mut();
        let enq = *self.enqueue_pos.get_mut();
        while pos != enq {
            let cell = &self.buffer[pos & self.mask];
            let slot = cell.value.get();
            #[expect(
                unsafe_code,
                reason = "cells in dequeue_pos..enqueue_pos hold initialised payloads"
            )]
            // SAFETY: every ticket in `dequeue_pos..enqueue_pos` names a cell a
            // producer published and no consumer took, so it holds an
            // initialised `T` we now own exclusively.
            unsafe {
                (*slot).assume_init_drop();
            }
            pos = pos.wrapping_add(1);
        }
    }
}

impl<T> fmt::Debug for MpmcQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpmcQueue")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .finish()
    }
}
