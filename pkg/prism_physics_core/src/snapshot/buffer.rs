//! A three-slot transport buffer for handing snapshots to the renderer.
//!
//! The fixed-step pipeline is a single-producer, single-consumer problem: the
//! physics step *writes* a fresh [`super::StateSnapshot`] after every tick, and
//! the render frame *reads* the two most recent snapshots to interpolate
//! between them. A [`TripleBuffer`] keeps exactly three slots so the producer
//! always has one free slot to write into without blocking or overwriting
//! either of the two snapshots the consumer needs.
//!
//! The published index is stored in an [`AtomicUsize`] so the publish/read
//! handshake is data-race-free by construction; wrapping the buffer in a shared
//! lock-free handle for a dedicated physics thread is a Layer-4 concern left to
//! the integration layer. No `unsafe` is used here.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! triple-buffer pattern is a standard, publicly documented concurrency
//! primitive implemented from scratch.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A three-slot rotating buffer that always exposes the two most recently
/// published values while keeping a third slot free for the next write.
///
/// The buffer starts with no publications; [`TripleBuffer::current`] and
/// [`TripleBuffer::previous`] return `None` until at least one (respectively
/// two) values have been published.
#[derive(Debug)]
pub struct TripleBuffer<T> {
    slots: [T; 3],
    /// Slot index most recently published (the "current" value).
    current: AtomicUsize,
    /// Slot index published just before `current` (the "previous" value).
    previous: AtomicUsize,
    /// Slot index currently free for the producer to write into.
    write: usize,
    /// Number of publications performed so far (saturating at 2 for state
    /// queries; used to report whether a full interpolation pair exists).
    published: usize,
}

impl<T: Clone> TripleBuffer<T> {
    /// Creates a buffer whose three slots are initialized to clones of `seed`.
    ///
    /// No slot is considered published yet: the seed values are placeholders
    /// the producer overwrites before publishing.
    #[must_use]
    pub fn new(seed: T) -> TripleBuffer<T> {
        TripleBuffer {
            slots: [seed.clone(), seed.clone(), seed],
            current: AtomicUsize::new(0),
            previous: AtomicUsize::new(0),
            write: 1,
            published: 0,
        }
    }

    /// Returns a mutable reference to the free slot the producer writes into
    /// before calling [`TripleBuffer::publish`].
    pub fn write_slot(&mut self) -> &mut T {
        &mut self.slots[self.write]
    }

    /// Publishes the value in the current write slot, rotating the free slot.
    ///
    /// After publishing, the just-written slot becomes "current", the former
    /// "current" becomes "previous", and the former "previous" slot is handed
    /// back to the producer as the next free write slot. This rotation is what
    /// keeps all three roles on distinct slots.
    pub fn publish(&mut self) {
        let new_current = self.write;
        let old_current = self.current.load(Ordering::Relaxed);
        // The next free slot is whichever of the three is neither the new
        // current nor the new previous (== old current).
        let next_write = 3 - new_current - old_current;
        self.previous.store(old_current, Ordering::Relaxed);
        self.current.store(new_current, Ordering::Release);
        self.write = next_write;
        self.published = (self.published + 1).min(2);
    }

    /// Returns the most recently published value, or `None` if nothing has been
    /// published yet.
    #[must_use]
    pub fn current(&self) -> Option<&T> {
        (self.published >= 1).then(|| &self.slots[self.current.load(Ordering::Acquire)])
    }

    /// Returns the value published just before [`TripleBuffer::current`], or
    /// `None` until at least two values have been published.
    #[must_use]
    pub fn previous(&self) -> Option<&T> {
        (self.published >= 2).then(|| &self.slots[self.previous.load(Ordering::Acquire)])
    }

    /// Returns the `(previous, current)` pair the renderer interpolates between,
    /// or `None` until a full pair has been published.
    #[must_use]
    pub fn pair(&self) -> Option<(&T, &T)> {
        match (self.previous(), self.current()) {
            (Some(prev), Some(curr)) => Some((prev, curr)),
            _ => None,
        }
    }

    /// Returns how many publications have occurred, saturating at `2`.
    #[must_use]
    pub fn published_count(&self) -> usize {
        self.published
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_has_no_published_values() {
        let buf = TripleBuffer::new(0_u32);
        assert!(buf.current().is_none());
        assert!(buf.previous().is_none());
        assert!(buf.pair().is_none());
        assert_eq!(buf.published_count(), 0);
    }

    #[test]
    fn first_publish_exposes_current_only() {
        let mut buf = TripleBuffer::new(0_u32);
        *buf.write_slot() = 10;
        buf.publish();
        assert_eq!(buf.current(), Some(&10));
        assert!(buf.previous().is_none());
        assert!(buf.pair().is_none());
    }

    #[test]
    fn second_publish_exposes_full_pair() {
        let mut buf = TripleBuffer::new(0_u32);
        *buf.write_slot() = 10;
        buf.publish();
        *buf.write_slot() = 20;
        buf.publish();
        assert_eq!(buf.pair(), Some((&10, &20)));
    }

    #[test]
    fn roles_stay_on_distinct_slots_across_many_publishes() {
        let mut buf = TripleBuffer::new(0_i32);
        // Publish a long sequence and check the pair always tracks the last two.
        for v in 1..=50_i32 {
            *buf.write_slot() = v;
            buf.publish();
            if v >= 2 {
                assert_eq!(buf.pair(), Some((&(v - 1), &v)));
            }
        }
    }
}
