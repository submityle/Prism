//! Deterministic command sequencing (design §9 / §13).
//!
//! Netcode needs one order every peer agrees on. A reliable-ordered transport
//! gives that for free, but the moment a channel is unreliable / unordered (or a
//! peer reconnects and replays), commands arrive out of order, twice, or with a
//! gap. This module supplies the two pure primitives the replication layer builds
//! on:
//!
//! - [`LamportClock`] — a logical clock. `tick` on send, `observe` on receive, so
//!   a total order can be derived from `(lamport, source)` without trusting wall
//!   clocks.
//! - [`DeliveryWindow`] — per-source in-order delivery over an unordered stream.
//!   It buffers a bounded window of early arrivals, hands items up strictly in
//!   sequence order, rejects duplicates and already-delivered (stale) frames, and
//!   rejects anything past the window so a lost frame forces an explicit resync
//!   instead of unbounded buffering.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A Lamport logical clock (design §9: "Lamport 定序").
///
/// `tick` on every send, `observe` on every receive: the value is a monotone
/// counter such that if `a` causally precedes `b` then `a < b`, and unrelated
/// events are ordered by `(value, source)` at the caller. Saturating arithmetic
/// keeps an absurd received stamp from wrapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LamportClock(u64);

impl LamportClock {
    /// A fresh clock at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// The current logical time.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Advance for a local event (a send) and return the stamp to attach.
    pub fn tick(&mut self) -> u64 {
        self.0 = self.0.saturating_add(1);
        self.0
    }

    /// Merge a received stamp: move to `max(local, received) + 1`, return it.
    pub fn observe(&mut self, received: u64) -> u64 {
        self.0 = self.0.max(received).saturating_add(1);
        self.0
    }
}

/// What [`DeliveryWindow::insert`] did with an offered item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delivery<T> {
    /// Accepted and buffered; a gap still separates it from the next expected
    /// sequence, so nothing is deliverable yet.
    Buffered,
    /// Accepted; carries every item that just became deliverable, in order (at
    /// least one). Filling a gap can release several buffered items at once.
    Advanced(Vec<(u64, T)>),
    /// `seq` was already delivered (it lies below the next expected sequence).
    Stale,
    /// `seq` is already buffered, awaiting its gap — the first copy wins.
    Duplicate,
    /// `seq` is at or past `next + capacity`: too far ahead to buffer. Reject it
    /// and resync (request the missing range) rather than grow without bound.
    OutOfWindow,
}

/// A per-source in-order delivery window over an unordered sequence stream.
///
/// The window expects sequence numbers starting at `0`. It buffers up to
/// `capacity - 1` items that arrive ahead of the next expected number, then
/// releases them as the gaps fill, so the caller observes a single gap-free
/// in-order stream no matter how the transport reorders or duplicates.
#[derive(Clone, Debug)]
pub struct DeliveryWindow<T> {
    next: u64,
    capacity: usize,
    buffered: BTreeMap<u64, T>,
}

impl<T> DeliveryWindow<T> {
    /// A window that buffers at most `capacity - 1` early arrivals.
    ///
    /// `capacity` is clamped to at least one (a window of one accepts only the
    /// exact next sequence).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            next: 0,
            capacity: capacity.max(1),
            buffered: BTreeMap::new(),
        }
    }

    /// The next sequence number that will be delivered.
    #[must_use]
    pub const fn next(&self) -> u64 {
        self.next
    }

    /// How many out-of-order items are currently buffered awaiting a gap.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffered.len()
    }

    /// Offer `item` tagged `seq` to the window.
    ///
    /// Returns [`Delivery::Advanced`] with the newly deliverable run when `seq`
    /// (or a filled gap) lets delivery progress, [`Delivery::Buffered`] for a
    /// valid early arrival, and one of the rejection variants otherwise. A
    /// rejected offer consumes `item`.
    pub fn insert(&mut self, seq: u64, item: T) -> Delivery<T> {
        if seq < self.next {
            return Delivery::Stale;
        }
        // `seq - next` is the distance ahead; anything at or past `capacity` is
        // past the window. Checked before buffering so a lost frame cannot grow
        // the buffer unboundedly.
        if seq - self.next >= self.capacity as u64 {
            return Delivery::OutOfWindow;
        }
        if seq != self.next && self.buffered.contains_key(&seq) {
            return Delivery::Duplicate;
        }
        self.buffered.insert(seq, item);

        if self.buffered.contains_key(&self.next) {
            let mut ready = Vec::new();
            while let Some(item) = self.buffered.remove(&self.next) {
                ready.push((self.next, item));
                self.next = self.next.saturating_add(1);
            }
            Delivery::Advanced(ready)
        } else {
            Delivery::Buffered
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_core,
        reason = "tests exercise the alloc/core-only core under std"
    )]

    use super::*;

    #[test]
    fn lamport_tick_and_observe_are_monotone() {
        let mut clock = LamportClock::new();
        assert_eq!(clock.value(), 0);
        assert_eq!(clock.tick(), 1);
        assert_eq!(clock.observe(5), 6, "observe jumps past the received stamp");
        assert_eq!(clock.observe(2), 7, "an older stamp still advances it");
        assert_eq!(clock.tick(), 8);
    }

    #[test]
    fn in_order_items_advance_one_at_a_time() {
        let mut window = DeliveryWindow::new(4);
        for (seq, item) in [(0, "a"), (1, "b"), (2, "c")] {
            assert_eq!(
                window.insert(seq, item),
                Delivery::Advanced(alloc::vec![(seq, item)])
            );
        }
        assert_eq!(window.next(), 3);
        assert_eq!(window.buffered(), 0);
    }

    #[test]
    fn a_gap_releases_every_buffered_item_in_order() {
        let mut window = DeliveryWindow::new(4);
        assert_eq!(window.insert(2, "c"), Delivery::Buffered);
        assert_eq!(window.insert(1, "b"), Delivery::Buffered);
        assert_eq!(window.buffered(), 2);

        // Frame 0 fills the gap; all three come out in order.
        assert_eq!(
            window.insert(0, "a"),
            Delivery::Advanced(alloc::vec![(0, "a"), (1, "b"), (2, "c")])
        );
        assert_eq!(window.next(), 3);
        assert_eq!(window.buffered(), 0);
    }

    #[test]
    fn duplicates_and_stale_frames_are_rejected() {
        let mut window = DeliveryWindow::new(4);
        assert_eq!(window.insert(1, "b"), Delivery::Buffered);
        assert_eq!(window.insert(1, "b-again"), Delivery::Duplicate);

        assert_eq!(
            window.insert(0, "a"),
            Delivery::Advanced(alloc::vec![(0, "a"), (1, "b")])
        );
        assert_eq!(window.insert(0, "a-again"), Delivery::Stale);
        assert_eq!(window.insert(1, "b-again"), Delivery::Stale);
        assert_eq!(window.buffered(), 0);
    }

    #[test]
    fn far_ahead_frames_are_out_of_window() {
        let mut window = DeliveryWindow::new(3);
        assert_eq!(window.insert(2, "c"), Delivery::Buffered, "depth 2 accepted");
        assert_eq!(window.insert(3, "d"), Delivery::OutOfWindow, "depth 3 past");
        assert_eq!(window.buffered(), 1, "the rejected item is not buffered");

        // A capacity of one (from zero) accepts only the exact next sequence.
        let mut one = DeliveryWindow::new(0);
        assert_eq!(one.insert(1, "b"), Delivery::OutOfWindow);
        assert_eq!(
            one.insert(0, "a"),
            Delivery::Advanced(alloc::vec![(0, "a")])
        );
    }
}
