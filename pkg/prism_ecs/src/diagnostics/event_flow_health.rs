//! Double-buffered event-queue flow and reader-backpressure census
//! (design §16.7 / §16.6).
//!
//! The event system (design §16.7) keeps each [`Events<E>`] queue in two
//! internal buffers and rotates them once per frame with
//! [`update`](Events::update): an event is readable the frame it is sent and
//! the frame after, then retired on the second rotation. Readers are detached
//! [`EventCursor`]s that each remember how far they have consumed. This gives a
//! full frame of grace regardless of system ordering — but only if every reader
//! actually drains the queue within that window. A cursor that falls two
//! rotations behind silently loses the events that were retired before it read
//! them, and nothing in the queue itself reports that a reader is running hot.
//!
//! This report resolves a queue together with the frame's live reader cursors
//! once and makes that backpressure legible, read-only, without consuming any
//! event or advancing any cursor:
//!
//! * **queue occupancy** — how many events are currently buffered across both
//!   halves versus the lifetime total ever sent, so a queue that is filling
//!   faster than it drains is visible;
//! * **reader lag** — per reader, how many buffered events it has yet to read,
//!   whether it is fully caught up, and whether it is *saturated* (its unread
//!   count equals the entire live buffer, i.e. it is pinned at or behind the
//!   oldest event and will drop the older half on the next
//!   [`update`](Events::update) unless it reads first);
//! * a rolled-up **backpressure summary** — caught-up / lagging / saturated
//!   reader counts, the slowest reader's lag as a permille of the live buffer
//!   (*backlog pressure*), and the mean fill across readers.
//!
//! Reader entries are reported in the caller's input order (design §14): a
//! cursor carries no identity of its own, so the census assigns each a stable
//! zero-based index matching the `cursors` slice. The census is `O(r)` over the
//! `r` reader cursors; each [`EventCursor::len`] lookup is itself `O(1)`.

use alloc::vec::Vec;

use crate::event::{Event, EventCursor, Events};

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One reader cursor's standing against the audited [`Events`] queue: how many
/// buffered events it has yet to consume, and the derived liveness flags
/// (design §16.7).
#[derive(Clone, Copy, Debug)]
pub struct EventReaderEntry {
    /// Zero-based index of this cursor in the audited `cursors` slice.
    pub reader_index: usize,
    /// Buffered events this cursor would still yield (its
    /// [`EventCursor::len`]).
    pub unread: usize,
    /// Whether the cursor has drained everything currently buffered.
    pub caught_up: bool,
    /// Whether the cursor's unread count equals the entire live buffer, i.e. it
    /// sits at or behind the oldest event and risks dropping the older half on
    /// the next [`update`](Events::update). Never set for an empty queue.
    pub saturated: bool,
}

/// Read-only flow / backpressure census of an [`Events<E>`] queue resolved
/// against the frame's live reader cursors: queue occupancy, per-reader lag,
/// and a rolled-up backpressure summary (design §16.7 / §16.6).
#[derive(Clone, Debug)]
pub struct EventFlowHealth {
    buffered: usize,
    lifetime_total: usize,
    total_unread: usize,
    caught_up_reader_count: usize,
    saturated_reader_count: usize,
    max_reader_lag: usize,
    min_reader_lag: usize,
    readers: Vec<EventReaderEntry>,
}

impl EventFlowHealth {
    /// Censuses an [`Events<E>`] queue against the frame's reader `cursors`:
    /// records queue occupancy and each cursor's unread backlog, then rolls up
    /// the backpressure summary. Read-only; neither the queue nor any cursor is
    /// mutated. `O(r)` over the reader cursors.
    pub fn from_cursors<E: Event>(events: &Events<E>, cursors: &[EventCursor<E>]) -> Self {
        let buffered = events.len();
        let lifetime_total = events.event_count();

        let mut readers: Vec<EventReaderEntry> = Vec::with_capacity(cursors.len());
        let mut total_unread = 0usize;
        let mut caught_up_reader_count = 0usize;
        let mut saturated_reader_count = 0usize;
        let mut max_reader_lag = 0usize;
        let mut min_reader_lag = usize::MAX;

        for (reader_index, cursor) in cursors.iter().enumerate() {
            let unread = cursor.len(events);
            let caught_up = unread == 0;
            let saturated = buffered > 0 && unread == buffered;

            total_unread += unread;
            if caught_up {
                caught_up_reader_count += 1;
            }
            if saturated {
                saturated_reader_count += 1;
            }
            if unread > max_reader_lag {
                max_reader_lag = unread;
            }
            if unread < min_reader_lag {
                min_reader_lag = unread;
            }

            readers.push(EventReaderEntry {
                reader_index,
                unread,
                caught_up,
                saturated,
            });
        }

        if readers.is_empty() {
            min_reader_lag = 0;
        }

        Self {
            buffered,
            lifetime_total,
            total_unread,
            caught_up_reader_count,
            saturated_reader_count,
            max_reader_lag,
            min_reader_lag,
            readers,
        }
    }

    /// Events currently buffered across both internal halves (equals
    /// [`Events::len`]).
    #[inline]
    pub fn buffered(&self) -> usize {
        self.buffered
    }

    /// Whether the queue currently holds no buffered events.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffered == 0
    }

    /// Total events ever sent into the queue over its lifetime (equals
    /// [`Events::event_count`]).
    #[inline]
    pub fn lifetime_total(&self) -> usize {
        self.lifetime_total
    }

    /// Number of reader cursors audited.
    #[inline]
    pub fn reader_count(&self) -> usize {
        self.readers.len()
    }

    /// Whether any reader cursor was audited.
    #[inline]
    pub fn has_readers(&self) -> bool {
        !self.readers.is_empty()
    }

    /// The per-reader lag breakdown, in the caller's input order.
    #[inline]
    pub fn readers(&self) -> &[EventReaderEntry] {
        &self.readers
    }

    /// Sum of every reader's unread backlog (double-counts shared buffered
    /// events across readers).
    #[inline]
    pub fn total_unread(&self) -> usize {
        self.total_unread
    }

    /// Readers that have drained everything currently buffered.
    #[inline]
    pub fn caught_up_reader_count(&self) -> usize {
        self.caught_up_reader_count
    }

    /// Readers with at least one unread buffered event.
    #[inline]
    pub fn lagging_reader_count(&self) -> usize {
        self.readers.len() - self.caught_up_reader_count
    }

    /// Whether every audited reader is fully caught up (vacuously `true` when
    /// no readers were audited).
    #[inline]
    pub fn all_readers_caught_up(&self) -> bool {
        self.caught_up_reader_count == self.readers.len()
    }

    /// Whether any audited reader has unread events.
    #[inline]
    pub fn has_lagging_readers(&self) -> bool {
        self.caught_up_reader_count < self.readers.len()
    }

    /// Readers whose unread count equals the entire live buffer — pinned at or
    /// behind the oldest event and at risk of dropping the older half on the
    /// next [`update`](Events::update).
    #[inline]
    pub fn saturated_reader_count(&self) -> usize {
        self.saturated_reader_count
    }

    /// Whether any reader is saturated (see
    /// [`saturated_reader_count`](Self::saturated_reader_count)).
    #[inline]
    pub fn has_saturated_readers(&self) -> bool {
        self.saturated_reader_count > 0
    }

    /// The slowest reader's unread backlog (`0` when no readers were audited).
    #[inline]
    pub fn max_reader_lag(&self) -> usize {
        self.max_reader_lag
    }

    /// The most up-to-date reader's unread backlog (`0` when no readers were
    /// audited).
    #[inline]
    pub fn min_reader_lag(&self) -> usize {
        self.min_reader_lag
    }

    /// Backlog pressure in permille: the slowest reader's lag as a fraction of
    /// the live buffer (`max_reader_lag × 1000 / buffered`). `1000` means a
    /// reader is pinned to the full buffer; `0` when the queue is empty.
    pub fn backlog_permille(&self) -> u64 {
        permille(self.max_reader_lag as u64, self.buffered as u64)
    }

    /// Mean reader fill in permille: the average unread backlog across readers
    /// as a fraction of the live buffer
    /// (`total_unread × 1000 / (reader_count × buffered)`). `0` when there are
    /// no readers or the queue is empty.
    pub fn mean_lag_permille(&self) -> u64 {
        let den = (self.readers.len() as u64) * (self.buffered as u64);
        permille(self.total_unread as u64, den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;

    struct Tick;
    impl Event for Tick {}

    #[test]
    fn empty_queue_without_readers_is_quiescent() {
        let events: Events<Tick> = Events::new();
        let health = EventFlowHealth::from_cursors(&events, &[]);
        assert!(health.is_empty());
        assert_eq!(health.buffered(), 0);
        assert_eq!(health.lifetime_total(), 0);
        assert_eq!(health.reader_count(), 0);
        assert!(!health.has_readers());
        assert_eq!(health.total_unread(), 0);
        assert_eq!(health.max_reader_lag(), 0);
        assert_eq!(health.min_reader_lag(), 0);
        assert!(health.all_readers_caught_up());
        assert!(!health.has_lagging_readers());
        assert!(!health.has_saturated_readers());
        assert_eq!(health.backlog_permille(), 0);
        assert_eq!(health.mean_lag_permille(), 0);
        assert!(health.readers().is_empty());
    }

    #[test]
    fn fresh_cursor_sees_full_backlog_and_is_saturated() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        events.send(Tick);
        let cursors = [EventCursor::default()];
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        assert_eq!(health.buffered(), 3);
        assert_eq!(health.lifetime_total(), 3);
        assert_eq!(health.reader_count(), 1);
        assert_eq!(health.readers()[0].unread, 3);
        assert!(health.readers()[0].saturated);
        assert!(!health.readers()[0].caught_up);
        assert_eq!(health.lagging_reader_count(), 1);
        assert_eq!(health.caught_up_reader_count(), 0);
        assert_eq!(health.saturated_reader_count(), 1);
        assert_eq!(health.max_reader_lag(), 3);
        assert_eq!(health.min_reader_lag(), 3);
        assert_eq!(health.backlog_permille(), 1000);
        assert_eq!(health.mean_lag_permille(), 1000);
    }

    #[test]
    fn end_positioned_cursor_has_no_lag() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        // A cursor obtained at the end only sees future events.
        let cursors = [events.get_cursor()];
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        assert_eq!(health.buffered(), 2);
        assert!(health.readers()[0].caught_up);
        assert!(!health.readers()[0].saturated);
        assert_eq!(health.caught_up_reader_count(), 1);
        assert_eq!(health.saturated_reader_count(), 0);
        assert!(health.all_readers_caught_up());
        assert_eq!(health.max_reader_lag(), 0);
        assert_eq!(health.backlog_permille(), 0);
        assert_eq!(health.mean_lag_permille(), 0);
    }

    #[test]
    fn partial_lag_is_below_saturation() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        // Snapshot the cursor at count 2, then send three more.
        let cursors = [events.get_cursor()];
        events.send(Tick);
        events.send(Tick);
        events.send(Tick);
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        assert_eq!(health.buffered(), 5);
        assert_eq!(health.readers()[0].unread, 3);
        assert!(!health.readers()[0].saturated);
        assert!(!health.readers()[0].caught_up);
        assert_eq!(health.lagging_reader_count(), 1);
        assert_eq!(health.saturated_reader_count(), 0);
        assert_eq!(health.backlog_permille(), 600);
    }

    #[test]
    fn multiple_readers_roll_up_mixed_lag() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        events.send(Tick);
        events.send(Tick);
        // One fresh reader (saturated), one caught up at the end.
        let cursors = [EventCursor::default(), events.get_cursor()];
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        assert_eq!(health.reader_count(), 2);
        assert_eq!(health.caught_up_reader_count(), 1);
        assert_eq!(health.lagging_reader_count(), 1);
        assert_eq!(health.saturated_reader_count(), 1);
        assert_eq!(health.max_reader_lag(), 4);
        assert_eq!(health.min_reader_lag(), 0);
        assert_eq!(health.total_unread(), 4);
        assert!(!health.all_readers_caught_up());
        // Average fill across the two readers: 4 of 2*4 buffered slots.
        assert_eq!(health.mean_lag_permille(), 500);
    }

    #[test]
    fn rotation_retires_old_events_but_keeps_lifetime_total() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        events.update(); // a=[2 events], b=[]
        events.send(Tick); // b=[1 event], buffered=3
        events.update(); // a=[1 event], b=[], buffered=1
        let cursors = [EventCursor::default()];
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        assert_eq!(health.buffered(), 1);
        assert_eq!(health.lifetime_total(), 3);
        // The fresh cursor sees only what survives, and that is the whole buffer.
        assert_eq!(health.readers()[0].unread, 1);
        assert!(health.readers()[0].saturated);
    }

    #[test]
    fn reader_entries_keep_input_order() {
        let mut events: Events<Tick> = Events::new();
        events.send(Tick);
        events.send(Tick);
        let caught_up = events.get_cursor(); // unread 0
        events.send(Tick); // buffered 3; caught_up now has unread 1
        let fresh = EventCursor::default(); // unread 3
        let cursors = [fresh, caught_up];
        let health = EventFlowHealth::from_cursors(&events, &cursors);
        let readers = health.readers();
        assert_eq!(readers.len(), 2);
        assert_eq!(readers[0].reader_index, 0);
        assert_eq!(readers[0].unread, 3);
        assert_eq!(readers[1].reader_index, 1);
        assert_eq!(readers[1].unread, 1);
    }

    #[test]
    fn clear_resets_buffer_but_not_lifetime_total() {
        let mut events: Events<Tick> = Events::new();
        for _ in 0..5 {
            events.send(Tick);
        }
        events.clear();
        events.send(Tick);
        events.send(Tick);
        let health = EventFlowHealth::from_cursors(&events, &[]);
        assert_eq!(health.buffered(), 2);
        assert_eq!(health.lifetime_total(), 7);
    }
}
