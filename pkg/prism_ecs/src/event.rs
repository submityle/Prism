//! Double-buffered, cross-frame events (design §16.7).
//!
//! This module implements the "double-buffered `EventReader`/`EventWriter`"
//! half of the event system described in design §16.7 (the observer-style,
//! same-frame immediate events are a separate, later subsystem). It is a
//! from-scratch reimagining of Bevy's `Events<E>` design, implemented here with
//! no `bevy_*` dependency.
//!
//! # Why double buffering
//!
//! A naive event queue that is cleared once per frame has a fatal ordering
//! hazard: a reader that runs *before* the writer this frame would miss the
//! event entirely, and a reader that runs *after* the clear would never see it.
//! The classic fix is to keep **two** buffers and rotate them once per frame:
//!
//! - [`Events::send`] always pushes into the current *write* buffer.
//! - [`Events::update`] swaps the two buffers, then clears the one that just
//!   became the write buffer.
//!
//! Because an event is only dropped on the *second* [`Events::update`] after it
//! was sent, every event is guaranteed to be readable for a full frame of grace
//! regardless of system ordering: it is visible the frame it is sent **and** the
//! frame after. On the second `update()` it is retired.
//!
//! # Reading without missing or duplicating
//!
//! Each buffer records the global [`Events::event_count`] value at the moment it
//! became the write buffer (`start_event_count`). Every stored event carries a
//! monotonically increasing [`EventId`]. A reader is just a cursor
//! ([`EventCursor`]) holding the id of the next event it has yet to observe
//! (`last_event_count`). To read, the cursor turns that absolute id into a slice
//! offset within each buffer, yields the remaining events oldest-first across
//! both buffers, and advances its cursor to the current count. This makes it
//! impossible to read the same event twice or to skip an event across a buffer
//! swap, which the module's tests exercise directly.

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::resource::Resource;

/// Marker trait for a type that can be sent as an event.
///
/// Events must be thread-shareable and own all of their data (`'static`) so the
/// scheduler can move them between frames and (in later milestones) between
/// worker threads.
pub trait Event: Send + Sync + 'static {}

/// A globally unique, monotonically increasing identifier for a single sent
/// event instance.
///
/// Ids are assigned in send order and never reused within an [`Events`]
/// buffer's lifetime, so ordering comparisons are meaningful: a lower id was
/// sent earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(
    /// The raw sequence number of this event.
    pub usize,
);

/// One stored event paired with the [`EventId`] assigned when it was sent.
struct EventInstance<E: Event> {
    id: EventId,
    event: E,
}

/// One of the two internal ring buffers.
///
/// `start_event_count` is the value of [`Events::event_count`] at the instant
/// this buffer became the write buffer, i.e. the absolute id of its first
/// element. Together with each element's position this lets a reader translate
/// an absolute [`EventId`] into an index without scanning.
struct EventSequence<E: Event> {
    events: Vec<EventInstance<E>>,
    start_event_count: usize,
}

impl<E: Event> Default for EventSequence<E> {
    #[inline]
    fn default() -> Self {
        Self {
            events: Vec::new(),
            start_event_count: 0,
        }
    }
}

/// A double-buffered queue of events of type `E` (design §16.7).
///
/// Send events with [`send`](Events::send) / [`send_batch`](Events::send_batch),
/// read them through an [`EventCursor`] obtained from
/// [`get_cursor`](Events::get_cursor) (or a freshly
/// [`default`](EventCursor::default)ed one), and rotate the buffers once per
/// frame with [`update`](Events::update).
///
/// An event sent during a frame is readable during that frame and the next;
/// the second [`update`](Events::update) after it was sent retires it.
pub struct Events<E: Event> {
    /// The older buffer: events sent before the most recent `update()`.
    events_a: EventSequence<E>,
    /// The newer buffer: the current write target.
    events_b: EventSequence<E>,
    /// The number of events ever sent; also the id to assign the next event.
    event_count: usize,
}

impl<E: Event> Default for Events<E> {
    #[inline]
    fn default() -> Self {
        Self {
            events_a: EventSequence::default(),
            events_b: EventSequence::default(),
            event_count: 0,
        }
    }
}

impl<E: Event> Events<E> {
    /// Create an empty event queue.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `event`, returning the [`EventId`] it was assigned.
    ///
    /// The event is pushed into the current write buffer and becomes readable
    /// immediately by any cursor that has not yet advanced past it.
    #[inline]
    pub fn send(&mut self, event: E) -> EventId {
        let id = EventId(self.event_count);
        self.events_b.events.push(EventInstance { id, event });
        self.event_count += 1;
        id
    }

    /// Send every event yielded by `iter`, in order.
    #[inline]
    pub fn send_batch(&mut self, iter: impl IntoIterator<Item = E>) {
        for event in iter {
            self.send(event);
        }
    }

    /// Send `E::default()`, returning its [`EventId`].
    #[inline]
    pub fn send_default(&mut self) -> EventId
    where
        E: Default,
    {
        self.send(E::default())
    }

    /// Rotate the double buffer.
    ///
    /// Swaps the two internal buffers, then clears the one that just became the
    /// write buffer and stamps it with the current event count. Events live for
    /// exactly two `update()` calls after being sent, giving one full frame of
    /// grace for readers regardless of system ordering.
    #[inline]
    pub fn update(&mut self) {
        core::mem::swap(&mut self.events_a, &mut self.events_b);
        self.events_b.events.clear();
        self.events_b.start_event_count = self.event_count;
    }

    /// Remove and yield every currently-buffered event, oldest first across both
    /// buffers, leaving the queue empty.
    ///
    /// The global event counter is preserved, so [`EventId`]s assigned after a
    /// drain continue to increase monotonically.
    pub fn drain(&mut self) -> impl Iterator<Item = E> + '_ {
        self.reset_start_event_count();
        self.events_a
            .events
            .drain(..)
            .chain(self.events_b.events.drain(..))
            .map(|instance| instance.event)
    }

    /// Discard all buffered events without yielding them.
    #[inline]
    pub fn clear(&mut self) {
        self.reset_start_event_count();
        self.events_a.events.clear();
        self.events_b.events.clear();
    }

    /// The number of events currently buffered across both buffers.
    #[inline]
    pub fn len(&self) -> usize {
        self.events_a.events.len() + self.events_b.events.len()
    }

    /// Whether no events are currently buffered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of events ever sent into this queue (the id that would be
    /// assigned to the next event).
    #[inline]
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Obtain a cursor positioned at the current end of the stream.
    ///
    /// The returned cursor reads only events sent *after* this call. To instead
    /// read everything currently buffered, use [`EventCursor::default`], whose
    /// `last_event_count` of `0` starts at the beginning of what is buffered.
    #[inline]
    pub fn get_cursor(&self) -> EventCursor<E> {
        EventCursor {
            last_event_count: self.event_count,
            _marker: PhantomData,
        }
    }

    /// Reset both buffers' start markers to the current count. Used by
    /// [`drain`](Events::drain) and [`clear`](Events::clear) so that a cursor
    /// which had not caught up does not later resurrect flushed ids.
    #[inline]
    fn reset_start_event_count(&mut self) {
        self.events_a.start_event_count = self.event_count;
        self.events_b.start_event_count = self.event_count;
    }
}

/// [`Events<E>`] is stored in a [`World`](crate::world::World) as a resource so
/// that systems can send and read events through [`Res`](crate::system::Res) /
/// [`ResMut`](crate::system::ResMut). The per-frame buffer rotation is driven by
/// a system calling [`Events::update`] (wired up by the application shell).
impl<E: Event> Resource for Events<E> {}

/// A manual reader cursor into an [`Events`] queue.
///
/// A cursor remembers the id of the next event it has yet to observe. Reading
/// through [`read`](EventCursor::read) yields every event with id at least that
/// value, oldest-first across both internal buffers, then advances the cursor so
/// those events are never yielded again — including across a buffer swap.
///
/// A [`default`](EventCursor::default)-constructed cursor starts at `0`, i.e. at
/// the beginning of whatever is currently buffered; use
/// [`Events::get_cursor`] for a cursor that only sees future events.
pub struct EventCursor<E: Event> {
    last_event_count: usize,
    _marker: PhantomData<E>,
}

impl<E: Event> Default for EventCursor<E> {
    #[inline]
    fn default() -> Self {
        Self {
            last_event_count: 0,
            _marker: PhantomData,
        }
    }
}

impl<E: Event> EventCursor<E> {
    /// Create a cursor at the beginning of the buffered stream.
    ///
    /// Equivalent to [`EventCursor::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read all events not yet seen, oldest-first across both buffers, advancing
    /// the cursor so they are never yielded again.
    #[inline]
    pub fn read<'a>(&mut self, events: &'a Events<E>) -> impl Iterator<Item = &'a E> {
        self.read_with_id(events).map(|(event, _id)| event)
    }

    /// Like [`read`](EventCursor::read) but also yields each event's
    /// [`EventId`].
    pub fn read_with_id<'a>(
        &mut self,
        events: &'a Events<E>,
    ) -> impl Iterator<Item = (&'a E, EventId)> {
        let last = self.last_event_count;

        // Translate the absolute cursor id into a slice offset within each
        // buffer. `saturating_sub` clamps the offset to 0 when the cursor is
        // older than a buffer's first id (events it missed are simply read from
        // the start of what survives).
        let a_index = last.saturating_sub(events.events_a.start_event_count);
        let b_index = last.saturating_sub(events.events_b.start_event_count);

        let a = events.events_a.events.get(a_index..).unwrap_or_default();
        let b = events.events_b.events.get(b_index..).unwrap_or_default();

        // We yield everything from here to the end, so advance eagerly.
        self.last_event_count = events.event_count;

        a.iter()
            .chain(b.iter())
            .map(|instance| (&instance.event, instance.id))
    }

    /// The number of unread events this cursor would yield from `events`.
    #[inline]
    pub fn len(&self, events: &Events<E>) -> usize {
        events
            .event_count
            .saturating_sub(self.last_event_count)
            .min(events.len())
    }

    /// Whether this cursor has no unread events in `events`.
    #[inline]
    pub fn is_empty(&self, events: &Events<E>) -> bool {
        self.len(events) == 0
    }

    /// Advance the cursor to the current end of `events` without reading, so
    /// that subsequent reads only see future events.
    #[inline]
    pub fn clear(&mut self, events: &Events<E>) {
        self.last_event_count = events.event_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct TestEvent(u32);
    impl Event for TestEvent {}

    #[test]
    fn send_and_read_basic() {
        let mut events = Events::<TestEvent>::new();
        let mut cursor = events.get_cursor();

        assert!(cursor.read(&events).next().is_none());

        events.send(TestEvent(1));
        events.send(TestEvent(2));

        let read: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(read, vec![TestEvent(1), TestEvent(2)]);

        // Reading again yields nothing: the cursor advanced.
        assert!(cursor.read(&events).next().is_none());
    }

    #[test]
    fn event_ids_are_monotonic() {
        let mut events = Events::<TestEvent>::new();
        let id0 = events.send(TestEvent(10));
        let id1 = events.send(TestEvent(11));
        assert_eq!(id0, EventId(0));
        assert_eq!(id1, EventId(1));
        assert!(id0 < id1);

        let mut cursor = EventCursor::<TestEvent>::default();
        let with_ids: Vec<_> = cursor
            .read_with_id(&events)
            .map(|(e, id)| (e.clone(), id))
            .collect();
        assert_eq!(
            with_ids,
            vec![(TestEvent(10), EventId(0)), (TestEvent(11), EventId(1))]
        );
    }

    #[test]
    fn event_survives_one_update_gone_after_two() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent(42));
        assert_eq!(events.len(), 1);

        // One full frame of grace: still present after a single update.
        events.update();
        assert_eq!(events.len(), 1);

        // A fresh default cursor can still observe it after one update.
        let mut cursor = EventCursor::<TestEvent>::default();
        let read: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(read, vec![TestEvent(42)]);

        // Retired on the second update.
        events.update();
        assert_eq!(events.len(), 0);
        let mut cursor = EventCursor::<TestEvent>::default();
        assert!(cursor.read(&events).next().is_none());
    }

    #[test]
    fn cursor_across_swap_no_dup_no_skip() {
        let mut events = Events::<TestEvent>::new();
        let mut cursor = events.get_cursor();

        events.send(TestEvent(1));
        let r: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(r, vec![TestEvent(1)]);

        events.update();
        events.send(TestEvent(2));
        // Must see only the new event, never re-read event 1 across the swap.
        let r: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(r, vec![TestEvent(2)]);

        events.update();
        // Nothing new; event 2 is in the older buffer but already read.
        assert!(cursor.read(&events).next().is_none());

        // And a brand-new reader at this point still sees the surviving event 2.
        let mut fresh = EventCursor::<TestEvent>::default();
        let r: Vec<_> = fresh.read(&events).cloned().collect();
        assert_eq!(r, vec![TestEvent(2)]);
    }

    #[test]
    fn interleaved_send_update_read() {
        let mut events = Events::<TestEvent>::new();
        let mut cursor = events.get_cursor();
        let mut seen: Vec<u32> = Vec::new();

        for frame in 0..10u32 {
            events.send(TestEvent(frame * 2));
            events.send(TestEvent(frame * 2 + 1));
            for e in cursor.read(&events) {
                seen.push(e.0);
            }
            events.update();
        }

        let expected: Vec<u32> = (0..20).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn drain_yields_oldest_first_across_buffers() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent(1));
        events.update();
        events.send(TestEvent(2));
        // event 1 is in the older buffer, event 2 in the newer buffer.
        assert_eq!(events.len(), 2);

        let drained: Vec<_> = events.drain().collect();
        assert_eq!(drained, vec![TestEvent(1), TestEvent(2)]);
        assert!(events.is_empty());

        // Ids keep increasing after a drain (two events were sent: ids 0 and 1).
        let id = events.send(TestEvent(3));
        assert_eq!(id, EventId(2));
    }

    #[test]
    fn send_batch_and_default() {
        let mut events = Events::<TestEvent>::new();
        events.send_batch([TestEvent(1), TestEvent(2), TestEvent(3)]);
        events.send_default();

        let mut cursor = EventCursor::<TestEvent>::default();
        let read: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(
            read,
            vec![TestEvent(1), TestEvent(2), TestEvent(3), TestEvent(0)]
        );
    }

    #[test]
    fn multiple_independent_cursors() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent(1));
        events.send(TestEvent(2));

        let mut a = EventCursor::<TestEvent>::default();
        let mut b = EventCursor::<TestEvent>::default();

        let ra: Vec<_> = a.read(&events).cloned().collect();
        assert_eq!(ra, vec![TestEvent(1), TestEvent(2)]);

        // `b` has not advanced and still sees both.
        assert_eq!(b.len(&events), 2);
        let rb: Vec<_> = b.read(&events).cloned().collect();
        assert_eq!(rb, vec![TestEvent(1), TestEvent(2)]);

        // Both are now caught up.
        assert!(a.is_empty(&events));
        assert!(b.is_empty(&events));
    }

    #[test]
    fn cursor_len_is_empty_and_clear() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent(1));
        events.send(TestEvent(2));

        let mut cursor = EventCursor::<TestEvent>::default();
        assert_eq!(cursor.len(&events), 2);
        assert!(!cursor.is_empty(&events));

        cursor.clear(&events);
        assert_eq!(cursor.len(&events), 0);
        assert!(cursor.is_empty(&events));
        assert!(cursor.read(&events).next().is_none());
    }

    #[test]
    fn get_cursor_reads_only_future_events() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent(1));

        // A cursor obtained now should ignore the already-buffered event.
        let mut cursor = events.get_cursor();
        assert!(cursor.read(&events).next().is_none());

        events.send(TestEvent(2));
        let read: Vec<_> = cursor.read(&events).cloned().collect();
        assert_eq!(read, vec![TestEvent(2)]);
    }

    #[test]
    fn events_clear_removes_everything() {
        let mut events = Events::<TestEvent>::new();
        events.send_batch([TestEvent(1), TestEvent(2)]);
        events.clear();
        assert!(events.is_empty());

        let mut cursor = EventCursor::<TestEvent>::default();
        assert!(cursor.read(&events).next().is_none());
    }
}
