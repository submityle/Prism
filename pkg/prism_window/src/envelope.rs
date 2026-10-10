//! Platform event stamping and the backend → kernel event envelope.
//!
//! The window kernel does not learn *which* window an [`WindowEvent`] belongs
//! to from the event itself (it stays a small [`Copy`] POD). Instead the
//! backend wraps every event in a [`WindowEventEnvelope`] carrying the target
//! [`WindowId`] and a [`PlatformEventStamp`] (monotonic time, global sequence,
//! and source). The kernel sorts by `(timestamp, sequence)` for a
//! deterministic, replayable stream, as specified in
//! `prism_window_refactor_zh.md` §4.2 and aligned with the winit backend
//! design §5.4.
//!
//! Timestamps are plain nanoseconds ([`MonotonicTimestamp`]) rather than
//! `std::time::Instant`, so recordings are byte-stable across machines and the
//! kernel stays `no_std`.

use crate::event::WindowEvent;
use crate::window::WindowId;

/// A monotonic timestamp in nanoseconds since a backend-chosen epoch.
///
/// This marks the moment the engine *received* a platform event, not the
/// hardware sampling instant (the backend may keep a separate hardware stamp).
/// It is only meaningful relative to other `MonotonicTimestamp`s from the same
/// backend session, and is guaranteed non-decreasing within a session.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct MonotonicTimestamp(pub u64);

impl MonotonicTimestamp {
    /// The zero instant (session epoch).
    pub const ZERO: Self = Self(0);

    /// Builds a timestamp from a nanosecond count.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// The nanosecond count since the session epoch.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Nanoseconds elapsed from `earlier` to `self`, saturating at `0` if
    /// `earlier` is later (clocks are assumed monotonic, but this never
    /// underflows).
    #[must_use]
    pub const fn saturating_duration_since(self, earlier: Self) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// A backend-entry global monotonic sequence number.
///
/// Assigned once per platform event as it enters the engine, across *all*
/// sources (windows, input, lifecycle). It provides a total order under a
/// single timestamp, so the kernel can sort deterministically even when many
/// events share a coarse timestamp.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct PlatformEventSequence(pub u64);

impl PlatformEventSequence {
    /// The first sequence value.
    pub const FIRST: Self = Self(0);

    /// Returns this value, then advances `self` by one. Used by backends at the
    /// single entry point where platform events are stamped.
    #[must_use]
    pub fn post_increment(&mut self) -> Self {
        let current = *self;
        self.0 = self.0.wrapping_add(1);
        current
    }

    /// The raw sequence value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Which platform subsystem produced an event.
///
/// Used only for routing/diagnostics and consumer fast-filtering; it never
/// participates in ordering (ordering is `(timestamp, sequence)` only).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum EventSource {
    /// A window-level event (resize, focus, close, scale change, ...).
    #[default]
    Window,
    /// An input-device event (keyboard, mouse, touch, gamepad).
    Input,
    /// A raw/relative device event (unassociated mouse motion, HID).
    Device,
    /// An application lifecycle event (resumed, suspended, memory warning).
    Lifecycle,
    /// A backend-internal diagnostic event.
    Backend,
}

/// The stamp every platform event carries as it crosses the ABI boundary.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct PlatformEventStamp {
    /// When the engine received the event (monotonic, non-decreasing).
    pub timestamp: MonotonicTimestamp,
    /// Global entry-order sequence, for stable ordering under one timestamp.
    pub sequence: PlatformEventSequence,
    /// The originating subsystem (diagnostics / filtering only).
    pub source: EventSource,
}

impl PlatformEventStamp {
    /// Builds a stamp.
    #[must_use]
    pub const fn new(
        timestamp: MonotonicTimestamp,
        sequence: PlatformEventSequence,
        source: EventSource,
    ) -> Self {
        Self {
            timestamp,
            sequence,
            source,
        }
    }

    /// The `(timestamp, sequence)` key the kernel sorts by. `source` is
    /// deliberately excluded so interleaving sources never changes order.
    #[must_use]
    pub const fn order_key(self) -> (u64, u64) {
        (self.timestamp.0, self.sequence.0)
    }
}

/// A single window event tagged with its target window and arrival stamp.
///
/// This is the one event type backends push across the ABI boundary. It stays
/// [`Copy`] because every field is `Copy`, so it can be batched through a
/// lock-free ring and recorded/replayed cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WindowEventEnvelope {
    /// Arrival stamp (time + global sequence + source).
    pub stamp: PlatformEventStamp,
    /// The window the event is routed to.
    pub window: WindowId,
    /// The window-level event payload.
    pub event: WindowEvent,
}

impl WindowEventEnvelope {
    /// Builds an envelope.
    #[must_use]
    pub const fn new(stamp: PlatformEventStamp, window: WindowId, event: WindowEvent) -> Self {
        Self {
            stamp,
            window,
            event,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PhysicalSize;

    #[test]
    fn timestamp_saturates_and_measures() {
        let a = MonotonicTimestamp::from_nanos(1_000);
        let b = MonotonicTimestamp::from_nanos(3_500);
        assert_eq!(b.saturating_duration_since(a), 2_500);
        // Reversed never underflows.
        assert_eq!(a.saturating_duration_since(b), 0);
        assert_eq!(MonotonicTimestamp::ZERO.as_nanos(), 0);
    }

    #[test]
    fn sequence_post_increment_is_monotonic() {
        let mut seq = PlatformEventSequence::FIRST;
        let first = seq.post_increment();
        let second = seq.post_increment();
        assert_eq!(first.get(), 0);
        assert_eq!(second.get(), 1);
        assert_eq!(seq.get(), 2);
        assert!(first < second);
    }

    #[test]
    fn stamp_orders_by_time_then_sequence_ignoring_source() {
        let earlier = PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(10),
            PlatformEventSequence(5),
            EventSource::Window,
        );
        let later_same_time = PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(10),
            PlatformEventSequence(6),
            EventSource::Input,
        );
        let later_time = PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(11),
            PlatformEventSequence(0),
            EventSource::Backend,
        );
        assert!(earlier < later_same_time);
        assert!(later_same_time < later_time);
    }

    #[test]
    fn stamp_source_does_not_affect_order() {
        // The kernel orders strictly by the `(timestamp, sequence)` key, which
        // excludes `source`; `sequence` is globally unique so `source` never
        // actually participates in the sort. (The derived `Ord` keeps `source`
        // as a final tiebreak purely so `Ord` stays consistent with `Eq`.)
        let key = (MonotonicTimestamp::from_nanos(42), PlatformEventSequence(7));
        let a = PlatformEventStamp::new(key.0, key.1, EventSource::Window);
        let b = PlatformEventStamp::new(key.0, key.1, EventSource::Backend);
        assert_eq!(a.order_key(), b.order_key());
    }

    #[test]
    fn envelope_is_copy_and_preserves_fields() {
        let stamp = PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(1),
            PlatformEventSequence(2),
            EventSource::Window,
        );
        let env = WindowEventEnvelope::new(
            stamp,
            WindowId(9),
            WindowEvent::Resized(PhysicalSize::new(640, 480)),
        );
        let copied = env; // Copy, not move.
        assert_eq!(copied.window, WindowId(9));
        assert_eq!(env.window, WindowId(9));
        assert_eq!(copied.stamp, stamp);
    }
}
