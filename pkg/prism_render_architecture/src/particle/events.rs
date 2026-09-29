//! Particle event system and cross-system data channels (design §14).
//!
//! Ember uses a two-tier event model, mirroring `Niagara`'s events plus data
//! channels and `VFX Graph`'s event attributes:
//!
//! * **GPU events** are lightweight, same-frame, per-emitter signals
//!   (`OnSpawn` / `OnDeath` / `OnCollision` / `OnCondition`) appended to a
//!   per-emitter ring and consumed by a later `PerEvent` stage that spawns
//!   inheriting the originator's attributes.
//! * **Data channels** are named, potentially cross-frame, cross-system rings:
//!   one system appends (for example "every explosion point") and others read
//!   and spawn (sparks / shockwave / smoke).
//!
//! This module owns the deterministic ring-buffer reservation and back-pressure
//! contract; the atomic append itself runs on the `GPU`.

/// The kind of same-frame `GPU` event an emitter can raise (design §14).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventKind {
    /// A particle was spawned this frame.
    OnSpawn,
    /// A particle died this frame.
    OnDeath,
    /// A particle collided with the scene (depth/`SDF`/ray, design §22).
    OnCollision,
    /// A user-authored predicate fired.
    OnCondition,
}

/// The outcome of reserving `count` slots in a bounded event ring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reservation {
    /// The first slot index granted (wrapped into `0..capacity`).
    pub start: u32,
    /// How many slots were actually granted (clamped to remaining space).
    pub granted: u32,
    /// How many were dropped because the ring was full (back-pressure count).
    pub dropped: u32,
}

/// A bounded append ring for `GPU` events with saturating back-pressure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventRing {
    capacity: u32,
    len: u32,
    dropped_total: u32,
}

impl EventRing {
    /// Creates an empty ring of the given capacity.
    #[must_use]
    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            len: 0,
            dropped_total: 0,
        }
    }

    /// Reserves up to `count` contiguous slots, clamping to the remaining space
    /// and accumulating the overflow into the dropped counter (design §14
    /// back-pressure: overflow is discarded and reported, never blocks).
    pub fn reserve(&mut self, count: u32) -> Reservation {
        let remaining = self.capacity - self.len;
        let granted = count.min(remaining);
        let dropped = count - granted;
        let start = self.len;
        self.len += granted;
        self.dropped_total = self.dropped_total.saturating_add(dropped);
        Reservation {
            start,
            granted,
            dropped,
        }
    }

    /// The number of live events queued.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the ring holds no events.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The total number of events dropped to back-pressure since creation.
    #[must_use]
    pub const fn dropped_total(&self) -> u32 {
        self.dropped_total
    }

    /// Clears the queue for the next frame, preserving the dropped counter.
    pub fn begin_frame(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_fits_within_capacity() {
        let mut ring = EventRing::new(8);
        let r = ring.reserve(5);
        assert_eq!(r.start, 0);
        assert_eq!(r.granted, 5);
        assert_eq!(r.dropped, 0);
        assert_eq!(ring.len(), 5);
    }

    #[test]
    fn overflow_is_clamped_and_counted() {
        let mut ring = EventRing::new(4);
        let a = ring.reserve(3);
        assert_eq!(a.granted, 3);
        let b = ring.reserve(5);
        assert_eq!(b.start, 3);
        assert_eq!(b.granted, 1);
        assert_eq!(b.dropped, 4);
        assert_eq!(ring.dropped_total(), 4);
        assert_eq!(ring.len(), 4);
    }

    #[test]
    fn begin_frame_resets_len_but_keeps_dropped() {
        let mut ring = EventRing::new(2);
        let _ = ring.reserve(5);
        assert_eq!(ring.dropped_total(), 3);
        ring.begin_frame();
        assert!(ring.is_empty());
        assert_eq!(ring.dropped_total(), 3);
    }

    #[test]
    fn event_kinds_are_distinct() {
        assert_ne!(EventKind::OnSpawn, EventKind::OnDeath);
        assert_ne!(EventKind::OnCollision, EventKind::OnCondition);
    }
}
