//! The ABI envelope layer that wraps raw [`InputEvent`]s with a deterministic,
//! platform-independent stamp.
//!
//! Backends translate OS/windowing messages into [`InputEventEnvelope`]s and
//! feed them to the kernel. Every envelope carries:
//! - a [`MonotonicTimestamp`] (integer nanoseconds, never a wall clock), so the
//!   same stream replays to byte-identical state on any machine;
//! - an [`InputEventSequence`], the backend's global monotonic counter that
//!   breaks ties between events sharing a timestamp (high-poll-rate mice);
//! - an [`InputDeviceId`], so multiple mice/keyboards/gamepads (split-screen,
//!   pen + mouse) stay distinguishable;
//! - an [`InputSource`] tag describing which device class produced the event.
//!
//! This mirrors the window kernel's envelope paradigm, but the types are
//! defined independently here: `prism_input` keeps its zero-dependency,
//! `no_std + alloc`, `unsafe`-free kernel contract and does not depend on the
//! window crate.

use crate::event::InputEvent;

/// A monotonic timestamp measured in integer nanoseconds.
///
/// This is deliberately an integer rather than a `std::time::Instant`: the
/// deterministic record/replay and lockstep contracts (see the design doc §5)
/// require byte-stable timestamps that serialize identically across platforms.
/// Backends are responsible for clamping any non-monotonic OS clock into a
/// monotonic sequence before stamping events.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct MonotonicTimestamp(pub u64);

impl MonotonicTimestamp {
    /// The zero instant (the start of the timeline).
    pub const ZERO: Self = Self(0);

    /// Builds a timestamp from a raw nanosecond count.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// The raw nanosecond count.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// The number of nanoseconds elapsed from `earlier` to `self`, saturating
    /// to `0` if `earlier` is actually later (a clock regression the backend
    /// failed to clamp). Never panics.
    #[must_use]
    pub const fn saturating_duration_since(self, earlier: Self) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// A backend-assigned, globally monotonic sequence number.
///
/// The backend increments this once per emitted event, giving every envelope a
/// unique, strictly increasing identity. It breaks ordering ties when several
/// events carry the same [`MonotonicTimestamp`] (common at 1k–8k Hz polling),
/// keeping sorts stable and deterministic.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct InputEventSequence(pub u64);

impl InputEventSequence {
    /// The first sequence value.
    pub const ZERO: Self = Self(0);

    /// The raw counter value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next sequence value, saturating at [`u64::MAX`] rather than
    /// wrapping so ordering never silently resets.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

/// Identifies a physical input device instance.
///
/// Unlike [`GamepadId`](crate::gamepad::GamepadId) (a logical gamepad slot),
/// this distinguishes *any* device across classes — two mice, a pen and a
/// mouse used together, several keyboards in split-screen — so the kernel can
/// aggregate or split state per device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct InputDeviceId(pub u64);

impl InputDeviceId {
    /// Builds a device id from a raw backend handle value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw backend handle value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The device class that produced an event.
///
/// This is a coarse routing/diagnostic tag on the envelope stamp; it is
/// intentionally distinct from the richer `InputSource` binding descriptor that
/// the (separate) Action-mapping layer defines.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum InputSource {
    /// A physical keyboard.
    Keyboard,
    /// A mouse (buttons, relative motion, wheel).
    Mouse,
    /// A touchscreen or touchpad touch surface.
    Touch,
    /// A gamepad / game controller.
    Gamepad,
    /// A stylus / pen digitizer.
    Pen,
    /// Any source not covered above.
    Other,
}

/// The deterministic stamp attached to every input event by the backend.
///
/// The primary order is `(timestamp, sequence)`, exposed as [`order_key`](Self::order_key)
/// and used everywhere a batch is sorted into causal order. `sequence` is a
/// global monotonic counter, so the primary key is already unique per event and
/// [`source`](Self::source) never actually breaks a tie in practice.
///
/// Ordering is **derived** (field order `timestamp` -> `sequence` -> `source`)
/// rather than hand-written, so [`Ord`] stays consistent with [`Eq`]: the std
/// contract requires `a == b` iff `a.cmp(&b) == Equal`. A hand-rolled `Ord`
/// that ignored `source` while `Eq` compared it would violate that invariant
/// and corrupt `BTreeMap`/`BTreeSet`/`binary_search`. Sort by [`order_key`](Self::order_key)
/// when you explicitly want source-agnostic ordering.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PlatformInputStamp {
    /// The monotonic nanosecond timestamp of the event.
    pub timestamp: MonotonicTimestamp,
    /// The backend's global monotonic sequence number for the event.
    pub sequence: InputEventSequence,
    /// The device class that produced the event.
    pub source: InputSource,
}

impl PlatformInputStamp {
    /// Builds a stamp from its parts.
    #[must_use]
    pub const fn new(
        timestamp: MonotonicTimestamp,
        sequence: InputEventSequence,
        source: InputSource,
    ) -> Self {
        Self {
            timestamp,
            sequence,
            source,
        }
    }

    /// The `(timestamp, sequence)` pair that defines the stamp's total order.
    #[must_use]
    pub const fn order_key(self) -> (u64, u64) {
        (self.timestamp.0, self.sequence.0)
    }
}

/// A single [`InputEvent`] wrapped with its deterministic ABI stamp and the
/// device it came from.
///
/// This is the unit the backend hands to the kernel. It stays `Copy` because
/// [`InputEvent`] is `Copy`, keeping the hot path allocation-free. It cannot be
/// `Eq`/`Ord` directly because `InputEvent` carries `f32` payloads; order
/// envelopes by their [`stamp`](Self::stamp) via [`stamp_order`](Self::stamp_order)
/// or the helpers in [`registry`](crate::registry).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InputEventEnvelope {
    /// The deterministic stamp (timestamp, sequence, source).
    pub stamp: PlatformInputStamp,
    /// The device instance that produced the event.
    pub device: InputDeviceId,
    /// The wrapped raw input event.
    pub event: InputEvent,
}

impl InputEventEnvelope {
    /// Builds an envelope from its parts.
    #[must_use]
    pub const fn new(stamp: PlatformInputStamp, device: InputDeviceId, event: InputEvent) -> Self {
        Self {
            stamp,
            device,
            event,
        }
    }

    /// The envelope's monotonic timestamp (shorthand for `self.stamp.timestamp`).
    #[must_use]
    pub const fn timestamp(self) -> MonotonicTimestamp {
        self.stamp.timestamp
    }

    /// The envelope's sequence number (shorthand for `self.stamp.sequence`).
    #[must_use]
    pub const fn sequence(self) -> InputEventSequence {
        self.stamp.sequence
    }

    /// Compares two envelopes by their stamp's `(timestamp, sequence)` order.
    ///
    /// Provided because [`InputEventEnvelope`] cannot implement [`Ord`]
    /// (its [`InputEvent`] payload is only `PartialEq`). Use this as the sort
    /// comparator for deterministic batching.
    #[must_use]
    pub fn stamp_order(&self, other: &Self) -> core::cmp::Ordering {
        self.stamp.cmp(&other.stamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyboard::{KeyCode, KeyboardInput};

    fn key_event(code: KeyCode) -> InputEvent {
        InputEvent::Keyboard(KeyboardInput {
            key_code: code,
            state: crate::event::ButtonState::Pressed,
            repeat: false,
        })
    }

    #[test]
    fn timestamp_zero_and_nanos_roundtrip() {
        assert_eq!(MonotonicTimestamp::ZERO.as_nanos(), 0);
        let t = MonotonicTimestamp::from_nanos(1_234);
        assert_eq!(t.as_nanos(), 1_234);
        assert_eq!(t, MonotonicTimestamp(1_234));
    }

    #[test]
    fn timestamp_is_monotonic_comparable() {
        let a = MonotonicTimestamp::from_nanos(10);
        let b = MonotonicTimestamp::from_nanos(20);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.min(b), a);
    }

    #[test]
    fn saturating_duration_since_handles_regression() {
        let later = MonotonicTimestamp::from_nanos(100);
        let earlier = MonotonicTimestamp::from_nanos(40);
        assert_eq!(later.saturating_duration_since(earlier), 60);
        // Regression saturates to zero rather than wrapping or panicking.
        assert_eq!(earlier.saturating_duration_since(later), 0);
        assert_eq!(later.saturating_duration_since(later), 0);
    }

    #[test]
    fn sequence_next_saturates() {
        assert_eq!(InputEventSequence::ZERO.next(), InputEventSequence(1));
        assert_eq!(InputEventSequence(5).get(), 5);
        assert_eq!(InputEventSequence(u64::MAX).next(), InputEventSequence(u64::MAX));
    }

    #[test]
    fn device_id_distinguishes_devices() {
        let a = InputDeviceId::from_raw(1);
        let b = InputDeviceId::from_raw(2);
        assert_ne!(a, b);
        assert_eq!(a.get(), 1);
        assert!(a < b);
    }

    #[test]
    fn stamp_orders_by_timestamp_then_sequence_only() {
        let base = MonotonicTimestamp::from_nanos(100);
        // Same timestamp, different sequence -> sequence breaks the tie.
        let s1 = PlatformInputStamp::new(base, InputEventSequence(1), InputSource::Mouse);
        let s2 = PlatformInputStamp::new(base, InputEventSequence(2), InputSource::Keyboard);
        assert!(s1 < s2);

        // Earlier timestamp wins regardless of sequence.
        let earlier =
            PlatformInputStamp::new(MonotonicTimestamp::from_nanos(50), InputEventSequence(99), InputSource::Gamepad);
        assert!(earlier < s1);
    }

    #[test]
    fn stamp_order_key_is_source_agnostic_but_ord_matches_eq() {
        let t = MonotonicTimestamp::from_nanos(7);
        let seq = InputEventSequence(3);
        let a = PlatformInputStamp::new(t, seq, InputSource::Keyboard);
        let b = PlatformInputStamp::new(t, seq, InputSource::Pen);
        // The source-agnostic primary key is identical...
        assert_eq!(a.order_key(), b.order_key());
        // ...and the two stamps are genuinely distinct values (source retained).
        assert_ne!(a, b);
        // Ord MUST stay consistent with Eq (std contract: a == b iff cmp == Equal).
        // Distinct values therefore compare non-equal; `source` is the final,
        // never-tied-in-practice tiebreaker after (timestamp, sequence).
        assert_ne!(a.cmp(&b), core::cmp::Ordering::Equal);
        assert_eq!(a.cmp(&a), core::cmp::Ordering::Equal);
        // Primary key still dominates: a later sequence always sorts after,
        // regardless of source.
        let later = PlatformInputStamp::new(t, InputEventSequence(4), InputSource::Keyboard);
        assert!(b < later);
    }

    #[test]
    fn envelope_is_copy_and_exposes_stamp_fields() {
        let stamp = PlatformInputStamp::new(
            MonotonicTimestamp::from_nanos(42),
            InputEventSequence(9),
            InputSource::Keyboard,
        );
        let env = InputEventEnvelope::new(stamp, InputDeviceId::from_raw(3), key_event(KeyCode::KeyA));
        // Copy semantics: using `env` after copying must still compile and hold.
        let copied = env;
        assert_eq!(copied, env);
        assert_eq!(env.timestamp(), MonotonicTimestamp::from_nanos(42));
        assert_eq!(env.sequence(), InputEventSequence(9));
        assert_eq!(env.device, InputDeviceId::from_raw(3));
    }

    #[test]
    fn envelope_stamp_order_matches_stamp() {
        let e_early = InputEventEnvelope::new(
            PlatformInputStamp::new(MonotonicTimestamp::from_nanos(1), InputEventSequence(0), InputSource::Mouse),
            InputDeviceId::from_raw(1),
            key_event(KeyCode::KeyA),
        );
        let e_late = InputEventEnvelope::new(
            PlatformInputStamp::new(MonotonicTimestamp::from_nanos(2), InputEventSequence(0), InputSource::Mouse),
            InputDeviceId::from_raw(1),
            key_event(KeyCode::KeyB),
        );
        assert_eq!(e_early.stamp_order(&e_late), core::cmp::Ordering::Less);
    }
}
