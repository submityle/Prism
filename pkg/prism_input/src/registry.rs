//! Deterministic batching of input envelopes: the drain-to-batch "frame fence".
//!
//! The simulation thread owns input state and advances it once per tick by
//! draining every queued [`InputEventEnvelope`] into a single, stably-sorted
//! batch (design doc §4.3/§5.2). Sorting is by the stamp's
//! `(timestamp, sequence)` only (see [`PlatformInputStamp`]), so the same
//! event stream always produces the same ordered batch — the basis for
//! record/replay and lockstep determinism.
//!
//! This module provides:
//! - [`sort_envelopes`]: a stable in-place sort for an existing batch;
//! - [`InputEnvelopeQueue`]: an accumulation buffer whose
//!   [`drain_sorted`](InputEnvelopeQueue::drain_sorted) /
//!   [`apply_sorted`](InputEnvelopeQueue::apply_sorted) implement the
//!   frame-fence entry point, mirroring the window kernel's
//!   `WindowRegistry::apply_sorted`.

use alloc::vec::Vec;

use crate::envelope::InputEventEnvelope;

/// Stably sorts `envelopes` in place by their stamp's `(timestamp, sequence)`.
///
/// The sort is *stable*: envelopes comparing equal (same timestamp and
/// sequence — which cannot happen for a well-behaved backend, since sequence is
/// globally unique) keep their original relative order, so the result is fully
/// deterministic.
pub fn sort_envelopes(envelopes: &mut [InputEventEnvelope]) {
    envelopes.sort_by(InputEventEnvelope::stamp_order);
}

/// An accumulation buffer for input envelopes with a deterministic
/// drain-to-batch entry point.
///
/// A backend (or the single-threaded host fallback) pushes envelopes as they
/// arrive; the simulation drains them once per tick into a sorted batch. The
/// buffer's capacity is retained across drains so the steady-state hot path is
/// allocation-free (see the zero-allocation proposition in the design doc).
#[derive(Clone, Default, Debug)]
pub struct InputEnvelopeQueue {
    buffer: Vec<InputEventEnvelope>,
}

impl InputEnvelopeQueue {
    /// Creates an empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: Vec::new(),
        }
    }

    /// Creates an empty queue preallocated for `capacity` envelopes, sized for
    /// the worst-case events-per-frame so steady-state pushes never allocate.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Appends an envelope in arrival order.
    pub fn push(&mut self, envelope: InputEventEnvelope) {
        self.buffer.push(envelope);
    }

    /// The number of queued envelopes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The current spare capacity (retained across drains).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Drains every queued envelope into a new, stably-sorted batch, leaving
    /// the queue empty but keeping its capacity for the next frame.
    ///
    /// This is the frame fence: the returned slice is the one deterministic
    /// view of input the simulation sees for this tick.
    pub fn drain_sorted(&mut self) -> Vec<InputEventEnvelope> {
        let mut batch: Vec<InputEventEnvelope> = self.buffer.drain(..).collect();
        sort_envelopes(&mut batch);
        batch
    }

    /// Drains and stably sorts the queue, then applies `apply` to each envelope
    /// in `(timestamp, sequence)` order. The queue is left empty with its
    /// capacity retained.
    ///
    /// This mirrors the window kernel's `WindowRegistry::apply_sorted`: a single
    /// deterministic entry point that feeds ordered events into whatever state
    /// the caller owns, without this crate having to know the state's shape.
    pub fn apply_sorted<F>(&mut self, mut apply: F)
    where
        F: FnMut(InputEventEnvelope),
    {
        let batch = self.drain_sorted();
        for envelope in batch {
            apply(envelope);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{
        InputDeviceId, InputEventSequence, InputSource, MonotonicTimestamp, PlatformInputStamp,
    };
    use crate::event::ButtonState;
    use crate::keyboard::{KeyCode, KeyboardInput};

    fn env(ts: u64, seq: u64, code: KeyCode) -> InputEventEnvelope {
        InputEventEnvelope::new(
            PlatformInputStamp::new(
                MonotonicTimestamp::from_nanos(ts),
                InputEventSequence(seq),
                InputSource::Keyboard,
            ),
            InputDeviceId::from_raw(1),
            key_event(code),
        )
    }

    fn key_event(code: KeyCode) -> crate::event::InputEvent {
        crate::event::InputEvent::Keyboard(KeyboardInput {
            key_code: code,
            state: ButtonState::Pressed,
            repeat: false,
        })
    }

    fn order_keys(batch: &[InputEventEnvelope]) -> Vec<(u64, u64)> {
        batch.iter().map(|e| e.stamp.order_key()).collect()
    }

    #[test]
    fn sort_envelopes_orders_by_timestamp_then_sequence() {
        let mut batch = [
            env(30, 0, KeyCode::KeyA),
            env(10, 5, KeyCode::KeyB),
            env(10, 2, KeyCode::KeyC),
            env(20, 1, KeyCode::KeyD),
        ];
        sort_envelopes(&mut batch);
        assert_eq!(order_keys(&batch), [(10, 2), (10, 5), (20, 1), (30, 0)]);
    }

    #[test]
    fn sort_is_deterministic_across_input_orderings() {
        let mut a = [
            env(5, 1, KeyCode::KeyA),
            env(5, 3, KeyCode::KeyB),
            env(1, 9, KeyCode::KeyC),
        ];
        let mut b = [
            env(1, 9, KeyCode::KeyC),
            env(5, 3, KeyCode::KeyB),
            env(5, 1, KeyCode::KeyA),
        ];
        sort_envelopes(&mut a);
        sort_envelopes(&mut b);
        assert_eq!(order_keys(&a), order_keys(&b));
        assert_eq!(a.to_vec(), b.to_vec());
    }

    #[test]
    fn queue_drain_sorted_empties_and_keeps_capacity() {
        let mut q = InputEnvelopeQueue::with_capacity(8);
        q.push(env(20, 1, KeyCode::KeyA));
        q.push(env(10, 2, KeyCode::KeyB));
        assert_eq!(q.len(), 2);
        assert!(!q.is_empty());

        let batch = q.drain_sorted();
        assert_eq!(order_keys(&batch), [(10, 2), (20, 1)]);
        assert!(q.is_empty());
        // Capacity retained for the next frame (allocation-free steady state).
        assert!(q.capacity() >= 8);
    }

    #[test]
    fn queue_apply_sorted_visits_in_order() {
        let mut q = InputEnvelopeQueue::new();
        q.push(env(30, 0, KeyCode::KeyA));
        q.push(env(10, 0, KeyCode::KeyB));
        q.push(env(20, 0, KeyCode::KeyC));

        let mut seen: Vec<u64> = Vec::new();
        q.apply_sorted(|e| seen.push(e.timestamp().as_nanos()));
        assert_eq!(seen, [10, 20, 30]);
        assert!(q.is_empty());
    }

    #[test]
    fn empty_queue_apply_is_noop() {
        let mut q = InputEnvelopeQueue::new();
        let mut count = 0_u32;
        q.apply_sorted(|_| count += 1);
        assert_eq!(count, 0);
    }
}
