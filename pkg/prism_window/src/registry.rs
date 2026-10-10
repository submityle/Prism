//! The multi-window registry: deterministic ownership and event routing.
//!
//! The kernel owns exactly one [`WindowRegistry`] on the simulation side (no
//! `Arc<Mutex>`, no shared mutable state; `prism_window_refactor_zh.md` §4.5).
//! Backends push [`WindowEventEnvelope`]s carrying a target [`WindowId`]; the
//! registry routes each to the right [`Window`] and advances it purely. For a
//! whole frame's batch it sorts by `(timestamp, sequence)` first so replay and
//! live runs produce identical state (§4.2 law 2).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::envelope::WindowEventEnvelope;
use crate::event::WindowEvent;
use crate::window::{Window, WindowAttributes, WindowId};

/// A deterministic collection of live windows keyed by [`WindowId`].
///
/// Backed by a [`BTreeMap`] so iteration order is stable (ascending id),
/// independent of insertion order or hashing — important for reproducible
/// multi-window frames.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WindowRegistry {
    windows: BTreeMap<WindowId, Window>,
    next_id: u64,
}

impl WindowRegistry {
    /// An empty registry. The first allocated id is `WindowId(1)` (`0` is left
    /// free as a sentinel for callers that want one).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            windows: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Allocates a fresh, previously-unused [`WindowId`] without inserting a
    /// window (used to pre-allocate the id for a `Create` command target).
    #[must_use]
    pub fn allocate_id(&mut self) -> WindowId {
        let id = WindowId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    /// Creates a window from desired attributes and returns its id.
    pub fn create(&mut self, attributes: WindowAttributes) -> WindowId {
        let id = self.allocate_id();
        self.windows.insert(id, Window::new(attributes));
        id
    }

    /// Inserts a window under a caller-chosen id (e.g. the id a `Create`
    /// command pre-allocated via [`allocate_id`](Self::allocate_id)). Returns
    /// `false` if the id was already occupied (the existing window is kept).
    /// The internal allocator is advanced past `id` so it is never reused.
    pub fn insert_with_id(&mut self, id: WindowId, attributes: WindowAttributes) -> bool {
        if self.windows.contains_key(&id) {
            return false;
        }
        self.next_id = self.next_id.max(id.0.wrapping_add(1));
        self.windows.insert(id, Window::new(attributes));
        true
    }

    /// Removes a window, returning it if it existed.
    pub fn remove(&mut self, id: WindowId) -> Option<Window> {
        self.windows.remove(&id)
    }

    /// Borrows a window.
    #[must_use]
    pub fn get(&self, id: WindowId) -> Option<&Window> {
        self.windows.get(&id)
    }

    /// Mutably borrows a window.
    pub fn get_mut(&mut self, id: WindowId) -> Option<&mut Window> {
        self.windows.get_mut(&id)
    }

    /// Whether a window with this id exists.
    #[must_use]
    pub fn contains(&self, id: WindowId) -> bool {
        self.windows.contains_key(&id)
    }

    /// The number of live windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether no windows are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Iterates windows in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (WindowId, &Window)> {
        self.windows.iter().map(|(id, w)| (*id, w))
    }

    /// Iterates windows mutably in ascending id order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (WindowId, &mut Window)> {
        self.windows.iter_mut().map(|(id, w)| (*id, w))
    }

    /// Collects the live window ids in ascending order.
    #[must_use]
    pub fn ids(&self) -> Vec<WindowId> {
        self.windows.keys().copied().collect()
    }

    /// Routes one envelope to its target window and advances it.
    ///
    /// Returns `true` if the event changed observable state. An envelope for an
    /// unknown window is ignored (returns `false`) — the kernel only handles
    /// events for windows it owns. A [`WindowEvent::Destroyed`] removes the
    /// window from the registry.
    pub fn apply(&mut self, envelope: WindowEventEnvelope) -> bool {
        if matches!(envelope.event, WindowEvent::Destroyed) {
            return self.windows.remove(&envelope.window).is_some();
        }
        match self.windows.get_mut(&envelope.window) {
            Some(window) => window.apply(envelope.event),
            None => false,
        }
    }

    /// Drains and applies a whole frame's event batch deterministically.
    ///
    /// The slice is sorted in place by `(timestamp, sequence)` first (the
    /// kernel assumes an ordered stream; §4.2 law 2), then each envelope is
    /// routed via [`apply`](Self::apply). Returns the number of envelopes that
    /// changed observable state. Sorting is stable, so envelopes sharing a key
    /// keep their relative order.
    pub fn apply_sorted(&mut self, batch: &mut [WindowEventEnvelope]) -> usize {
        batch.sort_by_key(|env| env.stamp.order_key());
        let mut changed = 0;
        for &envelope in batch.iter() {
            if self.apply(envelope) {
                changed += 1;
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{
        EventSource, MonotonicTimestamp, PlatformEventSequence, PlatformEventStamp,
    };
    use crate::geometry::{PhysicalPosition, PhysicalSize};

    fn stamp(ts: u64, seq: u64) -> PlatformEventStamp {
        PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(ts),
            PlatformEventSequence(seq),
            EventSource::Window,
        )
    }

    #[test]
    fn create_allocates_increasing_ids() {
        let mut reg = WindowRegistry::new();
        let a = reg.create(WindowAttributes::default());
        let b = reg.create(WindowAttributes::default());
        assert_eq!(a, WindowId(1));
        assert_eq!(b, WindowId(2));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.ids(), alloc::vec![WindowId(1), WindowId(2)]);
    }

    #[test]
    fn insert_with_id_advances_allocator_and_rejects_dups() {
        let mut reg = WindowRegistry::new();
        assert!(reg.insert_with_id(WindowId(10), WindowAttributes::default()));
        // Next auto id must be past 10.
        let next = reg.create(WindowAttributes::default());
        assert_eq!(next, WindowId(11));
        // Duplicate id rejected.
        assert!(!reg.insert_with_id(WindowId(10), WindowAttributes::default()));
    }

    #[test]
    fn apply_routes_to_the_targeted_window() {
        let mut reg = WindowRegistry::new();
        let a = reg.create(WindowAttributes::default());
        let b = reg.create(WindowAttributes::default());
        let changed = reg.apply(WindowEventEnvelope::new(
            stamp(1, 0),
            b,
            WindowEvent::Focused(false),
        ));
        assert!(changed);
        // Only window b lost focus; a is untouched.
        assert!(reg.get(a).unwrap().focused());
        assert!(!reg.get(b).unwrap().focused());
    }

    #[test]
    fn apply_ignores_unknown_windows() {
        let mut reg = WindowRegistry::new();
        let changed = reg.apply(WindowEventEnvelope::new(
            stamp(1, 0),
            WindowId(999),
            WindowEvent::CloseRequested,
        ));
        assert!(!changed);
    }

    #[test]
    fn destroyed_event_removes_the_window() {
        let mut reg = WindowRegistry::new();
        let a = reg.create(WindowAttributes::default());
        assert!(reg.contains(a));
        let changed = reg.apply(WindowEventEnvelope::new(
            stamp(1, 0),
            a,
            WindowEvent::Destroyed,
        ));
        assert!(changed);
        assert!(!reg.contains(a));
        // A second Destroyed for the same (now absent) window is a no-op.
        assert!(!reg.apply(WindowEventEnvelope::new(
            stamp(2, 1),
            a,
            WindowEvent::Destroyed,
        )));
    }

    #[test]
    fn apply_sorted_orders_by_stamp_regardless_of_input_order() {
        let mut reg = WindowRegistry::new();
        let w = reg.create(WindowAttributes::default());
        // Deliver out of order: a later Moved before an earlier one.
        let mut batch = alloc::vec![
            WindowEventEnvelope::new(
                stamp(100, 2),
                w,
                WindowEvent::Moved(PhysicalPosition::new(20, 20)),
            ),
            WindowEventEnvelope::new(
                stamp(50, 1),
                w,
                WindowEvent::Moved(PhysicalPosition::new(10, 10)),
            ),
        ];
        let changed = reg.apply_sorted(&mut batch);
        assert_eq!(changed, 2);
        // Final position must be the one with the greatest (ts, seq): (100,2).
        assert_eq!(
            reg.get(w).unwrap().position(),
            Some(PhysicalPosition::new(20, 20))
        );
    }

    #[test]
    fn apply_sorted_counts_only_observable_changes() {
        let mut reg = WindowRegistry::new();
        let w = reg.create(WindowAttributes::default());
        let size = PhysicalSize::new(800, 600);
        let mut batch = alloc::vec![
            WindowEventEnvelope::new(stamp(1, 0), w, WindowEvent::Resized(size)),
            // Same size again: no observable change.
            WindowEventEnvelope::new(stamp(2, 1), w, WindowEvent::Resized(size)),
        ];
        assert_eq!(reg.apply_sorted(&mut batch), 1);
    }
}
