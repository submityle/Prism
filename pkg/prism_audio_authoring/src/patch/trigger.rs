//! Lock-free gate and trigger cells for note-on style patch inputs.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports the trigger ports of design section 11 ("Patch exposes trigger
//! ports such as note-on, injected sample-accurately by the section 8
//! scheduler"). A [`GateCell`] carries a sustain gate plus a monotonically
//! increasing trigger counter; a primitive node compares the counter against
//! its last-seen value each block to detect a fresh edge. The host drives it
//! through a [`TriggerHandle`]. All operations are lock-free and real-time safe.

#[cfg(not(feature = "std"))]
use alloc::string::String;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

/// Shared gate state plus an edge counter, both lock-free.
///
/// The gate is a sustained boolean (open while a note is held); the counter
/// increments on every [`GateCell::trigger`] so a reader can detect one or more
/// note-on edges that occurred since it last looked, even across a block.
#[derive(Debug, Clone)]
pub struct GateCell {
    gate: Arc<AtomicU32>,
    edges: Arc<AtomicU32>,
}

impl GateCell {
    /// Creates a closed gate with a zeroed edge counter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gate: Arc::new(AtomicU32::new(0)),
            edges: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Opens or closes the sustain gate.
    #[inline]
    pub fn set_gate(&self, open: bool) {
        self.gate.store(u32::from(open), Ordering::Relaxed);
    }

    /// Returns `true` while the sustain gate is open.
    #[inline]
    #[must_use]
    pub fn gate(&self) -> bool {
        self.gate.load(Ordering::Relaxed) != 0
    }

    /// Registers a note-on edge, opening the gate and bumping the counter.
    #[inline]
    pub fn trigger(&self) {
        self.gate.store(1, Ordering::Relaxed);
        self.edges.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current edge counter.
    #[inline]
    #[must_use]
    pub fn edge_count(&self) -> u32 {
        self.edges.load(Ordering::Relaxed)
    }
}

impl Default for GateCell {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// A host-facing handle to a named trigger/gate input of a compiled patch.
#[derive(Debug, Clone)]
pub struct TriggerHandle {
    name: String,
    cell: GateCell,
}

impl TriggerHandle {
    /// Builds a handle binding `name` to `cell`.
    #[must_use]
    pub fn new(name: String, cell: GateCell) -> Self {
        Self { name, cell }
    }

    /// Returns the exposed trigger name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fires a note-on edge.
    #[inline]
    pub fn trigger(&self) {
        self.cell.trigger();
    }

    /// Opens or closes the sustain gate.
    #[inline]
    pub fn set_gate(&self, open: bool) {
        self.cell.set_gate(open);
    }

    /// Returns a clone of the underlying cell.
    #[must_use]
    pub fn cell(&self) -> GateCell {
        self.cell.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn gate_tracks_state() {
        let c = GateCell::new();
        assert!(!c.gate());
        c.set_gate(true);
        assert!(c.gate());
        c.set_gate(false);
        assert!(!c.gate());
    }

    #[test]
    fn trigger_increments_edges_and_opens_gate() {
        let c = GateCell::new();
        let start = c.edge_count();
        c.trigger();
        c.trigger();
        assert_eq!(c.edge_count(), start + 2);
        assert!(c.gate());
    }

    #[test]
    fn handle_shares_cell() {
        let handle = TriggerHandle::new("note_on".to_string(), GateCell::new());
        assert_eq!(handle.name(), "note_on");
        let cell = handle.cell();
        handle.trigger();
        assert_eq!(cell.edge_count(), 1);
    }
}
