//! Lock-free parameter cells shared between a host and a compiled patch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports design section 11 ("Patch exposes named inputs aligned with RTPC").
//! Because `prism_audio_core::graph::AudioGraph::node_mut` yields a trait object
//! with no downcast, exposed patch parameters are injected through a shared
//! atomic cell instead: a primitive node holds the reader end and the host (via
//! `ParamHandle`) holds a writer end. Reads and writes are lock-free and
//! real-time safe, so a running patch can be re-parameterized between blocks
//! without touching the audio-thread allocation or locking rules.

#[cfg(not(feature = "std"))]
use alloc::string::String;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

use prism_audio_core::Sample;

/// A shared scalar parameter value backed by a lock-free atomic.
///
/// The stored [`Sample`] is bit-cast into a [`u32`] so updates are atomic and
/// wait-free. Cloning a [`ParamCell`] shares the same underlying storage, which
/// is how a host-visible [`ParamHandle`] and the primitive node reading it stay
/// connected across the compile boundary.
#[derive(Debug, Clone)]
pub struct ParamCell {
    inner: Arc<AtomicU32>,
}

impl ParamCell {
    /// Creates a cell initialized to `value`.
    #[must_use]
    pub fn new(value: Sample) -> Self {
        Self {
            inner: Arc::new(AtomicU32::new(value.to_bits())),
        }
    }

    /// Atomically stores a new value.
    #[inline]
    pub fn set(&self, value: Sample) {
        self.inner.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Atomically loads the current value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> Sample {
        Sample::from_bits(self.inner.load(Ordering::Relaxed))
    }
}

impl Default for ParamCell {
    #[inline]
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// A host-facing handle to a named parameter exposed by a compiled patch.
///
/// Obtained from the compiled patch; writing through it updates the matching
/// primitive node on its next processed block.
#[derive(Debug, Clone)]
pub struct ParamHandle {
    name: String,
    cell: ParamCell,
}

impl ParamHandle {
    /// Builds a handle binding `name` to `cell`.
    #[must_use]
    pub fn new(name: String, cell: ParamCell) -> Self {
        Self { name, cell }
    }

    /// Returns the exposed parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the parameter value.
    #[inline]
    pub fn set(&self, value: Sample) {
        self.cell.set(value);
    }

    /// Reads the parameter value.
    #[inline]
    #[must_use]
    pub fn get(&self) -> Sample {
        self.cell.get()
    }

    /// Returns a clone of the underlying cell.
    #[must_use]
    pub fn cell(&self) -> ParamCell {
        self.cell.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    const EPS: Sample = 1.0e-6;

    #[test]
    fn shared_cell_sees_updates() {
        let a = ParamCell::new(0.5);
        let b = a.clone();
        a.set(0.25);
        assert!((b.get() - 0.25).abs() < EPS);
    }

    #[test]
    fn handle_round_trips() {
        let handle = ParamHandle::new("frequency".to_string(), ParamCell::new(440.0));
        assert_eq!(handle.name(), "frequency");
        handle.set(220.0);
        assert!((handle.get() - 220.0).abs() < EPS);
    }

    #[test]
    fn negative_and_extremes_round_trip() {
        let c = ParamCell::new(-1.0);
        assert!((c.get() + 1.0).abs() < EPS);
        c.set(1.0e9);
        assert!((c.get() - 1.0e9).abs() < 1.0);
    }
}
