//! A table of named snapshots keyed by [`SnapshotId`].
//!
//! The [`SnapshotRegistry`] owns every snapshot the mixer can transition to.
//! It is a thin, deterministic [`BTreeMap`] wrapper: snapshots are registered
//! once from authoring data and then looked up by id at runtime.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Stores [`crate::snapshot::Snapshot`] values. Read by [`crate::stack`] when
//! resolving blends and by [`crate::mixer::SnapshotMixer`] when starting a
//! transition.

use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Keys;

use crate::snapshot::{Snapshot, SnapshotId};

/// A deterministic collection of snapshots keyed by their id.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotRegistry {
    /// Registered snapshots, held in ascending id order.
    pub snapshots: BTreeMap<SnapshotId, Snapshot>,
}

impl SnapshotRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { snapshots: BTreeMap::new() }
    }

    /// Registers (or replaces) a snapshot under its own id, returning any
    /// snapshot previously stored under that id.
    pub fn register(&mut self, snapshot: Snapshot) -> Option<Snapshot> {
        self.snapshots.insert(snapshot.id, snapshot)
    }

    /// Returns the snapshot stored under `id`, if any.
    #[must_use]
    pub fn get(&self, id: SnapshotId) -> Option<&Snapshot> {
        self.snapshots.get(&id)
    }

    /// Returns `true` when a snapshot is stored under `id`.
    #[must_use]
    pub fn contains(&self, id: SnapshotId) -> bool {
        self.snapshots.contains_key(&id)
    }

    /// Removes and returns the snapshot stored under `id`, if any.
    pub fn remove(&mut self, id: SnapshotId) -> Option<Snapshot> {
        self.snapshots.remove(&id)
    }

    /// Iterates the registered ids in ascending order.
    pub fn ids(&self) -> Keys<'_, SnapshotId, Snapshot> {
        self.snapshots.keys()
    }

    /// Returns the number of registered snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` when no snapshots are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::{ParameterId, ParameterKind};
    use crate::target::ParameterTarget;

    fn sample_snapshot(id: u32) -> Snapshot {
        Snapshot::new(SnapshotId::new(id)).with_target(ParameterTarget::new(
            ParameterId::new(1),
            ParameterKind::Linear,
            1.0,
        ))
    }

    #[test]
    fn register_get_contains_remove() {
        let mut reg = SnapshotRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.register(sample_snapshot(1)).is_none());
        assert!(reg.contains(SnapshotId::new(1)));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(SnapshotId::new(1)).is_some());
        let removed = reg.remove(SnapshotId::new(1)).expect("was registered");
        assert_eq!(removed.id(), SnapshotId::new(1));
        assert!(!reg.contains(SnapshotId::new(1)));
    }

    #[test]
    fn register_replaces_and_returns_previous() {
        let mut reg = SnapshotRegistry::new();
        reg.register(sample_snapshot(1));
        let prev = reg.register(sample_snapshot(1)).expect("previous present");
        assert_eq!(prev.id(), SnapshotId::new(1));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn ids_are_sorted() {
        let mut reg = SnapshotRegistry::new();
        reg.register(sample_snapshot(5));
        reg.register(sample_snapshot(2));
        let ids: Vec<u32> = reg.ids().map(|id| id.get()).collect();
        assert_eq!(ids, vec![2, 5]);
    }
}
