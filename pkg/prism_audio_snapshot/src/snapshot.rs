//! A named snapshot: a set of target parameter values.
//!
//! A [`Snapshot`] captures where a group of mixer, bus, and effect parameters
//! should settle when the snapshot is fully active. It owns its targets in a
//! [`BTreeMap`] keyed by [`ParameterId`] so iteration and resolution are
//! deterministic regardless of insertion order.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Holds [`crate::target::ParameterTarget`] values keyed by
//! [`crate::parameter::ParameterId`]. Stored in [`crate::registry`], resolved
//! by [`crate::resolved`] and [`crate::stack`].

use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Values;

use crate::parameter::{ParameterId, ParameterKind};
use crate::target::ParameterTarget;
use prism_audio_core::math::Sample;

/// Opaque, stable identifier for one snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct SnapshotId(pub u32);

impl SnapshotId {
    /// Wraps a raw numeric handle.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw numeric handle.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for SnapshotId {
    fn from(raw: u32) -> Self {
        Self(raw)
    }
}

impl From<SnapshotId> for u32 {
    fn from(id: SnapshotId) -> Self {
        id.0
    }
}

/// A named set of parameter targets.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Snapshot {
    /// Identity of this snapshot.
    pub id: SnapshotId,
    /// Target values keyed by parameter, held in deterministic key order.
    pub targets: BTreeMap<ParameterId, ParameterTarget>,
}

impl Snapshot {
    /// Creates an empty snapshot with the given identity.
    #[must_use]
    pub fn new(id: SnapshotId) -> Self {
        Self { id, targets: BTreeMap::new() }
    }

    /// Inserts or replaces a target and returns the snapshot, for chaining.
    #[must_use]
    pub fn with_target(mut self, target: ParameterTarget) -> Self {
        self.targets.insert(target.id, target);
        self
    }

    /// Inserts or replaces the target for `id` with domain `kind` and `value`.
    pub fn set(&mut self, id: ParameterId, kind: ParameterKind, value: Sample) {
        self.targets.insert(id, ParameterTarget::new(id, kind, value));
    }

    /// Returns the target for `id`, if present.
    #[must_use]
    pub fn get(&self, id: ParameterId) -> Option<&ParameterTarget> {
        self.targets.get(&id)
    }

    /// Returns this snapshot's identity.
    #[must_use]
    pub const fn id(&self) -> SnapshotId {
        self.id
    }

    /// Iterates the targets in ascending parameter-id order.
    pub fn targets(&self) -> Values<'_, ParameterId, ParameterTarget> {
        self.targets.values()
    }

    /// Returns the number of targets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns `true` when the snapshot has no targets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn builder_and_accessors() {
        let snap = Snapshot::new(SnapshotId::new(1))
            .with_target(ParameterTarget::new(
                ParameterId::new(2),
                ParameterKind::Linear,
                0.5,
            ))
            .with_target(ParameterTarget::new(
                ParameterId::new(1),
                ParameterKind::Decibel,
                -3.0,
            ));
        assert_eq!(snap.id(), SnapshotId::new(1));
        assert_eq!(snap.len(), 2);
        assert!(!snap.is_empty());
        let got = snap.get(ParameterId::new(2)).expect("target present");
        assert!((got.value() - 0.5).abs() < EPS);
    }

    #[test]
    fn set_overwrites_existing_target() {
        let mut snap = Snapshot::new(SnapshotId::new(5));
        snap.set(ParameterId::new(1), ParameterKind::Linear, 0.1);
        snap.set(ParameterId::new(1), ParameterKind::Linear, 0.9);
        assert_eq!(snap.len(), 1);
        let got = snap.get(ParameterId::new(1)).expect("target present");
        assert!((got.value() - 0.9).abs() < EPS);
    }

    #[test]
    fn targets_iterate_in_key_order() {
        let snap = Snapshot::new(SnapshotId::new(1))
            .with_target(ParameterTarget::new(
                ParameterId::new(9),
                ParameterKind::Linear,
                1.0,
            ))
            .with_target(ParameterTarget::new(
                ParameterId::new(3),
                ParameterKind::Linear,
                2.0,
            ));
        let ids: Vec<u32> = snap.targets().map(|t| t.id().get()).collect();
        assert_eq!(ids, vec![3, 9]);
    }

    #[test]
    fn empty_snapshot_reports_empty() {
        let snap = Snapshot::new(SnapshotId::new(0));
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
    }
}
