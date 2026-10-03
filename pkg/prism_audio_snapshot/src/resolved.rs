//! A flattened parameter-to-value map produced by resolving snapshots.
//!
//! Resolving a [`Snapshot`] (or a weighted blend of several) collapses its
//! targets into a plain [`ParameterId`]-to-[`Sample`] map. The domain of each
//! parameter is applied during resolution and interpolation, so the resolved
//! map carries only final values; callers that still need the domain (for
//! continued interpolation) keep it alongside, as [`crate::transition`] does.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Produced from [`crate::snapshot::Snapshot`] and consumed by
//! [`crate::transition::Transition`] and [`crate::mixer::SnapshotMixer`] as the
//! live parameter state of the mix.

use alloc::collections::BTreeMap;
use alloc::collections::btree_map::{Iter, Keys};

use crate::parameter::ParameterId;
use crate::snapshot::Snapshot;
use prism_audio_core::math::Sample;

/// A deterministic map from parameter id to its current resolved value.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedParameters(BTreeMap<ParameterId, Sample>);

impl ResolvedParameters {
    /// Creates an empty resolved-parameter map.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Resolves a snapshot by taking each target's value.
    #[must_use]
    pub fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut map = BTreeMap::new();
        for target in snapshot.targets() {
            map.insert(target.id, target.value);
        }
        Self(map)
    }

    /// Returns the value for `id`, if present.
    #[must_use]
    pub fn get(&self, id: ParameterId) -> Option<Sample> {
        self.0.get(&id).copied()
    }

    /// Inserts or replaces the value for `id`.
    pub fn set(&mut self, id: ParameterId, value: Sample) {
        self.0.insert(id, value);
    }

    /// Iterates `(id, value)` pairs in ascending parameter-id order.
    pub fn iter(&self) -> Iter<'_, ParameterId, Sample> {
        self.0.iter()
    }

    /// Iterates the parameter ids in ascending order.
    pub fn param_ids(&self) -> Keys<'_, ParameterId, Sample> {
        self.0.keys()
    }

    /// Returns the number of resolved parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when no parameters are resolved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::ParameterKind;
    use crate::snapshot::SnapshotId;
    use crate::target::ParameterTarget;

    const EPS: Sample = 1e-6;

    #[test]
    fn from_snapshot_takes_target_values() {
        let snap = Snapshot::new(SnapshotId::new(1))
            .with_target(ParameterTarget::new(
                ParameterId::new(2),
                ParameterKind::Linear,
                0.75,
            ))
            .with_target(ParameterTarget::new(
                ParameterId::new(4),
                ParameterKind::Decibel,
                -6.0,
            ));
        let resolved = ResolvedParameters::from_snapshot(&snap);
        assert_eq!(resolved.len(), 2);
        assert!((resolved.get(ParameterId::new(2)).expect("present") - 0.75).abs() < EPS);
        assert!((resolved.get(ParameterId::new(4)).expect("present") - (-6.0)).abs() < EPS);
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut r = ResolvedParameters::new();
        assert!(r.is_empty());
        r.set(ParameterId::new(7), 1.5);
        assert_eq!(r.len(), 1);
        assert!((r.get(ParameterId::new(7)).expect("present") - 1.5).abs() < EPS);
        assert!(r.get(ParameterId::new(8)).is_none());
    }

    #[test]
    fn param_ids_are_ordered() {
        let mut r = ResolvedParameters::new();
        r.set(ParameterId::new(5), 0.0);
        r.set(ParameterId::new(1), 0.0);
        let ids: Vec<u32> = r.param_ids().map(|id| id.get()).collect();
        assert_eq!(ids, vec![1, 5]);
    }
}
