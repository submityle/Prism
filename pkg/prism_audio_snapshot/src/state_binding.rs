//! Mapping from global states to the snapshots they activate.
//!
//! This is where design section 13 snapshots meet design section 18 states.
//! Each [`StateGroupId`] maps its member [`StateId`]s to a [`SnapshotId`], so
//! when a group's active state changes the mix can follow. Resolving against a
//! live `StateManager` reads the active state of every bound group and returns
//! the snapshots to activate, in deterministic group order with duplicates
//! removed.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Joins `prism_audio_content::id::{StateGroupId, StateId}` and
//! `prism_audio_content::state::StateManager` to [`crate::snapshot::SnapshotId`].
//! The resolved set feeds [`crate::mixer::SnapshotMixer::drive_from_states`].

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use prism_audio_content::id::{StateGroupId, StateId};
use prism_audio_content::state::StateManager;

use crate::snapshot::SnapshotId;

/// Bindings from `(group, state)` pairs to the snapshot they activate.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateSnapshotBindings {
    /// For each group, the snapshot each of its states maps to.
    pub map: BTreeMap<StateGroupId, BTreeMap<StateId, SnapshotId>>,
}

impl StateSnapshotBindings {
    /// Creates an empty set of bindings.
    #[must_use]
    pub fn new() -> Self {
        Self { map: BTreeMap::new() }
    }

    /// Binds `(group, state)` to `snapshot`, replacing any prior binding.
    pub fn bind(&mut self, group: StateGroupId, state: StateId, snapshot: SnapshotId) {
        self.map.entry(group).or_default().insert(state, snapshot);
    }

    /// Returns the snapshot bound to `(group, state)`, if any.
    #[must_use]
    pub fn snapshot_for(&self, group: StateGroupId, state: StateId) -> Option<SnapshotId> {
        self.map.get(&group).and_then(|states| states.get(&state).copied())
    }

    /// Resolves the snapshots activated by the current active states.
    ///
    /// Only the groups that appear in these bindings are consulted. The result
    /// is ordered by group id with duplicate snapshots removed, so it is
    /// deterministic regardless of how the states were set.
    #[must_use]
    pub fn resolve_active(&self, states: &StateManager) -> Vec<SnapshotId> {
        let mut out: Vec<SnapshotId> = Vec::new();
        let mut seen: BTreeSet<SnapshotId> = BTreeSet::new();
        for (&group, bound) in &self.map {
            let Some(active) = states.active(group) else {
                continue;
            };
            let Some(&snapshot) = bound.get(&active) else {
                continue;
            };
            if seen.insert(snapshot) {
                out.push(snapshot);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_content::state::StateGroup;

    fn manager_with(group: u32, members: &[u32], default: u32) -> StateManager {
        let mut mgr = StateManager::new();
        let states: Vec<StateId> = members.iter().map(|&s| StateId::new(s)).collect();
        mgr.register(StateGroup::new(StateGroupId::new(group), states, StateId::new(default)));
        mgr
    }

    #[test]
    fn bind_and_lookup() {
        let mut b = StateSnapshotBindings::new();
        b.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(100));
        assert_eq!(
            b.snapshot_for(StateGroupId::new(1), StateId::new(10)),
            Some(SnapshotId::new(100))
        );
        assert!(b.snapshot_for(StateGroupId::new(1), StateId::new(11)).is_none());
    }

    #[test]
    fn resolve_active_reads_default_state() {
        let mgr = manager_with(1, &[10, 11], 10);
        let mut b = StateSnapshotBindings::new();
        b.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(100));
        b.bind(StateGroupId::new(1), StateId::new(11), SnapshotId::new(101));
        assert_eq!(b.resolve_active(&mgr), alloc::vec![SnapshotId::new(100)]);
    }

    #[test]
    fn resolve_active_follows_state_change() {
        let mut mgr = manager_with(1, &[10, 11], 10);
        assert!(mgr.set(StateGroupId::new(1), StateId::new(11)));
        let mut b = StateSnapshotBindings::new();
        b.bind(StateGroupId::new(1), StateId::new(11), SnapshotId::new(101));
        assert_eq!(b.resolve_active(&mgr), alloc::vec![SnapshotId::new(101)]);
    }

    #[test]
    fn resolve_active_dedups_shared_snapshot() {
        let mut mgr = StateManager::new();
        mgr.register(StateGroup::new(
            StateGroupId::new(1),
            alloc::vec![StateId::new(10)],
            StateId::new(10),
        ));
        mgr.register(StateGroup::new(
            StateGroupId::new(2),
            alloc::vec![StateId::new(20)],
            StateId::new(20),
        ));
        let mut b = StateSnapshotBindings::new();
        b.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(100));
        b.bind(StateGroupId::new(2), StateId::new(20), SnapshotId::new(100));
        assert_eq!(b.resolve_active(&mgr), alloc::vec![SnapshotId::new(100)]);
    }

    #[test]
    fn resolve_active_empty_without_bindings() {
        let mgr = manager_with(1, &[10], 10);
        let b = StateSnapshotBindings::new();
        assert!(b.resolve_active(&mgr).is_empty());
    }

    #[test]
    fn resolve_active_ignores_unbound_active_state() {
        let mgr = manager_with(1, &[10, 11], 10);
        let mut b = StateSnapshotBindings::new();
        // Only state 11 is bound, but 10 is active.
        b.bind(StateGroupId::new(1), StateId::new(11), SnapshotId::new(101));
        assert!(b.resolve_active(&mgr).is_empty());
    }
}
