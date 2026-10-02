//! Global **state groups**: mutually-exclusive game states (e.g. a `Combat`
//! group holding `Explore`/`Alert`/`Fight`) that drive mix snapshots and
//! event/container variation across the whole soundscape.
//!
//! Unlike per-object switches ([`crate::switch`]), a state group holds exactly
//! one active state globally. Changing it is a game-wide mood change: music
//! layering, snapshot interpolation, and container selection all read it.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The global
//! finite-state-group concept is modelled from scratch as plain data. No AI/ML.
//!
//! # Relationship
//!
//! [`StateManager`] lives inside [`crate::system::EventSystem`]; its active
//! states are read by container resolution ([`crate::container`]) and gate
//! which event variations run. (Mix-snapshot application is a higher mix-layer
//! concern and is intentionally out of scope for this data-only crate.)

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::id::{StateGroupId, StateId};

/// Authoring definition of one global state group: its member states and its
/// default (initial / fallback) state.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateGroup {
    /// Stable id of this group.
    pub id: StateGroupId,
    /// The states belonging to this group, in authored order.
    pub states: Vec<StateId>,
    /// The state active before any explicit change is posted.
    pub default_state: StateId,
}

impl StateGroup {
    /// Builds a group from its members and default state.
    #[must_use]
    pub fn new(id: StateGroupId, states: Vec<StateId>, default_state: StateId) -> Self {
        Self { id, states, default_state }
    }

    /// Returns `true` if `state` is a declared member of this group.
    #[must_use]
    pub fn contains(&self, state: StateId) -> bool {
        self.states.contains(&state)
    }
}

/// Live, mutable set of active states for every registered group.
///
/// Registering a group seeds its active state to the group default. Setting a
/// state that is not a member of its group is rejected (the active state is
/// left unchanged) so corrupt game input cannot push a group into an
/// undeclared state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateManager {
    groups: BTreeMap<StateGroupId, StateGroup>,
    active: BTreeMap<StateGroupId, StateId>,
}

impl StateManager {
    /// Creates an empty manager with no registered groups.
    #[must_use]
    pub fn new() -> Self {
        Self { groups: BTreeMap::new(), active: BTreeMap::new() }
    }

    /// Registers (or replaces) a group, seeding its active state to the
    /// group's declared default.
    pub fn register(&mut self, group: StateGroup) {
        self.active.insert(group.id, group.default_state);
        self.groups.insert(group.id, group);
    }

    /// Returns the currently active state of `group`, if the group exists.
    #[must_use]
    pub fn active(&self, group: StateGroupId) -> Option<StateId> {
        self.active.get(&group).copied()
    }

    /// Returns the registered definition of `group`, if any.
    #[must_use]
    pub fn group(&self, group: StateGroupId) -> Option<&StateGroup> {
        self.groups.get(&group)
    }

    /// Sets the active state of `group` to `state`.
    ///
    /// Returns `true` when the change was applied. The change is rejected
    /// (returns `false`) when the group is unknown or `state` is not a declared
    /// member of the group.
    pub fn set(&mut self, group: StateGroupId, state: StateId) -> bool {
        match self.groups.get(&group) {
            Some(def) if def.contains(state) => {
                self.active.insert(group, state);
                true
            }
            _ => false,
        }
    }

    /// Resets every registered group to its default state.
    pub fn reset_to_defaults(&mut self) {
        for (id, def) in &self.groups {
            self.active.insert(*id, def.default_state);
        }
    }

    /// Number of registered groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Returns `true` when no group is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> StateGroup {
        StateGroup::new(
            StateGroupId::new(1),
            alloc::vec![StateId::new(10), StateId::new(11), StateId::new(12)],
            StateId::new(10),
        )
    }

    #[test]
    fn contains_only_declared_members() {
        let g = group();
        assert!(g.contains(StateId::new(11)));
        assert!(!g.contains(StateId::new(99)));
    }

    #[test]
    fn register_seeds_active_to_default() {
        let mut mgr = StateManager::new();
        assert!(mgr.is_empty());
        mgr.register(group());
        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.active(StateGroupId::new(1)), Some(StateId::new(10)));
    }

    #[test]
    fn set_accepts_member_rejects_non_member() {
        let mut mgr = StateManager::new();
        mgr.register(group());
        assert!(mgr.set(StateGroupId::new(1), StateId::new(12)));
        assert_eq!(mgr.active(StateGroupId::new(1)), Some(StateId::new(12)));
        // Non-member is rejected and leaves the active state unchanged.
        assert!(!mgr.set(StateGroupId::new(1), StateId::new(99)));
        assert_eq!(mgr.active(StateGroupId::new(1)), Some(StateId::new(12)));
    }

    #[test]
    fn set_unknown_group_is_rejected() {
        let mut mgr = StateManager::new();
        assert!(!mgr.set(StateGroupId::new(7), StateId::new(10)));
        assert_eq!(mgr.active(StateGroupId::new(7)), None);
    }

    #[test]
    fn reset_to_defaults_restores_every_group() {
        let mut mgr = StateManager::new();
        mgr.register(group());
        assert!(mgr.set(StateGroupId::new(1), StateId::new(12)));
        mgr.reset_to_defaults();
        assert_eq!(mgr.active(StateGroupId::new(1)), Some(StateId::new(10)));
    }

    #[test]
    fn group_lookup_returns_definition() {
        let mut mgr = StateManager::new();
        mgr.register(group());
        assert_eq!(mgr.group(StateGroupId::new(1)).map(|g| g.id), Some(StateGroupId::new(1)));
        assert!(mgr.group(StateGroupId::new(2)).is_none());
    }
}
