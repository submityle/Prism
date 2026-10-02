//! Per-game-object **switch groups**: a discrete choice scoped to one emitter
//! (e.g. a `Surface` group of `Grass`/`Metal`/`Water` that selects a footstep
//! timbre for the walking character).
//!
//! Switches differ from states ([`crate::state`]) in scope: a switch is held
//! independently *per game object*, so two characters can stand on different
//! surfaces at once. Resolution falls back to the group default when an object
//! has no explicit switch set.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The per-object
//! discrete-switch concept is modelled from scratch as plain data. No AI/ML.
//!
//! # Relationship
//!
//! [`SwitchManager`] lives inside [`crate::system::EventSystem`]; its resolved
//! values pick branches in switch containers ([`crate::container`]).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::id::{GameObjectId, SwitchGroupId, SwitchId};

/// Authoring definition of a switch group: its member switches and default.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SwitchGroup {
    /// Stable id of this group.
    pub id: SwitchGroupId,
    /// The switches belonging to this group, in authored order.
    pub switches: Vec<SwitchId>,
    /// The switch used when an object has set none.
    pub default_switch: SwitchId,
}

impl SwitchGroup {
    /// Builds a switch group from its members and default.
    #[must_use]
    pub fn new(id: SwitchGroupId, switches: Vec<SwitchId>, default_switch: SwitchId) -> Self {
        Self { id, switches, default_switch }
    }

    /// Returns `true` if `switch` is a declared member of this group.
    #[must_use]
    pub fn contains(&self, switch: SwitchId) -> bool {
        self.switches.contains(&switch)
    }
}

/// Live, mutable per-object switch assignments.
///
/// State is stored sparsely: an object only appears once an explicit switch is
/// set for it, and resolution falls back to the group default otherwise. This
/// keeps memory proportional to objects that actually diverge from defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SwitchManager {
    groups: BTreeMap<SwitchGroupId, SwitchGroup>,
    // (object, group) -> switch; sparse.
    assignments: BTreeMap<(GameObjectId, SwitchGroupId), SwitchId>,
}

impl SwitchManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self { groups: BTreeMap::new(), assignments: BTreeMap::new() }
    }

    /// Registers (or replaces) a switch group definition.
    pub fn register(&mut self, group: SwitchGroup) {
        self.groups.insert(group.id, group);
    }

    /// Returns the registered definition of `group`, if any.
    #[must_use]
    pub fn group(&self, group: SwitchGroupId) -> Option<&SwitchGroup> {
        self.groups.get(&group)
    }

    /// Sets the switch for `(object, group)`.
    ///
    /// Returns `true` when applied; rejected (returns `false`) when the group
    /// is unknown or `switch` is not a member of it.
    pub fn set(&mut self, object: GameObjectId, group: SwitchGroupId, switch: SwitchId) -> bool {
        match self.groups.get(&group) {
            Some(def) if def.contains(switch) => {
                self.assignments.insert((object, group), switch);
                true
            }
            _ => false,
        }
    }

    /// Resolves the active switch for `(object, group)`, falling back to the
    /// group default when the object has no explicit assignment. Returns
    /// `None` only when the group itself is unknown.
    #[must_use]
    pub fn resolve(&self, object: GameObjectId, group: SwitchGroupId) -> Option<SwitchId> {
        let def = self.groups.get(&group)?;
        Some(
            self.assignments
                .get(&(object, group))
                .copied()
                .unwrap_or(def.default_switch),
        )
    }

    /// Clears every explicit assignment for `object` (it reverts to defaults).
    pub fn clear_object(&mut self, object: GameObjectId) {
        self.assignments.retain(|(obj, _), _| *obj != object);
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

    fn group() -> SwitchGroup {
        SwitchGroup::new(
            SwitchGroupId::new(1),
            alloc::vec![SwitchId::new(10), SwitchId::new(11)],
            SwitchId::new(10),
        )
    }

    #[test]
    fn resolve_falls_back_to_default_without_assignment() {
        let mut mgr = SwitchManager::new();
        mgr.register(group());
        let obj = GameObjectId::new(100);
        assert_eq!(mgr.resolve(obj, SwitchGroupId::new(1)), Some(SwitchId::new(10)));
    }

    #[test]
    fn resolve_unknown_group_is_none() {
        let mgr = SwitchManager::new();
        assert_eq!(mgr.resolve(GameObjectId::new(1), SwitchGroupId::new(9)), None);
    }

    #[test]
    fn set_is_scoped_per_object() {
        let mut mgr = SwitchManager::new();
        mgr.register(group());
        let a = GameObjectId::new(1);
        let b = GameObjectId::new(2);
        assert!(mgr.set(a, SwitchGroupId::new(1), SwitchId::new(11)));
        // Object `a` diverges, object `b` still resolves the default.
        assert_eq!(mgr.resolve(a, SwitchGroupId::new(1)), Some(SwitchId::new(11)));
        assert_eq!(mgr.resolve(b, SwitchGroupId::new(1)), Some(SwitchId::new(10)));
    }

    #[test]
    fn set_rejects_non_member_and_unknown_group() {
        let mut mgr = SwitchManager::new();
        mgr.register(group());
        let obj = GameObjectId::new(1);
        assert!(!mgr.set(obj, SwitchGroupId::new(1), SwitchId::new(99)));
        assert!(!mgr.set(obj, SwitchGroupId::new(2), SwitchId::new(10)));
        // Rejection leaves the resolve at the default.
        assert_eq!(mgr.resolve(obj, SwitchGroupId::new(1)), Some(SwitchId::new(10)));
    }

    #[test]
    fn clear_object_reverts_to_defaults() {
        let mut mgr = SwitchManager::new();
        mgr.register(group());
        let obj = GameObjectId::new(1);
        assert!(mgr.set(obj, SwitchGroupId::new(1), SwitchId::new(11)));
        mgr.clear_object(obj);
        assert_eq!(mgr.resolve(obj, SwitchGroupId::new(1)), Some(SwitchId::new(10)));
    }

    #[test]
    fn len_and_is_empty_track_registration() {
        let mut mgr = SwitchManager::new();
        assert!(mgr.is_empty());
        mgr.register(group());
        assert_eq!(mgr.len(), 1);
        assert!(!mgr.is_empty());
    }
}
