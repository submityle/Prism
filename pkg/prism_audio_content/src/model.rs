//! **Content model**: the immutable registry that holds every authored event,
//! container, game-sync group, and RTPC definition for a project.
//!
//! The model is a flat set of id-keyed tables, so it is cheap to clone,
//! serialise, and share. It owns no runtime state; the mutable cursors, live
//! switch/RTPC values, and RNG live in [`crate::system::EventSystem`], which
//! reads this model to resolve events. Keeping authoring data and live state
//! apart lets one model drive many independent emitters deterministically.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original, data-only registry built on ordered maps. No AI/ML.
//!
//! # Relationship
//!
//! [`ContentModel`] aggregates [`crate::event::Event`],
//! [`crate::container::Container`], [`crate::state::StateGroup`],
//! [`crate::switch::SwitchGroup`], [`crate::rtpc::RtpcDefinition`], and
//! [`crate::rtpc::RtpcBinding`]. [`crate::system::EventSystem::new`] builds its
//! live [`crate::state::StateManager`], [`crate::switch::SwitchManager`], and
//! [`crate::rtpc::RtpcRegistry`] from the tables exposed here.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::container::Container;
use crate::event::Event;
use crate::id::{ContainerId, EventId, RtpcId, StateGroupId, SwitchGroupId};
use crate::rtpc::{RtpcBinding, RtpcDefinition, RtpcRegistry};
use crate::state::{StateGroup, StateManager};
use crate::switch::{SwitchGroup, SwitchManager};

/// Immutable, id-keyed registry of all authored audio content.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentModel {
    events: BTreeMap<EventId, Event>,
    containers: BTreeMap<ContainerId, Container>,
    state_groups: BTreeMap<StateGroupId, StateGroup>,
    switch_groups: BTreeMap<SwitchGroupId, SwitchGroup>,
    rtpc_defs: BTreeMap<RtpcId, RtpcDefinition>,
    rtpc_bindings: Vec<RtpcBinding>,
}

impl ContentModel {
    /// Builds an empty model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) an event, keyed by its id.
    pub fn add_event(&mut self, event: Event) {
        self.events.insert(event.id, event);
    }

    /// Registers (or replaces) a container, keyed by its id.
    pub fn add_container(&mut self, container: Container) {
        self.containers.insert(container.id, container);
    }

    /// Registers (or replaces) a state group, keyed by its id.
    pub fn add_state_group(&mut self, group: StateGroup) {
        self.state_groups.insert(group.id, group);
    }

    /// Registers (or replaces) a switch group, keyed by its id.
    pub fn add_switch_group(&mut self, group: SwitchGroup) {
        self.switch_groups.insert(group.id, group);
    }

    /// Registers (or replaces) an RTPC definition, keyed by its id.
    pub fn add_rtpc(&mut self, def: RtpcDefinition) {
        self.rtpc_defs.insert(def.id, def);
    }

    /// Appends an RTPC binding (parameter mapping).
    pub fn add_rtpc_binding(&mut self, binding: RtpcBinding) {
        self.rtpc_bindings.push(binding);
    }

    /// Looks up an event by id.
    #[must_use]
    pub fn event(&self, id: EventId) -> Option<&Event> {
        self.events.get(&id)
    }

    /// Looks up a container by id.
    #[must_use]
    pub fn container(&self, id: ContainerId) -> Option<&Container> {
        self.containers.get(&id)
    }

    /// Looks up a state group by id.
    #[must_use]
    pub fn state_group(&self, id: StateGroupId) -> Option<&StateGroup> {
        self.state_groups.get(&id)
    }

    /// Looks up a switch group by id.
    #[must_use]
    pub fn switch_group(&self, id: SwitchGroupId) -> Option<&SwitchGroup> {
        self.switch_groups.get(&id)
    }

    /// Looks up an RTPC definition by id.
    #[must_use]
    pub fn rtpc(&self, id: RtpcId) -> Option<&RtpcDefinition> {
        self.rtpc_defs.get(&id)
    }

    /// Returns the number of registered events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns the number of registered containers.
    #[must_use]
    pub fn container_count(&self) -> usize {
        self.containers.len()
    }

    /// Builds a fresh [`StateManager`] seeded with every registered state
    /// group (each starting at its default state).
    #[must_use]
    pub fn build_state_manager(&self) -> StateManager {
        let mut mgr = StateManager::new();
        for group in self.state_groups.values() {
            mgr.register(group.clone());
        }
        mgr
    }

    /// Builds a fresh [`SwitchManager`] seeded with every registered switch
    /// group.
    #[must_use]
    pub fn build_switch_manager(&self) -> SwitchManager {
        let mut mgr = SwitchManager::new();
        for group in self.switch_groups.values() {
            mgr.register(group.clone());
        }
        mgr
    }

    /// Builds a fresh [`RtpcRegistry`] seeded with every registered RTPC
    /// definition and binding.
    #[must_use]
    pub fn build_rtpc_registry(&self) -> RtpcRegistry {
        let mut reg = RtpcRegistry::new();
        for def in self.rtpc_defs.values() {
            reg.define(def.clone());
        }
        for binding in &self.rtpc_bindings {
            reg.bind(binding.clone());
        }
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::container::{Container, ContainerKind, SequenceMode};
    use crate::curve::ParameterCurve;
    use crate::id::{Playable, SoundId, StateId, SwitchId};
    use crate::parameter::ParameterTarget;
    use crate::rtpc::{RtpcBinding, RtpcDefinition};
    use crate::state::StateGroup;
    use crate::switch::SwitchGroup;

    #[test]
    fn add_and_lookup_events_and_containers() {
        let mut model = ContentModel::new();
        let ev = Event::with_actions(
            EventId::new(1),
            alloc::vec![Action::Play(Playable::Sound(SoundId::new(10)))],
        );
        model.add_event(ev);
        model.add_container(Container::new(
            ContainerId::new(1),
            ContainerKind::Sequence {
                children: alloc::vec![Playable::Sound(SoundId::new(10))],
                mode: SequenceMode::Loop,
            },
        ));
        assert_eq!(model.event_count(), 1);
        assert_eq!(model.container_count(), 1);
        assert!(model.event(EventId::new(1)).is_some());
        assert!(model.event(EventId::new(2)).is_none());
        assert!(model.container(ContainerId::new(1)).is_some());
    }

    #[test]
    fn add_replaces_existing_entry_by_id() {
        let mut model = ContentModel::new();
        model.add_event(Event::new(EventId::new(1)));
        model.add_event(Event::with_actions(EventId::new(1), alloc::vec![Action::StopAll]));
        assert_eq!(model.event_count(), 1);
        assert_eq!(model.event(EventId::new(1)).map(Event::len), Some(1));
    }

    #[test]
    fn build_state_manager_seeds_defaults() {
        let mut model = ContentModel::new();
        model.add_state_group(StateGroup::new(
            StateGroupId::new(1),
            alloc::vec![StateId::new(10), StateId::new(11)],
            StateId::new(11),
        ));
        let mgr = model.build_state_manager();
        assert_eq!(mgr.active(StateGroupId::new(1)), Some(StateId::new(11)));
    }

    #[test]
    fn build_switch_manager_seeds_registration() {
        let mut model = ContentModel::new();
        model.add_switch_group(SwitchGroup::new(
            SwitchGroupId::new(1),
            alloc::vec![SwitchId::new(1), SwitchId::new(2)],
            SwitchId::new(1),
        ));
        let mgr = model.build_switch_manager();
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn build_rtpc_registry_seeds_defs_and_bindings() {
        let mut model = ContentModel::new();
        model.add_rtpc(RtpcDefinition::new(RtpcId::new(1), 0.0, 1.0, 0.0));
        model.add_rtpc_binding(RtpcBinding::new(
            RtpcId::new(1),
            ParameterTarget::VolumeDb,
            ParameterCurve::line(0.0, -60.0, 1.0, 0.0),
        ));
        let reg = model.build_rtpc_registry();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.bindings_for(RtpcId::new(1)).len(), 1);
    }

    #[test]
    fn empty_model_has_zero_counts() {
        let model = ContentModel::new();
        assert_eq!(model.event_count(), 0);
        assert_eq!(model.container_count(), 0);
        assert!(model.rtpc(RtpcId::new(1)).is_none());
        assert!(model.state_group(StateGroupId::new(1)).is_none());
        assert!(model.switch_group(SwitchGroupId::new(1)).is_none());
    }
}
