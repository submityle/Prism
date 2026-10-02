//! **Events**: the named triggers a game posts to drive audio.
//!
//! An [`Event`] is nothing more than a stable [`crate::id::EventId`] plus an
//! ordered list of [`crate::action::Action`]s. Posting it runs the actions in
//! order. This mirrors the universal "the game posts an event by name, the
//! audio engine runs its authored actions" contract shared by every major
//! middleware, modelled here as plain, serialisable data.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original, data-only event record. No AI/ML.
//!
//! # Relationship
//!
//! Events are stored in [`crate::model::ContentModel`] and executed by
//! [`crate::system::EventSystem::post_event`]. Their actions reference
//! containers, states, switches, and RTPCs by id.

use alloc::vec::Vec;

use crate::action::Action;
use crate::id::EventId;

/// A named trigger carrying the ordered actions to run when posted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Event {
    /// Stable id the game posts by.
    pub id: EventId,
    /// Actions executed in order when the event is posted.
    pub actions: Vec<Action>,
}

impl Event {
    /// Builds an event with no actions yet.
    #[must_use]
    pub fn new(id: EventId) -> Self {
        Self { id, actions: Vec::new() }
    }

    /// Builds an event from an existing action list.
    #[must_use]
    pub fn with_actions(id: EventId, actions: Vec<Action>) -> Self {
        Self { id, actions }
    }

    /// Appends an action, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with(mut self, action: Action) -> Self {
        self.actions.push(action);
        self
    }

    /// Appends an action in place.
    pub fn push(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Returns the ordered actions.
    #[must_use]
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Returns the number of actions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if the event has no actions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{Playable, SoundId};

    #[test]
    fn new_event_is_empty() {
        let e = Event::new(EventId::new(1));
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn with_builder_chains_actions_in_order() {
        let e = Event::new(EventId::new(1))
            .with(Action::Play(Playable::Sound(SoundId::new(10))))
            .with(Action::StopAll);
        assert_eq!(e.len(), 2);
        assert_eq!(e.actions()[0], Action::Play(Playable::Sound(SoundId::new(10))));
        assert_eq!(e.actions()[1], Action::StopAll);
    }

    #[test]
    fn push_appends_in_place() {
        let mut e = Event::new(EventId::new(2));
        e.push(Action::StopAll);
        assert!(!e.is_empty());
        assert_eq!(e.len(), 1);
    }

    #[test]
    fn with_actions_preserves_given_list() {
        let e = Event::with_actions(
            EventId::new(3),
            alloc::vec![Action::StopAll, Action::StopAll],
        );
        assert_eq!(e.len(), 2);
    }
}
