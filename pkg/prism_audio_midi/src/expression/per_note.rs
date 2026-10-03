//! Per-note expression state and the active-note slot table.
//!
//! Per-note expression is the heart of MIDI 2.0 and MPE: every sounding note
//! carries its own pitch bend, pressure, and a small set of continuous
//! controllers (brightness, modulation, ...), independent of its neighbours on
//! the same channel. [`PerNoteState`] is the fixed-size record for one note,
//! and [`PerNoteExpression`] is a fixed-capacity table of such records keyed by
//! `(channel, note)`. Both are plain data with no allocation: the table pre
//! allocates all of its slots, reuses a slot for a retriggered note, and steals
//! the oldest slot deterministically when it is full, so the real-time side can
//! read a note's expression without ever touching the allocator.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the per-note expression state of design section 52 (per-voice
//! expression bus / per-note controllers / per-note pitch bend). Driven by
//! [`crate::ump::message::ChannelVoice`] per-note messages and read by
//! [`crate::mapping`].

use crate::expression::controller::PerNoteController;
use crate::ump::message::PITCH_BEND_CENTER_32;

/// The maximum number of distinct per-note controllers stored per note.
pub const MAX_PER_NOTE_CONTROLLERS: usize = 8;

/// The maximum number of simultaneously active per-note slots.
pub const MAX_ACTIVE_NOTES: usize = 64;

/// The expression state carried by a single sounding note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PerNoteState {
    /// The note number (`0`-`127`).
    pub note: u8,
    /// The 16-bit attack velocity.
    pub velocity: u16,
    /// The per-note pitch bend, centred at `0x8000_0000`.
    pub pitch_bend: u32,
    /// The per-note pressure.
    pub pressure: u32,
    /// Whether the per-note controllers are detached from the note's lifetime.
    pub detached: bool,
    controllers: [(u8, u32); MAX_PER_NOTE_CONTROLLERS],
    controller_count: usize,
}

impl PerNoteState {
    /// Creates a note's state with the given note number and velocity, pitch
    /// bend centred, pressure zero, and no controllers set.
    #[must_use]
    pub const fn new(note: u8, velocity: u16) -> Self {
        Self {
            note,
            velocity,
            pitch_bend: PITCH_BEND_CENTER_32,
            pressure: 0,
            detached: false,
            controllers: [(0, 0); MAX_PER_NOTE_CONTROLLERS],
            controller_count: 0,
        }
    }

    /// Sets a per-note controller value by raw index. Updates the value when
    /// the controller is already present; otherwise inserts it while free slots
    /// remain. Returns `true` when the value was stored.
    pub fn set_controller(&mut self, index: u8, value: u32) -> bool {
        for entry in &mut self.controllers[..self.controller_count] {
            if entry.0 == index {
                entry.1 = value;
                return true;
            }
        }
        if self.controller_count < MAX_PER_NOTE_CONTROLLERS {
            self.controllers[self.controller_count] = (index, value);
            self.controller_count += 1;
            true
        } else {
            false
        }
    }

    /// Returns a per-note controller value by raw index, or `None` when unset.
    #[must_use]
    pub fn controller(&self, index: u8) -> Option<u32> {
        self.controllers[..self.controller_count]
            .iter()
            .find(|entry| entry.0 == index)
            .map(|entry| entry.1)
    }

    /// Returns a per-note controller value by named controller, or `None` when
    /// unset.
    #[must_use]
    pub fn named_controller(&self, controller: PerNoteController) -> Option<u32> {
        self.controller(controller.index())
    }

    /// Resets pitch bend, pressure, and all controllers to their defaults,
    /// keeping the note and velocity. Used by a per-note management reset.
    pub fn reset_expression(&mut self) {
        self.pitch_bend = PITCH_BEND_CENTER_32;
        self.pressure = 0;
        self.controller_count = 0;
    }
}

/// A key identifying an active note by channel and note number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PerNoteKey {
    /// The channel (`0`-`15`).
    pub channel: u8,
    /// The note number (`0`-`127`).
    pub note: u8,
}

impl PerNoteKey {
    /// Creates a key.
    #[must_use]
    pub const fn new(channel: u8, note: u8) -> Self {
        Self { channel, note }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct Slot {
    key: PerNoteKey,
    state: PerNoteState,
    order: u64,
    used: bool,
}

impl Slot {
    const EMPTY: Self = Self {
        key: PerNoteKey::new(0, 0),
        state: PerNoteState::new(0, 0),
        order: 0,
        used: false,
    };
}

/// A fixed-capacity table of per-note expression state keyed by channel/note.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PerNoteExpression {
    #[cfg_attr(feature = "serialize", serde(with = "slot_serde"))]
    slots: [Slot; MAX_ACTIVE_NOTES],
    next_order: u64,
}

impl Default for PerNoteExpression {
    fn default() -> Self {
        Self::new()
    }
}

impl PerNoteExpression {
    /// Creates an empty table with every slot free.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [Slot::EMPTY; MAX_ACTIVE_NOTES],
            next_order: 0,
        }
    }

    /// Begins a note, reusing the slot for a retriggered `(channel, note)`,
    /// then a free slot, then the oldest slot. Returns the chosen slot index.
    pub fn note_on(&mut self, channel: u8, note: u8, velocity: u16) -> usize {
        let key = PerNoteKey::new(channel, note);
        let order = self.next_order;
        self.next_order += 1;
        let index = self
            .find(key)
            .or_else(|| self.find_free())
            .unwrap_or_else(|| self.oldest());
        self.slots[index] = Slot {
            key,
            state: PerNoteState::new(note, velocity),
            order,
            used: true,
        };
        index
    }

    /// Ends a note. When the note's controllers are detached the slot is kept
    /// alive (so later per-note controller changes still land); otherwise the
    /// slot is freed. Returns `true` when a matching note was found.
    pub fn note_off(&mut self, channel: u8, note: u8) -> bool {
        if let Some(index) = self.find(PerNoteKey::new(channel, note)) {
            if !self.slots[index].state.detached {
                self.slots[index].used = false;
            }
            true
        } else {
            false
        }
    }

    /// Sets the per-note pitch bend for an active note; returns `true` on hit.
    pub fn set_pitch_bend(&mut self, channel: u8, note: u8, bend: u32) -> bool {
        self.with_state(channel, note, |state| state.pitch_bend = bend)
    }

    /// Sets the per-note pressure for an active note; returns `true` on hit.
    pub fn set_pressure(&mut self, channel: u8, note: u8, pressure: u32) -> bool {
        self.with_state(channel, note, |state| state.pressure = pressure)
    }

    /// Sets a per-note controller for an active note; returns `true` on hit.
    pub fn set_controller(&mut self, channel: u8, note: u8, index: u8, value: u32) -> bool {
        self.with_state(channel, note, |state| {
            state.set_controller(index, value);
        })
    }

    /// Applies a per-note management command: sets the detach flag and, when
    /// requested, resets the note's expression. Returns `true` on hit.
    pub fn manage(&mut self, channel: u8, note: u8, detach: bool, reset: bool) -> bool {
        self.with_state(channel, note, |state| {
            state.detached = detach;
            if reset {
                state.reset_expression();
            }
        })
    }

    /// Returns an active note's state, or `None` when the note is not active.
    #[must_use]
    pub fn get(&self, channel: u8, note: u8) -> Option<&PerNoteState> {
        self.find(PerNoteKey::new(channel, note))
            .map(|index| &self.slots[index].state)
    }

    /// Returns the number of currently active notes.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.used).count()
    }

    fn with_state<F: FnOnce(&mut PerNoteState)>(
        &mut self,
        channel: u8,
        note: u8,
        update: F,
    ) -> bool {
        if let Some(index) = self.find(PerNoteKey::new(channel, note)) {
            update(&mut self.slots[index].state);
            true
        } else {
            false
        }
    }

    fn find(&self, key: PerNoteKey) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.used && slot.key == key)
    }

    fn find_free(&self) -> Option<usize> {
        self.slots.iter().position(|slot| !slot.used)
    }

    fn oldest(&self) -> usize {
        let mut best = 0;
        let mut best_order = u64::MAX;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.order < best_order {
                best_order = slot.order;
                best = index;
            }
        }
        best
    }
}

/// Serializes and deserializes the fixed-capacity slot table as a sequence,
/// since serde does not implement its array traits for arrays this large.
#[cfg(feature = "serialize")]
mod slot_serde {
    use super::{MAX_ACTIVE_NOTES, Slot};
    use serde::{Deserialize, Deserializer, Serializer};
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    pub(super) fn serialize<S: Serializer>(
        value: &[Slot; MAX_ACTIVE_NOTES],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(value.iter())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<[Slot; MAX_ACTIVE_NOTES], D::Error> {
        let items = Vec::<Slot>::deserialize(deserializer)?;
        if items.len() != MAX_ACTIVE_NOTES {
            return Err(serde::de::Error::invalid_length(
                items.len(),
                &"MAX_ACTIVE_NOTES note slots",
            ));
        }
        let mut array = [Slot::EMPTY; MAX_ACTIVE_NOTES];
        for (dst, src) in array.iter_mut().zip(items) {
            *dst = src;
        }
        Ok(array)
    }
}


#[cfg(test)]
mod tests {
    use super::{MAX_ACTIVE_NOTES, PerNoteExpression};
    use crate::expression::controller::PerNoteController;
    use crate::ump::message::PITCH_BEND_CENTER_32;

    #[test]
    fn note_on_then_expression_updates() {
        let mut table = PerNoteExpression::new();
        table.note_on(1, 60, 0x8000);
        assert!(table.set_pitch_bend(1, 60, 0xC000_0000));
        assert!(table.set_pressure(1, 60, 0x4000_0000));
        assert!(table.set_controller(1, 60, PerNoteController::Brightness.index(), 0x1234));
        let state = table.get(1, 60).expect("active");
        assert_eq!(state.pitch_bend, 0xC000_0000);
        assert_eq!(state.pressure, 0x4000_0000);
        assert_eq!(state.named_controller(PerNoteController::Brightness), Some(0x1234));
    }

    #[test]
    fn note_off_frees_slot() {
        let mut table = PerNoteExpression::new();
        table.note_on(0, 64, 0x4000);
        assert_eq!(table.active_count(), 1);
        assert!(table.note_off(0, 64));
        assert_eq!(table.active_count(), 0);
        assert!(table.get(0, 64).is_none());
    }

    #[test]
    fn updates_miss_for_inactive_note() {
        let mut table = PerNoteExpression::new();
        assert!(!table.set_pitch_bend(0, 64, 0));
    }

    #[test]
    fn detach_keeps_slot_after_note_off() {
        let mut table = PerNoteExpression::new();
        table.note_on(0, 64, 0x4000);
        assert!(table.manage(0, 64, true, false));
        assert!(table.note_off(0, 64));
        // Detached: slot survives and still accepts controller changes.
        assert!(table.set_controller(0, 64, 74, 0x10));
    }

    #[test]
    fn reset_restores_defaults() {
        let mut table = PerNoteExpression::new();
        table.note_on(0, 64, 0x4000);
        table.set_pitch_bend(0, 64, 0x1000_0000);
        table.set_pressure(0, 64, 0x2000_0000);
        assert!(table.manage(0, 64, false, true));
        let state = table.get(0, 64).expect("active");
        assert_eq!(state.pitch_bend, PITCH_BEND_CENTER_32);
        assert_eq!(state.pressure, 0);
    }

    #[test]
    fn full_table_steals_oldest() {
        let mut table = PerNoteExpression::new();
        for note in 0..MAX_ACTIVE_NOTES {
            table.note_on(0, note as u8, 0x100);
        }
        assert_eq!(table.active_count(), MAX_ACTIVE_NOTES);
        // One more note steals the oldest (note 0) and keeps the table full.
        table.note_on(1, 100, 0x200);
        assert_eq!(table.active_count(), MAX_ACTIVE_NOTES);
        assert!(table.get(0, 0).is_none());
        assert!(table.get(1, 100).is_some());
    }
}
