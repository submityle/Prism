//! High-resolution controllers: 14-bit CC pairs and RPN/NRPN parsing.
//!
//! MIDI 1.0 reaches beyond 7-bit resolution in two standard ways, both decoded
//! here. First, continuous controllers 0-31 (coarse / MSB) pair with 32-63
//! (fine / LSB) to form a 14-bit value; [`HighResCc`] tracks those pairs and
//! yields the combined value up-scaled to 32 bits. Second, Registered and
//! Non-Registered Parameter Numbers (RPN / NRPN) are selected with CC 98-101
//! and written with the data-entry controllers (CC 6 / 38) or the
//! increment/decrement controllers (CC 96 / 97); [`RpnNrpnParser`] implements
//! that state machine, including the RPN Null deselect. The named per-note
//! controller indices used by MIDI 2.0 and MPE live in [`PerNoteController`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The controller
//! numbers and the RPN/NRPN convention are from the publicly published MIDI 1.0
//! and MIDI 2.0 specifications.
//!
//! # Relationship
//! Implements the high-resolution controller part of design section 52. Used by
//! [`crate::expression::channel_state`] to maintain per-channel controller and
//! pitch-bend-range state.

use crate::ump::scaling::scale_up;

/// A named per-note or channel controller, as assigned by MIDI 2.0 and MPE.
///
/// Only the handful of controllers this engine routes expressively are named;
/// anything else is preserved as [`PerNoteController::Other`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PerNoteController {
    /// Modulation (controller index 1).
    Modulation,
    /// Breath (controller index 2).
    Breath,
    /// Channel volume (controller index 7).
    Volume,
    /// Balance (controller index 8).
    Balance,
    /// Pan (controller index 10).
    Pan,
    /// Expression (controller index 11).
    Expression,
    /// Sound controller 5 / brightness, the MPE timbre dimension (index 74).
    Brightness,
    /// Any other controller index.
    Other {
        /// The raw controller index.
        index: u8,
    },
}

impl PerNoteController {
    /// Classifies a controller index.
    #[must_use]
    pub const fn from_index(index: u8) -> Self {
        match index {
            1 => Self::Modulation,
            2 => Self::Breath,
            7 => Self::Volume,
            8 => Self::Balance,
            10 => Self::Pan,
            11 => Self::Expression,
            74 => Self::Brightness,
            other => Self::Other { index: other },
        }
    }

    /// Returns the raw controller index.
    #[must_use]
    pub const fn index(self) -> u8 {
        match self {
            Self::Modulation => 1,
            Self::Breath => 2,
            Self::Volume => 7,
            Self::Balance => 8,
            Self::Pan => 10,
            Self::Expression => 11,
            Self::Brightness => 74,
            Self::Other { index } => index,
        }
    }
}

/// Tracker for the 32 coarse/fine MIDI 1.0 controller pairs.
///
/// Controllers 0-31 are the coarse (MSB) halves; 32-63 are their fine (LSB)
/// halves. Feeding a controller change updates the relevant half and returns
/// the current combined 14-bit value up-scaled to 32 bits, so a consumer sees
/// the same width as a native MIDI 2.0 controller.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HighResCc {
    msb: [u8; 32],
    lsb: [u8; 32],
}

impl Default for HighResCc {
    fn default() -> Self {
        Self::new()
    }
}

impl HighResCc {
    /// Creates a tracker with every pair at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            msb: [0; 32],
            lsb: [0; 32],
        }
    }

    /// Feeds a 7-bit controller change. For a coarse (0-31) or fine (32-63)
    /// controller this updates the pair and returns the logical controller
    /// index together with its combined 32-bit value. Other controllers return
    /// `None` because they are not part of a high-resolution pair.
    pub fn feed(&mut self, controller: u8, value: u8) -> Option<(u8, u32)> {
        let value = value & 0x7F;
        if controller < 32 {
            let slot = controller as usize;
            self.msb[slot] = value;
            Some((controller, self.combined(slot)))
        } else if controller < 64 {
            let slot = (controller - 32) as usize;
            self.lsb[slot] = value;
            Some((controller - 32, self.combined(slot)))
        } else {
            None
        }
    }

    /// Returns the current combined 32-bit value for a logical controller index
    /// (`0`-`31`); other indices return zero.
    #[must_use]
    pub fn value(&self, logical_index: u8) -> u32 {
        if logical_index < 32 {
            self.combined(logical_index as usize)
        } else {
            0
        }
    }

    fn combined(&self, slot: usize) -> u32 {
        let value14 = (u32::from(self.msb[slot]) << 7) | u32::from(self.lsb[slot]);
        scale_up(value14, 14, 32)
    }
}

/// Whether a parameter number is Registered (RPN) or Non-Registered (NRPN).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamKind {
    /// A Registered Parameter Number (standardised meaning).
    Registered,
    /// A Non-Registered Parameter Number (manufacturer-defined).
    Assignable,
}

/// A completed parameter write produced by [`RpnNrpnParser`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterUpdate {
    /// Whether the parameter is registered or assignable.
    pub kind: ParamKind,
    /// The parameter bank (MSB selector).
    pub bank: u8,
    /// The parameter index (LSB selector).
    pub index: u8,
    /// The 14-bit data value up-scaled to 32 bits.
    pub value: u32,
}

/// The MIDI 1.0 RPN/NRPN selection and data-entry state machine.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RpnNrpnParser {
    kind: ParamKind,
    bank: u8,
    index: u8,
    selected: bool,
    data_msb: u8,
    data_lsb: u8,
}

impl Default for RpnNrpnParser {
    fn default() -> Self {
        Self::new()
    }
}

impl RpnNrpnParser {
    /// Creates a parser with no parameter selected.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kind: ParamKind::Registered,
            bank: 0,
            index: 0,
            selected: false,
            data_msb: 0,
            data_lsb: 0,
        }
    }

    /// Feeds a 7-bit control change. Returns a [`ParameterUpdate`] when a data
    /// entry (or increment/decrement) completes a value for the selected
    /// parameter; returns `None` for selection changes or unrelated CCs.
    pub fn feed(&mut self, controller: u8, value: u8) -> Option<ParameterUpdate> {
        let value = value & 0x7F;
        match controller {
            101 => {
                self.kind = ParamKind::Registered;
                self.bank = value;
                self.update_selection();
                None
            }
            100 => {
                self.kind = ParamKind::Registered;
                self.index = value;
                self.update_selection();
                None
            }
            99 => {
                self.kind = ParamKind::Assignable;
                self.bank = value;
                self.update_selection();
                None
            }
            98 => {
                self.kind = ParamKind::Assignable;
                self.index = value;
                self.update_selection();
                None
            }
            6 => {
                self.data_msb = value;
                self.emit()
            }
            38 => {
                self.data_lsb = value;
                self.emit()
            }
            96 => {
                // Data increment: step the 14-bit value up by one.
                let next = self.data_value14().saturating_add(1).min(0x3FFF);
                self.data_msb = (next >> 7) as u8;
                self.data_lsb = (next & 0x7F) as u8;
                self.emit()
            }
            97 => {
                // Data decrement: step the 14-bit value down by one.
                let next = self.data_value14().saturating_sub(1);
                self.data_msb = (next >> 7) as u8;
                self.data_lsb = (next & 0x7F) as u8;
                self.emit()
            }
            _ => None,
        }
    }

    fn update_selection(&mut self) {
        // RPN Null (bank 0x7F, index 0x7F) deselects the current parameter.
        self.selected = !(self.kind == ParamKind::Registered
            && self.bank == 0x7F
            && self.index == 0x7F);
    }

    fn data_value14(&self) -> u16 {
        (u16::from(self.data_msb) << 7) | u16::from(self.data_lsb)
    }

    fn emit(&self) -> Option<ParameterUpdate> {
        if !self.selected {
            return None;
        }
        Some(ParameterUpdate {
            kind: self.kind,
            bank: self.bank,
            index: self.index,
            value: scale_up(u32::from(self.data_value14()), 14, 32),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HighResCc, ParamKind, PerNoteController, RpnNrpnParser};
    use crate::ump::scaling::scale_up;

    #[test]
    fn per_note_controller_round_trips() {
        for index in 0u8..=127 {
            assert_eq!(PerNoteController::from_index(index).index(), index);
        }
        assert_eq!(PerNoteController::from_index(74), PerNoteController::Brightness);
    }

    #[test]
    fn high_res_cc_combines_pair() {
        let mut cc = HighResCc::new();
        // Coarse value 0x40, fine value 0x00 -> 14-bit 0x2000 -> 32-bit center.
        let (index, value) = cc.feed(1, 0x40).expect("coarse");
        assert_eq!(index, 1);
        assert_eq!(value, scale_up(0x2000, 14, 32));
        // Add the fine half (controller 33) and confirm it refines the value.
        let (index, value) = cc.feed(33, 0x7F).expect("fine");
        assert_eq!(index, 1);
        assert_eq!(value, scale_up((0x40 << 7) | 0x7F, 14, 32));
    }

    #[test]
    fn high_res_cc_ignores_non_pair() {
        let mut cc = HighResCc::new();
        assert!(cc.feed(70, 0x10).is_none());
    }

    #[test]
    fn rpn_pitch_bend_range_update() {
        let mut parser = RpnNrpnParser::new();
        // Select RPN 0,0 (pitch bend sensitivity).
        assert!(parser.feed(101, 0).is_none());
        assert!(parser.feed(100, 0).is_none());
        // Data entry MSB = 2 semitones.
        let update = parser.feed(6, 2).expect("data entry");
        assert_eq!(update.kind, ParamKind::Registered);
        assert_eq!(update.bank, 0);
        assert_eq!(update.index, 0);
        assert_eq!(update.value, scale_up(2 << 7, 14, 32));
    }

    #[test]
    fn rpn_null_deselects() {
        let mut parser = RpnNrpnParser::new();
        parser.feed(101, 0);
        parser.feed(100, 0);
        // RPN Null.
        parser.feed(101, 0x7F);
        parser.feed(100, 0x7F);
        assert!(parser.feed(6, 5).is_none());
    }

    #[test]
    fn nrpn_data_entry() {
        let mut parser = RpnNrpnParser::new();
        parser.feed(99, 1);
        parser.feed(98, 2);
        let update = parser.feed(38, 3).expect("data entry lsb");
        assert_eq!(update.kind, ParamKind::Assignable);
        assert_eq!(update.bank, 1);
        assert_eq!(update.index, 2);
    }
}
