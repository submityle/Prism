//! Per-channel expression state tracking.
//!
//! A [`ChannelState`] folds the stream of channel-wide messages for one channel
//! into a compact snapshot: the latest pitch bend, channel pressure, every
//! control-change value, the selected program and bank, and the pitch-bend
//! range in semitones. It understands both protocols: MIDI 2.0
//! registered-controller messages set the pitch-bend range directly, while
//! MIDI 1.0 control changes are routed through an [`RpnNrpnParser`] and a
//! [`HighResCc`] tracker so the same range and high-resolution controller
//! values are recovered from a 7-bit stream. All values are stored at the
//! uniform 32-bit width.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the per-channel tracking of design section 52. Consumes
//! [`crate::ump::message::ChannelVoice`] values and feeds the channel pitch
//! bend and pressure into [`crate::mapping`].

use crate::expression::controller::{HighResCc, ParamKind, RpnNrpnParser};
use crate::ump::message::{ChannelVoice, PITCH_BEND_CENTER_32};
use crate::ump::scaling::scale_down;
use prism_audio_core::math::Sample;

/// The default pitch-bend range in semitones (plus or minus two semitones).
pub const DEFAULT_PITCH_BEND_RANGE: Sample = 2.0;

/// Compact per-channel expression state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelState {
    pitch_bend: u32,
    pressure: u32,
    #[cfg_attr(feature = "serialize", serde(with = "cc_serde"))]
    controllers: [u32; 128],
    program: u8,
    bank: u16,
    pitch_bend_range: Sample,
    high_res: HighResCc,
    rpn: RpnNrpnParser,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelState {
    /// Creates a channel in its reset state: pitch bend centred, no pressure,
    /// all controllers zero, program zero, and the default pitch-bend range.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pitch_bend: PITCH_BEND_CENTER_32,
            pressure: 0,
            controllers: [0; 128],
            program: 0,
            bank: 0,
            pitch_bend_range: DEFAULT_PITCH_BEND_RANGE,
            high_res: HighResCc::new(),
            rpn: RpnNrpnParser::new(),
        }
    }

    /// Applies a channel voice message, updating the tracked state. Per-note
    /// messages are ignored here because they are handled by
    /// [`crate::expression::per_note`].
    pub fn apply(&mut self, message: &ChannelVoice) {
        match *message {
            ChannelVoice::PitchBend { bend } => self.pitch_bend = bend,
            ChannelVoice::ChannelPressure { pressure } => self.pressure = pressure,
            ChannelVoice::ControlChange { index, value } => self.apply_cc(index, value),
            ChannelVoice::RegisteredController { bank, index, value } => {
                self.apply_registered(bank, index, value);
            }
            ChannelVoice::ProgramChange { program, bank } => {
                self.program = program;
                if let Some(bank) = bank {
                    self.bank = bank;
                }
            }
            _ => {}
        }
    }

    fn apply_cc(&mut self, index: u8, value: u32) {
        let slot = (index & 0x7F) as usize;
        self.controllers[slot] = value;
        // Recover the 7-bit value to drive the MIDI 1.0 RPN/NRPN and high-res
        // state machines; Min-Center-Max up-scaling keeps the top 7 bits equal
        // to the original, so down-scaling is exact.
        let value7 = scale_down(value, 32, 7) as u8;
        if let Some(update) = self.rpn.feed(index & 0x7F, value7)
            && update.kind == ParamKind::Registered
        {
            self.apply_registered(update.bank, update.index, update.value);
        }
        self.high_res.feed(index & 0x7F, value7);
    }

    fn apply_registered(&mut self, bank: u8, index: u8, value: u32) {
        // RPN 0,0 is pitch bend sensitivity: MSB semitones, LSB cents.
        if bank == 0 && index == 0 {
            let value14 = scale_down(value, 32, 14);
            let semitones = (value14 >> 7) as Sample;
            let cents = (value14 & 0x7F) as Sample;
            self.pitch_bend_range = semitones + cents / 100.0;
        }
    }

    /// Returns the raw 32-bit channel pitch bend (centred at `0x8000_0000`).
    #[must_use]
    pub const fn pitch_bend_raw(&self) -> u32 {
        self.pitch_bend
    }

    /// Returns the raw 32-bit channel pressure.
    #[must_use]
    pub const fn pressure_raw(&self) -> u32 {
        self.pressure
    }

    /// Returns the stored 32-bit value for a control-change index.
    #[must_use]
    pub fn controller(&self, index: u8) -> u32 {
        self.controllers[(index & 0x7F) as usize]
    }

    /// Returns the combined high-resolution value for a 14-bit controller pair
    /// (logical index `0`-`31`).
    #[must_use]
    pub fn high_res_controller(&self, logical_index: u8) -> u32 {
        self.high_res.value(logical_index)
    }

    /// Returns the selected program number.
    #[must_use]
    pub const fn program(&self) -> u8 {
        self.program
    }

    /// Returns the selected 14-bit bank.
    #[must_use]
    pub const fn bank(&self) -> u16 {
        self.bank
    }

    /// Returns the current pitch-bend range in semitones.
    #[must_use]
    pub const fn pitch_bend_range(&self) -> Sample {
        self.pitch_bend_range
    }

    /// Returns the current channel pitch bend expressed in semitones, combining
    /// the raw bend with the configured range.
    #[must_use]
    pub fn pitch_bend_semitones(&self) -> Sample {
        normalized_bend(self.pitch_bend) * self.pitch_bend_range
    }
}

/// Converts a 32-bit pitch bend (centred at `0x8000_0000`) into a bipolar
/// `[-1, 1]` fraction. The centre maps to exactly `0.0`.
#[must_use]
pub fn normalized_bend(bend: u32) -> Sample {
    let signed = i64::from(bend) - i64::from(PITCH_BEND_CENTER_32);
    (signed as Sample) / (PITCH_BEND_CENTER_32 as Sample)
}

/// Serializes and deserializes the 128-entry controller table as a sequence,
/// since serde does not implement its array traits for arrays this large.
#[cfg(feature = "serialize")]
mod cc_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    pub(super) fn serialize<S: Serializer>(
        value: &[u32; 128],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(value.iter())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<[u32; 128], D::Error> {
        let items = Vec::<u32>::deserialize(deserializer)?;
        if items.len() != 128 {
            return Err(serde::de::Error::invalid_length(
                items.len(),
                &"128 controller values",
            ));
        }
        let mut array = [0u32; 128];
        for (slot, value) in array.iter_mut().zip(items) {
            *slot = value;
        }
        Ok(array)
    }
}


#[cfg(test)]
mod tests {
    use super::{ChannelState, DEFAULT_PITCH_BEND_RANGE, normalized_bend};
    use crate::ump::message::{ChannelVoice, PITCH_BEND_CENTER_32};

    const EPS: f32 = 1.0e-5;

    #[test]
    fn center_bend_is_zero() {
        assert!(normalized_bend(PITCH_BEND_CENTER_32).abs() < EPS);
    }

    #[test]
    fn default_range_pitch_bend() {
        let mut state = ChannelState::new();
        assert!((state.pitch_bend_range() - DEFAULT_PITCH_BEND_RANGE).abs() < EPS);
        // Full positive bend should be +2 semitones at the default range.
        state.apply(&ChannelVoice::PitchBend { bend: 0xFFFF_FFFF });
        assert!((state.pitch_bend_semitones() - 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn rpn_sets_pitch_bend_range() {
        let mut state = ChannelState::new();
        // Select RPN 0,0 then enter 12 semitones.
        state.apply(&ChannelVoice::ControlChange {
            index: 101,
            value: 0,
        });
        state.apply(&ChannelVoice::ControlChange {
            index: 100,
            value: 0,
        });
        // Data entry MSB (CC 6) at 7-bit value 12 up-scaled to 32 bits.
        let value = crate::ump::scaling::scale_up(12, 7, 32);
        state.apply(&ChannelVoice::ControlChange { index: 6, value });
        assert!((state.pitch_bend_range() - 12.0).abs() < EPS);
    }

    #[test]
    fn program_and_bank_change() {
        let mut state = ChannelState::new();
        state.apply(&ChannelVoice::ProgramChange {
            program: 42,
            bank: Some(300),
        });
        assert_eq!(state.program(), 42);
        assert_eq!(state.bank(), 300);
    }

    #[test]
    fn controller_is_stored() {
        let mut state = ChannelState::new();
        state.apply(&ChannelVoice::ControlChange {
            index: 74,
            value: 0x1234_5678,
        });
        assert_eq!(state.controller(74), 0x1234_5678);
    }
}
