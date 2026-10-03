//! High-level MIDI messages decoded from Universal MIDI Packet words.
//!
//! This module lifts the raw [`crate::ump::word::UmpWord`] bit fields into a
//! strongly typed [`MidiMessage`] tree. MIDI 1.0 and MIDI 2.0 channel voice
//! messages are decoded into the same [`ChannelVoice`] enum: MIDI 1.0 values
//! are up-scaled to the MIDI 2.0 widths (16-bit velocity, 32-bit controllers
//! and pressure, 32-bit pitch bend) using [`crate::ump::scaling`], so every
//! downstream consumer sees one uniform, full-resolution representation. A MIDI
//! 1.0 note-on with velocity zero is normalised to a note-off, matching the
//! running-status convention; MIDI 2.0 note-ons keep a zero velocity because
//! the newer protocol does not overload it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The field layouts
//! are taken from the publicly published MIDI 1.0 and MIDI 2.0 UMP
//! specifications.
//!
//! # Relationship
//! Implements the decoding half of design section 52 (MIDI 2.0 UMP events).
//! Consumed by [`crate::ump::decoder`] and, through it, by the expression state
//! trackers in [`crate::expression`].

use crate::ump::scaling::scale_up;
use crate::ump::word::UmpWord;

/// A fully decoded, high-level MIDI message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MidiMessage {
    /// A utility message (clocking and timestamps); carries no group.
    Utility(UtilityMessage),
    /// A system real-time or system common message on a group.
    System {
        /// The UMP group (`0`-`15`).
        group: u8,
        /// The decoded system message.
        message: SystemMessage,
    },
    /// A channel voice message on a group and channel.
    ChannelVoice {
        /// The UMP group (`0`-`15`).
        group: u8,
        /// The channel within the group (`0`-`15`).
        channel: u8,
        /// The decoded channel voice message.
        message: ChannelVoice,
    },
}

/// A UMP utility message (message type `0x0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum UtilityMessage {
    /// No operation; a placeholder word that advances no state.
    NoOp,
    /// Jitter-reduction clock: a 16-bit sender clock reading.
    JrClock {
        /// The 16-bit clock reading.
        clock: u16,
    },
    /// Jitter-reduction timestamp: a 16-bit send time in units of 1/31250 s.
    JrTimestamp {
        /// The 16-bit timestamp.
        timestamp: u16,
    },
    /// Delta clockstamp ticks per quarter note (tempo resolution).
    DeltaClockstampTicksPerQuarterNote {
        /// Ticks per quarter note.
        ticks: u16,
    },
    /// Delta clockstamp: a 20-bit tick delta since the previous message.
    DeltaClockstamp {
        /// The 20-bit tick delta.
        ticks: u32,
    },
}

/// A UMP system real-time or system common message (message type `0x1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SystemMessage {
    /// MIDI time code quarter frame; carries the 7-bit nibble payload.
    MidiTimeCode {
        /// The quarter-frame data byte.
        data: u8,
    },
    /// Song position pointer: a 14-bit beat count since the start of the song.
    SongPositionPointer {
        /// The 14-bit position.
        position: u16,
    },
    /// Song select: a 7-bit song index.
    SongSelect {
        /// The 7-bit song number.
        song: u8,
    },
    /// Tune request.
    TuneRequest,
    /// Timing clock (24 pulses per quarter note).
    TimingClock,
    /// Start playing from the start of the song.
    Start,
    /// Continue playing from the current position.
    Continue,
    /// Stop playing.
    Stop,
    /// Active sensing keep-alive.
    ActiveSensing,
    /// System reset.
    Reset,
}

/// The per-note attribute carried by MIDI 2.0 note-on and note-off messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum NoteAttribute {
    /// No attribute (type `0x00`).
    None,
    /// Manufacturer-specific attribute (type `0x01`).
    ManufacturerSpecific {
        /// The 16-bit attribute payload.
        data: u16,
    },
    /// Profile-specific attribute (type `0x02`).
    ProfileSpecific {
        /// The 16-bit attribute payload.
        data: u16,
    },
    /// Pitch 7.9 attribute (type `0x03`): a Q7.9 fixed-point pitch in
    /// semitones (7 integer bits, 9 fractional bits).
    Pitch7_9 {
        /// The raw Q7.9 pitch value.
        pitch: u16,
    },
    /// Any other attribute type, retaining its raw type and data.
    Unknown {
        /// The raw attribute type byte.
        attribute_type: u8,
        /// The 16-bit attribute payload.
        data: u16,
    },
}

impl NoteAttribute {
    /// Decodes an attribute from its type byte and 16-bit data field.
    #[must_use]
    pub const fn decode(attribute_type: u8, data: u16) -> Self {
        match attribute_type {
            0x00 => Self::None,
            0x01 => Self::ManufacturerSpecific { data },
            0x02 => Self::ProfileSpecific { data },
            0x03 => Self::Pitch7_9 { pitch: data },
            other => Self::Unknown {
                attribute_type: other,
                data,
            },
        }
    }
}

/// A channel voice message in the uniform full-resolution representation.
///
/// Velocity is 16-bit, controllers and pressure are 32-bit, and pitch bend is
/// 32-bit centred at `0x8000_0000`. MIDI 1.0 messages are up-scaled into these
/// widths on decode so a consumer never has to branch on the source protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelVoice {
    /// Note off.
    NoteOff {
        /// The note number (`0`-`127`).
        note: u8,
        /// The 16-bit release velocity.
        velocity: u16,
        /// The per-note attribute.
        attribute: NoteAttribute,
    },
    /// Note on.
    NoteOn {
        /// The note number (`0`-`127`).
        note: u8,
        /// The 16-bit attack velocity.
        velocity: u16,
        /// The per-note attribute.
        attribute: NoteAttribute,
    },
    /// Polyphonic (per-note) key pressure.
    PolyPressure {
        /// The note number (`0`-`127`).
        note: u8,
        /// The 32-bit pressure.
        pressure: u32,
    },
    /// Registered per-note controller.
    RegisteredPerNoteController {
        /// The note number (`0`-`127`).
        note: u8,
        /// The controller index.
        index: u8,
        /// The 32-bit value.
        value: u32,
    },
    /// Assignable per-note controller.
    AssignablePerNoteController {
        /// The note number (`0`-`127`).
        note: u8,
        /// The controller index.
        index: u8,
        /// The 32-bit value.
        value: u32,
    },
    /// Per-note pitch bend, centred at `0x8000_0000`.
    PerNotePitchBend {
        /// The note number (`0`-`127`).
        note: u8,
        /// The 32-bit pitch bend.
        bend: u32,
    },
    /// Per-note management (detach and/or reset per-note controllers).
    PerNoteManagement {
        /// The note number (`0`-`127`).
        note: u8,
        /// Detach per-note controllers from the note's lifetime.
        detach: bool,
        /// Reset per-note controllers to their default values.
        reset: bool,
    },
    /// Control change (channel-level controller).
    ControlChange {
        /// The controller index (`0`-`127`).
        index: u8,
        /// The 32-bit value.
        value: u32,
    },
    /// Registered controller (RPN).
    RegisteredController {
        /// The RPN bank (MSB).
        bank: u8,
        /// The RPN index (LSB).
        index: u8,
        /// The 32-bit value.
        value: u32,
    },
    /// Assignable controller (NRPN).
    AssignableController {
        /// The NRPN bank (MSB).
        bank: u8,
        /// The NRPN index (LSB).
        index: u8,
        /// The 32-bit value.
        value: u32,
    },
    /// Relative registered controller (signed delta).
    RelativeRegisteredController {
        /// The RPN bank (MSB).
        bank: u8,
        /// The RPN index (LSB).
        index: u8,
        /// The signed 32-bit delta.
        value: i32,
    },
    /// Relative assignable controller (signed delta).
    RelativeAssignableController {
        /// The NRPN bank (MSB).
        bank: u8,
        /// The NRPN index (LSB).
        index: u8,
        /// The signed 32-bit delta.
        value: i32,
    },
    /// Program change, with an optional 14-bit bank select.
    ProgramChange {
        /// The 7-bit program number.
        program: u8,
        /// The 14-bit bank, present only when the bank-valid flag is set.
        bank: Option<u16>,
    },
    /// Channel pressure (aftertouch).
    ChannelPressure {
        /// The 32-bit pressure.
        pressure: u32,
    },
    /// Channel pitch bend, centred at `0x8000_0000`.
    PitchBend {
        /// The 32-bit pitch bend.
        bend: u32,
    },
}

impl ChannelVoice {
    /// Returns the note number for a per-note message, or `None` for a
    /// channel-wide message.
    #[must_use]
    pub const fn note(&self) -> Option<u8> {
        match *self {
            Self::NoteOff { note, .. }
            | Self::NoteOn { note, .. }
            | Self::PolyPressure { note, .. }
            | Self::RegisteredPerNoteController { note, .. }
            | Self::AssignablePerNoteController { note, .. }
            | Self::PerNotePitchBend { note, .. }
            | Self::PerNoteManagement { note, .. } => Some(note),
            _ => None,
        }
    }
}

/// The 32-bit value representing the exact centre of a bipolar pitch bend.
pub const PITCH_BEND_CENTER_32: u32 = 0x8000_0000;

/// Decodes a single-word utility message, or `None` for an unrecognised status.
#[must_use]
pub fn decode_utility(word: UmpWord) -> Option<MidiMessage> {
    let message = match word.status_nibble() {
        0x0 => UtilityMessage::NoOp,
        0x1 => UtilityMessage::JrClock {
            clock: word.low_u16(),
        },
        0x2 => UtilityMessage::JrTimestamp {
            timestamp: word.low_u16(),
        },
        0x3 => UtilityMessage::DeltaClockstampTicksPerQuarterNote {
            ticks: word.low_u16(),
        },
        0x4 => UtilityMessage::DeltaClockstamp {
            ticks: word.raw() & 0x000F_FFFF,
        },
        _ => return None,
    };
    Some(MidiMessage::Utility(message))
}

/// Decodes a single-word system message, or `None` for an unrecognised status.
#[must_use]
pub fn decode_system(word: UmpWord) -> Option<MidiMessage> {
    let message = match word.status_byte() {
        0xF1 => SystemMessage::MidiTimeCode {
            data: word.byte3() & 0x7F,
        },
        0xF2 => SystemMessage::SongPositionPointer {
            position: (u16::from(word.byte3() & 0x7F) << 7) | u16::from(word.byte2() & 0x7F),
        },
        0xF3 => SystemMessage::SongSelect {
            song: word.byte3() & 0x7F,
        },
        0xF6 => SystemMessage::TuneRequest,
        0xF8 => SystemMessage::TimingClock,
        0xFA => SystemMessage::Start,
        0xFB => SystemMessage::Continue,
        0xFC => SystemMessage::Stop,
        0xFE => SystemMessage::ActiveSensing,
        0xFF => SystemMessage::Reset,
        _ => return None,
    };
    Some(MidiMessage::System {
        group: word.group(),
        message,
    })
}

/// Decodes a single-word MIDI 1.0 channel voice message, up-scaling its values
/// to the uniform full-resolution representation. Returns `None` for an
/// unrecognised opcode.
#[must_use]
pub fn decode_midi1(word: UmpWord) -> Option<MidiMessage> {
    let note = word.byte2() & 0x7F;
    let data1 = word.byte2() & 0x7F;
    let data2 = word.byte3() & 0x7F;
    let message = match word.status_nibble() {
        0x8 => ChannelVoice::NoteOff {
            note,
            velocity: scale_up(u32::from(data2), 7, 16) as u16,
            attribute: NoteAttribute::None,
        },
        0x9 => {
            if data2 == 0 {
                // A MIDI 1.0 note-on with velocity zero is a note-off.
                ChannelVoice::NoteOff {
                    note,
                    velocity: 0,
                    attribute: NoteAttribute::None,
                }
            } else {
                ChannelVoice::NoteOn {
                    note,
                    velocity: scale_up(u32::from(data2), 7, 16) as u16,
                    attribute: NoteAttribute::None,
                }
            }
        }
        0xA => ChannelVoice::PolyPressure {
            note,
            pressure: scale_up(u32::from(data2), 7, 32),
        },
        0xB => ChannelVoice::ControlChange {
            index: data1,
            value: scale_up(u32::from(data2), 7, 32),
        },
        0xC => ChannelVoice::ProgramChange {
            program: data1,
            bank: None,
        },
        0xD => ChannelVoice::ChannelPressure {
            pressure: scale_up(u32::from(data1), 7, 32),
        },
        0xE => {
            let value14 = u32::from(data2) << 7 | u32::from(data1);
            ChannelVoice::PitchBend {
                bend: scale_up(value14, 14, 32),
            }
        }
        _ => return None,
    };
    Some(MidiMessage::ChannelVoice {
        group: word.group(),
        channel: word.channel(),
        message,
    })
}

/// Decodes a two-word MIDI 2.0 channel voice message. Returns `None` for an
/// unrecognised opcode.
#[must_use]
pub fn decode_midi2(word0: UmpWord, word1: u32) -> Option<MidiMessage> {
    let note = word0.byte2() & 0x7F;
    let message = match word0.status_nibble() {
        0x0 => ChannelVoice::RegisteredPerNoteController {
            note,
            index: word0.byte3(),
            value: word1,
        },
        0x1 => ChannelVoice::AssignablePerNoteController {
            note,
            index: word0.byte3(),
            value: word1,
        },
        0x2 => ChannelVoice::RegisteredController {
            bank: word0.byte2() & 0x7F,
            index: word0.byte3() & 0x7F,
            value: word1,
        },
        0x3 => ChannelVoice::AssignableController {
            bank: word0.byte2() & 0x7F,
            index: word0.byte3() & 0x7F,
            value: word1,
        },
        0x4 => ChannelVoice::RelativeRegisteredController {
            bank: word0.byte2() & 0x7F,
            index: word0.byte3() & 0x7F,
            value: word1 as i32,
        },
        0x5 => ChannelVoice::RelativeAssignableController {
            bank: word0.byte2() & 0x7F,
            index: word0.byte3() & 0x7F,
            value: word1 as i32,
        },
        0x6 => ChannelVoice::PerNotePitchBend { note, bend: word1 },
        0x8 => ChannelVoice::NoteOff {
            note,
            velocity: (word1 >> 16) as u16,
            attribute: NoteAttribute::decode(word0.byte3(), (word1 & 0xFFFF) as u16),
        },
        0x9 => ChannelVoice::NoteOn {
            note,
            velocity: (word1 >> 16) as u16,
            attribute: NoteAttribute::decode(word0.byte3(), (word1 & 0xFFFF) as u16),
        },
        0xA => ChannelVoice::PolyPressure {
            note,
            pressure: word1,
        },
        0xB => ChannelVoice::ControlChange {
            index: word0.byte2() & 0x7F,
            value: word1,
        },
        0xC => {
            let bank = if word0.byte3() & 0x01 != 0 {
                Some((u16::from((word1 >> 8) as u8 & 0x7F) << 7) | u16::from(word1 as u8 & 0x7F))
            } else {
                None
            };
            ChannelVoice::ProgramChange {
                program: (word1 >> 24) as u8 & 0x7F,
                bank,
            }
        }
        0xD => ChannelVoice::ChannelPressure { pressure: word1 },
        0xE => ChannelVoice::PitchBend { bend: word1 },
        0xF => ChannelVoice::PerNoteManagement {
            note,
            detach: word0.byte3() & 0x02 != 0,
            reset: word0.byte3() & 0x01 != 0,
        },
        _ => return None,
    };
    Some(MidiMessage::ChannelVoice {
        group: word0.group(),
        channel: word0.channel(),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelVoice, MidiMessage, NoteAttribute, PITCH_BEND_CENTER_32, SystemMessage,
        UtilityMessage, decode_midi1, decode_midi2, decode_system, decode_utility,
    };
    use crate::ump::word::UmpWord;

    #[test]
    fn midi1_note_on_upscales_velocity() {
        // Group 0, channel 0, note 60, velocity 127.
        let word = UmpWord::new(0x2090_3C7F);
        let msg = decode_midi1(word).expect("decoded");
        match msg {
            MidiMessage::ChannelVoice {
                channel,
                message: ChannelVoice::NoteOn { note, velocity, .. },
                ..
            } => {
                assert_eq!(channel, 0);
                assert_eq!(note, 60);
                assert_eq!(velocity, 0xFFFF);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi1_note_on_zero_velocity_is_note_off() {
        let word = UmpWord::new(0x2090_3C00);
        match decode_midi1(word).expect("decoded") {
            MidiMessage::ChannelVoice {
                message: ChannelVoice::NoteOff { note, .. },
                ..
            } => assert_eq!(note, 60),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi1_pitch_bend_center() {
        // LSB 0x00, MSB 0x40 -> 14-bit center 0x2000.
        let word = UmpWord::new(0x20E0_0040);
        match decode_midi1(word).expect("decoded") {
            MidiMessage::ChannelVoice {
                message: ChannelVoice::PitchBend { bend },
                ..
            } => assert_eq!(bend, PITCH_BEND_CENTER_32),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi2_note_on_keeps_zero_velocity() {
        // Group 0, channel 1, opcode 0x9, note 64.
        let word0 = UmpWord::new(0x4091_4000);
        match decode_midi2(word0, 0x0000_1234).expect("decoded") {
            MidiMessage::ChannelVoice {
                channel,
                message: ChannelVoice::NoteOn { note, velocity, .. },
                ..
            } => {
                assert_eq!(channel, 1);
                assert_eq!(note, 64);
                assert_eq!(velocity, 0);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi2_note_on_pitch_attribute() {
        let word0 = UmpWord::new(0x4090_3C03);
        match decode_midi2(word0, 0xABCD_1000).expect("decoded") {
            MidiMessage::ChannelVoice {
                message:
                    ChannelVoice::NoteOn {
                        velocity,
                        attribute: NoteAttribute::Pitch7_9 { pitch },
                        ..
                    },
                ..
            } => {
                assert_eq!(velocity, 0xABCD);
                assert_eq!(pitch, 0x1000);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi2_per_note_pitch_bend() {
        let word0 = UmpWord::new(0x4060_3C00);
        match decode_midi2(word0, PITCH_BEND_CENTER_32).expect("decoded") {
            MidiMessage::ChannelVoice {
                message: ChannelVoice::PerNotePitchBend { note, bend },
                ..
            } => {
                assert_eq!(note, 60);
                assert_eq!(bend, PITCH_BEND_CENTER_32);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn midi2_program_change_with_bank() {
        // Opcode 0xC, option bit0 set (bank valid).
        let word0 = UmpWord::new(0x40C0_0001);
        // program 7, bank msb 2, bank lsb 3.
        match decode_midi2(word0, 0x0700_0203).expect("decoded") {
            MidiMessage::ChannelVoice {
                message: ChannelVoice::ProgramChange { program, bank },
                ..
            } => {
                assert_eq!(program, 7);
                assert_eq!(bank, Some(2 << 7 | 3));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn utility_jr_timestamp() {
        let word = UmpWord::new(0x0020_1234);
        match decode_utility(word).expect("decoded") {
            MidiMessage::Utility(UtilityMessage::JrTimestamp { timestamp }) => {
                assert_eq!(timestamp, 0x1234);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn system_start_and_clock() {
        match decode_system(UmpWord::new(0x11FA_0000)).expect("decoded") {
            MidiMessage::System {
                group,
                message: SystemMessage::Start,
            } => assert_eq!(group, 1),
            other => panic!("unexpected {other:?}"),
        }
        match decode_system(UmpWord::new(0x10F8_0000)).expect("decoded") {
            MidiMessage::System {
                message: SystemMessage::TimingClock,
                ..
            } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unknown_statuses_decode_to_none() {
        assert!(decode_utility(UmpWord::new(0x00F0_0000)).is_none());
        assert!(decode_system(UmpWord::new(0x10F0_0000)).is_none());
        assert!(decode_midi1(UmpWord::new(0x2000_0000)).is_none());
        assert!(decode_midi2(UmpWord::new(0x4070_0000), 0).is_none());
    }
}
