//! The 32-bit Universal MIDI Packet word and message-type classification.
//!
//! Every UMP message is a sequence of one to four big-endian 32-bit words. The
//! top four bits of the first word hold the Message Type, which fixes both the
//! number of words in the packet and the meaning of the remaining fields. This
//! module wraps a single word with typed accessors for the common sub-fields
//! (group, status nibble, channel, data bytes) and provides the standard
//! word-count table so the decoder can assemble multi-word packets without ever
//! losing alignment on an unrecognised message type.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The word layout and
//! message-type word counts are taken from the publicly published MIDI 2.0 UMP
//! specification.
//!
//! # Relationship
//! Supports design section 52 (MIDI 2.0 UMP). The byte-level primitive beneath
//! [`crate::ump::message`] and [`crate::ump::decoder`].

/// The Message Type carried in the top four bits of a UMP word's first word.
///
/// The variant also determines how many 32-bit words the whole packet occupies
/// (see [`MessageType::word_count`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MessageType {
    /// Type `0x0`: utility messages (NOOP, JR clock, JR timestamp).
    Utility,
    /// Type `0x1`: system real-time and system common messages.
    System,
    /// Type `0x2`: MIDI 1.0 channel voice messages.
    Midi1ChannelVoice,
    /// Type `0x3`: 64-bit data messages (`SysEx7`).
    Data64,
    /// Type `0x4`: MIDI 2.0 channel voice messages.
    Midi2ChannelVoice,
    /// Type `0x5`: 128-bit data messages (`SysEx8`, mixed data set).
    Data128,
    /// Type `0xD`: flex data messages (128-bit).
    FlexData,
    /// Type `0xF`: UMP stream messages (128-bit).
    UmpStream,
    /// Any reserved message type, retaining its raw nibble so the decoder can
    /// still consume the correct number of words.
    Reserved(u8),
}

impl MessageType {
    /// Classifies the message type nibble (only the low four bits are used).
    #[must_use]
    pub const fn from_nibble(nibble: u8) -> Self {
        match nibble & 0x0F {
            0x0 => Self::Utility,
            0x1 => Self::System,
            0x2 => Self::Midi1ChannelVoice,
            0x3 => Self::Data64,
            0x4 => Self::Midi2ChannelVoice,
            0x5 => Self::Data128,
            0xD => Self::FlexData,
            0xF => Self::UmpStream,
            other => Self::Reserved(other),
        }
    }

    /// Returns the raw four-bit message type nibble.
    #[must_use]
    pub const fn nibble(self) -> u8 {
        match self {
            Self::Utility => 0x0,
            Self::System => 0x1,
            Self::Midi1ChannelVoice => 0x2,
            Self::Data64 => 0x3,
            Self::Midi2ChannelVoice => 0x4,
            Self::Data128 => 0x5,
            Self::FlexData => 0xD,
            Self::UmpStream => 0xF,
            Self::Reserved(other) => other & 0x0F,
        }
    }

    /// Returns the number of 32-bit words the packet occupies.
    ///
    /// The counts follow the fixed MIDI 2.0 UMP table: types `0x0`-`0x2` are one
    /// word, `0x3`-`0x4` two words, `0x5` four words, and the reserved ranges
    /// use the standard sizes (`0x6`-`0x7` one word, `0x8`-`0xA` two words,
    /// `0xB`-`0xC` three words, `0xD`-`0xF` four words).
    #[must_use]
    pub const fn word_count(self) -> usize {
        match self.nibble() {
            0x0 | 0x1 | 0x2 | 0x6 | 0x7 => 1,
            0x3 | 0x4 | 0x8 | 0x9 | 0xA => 2,
            0xB | 0xC => 3,
            _ => 4,
        }
    }
}

/// A single 32-bit Universal MIDI Packet word with typed field accessors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UmpWord(u32);

impl UmpWord {
    /// Wraps a raw 32-bit word.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw 32-bit value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Returns the message type carried in bits 28-31.
    #[must_use]
    pub const fn message_type(self) -> MessageType {
        MessageType::from_nibble((self.0 >> 28) as u8)
    }

    /// Returns the group field in bits 24-27 (`0`-`15`).
    #[must_use]
    pub const fn group(self) -> u8 {
        ((self.0 >> 24) & 0x0F) as u8
    }

    /// Returns the status nibble in bits 20-23 (opcode for channel voice
    /// messages, status for utility messages).
    #[must_use]
    pub const fn status_nibble(self) -> u8 {
        ((self.0 >> 20) & 0x0F) as u8
    }

    /// Returns the channel field in bits 16-19 (`0`-`15`).
    #[must_use]
    pub const fn channel(self) -> u8 {
        ((self.0 >> 16) & 0x0F) as u8
    }

    /// Returns the full status byte in bits 16-23 (used by system messages).
    #[must_use]
    pub const fn status_byte(self) -> u8 {
        ((self.0 >> 16) & 0xFF) as u8
    }

    /// Returns the byte in bits 8-15 (first data byte / note / index).
    #[must_use]
    pub const fn byte2(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    /// Returns the byte in bits 0-7 (second data byte / controller index).
    #[must_use]
    pub const fn byte3(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    /// Returns the low 16 bits (utility message payload).
    #[must_use]
    pub const fn low_u16(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::{MessageType, UmpWord};

    #[test]
    fn message_type_round_trips() {
        for n in 0u8..16 {
            let mt = MessageType::from_nibble(n);
            assert_eq!(mt.nibble(), n);
        }
    }

    #[test]
    fn word_counts_match_spec() {
        assert_eq!(MessageType::Utility.word_count(), 1);
        assert_eq!(MessageType::System.word_count(), 1);
        assert_eq!(MessageType::Midi1ChannelVoice.word_count(), 1);
        assert_eq!(MessageType::Data64.word_count(), 2);
        assert_eq!(MessageType::Midi2ChannelVoice.word_count(), 2);
        assert_eq!(MessageType::Data128.word_count(), 4);
        assert_eq!(MessageType::FlexData.word_count(), 4);
        assert_eq!(MessageType::UmpStream.word_count(), 4);
        assert_eq!(MessageType::Reserved(0xB).word_count(), 3);
    }

    #[test]
    fn field_extraction() {
        // MIDI 2.0 note on, group 2, channel 5, note 60.
        let word = UmpWord::new(0x4295_3C00);
        assert_eq!(word.message_type(), MessageType::Midi2ChannelVoice);
        assert_eq!(word.group(), 2);
        assert_eq!(word.status_nibble(), 0x9);
        assert_eq!(word.channel(), 5);
        assert_eq!(word.byte2(), 0x3C);
    }
}
