//! Incremental Universal MIDI Packet decoder.
//!
//! A UMP stream is a run of big-endian 32-bit words in which the first word of
//! each packet fixes how many words follow. [`UmpDecoder`] assembles those
//! multi-word packets one word (or one byte) at a time and emits a decoded
//! [`MidiMessage`] once a full packet has arrived. It never loses alignment:
//! even for message types this crate does not surface (data, flex data, stream)
//! it consumes exactly the right number of words using the standard word-count
//! table, returning `None` for the consumed-but-not-surfaced packet. This makes
//! it safe to feed an arbitrary device byte stream off the real-time thread.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the stream-assembly half of design section 52 (MIDI 2.0 UMP
//! events). Wraps [`crate::ump::message`] decoders and the
//! [`crate::ump::word`] word-count table.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::ump::message::{
    MidiMessage, decode_midi1, decode_midi2, decode_system, decode_utility,
};
use crate::ump::word::{MessageType, UmpWord};

/// An incremental decoder that assembles UMP words into [`MidiMessage`]s.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UmpDecoder {
    words: [u32; 4],
    have_words: usize,
    needed_words: usize,
    bytes: [u8; 4],
    have_bytes: usize,
}

impl Default for UmpDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl UmpDecoder {
    /// Creates an empty decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: [0; 4],
            have_words: 0,
            needed_words: 0,
            bytes: [0; 4],
            have_bytes: 0,
        }
    }

    /// Discards any partially assembled packet and byte fragment.
    pub fn reset(&mut self) {
        self.have_words = 0;
        self.needed_words = 0;
        self.have_bytes = 0;
    }

    /// Feeds one 32-bit word. Returns a decoded message once the current packet
    /// is complete, or `None` while more words are required or when the
    /// completed packet is not surfaced by this crate.
    pub fn push_word(&mut self, word: u32) -> Option<MidiMessage> {
        if self.have_words == 0 {
            let message_type = UmpWord::new(word).message_type();
            self.needed_words = message_type.word_count();
        }
        self.words[self.have_words] = word;
        self.have_words += 1;
        if self.have_words < self.needed_words {
            return None;
        }
        let decoded = self.decode_packet();
        self.have_words = 0;
        self.needed_words = 0;
        decoded
    }

    /// Feeds one byte of a big-endian word stream. Returns a decoded message
    /// when the byte completes a packet.
    pub fn push_byte(&mut self, byte: u8) -> Option<MidiMessage> {
        self.bytes[self.have_bytes] = byte;
        self.have_bytes += 1;
        if self.have_bytes < 4 {
            return None;
        }
        self.have_bytes = 0;
        let word = u32::from_be_bytes(self.bytes);
        self.push_word(word)
    }

    /// Decodes a slice of words, appending every surfaced message to `out`.
    pub fn push_words(&mut self, words: &[u32], out: &mut Vec<MidiMessage>) {
        for &word in words {
            if let Some(message) = self.push_word(word) {
                out.push(message);
            }
        }
    }

    /// Decodes a big-endian byte stream, appending every surfaced message to
    /// `out`. Trailing bytes that do not complete a word remain buffered for
    /// the next call.
    pub fn push_bytes(&mut self, bytes: &[u8], out: &mut Vec<MidiMessage>) {
        for &byte in bytes {
            if let Some(message) = self.push_byte(byte) {
                out.push(message);
            }
        }
    }

    fn decode_packet(&self) -> Option<MidiMessage> {
        let word0 = UmpWord::new(self.words[0]);
        match word0.message_type() {
            MessageType::Utility => decode_utility(word0),
            MessageType::System => decode_system(word0),
            MessageType::Midi1ChannelVoice => decode_midi1(word0),
            MessageType::Midi2ChannelVoice => decode_midi2(word0, self.words[1]),
            MessageType::Data64
            | MessageType::Data128
            | MessageType::FlexData
            | MessageType::UmpStream
            | MessageType::Reserved(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use super::UmpDecoder;
    use crate::ump::message::{ChannelVoice, MidiMessage};

    #[test]
    fn single_word_message_decodes_immediately() {
        let mut decoder = UmpDecoder::new();
        let msg = decoder.push_word(0x2090_3C7F).expect("note on");
        assert!(matches!(
            msg,
            MidiMessage::ChannelVoice {
                message: ChannelVoice::NoteOn { note: 60, .. },
                ..
            }
        ));
    }

    #[test]
    fn two_word_message_waits_for_second_word() {
        let mut decoder = UmpDecoder::new();
        assert!(decoder.push_word(0x4090_3C00).is_none());
        let msg = decoder.push_word(0xFFFF_0000).expect("note on");
        match msg {
            MidiMessage::ChannelVoice {
                message: ChannelVoice::NoteOn { note, velocity, .. },
                ..
            } => {
                assert_eq!(note, 60);
                assert_eq!(velocity, 0xFFFF);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn data_message_is_consumed_without_desync() {
        let mut decoder = UmpDecoder::new();
        // A 64-bit data (SysEx7) packet occupies two words and is not surfaced.
        assert!(decoder.push_word(0x3000_0000).is_none());
        assert!(decoder.push_word(0x0000_0000).is_none());
        // The following note on must still decode, proving alignment held.
        assert!(decoder.push_word(0x2090_3C7F).is_some());
    }

    #[test]
    fn byte_stream_matches_word_stream() {
        let words = [0x4090_3C00u32, 0xFFFF_0000, 0x2080_3C40];
        let mut from_words = Vec::new();
        let mut decoder_words = UmpDecoder::new();
        decoder_words.push_words(&words, &mut from_words);

        let mut bytes = Vec::new();
        for word in words {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        let mut from_bytes = Vec::new();
        let mut decoder_bytes = UmpDecoder::new();
        decoder_bytes.push_bytes(&bytes, &mut from_bytes);

        assert_eq!(from_words, from_bytes);
        assert_eq!(from_words.len(), 2);
    }

    #[test]
    fn reset_clears_partial_packet() {
        let mut decoder = UmpDecoder::new();
        assert!(decoder.push_word(0x4090_3C00).is_none());
        decoder.reset();
        // After reset a fresh single-word message decodes cleanly.
        assert!(decoder.push_word(0x2090_3C7F).is_some());
    }
}
