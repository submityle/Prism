//! The BW64 `chna` chunk: the channel-to-track mapping table.
//!
//! The `chna` chunk binds each physical track in the WAV container to its ADM
//! `audioTrackUID`, channel/track-format reference, and pack-format reference.
//! It is a fixed-width binary table: a four-byte header (`numTracks`,
//! `numUIDs`) followed by forty-byte `audioID` records. This module serialises
//! and parses that table deterministically so a write-then-read round trip is
//! byte identical.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements the publicly published `chna` chunk layout of BW64
//! (ITU-R BS.2088) and ADM (ITU-R BS.2076), modelled as owned Rust data.
//!
//! # Relationship
//!
//! Carried inside the [`crate::adm::bw64`] container alongside the `axml`
//! metadata produced by [`crate::adm::xml`] from the [`crate::adm::model`]
//! graph. The `uid`, `track_ref`, and `pack_ref` fields correspond to the
//! `audioTrackUID`, `audioChannelFormatIDRef`, and `audioPackFormatIDRef`
//! identifiers of that graph.

use alloc::string::String;
use alloc::vec::Vec;

/// Width in bytes of the `audioTrackUID` field of a `chna` record.
pub const UID_LEN: usize = 12;
/// Width in bytes of the channel/track-format reference field.
pub const TRACK_REF_LEN: usize = 14;
/// Width in bytes of the pack-format reference field.
pub const PACK_REF_LEN: usize = 11;
/// Total width in bytes of one `audioID` record (including the pad byte).
pub const RECORD_LEN: usize = 2 + UID_LEN + TRACK_REF_LEN + PACK_REF_LEN + 1;
/// Width in bytes of the `chna` header (`numTracks`, `numUIDs`).
pub const HEADER_LEN: usize = 4;

/// A failure while parsing a `chna` chunk from bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChnaError {
    /// The byte slice was shorter than a valid record stream requires.
    Truncated,
    /// A text field contained a non-ASCII byte.
    NonAscii,
}

/// One `audioID` record: a physical track bound to ADM identifiers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioId {
    /// One-based physical track index inside the WAV container.
    pub track_index: u16,
    /// The `audioTrackUID` string (up to [`UID_LEN`] ASCII bytes).
    pub uid: String,
    /// The channel/track-format reference (up to [`TRACK_REF_LEN`] bytes).
    pub track_ref: String,
    /// The pack-format reference (up to [`PACK_REF_LEN`] bytes).
    pub pack_ref: String,
}

impl AudioId {
    /// Builds an `audioID` record from borrowed identifiers.
    #[must_use]
    pub fn new(track_index: u16, uid: &str, track_ref: &str, pack_ref: &str) -> Self {
        Self {
            track_index,
            uid: String::from(uid),
            track_ref: String::from(track_ref),
            pack_ref: String::from(pack_ref),
        }
    }
}

/// The complete `chna` chunk: a track count plus the `audioID` records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChnaChunk {
    /// The number of distinct physical tracks the records cover.
    pub num_tracks: u16,
    /// The `audioID` records, one per non-silent track reference.
    pub audio_ids: Vec<AudioId>,
}

impl ChnaChunk {
    /// An empty `chna` chunk.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a record and widens `num_tracks` to cover its track index.
    pub fn push(&mut self, record: AudioId) {
        if record.track_index > self.num_tracks {
            self.num_tracks = record.track_index;
        }
        self.audio_ids.push(record);
    }

    /// The exact serialised length in bytes of this chunk body.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        HEADER_LEN + self.audio_ids.len() * RECORD_LEN
    }

    /// Serialises the chunk body (without the enclosing RIFF chunk header) into
    /// a deterministic little-endian byte vector.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let num_uids = u16::try_from(self.audio_ids.len()).unwrap_or(u16::MAX);
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(&self.num_tracks.to_le_bytes());
        out.extend_from_slice(&num_uids.to_le_bytes());
        for record in &self.audio_ids {
            out.extend_from_slice(&record.track_index.to_le_bytes());
            write_fixed(&mut out, &record.uid, UID_LEN);
            write_fixed(&mut out, &record.track_ref, TRACK_REF_LEN);
            write_fixed(&mut out, &record.pack_ref, PACK_REF_LEN);
            out.push(0);
        }
        out
    }

    /// Parses a `chna` chunk body from little-endian bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ChnaError::Truncated`] if the slice is too short for the
    /// declared record count and [`ChnaError::NonAscii`] if any text field
    /// holds a non-ASCII byte.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChnaError> {
        if bytes.len() < HEADER_LEN {
            return Err(ChnaError::Truncated);
        }
        let num_tracks = u16::from_le_bytes([bytes[0], bytes[1]]);
        let num_uids = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let mut audio_ids = Vec::with_capacity(num_uids);
        let mut offset = HEADER_LEN;
        for _ in 0..num_uids {
            if offset + RECORD_LEN > bytes.len() {
                return Err(ChnaError::Truncated);
            }
            let track_index = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            let mut cursor = offset + 2;
            let uid = read_fixed(bytes, cursor, UID_LEN)?;
            cursor += UID_LEN;
            let track_ref = read_fixed(bytes, cursor, TRACK_REF_LEN)?;
            cursor += TRACK_REF_LEN;
            let pack_ref = read_fixed(bytes, cursor, PACK_REF_LEN)?;
            audio_ids.push(AudioId {
                track_index,
                uid,
                track_ref,
                pack_ref,
            });
            offset += RECORD_LEN;
        }
        Ok(Self {
            num_tracks,
            audio_ids,
        })
    }
}

/// Writes `text` into `out` as exactly `width` bytes: truncated if longer,
/// null-padded if shorter.
fn write_fixed(out: &mut Vec<u8>, text: &str, width: usize) {
    let source = text.as_bytes();
    let copy = source.len().min(width);
    out.extend_from_slice(&source[..copy]);
    for _ in copy..width {
        out.push(0);
    }
}

/// Reads a `width`-byte fixed field at `offset`, stripping trailing nulls.
fn read_fixed(bytes: &[u8], offset: usize, width: usize) -> Result<String, ChnaError> {
    let field = &bytes[offset..offset + width];
    let end = field.iter().position(|&b| b == 0).unwrap_or(width);
    let trimmed = &field[..end];
    if !trimmed.is_ascii() {
        return Err(ChnaError::NonAscii);
    }
    let mut text = String::with_capacity(trimmed.len());
    for &byte in trimmed {
        text.push(byte as char);
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ChnaChunk {
        let mut chna = ChnaChunk::new();
        chna.push(AudioId::new(1, "ATU_00000001", "AC_00031001", "AP_00031001"));
        chna.push(AudioId::new(2, "ATU_00000002", "AC_00031002", "AP_00031001"));
        chna
    }

    #[test]
    fn byte_len_matches_output() {
        let chna = sample();
        assert_eq!(chna.to_bytes().len(), chna.byte_len());
        assert_eq!(chna.byte_len(), HEADER_LEN + 2 * RECORD_LEN);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let chna = sample();
        let bytes = chna.to_bytes();
        let parsed = ChnaChunk::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed, chna);
        assert_eq!(parsed.to_bytes(), bytes);
    }

    #[test]
    fn num_tracks_tracks_the_maximum_index() {
        let chna = sample();
        assert_eq!(chna.num_tracks, 2);
    }

    #[test]
    fn truncated_input_is_rejected() {
        assert_eq!(ChnaChunk::from_bytes(&[0, 0]), Err(ChnaError::Truncated));
        let mut bytes = sample().to_bytes();
        bytes.truncate(bytes.len() - 1);
        assert_eq!(ChnaChunk::from_bytes(&bytes), Err(ChnaError::Truncated));
    }

    #[test]
    fn long_fields_truncate_to_width() {
        let mut chna = ChnaChunk::new();
        chna.push(AudioId::new(1, "ATU_0000000123456", "AC", "AP"));
        let bytes = chna.to_bytes();
        let parsed = ChnaChunk::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed.audio_ids[0].uid.len(), UID_LEN);
    }
}
