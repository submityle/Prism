//! Media entry descriptors inside a bank: residency, format, and location.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The per-media record of design section 20's bank model. Each entry declares
//! where its encoded bytes live inside the bank blob, how it should be resident
//! (fully decoded in memory for short effects, or streamed for long audio), and
//! enough format metadata to budget memory before loading.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::codec::metadata::{AudioStreamInfo, CodecTag};

/// A stable identifier for a media entry within a single bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EntryId(pub u32);

/// How a media entry should reside in memory once its bank is loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Residency {
    /// Decode fully into a resident PCM buffer at load time (short effects).
    Memory,
    /// Stream at play time; `prefetch_frames` frames start resident for a
    /// zero-latency start (long music / ambience).
    Streaming {
        /// Number of leading frames kept resident for zero-latency start.
        prefetch_frames: u32,
    },
}

/// Lightweight format metadata carried by a bank entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaFormat {
    /// Codec family of the encoded bytes.
    pub codec: CodecTag,
    /// Channel count.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total frame count when known at authoring time.
    pub frame_count: Option<u64>,
}

impl MediaFormat {
    /// Creates a new format descriptor.
    #[must_use]
    pub fn new(
        codec: CodecTag,
        channels: u16,
        sample_rate: u32,
        frame_count: Option<u64>,
    ) -> Self {
        Self {
            codec,
            channels,
            sample_rate,
            frame_count,
        }
    }

    /// Returns the number of bytes a fully decoded (`f32`) resident copy would
    /// occupy, when the frame count is known.
    #[must_use]
    pub fn decoded_bytes(&self) -> Option<u64> {
        self.frame_count.map(|frames| {
            frames
                * u64::from(self.channels)
                * size_of::<prism_audio_core::math::Sample>() as u64
        })
    }

    /// Builds an [`AudioStreamInfo`] from this format.
    #[must_use]
    pub fn stream_info(&self) -> AudioStreamInfo {
        AudioStreamInfo::new(
            self.channels,
            self.sample_rate,
            self.frame_count,
            self.codec.clone(),
        )
    }
}

/// A single media item declared by a bank.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaEntry {
    /// Bank-local identifier.
    pub id: EntryId,
    /// Human-readable name used for name-based lookup.
    pub name: String,
    /// Format metadata.
    pub format: MediaFormat,
    /// Residency policy.
    pub residency: Residency,
    /// Byte offset of the encoded media within the bank blob.
    pub byte_offset: u64,
    /// Byte length of the encoded media within the bank blob.
    pub byte_len: u64,
}

impl MediaEntry {
    /// Creates a new media entry.
    #[must_use]
    pub fn new(
        id: EntryId,
        name: String,
        format: MediaFormat,
        residency: Residency,
        byte_offset: u64,
        byte_len: u64,
    ) -> Self {
        Self {
            id,
            name,
            format,
            residency,
            byte_offset,
            byte_len,
        }
    }

    /// Returns `true` when the entry is streamed rather than fully resident.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        matches!(self.residency, Residency::Streaming { .. })
    }

    /// Returns the number of prefetch frames for a streaming entry (0 for
    /// memory-resident entries).
    #[must_use]
    pub fn prefetch_frames(&self) -> u32 {
        match self.residency {
            Residency::Streaming { prefetch_frames } => prefetch_frames,
            Residency::Memory => 0,
        }
    }
}
