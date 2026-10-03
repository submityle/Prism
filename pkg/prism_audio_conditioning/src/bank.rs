//! Bank authoring: dependency de-duplication and streaming partitioning.
//!
//! A bank is assembled from a flattened list of asset references (the leaves of
//! an events-to-containers-to-assets dependency graph). References that resolve
//! to the same content are stored once, keyed by their content hash. Each
//! unique asset is then partitioned into a resident or streaming tier by its
//! frame count, streaming entries receive a prefetch hint, and the result is a
//! deterministic layout descriptor whose entries are ordered by content key.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the authoring leg of design section 51 and emits the entry types
//! of design section 20 (`prism_audio_assets::bank`), reusing [`MediaEntry`],
//! [`MediaFormat`], and [`Residency`] rather than inventing a parallel model.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_audio_assets::bank::{EntryId, MediaEntry, MediaFormat, Residency};
use prism_audio_assets::codec::CodecTag;

use crate::config::BankConfig;
use crate::content_hash::ContentHash;

/// One reference to a conditioned asset in the dependency graph.
///
/// Several references may share a `content_key`; those collapse to a single
/// stored entry during authoring.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AssetInput {
    /// Content hash identifying the asset's bytes; shared references match.
    pub content_key: ContentHash,
    /// Human-readable name used for name-based lookup in the bank.
    pub name: String,
    /// Codec family of the stored bytes.
    pub codec: CodecTag,
    /// Channel count.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total frame count of the asset.
    pub frame_count: u64,
    /// Encoded byte length of the asset within the bank blob.
    pub byte_len: u64,
}

/// The deterministic result of authoring a bank from a set of inputs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BankLayout {
    /// De-duplicated entries ordered by content key, with sequential ids and
    /// packed byte offsets.
    pub entries: Vec<MediaEntry>,
    /// Total encoded byte length of all stored entries.
    pub total_byte_len: u64,
    /// Number of memory-resident entries.
    pub resident_count: usize,
    /// Number of streaming entries.
    pub streaming_count: usize,
    /// Number of input references that collapsed onto an existing entry.
    pub deduplicated_references: usize,
}

/// Stateless bank-authoring entry point.
#[derive(Debug, Clone, Copy, Default)]
pub struct BankAuthoring;

impl BankAuthoring {
    /// Authors a [`BankLayout`] from a flattened reference list.
    ///
    /// References are de-duplicated by [`AssetInput::content_key`] (first
    /// occurrence wins for metadata), sorted by content key for a stable
    /// layout, assigned sequential [`EntryId`]s and packed byte offsets, and
    /// partitioned into [`Residency::Memory`] or [`Residency::Streaming`] by
    /// comparing their frame count against
    /// [`BankConfig::streaming_threshold_frames`].
    #[must_use]
    pub fn author(inputs: &[AssetInput], config: &BankConfig) -> BankLayout {
        let mut unique: BTreeMap<u64, &AssetInput> = BTreeMap::new();
        let mut duplicates = 0usize;
        for input in inputs {
            if unique.insert(input.content_key.0, input).is_some() {
                duplicates += 1;
            }
        }

        let mut entries: Vec<MediaEntry> = Vec::with_capacity(unique.len());
        let mut total_byte_len = 0u64;
        let mut resident_count = 0usize;
        let mut streaming_count = 0usize;
        let mut byte_offset = 0u64;

        for (index, (_key, input)) in unique.iter().enumerate() {
            let residency = if input.frame_count > config.streaming_threshold_frames {
                streaming_count += 1;
                Residency::Streaming {
                    prefetch_frames: config.prefetch_frames,
                }
            } else {
                resident_count += 1;
                Residency::Memory
            };
            let format = MediaFormat::new(
                input.codec.clone(),
                input.channels,
                input.sample_rate,
                Some(input.frame_count),
            );
            let entry = MediaEntry::new(
                EntryId(index as u32),
                input.name.clone(),
                format,
                residency,
                byte_offset,
                input.byte_len,
            );
            byte_offset += input.byte_len;
            total_byte_len += input.byte_len;
            entries.push(entry);
        }

        BankLayout {
            entries,
            total_byte_len,
            resident_count,
            streaming_count,
            deduplicated_references: duplicates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn input(key: u64, name: &str, frames: u64, bytes: u64) -> AssetInput {
        AssetInput {
            content_key: ContentHash(key),
            name: name.to_string(),
            codec: CodecTag::Pcm,
            channels: 1,
            sample_rate: 48_000,
            frame_count: frames,
            byte_len: bytes,
        }
    }

    #[test]
    fn duplicate_reference_is_stored_once() {
        let inputs = vec![
            input(1, "shot", 1_000, 4_000),
            input(1, "shot", 1_000, 4_000),
            input(2, "step", 2_000, 8_000),
        ];
        let layout = BankAuthoring::author(&inputs, &BankConfig::default());
        assert_eq!(layout.entries.len(), 2);
        assert_eq!(layout.deduplicated_references, 1);
        assert_eq!(layout.total_byte_len, 12_000);
    }

    #[test]
    fn long_asset_goes_to_streaming_tier() {
        let config = BankConfig {
            streaming_threshold_frames: 10_000,
            prefetch_frames: 2_048,
        };
        let inputs = vec![
            input(10, "sfx", 5_000, 2_000),
            input(20, "music", 500_000, 1_000_000),
        ];
        let layout = BankAuthoring::author(&inputs, &config);
        assert_eq!(layout.resident_count, 1);
        assert_eq!(layout.streaming_count, 1);

        let music = layout
            .entries
            .iter()
            .find(|e| e.name == "music")
            .expect("music entry");
        assert!(music.is_streaming());
        assert_eq!(music.prefetch_frames(), 2_048);

        let sfx = layout
            .entries
            .iter()
            .find(|e| e.name == "sfx")
            .expect("sfx entry");
        assert!(!sfx.is_streaming());
    }

    #[test]
    fn entries_are_ordered_by_content_key_with_packed_offsets() {
        let inputs = vec![
            input(30, "c", 1, 100),
            input(10, "a", 1, 200),
            input(20, "b", 1, 300),
        ];
        let layout = BankAuthoring::author(&inputs, &BankConfig::default());
        let names: Vec<&str> = layout.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        assert_eq!(layout.entries[0].byte_offset, 0);
        assert_eq!(layout.entries[1].byte_offset, 200);
        assert_eq!(layout.entries[2].byte_offset, 500);
    }
}
