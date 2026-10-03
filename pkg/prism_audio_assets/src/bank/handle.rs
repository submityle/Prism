//! The [`LoadedBank`]: a manifest plus its decoded / retained media payloads.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the load/unload semantics and memory accounting of design
//! section 20. Loading walks the manifest's media entries: memory-resident
//! entries are decoded to `f32` PCM once at load time (short effects live in a
//! resident pool); streaming entries keep their compact encoded bytes so a
//! [`PrefetchStream`](crate::streaming::prefetch::PrefetchStream) can be built
//! on demand without re-reading the blob. Unloading simply drops the bank,
//! freeing both pools. All decode work happens here, off the real-time thread.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::bank::entry::{EntryId, MediaEntry, Residency};
use crate::bank::manifest::{BankManifest, ManifestError};
use crate::codec::decoder::DecodeError;
use crate::codec::metadata::AudioStreamInfo;
use crate::codec::registry::DecoderRegistry;
use crate::streaming::prefetch::PrefetchStream;

/// Errors surfaced while loading a bank.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BankLoadError {
    /// The manifest failed structural validation against the blob.
    Manifest(ManifestError),
    /// A media entry could not be decoded.
    Decode {
        /// The entry whose bytes failed to decode.
        entry: EntryId,
        /// The underlying decode error.
        error: DecodeError,
    },
}

/// A media payload held by a [`LoadedBank`].
///
/// Memory-resident entries are fully decoded into interleaved `f32`; streaming
/// entries retain their encoded bytes so a prefetch stream can be constructed
/// lazily at play time.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadedMedia {
    /// A short effect decoded fully into a resident PCM block.
    Resident {
        /// Interleaved `f32` samples.
        pcm: Vec<Sample>,
        /// Stream description for the decoded block.
        info: AudioStreamInfo,
    },
    /// A long asset whose encoded bytes are retained for streaming.
    Streaming {
        /// Encoded media bytes copied out of the bank blob.
        encoded: Vec<u8>,
        /// Stream description reported by the decoder.
        info: AudioStreamInfo,
        /// Leading frames to keep resident for a zero-latency start.
        prefetch_frames: u32,
    },
}

impl LoadedMedia {
    /// Returns the stream description regardless of residency.
    #[must_use]
    pub fn info(&self) -> &AudioStreamInfo {
        match self {
            LoadedMedia::Resident { info, .. } | LoadedMedia::Streaming { info, .. } => info,
        }
    }

    /// Returns the resident PCM block, when this media is memory-resident.
    #[must_use]
    pub fn resident_pcm(&self) -> Option<&[Sample]> {
        match self {
            LoadedMedia::Resident { pcm, .. } => Some(pcm),
            LoadedMedia::Streaming { .. } => None,
        }
    }

    /// Returns `true` when this media is streamed rather than resident.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        matches!(self, LoadedMedia::Streaming { .. })
    }

    /// Builds a fresh [`PrefetchStream`] for a streaming entry.
    ///
    /// Returns `None` for memory-resident media (which is read directly from
    /// its PCM block). `ring_capacity_frames` sizes the streamed look-ahead.
    pub fn open_stream(
        &self,
        registry: &DecoderRegistry,
        ring_capacity_frames: usize,
    ) -> Option<Result<PrefetchStream, DecodeError>> {
        match self {
            LoadedMedia::Resident { .. } => None,
            LoadedMedia::Streaming {
                encoded,
                prefetch_frames,
                ..
            } => {
                let decoder = match registry.decode_bytes(encoded) {
                    Ok(decoder) => decoder,
                    Err(error) => return Some(Err(error)),
                };
                Some(PrefetchStream::new(
                    decoder,
                    *prefetch_frames as usize,
                    ring_capacity_frames,
                ))
            }
        }
    }

    /// Returns the number of resident PCM bytes this media occupies.
    #[must_use]
    pub fn resident_pcm_bytes(&self) -> u64 {
        match self {
            LoadedMedia::Resident { pcm, .. } => {
                (pcm.len() * size_of::<Sample>()) as u64
            }
            LoadedMedia::Streaming { .. } => 0,
        }
    }

    /// Returns the number of retained encoded bytes this media occupies.
    #[must_use]
    pub fn encoded_bytes(&self) -> u64 {
        match self {
            LoadedMedia::Resident { .. } => 0,
            LoadedMedia::Streaming { encoded, .. } => encoded.len() as u64,
        }
    }
}

/// A breakdown of a loaded bank's memory footprint, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BankMemoryUsage {
    /// Bytes held by fully decoded resident PCM blocks.
    pub resident_pcm_bytes: u64,
    /// Bytes held by retained encoded (streaming) media.
    pub encoded_bytes: u64,
    /// Approximate bytes held by metadata (manifest-derived bookkeeping).
    pub metadata_bytes: u64,
}

impl BankMemoryUsage {
    /// Returns the total footprint across all categories.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.resident_pcm_bytes
            .saturating_add(self.encoded_bytes)
            .saturating_add(self.metadata_bytes)
    }
}

/// A fully loaded bank: its manifest and the decoded / retained payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedBank {
    manifest: BankManifest,
    media: Vec<(EntryId, LoadedMedia)>,
}

impl LoadedBank {
    /// Loads a bank from its `manifest`, backing `blob`, and a decoder
    /// `registry`.
    ///
    /// The manifest is validated against the blob length first. Each media
    /// entry's byte slice is then taken from the blob: memory-resident entries
    /// are decoded fully to `f32`; streaming entries copy their encoded bytes
    /// out for later prefetch construction.
    pub fn load(
        manifest: BankManifest,
        blob: &[u8],
        registry: &DecoderRegistry,
    ) -> Result<Self, BankLoadError> {
        manifest
            .validate(blob.len() as u64)
            .map_err(BankLoadError::Manifest)?;
        let mut media: Vec<(EntryId, LoadedMedia)> = Vec::with_capacity(manifest.media.len());
        for entry in &manifest.media {
            let loaded = Self::load_entry(entry, blob, registry)?;
            media.push((entry.id, loaded));
        }
        Ok(Self { manifest, media })
    }

    fn load_entry(
        entry: &MediaEntry,
        blob: &[u8],
        registry: &DecoderRegistry,
    ) -> Result<LoadedMedia, BankLoadError> {
        let start = entry.byte_offset as usize;
        let end = start + entry.byte_len as usize;
        let bytes = &blob[start..end];
        match entry.residency {
            Residency::Memory => {
                let mut decoder = registry.decode_bytes(bytes).map_err(|error| {
                    BankLoadError::Decode {
                        entry: entry.id,
                        error,
                    }
                })?;
                let info = decoder.info();
                let pcm = decoder.decode_to_end().map_err(|error| BankLoadError::Decode {
                    entry: entry.id,
                    error,
                })?;
                Ok(LoadedMedia::Resident { pcm, info })
            }
            Residency::Streaming { prefetch_frames } => {
                // Probe the stream so the retained info matches the decoder's
                // own report rather than only the authored metadata.
                let decoder = registry.decode_bytes(bytes).map_err(|error| {
                    BankLoadError::Decode {
                        entry: entry.id,
                        error,
                    }
                })?;
                let info = decoder.info();
                drop(decoder);
                let mut encoded = Vec::with_capacity(bytes.len());
                encoded.extend_from_slice(bytes);
                Ok(LoadedMedia::Streaming {
                    encoded,
                    info,
                    prefetch_frames,
                })
            }
        }
    }

    /// Returns the bank's manifest.
    #[must_use]
    pub fn manifest(&self) -> &BankManifest {
        &self.manifest
    }

    /// Resolves a loaded media payload by its bank-local id.
    #[must_use]
    pub fn media_by_id(&self, id: EntryId) -> Option<&LoadedMedia> {
        self.media
            .iter()
            .find(|(entry_id, _)| *entry_id == id)
            .map(|(_, media)| media)
    }

    /// Resolves a loaded media payload by name via the manifest.
    #[must_use]
    pub fn media_by_name(&self, name: &str) -> Option<&LoadedMedia> {
        let id = self.manifest.media_by_name(name)?.id;
        self.media_by_id(id)
    }

    /// Returns an iterator over the bank's loaded media and their ids.
    pub fn iter_media(&self) -> impl Iterator<Item = (EntryId, &LoadedMedia)> {
        self.media.iter().map(|(id, media)| (*id, media))
    }

    /// Computes the bank's current memory footprint.
    #[must_use]
    pub fn memory_usage(&self) -> BankMemoryUsage {
        let mut usage = BankMemoryUsage::default();
        for (_, media) in &self.media {
            usage.resident_pcm_bytes = usage
                .resident_pcm_bytes
                .saturating_add(media.resident_pcm_bytes());
            usage.encoded_bytes = usage.encoded_bytes.saturating_add(media.encoded_bytes());
        }
        // Metadata estimate: a fixed per-entry record plus the media name.
        let per_entry = size_of::<MediaEntry>() as u64;
        let mut metadata = 0u64;
        for entry in &self.manifest.media {
            metadata = metadata
                .saturating_add(per_entry)
                .saturating_add(entry.name.len() as u64);
        }
        usage.metadata_bytes = metadata;
        usage
    }
}
