//! Stable references and resolved metadata for ECS-addressable audio media.
//!
//! A handle is only an identity: it does not imply that the bank is loaded or
//! that the codec is available. Callers must resolve it through a loaded
//! [`BankRegistry`](crate::bank::BankRegistry) before creating a playback
//! voice.

use crate::bank::{BankId, BankRegistry, EntryId, LoadedMedia};
use crate::codec::metadata::AudioStreamInfo;

/// Stable reference to one media entry in one audio bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioAssetHandle {
    /// Bank containing the encoded media.
    pub bank: BankId,
    /// Bank-local media entry id.
    pub entry: EntryId,
}

impl AudioAssetHandle {
    /// Creates a handle for a bank-local media entry.
    #[must_use]
    #[inline]
    pub const fn new(bank: BankId, entry: EntryId) -> Self {
        Self { bank, entry }
    }
}

/// Whether resolved media is decoded up front or retained for streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AudioAssetResidency {
    /// The bank owns decoded interleaved PCM for this entry.
    Resident,
    /// The bank owns encoded bytes and a decoder can be opened off-thread.
    Streaming,
}

/// Decoder-verified metadata for an ECS playback source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAudioAsset {
    /// Identity used to resolve this metadata.
    pub handle: AudioAssetHandle,
    /// Metadata reported by the actual decoder.
    pub stream: AudioStreamInfo,
    /// Loaded residency policy.
    pub residency: AudioAssetResidency,
}

/// Resolves a handle against already-loaded banks.
pub fn resolve_audio_asset(
    banks: &BankRegistry,
    handle: AudioAssetHandle,
) -> Option<ResolvedAudioAsset> {
    let media = banks.media(handle.bank, handle.entry)?;
    Some(ResolvedAudioAsset {
        handle,
        stream: media.info().clone(),
        residency: if matches!(media, LoadedMedia::Resident { .. }) {
            AudioAssetResidency::Resident
        } else {
            AudioAssetResidency::Streaming
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bank::{BankManifest, MediaEntry, MediaFormat, Residency};
    use crate::codec::{CodecTag, DecoderRegistry};
    use alloc::string::String;
    use alloc::vec::Vec;

    #[test]
    fn handle_resolves_only_loaded_decoder_verified_media() {
        let bytes = wav_mono(4);
        let manifest =
            BankManifest::new(BankId(2), String::from("test"), 1).with_media(MediaEntry::new(
                EntryId(9),
                String::from("clip"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(4)),
                Residency::Memory,
                0,
                bytes.len() as u64,
            ));
        let mut banks = BankRegistry::new();
        let decoders = DecoderRegistry::with_native();
        let handle = AudioAssetHandle::new(BankId(2), EntryId(9));
        assert!(resolve_audio_asset(&banks, handle).is_none());
        banks.load(manifest, &bytes, &decoders).expect("bank loads");
        let resolved = resolve_audio_asset(&banks, handle).expect("asset resolves");
        assert_eq!(resolved.stream.frame_count, Some(4));
        assert_eq!(resolved.residency, AudioAssetResidency::Resident);
    }

    fn wav_mono(frames: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            data.extend_from_slice(&(frame as i16).to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&96_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }
}
