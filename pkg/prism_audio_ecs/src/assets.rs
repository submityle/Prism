//! ECS-owned access to imported and decoder-verified audio bank media.

use alloc::vec::Vec;
use prism_audio_assets::asset::{resolve_audio_asset, AudioAssetHandle, ResolvedAudioAsset};
use prism_audio_assets::bank::{BankId, BankManifest, BankRegistry, BankRegistryError};
use prism_audio_assets::codec::metadata::AudioStreamInfo;
use prism_audio_assets::codec::registry::DecoderRegistry;
use prism_audio_core::math::Sample;
use prism_ecs::prelude::Resource;

/// Errors returned while loading a bank for ECS playback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioAssetLoadError {
    /// The bank registry rejected the manifest/blob/decoder combination.
    Registry(BankRegistryError),
}

/// A source payload that is safe to hand to an off-thread graph builder.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentAudioSource {
    /// Asset identity retained for route diagnostics.
    pub handle: AudioAssetHandle,
    /// Planar PCM owned by the route builder.
    pub channels: Vec<Vec<Sample>>,
    /// Decoder-reported stream metadata.
    pub stream: AudioStreamInfo,
}

/// Why an asset could not become a real source route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSourceRouteError {
    /// The bank or entry is not currently loaded.
    NotLoaded,
    /// Streaming decode is not yet connected to an `AudioGraph` source node.
    StreamingRequiresBackend,
    /// The decoder reported no valid channel count or sample rate.
    InvalidStreamMetadata,
    /// The resident payload has an invalid interleaved shape.
    InvalidResidentPcm,
    /// The generic resident route currently accepts mono and stereo sources.
    UnsupportedChannelCount,
}

/// ECS resource containing the loaded audio banks and their decoder registry.
///
/// Loading is an app/task-thread operation. The resource never performs I/O or
/// decoding from the audio callback; `BankRegistry::load` validates and decodes
/// resident entries before publishing the bank.
#[derive(Resource)]
pub struct AudioAssetStore {
    banks: BankRegistry,
    decoders: DecoderRegistry,
    revision: u64,
}

impl Default for AudioAssetStore {
    fn default() -> Self {
        Self {
            banks: BankRegistry::new(),
            decoders: DecoderRegistry::with_native(),
            revision: 0,
        }
    }
}

impl AudioAssetStore {
    /// Loads and validates one bank. No bank is inserted when loading fails.
    pub fn load_bank(
        &mut self,
        manifest: BankManifest,
        blob: &[u8],
    ) -> Result<BankId, AudioAssetLoadError> {
        let result = self
            .banks
            .load(manifest, blob, &self.decoders)
            .map_err(AudioAssetLoadError::Registry);
        if result.is_ok() {
            self.revision = self.revision.wrapping_add(1);
        }
        result
    }

    /// Unloads one bank after dependency checks.
    pub fn unload_bank(&mut self, id: BankId) -> Result<(), AudioAssetLoadError> {
        let result = self.banks.unload(id).map_err(AudioAssetLoadError::Registry);
        if result.is_ok() {
            self.revision = self.revision.wrapping_add(1);
        }
        result
    }

    /// Resolves a handle to metadata reported by the loaded decoder.
    #[must_use]
    pub fn resolve(&self, handle: AudioAssetHandle) -> Option<ResolvedAudioAsset> {
        resolve_audio_asset(&self.banks, handle)
    }

    /// Returns whether a bank is currently resident.
    #[must_use]
    pub fn is_loaded(&self, id: BankId) -> bool {
        self.banks.is_loaded(id)
    }

    /// Monotonic content revision used by graph-route caches.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Copies a resident interleaved payload into owned planar source data.
    /// Streaming media returns an explicit gate error instead of a fake voice.
    pub fn resident_source(
        &self,
        handle: AudioAssetHandle,
    ) -> Result<ResidentAudioSource, AudioSourceRouteError> {
        let media = self
            .banks
            .media(handle.bank, handle.entry)
            .ok_or(AudioSourceRouteError::NotLoaded)?;
        let stream = media.info().clone();
        if stream.channels == 0 || stream.sample_rate == 0 {
            return Err(AudioSourceRouteError::InvalidStreamMetadata);
        }
        let channels = usize::from(stream.channels);
        let pcm = media
            .resident_pcm()
            .ok_or(AudioSourceRouteError::StreamingRequiresBackend)?;
        if pcm.len() % channels != 0 {
            return Err(AudioSourceRouteError::InvalidResidentPcm);
        }
        if stream
            .frame_count
            .is_some_and(|frames| frames != (pcm.len() / channels) as u64)
        {
            return Err(AudioSourceRouteError::InvalidResidentPcm);
        }
        let frames = pcm.len() / channels;
        let mut planar = (0..channels)
            .map(|_| Vec::with_capacity(frames))
            .collect::<Vec<_>>();
        for frame in pcm.chunks_exact(channels) {
            for (channel, sample) in frame.iter().copied().enumerate() {
                planar[channel].push(sample);
            }
        }
        Ok(ResidentAudioSource {
            handle,
            channels: planar,
            stream,
        })
    }
}
