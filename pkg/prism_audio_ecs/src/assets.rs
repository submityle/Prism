//! ECS-owned access to imported and decoder-verified audio bank media.

use prism_audio_assets::asset::{resolve_audio_asset, AudioAssetHandle, ResolvedAudioAsset};
use prism_audio_assets::bank::{BankId, BankManifest, BankRegistry, BankRegistryError};
use prism_audio_assets::codec::registry::DecoderRegistry;
use prism_ecs::prelude::Resource;

/// Errors returned while loading a bank for ECS playback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioAssetLoadError {
    /// The bank registry rejected the manifest/blob/decoder combination.
    Registry(BankRegistryError),
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
}

impl Default for AudioAssetStore {
    fn default() -> Self {
        Self {
            banks: BankRegistry::new(),
            decoders: DecoderRegistry::with_native(),
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
        self.banks
            .load(manifest, blob, &self.decoders)
            .map_err(AudioAssetLoadError::Registry)
    }

    /// Unloads one bank after dependency checks.
    pub fn unload_bank(&mut self, id: BankId) -> Result<(), AudioAssetLoadError> {
        self.banks.unload(id).map_err(AudioAssetLoadError::Registry)
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
}
