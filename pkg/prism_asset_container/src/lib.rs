//! Versioned, deterministic runtime asset containers and priority overlays.
//!
//! This crate owns runtime consumption only. It does not cook assets, compress
//! payloads, authenticate publishers, or submit data to a GPU. The current
//! writer supports the `Raw` codec; other codec ids are preserved in metadata
//! by future-compatible readers only when explicitly supported (currently they
//! are rejected during mount).

extern crate alloc;

mod chunk;
mod format;
mod manifest;
mod manifest_reader;
mod overlay;
mod reader;
mod writer;

pub use chunk::{
    ChunkInput, ChunkPlacement, ChunkPlan, ChunkPlanError, ChunkPlanner, ChunkPolicy, PlannedChunk,
};
pub use format::{Codec, ContainerError, ContainerInfo, EntryInfo, FORMAT_VERSION, MAGIC};
pub use manifest::{
    ManifestError, ManifestRoute, RuntimeManifest, MANIFEST_MAGIC, MANIFEST_VERSION,
};
pub use manifest_reader::{ManifestReadError, ManifestReader};
pub use overlay::{MountedContainer, OverlaySet};
pub use reader::ContainerReader;
pub use writer::{ContainerWriter, WriteEntry};

/// A strict [`prism_asset::AssetReader`] backed by an owned runtime manifest
/// and overlay snapshot.
///
/// The reader derives the logical GUID from the VFS path using the frozen
/// `StableGuid` path contract. A `#label` selects the corresponding derived
/// sub-asset GUID. Manifest binding is checked for every operation, so a
/// remounted or priority-drifted container fails closed instead of silently
/// reading a different payload.
#[derive(Clone, Debug)]
pub struct ManifestAssetReader {
    manifest: RuntimeManifest,
    overlays: OverlaySet,
}

impl ManifestAssetReader {
    /// Creates a reader from an immutable manifest and mounted-container
    /// snapshot. Both values are cloned so the reader is `Send + Sync`.
    #[must_use]
    pub fn new(manifest: RuntimeManifest, overlays: OverlaySet) -> Self {
        Self { manifest, overlays }
    }

    /// Returns the owned manifest used by this reader.
    #[must_use]
    pub fn manifest(&self) -> &RuntimeManifest {
        &self.manifest
    }

    fn logical_guid(path: &str) -> Result<prism_asset::StableGuid, prism_asset::ReadError> {
        let parsed = prism_asset::AssetPath::parse(path);
        if parsed.has_scheme() || parsed.path().is_empty() || parsed.path().starts_with('/') {
            return Err(prism_asset::ReadError::InvalidPath);
        }
        if parsed
            .path()
            .split(['/', '\\'])
            .any(|segment| segment == "..")
        {
            return Err(prism_asset::ReadError::InvalidPath);
        }
        let parent = prism_asset::StableGuid::from_path(parsed.path());
        Ok(parsed.label().map_or(parent, |label| {
            prism_asset::StableGuid::derive_sub(parent, label)
        }))
    }

    fn bound(&self) -> Result<ManifestReader<'_, '_>, prism_asset::ReadError> {
        self.manifest
            .bind(&self.overlays)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))
    }
}

impl prism_asset::AssetReader for ManifestAssetReader {
    fn read(&self, path: &str) -> Result<Vec<u8>, prism_asset::ReadError> {
        let guid = Self::logical_guid(path)?;
        self.bound()?
            .read_all(guid)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))
    }

    fn read_range(
        &self,
        path: &str,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, prism_asset::ReadError> {
        let guid = Self::logical_guid(path)?;
        let len_usize = usize::try_from(len).map_err(|_| prism_asset::ReadError::OutOfRange {
            offset,
            len,
            size: 0,
        })?;
        let mut out = vec![0; len_usize];
        self.bound()?
            .read_range(guid, offset, &mut out)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))?;
        Ok(out)
    }

    fn metadata(&self, path: &str) -> Result<prism_asset::AssetMeta, prism_asset::ReadError> {
        let guid = Self::logical_guid(path)?;
        let entry = self
            .bound()?
            .entry_info(guid)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))?;
        Ok(prism_asset::AssetMeta {
            size: entry.raw_len,
            modified_ms: None,
            is_dir: false,
        })
    }
}

#[cfg(test)]
mod tests;
