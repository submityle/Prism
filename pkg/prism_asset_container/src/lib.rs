//! Versioned, deterministic runtime asset containers and priority overlays.
//!
//! This crate owns runtime consumption only. It does not cook assets,
//! authenticate publishers, or submit data to a GPU. The writer and reader
//! support deterministic raw and raw-DEFLATE payloads; higher-level chunk
//! compression indexes, encryption, signatures, and GPU decompression remain
//! outside this crate. Only [`FORMAT_VERSION`] is a supported container wire
//! version; planned versions are rejected until their complete semantics are
//! specified and implemented.

extern crate alloc;

mod chunk;
mod format;
mod install;
mod manifest;
mod manifest_reader;
mod overlay;
mod patch;
mod range_batch;
mod range_source;
mod reader;
mod writer;

pub use chunk::{
    ChunkInput, ChunkPlacement, ChunkPlan, ChunkPlanError, ChunkPlanner, ChunkPolicy, PlannedChunk,
};
pub use format::{
    Codec, ContainerError, ContainerInfo, EntryInfo, FORMAT_VERSION, MAGIC,
    SUPPORTED_FORMAT_VERSIONS,
};
pub use install::{install_runtime_package, InstallError, InstallMount, InstallTombstones};
pub use manifest::{
    ContainerIdentity, ManifestError, ManifestMetadata, ManifestRoute, RuntimeManifest,
    MANIFEST_MAGIC, MANIFEST_VERSION,
};
pub use manifest_reader::{ManifestRangeRequest, ManifestReadError, ManifestReader};
pub use overlay::{MountedContainer, OverlaySet};
pub use patch::{PatchDescriptorError, RuntimePatchDescriptor, PATCH_MAGIC, PATCH_VERSION};
pub use range_batch::{RangeBatchPolicy, RangeRequest};
pub use range_source::{FileRangeSource, MemoryRangeSource, RangeSource, RangeSourceError};
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
        // Validate the caller's range against the manifest entry before
        // allocating.  A remote or malformed caller must not be able to turn
        // an invalid u64 request into an unbounded Vec allocation.
        let entry = self
            .bound()?
            .entry_info(guid)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))?;
        let end = offset
            .checked_add(len)
            .ok_or(prism_asset::ReadError::OutOfRange {
                offset,
                len,
                size: entry.raw_len,
            })?;
        if end > entry.raw_len {
            return Err(prism_asset::ReadError::OutOfRange {
                offset,
                len,
                size: entry.raw_len,
            });
        }
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
