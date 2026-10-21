//! Versioned, deterministic runtime asset containers and priority overlays.
//!
//! This crate owns runtime consumption only. It does not cook assets,
//! authenticate publishers, or own platform I/O queues. The writer and reader
//! support deterministic raw and raw-DEFLATE payloads in both the per-payload
//! [`FORMAT_VERSION`] layout and the independently compressed
//! [`CHUNK_FORMAT_VERSION`] chunk-table layout, including independently
//! addressable `BlockDeflate` streams. The optional `gpu` feature adapts
//! `prism_gdeflate_gpu` to the injected block backend; the container validates
//! the decoded hashes and execution reports. Encryption and publisher
//! authentication remain outside this crate. Remote range
//! delivery is exposed as a transport-neutral [`RemoteRangeSource`] contract:
//! applications provide an immutable-version-aware HTTP/object-store adapter,
//! while this crate enforces exact identity, offset, length, and request-bound
//! checks before bytes reach the validated reader. The
//! optional `mmap` feature adds a read-only local-file [`MmapRangeSource`]; v2
//! [`ContainerReader`] instances retain a direct mapped directory/table index
//! without changing the validated container ABI or the default positional-I/O
//! path.

extern crate alloc;

pub mod block;
mod budget;
mod chunk;
mod format;
#[cfg(feature = "gpu")]
mod gpu;
mod install;
mod manifest;
mod manifest_reader;
mod overlay;
mod patch;
mod range_batch;
mod range_source;
mod reader;
mod writer;

pub use block::{
    BlockCodec, BlockDecodeBackend, BlockDecodePath, BlockDecodeReport, BlockDecodeRequest,
    BlockDecodeSummary, CpuBlockDecodeBackend,
};
pub use budget::{ContainerReadBudget, ContainerReadLimits};
pub use chunk::{
    ChunkInput, ChunkPlacement, ChunkPlan, ChunkPlanError, ChunkPlanner, ChunkPolicy, PlannedChunk,
};
pub use format::{
    ChunkInfo, Codec, ContainerError, ContainerInfo, EntryInfo, CHUNK_FORMAT_VERSION,
    FORMAT_VERSION, MAGIC, SUPPORTED_FORMAT_VERSIONS,
};
#[cfg(feature = "gpu")]
pub use gpu::{
    GpuBlockDecodeBackend, GpuResidentBlockPayload, GpuResidentBlockRange, GpuResidentBlockSlice,
};
pub use install::{
    install_runtime_package, install_runtime_package_transaction,
    install_runtime_package_transaction_with_policy, install_runtime_package_with_policy,
    InstallError, InstallMount, InstallTombstones, RecoveryError, RuntimeInstallHandle,
    RuntimeInstallPolicy, RuntimeInstallRecovery, RuntimePackageProvenance,
    MAX_RUNTIME_PROVENANCE_BYTES, MAX_RUNTIME_RECOVERY_BYTES,
};
pub use manifest::{
    ContainerIdentity, ManifestError, ManifestMetadata, ManifestRoute, RuntimeManifest,
    MANIFEST_MAGIC, MANIFEST_VERSION,
};
#[cfg(feature = "gpu")]
pub use manifest_reader::{GpuResidentAssetRange, GpuResidentManifestAsset};
pub use manifest_reader::{ManifestRangeRequest, ManifestReadError, ManifestReader};
pub use overlay::{MountedContainer, OverlaySet};
pub use patch::{PatchDescriptorError, RuntimePatchDescriptor, PATCH_MAGIC, PATCH_VERSION};
pub use range_batch::{RangeBatchPolicy, RangeRequest};
#[cfg(feature = "mmap")]
pub use range_source::MmapRangeSource;
pub use range_source::{
    FileRangeSource, MemoryRangeSource, RangeSource, RangeSourceError, RemoteRangeFetchError,
    RemoteRangeFetcher, RemoteRangeRequest, RemoteRangeResponse, RemoteRangeSource,
};
pub use reader::ContainerReader;
#[cfg(feature = "mmap")]
pub use reader::MmapV2DirectoryView;
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
        let bound = self.bound()?;
        let entry = bound
            .entry_info(guid)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > entry.raw_len)
        {
            return Err(prism_asset::ReadError::OutOfRange {
                offset,
                len,
                size: entry.raw_len,
            });
        }
        bound
            .read_range_owned(guid, offset, len)
            .map_err(|error| prism_asset::ReadError::Io(error.to_string()))
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
