#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![doc = "Engine-neutral coordination from world streaming policy to cell IO and scene hosting."]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU64;
use core::task::{Context, Poll};

use prism_world_io::ReadRequest;
use prism_world_stream::{
    validate_manifests, CellManifest, DemandSource, ManifestValidationError, SectorId,
    StreamAction, StreamBudget,
};

/// Default bounded storage reserved for validated prefetched cell bytes.
pub const DEFAULT_PREFETCH_CACHE_BYTES: u64 = 64 * 1024 * 1024;

mod routed_io;
pub use routed_io::RoutedCellIo;
mod decoded_scene_host;
pub use decoded_scene_host::{
    CellDecodeLimits, CellPayloadDecoder, DecodedCell, DecodedCellHost, DecodedSceneHost,
    DecodedSceneHostError,
};
mod cell_world;
pub use cell_world::{
    CellCommit, ResolvedWorldRef, StableRefDecoder, StableRefSceneHost, StableRefSceneHostError,
    StableWorldRef, WorldCellStore, WorldCellStoreError, WorldCellStoreLimits,
    WORLD_CELL_SNAPSHOT_VERSION,
};
mod world_partition_capability;
pub use world_partition_capability::{WorldPartitionCapability, WorldPartitionCapabilityError};
mod hlod_runtime;
pub use hlod_runtime::{
    HlodDistanceRequest, HlodRuntime, HlodRuntimeError, HlodSelection, HlodSelectionBudget,
    HlodSelectionPlan, HLOD_RUNTIME_SELECTION_VERSION,
};
mod sdf_local_volume;
pub use sdf_local_volume::{WorldSdfLocalVolume, WorldSdfLocalVolumeError};
mod sdf_objects;
pub use sdf_objects::{
    WorldSdfObject, WorldSdfObjectBake, WorldSdfObjectBakeBudget, WorldSdfObjectBakeError,
    WorldSdfObjectBakeReport, WorldSdfObjectConfig, WorldSdfObjectError, WorldSdfObjectLimits,
    WorldSdfObjectResource, WorldSdfObjectSample, WorldSdfObjectStore, WorldSdfObjectUpdate,
};
#[cfg(feature = "gpu_proxy")]
mod sdf_objects_gpu;
#[cfg(feature = "gpu_proxy")]
pub use sdf_objects_gpu::{WorldSdfGpuObjects, WorldSdfGpuObjectsConfig, WorldSdfGpuObjectsError};
mod sdf_clipmap;
pub use sdf_clipmap::{
    WorldSdfClipmapCommit, WorldSdfClipmapConfig, WorldSdfClipmapError, WorldSdfClipmapPlan,
    WorldSdfClipmapScheduler,
};
mod sdf_focus;
pub use sdf_focus::{
    WorldSdfFocusConfig, WorldSdfFocusDriver, WorldSdfFocusError, WorldSdfFocusUpdate,
    WorldSdfFocusUpdateError,
};
#[cfg(feature = "gpu_proxy")]
mod sdf_focus_gpu;
#[cfg(feature = "gpu_proxy")]
pub use sdf_focus_gpu::{WorldSdfGpuFocus, WorldSdfGpuFocusError};
#[cfg(any(test, feature = "gpu_proxy"))]
mod sdf_clipmap_upload;
#[cfg(feature = "gpu_proxy")]
pub use sdf_clipmap_upload::{
    WorldSdfClipmapUploadError, WorldSdfClipmapUploadStats, WorldSdfGpuClipmap,
    WorldSdfGpuClipmapError,
};
#[cfg(feature = "sdf_ecs_gpu")]
mod sdf_ecs_gpu;
#[cfg(feature = "sdf_ecs_gpu")]
pub use sdf_ecs_gpu::{
    WorldSdfEcsGpuBridge, WorldSdfEcsGpuConfig, WorldSdfEcsGpuError, WorldSdfEcsGpuPolicy,
    WorldSdfEcsRequest, WorldSdfEcsSample, WorldSdfEcsSampleReport, WorldSdfEcsSampleValue,
};
mod data_layers;
pub use data_layers::{
    DataLayerConfigError, DataLayerId, DataLayerState, LayeredCellFrameCoordinator,
    LayeredCellManifest, LayeredCellState, LayeredCoordinatorError, LayeredEvictionOutcome,
    LayeredEvictionReport, LayeredFrameReport, LayeredLoadedCell, LayeredPollReport,
    LayeredSceneHost, RuntimeDataLayers,
};
pub use prism_world_stream::{DataLayerMask, DataLayerMaskError, MAX_DATA_LAYERS};
#[cfg(feature = "scene_templates")]
mod scene_template_provider;
#[cfg(feature = "scene_templates")]
pub use scene_template_provider::{
    load_scene_template, NestedSceneTreeId, NestedSceneTreeRegistry, NestedSceneTreeRegistryError,
    SceneTemplateApplyOutcome, SceneTemplateBatchError, SceneTemplateCache,
    SceneTemplateInvalidation, SceneTemplateLoadError,
};
#[cfg(feature = "placement")]
mod placement_persist;
#[cfg(feature = "placement")]
pub use placement_persist::{
    decode_placement_deletions, encode_placement_deletions, PersistedHeightfieldPlacementOutput,
    PlacementDeletionBinding, PlacementDeletionCodecBudget, PlacementDeletionSnapshot,
    PlacementPersistenceError, PLACEMENT_DELETION_CODEC_VERSION, PLACEMENT_DELETION_HEADER_BYTES,
    PLACEMENT_DELETION_KEY_BYTES,
};
#[cfg(all(test, feature = "placement"))]
mod placement_persist_tests;
mod scatter_persist;
pub use scatter_persist::{
    decode_scatter_deletions, encode_scatter_deletions, PersistedFieldScatterOutput,
    PersistedScatterOutput, ScatterDeletionBinding, ScatterDeletionCodecBudget,
    ScatterDeletionSnapshot, ScatterPersistenceError, SCATTER_DELETION_HEADER_BYTES,
    SCATTER_DELETION_KEY_BYTES,
};
#[cfg(test)]
mod async_coordinator_tests;
#[cfg(test)]
mod decoded_scene_host_tests;
#[cfg(test)]
mod scatter_persist_tests;

#[cfg(all(feature = "asset_container", feature = "tasks"))]
mod asset_container_cell_io;
#[cfg(feature = "asset_container")]
mod asset_container_provider;
#[cfg(feature = "std")]
mod file_provider;
#[cfg(feature = "gpu_proxy")]
mod gpu_proxy_cell_host;
#[cfg(feature = "std")]
mod io_adapter;
#[cfg(feature = "tasks")]
mod priority_cell_io;
#[cfg(feature = "proxy_host")]
mod proxy_cell_host;
#[cfg(all(feature = "reactor", unix))]
mod reactor_cell_io;
#[cfg(feature = "residency_proxy_host")]
mod residency_proxy_host;
#[cfg(feature = "std")]
mod scheduled_io;
#[cfg(feature = "tasks")]
mod task_cell_io;
#[cfg(feature = "std")]
mod threaded_cell_io;

#[cfg(all(test, feature = "asset_container"))]
mod asset_container_provider_tests;
#[cfg(all(test, feature = "std"))]
mod file_provider_tests;
#[cfg(all(test, feature = "gpu_proxy"))]
mod gpu_proxy_cell_host_tests;
#[cfg(all(test, feature = "tasks"))]
mod priority_cell_io_tests;
#[cfg(all(test, feature = "proxy_host"))]
mod proxy_cell_host_tests;
#[cfg(all(test, feature = "reactor", unix))]
mod reactor_cell_io_tests;
#[cfg(all(test, feature = "residency_proxy_host"))]
mod residency_proxy_host_tests;
#[cfg(all(test, feature = "std"))]
mod scheduled_io_tests;
#[cfg(all(test, feature = "tasks"))]
mod task_cell_io_tests;

#[cfg(all(feature = "asset_container", feature = "tasks"))]
pub use asset_container_cell_io::{
    AssetContainerCellIo, AssetContainerCellIoLimits, AssetContainerCellIoLimitsError,
};
#[cfg(feature = "asset_container")]
pub use asset_container_provider::{
    AssetContainerCellProvider, AssetContainerCellRangePlan, AssetContainerProviderError,
    AssetContainerProviderLimits, DEFAULT_ASSET_CONTAINER_MAX_IO_BYTES,
    DEFAULT_ASSET_CONTAINER_MAX_RANGES, DEFAULT_ASSET_CONTAINER_MAX_RANGE_BYTES,
};
#[cfg(feature = "std")]
pub use file_provider::FileCellProvider;
#[cfg(feature = "gpu_proxy")]
pub use gpu_proxy_cell_host::{GpuProxyCellHost, GpuProxyCellHostError};
#[cfg(feature = "std")]
pub use io_adapter::{IoAdapterEvent, IoByteLimits, IoRequestId, IoSchedulerAdapter};
#[cfg(feature = "tasks")]
pub use priority_cell_io::{PrioritizedCellIo, PriorityCellIo};
#[cfg(feature = "proxy_host")]
pub use proxy_cell_host::{CellProxyDecoder, ProxyCellHost, ProxyCellHostError};
#[cfg(all(feature = "reactor", unix))]
pub use reactor_cell_io::ReactorCellIo;
#[cfg(feature = "residency_proxy_host")]
pub use residency_proxy_host::{ResidencyProxySceneHost, ResidencyProxySceneHostError};
#[cfg(feature = "std")]
pub use scheduled_io::ScheduledCellIo;
#[cfg(feature = "tasks")]
pub use task_cell_io::TaskCellIo;
#[cfg(feature = "std")]
pub use threaded_cell_io::ThreadedCellIo;

/// The byte width of the fixed cell manifest header.
pub const CELL_HEADER_SIZE: usize = 52;

const CELL_HEADER_MAGIC: [u8; 4] = *b"PRWC";

/// A request submitted to a [`ChunkProvider`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellReadRequest {
    manifest: CellManifest,
    priority: u32,
    byte_offset: u64,
    expected_checksum: Option<u64>,
}

impl CellReadRequest {
    /// Creates a cell request for the given manifest and stream priority.
    pub const fn new(manifest: CellManifest, priority: u32) -> Self {
        Self {
            manifest,
            priority,
            byte_offset: 0,
            expected_checksum: None,
        }
    }

    /// Returns the target manifest.
    #[must_use]
    pub const fn manifest(self) -> CellManifest {
        self.manifest
    }

    /// Returns the target sector.
    #[must_use]
    pub const fn sector(self) -> SectorId {
        self.manifest.sector
    }

    /// Returns the stream priority converted to the IO request priority type.
    #[must_use]
    pub const fn priority(self) -> u32 {
        self.priority
    }

    /// Adds an optional FNV-1a checksum for the complete cell stream.
    #[must_use]
    pub const fn with_expected_checksum(mut self, checksum: u64) -> Self {
        self.expected_checksum = Some(checksum);
        self
    }

    /// Selects a cell's starting offset within a file or byte-range provider.
    #[must_use]
    pub const fn with_byte_offset(mut self, offset: u64) -> Self {
        self.byte_offset = offset;
        self
    }

    /// Returns the corresponding engine-neutral world IO request.
    #[must_use]
    pub fn io_request(self) -> ReadRequest {
        let request =
            ReadRequest::new(self.byte_offset, self.manifest.io_bytes).with_priority(self.priority);
        match self.expected_checksum {
            Some(checksum) => request.with_expected_checksum(checksum),
            None => request,
        }
    }
}

/// A cancellation view passed to a provider for one poll.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CancellationToken {
    cancelled: bool,
}

impl CancellationToken {
    const fn new(cancelled: bool) -> Self {
        Self { cancelled }
    }

    /// Returns whether the current request has been cancelled.
    #[must_use]
    pub const fn is_cancelled(self) -> bool {
        self.cancelled
    }
}

/// Supplies opaque bytes for a cell request.
pub trait CellProvider {
    /// Provider-specific failure type.
    type Error;

    /// Reads one complete cell byte stream.
    fn read_cell(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error>;
}

/// Low-level provider seam owned by the bounded IO runtime.
pub trait ChunkProvider {
    /// Provider-specific failure type.
    type Error;

    /// Reads one cell's opaque byte stream.
    fn read_chunk(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error>;
}

impl<T> ChunkProvider for T
where
    T: CellProvider,
{
    type Error = T::Error;

    fn read_chunk(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        self.read_cell(request, cancel)
    }
}

/// The scene application boundary. The coordinator never parses the payload.
pub trait SceneHost {
    /// Host-specific failure type.
    type Error;

    /// Commits or replaces a validated cell payload in the host.
    ///
    /// A second call for the same sector is the streaming hot-reload path. The
    /// host owns replacement atomicity and may reject the operation; the
    /// coordinator keeps the previous manifest resident until success.
    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error>;

    /// Classifies a failed load for deterministic trace diagnostics.
    ///
    /// Existing hosts default to [`FailureStage::LoadHost`]. Adapters that
    /// perform decoding or staging may report a more specific stage.
    fn load_failure_stage(&self, _error: &Self::Error) -> FailureStage {
        FailureStage::LoadHost
    }

    /// Formats a stable diagnostic for a rejected load.
    fn load_failure_message(&self, _error: &Self::Error) -> String {
        String::from("scene host rejected load")
    }

    /// Removes a cell from the host.
    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error>;
}

/// A host-provided loaded cell snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadedCell {
    /// The identity and content version known to be loaded.
    pub manifest: CellManifest,
}

impl LoadedCell {
    /// Creates a loaded-state record from a manifest.
    #[must_use]
    pub const fn new(manifest: CellManifest) -> Self {
        Self { manifest }
    }
}

/// A bounded submission failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmitError {
    /// The owning task scope has been cancelled and admits no further work.
    Cancelled,
    /// The runtime has reached its outstanding request capacity.
    Backpressure {
        /// Maximum outstanding requests.
        capacity: usize,
    },
    /// All nonzero request identities have been used by this backend.
    RequestIdExhausted,
    /// Request metadata or provider admission was rejected.
    Provider {
        /// Human-readable failure detail.
        message: String,
    },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("IO task scope was cancelled"),
            Self::Backpressure { capacity } => {
                write!(formatter, "IO capacity {capacity} is full")
            }
            Self::RequestIdExhausted => formatter.write_str("IO request identities exhausted"),
            Self::Provider { message } => formatter.write_str(message),
        }
    }
}

/// Stable identity for one IO request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(pub(crate) u64);

impl RequestId {
    /// Creates an identity for a custom [`CellIo`] backend.
    ///
    /// The backend must never reuse this value during its lifetime.
    pub const fn new(value: NonZeroU64) -> Self {
        Self(value.get())
    }

    /// Returns the numeric identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One result emitted by [`IoRuntime::poll`].
pub enum IoCompletion<E> {
    /// A provider returned bytes.
    Completed {
        /// Request identity.
        request_id: RequestId,
        /// Request metadata.
        request: CellReadRequest,
        /// Opaque cell bytes.
        bytes: Vec<u8>,
    },
    /// The request was cancelled; its payload must not be committed.
    Cancelled {
        /// Request identity.
        request_id: RequestId,
        /// Request metadata.
        request: CellReadRequest,
    },
    /// The provider rejected the request.
    Failed {
        /// Request identity.
        request_id: RequestId,
        /// Request metadata.
        request: CellReadRequest,
        /// Provider failure.
        error: E,
    },
}

impl<E> IoCompletion<E> {
    /// Returns the identity of the retired request.
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::Completed { request_id, .. }
            | Self::Cancelled { request_id, .. }
            | Self::Failed { request_id, .. } => *request_id,
        }
    }

    /// Returns the original metadata of the retired request.
    #[must_use]
    pub const fn request(&self) -> CellReadRequest {
        match self {
            Self::Completed { request, .. }
            | Self::Cancelled { request, .. }
            | Self::Failed { request, .. } => *request,
        }
    }
}

impl<E: fmt::Debug> fmt::Debug for IoCompletion<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed {
                request_id,
                request,
                bytes,
            } => formatter
                .debug_struct("Completed")
                .field("request_id", request_id)
                .field("sector", &request.sector())
                .field("bytes", &bytes.len())
                .finish(),
            Self::Cancelled {
                request_id,
                request,
            } => formatter
                .debug_struct("Cancelled")
                .field("request_id", request_id)
                .field("sector", &request.sector())
                .finish(),
            Self::Failed {
                request_id,
                request,
                error,
            } => formatter
                .debug_struct("Failed")
                .field("request_id", request_id)
                .field("sector", &request.sector())
                .field("error", error)
                .finish(),
        }
    }
}

struct PendingRequest {
    request_id: RequestId,
    request: CellReadRequest,
    cancelled: bool,
}

/// A bounded synchronous compatibility runtime.
///
/// Polling invokes the provider directly and may block. Use [`CellIo`] with a
/// worker-backed provider when coordinating filesystem IO on a frame thread.
pub struct IoRuntime<P: ChunkProvider> {
    provider: P,
    capacity: usize,
    next_request_id: Option<u64>,
    pending: Vec<PendingRequest>,
}

impl<P: ChunkProvider> IoRuntime<P> {
    /// Creates a runtime with a maximum number of outstanding requests.
    pub fn new(provider: P, capacity: usize) -> Result<Self, SubmitError> {
        if capacity == 0 {
            return Err(SubmitError::Backpressure { capacity: 0 });
        }
        Ok(Self {
            provider,
            capacity,
            next_request_id: Some(1),
            pending: Vec::new(),
        })
    }

    /// Submits one request without performing provider work.
    pub fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        if self.pending.len() >= self.capacity {
            return Err(SubmitError::Backpressure {
                capacity: self.capacity,
            });
        }
        let value = self
            .next_request_id
            .ok_or(SubmitError::RequestIdExhausted)?;
        let request_id = RequestId(value);
        self.next_request_id = value.checked_add(1);
        self.pending.push(PendingRequest {
            request_id,
            request,
            cancelled: false,
        });
        Ok(request_id)
    }

    /// Marks a pending request cancelled. The next poll emits [`IoCompletion::Cancelled`].
    pub fn cancel(&mut self, request_id: RequestId) -> bool {
        if let Some(pending) = self
            .pending
            .iter_mut()
            .find(|pending| pending.request_id == request_id)
        {
            if pending.cancelled {
                false
            } else {
                pending.cancelled = true;
                true
            }
        } else {
            false
        }
    }

    /// Returns the number of outstanding requests.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Polls the highest-priority request, preserving FIFO order for ties.
    pub fn poll(&mut self) -> Option<IoCompletion<P::Error>> {
        let index = self
            .pending
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.request
                    .priority()
                    .cmp(&right.request.priority())
                    .then_with(|| right.request_id.cmp(&left.request_id))
            })?
            .0;
        let pending = self.pending.swap_remove(index);
        if pending.cancelled {
            return Some(IoCompletion::Cancelled {
                request_id: pending.request_id,
                request: pending.request,
            });
        }
        let result = self
            .provider
            .read_chunk(&pending.request, CancellationToken::new(false));
        Some(match result {
            Ok(bytes) => IoCompletion::Completed {
                request_id: pending.request_id,
                request: pending.request,
                bytes,
            },
            Err(error) => IoCompletion::Failed {
                request_id: pending.request_id,
                request: pending.request,
                error,
            },
        })
    }
}

/// Bounded cell IO driven by the caller's task context.
///
/// Accepted requests retain their admission until a terminal completion is
/// consumed. A pending nonempty backend must arrange a wake when it can progress.
/// Identities are nonzero, never reused within a backend, and local to that backend.
pub trait CellIo {
    /// Provider or backend failure type.
    type Error;

    /// Admits a request without waiting for its bytes.
    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError>;
    /// Requests cancellation; terminal retirement occurs during polling.
    fn cancel(&mut self, id: RequestId) -> bool;
    /// Counts accepted requests whose terminal event has not been consumed.
    fn in_flight(&self) -> usize;
    /// Polls a terminal cell event or an internal backend error.
    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>>;
}

/// Waits for one terminal cell event using the executor's task context.
///
/// This future borrows the backend exclusively and does not spawn work or park
/// the calling thread. An idle backend may remain pending until its owner drops
/// the future. The backend controls admission, cancellation, and polling cost.
pub async fn next_cell_completion<I: CellIo + ?Sized>(
    io: &mut I,
) -> Result<IoCompletion<I::Error>, I::Error> {
    core::future::poll_fn(|cx| io.poll_completion(cx)).await
}

impl<P: ChunkProvider> CellIo for IoRuntime<P> {
    type Error = P::Error;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        IoRuntime::submit(self, request)
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        IoRuntime::cancel(self, id)
    }

    fn in_flight(&self) -> usize {
        IoRuntime::in_flight(self)
    }

    fn poll_completion(
        &mut self,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        match self.poll() {
            Some(completion) => Poll::Ready(Ok(completion)),
            None => Poll::Pending,
        }
    }
}

/// A stable manifest header encoded at the beginning of every cell stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestHeader {
    /// Stable scene/template key.
    pub scene_key: u128,
    /// Scene content version.
    pub content_version: u64,
    /// Derived payload hash/version.
    pub derived_hash: u128,
    /// Canonical Data Layer membership bits.
    pub data_layers: prism_world_stream::DataLayerMask,
}

impl ManifestHeader {
    /// Creates a header from a stream manifest.
    #[must_use]
    pub const fn from_manifest(manifest: CellManifest) -> Self {
        Self {
            scene_key: manifest.scene_key,
            content_version: manifest.content_version,
            derived_hash: manifest.derived_hash,
            data_layers: manifest.data_layers,
        }
    }

    /// Encodes the fixed header into a new byte vector.
    #[must_use]
    pub fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CELL_HEADER_SIZE);
        bytes.extend_from_slice(&CELL_HEADER_MAGIC);
        bytes.extend_from_slice(&self.scene_key.to_le_bytes());
        bytes.extend_from_slice(&self.content_version.to_le_bytes());
        bytes.extend_from_slice(&self.derived_hash.to_le_bytes());
        bytes.extend_from_slice(&self.data_layers.bits().to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<(Self, &[u8]), HeaderError> {
        if bytes.len() < CELL_HEADER_SIZE {
            return Err(HeaderError::TooShort {
                actual: bytes.len(),
            });
        }
        if bytes[..4] != CELL_HEADER_MAGIC {
            return Err(HeaderError::BadMagic);
        }
        let scene_key = u128::from_le_bytes(bytes[4..20].try_into().expect("fixed header slice"));
        let content_version =
            u64::from_le_bytes(bytes[20..28].try_into().expect("fixed header slice"));
        let derived_hash =
            u128::from_le_bytes(bytes[28..44].try_into().expect("fixed header slice"));
        let data_layers = prism_world_stream::DataLayerMask::from_bits(u64::from_le_bytes(
            bytes[44..52].try_into().expect("fixed header slice"),
        ));
        Ok((
            Self {
                scene_key,
                content_version,
                derived_hash,
                data_layers,
            },
            &bytes[CELL_HEADER_SIZE..],
        ))
    }
}

/// A manifest-header validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeaderError {
    /// The cell stream is shorter than the fixed header.
    TooShort {
        /// Actual byte count.
        actual: usize,
    },
    /// The stream does not begin with `PRWC`.
    BadMagic,
    /// The scene key does not match the manifest.
    SceneKeyMismatch {
        /// Expected manifest value.
        expected: u128,
        /// Encoded value.
        actual: u128,
    },
    /// The content version does not match the manifest.
    ContentVersionMismatch {
        /// Expected manifest value.
        expected: u64,
        /// Encoded value.
        actual: u64,
    },
    /// The derived hash does not match the manifest.
    DerivedHashMismatch {
        /// Expected manifest value.
        expected: u128,
        /// Encoded value.
        actual: u128,
    },
    /// The Data Layer mask does not match the manifest.
    DataLayerMaskMismatch {
        /// Expected manifest value.
        expected: prism_world_stream::DataLayerMask,
        /// Encoded value.
        actual: prism_world_stream::DataLayerMask,
    },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { actual } => write!(formatter, "cell header is short: {actual} bytes"),
            Self::BadMagic => formatter.write_str("cell header has invalid magic"),
            Self::SceneKeyMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scene key mismatch: {actual:#x} != {expected:#x}"
                )
            }
            Self::ContentVersionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "content version mismatch: {actual} != {expected}"
                )
            }
            Self::DerivedHashMismatch { expected, actual } => {
                write!(
                    formatter,
                    "derived hash mismatch: {actual:#x} != {expected:#x}"
                )
            }
            Self::DataLayerMaskMismatch { expected, actual } => write!(
                formatter,
                "Data Layer mask mismatch: {actual:?} != {expected:?}"
            ),
        }
    }
}

fn validate_header(manifest: CellManifest, bytes: &[u8]) -> Result<&[u8], HeaderError> {
    let (header, payload) = ManifestHeader::decode(bytes)?;
    if header.scene_key != manifest.scene_key {
        return Err(HeaderError::SceneKeyMismatch {
            expected: manifest.scene_key,
            actual: header.scene_key,
        });
    }
    if header.content_version != manifest.content_version {
        return Err(HeaderError::ContentVersionMismatch {
            expected: manifest.content_version,
            actual: header.content_version,
        });
    }
    if header.derived_hash != manifest.derived_hash {
        return Err(HeaderError::DerivedHashMismatch {
            expected: manifest.derived_hash,
            actual: header.derived_hash,
        });
    }
    if header.data_layers != manifest.data_layers {
        return Err(HeaderError::DataLayerMaskMismatch {
            expected: manifest.data_layers,
            actual: header.data_layers,
        });
    }
    Ok(payload)
}

/// The operation that caused a trace entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceOp {
    /// A required load was admitted.
    Load,
    /// A prefetched payload was admitted without scene instantiation.
    Prefetch,
    /// A resident cell revision replacement was admitted.
    Reload,
    /// A loaded cell was evicted.
    Evict,
    /// An operation failed.
    Failed,
}

/// The stage at which an operation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureStage {
    /// IO queue capacity rejected the request.
    Backpressure,
    /// The request was cancelled.
    Cancelled,
    /// The provider returned an error.
    Provider,
    /// The fixed manifest header was invalid.
    ManifestHeader,
    /// The scene host rejected a load.
    LoadHost,
    /// The application payload decoder rejected a cell.
    PayloadDecode,
    /// A cell's decoded footprint exceeded its configured budget.
    PayloadBudget,
    /// The typed host rejected preparation before publication.
    CellPrepare,
    /// The scene host rejected an eviction.
    EvictHost,
}

/// An observable event emitted by the coordinator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceEvent {
    /// A frame started planning.
    FrameStarted {
        /// Frame number.
        frame: u64,
    },
    /// A load was submitted to IO.
    LoadSubmitted {
        /// Frame number.
        frame: u64,
        /// Cell identity.
        sector: SectorId,
        /// IO request identity.
        request_id: RequestId,
    },
    /// A cell was committed to the host.
    Loaded {
        /// Frame number that admitted the load.
        frame: u64,
        /// Cell identity.
        sector: SectorId,
    },
    /// A prefetched payload was validated and retained without entering the scene host.
    Prefetched {
        /// Frame number that admitted the prefetch.
        frame: u64,
        /// Cell identity.
        sector: SectorId,
    },
    /// A cell was evicted from the host.
    Evicted {
        /// Frame number.
        frame: u64,
        /// Cell identity.
        sector: SectorId,
    },
    /// An operation failed and did not cross the host boundary.
    Failed {
        /// Frame number.
        frame: u64,
        /// Cell identity.
        sector: SectorId,
        /// Operation category.
        operation: TraceOp,
        /// Failure stage.
        stage: FailureStage,
        /// Stable diagnostic text.
        message: String,
    },
    /// A frame finished with the number of policy actions emitted.
    FrameFinished {
        /// Frame number.
        frame: u64,
        /// Number of stream actions emitted during planning.
        actions: usize,
    },
}

/// Append-only coordinator trace with deterministic replay validation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Trace {
    events: Vec<TraceEvent>,
}

impl Trace {
    /// Creates an empty trace.
    #[must_use]
    pub const fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Appends an event.
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Returns all recorded events.
    #[must_use]
    pub fn as_slice(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Validates frame nesting and returns the number of complete frames.
    pub fn replay(&self) -> Result<usize, TraceError> {
        let mut open = None;
        let mut count = 0;
        let mut started = BTreeSet::new();
        let mut finished = BTreeSet::new();
        for event in &self.events {
            match event {
                TraceEvent::FrameStarted { frame } => {
                    if open.is_some() || !started.insert(*frame) {
                        return Err(TraceError::NestedFrame);
                    }
                    open = Some(*frame);
                }
                TraceEvent::FrameFinished { frame, .. } => {
                    if open != Some(*frame) || !finished.insert(*frame) {
                        return Err(TraceError::FrameMismatch);
                    }
                    open = None;
                    count += 1;
                }
                TraceEvent::LoadSubmitted { frame, .. }
                | TraceEvent::Loaded { frame, .. }
                | TraceEvent::Prefetched { frame, .. }
                | TraceEvent::Evicted { frame, .. }
                | TraceEvent::Failed { frame, .. } => {
                    if open != Some(*frame) && !finished.contains(frame) {
                        return Err(TraceError::EventOutsideFrame);
                    }
                }
            }
        }
        if open.is_some() {
            return Err(TraceError::UnclosedFrame);
        }
        Ok(count)
    }
}

/// A trace replay failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceError {
    /// A frame began while another frame was open.
    NestedFrame,
    /// A finish event did not match the open frame.
    FrameMismatch,
    /// An operation event appeared outside a frame.
    EventOutsideFrame,
    /// The trace ended with an open frame.
    UnclosedFrame,
}

/// The result of one planning pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameReport {
    /// Frame number assigned by the coordinator.
    pub frame: u64,
    /// Number of stream actions produced by the policy.
    pub actions: usize,
    /// Number of IO requests accepted by the runtime.
    pub submitted: usize,
    /// Number of policy submissions rejected by backpressure.
    pub failed: usize,
}

/// Invalid frame input rejected before coordinator state or host state changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramePlanError {
    /// The manifest slice contains duplicate sector identities.
    DuplicateManifest(ManifestValidationError),
    /// The loaded snapshot contains duplicate sector identities.
    DuplicateLoadedCell {
        /// The duplicated sector.
        sector: SectorId,
        /// Index of the first loaded entry.
        first_index: usize,
        /// Index of the duplicate loaded entry.
        duplicate_index: usize,
    },
}

impl fmt::Display for FramePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateManifest(error) => write!(formatter, "{error}"),
            Self::DuplicateLoadedCell {
                sector,
                first_index,
                duplicate_index,
            } => write!(
                formatter,
                "duplicate loaded cell sector {sector:?} at indices {first_index} and {duplicate_index}"
            ),
        }
    }
}

impl core::error::Error for FramePlanError {}

impl From<ManifestValidationError> for FramePlanError {
    fn from(error: ManifestValidationError) -> Self {
        Self::DuplicateManifest(error)
    }
}

struct PendingLoad {
    frame: u64,
    manifest: CellManifest,
    previous: Option<CellManifest>,
    operation: TraceOp,
    prefetch: bool,
    cancel_requested: bool,
}

struct CachedCell {
    manifest: CellManifest,
    bytes: Vec<u8>,
}

/// Coordinates stream policy, bounded IO, manifest validation, and scene host calls.
pub struct CellFrameCoordinator<I: CellIo> {
    io: I,
    stream: prism_world_stream::StreamScheduler,
    known_loaded: BTreeMap<SectorId, CellManifest>,
    pending: BTreeMap<RequestId, PendingLoad>,
    trace: Trace,
    frame: u64,
    prefetch_cache: BTreeMap<SectorId, CachedCell>,
    prefetch_cache_bytes: u64,
    prefetch_cache_limit: u64,
}

/// Synchronous compatibility coordinator over a [`ChunkProvider`].
pub type WorldFrameCoordinator<P> = CellFrameCoordinator<IoRuntime<P>>;

impl<P: ChunkProvider> CellFrameCoordinator<IoRuntime<P>> {
    /// Creates a coordinator with a bounded IO runtime.
    pub fn new(provider: P, io_capacity: usize) -> Result<Self, SubmitError> {
        Ok(Self::with_io(IoRuntime::new(provider, io_capacity)?))
    }

    /// Runs one synchronous provider request and applies its terminal result.
    pub fn poll_io<H: SceneHost>(&mut self, host: &mut H) -> bool
    where
        P::Error: fmt::Debug,
    {
        let Some(completion) = self.io.poll() else {
            return false;
        };
        self.apply_completion(completion, host);
        true
    }
}

impl<I: CellIo> CellFrameCoordinator<I> {
    /// Creates a coordinator using an explicitly selected IO backend.
    pub fn with_io(io: I) -> Self {
        Self::with_io_and_prefetch_cache(io, DEFAULT_PREFETCH_CACHE_BYTES)
    }

    /// Creates a coordinator with an explicit bounded prefetch cache.
    ///
    /// A zero limit disables retention while preserving prefetch validation;
    /// prefetched bytes are then discarded after the provider completes.
    pub fn with_io_and_prefetch_cache(io: I, prefetch_cache_limit: u64) -> Self {
        Self {
            io,
            stream: prism_world_stream::StreamScheduler::new(),
            known_loaded: BTreeMap::new(),
            pending: BTreeMap::new(),
            trace: Trace::new(),
            frame: 0,
            prefetch_cache: BTreeMap::new(),
            prefetch_cache_bytes: 0,
            prefetch_cache_limit,
        }
    }

    /// Returns the bytes currently retained by the prefetch cache.
    #[must_use]
    pub const fn prefetch_cache_bytes(&self) -> u64 {
        self.prefetch_cache_bytes
    }

    /// Returns the configured prefetch cache limit.
    #[must_use]
    pub const fn prefetch_cache_limit(&self) -> u64 {
        self.prefetch_cache_limit
    }

    /// Plans a frame, submits load actions, and applies evictions to `host`.
    pub fn try_plan_frame<H: SceneHost>(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        loaded: &[LoadedCell],
        budget: StreamBudget,
        host: &mut H,
    ) -> Result<FrameReport, FramePlanError> {
        validate_manifests(manifests)?;
        validate_loaded_cells(loaded)?;
        self.frame = self.frame.wrapping_add(1);
        let frame = self.frame;
        self.trace.push(TraceEvent::FrameStarted { frame });
        self.sync_loaded(loaded);
        let plan = self
            .stream
            .try_plan(demands, manifests, budget)
            .expect("manifest validation completed before stream planning");
        let mut report = FrameReport {
            frame,
            actions: plan.actions.len(),
            submitted: 0,
            failed: 0,
        };
        for action in plan.actions {
            match action {
                StreamAction::Load(request) => {
                    let manifest = manifests
                        .iter()
                        .find(|manifest| manifest.sector == request.sector)
                        .expect("stream request must refer to a validated manifest");
                    self.submit_stream_request(
                        frame,
                        request,
                        *manifest,
                        false,
                        false,
                        host,
                        &mut report,
                    );
                }
                StreamAction::Reload(request) => {
                    let manifest = manifests
                        .iter()
                        .find(|manifest| manifest.sector == request.sector)
                        .expect("stream request must refer to a validated manifest");
                    self.submit_stream_request(
                        frame,
                        request,
                        *manifest,
                        false,
                        true,
                        host,
                        &mut report,
                    );
                }
                StreamAction::Prefetch(request) => {
                    let manifest = manifests
                        .iter()
                        .find(|manifest| manifest.sector == request.sector)
                        .expect("stream request must refer to a validated manifest");
                    self.submit_stream_request(
                        frame,
                        request,
                        *manifest,
                        true,
                        false,
                        host,
                        &mut report,
                    );
                }
                StreamAction::Unload(sector) => {
                    if host.evict_cell(sector).is_ok() {
                        self.known_loaded.remove(&sector);
                        let _ = self.remove_prefetch_cache(sector);
                        self.trace.push(TraceEvent::Evicted { frame, sector });
                    } else {
                        if let Some(manifest) = self.known_loaded.get(&sector).copied() {
                            self.stream.mark_resident_manifest(manifest);
                        }
                        self.record_failure(
                            frame,
                            sector,
                            TraceOp::Evict,
                            FailureStage::EvictHost,
                            "scene host rejected eviction",
                        );
                        report.failed += 1;
                    }
                }
            }
        }
        self.trace.push(TraceEvent::FrameFinished {
            frame,
            actions: report.actions,
        });
        Ok(report)
    }

    /// Plans a frame and panics on duplicate manifest or loaded-cell input.
    ///
    /// New callers that receive external or dynamically assembled snapshots
    /// should use [`Self::try_plan_frame`] to propagate validation errors.
    pub fn plan_frame<H: SceneHost>(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        loaded: &[LoadedCell],
        budget: StreamBudget,
        host: &mut H,
    ) -> FrameReport {
        self.try_plan_frame(demands, manifests, loaded, budget, host)
            .expect("world frame inputs must contain unique sectors")
    }

    /// Polls the backend once and commits one validated terminal result to `host`.
    ///
    /// `Pending` preserves all pending loads. Internal backend errors are returned
    /// to the caller; per-request failures are recorded in the trace and retired.
    /// Polling is nonblocking when the selected [`CellIo`] implementation is.
    pub fn poll_io_with_context<H: SceneHost>(
        &mut self,
        cx: &mut Context<'_>,
        host: &mut H,
    ) -> Poll<Result<(), I::Error>>
    where
        I::Error: fmt::Debug,
    {
        match self.io.poll_completion(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(completion)) => {
                self.apply_completion(completion, host);
                Poll::Ready(Ok(()))
            }
        }
    }

    fn remove_prefetch_cache(&mut self, sector: SectorId) -> Option<CachedCell> {
        let cached = self.prefetch_cache.remove(&sector)?;
        self.prefetch_cache_bytes = self
            .prefetch_cache_bytes
            .saturating_sub(cached.bytes.len() as u64);
        Some(cached)
    }

    fn retain_prefetch(&mut self, manifest: CellManifest, bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        let _ = self.remove_prefetch_cache(manifest.sector);
        if size == 0 || size > self.prefetch_cache_limit {
            return;
        }
        while self.prefetch_cache_bytes.saturating_add(size) > self.prefetch_cache_limit {
            let Some(sector) = self.prefetch_cache.keys().next().copied() else {
                return;
            };
            let _ = self.remove_prefetch_cache(sector);
        }
        self.prefetch_cache_bytes = self.prefetch_cache_bytes.saturating_add(size);
        self.prefetch_cache
            .insert(manifest.sector, CachedCell { manifest, bytes });
    }

    fn submit_stream_request<H: SceneHost>(
        &mut self,
        frame: u64,
        request: prism_world_stream::StreamRequest,
        manifest: CellManifest,
        prefetch: bool,
        explicit_reload: bool,
        host: &mut H,
        report: &mut FrameReport,
    ) {
        let previous_loaded = self.known_loaded.get(&request.sector).copied();
        let revision_changed = previous_loaded.is_some_and(|old| {
            old.scene_key != manifest.scene_key
                || old.content_version != manifest.content_version
                || old.derived_hash != manifest.derived_hash
                || old.data_layers != manifest.data_layers
        });
        let operation = if prefetch {
            TraceOp::Prefetch
        } else if explicit_reload || revision_changed {
            TraceOp::Reload
        } else {
            TraceOp::Load
        };
        let previous = (prefetch || operation == TraceOp::Reload)
            .then_some(previous_loaded)
            .flatten();

        if let Some(cached) = self.prefetch_cache.get(&request.sector) {
            if explicit_reload || cached.manifest != manifest {
                let _ = self.remove_prefetch_cache(request.sector);
            } else if prefetch {
                self.stream.mark_prefetched_manifest(manifest);
                return;
            } else if let Some(cached) = self.remove_prefetch_cache(request.sector) {
                match validate_header(manifest, &cached.bytes) {
                    Ok(payload) => match host.load_cell(&manifest, payload) {
                        Ok(()) => {
                            self.known_loaded.insert(manifest.sector, manifest);
                            self.stream.mark_resident_manifest(manifest);
                            self.trace.push(TraceEvent::Loaded {
                                frame,
                                sector: manifest.sector,
                            });
                        }
                        Err(error) => {
                            let stage = host.load_failure_stage(&error);
                            let message = host.load_failure_message(&error);
                            self.retain_prefetch(cached.manifest, cached.bytes);
                            self.restore_stream_state(request.sector, previous);
                            self.record_failure(frame, request.sector, operation, stage, &message);
                            report.failed += 1;
                        }
                    },
                    Err(error) => {
                        self.restore_stream_state(request.sector, previous);
                        self.record_failure(
                            frame,
                            request.sector,
                            operation,
                            FailureStage::ManifestHeader,
                            &error.to_string(),
                        );
                        report.failed += 1;
                    }
                }
                return;
            }
        }

        let io_request = CellReadRequest::new(manifest, u32::from(request.priority.0));
        match self.io.submit(io_request) {
            Ok(request_id) => {
                self.pending.insert(
                    request_id,
                    PendingLoad {
                        frame,
                        manifest,
                        previous,
                        operation,
                        prefetch,
                        cancel_requested: false,
                    },
                );
                self.trace.push(TraceEvent::LoadSubmitted {
                    frame,
                    sector: request.sector,
                    request_id,
                });
                report.submitted += 1;
            }
            Err(error) => {
                self.restore_stream_state(request.sector, previous);
                let stage = match &error {
                    SubmitError::Backpressure { .. } => FailureStage::Backpressure,
                    _ => FailureStage::Provider,
                };
                self.record_failure(frame, request.sector, operation, stage, &error.to_string());
                report.failed += 1;
            }
        }
    }

    fn apply_completion<H: SceneHost>(&mut self, completion: IoCompletion<I::Error>, host: &mut H)
    where
        I::Error: fmt::Debug,
    {
        match completion {
            IoCompletion::Completed {
                request_id,
                request,
                bytes,
            } => {
                let pending = self.pending.remove(&request_id);
                let Some(pending) = pending else {
                    return;
                };
                if pending.cancel_requested {
                    self.restore_stream_state(pending.manifest.sector, pending.previous);
                    self.record_failure(
                        pending.frame,
                        request.sector(),
                        pending.operation,
                        FailureStage::Cancelled,
                        "IO request cancelled before a late completion was consumed",
                    );
                    return;
                }
                let previous = pending.previous;
                let operation = pending.operation;
                if pending.prefetch {
                    match validate_header(pending.manifest, &bytes) {
                        Ok(_) => {
                            self.retain_prefetch(pending.manifest, bytes);
                            self.stream.mark_prefetched_manifest(pending.manifest);
                            self.trace.push(TraceEvent::Prefetched {
                                frame: pending.frame,
                                sector: pending.manifest.sector,
                            });
                        }
                        Err(error) => {
                            self.restore_stream_state(pending.manifest.sector, previous);
                            self.record_failure(
                                pending.frame,
                                pending.manifest.sector,
                                operation,
                                FailureStage::ManifestHeader,
                                &error.to_string(),
                            );
                        }
                    }
                } else {
                    match validate_header(pending.manifest, &bytes) {
                        Ok(payload) => match host.load_cell(&pending.manifest, payload) {
                            Ok(()) => {
                                self.known_loaded
                                    .insert(pending.manifest.sector, pending.manifest);
                                self.stream.mark_resident_manifest(pending.manifest);
                                self.trace.push(TraceEvent::Loaded {
                                    frame: pending.frame,
                                    sector: pending.manifest.sector,
                                });
                            }
                            Err(error) => {
                                let stage = host.load_failure_stage(&error);
                                let message = host.load_failure_message(&error);
                                self.restore_stream_state(pending.manifest.sector, previous);
                                self.record_failure(
                                    pending.frame,
                                    pending.manifest.sector,
                                    operation,
                                    stage,
                                    &message,
                                );
                            }
                        },
                        Err(error) => {
                            self.restore_stream_state(pending.manifest.sector, previous);
                            self.record_failure(
                                pending.frame,
                                pending.manifest.sector,
                                operation,
                                FailureStage::ManifestHeader,
                                &error.to_string(),
                            );
                        }
                    }
                }
            }
            IoCompletion::Cancelled {
                request_id,
                request,
            } => {
                let pending = self.pending.remove(&request_id);
                let (frame, previous, operation) = pending
                    .map_or((self.frame, None, TraceOp::Load), |pending| {
                        (pending.frame, pending.previous, pending.operation)
                    });
                self.restore_stream_state(request.sector(), previous);
                self.record_failure(
                    frame,
                    request.sector(),
                    operation,
                    FailureStage::Cancelled,
                    "IO request cancelled",
                );
            }
            IoCompletion::Failed {
                request_id,
                request,
                error,
            } => {
                let pending = self.pending.remove(&request_id);
                if let Some(pending) = pending.as_ref().filter(|pending| pending.cancel_requested) {
                    let frame = pending.frame;
                    let previous = pending.previous;
                    let operation = pending.operation;
                    self.restore_stream_state(request.sector(), previous);
                    self.record_failure(
                        frame,
                        request.sector(),
                        operation,
                        FailureStage::Cancelled,
                        "IO request cancelled before a late failure was consumed",
                    );
                    return;
                }
                let (frame, previous, operation) = pending
                    .map_or((self.frame, None, TraceOp::Load), |pending| {
                        (pending.frame, pending.previous, pending.operation)
                    });
                self.restore_stream_state(request.sector(), previous);
                self.record_failure(
                    frame,
                    request.sector(),
                    operation,
                    FailureStage::Provider,
                    &format!("{error:?}"),
                );
            }
        }
    }

    /// Cancels the pending request for `sector`, if any.
    pub fn cancel_sector(&mut self, sector: SectorId) -> bool {
        let Some((request_id, pending)) = self
            .pending
            .iter_mut()
            .find(|(_, pending)| pending.manifest.sector == sector)
        else {
            return false;
        };
        if pending.cancel_requested {
            return false;
        }
        pending.cancel_requested = true;
        let request_id = *request_id;
        let _ = self.io.cancel(request_id);
        true
    }

    /// Returns the observable trace.
    #[must_use]
    pub const fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Returns the number of outstanding IO requests.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.io.in_flight()
    }

    /// Schedules a budgeted reload for a resident sector after external content
    /// dependency invalidation. The existing resident revision remains the
    /// rollback target until the replacement is committed.
    pub fn request_reload(&mut self, sector: SectorId) -> bool {
        self.known_loaded.contains_key(&sector) && self.stream.request_reload(sector)
    }

    /// Converts scene-template dependency invalidations into budgeted reload
    /// requests for currently resident cells whose scene key is the invalidated
    /// template GUID. Returned sectors are unique and sorted.
    #[cfg(feature = "scene_templates")]
    pub fn schedule_template_invalidations(
        &mut self,
        invalidations: &[SceneTemplateInvalidation],
    ) -> Vec<SectorId> {
        let affected: BTreeSet<_> = invalidations
            .iter()
            .filter(|event| event.reloadable)
            .map(|event| event.dependent.to_u128())
            .collect();
        let sectors: Vec<_> = self
            .known_loaded
            .iter()
            .filter_map(|(&sector, manifest)| {
                affected.contains(&manifest.scene_key).then_some(sector)
            })
            .collect();
        for &sector in &sectors {
            self.stream.request_reload(sector);
        }
        sectors
    }

    /// Applies one scene-template event/snapshot batch to `cache`, then queues
    /// budgeted reloads for its newly computed resident dependents. A rejected
    /// cache batch leaves scheduler state unchanged.
    #[cfg(feature = "scene_templates")]
    pub fn apply_scene_template_assets(
        &mut self,
        cache: &mut SceneTemplateCache,
        assets: &mut prism_asset::Assets<prism_scene::SceneTemplate>,
    ) -> Result<SceneTemplateApplyOutcome, SceneTemplateBatchError> {
        let invalidations = cache.apply_assets(assets)?;
        let reload_sectors = self.schedule_template_invalidations(&invalidations);
        Ok(SceneTemplateApplyOutcome {
            invalidations,
            reload_sectors,
        })
    }

    fn sync_loaded(&mut self, loaded: &[LoadedCell]) {
        let current: BTreeSet<_> = loaded.iter().map(|cell| cell.manifest.sector).collect();
        let stale: Vec<_> = self
            .known_loaded
            .keys()
            .copied()
            .filter(|sector| !current.contains(sector))
            .collect();
        for sector in stale {
            self.known_loaded.remove(&sector);
            self.stream.mark_unloaded(sector);
        }
        for cell in loaded {
            let _ = self.remove_prefetch_cache(cell.manifest.sector);
            self.known_loaded
                .insert(cell.manifest.sector, cell.manifest);
            // This is an observation of host state, not a replacement commit;
            // retain any explicit invalidation awaiting successful reload.
            self.stream.restore_resident_manifest(cell.manifest);
        }
    }

    fn restore_stream_state(&mut self, sector: SectorId, previous: Option<CellManifest>) {
        match previous {
            Some(manifest) => self.stream.restore_resident_manifest(manifest),
            None => self.stream.mark_unloaded(sector),
        }
    }

    fn record_failure(
        &mut self,
        frame: u64,
        sector: SectorId,
        operation: TraceOp,
        stage: FailureStage,
        message: &str,
    ) {
        self.trace.push(TraceEvent::Failed {
            frame,
            sector,
            operation,
            stage,
            message: String::from(message),
        });
    }
}

fn validate_loaded_cells(loaded: &[LoadedCell]) -> Result<(), FramePlanError> {
    let mut indexed: Vec<_> = loaded
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.manifest.sector, index))
        .collect();
    indexed.sort_unstable_by_key(|(sector, index)| (*sector, *index));
    for pair in indexed.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(FramePlanError::DuplicateLoadedCell {
                sector: pair[0].0,
                first_index: pair[0].1,
                duplicate_index: pair[1].1,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use prism_world_stream::{AabbI64, DataLayerMask, GlobalPos};

    fn manifest(x: i64, scene_key: u128, version: u64, derived_hash: u128) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
                .expect("valid bounds"),
            scene_key,
            content_version: version,
            derived_hash,
            io_bytes: 1,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        }
    }

    fn bytes(manifest: CellManifest, payload: &[u8]) -> Vec<u8> {
        let mut result = ManifestHeader::from_manifest(manifest).encode();
        result.extend_from_slice(payload);
        result
    }

    struct MemoryProvider {
        bytes: BTreeMap<SectorId, Vec<u8>>,
        calls: Arc<AtomicUsize>,
    }

    impl CellProvider for MemoryProvider {
        type Error = ();

        fn read_cell(
            &self,
            request: &CellReadRequest,
            cancel: CancellationToken,
        ) -> Result<Vec<u8>, Self::Error> {
            if cancel.is_cancelled() {
                return Err(());
            }
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.bytes.get(&request.sector()).cloned().ok_or(())
        }
    }

    #[derive(Default)]
    struct Host {
        loaded: Vec<SectorId>,
        evicted: Vec<SectorId>,
        payloads: Vec<Vec<u8>>,
        fail_loads: usize,
    }

    impl SceneHost for Host {
        type Error = ();

        fn load_cell(
            &mut self,
            manifest: &CellManifest,
            payload: &[u8],
        ) -> Result<(), Self::Error> {
            self.loaded.push(manifest.sector);
            self.payloads.push(payload.to_vec());
            if self.fail_loads > 0 {
                self.fail_loads -= 1;
                return Err(());
            }
            Ok(())
        }

        fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
            self.evicted.push(sector);
            Ok(())
        }
    }

    fn coordinator(
        manifest: CellManifest,
        capacity: usize,
    ) -> WorldFrameCoordinator<MemoryProvider> {
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(manifest.sector, bytes(manifest, b"opaque"));
        WorldFrameCoordinator::new(
            MemoryProvider {
                bytes: bytes_by_sector,
                calls: Arc::new(AtomicUsize::new(0)),
            },
            capacity,
        )
        .expect("runtime config")
    }

    #[test]
    fn duplicate_manifest_and_loaded_inputs_fail_before_runtime_mutation() {
        let first = manifest(0, 7, 1, 9);
        let duplicate = CellManifest {
            scene_key: 8,
            io_bytes: 64,
            ..first
        };
        let demand = DemandSource::new(first.sector, 0, 1, 10).expect("demand");
        let mut coordinator = coordinator(first, 1);
        let mut host = Host::default();

        assert_eq!(
            coordinator.try_plan_frame(
                &[demand],
                &[first, duplicate],
                &[],
                StreamBudget {
                    io_bytes: 64,
                    max_actions: 2,
                },
                &mut host,
            ),
            Err(FramePlanError::DuplicateManifest(
                ManifestValidationError::DuplicateSector {
                    sector: first.sector,
                    first_index: 0,
                    duplicate_index: 1,
                }
            ))
        );
        assert!(coordinator.trace().as_slice().is_empty());
        assert_eq!(coordinator.in_flight(), 0);
        assert!(host.loaded.is_empty());

        assert_eq!(
            coordinator.try_plan_frame(
                &[demand],
                &[first],
                &[LoadedCell::new(first), LoadedCell::new(duplicate)],
                StreamBudget {
                    io_bytes: 64,
                    max_actions: 2,
                },
                &mut host,
            ),
            Err(FramePlanError::DuplicateLoadedCell {
                sector: first.sector,
                first_index: 0,
                duplicate_index: 1,
            })
        );
        assert!(coordinator.trace().as_slice().is_empty());
        assert_eq!(coordinator.in_flight(), 0);
        assert!(host.loaded.is_empty());
    }

    #[test]
    fn memory_provider_loads_then_evicts_and_trace_replays() {
        let loaded_manifest = manifest(0, 7, 2, 9);
        let mut coordinator = coordinator(loaded_manifest, 2);
        let mut host = Host::default();
        let demand = DemandSource::new(loaded_manifest.sector, 0, 1, 10).expect("demand");
        let report = coordinator.plan_frame(
            &[demand],
            &[loaded_manifest],
            &[],
            StreamBudget {
                io_bytes: 2,
                max_actions: 2,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(host.loaded, vec![loaded_manifest.sector]);

        let report = coordinator.plan_frame(
            &[],
            &[loaded_manifest],
            &[LoadedCell::new(loaded_manifest)],
            StreamBudget {
                io_bytes: 2,
                max_actions: 2,
            },
            &mut host,
        );
        assert_eq!(report.actions, 1);
        assert_eq!(host.evicted, vec![loaded_manifest.sector]);
        assert_eq!(coordinator.trace().replay(), Ok(2));
    }

    #[test]
    fn prefetch_validates_and_promotes_from_bounded_cache_without_scene_load() {
        let manifest = CellManifest {
            io_bytes: (CELL_HEADER_SIZE + b"prefetched".len()) as u64,
            ..manifest(1, 17, 1, 9)
        };
        let payload = bytes(manifest, b"prefetched");
        let calls = Arc::new(AtomicUsize::new(0));
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(manifest.sector, payload);
        let provider = MemoryProvider {
            bytes: bytes_by_sector,
            calls: Arc::clone(&calls),
        };
        let mut coordinator = CellFrameCoordinator::with_io_and_prefetch_cache(
            IoRuntime::new(provider, 1).expect("runtime config"),
            1024,
        );
        let mut host = Host::default();
        let center = SectorId::new(0, 0, 0, 0);
        let demand = DemandSource::new(center, 0, 2, 10)
            .expect("demand")
            .with_prediction(manifest.sector);
        let budget = StreamBudget {
            io_bytes: 64,
            max_actions: 1,
        };

        let prefetch = coordinator
            .try_plan_frame(&[demand], &[manifest], &[], budget, &mut host)
            .expect("prefetch plan");
        assert_eq!(prefetch.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert!(host.loaded.is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(coordinator.prefetch_cache_bytes(), manifest.io_bytes);
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Prefetched { sector, .. } if *sector == manifest.sector
        )));

        let repeated = coordinator
            .try_plan_frame(&[demand], &[manifest], &[], budget, &mut host)
            .expect("cached prefetch plan");
        assert_eq!(repeated.actions, 0);
        assert_eq!(repeated.submitted, 0);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let required = DemandSource::new(manifest.sector, 0, 1, 10).expect("demand");
        let promotion = coordinator
            .try_plan_frame(&[required], &[manifest], &[], budget, &mut host)
            .expect("promotion plan");
        assert_eq!(promotion.submitted, 0);
        assert_eq!(promotion.failed, 0);
        assert!(host.loaded.contains(&manifest.sector));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(coordinator.prefetch_cache_bytes(), 0);
        assert_eq!(coordinator.in_flight(), 0);
    }

    #[test]
    fn changed_manifest_reloads_existing_cell_and_keeps_old_state_on_failure() {
        let old = manifest(0, 7, 1, 9);
        let updated = manifest(0, 7, 2, 10);
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(updated.sector, bytes(updated, b"new"));
        let provider = MemoryProvider {
            bytes: bytes_by_sector,
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host::default();
        let demand = DemandSource::new(updated.sector, 0, 1, 10).expect("demand");

        let report = coordinator.plan_frame(
            &[demand],
            &[updated],
            &[LoadedCell::new(old)],
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(host.loaded, vec![updated.sector]);
        assert!(coordinator
            .trace()
            .as_slice()
            .iter()
            .any(|event| matches!(event, TraceEvent::Loaded { .. })));

        // A later revision with a bad header must not make the already loaded
        // revision disappear from the coordinator's residency state.
        let broken = manifest(0, 7, 3, 11);
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(broken.sector, bytes(updated, b"stale-header"));
        let provider = MemoryProvider {
            bytes: bytes_by_sector,
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut failing = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut failing_host = Host::default();
        let first = failing.plan_frame(
            &[demand],
            &[broken],
            &[LoadedCell::new(updated)],
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut failing_host,
        );
        assert_eq!(first.submitted, 1);
        assert!(failing.poll_io(&mut failing_host));
        assert!(failing.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Failed {
                operation: TraceOp::Reload,
                stage: FailureStage::ManifestHeader,
                ..
            }
        )));
        let retry = failing.plan_frame(
            &[demand],
            &[broken],
            &[LoadedCell::new(updated)],
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut failing_host,
        );
        assert_eq!(retry.submitted, 1);
    }

    #[cfg(feature = "scene_templates")]
    #[test]
    fn template_dependency_invalidation_enters_budgeted_reload_commit_path() {
        let resident = manifest(0, 0x5151, 1, 9);
        let provider = MemoryProvider {
            bytes: BTreeMap::from([(resident.sector, bytes(resident, b"replacement"))]),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host::default();
        let loaded = [LoadedCell::new(resident)];
        coordinator.sync_loaded(&loaded);

        let invalidation = SceneTemplateInvalidation {
            changed: prism_asset::StableGuid::from_u128(resident.scene_key),
            dependent: prism_asset::StableGuid::from_u128(resident.scene_key),
            reloadable: true,
        };
        let scheduled = coordinator.schedule_template_invalidations(&[invalidation]);
        assert_eq!(scheduled, vec![resident.sector]);
        let report = coordinator.plan_frame(
            &[DemandSource::new(resident.sector, 0, 1, 10).unwrap()],
            &[resident],
            &loaded,
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(host.loaded, vec![resident.sector]);
        let next = coordinator.plan_frame(
            &[DemandSource::new(resident.sector, 0, 1, 10).unwrap()],
            &[resident],
            &loaded,
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(next.actions, 0);
    }

    #[cfg(feature = "scene_templates")]
    #[test]
    fn scene_template_asset_batch_api_atomically_queues_resident_reload() {
        let guid = prism_asset::StableGuid::from_u128(0x5252);
        let resident = manifest(0, guid.to_u128(), 1, 9);
        let provider = MemoryProvider {
            bytes: BTreeMap::from([(resident.sector, bytes(resident, b"replacement"))]),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        coordinator.sync_loaded(&[LoadedCell::new(resident)]);
        let mut cache = SceneTemplateCache::new();
        let mut assets = prism_asset::Assets::new();
        let mut invalid = prism_scene::SceneTemplate::empty(guid);
        invalid.schema = 0;
        assets.insert(invalid);
        assert!(matches!(
            coordinator.apply_scene_template_assets(&mut cache, &mut assets),
            Err(SceneTemplateBatchError::InvalidSnapshot(_))
        ));
        let demand = DemandSource::new(resident.sector, 0, 1, 10).unwrap();
        let unmodified = coordinator.plan_frame(
            &[demand],
            &[resident],
            &[LoadedCell::new(resident)],
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut Host::default(),
        );
        assert_eq!(unmodified.submitted, 0);

        assets.insert(prism_scene::SceneTemplate::empty(guid));

        let outcome = coordinator
            .apply_scene_template_assets(&mut cache, &mut assets)
            .unwrap();
        assert_eq!(outcome.reload_sectors, vec![resident.sector]);
        assert_eq!(outcome.invalidations.len(), 1);
        assert_eq!(outcome.invalidations[0].changed, guid);
        assert_eq!(outcome.invalidations[0].dependent, guid);
        assert!(outcome.invalidations[0].reloadable);
        assert!(cache.get(guid).is_some());
    }

    #[cfg(feature = "scene_templates")]
    #[test]
    fn explicit_same_revision_reload_failure_preserves_resident_and_retries_as_reload() {
        let resident = manifest(0, 0x5252, 1, 9);
        let provider = MemoryProvider {
            bytes: BTreeMap::from([(resident.sector, bytes(resident, b"replacement"))]),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host {
            fail_loads: 1,
            ..Host::default()
        };
        let loaded = [LoadedCell::new(resident)];
        coordinator.sync_loaded(&loaded);

        let invalidation = SceneTemplateInvalidation {
            changed: prism_asset::StableGuid::from_u128(resident.scene_key),
            dependent: prism_asset::StableGuid::from_u128(resident.scene_key),
            reloadable: true,
        };
        assert_eq!(
            coordinator.schedule_template_invalidations(&[invalidation]),
            vec![resident.sector]
        );
        let demand = DemandSource::new(resident.sector, 0, 1, 10).expect("demand");
        let budget = StreamBudget {
            io_bytes: 1,
            max_actions: 1,
        };

        let first = coordinator.plan_frame(&[demand], &[resident], &loaded, budget, &mut host);
        assert_eq!(first.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Failed {
                sector,
                operation: TraceOp::Reload,
                stage: FailureStage::LoadHost,
                ..
            } if *sector == resident.sector
        )));

        let retry = coordinator.plan_frame(&[demand], &[resident], &loaded, budget, &mut host);
        assert_eq!(retry.submitted, 1);
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::LoadSubmitted { frame, sector, .. }
                if *frame == retry.frame && *sector == resident.sector
        )));
        assert_eq!(
            coordinator.known_loaded.get(&resident.sector),
            Some(&resident)
        );
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(host.loaded, vec![resident.sector, resident.sector]);
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Loaded { frame, sector }
                if *frame == retry.frame && *sector == resident.sector
        )));
    }

    #[cfg(feature = "scene_templates")]
    #[test]
    fn explicit_reload_discards_matching_prefetch_cache_and_reads_fresh_payload() {
        let resident = manifest(0, 0x5353, 1, 9);
        let prefetched = bytes(resident, b"stale-prefetch");
        let fresh = bytes(resident, b"fresh-reload");
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MemoryProvider {
            bytes: BTreeMap::from([(resident.sector, fresh)]),
            calls: calls.clone(),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host::default();
        let loaded = [LoadedCell::new(resident)];
        coordinator.sync_loaded(&loaded);
        coordinator.retain_prefetch(resident, prefetched);

        let invalidation = SceneTemplateInvalidation {
            changed: prism_asset::StableGuid::from_u128(resident.scene_key),
            dependent: prism_asset::StableGuid::from_u128(resident.scene_key),
            reloadable: true,
        };
        assert_eq!(
            coordinator.schedule_template_invalidations(&[invalidation]),
            vec![resident.sector]
        );
        let report = coordinator.plan_frame(
            &[DemandSource::new(resident.sector, 0, 1, 10).expect("demand")],
            &[resident],
            &loaded,
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert_eq!(coordinator.prefetch_cache_bytes(), 0);
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(host.payloads, vec![b"fresh-reload".to_vec()]);
    }

    #[test]
    fn manifest_header_mismatch_is_failed_without_scene_parse() {
        let layer_mask = DataLayerMask::from_layers(&[prism_world_stream::DataLayerId::new(2)])
            .expect("valid layer mask");
        let expected = CellManifest {
            data_layers: layer_mask,
            ..manifest(0, 7, 2, 9)
        };
        let wrong = CellManifest {
            data_layers: DataLayerMask::EMPTY,
            ..manifest(0, 7, 2, 10)
        };
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(expected.sector, bytes(wrong, b"opaque"));
        let provider = MemoryProvider {
            bytes: bytes_by_sector,
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host::default();
        coordinator.plan_frame(
            &[DemandSource::new(expected.sector, 0, 1, 1)
                .expect("demand")
                .with_data_layers(layer_mask)],
            &[expected],
            &[],
            StreamBudget {
                io_bytes: 1,
                max_actions: 1,
            },
            &mut host,
        );
        coordinator.poll_io(&mut host);
        assert!(host.loaded.is_empty());
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Failed {
                stage: FailureStage::ManifestHeader,
                ..
            }
        )));
    }

    #[test]
    fn cancellation_and_backpressure_are_observable() {
        let first = manifest(0, 1, 1, 1);
        let second = manifest(1, 2, 1, 2);
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(first.sector, bytes(first, b"a"));
        bytes_by_sector.insert(second.sector, bytes(second, b"b"));
        let mut coordinator = WorldFrameCoordinator::new(
            MemoryProvider {
                bytes: bytes_by_sector,
                calls: Arc::new(AtomicUsize::new(0)),
            },
            1,
        )
        .expect("runtime config");
        let mut host = Host::default();
        let demand = DemandSource::new(first.sector, 1, 2, 10).expect("demand");
        let report = coordinator.plan_frame(
            &[demand],
            &[first, second],
            &[],
            StreamBudget {
                io_bytes: 4,
                max_actions: 4,
            },
            &mut host,
        );
        assert_eq!(report.failed, 1);
        assert!(coordinator.cancel_sector(first.sector));
        coordinator.poll_io(&mut host);
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Failed {
                stage: FailureStage::Cancelled,
                ..
            }
        )));
        assert!(coordinator.trace().as_slice().iter().any(|event| matches!(
            event,
            TraceEvent::Failed {
                stage: FailureStage::Backpressure,
                ..
            }
        )));
    }
}
