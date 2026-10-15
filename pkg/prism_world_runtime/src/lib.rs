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

use prism_world_io::ReadRequest;
use prism_world_stream::{CellManifest, DemandSource, SectorId, StreamAction, StreamBudget};

/// The byte width of the fixed cell manifest header.
pub const CELL_HEADER_SIZE: usize = 44;

const CELL_HEADER_MAGIC: [u8; 4] = *b"PRWC";

/// A request submitted to a [`ChunkProvider`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellReadRequest {
    manifest: CellManifest,
    priority: u32,
}

impl CellReadRequest {
    fn new(manifest: CellManifest, priority: u32) -> Self {
        Self { manifest, priority }
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

    /// Returns the corresponding engine-neutral world IO request.
    #[must_use]
    pub fn io_request(self) -> ReadRequest {
        ReadRequest::new(0, self.manifest.io_bytes).with_priority(self.priority)
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

    /// Commits a validated cell payload to the host.
    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error>;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitError {
    /// The runtime has reached its outstanding request capacity.
    Backpressure {
        /// Maximum outstanding requests.
        capacity: usize,
    },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backpressure { capacity } => {
                write!(formatter, "IO capacity {capacity} is full")
            }
        }
    }
}

/// Stable identity for one IO request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u64);

impl RequestId {
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
    /// The request was cancelled before provider work started.
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
    sequence: u64,
    cancelled: bool,
}

/// A bounded, deterministic, poll-driven IO runtime.
pub struct IoRuntime<P: ChunkProvider> {
    provider: P,
    capacity: usize,
    next_request_id: u64,
    next_sequence: u64,
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
            next_request_id: 1,
            next_sequence: 0,
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
        let request_id = RequestId(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1);
        self.pending.push(PendingRequest {
            request_id,
            request,
            sequence: self.next_sequence,
            cancelled: false,
        });
        self.next_sequence = self.next_sequence.wrapping_add(1);
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
                    .then_with(|| right.sequence.cmp(&left.sequence))
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

/// A stable manifest header encoded at the beginning of every cell stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestHeader {
    /// Stable scene/template key.
    pub scene_key: u128,
    /// Scene content version.
    pub content_version: u64,
    /// Derived payload hash/version.
    pub derived_hash: u128,
}

impl ManifestHeader {
    /// Creates a header from a stream manifest.
    #[must_use]
    pub const fn from_manifest(manifest: CellManifest) -> Self {
        Self {
            scene_key: manifest.scene_key,
            content_version: manifest.content_version,
            derived_hash: manifest.derived_hash,
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
        Ok((
            Self {
                scene_key,
                content_version,
                derived_hash,
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
    Ok(payload)
}

/// The operation that caused a trace entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceOp {
    /// A required or prefetched load was admitted.
    Load,
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

struct PendingLoad {
    frame: u64,
    manifest: CellManifest,
}

/// Coordinates stream policy, bounded IO, manifest validation, and scene host calls.
pub struct WorldFrameCoordinator<P: ChunkProvider> {
    io: IoRuntime<P>,
    stream: prism_world_stream::StreamScheduler,
    known_loaded: BTreeMap<SectorId, CellManifest>,
    pending: BTreeMap<RequestId, PendingLoad>,
    trace: Trace,
    frame: u64,
}

impl<P: ChunkProvider> WorldFrameCoordinator<P> {
    /// Creates a coordinator with a bounded IO runtime.
    pub fn new(provider: P, io_capacity: usize) -> Result<Self, SubmitError> {
        Ok(Self {
            io: IoRuntime::new(provider, io_capacity)?,
            stream: prism_world_stream::StreamScheduler::new(),
            known_loaded: BTreeMap::new(),
            pending: BTreeMap::new(),
            trace: Trace::new(),
            frame: 0,
        })
    }

    /// Plans a frame, submits load actions, and applies evictions to `host`.
    pub fn plan_frame<H: SceneHost>(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        loaded: &[LoadedCell],
        budget: StreamBudget,
        host: &mut H,
    ) -> FrameReport {
        self.frame = self.frame.wrapping_add(1);
        let frame = self.frame;
        self.trace.push(TraceEvent::FrameStarted { frame });
        self.sync_loaded(loaded);
        let plan = self.stream.plan(demands, manifests, budget);
        let mut report = FrameReport {
            frame,
            actions: plan.actions.len(),
            submitted: 0,
            failed: 0,
        };
        for action in plan.actions {
            match action {
                StreamAction::Load(request) | StreamAction::Prefetch(request) => {
                    let Some(manifest) = manifests.iter().find(|m| m.sector == request.sector)
                    else {
                        self.record_failure(
                            frame,
                            request.sector,
                            TraceOp::Failed,
                            FailureStage::Provider,
                            "stream action has no manifest",
                        );
                        self.stream.mark_unloaded(request.sector);
                        report.failed += 1;
                        continue;
                    };
                    let io_request = CellReadRequest::new(*manifest, u32::from(request.priority.0));
                    match self.io.submit(io_request) {
                        Ok(request_id) => {
                            self.pending.insert(
                                request_id,
                                PendingLoad {
                                    frame,
                                    manifest: *manifest,
                                },
                            );
                            self.trace.push(TraceEvent::LoadSubmitted {
                                frame,
                                sector: request.sector,
                                request_id,
                            });
                            report.submitted += 1;
                        }
                        Err(SubmitError::Backpressure { capacity }) => {
                            self.stream.mark_unloaded(request.sector);
                            self.record_failure(
                                frame,
                                request.sector,
                                TraceOp::Load,
                                FailureStage::Backpressure,
                                &format!("IO capacity {capacity} is full"),
                            );
                            report.failed += 1;
                        }
                    }
                }
                StreamAction::Unload(sector) => {
                    if host.evict_cell(sector).is_ok() {
                        self.known_loaded.remove(&sector);
                        self.trace.push(TraceEvent::Evicted { frame, sector });
                    } else {
                        if let Some(manifest) = self.known_loaded.get(&sector).copied() {
                            self.stream.mark_resident(manifest.sector);
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
        report
    }

    /// Polls one submitted IO request and, on success, commits it to `host`.
    pub fn poll_io<H: SceneHost>(&mut self, host: &mut H) -> bool {
        let Some(completion) = self.io.poll() else {
            return false;
        };
        match completion {
            IoCompletion::Completed {
                request_id,
                request,
                bytes,
            } => {
                let pending = self.pending.remove(&request_id);
                let Some(pending) = pending else {
                    return true;
                };
                match validate_header(request.manifest(), &bytes) {
                    Ok(payload) => match host.load_cell(&pending.manifest, payload) {
                        Ok(()) => {
                            self.known_loaded
                                .insert(pending.manifest.sector, pending.manifest);
                            self.stream.mark_resident(pending.manifest.sector);
                            self.trace.push(TraceEvent::Loaded {
                                frame: pending.frame,
                                sector: pending.manifest.sector,
                            });
                        }
                        Err(_) => {
                            self.stream.mark_unloaded(pending.manifest.sector);
                            self.record_failure(
                                pending.frame,
                                pending.manifest.sector,
                                TraceOp::Load,
                                FailureStage::LoadHost,
                                "scene host rejected load",
                            );
                        }
                    },
                    Err(error) => {
                        self.stream.mark_unloaded(pending.manifest.sector);
                        self.record_failure(
                            pending.frame,
                            pending.manifest.sector,
                            TraceOp::Load,
                            FailureStage::ManifestHeader,
                            &error.to_string(),
                        );
                    }
                }
            }
            IoCompletion::Cancelled {
                request_id,
                request,
            } => {
                self.pending.remove(&request_id);
                self.stream.mark_unloaded(request.sector());
                self.record_failure(
                    self.frame,
                    request.sector(),
                    TraceOp::Load,
                    FailureStage::Cancelled,
                    "IO request cancelled",
                );
            }
            IoCompletion::Failed {
                request_id,
                request,
                error: _,
            } => {
                let frame = self
                    .pending
                    .remove(&request_id)
                    .map_or(self.frame, |pending| pending.frame);
                self.stream.mark_unloaded(request.sector());
                self.record_failure(
                    frame,
                    request.sector(),
                    TraceOp::Load,
                    FailureStage::Provider,
                    "cell provider rejected IO",
                );
            }
        }
        true
    }

    /// Cancels the pending request for `sector`, if any.
    pub fn cancel_sector(&mut self, sector: SectorId) -> bool {
        let Some(request_id) = self
            .pending
            .iter()
            .find(|(_, pending)| pending.manifest.sector == sector)
            .map(|(request_id, _)| *request_id)
        else {
            return false;
        };
        self.io.cancel(request_id)
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
            self.known_loaded
                .insert(cell.manifest.sector, cell.manifest);
            self.stream.mark_resident(cell.manifest.sector);
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use prism_world_stream::{AabbI64, GlobalPos};

    fn manifest(x: i64, scene_key: u128, version: u64, derived_hash: u128) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
                .expect("valid bounds"),
            scene_key,
            content_version: version,
            derived_hash,
            io_bytes: 1,
        }
    }

    fn bytes(manifest: CellManifest, payload: &[u8]) -> Vec<u8> {
        let mut result = ManifestHeader::from_manifest(manifest).encode();
        result.extend_from_slice(payload);
        result
    }

    struct MemoryProvider {
        bytes: BTreeMap<SectorId, Vec<u8>>,
        calls: AtomicUsize,
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
                calls: AtomicUsize::new(0),
            },
            capacity,
        )
        .expect("runtime config")
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
    fn manifest_header_mismatch_is_failed_without_scene_parse() {
        let expected = manifest(0, 7, 2, 9);
        let wrong = manifest(0, 7, 2, 10);
        let mut bytes_by_sector = BTreeMap::new();
        bytes_by_sector.insert(expected.sector, bytes(wrong, b"opaque"));
        let provider = MemoryProvider {
            bytes: bytes_by_sector,
            calls: AtomicUsize::new(0),
        };
        let mut coordinator = WorldFrameCoordinator::new(provider, 1).expect("runtime config");
        let mut host = Host::default();
        coordinator.plan_frame(
            &[DemandSource::new(expected.sector, 0, 1, 1).expect("demand")],
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
                calls: AtomicUsize::new(0),
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
