//! Bounded asynchronous cell reads over the host-owned named IO lane.
//!
//! The adapter owns only a small dispatch queue. It never creates a thread per
//! request: accepted work is dispatched to [`prism_tasks::NamedThreads`]' IO
//! lane, while the adapter's active limit prevents that shared lane from being
//! flooded by this provider. A request remains admitted until its terminal
//! completion is consumed.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::task::{Context, Poll, Waker};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Mutex;

use prism_tasks::{NamedThreads, Priority, PriorityCell, ThreadCategory};

use crate::{
    AssetContainerCellProvider, AssetContainerProviderError, CancellationToken, CellIo,
    CellProvider, CellReadRequest, IoCompletion, PriorityCellIo, RequestId, SubmitError,
};

/// Bounds for accepted requests and their declared cell byte budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetContainerCellIoLimits {
    /// Maximum number of requests retained before terminal consumption.
    pub max_in_flight: usize,
    /// Maximum sum of declared `CellManifest::io_bytes` for retained requests.
    pub max_declared_bytes: u64,
    /// Maximum number of requests dispatched to the shared IO lane at once.
    pub max_active: usize,
}

impl AssetContainerCellIoLimits {
    /// Creates validated adapter limits.
    pub const fn new(
        max_in_flight: usize,
        max_declared_bytes: u64,
        max_active: usize,
    ) -> Result<Self, AssetContainerCellIoLimitsError> {
        if max_in_flight == 0 || max_declared_bytes == 0 || max_active == 0 {
            return Err(AssetContainerCellIoLimitsError::ZeroLimit);
        }
        if max_active > max_in_flight {
            return Err(AssetContainerCellIoLimitsError::ActiveExceedsAdmission {
                max_active,
                max_in_flight,
            });
        }
        Ok(Self {
            max_in_flight,
            max_declared_bytes,
            max_active,
        })
    }
}

/// Invalid bounds supplied to [`AssetContainerCellIoLimits::new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetContainerCellIoLimitsError {
    /// Every limit must be nonzero.
    ZeroLimit,
    /// Active dispatch cannot exceed retained admission.
    ActiveExceedsAdmission {
        /// Requested active dispatch bound.
        max_active: usize,
        /// Requested retained request bound.
        max_in_flight: usize,
    },
}

impl core::fmt::Display for AssetContainerCellIoLimitsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroLimit => {
                formatter.write_str("asset-container cell IO limits must be nonzero")
            }
            Self::ActiveExceedsAdmission {
                max_active,
                max_in_flight,
            } => write!(
                formatter,
                "active limit {max_active} exceeds admission limit {max_in_flight}"
            ),
        }
    }
}

impl std::error::Error for AssetContainerCellIoLimitsError {}

struct QueuedRequest {
    id: RequestId,
    request: CellReadRequest,
    priority: Option<Arc<PriorityCell>>,
    sequence: u64,
    cancelled: Arc<AtomicBool>,
}

enum RecordState {
    Queued(QueuedRequest),
    Dispatched { cancelled: Arc<AtomicBool> },
    Ready(IoCompletion<AssetContainerProviderError>),
}

struct State {
    records: HashMap<RequestId, RecordState>,
    ready: VecDeque<RequestId>,
    waiters: Vec<Waker>,
    retained_bytes: u64,
    active: usize,
    next_sequence: u64,
}

struct Shared {
    provider: AssetContainerCellProvider,
    workers: NamedThreads,
    limits: AssetContainerCellIoLimits,
    state: Mutex<State>,
}

impl Shared {
    fn new(
        provider: AssetContainerCellProvider,
        workers: NamedThreads,
        limits: AssetContainerCellIoLimits,
    ) -> Arc<Self> {
        Arc::new(Self {
            provider,
            workers,
            limits,
            state: Mutex::new(State {
                records: HashMap::new(),
                ready: VecDeque::new(),
                waiters: Vec::new(),
                retained_bytes: 0,
                active: 0,
                next_sequence: 0,
            }),
        })
    }

    fn wake_waiters(&self) {
        let waiters = {
            let mut state = self.state.lock().unwrap();
            core::mem::take(&mut state.waiters)
        };
        for waiter in waiters {
            waiter.wake();
        }
    }

    fn complete(
        self: &Arc<Self>,
        id: RequestId,
        request: CellReadRequest,
        cancelled: Arc<AtomicBool>,
        result: Result<Vec<u8>, AssetContainerProviderError>,
    ) {
        {
            let mut state = self.state.lock().unwrap();
            if !state.records.contains_key(&id) {
                return;
            }
            state.active = state.active.saturating_sub(1);
            let completion = if cancelled.load(AtomicOrdering::Acquire) {
                IoCompletion::Cancelled {
                    request_id: id,
                    request,
                }
            } else {
                match result {
                    Ok(bytes) => IoCompletion::Completed {
                        request_id: id,
                        request,
                        bytes,
                    },
                    Err(error) => IoCompletion::Failed {
                        request_id: id,
                        request,
                        error,
                    },
                }
            };
            state.records.insert(id, RecordState::Ready(completion));
            state.ready.push_back(id);
        }
        self.wake_waiters();
        self.pump();
    }

    fn pump(self: &Arc<Self>) {
        loop {
            let work = {
                let mut state = self.state.lock().unwrap();
                if state.active >= self.limits.max_active {
                    None
                } else {
                    let selected = state
                        .records
                        .values()
                        .filter_map(|record| match record {
                            RecordState::Queued(request) => Some(request),
                            RecordState::Dispatched { .. } | RecordState::Ready(_) => None,
                        })
                        .min_by(|left, right| compare_queued(left, right));
                    let Some(selected) = selected else {
                        return;
                    };
                    let id = selected.id;
                    let (request, cancelled) = match state.records.remove(&id) {
                        Some(RecordState::Queued(queued)) => (queued.request, queued.cancelled),
                        _ => unreachable!("queued record disappeared while locked"),
                    };
                    state.records.insert(
                        id,
                        RecordState::Dispatched {
                            cancelled: Arc::clone(&cancelled),
                        },
                    );
                    state.active += 1;
                    Some((id, request, cancelled))
                }
            };
            let Some((id, request, cancelled)) = work else {
                return;
            };
            let shared = Arc::clone(self);
            let provider = self.provider.clone();
            self.workers.dispatch(ThreadCategory::Io, move || {
                let result = if cancelled.load(AtomicOrdering::Acquire) {
                    Err(AssetContainerProviderError::Cancelled)
                } else {
                    provider.read_cell(&request, CancellationToken::new(false))
                };
                shared.complete(id, request, cancelled, result);
            });
        }
    }

    fn cancel(self: &Arc<Self>, id: RequestId) -> bool {
        let mut notify = false;
        {
            let mut state = self.state.lock().unwrap();
            let Some(record) = state.records.get_mut(&id) else {
                return false;
            };
            match record {
                RecordState::Queued(queued) => {
                    queued.cancelled.store(true, AtomicOrdering::Release);
                    let request = queued.request;
                    *record = RecordState::Ready(IoCompletion::Cancelled {
                        request_id: id,
                        request,
                    });
                    state.ready.push_back(id);
                    notify = true;
                }
                RecordState::Dispatched { cancelled, .. } => {
                    if cancelled.swap(true, AtomicOrdering::AcqRel) {
                        return false;
                    }
                }
                RecordState::Ready(completion) => {
                    let _ = completion;
                    return false;
                }
            }
        }
        if notify {
            self.wake_waiters();
        }
        true
    }

    fn poll(&self, cx: &mut Context<'_>) -> Poll<IoCompletion<AssetContainerProviderError>> {
        let completion = {
            let mut state = self.state.lock().unwrap();
            let Some(id) = state.ready.pop_front() else {
                if !state
                    .waiters
                    .iter()
                    .any(|waiter| waiter.will_wake(cx.waker()))
                {
                    state.waiters.push(cx.waker().clone());
                }
                return Poll::Pending;
            };
            let record = state.records.remove(&id).expect("ready record exists");
            let completion = match record {
                RecordState::Ready(completion) => completion,
                RecordState::Queued(_) | RecordState::Dispatched { .. } => {
                    unreachable!("only ready records are queued for polling")
                }
            };
            state.retained_bytes -= completion_bytes(&completion);
            completion
        };
        Poll::Ready(completion)
    }
}

fn completion_bytes(completion: &IoCompletion<AssetContainerProviderError>) -> u64 {
    match completion {
        IoCompletion::Completed { request, .. }
        | IoCompletion::Cancelled { request, .. }
        | IoCompletion::Failed { request, .. } => request.manifest().io_bytes,
    }
}

fn compare_queued(left: &QueuedRequest, right: &QueuedRequest) -> Ordering {
    let left_priority = left
        .priority
        .as_ref()
        .map_or(Priority::Normal, |cell| cell.get());
    let right_priority = right
        .priority
        .as_ref()
        .map_or(Priority::Normal, |cell| cell.get());
    right_priority
        .cmp(&left_priority)
        .then_with(|| right.request.priority().cmp(&left.request.priority()))
        .then_with(|| left.sequence.cmp(&right.sequence))
}

/// A bounded asynchronous adapter for [`AssetContainerCellProvider`].
pub struct AssetContainerCellIo {
    shared: Arc<Shared>,
    next_id: AtomicU64,
}

impl AssetContainerCellIo {
    /// Creates an adapter using the caller-owned named worker lanes.
    pub fn new(
        provider: AssetContainerCellProvider,
        workers: NamedThreads,
        limits: AssetContainerCellIoLimits,
    ) -> Self {
        Self {
            shared: Shared::new(provider, workers, limits),
            next_id: AtomicU64::new(1),
        }
    }

    /// Returns the current retained declared byte count.
    #[must_use]
    pub fn retained_declared_bytes(&self) -> u64 {
        self.shared.state.lock().unwrap().retained_bytes
    }

    /// Returns the configured adapter limits.
    #[must_use]
    pub fn limits(&self) -> AssetContainerCellIoLimits {
        self.shared.limits
    }

    fn submit_inner(
        &mut self,
        request: CellReadRequest,
        priority: Option<Arc<PriorityCell>>,
    ) -> Result<RequestId, SubmitError> {
        let declared = request.manifest().io_bytes;
        let id_value = self
            .next_id
            .fetch_update(AtomicOrdering::AcqRel, AtomicOrdering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| SubmitError::RequestIdExhausted)?;
        let id = RequestId::new(core::num::NonZeroU64::new(id_value).unwrap());
        {
            let mut state = self.shared.state.lock().unwrap();
            if state.records.len() >= self.shared.limits.max_in_flight
                || state
                    .retained_bytes
                    .checked_add(declared)
                    .is_none_or(|bytes| bytes > self.shared.limits.max_declared_bytes)
            {
                return Err(SubmitError::Backpressure {
                    capacity: self.shared.limits.max_in_flight,
                });
            }
            let sequence = state.next_sequence;
            state.next_sequence = state.next_sequence.wrapping_add(1);
            state.retained_bytes += declared;
            let cancelled = Arc::new(AtomicBool::new(false));
            state.records.insert(
                id,
                RecordState::Queued(QueuedRequest {
                    id,
                    request,
                    priority,
                    sequence,
                    cancelled,
                }),
            );
        }
        self.shared.pump();
        Ok(id)
    }
}

impl CellIo for AssetContainerCellIo {
    type Error = AssetContainerProviderError;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        self.submit_inner(request, None)
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        let cancelled = self.shared.cancel(id);
        if cancelled {
            self.shared.pump();
        }
        cancelled
    }

    fn in_flight(&self) -> usize {
        self.shared.state.lock().unwrap().records.len()
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        self.shared.poll(cx).map(Ok)
    }
}

impl PriorityCellIo for AssetContainerCellIo {
    fn submit_with_priority(
        &mut self,
        request: CellReadRequest,
        priority: Arc<PriorityCell>,
    ) -> Result<RequestId, SubmitError> {
        self.submit_inner(request, Some(priority))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManifestHeader, CELL_HEADER_SIZE};
    use alloc::sync::Arc;
    use core::task::Poll;
    use prism_asset::StableGuid;
    use prism_asset_container::{
        ContainerReader, ContainerWriter, MemoryRangeSource, OverlaySet, RangeSource,
        RangeSourceError, RuntimeManifest,
    };
    use prism_tasks::NamedThreadsConfig;
    use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

    #[derive(Debug)]
    struct ReadGate {
        block: AtomicBool,
        entered: AtomicBool,
        released: AtomicBool,
        reads: AtomicUsize,
    }

    #[derive(Debug)]
    struct RecordingSource {
        bytes: Arc<[u8]>,
        names: Arc<Mutex<Vec<String>>>,
        gate: Option<Arc<ReadGate>>,
    }

    impl RangeSource for RecordingSource {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<(), RangeSourceError> {
            self.names.lock().unwrap().push(
                std::thread::current()
                    .name()
                    .unwrap_or("unnamed")
                    .to_owned(),
            );
            if let Some(gate) = &self.gate {
                if gate.block.load(StdOrdering::Acquire) {
                    gate.entered.store(true, StdOrdering::Release);
                    while !gate.released.load(StdOrdering::Acquire) {
                        std::thread::yield_now();
                    }
                }
                gate.reads.fetch_add(1, StdOrdering::AcqRel);
            }
            let start = usize::try_from(offset).map_err(|_| RangeSourceError::OutOfBounds)?;
            let end = start
                .checked_add(out.len())
                .ok_or(RangeSourceError::OutOfBounds)?;
            let bytes = self
                .bytes
                .get(start..end)
                .ok_or(RangeSourceError::OutOfBounds)?;
            out.copy_from_slice(bytes);
            Ok(())
        }
    }

    fn recording_provider(
        manifest: CellManifest,
        gate: Option<Arc<ReadGate>>,
    ) -> (
        AssetContainerCellProvider,
        Arc<Mutex<Vec<String>>>,
        Option<Arc<ReadGate>>,
    ) {
        let mut payload = ManifestHeader::from_manifest(manifest).encode();
        payload.extend_from_slice(b"test payload");
        let mut writer = ContainerWriter::new();
        writer
            .add(StableGuid::from_u128(manifest.scene_key), payload)
            .unwrap();
        let names = Arc::new(Mutex::new(Vec::new()));
        let reader = ContainerReader::from_range_source(Arc::new(RecordingSource {
            bytes: Arc::from(writer.finish().unwrap()),
            names: Arc::clone(&names),
            gate: gate.clone(),
        }))
        .unwrap();
        names.lock().unwrap().clear();
        let mut overlays = OverlaySet::new();
        overlays.mount("test", 1, reader);
        let runtime = overlays.runtime_manifest().unwrap();
        if let Some(gate) = &gate {
            gate.block.store(true, StdOrdering::Release);
        }
        (
            AssetContainerCellProvider::with_max_io_bytes(runtime, overlays, manifest.io_bytes)
                .unwrap(),
            names,
            gate,
        )
    }

    fn manifest(scene_key: u128, io_bytes: usize) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, 1, 2, scene_key as i64),
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
            scene_key,
            content_version: 1,
            derived_hash: 2,
            io_bytes: io_bytes as u64,
            data_layers: DataLayerMask::EMPTY,
        }
    }

    fn provider(manifest: CellManifest) -> AssetContainerCellProvider {
        let mut bytes = ManifestHeader::from_manifest(manifest).encode();
        bytes.extend_from_slice(b"test payload");
        let mut writer = ContainerWriter::new();
        writer
            .add(StableGuid::from_u128(manifest.scene_key), bytes)
            .unwrap();
        let reader = ContainerReader::from_range_source(Arc::new(MemoryRangeSource::new(
            writer.finish().unwrap(),
        )))
        .unwrap();
        let mut overlays = OverlaySet::new();
        overlays.mount("test", 1, reader);
        let runtime: RuntimeManifest = overlays.runtime_manifest().unwrap();
        AssetContainerCellProvider::with_max_io_bytes(runtime, overlays, manifest.io_bytes).unwrap()
    }

    fn limits(max_in_flight: usize, max_bytes: u64, active: usize) -> AssetContainerCellIoLimits {
        AssetContainerCellIoLimits::new(max_in_flight, max_bytes, active).unwrap()
    }

    fn poll_once(
        io: &mut AssetContainerCellIo,
    ) -> Poll<Result<IoCompletion<AssetContainerProviderError>, AssetContainerProviderError>> {
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        io.poll_completion(&mut cx)
    }

    fn wait_completion(io: &mut AssetContainerCellIo) -> IoCompletion<AssetContainerProviderError> {
        for _ in 0..10_000 {
            if let Poll::Ready(result) = poll_once(io) {
                return result.unwrap();
            }
            std::thread::yield_now();
        }
        panic!("timed out waiting for cell completion")
    }

    #[derive(Debug)]
    struct WakeFlag(AtomicBool);

    impl std::task::Wake for WakeFlag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, StdOrdering::Release);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, StdOrdering::Release);
        }
    }

    #[test]
    fn bounds_retain_count_and_declared_bytes_until_consumed() {
        let m = manifest(1, CELL_HEADER_SIZE + 12);
        let mut io = AssetContainerCellIo::new(
            provider(m),
            NamedThreads::with_config(NamedThreadsConfig {
                io_threads: 1,
                async_compute_threads: 1,
            }),
            limits(2, m.io_bytes, 1),
        );
        io.submit(CellReadRequest::new(m, 0)).unwrap();
        assert_eq!(io.in_flight(), 1);
        assert_eq!(io.retained_declared_bytes(), m.io_bytes);
        assert!(matches!(
            io.submit(CellReadRequest::new(m, 0)),
            Err(SubmitError::Backpressure { .. })
        ));
        let _ = wait_completion(&mut io);
        assert_eq!(io.in_flight(), 0);
        assert_eq!(io.retained_declared_bytes(), 0);
    }

    #[test]
    fn queued_cancellation_does_not_dispatch_read() {
        let m = manifest(2, CELL_HEADER_SIZE + 12);
        let gate = Arc::new(ReadGate {
            block: AtomicBool::new(false),
            entered: AtomicBool::new(false),
            released: AtomicBool::new(false),
            reads: AtomicUsize::new(0),
        });
        let (provider, _names, gate) = recording_provider(m, Some(Arc::clone(&gate)));
        let gate = gate.unwrap();
        let mut io =
            AssetContainerCellIo::new(provider, NamedThreads::new(), limits(2, m.io_bytes * 2, 1));
        let first = io.submit(CellReadRequest::new(m, 0)).unwrap();
        for _ in 0..10_000 {
            if gate.entered.load(StdOrdering::Acquire) {
                break;
            }
            std::thread::yield_now();
        }
        assert!(gate.entered.load(StdOrdering::Acquire));
        let second = io.submit(CellReadRequest::new(m, 0)).unwrap();
        assert!(io.cancel(second));
        gate.released.store(true, StdOrdering::Release);
        let mut completed = false;
        let mut cancelled = false;
        for _ in 0..10_000 {
            if let Poll::Ready(Ok(completion)) = poll_once(&mut io) {
                match completion {
                    IoCompletion::Completed { request_id, .. } => {
                        assert_eq!(request_id, first);
                        completed = true;
                        break;
                    }
                    IoCompletion::Cancelled { request_id, .. } => {
                        assert_eq!(request_id, second);
                        cancelled = true;
                    }
                    IoCompletion::Failed { .. } => panic!("first request unexpectedly failed"),
                }
            }
            std::thread::yield_now();
        }
        assert!(completed);
        let reads_before_second_poll = gate.reads.load(StdOrdering::Acquire);
        if !cancelled {
            let second_completion = wait_completion(&mut io);
            assert!(matches!(
                second_completion,
                IoCompletion::Cancelled { request_id, .. } if request_id == second
            ));
        }
        assert_eq!(
            gate.reads.load(StdOrdering::Acquire),
            reads_before_second_poll
        );
        assert!(!io.cancel(second));
        assert!(io.cancel(first) || io.in_flight() == 0);
    }

    #[test]
    fn dispatched_cancellation_suppresses_success() {
        let m = manifest(3, CELL_HEADER_SIZE + 12);
        let mut io =
            AssetContainerCellIo::new(provider(m), NamedThreads::new(), limits(1, m.io_bytes, 1));
        let id = io.submit(CellReadRequest::new(m, 0)).unwrap();
        assert!(io.cancel(id));
        assert!(
            matches!(wait_completion(&mut io), IoCompletion::Cancelled { request_id, .. } if request_id == id)
        );
    }

    #[test]
    fn provider_error_identity_is_preserved() {
        let good = manifest(4, CELL_HEADER_SIZE + 12);
        let bad = manifest(0xDEAD, good.io_bytes as usize);
        let mut io = AssetContainerCellIo::new(
            provider(good),
            NamedThreads::new(),
            limits(1, bad.io_bytes, 1),
        );
        io.submit(CellReadRequest::new(bad, 0)).unwrap();
        assert!(matches!(
            wait_completion(&mut io),
            IoCompletion::Failed {
                error: AssetContainerProviderError::Manifest(_),
                ..
            }
        ));
    }

    #[test]
    fn worker_execution_uses_the_named_io_lane() {
        let m = manifest(7, CELL_HEADER_SIZE + 12);
        let (provider, names, _gate) = recording_provider(m, None);
        let mut io =
            AssetContainerCellIo::new(provider, NamedThreads::new(), limits(1, m.io_bytes, 1));
        io.submit(CellReadRequest::new(m, 0)).unwrap();
        let _ = wait_completion(&mut io);
        assert!(names
            .lock()
            .unwrap()
            .iter()
            .any(|name| name.starts_with("prism-io-")));
    }

    #[test]
    fn pending_poll_is_woken_by_terminal_completion() {
        let m = manifest(8, CELL_HEADER_SIZE + 12);
        let mut io =
            AssetContainerCellIo::new(provider(m), NamedThreads::new(), limits(1, m.io_bytes, 1));
        let flag = Arc::new(WakeFlag(AtomicBool::new(false)));
        let waker = std::task::Waker::from(Arc::clone(&flag));
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(io.poll_completion(&mut cx), Poll::Pending));
        io.submit(CellReadRequest::new(m, 0)).unwrap();
        for _ in 0..10_000 {
            if flag.0.load(StdOrdering::Acquire) {
                break;
            }
            std::thread::yield_now();
        }
        assert!(flag.0.load(StdOrdering::Acquire));
        assert!(matches!(
            poll_once(&mut io),
            Poll::Ready(Ok(IoCompletion::Completed { .. }))
        ));
    }

    #[test]
    fn queued_priority_uses_live_priority_cell() {
        let m1 = manifest(5, CELL_HEADER_SIZE + 12);
        let m2 = manifest(6, CELL_HEADER_SIZE + 12);
        let workers = NamedThreads::with_config(NamedThreadsConfig {
            io_threads: 1,
            async_compute_threads: 1,
        });
        let mut io =
            AssetContainerCellIo::new(provider(m1), workers, limits(3, m1.io_bytes * 3, 1));
        let low = Arc::new(PriorityCell::new(Priority::Low));
        let high = Arc::new(PriorityCell::new(Priority::Low));
        let first = io
            .submit_with_priority(CellReadRequest::new(m1, 0), Arc::clone(&low))
            .unwrap();
        let low_id = io
            .submit_with_priority(CellReadRequest::new(m2, 0), Arc::clone(&low))
            .unwrap();
        let high_id = io
            .submit_with_priority(CellReadRequest::new(m2, 0), Arc::clone(&high))
            .unwrap();
        high.boost_to(Priority::Critical);
        assert!(matches!(
            wait_completion(&mut io),
            IoCompletion::Completed { request_id, .. } if request_id == first
        ));
        assert!(matches!(
            wait_completion(&mut io),
            IoCompletion::Failed { request_id, .. } if request_id == high_id
        ));
        assert!(matches!(
            wait_completion(&mut io),
            IoCompletion::Failed { request_id, .. } if request_id == low_id
        ));
    }
}
