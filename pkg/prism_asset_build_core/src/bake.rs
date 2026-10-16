//! Bounded in-process scheduling and runtime caching for `RuntimeDerived` bake
//! kernels.
//!
//! [`BakeService`] owns only an explicitly configured, process-local cache. It
//! never writes DDC/CAS, manifests, release packages, or remote services. A
//! full service rejects immediately; it does not create an unbounded waiting
//! queue.

use crate::types::{ArtifactKey, ContentHash};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use prism_tasks::{Counter, TaskPool};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, RwLock};

const BAKE_KEY_PROTOCOL_VERSION: u32 = 1;
const MAX_BAKE_WORKERS: usize = 64;

/// Caller-supplied stable identity for the input represented by a bake request.
///
/// The bytes or hash are part of the request key exactly as supplied. The
/// service never substitutes a timestamp, random value, path, or machine
/// state. When using [`Self::Hash`], the caller is responsible for ensuring
/// that the hash describes `BakeRequest::input_bytes`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StableInputKey {
    /// Canonical, stable bytes selected by the caller as the input identity.
    Bytes(Vec<u8>),
    /// A caller-computed stable content hash for the input.
    Hash(ContentHash),
}

impl StableInputKey {
    /// Creates a byte-based stable input identity by copying `bytes`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::Bytes(bytes.to_vec())
    }

    /// Creates a hash-based stable input identity.
    #[must_use]
    pub const fn from_hash(hash: ContentHash) -> Self {
        Self::Hash(hash)
    }
}

/// Complete semantic input and output contract for one in-process bake.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BakeRequest {
    /// Stable input identity supplied by the caller.
    pub stable_input: StableInputKey,
    /// Owned bytes passed to the kernel.
    pub input_bytes: Vec<u8>,
    /// Stable identity of the bake kernel implementation.
    pub kernel_id: u128,
    /// Stable identity of the recipe being applied.
    pub recipe_id: u128,
    /// Version of the kernel/recipe contract.
    pub version: u32,
    /// Already-canonical recipe settings.
    pub canonical_settings: Vec<u8>,
    /// Runtime-derived output kind.
    pub output_kind: u128,
    /// Runtime-derived output schema version.
    pub output_schema: u32,
    /// Required output alignment, as a non-zero power of two.
    pub output_alignment: u32,
}

impl BakeRequest {
    /// Creates a bake request. Settings must already be canonical; this API
    /// does not guess defaults or normalize caller data.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor makes every deterministic key and descriptor field explicit"
    )]
    pub fn new(
        stable_input: StableInputKey,
        input_bytes: Vec<u8>,
        kernel_id: u128,
        recipe_id: u128,
        version: u32,
        canonical_settings: Vec<u8>,
        output_kind: u128,
        output_schema: u32,
        output_alignment: u32,
    ) -> Self {
        Self {
            stable_input,
            input_bytes,
            kernel_id,
            recipe_id,
            version,
            canonical_settings,
            output_kind,
            output_schema,
            output_alignment,
        }
    }

    /// Validates the output contract before a request consumes an in-flight
    /// slot.
    pub fn validate(&self) -> Result<(), InvalidRequest> {
        if self.output_alignment == 0 || !self.output_alignment.is_power_of_two() {
            return Err(InvalidRequest::InvalidAlignment {
                alignment: self.output_alignment,
            });
        }
        Ok(())
    }

    /// Derives the deterministic semantic request key.
    #[must_use]
    pub fn request_key(&self) -> ArtifactKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"Prism.AssetBuild.RuntimeDerivedBakeKey\0");
        hasher.update(&BAKE_KEY_PROTOCOL_VERSION.to_le_bytes());
        match &self.stable_input {
            StableInputKey::Bytes(bytes) => {
                hasher.update(&[0]);
                update_len_prefixed(&mut hasher, bytes);
            }
            StableInputKey::Hash(hash) => {
                hasher.update(&[1]);
                hasher.update(&hash.0);
            }
        }
        hasher.update(&self.kernel_id.to_le_bytes());
        hasher.update(&self.recipe_id.to_le_bytes());
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.output_kind.to_le_bytes());
        hasher.update(&self.output_schema.to_le_bytes());
        hasher.update(&self.output_alignment.to_le_bytes());
        update_len_prefixed(&mut hasher, &self.canonical_settings);
        ArtifactKey(*hasher.finalize().as_bytes())
    }
}

fn update_len_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Pure, synchronous algorithm invoked by [`BakeService`].
///
/// The implementation must not publish to DDC/CAS/Manifest or depend on
/// ambient state. Cancellation is cooperative: a non-preemptible kernel may
/// finish its current call, but the service will not publish its result after
/// cancellation wins the publication race.
pub trait BakeKernel: Send + Sync + 'static {
    /// Produces owned output bytes for `request`.
    fn bake(
        &self,
        request: &BakeRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, KernelError>;
}

impl<F> BakeKernel for F
where
    F: Fn(&BakeRequest, &CancellationToken) -> Result<Vec<u8>, KernelError> + Send + Sync + 'static,
{
    fn bake(
        &self,
        request: &BakeRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, KernelError> {
        self(request, cancellation)
    }
}

/// Explicit kernel lookup seam used by [`BakeService`].
pub trait BakeKernelDispatch: Send + Sync + 'static {
    /// Resolves a stable request `kernel_id` to the kernel that executes it.
    fn resolve(&self, kernel_id: u128) -> Result<Arc<dyn BakeKernel>, KernelDispatchError>;
}

/// Thread-safe caller-owned registry of stable kernel IDs.
///
/// The registry has no global instance. A caller may register kernels before or
/// after constructing a service, and all lookups observe the same registry
/// instance that was explicitly injected into that service.
#[derive(Default)]
pub struct BakeKernelRegistry {
    kernels: RwLock<BTreeMap<u128, Arc<dyn BakeKernel>>>,
}

impl BakeKernelRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one stable kernel ID, rejecting replacement of an existing ID.
    pub fn register(
        &self,
        kernel_id: u128,
        kernel: Arc<dyn BakeKernel>,
    ) -> Result<(), KernelRegistryError> {
        let mut kernels = self
            .kernels
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let std::collections::btree_map::Entry::Vacant(entry) = kernels.entry(kernel_id) {
            entry.insert(kernel);
            Ok(())
        } else {
            Err(KernelRegistryError::DuplicateKernelId { kernel_id })
        }
    }

    /// Returns whether the registry contains `kernel_id`.
    #[must_use]
    pub fn contains(&self, kernel_id: u128) -> bool {
        self.kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&kernel_id)
    }
}

impl BakeKernelDispatch for BakeKernelRegistry {
    fn resolve(&self, kernel_id: u128) -> Result<Arc<dyn BakeKernel>, KernelDispatchError> {
        self.kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&kernel_id)
            .cloned()
            .ok_or(KernelDispatchError::UnknownKernel { kernel_id })
    }
}

/// Registration failure for [`BakeKernelRegistry`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelRegistryError {
    /// A stable ID may not be rebound after registration.
    DuplicateKernelId {
        /// The already registered stable kernel ID.
        kernel_id: u128,
    },
}

impl fmt::Display for KernelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKernelId { kernel_id } => {
                write!(formatter, "duplicate bake kernel id {kernel_id:#034x}")
            }
        }
    }
}

impl std::error::Error for KernelRegistryError {}

/// Structured failure from a kernel dispatch seam.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelDispatchError {
    /// No registered kernel owns the requested stable ID.
    UnknownKernel {
        /// The stable kernel ID requested by the bake request.
        kernel_id: u128,
    },
}

impl fmt::Display for KernelDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownKernel { kernel_id } => {
                write!(formatter, "unknown bake kernel id {kernel_id:#034x}")
            }
        }
    }
}

impl std::error::Error for KernelDispatchError {}

/// Explicit bounds for the process-local `RuntimeDerived` cache.
///
/// A zero limit disables storage for that dimension; zero never means
/// unbounded. The cache is always best-effort and is not DDC/CAS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeCacheConfig {
    /// Maximum number of cached successful results.
    pub max_entries: usize,
    /// Maximum sum of cached output byte lengths.
    pub max_bytes: u64,
}

impl RuntimeCacheConfig {
    /// Creates explicit entry and byte limits.
    #[must_use]
    pub const fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }

    /// Disables cache storage while retaining the same service API.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::new(0, 0)
    }
}

/// Observable cache counters and current bounded occupancy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeCacheStats {
    /// Current number of cached results.
    pub entries: usize,
    /// Current sum of cached output byte lengths.
    pub bytes: u64,
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of lookup misses, including disabled-cache misses.
    pub misses: u64,
    /// Number of entries removed by deterministic capacity eviction.
    pub evictions: u64,
}

#[derive(Debug)]
struct RuntimeCache {
    config: RuntimeCacheConfig,
    state: Mutex<RuntimeCacheState>,
}

#[derive(Debug, Default)]
struct RuntimeCacheState {
    entries: BTreeMap<ArtifactKey, BakeResult>,
    bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl RuntimeCache {
    fn new(config: RuntimeCacheConfig) -> Self {
        Self {
            config,
            state: Mutex::new(RuntimeCacheState::default()),
        }
    }

    fn get(&self, key: ArtifactKey) -> Option<BakeResult> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = state.entries.get(&key).cloned();
        if result.is_some() {
            state.hits = state.hits.saturating_add(1);
        } else {
            state.misses = state.misses.saturating_add(1);
        }
        result
    }

    fn contains(&self, key: ArtifactKey) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .contains_key(&key)
    }

    fn insert(&self, result: BakeResult) {
        if self.config.max_entries == 0 || result.descriptor.byte_len > self.config.max_bytes {
            return;
        }
        let key = result.key;
        let byte_len = result.descriptor.byte_len;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(previous) = state.entries.insert(key, result) {
            state.bytes = state.bytes.saturating_sub(previous.descriptor.byte_len);
        }
        state.bytes = state.bytes.saturating_add(byte_len);
        while state.entries.len() > self.config.max_entries || state.bytes > self.config.max_bytes {
            let Some((&key, _)) = state.entries.iter().next_back() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&key) {
                state.bytes = state.bytes.saturating_sub(evicted.descriptor.byte_len);
                state.evictions = state.evictions.saturating_add(1);
            }
        }
    }

    fn stats(&self) -> RuntimeCacheStats {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        RuntimeCacheStats {
            entries: state.entries.len(),
            bytes: state.bytes,
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
        }
    }
}

/// Structured error returned by a bake kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelError {
    message: String,
}

impl KernelError {
    /// Creates a kernel failure with a caller-owned diagnostic message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the diagnostic message without exposing any scheduler state.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for KernelError {}

/// Invalid request details reported before admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidRequest {
    /// Output alignment must be representable by the descriptor contract.
    InvalidAlignment {
        /// The alignment supplied by the caller.
        alignment: u32,
    },
}

impl fmt::Display for InvalidRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlignment { alignment } => {
                write!(
                    formatter,
                    "output alignment must be a non-zero power of two, got {alignment}"
                )
            }
        }
    }
}

impl std::error::Error for InvalidRequest {}

/// Failure returned by request admission or execution.
#[derive(Clone, Debug)]
pub enum BakeError {
    /// No in-flight slot was available; the request was not queued.
    QueueFull {
        /// The configured admission limit.
        max_in_flight: usize,
    },
    /// Cancellation won before result publication.
    Cancelled,
    /// The injected dispatch seam could not resolve the request kernel.
    KernelDispatch(KernelDispatchError),
    /// The request was rejected before consuming an in-flight slot.
    InvalidRequest(InvalidRequest),
    /// The kernel returned a structured failure.
    Kernel(KernelError),
    /// The kernel panicked; the service recovered its in-flight slot.
    KernelPanicked,
    /// The worker could not be started.
    WorkerStart,
    /// The worker ended before delivering its one result.
    WorkerDisconnected,
}

impl fmt::Display for BakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { max_in_flight } => {
                write!(
                    formatter,
                    "bake service is full (max in flight: {max_in_flight})"
                )
            }
            Self::Cancelled => formatter.write_str("bake request was cancelled"),
            Self::KernelDispatch(error) => {
                write!(formatter, "bake kernel dispatch failed: {error}")
            }
            Self::InvalidRequest(error) => write!(formatter, "invalid bake request: {error}"),
            Self::Kernel(error) => write!(formatter, "bake kernel failed: {error}"),
            Self::KernelPanicked => formatter.write_str("bake kernel panicked"),
            Self::WorkerStart => formatter.write_str("bake worker could not be started"),
            Self::WorkerDisconnected => formatter.write_str("bake worker disconnected"),
        }
    }
}

impl std::error::Error for BakeError {}

/// Descriptor for an owned, in-memory runtime-derived result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BakeDescriptor {
    /// Semantic key derived from the request.
    pub key: ArtifactKey,
    /// Output kind from the request contract.
    pub kind: u128,
    /// Output schema from the request contract.
    pub schema: u32,
    /// BLAKE3 digest of `BakeResult::output_bytes`.
    pub content_hash: ContentHash,
    /// Length of `BakeResult::output_bytes`.
    pub byte_len: u64,
    /// Alignment from the request contract.
    pub alignment: u32,
}

/// Successful result. The output remains owned by the caller after `wait`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BakeResult {
    /// Deterministic semantic request key.
    pub key: ArtifactKey,
    /// Self-contained descriptor for the returned bytes.
    pub descriptor: BakeDescriptor,
    /// Owned kernel output. A configured service may retain a clone in its
    /// process-local runtime cache after successful publication.
    pub output_bytes: Vec<u8>,
}

/// Cooperative cancellation handle shared by the caller and kernel.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    control: Arc<JobControl>,
}

impl CancellationToken {
    /// Requests cancellation. Returns `true` only if publication had not won.
    pub fn cancel(&self) -> bool {
        self.control.request_cancel()
    }

    /// Returns whether cancellation has won the job state race.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.control.is_cancelled()
    }
}

/// A submitted bake request. Dropping the handle does not cancel the kernel;
/// call [`Self::cancel`] when cancellation is required. Duplicate requests
/// share one kernel execution, while each handle receives an independent
/// cancellation result.
pub struct BakeHandle {
    receiver: Receiver<Result<BakeResult, BakeError>>,
    subscription: Option<Arc<Subscription>>,
    key: ArtifactKey,
    jobs: Arc<Mutex<BTreeMap<ArtifactKey, Arc<SharedJob>>>>,
}

impl BakeHandle {
    /// Requests cancellation before the result publication boundary.
    pub fn cancel(&self) -> bool {
        let Some(subscription) = &self.subscription else {
            return false;
        };
        if !subscription.cancel() {
            return false;
        }
        if subscription.job.should_cancel() {
            subscription.job.control.request_cancel();
            let mut jobs = self
                .jobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if jobs
                .get(&self.key)
                .is_some_and(|job| Arc::ptr_eq(job, &subscription.job))
            {
                jobs.remove(&self.key);
            }
        }
        true
    }

    /// Waits for the pool task and returns the sole result message.
    pub fn wait(self) -> Result<BakeResult, BakeError> {
        self.receiver
            .recv()
            .unwrap_or(Err(BakeError::WorkerDisconnected))
    }
}

const SUBSCRIBER_OPEN: u8 = 0;
const SUBSCRIBER_CANCELLED: u8 = 1;
const SUBSCRIBER_DELIVERED: u8 = 2;

struct Subscription {
    state: std::sync::atomic::AtomicU8,
    sender: mpsc::Sender<Result<BakeResult, BakeError>>,
    job: Arc<SharedJob>,
}

impl Subscription {
    fn cancel(&self) -> bool {
        if self
            .state
            .compare_exchange(
                SUBSCRIBER_OPEN,
                SUBSCRIBER_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        true
    }

    fn deliver(&self, result: Result<BakeResult, BakeError>) {
        if self
            .state
            .compare_exchange(
                SUBSCRIBER_OPEN,
                SUBSCRIBER_DELIVERED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            let _ = self.sender.send(result);
        } else if self
            .state
            .compare_exchange(
                SUBSCRIBER_CANCELLED,
                SUBSCRIBER_DELIVERED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            let _ = self.sender.send(Err(BakeError::Cancelled));
        }
    }
}

struct SharedJob {
    control: Arc<JobControl>,
    state: Mutex<SharedJobState>,
}

#[derive(Default)]
struct SharedJobState {
    completion: Option<Result<BakeResult, BakeError>>,
    subscribers: Vec<Arc<Subscription>>,
}

impl SharedJob {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            control: Arc::new(JobControl::new()),
            state: Mutex::new(SharedJobState::default()),
        })
    }

    fn subscribe(
        self: &Arc<Self>,
        sender: mpsc::Sender<Result<BakeResult, BakeError>>,
    ) -> Arc<Subscription> {
        let subscription = Arc::new(Subscription {
            state: std::sync::atomic::AtomicU8::new(SUBSCRIBER_OPEN),
            sender,
            job: Arc::clone(self),
        });
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(completion) = &state.completion {
            subscription.deliver(completion.clone());
        } else if self.control.is_cancelled() {
            subscription.deliver(Err(BakeError::Cancelled));
        } else {
            state.subscribers.push(Arc::clone(&subscription));
        }
        subscription
    }

    fn should_cancel(&self) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.completion.is_none()
            && state
                .subscribers
                .iter()
                .all(|subscription| subscription.state.load(Ordering::Acquire) != SUBSCRIBER_OPEN)
    }

    fn complete(&self, result: Result<BakeResult, BakeError>) {
        let subscribers = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.completion.is_some() {
                return;
            }
            state.completion = Some(result.clone());
            std::mem::take(&mut state.subscribers)
        };
        for subscriber in subscribers {
            subscriber.deliver(result.clone());
        }
    }
}

#[derive(Debug)]
struct JobControl {
    state: Mutex<JobState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JobState {
    Open,
    Cancelled,
    Published,
    Finished,
}

impl JobControl {
    fn new() -> Self {
        Self {
            state: Mutex::new(JobState::Open),
        }
    }

    fn request_cancel(&self) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *state == JobState::Open {
            *state = JobState::Cancelled;
            true
        } else {
            false
        }
    }

    fn is_cancelled(&self) -> bool {
        *self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            == JobState::Cancelled
    }

    fn outcome(
        &self,
        requested: Result<BakeResult, BakeError>,
    ) -> Option<Result<BakeResult, BakeError>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match *state {
            JobState::Open => {
                *state = if requested.is_ok() {
                    JobState::Published
                } else {
                    JobState::Finished
                };
                Some(requested)
            }
            JobState::Cancelled => {
                *state = JobState::Finished;
                Some(Err(BakeError::Cancelled))
            }
            JobState::Published | JobState::Finished => None,
        }
    }
}

struct InFlightPermit {
    count: Arc<AtomicUsize>,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// In-process, bounded `BakeService` for `RuntimeDerived` work.
#[derive(Clone)]
pub struct BakeService {
    max_in_flight: usize,
    in_flight: Arc<AtomicUsize>,
    dispatch: Arc<dyn BakeKernelDispatch>,
    cache: Arc<RuntimeCache>,
    jobs: Arc<Mutex<BTreeMap<ArtifactKey, Arc<SharedJob>>>>,
    pool: Arc<TaskPool>,
}

struct DirectKernelDispatch {
    kernel: Arc<dyn BakeKernel>,
}

impl BakeKernelDispatch for DirectKernelDispatch {
    fn resolve(&self, _kernel_id: u128) -> Result<Arc<dyn BakeKernel>, KernelDispatchError> {
        Ok(Arc::clone(&self.kernel))
    }
}

impl BakeService {
    /// Creates a service with no waiting queue and no runtime cache.
    ///
    /// This compatibility constructor dispatches every request to `kernel`,
    /// regardless of `BakeRequest::kernel_id`. Use [`Self::new_with_registry`]
    /// when stable kernel ID validation is required.
    pub fn new(
        max_in_flight: usize,
        kernel: Arc<dyn BakeKernel>,
    ) -> Result<Self, BakeServiceConfigError> {
        Self::new_with_dispatch(
            max_in_flight,
            RuntimeCacheConfig::disabled(),
            Arc::new(DirectKernelDispatch { kernel }),
        )
    }

    /// Creates a service with an explicit bounded process-local runtime cache.
    pub fn new_with_cache(
        max_in_flight: usize,
        cache_config: RuntimeCacheConfig,
        kernel: Arc<dyn BakeKernel>,
    ) -> Result<Self, BakeServiceConfigError> {
        Self::new_with_dispatch(
            max_in_flight,
            cache_config,
            Arc::new(DirectKernelDispatch { kernel }),
        )
    }

    /// Creates a service that resolves each request through an injected
    /// caller-owned stable kernel registry.
    pub fn new_with_registry(
        max_in_flight: usize,
        cache_config: RuntimeCacheConfig,
        registry: Arc<BakeKernelRegistry>,
    ) -> Result<Self, BakeServiceConfigError> {
        Self::new_with_dispatch(max_in_flight, cache_config, registry)
    }

    /// Creates a service with a custom explicit kernel dispatch seam.
    pub fn new_with_dispatch(
        max_in_flight: usize,
        cache_config: RuntimeCacheConfig,
        dispatch: Arc<dyn BakeKernelDispatch>,
    ) -> Result<Self, BakeServiceConfigError> {
        if max_in_flight == 0 {
            return Err(BakeServiceConfigError::ZeroInFlight);
        }
        Ok(Self {
            max_in_flight,
            in_flight: Arc::new(AtomicUsize::new(0)),
            dispatch,
            cache: Arc::new(RuntimeCache::new(cache_config)),
            jobs: Arc::new(Mutex::new(BTreeMap::new())),
            pool: Arc::new(TaskPool::with_threads(
                max_in_flight.clamp(1, MAX_BAKE_WORKERS),
            )),
        })
    }

    /// Returns the configured admission limit.
    #[must_use]
    pub const fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// Returns the number of currently admitted requests.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Returns current cache occupancy and counters.
    #[must_use]
    pub fn cache_stats(&self) -> RuntimeCacheStats {
        self.cache.stats()
    }

    /// Returns whether `key` currently has a successful cached result.
    #[must_use]
    pub fn is_cached(&self, key: ArtifactKey) -> bool {
        self.cache.contains(key)
    }

    /// Admits and starts one request, or rejects it immediately when full.
    pub fn submit(&self, request: BakeRequest) -> Result<BakeHandle, BakeError> {
        request.validate().map_err(BakeError::InvalidRequest)?;
        let key = request.request_key();
        let (sender, receiver) = mpsc::channel();
        if let Some(result) = self.cache.get(key) {
            let _ = sender.send(Ok(result));
            return Ok(BakeHandle {
                receiver,
                subscription: None,
                key,
                jobs: Arc::clone(&self.jobs),
            });
        }

        let kernel = self
            .dispatch
            .resolve(request.kernel_id)
            .map_err(BakeError::KernelDispatch)?;
        let mut jobs = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(job) = jobs.get(&key) {
            let subscription = job.subscribe(sender);
            return Ok(BakeHandle {
                receiver,
                subscription: Some(subscription),
                key,
                jobs: Arc::clone(&self.jobs),
            });
        }

        let permit = self.try_acquire().ok_or(BakeError::QueueFull {
            max_in_flight: self.max_in_flight,
        })?;
        let job = SharedJob::new();
        let subscription = job.subscribe(sender);
        jobs.insert(key, Arc::clone(&job));
        drop(jobs);

        let worker_cache = Arc::clone(&self.cache);
        let worker_jobs = Arc::clone(&self.jobs);
        let worker_request = request;
        let worker_kernel = kernel;
        let worker_job = Arc::clone(&job);
        let worker_key = key;
        let pool = Arc::clone(&self.pool);
        let counter = Counter::new();
        pool.spawn(&counter, move || {
            let cancellation = CancellationToken {
                control: Arc::clone(&worker_job.control),
            };
            let requested = if cancellation.is_cancelled() {
                Err(BakeError::Cancelled)
            } else {
                match catch_unwind(AssertUnwindSafe(|| {
                    worker_kernel.bake(&worker_request, &cancellation)
                })) {
                    Ok(Ok(output_bytes)) => {
                        if cancellation.is_cancelled() {
                            Err(BakeError::Cancelled)
                        } else {
                            let descriptor = BakeDescriptor {
                                key: worker_key,
                                kind: worker_request.output_kind,
                                schema: worker_request.output_schema,
                                content_hash: ContentHash::digest(&output_bytes),
                                byte_len: output_bytes.len() as u64,
                                alignment: worker_request.output_alignment,
                            };
                            Ok(BakeResult {
                                key: worker_key,
                                descriptor,
                                output_bytes,
                            })
                        }
                    }
                    Ok(Err(error)) => Err(BakeError::Kernel(error)),
                    Err(_) => Err(BakeError::KernelPanicked),
                }
            };
            drop(permit);
            if let Some(outcome) = worker_job.control.outcome(requested) {
                if let Ok(result) = &outcome {
                    worker_cache.insert(result.clone());
                }
                worker_job.complete(outcome);
            }
            let mut jobs = worker_jobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if jobs
                .get(&worker_key)
                .is_some_and(|job| Arc::ptr_eq(job, &worker_job))
            {
                jobs.remove(&worker_key);
            }
        });
        Ok(BakeHandle {
            receiver,
            subscription: Some(subscription),
            key,
            jobs: Arc::clone(&self.jobs),
        })
    }

    fn try_acquire(&self) -> Option<InFlightPermit> {
        let mut current = self.in_flight.load(Ordering::Acquire);
        loop {
            if current >= self.max_in_flight {
                return None;
            }
            match self.in_flight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(InFlightPermit {
                        count: Arc::clone(&self.in_flight),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

/// Configuration failure for [`BakeService`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BakeServiceConfigError {
    /// At least one in-flight slot is required.
    ZeroInFlight,
}

impl fmt::Display for BakeServiceConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInFlight => formatter.write_str("max_in_flight must be greater than zero"),
        }
    }
}

impl std::error::Error for BakeServiceConfigError {}
