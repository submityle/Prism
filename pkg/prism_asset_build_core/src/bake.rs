//! Bounded in-process scheduling for `RuntimeDerived` bake kernels.
//!
//! [`BakeService`] deliberately owns no cache or publication sink. It admits a
//! finite number of requests, runs the caller-provided pure kernel, and hands
//! an owned result back to the caller. A full service rejects immediately; it
//! does not create an unbounded waiting queue.

use crate::types::{ArtifactKey, ContentHash};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

const BAKE_KEY_PROTOCOL_VERSION: u32 = 1;

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
#[derive(Debug)]
pub enum BakeError {
    /// No in-flight slot was available; the request was not queued.
    QueueFull {
        /// The configured admission limit.
        max_in_flight: usize,
    },
    /// Cancellation won before result publication.
    Cancelled,
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
    /// Owned kernel output; no service-side cache retains it.
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
/// call [`Self::cancel`] when cancellation is required.
pub struct BakeHandle {
    receiver: Receiver<Result<BakeResult, BakeError>>,
    control: Arc<JobControl>,
    worker: Option<JoinHandle<()>>,
}

impl BakeHandle {
    /// Requests cancellation before the result publication boundary.
    pub fn cancel(&self) -> bool {
        self.control.request_cancel()
    }

    /// Waits for the worker and returns the sole result message.
    pub fn wait(mut self) -> Result<BakeResult, BakeError> {
        let result = self
            .receiver
            .recv()
            .unwrap_or(Err(BakeError::WorkerDisconnected));
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            return Err(BakeError::KernelPanicked);
        }
        result
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
    kernel: Arc<dyn BakeKernel>,
}

impl BakeService {
    /// Creates a service with no waiting queue and a caller-selected limit.
    pub fn new(
        max_in_flight: usize,
        kernel: Arc<dyn BakeKernel>,
    ) -> Result<Self, BakeServiceConfigError> {
        if max_in_flight == 0 {
            return Err(BakeServiceConfigError::ZeroInFlight);
        }
        Ok(Self {
            max_in_flight,
            in_flight: Arc::new(AtomicUsize::new(0)),
            kernel,
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

    /// Admits and starts one request, or rejects it immediately when full.
    pub fn submit(&self, request: BakeRequest) -> Result<BakeHandle, BakeError> {
        request.validate().map_err(BakeError::InvalidRequest)?;
        let permit = self.try_acquire().ok_or(BakeError::QueueFull {
            max_in_flight: self.max_in_flight,
        })?;
        let (sender, receiver) = mpsc::channel();
        let control = Arc::new(JobControl::new());
        let worker_control = Arc::clone(&control);
        let worker_kernel = Arc::clone(&self.kernel);
        let worker = thread::Builder::new()
            .name("prism-bake".to_owned())
            .spawn(move || {
                let _permit = permit;
                let cancellation = CancellationToken {
                    control: Arc::clone(&worker_control),
                };
                let outcome = if cancellation.is_cancelled() {
                    Err(BakeError::Cancelled)
                } else {
                    match catch_unwind(AssertUnwindSafe(|| {
                        worker_kernel.bake(&request, &cancellation)
                    })) {
                        Ok(Ok(output_bytes)) => {
                            if cancellation.is_cancelled() {
                                Err(BakeError::Cancelled)
                            } else {
                                let key = request.request_key();
                                let descriptor = BakeDescriptor {
                                    key,
                                    kind: request.output_kind,
                                    schema: request.output_schema,
                                    content_hash: ContentHash::digest(&output_bytes),
                                    byte_len: output_bytes.len() as u64,
                                    alignment: request.output_alignment,
                                };
                                Ok(BakeResult {
                                    key,
                                    descriptor,
                                    output_bytes,
                                })
                            }
                        }
                        Ok(Err(error)) => Err(BakeError::Kernel(error)),
                        Err(_) => Err(BakeError::KernelPanicked),
                    }
                };
                if let Some(outcome) = worker_control.outcome(outcome) {
                    let _ = sender.send(outcome);
                }
            })
            .map_err(|_| BakeError::WorkerStart)?;
        Ok(BakeHandle {
            receiver,
            control,
            worker: Some(worker),
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
