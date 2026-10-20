//! Bounded asynchronous cell I/O contracts.
//!
//! The provider owns storage and decoding policy. This crate only transports
//! opaque payload bytes, reserves bounded memory, and reports terminal events
//! in a stable scan of ready requests. Cancelling an accepted request queues
//! a terminal event so the owner can restore the cell's last safe state.
//! Before dropping the driver, drain or cancel/reconcile all accepted requests.

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use super::{CellCoord, CellFileIdentity, TransitionToken};

/// A boxed provider future that returns one opaque cell payload.
pub type CellLoadFuture<E> = Pin<Box<dyn Future<Output = Result<Vec<u8>, E>> + Send + 'static>>;

/// A boxed provider future that commits one opaque cell payload.
pub type CellSaveFuture<E> = Pin<Box<dyn Future<Output = Result<(), E>> + Send + 'static>>;

/// Admission budget for asynchronous cell I/O.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoBudget {
    max_in_flight: usize,
    max_bytes: usize,
}

impl IoBudget {
    /// Creates a budget with non-zero request and reserved-byte capacities.
    pub const fn new(max_in_flight: usize, max_bytes: usize) -> Result<Self, IoSubmitError<()>> {
        if max_in_flight == 0 {
            return Err(IoSubmitError::ZeroInFlight);
        }
        if max_bytes == 0 {
            return Err(IoSubmitError::ZeroBytes);
        }
        Ok(Self {
            max_in_flight,
            max_bytes,
        })
    }

    /// Maximum number of accepted operations.
    #[must_use]
    pub const fn max_in_flight(self) -> usize {
        self.max_in_flight
    }

    /// Maximum bytes reserved by accepted operations.
    #[must_use]
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

/// A load request tied to one model transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellLoadRequest {
    token: TransitionToken,
    file: CellFileIdentity,
    max_bytes: usize,
}

impl CellLoadRequest {
    /// Creates a bounded load request.
    pub const fn new(token: TransitionToken, file: CellFileIdentity, max_bytes: usize) -> Self {
        Self {
            token,
            file,
            max_bytes,
        }
    }

    /// Transition being completed by this request.
    #[must_use]
    pub const fn token(&self) -> TransitionToken {
        self.token
    }

    /// Cell being loaded.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        self.token.cell()
    }

    /// Stable storage identity.
    #[must_use]
    pub const fn file(&self) -> &CellFileIdentity {
        &self.file
    }

    /// Maximum accepted payload size.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

/// A save request for an already resident cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CellSaveRequest {
    cell: CellCoord,
    file: CellFileIdentity,
    revision: u64,
}

impl CellSaveRequest {
    /// Creates a save request with the model revision observed by the caller.
    pub const fn new(cell: CellCoord, file: CellFileIdentity, revision: u64) -> Self {
        Self {
            cell,
            file,
            revision,
        }
    }

    /// Cell being saved.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        self.cell
    }

    /// Stable storage identity.
    #[must_use]
    pub const fn file(&self) -> &CellFileIdentity {
        &self.file
    }

    /// Cell authoring revision captured for this write.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}

/// A provider operation that can be cancelled or reported as failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoOperation {
    /// Read one cell payload.
    Load(CellLoadRequest),
    /// Commit one cell payload.
    Save(CellSaveRequest),
}

impl IoOperation {
    /// Returns the cell affected by the operation.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        match self {
            Self::Load(request) => request.cell(),
            Self::Save(request) => request.cell(),
        }
    }
}

/// Stable identity of an accepted asynchronous operation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IoRequestId(u64);

impl IoRequestId {
    /// Returns the numeric request identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Provider or driver validation failure attached to a terminal completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoFailure<E> {
    /// The provider rejected or failed the operation.
    Provider(E),
    /// The provider returned more bytes than the request admitted.
    PayloadTooLarge {
        /// Declared request limit.
        limit: usize,
        /// Actual returned payload size.
        actual: usize,
    },
}

impl<E: fmt::Display> fmt::Display for IoFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => error.fmt(formatter),
            Self::PayloadTooLarge { limit, actual } => {
                write!(
                    formatter,
                    "cell payload {actual} exceeds request limit {limit}"
                )
            }
        }
    }
}

/// Terminal event emitted by [`WorldIoDriver`].
#[derive(Debug, Eq, PartialEq)]
pub enum IoCompletion<E> {
    /// A load completed with an opaque payload.
    Loaded {
        /// Request identity.
        request_id: IoRequestId,
        /// Original load request.
        request: CellLoadRequest,
        /// Opaque payload bytes.
        bytes: Vec<u8>,
    },
    /// A save completed successfully.
    Saved {
        /// Request identity.
        request_id: IoRequestId,
        /// Original save request.
        request: CellSaveRequest,
    },
    /// An operation failed without changing model state.
    Failed {
        /// Request identity.
        request_id: IoRequestId,
        /// Original operation.
        operation: IoOperation,
        /// Failure detail.
        error: IoFailure<E>,
    },
    /// An operation was cancelled and must be reconciled by its owner.
    Cancelled {
        /// Request identity.
        request_id: IoRequestId,
        /// Original operation.
        operation: IoOperation,
    },
}

impl<E> IoCompletion<E> {
    /// Returns the request identity of this terminal event.
    #[must_use]
    pub const fn request_id(&self) -> IoRequestId {
        match self {
            Self::Loaded { request_id, .. }
            | Self::Saved { request_id, .. }
            | Self::Failed { request_id, .. }
            | Self::Cancelled { request_id, .. } => *request_id,
        }
    }

    /// Returns the affected cell.
    #[must_use]
    pub const fn cell(&self) -> CellCoord {
        match self {
            Self::Loaded { request, .. } => request.cell(),
            Self::Saved { request, .. } => request.cell(),
            Self::Failed { operation, .. } | Self::Cancelled { operation, .. } => operation.cell(),
        }
    }
}

/// Provider contract for opaque asynchronous cell payloads.
///
/// Startup must not block. Loads must enforce `request.max_bytes()` before
/// allocating the full payload. Dropping a future must cancel its work; a save
/// must not publish partial bytes, and may report success only after its atomic
/// commit is durable according to the provider's persistence policy. The driver
/// bounds reservations and validates results, not a provider's internal caches.
pub trait CellIoProvider {
    /// Provider-specific error type.
    type Error: Send + 'static;

    /// Starts a load without borrowing the returned future from the provider.
    fn load_cell(
        &self,
        request: CellLoadRequest,
    ) -> Result<CellLoadFuture<Self::Error>, Self::Error>;

    /// Starts a save and takes ownership of the bytes until completion.
    fn save_cell(
        &self,
        request: CellSaveRequest,
        payload: Vec<u8>,
    ) -> Result<CellSaveFuture<Self::Error>, Self::Error>;
}

enum ActiveOperation<E> {
    Load {
        request: CellLoadRequest,
        future: CellLoadFuture<E>,
        reserved_bytes: usize,
    },
    Save {
        request: CellSaveRequest,
        future: CellSaveFuture<E>,
        reserved_bytes: usize,
    },
}

impl<E> ActiveOperation<E> {
    fn operation(&self) -> IoOperation {
        match self {
            Self::Load { request, .. } => IoOperation::Load(request.clone()),
            Self::Save { request, .. } => IoOperation::Save(request.clone()),
        }
    }

    const fn reserved_bytes(&self) -> usize {
        match self {
            Self::Load { reserved_bytes, .. } | Self::Save { reserved_bytes, .. } => {
                *reserved_bytes
            }
        }
    }
}

/// Bounded, deterministic driver for provider futures.
pub struct WorldIoDriver<P: CellIoProvider> {
    provider: P,
    budget: IoBudget,
    active: BTreeMap<IoRequestId, ActiveOperation<P::Error>>,
    cancelled: VecDeque<IoCompletion<P::Error>>,
    next_id: u64,
    reserved_bytes: usize,
}

impl<P: CellIoProvider> WorldIoDriver<P> {
    /// Creates an idle driver with a fixed request and memory budget.
    #[must_use]
    pub fn new(provider: P, budget: IoBudget) -> Self {
        Self {
            provider,
            budget,
            active: BTreeMap::new(),
            cancelled: VecDeque::new(),
            next_id: 0,
            reserved_bytes: 0,
        }
    }

    /// Returns the configured admission budget.
    #[must_use]
    pub const fn budget(&self) -> IoBudget {
        self.budget
    }

    /// Returns the number of operations awaiting terminal delivery.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.active.len() + self.cancelled.len()
    }

    /// Returns currently reserved payload bytes.
    #[must_use]
    pub const fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    /// Returns a shared provider reference for diagnostics or adapter setup.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Submits a load after reserving its declared maximum payload size.
    pub fn submit_load(
        &mut self,
        request: CellLoadRequest,
    ) -> Result<IoRequestId, IoSubmitError<P::Error>> {
        let reserved_bytes = request.max_bytes();
        if reserved_bytes == 0
            || request.token().kind() != super::TransitionKind::Load
            || request.file().coord() != request.cell()
        {
            return Err(IoSubmitError::InvalidRequest);
        }
        self.admit(reserved_bytes)?;
        self.check_cell_idle(request.cell())?;
        let id = self.allocate_id()?;
        let future = match self.provider.load_cell(request.clone()) {
            Ok(future) => future,
            Err(error) => return Err(IoSubmitError::Provider(error)),
        };
        self.reserved_bytes += reserved_bytes;
        self.active.insert(
            id,
            ActiveOperation::Load {
                request,
                future,
                reserved_bytes,
            },
        );
        Ok(id)
    }

    /// Submits a save while reserving the owned payload bytes.
    pub fn submit_save(
        &mut self,
        request: CellSaveRequest,
        payload: Vec<u8>,
    ) -> Result<IoRequestId, IoSubmitError<P::Error>> {
        let reserved_bytes = payload.len();
        if request.file().coord() != request.cell() {
            return Err(IoSubmitError::InvalidRequest);
        }
        self.admit(reserved_bytes)?;
        self.check_cell_idle(request.cell())?;
        let id = self.allocate_id()?;
        let future = match self.provider.save_cell(request.clone(), payload) {
            Ok(future) => future,
            Err(error) => return Err(IoSubmitError::Provider(error)),
        };
        self.reserved_bytes += reserved_bytes;
        self.active.insert(
            id,
            ActiveOperation::Save {
                request,
                future,
                reserved_bytes,
            },
        );
        Ok(id)
    }

    /// Cancels an accepted operation and queues its terminal cancellation.
    pub fn cancel(&mut self, request_id: IoRequestId) -> bool {
        let Some(operation) = self.active.remove(&request_id) else {
            return false;
        };
        self.reserved_bytes = self
            .reserved_bytes
            .saturating_sub(operation.reserved_bytes());
        self.cancelled.push_back(IoCompletion::Cancelled {
            request_id,
            operation: operation.operation(),
        });
        true
    }

    /// Delivers queued cancellations, then polls ready work in request-id order.
    /// A pending earlier request does not block ready later work.
    pub fn poll(&mut self, context: &mut Context<'_>) -> Poll<Option<IoCompletion<P::Error>>> {
        if let Some(completion) = self.cancelled.pop_front() {
            return Poll::Ready(Some(completion));
        }
        if self.active.is_empty() {
            return Poll::Ready(None);
        }

        let ids: Vec<_> = self.active.keys().copied().collect();
        for request_id in ids {
            let polled = match self.active.get_mut(&request_id) {
                Some(ActiveOperation::Load { future, .. }) => {
                    PollResult::Load(future.as_mut().poll(context))
                }
                Some(ActiveOperation::Save { future, .. }) => {
                    PollResult::Save(future.as_mut().poll(context))
                }
                None => continue,
            };
            match polled {
                PollResult::Load(Poll::Ready(result)) => {
                    let operation = self
                        .active
                        .remove(&request_id)
                        .expect("polled operation remains registered");
                    self.release(&operation);
                    let ActiveOperation::Load { request, .. } = operation else {
                        unreachable!("load poll result matches load operation")
                    };
                    return Poll::Ready(Some(match result {
                        Ok(bytes) if bytes.len() <= request.max_bytes() => IoCompletion::Loaded {
                            request_id,
                            request,
                            bytes,
                        },
                        Ok(bytes) => IoCompletion::Failed {
                            request_id,
                            operation: IoOperation::Load(request.clone()),
                            error: IoFailure::PayloadTooLarge {
                                limit: request.max_bytes(),
                                actual: bytes.len(),
                            },
                        },
                        Err(error) => IoCompletion::Failed {
                            request_id,
                            operation: IoOperation::Load(request),
                            error: IoFailure::Provider(error),
                        },
                    }));
                }
                PollResult::Save(Poll::Ready(result)) => {
                    let operation = self
                        .active
                        .remove(&request_id)
                        .expect("polled operation remains registered");
                    self.release(&operation);
                    let ActiveOperation::Save { request, .. } = operation else {
                        unreachable!("save poll result matches save operation")
                    };
                    return Poll::Ready(Some(match result {
                        Ok(()) => IoCompletion::Saved {
                            request_id,
                            request,
                        },
                        Err(error) => IoCompletion::Failed {
                            request_id,
                            operation: IoOperation::Save(request),
                            error: IoFailure::Provider(error),
                        },
                    }));
                }
                PollResult::Load(Poll::Pending) | PollResult::Save(Poll::Pending) => {}
            }
        }
        Poll::Pending
    }

    fn admit(&self, bytes: usize) -> Result<(), IoSubmitError<P::Error>> {
        if self.active.len() + self.cancelled.len() >= self.budget.max_in_flight {
            return Err(IoSubmitError::Backpressure {
                max_in_flight: self.budget.max_in_flight,
            });
        }
        let reserved =
            self.reserved_bytes
                .checked_add(bytes)
                .ok_or(IoSubmitError::ByteBudgetExceeded {
                    requested: bytes,
                    max_bytes: self.budget.max_bytes,
                })?;
        if reserved > self.budget.max_bytes {
            return Err(IoSubmitError::ByteBudgetExceeded {
                requested: bytes,
                max_bytes: self.budget.max_bytes,
            });
        }
        Ok(())
    }

    fn check_cell_idle(&self, cell: CellCoord) -> Result<(), IoSubmitError<P::Error>> {
        if self
            .active
            .values()
            .any(|operation| operation.operation().cell() == cell)
            || self
                .cancelled
                .iter()
                .any(|completion| completion.cell() == cell)
        {
            return Err(IoSubmitError::CellBusy(cell));
        }
        Ok(())
    }

    pub(crate) fn contains_load(&self, request: &CellLoadRequest) -> bool {
        self.active.values().any(|operation| {
            matches!(operation, ActiveOperation::Load { request: active, .. } if active == request)
        }) || self.cancelled.iter().any(|completion| {
            matches!(completion, IoCompletion::Cancelled { operation: IoOperation::Load(active), .. } if active == request)
        })
    }

    fn allocate_id(&mut self) -> Result<IoRequestId, IoSubmitError<P::Error>> {
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(IoSubmitError::RequestIdExhausted)?;
        Ok(IoRequestId(self.next_id))
    }

    fn release(&mut self, operation: &ActiveOperation<P::Error>) {
        self.reserved_bytes = self
            .reserved_bytes
            .saturating_sub(operation.reserved_bytes());
    }
}

enum PollResult<E> {
    Load(Poll<Result<Vec<u8>, E>>),
    Save(Poll<Result<(), E>>),
}

/// Submission failures. No future is retained on rejection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoSubmitError<E> {
    /// The request budget has no free operation slot.
    Backpressure {
        /// Configured operation capacity.
        max_in_flight: usize,
    },
    /// The request would exceed the reserved memory budget.
    ByteBudgetExceeded {
        /// Bytes requested by the rejected operation.
        requested: usize,
        /// Configured byte capacity.
        max_bytes: usize,
    },
    /// No operation slots were configured.
    ZeroInFlight,
    /// No byte budget was configured.
    ZeroBytes,
    /// The monotonic operation ID space is exhausted.
    RequestIdExhausted,
    /// A malformed request did not agree with its transition or file identity.
    InvalidRequest,
    /// A cell already has an operation awaiting terminal delivery.
    CellBusy(CellCoord),
    /// The provider rejected operation startup.
    Provider(E),
}

impl<E: fmt::Display> fmt::Display for IoSubmitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backpressure { max_in_flight } => {
                write!(formatter, "cell I/O capacity {max_in_flight} is full")
            }
            Self::ByteBudgetExceeded {
                requested,
                max_bytes,
            } => write!(
                formatter,
                "cell I/O reservation {requested} exceeds byte budget {max_bytes}"
            ),
            Self::ZeroInFlight => {
                formatter.write_str("cell I/O in-flight capacity must be non-zero")
            }
            Self::ZeroBytes => formatter.write_str("cell I/O byte capacity must be non-zero"),
            Self::RequestIdExhausted => formatter.write_str("cell I/O request IDs exhausted"),
            Self::InvalidRequest => formatter.write_str("invalid cell I/O request"),
            Self::CellBusy(cell) => write!(
                formatter,
                "cell ({},{},{}) already has I/O in flight",
                cell.x, cell.y, cell.z
            ),
            Self::Provider(error) => error.fmt(formatter),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for IoSubmitError<E> {}
