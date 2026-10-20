//! Bounded asynchronous request scheduling and dependency assembly layered over
//! the synchronous [`AssetServer`](crate::AssetServer).
//!
//! This API coalesces concurrent requests for the same path, bounds admission,
//! snapshots source/loader state under the server mutex, and performs validated
//! loader-declared range reads on an I/O pool plus synchronous CPU decoding on a CPU pool before
//! committing one dependency-aware revision. Awaiting a dependency yields the
//! CPU worker, so a fixed worker count cannot deadlock solely because parent
//! assemblies are waiting for child flights. Reload flights support a narrow
//! resident-root revision replacement with per-path epoch supersede and direct
//! labeled-path loads that reuse the source assembly; reload candidates also
//! decode and stage newly introduced ordinary dependencies and sibling labels
//! before one atomic commit. Loader ranges use bounded batches on the existing
//! I/O pool, concatenated in declaration order; preemptive cancellation is outside this slice. Live priority cells order queued I/O and CPU work,
//! including wakeups and dependencies; ticket
//! cancellation stops that caller from waiting but does not kill shared or
//! already-running work.

use crate::loader_exec::{AssetLoaders, FullLoadOutput};
use crate::server::{
    AsyncLoadPreparation, AsyncReloadCandidate, AsyncReloadCandidateAsset, AsyncReloadCommitError,
    SoftLoadError,
};
use crate::source::AssetSources;
use crate::{Asset, AssetPath, AssetServer, AssetTypeId, SoftHandle, UntypedHandle};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::sync::{Arc, Weak};
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use prism_tasks::{Priority, PriorityCell, TaskPool};
use std::sync::Mutex;

/// Asset-loading urgency, applied to both I/O dispatch and CPU continuations.
/// Running synchronous reads/decoders are not preempted. Promotion is monotonic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoadPriority {
    /// Work which may wait for idle time.
    Background,
    /// Speculative work for assets which may soon be needed.
    Prefetch,
    /// Visible-content streaming, ahead of speculative work.
    #[default]
    Streaming,
    /// Required by the current interactive critical path.
    Immediate,
}

impl From<LoadPriority> for Priority {
    fn from(priority: LoadPriority) -> Self {
        match priority {
            LoadPriority::Immediate => Self::Critical,
            LoadPriority::Streaming => Self::High,
            LoadPriority::Prefetch => Self::Low,
            LoadPriority::Background => Self::Background,
        }
    }
}

/// Hard admission limits for one caller-controlled soft-reference batch.
/// These count roots, not transitive dependencies, decoded memory, or I/O bytes.
/// Call once per frame/tick with that frame's allowance to implement batching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftLoadBudget {
    /// Maximum input references, including duplicates.
    pub max_requests: usize,
    /// Maximum distinct root paths, including already in-flight roots.
    pub max_unique_assets: usize,
}

/// An entire soft-reference batch was rejected before any flight was admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SoftBatchError {
    /// The input exceeds the per-batch reference allowance.
    RequestBudgetExceeded {
        /// Number of input references.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Distinct resolved roots exceed the per-batch asset allowance.
    UniqueAssetBudgetExceeded {
        /// Number of distinct paths.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// A reference could not be resolved without I/O.
    InvalidHandle {
        /// Input index of the first invalid reference.
        index: usize,
        /// Exact identity/type/catalog diagnostic.
        reason: SoftLoadError,
    },
    /// Existing shared work has an incompatible or unresolved requested type.
    ConflictingFlight {
        /// First input index referring to the conflicting flight.
        index: usize,
        /// Resolved root path.
        path: AssetPath,
    },
    /// All new roots cannot fit; duplicates do not consume new slots.
    AtCapacity {
        /// New execution slots required by the batch.
        required: usize,
        /// Slots available without bypassing dependency/admission waiters.
        available: usize,
    },
}

impl core::fmt::Display for SoftBatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RequestBudgetExceeded { requested, limit } => write!(
                f,
                "soft batch request budget exceeded: {requested} > {limit}"
            ),
            Self::UniqueAssetBudgetExceeded { requested, limit } => write!(
                f,
                "soft batch unique-root budget exceeded: {requested} > {limit}"
            ),
            Self::InvalidHandle { index, reason } => {
                write!(f, "soft batch reference {index}: {reason}")
            }
            Self::ConflictingFlight { index, path } => write!(
                f,
                "soft batch reference {index} conflicts with in-flight type: {path}"
            ),
            Self::AtCapacity {
                required,
                available,
            } => write!(
                f,
                "soft batch needs {required} new slots but only {available} are available"
            ),
        }
    }
}

impl std::error::Error for SoftBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidHandle { reason, .. } => Some(reason),
            _ => None,
        }
    }
}

/// Weak dependency links avoid retaining completed flights or creating Arc
/// cycles. Link insertion and promotion may race: inserting first then sampling
/// the parent's cell ensures either operation observes the other's promotion.
#[derive(Clone)]
struct FlightPriority(Arc<PriorityNode>);

struct PriorityNode {
    cell: Arc<PriorityCell>,
    dependencies: Mutex<Vec<Weak<PriorityNode>>>,
}

impl FlightPriority {
    fn new(priority: Priority) -> Self {
        Self(Arc::new(PriorityNode {
            cell: Arc::new(PriorityCell::new(priority)),
            dependencies: Mutex::new(Vec::new()),
        }))
    }

    fn effective_priority(&self) -> Priority {
        self.0.cell.get()
    }

    fn priority_cell(&self) -> Arc<PriorityCell> {
        Arc::clone(&self.0.cell)
    }

    fn inherit(&self, dependency: &Self) {
        {
            let mut dependencies = self.0.dependencies.lock().expect("priority links poisoned");
            dependencies.retain(|entry| entry.strong_count() != 0);
            let weak = Arc::downgrade(&dependency.0);
            if !dependencies.iter().any(|entry| entry.ptr_eq(&weak)) {
                dependencies.push(weak);
            }
        }
        dependency.boost_to(self.effective_priority());
    }

    fn boost_to(&self, priority: Priority) {
        let mut pending = vec![Arc::clone(&self.0)];
        let mut visited = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !visited.insert(Arc::as_ptr(&node) as usize) {
                continue;
            }
            node.cell.boost_to(priority);
            let dependencies = node.dependencies.lock().expect("priority links poisoned");
            pending.extend(dependencies.iter().filter_map(Weak::upgrade));
        }
    }
}

/// The result of a scheduled load.
pub type ScheduleOutcome = Result<UntypedHandle, LoadFailure>;

/// Failure returned to a scheduled-load consumer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadFailure {
    /// No registered loader claims the requested path.
    NoLoader,
    /// The loader or server panicked while processing the request.
    WorkerPanicked,
    /// This consumer cancelled its wait. The shared load may still complete.
    Cancelled,
    /// The asset or one of its dependencies failed before publication.
    AssemblyFailed(String),
    /// A newer reload started before this reload could publish its staged
    /// result. The resident revision was left untouched.
    Superseded,
}

/// A root request could not be admitted to the bounded scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The configured maximum number of unique requests is already in flight.
    AtCapacity {
        /// Maximum number of concurrently executing assembly flights.
        limit: usize,
    },
    /// A dependency would exceed the bounded waiting queue.
    DependencyQueueFull {
        /// Maximum number of dependency flights waiting for execution.
        limit: usize,
    },
    /// The internal reload request sequence exhausted its finite space.
    RequestIdExhausted,
}

impl core::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AtCapacity { limit } => {
                write!(f, "asset load scheduler is at capacity ({limit})")
            }
            Self::DependencyQueueFull { limit } => {
                write!(f, "asset dependency queue is full ({limit})")
            }
            Self::RequestIdExhausted => f.write_str("asset reload request id space exhausted"),
        }
    }
}

impl std::error::Error for ScheduleError {}

struct Flight {
    notification: Mutex<FlightNotification>,
    priority: FlightPriority,
    requested_type: Option<AssetTypeId>,
}

struct FlightNotification {
    result: Option<ScheduleOutcome>,
    waiters: Vec<(Arc<()>, Waker)>,
}

impl Flight {
    /// Publish the result while the scheduler state lock is held, then return
    /// registered wakers for notification after that lock has been released.
    fn complete(&self, result: ScheduleOutcome) -> Vec<Waker> {
        let mut notification = self.notification.lock().expect("load flight poisoned");
        notification.result = Some(result);
        core::mem::take(&mut notification.waiters)
            .into_iter()
            .map(|(_, waker)| waker)
            .collect()
    }

    fn wake_all(wakers: Vec<Waker>) {
        for waker in wakers {
            waker.wake();
        }
    }

    fn poll(&self, token: &Arc<()>, cx: &Context<'_>) -> Poll<ScheduleOutcome> {
        let mut notification = self.notification.lock().expect("load flight poisoned");
        if let Some(result) = notification.result.clone() {
            return Poll::Ready(result);
        }
        if let Some((_, waker)) = notification
            .waiters
            .iter_mut()
            .find(|(waiter, _)| Arc::ptr_eq(waiter, token))
        {
            if !waker.will_wake(cx.waker()) {
                *waker = cx.waker().clone();
            }
        } else {
            notification
                .waiters
                .push((Arc::clone(token), cx.waker().clone()));
        }
        // Recheck after registering to close completion-vs-registration race.
        if let Some(result) = notification.result.clone() {
            Poll::Ready(result)
        } else {
            Poll::Pending
        }
    }

    fn unregister(&self, token: &Arc<()>) {
        let mut notification = self.notification.lock().expect("load flight poisoned");
        notification
            .waiters
            .retain(|(waiter, _)| !Arc::ptr_eq(waiter, token));
    }

    #[cfg(test)]
    fn waiter_count(&self) -> usize {
        self.notification
            .lock()
            .expect("load flight poisoned")
            .waiters
            .len()
    }
}

struct SchedulerState {
    in_flight: BTreeMap<FlightKey, Arc<Flight>>,
    running: BTreeMap<FlightKey, Arc<Flight>>,
    reload_reads: BTreeMap<ReloadReadKey, Arc<ReloadReadFlight>>,
    queued: VecDeque<QueuedFlight>,
    admission_waiters: Vec<AdmissionWaiter>,
    next_admission_seq: u64,
    next_reload_serial: u64,
}

struct AdmissionWaiter {
    token: Arc<()>,
    priority: FlightPriority,
    seq: u64,
    waker: Waker,
    granted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FlightKey {
    Load(AssetPath),
    Reload { path: AssetPath, serial: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReloadReadKey {
    source_revision: u64,
    path: AssetPath,
    type_id: AssetTypeId,
}

struct ReloadReadFlight {
    notification: Mutex<ReloadReadNotification>,
    priority: FlightPriority,
}

struct ReloadReadNotification {
    result: Option<Result<Arc<[u8]>, String>>,
    waiters: Vec<(Arc<()>, Waker)>,
}

impl ReloadReadFlight {
    fn complete(&self, result: Result<Arc<[u8]>, String>) -> Vec<Waker> {
        let mut notification = self
            .notification
            .lock()
            .expect("reload read flight poisoned");
        notification.result = Some(result);
        core::mem::take(&mut notification.waiters)
            .into_iter()
            .map(|(_, waker)| waker)
            .collect()
    }

    fn poll(&self, token: &Arc<()>, cx: &Context<'_>) -> Poll<Result<Arc<[u8]>, String>> {
        let mut notification = self
            .notification
            .lock()
            .expect("reload read flight poisoned");
        if let Some(result) = notification.result.clone() {
            return Poll::Ready(result);
        }
        if let Some((_, waker)) = notification
            .waiters
            .iter_mut()
            .find(|(waiter, _)| Arc::ptr_eq(waiter, token))
        {
            if !waker.will_wake(cx.waker()) {
                *waker = cx.waker().clone();
            }
        } else {
            notification
                .waiters
                .push((Arc::clone(token), cx.waker().clone()));
        }
        if let Some(result) = notification.result.clone() {
            Poll::Ready(result)
        } else {
            Poll::Pending
        }
    }

    fn unregister(&self, token: &Arc<()>) {
        let mut notification = self
            .notification
            .lock()
            .expect("reload read flight poisoned");
        notification
            .waiters
            .retain(|(waiter, _)| !Arc::ptr_eq(waiter, token));
    }

    fn wake_all(wakers: Vec<Waker>) {
        for waker in wakers {
            waker.wake();
        }
    }
}

struct ReloadReadTicket {
    flight: Arc<ReloadReadFlight>,
    token: Arc<()>,
}

impl Future for ReloadReadTicket {
    type Output = Result<Arc<[u8]>, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.flight.poll(&self.token, cx)
    }
}

impl Drop for ReloadReadTicket {
    fn drop(&mut self) {
        self.flight.unregister(&self.token);
    }
}

struct QueuedFlight {
    key: FlightKey,
    flight: Arc<Flight>,
    priority: Priority,
    ancestors: BTreeSet<AssetPath>,
    reload: bool,
}

/// Concurrent facade for scheduling path loads through an existing server.
///
/// Clones share admission limits and in-flight deduplication. A duplicate path
/// joins its existing request even when the scheduler is at capacity.
pub struct LoadScheduler {
    server: Arc<AssetServer>,
    io_pool: TaskPool,
    cpu_pool: TaskPool,
    limit: usize,
    state: Arc<Mutex<SchedulerState>>,
}

impl Clone for LoadScheduler {
    fn clone(&self) -> Self {
        Self {
            server: Arc::clone(&self.server),
            io_pool: self.io_pool.clone(),
            cpu_pool: self.cpu_pool.clone(),
            limit: self.limit,
            state: Arc::clone(&self.state),
        }
    }
}

impl LoadScheduler {
    /// Create a scheduler with an explicit pool and a maximum number of
    /// concurrently executing assembly flights. The supplied pool remains the
    /// CPU pool for source compatibility; an independent I/O pool with the
    /// same worker count is created for blocking source reads. `max_in_flight`
    /// must be nonzero. Root requests fail fast at capacity; dependency
    /// requests enter a bounded priority queue so a parent can yield its slot
    /// while waiting for its dependency closure.
    #[must_use]
    pub fn new(server: Arc<AssetServer>, pool: TaskPool, max_in_flight: usize) -> Self {
        let io_pool = TaskPool::with_threads(pool.worker_count());
        Self::with_io_cpu_pools(server, io_pool, pool, max_in_flight)
    }

    /// Create a scheduler with explicitly separated I/O and CPU pools.
    ///
    /// I/O work is submitted to `io_pool`; decoding, dependency assembly, and
    /// commit work run on `cpu_pool`. The pools must be independent: passing
    /// the same single-threaded pool for both can deadlock a read future that
    /// is awaited by its only worker. Dependency waits are future awaits, not
    /// blocking joins, so CPU workers are available to run newly admitted
    /// dependency flights while parents are pending.
    #[must_use]
    pub fn with_io_cpu_pools(
        server: Arc<AssetServer>,
        io_pool: TaskPool,
        cpu_pool: TaskPool,
        max_in_flight: usize,
    ) -> Self {
        assert!(max_in_flight > 0, "max_in_flight must be nonzero");
        Self {
            server,
            io_pool,
            cpu_pool,
            limit: max_in_flight,
            state: Arc::new(Mutex::new(SchedulerState {
                in_flight: BTreeMap::new(),
                running: BTreeMap::new(),
                reload_reads: BTreeMap::new(),
                queued: VecDeque::new(),
                admission_waiters: Vec::new(),
                next_admission_seq: 0,
                next_reload_serial: 0,
            })),
        }
    }

    /// Schedule an untyped load. Requests for an identical [`AssetPath`] share
    /// one server invocation and receive clones of its resulting handle.
    pub fn schedule(
        &self,
        path: impl Into<AssetPath>,
        priority: impl Into<Priority>,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_with_ancestors(path.into(), priority.into(), BTreeSet::new())
    }

    /// Schedule an asynchronous reload of an already resident asset.
    ///
    /// Reload flights are independent from ordinary load flights for the same
    /// path. Decoding and dependency resolution happen against staging state;
    /// only the latest reload epoch may replace the resident revision.
    pub fn schedule_reload(
        &self,
        path: impl Into<AssetPath>,
        priority: impl Into<Priority>,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_reload_with_ancestors(path.into(), priority.into(), BTreeSet::new())
    }

    /// Atomically admits a typed soft-reference batch, returning one separately
    /// cancellable ticket per input, in input order. Duplicates share work.
    ///
    /// Refusal performs no I/O and admits/promotes no flights. Request and
    /// distinct-root budgets include duplicates/resident work as documented by
    /// [`SoftLoadBudget`]; scheduler capacity charges only new distinct flights.
    /// Dependencies retain the scheduler's separate bounded queue policy.
    #[expect(
        clippy::result_large_err,
        reason = "structured soft-load diagnostics are part of the scheduler contract"
    )]
    pub fn schedule_soft_batch<A: Asset>(
        &self,
        soft: &[SoftHandle<A>],
        priority: impl Into<Priority>,
        budget: SoftLoadBudget,
    ) -> Result<Vec<LoadTicket>, SoftBatchError> {
        if soft.len() > budget.max_requests {
            return Err(SoftBatchError::RequestBudgetExceeded {
                requested: soft.len(),
                limit: budget.max_requests,
            });
        }
        let paths = self
            .server
            .resolve_soft_batch(soft)
            .map_err(|(index, reason)| SoftBatchError::InvalidHandle { index, reason })?;
        let unique: BTreeSet<_> = paths.iter().cloned().collect();
        if unique.len() > budget.max_unique_assets {
            return Err(SoftBatchError::UniqueAssetBudgetExceeded {
                requested: unique.len(),
                limit: budget.max_unique_assets,
            });
        }
        let priority = priority.into();
        let requested_type = AssetTypeId::of::<A>();
        let (tickets, starts) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            for (index, path) in paths.iter().enumerate() {
                if let Some(existing) = state.in_flight.get(&FlightKey::Load(path.clone()))
                    && existing.requested_type != Some(requested_type)
                {
                    return Err(SoftBatchError::ConflictingFlight {
                        index,
                        path: path.clone(),
                    });
                }
            }
            let required = unique
                .iter()
                .filter(|path| {
                    !state
                        .in_flight
                        .contains_key(&FlightKey::Load((*path).clone()))
                })
                .count();
            let available = if state.queued.is_empty() && state.admission_waiters.is_empty() {
                self.limit.saturating_sub(state.running.len())
            } else {
                0
            };
            if required > available {
                return Err(SoftBatchError::AtCapacity {
                    required,
                    available,
                });
            }
            let mut starts = Vec::with_capacity(required);
            // Keep first occurrence order for equal-priority FIFO dispatch.
            let mut tickets = Vec::with_capacity(paths.len());
            for path in paths {
                let key = FlightKey::Load(path);
                let flight = if let Some(existing) = state.in_flight.get(&key) {
                    existing.priority.boost_to(priority);
                    Arc::clone(existing)
                } else {
                    let flight = Arc::new(Flight {
                        notification: Mutex::new(FlightNotification {
                            result: None,
                            waiters: Vec::new(),
                        }),
                        priority: FlightPriority::new(priority),
                        requested_type: Some(requested_type),
                    });
                    state.in_flight.insert(key.clone(), Arc::clone(&flight));
                    state.running.insert(key.clone(), Arc::clone(&flight));
                    starts.push((key, Arc::clone(&flight)));
                    flight
                };
                tickets.push(LoadTicket::new(flight));
            }
            (tickets, starts)
        };
        for (key, flight) in starts {
            self.spawn_flight(key, flight, priority, BTreeSet::new(), false);
        }
        Ok(tickets)
    }

    fn schedule_with_ancestors(
        &self,
        path: AssetPath,
        priority: Priority,
        ancestors: BTreeSet<AssetPath>,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_typed_with_ancestors(path, priority, None, ancestors)
    }

    fn schedule_reload_with_ancestors(
        &self,
        path: AssetPath,
        priority: Priority,
        ancestors: BTreeSet<AssetPath>,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_reload_typed_with_ancestors(path, priority, None, ancestors)
    }

    fn schedule_typed_with_ancestors(
        &self,
        path: AssetPath,
        priority: Priority,
        requested_type: Option<AssetTypeId>,
        ancestors: BTreeSet<AssetPath>,
    ) -> Result<LoadTicket, ScheduleError> {
        // Normalize untyped requests before admission so a typed dependency and
        // an untyped root request for the same path can share a flight only
        // when loader selection resolves them to the same produced type.
        let requested_type = requested_type.or_else(|| {
            (!path.has_label())
                .then(|| self.server.inferred_asset_type(&path))
                .flatten()
        });
        self.schedule_keyed(
            FlightKey::Load(path),
            priority,
            requested_type,
            ancestors,
            false,
        )
    }

    fn schedule_reload_typed_with_ancestors(
        &self,
        path: AssetPath,
        priority: Priority,
        requested_type: Option<AssetTypeId>,
        ancestors: BTreeSet<AssetPath>,
    ) -> Result<LoadTicket, ScheduleError> {
        let serial = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            let serial = state.next_reload_serial;
            state.next_reload_serial = state
                .next_reload_serial
                .checked_add(1)
                .ok_or(ScheduleError::RequestIdExhausted)?;
            serial
        };
        self.schedule_keyed(
            FlightKey::Reload { path, serial },
            priority,
            requested_type,
            ancestors,
            true,
        )
    }

    fn schedule_keyed(
        &self,
        key: FlightKey,
        priority: Priority,
        requested_type: Option<AssetTypeId>,
        ancestors: BTreeSet<AssetPath>,
        reload: bool,
    ) -> Result<LoadTicket, ScheduleError> {
        let path = match &key {
            FlightKey::Load(path) | FlightKey::Reload { path, .. } => path.clone(),
        };
        let requested_type = requested_type.or_else(|| {
            (!path.has_label())
                .then(|| self.server.inferred_asset_type(&path))
                .flatten()
        });
        let (flight, start_now) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            if let Some(existing) = state.in_flight.get(&key) {
                if let (Some(existing_type), Some(requested_type)) =
                    (existing.requested_type, requested_type)
                    && existing_type != requested_type
                {
                    let conflict = Arc::new(Flight {
                        notification: Mutex::new(FlightNotification {
                            result: None,
                            waiters: Vec::new(),
                        }),
                        priority: FlightPriority::new(priority),
                        requested_type: Some(requested_type),
                    });
                    conflict.complete(Err(LoadFailure::AssemblyFailed(format!(
                        "same asset path requested with conflicting types: {path}"
                    ))));
                    return Ok(LoadTicket {
                        flight: conflict,
                        waiter_token: Arc::new(()),
                        cancellation: Arc::new(Mutex::new(Cancellation {
                            cancelled: false,
                            waker: None,
                        })),
                    });
                }
                existing.priority.boost_to(priority);
                (Arc::clone(existing), false)
            } else {
                let can_start = state.running.len() < self.limit
                    && state.queued.is_empty()
                    && state.admission_waiters.is_empty();
                if !can_start && ancestors.is_empty() {
                    return Err(ScheduleError::AtCapacity { limit: self.limit });
                }
                if !can_start && !ancestors.is_empty() && state.queued.len() >= self.limit {
                    return Err(ScheduleError::DependencyQueueFull { limit: self.limit });
                }
                let flight = Arc::new(Flight {
                    notification: Mutex::new(FlightNotification {
                        result: None,
                        waiters: Vec::new(),
                    }),
                    priority: FlightPriority::new(priority),
                    requested_type,
                });
                state.in_flight.insert(key.clone(), Arc::clone(&flight));
                if can_start {
                    state.running.insert(key.clone(), Arc::clone(&flight));
                    (flight, true)
                } else {
                    state.queued.push_back(QueuedFlight {
                        key: key.clone(),
                        flight: Arc::clone(&flight),
                        priority,
                        ancestors: ancestors.clone(),
                        reload,
                    });
                    (flight, false)
                }
            }
        };

        if start_now {
            self.spawn_flight(key, flight.clone(), priority, ancestors, reload);
        }

        Ok(LoadTicket {
            flight,
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        })
    }

    /// Number of admitted unique requests not yet completed, including
    /// dependency flights waiting for an execution slot.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.state
            .lock()
            .expect("load scheduler poisoned")
            .in_flight
            .len()
    }

    /// Maximum number of concurrently executing assembly flights.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.limit
    }

    fn spawn_flight(
        &self,
        key: FlightKey,
        flight: Arc<Flight>,
        priority: Priority,
        ancestors: BTreeSet<AssetPath>,
        reload: bool,
    ) {
        let scheduler = self.clone();
        let server = Arc::clone(&self.server);
        let pool = self.cpu_pool.clone();
        let task = pool.spawn_future_with_priority(flight.priority.priority_cell(), async move {
            let priority = flight.priority.effective_priority().max(priority);
            let path = match &key {
                FlightKey::Load(path) | FlightKey::Reload { path, .. } => path.clone(),
            };
            let panic_server = Arc::clone(&server);
            let worker_scheduler = scheduler.clone();
            let worker_path = path.clone();
            let worker_flight = Arc::clone(&flight);
            let outcome = match PanicBoundary::new(async move {
                if reload {
                    execute_async_reload(
                        worker_scheduler.clone(),
                        Arc::clone(&server),
                        worker_path.clone(),
                        worker_flight.requested_type,
                        Arc::clone(&worker_flight),
                    )
                    .await
                } else {
                    execute_async_load(
                        worker_scheduler,
                        Arc::clone(&server),
                        worker_path,
                        Arc::clone(&worker_flight),
                        worker_flight.requested_type,
                        priority,
                        ancestors,
                    )
                    .await
                }
            })
            .await
            {
                Ok(outcome) => outcome,
                Err(()) => {
                    if !reload {
                        panic_server.fail_async_load_path(&path, "async load worker panicked");
                    }
                    Err(LoadFailure::WorkerPanicked)
                }
            };
            scheduler.finish_flight(key, flight, outcome);
        });
        task.detach();
    }

    fn finish_flight(&self, key: FlightKey, flight: Arc<Flight>, outcome: ScheduleOutcome) {
        let (wakers, admission_wakers, queued) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            if state
                .running
                .get(&key)
                .is_some_and(|running| Arc::ptr_eq(running, &flight))
            {
                state.running.remove(&key);
            }
            let wakers = flight.complete(outcome);
            if state
                .in_flight
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(current, &flight))
            {
                state.in_flight.remove(&key);
            }
            let queued = self.admit_queued_locked(&mut state);
            let admission_wakers = self.grant_admission_locked(&mut state);
            (wakers, admission_wakers, queued)
        };
        Flight::wake_all(wakers);
        for queued in queued {
            self.spawn_flight(
                queued.key,
                queued.flight,
                queued.priority,
                queued.ancestors,
                queued.reload,
            );
        }
        Flight::wake_all(admission_wakers);
    }

    fn admit_queued_locked(&self, state: &mut SchedulerState) -> Vec<QueuedFlight> {
        let mut admitted = Vec::new();
        let reservations = state
            .admission_waiters
            .iter()
            .filter(|waiter| waiter.granted)
            .count();
        while state.running.len().saturating_add(reservations) < self.limit {
            if state.queued.is_empty() {
                break;
            }
            let mut index = 0;
            for candidate in 1..state.queued.len() {
                let candidate_priority =
                    state.queued[candidate].flight.priority.effective_priority();
                let selected_priority = state.queued[index].flight.priority.effective_priority();
                if candidate_priority > selected_priority {
                    index = candidate;
                }
            }
            let queued = state.queued.remove(index).expect("queued index is valid");
            state
                .running
                .insert(queued.key.clone(), Arc::clone(&queued.flight));
            admitted.push(queued);
        }
        admitted
    }

    fn release_execution_slot(&self, key: &FlightKey, flight: &Arc<Flight>) {
        let (admission_wakers, queued) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            if state
                .running
                .get(key)
                .is_some_and(|running| Arc::ptr_eq(running, flight))
            {
                state.running.remove(key);
            }
            let queued = self.admit_queued_locked(&mut state);
            let admission_wakers = self.grant_admission_locked(&mut state);
            (admission_wakers, queued)
        };
        for queued in queued {
            self.spawn_flight(
                queued.key,
                queued.flight,
                queued.priority,
                queued.ancestors,
                queued.reload,
            );
        }
        Flight::wake_all(admission_wakers);
    }

    fn acquire_execution_slot(
        &self,
        key: FlightKey,
        flight: Arc<Flight>,
        priority: Priority,
    ) -> ExecutionSlotFuture {
        flight.priority.boost_to(priority);
        ExecutionSlotFuture {
            scheduler: self.clone(),
            key,
            flight,
            token: Arc::new(()),
        }
    }

    fn reload_read(
        &self,
        source_revision: u64,
        sources: &AssetSources,
        loaders: &AssetLoaders,
        path: &AssetPath,
        type_id: AssetTypeId,
        priority: &FlightPriority,
    ) -> ReloadReadTicket {
        let key = ReloadReadKey {
            source_revision,
            path: path.clone(),
            type_id,
        };
        let (flight, start) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            // Source revisions are immutable snapshots for a reload. Drop
            // flights from older snapshots so a later mount can never reuse
            // old bytes and the cache remains bounded by one revision.
            state
                .reload_reads
                .retain(|candidate, _| candidate.source_revision == source_revision);
            if let Some(flight) = state.reload_reads.get(&key) {
                (Arc::clone(flight), false)
            } else {
                let flight = Arc::new(ReloadReadFlight {
                    notification: Mutex::new(ReloadReadNotification {
                        result: None,
                        waiters: Vec::new(),
                    }),
                    priority: FlightPriority::new(priority.effective_priority()),
                });
                state.reload_reads.insert(key, Arc::clone(&flight));
                (flight, true)
            }
        };
        priority.inherit(&flight.priority);
        if start {
            let read_sources = sources.clone();
            let read_loaders = loaders.clone();
            let read_path = path.clone();
            let read_flight = Arc::clone(&flight);
            if self.io_pool.is_single_threaded() {
                let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    read_loaders.read_for_type(&read_sources, &read_path, type_id)
                })) {
                    Ok(Ok(bytes)) => Ok(Arc::<[u8]>::from(bytes)),
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err("asset source or loader panicked".to_string()),
                };
                ReloadReadFlight::wake_all(read_flight.complete(result));
            } else {
                let pool = self.io_pool.clone();
                let priority = flight.priority.priority_cell();
                self.io_pool
                    .spawn_future_with_priority(flight.priority.priority_cell(), async move {
                        let result = read_loader_bytes_async(
                            &read_loaders,
                            &read_sources,
                            &read_path,
                            type_id,
                            &pool,
                            priority,
                        )
                        .await
                        .map(Arc::<[u8]>::from);
                        ReloadReadFlight::wake_all(read_flight.complete(result));
                    })
                    .detach();
            }
        }
        ReloadReadTicket {
            flight,
            token: Arc::new(()),
        }
    }

    /// Reserve available execution slots for the highest-priority waiters.
    ///
    /// The waiter stays in the admission list until it polls again and consumes
    /// its reservation. This is an explicit handoff: a newly-created waiter
    /// cannot observe the temporarily-free slot and bypass a waiter that has
    /// already been selected. Wakers are returned for notification after the
    /// scheduler lock is released.
    fn grant_admission_locked(&self, state: &mut SchedulerState) -> Vec<Waker> {
        let granted = state
            .admission_waiters
            .iter()
            .filter(|waiter| waiter.granted)
            .count();
        let available = self
            .limit
            .saturating_sub(state.running.len().saturating_add(granted));
        let mut wakers = Vec::with_capacity(available.min(state.admission_waiters.len()));
        for _ in 0..available {
            let Some(index) = state
                .admission_waiters
                .iter()
                .enumerate()
                .filter(|(_, waiter)| !waiter.granted)
                .max_by(|(_, left), (_, right)| {
                    left.priority
                        .effective_priority()
                        .cmp(&right.priority.effective_priority())
                        .then_with(|| right.seq.cmp(&left.seq))
                })
                .map(|(index, _)| index)
            else {
                break;
            };
            let waiter = &mut state.admission_waiters[index];
            waiter.granted = true;
            wakers.push(waiter.waker.clone());
        }
        wakers
    }
}

async fn execute_async_load(
    scheduler: LoadScheduler,
    server: Arc<AssetServer>,
    path: AssetPath,
    flight: Arc<Flight>,
    requested_type: Option<AssetTypeId>,
    priority: Priority,
    mut ancestors: BTreeSet<AssetPath>,
) -> ScheduleOutcome {
    let requested_path = path.clone();
    let source_path = path.clone().without_label();
    let requested_label = path.has_label();
    let source_was_ancestor = ancestors.contains(&source_path);
    if !ancestors.insert(requested_path.clone()) {
        return Err(LoadFailure::AssemblyFailed(format!(
            "dependency cycle at {requested_path}"
        )));
    }
    if requested_label {
        if source_was_ancestor {
            return Err(LoadFailure::AssemblyFailed(format!(
                "dependency cycle at {source_path}"
            )));
        }
        scheduler.release_execution_slot(&FlightKey::Load(requested_path.clone()), &flight);
        let source_ticket = match scheduler.schedule_with_ancestors(
            source_path.clone(),
            flight.priority.effective_priority(),
            ancestors.clone(),
        ) {
            Ok(ticket) => ticket,
            Err(error) => return Err(LoadFailure::AssemblyFailed(error.to_string())),
        };
        flight.priority.inherit(&source_ticket.flight.priority);
        let source_result = source_ticket.await;
        return match source_result {
            Ok(handle) if server.is_loaded(handle.id()) => {
                let handle = server.load_untyped(requested_path.clone()).ok_or_else(|| {
                    LoadFailure::AssemblyFailed(format!(
                        "labeled asset was not emitted by {source_path}: {requested_path}"
                    ))
                })?;
                if requested_type.is_some_and(|expected| expected != handle.id().type_id()) {
                    return Err(LoadFailure::AssemblyFailed(format!(
                        "labeled asset type does not match requested type: {requested_path}"
                    )));
                }
                Ok(handle)
            }
            Ok(_) => Err(LoadFailure::AssemblyFailed(format!(
                "source asset failed to load: {source_path}"
            ))),
            Err(error) => Err(LoadFailure::AssemblyFailed(format!(
                "source asset {source_path} failed: {error:?}"
            ))),
        };
    }
    let preparation = match server.prepare_async_load(source_path.clone(), requested_type) {
        Ok(preparation) => preparation,
        Err(error) if error.starts_with("no loader for ") => return Err(LoadFailure::NoLoader),
        Err(error) => return Err(LoadFailure::AssemblyFailed(error)),
    };
    let plan = match preparation {
        AsyncLoadPreparation::Existing(handle) => {
            if server.is_loaded(handle.id()) {
                return Ok(handle);
            }
            return Err(LoadFailure::AssemblyFailed(format!(
                "asset is already loading outside this scheduler: {source_path}"
            )));
        }
        AsyncLoadPreparation::New(plan) => plan,
    };
    let bytes = if scheduler.io_pool.is_single_threaded() {
        read_bytes_inline(&plan)
    } else {
        let read_sources = plan.sources.clone();
        let read_loaders = plan.loaders.clone();
        let read_path = plan.path.clone();
        let read_type_id = plan.type_id;
        let pool = scheduler.io_pool.clone();
        let priority = flight.priority.priority_cell();
        scheduler
            .io_pool
            .spawn_future_with_priority(flight.priority.priority_cell(), async move {
                read_loader_bytes_async(
                    &read_loaders,
                    &read_sources,
                    &read_path,
                    read_type_id,
                    &pool,
                    priority,
                )
                .await
            })
            .await
    };
    let bytes = match bytes {
        Ok(bytes) => bytes,
        Err(error) => {
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
    };
    let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.loaders
            .load_for_type(&plan.path, plan.type_id, &bytes)
            .map_err(|error| error.to_string())
    })) {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
        Err(_) => {
            let error = "asset source or loader panicked".to_string();
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
    };
    for dependency in &output.primary.dependencies {
        if ancestors.contains(&dependency.path) {
            let error = format!("dependency cycle at {}", dependency.path);
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
    }
    let has_labeled_dependencies = output
        .labeled
        .iter()
        .any(|labeled| !labeled.dependencies.is_empty());
    if !output.primary.dependencies.is_empty() || has_labeled_dependencies {
        scheduler.release_execution_slot(&FlightKey::Load(requested_path.clone()), &flight);
    }
    let has_dependencies = !output.primary.dependencies.is_empty();
    let mut dependencies = Vec::with_capacity(output.primary.dependencies.len());
    for dependency in &output.primary.dependencies {
        let ticket = match scheduler.schedule_typed_with_ancestors(
            dependency.path.clone(),
            flight.priority.effective_priority(),
            dependency.expected_type,
            ancestors.clone(),
        ) {
            Ok(ticket) => ticket,
            Err(error) => {
                let error = error.to_string();
                server.fail_async_load(&plan, &error);
                return Err(LoadFailure::AssemblyFailed(error));
            }
        };
        flight.priority.inherit(&ticket.flight.priority);
        let dependency_path = dependency.path.clone();
        match ticket.await {
            Ok(handle) if server.is_loaded(handle.id()) => dependencies.push(handle),
            Ok(_) => {
                let error = format!("dependency {} failed to load", dependency_path);
                server.fail_async_load(&plan, &error);
                return Err(LoadFailure::AssemblyFailed(error));
            }
            Err(error) => {
                let error = format!("dependency {} failed: {error:?}", dependency_path);
                server.fail_async_load(&plan, &error);
                return Err(LoadFailure::AssemblyFailed(error));
            }
        }
    }
    // Do not hold a borrow of `ErasedLoadedAsset` across dependency awaits:
    // its boxed payload is `dyn Any + Send`, not `Sync`, so retaining that
    // borrow would make the worker future fail the task pool's `Send` bound.
    // Dependency metadata is owned and `Send`, while the erased payload stays
    // parked in `output` until the atomic commit below.
    let labeled_requests: Vec<_> = output
        .labeled
        .iter()
        .map(|labeled| (labeled.label.clone(), labeled.dependencies.clone()))
        .collect();
    let mut labeled_dependencies = Vec::with_capacity(labeled_requests.len());
    for (label, requests) in labeled_requests {
        let label_path = source_path.clone().with_label(label);
        let mut label_ancestors = ancestors.clone();
        let is_requested_source_label =
            ancestors.contains(&label_path) && label_path.clone().without_label() == source_path;
        if !is_requested_source_label && !label_ancestors.insert(label_path.clone()) {
            let error = format!("dependency cycle at {label_path}");
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
        let mut resolved = Vec::with_capacity(requests.len());
        for dependency in requests {
            if label_ancestors.contains(&dependency.path) {
                let error = format!("dependency cycle at {}", dependency.path);
                server.fail_async_load(&plan, &error);
                return Err(LoadFailure::AssemblyFailed(error));
            }
            let ticket = match scheduler.schedule_typed_with_ancestors(
                dependency.path.clone(),
                flight.priority.effective_priority(),
                dependency.expected_type,
                label_ancestors.clone(),
            ) {
                Ok(ticket) => ticket,
                Err(error) => {
                    let error = error.to_string();
                    server.fail_async_load(&plan, &error);
                    return Err(LoadFailure::AssemblyFailed(error));
                }
            };
            flight.priority.inherit(&ticket.flight.priority);
            let dependency_path = dependency.path.clone();
            match ticket.await {
                Ok(handle) if server.is_loaded(handle.id()) => resolved.push(handle),
                Ok(_) => {
                    let error = format!("dependency {} failed to load", dependency_path);
                    server.fail_async_load(&plan, &error);
                    return Err(LoadFailure::AssemblyFailed(error));
                }
                Err(error) => {
                    let error = format!("dependency {} failed: {error:?}", dependency_path);
                    server.fail_async_load(&plan, &error);
                    return Err(LoadFailure::AssemblyFailed(error));
                }
            }
        }
        labeled_dependencies.push(resolved);
    }
    if has_dependencies || has_labeled_dependencies {
        scheduler
            .acquire_execution_slot(
                FlightKey::Load(requested_path.clone()),
                Arc::clone(&flight),
                priority,
            )
            .await;
    }
    server
        .commit_async_load(*plan, output, dependencies, labeled_dependencies)
        .map_err(LoadFailure::AssemblyFailed)
}

async fn execute_async_reload(
    scheduler: LoadScheduler,
    server: Arc<AssetServer>,
    path: AssetPath,
    requested_type: Option<AssetTypeId>,
    flight: Arc<Flight>,
) -> ScheduleOutcome {
    let plan = match server.prepare_async_reload(path.clone(), requested_type) {
        Ok(plan) => plan,
        Err(error) if error.starts_with("no loader for ") => return Err(LoadFailure::NoLoader),
        Err(error) => return Err(LoadFailure::AssemblyFailed(error)),
    };
    let output = decode_reload_source(
        &scheduler,
        plan.source_revision,
        &plan.sources,
        &plan.loaders,
        &plan.path,
        plan.type_id,
        &flight.priority,
    )
    .await
    .map_err(LoadFailure::AssemblyFailed)?;
    let mut decoded = BTreeMap::new();
    let mut pending = VecDeque::new();
    decoded.insert(plan.path.clone(), (plan.type_id, output));
    pending.push_back(plan.path.clone());
    const MAX_REVISION_CLOSURE: usize = 4096;
    while let Some(current_path) = pending.pop_front() {
        if decoded.len() > MAX_REVISION_CLOSURE {
            return Err(LoadFailure::AssemblyFailed(
                "async reload dependency closure exceeds the bounded candidate limit".into(),
            ));
        }
        let requests = {
            let (_, output) = decoded
                .get(&current_path)
                .expect("pending paths are inserted before they are queued");
            let mut requests = Vec::new();
            requests.extend(output.primary.dependencies.iter().cloned());
            for labeled in &output.labeled {
                requests.extend(labeled.dependencies.iter().cloned());
            }
            requests
        };
        for dependency in requests {
            let source_path = dependency.path.clone().without_label();
            if decoded.contains_key(&source_path) {
                continue;
            }
            let dependency_type = if dependency.path.has_label() {
                plan.loaders
                    .policy()
                    .resolve_untyped(&source_path)
                    .and_then(|loader| plan.loaders.policy().produced_type(loader))
                    .ok_or_else(|| {
                        LoadFailure::AssemblyFailed(format!("no loader for {source_path}"))
                    })?
            } else {
                dependency
                    .expected_type
                    .or_else(|| {
                        plan.loaders
                            .policy()
                            .resolve_untyped(&source_path)
                            .and_then(|loader| plan.loaders.policy().produced_type(loader))
                    })
                    .ok_or_else(|| {
                        LoadFailure::AssemblyFailed(format!("no loader for {source_path}"))
                    })?
            };
            let output = decode_reload_source(
                &scheduler,
                plan.source_revision,
                &plan.sources,
                &plan.loaders,
                &source_path,
                dependency_type,
                &flight.priority,
            )
            .await
            .map_err(LoadFailure::AssemblyFailed)?;
            if output.primary.type_id != dependency_type {
                return Err(LoadFailure::AssemblyFailed(format!(
                    "loader produced an unexpected asset type for {source_path}"
                )));
            }
            decoded.insert(source_path.clone(), (dependency_type, output));
            pending.push_back(source_path);
        }
    }
    let candidate = AsyncReloadCandidate {
        assets: decoded
            .into_iter()
            .map(|(path, (type_id, output))| AsyncReloadCandidateAsset {
                path,
                type_id,
                output,
            })
            .collect(),
    };
    match server.commit_async_reload(plan, candidate) {
        Ok(handle) => Ok(handle),
        Err(AsyncReloadCommitError::Superseded) => Err(LoadFailure::Superseded),
        Err(AsyncReloadCommitError::Rejected(error)) => Err(LoadFailure::AssemblyFailed(error)),
    }
}

/// Reads and decodes one source revision outside the server state mutex.
/// Reload assembly deliberately owns its decoded outputs until commit so a
/// newly introduced dependency, label, or recursive dependency cannot observe
/// a partially published resident graph.
async fn decode_reload_source(
    scheduler: &LoadScheduler,
    source_revision: u64,
    sources: &AssetSources,
    loaders: &AssetLoaders,
    path: &AssetPath,
    type_id: AssetTypeId,
    priority: &FlightPriority,
) -> Result<FullLoadOutput, String> {
    let bytes = scheduler
        .reload_read(source_revision, sources, loaders, path, type_id, priority)
        .await?;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loaders
            .load_for_type(path, type_id, &bytes)
            .map_err(|error| error.to_string())
    })) {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("asset source or loader panicked".to_string()),
    }
}

struct PanicBoundary<F> {
    future: Pin<Box<F>>,
}

impl<F> PanicBoundary<F> {
    fn new(future: F) -> Self {
        Self {
            future: Box::pin(future),
        }
    }
}

impl<F> Future for PanicBoundary<F>
where
    F: Future,
{
    type Output = Result<F::Output, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            this.future.as_mut().poll(cx)
        })) {
            Ok(Poll::Ready(output)) => Poll::Ready(Ok(output)),
            Ok(Poll::Pending) => Poll::Pending,
            Err(_) => Poll::Ready(Err(())),
        }
    }
}

struct ExecutionSlotFuture {
    scheduler: LoadScheduler,
    key: FlightKey,
    flight: Arc<Flight>,
    token: Arc<()>,
}

impl Future for ExecutionSlotFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let (result, wakers) = {
            let mut state = this
                .scheduler
                .state
                .lock()
                .expect("load scheduler poisoned");
            let own_index = state
                .admission_waiters
                .iter()
                .position(|waiter| Arc::ptr_eq(&waiter.token, &this.token));

            if own_index.is_some_and(|index| state.admission_waiters[index].granted) {
                let index = own_index.expect("granted waiter must have an index");
                state.admission_waiters.remove(index);
                state
                    .running
                    .insert(this.key.clone(), Arc::clone(&this.flight));
                (Poll::Ready(()), Vec::new())
            } else if own_index.is_none()
                && state.running.len() < this.scheduler.limit
                && state.queued.is_empty()
                && state.admission_waiters.is_empty()
            {
                state
                    .running
                    .insert(this.key.clone(), Arc::clone(&this.flight));
                (Poll::Ready(()), Vec::new())
            } else {
                if let Some(index) = own_index {
                    let waiter = &mut state.admission_waiters[index];
                    if !waiter.waker.will_wake(cx.waker()) {
                        waiter.waker = cx.waker().clone();
                    }
                } else {
                    let seq = state.next_admission_seq;
                    state.next_admission_seq = state.next_admission_seq.saturating_add(1);
                    state.admission_waiters.push(AdmissionWaiter {
                        token: Arc::clone(&this.token),
                        priority: this.flight.priority.clone(),
                        seq,
                        waker: cx.waker().clone(),
                        granted: false,
                    });
                }
                let wakers = if state.queued.is_empty() {
                    this.scheduler.grant_admission_locked(&mut state)
                } else {
                    Vec::new()
                };
                let own_granted = state
                    .admission_waiters
                    .iter()
                    .position(|waiter| Arc::ptr_eq(&waiter.token, &this.token))
                    .is_some_and(|index| state.admission_waiters[index].granted);
                if own_granted {
                    let index = state
                        .admission_waiters
                        .iter()
                        .position(|waiter| Arc::ptr_eq(&waiter.token, &this.token))
                        .expect("granted waiter must have an index");
                    state.admission_waiters.remove(index);
                    state
                        .running
                        .insert(this.key.clone(), Arc::clone(&this.flight));
                    (Poll::Ready(()), wakers)
                } else {
                    (Poll::Pending, wakers)
                }
            }
        };
        Flight::wake_all(wakers);
        result
    }
}

impl Drop for ExecutionSlotFuture {
    fn drop(&mut self) {
        let (wakers, queued) = {
            let mut state = self
                .scheduler
                .state
                .lock()
                .expect("load scheduler poisoned");
            let was_granted = state
                .admission_waiters
                .iter()
                .position(|waiter| Arc::ptr_eq(&waiter.token, &self.token))
                .map(|index| {
                    let waiter = state.admission_waiters.remove(index);
                    waiter.granted
                })
                .unwrap_or(false);
            if was_granted {
                let queued = self.scheduler.admit_queued_locked(&mut state);
                let wakers = self.scheduler.grant_admission_locked(&mut state);
                (wakers, queued)
            } else {
                (Vec::new(), Vec::new())
            }
        };
        for queued in queued {
            self.scheduler.spawn_flight(
                queued.key,
                queued.flight,
                queued.priority,
                queued.ancestors,
                queued.reload,
            );
        }
        Flight::wake_all(wakers);
    }
}

fn read_bytes_inline(plan: &crate::server::AsyncLoadPlan) -> Result<Vec<u8>, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        read_bytes_for_plan(&plan.sources, plan)
    })) {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err("asset source or loader panicked".to_string()),
    }
}

async fn read_loader_bytes_async(
    loaders: &AssetLoaders,
    sources: &AssetSources,
    path: &AssetPath,
    requested: AssetTypeId,
    pool: &TaskPool,
    priority: Arc<PriorityCell>,
) -> Result<Vec<u8>, String> {
    PanicBoundary::new(loaders.read_for_type_async(sources, path, requested, pool, priority))
        .await
        .unwrap_or_else(|()| Err("asset source or loader panicked".to_string()))
}

fn read_bytes_for_plan(
    sources: &AssetSources,
    plan: &crate::server::AsyncLoadPlan,
) -> Result<Vec<u8>, String> {
    plan.loaders
        .read_for_type(sources, &plan.path, plan.type_id)
}

/// A future for one consumer of a scheduled load. Cancelling a ticket only
/// cancels that consumer's wait; shared server work is allowed to finish.
pub struct LoadTicket {
    flight: Arc<Flight>,
    waiter_token: Arc<()>,
    cancellation: Arc<Mutex<Cancellation>>,
}

struct Cancellation {
    cancelled: bool,
    waker: Option<Waker>,
}

impl LoadTicket {
    fn new(flight: Arc<Flight>) -> Self {
        Self {
            flight,
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        }
    }

    /// Promotes this shared flight, its queued I/O/CPU continuations, and known
    /// dependencies. Lower priorities never demote work needed by another
    /// consumer. A cancelled consumer cannot promote shared work.
    pub fn reprioritize(&self, priority: impl Into<Priority>) {
        let cancellation = self.cancellation.lock().expect("ticket waker poisoned");
        if !cancellation.cancelled {
            self.flight.priority.boost_to(priority.into());
        }
    }

    /// Current effective task-pool priority, including inheritance/promotions.
    #[must_use]
    pub fn effective_priority(&self) -> Priority {
        self.flight.priority.effective_priority()
    }

    /// Stop waiting for this request. This is cooperative at the API boundary:
    /// shared work and work already running on the synchronous server continue.
    pub fn cancel(&self) {
        let waker = self
            .cancellation
            .lock()
            .expect("ticket waker poisoned")
            .cancel();
        self.flight.unregister(&self.waiter_token);
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// Whether this ticket has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .lock()
            .expect("ticket waker poisoned")
            .cancelled
    }
}

impl Future for LoadTicket {
    type Output = ScheduleOutcome;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut cancellation = self.cancellation.lock().expect("ticket waker poisoned");
        if cancellation.cancelled {
            return Poll::Ready(Err(LoadFailure::Cancelled));
        }
        cancellation.waker = Some(cx.waker().clone());
        let result = self.flight.poll(&self.waiter_token, cx);
        if cancellation.cancelled {
            cancellation.waker = None;
            self.flight.unregister(&self.waiter_token);
            Poll::Ready(Err(LoadFailure::Cancelled))
        } else {
            if result.is_ready() {
                cancellation.waker = None;
                self.flight.unregister(&self.waiter_token);
            }
            result
        }
    }
}

impl Drop for LoadTicket {
    fn drop(&mut self) {
        self.flight.unregister(&self.waiter_token);
    }
}

impl Cancellation {
    fn cancel(&mut self) -> Option<Waker> {
        self.cancelled = true;
        self.waker.take()
    }
}

#[cfg(test)]
#[path = "../text/unit/scheduler_priority.rs"]
mod priority_tests;

#[cfg(test)]
#[path = "scheduler_range_tests.rs"]
mod range_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Asset, AssetLoader, AssetMeta, AssetPath, AssetReadPlan, AssetReader, LoadContext,
        LoadError, MemSource, RangePart, ReadError, RecursiveDependencyLoadState,
    };
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use prism_tasks::TaskPool;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct TestAsset;

    impl Asset for TestAsset {
        const TYPE_NAME: &'static str = "prism_asset::scheduler::tests::TestAsset";
    }

    #[derive(Debug)]
    struct OtherAsset;

    impl Asset for OtherAsset {
        const TYPE_NAME: &'static str = "prism_asset::scheduler::tests::OtherAsset";
    }

    #[derive(Debug)]
    struct ReloadAsset(u8);

    impl Asset for ReloadAsset {
        const TYPE_NAME: &'static str = "prism_asset::scheduler::tests::ReloadAsset";
    }

    struct ReloadLoader;

    impl AssetLoader for ReloadLoader {
        type Asset = ReloadAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["reload"]
        }

        fn load(&self, bytes: &[u8], _ctx: &mut LoadContext) -> Result<ReloadAsset, LoadError> {
            match bytes.first().copied() {
                Some(value) if value > 0 => Ok(ReloadAsset(value)),
                _ => Err(LoadError::Malformed("reload payload is invalid".into())),
            }
        }
    }

    struct ReloadGateSource {
        value: u8,
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
    }

    impl AssetReader for ReloadGateSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Ok(vec![self.value])
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 1,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct ReloadRootLoader;

    impl AssetLoader for ReloadRootLoader {
        type Asset = TestAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["rreload"]
        }

        fn load(&self, bytes: &[u8], ctx: &mut LoadContext) -> Result<TestAsset, LoadError> {
            let dependency = match bytes.first().copied() {
                Some(b'a') => "a.rdep",
                Some(b'b') => "b.rdep",
                _ => return Err(LoadError::Malformed("unknown reload dependency".into())),
            };
            ctx.add_typed_dependency(
                AssetPath::parse(dependency),
                AssetTypeId::of::<OtherAsset>(),
            );
            Ok(TestAsset)
        }
    }

    struct ReloadDependencyLoader;

    impl AssetLoader for ReloadDependencyLoader {
        type Asset = OtherAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["rdep"]
        }

        fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext) -> Result<OtherAsset, LoadError> {
            Ok(OtherAsset)
        }
    }

    struct TestLoader;

    impl AssetLoader for TestLoader {
        type Asset = TestAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["sched"]
        }

        fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext) -> Result<TestAsset, LoadError> {
            Ok(TestAsset)
        }
    }

    struct LabeledLoader;

    impl AssetLoader for LabeledLoader {
        type Asset = TestAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["scene"]
        }

        fn load(&self, _bytes: &[u8], ctx: &mut LoadContext) -> Result<TestAsset, LoadError> {
            ctx.add_labeled_asset("mesh", OtherAsset);
            Ok(TestAsset)
        }
    }

    struct OtherLoader;

    impl AssetLoader for OtherLoader {
        type Asset = OtherAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["sched"]
        }

        fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext) -> Result<OtherAsset, LoadError> {
            Ok(OtherAsset)
        }
    }

    struct GateSource {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        block_once: AtomicBool,
    }

    impl AssetReader for GateSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            if self.block_once.swap(false, Ordering::AcqRel) {
                self.started.store(true, Ordering::Release);
                while !self.release.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
            Ok(Vec::new())
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 0,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct PanicSource;

    impl AssetReader for PanicSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            panic!("test reader panic")
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 0,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct RangeOnlySource {
        bytes: Arc<[u8]>,
        reads: Arc<AtomicUsize>,
        ranges: Arc<AtomicUsize>,
    }

    impl AssetReader for RangeOnlySource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::AcqRel);
            Err(ReadError::Unsupported)
        }

        fn read_range(&self, _path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
            self.ranges.fetch_add(1, Ordering::AcqRel);
            let end = offset.checked_add(len).ok_or(ReadError::OutOfRange {
                offset,
                len,
                size: self.bytes.len() as u64,
            })?;
            if end > self.bytes.len() as u64 {
                return Err(ReadError::OutOfRange {
                    offset,
                    len,
                    size: self.bytes.len() as u64,
                });
            }
            Ok(self.bytes[offset as usize..end as usize].to_vec())
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: self.bytes.len() as u64,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct RangeAwareReloadLoader;

    impl AssetLoader for RangeAwareReloadLoader {
        type Asset = ReloadAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["ranged"]
        }

        fn read_plan(
            &self,
            _path: &AssetPath,
            _metadata: AssetMeta,
        ) -> Result<AssetReadPlan, LoadError> {
            AssetReadPlan::ranges([RangePart::new(2, 1), RangePart::new(0, 2)])
                .map_err(|error| LoadError::LimitExceeded(error.to_string()))
        }

        fn load(&self, bytes: &[u8], _ctx: &mut LoadContext) -> Result<ReloadAsset, LoadError> {
            if bytes.len() != 3 || bytes[0] != bytes[2] + bytes[1] {
                return Err(LoadError::Malformed(
                    "range declaration order was not preserved".into(),
                ));
            }
            bytes
                .first()
                .copied()
                .filter(|value| *value > 0)
                .map(ReloadAsset)
                .ok_or_else(|| LoadError::Malformed("ranged payload is invalid".into()))
        }
    }

    struct ParallelSource {
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        release: Arc<AtomicBool>,
    }

    impl AssetReader for ParallelSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
            self.peak.fetch_max(active, Ordering::AcqRel);
            while !self.release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            self.active.fetch_sub(1, Ordering::AcqRel);
            Ok(Vec::new())
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 0,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct ParentLoader;

    impl AssetLoader for ParentLoader {
        type Asset = TestAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["parent"]
        }

        fn load(&self, _bytes: &[u8], ctx: &mut LoadContext) -> Result<TestAsset, LoadError> {
            ctx.add_dependency(AssetPath::parse("child.sched"));
            Ok(TestAsset)
        }
    }

    struct CycleLoader;

    impl AssetLoader for CycleLoader {
        type Asset = TestAsset;

        fn extensions(&self) -> &'static [&'static str] {
            &["cyc"]
        }

        fn load(&self, _bytes: &[u8], ctx: &mut LoadContext) -> Result<TestAsset, LoadError> {
            let dependency = if ctx.path().path().starts_with("a.") {
                "b.cyc"
            } else {
                "a.cyc"
            };
            ctx.add_dependency(AssetPath::parse(dependency));
            Ok(TestAsset)
        }
    }

    fn gated_server() -> (Arc<AssetServer>, Arc<AtomicBool>, Arc<AtomicBool>) {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.mount(
            "gate",
            Arc::new(GateSource {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
                block_once: AtomicBool::new(true),
            }),
            0,
        );
        (server, started, release)
    }

    #[test]
    fn async_reload_replaces_value_without_changing_handle_identity() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(ReloadLoader);
        let mut initial = MemSource::new();
        initial.insert("hero.reload", [1_u8]);
        server.mount("mem", Arc::new(initial), 0);
        let resident = server.load::<ReloadAsset>("mem://hero.reload");
        let id = resident.id();

        let mut revised = MemSource::new();
        revised.insert("hero.reload", [2_u8]);
        server.mount("mem", Arc::new(revised), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
        let result = scheduler
            .cpu_pool
            .block_on(
                scheduler
                    .schedule_reload("mem://hero.reload", Priority::Normal)
                    .unwrap(),
            )
            .expect("reload should commit");

        assert_eq!(result.id(), resident.untyped_id());
        assert_eq!(result.id(), id.untyped());
        assert_eq!(server.with_asset(id, |asset| asset.0), Some(2));
        assert!(server.is_loaded(id.untyped()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn async_initial_load_and_reload_use_declared_range_reads() {
        let reads = Arc::new(AtomicUsize::new(0));
        let ranges = Arc::new(AtomicUsize::new(0));
        let server = Arc::new(AssetServer::new());
        server.register_loader(RangeAwareReloadLoader);
        server.mount(
            "base",
            Arc::new(RangeOnlySource {
                bytes: Arc::<[u8]>::from([1_u8, 2, 3]),
                reads: Arc::clone(&reads),
                ranges: Arc::clone(&ranges),
            }),
            0,
        );
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
        let resident = scheduler
            .cpu_pool
            .block_on(scheduler.schedule("hero.ranged", Priority::Normal).unwrap())
            .expect("initial range load should commit");
        let asset_id = resident
            .id()
            .typed::<ReloadAsset>()
            .expect("declared loader type should match the resident asset");
        assert_eq!(server.with_asset(asset_id, |asset| asset.0), Some(3));

        server.mount(
            "patch",
            Arc::new(RangeOnlySource {
                bytes: Arc::<[u8]>::from([2_u8, 4, 6]),
                reads: Arc::clone(&reads),
                ranges: Arc::clone(&ranges),
            }),
            10,
        );
        scheduler
            .cpu_pool
            .block_on(
                scheduler
                    .schedule_reload("hero.ranged", Priority::Normal)
                    .unwrap(),
            )
            .expect("range reload should commit");
        assert_eq!(server.with_asset(asset_id, |asset| asset.0), Some(6));
        assert_eq!(reads.load(Ordering::Acquire), 0);
        assert_eq!(ranges.load(Ordering::Acquire), 4);
    }

    #[test]
    fn synchronous_load_uses_reordered_declared_ranges() {
        let reads = Arc::new(AtomicUsize::new(0));
        let ranges = Arc::new(AtomicUsize::new(0));
        let server = AssetServer::new();
        server.register_loader(RangeAwareReloadLoader);
        server.mount(
            "mem",
            Arc::new(RangeOnlySource {
                bytes: Arc::<[u8]>::from([1_u8, 2, 3]),
                reads: Arc::clone(&reads),
                ranges: Arc::clone(&ranges),
            }),
            0,
        );
        let asset = server.load::<ReloadAsset>("hero.ranged");
        assert_eq!(server.with_asset(asset.id(), |value| value.0), Some(3));
        assert_eq!(reads.load(Ordering::Acquire), 0);
        assert_eq!(ranges.load(Ordering::Acquire), 2);
    }

    #[test]
    fn truncated_range_reload_retains_the_resident_revision() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(RangeAwareReloadLoader);
        let mut initial = MemSource::new();
        initial.insert("hero.ranged", [1_u8, 2, 3]);
        server.mount("mem", Arc::new(initial), 0);
        let asset = server.load::<ReloadAsset>("hero.ranged");
        let mut truncated = MemSource::new();
        truncated.insert("hero.ranged", [1_u8]);
        server.mount("mem", Arc::new(truncated), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
        assert!(scheduler
            .cpu_pool
            .block_on(
                scheduler
                    .schedule_reload("hero.ranged", Priority::Normal)
                    .unwrap()
            )
            .is_err());
        assert_eq!(server.with_asset(asset.id(), |value| value.0), Some(3));
    }

    #[test]
    fn async_reload_failure_keeps_last_known_good_value() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(ReloadLoader);
        let mut initial = MemSource::new();
        initial.insert("hero.reload", [7_u8]);
        server.mount("mem", Arc::new(initial), 0);
        let resident = server.load::<ReloadAsset>("mem://hero.reload");

        let mut broken = MemSource::new();
        broken.insert("hero.reload", [0_u8]);
        server.mount("mem", Arc::new(broken), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
        let result = scheduler.cpu_pool.block_on(
            scheduler
                .schedule_reload("mem://hero.reload", Priority::Normal)
                .unwrap(),
        );

        assert!(matches!(result, Err(LoadFailure::AssemblyFailed(_))));
        assert_eq!(server.with_asset(resident.id(), |asset| asset.0), Some(7));
        assert!(server.is_loaded(resident.untyped_id()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn older_async_reload_is_superseded_without_overwriting_newer_value() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(ReloadLoader);
        let mut initial = MemSource::new();
        initial.insert("hero.reload", [1_u8]);
        server.mount("base", Arc::new(initial), 0);
        let resident = server.load::<ReloadAsset>("hero.reload");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        server.mount(
            "reload-one",
            Arc::new(ReloadGateSource {
                value: 2,
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            10,
        );
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 2);
        let first = scheduler
            .schedule_reload("hero.reload", Priority::Normal)
            .expect("first reload admitted");
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }

        let mut newer = MemSource::new();
        newer.insert("hero.reload", [3_u8]);
        server.mount("reload-two", Arc::new(newer), 20);
        let second = scheduler
            .schedule_reload("hero.reload", Priority::Critical)
            .expect("newer reload admitted");
        let second_result = scheduler
            .cpu_pool
            .block_on(second)
            .expect("newer reload commits");
        release.store(true, Ordering::Release);
        let first_result = scheduler.cpu_pool.block_on(first);

        assert_eq!(second_result.id(), resident.untyped_id());
        assert_eq!(first_result, Err(LoadFailure::Superseded));
        assert_eq!(server.with_asset(resident.id(), |asset| asset.0), Some(3));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn async_reload_publishes_a_new_dependency_closure_atomically() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(ReloadRootLoader);
        server.register_loader(ReloadDependencyLoader);
        let mut initial = MemSource::new();
        initial.insert("hero.rreload", [b'a']);
        initial.insert("a.rdep", [0_u8]);
        initial.insert("b.rdep", [0_u8]);
        server.mount("mem", Arc::new(initial), 0);
        let root = server.load::<TestAsset>("mem://hero.rreload");
        assert_eq!(server.total_assets(), 2);

        let mut revised = MemSource::new();
        revised.insert("hero.rreload", [b'b']);
        revised.insert("a.rdep", [0_u8]);
        revised.insert("b.rdep", [0_u8]);
        server.mount("mem", Arc::new(revised), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);
        let result = scheduler.cpu_pool.block_on(
            scheduler
                .schedule_reload("mem://hero.rreload", Priority::Normal)
                .unwrap(),
        );

        let result = result.expect("the revision-local dependency must be assembled");
        assert_eq!(server.total_assets(), 3);
        assert!(server.is_loaded(root.untyped_id()));
        assert!(server.is_loaded(result.id()));
        let dependency = server
            .load_untyped("b.rdep")
            .expect("new dependency is resident after the commit");
        assert!(server.is_loaded(dependency.id()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn duplicate_paths_share_one_capacity_slot_and_cleanup_is_atomic() {
        let (server, started, release) = gated_server();
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 1);

        let mut first = scheduler
            .schedule("gate://first.sched", Priority::Normal)
            .expect("first request admitted");
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        assert_eq!(scheduler.in_flight(), 1);

        let duplicate = scheduler
            .schedule("gate://first.sched", Priority::Critical)
            .expect("duplicate joins at capacity");
        assert_eq!(scheduler.in_flight(), 1);
        assert!(matches!(
            scheduler.schedule("gate://second.sched", Priority::Normal),
            Err(ScheduleError::AtCapacity { limit: 1 })
        ));

        release.store(true, Ordering::Release);
        assert!(scheduler.cpu_pool.block_on(&mut first).is_ok());
        assert!(scheduler.cpu_pool.block_on(duplicate).is_ok());
        assert_eq!(scheduler.in_flight(), 0);

        // The completion path released the exact flight's slot before a new
        // path can be admitted; no stale atomic count can reject this load.
        let next = scheduler
            .schedule("gate://second.sched", Priority::Normal)
            .expect("capacity released after completion");
        assert!(scheduler.cpu_pool.block_on(next).is_ok());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn no_loader_is_a_normal_terminal_failure() {
        let scheduler =
            LoadScheduler::new(Arc::new(AssetServer::new()), TaskPool::with_threads(0), 1);
        let ticket = scheduler
            .schedule("missing.unknown", Priority::Normal)
            .expect("request admitted");
        assert_eq!(
            scheduler.cpu_pool.block_on(ticket),
            Err(LoadFailure::NoLoader)
        );
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn direct_labeled_path_reuses_async_source_assembly() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(LabeledLoader);
        let mut source = MemSource::new();
        source.insert("scene.scene", Arc::<[u8]>::from([]));
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);

        let label = scheduler
            .cpu_pool
            .block_on(
                scheduler
                    .schedule("mem://scene.scene#mesh", Priority::Normal)
                    .expect("labeled request admitted"),
            )
            .expect("labeled request should commit");

        assert_eq!(label.id().type_id(), AssetTypeId::of::<OtherAsset>());
        assert!(server.is_loaded(label.id()));
        assert!(server.is_loaded(
            server
                .load_untyped("mem://scene.scene")
                .expect("source should be resident")
                .id()
        ));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn direct_labeled_path_resolves_already_resident_source() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(LabeledLoader);
        let mut source = MemSource::new();
        source.insert("scene.scene", Arc::<[u8]>::from([]));
        server.mount("mem", Arc::new(source), 0);
        let primary = server
            .load_untyped("mem://scene.scene")
            .expect("source should load");
        assert!(server.is_loaded(primary.id()));
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);

        let label = scheduler
            .cpu_pool
            .block_on(
                scheduler
                    .schedule("mem://scene.scene#mesh", Priority::Normal)
                    .expect("labeled request admitted"),
            )
            .expect("resident source should resolve its label");

        assert_eq!(label.id().type_id(), AssetTypeId::of::<OtherAsset>());
        assert!(server.is_loaded(label.id()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn unknown_direct_labeled_path_does_not_poison_source_revision() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(LabeledLoader);
        let mut source = MemSource::new();
        source.insert("scene.scene", Arc::<[u8]>::from([]));
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(1), 1);

        let result = scheduler.cpu_pool.block_on(
            scheduler
                .schedule("mem://scene.scene#missing", Priority::Normal)
                .expect("unknown label request admitted"),
        );
        assert!(matches!(
            result,
            Err(LoadFailure::AssemblyFailed(message)) if message.contains("was not emitted")
        ));
        let primary = server
            .load_untyped("mem://scene.scene")
            .expect("source should remain queryable");
        assert!(server.is_loaded(primary.id()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn source_panics_fail_ticket_and_release_capacity() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.mount("panic", Arc::new(PanicSource), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(0), 1);
        let ticket = scheduler
            .schedule("panic://boom.sched", Priority::Normal)
            .expect("request admitted");
        let error = scheduler
            .cpu_pool
            .block_on(ticket)
            .expect_err("source panic must fail the scheduled assembly");
        assert_eq!(
            error,
            LoadFailure::AssemblyFailed("asset source or loader panicked".into())
        );
        let handle = server
            .load_untyped("panic://boom.sched")
            .expect("failed asset remains queryable");
        assert!(server.load_state(handle.id()).is_failed());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn distinct_paths_read_outside_the_server_mutex() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.mount(
            "parallel",
            Arc::new(ParallelSource {
                active: Arc::clone(&active),
                peak: Arc::clone(&peak),
                release: Arc::clone(&release),
            }),
            0,
        );
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 2);
        let first = scheduler
            .schedule("parallel://first.sched", Priority::Normal)
            .expect("first request admitted");
        let second = scheduler
            .schedule("parallel://second.sched", Priority::Normal)
            .expect("second request admitted");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while peak.load(Ordering::Acquire) < 2 && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(peak.load(Ordering::Acquire), 2);
        release.store(true, Ordering::Release);
        assert!(scheduler.cpu_pool.block_on(first).is_ok());
        assert!(scheduler.cpu_pool.block_on(second).is_ok());
    }

    #[test]
    fn parent_waits_for_dependency_before_commit() {
        let mut source = MemSource::new();
        source.insert("parent.parent", Arc::<[u8]>::from([]));
        source.insert("child.sched", Arc::<[u8]>::from([]));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.register_loader(ParentLoader);
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(1),
            TaskPool::with_threads(1),
            4,
        );
        let ticket = scheduler
            .schedule("mem://parent.parent", Priority::Normal)
            .expect("parent request admitted");
        let handle = scheduler
            .cpu_pool
            .block_on(ticket)
            .expect("parent and child commit");
        assert!(server.is_loaded(handle.id()));
        assert!(matches!(
            server.recursive_state(handle.id()),
            RecursiveDependencyLoadState::Loaded
        ));
    }

    #[test]
    fn parent_dependency_does_not_deadlock_when_execution_capacity_is_one() {
        let mut source = MemSource::new();
        source.insert("parent.parent", Arc::<[u8]>::from([]));
        source.insert("child.sched", Arc::<[u8]>::from([]));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.register_loader(ParentLoader);
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::with_io_cpu_pools(
            Arc::clone(&server),
            TaskPool::with_threads(1),
            TaskPool::with_threads(1),
            1,
        );
        let ticket = scheduler
            .schedule("mem://parent.parent", Priority::Normal)
            .expect("parent request admitted");
        let handle = scheduler.cpu_pool.block_on(ticket);
        assert!(handle.is_ok(), "parent assembly failed: {handle:?}");
        let handle = handle.expect("parent and child commit");
        assert!(server.is_loaded(handle.id()));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn dependency_failure_does_not_publish_parent_revision() {
        let mut source = MemSource::new();
        source.insert("parent.parent", Arc::<[u8]>::from([]));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.register_loader(ParentLoader);
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 4);
        let ticket = scheduler
            .schedule("mem://parent.parent", Priority::Normal)
            .expect("parent request admitted");
        let error = scheduler
            .cpu_pool
            .block_on(ticket)
            .expect_err("dependency failure must be terminal");
        assert!(matches!(error, LoadFailure::AssemblyFailed(_)));
        let handle = server
            .load_untyped("mem://parent.parent")
            .expect("failed parent remains queryable");
        assert!(server.load_state(handle.id()).is_failed());
        assert!(server.recursive_state(handle.id()).is_failed());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn dependency_cycle_fails_terminally_without_hanging() {
        let mut source = MemSource::new();
        source.insert("a.cyc", Arc::<[u8]>::from([]));
        source.insert("b.cyc", Arc::<[u8]>::from([]));
        let server = Arc::new(AssetServer::new());
        server.register_loader(CycleLoader);
        server.mount("mem", Arc::new(source), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 4);
        let ticket = scheduler
            .schedule("mem://a.cyc", Priority::Normal)
            .expect("cycle request admitted");
        let error = scheduler
            .cpu_pool
            .block_on(ticket)
            .expect_err("dependency cycle must be terminal");
        assert!(matches!(error, LoadFailure::AssemblyFailed(_)));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn conflicting_typed_requests_do_not_share_a_flight() {
        let (server, started, release) = gated_server();
        server.register_loader(OtherLoader);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 2);
        let mut first = scheduler
            .schedule_typed_with_ancestors(
                AssetPath::parse("gate://typed.sched"),
                Priority::Normal,
                Some(AssetTypeId::of::<TestAsset>()),
                BTreeSet::new(),
            )
            .expect("first typed request admitted");
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let second = scheduler
            .schedule_typed_with_ancestors(
                AssetPath::parse("gate://typed.sched"),
                Priority::Normal,
                Some(AssetTypeId::of::<OtherAsset>()),
                BTreeSet::new(),
            )
            .expect("conflict is represented by a completed ticket");
        assert_eq!(
            scheduler.cpu_pool.block_on(second),
            Err(LoadFailure::AssemblyFailed(
                "same asset path requested with conflicting types: gate://typed.sched".into()
            ))
        );
        release.store(true, Ordering::Release);
        assert!(scheduler.cpu_pool.block_on(&mut first).is_ok());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn async_commit_rejects_a_slot_that_is_no_longer_loading() {
        let mut source = MemSource::new();
        source.insert("stale.sched", Arc::<[u8]>::from([]));
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.mount("mem", Arc::new(source), 0);
        let preparation = server
            .prepare_async_load(
                AssetPath::parse("mem://stale.sched"),
                Some(AssetTypeId::of::<TestAsset>()),
            )
            .expect("preparation succeeds");
        let plan = match preparation {
            AsyncLoadPreparation::New(plan) => plan,
            AsyncLoadPreparation::Existing(_) => panic!("fresh path must reserve a slot"),
        };
        server.fail_async_load(&plan, "forced pre-commit failure");
        let output = plan
            .loaders
            .load_for_type(&plan.path, plan.type_id, &[])
            .expect("test output decodes");
        let error = server
            .commit_async_load(*plan, output, Vec::new(), Vec::new())
            .expect_err("failed slot must not be resurrected");
        assert!(error.contains("no longer loading"));
    }

    #[test]
    fn completion_before_poll_is_not_a_notification_loss() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: FlightPriority::new(Priority::Normal),
            requested_type: None,
        });
        Flight::wake_all(flight.complete(Err(LoadFailure::NoLoader)));

        let mut ticket = LoadTicket {
            flight,
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        };
        let waker = Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))));
        let mut cx = Context::from_waker(&waker);
        assert_eq!(
            Pin::new(&mut ticket).poll(&mut cx),
            Poll::Ready(Err(LoadFailure::NoLoader))
        );
    }

    #[test]
    fn panic_boundary_turns_worker_panics_into_terminal_results() {
        let pool = TaskPool::with_threads(0);
        let result = pool.block_on(PanicBoundary::new(async {
            panic!("synthetic async worker panic")
        }));
        assert_eq!(result, Err(()));
    }

    fn test_scheduler(limit: usize) -> LoadScheduler {
        LoadScheduler::with_io_cpu_pools(
            Arc::new(AssetServer::new()),
            TaskPool::with_threads(0),
            TaskPool::with_threads(0),
            limit,
        )
    }

    fn test_flight(_scheduler: &LoadScheduler, priority: Priority) -> Arc<Flight> {
        Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: FlightPriority::new(priority),
            requested_type: None,
        })
    }

    fn test_waker() -> Waker {
        Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))))
    }

    #[test]
    fn execution_slot_handoff_cannot_be_stolen_by_a_new_waiter() {
        let scheduler = test_scheduler(1);
        let blocker_key = FlightKey::Load(AssetPath::parse("blocker.sched"));
        let blocker = test_flight(&scheduler, Priority::Normal);
        scheduler
            .state
            .lock()
            .expect("load scheduler poisoned")
            .running
            .insert(blocker_key.clone(), Arc::clone(&blocker));

        let first_key = FlightKey::Load(AssetPath::parse("first.sched"));
        let first_flight = test_flight(&scheduler, Priority::Normal);
        let first = scheduler.acquire_execution_slot(
            first_key.clone(),
            Arc::clone(&first_flight),
            Priority::Normal,
        );
        let second_key = FlightKey::Load(AssetPath::parse("second.sched"));
        let second_flight = test_flight(&scheduler, Priority::Critical);
        let second = scheduler.acquire_execution_slot(
            second_key.clone(),
            Arc::clone(&second_flight),
            Priority::Critical,
        );
        let third_key = FlightKey::Load(AssetPath::parse("third.sched"));
        let third_flight = test_flight(&scheduler, Priority::Low);
        let third =
            scheduler.acquire_execution_slot(third_key, Arc::clone(&third_flight), Priority::Low);
        let mut first = Box::pin(first);
        let mut second = Box::pin(second);
        let mut third = Box::pin(third);
        let waker = test_waker();
        let mut context = Context::from_waker(&waker);

        assert!(first.as_mut().poll(&mut context).is_pending());
        assert!(second.as_mut().poll(&mut context).is_pending());
        scheduler.release_execution_slot(&blocker_key, &blocker);

        // The critical waiter owns the released slot even before its waker is
        // polled. A newly-arrived low-priority waiter must remain pending.
        assert!(third.as_mut().poll(&mut context).is_pending());
        assert!(second.as_mut().poll(&mut context).is_ready());
        assert!(first.as_mut().poll(&mut context).is_pending());

        scheduler.release_execution_slot(&second_key, &second_flight);
        assert!(first.as_mut().poll(&mut context).is_ready());
    }

    #[test]
    fn execution_slot_handoff_is_priority_first_and_fifo_within_a_priority() {
        let scheduler = test_scheduler(1);
        let blocker_key = FlightKey::Load(AssetPath::parse("blocker.sched"));
        let blocker = test_flight(&scheduler, Priority::Normal);
        scheduler
            .state
            .lock()
            .expect("load scheduler poisoned")
            .running
            .insert(blocker_key.clone(), Arc::clone(&blocker));

        let entries = [
            ("normal.sched", Priority::Normal),
            ("critical-a.sched", Priority::Critical),
            ("critical-b.sched", Priority::Critical),
            ("low.sched", Priority::Low),
        ];
        let mut futures = Vec::new();
        let mut flights = Vec::new();
        let mut keys = Vec::new();
        for (path, priority) in entries {
            let key = FlightKey::Load(AssetPath::parse(path));
            let flight = test_flight(&scheduler, priority);
            futures.push(Box::pin(scheduler.acquire_execution_slot(
                key.clone(),
                Arc::clone(&flight),
                priority,
            )));
            keys.push(key);
            flights.push(flight);
        }
        let waker = test_waker();
        let mut context = Context::from_waker(&waker);
        for future in &mut futures {
            assert!(future.as_mut().poll(&mut context).is_pending());
        }

        scheduler.release_execution_slot(&blocker_key, &blocker);
        assert!(futures[1].as_mut().poll(&mut context).is_ready());
        scheduler.release_execution_slot(&keys[1], &flights[1]);
        assert!(futures[2].as_mut().poll(&mut context).is_ready());
        scheduler.release_execution_slot(&keys[2], &flights[2]);
        assert!(futures[0].as_mut().poll(&mut context).is_ready());
        scheduler.release_execution_slot(&keys[0], &flights[0]);
        assert!(futures[3].as_mut().poll(&mut context).is_ready());
    }

    #[test]
    fn cancellation_wakes_waiter_and_wins_over_completion() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: FlightPriority::new(Priority::Normal),
            requested_type: None,
        });
        let mut ticket = LoadTicket {
            flight: Arc::clone(&flight),
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        };
        let wakes = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountWake(Arc::clone(&wakes))));
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut ticket).poll(&mut cx).is_pending());
        assert_eq!(flight.waiter_count(), 1);

        ticket.cancel();
        assert_eq!(wakes.load(Ordering::Acquire), 1);
        assert_eq!(flight.waiter_count(), 0);
        Flight::wake_all(flight.complete(Err(LoadFailure::NoLoader)));
        assert_eq!(
            Pin::new(&mut ticket).poll(&mut cx),
            Poll::Ready(Err(LoadFailure::Cancelled))
        );
    }

    #[test]
    fn dropping_pending_ticket_unregisters_its_waker() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: FlightPriority::new(Priority::Normal),
            requested_type: None,
        });
        let mut ticket = LoadTicket {
            flight: Arc::clone(&flight),
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        };
        let waker = Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))));
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut ticket).poll(&mut cx).is_pending());
        assert_eq!(flight.waiter_count(), 1);

        drop(ticket);
        assert_eq!(flight.waiter_count(), 0);
    }

    #[test]
    fn repeated_poll_updates_one_waiter_token_instead_of_growing() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: FlightPriority::new(Priority::Normal),
            requested_type: None,
        });
        let mut ticket = LoadTicket {
            flight: Arc::clone(&flight),
            waiter_token: Arc::new(()),
            cancellation: Arc::new(Mutex::new(Cancellation {
                cancelled: false,
                waker: None,
            })),
        };
        let first_waker = Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))));
        let second_waker = Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))));
        let mut first_cx = Context::from_waker(&first_waker);
        let mut second_cx = Context::from_waker(&second_waker);
        assert!(Pin::new(&mut ticket).poll(&mut first_cx).is_pending());
        assert!(Pin::new(&mut ticket).poll(&mut second_cx).is_pending());
        assert_eq!(flight.waiter_count(), 1);

        Flight::wake_all(flight.complete(Err(LoadFailure::NoLoader)));
        assert_eq!(
            Pin::new(&mut ticket).poll(&mut second_cx),
            Poll::Ready(Err(LoadFailure::NoLoader))
        );
        assert_eq!(flight.waiter_count(), 0);
    }

    #[test]
    fn shared_path_tickets_have_independent_removable_waiters() {
        let (server, started, release) = gated_server();
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(2), 1);
        let mut first = scheduler
            .schedule("gate://shared.sched", Priority::Normal)
            .expect("request admitted");
        while !started.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let flight = Arc::clone(&first.flight);
        let mut duplicate = scheduler
            .schedule("gate://shared.sched", Priority::Critical)
            .expect("duplicate joins existing request");
        let waker = Waker::from(Arc::new(CountWake(Arc::new(AtomicUsize::new(0)))));
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut first).poll(&mut cx).is_pending());
        assert!(Pin::new(&mut duplicate).poll(&mut cx).is_pending());
        assert_eq!(flight.waiter_count(), 2);

        drop(first);
        assert_eq!(flight.waiter_count(), 1);
        release.store(true, Ordering::Release);
        assert!(scheduler.cpu_pool.block_on(duplicate).is_ok());
        assert_eq!(flight.waiter_count(), 0);
    }

    struct CountWake(Arc<AtomicUsize>);

    impl alloc::task::Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }
}
