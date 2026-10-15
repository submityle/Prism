//! Bounded asynchronous request scheduling and dependency assembly layered over
//! the synchronous [`AssetServer`](crate::AssetServer).
//!
//! This API coalesces concurrent requests for the same path, bounds admission,
//! snapshots source/loader state under the server mutex, and performs full
//! reads on an I/O pool plus synchronous CPU decoding on a CPU pool before
//! committing one dependency-aware revision. Awaiting a dependency yields the
//! CPU worker, so a fixed worker count cannot deadlock solely because parent
//! assemblies are waiting for child flights. Reload flights support a narrow
//! resident-root revision replacement with per-path epoch supersede; labeled
//! revisions and automatic loading of new reload dependencies remain
//! unsupported. Range reads and preemptive cancellation are also outside this
//! slice. Priority orders queued CPU-pool work where possible; ticket
//! cancellation stops that caller from waiting but does not kill shared or
//! already-running work.

use crate::server::{AsyncLoadPreparation, AsyncReloadCommitError, AsyncReloadPlan};
use crate::{AssetPath, AssetServer, AssetTypeId, UntypedHandle};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use prism_tasks::{Priority, PriorityGroup, TaskPool};
use std::sync::Mutex;

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
    priority: PriorityGroup,
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
    queued: VecDeque<QueuedFlight>,
    admission_waiters: Vec<(Arc<()>, Priority, Waker)>,
    next_reload_serial: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FlightKey {
    Load(AssetPath),
    Reload { path: AssetPath, serial: u64 },
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
                queued: VecDeque::new(),
                admission_waiters: Vec::new(),
                next_reload_serial: 0,
            })),
        }
    }

    /// Schedule an untyped load. Requests for an identical [`AssetPath`] share
    /// one server invocation and receive clones of its resulting handle.
    pub fn schedule(
        &self,
        path: impl Into<AssetPath>,
        priority: Priority,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_with_ancestors(path.into(), priority, BTreeSet::new())
    }

    /// Schedule an asynchronous reload of an already resident asset.
    ///
    /// Reload flights are independent from ordinary load flights for the same
    /// path. Decoding and dependency resolution happen against staging state;
    /// only the latest reload epoch may replace the resident revision.
    pub fn schedule_reload(
        &self,
        path: impl Into<AssetPath>,
        priority: Priority,
    ) -> Result<LoadTicket, ScheduleError> {
        self.schedule_reload_with_ancestors(path.into(), priority, BTreeSet::new())
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
        let requested_type = requested_type.or_else(|| self.server.inferred_asset_type(&path));
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
        let requested_type = requested_type.or_else(|| self.server.inferred_asset_type(&path));
        let (flight, start_now) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            if let Some(existing) = state.in_flight.get(&key) {
                if let (Some(existing_type), Some(requested_type)) =
                    (existing.requested_type, requested_type)
                {
                    if existing_type != requested_type {
                        let conflict = Arc::new(Flight {
                            notification: Mutex::new(FlightNotification {
                                result: None,
                                waiters: Vec::new(),
                            }),
                            priority: self.cpu_pool.priority_group(priority),
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
                    priority: self.cpu_pool.priority_group(priority),
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
        let task = pool.spawn_async(async move {
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
            let admission_wakers = self.take_admission_wakers_locked(&mut state);
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
        while state.running.len() < self.limit {
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
            let admission_wakers = self.take_admission_wakers_locked(&mut state);
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
        ExecutionSlotFuture {
            scheduler: self.clone(),
            key,
            flight,
            token: Arc::new(()),
            priority,
        }
    }

    fn take_admission_wakers_locked(&self, state: &mut SchedulerState) -> Vec<Waker> {
        let available = self.limit.saturating_sub(state.running.len());
        let mut wakers = Vec::with_capacity(available.min(state.admission_waiters.len()));
        for _ in 0..available {
            let Some(index) = state
                .admission_waiters
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, priority, _))| *priority)
                .map(|(index, _)| index)
            else {
                break;
            };
            let (_, _, waker) = state.admission_waiters.remove(index);
            wakers.push(waker);
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
    if !ancestors.insert(path.clone()) {
        return Err(LoadFailure::AssemblyFailed(format!(
            "dependency cycle at {path}"
        )));
    }
    let preparation = match server.prepare_async_load(path.clone(), requested_type) {
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
                "asset is already loading outside this scheduler: {path}"
            )));
        }
        AsyncLoadPreparation::New(plan) => plan,
    };
    let bytes = if scheduler.io_pool.is_single_threaded() {
        read_bytes_inline(&plan)
    } else {
        let read_sources = plan.sources.clone();
        let read_path = plan.path.clone();
        scheduler
            .io_pool
            .spawn_async(async move {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    read_sources.read(&read_path)
                })) {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(error)) => Err(error.to_string()),
                    Err(_) => Err("asset source or loader panicked".to_string()),
                }
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
    if !output.labeled.is_empty() {
        let error = "labeled assets are not supported by async assembly".to_string();
        server.fail_async_load(&plan, &error);
        return Err(LoadFailure::AssemblyFailed(error));
    }
    for dependency in &output.primary.dependencies {
        if ancestors.contains(&dependency.path) {
            let error = format!("dependency cycle at {}", dependency.path);
            server.fail_async_load(&plan, &error);
            return Err(LoadFailure::AssemblyFailed(error));
        }
    }
    if !output.primary.dependencies.is_empty() {
        scheduler.release_execution_slot(&FlightKey::Load(path.clone()), &flight);
    }
    let has_dependencies = !output.primary.dependencies.is_empty();
    let mut dependencies = Vec::with_capacity(output.primary.dependencies.len());
    for dependency in &output.primary.dependencies {
        let ticket = match scheduler.schedule_typed_with_ancestors(
            dependency.path.clone(),
            priority,
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
    if has_dependencies {
        scheduler
            .acquire_execution_slot(FlightKey::Load(path.clone()), Arc::clone(&flight), priority)
            .await;
    }
    server
        .commit_async_load(plan, output, dependencies)
        .map_err(LoadFailure::AssemblyFailed)
}

async fn execute_async_reload(
    scheduler: LoadScheduler,
    server: Arc<AssetServer>,
    path: AssetPath,
    requested_type: Option<AssetTypeId>,
) -> ScheduleOutcome {
    let plan = match server.prepare_async_reload(path.clone(), requested_type) {
        Ok(plan) => plan,
        Err(error) if error.starts_with("no loader for ") => return Err(LoadFailure::NoLoader),
        Err(error) => return Err(LoadFailure::AssemblyFailed(error)),
    };
    let bytes = if scheduler.io_pool.is_single_threaded() {
        read_reload_bytes_inline(&plan)
    } else {
        let read_sources = plan.sources.clone();
        let read_path = plan.path.clone();
        scheduler
            .io_pool
            .spawn_async(async move {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    read_sources.read(&read_path)
                })) {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(error)) => Err(error.to_string()),
                    Err(_) => Err("asset source or loader panicked".to_string()),
                }
            })
            .await
    };
    let bytes = bytes.map_err(LoadFailure::AssemblyFailed)?;
    let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.loaders
            .load_for_type(&plan.path, plan.type_id, &bytes)
            .map_err(|error| error.to_string())
    })) {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => return Err(LoadFailure::AssemblyFailed(error)),
        Err(_) => return Err(LoadFailure::WorkerPanicked),
    };
    let mut dependencies = Vec::with_capacity(output.primary.dependencies.len());
    for dependency in &output.primary.dependencies {
        // A reload has no revision-local catalog or graph journal yet. Do not
        // route dependencies through ordinary load: that would publish new
        // live slots even when this reload later fails or is superseded.
        let dependency = server
            .resident_async_dependency(&dependency.path, dependency.expected_type)
            .map_err(LoadFailure::AssemblyFailed)?;
        dependencies.push(dependency);
    }
    let mut labeled_dependencies = Vec::with_capacity(output.labeled.len());
    for labeled in &output.labeled {
        let mut resolved = Vec::with_capacity(labeled.dependencies.len());
        for dependency in &labeled.dependencies {
            let dependency = server
                .resident_async_dependency(&dependency.path, dependency.expected_type)
                .map_err(LoadFailure::AssemblyFailed)?;
            resolved.push(dependency);
        }
        labeled_dependencies.push(resolved);
    }
    match server.commit_async_reload(plan, output, dependencies, labeled_dependencies) {
        Ok(handle) => Ok(handle),
        Err(AsyncReloadCommitError::Superseded) => Err(LoadFailure::Superseded),
        Err(AsyncReloadCommitError::Rejected(error)) => Err(LoadFailure::AssemblyFailed(error)),
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
    priority: Priority,
}

impl Future for ExecutionSlotFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this
            .scheduler
            .state
            .lock()
            .expect("load scheduler poisoned");
        if state.running.len() < this.scheduler.limit && state.queued.is_empty() {
            state
                .running
                .insert(this.key.clone(), Arc::clone(&this.flight));
            state
                .admission_waiters
                .retain(|(token, _, _)| !Arc::ptr_eq(token, &this.token));
            return Poll::Ready(());
        }
        if let Some((_, priority, waker)) = state
            .admission_waiters
            .iter_mut()
            .find(|(token, _, _)| Arc::ptr_eq(token, &this.token))
        {
            *priority = this.priority;
            if !waker.will_wake(cx.waker()) {
                *waker = cx.waker().clone();
            }
        } else {
            state.admission_waiters.push((
                Arc::clone(&this.token),
                this.priority,
                cx.waker().clone(),
            ));
        }
        Poll::Pending
    }
}

impl Drop for ExecutionSlotFuture {
    fn drop(&mut self) {
        let mut state = self
            .scheduler
            .state
            .lock()
            .expect("load scheduler poisoned");
        state
            .admission_waiters
            .retain(|(token, _, _)| !Arc::ptr_eq(token, &self.token));
    }
}

fn read_bytes_inline(plan: &crate::server::AsyncLoadPlan) -> Result<Vec<u8>, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.sources.read(&plan.path)
    })) {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err("asset source or loader panicked".to_string()),
    }
}

fn read_reload_bytes_inline(plan: &AsyncReloadPlan) -> Result<Vec<u8>, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.sources.read(&plan.path)
    })) {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err("asset source or loader panicked".to_string()),
    }
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
mod tests {
    use super::*;
    use crate::{
        Asset, AssetLoader, AssetMeta, AssetPath, AssetReader, LoadContext, LoadError, MemSource,
        ReadError, RecursiveDependencyLoadState,
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
    fn async_reload_does_not_publish_unresident_dependencies() {
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

        assert!(
            matches!(result, Err(LoadFailure::AssemblyFailed(message)) if message.contains("not resident"))
        );
        assert_eq!(server.total_assets(), 2);
        assert!(server.is_loaded(root.untyped_id()));
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
            .commit_async_load(plan, output, Vec::new())
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
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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

    #[test]
    fn cancellation_wakes_waiter_and_wins_over_completion() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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
