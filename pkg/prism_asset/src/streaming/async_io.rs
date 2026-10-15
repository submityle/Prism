//! Lock-out asynchronous range-read execution for runtime streaming.
//!
//! The executor owns the source table and the controller through `Arc`s. A
//! request is admitted under the controller lock, but the source read itself
//! runs on the caller-provided [`prism_tasks::TaskPool`] without that lock.
//! Completion takes the same request gate as cancellation before committing a
//! deterministic controller transition.

use super::{
    StreamError, StreamFailure, StreamIoError, StreamRequest, StreamRequestId, StreamState,
    StreamingController, SubmitError,
};
use crate::AssetSources;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use prism_tasks::{Task, TaskPool};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

/// A task returned by [`AsyncStreamIoExecutor::execute`].
pub type AsyncStreamTask = Task<Result<Vec<u8>, StreamIoError>>;

#[derive(Debug, Default)]
struct RequestGate {
    phase: Mutex<RequestPhase>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RequestPhase {
    #[default]
    Ready,
    Reading,
    Cancelled,
}

impl RequestGate {
    /// Claims the read start. This is the cancellation/read linearization
    /// point: cancellation winning this lock prevents the source call.
    fn begin_read(&self) -> bool {
        let mut phase = self.phase.lock().unwrap();
        match *phase {
            RequestPhase::Ready => {
                *phase = RequestPhase::Reading;
                true
            }
            RequestPhase::Reading | RequestPhase::Cancelled => false,
        }
    }
}

/// Shared, lock-out asynchronous executor for runtime range reads.
///
/// `submit` only records a request. `execute` admits it and schedules one
/// source read on the supplied [`TaskPool`]. The task never holds the
/// controller mutex while calling [`AssetSources::read_range`]. Dropping the
/// returned task detaches it; the read and its state commit continue in the
/// pool.
#[derive(Clone)]
pub struct AsyncStreamIoExecutor {
    sources: Arc<AssetSources>,
    controller: Arc<Mutex<StreamingController>>,
    pool: TaskPool,
    gates: Arc<Mutex<BTreeMap<StreamRequestId, Arc<RequestGate>>>>,
}

impl AsyncStreamIoExecutor {
    /// Creates an executor over shared sources, controller, and task pool.
    #[must_use]
    pub fn new(
        sources: Arc<AssetSources>,
        controller: StreamingController,
        pool: TaskPool,
    ) -> Self {
        Self {
            sources,
            controller: Arc::new(Mutex::new(controller)),
            pool,
            gates: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Submits a request without admitting or reading it.
    pub fn submit(&self, request: StreamRequest) -> Result<StreamRequestId, SubmitError> {
        // Publish the controller record and its gate as one operation. All
        // paths that need a gate take the gate-table lock before the
        // controller lock, so no caller can observe a request without its
        // cancellation state.
        let mut gates = self.gates.lock().unwrap();
        let mut controller = self.controller.lock().unwrap();
        let id = controller.submit(request)?;
        gates.insert(id, Arc::new(RequestGate::default()));
        Ok(id)
    }

    /// Admits one pending request and schedules its source range read.
    ///
    /// Admission and the `Pending -> Loading` transition are performed before
    /// the task is queued. The task then rechecks the request gate immediately
    /// before calling the source, so a cancellation that wins first performs
    /// no source I/O.
    pub fn execute(&self, id: StreamRequestId) -> Result<AsyncStreamTask, StreamIoError> {
        let Ok(gate) = self.gate(id) else {
            let state = self
                .controller
                .lock()
                .unwrap()
                .state(id)
                .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
            return Err(match state {
                StreamState::Cancelled => StreamIoError::Cancelled { id },
                state => StreamIoError::AlreadyExecuted { id, state },
            });
        };
        let gate_phase = gate.phase.lock().unwrap();
        let mut controller = self.controller.lock().unwrap();

        let state = controller
            .state(id)
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
        if state != StreamState::Pending {
            return Err(match state {
                StreamState::Cancelled => StreamIoError::Cancelled { id },
                state => StreamIoError::AlreadyExecuted { id, state },
            });
        }
        if *gate_phase == RequestPhase::Cancelled {
            return Err(StreamIoError::Cancelled { id });
        }
        let request = controller
            .request(id)
            .cloned()
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
        if let Err(error) = controller.begin_load(id) {
            return match error {
                StreamError::BudgetExceeded { .. } | StreamError::BudgetOverflow => {
                    Err(StreamIoError::AdmissionRejected { id, error })
                }
                error => Err(StreamIoError::Controller(error)),
            };
        }
        drop(controller);
        drop(gate_phase);

        let sources = Arc::clone(&self.sources);
        let controller = Arc::clone(&self.controller);
        let gates = Arc::clone(&self.gates);
        let gate_for_task = Arc::clone(&gate);
        Ok(self.pool.spawn_async(async move {
            let outcome = if !gate_for_task.begin_read() {
                // Cancellation already performed the terminal controller
                // transition and released its reservation.
                Err(StreamIoError::Cancelled { id })
            } else {
                // This is the only blocking operation in the task. No
                // controller mutex or mutable controller borrow is held
                // across this call.
                let result = catch_unwind(AssertUnwindSafe(|| {
                    sources.read_range(&request.path, request.offset, request.len)
                }))
                .unwrap_or_else(|_| Err(crate::ReadError::Io("source read panicked".into())));

                // The gate is the linearization point shared with cancel:
                // once this lock is held, cancellation cannot invalidate the
                // result before the controller transition is committed.
                let gate_phase = gate_for_task.phase.lock().unwrap();
                let mut controller = controller.lock().unwrap();
                let outcome = if *gate_phase == RequestPhase::Cancelled {
                    Err(StreamIoError::Cancelled { id })
                } else {
                    match result {
                        Ok(bytes) if u64::try_from(bytes.len()).ok() == Some(request.len) => {
                            controller
                                .complete_load(id)
                                .map(|()| bytes)
                                .map_err(StreamIoError::Controller)
                        }
                        Ok(bytes) => {
                            let error = crate::ReadError::Io(format!(
                                "short read: requested {} bytes, received {}",
                                request.len,
                                bytes.len()
                            ));
                            controller
                                .fail(id, StreamFailure::Read)
                                .map(|()| Err(StreamIoError::Read { id, error }))
                                .unwrap_or_else(|error| Err(StreamIoError::Controller(error)))
                        }
                        Err(error) => controller
                            .fail(id, StreamFailure::Read)
                            .map(|()| Err(StreamIoError::Read { id, error }))
                            .unwrap_or_else(|error| Err(StreamIoError::Controller(error))),
                    }
                };
                drop(controller);
                drop(gate_phase);
                outcome
            };
            remove_gate(&gates, id, &gate_for_task);
            outcome
        }))
    }

    /// Executes the controller's deterministic highest-priority pending
    /// request, if one is available.
    pub fn execute_next(&self) -> Result<Option<AsyncStreamTask>, StreamIoError> {
        let id = self
            .controller
            .lock()
            .unwrap()
            .next_pending()
            .map(|(id, _)| id);
        id.map(|id| self.execute(id)).transpose()
    }

    /// Cancels a pending or loading request. A pending cancellation prevents
    /// the task from calling the source. A loading cancellation releases the
    /// controller's reservation immediately; if the source is already in its
    /// synchronous read, its result is discarded and the request stays
    /// `Cancelled`.
    pub fn cancel(&self, id: StreamRequestId) -> Result<(), StreamIoError> {
        let state = self
            .controller
            .lock()
            .unwrap()
            .state(id)
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
        if !matches!(state, StreamState::Pending | StreamState::Loading) {
            return Err(match state {
                StreamState::Cancelled => StreamIoError::Cancelled { id },
                state => StreamIoError::AlreadyExecuted { id, state },
            });
        }
        let Ok(gate) = self.gate(id) else {
            let state = self
                .controller
                .lock()
                .unwrap()
                .state(id)
                .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
            return Err(match state {
                StreamState::Cancelled => StreamIoError::Cancelled { id },
                state => StreamIoError::AlreadyExecuted { id, state },
            });
        };
        let mut gate_phase = gate.phase.lock().unwrap();
        let mut controller = self.controller.lock().unwrap();
        let state = controller
            .state(id)
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))?;
        match state {
            StreamState::Pending | StreamState::Loading => {
                *gate_phase = RequestPhase::Cancelled;
                let result = controller.cancel(id).map_err(StreamIoError::Controller);
                drop(controller);
                drop(gate_phase);
                if result.is_ok() {
                    remove_gate(&self.gates, id, &gate);
                }
                result
            }
            StreamState::Cancelled => Err(StreamIoError::Cancelled { id }),
            state => Err(StreamIoError::AlreadyExecuted { id, state }),
        }
    }

    /// Returns the current state of a request.
    #[must_use]
    pub fn state(&self, id: StreamRequestId) -> Option<StreamState> {
        self.controller.lock().unwrap().state(id)
    }

    /// Returns the current aggregate residency accounting.
    #[must_use]
    pub fn used(&self) -> super::Residency {
        self.controller.lock().unwrap().used()
    }

    /// Returns the task pool used by this executor.
    #[must_use]
    pub fn pool(&self) -> TaskPool {
        self.pool.clone()
    }

    fn gate(&self, id: StreamRequestId) -> Result<Arc<RequestGate>, StreamIoError> {
        self.gates
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or(StreamIoError::Controller(StreamError::UnknownRequest(id)))
    }
}

fn remove_gate(
    gates: &Mutex<BTreeMap<StreamRequestId, Arc<RequestGate>>>,
    id: StreamRequestId,
    gate: &Arc<RequestGate>,
) {
    let mut gates = gates.lock().unwrap();
    let is_same_gate = gates
        .get(&id)
        .map(|current| Arc::ptr_eq(current, gate))
        .unwrap_or(false);
    if is_same_gate {
        gates.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetMeta, AssetReader, MemSource, ReadError, Residency, ResidencyLimits};
    use prism_tasks::Priority;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex as StdMutex};
    fn request(path: &str, len: u64) -> StreamRequest {
        StreamRequest {
            path: path.into(),
            offset: 0,
            len,
            residency: Residency::new(len, len, len),
            priority: Priority::Normal,
            importance: 0,
        }
    }

    fn sources_with(path: &str, bytes: &[u8]) -> Arc<AssetSources> {
        let mut source = MemSource::new();
        source.insert(path, Arc::<[u8]>::from(bytes));
        let mut sources = AssetSources::new();
        sources.mount("mem", Arc::new(source), 0);
        Arc::new(sources)
    }

    struct CountingSource {
        reads: Arc<AtomicUsize>,
        bytes: &'static [u8],
    }

    impl AssetReader for CountingSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(self.bytes.to_vec())
        }

        fn read_range(&self, _path: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let end = offset.checked_add(len).ok_or(ReadError::OutOfRange {
                offset,
                len,
                size: self.bytes.len() as u64,
            })?;
            let start = usize::try_from(offset).map_err(|_| ReadError::OutOfRange {
                offset,
                len,
                size: self.bytes.len() as u64,
            })?;
            let end = usize::try_from(end).map_err(|_| ReadError::OutOfRange {
                offset,
                len,
                size: self.bytes.len() as u64,
            })?;
            self.bytes
                .get(start..end)
                .map(ToOwned::to_owned)
                .ok_or(ReadError::OutOfRange {
                    offset,
                    len,
                    size: self.bytes.len() as u64,
                })
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: self.bytes.len() as u64,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    struct BlockingSource {
        entered: Arc<(StdMutex<bool>, Condvar)>,
        release: Arc<(StdMutex<bool>, Condvar)>,
        reads: Arc<AtomicUsize>,
    }

    struct ParallelSource {
        entered: Arc<AtomicUsize>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        release: Arc<(StdMutex<bool>, Condvar)>,
    }

    struct ShortReadSource {
        bytes: &'static [u8],
    }

    impl AssetReader for ShortReadSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            Ok(self.bytes.to_vec())
        }

        fn read_range(&self, _path: &str, _offset: u64, _len: u64) -> Result<Vec<u8>, ReadError> {
            Ok(self.bytes.to_vec())
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: self.bytes.len() as u64,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    impl AssetReader for BlockingSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let (lock, wake) = &*self.entered;
            *lock.lock().unwrap() = true;
            wake.notify_all();
            let (lock, wake) = &*self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            Ok(b"blocked".to_vec())
        }

        fn read_range(&self, path: &str, _offset: u64, _len: u64) -> Result<Vec<u8>, ReadError> {
            self.read(path)
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 7,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    impl AssetReader for ParallelSource {
        fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
            Ok(b"data".to_vec())
        }

        fn read_range(&self, _path: &str, _offset: u64, _len: u64) -> Result<Vec<u8>, ReadError> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            self.entered.fetch_add(1, Ordering::SeqCst);

            let (lock, wake) = &*self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(b"data".to_vec())
        }

        fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
            Ok(AssetMeta {
                size: 4,
                modified_ms: None,
                is_dir: false,
            })
        }
    }

    #[test]
    fn cancellation_before_execution_performs_no_io_and_releases_no_budget() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abcdef",
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("counting://hero.bin", 3)).unwrap();

        executor.cancel(id).unwrap();

        assert_eq!(executor.state(id), Some(StreamState::Cancelled));
        assert_eq!(executor.used(), Residency::default());
        assert!(matches!(
            executor.execute(id),
            Err(StreamIoError::Cancelled { id: actual }) if actual == id
        ));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
        assert_eq!(executor.gates.lock().unwrap().len(), 0);
    }

    #[test]
    fn cancellation_before_read_starts_performs_no_io() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abcdef",
            }),
            0,
        );
        let pool = TaskPool::with_threads(1);
        let blocker_release = Arc::new((StdMutex::new(false), Condvar::new()));
        let blocker_started = Arc::new((StdMutex::new(false), Condvar::new()));
        let blocker_release_for_job = Arc::clone(&blocker_release);
        let blocker_started_for_job = Arc::clone(&blocker_started);
        let blocker = pool.spawn_catch(move || {
            let (lock, wake) = &*blocker_started_for_job;
            *lock.lock().unwrap() = true;
            wake.notify_all();
            let (lock, wake) = &*blocker_release_for_job;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
        });
        let (lock, wake) = &*blocker_started;
        let mut started = lock.lock().unwrap();
        while !*started {
            started = wake.wait(started).unwrap();
        }

        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            pool.clone(),
        );
        let id = executor.submit(request("counting://hero.bin", 3)).unwrap();
        let task = executor.execute(id).unwrap();
        executor.cancel(id).unwrap();
        let (lock, wake) = &*blocker_release;
        *lock.lock().unwrap() = true;
        wake.notify_all();
        let _: () = blocker.join();
        assert!(matches!(
            pool.block_on(task),
            Err(StreamIoError::Cancelled { id: actual }) if actual == id
        ));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn successful_async_range_read_commits_resident_after_io() {
        let executor = AsyncStreamIoExecutor::new(
            sources_with("hero.bin", b"abcdef"),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor
            .submit(StreamRequest {
                path: "mem://hero.bin".into(),
                offset: 2,
                len: 3,
                residency: Residency::new(3, 3, 3),
                priority: Priority::Normal,
                importance: 0,
            })
            .unwrap();
        let task = executor.execute(id).unwrap();
        assert_eq!(executor.pool().block_on(task).unwrap(), b"cde");
        assert_eq!(executor.state(id), Some(StreamState::Resident));
        assert_eq!(executor.used(), Residency::new(3, 3, 3));
        assert_eq!(executor.gates.lock().unwrap().len(), 0);
    }

    #[test]
    fn short_async_range_read_fails_and_releases_all_residency() {
        let mut sources = AssetSources::new();
        sources.mount("short", Arc::new(ShortReadSource { bytes: b"abc" }), 0);
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("short://hero.bin", 5)).unwrap();
        let task = executor.execute(id).unwrap();

        let error = executor.pool().block_on(task).unwrap_err();
        assert!(matches!(
            error,
            StreamIoError::Read {
                id: actual,
                error: ReadError::Io(message),
            } if actual == id && message == "short read: requested 5 bytes, received 3"
        ));
        assert_eq!(executor.state(id), Some(StreamState::Failed));
        assert_eq!(executor.used(), Residency::default());
    }

    #[test]
    fn cancellation_during_read_discards_result_and_releases_budget() {
        let entered = Arc::new((StdMutex::new(false), Condvar::new()));
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "blocking",
            Arc::new(BlockingSource {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
                reads: Arc::clone(&reads),
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("blocking://hero.bin", 7)).unwrap();
        let task = executor.execute(id).unwrap();

        let (entered_lock, entered_wake) = &*entered;
        let mut did_enter = entered_lock.lock().unwrap();
        while !*did_enter {
            did_enter = entered_wake.wait(did_enter).unwrap();
        }
        // The source is blocked, yet the controller remains available to
        // unrelated callers. This proves the blocking read is outside the
        // controller mutex critical section.
        assert!(executor.controller.try_lock().is_ok());
        executor.cancel(id).unwrap();
        let (release_lock, release_wake) = &*release;
        *release_lock.lock().unwrap() = true;
        release_wake.notify_all();

        assert_eq!(
            executor.pool().block_on(task),
            Err(StreamIoError::Cancelled { id })
        );
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(executor.state(id), Some(StreamState::Cancelled));
        assert_eq!(executor.used(), Residency::default());
        assert_eq!(executor.gates.lock().unwrap().len(), 0);
    }

    #[test]
    fn dropping_task_does_not_skip_residency_commit_or_gate_cleanup() {
        let entered = Arc::new((StdMutex::new(false), Condvar::new()));
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let mut sources = AssetSources::new();
        sources.mount(
            "blocking",
            Arc::new(BlockingSource {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
                reads: Arc::new(AtomicUsize::new(0)),
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("blocking://hero.bin", 7)).unwrap();
        let task = executor.execute(id).unwrap();

        let (entered_lock, entered_wake) = &*entered;
        let mut did_enter = entered_lock.lock().unwrap();
        while !*did_enter {
            did_enter = entered_wake.wait(did_enter).unwrap();
        }
        drop(task);

        let (release_lock, release_wake) = &*release;
        *release_lock.lock().unwrap() = true;
        release_wake.notify_all();

        let mut committed = false;
        for _ in 0..10_000 {
            if executor.state(id) == Some(StreamState::Resident)
                && executor.gates.lock().unwrap().is_empty()
            {
                committed = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(committed, "detached task did not commit its terminal state");
        assert_eq!(executor.used(), Residency::new(7, 7, 7));
    }

    #[test]
    fn independent_requests_read_in_parallel_without_controller_lock() {
        let entered = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let release = Arc::new((StdMutex::new(false), Condvar::new()));
        let mut sources = AssetSources::new();
        sources.mount(
            "parallel",
            Arc::new(ParallelSource {
                entered: Arc::clone(&entered),
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
                release: Arc::clone(&release),
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let first = executor.submit(request("parallel://first.bin", 4)).unwrap();
        let second = executor
            .submit(request("parallel://second.bin", 4))
            .unwrap();
        let first_task = executor.execute(first).unwrap();
        let second_task = executor.execute(second).unwrap();

        for _ in 0..10_000 {
            if entered.load(Ordering::SeqCst) == 2 {
                break;
            }
            std::thread::yield_now();
        }
        let both_entered = entered.load(Ordering::SeqCst) == 2;
        assert!(executor.controller.try_lock().is_ok());

        let (release_lock, release_wake) = &*release;
        *release_lock.lock().unwrap() = true;
        release_wake.notify_all();

        assert_eq!(executor.pool().block_on(first_task).unwrap(), b"data");
        assert_eq!(executor.pool().block_on(second_task).unwrap(), b"data");
        assert!(
            both_entered,
            "both requests must reach the source concurrently"
        );
        assert!(max_active.load(Ordering::SeqCst) >= 2);
        assert_eq!(executor.state(first), Some(StreamState::Resident));
        assert_eq!(executor.state(second), Some(StreamState::Resident));
        assert_eq!(executor.used(), Residency::new(8, 8, 8));
    }

    #[test]
    fn concurrent_cancel_and_execute_have_one_deterministic_commit() {
        let executor = AsyncStreamIoExecutor::new(
            sources_with("hero.bin", b"abcdef"),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("mem://hero.bin", 3)).unwrap();
        let cancel_executor = executor.clone();
        let cancel = std::thread::spawn(move || cancel_executor.cancel(id));
        let task = executor.execute(id);
        let cancel_result = cancel.join().unwrap();
        match task {
            Ok(task) => {
                let result = executor.pool().block_on(task);
                match result {
                    Ok(_) => {}
                    Err(StreamIoError::Cancelled { id: actual }) => assert_eq!(actual, id),
                    Err(error) => panic!("unexpected task error: {error:?}"),
                }
            }
            Err(StreamIoError::Cancelled { id: actual }) => assert_eq!(actual, id),
            Err(error) => panic!("unexpected execute error: {error:?}"),
        }
        assert!(
            cancel_result.is_ok()
                || matches!(cancel_result, Err(StreamIoError::AlreadyExecuted { .. }))
        );
    }

    #[test]
    fn source_failure_releases_all_residency() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abc",
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(16, 16, 16)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("counting://hero.bin", 9)).unwrap();
        let task = executor.execute(id).unwrap();
        assert!(matches!(
            executor.pool().block_on(task),
            Err(StreamIoError::Read { id: actual, .. }) if actual == id
        ));
        assert_eq!(executor.state(id), Some(StreamState::Failed));
        assert_eq!(executor.used(), Residency::default());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert_eq!(executor.gates.lock().unwrap().len(), 0);
    }

    #[test]
    fn budget_rejection_does_not_schedule_a_read() {
        let reads = Arc::new(AtomicUsize::new(0));
        let mut sources = AssetSources::new();
        sources.mount(
            "counting",
            Arc::new(CountingSource {
                reads: Arc::clone(&reads),
                bytes: b"abcdef",
            }),
            0,
        );
        let executor = AsyncStreamIoExecutor::new(
            Arc::new(sources),
            StreamingController::new(ResidencyLimits::new(2, 2, 2)),
            TaskPool::with_threads(2),
        );
        let id = executor.submit(request("counting://hero.bin", 3)).unwrap();
        assert!(matches!(
            executor.execute(id),
            Err(StreamIoError::AdmissionRejected { id: actual, .. }) if actual == id
        ));
        assert_eq!(reads.load(Ordering::SeqCst), 0);
        assert_eq!(executor.state(id), Some(StreamState::Pending));
    }
}
