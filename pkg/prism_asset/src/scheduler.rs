//! Bounded asynchronous request scheduling layered over the synchronous
//! [`AssetServer`](crate::AssetServer).
//!
//! This API moves the server call onto `prism_tasks::TaskPool`, coalesces
//! concurrent requests for the same path, and bounds the number of admitted
//! unique requests. It does **not** make the server's internal read/decode
//! pipeline parallel: that pipeline currently holds one server mutex for its
//! full duration. Priority orders queued task-pool work where possible; it
//! cannot preempt work already running. Ticket cancellation stops that caller
//! from waiting but does not kill shared or already-running work.

use crate::{AssetPath, AssetServer, UntypedHandle};
use alloc::collections::BTreeMap;
use alloc::sync::{Arc, Weak};
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
}

/// A request could not be admitted to the bounded in-flight set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The configured maximum number of unique requests is already in flight.
    AtCapacity {
        /// Maximum number of unique in-flight requests.
        limit: usize,
    },
}

impl core::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AtCapacity { limit } => {
                write!(f, "asset load scheduler is at capacity ({limit})")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

struct Flight {
    notification: Mutex<FlightNotification>,
    priority: PriorityGroup,
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
    in_flight: BTreeMap<AssetPath, Arc<Flight>>,
}

/// Concurrent facade for scheduling path loads through an existing server.
///
/// Clones share admission limits and in-flight deduplication. A duplicate path
/// joins its existing request even when the scheduler is at capacity.
pub struct LoadScheduler {
    server: Arc<AssetServer>,
    pool: TaskPool,
    limit: usize,
    state: Arc<Mutex<SchedulerState>>,
}

impl Clone for LoadScheduler {
    fn clone(&self) -> Self {
        Self {
            server: Arc::clone(&self.server),
            pool: self.pool.clone(),
            limit: self.limit,
            state: Arc::clone(&self.state),
        }
    }
}

impl LoadScheduler {
    /// Create a scheduler with an explicit pool and a maximum number of
    /// concurrently admitted unique paths. `max_in_flight` must be nonzero.
    #[must_use]
    pub fn new(server: Arc<AssetServer>, pool: TaskPool, max_in_flight: usize) -> Self {
        assert!(max_in_flight > 0, "max_in_flight must be nonzero");
        Self {
            server,
            pool,
            limit: max_in_flight,
            state: Arc::new(Mutex::new(SchedulerState {
                in_flight: BTreeMap::new(),
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
        let path = path.into();
        let (flight, is_new) = {
            let mut state = self.state.lock().expect("load scheduler poisoned");
            if let Some(existing) = state.in_flight.get(&path) {
                existing.priority.boost_to(priority);
                (Arc::clone(existing), false)
            } else {
                if state.in_flight.len() >= self.limit {
                    return Err(ScheduleError::AtCapacity { limit: self.limit });
                }
                let flight = Arc::new(Flight {
                    notification: Mutex::new(FlightNotification {
                        result: None,
                        waiters: Vec::new(),
                    }),
                    priority: self.pool.priority_group(priority),
                });
                state.in_flight.insert(path.clone(), Arc::clone(&flight));
                (flight, true)
            }
        };

        if is_new {
            let server = Arc::clone(&self.server);
            let state: Weak<Mutex<SchedulerState>> = Arc::downgrade(&self.state);
            let job_flight = Arc::clone(&flight);
            let job_path = path.clone();
            self.pool.spawn_in_group(&flight.priority, move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    server
                        .load_untyped(job_path.clone())
                        .ok_or(LoadFailure::NoLoader)
                }))
                .unwrap_or(Err(LoadFailure::WorkerPanicked));
                // Completion and removal share the same lock as admission. A
                // racing duplicate therefore either joins this flight before
                // it completes, or starts a fresh flight after its slot has
                // been released.
                let wakers = if let Some(state) = state.upgrade() {
                    let mut state = state.lock().expect("load scheduler poisoned");
                    let wakers = job_flight.complete(outcome);
                    if state
                        .in_flight
                        .get(&job_path)
                        .is_some_and(|f| Arc::ptr_eq(f, &job_flight))
                    {
                        state.in_flight.remove(&job_path);
                    }
                    wakers
                } else {
                    // Tickets may outlive the scheduler itself; they must
                    // still observe the worker's terminal result.
                    job_flight.complete(outcome)
                };
                Flight::wake_all(wakers);
            });
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

    /// Number of admitted unique requests not yet completed.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.state
            .lock()
            .expect("load scheduler poisoned")
            .in_flight
            .len()
    }

    /// Maximum number of admitted unique requests.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.limit
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
    use crate::{Asset, AssetLoader, AssetMeta, AssetReader, LoadContext, LoadError, ReadError};
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use prism_tasks::TaskPool;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct TestAsset;

    impl Asset for TestAsset {
        const TYPE_NAME: &'static str = "prism_asset::scheduler::tests::TestAsset";
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
        assert!(scheduler.pool.block_on(&mut first).is_ok());
        assert!(scheduler.pool.block_on(duplicate).is_ok());
        assert_eq!(scheduler.in_flight(), 0);

        // The completion path released the exact flight's slot before a new
        // path can be admitted; no stale atomic count can reject this load.
        let next = scheduler
            .schedule("gate://second.sched", Priority::Normal)
            .expect("capacity released after completion");
        assert!(scheduler.pool.block_on(next).is_ok());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn no_loader_is_a_normal_terminal_failure() {
        let scheduler =
            LoadScheduler::new(Arc::new(AssetServer::new()), TaskPool::with_threads(0), 1);
        let ticket = scheduler
            .schedule("missing.unknown", Priority::Normal)
            .expect("request admitted");
        assert_eq!(scheduler.pool.block_on(ticket), Err(LoadFailure::NoLoader));
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn source_panics_become_failed_handles_and_release_capacity() {
        let server = Arc::new(AssetServer::new());
        server.register_loader(TestLoader);
        server.mount("panic", Arc::new(PanicSource), 0);
        let scheduler = LoadScheduler::new(Arc::clone(&server), TaskPool::with_threads(0), 1);
        let ticket = scheduler
            .schedule("panic://boom.sched", Priority::Normal)
            .expect("request admitted");
        let handle = scheduler
            .pool
            .block_on(ticket)
            .expect("source panic is converted into a failed asset handle");
        assert!(server.load_state(handle.id()).is_failed());
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn completion_before_poll_is_not_a_notification_loss() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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
    fn cancellation_wakes_waiter_and_wins_over_completion() {
        let flight = Arc::new(Flight {
            notification: Mutex::new(FlightNotification {
                result: None,
                waiters: Vec::new(),
            }),
            priority: TaskPool::with_threads(0).priority_group(Priority::Normal),
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
        assert!(scheduler.pool.block_on(duplicate).is_ok());
        assert_eq!(flight.waiter_count(), 0);
    }

    struct CountWake(Arc<AtomicUsize>);

    impl alloc::task::Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }
}
