//! Safe Wake probes exercise final destruction and notification through the
//! real ticket/flight/admission paths. try_lock records regressions instead of
//! hanging a test. These are not custom RawWaker clone-callback models.

use super::*;
use std::sync::atomic::AtomicUsize;
use std::task::Wake;

#[derive(Default)]
struct Audit {
    drops: AtomicUsize,
    wakes: AtomicUsize,
    locked: AtomicUsize,
    reentered: AtomicUsize,
}

struct Probe {
    audit: Arc<Audit>,
    observe: Box<dyn Fn() -> bool + Send + Sync>,
}

impl Probe {
    fn observe(&self) {
        if (self.observe)() {
            self.audit.reentered.fetch_add(1, Ordering::SeqCst);
        } else {
            self.audit.locked.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Wake for Probe {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.audit.wakes.fetch_add(1, Ordering::SeqCst);
        self.observe();
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.audit.drops.fetch_add(1, Ordering::SeqCst);
        self.observe();
    }
}

fn probe(observe: impl Fn() -> bool + Send + Sync + 'static) -> (Waker, Arc<Audit>) {
    let audit = Arc::new(Audit::default());
    let waker = Waker::from(Arc::new(Probe {
        audit: Arc::clone(&audit),
        observe: Box::new(observe),
    }));
    (waker, audit)
}

fn assert_audit(audit: &Audit, wakes: usize) {
    assert_eq!(audit.locked.load(Ordering::SeqCst), 0);
    assert_eq!(audit.drops.load(Ordering::SeqCst), 1);
    assert_eq!(audit.wakes.load(Ordering::SeqCst), wakes);
    assert_eq!(audit.reentered.load(Ordering::SeqCst), wakes + 1);
}

fn flight() -> Arc<Flight> {
    Arc::new(Flight {
        notification: Mutex::new(FlightNotification {
            result: None,
            waiters: Vec::new(),
        }),
        priority: FlightPriority::new(Priority::Normal),
        requested_type: None,
        cancellation: RangeAssemblyCancellation::new(),
        lifecycle: Mutex::new(FlightLifecycle::default()),
    })
}

fn flight_probe(flight: &Arc<Flight>) -> (Waker, Arc<Audit>) {
    let weak = Arc::downgrade(flight);
    probe(move || {
        let flight = weak.upgrade().unwrap();
        if flight.notification.try_lock().is_err() || flight.lifecycle.try_lock().is_err() {
            return false;
        }
        let _ = flight.waiter_count();
        true
    })
}

#[test]
fn flight_unregister_retires_waker_after_unlocking() {
    let flight = flight();
    let token = Arc::new(());
    let (waker, audit) = flight_probe(&flight);
    assert!(flight
        .poll(&token, &Context::from_waker(&waker))
        .is_pending());
    drop(waker);
    flight.unregister(&token);
    assert_eq!(flight.waiter_count(), 0);
    assert_audit(&audit, 0);
}

#[test]
fn flight_repoll_and_completion_retire_and_wake_after_unlocking() {
    let flight = flight();
    let token = Arc::new(());
    let (old, old_audit) = flight_probe(&flight);
    assert!(flight.poll(&token, &Context::from_waker(&old)).is_pending());
    drop(old);
    let (new, new_audit) = flight_probe(&flight);
    assert!(flight.poll(&token, &Context::from_waker(&new)).is_pending());
    assert_audit(&old_audit, 0);
    drop(new);
    Flight::wake_all(flight.complete(Err(LoadFailure::NoLoader)));
    assert_audit(&new_audit, 1);
}

fn read_flight() -> Arc<ReloadReadFlight> {
    Arc::new(ReloadReadFlight {
        notification: Mutex::new(ReloadReadNotification {
            result: None,
            waiters: Vec::new(),
        }),
        priority: FlightPriority::new(Priority::Normal),
    })
}

fn read_probe(flight: &Arc<ReloadReadFlight>) -> (Waker, Arc<Audit>) {
    let weak = Arc::downgrade(flight);
    probe(move || {
        let flight = weak.upgrade().unwrap();
        if flight.notification.try_lock().is_err() {
            return false;
        }
        flight.unregister(&Arc::new(()));
        true
    })
}

#[test]
fn reload_read_unregister_retires_waker_after_unlocking() {
    let flight = read_flight();
    let token = Arc::new(());
    let (waker, audit) = read_probe(&flight);
    assert!(flight
        .poll(&token, &Context::from_waker(&waker))
        .is_pending());
    drop(waker);
    flight.unregister(&token);
    assert_audit(&audit, 0);
}

#[test]
fn reload_read_repoll_and_completion_notify_after_unlocking() {
    let flight = read_flight();
    let token = Arc::new(());
    let (old, old_audit) = read_probe(&flight);
    assert!(flight.poll(&token, &Context::from_waker(&old)).is_pending());
    drop(old);
    let (new, new_audit) = read_probe(&flight);
    assert!(flight.poll(&token, &Context::from_waker(&new)).is_pending());
    assert_audit(&old_audit, 0);
    drop(new);
    ReloadReadFlight::wake_all(flight.complete(Ok(Arc::from([1_u8, 2]))));
    assert_audit(&new_audit, 1);
}

#[test]
fn ticket_repoll_handles_cancellation_reentering_from_retired_flight_waker() {
    let flight = flight();
    let mut ticket = LoadTicket::new(Arc::clone(&flight));
    let weak_flight = Arc::downgrade(&flight);
    let weak_cancellation = Arc::downgrade(&ticket.cancellation);
    let token = Arc::clone(&ticket.waiter_token);
    let (waker, audit) = probe(move || {
        let flight = weak_flight.upgrade().unwrap();
        let cancellation = weak_cancellation.upgrade().unwrap();
        if cancellation.try_lock().is_err()
            || flight.notification.try_lock().is_err()
            || flight.lifecycle.try_lock().is_err()
        {
            return false;
        }
        let wake = cancellation.lock().unwrap().cancel();
        flight.unregister(&token);
        if let Some(wake) = wake {
            wake.wake();
        }
        true
    });
    assert!(Pin::new(&mut ticket)
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(waker);
    // The old task waker has references in both the ticket and the flight.
    // Replacing the latter runs its destructor, which cancels after a new
    // notification was registered. The final check must remove that waiter.
    let next = Waker::noop();
    assert_eq!(
        Pin::new(&mut ticket).poll(&mut Context::from_waker(next)),
        Poll::Ready(Err(LoadFailure::Cancelled))
    );
    assert_eq!(flight.waiter_count(), 0);
    assert_eq!(flight.lifecycle.lock().unwrap().consumers, 0);
    assert!(flight.is_cancelled());
    assert_audit(&audit, 0);
}

fn scheduler() -> LoadScheduler {
    LoadScheduler::with_io_cpu_pools(
        Arc::new(AssetServer::new()),
        TaskPool::with_threads(0),
        TaskPool::with_threads(0),
        1,
    )
}

fn scheduler_probe(scheduler: &LoadScheduler, flight: &Arc<Flight>) -> (Waker, Arc<Audit>) {
    let scheduler = scheduler.clone();
    let weak_flight = Arc::downgrade(flight);
    probe(move || {
        let flight = weak_flight.upgrade().unwrap();
        if scheduler.state.try_lock().is_err() || flight.lifecycle.try_lock().is_err() {
            return false;
        }
        let _ = scheduler.in_flight();
        true
    })
}

#[test]
fn admission_repoll_retires_worker_and_waiter_wakers_after_unlocking() {
    let scheduler = scheduler();
    let blocker = flight();
    let blocker_key = FlightKey::Load(AssetPath::from("blocker.notify"));
    scheduler
        .state
        .lock()
        .unwrap()
        .running
        .insert(blocker_key.clone(), Arc::clone(&blocker));
    let flight = flight();
    let mut slot = Box::pin(scheduler.acquire_execution_slot(
        FlightKey::Load(AssetPath::from("waiter.notify")),
        Arc::clone(&flight),
        Priority::Normal,
    ));
    let (old, old_audit) = scheduler_probe(&scheduler, &flight);
    assert!(slot
        .as_mut()
        .poll(&mut Context::from_waker(&old))
        .is_pending());
    drop(old);
    let (new, new_audit) = scheduler_probe(&scheduler, &flight);
    assert!(slot
        .as_mut()
        .poll(&mut Context::from_waker(&new))
        .is_pending());
    assert_audit(&old_audit, 0);
    drop(new);
    scheduler.release_execution_slot(&blocker_key, &blocker);
    assert_eq!(new_audit.locked.load(Ordering::SeqCst), 0);
    assert_eq!(new_audit.wakes.load(Ordering::SeqCst), 1);
    assert!(slot
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_ready());
    assert_audit(&new_audit, 1);
}

#[test]
fn dropping_admission_future_retires_waiter_after_unlocking() {
    let scheduler = scheduler();
    let blocker = flight();
    scheduler
        .state
        .lock()
        .unwrap()
        .running
        .insert(FlightKey::Load(AssetPath::from("blocker.notify")), blocker);
    let flight = flight();
    let mut slot = Box::pin(scheduler.acquire_execution_slot(
        FlightKey::Load(AssetPath::from("waiter.notify")),
        Arc::clone(&flight),
        Priority::Normal,
    ));
    let (waker, audit) = scheduler_probe(&scheduler, &flight);
    assert!(slot
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(waker);
    drop(slot);
    // The lifecycle still owns a worker notification. Finishing must retire
    // the final reference outside both locks, just like normal completion.
    Flight::wake_all(flight.complete(Err(LoadFailure::Cancelled)));
    assert!(scheduler.state.lock().unwrap().admission_waiters.is_empty());
    assert_audit(&audit, 1);
}
