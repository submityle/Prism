use super::tests::manifest;
use super::*;
use std::sync::atomic::AtomicUsize;
use std::sync::{mpsc, Weak};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

struct CallbackWake {
    on_wake: Box<dyn Fn() + Send + Sync>,
    on_drop: Box<dyn Fn() + Send + Sync>,
}

impl Wake for CallbackWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        (self.on_wake)();
    }
}

impl Drop for CallbackWake {
    fn drop(&mut self) {
        (self.on_drop)();
    }
}

fn callbacks(
    on_wake: impl Fn() + Send + Sync + 'static,
    on_drop: impl Fn() + Send + Sync + 'static,
) -> Waker {
    Waker::from(Arc::new(CallbackWake {
        on_wake: Box::new(on_wake),
        on_drop: Box::new(on_drop),
    }))
}

fn counter(count: &Arc<AtomicUsize>) -> Waker {
    let count = Arc::clone(count);
    callbacks(
        move || {
            count.fetch_add(1, Ordering::Relaxed);
        },
        || {},
    )
}

struct PollProvider {
    wake: bool,
    ready: bool,
}

struct PollOperation {
    wake: bool,
    ready: bool,
    cancel_waker: Option<Waker>,
}

impl ReadProvider for PollProvider {
    type Operation = PollOperation;

    fn start(&self, _: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        Ok(PollOperation {
            wake: self.wake,
            ready: self.ready,
            cancel_waker: None,
        })
    }
}

impl ChunkedRead for PollOperation {
    fn poll_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        self.cancel_waker = Some(cx.waker().clone());
        if self.wake {
            cx.waker().wake_by_ref();
        }
        if self.ready {
            Poll::Ready(Ok(None))
        } else {
            Poll::Pending
        }
    }

    fn cancel(&mut self) {
        if let Some(waker) = self.cancel_waker.take() {
            waker.wake();
        }
    }
}

fn adapter(wake: bool, ready: bool) -> Arc<IoSchedulerAdapter<PollProvider>> {
    Arc::new(IoSchedulerAdapter::new(PollProvider { wake, ready }, 2, 2).expect("adapter"))
}

// Check before actually reentering so a regression fails instead of hanging the
// test process. No callback runs assertions that panic isolation might swallow.
fn can_reenter<P: ReadProvider>(adapter: &IoSchedulerAdapter<P>) -> bool {
    let adapter_unlocked = adapter.state.try_lock().is_ok();
    let wake_unlocked = adapter.wake.state.try_lock().is_ok();
    if adapter_unlocked && wake_unlocked {
        let _ = adapter.len();
        true
    } else {
        false
    }
}

#[test]
fn changed_public_wakers_keep_only_the_latest_registration() {
    let adapter = adapter(false, false);
    adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("submit");
    let wakes = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    for index in 0..256 {
        let wakes = Arc::clone(&wakes);
        let drops_for_callback = Arc::clone(&drops);
        let waker = callbacks(
            move || {
                wakes.fetch_add(1, Ordering::Relaxed);
            },
            move || {
                drops_for_callback.fetch_add(1, Ordering::Relaxed);
            },
        );
        assert!(adapter.poll(&mut Context::from_waker(&waker)).is_pending());
        drop(waker);
        let state = adapter.wake.lock();
        assert!(state.waiters.public.is_some());
        assert!(state.waiters.blocking.is_empty());
        drop(state);
        assert_eq!(drops.load(Ordering::Relaxed), index);
    }
    adapter.wake.wake_by_ref();
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 256);
    assert!(adapter.wake.lock().waiters.public.is_none());
}

#[test]
fn replacing_a_waker_runs_its_reentrant_final_drop_outside_both_locks() {
    let adapter = adapter(false, false);
    adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("submit");
    let reentered = Arc::new(AtomicUsize::new(0));
    let violations = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&adapter);
    let drop_reentered = Arc::clone(&reentered);
    let drop_violations = Arc::clone(&violations);
    let waker = callbacks(
        || {},
        move || {
            let adapter = weak.upgrade().expect("live adapter");
            if can_reenter(&adapter) {
                drop_reentered.fetch_add(1, Ordering::Relaxed);
            } else {
                drop_violations.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    assert!(adapter.poll(&mut Context::from_waker(&waker)).is_pending());
    drop(waker);
    assert!(adapter
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    assert_eq!(reentered.load(Ordering::Relaxed), 1);
    assert_eq!(violations.load(Ordering::Relaxed), 0);
}

#[test]
fn synchronous_provider_wake_is_deferred_until_adapter_state_unlocks() {
    for ready in [false, true] {
        let adapter = adapter(true, ready);
        adapter
            .submit(CellReadRequest::new(manifest(0), 1))
            .expect("submit");
        let calls = Arc::new(AtomicUsize::new(0));
        let violations = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&adapter);
        let on_wake_calls = Arc::clone(&calls);
        let on_wake_violations = Arc::clone(&violations);
        let waker = callbacks(
            move || {
                let adapter = weak.upgrade().expect("live adapter");
                on_wake_calls.fetch_add(1, Ordering::Relaxed);
                if !can_reenter(&adapter) {
                    on_wake_violations.fetch_add(1, Ordering::Relaxed);
                }
            },
            || {},
        );
        let result = adapter.poll(&mut Context::from_waker(&waker));
        assert_eq!(result.is_ready(), ready);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(violations.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn synchronous_cancel_wake_is_deferred_until_adapter_state_unlocks() {
    let adapter = adapter(false, false);
    let id = adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("submit");
    let calls = Arc::new(AtomicUsize::new(0));
    let violations = Arc::new(AtomicUsize::new(0));
    let weak = Arc::downgrade(&adapter);
    let callback_calls = Arc::clone(&calls);
    let callback_violations = Arc::clone(&violations);
    let waker = callbacks(
        move || {
            let adapter = weak.upgrade().expect("live adapter");
            callback_calls.fetch_add(1, Ordering::Relaxed);
            if !can_reenter(&adapter) {
                callback_violations.fetch_add(1, Ordering::Relaxed);
            }
        },
        || {},
    );
    assert!(adapter.poll(&mut Context::from_waker(&waker)).is_pending());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(adapter.cancel(id));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(violations.load(Ordering::Relaxed), 0);
    assert!(
        matches!(adapter.poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(IoAdapterEvent::Failed {
            request_id, error: IoError::Cancelled { request_id: cancelled }, ..
        })) if request_id == id && cancelled == id)
    );
    assert!(adapter.is_empty());
}

#[test]
fn ready_poll_does_not_unregister_a_newer_reentrant_poll() {
    let adapter = adapter(true, true);
    adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("submit");
    let newer_wakes = Arc::new(AtomicUsize::new(0));
    let newer_waker = counter(&newer_wakes);
    let reentered = Arc::new(AtomicUsize::new(0));
    let reentered_callback = Arc::clone(&reentered);
    let weak = Arc::downgrade(&adapter);
    let waker = callbacks(
        move || {
            let adapter = weak.upgrade().expect("live adapter");
            if can_reenter(&adapter)
                && adapter
                    .poll(&mut Context::from_waker(&newer_waker))
                    .is_pending()
            {
                reentered_callback.fetch_add(1, Ordering::Relaxed);
            }
        },
        || {},
    );
    assert!(matches!(
        adapter.poll(&mut Context::from_waker(&waker)),
        Poll::Ready(Ok(IoAdapterEvent::Completed { .. }))
    ));
    assert_eq!(reentered.load(Ordering::Relaxed), 1);
    assert!(adapter.wake.lock().waiters.public.is_some());
    adapter.wake.wake_by_ref();
    assert_eq!(newer_wakes.load(Ordering::Relaxed), 1);
}

#[test]
fn stale_registration_cannot_remove_a_new_poll_with_the_same_waker() {
    let wake = SchedulerWake::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let waker = counter(&calls);
    let old = wake.register(None, &waker);
    let new = wake.register(None, &waker);
    wake.unregister(&old);
    assert!(wake.lock().waiters.public.is_some());
    wake.notify();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    wake.unregister(&new);
}

#[test]
fn blocking_registrations_are_keyed_by_target_even_with_the_same_waker() {
    let adapter = adapter(false, false);
    let first = adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("first");
    let second = adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("second");
    let calls = Arc::new(AtomicUsize::new(0));
    let waker = counter(&calls);
    let first_registration = adapter.wake.register(Some(first), &waker);
    let second_registration = adapter.wake.register(Some(second), &waker);
    let public_registration = adapter.wake.register(None, &waker);
    adapter.wake.unregister(&first_registration);
    adapter.wake.unregister(&public_registration);
    assert_eq!(adapter.wake.lock().waiters.blocking.len(), 1);
    adapter.wake.wake_by_ref();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    adapter.wake.unregister(&second_registration);
}

#[test]
fn concurrent_worker_notification_waits_for_busy_guard_without_losing_wake() {
    let adapter = adapter(false, false);
    let calls = Arc::new(AtomicUsize::new(0));
    let violations = Arc::new(AtomicUsize::new(0));
    for _ in 0..64 {
        let weak = Arc::downgrade(&adapter);
        let callback_calls = Arc::clone(&calls);
        let callback_violations = Arc::clone(&violations);
        let waker = callbacks(
            move || {
                let adapter = weak.upgrade().expect("live adapter");
                callback_calls.fetch_add(1, Ordering::Relaxed);
                if !can_reenter(&adapter) {
                    callback_violations.fetch_add(1, Ordering::Relaxed);
                }
            },
            || {},
        );
        let registration = adapter.wake.register(None, &waker);
        let state = adapter.lock_state().expect("state");
        let before = calls.load(Ordering::Relaxed);
        let wake = Arc::clone(&adapter.wake);
        thread::spawn(move || wake.wake_by_ref())
            .join()
            .expect("worker notify");
        assert_eq!(calls.load(Ordering::Relaxed), before);
        drop(state);
        assert_eq!(calls.load(Ordering::Relaxed), before + 1);
        adapter.wake.unregister(&registration);
    }
    assert_eq!(violations.load(Ordering::Relaxed), 0);
}

#[test]
fn worker_notification_during_dispatch_reaches_new_registration() {
    let wake = Arc::new(SchedulerWake::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let newer = counter(&calls);
    let weak = Arc::downgrade(&wake);
    let waker = callbacks(
        move || {
            let wake = weak.upgrade().expect("live scheduler wake");
            drop(wake.register(None, &newer));
            let worker_wake = Arc::clone(&wake);
            thread::spawn(move || worker_wake.wake_by_ref())
                .join()
                .expect("worker notify");
        },
        || {},
    );
    drop(wake.register(None, &waker));
    wake.wake_by_ref();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let state = wake.lock();
    assert!(!state.dispatching);
    assert!(!state.notified);
    assert!(state.waiters.public.is_none());
}

#[test]
fn wake_and_final_drop_panics_do_not_skip_other_registered_targets() {
    let adapter = adapter(false, false);
    let first = adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("first");
    let second = adapter
        .submit(CellReadRequest::new(manifest(0), 1))
        .expect("second");
    let calls = Arc::new(AtomicUsize::new(0));
    let panicked = Arc::new(AtomicUsize::new(0));
    let drop_panicked = Arc::clone(&panicked);
    let bad = callbacks(
        || panic!("expected wake panic"),
        move || {
            drop_panicked.fetch_add(1, Ordering::Relaxed);
            panic!("expected final drop panic");
        },
    );
    drop(adapter.wake.register(None, &bad));
    drop(bad);
    let good = counter(&calls);
    drop(adapter.wake.register(Some(first), &good));
    drop(adapter.wake.register(Some(second), &good));
    adapter.wake.wake_by_ref();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(panicked.load(Ordering::Relaxed), 1);
    assert!(!adapter.wake.lock().dispatching);
    assert!(
        adapter.wake.state.lock().is_ok(),
        "callbacks must not poison metadata"
    );
}

struct Gate {
    released: AtomicBool,
    entered: mpsc::Sender<IoRequestId>,
    waker: Mutex<Option<Waker>>,
}

struct GateProvider(Arc<Gate>);

struct GateOperation {
    gate: Arc<Gate>,
    id: IoRequestId,
    offset: u64,
    announced: bool,
    delivered: bool,
}

impl ReadProvider for GateProvider {
    type Operation = GateOperation;

    fn start(&self, request: &ReadRequest, id: IoRequestId) -> Result<Self::Operation, IoError> {
        Ok(GateOperation {
            gate: Arc::clone(&self.0),
            id,
            offset: request.offset(),
            announced: false,
            delivered: false,
        })
    }
}

impl ChunkedRead for GateOperation {
    fn poll_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if !self.gate.released.load(Ordering::Acquire) {
            let replacement = cx.waker().clone();
            let previous = self
                .gate
                .waker
                .lock()
                .expect("gate lock")
                .replace(replacement);
            drop(previous);
            if !self.announced {
                self.announced = true;
                self.gate.entered.send(self.id).expect("announce pending");
            }
            if !self.gate.released.load(Ordering::Acquire) {
                return Poll::Pending;
            }
        }
        if self.delivered {
            return Poll::Ready(Ok(None));
        }
        self.delivered = true;
        Poll::Ready(Ok(Some(ReadChunk::new(
            self.offset,
            vec![self.offset as u8],
        ))))
    }

    fn cancel(&mut self) {}
}

fn count_reentrant_drop(
    weak: Weak<IoSchedulerAdapter<GateProvider>>,
    count: Arc<AtomicUsize>,
) -> impl Fn() + Send + Sync {
    move || {
        if let Some(adapter) = weak.upgrade()
            && can_reenter(&adapter)
        {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
fn changing_public_consumer_and_panicking_wake_preserve_multiple_blocking_readers() {
    let (entered, pending) = mpsc::channel();
    let gate = Arc::new(Gate {
        released: AtomicBool::new(false),
        entered,
        waker: Mutex::new(None),
    });
    let adapter =
        Arc::new(IoSchedulerAdapter::new(GateProvider(Arc::clone(&gate)), 2, 2).expect("adapter"));
    let (send, results) = mpsc::channel();
    let threads: Vec<_> = [17, 29]
        .into_iter()
        .map(|offset| {
            let adapter = Arc::clone(&adapter);
            let send = send.clone();
            thread::spawn(move || {
                let result = adapter.read_to_completion(
                    &CellReadRequest::new(manifest(1), 1).with_byte_offset(offset),
                    CancellationToken::new(false),
                );
                send.send((offset, result)).expect("reader result");
            })
        })
        .collect();
    let first = pending.recv_timeout(TIMEOUT).expect("first reader pending");
    let second = pending
        .recv_timeout(TIMEOUT)
        .expect("second reader pending");
    assert_ne!(first, second);
    let drops = Arc::new(AtomicUsize::new(0));
    for _ in 0..64 {
        let waker = callbacks(
            || {},
            count_reentrant_drop(Arc::downgrade(&adapter), Arc::clone(&drops)),
        );
        assert!(adapter.poll(&mut Context::from_waker(&waker)).is_pending());
    }
    let waker = callbacks(|| panic!("expected public consumer panic"), || {});
    assert!(adapter.poll(&mut Context::from_waker(&waker)).is_pending());
    assert_eq!(drops.load(Ordering::Relaxed), 64);
    gate.released.store(true, Ordering::Release);
    let provider_waker = gate
        .waker
        .lock()
        .expect("gate lock")
        .take()
        .expect("provider waker");
    provider_waker.wake();
    for _ in 0..2 {
        let (offset, result) = results
            .recv_timeout(TIMEOUT)
            .expect("both readers must wake");
        assert_eq!(result, Ok(vec![offset as u8]));
    }
    for reader in threads {
        reader.join().expect("reader thread");
    }
    assert!(adapter.is_empty());
    assert!(adapter.wake.lock().waiters.blocking.is_empty());
}
