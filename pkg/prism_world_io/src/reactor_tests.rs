//! Live native tests. Unsupported platforms assert Unsupported, never substitute
//! synchronous reads. The host/main validation run is responsible for running them.

use super::*;
use crate::reactor::{test_support, TestEvent};
use prism_tasks::{IoReactor, NamedThreads, NamedThreadsConfig};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    task::{Wake, Waker},
    thread,
    time::{Duration, Instant},
};

const DEADLINE: Duration = Duration::from_secs(10);

// Declaration order joins the existing workers before the final reactor drop.
struct Services {
    workers: NamedThreads,
    reactor: Arc<IoReactor>,
}

impl Services {
    fn new() -> Option<Self> {
        if !IoReactor::supported() {
            assert!(matches!(
                IoReactor::new(),
                Err(prism_tasks::AioIoError::Unsupported)
            ));
            return None;
        }
        let reactor = Arc::new(IoReactor::new().expect("real native reactor"));
        Some(Self {
            workers: NamedThreads::with_config(NamedThreadsConfig {
                io_threads: 2,
                async_compute_threads: 1,
            }),
            reactor,
        })
    }

    fn provider(&self, path: &PathBuf, config: ReactorFileConfig) -> ReactorFileProvider<'_> {
        ReactorFileProvider::new(path, &self.reactor, &self.workers, config).unwrap()
    }
}

struct Source(PathBuf);

impl Source {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-native-world-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

fn config(capacity: usize, max_chunk_size: usize) -> ReactorFileConfig {
    ReactorFileConfig {
        capacity,
        max_chunk_size,
        ..Default::default()
    }
}

struct Signal(mpsc::Sender<()>);

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn signal() -> (Waker, mpsc::Receiver<()>) {
    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(Signal(tx))), rx)
}

fn poll_until<T>(mut poll: impl FnMut(&mut Context<'_>) -> Poll<T>) -> T {
    let (waker, rx) = signal();
    let mut context = Context::from_waker(&waker);
    let end = Instant::now() + DEADLINE;
    loop {
        match poll(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => rx
                .recv_timeout(end.saturating_duration_since(Instant::now()))
                .unwrap(),
        }
    }
}

fn next(operation: &mut impl ChunkedRead) -> Result<Option<ReadChunk>, IoError> {
    poll_until(|context| operation.poll_chunk(context))
}

fn until(mut predicate: impl FnMut() -> bool) {
    let end = Instant::now() + DEADLINE;
    while !predicate() {
        assert!(
            Instant::now() < end,
            "IO worker did not reach expected state"
        );
        thread::yield_now();
    }
}

struct Gate {
    release: Arc<(Mutex<bool>, Condvar)>,
    entered: mpsc::Receiver<()>,
}

impl Gate {
    fn new() -> (Self, Arc<dyn Fn() + Send + Sync>) {
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_release = Arc::clone(&release);
        let (tx, entered) = mpsc::channel();
        let wait = Arc::new(move || {
            let _ = tx.send(());
            let (mutex, changed) = &*worker_release;
            let guard = mutex.lock().unwrap();
            drop(changed.wait_while(guard, |released| !*released).unwrap());
        });
        (Self { release, entered }, wait)
    }

    fn entered(&self) {
        self.entered.recv_timeout(DEADLINE).unwrap();
    }

    fn open(&self) {
        let (mutex, changed) = &*self.release;
        *mutex.lock().unwrap() = true;
        changed.notify_all();
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
    }
}

type CallbackSnapshot = (usize, u64, usize);

enum CallbackRequest {
    Snapshot(mpsc::Sender<CallbackSnapshot>),
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallbackEvent {
    Wake,
    Drop,
}

type CallbackObservation = (CallbackEvent, Option<CallbackSnapshot>);

struct CallbackProbe(mpsc::Sender<CallbackRequest>);

impl Drop for CallbackProbe {
    fn drop(&mut self) {
        let _ = self.0.send(CallbackRequest::Stop);
    }
}

struct ProbedWake {
    requests: mpsc::Sender<CallbackRequest>,
    observations: mpsc::Sender<CallbackObservation>,
    panic_on_wake: bool,
}

impl ProbedWake {
    fn observe(&self, event: CallbackEvent) {
        let (reply, result) = mpsc::channel();
        let snapshot = if self.requests.send(CallbackRequest::Snapshot(reply)).is_ok() {
            result.recv_timeout(DEADLINE).ok()
        } else {
            None
        };
        // Assert on the test thread: provider panic isolation must not hide a
        // failed reentrancy check, including during the final Arc destruction.
        let _ = self.observations.send((event, snapshot));
    }
}

impl Wake for ProbedWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.observe(CallbackEvent::Wake);
        assert!(!self.panic_on_wake, "queued host wake panic");
    }
}

impl Drop for ProbedWake {
    fn drop(&mut self) {
        self.observe(CallbackEvent::Drop);
    }
}

impl CallbackProbe {
    fn waker(&self, panic_on_wake: bool) -> (Waker, mpsc::Receiver<CallbackObservation>) {
        let (observations, observed) = mpsc::channel();
        (
            Waker::from(Arc::new(ProbedWake {
                requests: self.0.clone(),
                observations,
                panic_on_wake,
            })),
            observed,
        )
    }
}

fn with_callback_probe(provider: &ReactorFileProvider<'_>, test: impl FnOnce(&CallbackProbe)) {
    thread::scope(|scope| {
        let (requests, pending) = mpsc::channel();
        let queued = provider.clone().queued();
        scope.spawn(move || {
            while let Ok(CallbackRequest::Snapshot(reply)) = pending.recv() {
                let snapshot = (
                    queued.in_flight(),
                    queued.reserved_bytes(),
                    queued.pending_admissions(),
                );
                let _ = reply.send(snapshot);
            }
        });
        // A 'static waker owns only channels, never the borrowed services. The
        // callback waits for a reentrant query to finish before returning. A
        // lock-held regression times out, releases the lock and fails below;
        // it cannot leave the scoped query thread blocked at teardown.
        let probe = CallbackProbe(requests);
        test(&probe);
    });
}

fn expect_callbacks(
    observed: &mpsc::Receiver<CallbackObservation>,
    events: &[CallbackEvent],
    snapshot: CallbackSnapshot,
) {
    for &event in events {
        assert_eq!(
            observed.recv_timeout(DEADLINE).unwrap(),
            (event, Some(snapshot)),
            "host callback must allow provider queries before it returns"
        );
    }
    assert!(observed.try_recv().is_err(), "unexpected extra callback");
}

#[test]
fn reactor_validates_limits_without_services_or_files() {
    for limits in [
        config(0, 1),
        config(1, 0),
        config(1, usize::MAX),
        config(usize::MAX, 2),
    ] {
        assert_eq!(
            test_support::validate(limits).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
    assert!(test_support::validate(ReactorFileConfig {
        max_request_bytes: 0,
        max_total_bytes: 0,
        ..config(1, 1)
    })
    .is_ok());
}

#[test]
fn reactor_real_file_multichunk_offset_and_scheduler_checksum() {
    let Some(services) = Services::new() else {
        return;
    };
    let bytes: Vec<_> = (0..1031).map(|n| (n % 251) as u8).collect();
    let source = Source::new(&bytes);
    let provider = services.provider(&source.0, config(2, 73));
    let monitor = provider.clone();
    let mut scheduler = IoScheduler::new(provider, 2, 2).unwrap();
    let expected = &bytes[11..1018];
    let id = scheduler
        .submit(
            ReadRequest::new(11, expected.len() as u64)
                .with_chunk_size(101)
                .unwrap()
                .with_expected_checksum(checksum(expected)),
        )
        .unwrap();
    let mut result = Vec::new();
    let mut chunks = 0;
    loop {
        match poll_until(|cx| scheduler.poll(cx)).unwrap() {
            IoEvent::Chunk { request_id, chunk } => {
                assert_eq!(request_id, id);
                assert_eq!(chunk.offset(), 11 + result.len() as u64);
                assert!(chunk.bytes().len() <= 73);
                result.extend_from_slice(chunk.bytes());
                chunks += 1;
            }
            IoEvent::Completed(summary) => {
                assert_eq!(summary.request_id(), id);
                assert_eq!(summary.checksum(), checksum(expected));
                assert_eq!(summary.length(), expected.len() as u64);
                break;
            }
        }
    }
    assert!(chunks > 2);
    assert_eq!(result, expected);
    assert_eq!(monitor.in_flight(), 0);
    assert_eq!(monitor.reserved_bytes(), 0);
}

#[test]
fn reactor_chunk_ack_holds_capacity_and_avoids_read_ahead() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefghij");
    let provider = services.provider(&source.0, config(1, 3));
    let submissions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&submissions);
    test_support::set_hook(
        &provider,
        Arc::new(move |event| {
            if event == TestEvent::Submitted {
                observed.fetch_add(1, Ordering::SeqCst);
            }
        }),
    );
    let request = ReadRequest::new(0, 10);
    let mut operation = provider.start(&request, RequestId(1)).unwrap();
    until(|| test_support::ready(&provider));
    assert_eq!(submissions.load(Ordering::SeqCst), 1);
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"abc");
    assert_eq!(
        submissions.load(Ordering::SeqCst),
        1,
        "delivering a chunk is not an ACK"
    );
    assert_eq!(provider.in_flight(), 1);
    assert_eq!(provider.reserved_bytes(), 10);
    assert!(matches!(
        provider.start(&request, RequestId(2)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"def");
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"ghi");
    // Final native completion/close does not release quota before consumption.
    assert!(operation
        .poll_chunk(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    until(|| test_support::ready(&provider));
    assert_eq!(provider.reserved_bytes(), 10);
    assert_eq!(submissions.load(Ordering::SeqCst), 4);
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"j");
    assert_eq!(provider.in_flight(), 0);
    assert!(next(&mut operation).unwrap().is_none());
}

#[test]
fn reactor_cancel_after_submission_keeps_fd_and_quota_through_reap_and_close() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(&vec![37; 65536]);
    let provider = services.provider(&source.0, config(1, 65536));
    let (gate, wait) = Gate::new();
    let (tx, events) = mpsc::channel();
    test_support::set_hook(
        &provider,
        Arc::new(move |event| {
            tx.send((event, thread::current().name().unwrap_or("").to_string()))
                .unwrap();
            if event == TestEvent::Submitted {
                wait();
            }
        }),
    );
    let request = ReadRequest::new(0, 65536);
    let mut operation = provider.start(&request, RequestId(1)).unwrap();
    gate.entered();
    let (waker, woke) = signal();
    assert!(operation
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    operation.cancel();
    woke.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        next(&mut operation),
        Err(IoError::Cancelled {
            request_id: RequestId(1)
        })
    );
    // Drop must return while the worker still owns a submitted, unobserved future.
    drop(operation);
    assert_eq!(provider.in_flight(), 1);
    assert_eq!(provider.reserved_bytes(), 65536);
    assert!(matches!(
        provider.start(&request, RequestId(2)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    gate.open();
    until(|| provider.in_flight() == 0);
    assert_eq!(provider.reserved_bytes(), 0);
    let observed: Vec<_> = events.try_iter().collect();
    assert_eq!(
        observed.iter().map(|(event, _)| *event).collect::<Vec<_>>(),
        vec![
            TestEvent::BeforeOpen,
            TestEvent::Submitted,
            TestEvent::Reaped { file_alive: true },
            TestEvent::Closed,
        ]
    );
    assert!(observed
        .iter()
        .all(|(_, name)| name.starts_with("prism-io-")));
    let mut replacement = provider
        .start(&ReadRequest::new(0, 1), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), &[37]);
}

#[test]
fn reactor_cancellation_before_open_performs_no_native_read() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let provider = services.provider(&source.0, config(1, 2));
    let (gate, wait) = Gate::new();
    let submitted = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&submitted);
    test_support::set_hook(
        &provider,
        Arc::new(move |event| match event {
            TestEvent::BeforeOpen => wait(),
            TestEvent::Submitted => {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }),
    );
    let mut operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    operation.cancel();
    drop(operation);
    assert_eq!(provider.in_flight(), 1);
    gate.open();
    until(|| provider.in_flight() == 0);
    assert_eq!(submitted.load(Ordering::SeqCst), 0);
}

#[test]
fn reactor_drop_parked_or_ready_work_dispatches_close_to_existing_io_workers() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    for consume_chunk in [false, true] {
        let provider = services.provider(&source.0, config(1, 3));
        let (gate, wait) = Gate::new();
        test_support::set_hook(
            &provider,
            Arc::new(move |event| {
                if event == TestEvent::Closed {
                    assert!(thread::current().name().unwrap().starts_with("prism-io-"));
                    wait();
                }
            }),
        );
        let mut operation = provider
            .start(&ReadRequest::new(0, 8), RequestId(1))
            .unwrap();
        until(|| test_support::ready(&provider));
        if consume_chunk {
            assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"abc");
        }
        drop(operation);
        gate.entered();
        assert_eq!(provider.in_flight(), 1, "cleanup still owns its admission");
        assert_eq!(provider.reserved_bytes(), 8);
        gate.open();
        until(|| provider.in_flight() == 0);
    }
}

#[test]
fn reactor_drop_does_not_join_or_destroy_last_service_owner_on_compute() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let provider = services.provider(&source.0, config(1, 2));
    let monitor = provider.clone();
    let (gate, wait) = Gate::new();
    test_support::set_hook(
        &provider,
        Arc::new(move |event| {
            if event == TestEvent::Submitted {
                wait();
            }
        }),
    );
    let operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        scope.spawn(move || {
            drop(provider);
            drop(operation);
            tx.send(()).unwrap();
        });
        let returned = rx.recv_timeout(Duration::from_secs(1));
        // Release before asserting, so even a regression cannot hang scoped join.
        gate.open();
        assert!(returned.is_ok(), "compute Drop waited for native IO");
    });
    until(|| monitor.in_flight() == 0);
    drop(monitor);
    assert_eq!(Arc::strong_count(&services.reactor), 1);
}

#[test]
fn reactor_errors_are_terminal_and_release_quota_only_when_consumed() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abc");
    let missing = source.0.with_extension("missing");
    for path in [missing, source.0.parent().unwrap().to_path_buf()] {
        let provider = services.provider(&path, config(1, 2));
        let mut operation = provider
            .start(&ReadRequest::new(0, 2), RequestId(1))
            .unwrap();
        until(|| test_support::ready(&provider));
        assert_eq!(provider.reserved_bytes(), 2);
        assert!(matches!(
            next(&mut operation),
            Err(IoError::Provider { .. })
        ));
        assert_eq!(provider.in_flight(), 0);
        assert_eq!(provider.reserved_bytes(), 0);
        assert!(next(&mut operation).unwrap().is_none());
    }
    let provider = services.provider(&source.0, config(1, 10));
    let mut operation = provider
        .start(&ReadRequest::new(2, 5), RequestId(1))
        .unwrap();
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"c");
    assert_eq!(
        next(&mut operation),
        Err(IoError::ShortRead {
            expected: 5,
            actual: 1
        })
    );
    assert_eq!(provider.reserved_bytes(), 0);
}

#[test]
fn reactor_request_and_total_byte_quotas_are_shared_across_clones() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    let provider = services.provider(
        &source.0,
        ReactorFileConfig {
            max_request_bytes: 5,
            max_total_bytes: 8,
            ..config(4, 8)
        },
    );
    let clone = provider.clone();
    assert!(matches!(
        provider.start(&ReadRequest::new(0, 6), RequestId(1)),
        Err(IoError::Provider { .. })
    ));
    let mut a = provider
        .start(&ReadRequest::new(0, 5), RequestId(2))
        .unwrap();
    assert!(matches!(
        clone.start(&ReadRequest::new(0, 4), RequestId(3)),
        Err(IoError::Provider { .. })
    ));
    let b = clone.start(&ReadRequest::new(0, 3), RequestId(4)).unwrap();
    assert_eq!(clone.reserved_bytes(), 8);
    assert!(matches!(
        provider.start(&ReadRequest::new(0, 1), RequestId(5)),
        Err(IoError::Provider { .. })
    ));
    assert_eq!(next(&mut a).unwrap().unwrap().bytes(), b"abcde");
    assert_eq!(clone.reserved_bytes(), 3);
    drop(b);
    until(|| provider.in_flight() == 0);
    assert_eq!(provider.reserved_bytes(), 0);
}

#[test]
fn reactor_range_checks_zero_length_and_total_quota_overflow() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abc");
    let provider = services.provider(&source.0, config(3, 1));
    assert!(matches!(
        provider.start(&ReadRequest::new(u64::MAX, 1), RequestId(1)),
        Err(IoError::RangeOverflow)
    ));
    assert!(matches!(
        provider.start(&ReadRequest::new(i64::MAX as u64, 2), RequestId(2)),
        Err(IoError::Provider { .. })
    ));
    assert_eq!(provider.in_flight(), 0);
    // Each range fits signed offsets, but their sum does not fit u64.
    let a = provider
        .start(&ReadRequest::new(0, 1_u64 << 63), RequestId(3))
        .unwrap();
    assert!(matches!(
        provider.start(&ReadRequest::new(0, 1_u64 << 63), RequestId(4)),
        Err(IoError::Provider { .. })
    ));
    assert_eq!(provider.reserved_bytes(), 1_u64 << 63);
    drop(a);
    until(|| provider.in_flight() == 0);
    let empty = services.provider(
        &source.0.with_extension("not-opened"),
        ReactorFileConfig {
            max_request_bytes: 0,
            max_total_bytes: 0,
            ..config(1, 1)
        },
    );
    let mut operation = empty
        .start(&ReadRequest::new(u64::MAX, 0), RequestId(5))
        .unwrap();
    assert!(next(&mut operation).unwrap().is_none());
    assert_eq!(empty.in_flight(), 0);
}

#[test]
fn reactor_submitted_future_drains_even_if_worker_turn_unwinds() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let provider = services.provider(&source.0, config(1, 4));
    let closed = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&closed);
    test_support::set_hook(
        &provider,
        Arc::new(move |event| match event {
            TestEvent::Submitted => panic!("test unwind with a live native future"),
            TestEvent::Closed => {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }),
    );
    let mut operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    assert!(matches!(
        next(&mut operation),
        Err(IoError::Provider { .. })
    ));
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert_eq!(provider.in_flight(), 0);
    test_support::set_hook(&provider, Arc::new(|_| {}));
    let mut replacement = provider
        .start(&ReadRequest::new(0, 4), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), b"abcd");
}

#[test]
fn reactor_host_waker_panic_does_not_reach_native_reactor_or_poison_provider() {
    struct PanickingWake;
    impl Wake for PanickingWake {
        fn wake(self: Arc<Self>) {
            panic!("host wake panic");
        }
    }
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let provider = services.provider(&source.0, config(1, 4));
    let (gate, wait) = Gate::new();
    test_support::set_hook(
        &provider,
        Arc::new(move |event| {
            if event == TestEvent::Submitted {
                wait();
            }
        }),
    );
    let mut operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    let waker = Waker::from(Arc::new(PanickingWake));
    assert!(operation
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    gate.open();
    until(|| test_support::ready(&provider));
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"abcd");
    let mut replacement = provider
        .start(&ReadRequest::new(0, 4), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), b"abcd");
}

#[test]
fn reactor_two_files_and_many_operations_share_host_services() {
    let Some(services) = Services::new() else {
        return;
    };
    let a = Source::new(b"abcdef");
    let b = Source::new(b"UVWXYZ");
    let left = services.provider(&a.0, config(20, 2));
    let right = services.provider(&b.0, config(20, 2));
    let mut operations = Vec::new();
    for index in 0..40 {
        let provider = if index % 2 == 0 { &left } else { &right };
        operations.push(
            provider
                .start(&ReadRequest::new(1, 4), RequestId(index + 1))
                .unwrap(),
        );
    }
    for (index, mut operation) in operations.into_iter().enumerate() {
        let mut bytes = Vec::new();
        while let Some(chunk) = next(&mut operation).unwrap() {
            bytes.extend_from_slice(chunk.bytes());
        }
        assert_eq!(bytes, if index % 2 == 0 { b"bcde" } else { b"VWXY" });
    }
    assert_eq!(left.in_flight(), 0);
    assert_eq!(right.in_flight(), 0);
}

#[test]
fn queued_reactor_replacement_waits_for_cancelled_native_cleanup_without_retry_wakes() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    let direct = services.provider(&source.0, config(1, 8));
    let queued = direct.clone().queued();
    let (gate, wait) = Gate::new();
    test_support::set_hook(
        &direct,
        Arc::new(move |event| {
            if event == TestEvent::Submitted {
                wait();
            }
        }),
    );
    let request = ReadRequest::new(0, 8);
    let mut old = direct.start(&request, RequestId(1)).unwrap();
    gate.entered();
    old.cancel();
    drop(old);
    let mut replacement = queued.start(&request, RequestId(2)).unwrap();
    let (waker, woke) = signal();
    assert!(replacement
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(queued.pending_admissions(), 1);
    assert_eq!(queued.in_flight(), 1);
    assert_eq!(queued.reserved_bytes(), 8);
    assert!(woke.try_recv().is_err(), "capacity wait must not self-wake");
    assert!(matches!(
        queued.clone().start(&request, RequestId(3)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    gate.open();
    woke.recv_timeout(DEADLINE)
        .expect("cleanup grants and wakes replacement");
    // Grant owns the slot even before its receiver is polled.
    assert!(matches!(
        direct.start(&request, RequestId(4)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    assert_eq!(
        next(&mut replacement).unwrap().unwrap().bytes(),
        b"abcdefgh"
    );
    assert_eq!(queued.in_flight(), 0);
    assert_eq!(queued.pending_admissions(), 0);
    assert_eq!(queued.reserved_bytes(), 0);
}

#[test]
fn queued_reactor_fifo_byte_wait_head_cancellation_grants_all_fitting_waiters() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    let direct = services.provider(
        &source.0,
        ReactorFileConfig {
            max_request_bytes: 8,
            max_total_bytes: 8,
            ..config(4, 8)
        },
    );
    let queued = direct.clone().queued();
    let mut active = direct.start(&ReadRequest::new(0, 5), RequestId(1)).unwrap();
    let mut head = queued.start(&ReadRequest::new(0, 4), RequestId(2)).unwrap();
    let mut a = queued.start(&ReadRequest::new(0, 1), RequestId(3)).unwrap();
    let mut b = queued.start(&ReadRequest::new(1, 2), RequestId(4)).unwrap();
    let (head_waker, head_woke) = signal();
    let (a_waker, a_woke) = signal();
    let (b_waker, b_woke) = signal();
    for (operation, waker) in [
        (&mut head, &head_waker),
        (&mut a, &a_waker),
        (&mut b, &b_waker),
    ] {
        assert!(operation
            .poll_chunk(&mut Context::from_waker(waker))
            .is_pending());
    }
    assert_eq!(queued.in_flight(), 1, "byte head must not be bypassed");
    assert!(a_woke.try_recv().is_err());
    assert!(b_woke.try_recv().is_err());
    head.cancel();
    head_woke.recv_timeout(DEADLINE).unwrap();
    a_woke.recv_timeout(DEADLINE).unwrap();
    b_woke.recv_timeout(DEADLINE).unwrap();
    assert_eq!(queued.in_flight(), 3);
    assert_eq!(queued.reserved_bytes(), 8);
    assert_eq!(queued.pending_admissions(), 2);
    assert!(matches!(next(&mut head), Err(IoError::Cancelled { .. })));
    // Dropping a granted-but-unpolled stream is synchronous and submits no read.
    drop(a);
    assert_eq!(queued.in_flight(), 2);
    assert_eq!(queued.reserved_bytes(), 7);
    assert_eq!(next(&mut b).unwrap().unwrap().bytes(), b"bc");
    assert_eq!(next(&mut active).unwrap().unwrap().bytes(), b"abcde");
    assert_eq!(queued.reserved_bytes(), 0);
    assert_eq!(queued.pending_admissions(), 0);
}

#[test]
fn queued_reactor_release_grants_in_fifo_order_and_cancelled_grant_hands_off() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let direct = services.provider(&source.0, config(2, 4));
    let queued = direct.clone().queued();
    let mut left = direct.start(&ReadRequest::new(0, 4), RequestId(1)).unwrap();
    let mut right = direct.start(&ReadRequest::new(0, 4), RequestId(2)).unwrap();
    let mut first = queued.start(&ReadRequest::new(0, 4), RequestId(3)).unwrap();
    let mut second = queued.start(&ReadRequest::new(0, 4), RequestId(4)).unwrap();
    let (first_waker, first_woke) = signal();
    let (second_waker, second_woke) = signal();
    assert!(first
        .poll_chunk(&mut Context::from_waker(&first_waker))
        .is_pending());
    assert!(second
        .poll_chunk(&mut Context::from_waker(&second_waker))
        .is_pending());
    assert_eq!(next(&mut left).unwrap().unwrap().bytes(), b"abcd");
    first_woke.recv_timeout(DEADLINE).unwrap();
    assert!(second_woke.try_recv().is_err());
    assert_eq!(queued.in_flight(), 2);
    drop(first);
    second_woke.recv_timeout(DEADLINE).unwrap();
    assert_eq!(queued.in_flight(), 2);
    assert_eq!(next(&mut second).unwrap().unwrap().bytes(), b"abcd");
    assert_eq!(next(&mut right).unwrap().unwrap().bytes(), b"abcd");
    assert_eq!(queued.in_flight(), 0);
    assert_eq!(queued.pending_admissions(), 0);
}

#[test]
fn queued_reactor_latest_waiting_waker_and_release_cancel_grant_callbacks_are_reentrant() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let direct = services.provider(&source.0, config(2, 4));
    let queued = direct.clone().queued();
    with_callback_probe(&direct, |probe| {
        let request = ReadRequest::new(0, 4);
        let mut left = direct.start(&request, RequestId(1)).unwrap();
        let mut right = direct.start(&request, RequestId(2)).unwrap();
        let mut first = queued.start(&request, RequestId(3)).unwrap();
        let mut second = queued.start(&request, RequestId(4)).unwrap();
        let (old, old_callbacks) = probe.waker(false);
        assert!(first
            .poll_chunk(&mut Context::from_waker(&old))
            .is_pending());
        drop(old); // Only the waiting admission now owns this callback.
        let (latest, latest_callbacks) = probe.waker(false);
        assert!(first
            .poll_chunk(&mut Context::from_waker(&latest))
            .is_pending());
        drop(latest);
        expect_callbacks(&old_callbacks, &[CallbackEvent::Drop], (2, 8, 2));

        let (second_waker, second_callbacks) = probe.waker(false);
        assert!(second
            .poll_chunk(&mut Context::from_waker(&second_waker))
            .is_pending());
        drop(second_waker);
        assert!(
            latest_callbacks.try_recv().is_err(),
            "wait must not self-wake"
        );
        assert_eq!(next(&mut left).unwrap().unwrap().bytes(), b"abcd");
        expect_callbacks(
            &latest_callbacks,
            &[CallbackEvent::Wake, CallbackEvent::Drop],
            (2, 8, 2),
        );
        assert!(old_callbacks.try_recv().is_err(), "stale waker was woken");
        assert!(second_callbacks.try_recv().is_err(), "FIFO was bypassed");

        // An unpolled grant has no IO work. Its cancellation synchronously
        // transfers the charged slot/bytes, then invokes the next host waker.
        drop(first);
        expect_callbacks(
            &second_callbacks,
            &[CallbackEvent::Wake, CallbackEvent::Drop],
            (2, 8, 1),
        );
        assert_eq!(next(&mut second).unwrap().unwrap().bytes(), b"abcd");
        assert_eq!(next(&mut right).unwrap().unwrap().bytes(), b"abcd");
        assert_eq!(queued.in_flight(), 0);
        assert_eq!(queued.reserved_bytes(), 0);
        assert_eq!(queued.pending_admissions(), 0);
    });
}

#[test]
fn queued_reactor_active_latest_waker_panic_is_reentrant_and_worker_survives() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let direct = services.provider(&source.0, config(1, 4));
    let queued = direct.clone().queued();
    with_callback_probe(&direct, |probe| {
        // Keep the RAII gate inside the scope: any assertion unwind opens it
        // before either scoped joining or host-owned IO-worker teardown.
        let (gate, wait) = Gate::new();
        test_support::set_hook(
            &direct,
            Arc::new(move |event| {
                if event == TestEvent::Submitted {
                    wait();
                }
            }),
        );
        let request = ReadRequest::new(0, 4);
        let mut operation = queued.start(&request, RequestId(1)).unwrap();
        assert!(operation
            .poll_chunk(&mut Context::from_waker(Waker::noop()))
            .is_pending());
        gate.entered();
        let (old, old_callbacks) = probe.waker(false);
        assert!(operation
            .poll_chunk(&mut Context::from_waker(&old))
            .is_pending());
        drop(old);
        let (latest, latest_callbacks) = probe.waker(true);
        assert!(operation
            .poll_chunk(&mut Context::from_waker(&latest))
            .is_pending());
        drop(latest);
        expect_callbacks(&old_callbacks, &[CallbackEvent::Drop], (1, 4, 0));
        assert!(latest_callbacks.try_recv().is_err());
        gate.open();
        expect_callbacks(
            &latest_callbacks,
            &[CallbackEvent::Wake, CallbackEvent::Drop],
            (1, 4, 0),
        );
        assert!(old_callbacks.try_recv().is_err(), "stale waker was woken");
        assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"abcd");
        assert!(next(&mut operation).unwrap().is_none());
        let mut replacement = queued.start(&request, RequestId(2)).unwrap();
        assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), b"abcd");
        assert_eq!(queued.in_flight(), 0);
        assert_eq!(queued.reserved_bytes(), 0);
        assert_eq!(queued.pending_admissions(), 0);
    });
}

#[test]
fn queued_reactor_cancel_head_and_grant_wake_panics_do_not_skip_later_callbacks() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    let direct = services.provider(
        &source.0,
        ReactorFileConfig {
            max_total_bytes: 8,
            ..config(4, 8)
        },
    );
    let queued = direct.clone().queued();
    with_callback_probe(&direct, |probe| {
        let mut active = direct.start(&ReadRequest::new(0, 5), RequestId(1)).unwrap();
        let mut head = queued.start(&ReadRequest::new(0, 4), RequestId(2)).unwrap();
        let mut a = queued.start(&ReadRequest::new(0, 1), RequestId(3)).unwrap();
        let mut b = queued.start(&ReadRequest::new(1, 2), RequestId(4)).unwrap();
        let (head_waker, head_callbacks) = probe.waker(true);
        let (a_waker, a_callbacks) = probe.waker(true);
        let (b_waker, b_callbacks) = probe.waker(false);
        for (operation, waker) in [
            (&mut head, &head_waker),
            (&mut a, &a_waker),
            (&mut b, &b_waker),
        ] {
            assert!(operation
                .poll_chunk(&mut Context::from_waker(waker))
                .is_pending());
        }
        drop((head_waker, a_waker, b_waker));
        assert_eq!(queued.in_flight(), 1, "byte-quota head must block FIFO");
        for observed in [&head_callbacks, &a_callbacks, &b_callbacks] {
            assert!(observed.try_recv().is_err(), "wait must not self-wake");
        }

        // Both the cancelled admission's callback and the first grant's
        // callback panic. Neither may unwind cancel or skip the second grant.
        head.cancel();
        for observed in [&head_callbacks, &a_callbacks, &b_callbacks] {
            expect_callbacks(
                observed,
                &[CallbackEvent::Wake, CallbackEvent::Drop],
                (3, 8, 2),
            );
        }
        assert_eq!(
            next(&mut head),
            Err(IoError::Cancelled {
                request_id: RequestId(2),
            })
        );
        head.cancel(); // Repeated cancellation must not call the host again.
        assert!(head_callbacks.try_recv().is_err());
        drop(a);
        assert_eq!(queued.in_flight(), 2);
        assert_eq!(queued.reserved_bytes(), 7);
        assert_eq!(next(&mut b).unwrap().unwrap().bytes(), b"bc");
        assert_eq!(next(&mut active).unwrap().unwrap().bytes(), b"abcde");
        assert_eq!(queued.in_flight(), 0);
        assert_eq!(queued.reserved_bytes(), 0);
        assert_eq!(queued.pending_admissions(), 0);
    });
}

#[test]
fn queued_reactor_unpolled_starts_are_bounded_and_impossible_ranges_fail_before_admission() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcd");
    let queued = services
        .provider(
            &source.0,
            ReactorFileConfig {
                max_request_bytes: 8,
                max_total_bytes: 4,
                ..config(1, 4)
            },
        )
        .queued();
    for request in [
        ReadRequest::new(0, 5),
        ReadRequest::new(u64::MAX, 1),
        ReadRequest::new(i64::MAX as u64, 2),
    ] {
        assert!(queued.start(&request, RequestId(1)).is_err());
        assert_eq!(queued.pending_admissions(), 0);
    }
    let a = queued.start(&ReadRequest::new(0, 4), RequestId(2)).unwrap();
    assert_eq!(queued.pending_admissions(), 1);
    assert_eq!(queued.reserved_bytes(), 0);
    assert!(matches!(
        queued.clone().start(&ReadRequest::new(0, 1), RequestId(3)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    drop(a);
    assert_eq!(queued.pending_admissions(), 0);
    let mut b = queued.start(&ReadRequest::new(0, 4), RequestId(4)).unwrap();
    assert_eq!(next(&mut b).unwrap().unwrap().bytes(), b"abcd");
    assert_eq!(queued.pending_admissions(), 0);
}

#[test]
fn queued_reactor_scheduler_recovers_from_many_cancellations_on_one_slot() {
    let Some(services) = Services::new() else {
        return;
    };
    let source = Source::new(b"abcdefgh");
    let queued = services.provider(&source.0, config(1, 2)).queued();
    let monitor = queued.clone();
    let mut scheduler = IoScheduler::new(queued, 1, 2).unwrap();
    for index in 0..64 {
        let id = scheduler.submit(ReadRequest::new(0, 8)).unwrap();
        // Starting the physical stream is racy with completion by design.
        let _ = scheduler.poll(&mut Context::from_waker(Waker::noop()));
        assert!(scheduler.cancel(id));
        assert!(
            matches!(poll_until(|cx| scheduler.poll(cx)), Err(IoError::Cancelled { request_id }) if request_id == id)
        );
        let id = scheduler
            .submit(ReadRequest::new(0, 8).with_expected_checksum(checksum(b"abcdefgh")))
            .unwrap();
        let mut bytes = Vec::new();
        loop {
            match poll_until(|cx| scheduler.poll(cx)).unwrap() {
                IoEvent::Chunk { chunk, .. } => bytes.extend_from_slice(chunk.bytes()),
                IoEvent::Completed(summary) => {
                    assert_eq!(summary.request_id(), id);
                    break;
                }
            }
        }
        assert_eq!(bytes, b"abcdefgh", "replacement {index}");
    }
    assert_eq!(monitor.in_flight(), 0);
    assert_eq!(monitor.pending_admissions(), 0);
}
