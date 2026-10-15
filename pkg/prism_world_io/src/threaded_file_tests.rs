use super::*;
use crate::threaded_file::{test_support, QueuedThreadedFileProvider};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    task::{Wake, Waker},
    thread,
    time::{Duration, Instant},
};

const DEADLINE: Duration = Duration::from_secs(5);

struct Source(PathBuf);

impl Source {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-threaded-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .and_then(|mut file| std::io::Write::write_all(&mut file, bytes))
            .unwrap();
        Self(path)
    }

    fn provider(&self, capacity: usize, chunk: usize) -> ThreadedFileProvider {
        ThreadedFileProvider::new(
            self.0.clone(),
            ThreadedFileConfig {
                workers: 1,
                capacity,
                max_chunk_size: chunk,
            },
        )
        .unwrap()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        // Tests may still have a detached worker closing its read-only handle.
        fs::remove_file(&self.0).unwrap();
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

fn next(operation: &mut impl ChunkedRead) -> Result<Option<ReadChunk>, IoError> {
    let (waker, rx) = signal();
    let mut context = Context::from_waker(&waker);
    let end = Instant::now() + DEADLINE;
    loop {
        match operation.poll_chunk(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => rx
                .recv_timeout(end.saturating_duration_since(Instant::now()))
                .unwrap(),
        }
    }
}

fn until(mut predicate: impl FnMut() -> bool) {
    let end = Instant::now() + DEADLINE;
    while !predicate() {
        assert!(Instant::now() < end, "worker did not reach expected state");
        thread::yield_now();
    }
}

// A RAII latch prevents a failed assertion from permanently parking a detached
// test worker. Read entry is signalled separately from the release condition.
struct Gate {
    release: Arc<(Mutex<bool>, Condvar)>,
    entered: mpsc::Receiver<()>,
}

impl Gate {
    fn install(provider: &ThreadedFileProvider) -> Self {
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_release = Arc::clone(&release);
        let (tx, entered) = mpsc::channel();
        test_support::set_hook(
            provider,
            Arc::new(move || {
                let _ = tx.send(());
                let (lock, changed) = &*worker_release;
                let guard = lock.lock().unwrap();
                drop(changed.wait_while(guard, |released| !*released).unwrap());
            }),
        );
        Self { release, entered }
    }

    fn entered(&self) {
        self.entered.recv_timeout(DEADLINE).unwrap();
    }

    fn open(&self) {
        let (lock, changed) = &*self.release;
        *lock.lock().unwrap() = true;
        changed.notify_all();
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
    }
}

#[test]
fn threaded_validates_limits_and_reservations_without_opening_path() {
    let valid = ThreadedFileConfig {
        workers: 1,
        capacity: 1,
        max_chunk_size: 1,
    };
    for config in [
        ThreadedFileConfig {
            workers: 0,
            ..valid
        },
        ThreadedFileConfig {
            capacity: 0,
            ..valid
        },
        ThreadedFileConfig {
            max_chunk_size: 0,
            ..valid
        },
        ThreadedFileConfig {
            max_chunk_size: usize::MAX,
            ..valid
        },
        ThreadedFileConfig {
            capacity: usize::MAX,
            ..valid
        },
    ] {
        assert!(ThreadedFileProvider::new("not-opened", config).is_err());
    }
    assert!(test_support::allocation_error(usize::MAX).is_err());
    let provider = ThreadedFileProvider::new("not-opened", valid).unwrap();
    assert_eq!(provider.path(), std::path::Path::new("not-opened"));
    assert_eq!(provider.config(), valid);
    assert_eq!(test_support::allocated_slot_count(&provider), 1);
    provider.shutdown();
    until(|| provider.is_shutdown_complete());
}

#[test]
fn threaded_reads_exact_range_caps_chunks_and_releases_terminal_slot() {
    let source = Source::new(b"0123456789");
    let provider = source.provider(1, 2);
    let request = ReadRequest::new(2, 5).with_chunk_size(9).unwrap();
    let mut operation = provider.start(&request, RequestId(71)).unwrap();
    for (offset, bytes) in [(2, &b"23"[..]), (4, &b"45"[..]), (6, &b"6"[..])] {
        let chunk = next(&mut operation).unwrap().unwrap();
        assert_eq!(chunk.offset(), offset);
        assert_eq!(chunk.bytes(), bytes);
    }
    assert_eq!(provider.outstanding(), 0);
    assert_eq!(next(&mut operation), Ok(None));
    assert_eq!(next(&mut operation), Ok(None));
    operation.cancel();
    assert_eq!(next(&mut operation), Ok(None));
}

#[test]
fn threaded_range_overflow_rejected_before_admission() {
    let source = Source::new(b"a");
    let provider = source.provider(1, 1);
    assert!(matches!(
        provider.start(&ReadRequest::new(u64::MAX, 1), RequestId(1)),
        Err(IoError::RangeOverflow)
    ));
    assert_eq!(provider.outstanding(), 0);
}

#[test]
fn threaded_empty_request_does_not_open_missing_source() {
    let source = Source::new(b"a");
    let missing = source.0.with_extension("missing");
    let provider = ThreadedFileProvider::new(missing, ThreadedFileConfig::default()).unwrap();
    let mut empty = provider
        .start(&ReadRequest::new(u64::MAX, 0), RequestId(1))
        .unwrap();
    assert_eq!(next(&mut empty), Ok(None));
    let mut missing = provider
        .start(&ReadRequest::new(0, 1), RequestId(2))
        .unwrap();
    assert!(matches!(next(&mut missing), Err(IoError::Provider { .. })));
    assert_eq!(provider.outstanding(), 0);
}

#[test]
fn threaded_short_read_keeps_cumulative_actual_length() {
    let source = Source::new(b"abc");
    let provider = source.provider(1, 2);
    let mut operation = provider
        .start(&ReadRequest::new(0, 5), RequestId(1))
        .unwrap();
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"ab");
    assert_eq!(
        next(&mut operation),
        Err(IoError::ShortRead {
            expected: 5,
            actual: 3
        })
    );
    assert_eq!(next(&mut operation), Ok(None));
    assert_eq!(provider.outstanding(), 0);
}

#[test]
fn threaded_parked_chunks_keep_admission_but_do_not_occupy_worker() {
    let source = Source::new(b"abcdef");
    let provider = source.provider(2, 2);
    let mut parked = provider
        .start(&ReadRequest::new(0, 6), RequestId(1))
        .unwrap();
    assert_eq!(next(&mut parked).unwrap().unwrap().bytes(), b"ab");
    let mut other = provider
        .start(&ReadRequest::new(4, 2), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut other).unwrap().unwrap().bytes(), b"ef");
    assert_eq!(provider.outstanding(), 1);
    assert_eq!(next(&mut parked).unwrap().unwrap().bytes(), b"cd");
    assert_eq!(next(&mut parked).unwrap().unwrap().bytes(), b"ef");
    assert_eq!(provider.outstanding(), 0);
}

#[test]
fn threaded_running_cancel_is_immediate_but_keeps_slot_until_cleanup() {
    let source = Source::new(b"abcd");
    let provider = source.provider(1, 2);
    let gate = Gate::install(&provider);
    let mut operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    operation.cancel();
    operation.cancel();
    assert_eq!(
        next(&mut operation),
        Err(IoError::Cancelled {
            request_id: RequestId(1)
        })
    );
    assert_eq!(provider.outstanding(), 1);
    assert!(matches!(
        provider.start(&ReadRequest::new(0, 1), RequestId(2)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    gate.open();
    until(|| provider.outstanding() == 0);
    let mut retry = provider
        .start(&ReadRequest::new(0, 1), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut retry).unwrap().unwrap().bytes(), b"a");
}

#[test]
fn threaded_queued_cancel_never_enters_io_and_does_not_duplicate_queue_entry() {
    let source = Source::new(b"abcd");
    let provider = source.provider(2, 2);
    let gate = Gate::install(&provider);
    let first = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    let mut queued = provider
        .start(&ReadRequest::new(0, 1), RequestId(2))
        .unwrap();
    queued.cancel();
    queued.cancel();
    assert_eq!(
        next(&mut queued),
        Err(IoError::Cancelled {
            request_id: RequestId(2)
        })
    );
    drop(first);
    gate.open();
    until(|| provider.outstanding() == 0);
    assert!(gate.entered.try_recv().is_err());
}

#[test]
fn threaded_drop_and_repeated_shutdown_never_wait_for_blocked_read() {
    let source = Source::new(b"abcd");
    let provider = source.provider(1, 2);
    let observer = provider.clone();
    let gate = Gate::install(&provider);
    let operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(1))
        .unwrap();
    gate.entered();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        drop(operation);
        provider.shutdown();
        provider.shutdown();
        drop(provider);
        tx.send(()).unwrap();
    });
    rx.recv_timeout(DEADLINE).unwrap();
    assert!(observer.is_shutdown());
    assert!(!observer.is_shutdown_complete());
    assert_eq!(observer.outstanding(), 1);
    assert!(matches!(
        observer.start(&ReadRequest::new(0, 1), RequestId(2)),
        Err(IoError::Provider { .. })
    ));
    gate.open();
    until(|| observer.is_shutdown_complete());
    assert_eq!(observer.outstanding(), 0);
}

#[test]
fn threaded_last_owner_drop_cancels_surviving_operation_without_join() {
    let source = Source::new(b"abcd");
    let provider = source.provider(1, 2);
    let gate = Gate::install(&provider);
    let mut operation = provider
        .start(&ReadRequest::new(0, 4), RequestId(4))
        .unwrap();
    gate.entered();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        drop(provider);
        tx.send(()).unwrap();
    });
    rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        next(&mut operation),
        Err(IoError::Cancelled {
            request_id: RequestId(4)
        })
    );
    gate.open();
}

#[test]
fn threaded_waker_replacement_has_no_lost_wake_or_self_spin() {
    let source = Source::new(b"ab");
    let provider = source.provider(1, 1);
    let gate = Gate::install(&provider);
    let mut operation = provider
        .start(&ReadRequest::new(0, 2), RequestId(1))
        .unwrap();
    gate.entered();
    let (old, old_rx) = signal();
    let (current, current_rx) = signal();
    assert!(operation
        .poll_chunk(&mut Context::from_waker(&old))
        .is_pending());
    assert!(operation
        .poll_chunk(&mut Context::from_waker(&current))
        .is_pending());
    assert!(current_rx.try_recv().is_err());
    gate.open();
    current_rx.recv_timeout(DEADLINE).unwrap();
    assert!(old_rx.try_recv().is_err());
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"a");
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"b");
}

struct HostWake {
    provider: ThreadedFileProvider,
    called: AtomicUsize,
    panic: AtomicBool,
    tx: mpsc::Sender<()>,
}

impl Wake for HostWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        // Reentry would deadlock if callbacks were invoked with the pool locked.
        let _ = self.provider.outstanding();
        self.called.fetch_add(1, Ordering::SeqCst);
        let _ = self.tx.send(());
        assert!(
            !self.panic.swap(false, Ordering::SeqCst),
            "injected host wake panic"
        );
    }
}

#[test]
fn threaded_reentrant_panicking_wake_does_not_kill_the_only_worker() {
    let source = Source::new(b"ab");
    let provider = source.provider(1, 1);
    let gate = Gate::install(&provider);
    let mut operation = provider
        .start(&ReadRequest::new(0, 2), RequestId(1))
        .unwrap();
    gate.entered();
    let (tx, rx) = mpsc::channel();
    let host = Arc::new(HostWake {
        provider: provider.clone(),
        called: AtomicUsize::new(0),
        panic: AtomicBool::new(true),
        tx,
    });
    let waker = Waker::from(Arc::clone(&host));
    assert!(operation
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    gate.open();
    rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"a");
    assert_eq!(next(&mut operation).unwrap().unwrap().bytes(), b"b");
    assert_eq!(host.called.load(Ordering::SeqCst), 1);
}

#[test]
fn threaded_shutdown_wakes_every_waiter_despite_host_panic() {
    let source = Source::new(b"ab");
    let provider = source.provider(2, 1);
    let gate = Gate::install(&provider);
    let mut first = provider
        .start(&ReadRequest::new(0, 2), RequestId(1))
        .unwrap();
    gate.entered();
    let mut second = provider
        .start(&ReadRequest::new(0, 2), RequestId(2))
        .unwrap();
    let (tx, rx) = mpsc::channel();
    let waker = Waker::from(Arc::new(HostWake {
        provider: provider.clone(),
        called: AtomicUsize::new(0),
        panic: AtomicBool::new(true),
        tx,
    }));
    assert!(first
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    let (other, other_rx) = signal();
    assert!(second
        .poll_chunk(&mut Context::from_waker(&other))
        .is_pending());
    provider.shutdown();
    rx.recv_timeout(DEADLINE).unwrap();
    other_rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        next(&mut first),
        Err(IoError::Cancelled {
            request_id: RequestId(1)
        })
    );
    assert_eq!(
        next(&mut second),
        Err(IoError::Cancelled {
            request_id: RequestId(2)
        })
    );
    gate.open();
    until(|| provider.is_shutdown_complete());
    assert_eq!(provider.outstanding(), 0);
}

#[test]
fn threaded_result_before_registration_is_observed_without_needing_a_wake() {
    let source = Source::new(b"a");
    let provider = source.provider(1, 1);
    let mut operation = provider
        .start(&ReadRequest::new(0, 1), RequestId(1))
        .unwrap();
    test_support::wait_ready(&provider);
    let (waker, rx) = signal();
    assert!(matches!(
        operation.poll_chunk(&mut Context::from_waker(&waker)),
        Poll::Ready(Ok(Some(_)))
    ));
    assert!(rx.try_recv().is_err());
}

#[test]
fn threaded_cancel_ready_result_and_awaiting_ack_reclaims_slots() {
    let source = Source::new(b"ab");
    let provider = source.provider(1, 1);
    let mut operation = provider
        .start(&ReadRequest::new(0, 2), RequestId(1))
        .unwrap();
    test_support::wait_ready(&provider);
    operation.cancel();
    operation.cancel();
    until(|| provider.outstanding() == 0);
    let mut replacement = provider
        .start(&ReadRequest::new(0, 2), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), b"a");
    drop(replacement);
    until(|| provider.outstanding() == 0);
    // Dropping the old, already cancelled operation cannot cancel a reused slot.
    let mut live = provider
        .start(&ReadRequest::new(0, 1), RequestId(3))
        .unwrap();
    drop(operation);
    assert_eq!(next(&mut live).unwrap().unwrap().bytes(), b"a");
}

#[test]
fn threaded_worker_read_panic_is_terminal_and_pool_can_recover() {
    let source = Source::new(b"ab");
    let provider = source.provider(1, 1);
    let panic_once = Arc::new(AtomicBool::new(true));
    test_support::set_hook(
        &provider,
        Arc::new(move || {
            assert!(
                !panic_once.swap(false, Ordering::SeqCst),
                "injected read panic"
            );
        }),
    );
    let mut operation = provider
        .start(&ReadRequest::new(0, 1), RequestId(1))
        .unwrap();
    assert!(matches!(
        next(&mut operation),
        Err(IoError::Provider { .. })
    ));
    let mut retry = provider
        .start(&ReadRequest::new(0, 1), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut retry).unwrap().unwrap().bytes(), b"a");
}

fn scheduler_next(
    scheduler: &mut IoScheduler<QueuedThreadedFileProvider>,
) -> Result<IoEvent, IoError> {
    let (waker, rx) = signal();
    let end = Instant::now() + DEADLINE;
    loop {
        match scheduler.poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => rx
                .recv_timeout(end.saturating_duration_since(Instant::now()))
                .unwrap(),
        }
    }
}

#[test]
fn queued_scheduler_retries_exhausted_cancellation_cleanup_and_reads_two_mib() {
    let bytes = vec![0x5a; 2 * 1024 * 1024];
    let source = Source::new(&bytes);
    let direct = source.provider(1, 64 * 1024);
    let gate = Gate::install(&direct);
    let provider = direct.clone().queued();
    let mut scheduler = IoScheduler::new(provider.clone(), 1, 1).unwrap();
    let first = scheduler.submit(ReadRequest::new(0, 1)).unwrap();
    let (waker, rx) = signal();
    assert!(scheduler
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    gate.entered();
    assert!(scheduler.cancel(first));
    assert_eq!(
        scheduler_next(&mut scheduler),
        Err(IoError::Cancelled { request_id: first })
    );
    let retry = scheduler
        .submit(ReadRequest::new(0, bytes.len() as u64).with_expected_checksum(checksum(&bytes)))
        .unwrap();
    while rx.try_recv().is_ok() {}
    assert!(scheduler
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(provider.outstanding(), 1);
    assert_eq!(provider.pending_admissions(), 1);
    assert!(matches!(
        direct.start(&ReadRequest::new(0, 1), RequestId(99)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    assert!(rx.try_recv().is_err(), "waiting must not self-wake");
    gate.open();
    rx.recv_timeout(DEADLINE).unwrap();
    let mut actual = Vec::new();
    loop {
        match scheduler_next(&mut scheduler).unwrap() {
            IoEvent::Chunk { request_id, chunk } => {
                assert_eq!(request_id, retry);
                actual.extend_from_slice(chunk.bytes());
            }
            IoEvent::Completed(summary) => {
                assert_eq!(summary.request_id, retry);
                assert_eq!(summary.length, bytes.len() as u64);
                break;
            }
        }
    }
    assert_eq!(actual, bytes);
    assert!(scheduler.is_empty());
    assert_eq!(provider.outstanding(), 0);
    assert_eq!(provider.pending_admissions(), 0);
    provider.shutdown();
    until(|| provider.is_shutdown_complete());
}

#[test]
fn queued_start_is_lazy_bounded_and_cancel_drop_unregister_before_poll() {
    let source = Source::new(b"ab");
    let provider = source.provider(1, 1).queued();
    assert_eq!(provider.path(), source.0.as_path());
    assert_eq!(provider.config().capacity, 1);
    assert!(matches!(
        provider.start(&ReadRequest::new(u64::MAX, 1), RequestId(0)),
        Err(IoError::RangeOverflow)
    ));
    let mut first = provider
        .start(&ReadRequest::new(0, 1), RequestId(1))
        .unwrap();
    assert_eq!(provider.outstanding(), 0, "start must not enqueue IO");
    assert_eq!(provider.pending_admissions(), 1);
    assert!(matches!(
        provider
            .clone()
            .start(&ReadRequest::new(0, 1), RequestId(2)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    first.cancel();
    first.cancel();
    assert_eq!(provider.pending_admissions(), 0);
    assert_eq!(
        next(&mut first),
        Err(IoError::Cancelled {
            request_id: RequestId(1)
        })
    );
    let second = provider
        .start(&ReadRequest::new(0, 1), RequestId(1))
        .unwrap();
    drop(first); // Reusing an admission index or request identity is safe.
    assert_eq!(provider.pending_admissions(), 1);
    drop(second);
    assert_eq!(provider.pending_admissions(), 0);
    let mut third = provider
        .start(&ReadRequest::new(1, 1), RequestId(1))
        .unwrap();
    assert_eq!(next(&mut third).unwrap().unwrap().bytes(), b"b");
}

#[test]
fn queued_cancel_registered_waiter_replaces_waker_without_spin_or_stale_wake() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let gate = Gate::install(&direct);
    let occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let provider = direct.clone().queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(1))
        .unwrap();
    let (old, old_rx) = signal();
    let (current, current_rx) = signal();
    assert!(waiting
        .poll_chunk(&mut Context::from_waker(&old))
        .is_pending());
    assert!(waiting
        .poll_chunk(&mut Context::from_waker(&current))
        .is_pending());
    assert!(old_rx.try_recv().is_err());
    assert!(current_rx.try_recv().is_err());
    waiting.cancel();
    current_rx.recv_timeout(DEADLINE).unwrap();
    assert!(old_rx.try_recv().is_err());
    assert_eq!(provider.pending_admissions(), 0);
    let mut live = provider
        .start(&ReadRequest::new(1, 1), RequestId(1))
        .unwrap();
    drop(waiting);
    assert!(live
        .poll_chunk(&mut Context::from_waker(&current))
        .is_pending());
    drop(occupied);
    gate.open();
    current_rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(next(&mut live).unwrap().unwrap().bytes(), b"b");
    assert!(old_rx.try_recv().is_err());
}

#[test]
fn queued_terminal_consumption_grants_slot_before_wake_and_drop_cleans_grant() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let mut occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    test_support::wait_ready(&direct);
    let provider = direct.clone().queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    let (waker, rx) = signal();
    assert!(waiting
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(next(&mut occupied).unwrap().unwrap().bytes(), b"a");
    rx.recv_timeout(DEADLINE).unwrap();
    // The grant itself owns the slot even if its scheduler hasn't polled again.
    assert_eq!(provider.outstanding(), 1);
    assert!(matches!(
        direct.start(&ReadRequest::new(0, 1), RequestId(3)),
        Err(IoError::Backpressure { capacity: 1 })
    ));
    drop(waiting);
    assert_eq!(provider.pending_admissions(), 0);
    until(|| provider.outstanding() == 0);
    let mut replacement = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut replacement).unwrap().unwrap().bytes(), b"b");
}

#[test]
fn queued_capacity_freed_before_registration_is_seen_without_a_wake() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let mut occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    let provider = direct.clone().queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    assert_eq!(next(&mut occupied).unwrap().unwrap().bytes(), b"a");
    assert_eq!(next(&mut waiting).unwrap().unwrap().bytes(), b"b");
}

#[test]
fn queued_multiple_schedulers_share_fifo_grants_with_colliding_request_ids() {
    let source = Source::new(b"ab");
    let direct = source.provider(2, 1);
    let gate = Gate::install(&direct);
    let first = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let second = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    let provider = direct.clone().queued();
    let mut a = IoScheduler::new(provider.clone(), 1, 1).unwrap();
    let mut b = IoScheduler::new(direct.clone().queued(), 1, 1).unwrap();
    assert_eq!(a.submit(ReadRequest::new(0, 1)).unwrap(), RequestId(1));
    assert_eq!(b.submit(ReadRequest::new(1, 1)).unwrap(), RequestId(1));
    let (aw, arx) = signal();
    let (bw, brx) = signal();
    assert!(a.poll(&mut Context::from_waker(&aw)).is_pending());
    assert!(b.poll(&mut Context::from_waker(&bw)).is_pending());
    assert_eq!(provider.pending_admissions(), 2);
    assert!(arx.try_recv().is_err());
    assert!(brx.try_recv().is_err());
    drop(first);
    drop(second);
    gate.open();
    arx.recv_timeout(DEADLINE).unwrap();
    brx.recv_timeout(DEADLINE).unwrap();
    for (scheduler, expected) in [(&mut a, b"a"), (&mut b, b"b")] {
        assert!(
            matches!(scheduler_next(scheduler), Ok(IoEvent::Chunk { chunk, .. }) if chunk.bytes() == expected)
        );
        assert!(matches!(
            scheduler_next(scheduler),
            Ok(IoEvent::Completed(_))
        ));
    }
    assert_eq!(provider.outstanding(), 0);
    assert_eq!(provider.pending_admissions(), 0);
}

#[test]
fn queued_fifo_grants_only_head_and_canceling_grant_advances_next_waiter() {
    let source = Source::new(b"ab");
    let direct = source.provider(2, 1);
    let mut occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    test_support::wait_ready(&direct);
    let blocker = direct.start(&ReadRequest::new(0, 2), RequestId(2)).unwrap();
    let provider = direct.clone().queued();
    // FIFO follows poll registration, not allocation index or request identity.
    let mut second = provider
        .start(&ReadRequest::new(1, 1), RequestId(1))
        .unwrap();
    let mut first = provider
        .start(&ReadRequest::new(0, 1), RequestId(1))
        .unwrap();
    let (first_waker, first_rx) = signal();
    let (second_waker, second_rx) = signal();
    assert!(first
        .poll_chunk(&mut Context::from_waker(&first_waker))
        .is_pending());
    assert!(second
        .poll_chunk(&mut Context::from_waker(&second_waker))
        .is_pending());
    assert_eq!(next(&mut occupied).unwrap().unwrap().bytes(), b"a");
    first_rx.recv_timeout(DEADLINE).unwrap();
    assert!(second_rx.try_recv().is_err(), "only FIFO head gets a slot");
    assert_eq!(provider.outstanding(), 2);
    drop(first); // Cancel the grant before its scheduler polls again.
    second_rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(next(&mut second).unwrap().unwrap().bytes(), b"b");
    drop(blocker);
    until(|| provider.outstanding() == 0);
}

#[test]
fn queued_shutdown_wakes_every_waiter_despite_reentrant_panicking_callback() {
    let source = Source::new(b"ab");
    let direct = source.provider(2, 1);
    let gate = Gate::install(&direct);
    let occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let other = direct.start(&ReadRequest::new(0, 1), RequestId(2)).unwrap();
    let provider = direct.clone().queued();
    let mut first = provider
        .start(&ReadRequest::new(0, 1), RequestId(3))
        .unwrap();
    let mut second = provider
        .start(&ReadRequest::new(1, 1), RequestId(4))
        .unwrap();
    let (tx, rx) = mpsc::channel();
    let host = Waker::from(Arc::new(HostWake {
        provider: direct.clone(),
        called: AtomicUsize::new(0),
        panic: AtomicBool::new(true),
        tx,
    }));
    let (waker, other_rx) = signal();
    assert!(first
        .poll_chunk(&mut Context::from_waker(&host))
        .is_pending());
    assert!(second
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    provider.shutdown();
    provider.shutdown();
    rx.recv_timeout(DEADLINE).unwrap();
    other_rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        next(&mut first),
        Err(IoError::Cancelled {
            request_id: RequestId(3)
        })
    );
    assert_eq!(
        next(&mut second),
        Err(IoError::Cancelled {
            request_id: RequestId(4)
        })
    );
    assert!(matches!(
        provider.start(&ReadRequest::new(0, 1), RequestId(5)),
        Err(IoError::Provider { .. })
    ));
    drop((first, second, occupied, other));
    assert_eq!(provider.pending_admissions(), 0);
    gate.open();
    until(|| provider.is_shutdown_complete());
}

#[test]
fn queued_last_owner_drop_wakes_surviving_admission_without_join() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let gate = Gate::install(&direct);
    let occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let provider = direct.queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    let (waker, rx) = signal();
    assert!(waiting
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    drop(provider);
    rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(
        next(&mut waiting),
        Err(IoError::Cancelled {
            request_id: RequestId(2)
        })
    );
    drop((waiting, occupied));
    gate.open();
}

#[test]
fn queued_capacity_wake_panic_reenters_pool_without_killing_worker() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let gate = Gate::install(&direct);
    let occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let provider = direct.clone().queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    let (tx, rx) = mpsc::channel();
    let host = Arc::new(HostWake {
        provider: direct,
        called: AtomicUsize::new(0),
        panic: AtomicBool::new(true),
        tx,
    });
    let waker = Waker::from(Arc::clone(&host));
    assert!(waiting
        .poll_chunk(&mut Context::from_waker(&waker))
        .is_pending());
    drop(occupied);
    gate.open();
    rx.recv_timeout(DEADLINE).unwrap();
    assert_eq!(next(&mut waiting).unwrap().unwrap().bytes(), b"b");
    assert_eq!(host.called.load(Ordering::SeqCst), 1);
    assert_eq!(provider.outstanding(), 0);
}

struct HostDrop {
    provider: QueuedThreadedFileProvider,
    tx: mpsc::Sender<()>,
}

#[allow(
    clippy::manual_noop_waker,
    reason = "custom final drop tests reentry and panic isolation"
)]
impl Wake for HostDrop {
    fn wake(self: Arc<Self>) {}
}

impl Drop for HostDrop {
    fn drop(&mut self) {
        let _ = self.provider.outstanding();
        let _ = self.provider.pending_admissions();
        self.tx.send(()).unwrap();
        panic!("injected host waker drop panic");
    }
}

#[test]
fn queued_replaced_and_canceled_waker_drop_can_reenter_and_panic() {
    let source = Source::new(b"ab");
    let direct = source.provider(1, 1);
    let gate = Gate::install(&direct);
    let occupied = direct.start(&ReadRequest::new(0, 1), RequestId(1)).unwrap();
    gate.entered();
    let provider = direct.queued();
    let mut waiting = provider
        .start(&ReadRequest::new(1, 1), RequestId(2))
        .unwrap();
    for cancel in [false, true] {
        let (tx, rx) = mpsc::channel();
        let host = Waker::from(Arc::new(HostDrop {
            provider: provider.clone(),
            tx,
        }));
        assert!(waiting
            .poll_chunk(&mut Context::from_waker(&host))
            .is_pending());
        drop(host); // The pool now owns the final waker reference.
        if cancel {
            waiting.cancel();
        } else {
            let (replacement, _) = signal();
            assert!(waiting
                .poll_chunk(&mut Context::from_waker(&replacement))
                .is_pending());
        }
        rx.recv_timeout(DEADLINE).unwrap();
    }
    assert_eq!(provider.pending_admissions(), 0);
    drop(occupied);
    gate.open();
    until(|| provider.outstanding() == 0);
    let mut retry = provider
        .start(&ReadRequest::new(1, 1), RequestId(3))
        .unwrap();
    assert_eq!(next(&mut retry).unwrap().unwrap().bytes(), b"b");
}

#[test]
fn queued_concurrent_schedulers_cancel_and_retry_within_shared_bounds() {
    let source = Source::new(b"abcdefgh");
    let provider = source.provider(4, 1).queued();
    let mut threads = Vec::new();
    for _ in 0..4 {
        let provider = provider.clone();
        threads.push(thread::spawn(move || {
            let mut scheduler = IoScheduler::new(provider.clone(), 1, 1).unwrap();
            let (waker, _) = signal();
            for _ in 0..32 {
                let canceled = scheduler.submit(ReadRequest::new(0, 8)).unwrap();
                let _ = scheduler.poll(&mut Context::from_waker(&waker));
                assert!(scheduler.cancel(canceled));
                assert_eq!(
                    scheduler_next(&mut scheduler),
                    Err(IoError::Cancelled {
                        request_id: canceled
                    })
                );
                scheduler.submit(ReadRequest::new(0, 2)).unwrap();
                for _ in 0..2 {
                    assert!(matches!(
                        scheduler_next(&mut scheduler),
                        Ok(IoEvent::Chunk { .. })
                    ));
                }
                assert!(matches!(
                    scheduler_next(&mut scheduler),
                    Ok(IoEvent::Completed(_))
                ));
                assert!(provider.outstanding() <= 4);
                assert!(provider.pending_admissions() <= 4);
            }
        }));
    }
    for worker in threads {
        worker.join().unwrap();
    }
    until(|| provider.outstanding() == 0);
    assert_eq!(provider.pending_admissions(), 0);
    provider.shutdown();
    until(|| provider.is_shutdown_complete());
}
