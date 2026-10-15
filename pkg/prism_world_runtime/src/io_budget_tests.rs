use super::tests::{context, manifest, MemoryProvider};
use super::*;
use std::sync::atomic::AtomicUsize;
use std::sync::{mpsc, Barrier};
use std::time::Duration;

fn request(bytes: u64) -> CellReadRequest {
    let mut manifest = manifest(0);
    manifest.io_bytes = bytes;
    CellReadRequest::new(manifest, 1)
}

fn bounded(single: u64, total: u64) -> IoSchedulerAdapter<MemoryProvider> {
    IoSchedulerAdapter::with_byte_limits(
        MemoryProvider {
            bytes: Arc::new(vec![7]),
        },
        1,
        32,
        1,
        IoByteLimits {
            max_request_bytes: single,
            max_total_bytes: total,
        },
    )
    .expect("bounded adapter")
}

fn provider_error<T: core::fmt::Debug>(result: Result<T, IoError>, text: &str) {
    assert!(
        matches!(result, Err(IoError::Provider { ref message }) if message.contains(text)),
        "expected {text:?}, got {result:?}"
    );
}

fn next<P: ReadProvider>(adapter: &IoSchedulerAdapter<P>) -> IoAdapterEvent {
    match adapter.poll(&mut context()) {
        Poll::Ready(Ok(event)) => event,
        other => panic!("expected an event, got {other:?}"),
    }
}

#[test]
fn byte_limits_are_exact_and_rejections_do_not_consume_quota_or_ids() {
    let adapter = bounded(3, 4);
    provider_error(adapter.submit(request(4)), "per-request limit 3");
    assert_eq!(adapter.admitted_bytes(), Ok(0));
    assert!(adapter.is_empty());
    let first = adapter.submit(request(3)).expect("exact single limit");
    provider_error(adapter.submit(request(2)), "total limit 4");
    assert_eq!(adapter.len(), 1);
    assert_eq!(adapter.admitted_bytes(), Ok(3));
    let second = adapter.submit(request(1)).expect("exact total limit");
    assert_eq!(adapter.admitted_bytes(), Ok(4));
    // Rejected admission must not advance the scheduler's ID sequence.
    let reference = bounded(3, 4);
    assert_eq!(reference.submit(request(3)), Ok(first));
    assert_eq!(reference.submit(request(1)), Ok(second));
    assert!(adapter.cancel(first));
    assert_eq!(adapter.admitted_bytes(), Ok(4));
    assert!(
        matches!(next(&adapter), IoAdapterEvent::Failed { request_id, .. } if request_id == first)
    );
    assert_eq!(adapter.admitted_bytes(), Ok(1));
    adapter.submit(request(3)).expect("released three bytes");
}

#[test]
fn zero_and_independent_byte_limits_are_preserved_without_clamping() {
    let zero = bounded(0, 0);
    assert_eq!(
        zero.byte_limits(),
        Some(IoByteLimits {
            max_request_bytes: 0,
            max_total_bytes: 0
        })
    );
    provider_error(zero.submit(request(1)), "per-request limit 0");
    let id = zero.submit(request(0)).expect("zero declaration admitted");
    assert_eq!(zero.admitted_bytes(), Ok(0));
    assert!(zero.cancel(id));
    assert!(matches!(next(&zero), IoAdapterEvent::Failed { request_id, .. } if request_id == id));
    assert!(zero.is_empty());

    let independent = bounded(9, 2);
    assert_eq!(
        independent.byte_limits(),
        Some(IoByteLimits {
            max_request_bytes: 9,
            max_total_bytes: 2
        })
    );
    provider_error(independent.submit(request(3)), "total limit 2");
    assert_eq!(independent.admitted_bytes(), Ok(0));
    independent.submit(request(2)).expect("exact total");
    let total_zero = bounded(9, 0);
    provider_error(total_zero.submit(request(1)), "total limit 0");
    let (empty, _) = probe(Outcome::Success, 0, 0);
    assert_eq!(
        empty.read_to_completion(&request(0), CancellationToken::new(false)),
        Ok(vec![])
    );
    assert_eq!(empty.admitted_bytes(), Ok(0));
    assert!(empty.is_empty());
}

#[test]
fn declaration_sum_and_range_overflow_never_consume_quota() {
    for adapter in [
        bounded(u64::MAX, u64::MAX),
        IoSchedulerAdapter::new(
            MemoryProvider {
                bytes: Arc::new(vec![7]),
            },
            1,
            32,
        )
        .expect("legacy constructor"),
    ] {
        assert_eq!(
            adapter.submit(request(1).with_byte_offset(u64::MAX)),
            Err(IoError::RangeOverflow)
        );
        assert_eq!(adapter.admitted_bytes(), Ok(0));
        assert!(adapter.is_empty());
        let id = adapter.submit(request(u64::MAX)).expect("exact u64 bound");
        provider_error(adapter.submit(request(1)), "admission overflow");
        assert_eq!(adapter.len(), 1);
        assert_eq!(adapter.admitted_bytes(), Ok(u64::MAX));
        assert!(adapter.cancel(id));
        assert!(
            matches!(next(&adapter), IoAdapterEvent::Failed { request_id, .. } if request_id == id)
        );
        assert_eq!(adapter.admitted_bytes(), Ok(0));
        adapter.submit(request(1)).expect("quota recovered");
    }
}

#[derive(Clone, Copy, Debug)]
enum Outcome {
    Success,
    StartError,
    PollError,
    ShortRead,
}

#[derive(Default)]
struct Counts {
    starts: AtomicUsize,
    cancels: AtomicUsize,
    drops: AtomicUsize,
}

struct ProbeProvider {
    counts: Arc<Counts>,
    outcome: Outcome,
}

struct ProbeOperation {
    counts: Arc<Counts>,
    offset: u64,
    delivered: bool,
    outcome: Outcome,
}

impl ReadProvider for ProbeProvider {
    type Operation = ProbeOperation;

    fn start(&self, request: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        self.counts.starts.fetch_add(1, Ordering::Relaxed);
        // Offset zero exercises each failure; the unrelated request succeeds.
        let outcome = if request.offset() == 0 {
            self.outcome
        } else {
            Outcome::Success
        };
        if matches!(outcome, Outcome::StartError) {
            return Err(internal_error("probe start failure"));
        }
        Ok(ProbeOperation {
            counts: Arc::clone(&self.counts),
            offset: request.offset(),
            delivered: request.length() == 0,
            outcome,
        })
    }
}

impl ChunkedRead for ProbeOperation {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if matches!(self.outcome, Outcome::PollError) {
            return Poll::Ready(Err(internal_error("probe poll failure")));
        }
        if self.delivered || matches!(self.outcome, Outcome::ShortRead) {
            Poll::Ready(Ok(None))
        } else {
            self.delivered = true;
            Poll::Ready(Ok(Some(ReadChunk::new(self.offset, vec![7]))))
        }
    }

    fn cancel(&mut self) {
        self.counts.cancels.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for ProbeOperation {
    fn drop(&mut self) {
        self.counts.drops.fetch_add(1, Ordering::Relaxed);
    }
}

fn probe(
    outcome: Outcome,
    single: u64,
    total: u64,
) -> (IoSchedulerAdapter<ProbeProvider>, Arc<Counts>) {
    let counts = Arc::new(Counts::default());
    let adapter = IoSchedulerAdapter::with_byte_limits(
        ProbeProvider {
            counts: Arc::clone(&counts),
            outcome,
        },
        1,
        4,
        1,
        IoByteLimits {
            max_request_bytes: single,
            max_total_bytes: total,
        },
    )
    .expect("probe adapter");
    (adapter, counts)
}

#[test]
fn buffered_success_errors_checksum_and_cancellation_hold_bytes_until_consumed() {
    // Success, start/poll errors, short reads, checksum, queued and active cancel.
    for case in 0..7 {
        let outcome = match case {
            1 => Outcome::StartError,
            2 => Outcome::PollError,
            3 => Outcome::ShortRead,
            _ => Outcome::Success,
        };
        let (adapter, counts) = probe(outcome, 2, 2);
        let first_request = if case == 4 {
            request(1).with_expected_checksum(0)
        } else {
            request(1)
        };
        let first = adapter.submit(first_request).expect("first");
        if case == 5 {
            assert!(adapter.cancel(first));
        }
        if case == 6 {
            assert!(matches!(next(&adapter), IoAdapterEvent::Chunk { .. }));
            assert_eq!(adapter.admitted_bytes(), Ok(1));
            assert!(adapter.cancel(first));
        }
        let second = adapter
            .submit(request(1).with_byte_offset(1))
            .expect("second");
        assert_eq!(adapter.admitted_bytes(), Ok(2));
        let mut completed = false;
        for _ in 0..8 {
            if let Poll::Ready(Ok(event)) = adapter.poll_for(&mut context(), Some(second)) {
                assert_eq!(event.request_id(), second);
                if matches!(event, IoAdapterEvent::Completed { .. }) {
                    completed = true;
                    break;
                }
                assert!(matches!(event, IoAdapterEvent::Chunk { .. }));
            }
        }
        assert!(completed, "unrelated request case {case}");
        assert_eq!(adapter.admitted_bytes(), Ok(1));
        assert_eq!(adapter.len(), 1);
        assert!(adapter.lock_state().expect("state").scheduler.is_empty());
        provider_error(adapter.submit(request(2)), "total limit 2");
        assert_eq!(
            counts.starts.load(Ordering::Relaxed),
            if case == 5 { 1 } else { 2 }
        );
        loop {
            let event = next(&adapter);
            assert_eq!(event.request_id(), first);
            match event {
                IoAdapterEvent::Chunk { .. } => assert_eq!(adapter.admitted_bytes(), Ok(1)),
                IoAdapterEvent::Completed { bytes, .. } => {
                    assert_eq!(case, 0);
                    assert_eq!(bytes, [7]);
                    break;
                }
                IoAdapterEvent::Failed { error, .. } => {
                    assert_ne!(case, 0);
                    match case {
                        1 | 2 => assert!(matches!(error, IoError::Provider { .. })),
                        3 => assert!(matches!(error, IoError::ShortRead { .. })),
                        4 => assert!(matches!(error, IoError::ChecksumMismatch { .. })),
                        5 | 6 => assert_eq!(error, IoError::Cancelled { request_id: first }),
                        _ => unreachable!(),
                    }
                    break;
                }
            }
        }
        assert_eq!(adapter.admitted_bytes(), Ok(0));
        assert!(adapter.is_empty());
        assert!(
            adapter.poll(&mut context()).is_pending(),
            "no duplicate terminal"
        );
        let replacement = adapter.submit(request(2)).expect("failure quota released");
        assert!(adapter.cancel(replacement));
        assert!(matches!(next(&adapter), IoAdapterEvent::Failed { .. }));
        assert_eq!(adapter.admitted_bytes(), Ok(0));
    }
}

#[test]
fn allocation_failure_cancels_drops_and_keeps_identity_and_quota_until_consumed() {
    let (adapter, counts) = probe(Outcome::Success, u64::MAX, u64::MAX);
    // This always exceeds Vec's representable capacity, without requesting OOM.
    let huge = adapter
        .submit(request(u64::MAX - 1))
        .expect("huge declaration");
    let small = adapter
        .submit(request(1).with_byte_offset(1))
        .expect("small declaration");
    assert_eq!(adapter.admitted_bytes(), Ok(u64::MAX));
    assert!(adapter.poll_for(&mut context(), Some(small)).is_pending());
    assert_eq!(counts.cancels.load(Ordering::Relaxed), 1);
    assert_eq!(
        counts.drops.load(Ordering::Relaxed),
        0,
        "cancellation not yet drained"
    );
    assert_eq!(adapter.admitted_bytes(), Ok(u64::MAX));
    assert!(adapter.poll_for(&mut context(), Some(small)).is_pending());
    assert_eq!(
        counts.drops.load(Ordering::Relaxed),
        1,
        "failed operation dropped before delivery"
    );
    assert!(matches!(
        adapter.poll_for(&mut context(), Some(small)),
        Poll::Ready(Ok(IoAdapterEvent::Chunk { .. }))
    ));
    assert!(matches!(
        adapter.poll_for(&mut context(), Some(small)),
        Poll::Ready(Ok(IoAdapterEvent::Completed { .. }))
    ));
    assert_eq!(adapter.admitted_bytes(), Ok(u64::MAX - 1));
    provider_error(adapter.submit(request(2)), "admission overflow");
    assert_eq!(counts.starts.load(Ordering::Relaxed), 2);
    match next(&adapter) {
        IoAdapterEvent::Failed {
            request_id,
            request: failed,
            error,
        } => {
            assert_eq!(request_id, huge);
            assert_eq!(failed, request(u64::MAX - 1));
            provider_error::<()>(Err(error), "allocation failed");
        }
        event => panic!("expected allocation failure, got {event:?}"),
    }
    assert_eq!(adapter.admitted_bytes(), Ok(0));
    assert!(adapter.is_empty());
    assert!(adapter.poll(&mut context()).is_pending());
    assert_eq!(
        adapter.read_to_completion(&request(1), CancellationToken::new(false)),
        Ok(vec![7])
    );
    assert_eq!(adapter.admitted_bytes(), Ok(0));
    assert_eq!(counts.cancels.load(Ordering::Relaxed), 1);
    assert_eq!(counts.drops.load(Ordering::Relaxed), 3);
}

#[test]
fn blocking_allocation_failure_wakes_and_releases_quota_for_next_reader() {
    let (adapter, counts) = probe(Outcome::Success, u64::MAX, u64::MAX);
    let adapter = Arc::new(adapter);
    let reader = Arc::clone(&adapter);
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || {
        sender
            .send(reader.read_to_completion(&request(u64::MAX), CancellationToken::new(false)))
            .expect("send allocation failure");
    });
    provider_error(
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("allocation failure must wake"),
        "allocation failed",
    );
    worker.join().expect("reader thread");
    assert_eq!(counts.cancels.load(Ordering::Relaxed), 1);
    assert_eq!(counts.drops.load(Ordering::Relaxed), 1);
    assert_eq!(adapter.admitted_bytes(), Ok(0));
    assert_eq!(
        adapter.read_to_completion(&request(1), CancellationToken::new(false)),
        Ok(vec![7])
    );
    assert_eq!(adapter.admitted_bytes(), Ok(0));
}

#[test]
fn simultaneous_admission_is_atomic_and_concurrent_cancellation_recovers_bytes() {
    let adapter = Arc::new(bounded(1, 4));
    let barrier = Arc::new(Barrier::new(16));
    let workers: Vec<_> = (0..16)
        .map(|_| {
            let adapter = Arc::clone(&adapter);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                adapter.submit(request(1))
            })
        })
        .collect();
    let mut ids = Vec::new();
    for worker in workers {
        match worker.join().expect("admission worker") {
            Ok(id) => ids.push(id),
            error => provider_error(error, "total limit 4"),
        }
    }
    assert_eq!(ids.len(), 4);
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 4);
    assert_eq!(adapter.admitted_bytes(), Ok(4));
    let workers: Vec<_> = ids
        .into_iter()
        .map(|id| {
            let adapter = Arc::clone(&adapter);
            thread::spawn(move || assert!(adapter.cancel(id)))
        })
        .collect();
    for worker in workers {
        worker.join().expect("cancellation worker");
    }
    assert_eq!(adapter.admitted_bytes(), Ok(4));
    for remaining in (0..4).rev() {
        assert!(matches!(next(&adapter), IoAdapterEvent::Failed { .. }));
        assert_eq!(adapter.admitted_bytes(), Ok(remaining));
    }
    assert!(adapter.is_empty());
    assert!(adapter.poll(&mut context()).is_pending());
    adapter.submit(request(1)).expect("quota reusable");
}
