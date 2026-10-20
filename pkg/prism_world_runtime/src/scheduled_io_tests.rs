use crate::scheduled_io::ScheduledCellIo;
use crate::{CellIo, CellReadRequest, IoByteLimits, IoCompletion, RequestId, SubmitError};
use core::task::{Context, Poll, Waker};
use prism_world_io::{
    checksum, ChunkedRead, ConfigError, IoError, ReadChunk, ReadProvider, ReadRequest,
    RequestId as IoRequestId, ThreadedFileConfig, ThreadedFileProvider,
};
use prism_world_stream::{AabbI64, CellManifest, GlobalPos, SectorId};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::task::Wake;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(5);

fn request(offset: u64, length: u64) -> CellReadRequest {
    CellReadRequest::new(
        CellManifest {
            sector: SectorId::new(0, offset as i64, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
                .expect("valid bounds"),
            scene_key: 7,
            content_version: 3,
            derived_hash: 9,
            io_bytes: length,
            data_layers: prism_world_stream::DataLayerMask::EMPTY,
        },
        11,
    )
    .with_byte_offset(offset)
}

struct Signal {
    sender: SyncSender<()>,
    wakes: AtomicUsize,
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.wakes.fetch_add(1, Ordering::Relaxed);
        let _ = self.sender.try_send(());
    }
}

fn signal() -> (Arc<Signal>, Waker, Receiver<()>) {
    let (sender, receiver) = mpsc::sync_channel(1);
    let signal = Arc::new(Signal {
        sender,
        wakes: AtomicUsize::new(0),
    });
    let waker = Waker::from(Arc::clone(&signal));
    (signal, waker, receiver)
}

fn next(io: &mut impl CellIo<Error = IoError>) -> IoCompletion<IoError> {
    let (_, waker, receiver) = signal();
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match io.poll_completion(&mut cx) {
            Poll::Ready(Ok(completion)) => return completion,
            Poll::Ready(Err(error)) => panic!("internal IO error: {error}"),
            Poll::Pending => receiver
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("pending IO must wake its caller"),
        }
    }
}

struct Probe {
    released: AtomicBool,
    starts: AtomicUsize,
    polls: AtomicUsize,
    cancellations: AtomicUsize,
    waker: Mutex<Option<Waker>>,
}

impl Probe {
    fn release(&self) {
        self.released.store(true, Ordering::Release);
        let waker = self.waker.lock().expect("probe lock").take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct FakeProvider(Arc<Probe>);

fn provider(released: bool) -> (FakeProvider, Arc<Probe>) {
    let probe = Arc::new(Probe {
        released: AtomicBool::new(released),
        starts: AtomicUsize::new(0),
        polls: AtomicUsize::new(0),
        cancellations: AtomicUsize::new(0),
        waker: Mutex::new(None),
    });
    (FakeProvider(Arc::clone(&probe)), probe)
}

struct FakeOperation {
    probe: Arc<Probe>,
    gated: bool,
    offset: u64,
    remaining: u64,
    chunk_size: usize,
}

impl ReadProvider for FakeProvider {
    type Operation = FakeOperation;

    fn start(&self, request: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        self.0.starts.fetch_add(1, Ordering::Relaxed);
        Ok(FakeOperation {
            probe: Arc::clone(&self.0),
            gated: request.offset() == 0,
            offset: request.offset(),
            remaining: request.length(),
            chunk_size: request.chunk_size(),
        })
    }
}

impl ChunkedRead for FakeOperation {
    fn poll_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        self.probe.polls.fetch_add(1, Ordering::Relaxed);
        if self.gated && !self.probe.released.load(Ordering::Acquire) {
            *self.probe.waker.lock().expect("probe lock") = Some(cx.waker().clone());
            if !self.probe.released.load(Ordering::Acquire) {
                return Poll::Pending;
            }
        }
        if self.remaining == 0 {
            return Poll::Ready(Ok(None));
        }
        let length = self.remaining.min(self.chunk_size as u64);
        let offset = self.offset;
        let end = offset + length;
        let bytes = (offset..end).map(|index| index as u8).collect();
        self.offset = end;
        self.remaining -= length;
        Poll::Ready(Ok(Some(ReadChunk::new(offset, bytes))))
    }

    fn cancel(&mut self) {
        self.probe.cancellations.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn pending_provider_returns_promptly_and_wakes_the_cell_io_caller() {
    let (provider, probe) = provider(false);
    let mut io = ScheduledCellIo::new(provider, 1, 1, 2).expect("backend");
    let bytes: Vec<_> = (0..5).collect();
    let request = request(0, 5).with_expected_checksum(checksum(&bytes));
    let (_, waker, receiver) = signal();
    let mut cx = Context::from_waker(&waker);
    let backend: &mut dyn CellIo<Error = IoError> = &mut io;
    let id = backend.submit(request).expect("submit");

    assert!(backend.poll_completion(&mut cx).is_pending());
    assert_eq!(probe.polls.load(Ordering::Relaxed), 1);
    assert_eq!(backend.in_flight(), 1);
    assert!(
        receiver.try_recv().is_err(),
        "no wake before provider release"
    );
    probe.release();
    receiver.recv_timeout(TIMEOUT).expect("provider wake");
    match backend.poll_completion(&mut cx) {
        Poll::Ready(Ok(IoCompletion::Completed {
            request_id,
            request: completed_request,
            bytes: actual,
        })) => {
            assert_eq!(request_id, id);
            assert_eq!(completed_request, request);
            assert_eq!(actual, bytes);
        }
        other => panic!("expected completion, got {other:?}"),
    }
    assert_eq!(backend.in_flight(), 0);
    assert!(!backend.cancel(id));
}

#[test]
fn continuously_ready_chunks_yield_with_a_wake_and_retain_admission() {
    let (provider, probe) = provider(true);
    let mut io = ScheduledCellIo::new(provider, 1, 1, 1).expect("backend");
    let id = io.submit(request(0, 64)).expect("submit");
    let (signal, waker, receiver) = signal();
    let mut cx = Context::from_waker(&waker);

    assert!(io.poll_completion(&mut cx).is_pending());
    let polls = probe.polls.load(Ordering::Relaxed);
    assert!(polls > 0 && polls <= 32, "unbounded chunk drain: {polls}");
    assert!(signal.wakes.load(Ordering::Relaxed) > 0);
    receiver.recv_timeout(TIMEOUT).expect("cooperative wake");
    assert_eq!(io.in_flight(), 1);
    assert!(matches!(
        io.submit(request(100, 1)),
        Err(SubmitError::Backpressure { capacity: 1 })
    ));

    match next(&mut io) {
        IoCompletion::Completed {
            request_id, bytes, ..
        } => {
            assert_eq!(request_id, id);
            assert_eq!(bytes, (0..64).collect::<Vec<u8>>());
        }
        other => panic!("expected completed aggregate, got {other:?}"),
    }
    assert_eq!(probe.polls.load(Ordering::Relaxed), 65);
    assert_eq!(io.in_flight(), 0);
    assert!(io.submit(request(100, 1)).is_ok());
}

#[test]
fn pending_read_does_not_starve_another_active_cell() {
    let (provider, probe) = provider(false);
    let mut io = ScheduledCellIo::new(provider, 2, 2, 1).expect("backend");
    let slow = io.submit(request(0, 3)).expect("slow submit");
    let fast = io.submit(request(100, 1)).expect("fast submit");

    assert!(matches!(
        next(&mut io),
        IoCompletion::Completed { request_id, bytes, .. }
            if request_id == fast && bytes == [100]
    ));
    assert_eq!(probe.starts.load(Ordering::Relaxed), 2);
    assert_eq!(io.in_flight(), 1);
    probe.release();
    assert!(matches!(
        next(&mut io),
        IoCompletion::Completed { request_id, bytes, .. }
            if request_id == slow && bytes == [0, 1, 2]
    ));
    assert_eq!(io.in_flight(), 0);
}

#[test]
fn queued_cancellation_keeps_capacity_until_consumed_and_does_not_reuse_ids() {
    let (provider, probe) = provider(false);
    let mut io = ScheduledCellIo::new(provider, 1, 1, 1).expect("backend");
    let original = request(0, 1);
    let first = io.submit(original).expect("submit");
    assert!(io.cancel(first));
    assert!(!io.cancel(first));
    assert_eq!(io.in_flight(), 1);
    assert!(matches!(
        io.submit(request(2, 1)),
        Err(SubmitError::Backpressure { capacity: 1 })
    ));
    assert!(matches!(
        next(&mut io),
        IoCompletion::Cancelled { request_id, request }
            if request_id == first && request == original
    ));
    assert_eq!(probe.starts.load(Ordering::Relaxed), 0);
    assert_eq!(io.in_flight(), 0);

    let second = io.submit(request(2, 1)).expect("released capacity");
    assert!(second.get() > first.get());
    assert!(!io.cancel(first));
    assert!(!io.cancel(RequestId(u64::MAX)));
    assert!(matches!(
        next(&mut io),
        IoCompletion::Completed { request_id, .. } if request_id == second
    ));
}

#[test]
fn active_cancellation_wakes_pending_poll_and_reports_exactly_one_terminal() {
    let (provider, probe) = provider(false);
    let mut io = ScheduledCellIo::new(provider, 1, 1, 1).expect("backend");
    let id = io.submit(request(0, 1)).expect("submit");
    let (_, waker, receiver) = signal();
    let mut cx = Context::from_waker(&waker);
    assert!(io.poll_completion(&mut cx).is_pending());
    assert!(io.cancel(id));
    receiver.recv_timeout(TIMEOUT).expect("cancel wake");
    assert_eq!(probe.cancellations.load(Ordering::Relaxed), 1);
    assert_eq!(io.in_flight(), 1);
    assert!(matches!(
        io.submit(request(10, 1)),
        Err(SubmitError::Backpressure { capacity: 1 })
    ));
    assert!(matches!(
        io.poll_completion(&mut cx),
        Poll::Ready(Ok(IoCompletion::Cancelled { request_id, .. })) if request_id == id
    ));
    assert_eq!(io.in_flight(), 0);
    assert!(!io.cancel(id));
    assert!(io.poll_completion(&mut cx).is_pending());
    assert_eq!(probe.polls.load(Ordering::Relaxed), 1);
}

#[test]
fn declared_byte_limits_reject_without_ids_and_release_only_on_terminal_poll() {
    let (provider, _) = provider(true);
    let mut io = ScheduledCellIo::with_byte_limits(
        provider,
        2,
        4,
        1,
        IoByteLimits {
            max_request_bytes: 3,
            max_total_bytes: 4,
        },
    )
    .expect("backend");
    assert!(matches!(
        io.submit(request(1, 4)),
        Err(SubmitError::Provider { message }) if message.contains("per-request limit")
    ));
    let first = io.submit(request(1, 3)).expect("single limit");
    assert_eq!(first.get(), 1);
    assert!(matches!(
        io.submit(request(4, 2)),
        Err(SubmitError::Provider { message }) if message.contains("total limit")
    ));
    let second = io.submit(request(4, 1)).expect("total limit");
    assert_eq!(second.get(), first.get() + 1);
    assert!(io.cancel(first));
    assert_eq!(io.in_flight(), 2);
    assert!(matches!(
        io.submit(request(5, 1)),
        Err(SubmitError::Provider { message }) if message.contains("total limit")
    ));
    assert!(matches!(
        next(&mut io),
        IoCompletion::Cancelled { request_id, .. } if request_id == first
    ));
    let third = io.submit(request(5, 3)).expect("released byte quota");
    assert!(third.get() > second.get());
    assert_eq!(io.in_flight(), 2);
    for _ in 0..2 {
        assert!(matches!(next(&mut io), IoCompletion::Completed { .. }));
    }
    assert_eq!(io.in_flight(), 0);
}

#[test]
fn invalid_range_is_a_submission_failure_and_checksum_failure_is_a_completion() {
    let (provider, _) = provider(true);
    let mut io = ScheduledCellIo::new(provider, 1, 1, 2).expect("backend");
    assert!(matches!(
        io.submit(request(u64::MAX, 1)),
        Err(SubmitError::Provider { message }) if message.contains("overflows")
    ));
    assert_eq!(io.in_flight(), 0);
    let id = io
        .submit(request(1, 3).with_expected_checksum(0))
        .expect("submit bad checksum");
    assert_eq!(id.get(), 1);
    assert!(matches!(
        next(&mut io),
        IoCompletion::Failed {
            request_id,
            error: IoError::ChecksumMismatch { .. },
            ..
        } if request_id == id
    ));
    assert_eq!(io.in_flight(), 0);
    assert!(io.submit(request(1, 3)).is_ok());
}

struct FaultProvider {
    panic_on_poll: bool,
}

struct FaultOperation;

impl ReadProvider for FaultProvider {
    type Operation = FaultOperation;

    fn start(&self, _: &ReadRequest, _: IoRequestId) -> Result<Self::Operation, IoError> {
        if self.panic_on_poll {
            Ok(FaultOperation)
        } else {
            Err(IoError::Provider {
                message: "start rejected".into(),
            })
        }
    }
}

impl ChunkedRead for FaultOperation {
    fn poll_chunk(&mut self, _: &mut Context<'_>) -> Poll<Result<Option<ReadChunk>, IoError>> {
        panic!("deliberate provider panic to poison the adapter lock")
    }

    fn cancel(&mut self) {}
}

#[test]
fn provider_start_error_keeps_its_request_identity_and_releases_admission() {
    let mut io = ScheduledCellIo::new(
        FaultProvider {
            panic_on_poll: false,
        },
        1,
        1,
        1,
    )
    .expect("backend");
    let original = request(0, 1);
    let id = io.submit(original).expect("submit");
    assert!(matches!(
        next(&mut io),
        IoCompletion::Failed {
            request_id,
            request,
            error: IoError::Provider { message },
        } if request_id == id && request == original && message == "start rejected"
    ));
    assert_eq!(io.in_flight(), 0);
    assert!(io.submit(original).is_ok());
}

#[cfg(panic = "unwind")]
#[test]
fn internal_adapter_error_propagates_without_fabricating_a_cell_completion() {
    let mut io = ScheduledCellIo::new(
        FaultProvider {
            panic_on_poll: true,
        },
        1,
        1,
        1,
    )
    .expect("backend");
    io.submit(request(0, 1)).expect("submit");
    let (_, waker, _) = signal();
    let mut cx = Context::from_waker(&waker);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        io.poll_completion(&mut cx)
    }))
    .is_err());
    assert!(matches!(
        io.poll_completion(&mut cx),
        Poll::Ready(Err(IoError::Provider { message })) if message.contains("poisoned")
    ));
    assert_eq!(io.in_flight(), 1);
    assert!(matches!(
        io.submit(request(1, 1)),
        Err(SubmitError::Provider { message }) if message.contains("poisoned")
    ));
}

#[test]
fn constructors_preserve_invalid_scheduler_configuration_errors() {
    let (p, _) = provider(true);
    assert!(matches!(
        ScheduledCellIo::new(p, 0, 1, 1),
        Err(ConfigError::ZeroInFlight)
    ));
    let (p, _) = provider(true);
    assert!(matches!(
        ScheduledCellIo::new(p, 1, 0, 1),
        Err(ConfigError::ZeroCapacity)
    ));
    let (p, _) = provider(true);
    assert!(matches!(
        ScheduledCellIo::new(p, 1, 1, 0),
        Err(ConfigError::ZeroChunkSize)
    ));
}

struct TempFile(PathBuf);

impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-scheduled-cell-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create unique test file");
        file.write_all(bytes).expect("write test file");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).expect("remove test file");
    }
}

#[test]
fn threaded_file_provider_completes_resolved_ranges_and_checksum_failures() {
    let source = TempFile::new(b"prefix:abcde:tail");
    let provider = ThreadedFileProvider::new(
        source.0.clone(),
        ThreadedFileConfig {
            workers: 1,
            capacity: 2,
            max_chunk_size: 2,
        },
    )
    .expect("threaded provider");
    let mut io = ScheduledCellIo::with_byte_limits(
        provider,
        2,
        2,
        2,
        IoByteLimits {
            max_request_bytes: 5,
            max_total_bytes: 9,
        },
    )
    .expect("backend");
    let resolved = request(7, 5).with_expected_checksum(checksum(b"abcde"));
    let good = io.submit(resolved).expect("resolved read");
    let bad = io
        .submit(request(13, 4).with_expected_checksum(0))
        .expect("invalid checksum read");
    let mut completed = false;
    let mut failed = false;
    for _ in 0..2 {
        match next(&mut io) {
            IoCompletion::Completed {
                request_id,
                request,
                bytes,
            } => {
                assert_eq!(request_id, good);
                assert_eq!(request, resolved);
                assert_eq!(bytes, b"abcde");
                assert!(!completed);
                completed = true;
            }
            IoCompletion::Failed {
                request_id,
                error: IoError::ChecksumMismatch { .. },
                ..
            } => {
                assert_eq!(request_id, bad);
                assert!(!failed);
                failed = true;
            }
            other => panic!("unexpected file completion: {other:?}"),
        }
    }
    assert!(completed && failed);
    assert_eq!(io.in_flight(), 0);
}
