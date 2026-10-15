use std::{
    fs,
    future::Future,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Barrier, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::Duration,
};

use super::*;

#[test]
fn checksum_matches_the_crc32_reference_vector() {
    assert_eq!(checksum32(b"123456789"), 0xcbf4_3926);
}

#[test]
fn ranges_and_validation_are_deterministic() {
    let range = ChunkRange::new(8, 4).expect("valid range");
    assert_eq!(range.end(), 12);
    assert_eq!(ChunkRange::new(u64::MAX, 1), Err(RangeError::Overflow));
    assert_eq!(
        validate_chunk(range, b"abc", None),
        Err(ChunkValidationError::ShortRead {
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(
        validate_chunk(range, b"abcd", Some(0)),
        Err(ChunkValidationError::ChecksumMismatch {
            expected: 0,
            actual: checksum32(b"abcd")
        })
    );
    let chunk =
        Chunk::from_bytes(range, b"abcd".to_vec(), Some(checksum32(b"abcd"))).expect("valid chunk");
    assert_eq!(chunk.bytes(), b"abcd");
}

#[test]
fn priority_queue_is_high_priority_first_and_fifo_for_ties() {
    let gate = Arc::new(Barrier::new(2));
    let order = Arc::new(Mutex::new(Vec::new()));
    let provider = OrderedProvider {
        gate: Arc::clone(&gate),
        order: Arc::clone(&order),
        first: AtomicBool::new(true),
    };
    let runtime = IoRuntime::new(
        provider,
        IoConfig {
            workers: 1,
            max_in_flight: 4,
        },
    )
    .expect("runtime");

    let first = runtime.submit(request(0, 1, 0)).expect("first submit");
    wait_until(|| runtime.in_flight() == 1);
    let low = runtime.submit(request(1, 1, 1)).expect("low submit");
    let high = runtime.submit(request(2, 9, 2)).expect("high submit");
    gate.wait();

    assert_eq!(block_on(first).expect("first result").bytes(), &[0]);
    assert_eq!(block_on(high).expect("high result").bytes(), &[2]);
    assert_eq!(block_on(low).expect("low result").bytes(), &[1]);
    assert_eq!(*order.lock().expect("order lock"), vec![0, 2, 1]);
}

#[test]
fn bounded_queue_reports_backpressure_without_blocking_submission() {
    let gate = Arc::new(Barrier::new(2));
    let provider = GateProvider {
        gate: Arc::clone(&gate),
        entered: Arc::new(AtomicBool::new(false)),
        calls: Arc::new(AtomicU64::new(0)),
    };
    let runtime = IoRuntime::new(
        provider,
        IoConfig {
            workers: 1,
            max_in_flight: 1,
        },
    )
    .expect("runtime");
    let first = runtime.submit(request(0, 0, 0)).expect("first submit");
    wait_until(|| runtime.in_flight() == 1);
    assert!(matches!(
        runtime.submit(request(1, 0, 1)),
        Err(SubmitError::Backpressure { capacity: 1 })
    ));
    gate.wait();
    assert!(block_on(first).is_ok());
}

#[test]
fn cancellation_wakes_a_polled_ticket_and_prevents_provider_work() {
    let gate = Arc::new(Barrier::new(2));
    let provider = GateProvider {
        gate: Arc::clone(&gate),
        entered: Arc::new(AtomicBool::new(false)),
        calls: Arc::new(AtomicU64::new(0)),
    };
    let calls = Arc::clone(&provider.calls);
    let runtime = IoRuntime::new(
        provider,
        IoConfig {
            workers: 1,
            max_in_flight: 2,
        },
    )
    .expect("runtime");
    let first = runtime.submit(request(0, 0, 1)).expect("first submit");
    wait_until(|| runtime.in_flight() == 1);
    let ticket = runtime.submit(request(1, 0, 1)).expect("second submit");
    ticket.cancel();
    gate.wait();
    assert!(block_on(first).is_ok());
    assert!(ticket.is_cancelled());
    assert!(matches!(block_on(ticket), Err(IoError::Cancelled)));
    assert_eq!(calls.load(Ordering::Acquire), 0);
}

#[test]
fn checksum_and_short_read_errors_are_reported() {
    let provider = StaticProvider {
        bytes: b"abcdefgh".to_vec(),
    };
    let runtime = IoRuntime::new(provider, IoConfig::default()).expect("runtime");
    let checksum_request = request(0, 0, 4).with_checksum(0);
    assert!(matches!(
        block_on(runtime.submit(checksum_request).expect("submit")),
        Err(IoError::ChecksumMismatch { .. })
    ));

    let short_request = request(6, 0, 4);
    assert!(matches!(
        block_on(runtime.submit(short_request).expect("submit")),
        Err(IoError::ShortRead {
            expected: 4,
            actual: 2
        })
    ));
}

#[test]
fn file_provider_reads_chunks_and_reports_short_reads() {
    let path = unique_temp_path();
    fs::write(&path, b"0123456789abcdef").expect("write test file");
    let runtime = IoRuntime::new(
        FileProvider::new(&path),
        IoConfig {
            workers: 2,
            max_in_flight: 8,
        },
    )
    .expect("runtime");

    let request = ReadRequest::new(ChunkRange::new(4, 6).expect("range"), 3)
        .with_checksum(checksum32(b"456789"));
    let chunk = block_on(runtime.submit(request).expect("submit")).expect("file read");
    assert_eq!(chunk.range(), ChunkRange::new(4, 6).expect("range"));
    assert_eq!(chunk.bytes(), b"456789");

    let short = ReadRequest::new(ChunkRange::new(14, 4).expect("range"), 0);
    assert!(matches!(
        block_on(runtime.submit(short).expect("submit")),
        Err(IoError::ShortRead {
            expected: 4,
            actual: 2
        })
    ));
    drop(runtime);
    fs::remove_file(path).expect("remove test file");
}

#[test]
fn async_reader_trait_submits_and_polls_a_request() {
    let runtime = IoRuntime::new(
        StaticProvider {
            bytes: b"abcd".to_vec(),
        },
        IoConfig::default(),
    )
    .expect("runtime");
    let request = request(0, 0, 4).with_checksum(checksum32(b"abcd"));
    let chunk = block_on(runtime.read_chunk(request)).expect("async read");
    assert_eq!(chunk.bytes(), b"abcd");
}

fn request(offset: u64, priority: u8, length: usize) -> ReadRequest {
    ReadRequest::new(
        ChunkRange::new(offset, length).expect("test range"),
        priority,
    )
}

fn wait_until(predicate: impl Fn() -> bool) {
    for _ in 0..1000 {
        if predicate() {
            return;
        }
        thread::yield_now();
    }
    panic!("condition did not become true");
}

fn block_on<F: Future>(future: F) -> F::Output {
    let parker = Arc::new(ThreadParker {
        thread: thread::current(),
    });
    let waker = Waker::from(Arc::clone(&parker));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park_timeout(Duration::from_millis(10)),
        }
    }
}

struct ThreadParker {
    thread: thread::Thread,
}

impl Wake for ThreadParker {
    fn wake(self: Arc<Self>) {
        self.thread.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.thread.unpark();
    }
}

struct StaticProvider {
    bytes: Vec<u8>,
}

impl ChunkProvider for StaticProvider {
    type Error = &'static str;

    fn read_chunk(
        &self,
        range: ChunkRange,
        _cancel: &CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        let start = usize::try_from(range.offset()).map_err(|_| "offset")?;
        let end = start.saturating_add(range.length()).min(self.bytes.len());
        Ok(self.bytes[start.min(self.bytes.len())..end].to_vec())
    }
}

struct CountingProvider {
    calls: Arc<AtomicU64>,
}

impl ChunkProvider for CountingProvider {
    type Error = &'static str;

    fn read_chunk(
        &self,
        range: ChunkRange,
        _cancel: &CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        Ok(vec![0; range.length()])
    }
}

struct GateProvider {
    gate: Arc<Barrier>,
    entered: Arc<AtomicBool>,
    calls: Arc<AtomicU64>,
}

impl ChunkProvider for GateProvider {
    type Error = &'static str;

    fn read_chunk(
        &self,
        range: ChunkRange,
        _cancel: &CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        if !self.entered.swap(true, Ordering::AcqRel) {
            self.gate.wait();
        }
        Ok(vec![0; range.length()])
    }
}

struct OrderedProvider {
    gate: Arc<Barrier>,
    order: Arc<Mutex<Vec<u64>>>,
    first: AtomicBool,
}

impl ChunkProvider for OrderedProvider {
    type Error = &'static str;

    fn read_chunk(
        &self,
        range: ChunkRange,
        _cancel: &CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        if self.first.swap(false, Ordering::AcqRel) {
            self.gate.wait();
        }
        self.order.lock().expect("order lock").push(range.offset());
        Ok(vec![range.offset() as u8; range.length()])
    }
}

fn unique_temp_path() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("prism-world-io-{}-{id}.bin", std::process::id()))
}
