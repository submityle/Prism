use super::*;
use prism_world_io::{checksum, FileProvider, IoError};
use prism_world_stream::{AabbI64, GlobalPos};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

struct TempFile(PathBuf);

impl TempFile {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism_runtime_{}_{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create unique temporary file");
        file.write_all(bytes).expect("write temporary file");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        match fs::remove_file(&self.0) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("temporary file cleanup: {error}"),
        }
    }
}

fn manifest(x: i64, length: usize) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, x, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
            .expect("valid bounds"),
        scene_key: (7 + x) as u128,
        content_version: 2,
        derived_hash: 123,
        io_bytes: length as u64,
    }
}

fn cell_bytes(manifest: CellManifest, payload: &[u8]) -> Vec<u8> {
    let mut bytes = ManifestHeader::from_manifest(manifest).encode();
    bytes.extend_from_slice(payload);
    bytes
}

#[derive(Default)]
struct Host(BTreeMap<SectorId, Vec<u8>>);

impl SceneHost for Host {
    type Error = ();

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), ()> {
        assert!(self.0.insert(manifest.sector, payload.to_vec()).is_none());
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), ()> {
        self.0.remove(&sector);
        Ok(())
    }
}

#[test]
fn worker_archive_routes_validates_and_commits_without_blocking_frame_io() {
    use prism_world_io::ThreadedFileConfig;
    use std::sync::{mpsc, Arc};
    use std::task::Wake;
    use std::time::{Duration, Instant};

    struct Signal(mpsc::SyncSender<()>);
    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            let _ = self.0.try_send(());
        }
        fn wake_by_ref(self: &Arc<Self>) {
            let _ = self.0.try_send(());
        }
    }

    // One worker slot, multiple logical requests, and headers split over chunks.
    let payloads: [&[u8]; 3] = [b"opaque-one", b"checksum-rejected", b"header-rejected"];
    let manifests: Vec<_> = payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| manifest(index as i64, CELL_HEADER_SIZE + payload.len()))
        .collect();
    let mut archive = b"prefix".to_vec();
    let mut sources = Vec::new();
    for (index, &m) in manifests.iter().enumerate() {
        let mut header = m;
        if index == 2 {
            header.content_version += 1;
        }
        let bytes = cell_bytes(header, payloads[index]);
        let checksum = checksum(&bytes) ^ u64::from(index == 1);
        sources.push(
            CellReadRequest::new(m, 0)
                .with_byte_offset(archive.len() as u64)
                .with_expected_checksum(checksum),
        );
        archive.extend_from_slice(&bytes);
        archive.extend_from_slice(b"gap");
    }
    let file = TempFile::new(&archive);
    let mut io = ThreadedCellIo::threaded(
        &file.0,
        ThreadedFileConfig {
            workers: 1,
            capacity: 1,
            max_chunk_size: 7,
        },
        1,
        3,
        IoByteLimits {
            max_request_bytes: 128,
            max_total_bytes: 384,
        },
    )
    .unwrap();
    for source in sources {
        io.register(source).unwrap();
    }
    let mut coordinator = CellFrameCoordinator::with_io(io);
    let mut host = Host::default();
    let demands: Vec<_> = manifests
        .iter()
        .map(|m| DemandSource::new(m.sector, 0, 1, 10).unwrap())
        .collect();
    assert_eq!(
        coordinator
            .plan_frame(
                &demands,
                &manifests,
                &[],
                StreamBudget {
                    io_bytes: 384,
                    max_actions: 3
                },
                &mut host
            )
            .submitted,
        3
    );
    let (send, receive) = mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(Signal(send)));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(5);
    while coordinator.in_flight() != 0 {
        match coordinator.poll_io_with_context(&mut cx, &mut host) {
            Poll::Ready(result) => result.unwrap(),
            Poll::Pending => receive
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .expect("worker completion must wake frame driver"),
        }
    }
    assert_eq!(host.0.len(), 1);
    assert_eq!(host.0[&manifests[0].sector], payloads[0]);
    assert_eq!(coordinator.trace().replay(), Ok(1));
    for stage in [FailureStage::Provider, FailureStage::ManifestHeader] {
        assert!(coordinator
            .trace()
            .as_slice()
            .iter()
            .any(|event| matches!(event,
            TraceEvent::Failed { stage: actual, .. } if *actual == stage)));
    }
}

fn run_frame(provider: FileCellProvider, manifests: &[CellManifest]) -> (Host, Trace) {
    let mut coordinator = WorldFrameCoordinator::new(provider, manifests.len()).expect("runtime");
    let mut host = Host::default();
    let demands: Vec<_> = manifests
        .iter()
        .map(|m| DemandSource::new(m.sector, 0, 1, 10).expect("demand"))
        .collect();
    let report = coordinator.plan_frame(
        &demands,
        manifests,
        &[],
        StreamBudget {
            io_bytes: manifests.iter().map(|m| m.io_bytes).sum(),
            max_actions: manifests.len(),
        },
        &mut host,
    );
    assert_eq!(report.submitted, manifests.len());
    for _ in manifests {
        assert!(coordinator.poll_io(&mut host));
    }
    assert!(!coordinator.poll_io(&mut host));
    assert_eq!(coordinator.in_flight(), 0);
    (host, coordinator.trace().clone())
}

#[test]
fn packed_file_ranges_cross_chunk_headers_and_reach_scene_host() {
    let payloads: [&[u8]; 2] = [b"\0first\xffpayload", b"second\0scene"];
    let manifests = [
        manifest(0, CELL_HEADER_SIZE + payloads[0].len()),
        manifest(1, CELL_HEADER_SIZE + payloads[1].len()),
    ];
    let cells = [
        cell_bytes(manifests[0], payloads[0]),
        cell_bytes(manifests[1], payloads[1]),
    ];
    let mut archive = b"prefix".to_vec();
    let first_offset = archive.len() as u64;
    archive.extend_from_slice(&cells[0]);
    archive.extend_from_slice(b"gap");
    let second_offset = archive.len() as u64;
    archive.extend_from_slice(&cells[1]);
    archive.extend_from_slice(b"trailer");
    let file = TempFile::new(&archive);
    let mut provider = FileCellProvider::with_chunk_size(&file.0, 2, 2, 7).expect("provider");
    for (index, offset) in [first_offset, second_offset].into_iter().enumerate() {
        provider
            .register(
                CellReadRequest::new(manifests[index], 0)
                    .with_byte_offset(offset)
                    .with_expected_checksum(checksum(&cells[index])),
            )
            .expect("register");
    }
    let (host, trace) = run_frame(provider, &manifests);
    for (m, payload) in manifests.iter().zip(payloads) {
        assert_eq!(host.0[&m.sector], payload);
    }
    assert_eq!(
        trace
            .as_slice()
            .iter()
            .filter(|e| matches!(e, TraceEvent::Loaded { .. }))
            .count(),
        2
    );
    assert!(!trace
        .as_slice()
        .iter()
        .any(|e| matches!(e, TraceEvent::Failed { .. })));
}

fn next_event<P: prism_world_io::ReadProvider>(adapter: &IoSchedulerAdapter<P>) -> IoAdapterEvent {
    match adapter.poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(Ok(event)) => event,
        other => panic!("immediate provider must emit one event: {other:?}"),
    }
}

#[test]
fn bounded_file_admission_counts_header_and_delivers_only_admitted_payloads() {
    let payload = b"bounded scene";
    let m = manifest(0, CELL_HEADER_SIZE + payload.len());
    let bytes = cell_bytes(m, payload);
    let mut archive = b"prefix".to_vec();
    let offset = archive.len() as u64;
    archive.extend_from_slice(&bytes);
    let file = TempFile::new(&archive);
    for (single, total, admitted) in [
        (m.io_bytes - 1, m.io_bytes, false),
        (m.io_bytes, m.io_bytes - 1, false),
        (m.io_bytes, m.io_bytes, true),
    ] {
        let mut provider = FileCellProvider::with_byte_limits(
            &file.0,
            1,
            2,
            3,
            IoByteLimits {
                max_request_bytes: single,
                max_total_bytes: total,
            },
        )
        .expect("bounded file provider");
        let request = CellReadRequest::new(m, 1);
        provider
            .register(
                request
                    .with_byte_offset(offset)
                    .with_expected_checksum(checksum(&bytes)),
            )
            .expect("registration is independent of admission");
        assert_eq!(provider.scheduler().admitted_bytes(), Ok(0));
        let result = provider.read_chunk(&request, CancellationToken::new(false));
        if admitted {
            assert_eq!(result, Ok(bytes.clone()));
        } else {
            assert!(matches!(result, Err(IoError::Provider { .. })));
        }
        assert_eq!(provider.scheduler().admitted_bytes(), Ok(0));
        assert!(provider.scheduler().is_empty());
        let (host, trace) = run_frame(provider, &[m]);
        if admitted {
            assert_eq!(host.0[&m.sector], payload);
            assert!(trace
                .as_slice()
                .iter()
                .any(|e| matches!(e, TraceEvent::Loaded { .. })));
        } else {
            assert!(host.0.is_empty());
            assert!(trace.as_slice().iter().any(|e| matches!(
                e,
                TraceEvent::Failed {
                    stage: FailureStage::Provider,
                    ..
                }
            )));
        }
    }
}

#[test]
fn real_file_chunks_summary_checksum_backpressure_and_cancel() {
    let file = TempFile::new(b"prefix0123456789suffix");
    let adapter =
        IoSchedulerAdapter::with_chunk_size(FileProvider::new(&file.0), 1, 1, 3).expect("adapter");
    let request = CellReadRequest::new(manifest(0, 10), 4)
        .with_byte_offset(6)
        .with_expected_checksum(checksum(b"0123456789"));
    let id = adapter.submit(request).expect("submit");
    assert_eq!(
        adapter.submit(request),
        Err(IoError::Backpressure { capacity: 1 })
    );
    let mut offset = 6;
    let mut chunks = 0;
    loop {
        let event = next_event(&adapter);
        assert_eq!(event.request_id(), id);
        match event {
            IoAdapterEvent::Chunk { chunk, .. } => {
                assert_eq!(chunk.offset(), offset);
                assert!(chunk.bytes().len() <= 3);
                offset += chunk.bytes().len() as u64;
                chunks += 1;
            }
            IoAdapterEvent::Completed { bytes, summary, .. } => {
                assert_eq!(bytes, b"0123456789");
                assert_eq!(summary.offset(), 6);
                assert_eq!(summary.length(), 10);
                assert_eq!(summary.checksum(), checksum(&bytes));
                break;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!((offset, chunks), (16, 4));
    assert!(adapter.is_empty());
    let id = adapter.submit(request).expect("released capacity");
    assert!(adapter.cancel(id));
    assert!(matches!(next_event(&adapter), IoAdapterEvent::Failed {
        request_id, error: IoError::Cancelled { request_id: cancelled }, ..
    } if request_id == id && cancelled == id));
    assert!(adapter.is_empty());
}

#[test]
fn real_file_short_read_checksum_and_open_errors_keep_request_identity() {
    let file = TempFile::new(b"abcd");
    let adapter =
        IoSchedulerAdapter::with_chunk_size(FileProvider::new(&file.0), 2, 2, 2).expect("adapter");
    let checksum_id = adapter
        .submit(CellReadRequest::new(manifest(0, 4), 1).with_expected_checksum(0))
        .expect("checksum request");
    let short_id = adapter
        .submit(CellReadRequest::new(manifest(1, 6), 1))
        .expect("short request");
    let mut errors = BTreeMap::new();
    while !adapter.is_empty() {
        match next_event(&adapter) {
            IoAdapterEvent::Failed {
                request_id, error, ..
            } => {
                assert!(errors.insert(request_id, error).is_none());
            }
            IoAdapterEvent::Chunk { .. } => {}
            event => panic!("unexpected completion: {event:?}"),
        }
    }
    assert!(matches!(
        errors[&checksum_id],
        IoError::ChecksumMismatch { .. }
    ));
    assert!(matches!(errors[&short_id], IoError::ShortRead { .. }));
    fs::remove_file(&file.0).expect("remove owned file to test failed open");
    let id = adapter
        .submit(CellReadRequest::new(manifest(0, 1), 1))
        .expect("submit missing file");
    assert!(matches!(next_event(&adapter), IoAdapterEvent::Failed {
        request_id, error: IoError::Provider { .. }, ..
    } if request_id == id));
    assert_eq!(
        adapter.submit(CellReadRequest::new(manifest(0, 2), 1).with_byte_offset(u64::MAX)),
        Err(IoError::RangeOverflow)
    );
    assert!(adapter.is_empty());
}

#[test]
fn invalid_file_bytes_never_cross_scene_host_boundary() {
    let original = manifest(0, CELL_HEADER_SIZE + 5);
    let valid = cell_bytes(original, b"scene");
    for case in 0..7 {
        let mut m = original;
        let mut bytes = valid.clone();
        let stage = match case {
            0 => {
                bytes.pop();
                FailureStage::Provider
            }
            1 => FailureStage::Provider,
            2..=5 => {
                bytes[[0, 4, 20, 28][case - 2]] ^= 1;
                FailureStage::ManifestHeader
            }
            6 => {
                bytes.truncate(CELL_HEADER_SIZE - 1);
                m.io_bytes = bytes.len() as u64;
                FailureStage::ManifestHeader
            }
            _ => unreachable!(),
        };
        let file = TempFile::new(&bytes);
        let mut provider = FileCellProvider::with_chunk_size(&file.0, 1, 1, 3).expect("provider");
        provider
            .register(
                CellReadRequest::new(m, 0).with_expected_checksum(if case == 1 {
                    0
                } else {
                    checksum(&bytes)
                }),
            )
            .expect("register");
        let (host, trace) = run_frame(provider, &[m]);
        assert!(host.0.is_empty(), "case {case}");
        assert!(
            trace
                .as_slice()
                .iter()
                .any(|e| matches!(e, TraceEvent::Failed { stage: actual, .. } if *actual == stage)),
            "case {case}: {trace:?}"
        );
    }
}

#[test]
fn file_registration_preserves_range_priority_and_manifest_freshness() {
    let file = TempFile::new(b"unused");
    let mut provider = FileCellProvider::new(&file.0, 1, 1).expect("provider");
    let m = manifest(0, 4);
    let request = CellReadRequest::new(m, 99);
    assert!(provider.resolve(&request).is_err());
    let source = CellReadRequest::new(m, 1)
        .with_byte_offset(2)
        .with_expected_checksum(19);
    assert_eq!(provider.register(source), Ok(None));
    let resolved = provider.resolve(&request).expect("resolved").io_request();
    assert_eq!(resolved.offset(), 2);
    assert_eq!(resolved.expected_checksum(), Some(19));
    assert_eq!(resolved.priority(), 99);
    let mut stale = m;
    stale.content_version += 1;
    assert!(provider.resolve(&CellReadRequest::new(stale, 1)).is_err());
    assert_eq!(
        provider.register(source.with_byte_offset(u64::MAX)),
        Err(IoError::RangeOverflow)
    );
    assert_eq!(
        provider
            .resolve(&request)
            .expect("unchanged source")
            .io_request(),
        resolved
    );
    assert_eq!(provider.register(request), Ok(Some(source)));
}

#[test]
fn blocking_file_read_preserves_unrelated_failures_and_buffered_backpressure() {
    let file = TempFile::new(b"abcd");
    let adapter =
        IoSchedulerAdapter::with_chunk_size(FileProvider::new(&file.0), 2, 4, 2).expect("adapter");
    let request = CellReadRequest::new(manifest(0, 4), 1);
    let bad_checksum = adapter
        .submit(request.with_expected_checksum(0))
        .expect("bad checksum");
    let short = adapter
        .submit(CellReadRequest::new(manifest(1, 6), 1))
        .expect("short");
    let cancelled = adapter.submit(request).expect("cancelled");
    assert!(adapter.cancel(cancelled));
    assert_eq!(
        adapter.read_to_completion(&request, CancellationToken::new(false)),
        Ok(b"abcd".to_vec())
    );
    assert_eq!(adapter.len(), 3, "buffered terminal events retain capacity");
    let fourth = adapter.submit(request).expect("one slot remains");
    assert_eq!(
        adapter.submit(request),
        Err(IoError::Backpressure { capacity: 4 })
    );
    let mut failures = BTreeMap::new();
    while !adapter.is_empty() {
        match next_event(&adapter) {
            IoAdapterEvent::Failed {
                request_id, error, ..
            } => {
                assert!(failures.insert(request_id, error).is_none());
            }
            IoAdapterEvent::Completed {
                request_id, bytes, ..
            } => {
                assert_eq!(request_id, fourth);
                assert_eq!(bytes, b"abcd");
            }
            IoAdapterEvent::Chunk { .. } => {}
        }
    }
    assert_eq!(failures.len(), 3);
    assert!(matches!(
        failures[&bad_checksum],
        IoError::ChecksumMismatch { .. }
    ));
    assert!(matches!(failures[&short], IoError::ShortRead { .. }));
    assert_eq!(
        failures[&cancelled],
        IoError::Cancelled {
            request_id: cancelled
        }
    );
}

#[test]
fn buffered_success_retains_capacity_and_full_chunk_sequence_until_consumed() {
    let file = TempFile::new(b"abcdefgh");
    let adapter =
        IoSchedulerAdapter::with_chunk_size(FileProvider::new(&file.0), 1, 2, 2).expect("adapter");
    let first = CellReadRequest::new(manifest(0, 4), 2);
    let first_id = adapter.submit(first).expect("first");
    let second = CellReadRequest::new(manifest(1, 4), 1).with_byte_offset(4);
    assert_eq!(
        adapter.read_to_completion(&second, CancellationToken::new(false)),
        Ok(b"efgh".to_vec())
    );
    assert_eq!(adapter.len(), 1);
    let second_id = adapter.submit(second).expect("one free slot");
    assert_eq!(
        adapter.submit(first),
        Err(IoError::Backpressure { capacity: 2 })
    );
    for expected_offset in [0, 2] {
        match next_event(&adapter) {
            IoAdapterEvent::Chunk {
                request_id, chunk, ..
            } => {
                assert_eq!(request_id, first_id);
                assert_eq!(chunk.offset(), expected_offset);
            }
            other => panic!("buffered chunk order: {other:?}"),
        }
        assert_eq!(adapter.len(), 2, "chunks do not release capacity");
    }
    assert!(matches!(next_event(&adapter), IoAdapterEvent::Completed {
        request_id, bytes, ..
    } if request_id == first_id && bytes == b"abcd"));
    assert_eq!(adapter.len(), 1);
    assert!(adapter.cancel(second_id));
    assert!(matches!(next_event(&adapter), IoAdapterEvent::Failed {
        request_id, error: IoError::Cancelled { .. }, ..
    } if request_id == second_id));
    assert!(adapter.is_empty());
}
