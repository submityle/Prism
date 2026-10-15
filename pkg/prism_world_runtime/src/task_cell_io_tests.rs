use super::*;
use alloc::collections::BTreeMap;
use core::future::poll_fn;
use core::task::Waker;
use prism_tasks::{CancelToken, TaskPool};
use prism_world_io::{checksum, IoError, ThreadedFileConfig};
use prism_world_stream::{AabbI64, GlobalPos};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::task::Wake;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

fn request(x: i64) -> CellReadRequest {
    CellReadRequest::new(
        CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1)).unwrap(),
            scene_key: x as u128,
            content_version: 1,
            derived_hash: 2,
            io_bytes: CELL_HEADER_SIZE as u64,
        },
        10,
    )
}

struct ProbeIo {
    pending: BTreeMap<RequestId, (CellReadRequest, bool)>,
    next: u64,
    cancels: Arc<AtomicUsize>,
    ready: bool,
    late_success: bool,
    cancel_on_poll: Option<CancelToken>,
    fail_poll: bool,
    panic_cancel: bool,
}

impl ProbeIo {
    fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
            next: 1,
            cancels: Arc::new(AtomicUsize::new(0)),
            ready: false,
            late_success: false,
            cancel_on_poll: None,
            fail_poll: false,
            panic_cancel: false,
        }
    }
}

impl CellIo for ProbeIo {
    type Error = ();
    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        let id = RequestId(self.next);
        self.next += 1;
        self.pending.insert(id, (request, false));
        Ok(id)
    }
    fn cancel(&mut self, id: RequestId) -> bool {
        self.cancels.fetch_add(1, Ordering::Relaxed);
        assert!(!self.panic_cancel, "backend cancel panic");
        if self.late_success {
            return false;
        }
        self.pending.get_mut(&id).is_some_and(|(_, cancelled)| {
            let changed = !*cancelled;
            *cancelled = true;
            changed
        })
    }
    fn in_flight(&self) -> usize {
        self.pending.len()
    }
    fn poll_completion(&mut self, _: &mut Context<'_>) -> Poll<Result<IoCompletion<()>, ()>> {
        if let Some(token) = self.cancel_on_poll.take() {
            token.cancel();
        }
        if self.fail_poll {
            return Poll::Ready(Err(()));
        }
        let Some((&id, &(request, cancelled))) = self.pending.first_key_value() else {
            return Poll::Pending;
        };
        if !cancelled && !self.ready && !self.late_success {
            return Poll::Pending;
        }
        self.pending.remove(&id);
        Poll::Ready(Ok(if cancelled {
            IoCompletion::Cancelled {
                request_id: id,
                request,
            }
        } else {
            IoCompletion::Completed {
                request_id: id,
                request,
                bytes: ManifestHeader::from_manifest(request.manifest()).encode(),
            }
        }))
    }
}

struct CountWake(AtomicUsize);
impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn parent_cancellation_wakes_a_parked_task_without_any_provider_wake() {
    let pool = TaskPool::with_threads(1);
    let root = CancelToken::new();
    let mut io = TaskCellIo::new(ProbeIo::new(), &root).ok().unwrap();
    let id = io.submit(request(0)).unwrap();
    let (parked_tx, parked_rx) = mpsc::sync_channel(1);
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let task = pool.spawn_async(async move {
        let mut first = true;
        let terminal = poll_fn(|cx| {
            let result = io.poll_completion(cx);
            if first {
                assert!(result.is_pending());
                first = false;
                parked_tx.send(()).unwrap();
            }
            result
        })
        .await
        .unwrap();
        assert!(matches!(terminal, IoCompletion::Cancelled { request_id, .. } if request_id == id));
        assert_eq!(io.in_flight(), 0);
        assert_eq!(io.submit(request(1)), Err(SubmitError::Cancelled));
        done_tx.send(()).unwrap();
    });
    parked_rx
        .recv_timeout(TIMEOUT)
        .expect("task polled and parked");
    let (other_tx, other_rx) = mpsc::sync_channel(1);
    let other = pool.spawn_async(async move {
        other_tx.send(()).unwrap();
    });
    other_rx
        .recv_timeout(TIMEOUT)
        .expect("the only compute worker remains available");
    root.cancel();
    done_rx
        .recv_timeout(TIMEOUT)
        .expect("scope notification resumes parked IO task");
    pool.block_on(task);
    pool.block_on(other);
}

#[test]
fn cancellation_is_batched_and_admission_is_retained_until_terminals_are_consumed() {
    let root = CancelToken::new();
    let backend = ProbeIo::new();
    let cancels = backend.cancels.clone();
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    for x in 0..35 {
        io.submit(request(x)).unwrap();
    }
    root.cancel();
    let wake = Arc::new(CountWake(AtomicUsize::new(0)));
    let waker = Waker::from(wake.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(io.poll_completion(&mut cx).is_pending());
    assert_eq!(cancels.load(Ordering::Relaxed), 16);
    assert_eq!(io.in_flight(), 35);
    assert_eq!(wake.0.load(Ordering::Relaxed), 1);
    assert!(io.poll_completion(&mut cx).is_pending());
    assert_eq!(cancels.load(Ordering::Relaxed), 32);
    for left in (0..35).rev() {
        assert!(matches!(
            io.poll_completion(&mut cx),
            Poll::Ready(Ok(IoCompletion::Cancelled { .. }))
        ));
        assert_eq!(io.in_flight(), left);
    }
    assert_eq!(cancels.load(Ordering::Relaxed), 35);
    assert!(io.poll_completion(&mut cx).is_pending());
}

#[test]
fn cancellation_suppresses_success_even_when_backend_cannot_cancel_it() {
    let root = CancelToken::new();
    let mut backend = ProbeIo::new();
    backend.late_success = true;
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    let id = io.submit(request(0)).unwrap();
    root.cancel();
    let event = TaskPool::with_threads(0)
        .block_on(next_cell_completion(&mut io))
        .unwrap();
    assert_eq!(event.request_id(), id);
    assert_eq!(event.request(), request(0));
    assert!(matches!(event, IoCompletion::Cancelled { .. }));
}

#[test]
fn cancellation_during_backend_ready_poll_suppresses_the_payload() {
    let root = CancelToken::new();
    let mut backend = ProbeIo::new();
    backend.ready = true;
    backend.cancel_on_poll = Some(root.clone());
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    let id = io.submit(request(0)).unwrap();
    let event = TaskPool::with_threads(0)
        .block_on(next_cell_completion(&mut io))
        .unwrap();
    assert!(matches!(event, IoCompletion::Cancelled { request_id, .. } if request_id == id));
    assert_eq!(io.in_flight(), 0);
}

#[test]
fn individual_cancellation_leaves_scope_open_and_backend_errors_preserve_ownership() {
    let root = CancelToken::new();
    let mut io = TaskCellIo::new(ProbeIo::new(), &root).ok().unwrap();
    let id = io.submit(request(0)).unwrap();
    assert!(io.cancel(id));
    assert!(!io.cancel(id));
    let event = TaskPool::with_threads(0)
        .block_on(next_cell_completion(&mut io))
        .unwrap();
    assert_eq!(event.request_id(), id);
    assert!(!io.cancel_token().is_cancelled());
    assert!(io.submit(request(1)).is_ok());

    let mut backend = ProbeIo::new();
    backend.fail_poll = true;
    let cancels = backend.cancels.clone();
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    io.submit(request(2)).unwrap();
    assert!(TaskPool::with_threads(0)
        .block_on(next_cell_completion(&mut io))
        .is_err());
    assert_eq!(io.in_flight(), 1);
    drop(io);
    assert_eq!(cancels.load(Ordering::Relaxed), 1);
}

struct PanicWake;
impl Wake for PanicWake {
    fn wake(self: Arc<Self>) {
        panic!("notification panic");
    }
}

#[test]
fn drop_notifies_every_request_even_after_waker_and_backend_panics() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let root = CancelToken::new();
    let mut backend = ProbeIo::new();
    backend.panic_cancel = true;
    let cancels = backend.cancels.clone();
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    for x in 0..3 {
        io.submit(request(x)).unwrap();
    }
    let child = io.cancel_token().clone();
    let waker = Waker::from(Arc::new(PanicWake));
    let mut notified = Box::pin(child.cancelled());
    assert!(notified
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert!(catch_unwind(AssertUnwindSafe(|| drop(io))).is_err());
    assert!(child.is_cancelled());
    assert_eq!(cancels.load(Ordering::Relaxed), 3);
    assert!(!root.is_cancelled());
}

#[test]
fn drop_during_unwind_completes_cancellation_without_a_second_panic() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let root = CancelToken::new();
    let mut backend = ProbeIo::new();
    backend.panic_cancel = true;
    let cancels = backend.cancels.clone();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
        io.submit(request(0)).unwrap();
        io.submit(request(1)).unwrap();
        panic!("outer panic");
    }))
    .is_err());
    assert_eq!(cancels.load(Ordering::Relaxed), 2);
}

#[test]
fn child_drop_cancels_only_its_requests_and_scope() {
    let root = CancelToken::new();
    let sibling = root.child();
    let backend = ProbeIo::new();
    let cancels = backend.cancels.clone();
    let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
    let child = io.cancel_token().clone();
    io.submit(request(0)).unwrap();
    io.submit(request(1)).unwrap();
    drop(io);
    assert_eq!(cancels.load(Ordering::Relaxed), 2);
    assert!(child.is_cancelled());
    assert!(!root.is_cancelled());
    assert!(!sibling.is_cancelled());
}

#[test]
fn nonidle_backend_is_returned_and_already_cancelled_parent_rejects_submission() {
    let root = CancelToken::new();
    let mut backend = ProbeIo::new();
    let id = backend.submit(request(0)).unwrap();
    let backend = TaskCellIo::new(backend, &root).err().unwrap();
    assert!(backend.pending.contains_key(&id));
    root.cancel();
    let mut io = TaskCellIo::new(ProbeIo::new(), &root).ok().unwrap();
    assert!(io.cancel_token().is_cancelled());
    assert_eq!(io.submit(request(0)), Err(SubmitError::Cancelled));
    assert_eq!(io.in_flight(), 0);
}

struct Archive(PathBuf);
impl Drop for Archive {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

#[derive(Default)]
struct Host(Vec<(SectorId, Vec<u8>)>);
impl SceneHost for Host {
    type Error = ();
    fn load_cell(&mut self, m: &CellManifest, payload: &[u8]) -> Result<(), ()> {
        self.0.push((m.sector, payload.to_vec()));
        Ok(())
    }
    fn evict_cell(&mut self, _: SectorId) -> Result<(), ()> {
        Ok(())
    }
}

#[test]
fn task_pool_drives_archive_checksum_header_and_scene_host_on_single_and_multi_worker_pools() {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "prism-task-cell-{}-{}.bin",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let archive = Archive(path);
    let mut sources = Vec::new();
    let mut offset = 0;
    for x in 0..3 {
        let mut m = request(x).manifest();
        m.io_bytes += 512;
        let mut bytes = ManifestHeader::from_manifest(m).encode();
        bytes.resize(m.io_bytes as usize, x as u8 + 1);
        file.write_all(&bytes).unwrap();
        sources.push(
            CellReadRequest::new(m, 10)
                .with_byte_offset(offset)
                .with_expected_checksum(checksum(&bytes)),
        );
        offset += m.io_bytes;
    }
    drop(file);
    for threads in [0, 1, 2] {
        let pool = TaskPool::with_threads(threads);
        let root = CancelToken::new();
        let mut backend = ThreadedCellIo::threaded(
            &archive.0,
            ThreadedFileConfig {
                workers: 1,
                capacity: 1,
                max_chunk_size: 7,
            },
            1,
            3,
            IoByteLimits {
                max_request_bytes: 1024,
                max_total_bytes: 3072,
            },
        )
        .unwrap();
        for source in &sources {
            backend.register(*source).unwrap();
        }
        let io = TaskCellIo::new(backend, &root).ok().unwrap();
        let manifests: Vec<_> = sources.iter().map(|s| s.manifest()).collect();
        let demands: Vec<_> = manifests
            .iter()
            .map(|m| DemandSource::new(m.sector, 0, 1, 10).unwrap())
            .collect();
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let task = pool.spawn_async(async move {
            let mut coordinator = CellFrameCoordinator::with_io(io);
            let mut host = Host::default();
            let report = coordinator.plan_frame(
                &demands,
                &manifests,
                &[],
                StreamBudget {
                    io_bytes: 3072,
                    max_actions: 3,
                },
                &mut host,
            );
            assert_eq!(report.submitted, 3);
            while coordinator.in_flight() != 0 {
                poll_fn(|cx| coordinator.poll_io_with_context(cx, &mut host)).await?;
            }
            host.0.sort_by_key(|(sector, _)| *sector);
            assert_eq!(host.0.len(), 3);
            for (x, (sector, payload)) in host.0.iter().enumerate() {
                assert_eq!(*sector, SectorId::new(0, x as i64, 0, 0));
                assert_eq!(*payload, vec![x as u8 + 1; 512]);
            }
            done_tx.send(()).unwrap();
            Ok::<_, IoError>(())
        });
        if threads != 0 {
            done_rx
                .recv_timeout(TIMEOUT)
                .expect("real file task completes");
        }
        pool.block_on(task).unwrap();
        assert!(!root.is_cancelled());
    }
}
