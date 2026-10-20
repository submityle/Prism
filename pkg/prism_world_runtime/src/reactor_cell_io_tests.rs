use super::*;
use core::future::poll_fn;
use prism_tasks::{CancelToken, IoReactor, NamedThreads, NamedThreadsConfig, TaskPool};
use prism_world_io::{checksum, IoError, ReactorFileConfig};
use prism_world_stream::{AabbI64, GlobalPos};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::Waker,
};

// Host owns the final Arc, and declaration order drains workers first.
struct Services {
    workers: NamedThreads,
    reactor: Arc<IoReactor>,
}
impl Services {
    fn new() -> Option<Arc<Self>> {
        if !IoReactor::supported() {
            return None;
        }
        Some(Arc::new(Self {
            workers: NamedThreads::with_config(NamedThreadsConfig {
                io_threads: 2,
                async_compute_threads: 1,
            }),
            reactor: Arc::new(IoReactor::new().unwrap()),
        }))
    }
}

struct Archive(PathBuf);
impl Archive {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-native-cell-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
}
impl Drop for Archive {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

fn manifest(x: i64, payload: usize) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, x, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1)).unwrap(),
        scene_key: x as u128,
        content_version: 2,
        derived_hash: 17,
        io_bytes: (CELL_HEADER_SIZE + payload) as u64,
        data_layers: prism_world_stream::DataLayerMask::EMPTY,
    }
}

fn limits() -> IoByteLimits {
    IoByteLimits {
        max_request_bytes: 4096,
        max_total_bytes: 16384,
    }
}
fn native() -> ReactorFileConfig {
    ReactorFileConfig {
        capacity: 1,
        max_chunk_size: 7,
        max_request_bytes: 4096,
        max_total_bytes: 4096,
    }
}

#[derive(Default)]
struct Host(BTreeMap<SectorId, Vec<u8>>);
impl SceneHost for Host {
    type Error = ();
    fn load_cell(&mut self, manifest: &CellManifest, bytes: &[u8]) -> Result<(), ()> {
        assert!(self.0.insert(manifest.sector, bytes.to_vec()).is_none());
        Ok(())
    }
    fn evict_cell(&mut self, sector: SectorId) -> Result<(), ()> {
        self.0.remove(&sector);
        Ok(())
    }
}

#[test]
fn native_archive_reaches_scene_host_on_zero_one_two_compute_workers_and_rejects_bad_content() {
    let Some(services) = Services::new() else {
        return;
    };
    let payloads: [&[u8]; 3] = [b"valid opaque payload", b"bad checksum", b"bad manifest"];
    let manifests: Vec<_> = payloads
        .iter()
        .enumerate()
        .map(|(i, p)| manifest(i as i64, p.len()))
        .collect();
    let mut archive = b"archive prefix".to_vec();
    let mut sources = Vec::new();
    for (index, &manifest) in manifests.iter().enumerate() {
        let mut header = manifest;
        if index == 2 {
            header.content_version += 1;
        }
        let mut bytes = ManifestHeader::from_manifest(header).encode();
        bytes.extend_from_slice(payloads[index]);
        sources.push(
            CellReadRequest::new(manifest, 10)
                .with_byte_offset(archive.len() as u64)
                .with_expected_checksum(checksum(&bytes) ^ u64::from(index == 1)),
        );
        archive.extend_from_slice(&bytes);
        archive.extend_from_slice(b"gap");
    }
    let archive = Archive::new(&archive);
    for threads in [0, 1, 2] {
        let pool = TaskPool::with_threads(threads);
        let task_services = Arc::clone(&services);
        let sources = sources.clone();
        let manifests = manifests.clone();
        let path = archive.0.clone();
        let task = pool.spawn_async(async move {
            let root = CancelToken::new();
            let mut backend = ReactorCellIo::reactor(
                path,
                &task_services.reactor,
                &task_services.workers,
                native(),
                1,
                3,
                limits(),
            )
            .unwrap();
            for source in sources {
                backend.register(source).unwrap();
            }
            let io = TaskCellIo::new(backend, &root).ok().unwrap();
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
                            io_bytes: 16384,
                            max_actions: 3
                        },
                        &mut host
                    )
                    .submitted,
                3
            );
            while coordinator.in_flight() != 0 {
                poll_fn(|cx| coordinator.poll_io_with_context(cx, &mut host))
                    .await
                    .unwrap();
            }
            assert_eq!(host.0.len(), 1);
            assert_eq!(host.0[&manifests[0].sector], payloads[0]);
            assert_eq!(coordinator.trace().replay(), Ok(1));
            for stage in [FailureStage::Provider, FailureStage::ManifestHeader] {
                assert!(coordinator.trace().as_slice().iter().any(|event|
                    matches!(event, TraceEvent::Failed { stage: actual, .. } if *actual == stage)));
            }
            assert!(!root.is_cancelled());
        });
        pool.block_on(task);
        assert_eq!(
            Arc::strong_count(&services),
            1,
            "compute task released its service clone"
        );
    }
}

#[test]
fn native_runtime_cancel_replacement_and_parent_scope_do_not_deliver_late_payloads() {
    let Some(services) = Services::new() else {
        return;
    };
    let manifest = manifest(0, 512);
    let mut bytes = ManifestHeader::from_manifest(manifest).encode();
    bytes.resize(manifest.io_bytes as usize, 37);
    let source = CellReadRequest::new(manifest, 10).with_expected_checksum(checksum(&bytes));
    let archive = Archive::new(&bytes);
    let pool = TaskPool::with_threads(0);
    pool.block_on(async {
        let mut backend = ReactorCellIo::reactor(
            &archive.0,
            &services.reactor,
            &services.workers,
            native(),
            1,
            2,
            limits(),
        )
        .unwrap();
        backend.register(source).unwrap();
        let root = CancelToken::new();
        let mut io = TaskCellIo::new(backend, &root).ok().unwrap();
        for _ in 0..32 {
            let old = io.submit(source).unwrap();
            // One scheduler poll cannot finish a 556-byte stream of 7-byte chunks.
            assert!(io
                .poll_completion(&mut Context::from_waker(Waker::noop()))
                .is_pending());
            assert!(io.cancel(old));
            assert!(matches!(next_cell_completion(&mut io).await.unwrap(),
                IoCompletion::Cancelled { request_id, .. } if request_id == old));
            let next = io.submit(source).unwrap();
            match next_cell_completion(&mut io).await.unwrap() {
                IoCompletion::Completed {
                    request_id,
                    bytes: actual,
                    ..
                } => {
                    assert_eq!(request_id, next);
                    assert_eq!(actual, bytes);
                }
                other => panic!("native replacement failed: {other:?}"),
            }
        }
        let old = io.submit(source).unwrap();
        assert!(io
            .poll_completion(&mut Context::from_waker(Waker::noop()))
            .is_pending());
        root.cancel();
        assert!(matches!(next_cell_completion(&mut io).await.unwrap(),
            IoCompletion::Cancelled { request_id, .. } if request_id == old));
        assert_eq!(io.in_flight(), 0);
        assert_eq!(io.submit(source), Err(SubmitError::Cancelled));
    });
}

#[test]
fn native_runtime_routing_is_checked_before_admission_and_errors_retire_once() {
    let Some(services) = Services::new() else {
        return;
    };
    let archive = Archive::new(b"nothing");
    let source = CellReadRequest::new(manifest(0, 1), 10);
    let mut io = ReactorCellIo::reactor(
        archive.0.with_extension("missing"),
        &services.reactor,
        &services.workers,
        native(),
        1,
        1,
        limits(),
    )
    .unwrap();
    assert!(matches!(
        io.submit(source),
        Err(SubmitError::Provider { .. })
    ));
    io.register(source).unwrap();
    let mut stale = source.manifest();
    stale.content_version += 1;
    assert!(matches!(
        io.submit(CellReadRequest::new(stale, 1)),
        Err(SubmitError::Provider { .. })
    ));
    assert_eq!(io.in_flight(), 0);
    let id = io.submit(source).unwrap();
    let completion = TaskPool::with_threads(0)
        .block_on(next_cell_completion(&mut io))
        .unwrap();
    assert!(
        matches!(completion, IoCompletion::Failed { request_id, error: IoError::Provider { .. }, .. } if request_id == id)
    );
    assert_eq!(io.in_flight(), 0);
    assert!(!io.cancel(id));
}

#[test]
fn native_runtime_factory_rejects_invalid_limits_without_opening_files() {
    let Some(services) = Services::new() else {
        return;
    };
    for (active, capacity) in [(0, 1), (2, 1), (1, 0)] {
        let result = ReactorCellIo::reactor(
            "not-opened",
            &services.reactor,
            &services.workers,
            native(),
            active,
            capacity,
            limits(),
        );
        assert_eq!(
            result.err().unwrap().kind(),
            std::io::ErrorKind::InvalidInput
        );
    }
}
