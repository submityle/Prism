use std::{
    future::Future,
    pin::Pin,
    sync::{atomic::AtomicU64, mpsc},
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;

use super::*;
use crate::{EditorWorld, IoBudget, IoCompletion, WorldIoDriver};

pub(super) struct Temp(pub(super) PathBuf);

impl Temp {
    pub(super) fn new() -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-editor-fs-unit-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create test directory: {e}"),
            }
        }
    }

    fn provider(&self, max: usize) -> FileCellIoProvider {
        FileCellIoProvider::new(&self.0, FileCellIoLimits::new(1024, max).unwrap()).unwrap()
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove private test directory");
    }
}

struct Notify(thread::Thread);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

pub(super) fn run<T>(mut future: Pin<Box<dyn Future<Output = T> + Send>>) -> T {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        assert!(Instant::now() < deadline, "worker did not complete");
        thread::park_timeout(Duration::from_millis(10));
    }
}

pub(super) fn idle(provider: &FileCellIoProvider) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while provider.outstanding_operations() != 0 {
        assert!(Instant::now() < deadline, "worker did not release capacity");
        thread::yield_now();
    }
}

fn request(revision: u64) -> CellSaveRequest {
    let mut model = EditorWorld::new(StableGuid::from_path("world/fs-tests")).unwrap();
    let coord = CellCoord::new(-1, 2, 3);
    let file = model.add_cell(coord, []).unwrap().file().clone();
    CellSaveRequest::new(coord, file, revision)
}

fn save(provider: &FileCellIoProvider, revision: u64, bytes: &[u8]) -> Result<(), FileCellIoError> {
    let result = run(provider
        .save_cell(request(revision), bytes.to_vec())
        .unwrap());
    idle(provider);
    result
}

#[test]
fn pre_rename_failure_preserves_original_and_retries_without_stage_leaks() {
    let temp = Temp::new();
    let provider = temp.provider(2);
    save(&provider, 1, b"old").unwrap();
    let path = temp.0.join("cells/-1_2_3.prismcell");
    let original = fs::read(&path).unwrap();
    *lock(&provider.shared.hook) = Some(Arc::new(|phase| {
        if phase == TestPhase::Staged {
            Err(io::Error::other("injected pre-rename failure"))
        } else {
            Ok(())
        }
    }));
    assert!(matches!(
        save(&provider, 2, b"new"),
        Err(FileCellIoError::Io { .. })
    ));
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(!path.with_extension("prismcell.pending").exists());
    *lock(&provider.shared.hook) = None;
    save(&provider, 2, b"new").unwrap();
    assert_ne!(fs::read(&path).unwrap(), original);
}

#[test]
fn post_rename_failure_is_uncertain_and_identical_retry_resyncs() {
    let temp = Temp::new();
    let provider = temp.provider(2);
    save(&provider, 1, b"old").unwrap();
    *lock(&provider.shared.hook) = Some(Arc::new(|phase| {
        if phase == TestPhase::Renamed {
            Err(io::Error::other("injected directory sync failure"))
        } else {
            Ok(())
        }
    }));
    assert!(matches!(
        save(&provider, 2, b"new"),
        Err(FileCellIoError::CommitUncertain(_))
    ));
    let path = temp.0.join("cells/-1_2_3.prismcell");
    assert_eq!(&fs::read(&path).unwrap()[HEADER_BYTES..], b"new");
    *lock(&provider.shared.hook) = None;
    save(&provider, 2, b"new").unwrap();
    assert!(matches!(
        save(&provider, 2, b"divergent"),
        Err(FileCellIoError::RevisionConflict {
            current: Some(2),
            ..
        })
    ));
}

#[test]
fn staged_driver_cancellation_prevents_publication_and_keeps_worker_bounded() {
    let temp = Temp::new();
    let provider = temp.provider(1);
    save(&provider, 1, b"old").unwrap();
    let (staged_tx, staged_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Mutex::new(release_rx);
    *lock(&provider.shared.hook) = Some(Arc::new(move |phase| {
        if phase == TestPhase::Staged {
            staged_tx.send(()).unwrap();
            lock(&release_rx)
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        Ok(())
    }));
    let mut driver = WorldIoDriver::new(provider.clone(), IoBudget::new(2, 2048).unwrap());
    let id = driver.submit_save(request(2), b"new".to_vec()).unwrap();
    staged_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let path = temp.0.join("cells/-1_2_3.prismcell");
    assert_eq!(&fs::read(&path).unwrap()[HEADER_BYTES..], b"old");
    let other = temp.provider(2);
    assert!(matches!(
        save(&other, 3, b"conflict"),
        Err(FileCellIoError::Busy)
    ));
    assert!(driver.cancel(id));
    assert_eq!(driver.reserved_bytes(), 0);
    assert!(matches!(
        driver.poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Some(IoCompletion::Cancelled { .. }))
    ));
    // Cancelling does not admit a new detached worker while the old one exits.
    assert!(matches!(
        provider.save_cell(request(3), vec![]),
        Err(FileCellIoError::Capacity)
    ));
    release_tx.send(()).unwrap();
    idle(&provider);
    assert_eq!(&fs::read(&path).unwrap()[HEADER_BYTES..], b"old");
    assert!(!path.with_extension("prismcell.pending").exists());
    *lock(&provider.shared.hook) = None;
    save(&provider, 3, b"retry").unwrap();
}

#[test]
fn tampered_relative_identity_is_rejected_without_starting_io() {
    let temp = Temp::new();
    let provider = temp.provider(1);
    for path in [
        "../escape",
        "/absolute",
        "cells/../../escape",
        "cells\\..\\escape",
    ] {
        let mut identity = request(1).file().clone();
        identity.relative_path = path.into();
        let request = CellSaveRequest::new(identity.coord(), identity, 1);
        assert!(matches!(
            provider.save_cell(request, vec![]),
            Err(FileCellIoError::InvalidRequest)
        ));
    }
    assert_eq!(provider.outstanding_operations(), 0);
    assert!(!temp.0.join("cells").exists());
}

#[test]
fn crashed_stage_is_never_loaded_and_is_reclaimed_by_next_save() {
    let temp = Temp::new();
    let provider = temp.provider(2);
    save(&provider, 1, b"committed").unwrap();
    let stage = temp.0.join("cells/-1_2_3.prismcell.pending");
    fs::write(&stage, b"partial crash stage").unwrap();
    save(&provider, 2, b"next").unwrap();
    assert!(!stage.exists());
    assert_eq!(
        &fs::read(temp.0.join("cells/-1_2_3.prismcell")).unwrap()[HEADER_BYTES..],
        b"next"
    );
}
