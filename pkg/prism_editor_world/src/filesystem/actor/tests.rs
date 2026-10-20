use super::super::{
    tests::{run, Temp},
    TestPhase,
};
use super::*;
use crate::CellCoord;
use prism_asset::StableGuid;
use std::{
    sync::mpsc,
    task::{Context, Poll, Wake, Waker},
};

const TIMEOUT: Duration = Duration::from_secs(5);
fn identity() -> ActorFileIdentity {
    ActorFileIdentity::new(
        StableGuid::from_path("actor/failure"),
        CellCoord::new(1, 2, 3),
    )
    .unwrap()
}
fn provider(temp: &Temp, requests: usize, bytes: usize) -> FileActorIoProvider {
    FileActorIoProvider::new(&temp.0, ActorIoLimits::new(1024, requests, bytes).unwrap()).unwrap()
}
fn idle(provider: &FileActorIoProvider) {
    let deadline = Instant::now() + TIMEOUT;
    while provider.outstanding_operations() != 0 || provider.reserved_bytes() != 0 {
        assert!(Instant::now() < deadline, "worker retained resources");
        std::thread::yield_now();
    }
}
fn request(revision: u64) -> ActorSaveRequest {
    ActorSaveRequest::new(
        identity(),
        revision,
        if revision == 1 {
            ActorRevisionBase::Missing
        } else {
            ActorRevisionBase::Revision(revision - 1)
        },
    )
    .unwrap()
}
fn save(
    provider: &FileActorIoProvider,
    revision: u64,
    bytes: &[u8],
) -> Result<ActorSaveReceipt, FileCellIoError> {
    let result = run(provider
        .save_actor(request(revision), bytes.into(), TIMEOUT)
        .unwrap());
    idle(provider);
    result
}
fn checkpoint(provider: &FileActorIoProvider) -> (mpsc::Receiver<()>, mpsc::SyncSender<()>) {
    let (staged_tx, staged_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Mutex::new(release_rx);
    *lock(&provider.cell.shared.hook) = Some(Arc::new(move |phase| {
        if phase == TestPhase::Staged {
            staged_tx.send(()).unwrap();
            lock(&release_rx).recv_timeout(TIMEOUT).unwrap();
        }
        Ok(())
    }));
    (staged_rx, release_tx)
}

#[test]
fn cancelled_actor_retains_worker_and_byte_bounds_until_stage_cleanup() {
    let temp = Temp::new();
    let provider = provider(&temp, 4, 1024);
    save(&provider, 1, b"old").unwrap();
    let path = temp.0.join(identity().relative_path());
    let old = fs::read(&path).unwrap();
    let (staged, release) = checkpoint(&provider);
    let future = provider
        .save_actor(request(2), vec![1; 1024].into(), TIMEOUT)
        .unwrap();
    staged.recv_timeout(TIMEOUT).unwrap();
    drop(future);
    assert_eq!(provider.outstanding_operations(), 1);
    assert_eq!(provider.reserved_bytes(), 1024);
    assert!(matches!(
        provider.load_actor(identity(), 1, TIMEOUT),
        Err(FileCellIoError::Capacity)
    ));
    release.send(()).unwrap();
    idle(&provider);
    assert_eq!(provider.reserved_bytes(), 0);
    assert_eq!(fs::read(&path).unwrap(), old);
    assert!(!path.with_extension("pending").exists());
}

#[test]
fn expired_deadline_after_staging_never_publishes() {
    let temp = Temp::new();
    let provider = provider(&temp, 1, 1024);
    save(&provider, 1, b"old").unwrap();
    let path = temp.0.join(identity().relative_path());
    let old = fs::read(&path).unwrap();
    let (staged, release) = checkpoint(&provider);
    let timeout = Duration::from_millis(100);
    let started = Instant::now();
    let future = provider
        .save_actor(request(2), Arc::from(b"new".as_slice()), timeout)
        .unwrap();
    staged.recv_timeout(TIMEOUT).unwrap();
    while Instant::now().duration_since(started) <= timeout {
        std::thread::sleep(Duration::from_millis(5));
    }
    release.send(()).unwrap();
    assert!(matches!(run(future), Err(FileCellIoError::TimedOut)));
    idle(&provider);
    assert_eq!(fs::read(&path).unwrap(), old);
    assert!(!path.with_extension("pending").exists());
    assert_eq!(provider.reserved_bytes(), 0);
}

#[test]
fn actor_pre_rename_failure_and_uncertain_post_rename_retry() {
    let temp = Temp::new();
    let provider = provider(&temp, 2, 2048);
    save(&provider, 1, b"old").unwrap();
    let path = temp.0.join(identity().relative_path());
    let old = fs::read(&path).unwrap();
    for phase in [TestPhase::Staged, TestPhase::Renamed] {
        *lock(&provider.cell.shared.hook) = Some(Arc::new(move |at| {
            if at == phase {
                Err(io::Error::other("injected failure"))
            } else {
                Ok(())
            }
        }));
        let error = save(&provider, 2, b"new").unwrap_err();
        if phase == TestPhase::Staged {
            assert!(matches!(error, FileCellIoError::Io { .. }));
            assert_eq!(fs::read(&path).unwrap(), old);
        } else {
            assert!(matches!(error, FileCellIoError::CommitUncertain(_)));
            assert_eq!(&fs::read(&path).unwrap()[64..], b"new");
        }
        assert!(!path.with_extension("pending").exists());
    }
    *lock(&provider.cell.shared.hook) = None;
    save(&provider, 2, b"new").unwrap();
}

#[test]
fn worker_panic_is_terminal_removes_stage_and_releases_os_lock() {
    let temp = Temp::new();
    let provider = provider(&temp, 1, 1024);
    save(&provider, 1, b"old").unwrap();
    *lock(&provider.cell.shared.hook) = Some(Arc::new(|phase| {
        assert!(phase != TestPhase::Staged, "injected panic");
        Ok(())
    }));
    assert!(matches!(
        save(&provider, 2, b"new"),
        Err(FileCellIoError::Io { .. })
    ));
    let path = temp.0.join(identity().relative_path());
    assert!(!path.with_extension("pending").exists());
    assert_eq!(provider.reserved_bytes(), 0);
    *lock(&provider.cell.shared.hook) = None;
    save(&provider, 2, b"retry").unwrap();
}

#[test]
fn completed_unpolled_result_keeps_aggregate_reservation() {
    let temp = Temp::new();
    let provider = provider(&temp, 4, 1024);
    save(&provider, 1, b"old").unwrap();
    let (staged, release) = checkpoint(&provider);
    let mut future = provider
        .save_actor(request(2), vec![1; 1024].into(), TIMEOUT)
        .unwrap();
    staged.recv_timeout(TIMEOUT).unwrap();
    struct Notify(mpsc::SyncSender<()>);
    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.send(()).unwrap();
        }
    }
    let (done, completed) = mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(Notify(done)));
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    release.send(()).unwrap();
    // A wake occurs only after the worker has stored its result. Leave that
    // completed result inside the future to test its retained allocation.
    completed.recv_timeout(TIMEOUT).unwrap();
    assert_eq!(provider.reserved_bytes(), 1024);
    assert!(matches!(
        provider.load_actor(identity(), 1, TIMEOUT),
        Err(FileCellIoError::Capacity)
    ));
    drop(future);
    idle(&provider);
    assert_eq!(provider.reserved_bytes(), 0);
}
