//! Actual opaque actor storage, guarded reconciliation and process recovery.
use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;
use prism_editor_world::*;
use std::{
    fs::{self, File},
    future::Future,
    path::PathBuf,
    pin::Pin,
    process::{Child, Command},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(5);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-ofpa-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => panic!("create private directory: {e}"),
            }
        }
    }
    fn provider(&self) -> FileActorIoProvider {
        FileActorIoProvider::new(&self.0, ActorIoLimits::new(1024, 4, 4096).unwrap()).unwrap()
    }
    fn path(&self) -> PathBuf {
        self.0.join(identity().relative_path())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
struct Notify(thread::Thread);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn run<T>(mut future: Pin<Box<dyn Future<Output = T> + Send>>) -> T {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        assert!(Instant::now() < deadline, "worker did not finish");
        thread::park_timeout(Duration::from_millis(5));
    }
}
fn idle(provider: &FileActorIoProvider) {
    let deadline = Instant::now() + TIMEOUT;
    while provider.outstanding_operations() != 0 || provider.reserved_bytes() != 0 {
        assert!(Instant::now() < deadline, "worker retained resources");
        thread::yield_now();
    }
}
fn identity() -> ActorFileIdentity {
    ActorFileIdentity::new(
        StableGuid::from_path("actor/opaque"),
        CellCoord::new(i32::MIN, -2, i32::MAX),
    )
    .unwrap()
}
fn request(revision: u64, base: ActorRevisionBase) -> ActorSaveRequest {
    ActorSaveRequest::new(identity(), revision, base).unwrap()
}
fn save(
    provider: &FileActorIoProvider,
    revision: u64,
    base: ActorRevisionBase,
    bytes: &[u8],
) -> Result<ActorSaveReceipt, FileCellIoError> {
    let result = run(provider
        .save_actor(request(revision, base), bytes.into(), TIMEOUT)
        .unwrap());
    idle(provider);
    result
}

#[test]
fn versioned_roundtrip_restart_revision_bases_and_idempotency() {
    let temp = Temp::new();
    let provider = temp.provider();
    save(&provider, 7, ActorRevisionBase::Missing, &[0, 255, 1]).unwrap();
    let raw = fs::read(temp.path()).unwrap();
    assert_eq!(&raw[..8], b"PEWACTR\0");
    assert_eq!(&raw[8..12], &1u32.to_le_bytes());
    assert_eq!(raw.len(), 67);
    let reopened = temp.provider();
    let snapshot = run(reopened.load_actor(identity(), 1024, TIMEOUT).unwrap()).unwrap();
    assert_eq!(snapshot.revision(), 7);
    assert_eq!(snapshot.bytes(), &[0, 255, 1]);
    for (revision, base) in [
        (8, ActorRevisionBase::Missing),
        (9, ActorRevisionBase::Revision(6)),
        (7, ActorRevisionBase::Revision(7)),
    ] {
        assert!(matches!(
            save(&provider, revision, base, b"divergent"),
            Err(FileCellIoError::RevisionConflict {
                current: Some(7),
                ..
            })
        ));
        assert_eq!(fs::read(temp.path()).unwrap(), raw);
    }
    save(&provider, 7, ActorRevisionBase::Missing, &[0, 255, 1]).unwrap();
    save(&provider, 8, ActorRevisionBase::Revision(7), b"").unwrap();
    assert!(run(reopened.load_actor(identity(), 1024, TIMEOUT).unwrap())
        .unwrap()
        .bytes()
        .is_empty());
    idle(&reopened);
}

#[test]
fn corrupt_envelopes_lengths_identity_and_versions_fail_closed() {
    let temp = Temp::new();
    let provider = temp.provider();
    save(&provider, 1, ActorRevisionBase::Missing, b"opaque").unwrap();
    let original = fs::read(temp.path()).unwrap();
    for offset in [0, 8, 12, 24, 40, 48, 56, 64] {
        let mut broken = original.clone();
        broken[offset] ^= 1;
        fs::write(temp.path(), broken).unwrap();
        assert!(run(provider.load_actor(identity(), 1024, TIMEOUT).unwrap()).is_err());
        assert!(save(&provider, 2, ActorRevisionBase::Revision(1), b"replacement").is_err());
    }
    for broken in [
        &original[..12],
        &[original.as_slice(), b"extra"].concat()[..],
    ] {
        fs::write(temp.path(), broken).unwrap();
        assert!(run(provider.load_actor(identity(), 1024, TIMEOUT).unwrap()).is_err());
    }
    File::options()
        .write(true)
        .open(temp.path())
        .unwrap()
        .set_len(2048)
        .unwrap();
    assert!(matches!(
        run(provider.load_actor(identity(), 1024, TIMEOUT).unwrap()),
        Err(FileCellIoError::TooLarge { .. })
    ));
    assert!(ActorFileIdentity::new(StableGuid::from_u128(0), identity().cell()).is_err());
    assert!(ActorSaveRequest::new(identity(), 1, ActorRevisionBase::Revision(2)).is_err());
    assert!(provider.load_actor(identity(), 0, TIMEOUT).is_err());
    assert!(provider
        .load_actor(identity(), 1024, Duration::ZERO)
        .is_err());
    assert!(provider
        .save_actor(
            request(2, ActorRevisionBase::Revision(1)),
            vec![0; 1025].into(),
            TIMEOUT
        )
        .is_err());
    idle(&provider);
}

#[test]
fn retained_future_holds_request_and_byte_admission() {
    let temp = Temp::new();
    let provider =
        FileActorIoProvider::new(&temp.0, ActorIoLimits::new(1024, 1, 1024).unwrap()).unwrap();
    save(&provider, 1, ActorRevisionBase::Missing, b"payload").unwrap();
    let future = provider.load_actor(identity(), 1024, TIMEOUT).unwrap();
    assert!(matches!(
        provider.load_actor(identity(), 1024, TIMEOUT),
        Err(FileCellIoError::Capacity)
    ));
    assert_eq!(provider.reserved_bytes(), 1024);
    assert_eq!(run(future).unwrap().bytes(), b"payload");
    idle(&provider);
}

#[test]
fn independent_actor_writers_cannot_replace_a_divergent_revision() {
    let temp = Temp::new();
    let first = temp.provider();
    let second = temp.provider();
    save(&first, 1, ActorRevisionBase::Missing, b"base").unwrap();
    let gate = Arc::new(std::sync::Barrier::new(3));
    let mut workers = Vec::new();
    for (provider, bytes) in [
        (first.clone(), b"first".as_slice()),
        (second.clone(), b"second".as_slice()),
    ] {
        let gate = gate.clone();
        workers.push(thread::spawn(move || {
            gate.wait();
            save(&provider, 2, ActorRevisionBase::Revision(1), bytes)
        }));
    }
    gate.wait();
    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert!(results.iter().any(|result| matches!(
        result,
        Err(FileCellIoError::Busy | FileCellIoError::RevisionConflict { .. })
    )));
    let winner = results.into_iter().find_map(Result::ok).unwrap();
    let snapshot = run(first.load_actor(identity(), 1024, TIMEOUT).unwrap()).unwrap();
    assert_eq!(snapshot.revision(), 2);
    assert_eq!(snapshot.bytes(), winner.payload());
    idle(&first);
    idle(&second);
}

fn world() -> (EditorWorldData, CellLockToken) {
    let mut world = EditorWorldData::new(
        StableGuid::from_path("world"),
        EditorWorldLimits::new(2, 32, 4, 1, 1).unwrap(),
    )
    .unwrap();
    let cell = identity().cell();
    world.add_cell(cell, []).unwrap();
    let mut demand = ResidencyDemand::new();
    demand.require(cell);
    let (_, tokens) = world.begin_residency(&demand, 0).unwrap();
    world
        .complete_load(tokens[0], b"manifest".to_vec())
        .unwrap();
    world.register_actor(identity().actor(), cell).unwrap();
    let lease = world
        .acquire_lock(cell, StableGuid::from_path("owner"), 0, 100)
        .unwrap();
    (world, lease)
}
fn write_pending(provider: &FileActorIoProvider, pending: &PendingActorSave) -> ActorSaveReceipt {
    let receipt = run(provider
        .save_actor(pending.request().clone(), pending.payload(), TIMEOUT)
        .unwrap())
    .unwrap();
    idle(provider);
    receipt
}

#[test]
fn receipt_keeps_newer_edits_and_expired_lease_dirty_then_retries_base() {
    let temp = Temp::new();
    let provider = temp.provider();
    let (mut world, lease) = world();
    let actor = identity().actor();
    world
        .replace_actor_payload(actor, &lease, 1, Arc::from(b"old".as_slice()))
        .unwrap();
    let pending = world
        .prepare_actor_save(actor, &lease, 1, &provider)
        .unwrap();
    world
        .replace_actor_payload(actor, &lease, 2, pending.payload())
        .unwrap(); // identical bytes still a new edit
    let receipt = write_pending(&provider, &pending);
    assert!(world.commit_ofpa(&pending, &receipt, 2).is_err());
    assert_eq!(world.plan_ofpa().operations().len(), 1);
    let current = world
        .prepare_actor_save(actor, &lease, 2, &provider)
        .unwrap();
    assert_eq!(current.request().base(), ActorRevisionBase::Revision(1));
    let receipt = write_pending(&provider, &current);
    assert!(world.commit_ofpa(&current, &receipt, 100).is_err());
    assert_eq!(world.plan_ofpa().operations().len(), 1);
    let lease = world
        .acquire_lock(identity().cell(), StableGuid::from_path("owner"), 100, 100)
        .unwrap();
    let retry = world
        .prepare_actor_save(actor, &lease, 100, &provider)
        .unwrap();
    let receipt = write_pending(&provider, &retry);
    world.commit_ofpa(&retry, &receipt, 100).unwrap();
    assert!(world.plan_ofpa().operations().is_empty());
    assert!(world.model().cell(identity().cell()).unwrap().is_dirty());
}

#[test]
fn wrong_root_or_bytes_receipts_cannot_acknowledge_and_stale_load_cannot_overwrite() {
    let temp = Temp::new();
    let other = Temp::new();
    let provider = temp.provider();
    let alien = other.provider();
    let (mut world, lease) = world();
    let actor = identity().actor();
    world
        .replace_actor_payload(actor, &lease, 1, Arc::from(b"mine".as_slice()))
        .unwrap();
    let pending = world
        .prepare_actor_save(actor, &lease, 1, &provider)
        .unwrap();
    let wrong_root = write_pending(&alien, &pending);
    assert!(matches!(
        world.commit_ofpa(&pending, &wrong_root, 1),
        Err(EditorWorldDataError::UnexpectedCompletion)
    ));
    let wrong_bytes = save(&provider, 1, ActorRevisionBase::Missing, b"other").unwrap();
    assert!(matches!(
        world.commit_ofpa(&pending, &wrong_bytes, 1),
        Err(EditorWorldDataError::UnexpectedCompletion)
    ));
    // A fresh clean catalog can load the actual opaque disk bytes.
    let (mut restored, lease) = self::world();
    let load = restored
        .prepare_actor_load(actor, &lease, 1, &provider)
        .unwrap();
    let snapshot = run(provider
        .load_actor(load.identity().clone(), 32, TIMEOUT)
        .unwrap())
    .unwrap();
    restored.apply_actor_load(&load, snapshot, 1).unwrap();
    assert_eq!(restored.actor_payload(actor), Some(b"other".as_slice()));
    let load = restored
        .prepare_actor_load(actor, &lease, 2, &provider)
        .unwrap();
    let snapshot = run(provider
        .load_actor(load.identity().clone(), 32, TIMEOUT)
        .unwrap())
    .unwrap();
    restored
        .replace_actor_payload(actor, &lease, 2, Arc::from(b"newer".as_slice()))
        .unwrap();
    assert!(restored.apply_actor_load(&load, snapshot, 2).is_err());
    assert_eq!(restored.actor_payload(actor), Some(b"newer".as_slice()));
    assert!(restored
        .replace_actor_payload(actor, &lease, 2, vec![0; 33].into())
        .is_err());
    idle(&provider);
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn actor_lock_child() {
    let Some(root) = std::env::var_os("PRISM_OFPA_LOCK_CHILD") else {
        return;
    };
    let root = PathBuf::from(root);
    let path = root.join(identity().relative_path());
    let guard = File::options()
        .read(true)
        .write(true)
        .open(path.with_extension("lock"))
        .unwrap();
    guard.try_lock().unwrap();
    fs::write(path.with_extension("pending"), b"crashed partial stage").unwrap();
    fs::write(root.join("ready"), b"locked").unwrap();
    loop {
        thread::park();
    }
}
#[test]
fn process_death_releases_actor_lock_and_reclaims_only_its_stage() {
    let temp = Temp::new();
    let provider = temp.provider();
    save(&provider, 1, ActorRevisionBase::Missing, b"old").unwrap();
    let unrelated = temp.path().with_extension("unrelated");
    fs::write(&unrelated, b"retain").unwrap();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "actor_lock_child", "--nocapture"])
            .env("PRISM_OFPA_LOCK_CHILD", &temp.0)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + TIMEOUT;
    while !temp.0.join("ready").exists() {
        assert!(Instant::now() < deadline);
        assert!(child.0.try_wait().unwrap().is_none());
        thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(
        save(&provider, 2, ActorRevisionBase::Revision(1), b"new"),
        Err(FileCellIoError::Busy)
    ));
    assert!(matches!(
        run(provider.load_actor(identity(), 1024, TIMEOUT).unwrap()),
        Err(FileCellIoError::Busy)
    ));
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    save(&provider, 2, ActorRevisionBase::Revision(1), b"new").unwrap();
    assert!(!temp.path().with_extension("pending").exists());
    assert_eq!(fs::read(unrelated).unwrap(), b"retain");
    assert_eq!(
        run(provider.load_actor(identity(), 1024, TIMEOUT).unwrap())
            .unwrap()
            .bytes(),
        b"new"
    );
    idle(&provider);
}

#[cfg(unix)]
#[test]
fn actor_directory_file_lock_and_stage_symlinks_are_rejected() {
    use std::os::unix::fs::symlink;
    let outside = Temp::new();
    let victim = outside.0.join("victim");
    fs::write(&victim, b"safe").unwrap();
    for kind in ["directory", "file", "lock", "pending"] {
        let temp = Temp::new();
        let provider = temp.provider();
        let path = temp.path();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        if kind == "directory" {
            fs::remove_dir(path.parent().unwrap()).unwrap();
            symlink(&outside.0, path.parent().unwrap()).unwrap();
        } else {
            symlink(
                &victim,
                if kind == "file" {
                    path.clone()
                } else {
                    path.with_extension(kind)
                },
            )
            .unwrap();
        }
        assert!(matches!(
            save(&provider, 1, ActorRevisionBase::Missing, b"bad"),
            Err(FileCellIoError::UnsafePath(_))
        ));
        assert_eq!(fs::read(&victim).unwrap(), b"safe");
    }
}

#[test]
fn actor_then_cell_real_save_and_restart_reload_are_separate_commits() {
    let temp = Temp::new();
    let actors = temp.provider();
    let (mut world, lease) = world();
    let actor = identity().actor();
    let cell = identity().cell();
    world
        .replace_actor_payload(actor, &lease, 1, Arc::from([0u8, 255, 3].as_slice()))
        .unwrap();
    let pending = world.prepare_actor_save(actor, &lease, 1, &actors).unwrap();
    let receipt = write_pending(&actors, &pending);
    world.commit_ofpa(&pending, &receipt, 1).unwrap();
    assert!(world.model().cell(cell).unwrap().is_dirty());
    let cells = FileCellIoProvider::new(&temp.0, FileCellIoLimits::new(1024, 1).unwrap()).unwrap();
    let mut driver = WorldIoDriver::new(cells.clone(), IoBudget::new(1, 1024).unwrap());
    let pending = world.prepare_save(cell, &lease, 1).unwrap();
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    let completion = run(Box::pin(std::future::poll_fn(move |cx| driver.poll(cx)))).unwrap();
    world
        .apply_save_completion(&pending, completion, 1)
        .unwrap();
    assert!(!world.model().cell(cell).unwrap().is_dirty());
    world.release_lock(&lease).unwrap();
    let (_, tokens) = world.begin_residency(&ResidencyDemand::new(), 2).unwrap();
    world.complete_unload(tokens[0]).unwrap();
    assert!(world.actor_payload(actor).is_none());
    let (mut restarted, lease) = self::world();
    let pending = restarted
        .prepare_actor_load(actor, &lease, 1, &actors)
        .unwrap();
    let snapshot = run(actors
        .load_actor(pending.identity().clone(), 32, TIMEOUT)
        .unwrap())
    .unwrap();
    restarted.apply_actor_load(&pending, snapshot, 1).unwrap();
    assert_eq!(
        restarted.actor_payload(actor),
        Some([0u8, 255, 3].as_slice())
    );
    restarted
        .replace_actor_payload(actor, &lease, 2, Arc::from(b"edit".as_slice()))
        .unwrap();
    let pending = restarted
        .prepare_actor_save(actor, &lease, 2, &actors)
        .unwrap();
    assert_eq!(pending.request().revision(), 2);
    assert_eq!(pending.request().base(), ActorRevisionBase::Revision(1));
    idle(&actors);
    let deadline = Instant::now() + TIMEOUT;
    while cells.outstanding_operations() != 0 {
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
}

#[test]
fn load_captured_before_unload_rejects_same_revision_after_reload() {
    let temp = Temp::new();
    let provider = temp.provider();
    save(&provider, 1, ActorRevisionBase::Missing, b"old").unwrap();
    let (mut world, lease) = world();
    let actor = identity().actor();
    let cell = identity().cell();
    let pending = world
        .prepare_actor_load(actor, &lease, 1, &provider)
        .unwrap();
    let snapshot = run(provider.load_actor(identity(), 32, TIMEOUT).unwrap()).unwrap();
    // A completion captured in an earlier residency cannot populate a reloaded
    // cell, even when its numeric revision is unchanged.
    world.release_lock(&lease).unwrap();
    let (_, tokens) = world.begin_residency(&ResidencyDemand::new(), 1).unwrap();
    world.complete_unload(tokens[0]).unwrap();
    let mut demand = ResidencyDemand::new();
    demand.require(cell);
    let (_, tokens) = world.begin_residency(&demand, 1).unwrap();
    world
        .complete_load(tokens[0], b"manifest".to_vec())
        .unwrap();
    assert!(world.apply_actor_load(&pending, snapshot, 1).is_err());
    assert!(world.actor_payload(actor).is_none());
    idle(&provider);
}

#[test]
fn actor_pool_is_bounded_and_other_cell_edits_keep_receipt_stale() {
    let temp = Temp::new();
    let provider = temp.provider();
    let (mut world, lease) = world();
    let actor = identity().actor();
    let second = StableGuid::from_path("actor/second");
    world.register_actor(second, identity().cell()).unwrap();
    world
        .replace_actor_payload(actor, &lease, 1, vec![1; 20].into())
        .unwrap();
    assert!(world
        .replace_actor_payload(second, &lease, 1, vec![2; 13].into())
        .is_err());
    assert!(world.actor_payload(second).is_none());
    let pending = world
        .prepare_actor_save(actor, &lease, 1, &provider)
        .unwrap();
    let receipt = write_pending(&provider, &pending);
    world
        .replace_payload(identity().cell(), &lease, 2, b"changed manifest".to_vec())
        .unwrap();
    assert!(world.commit_ofpa(&pending, &receipt, 2).is_err());
    assert_eq!(world.plan_ofpa().operations().len(), 1);
    let retry = world
        .prepare_actor_save(actor, &lease, 2, &provider)
        .unwrap();
    let receipt = write_pending(&provider, &retry);
    world.commit_ofpa(&retry, &receipt, 2).unwrap();
    assert!(world.plan_ofpa().operations().is_empty());
}
