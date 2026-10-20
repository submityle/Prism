//! Real filesystem, driver and editor-facade acceptance tests (no Bevy).

use std::{
    fs::{self, File, OpenOptions},
    future::Future,
    io,
    path::{Path, PathBuf},
    pin::Pin,
    process::{Child, Command},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier,
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;
use prism_editor_world::{
    CellFileIdentity, CellIoProvider, CellLoadRequest, CellResidency, CellSaveRequest,
    EditorWorldData, EditorWorldDataError, EditorWorldLimits, FileCellIoError, FileCellIoLimits,
    FileCellIoProvider, IoBudget, IoCompletion, ResidencyDemand, WorldIoDriver,
};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-editor-fs-{}-{}",
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

    fn provider(&self, limit: usize) -> FileCellIoProvider {
        FileCellIoProvider::new(&self.0, FileCellIoLimits::new(limit, 8).unwrap()).unwrap()
    }

    fn path(&self, identity: &CellFileIdentity) -> PathBuf {
        self.0.join(identity.relative_path())
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

fn run<T>(mut future: Pin<Box<dyn Future<Output = T> + Send>>) -> T {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
            return result;
        }
        assert!(Instant::now() < deadline, "future did not finish");
        thread::park_timeout(Duration::from_millis(10));
    }
}

fn event(driver: &mut WorldIoDriver<FileCellIoProvider>) -> IoCompletion<FileCellIoError> {
    let waker = Waker::from(Arc::new(Notify(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Poll::Ready(Some(result)) = driver.poll(&mut cx) {
            return result;
        }
        assert!(Instant::now() < deadline, "driver did not finish");
        thread::park_timeout(Duration::from_millis(10));
    }
}

fn idle(provider: &FileCellIoProvider) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while provider.outstanding_operations() != 0 {
        assert!(Instant::now() < deadline, "worker did not exit");
        thread::yield_now();
    }
}

fn world_at(name: &str, coord: CellCoord, limit: usize) -> EditorWorldData {
    let mut world = EditorWorldData::new(
        StableGuid::from_path(name),
        EditorWorldLimits::new(8, limit, 8, 2, 8).unwrap(),
    )
    .unwrap();
    world.add_cell(coord, []).unwrap();
    world
}

fn world() -> (EditorWorldData, CellCoord) {
    let coord = CellCoord::new(-5, 2, i32::MAX);
    (world_at("world/fs", coord, 256 * 1024), coord)
}

fn identity(world: &EditorWorldData, coord: CellCoord) -> CellFileIdentity {
    world.model().cell(coord).unwrap().file().clone()
}

fn load_request(world: &mut EditorWorldData, coord: CellCoord, limit: usize) -> CellLoadRequest {
    let mut demand = ResidencyDemand::new();
    demand.require(coord);
    let (_, tokens) = world.begin_residency(&demand, 0).unwrap();
    CellLoadRequest::new(tokens[0], identity(world, coord), limit)
}

fn save(
    provider: &FileCellIoProvider,
    identity: &CellFileIdentity,
    revision: u64,
    bytes: &[u8],
) -> Result<(), FileCellIoError> {
    run(provider
        .save_cell(
            CellSaveRequest::new(identity.coord(), identity.clone(), revision),
            bytes.to_vec(),
        )
        .unwrap())
}

#[test]
fn binary_roundtrip_reopen_stable_format_and_extreme_coordinates() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    for (index, coord) in [
        CellCoord::new(i32::MIN, 0, i32::MAX),
        CellCoord::new(-1, -2, -3),
    ]
    .into_iter()
    .enumerate()
    {
        let mut world = world_at("world/roundtrip", coord, 256 * 1024);
        let identity = identity(&world, coord);
        let payload: Vec<_> = (0..150_001).map(|n| (n % 251) as u8).collect();
        let payload = if index == 0 { payload } else { vec![] };
        save(&provider, &identity, 17, &payload).unwrap();
        let path = temp.path(&identity);
        assert_eq!(path, identity.scene_store_path(temp.0.join("cells")));
        let encoded = fs::read(&path).unwrap();
        assert_eq!(&encoded[..8], b"PEWCELL\0");
        assert_eq!(&encoded[8..12], &1_u32.to_le_bytes());
        assert_eq!(&encoded[40..48], &17_u64.to_le_bytes());
        assert_eq!(encoded.len(), payload.len() + 64);
        assert_eq!(&encoded[64..], payload);
        let reopened = temp.provider(256 * 1024);
        let loaded = run(reopened
            .load_snapshot(load_request(&mut world, coord, 256 * 1024))
            .unwrap())
        .unwrap();
        assert_eq!(loaded.bytes, payload);
        assert_eq!(loaded.revision, Some(17));
        save(&reopened, &identity, 17, &payload).unwrap();
        assert_eq!(fs::read(path).unwrap(), encoded);
        idle(&reopened);
    }
    idle(&provider);
}

#[test]
fn restart_restores_revision_and_stale_completion_does_not_lose_later_edits() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (mut world, coord) = world();
    save(&provider, &identity(&world, coord), 42, b"disk").unwrap();
    let request = load_request(&mut world, coord, 256 * 1024);
    let mut driver = WorldIoDriver::new(provider.clone(), IoBudget::new(4, 512 * 1024).unwrap());
    world.submit_load(&mut driver, request.token()).unwrap();
    assert!(world.apply_load_completion(event(&mut driver)).unwrap());
    assert_eq!(world.model().cell(coord).unwrap().revision(), 42);
    let lease = world
        .acquire_lock(coord, StableGuid::from_path("editor"), 0, 100)
        .unwrap();
    world
        .replace_payload(coord, &lease, 1, b"snapshot".to_vec())
        .unwrap();
    let pending = world.prepare_save(coord, &lease, 1).unwrap();
    assert_eq!(pending.request().revision(), 43);
    assert_eq!(pending.request().base_revision(), Some(42));
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    world
        .replace_payload(coord, &lease, 2, b"later edit".to_vec())
        .unwrap();
    assert!(matches!(
        world.apply_save_completion(&pending, event(&mut driver), 2),
        Err(EditorWorldDataError::Model(_))
    ));
    assert!(world.model().cell(coord).unwrap().is_dirty());
    assert_eq!(world.payload(coord), Some(&b"later edit"[..]));
    let retry = world.prepare_save(coord, &lease, 2).unwrap();
    assert_eq!(retry.request().base_revision(), Some(43));
    driver
        .submit_save(retry.request().clone(), retry.payload().to_vec())
        .unwrap();
    assert!(world
        .apply_save_completion(&retry, event(&mut driver), 2)
        .unwrap());
    assert!(!world.model().cell(coord).unwrap().is_dirty());
    let mut restarted = world_at("world/fs", coord, 256 * 1024);
    let snapshot = run(provider
        .load_snapshot(load_request(&mut restarted, coord, 256 * 1024))
        .unwrap())
    .unwrap();
    assert_eq!(snapshot.revision, Some(44));
    assert_eq!(snapshot.bytes, b"later edit");
    idle(&provider);
}

#[test]
fn expired_lease_cannot_acknowledge_a_real_disk_save_and_same_snapshot_can_retry() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (mut world, coord) = world();
    let request = load_request(&mut world, coord, 256 * 1024);
    world
        .complete_load(request.token(), b"initial".to_vec())
        .unwrap();
    let lease = world
        .acquire_lock(coord, StableGuid::from_path("editor"), 0, 2)
        .unwrap();
    world
        .replace_payload(coord, &lease, 1, b"edited".to_vec())
        .unwrap();
    let pending = world.prepare_save(coord, &lease, 1).unwrap();
    let mut driver = WorldIoDriver::new(provider.clone(), IoBudget::new(2, 512 * 1024).unwrap());
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    assert!(matches!(
        world.apply_save_completion(&pending, event(&mut driver), 2),
        Err(EditorWorldDataError::Lock(_))
    ));
    assert!(world.model().cell(coord).unwrap().is_dirty());
    let lease = world
        .acquire_lock(coord, StableGuid::from_path("editor"), 2, 10)
        .unwrap();
    let retry = world.prepare_save(coord, &lease, 3).unwrap();
    // Identical retries are permitted even when base == revision.
    driver
        .submit_save(retry.request().clone(), retry.payload().to_vec())
        .unwrap();
    assert!(world
        .apply_save_completion(&retry, event(&mut driver), 3)
        .unwrap());
    idle(&provider);
}

#[test]
fn divergent_editor_with_more_local_edits_cannot_overwrite_changed_disk_base() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (mut left, coord) = world();
    let mut right = world_at("world/fs", coord, 256 * 1024);
    let identity = identity(&left, coord);
    save(&provider, &identity, 10, b"shared base").unwrap();
    let mut a = WorldIoDriver::new(provider.clone(), IoBudget::new(2, 512 * 1024).unwrap());
    let other = temp.provider(256 * 1024);
    let mut b = WorldIoDriver::new(other.clone(), IoBudget::new(2, 512 * 1024).unwrap());
    for (world, driver) in [(&mut left, &mut a), (&mut right, &mut b)] {
        let request = load_request(world, coord, 256 * 1024);
        world.submit_load(driver, request.token()).unwrap();
        assert!(world.apply_load_completion(event(driver)).unwrap());
    }
    let left_lease = left
        .acquire_lock(coord, StableGuid::from_path("left editor"), 0, 100)
        .unwrap();
    let right_lease = right
        .acquire_lock(coord, StableGuid::from_path("right editor"), 0, 100)
        .unwrap();
    left.replace_payload(coord, &left_lease, 1, b"left edit".to_vec())
        .unwrap();
    let pending = left.prepare_save(coord, &left_lease, 1).unwrap();
    a.submit_save(pending.request().clone(), pending.payload().to_vec())
        .unwrap();
    assert!(left
        .apply_save_completion(&pending, event(&mut a), 1)
        .unwrap());
    let authoritative = fs::read(temp.path(&identity)).unwrap();
    for bytes in [
        b"right first edit".as_slice(),
        b"right second edit".as_slice(),
    ] {
        right
            .replace_payload(coord, &right_lease, 1, bytes.to_vec())
            .unwrap();
    }
    let stale = right.prepare_save(coord, &right_lease, 1).unwrap();
    assert_eq!(stale.request().revision(), 12);
    assert_eq!(stale.request().base_revision(), Some(10));
    b.submit_save(stale.request().clone(), stale.payload().to_vec())
        .unwrap();
    let completion = event(&mut b);
    assert!(matches!(
        &completion,
        IoCompletion::Failed {
            error: prism_editor_world::IoFailure::Provider(FileCellIoError::RevisionConflict {
                requested: 12,
                current: Some(11),
            }),
            ..
        }
    ));
    assert!(!right.apply_save_completion(&stale, completion, 1).unwrap());
    assert!(right.model().cell(coord).unwrap().is_dirty());
    assert_eq!(right.payload(coord), Some(b"right second edit".as_slice()));
    assert_eq!(fs::read(temp.path(&identity)).unwrap(), authoritative);
    idle(&provider);
    idle(&other);
}

#[test]
fn stale_equal_divergent_and_wrong_base_saves_never_replace_newer_data() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (world, coord) = world();
    let identity = identity(&world, coord);
    save(&provider, &identity, 10, b"newer").unwrap();
    let original = fs::read(temp.path(&identity)).unwrap();
    for revision in [0, 9, 10] {
        assert!(matches!(
            save(&provider, &identity, revision, b"stale"),
            Err(FileCellIoError::RevisionConflict {
                current: Some(10),
                ..
            })
        ));
        assert_eq!(fs::read(temp.path(&identity)).unwrap(), original);
    }
    let request = CellSaveRequest::new(coord, identity.clone(), 100).with_base_revision(9);
    assert!(matches!(
        run(provider
            .save_cell(request, b"divergent branch".to_vec())
            .unwrap()),
        Err(FileCellIoError::RevisionConflict { .. })
    ));
    save(&provider, &identity, 10, b"newer").unwrap();
    assert_eq!(fs::read(temp.path(&identity)).unwrap(), original);
    idle(&provider);
}

#[test]
fn malformed_files_are_rejected_and_preserved_for_recovery() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (mut world, coord) = world();
    let identity = identity(&world, coord);
    save(&provider, &identity, 1, b"payload").unwrap();
    let request = load_request(&mut world, coord, 256 * 1024);
    let path = temp.path(&identity);
    let original = fs::read(&path).unwrap();
    let mut cases = vec![
        vec![],
        original[..63].to_vec(),
        original[..original.len() - 1].to_vec(),
    ];
    for offset in [0, 8, 12, 24, 40, 56, 64] {
        let mut bytes = original.clone();
        bytes[offset] ^= 0x80;
        cases.push(bytes);
    }
    let mut huge = original.clone();
    huge[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
    cases.push(huge);
    let mut trailing = original.clone();
    trailing.push(0);
    cases.push(trailing);
    for corrupt in cases {
        fs::write(&path, &corrupt).unwrap();
        assert!(run(provider.load_cell(request.clone()).unwrap()).is_err());
        assert!(save(&provider, &identity, 2, b"replacement").is_err());
        assert_eq!(fs::read(&path).unwrap(), corrupt);
    }
    // A valid other-world envelope at the same numeric coordinate is not ours.
    fs::write(&path, &original).unwrap();
    let mut other_world = world_at("different/world", coord, 256 * 1024);
    assert!(matches!(
        run(provider
            .load_cell(load_request(&mut other_world, coord, 256 * 1024))
            .unwrap()),
        Err(FileCellIoError::Corrupt("stable identity mismatch"))
    ));
    idle(&provider);
}

#[test]
fn missing_and_corrupt_driver_loads_restore_unloaded_state() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (mut world, coord) = world();
    let mut driver = WorldIoDriver::new(provider.clone(), IoBudget::new(2, 512 * 1024).unwrap());
    for bytes in [None, Some(b"broken".as_slice())] {
        if let Some(bytes) = bytes {
            fs::create_dir_all(temp.0.join("cells")).unwrap();
            fs::write(temp.path(&identity(&world, coord)), bytes).unwrap();
        }
        let request = load_request(&mut world, coord, 256 * 1024);
        world.submit_load(&mut driver, request.token()).unwrap();
        assert!(!world.apply_load_completion(event(&mut driver)).unwrap());
        assert_eq!(
            world.model().cell(coord).unwrap().residency(),
            CellResidency::Unloaded
        );
        assert!(world.payload(coord).is_none());
        assert_eq!(driver.reserved_bytes(), 0);
    }
    idle(&provider);
}

#[test]
fn limits_are_enforced_before_read_allocation_and_submission() {
    assert!(FileCellIoLimits::new(0, 1).is_err());
    assert!(FileCellIoLimits::new(1, 0).is_err());
    assert!(FileCellIoLimits::new(prism_scene::partition::MAX_CELL_BYTES, 1).is_err());
    assert!(FileCellIoLimits::new(prism_scene::partition::MAX_CELL_BYTES - 64, 1).is_ok());
    let temp = Temp::new();
    let provider = temp.provider(4);
    let (mut world, coord) = world();
    let identity = identity(&world, coord);
    let request = CellSaveRequest::new(coord, identity.clone(), 1);
    assert!(matches!(
        provider.save_cell(request.clone(), vec![0; 5]),
        Err(FileCellIoError::TooLarge {
            limit: 4,
            actual: 5
        })
    ));
    assert_eq!(provider.outstanding_operations(), 0);
    save(&provider, &identity, 1, b"1234").unwrap();
    let request = load_request(&mut world, coord, 3);
    let zero = CellLoadRequest::new(request.token(), identity.clone(), 0);
    assert!(matches!(
        run(provider.load_cell(request).unwrap()),
        Err(FileCellIoError::TooLarge { limit: 3, .. })
    ));
    assert!(matches!(
        provider.load_cell(zero),
        Err(FileCellIoError::InvalidRequest)
    ));
    // Sparse oversized file must be rejected by stat, not read into a buffer.
    let file = OpenOptions::new()
        .write(true)
        .open(temp.path(&identity))
        .unwrap();
    file.set_len(1024 * 1024 * 1024).unwrap();
    let mut other = world_at("world/fs", coord, 256 * 1024);
    assert!(matches!(
        run(provider
            .load_cell(load_request(&mut other, coord, 256 * 1024))
            .unwrap()),
        Err(FileCellIoError::TooLarge { .. })
    ));
    idle(&provider);
}

#[test]
fn independently_opened_providers_serialize_same_revision_writers() {
    let temp = Temp::new();
    let a = temp.provider(256 * 1024);
    let b = temp.provider(256 * 1024);
    let (world, coord) = world();
    let identity = identity(&world, coord);
    save(&a, &identity, 1, b"original").unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = [
        (a.clone(), b"left".as_slice()),
        (b.clone(), b"right".as_slice()),
    ]
    .into_iter()
    .map(|(provider, bytes)| {
        let barrier = barrier.clone();
        let identity = identity.clone();
        thread::spawn(move || {
            barrier.wait();
            save(&provider, &identity, 2, bytes)
        })
    })
    .collect();
    barrier.wait();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(results.iter().any(|r| matches!(
        r,
        Err(FileCellIoError::Busy | FileCellIoError::RevisionConflict { .. })
    )));
    let encoded = fs::read(temp.path(&identity)).unwrap();
    assert!([b"left".as_slice(), b"right".as_slice()].contains(&&encoded[64..]));
    idle(&a);
    idle(&b);
}

#[cfg(unix)]
#[test]
fn symlink_directory_cell_lock_and_stage_cannot_escape_root() {
    use std::os::unix::fs::symlink;
    let outside = Temp::new();
    fs::write(outside.0.join("victim"), b"untouched").unwrap();
    let (world, coord) = world();
    let identity = identity(&world, coord);
    for extension in [
        "directory",
        "prismcell",
        "prismcell.lock",
        "prismcell.pending",
    ] {
        let temp = Temp::new();
        let provider = temp.provider(256 * 1024);
        if extension == "directory" {
            symlink(&outside.0, temp.0.join("cells")).unwrap();
        } else {
            fs::create_dir(temp.0.join("cells")).unwrap();
            symlink(
                outside.0.join("victim"),
                temp.path(&identity).with_extension(extension),
            )
            .unwrap();
        }
        assert!(matches!(
            save(&provider, &identity, 1, b"escape"),
            Err(FileCellIoError::UnsafePath(_))
        ));
        assert_eq!(fs::read(outside.0.join("victim")).unwrap(), b"untouched");
        idle(&provider);
    }
    let limits = FileCellIoLimits::new(8, 1).unwrap();
    assert!(matches!(
        FileCellIoProvider::new(outside.0.join(".."), limits),
        Err(FileCellIoError::UnsafePath(_))
    ));
}

// The subprocess holds the actual OS lock and is killed by its parent. This
// exercises lock release after process death, not just a shared Rust mutex.
#[test]
fn filesystem_lock_child() {
    let Some(root) = std::env::var_os("PRISM_EDITOR_FS_LOCK_CHILD") else {
        return;
    };
    let root = PathBuf::from(root);
    let lock = File::options()
        .read(true)
        .write(true)
        .open(root.join("cells/-5_2_2147483647.prismcell.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    fs::write(root.join("child-ready"), b"locked").unwrap();
    loop {
        thread::park();
    }
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn process_lock_is_busy_then_recovers_after_process_death() {
    let temp = Temp::new();
    let provider = temp.provider(256 * 1024);
    let (world, coord) = world();
    let identity = identity(&world, coord);
    save(&provider, &identity, 1, b"original").unwrap();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "filesystem_lock_child", "--nocapture"])
            .env("PRISM_EDITOR_FS_LOCK_CHILD", &temp.0)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !Path::new(&temp.0.join("child-ready")).exists() {
        assert!(Instant::now() < deadline, "lock child did not start");
        assert!(child.0.try_wait().unwrap().is_none(), "lock child exited");
        thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(
        save(&provider, &identity, 2, b"blocked"),
        Err(FileCellIoError::Busy)
    ));
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    // Leave a crashed writer's exact stage path; the next holder reclaims it.
    fs::write(
        temp.path(&identity).with_extension("prismcell.pending"),
        b"partial",
    )
    .unwrap();
    save(&provider, &identity, 2, b"recovered").unwrap();
    assert_eq!(&fs::read(temp.path(&identity)).unwrap()[64..], b"recovered");
    idle(&provider);
}
