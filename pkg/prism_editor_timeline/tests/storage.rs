//! Real files, exact-byte conflicts and explicit host adoption of async results.
extern crate alloc;
use alloc::sync::Arc;
use prism_editor_timeline::*;
use std::{
    fs::{self, File, FileTimes},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    },
    task::Poll,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(10);
static SERIAL: AtomicU64 = AtomicU64::new(1);
struct Project(PathBuf);
impl Project {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-timeline-files-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self) -> PathBuf {
        self.0.join("camera.timeline")
    }
    fn service(&self, max_requests: usize) -> TimelineStorageService {
        TimelineStorageService::new(
            self.path(),
            TimelineStorageLimits {
                max_requests,
                ..Default::default()
            },
        )
        .unwrap()
    }
}
impl Drop for Project {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn document(rate: u64) -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(u128::MAX), "相机 translation", 2).unwrap();
    let mut key = Keyframe::new(
        KeyId::from_raw(0),
        10,
        vec![-0.0, 3.5],
        Interpolation::Cubic,
    );
    key.set_tangents(vec![0.1, -0.2], vec![0.3, -0.4]).unwrap();
    track.insert(key).unwrap();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: u128::MAX,
            tick: u64::MAX,
            name: "镜头切换".into(),
        })
        .unwrap();
    TimelineDocument {
        ticks_per_second: rate,
        timeline,
    }
}
fn bytes(rate: u64) -> Vec<u8> {
    document(rate)
        .encode(TimelineLimits::default(), 8 * 1024 * 1024)
        .unwrap()
}
fn token() -> TimelineSnapshotToken {
    TimelineEditor::new(Timeline::default(), TimelineLimits::default())
        .unwrap()
        .snapshot_token()
}
fn wait(
    request: &mut TimelineStorageRequest,
) -> Result<TimelineStorageResult, TimelineStorageError> {
    let until = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(result) = request.poll() {
            return result;
        }
        assert!(
            Instant::now() < until,
            "worker did not finish request {}",
            request.id()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn saved(request: &mut TimelineStorageRequest) -> TimelineSaveReceipt {
    match wait(request).unwrap() {
        TimelineStorageResult::Saved(receipt) => receipt,
        other => panic!("{other:?}"),
    }
}
fn loaded(request: &mut TimelineStorageRequest) -> TimelineLoadCandidate {
    match wait(request).unwrap() {
        TimelineStorageResult::Loaded(candidate) => candidate,
        other => panic!("{other:?}"),
    }
}
fn error(request: &mut TimelineStorageRequest, expected: TimelineStorageError) {
    assert_eq!(wait(request).unwrap_err(), expected);
}

#[test]
fn real_worker_roundtrip_preserves_document_exact_bytes_and_snapshot_identity() {
    let project = Project::new();
    let service = project.service(2);
    let snapshot_token = token();
    let original = bytes(120);
    let mut request = service
        .save(
            original.clone(),
            snapshot_token.clone(),
            TimelineFileRevision::Missing,
            TIMEOUT,
        )
        .unwrap();
    let receipt = saved(&mut request);
    assert_eq!(receipt.snapshot_token, snapshot_token);
    assert_eq!(
        receipt.revision,
        TimelineFileRevision::Present(original.clone().into())
    );
    assert_eq!(receipt.durability, TimelineSaveDurability::Synced);
    assert!(!receipt.deadline_exceeded);
    assert_eq!(request.cancel(), TimelineCancelOutcome::TooLate);
    assert!(matches!(
        request.poll(),
        Poll::Ready(Err(TimelineStorageError::Consumed))
    ));
    let mut load = service.load(TIMEOUT).unwrap();
    assert!(load.id() > request.id());
    let candidate = loaded(&mut load);
    assert_eq!(candidate.document, document(120));
    assert_eq!(candidate.revision, receipt.revision);
    assert_eq!(fs::read(project.path()).unwrap(), original);
    assert_eq!(
        fs::read_dir(&project.0).unwrap().count(),
        2,
        "only document and sidecar lock"
    );
}

#[test]
fn same_size_same_mtime_edit_is_a_conflict_and_deleted_files_are_distinct_revisions() {
    let project = Project::new();
    let service = project.service(2);
    let old = bytes(120);
    let external = bytes(240);
    assert_eq!(old.len(), external.len());
    fs::write(project.path(), &old).unwrap();
    let candidate = loaded(&mut service.load(TIMEOUT).unwrap());
    let previous_time = fs::metadata(project.path()).unwrap().modified().unwrap();
    fs::write(project.path(), &external).unwrap();
    File::options()
        .write(true)
        .open(project.path())
        .unwrap()
        .set_times(FileTimes::new().set_modified(previous_time))
        .unwrap();
    assert_eq!(
        fs::metadata(project.path()).unwrap().modified().unwrap(),
        previous_time
    );
    error(
        &mut service
            .save(bytes(480), token(), candidate.revision.clone(), TIMEOUT)
            .unwrap(),
        TimelineStorageError::Conflict,
    );
    assert_eq!(fs::read(project.path()).unwrap(), external);
    fs::remove_file(project.path()).unwrap();
    error(
        &mut service
            .save(bytes(480), token(), candidate.revision, TIMEOUT)
            .unwrap(),
        TimelineStorageError::Conflict,
    );
    let receipt = saved(
        &mut service
            .save(bytes(480), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    error(
        &mut service
            .save(bytes(960), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
        TimelineStorageError::Conflict,
    );
    assert_eq!(
        receipt.revision,
        TimelineFileRevision::Present(bytes(480).into())
    );
}

#[test]
fn independent_workers_do_not_silently_overwrite_one_anothers_revision() {
    let project = Project::new();
    let a = project.service(2);
    let b = project.service(2);
    let mut first = a
        .save(bytes(120), token(), TimelineFileRevision::Missing, TIMEOUT)
        .unwrap();
    let mut second = b
        .save(bytes(240), token(), TimelineFileRevision::Missing, TIMEOUT)
        .unwrap();
    let results = [wait(&mut first), wait(&mut second)];
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Ok(TimelineStorageResult::Saved(_))))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(
                result,
                Err(TimelineStorageError::Busy | TimelineStorageError::Conflict)
            ))
            .count(),
        1
    );
    let actual = fs::read(project.path()).unwrap();
    assert!(actual == bytes(120) || actual == bytes(240));
}

#[test]
fn output_slots_and_input_capacity_remain_bounded_until_request_handles_drop() {
    let project = Project::new();
    let service = project.service(2);
    fs::write(project.path(), bytes(120)).unwrap();
    let mut first = service.load(TIMEOUT).unwrap();
    let mut second = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        service.load(TIMEOUT),
        Err(TimelineStorageError::Busy)
    ));
    loaded(&mut first);
    loaded(&mut second);
    assert!(
        matches!(service.load(TIMEOUT), Err(TimelineStorageError::Busy)),
        "consumed handles still reserve their slots"
    );
    drop(first);
    let mut third = service.load(TIMEOUT).unwrap();
    loaded(&mut third);
    let small = TimelineStorageService::new(
        project.path(),
        TimelineStorageLimits {
            max_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap();
    let mut oversized_capacity = Vec::with_capacity(257);
    oversized_capacity.extend_from_slice(b"tiny");
    assert!(matches!(
        small.save(
            oversized_capacity,
            token(),
            TimelineFileRevision::Missing,
            TIMEOUT
        ),
        Err(TimelineStorageError::ByteLimit)
    ));
    assert!(matches!(
        small.save(
            vec![],
            token(),
            TimelineFileRevision::Present(vec![0; 257].into()),
            TIMEOUT
        ),
        Err(TimelineStorageError::ByteLimit)
    ));
    File::create(project.path()).unwrap().set_len(257).unwrap();
    error(
        &mut small.load(TIMEOUT).unwrap(),
        TimelineStorageError::ByteLimit,
    );
}

#[test]
fn malformed_and_policy_rejected_loads_and_saves_preserve_the_file() {
    let project = Project::new();
    let service = project.service(2);
    let original = bytes(120);
    fs::write(project.path(), &original).unwrap();
    for bad in [
        vec![],
        b"wrong format".to_vec(),
        original[..original.len() - 1].to_vec(),
        {
            let mut bad = original.clone();
            bad.push(0);
            bad
        },
        {
            let mut bad = original.clone();
            bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
            bad
        },
    ] {
        let mut save = service
            .save(
                bad.clone(),
                token(),
                TimelineFileRevision::Present(original.clone().into()),
                TIMEOUT,
            )
            .unwrap();
        assert!(matches!(
            wait(&mut save),
            Err(TimelineStorageError::Codec(_))
        ));
        assert_eq!(fs::read(project.path()).unwrap(), original);
        fs::write(project.path(), &bad).unwrap();
        assert!(matches!(
            wait(&mut service.load(TIMEOUT).unwrap()),
            Err(TimelineStorageError::Codec(_))
        ));
        assert_eq!(fs::read(project.path()).unwrap(), bad);
        fs::write(project.path(), &original).unwrap();
    }
    let restrictive = TimelineStorageService::new(
        project.path(),
        TimelineStorageLimits {
            document: TimelineLimits {
                max_tracks: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        wait(&mut restrictive.load(TIMEOUT).unwrap()),
        Err(TimelineStorageError::Codec(TimelineError::LimitExceeded(
            "tracks"
        )))
    ));
}

#[test]
fn cancellation_and_zero_timeout_fence_results_without_replacing_bytes() {
    let project = Project::new();
    let service = project.service(2);
    let original = bytes(120);
    fs::write(project.path(), &original).unwrap();
    let mut cancelled = service.load(TIMEOUT).unwrap();
    assert_eq!(cancelled.cancel(), TimelineCancelOutcome::Cancelled);
    error(&mut cancelled, TimelineStorageError::Cancelled);
    drop(cancelled);
    let mut expired = service
        .save(
            bytes(240),
            token(),
            TimelineFileRevision::Present(original.clone().into()),
            Duration::ZERO,
        )
        .unwrap();
    error(&mut expired, TimelineStorageError::TimedOut);
    assert!(matches!(
        expired.poll(),
        Poll::Ready(Err(TimelineStorageError::Consumed))
    ));
    assert_eq!(fs::read(project.path()).unwrap(), original);
}

#[test]
fn old_load_and_save_results_require_host_generation_and_snapshot_checks() {
    let project = Project::new();
    let service = project.service(2);
    fs::write(project.path(), bytes(120)).unwrap();
    let mut old_load = service.load(TIMEOUT).unwrap();
    let old = loaded(&mut old_load);
    let mut editor = TimelineEditor::new(old.document.timeline, TimelineLimits::default()).unwrap();
    let snapshot = editor.snapshot_token();
    let mut save = service
        .save(bytes(120), snapshot.clone(), old.revision, TIMEOUT)
        .unwrap();
    drop(old_load);
    editor
        .execute(
            "edit after save submission",
            vec![TimelineEdit::RemoveEvent(u128::MAX)],
        )
        .unwrap();
    let receipt = saved(&mut save);
    assert_eq!(receipt.snapshot_token, snapshot);
    assert_ne!(
        receipt.snapshot_token,
        editor.snapshot_token(),
        "host must not confirm this old save as current"
    );
    assert_eq!(
        editor.timeline().event_count(),
        0,
        "worker did not replace live editor"
    );
    let mut new_load = service.load(TIMEOUT).unwrap();
    assert!(new_load.id() > save.id());
    let newer = loaded(&mut new_load);
    assert_eq!(newer.document.timeline.event_count(), 1);
    assert_eq!(
        editor.timeline().event_count(),
        0,
        "load candidate is not auto-adopted"
    );
}

#[test]
fn uncoordinated_readers_observe_only_complete_atomic_generations() {
    let project = Project::new();
    let service = project.service(1);
    let old = bytes(120);
    let new = bytes(240);
    fs::write(project.path(), &old).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let reader_barrier = barrier.clone();
    let path = project.path();
    let reader_old = old.clone();
    let reader_new = new.clone();
    let reader = std::thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..4096 {
            let actual = fs::read(&path).unwrap();
            assert!(
                actual == reader_old || actual == reader_new,
                "partial generation"
            );
        }
    });
    barrier.wait();
    let mut revision = TimelineFileRevision::Present(old.clone().into());
    for index in 0..16 {
        let next = if index % 2 == 0 {
            new.clone()
        } else {
            old.clone()
        };
        revision = saved(&mut service.save(next, token(), revision, TIMEOUT).unwrap()).revision;
    }
    reader.join().unwrap();
    assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
}

#[test]
fn sidecar_lock_errors_missing_parent_and_missing_file_are_explicit() {
    let project = Project::new();
    let service = project.service(2);
    error(
        &mut service.load(TIMEOUT).unwrap(),
        TimelineStorageError::Missing,
    );
    let lock = File::options()
        .read(true)
        .write(true)
        .open(project.0.join("camera.timeline.prism-timeline.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    error(
        &mut service
            .save(bytes(120), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
        TimelineStorageError::Busy,
    );
    assert!(!project.path().exists());
    drop(lock);
    saved(
        &mut service
            .save(bytes(120), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    let missing = TimelineStorageService::new(
        project.0.join("not-created/file.timeline"),
        TimelineStorageLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        wait(&mut missing.load(TIMEOUT).unwrap()),
        Err(TimelineStorageError::Io {
            operation: "resolve parent",
            kind: std::io::ErrorKind::NotFound
        })
    ));
    assert!(matches!(
        TimelineStorageService::new("relative.timeline", TimelineStorageLimits::default()),
        Err(TimelineStorageError::InvalidConfiguration)
    ));
}

#[test]
fn independent_process_loads_only_real_saved_document_bytes() {
    const CHILD: &str = "PRISM_TIMELINE_STORAGE_CHILD_FILE";
    if let Some(path) = std::env::var_os(CHILD) {
        let service =
            TimelineStorageService::new(PathBuf::from(path), TimelineStorageLimits::default())
                .unwrap();
        let candidate = loaded(&mut service.load(TIMEOUT).unwrap());
        assert_eq!(candidate.document, document(120));
        saved(
            &mut service
                .save(bytes(240), token(), candidate.revision, TIMEOUT)
                .unwrap(),
        );
        return;
    }
    let project = Project::new();
    saved(
        &mut project
            .service(1)
            .save(bytes(120), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
    );
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("independent_process_loads_only_real_saved_document_bytes")
        .env(CHILD, project.path())
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
}

#[cfg(unix)]
#[test]
fn target_and_lock_symlinks_or_directories_are_rejected() {
    use std::os::unix::fs::symlink;
    let project = Project::new();
    let outside = Project::new();
    fs::write(outside.path(), bytes(120)).unwrap();
    symlink(outside.path(), project.path()).unwrap();
    let service = project.service(1);
    error(
        &mut service.load(TIMEOUT).unwrap(),
        TimelineStorageError::UnsafePath,
    );
    fs::remove_file(project.path()).unwrap();
    fs::create_dir(project.path()).unwrap();
    error(
        &mut service
            .save(bytes(240), token(), TimelineFileRevision::Missing, TIMEOUT)
            .unwrap(),
        TimelineStorageError::UnsafePath,
    );
    fs::remove_dir(project.path()).unwrap();
    fs::remove_file(project.0.join("camera.timeline.prism-timeline.lock")).unwrap();
    symlink(
        outside.path(),
        project.0.join("camera.timeline.prism-timeline.lock"),
    )
    .unwrap();
    error(
        &mut service.load(TIMEOUT).unwrap(),
        TimelineStorageError::UnsafePath,
    );
    assert_eq!(fs::read(outside.path()).unwrap(), bytes(120));
}
