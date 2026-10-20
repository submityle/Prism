//! Actual project files through the shared async CAS/publication worker.
extern crate alloc;
use alloc::sync::Arc;
use prism_editor_timeline::*;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    task::Poll,
    time::{Duration, Instant},
};

static SERIAL: AtomicU64 = AtomicU64::new(1);
const TIMEOUT: Duration = Duration::from_secs(10);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-sequence-file-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self) -> PathBuf {
        self.0.join("project.prseq")
    }
    fn service(&self, slots: usize) -> SequenceProjectStorageService {
        SequenceProjectStorageService::new(self.path(), slots, SequenceDocumentLimits::default())
            .unwrap()
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn editor() -> SequenceProjectEditor {
    let mut editor = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits::default(),
    )
    .unwrap();
    let mut timeline = Timeline::default();
    timeline
        .add_event(TimelineEvent {
            id: 8,
            tick: 3,
            name: "real-provider-event".into(),
        })
        .unwrap();
    editor
        .execute(SequenceProjectCommand {
            label: "project".into(),
            edits: vec![
                SequenceProjectEdit::PutDocument {
                    id: SequenceId(12),
                    document: TimelineDocument {
                        ticks_per_second: 24,
                        timeline,
                    },
                },
                SequenceProjectEdit::SetRoot(Some(SequenceId(12))),
            ],
        })
        .unwrap();
    editor
}
fn wait(
    request: &mut SequenceProjectStorageRequest,
) -> Result<SequenceProjectStorageResult, TimelineStorageError> {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(result) = request.poll() {
            return result;
        }
        assert!(Instant::now() < deadline, "worker did not complete");
        std::thread::yield_now();
    }
}
fn admit(
    mut submit: impl FnMut() -> Result<SequenceProjectStorageRequest, TimelineStorageError>,
) -> SequenceProjectStorageRequest {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match submit() {
            Ok(request) => return request,
            Err(TimelineStorageError::Busy) => {
                assert!(Instant::now() < deadline, "worker slot did not retire");
                std::thread::yield_now();
            }
            Err(error) => panic!("{error}"),
        }
    }
}
fn save(
    service: &SequenceProjectStorageService,
    editor: &SequenceProjectEditor,
    expected: TimelineFileRevision,
) -> SequenceProjectSaveReceipt {
    let bytes = editor.document().encode(editor.limits()).unwrap();
    let mut request = service
        .save(bytes, editor.snapshot_token(), expected, TIMEOUT)
        .unwrap();
    let SequenceProjectStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
        panic!("save result")
    };
    receipt
}

#[test]
fn actual_async_save_load_full_project_roundtrip_and_provider_registry() {
    let files = Files::new();
    let service = files.service(2);
    let editor = editor();
    let receipt = save(&service, &editor, TimelineFileRevision::Missing);
    assert_eq!(receipt.snapshot_token, editor.snapshot_token());
    assert!(matches!(
        receipt.durability,
        TimelineSaveDurability::Synced | TimelineSaveDurability::Uncertain { .. }
    ));
    let disk = fs::read(files.path()).unwrap();
    assert_eq!(disk, editor.document().encode(editor.limits()).unwrap());
    let mut request = service.load(TIMEOUT).unwrap();
    let SequenceProjectStorageResult::Loaded(candidate) = wait(&mut request).unwrap() else {
        panic!("load result")
    };
    assert_eq!(&candidate.document, editor.document());
    assert_eq!(candidate.revision, receipt.revision);
    let registry = candidate
        .document
        .registry(SequenceLimits::default())
        .unwrap();
    let mut player = SequencePlayer::new(
        &registry,
        candidate.document.root().unwrap(),
        SequenceRange::new(0, 24).unwrap(),
        PlaybackMode::Once,
    )
    .unwrap();
    player.play();
    let frame = player.advance(Duration::from_millis(125)).unwrap();
    assert_eq!(frame.events[0].event.name, "real-provider-event");
    assert!(matches!(
        request.poll(),
        Poll::Ready(Err(TimelineStorageError::Consumed))
    ));
}

#[test]
fn exact_same_length_external_bytes_conflict_without_replacing_file() {
    let files = Files::new();
    let service = files.service(2);
    let editor = editor();
    let receipt = save(&service, &editor, TimelineFileRevision::Missing);
    let mut external = fs::read(files.path()).unwrap();
    let last = external.len() - 1;
    external[last] ^= 1;
    fs::write(files.path(), &external).unwrap();
    let mut request = service
        .save(
            editor.document().encode(editor.limits()).unwrap(),
            editor.snapshot_token(),
            receipt.revision,
            TIMEOUT,
        )
        .unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::Conflict)
    ));
    assert_eq!(fs::read(files.path()).unwrap(), external);
}

#[test]
fn corrupt_load_and_invalid_owned_save_have_precise_codec_error_and_preserve_target() {
    let files = Files::new();
    let service = files.service(2);
    let editor = editor();
    fs::write(files.path(), b"broken project").unwrap();
    let mut load = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        wait(&mut load),
        Err(TimelineStorageError::SequenceCodec(_))
    ));
    drop(load);
    let mut save = service
        .save(
            b"still broken".to_vec(),
            editor.snapshot_token(),
            TimelineFileRevision::Present(Arc::from(&b"broken project"[..])),
            TIMEOUT,
        )
        .unwrap();
    assert!(matches!(
        wait(&mut save),
        Err(TimelineStorageError::SequenceCodec(_))
    ));
    assert_eq!(fs::read(files.path()).unwrap(), b"broken project");
}

#[test]
fn missing_and_existing_empty_files_are_distinct_cas_revisions() {
    let files = Files::new();
    let service = files.service(2);
    let editor = editor();
    let mut missing = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        wait(&mut missing),
        Err(TimelineStorageError::Missing)
    ));
    drop(missing);
    fs::write(files.path(), []).unwrap();
    let mut request = service
        .save(
            editor.document().encode(editor.limits()).unwrap(),
            editor.snapshot_token(),
            TimelineFileRevision::Missing,
            TIMEOUT,
        )
        .unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::Conflict)
    ));
    drop(request);
    save(
        &service,
        &editor,
        TimelineFileRevision::Present(Arc::from([])),
    );
}

#[test]
fn retained_output_capacity_and_stale_host_tokens_are_visible() {
    let files = Files::new();
    let service = files.service(1);
    let mut editor = editor();
    let receipt = save(&service, &editor, TimelineFileRevision::Missing);
    let mut load = admit(|| service.load(TIMEOUT));
    let old_id = load.id();
    let _candidate = wait(&mut load).unwrap();
    assert!(matches!(
        service.load(TIMEOUT),
        Err(TimelineStorageError::Busy)
    ));
    drop(load);
    let mut newer = admit(|| service.load(TIMEOUT));
    assert!(newer.id() > old_id);
    wait(&mut newer).unwrap();
    drop(newer);
    editor.undo().unwrap();
    assert_ne!(receipt.snapshot_token, editor.snapshot_token());
    let bytes = fs::read(files.path()).unwrap();
    assert_eq!(
        receipt.revision,
        TimelineFileRevision::Present(bytes.into())
    );
}

#[test]
fn timeout_and_explicit_cancel_fence_late_worker_output_and_publication() {
    let files = Files::new();
    let service = files.service(2);
    let editor = editor();
    let receipt = save(&service, &editor, TimelineFileRevision::Missing);
    let original = fs::read(files.path()).unwrap();
    let mut timeout = service
        .save(
            original.clone(),
            editor.snapshot_token(),
            receipt.revision.clone(),
            Duration::ZERO,
        )
        .unwrap();
    assert!(matches!(
        wait(&mut timeout),
        Err(TimelineStorageError::TimedOut)
    ));
    drop(timeout);
    // Zero deadline ensures the worker cannot cross publication before cancellation.
    let mut cancelled = admit(|| {
        service.save(
            original.clone(),
            editor.snapshot_token(),
            receipt.revision.clone(),
            Duration::ZERO,
        )
    });
    assert_eq!(cancelled.cancel(), TimelineCancelOutcome::Cancelled);
    assert!(matches!(
        wait(&mut cancelled),
        Err(TimelineStorageError::Cancelled)
    ));
    drop(cancelled);
    // Serial worker barrier: completion of this load proves earlier jobs retired.
    let deadline = Instant::now() + TIMEOUT;
    let mut barrier = loop {
        match service.load(TIMEOUT) {
            Ok(request) => break request,
            Err(TimelineStorageError::Busy) => {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            Err(error) => panic!("{error}"),
        }
    };
    wait(&mut barrier).unwrap();
    assert_eq!(fs::read(files.path()).unwrap(), original);
    assert!(!fs::read_dir(&files.0).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".stage")));
}

#[test]
fn project_policy_byte_admission_unsafe_paths_and_cooperating_lock_are_enforced() {
    let files = Files::new();
    let editor = editor();
    assert!(SequenceProjectStorageService::new("relative", 2, editor.limits()).is_err());
    assert!(SequenceProjectStorageService::new(files.path(), 0, editor.limits()).is_err());
    let limits = SequenceDocumentLimits {
        max_bytes: 128,
        ..editor.limits()
    };
    let service = SequenceProjectStorageService::new(files.path(), 2, limits).unwrap();
    let mut bytes = Vec::with_capacity(129);
    bytes.push(0);
    assert!(matches!(
        service.save(
            bytes,
            editor.snapshot_token(),
            TimelineFileRevision::Missing,
            TIMEOUT
        ),
        Err(TimelineStorageError::ByteLimit)
    ));
    let lock = fs::OpenOptions::new()
        .write(true)
        .read(true)
        .create_new(true)
        .open(files.0.join("project.prseq.prism-timeline.lock"))
        .unwrap();
    lock.lock().unwrap();
    let mut request = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::Busy)
    ));
}

#[cfg(unix)]
#[test]
fn symlink_target_is_never_replaced() {
    let files = Files::new();
    let real = files.0.join("unowned");
    fs::write(&real, b"preserve").unwrap();
    std::os::unix::fs::symlink(&real, files.path()).unwrap();
    let service = files.service(2);
    let editor = editor();
    let mut request = service
        .save(
            editor.document().encode(editor.limits()).unwrap(),
            editor.snapshot_token(),
            TimelineFileRevision::Missing,
            TIMEOUT,
        )
        .unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::UnsafePath)
    ));
    assert_eq!(fs::read(real).unwrap(), b"preserve");
}
