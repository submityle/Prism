//! Actual project publication through the shared document worker and OFPA IO.
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_timeline::{TimelineCancelOutcome, TimelineFileRevision, TimelineStorageError};
use prism_editor_world::{
    project::*, ActorFileIdentity, ActorIoLimits, ActorRevisionBase, ActorSaveRequest,
    EditorWorldLimits, FileActorIoProvider,
};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::Poll,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(10);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-world-project-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => panic!("private test directory: {e}"),
            }
        }
    }
    fn path(&self) -> PathBuf {
        self.0.join("world.prworld")
    }
    fn service(&self, slots: usize) -> WorldProjectStorageService {
        WorldProjectStorageService::new(self.path(), id(7), slots, ProjectLimits::default())
            .unwrap()
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn id(n: u128) -> StableGuid {
    StableGuid::from_u128(n)
}
fn cell() -> CellCoord {
    CellCoord::new(-12, i32::MIN, i32::MAX)
}
fn project() -> WorldProject {
    let mut p = WorldProject::new(id(7)).unwrap();
    p.apply(
        0,
        &[
            ProjectEdit::Layer {
                id: DataLayerId::new(0),
                value: Some(ProjectLayer {
                    id: DataLayerId::new(0),
                    name: "真实项目层".into(),
                    parent: None,
                    visible: true,
                    editable: true,
                }),
            },
            ProjectEdit::Cell {
                cell: cell(),
                layers: Some([DataLayerId::new(0)].into()),
            },
            ProjectEdit::Actor {
                actor: id(99),
                cell: Some(cell()),
            },
        ],
        ProjectLimits::default(),
        8192,
    )
    .unwrap();
    p
}
fn wait(
    request: &mut WorldProjectStorageRequest,
) -> Result<WorldProjectStorageResult, TimelineStorageError> {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(output) = request.poll() {
            return output;
        }
        assert!(Instant::now() < deadline, "shared worker did not finish");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn save(
    service: &WorldProjectStorageService,
    p: &WorldProject,
    expected: TimelineFileRevision,
) -> WorldProjectSaveReceipt {
    let mut request = service.save(p, expected, TIMEOUT).unwrap();
    let WorldProjectStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
        panic!("save output")
    };
    receipt
}

#[test]
fn actual_save_restart_load_and_materialization_use_complete_real_membership() {
    let files = Files::new();
    let p = project();
    let receipt = save(&files.service(2), &p, TimelineFileRevision::Missing);
    assert!(p.matches_token(&receipt.snapshot_token));
    let bytes = fs::read(files.path()).unwrap();
    assert_eq!(bytes, p.encode(ProjectLimits::default()).unwrap());
    assert_eq!(
        receipt.revision,
        TimelineFileRevision::Present(Arc::from(bytes))
    );
    let reopened = files.service(2);
    let mut request = reopened.load(TIMEOUT).unwrap();
    let WorldProjectStorageResult::Loaded(candidate) = wait(&mut request).unwrap() else {
        panic!("load output")
    };
    assert_eq!(candidate.document, p);
    assert_eq!(candidate.revision, receipt.revision);
    assert!(!candidate.document.matches_token(&receipt.snapshot_token));
    let world = candidate
        .document
        .materialize(
            EditorWorldLimits::editor_default(),
            ProjectLimits::default(),
        )
        .unwrap();
    assert_eq!(
        world.actor_record(id(99)).unwrap().identity().cell(),
        cell()
    );
    assert_eq!(
        world
            .model()
            .cell(cell())
            .unwrap()
            .layers()
            .collect::<Vec<_>>(),
        [DataLayerId::new(0)]
    );
    assert!(matches!(
        request.poll(),
        Poll::Ready(Err(TimelineStorageError::Consumed))
    ));
}

#[test]
fn stale_file_revision_and_same_length_external_change_preserve_actual_disk_bytes() {
    let files = Files::new();
    let service = files.service(2);
    let mut p = project();
    let receipt = save(&service, &p, TimelineFileRevision::Missing);
    let mut external = fs::read(files.path()).unwrap();
    external[36] ^= 1;
    fs::write(files.path(), &external).unwrap();
    p.apply(
        p.revision(),
        &[ProjectEdit::Actor {
            actor: id(99),
            cell: None,
        }],
        ProjectLimits::default(),
        8192,
    )
    .unwrap();
    let mut request = service.save(&p, receipt.revision, TIMEOUT).unwrap();
    assert_eq!(
        wait(&mut request).unwrap_err(),
        TimelineStorageError::Conflict
    );
    assert_eq!(fs::read(files.path()).unwrap(), external);
    assert!(!p.matches_token(&receipt.snapshot_token));
}

#[test]
fn codec_domain_wrong_world_corruption_and_file_budget_fail_without_adopting_host() {
    let files = Files::new();
    let service = files.service(2);
    let p = project();
    let token = p.token();
    fs::write(files.path(), b"corrupt").unwrap();
    let mut request = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::DocumentCodec {
            domain: "world-project",
            ..
        })
    ));
    drop(request);
    let foreign = WorldProject::new(id(88)).unwrap();
    fs::write(
        files.path(),
        foreign.encode(ProjectLimits::default()).unwrap(),
    )
    .unwrap();
    let mut request = service.load(TIMEOUT).unwrap();
    assert!(matches!(
        wait(&mut request),
        Err(TimelineStorageError::DocumentCodec {
            domain: "world-project",
            ..
        })
    ));
    assert!(matches!(
        service.save(&foreign, TimelineFileRevision::Missing, TIMEOUT),
        Err(WorldProjectStorageError::Project(ProjectError::Invalid(_)))
    ));
    assert!(p.matches_token(&token));
    let bounded = WorldProjectStorageService::new(
        files.path(),
        id(7),
        2,
        ProjectLimits {
            max_bytes: 64,
            ..ProjectLimits::default()
        },
    )
    .unwrap();
    fs::write(files.path(), p.encode(ProjectLimits::default()).unwrap()).unwrap();
    let mut request = bounded.load(TIMEOUT).unwrap();
    assert_eq!(
        wait(&mut request).unwrap_err(),
        TimelineStorageError::ByteLimit
    );
    assert!(matches!(
        bounded.save(&p, TimelineFileRevision::Missing, TIMEOUT),
        Err(WorldProjectStorageError::Project(ProjectError::Budget(_)))
    ));
}

#[test]
fn retained_handles_bound_requests_and_cancel_has_real_publication_fence() {
    let files = Files::new();
    let service = files.service(1);
    let p = project();
    let mut request = service
        .save(&p, TimelineFileRevision::Missing, TIMEOUT)
        .unwrap();
    assert!(matches!(
        service.load(TIMEOUT),
        Err(TimelineStorageError::Busy)
    ));
    match request.cancel() {
        TimelineCancelOutcome::Cancelled => {
            assert_eq!(
                wait(&mut request).unwrap_err(),
                TimelineStorageError::Cancelled
            );
            assert!(
                !files.path().exists(),
                "cancelled operation must never publish"
            );
        }
        TimelineCancelOutcome::TooLate => {
            assert!(matches!(
                wait(&mut request).unwrap(),
                WorldProjectStorageResult::Saved(_)
            ));
            assert_eq!(
                fs::read(files.path()).unwrap(),
                p.encode(ProjectLimits::default()).unwrap()
            );
        }
    }
    assert!(
        WorldProjectStorageService::new(files.path(), id(7), 0, ProjectLimits::default()).is_err()
    );
    assert!(
        WorldProjectStorageService::new(files.path(), id(0), 1, ProjectLimits::default()).is_err()
    );
    assert!(WorldProjectStorageService::new(
        "relative.prworld",
        id(7),
        1,
        ProjectLimits::default()
    )
    .is_err());
}

#[test]
fn edit_undo_and_independent_decode_cannot_accept_an_old_save_receipt() {
    let files = Files::new();
    let service = files.service(2);
    let mut p = project();
    let receipt = save(&service, &p, TimelineFileRevision::Missing);
    let revision = p.revision();
    let inverse = p
        .apply(
            revision,
            &[ProjectEdit::Actor {
                actor: id(99),
                cell: None,
            }],
            ProjectLimits::default(),
            8192,
        )
        .unwrap();
    inverse.undo(&mut p, ProjectLimits::default()).unwrap();
    assert!(!p.matches_token(&receipt.snapshot_token));
    let current = save(&service, &p, receipt.revision);
    assert!(p.matches_token(&current.snapshot_token));
    let decoded =
        WorldProject::decode(&fs::read(files.path()).unwrap(), ProjectLimits::default()).unwrap();
    assert_eq!(decoded, p);
    assert!(!decoded.matches_token(&current.snapshot_token));
}

fn run<T>(mut future: std::pin::Pin<Box<dyn Future<Output = T> + Send>>) -> T {
    struct Notify(std::thread::Thread);
    impl std::task::Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(Notify(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        assert!(Instant::now() < deadline, "OFPA worker did not finish");
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[test]
fn reloaded_membership_resolves_actual_existing_ofpa_actor_file() {
    let files = Files::new();
    let p = project();
    let provider =
        FileActorIoProvider::new(&files.0, ActorIoLimits::new(1024, 2, 2048).unwrap()).unwrap();
    let identity = ActorFileIdentity::new(id(99), cell()).unwrap();
    let payload: Arc<[u8]> = Arc::from(&b"actual opaque project-owned actor bytes"[..]);
    let request = ActorSaveRequest::new(identity.clone(), 3, ActorRevisionBase::Missing).unwrap();
    run(provider
        .save_actor(request, payload.clone(), TIMEOUT)
        .unwrap())
    .unwrap();
    save(&files.service(2), &p, TimelineFileRevision::Missing);
    let service = files.service(2);
    let mut request = service.load(TIMEOUT).unwrap();
    let WorldProjectStorageResult::Loaded(candidate) = wait(&mut request).unwrap() else {
        panic!("load output")
    };
    let restored = candidate
        .document
        .materialize(
            EditorWorldLimits::editor_default(),
            ProjectLimits::default(),
        )
        .unwrap();
    let restored_identity = restored.actor_record(id(99)).unwrap().identity();
    assert_eq!(restored_identity, &identity);
    let reopened =
        FileActorIoProvider::new(&files.0, ActorIoLimits::new(1024, 2, 2048).unwrap()).unwrap();
    let snapshot = run(reopened
        .load_actor(restored_identity.clone(), 1024, TIMEOUT)
        .unwrap())
    .unwrap();
    assert_eq!(snapshot.revision(), 3);
    assert_eq!(snapshot.bytes(), payload.as_ref());
    assert!(
        restored.actor_payload(id(99)).is_none(),
        "membership load never fabricates residency"
    );
}
