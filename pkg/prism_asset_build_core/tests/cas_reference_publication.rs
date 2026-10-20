use prism_asset_build_core::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, CasError, ContentHash, PublishError,
    PublishedManifest, RevisionStore,
};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const BASELINE_BYTES: &[u8] = b"previous complete revision";
const CHILD_ROOT_ENV: &str = "PRISM_CAS_REFERENCE_LEASE_CHILD_ROOT";
const CHILD_READY: &str = "PRISM_CAS_REFERENCE_LEASE_READY";
const CHILD_TIMEOUT: Duration = Duration::from_secs(10);

struct Fixture {
    store: RevisionStore,
    baseline: PublishedManifest,
    // Keep the directory alive until the store and baseline have been dropped.
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = RevisionStore::open(root.path()).unwrap();
        let mut transaction = store.begin(BuildRevision(0), None);
        transaction
            .stage(
                AssetId(100),
                descriptor(100, BASELINE_BYTES, vec![]),
                BASELINE_BYTES.to_vec(),
            )
            .unwrap();
        let baseline = transaction.commit().unwrap();
        Self {
            store,
            baseline,
            root,
        }
    }

    fn assert_baseline(&self) {
        assert_eq!(
            fs::read(self.root.path().join("CURRENT")).unwrap(),
            0u64.to_le_bytes()
        );
        assert_publication(&self.store, &self.baseline);
        // Reopening exercises recovery after a rejected CAS reference, which
        // may have left Prepare in the journal without replacing CURRENT.
        let reopened = RevisionStore::open(self.root.path()).unwrap();
        assert_publication(&reopened, &self.baseline);
        assert_eq!(
            reopened
                .cas()
                .read(ContentHash::digest(BASELINE_BYTES))
                .unwrap(),
            BASELINE_BYTES
        );
    }
}

fn descriptor(key: u8, bytes: &[u8], dependencies: Vec<ArtifactKey>) -> ArtifactDescriptor {
    ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 4, 1, 16, dependencies, bytes)
}

fn assert_publication(store: &RevisionStore, published: &PublishedManifest) {
    assert_eq!(
        store.current_revision().unwrap(),
        Some(published.manifest.revision)
    );
    assert_eq!(
        store.manifest(published.manifest.revision).unwrap(),
        published.manifest
    );
    assert_eq!(
        ContentHash::digest(&published.manifest.canonical_bytes()),
        published.manifest_hash
    );
    for descriptor in published.manifest.entries.values() {
        assert_eq!(
            store.cas().verify(descriptor.content_hash).unwrap(),
            descriptor.byte_len
        );
    }
}

#[test]
fn mixed_owned_and_existing_cas_outputs_publish_one_complete_dependency_closure() {
    let fixture = Fixture::new();
    let leaf_bytes = b"owned leaf";
    let reference_bytes = b"existing CAS middle";
    let root_bytes = b"owned root";
    let leaf = descriptor(1, leaf_bytes, vec![]);
    let reference = descriptor(2, reference_bytes, vec![leaf.key]);
    let root = descriptor(3, root_bytes, vec![reference.key]);
    fixture.store.cas().put(reference_bytes).unwrap();
    let mut transaction = fixture
        .store
        .begin(BuildRevision(1), Some(BuildRevision(0)));
    // Deliberately stage in reverse dependency order.
    transaction
        .stage(AssetId(3), root.clone(), root_bytes.to_vec())
        .unwrap();
    transaction
        .stage_existing(AssetId(2), reference.clone())
        .unwrap();
    transaction
        .stage(AssetId(1), leaf.clone(), leaf_bytes.to_vec())
        .unwrap();
    assert!(!fixture.store.cas().blob_path(leaf.content_hash).exists());
    assert!(!fixture.store.cas().blob_path(root.content_hash).exists());

    let published = transaction.commit().unwrap();
    assert_eq!(
        published.manifest.entries,
        BTreeMap::from([
            (AssetId(1), leaf.clone()),
            (AssetId(2), reference.clone()),
            (AssetId(3), root.clone()),
        ])
    );
    for (descriptor, bytes) in [
        (&leaf, leaf_bytes.as_slice()),
        (&reference, reference_bytes.as_slice()),
        (&root, root_bytes.as_slice()),
    ] {
        assert_eq!(
            fixture.store.cas().read(descriptor.content_hash).unwrap(),
            bytes
        );
    }
    assert_publication(&fixture.store, &published);
    let reopened = RevisionStore::open(fixture.root.path()).unwrap();
    assert_publication(&reopened, &published);
    assert_eq!(
        reopened.manifest(BuildRevision(0)).unwrap(),
        fixture.baseline.manifest
    );
}

#[test]
fn duplicate_asset_preserves_first_staging_across_both_payload_sources() {
    for first_existing in [false, true] {
        for duplicate_existing in [false, true] {
            let fixture = Fixture::new();
            let first_bytes = b"first staged payload";
            let duplicate_bytes = b"rejected replacement payload";
            let first = descriptor(1, first_bytes, vec![]);
            let duplicate = descriptor(2, duplicate_bytes, vec![]);
            let mut transaction = fixture
                .store
                .begin(BuildRevision(1), Some(BuildRevision(0)));
            if first_existing {
                fixture.store.cas().put(first_bytes).unwrap();
                transaction
                    .stage_existing(AssetId(1), first.clone())
                    .unwrap();
            } else {
                transaction
                    .stage(AssetId(1), first.clone(), first_bytes.to_vec())
                    .unwrap();
            }
            let result = if duplicate_existing {
                // Intentionally absent: accepting this replacement would make
                // commit fail rather than silently preserve the first value.
                transaction.stage_existing(AssetId(1), duplicate.clone())
            } else {
                transaction.stage(AssetId(1), duplicate.clone(), duplicate_bytes.to_vec())
            };
            assert!(matches!(
                result,
                Err(PublishError::Invalid("duplicate staged AssetId"))
            ));

            let published = transaction.commit().unwrap();
            assert_eq!(
                published.manifest.entries,
                BTreeMap::from([(AssetId(1), first.clone())])
            );
            assert_eq!(
                fixture.store.cas().read(first.content_hash).unwrap(),
                first_bytes
            );
            assert!(!fixture
                .store
                .cas()
                .blob_path(duplicate.content_hash)
                .exists());
            assert_publication(&fixture.store, &published);
        }
    }
}

#[derive(Clone, Copy)]
enum ReferenceFailure {
    Missing,
    Tampered,
    WrongLength,
}

fn check_rejected_reference(failure: ReferenceFailure) {
    let fixture = Fixture::new();
    let bytes = vec![b'c'; 64 * 1024 + 17];
    let mut candidate = descriptor(1, &bytes, vec![]);
    let hash = fixture.store.cas().put(&bytes).unwrap();
    if matches!(failure, ReferenceFailure::WrongLength) {
        candidate.byte_len += 1;
    }
    let mut transaction = fixture
        .store
        .begin(BuildRevision(1), Some(BuildRevision(0)));
    transaction.stage_existing(AssetId(1), candidate).unwrap();
    let mut tampered = bytes.clone();
    *tampered.last_mut().unwrap() ^= 1;
    match failure {
        ReferenceFailure::Missing => {
            fs::remove_file(fixture.store.cas().blob_path(hash)).unwrap();
        }
        ReferenceFailure::Tampered => {
            // The unchanged prefix fills the streaming verification buffer.
            // Only hashing the tail, with length unchanged, can reject this.
            fs::write(fixture.store.cas().blob_path(hash), &tampered).unwrap();
        }
        ReferenceFailure::WrongLength => {}
    }
    let error = transaction.commit().unwrap_err();
    match failure {
        ReferenceFailure::Missing => assert!(matches!(
            error,
            PublishError::Cas(CasError::Io(ref error)) if error.kind() == io::ErrorKind::NotFound
        )),
        ReferenceFailure::Tampered => assert!(matches!(
            error,
            PublishError::Cas(CasError::Corrupt { expected, actual })
                if expected == hash && actual == ContentHash::digest(&tampered)
        )),
        ReferenceFailure::WrongLength => assert!(matches!(
            error,
            PublishError::DescriptorMismatch(AssetId(1))
        )),
    }
    fixture.assert_baseline();
}

#[test]
fn reference_removed_after_staging_is_rejected_without_replacing_current() {
    check_rejected_reference(ReferenceFailure::Missing);
}

#[test]
fn reference_tampered_after_staging_is_rejected_without_replacing_current() {
    check_rejected_reference(ReferenceFailure::Tampered);
}

#[test]
fn reference_with_wrong_declared_length_is_rejected_without_replacing_current() {
    check_rejected_reference(ReferenceFailure::WrongLength);
}

#[derive(Clone, Copy, Debug)]
enum ClosureFailure {
    MissingDependency,
    Cycle,
    DuplicateDependency,
    ConflictingKey,
}

#[test]
fn invalid_mixed_dependency_closure_never_writes_owned_payloads_to_cas() {
    for failure in [
        ClosureFailure::MissingDependency,
        ClosureFailure::Cycle,
        ClosureFailure::DuplicateDependency,
        ClosureFailure::ConflictingKey,
    ] {
        let fixture = Fixture::new();
        let owned_bytes = b"owned bytes must stay unstored";
        let existing_bytes = b"preexisting closure member";
        let mut owned = descriptor(1, owned_bytes, vec![]);
        let mut existing = descriptor(2, existing_bytes, vec![]);
        let missing = ArtifactKey([9; 32]);
        match failure {
            ClosureFailure::MissingDependency => existing.dependencies = vec![missing],
            ClosureFailure::Cycle => {
                owned.dependencies = vec![existing.key];
                existing.dependencies = vec![owned.key];
            }
            ClosureFailure::DuplicateDependency => {
                existing.dependencies = vec![owned.key, owned.key];
            }
            ClosureFailure::ConflictingKey => existing.key = owned.key,
        }
        fixture.store.cas().put(existing_bytes).unwrap();
        let mut transaction = fixture
            .store
            .begin(BuildRevision(1), Some(BuildRevision(0)));
        // The owned asset sorts first. Validating closure only while walking
        // and writing staged entries would leak these bytes into CAS.
        transaction
            .stage(AssetId(1), owned.clone(), owned_bytes.to_vec())
            .unwrap();
        transaction
            .stage_existing(AssetId(2), existing.clone())
            .unwrap();
        let error = transaction.commit().unwrap_err();
        match failure {
            ClosureFailure::MissingDependency => assert!(matches!(
                error,
                PublishError::MissingDependency { asset: AssetId(2), dependency }
                    if dependency == missing
            )),
            ClosureFailure::Cycle => assert!(matches!(
                error,
                PublishError::DependencyCycle(keys) if keys == vec![owned.key, existing.key]
            )),
            ClosureFailure::DuplicateDependency => assert!(matches!(
                error,
                PublishError::Invalid("artifact descriptor contains a duplicate dependency")
            )),
            ClosureFailure::ConflictingKey => assert!(matches!(
                error,
                PublishError::Invalid("one ArtifactKey has conflicting staged descriptors")
            )),
        }
        assert!(
            !fixture.store.cas().blob_path(owned.content_hash).exists(),
            "owned CAS write leaked for {failure:?}"
        );
        assert_eq!(
            fixture.store.cas().read(existing.content_hash).unwrap(),
            existing_bytes
        );
        fixture.assert_baseline();
    }
}

fn assert_deferred(store: &RevisionStore, orphan: ContentHash) {
    let report = store.sweep_unreachable_cas(Duration::ZERO).unwrap();
    assert!(report.deferred_due_to_active_build);
    assert!(!report.deferred_due_to_pending_transaction);
    assert_eq!(report.deleted, 0);
    assert!(store.cas().blob_path(orphan).exists());
}

fn assert_gc_resumed(fixture: &Fixture, store: &RevisionStore, orphan: ContentHash) {
    let report = store.sweep_unreachable_cas(Duration::ZERO).unwrap();
    assert!(!report.deferred_due_to_active_build);
    assert!(!report.deferred_due_to_pending_transaction);
    assert_eq!(report.deleted, 1);
    assert!(!store.cas().blob_path(orphan).exists());
    assert_publication(store, &fixture.baseline);
}

#[test]
fn gc_resumes_only_after_every_shared_build_output_lease_is_dropped() {
    let fixture = Fixture::new();
    let builder = RevisionStore::open(fixture.root.path()).unwrap();
    let independent_builder = RevisionStore::open(fixture.root.path()).unwrap();
    let sweeper = RevisionStore::open(fixture.root.path()).unwrap();
    let first = builder.protect_build_outputs().unwrap();
    let second = builder.protect_build_outputs().unwrap();
    let third = independent_builder.protect_build_outputs().unwrap();
    let orphan = builder.cas().put(b"live unpublished build output").unwrap();
    // The guards must outlive the handles from which they were acquired.
    drop(builder);
    drop(independent_builder);
    assert_deferred(&sweeper, orphan);
    drop(first);
    assert_deferred(&sweeper, orphan);
    drop(third);
    assert_deferred(&sweeper, orphan);
    drop(second);
    assert_gc_resumed(&fixture, &sweeper, orphan);
}

// Kill and reap before TempDir cleanup, including on a readiness timeout or
// failed parent assertion. Drain stdout concurrently so the child cannot block
// on a full pipe while the parent waits for its readiness marker.
struct LeaseChild {
    child: Child,
    lines: Receiver<String>,
    reader: Option<JoinHandle<()>>,
}

impl LeaseChild {
    fn spawn(root: &Path) -> Self {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "subprocess_output_lease_holder",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ROOT_ENV, root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.unwrap_or_else(|error| format!("child stdout error: {error}"));
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            lines,
            reader: Some(reader),
        }
    }

    fn await_lease(&mut self) {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        let mut transcript = String::new();
        loop {
            let line = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| {
                    panic!("lease child did not become ready: {error}\n{transcript}")
                });
            transcript.push_str(&line);
            transcript.push('\n');
            if line.ends_with(CHILD_READY) {
                assert!(self.child.try_wait().unwrap().is_none());
                return;
            }
        }
    }

    fn kill_and_reap(&mut self) {
        assert!(self.child.try_wait().unwrap().is_none());
        self.child.kill().unwrap();
        assert!(!self.child.wait().unwrap().success());
        self.reader.take().unwrap().join().unwrap();
    }
}

impl Drop for LeaseChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[test]
fn killed_independent_process_releases_output_lease_and_gc_preserves_baseline() {
    let fixture = Fixture::new();
    let orphan = fixture.store.cas().put(b"child-protected orphan").unwrap();
    let sweeper = RevisionStore::open(fixture.root.path()).unwrap();
    let mut child = LeaseChild::spawn(fixture.root.path());
    child.await_lease();
    assert_deferred(&sweeper, orphan);
    child.kill_and_reap();
    assert_gc_resumed(&fixture, &sweeper, orphan);
    fixture.assert_baseline();
}

// Only the parent invocation supplies this binary-specific helper environment.
#[test]
fn subprocess_output_lease_holder() {
    let Some(root) = std::env::var_os(CHILD_ROOT_ENV) else {
        return;
    };
    let store = RevisionStore::open(root).unwrap();
    let lease = store.protect_build_outputs().unwrap();
    println!("{CHILD_READY}");
    io::stdout().flush().unwrap();
    let mut command = String::new();
    io::stdin().read_line(&mut command).unwrap();
    // Parent kills the process while it is blocked above, without running Drop.
    drop(lease);
    panic!("lease child was released without being killed: {command:?}");
}
