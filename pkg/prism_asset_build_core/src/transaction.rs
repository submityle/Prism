use crate::cas::{CasError, ContentAddressedStore};
use crate::types::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, Manifest, ManifestEntry,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum PublishError {
    Io(io::Error),
    Cas(CasError),
    Invalid(&'static str),
    Conflict {
        expected: Option<BuildRevision>,
        actual: Option<BuildRevision>,
    },
    MissingPayload(AssetId),
    DescriptorMismatch(AssetId),
    MissingDependency {
        asset: AssetId,
        dependency: ArtifactKey,
    },
    DependencyCycle(Vec<ArtifactKey>),
}
impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PublishError {}
impl From<io::Error> for PublishError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<CasError> for PublishError {
    fn from(value: CasError) -> Self {
        Self::Cas(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedManifest {
    pub manifest: Manifest,
    pub manifest_hash: crate::types::ContentHash,
}

/// Local revision store. Publication becomes visible at atomic replacement of
/// CURRENT. A failure while syncing the containing directory after that rename
/// has an indeterminate durability outcome: the new pointer may already be
/// visible even though `commit` returns an I/O error.
#[derive(Clone, Debug)]
pub struct RevisionStore {
    root: PathBuf,
    cas: ContentAddressedStore,
    /// Serializes optimistic check-and-publish within this process. Cross-process
    /// coordination is intentionally not implied by this local store.
    publish_lock: Arc<Mutex<()>>,
}

/// Cross-process publication lease. On Unix this is an advisory `flock` held
/// on a stable file, so separate `RevisionStore` instances and worker
/// processes sharing one root serialize the compare-and-publish window.
#[cfg(unix)]
struct FilePublishLock(File);

#[cfg(unix)]
impl FilePublishLock {
    fn acquire(root: &Path) -> Result<Self, PublishError> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(root.join(".publish.lock"))?;
        rustix::fs::flock(&file, rustix::fs::FlockOperation::LockExclusive)
            .map_err(|error| PublishError::Io(io::Error::from(error)))?;
        Ok(Self(file))
    }
}

#[cfg(unix)]
impl Drop for FilePublishLock {
    fn drop(&mut self) {
        // The descriptor is closed immediately afterwards; an unlock failure
        // cannot make the already-completed publication observable as failed.
        let _ = rustix::fs::flock(&self.0, rustix::fs::FlockOperation::Unlock);
    }
}

#[cfg(not(unix))]
struct FilePublishLock;

#[cfg(not(unix))]
impl FilePublishLock {
    fn acquire(root: &Path) -> Result<Self, PublishError> {
        let _ = root;
        Err(PublishError::Invalid(
            "cross-process publication locking is unavailable on this platform",
        ))
    }
}

impl RevisionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, PublishError> {
        let root = root.into();
        // Persist each newly-created directory entry in its parent. Otherwise a
        // durable CURRENT/revision file could survive while the directory that
        // contains it (or the store root itself) disappears after a crash.
        create_dir_all_durable(&root.join("revisions"))?;
        let cas_root = root.join("cas");
        create_dir_all_durable(&cas_root)?;
        let cas = ContentAddressedStore::open(cas_root)?;
        Ok(Self {
            root,
            cas,
            publish_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn cas(&self) -> &ContentAddressedStore {
        &self.cas
    }

    pub fn current_revision(&self) -> Result<Option<BuildRevision>, PublishError> {
        let path = self.root.join("CURRENT");
        match fs::read(path) {
            Ok(bytes) if bytes.len() == 8 => Ok(Some(BuildRevision(u64::from_le_bytes(
                bytes.try_into().unwrap(),
            )))),
            Ok(_) => Err(PublishError::Invalid("CURRENT pointer has invalid length")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn begin(
        &self,
        revision: BuildRevision,
        expected_current: Option<BuildRevision>,
    ) -> PublishTransaction {
        PublishTransaction {
            store: self.clone(),
            revision,
            expected_current,
            staged: BTreeMap::new(),
        }
    }

    pub fn manifest_bytes(&self, revision: BuildRevision) -> Result<Vec<u8>, PublishError> {
        Ok(fs::read(
            self.root
                .join("revisions")
                .join(format!("{}.manifest", revision.0)),
        )?)
    }
}

pub struct PublishTransaction {
    store: RevisionStore,
    revision: BuildRevision,
    expected_current: Option<BuildRevision>,
    staged: BTreeMap<AssetId, (ArtifactDescriptor, Vec<u8>)>,
}

impl PublishTransaction {
    pub fn stage(
        &mut self,
        asset: AssetId,
        descriptor: ArtifactDescriptor,
        bytes: Vec<u8>,
    ) -> Result<(), PublishError> {
        descriptor.validate().map_err(PublishError::Invalid)?;
        if descriptor.byte_len != bytes.len() as u64
            || descriptor.content_hash != crate::types::ContentHash::digest(&bytes)
        {
            return Err(PublishError::DescriptorMismatch(asset));
        }
        if self.staged.contains_key(&asset) {
            return Err(PublishError::Invalid("duplicate staged AssetId"));
        }
        self.staged.insert(asset, (descriptor, bytes));
        Ok(())
    }

    pub fn commit(self) -> Result<PublishedManifest, PublishError> {
        let _publish_guard = self
            .store
            .publish_lock
            .lock()
            .map_err(|_| PublishError::Invalid("publish lock poisoned"))?;
        let _cross_process_guard = FilePublishLock::acquire(&self.store.root)?;
        let actual = self.store.current_revision()?;
        if actual != self.expected_current {
            return Err(PublishError::Conflict {
                expected: self.expected_current,
                actual,
            });
        }
        let next = actual.map_or(0, |r| r.0.checked_add(1).unwrap_or(u64::MAX));
        if self.revision.0 != next || (actual == Some(BuildRevision(u64::MAX))) {
            return Err(PublishError::Invalid(
                "revision must be the next monotonic revision",
            ));
        }

        validate_dependency_closure(&self.staged)?;

        let mut entries = Vec::with_capacity(self.staged.len());
        for (asset, (descriptor, bytes)) in &self.staged {
            let hash = self.store.cas.put(bytes)?;
            if hash != descriptor.content_hash {
                return Err(PublishError::DescriptorMismatch(*asset));
            }
            entries.push(ManifestEntry {
                asset: *asset,
                descriptor: descriptor.clone(),
            });
        }
        let manifest = Manifest::new(self.revision, entries).map_err(PublishError::Invalid)?;
        let bytes = manifest.canonical_bytes();
        let manifest_hash = crate::types::ContentHash::digest(&bytes);

        let revision_path = self
            .store
            .root
            .join("revisions")
            .join(format!("{}.manifest", self.revision.0));
        write_immutable(&revision_path, &bytes)?;

        // Re-read immediately before pointer replacement. The file lock makes
        // this a compare-and-publish window across cooperating processes; the
        // second read also prevents a future prepare step from widening the
        // check-to-publish gap.
        let final_current = self.store.current_revision()?;
        if final_current != self.expected_current {
            return Err(PublishError::Conflict {
                expected: self.expected_current,
                actual: final_current,
            });
        }

        // Last mutating operation is atomic pointer replacement. If anything above
        // fails, CURRENT still names the prior complete revision.
        atomic_replace(&self.store.root, "CURRENT", &self.revision.0.to_le_bytes())?;
        Ok(PublishedManifest {
            manifest,
            manifest_hash,
        })
    }
}

/// Verifies that the manifest being published contains a complete, acyclic
/// closure for every staged artifact. This runs before the first CAS write so
/// invalid revisions cannot create blobs as a side effect of validation.
fn validate_dependency_closure(
    staged: &BTreeMap<AssetId, (ArtifactDescriptor, Vec<u8>)>,
) -> Result<(), PublishError> {
    let mut artifact_owners = BTreeMap::<ArtifactKey, AssetId>::new();
    let mut descriptors = BTreeMap::<ArtifactKey, &ArtifactDescriptor>::new();

    for (asset, (descriptor, _)) in staged {
        if let Some(existing) = descriptors.insert(descriptor.key, descriptor)
            && existing != descriptor
        {
            return Err(PublishError::Invalid(
                "one ArtifactKey has conflicting staged descriptors",
            ));
        }
        artifact_owners.entry(descriptor.key).or_insert(*asset);
    }

    let mut indegree = descriptors
        .keys()
        .copied()
        .map(|key| (key, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<ArtifactKey, BTreeSet<ArtifactKey>>::new();

    for (key, descriptor) in &descriptors {
        let mut unique_dependencies = BTreeSet::new();
        for dependency in &descriptor.dependencies {
            if !unique_dependencies.insert(*dependency) {
                return Err(PublishError::Invalid(
                    "artifact descriptor contains a duplicate dependency",
                ));
            }
            if !artifact_owners.contains_key(dependency) {
                return Err(PublishError::MissingDependency {
                    asset: artifact_owners[key],
                    dependency: *dependency,
                });
            }
            if dependency == key {
                return Err(PublishError::DependencyCycle(vec![*key]));
            }
            if dependents.entry(*dependency).or_default().insert(*key) {
                *indegree
                    .get_mut(key)
                    .expect("all staged artifact keys have indegree entries") += 1;
            }
        }
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(&key, &degree)| (degree == 0).then_some(key))
        .collect::<BTreeSet<_>>();
    let mut visited = 0usize;
    while let Some(key) = ready.pop_first() {
        visited += 1;
        if let Some(children) = dependents.get(&key) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .expect("dependent artifact key has an indegree entry");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*child);
                }
            }
        }
    }
    if visited != indegree.len() {
        return Err(PublishError::DependencyCycle(
            indegree
                .into_iter()
                .filter_map(|(key, degree)| (degree != 0).then_some(key))
                .collect(),
        ));
    }
    Ok(())
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), PublishError> {
    let parent = path
        .parent()
        .ok_or(PublishError::Invalid("manifest path has no parent"))?;
    if path.exists() {
        if fs::read(path)? == bytes {
            // This can be a retry after rename succeeded but the prior
            // directory sync failed. Re-sync before allowing CURRENT to point
            // at this manifest.
            return sync_directory(parent);
        }
        return Err(PublishError::Invalid(
            "revision manifest already exists with different contents",
        ));
    }
    let tmp = temp_path(parent);
    let result = (|| -> Result<(), PublishError> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        match fs::rename(&tmp, path) {
            Ok(()) => sync_directory(parent),
            Err(_e) if path.exists() && fs::read(path)? == bytes => {
                let _ = fs::remove_file(&tmp);
                // A concurrent publisher may have installed the identical
                // immutable manifest; its directory entry still needs to be
                // durable before CURRENT can reference it.
                sync_directory(parent)
            }
            Err(e) => Err(PublishError::Io(e)),
        }
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

fn atomic_replace(root: &Path, name: &str, bytes: &[u8]) -> Result<(), PublishError> {
    let tmp = temp_path(root);
    let result = (|| -> Result<(), PublishError> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, root.join(name))?;
        sync_directory(root)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

fn sync_directory(path: &Path) -> Result<(), PublishError> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Creates a directory tree and durably records every newly-created child in
/// its parent, from the deepest new entry back to the pre-existing ancestor.
fn create_dir_all_durable(path: &Path) -> Result<(), PublishError> {
    let mut missing = Vec::new();
    let mut cursor = path;
    while !cursor.exists() {
        missing.push(cursor.to_path_buf());
        cursor = cursor.parent().ok_or(PublishError::Invalid(
            "directory path has no existing ancestor",
        ))?;
    }
    if !cursor.is_dir() {
        return Err(PublishError::Io(io::Error::new(
            io::ErrorKind::NotADirectory,
            "existing ancestor is not a directory",
        )));
    }
    fs::create_dir_all(path)?;
    for created in missing {
        let parent = created
            .parent()
            .ok_or(PublishError::Invalid("created directory has no parent"))?;
        sync_directory(parent)?;
    }
    Ok(())
}

fn temp_path(parent: &Path) -> PathBuf {
    let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(".txn-{}-{id}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ArtifactKey, ContentHash};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-revisions-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn descriptor(bytes: &[u8]) -> ArtifactDescriptor {
        ArtifactDescriptor {
            key: ArtifactKey([7; 32]),
            kind: 4,
            schema: 1,
            content_hash: ContentHash::digest(bytes),
            byte_len: bytes.len() as u64,
            alignment: 16,
            dependencies: vec![],
        }
    }

    fn descriptor_with_dependencies(
        key: u8,
        bytes: &[u8],
        dependencies: Vec<ArtifactKey>,
    ) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 4, 1, 16, dependencies, bytes)
    }

    #[test]
    fn publishes_and_failed_transaction_keeps_previous_revision() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let mut first = store.begin(BuildRevision(0), None);
        first
            .stage(AssetId(1), descriptor(b"good"), b"good".to_vec())
            .unwrap();
        first.commit().unwrap();
        let previous = store.manifest_bytes(BuildRevision(0)).unwrap();

        let mut failed = store.begin(BuildRevision(1), Some(BuildRevision(0)));
        failed
            .stage(AssetId(2), descriptor(b"expected"), b"wrong".to_vec())
            .unwrap_err();
        // No commit was possible: the prior publication remains intact.
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
        assert_eq!(store.manifest_bytes(BuildRevision(0)).unwrap(), previous);

        let mut next = store.begin(BuildRevision(1), Some(BuildRevision(0)));
        next.stage(AssetId(2), descriptor(b"next"), b"next".to_vec())
            .unwrap();
        next.commit().unwrap();
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(1)));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reopened_store_recovers_published_pointer_manifest_and_payload() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"durable payload";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        drop(store);

        let reopened = RevisionStore::open(&dir).unwrap();
        assert_eq!(reopened.current_revision().unwrap(), Some(BuildRevision(0)));
        let manifest_bytes = reopened.manifest_bytes(BuildRevision(0)).unwrap();
        assert!(!manifest_bytes.is_empty());
        assert_eq!(
            reopened.cas().read(ContentHash::digest(payload)).unwrap(),
            payload
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn orphaned_prepared_manifest_does_not_advance_current_and_can_be_published() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"prepared before pointer update";
        let manifest = Manifest::new(
            BuildRevision(0),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(payload),
            }],
        )
        .unwrap();
        // Model a process stopping after the immutable manifest rename but
        // before CURRENT is replaced.
        fs::write(dir.join("revisions/0.manifest"), manifest.canonical_bytes()).unwrap();
        drop(store);

        let reopened = RevisionStore::open(&dir).unwrap();
        assert_eq!(reopened.current_revision().unwrap(), None);
        let mut tx = reopened.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(reopened.current_revision().unwrap(), Some(BuildRevision(0)));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn optimistic_conflict_does_not_replace_current() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let tx = store.begin(BuildRevision(0), Some(BuildRevision(9)));
        assert!(matches!(tx.commit(), Err(PublishError::Conflict { .. })));
        assert_eq!(store.current_revision().unwrap(), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_duplicate_staged_asset_without_overwriting_first_value() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(b"first"), b"first".to_vec())
            .unwrap();
        assert!(matches!(
            tx.stage(AssetId(1), descriptor(b"second"), b"second".to_vec()),
            Err(PublishError::Invalid("duplicate staged AssetId"))
        ));
        let published = tx.commit().unwrap();
        assert_eq!(
            published.manifest.entries[&AssetId(1)].content_hash,
            ContentHash::digest(b"first")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_missing_dependency_before_writing_cas_or_current() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let missing = ArtifactKey([9; 32]);
        let bytes = b"root";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(
            AssetId(1),
            descriptor_with_dependencies(1, bytes, vec![missing]),
            bytes.to_vec(),
        )
        .unwrap();

        assert!(matches!(
            tx.commit(),
            Err(PublishError::MissingDependency {
                asset: AssetId(1),
                dependency
            }) if dependency == missing
        ));
        assert_eq!(store.current_revision().unwrap(), None);
        assert!(!store.cas().blob_path(ContentHash::digest(bytes)).exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_self_dependency_and_multi_artifact_cycle() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let key_a = ArtifactKey([1; 32]);
        let key_b = ArtifactKey([2; 32]);

        let mut self_cycle = store.begin(BuildRevision(0), None);
        self_cycle
            .stage(
                AssetId(1),
                descriptor_with_dependencies(1, b"self", vec![key_a]),
                b"self".to_vec(),
            )
            .unwrap();
        assert!(matches!(
            self_cycle.commit(),
            Err(PublishError::DependencyCycle(keys)) if keys == vec![key_a]
        ));

        let mut cycle = store.begin(BuildRevision(0), None);
        cycle
            .stage(
                AssetId(1),
                descriptor_with_dependencies(1, b"a", vec![key_b]),
                b"a".to_vec(),
            )
            .unwrap();
        cycle
            .stage(
                AssetId(2),
                descriptor_with_dependencies(2, b"b", vec![key_a]),
                b"b".to_vec(),
            )
            .unwrap();
        assert!(matches!(
            cycle.commit(),
            Err(PublishError::DependencyCycle(keys)) if keys == vec![key_a, key_b]
        ));
        assert_eq!(store.current_revision().unwrap(), None);
        assert!(!store.cas().blob_path(ContentHash::digest(b"a")).exists());
        assert!(!store.cas().blob_path(ContentHash::digest(b"b")).exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_duplicate_dependency_edges_before_publication() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let dependency = ArtifactKey([1; 32]);
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(
            AssetId(1),
            descriptor_with_dependencies(1, b"leaf", vec![]),
            b"leaf".to_vec(),
        )
        .unwrap();
        tx.stage(
            AssetId(2),
            descriptor_with_dependencies(2, b"root", vec![dependency, dependency]),
            b"root".to_vec(),
        )
        .unwrap();

        assert!(matches!(
            tx.commit(),
            Err(PublishError::Invalid(
                "artifact descriptor contains a duplicate dependency"
            ))
        ));
        assert_eq!(store.current_revision().unwrap(), None);
        assert!(!store.cas().blob_path(ContentHash::digest(b"leaf")).exists());
        assert!(!store.cas().blob_path(ContentHash::digest(b"root")).exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn separate_store_handles_serialize_concurrent_revision_publication() {
        let dir = temp_dir();
        let first_store = RevisionStore::open(&dir).unwrap();
        let second_store = RevisionStore::open(&dir).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let handles = [first_store, second_store]
            .into_iter()
            .enumerate()
            .map(|(index, store)| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut transaction = store.begin(BuildRevision(0), None);
                    let bytes: &[u8] = if index == 0 { b"first" } else { b"second" };
                    transaction
                        .stage(
                            AssetId(index as u128 + 1),
                            ArtifactDescriptor::for_bytes(
                                ArtifactKey([index as u8 + 1; 32]),
                                4,
                                1,
                                16,
                                vec![],
                                bytes,
                            ),
                            bytes.to_vec(),
                        )
                        .unwrap();
                    barrier.wait();
                    transaction.commit()
                })
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(PublishError::Conflict { .. })))
                .count(),
            1
        );
        assert_eq!(
            RevisionStore::open(&dir)
                .unwrap()
                .current_revision()
                .unwrap(),
            Some(BuildRevision(0))
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
