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

const MANIFEST_MAGIC: &[u8] = b"PRISM-MANIFEST\0\x01";
const MANIFEST_ENTRY_FIXED_BYTES: usize = 16 + 32 + 16 + 4 + 32 + 8 + 4 + 8;

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
        let store = Self {
            root,
            cas,
            publish_lock: Arc::new(Mutex::new(())),
        };
        // CURRENT is the publication boundary. Never open a store that points
        // at a missing, malformed, or differently-numbered manifest.
        store.current_revision()?;
        Ok(store)
    }

    pub fn cas(&self) -> &ContentAddressedStore {
        &self.cas
    }

    pub fn current_revision(&self) -> Result<Option<BuildRevision>, PublishError> {
        let current = match fs::read(self.root.join("CURRENT")) {
            Ok(bytes) if bytes.len() == 8 => Ok(Some(BuildRevision(u64::from_le_bytes(
                bytes.try_into().unwrap(),
            )))),
            Ok(_) => Err(PublishError::Invalid("CURRENT pointer has invalid length")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }?;
        if let Some(revision) = current {
            self.validate_current_manifest(revision)?;
        }
        Ok(current)
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

    fn validate_current_manifest(&self, revision: BuildRevision) -> Result<(), PublishError> {
        let path = self
            .root
            .join("revisions")
            .join(format!("{}.manifest", revision.0));
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(PublishError::Invalid(
                    "CURRENT points to a missing manifest",
                ));
            }
            Err(error) => return Err(PublishError::Io(error)),
        };
        let manifest = decode_manifest(&bytes, revision)?;
        // A syntactically valid manifest is not a valid publication if any
        // immutable payload it names has disappeared or no longer matches its
        // declared length.  Verify the closure at the publication boundary so
        // a later build never treats a dangling CURRENT as a usable base.
        for descriptor in manifest.entries.values() {
            let payload = self
                .cas
                .read(descriptor.content_hash)
                .map_err(|error| PublishError::Cas(error))?;
            let actual_len = u64::try_from(payload.len())
                .map_err(|_| PublishError::Invalid("CAS payload length overflows"))?;
            if actual_len != descriptor.byte_len {
                return Err(PublishError::Invalid(
                    "CURRENT manifest payload length does not match CAS",
                ));
            }
        }
        Ok(())
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

/// Decodes the local manifest wire format and proves that it is the exact
/// canonical representation produced by [`Manifest::canonical_bytes`]. This
/// is intentionally kept next to publication: `CURRENT` must never make an
/// untrusted or partially-written manifest visible to callers.
fn decode_manifest(
    bytes: &[u8],
    expected_revision: BuildRevision,
) -> Result<Manifest, PublishError> {
    if bytes.len() < MANIFEST_MAGIC.len() || !bytes.starts_with(MANIFEST_MAGIC) {
        return Err(PublishError::Invalid(
            "manifest has invalid magic or version",
        ));
    }
    let mut reader = ManifestReader { bytes, offset: 0 };
    if reader.take(MANIFEST_MAGIC.len())? != MANIFEST_MAGIC {
        return Err(PublishError::Invalid(
            "manifest has invalid magic or version",
        ));
    }
    let revision = BuildRevision(reader.u64()?);
    if revision != expected_revision {
        return Err(PublishError::Invalid(
            "CURRENT revision does not match manifest revision",
        ));
    }
    let entry_count = reader.usize_count("manifest entry count")?;
    let minimum_remaining = entry_count
        .checked_mul(MANIFEST_ENTRY_FIXED_BYTES)
        .ok_or(PublishError::Invalid("manifest entry count overflows"))?;
    if minimum_remaining > reader.remaining() {
        return Err(PublishError::Invalid("manifest is truncated"));
    }

    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let asset = AssetId(reader.u128()?);
        let key = ArtifactKey(reader.array::<32>()?);
        let kind = reader.u128()?;
        let schema = reader.u32()?;
        let content_hash = crate::types::ContentHash(reader.array::<32>()?);
        let byte_len = reader.u64()?;
        let alignment = reader.u32()?;
        let dependency_count = reader.usize_count("manifest dependency count")?;
        let dependency_bytes = dependency_count
            .checked_mul(std::mem::size_of::<ArtifactKey>())
            .ok_or(PublishError::Invalid("manifest dependency count overflows"))?;
        if dependency_bytes > reader.remaining() {
            return Err(PublishError::Invalid("manifest dependencies are truncated"));
        }
        let mut dependencies = Vec::with_capacity(dependency_count);
        for _ in 0..dependency_count {
            dependencies.push(ArtifactKey(reader.array::<32>()?));
        }
        entries.push(ManifestEntry {
            asset,
            descriptor: ArtifactDescriptor {
                key,
                kind,
                schema,
                content_hash,
                byte_len,
                alignment,
                dependencies,
            },
        });
    }
    if reader.remaining() != 0 {
        return Err(PublishError::Invalid("manifest has trailing bytes"));
    }

    let manifest = Manifest::new(revision, entries).map_err(PublishError::Invalid)?;
    let staged = manifest
        .entries
        .iter()
        .map(|(&asset, descriptor)| (asset, (descriptor.clone(), Vec::new())))
        .collect::<BTreeMap<_, _>>();
    validate_dependency_closure(&staged)?;
    if manifest.canonical_bytes() != bytes {
        return Err(PublishError::Invalid("manifest is not canonical"));
    }
    Ok(manifest)
}

struct ManifestReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ManifestReader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], PublishError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(PublishError::Invalid("manifest length overflows"))?;
        if end > self.bytes.len() {
            return Err(PublishError::Invalid("manifest is truncated"));
        }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PublishError> {
        self.take(N)?
            .try_into()
            .map_err(|_| PublishError::Invalid("manifest field has an invalid length"))
    }

    fn u32(&mut self) -> Result<u32, PublishError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, PublishError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn u128(&mut self) -> Result<u128, PublishError> {
        Ok(u128::from_le_bytes(self.array()?))
    }

    fn usize_count(&mut self, what: &'static str) -> Result<usize, PublishError> {
        usize::try_from(self.u64()?).map_err(|_| PublishError::Invalid(what))
    }
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
    fn open_fails_closed_when_current_manifest_is_missing() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("revisions")).unwrap();
        fs::write(dir.join("CURRENT"), BuildRevision(7).0.to_le_bytes()).unwrap();

        assert!(matches!(
            RevisionStore::open(&dir),
            Err(PublishError::Invalid(
                "CURRENT points to a missing manifest"
            ))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn open_fails_closed_when_current_manifest_is_malformed() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("revisions")).unwrap();
        fs::write(dir.join("CURRENT"), BuildRevision(3).0.to_le_bytes()).unwrap();
        fs::write(dir.join("revisions/3.manifest"), b"truncated").unwrap();

        assert!(matches!(
            RevisionStore::open(&dir),
            Err(PublishError::Invalid(
                "manifest has invalid magic or version"
            ))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn open_fails_closed_when_manifest_revision_does_not_match_current() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("revisions")).unwrap();
        fs::write(dir.join("CURRENT"), BuildRevision(3).0.to_le_bytes()).unwrap();
        let manifest = Manifest::new(BuildRevision(2), Vec::<ManifestEntry>::new()).unwrap();
        fs::write(dir.join("revisions/3.manifest"), manifest.canonical_bytes()).unwrap();

        assert!(matches!(
            RevisionStore::open(&dir),
            Err(PublishError::Invalid(
                "CURRENT revision does not match manifest revision"
            ))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn current_revision_revalidates_manifest_after_open() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(b"payload"), b"payload".to_vec())
            .unwrap();
        tx.commit().unwrap();
        fs::remove_file(dir.join("revisions/0.manifest")).unwrap();

        assert!(matches!(
            store.current_revision(),
            Err(PublishError::Invalid(
                "CURRENT points to a missing manifest"
            ))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn open_fails_closed_when_current_manifest_payload_is_missing() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"payload missing after publication";
        let hash = ContentHash::digest(payload);
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        fs::remove_file(store.cas().blob_path(hash)).unwrap();
        drop(store);

        assert!(matches!(
            RevisionStore::open(&dir),
            Err(PublishError::Cas(CasError::Io(error)))
                if error.kind() == io::ErrorKind::NotFound
        ));
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
