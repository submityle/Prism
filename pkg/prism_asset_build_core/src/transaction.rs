use crate::cas::{CasError, ContentAddressedStore};
use crate::journal::{JournalRecord, TransactionJournal};
use crate::package::RuntimePackage;
use crate::publication::PublicationDescriptor;
use crate::types::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, Manifest, ManifestEntry,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use prism_asset_container::{ContainerReader, RuntimeManifest};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

const MANIFEST_MAGIC: &[u8] = b"PRISM-MANIFEST\0\x01";
const MANIFEST_ENTRY_FIXED_BYTES: usize = 16 + 32 + 16 + 4 + 32 + 8 + 4 + 8;

#[derive(Debug)]
pub enum PublishError {
    Io(io::Error),
    Cas(CasError),
    Invalid(&'static str),
    /// `CURRENT` was replaced, but the durable commit marker could not be
    /// written. The revision may already be visible; callers must reconcile
    /// it instead of retrying the same revision as a fresh transaction.
    UnknownCommit {
        revision: BuildRevision,
        manifest_hash: crate::types::ContentHash,
        publication_hash: Option<crate::types::ContentHash>,
        source: io::Error,
    },
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
    /// Immutable binding for runtime package outputs, when this is a packaged
    /// publication. Build-only revisions intentionally have no runtime package.
    pub publication: Option<PublicationDescriptor>,
}

/// Durable state observed for one local publication revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionState {
    /// No journal record exists for this revision.
    Unknown,
    /// A prepare record exists but publication has not reached a durable commit
    /// marker.  [`RevisionStore::open`] resolves this state before returning.
    Prepared,
    /// `CURRENT` names a prepared revision but its commit marker is absent.
    UnknownCommit,
    /// The revision reached the publication boundary and has a commit marker.
    Committed,
    /// Recovery discarded an uncommitted prepare and recorded the decision.
    Aborted,
}

/// Local revision store. Publication becomes visible at atomic replacement of
/// CURRENT. The append-only journal records the prepare/commit boundary so a
/// failure while syncing the containing directory after that rename can be
/// resolved on the next open instead of being silently classified as a failed
/// or successful build.
#[derive(Clone, Debug)]
pub struct RevisionStore {
    root: PathBuf,
    cas: ContentAddressedStore,
    journal: TransactionJournal,
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
        let transactions_root = root.join("transactions");
        create_dir_all_durable(&transactions_root)?;
        // Serialize journal initialization and recovery with publishers in
        // other Unix processes.  The root exists by this point, so the lock
        // itself is also a durable store entry.
        #[cfg(unix)]
        let _recovery_guard = FilePublishLock::acquire(&root)?;
        let cas = ContentAddressedStore::open(cas_root)?;
        let journal = TransactionJournal::open(transactions_root.join("journal.bin"))?;
        let store = Self {
            root,
            cas,
            journal,
            publish_lock: Arc::new(Mutex::new(())),
        };
        store.recover_transactions()?;
        // CURRENT is the publication boundary. Never open a store that points
        // at a missing, malformed, or differently-numbered manifest.
        store.current_revision()?;
        Ok(store)
    }

    pub fn cas(&self) -> &ContentAddressedStore {
        &self.cas
    }

    pub fn current_revision(&self) -> Result<Option<BuildRevision>, PublishError> {
        let current = self.read_current_pointer()?;
        if let Some(revision) = current {
            self.validate_current_manifest(revision)?;
        }
        Ok(current)
    }

    /// Returns the last durable journal state for `revision`.
    pub fn transaction_state(
        &self,
        revision: BuildRevision,
    ) -> Result<TransactionState, PublishError> {
        let records = self.journal.scan()?;
        let mut state = TransactionState::Unknown;
        for record in records {
            match record {
                JournalRecord::Prepare {
                    revision: candidate,
                    ..
                }
                | JournalRecord::Commit {
                    revision: candidate,
                    ..
                }
                | JournalRecord::Abort {
                    revision: candidate,
                    ..
                } if candidate == revision => {
                    state = match record {
                        JournalRecord::Prepare { .. } => TransactionState::Prepared,
                        JournalRecord::Commit { .. } => TransactionState::Committed,
                        JournalRecord::Abort { .. } => TransactionState::Aborted,
                    };
                }
                _ => {}
            }
        }
        if state == TransactionState::Prepared && self.read_current_pointer()? == Some(revision) {
            return Ok(TransactionState::UnknownCommit);
        }
        Ok(state)
    }

    /// Returns committed revisions after `after`, suitable for replaying a
    /// notification stream after a frontend reconnects.
    pub fn committed_revisions_since(
        &self,
        after: Option<BuildRevision>,
    ) -> Result<Vec<BuildRevision>, PublishError> {
        let records = self.journal.scan()?;
        let mut revisions = BTreeSet::new();
        for record in records {
            if let JournalRecord::Commit { revision, .. } = record
                && after.is_none_or(|previous| revision > previous)
            {
                revisions.insert(revision);
            }
        }
        Ok(revisions.into_iter().collect())
    }

    fn read_current_pointer(&self) -> Result<Option<BuildRevision>, PublishError> {
        match fs::read(self.root.join("CURRENT")) {
            Ok(bytes) if bytes.len() == 8 => Ok(Some(BuildRevision(u64::from_le_bytes(
                bytes.try_into().unwrap(),
            )))),
            Ok(_) => Err(PublishError::Invalid("CURRENT pointer has invalid length")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn recover_transactions(&self) -> Result<(), PublishError> {
        let records = self.journal.scan()?;
        let mut pending = BTreeMap::new();
        let mut states = BTreeMap::new();
        for record in records {
            match record {
                JournalRecord::Prepare {
                    revision,
                    expected_current,
                    manifest_hash,
                    publication_hash,
                } => {
                    if matches!(states.get(&revision), Some(TransactionState::Committed)) {
                        return Err(PublishError::Invalid(
                            "transaction journal prepares a committed revision",
                        ));
                    }
                    if matches!(states.get(&revision), Some(TransactionState::Prepared)) {
                        return Err(PublishError::Invalid(
                            "transaction journal contains duplicate prepare",
                        ));
                    }
                    pending.insert(
                        revision,
                        (expected_current, manifest_hash, publication_hash),
                    );
                    states.insert(revision, TransactionState::Prepared);
                }
                JournalRecord::Commit {
                    revision,
                    manifest_hash,
                    publication_hash,
                } => {
                    let Some((_, prepared_hash, prepared_publication_hash)) =
                        pending.remove(&revision)
                    else {
                        return Err(PublishError::Invalid(
                            "transaction journal commits without a prepare",
                        ));
                    };
                    if prepared_hash != manifest_hash {
                        return Err(PublishError::Invalid(
                            "transaction journal commit hash differs from prepare",
                        ));
                    }
                    if prepared_publication_hash != publication_hash {
                        return Err(PublishError::Invalid(
                            "transaction journal commit publication hash differs from prepare",
                        ));
                    }
                    states.insert(revision, TransactionState::Committed);
                }
                JournalRecord::Abort {
                    revision,
                    manifest_hash,
                    publication_hash,
                } => {
                    let Some((_, prepared_hash, prepared_publication_hash)) =
                        pending.remove(&revision)
                    else {
                        return Err(PublishError::Invalid(
                            "transaction journal aborts without a prepare",
                        ));
                    };
                    if prepared_hash != manifest_hash {
                        return Err(PublishError::Invalid(
                            "transaction journal abort hash differs from prepare",
                        ));
                    }
                    if prepared_publication_hash != publication_hash {
                        return Err(PublishError::Invalid(
                            "transaction journal abort publication hash differs from prepare",
                        ));
                    }
                    states.insert(revision, TransactionState::Aborted);
                }
            }
        }

        let current = self.read_current_pointer()?;
        for (revision, (_, manifest_hash, publication_hash)) in pending {
            let manifest_path = self
                .root
                .join("revisions")
                .join(format!("{}.manifest", revision.0));
            if current == Some(revision) {
                let bytes = fs::read(&manifest_path).map_err(|error| {
                    PublishError::Io(io::Error::new(
                        error.kind(),
                        format!("prepared current manifest is unavailable: {error}"),
                    ))
                })?;
                if crate::types::ContentHash::digest(&bytes) != manifest_hash {
                    return Err(PublishError::Invalid(
                        "CURRENT manifest differs from transaction journal",
                    ));
                }
                if let Some(publication_hash) = publication_hash {
                    let publication_path = self
                        .root
                        .join("revisions")
                        .join(format!("{}.publication", revision.0));
                    let publication_bytes = fs::read(&publication_path).map_err(|error| {
                        PublishError::Io(io::Error::new(
                            error.kind(),
                            format!("prepared current publication is unavailable: {error}"),
                        ))
                    })?;
                    if crate::types::ContentHash::digest(&publication_bytes) != publication_hash {
                        return Err(PublishError::Invalid(
                            "CURRENT publication differs from transaction journal",
                        ));
                    }
                }
                // A pointer replacement may have succeeded immediately before
                // the commit marker write.  Validate the complete payload
                // closure before converting that ambiguous state to committed.
                self.validate_current_manifest(revision)?;
                self.journal.append(JournalRecord::Commit {
                    revision,
                    manifest_hash,
                    publication_hash,
                })?;
            } else {
                if manifest_path.exists() {
                    let bytes = fs::read(&manifest_path)?;
                    if crate::types::ContentHash::digest(&bytes) != manifest_hash {
                        return Err(PublishError::Invalid(
                            "orphaned transaction manifest differs from journal",
                        ));
                    }
                    fs::remove_file(&manifest_path)?;
                    sync_directory(
                        manifest_path
                            .parent()
                            .ok_or(PublishError::Invalid("manifest path has no parent"))?,
                    )?;
                }
                let publication_path = self
                    .root
                    .join("revisions")
                    .join(format!("{}.publication", revision.0));
                if let Some(publication_hash) = publication_hash {
                    if publication_path.exists() {
                        let bytes = fs::read(&publication_path)?;
                        if crate::types::ContentHash::digest(&bytes) != publication_hash {
                            return Err(PublishError::Invalid(
                                "orphaned transaction publication differs from journal",
                            ));
                        }
                        fs::remove_file(&publication_path)?;
                        sync_directory(
                            publication_path
                                .parent()
                                .ok_or(PublishError::Invalid("publication path has no parent"))?,
                        )?;
                    }
                }
                let runtime_root = self.root.join("runtime").join(revision.0.to_string());
                if publication_hash.is_some() && runtime_root.exists() {
                    fs::remove_dir_all(&runtime_root)?;
                    sync_directory(
                        runtime_root
                            .parent()
                            .ok_or(PublishError::Invalid("runtime path has no parent"))?,
                    )?;
                }
                self.journal.append(JournalRecord::Abort {
                    revision,
                    manifest_hash,
                    publication_hash,
                })?;
            }
        }
        cleanup_transaction_temps(&self.root)?;
        Ok(())
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
            runtime_package: None,
        }
    }

    pub fn manifest_bytes(&self, revision: BuildRevision) -> Result<Vec<u8>, PublishError> {
        Ok(fs::read(
            self.root
                .join("revisions")
                .join(format!("{}.manifest", revision.0)),
        )?)
    }

    /// Reads and validates the immutable publication descriptor for a revision.
    pub fn publication_bytes(&self, revision: BuildRevision) -> Result<Vec<u8>, PublishError> {
        Ok(fs::read(
            self.root
                .join("revisions")
                .join(format!("{}.publication", revision.0)),
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
        self.validate_publication_if_present(revision, &bytes)?;
        Ok(())
    }

    fn validate_publication_if_present(
        &self,
        revision: BuildRevision,
        build_manifest_bytes: &[u8],
    ) -> Result<(), PublishError> {
        let publication_required = self.publication_requirement(revision)?;
        let descriptor_path = self
            .root
            .join("revisions")
            .join(format!("{}.publication", revision.0));
        let descriptor_bytes = match fs::read(&descriptor_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if publication_required == Some(true) {
                    return Err(PublishError::Invalid(
                        "CURRENT package publication descriptor is missing",
                    ));
                }
                // Legacy build-only stores and current build-only revisions do
                // not have a runtime publication descriptor.
                return Ok(());
            }
            Err(error) => return Err(PublishError::Io(error)),
        };
        if publication_required == Some(false) {
            return Err(PublishError::Invalid(
                "build-only revision unexpectedly has a runtime publication",
            ));
        }
        let publication =
            PublicationDescriptor::decode(&descriptor_bytes).map_err(PublishError::Invalid)?;
        if publication.revision != revision
            || publication.build_manifest_hash
                != crate::types::ContentHash::digest(build_manifest_bytes)
        {
            return Err(PublishError::Invalid(
                "publication descriptor does not bind CURRENT manifest",
            ));
        }
        let runtime_root = self.root.join("runtime").join(revision.0.to_string());
        let runtime_manifest_path = runtime_root.join("manifest.bin");
        let runtime_manifest = fs::read(&runtime_manifest_path).map_err(|error| {
            PublishError::Io(io::Error::new(
                error.kind(),
                format!("published runtime manifest is unavailable: {error}"),
            ))
        })?;
        if crate::types::ContentHash::digest(&runtime_manifest) != publication.runtime_manifest_hash
        {
            return Err(PublishError::Invalid(
                "published runtime manifest hash differs from publication descriptor",
            ));
        }
        let runtime_manifest = RuntimeManifest::decode(&runtime_manifest)
            .map_err(|_| PublishError::Invalid("published runtime manifest is invalid"))?;
        let mut container_bytes = BTreeMap::new();
        for container in &publication.containers {
            let path = runtime_root.join("containers").join(&container.name);
            let bytes = fs::read(path).map_err(|error| {
                PublishError::Io(io::Error::new(
                    error.kind(),
                    format!("published runtime container is unavailable: {error}"),
                ))
            })?;
            if u64::try_from(bytes.len()).ok() != Some(container.byte_len)
                || crate::types::ContentHash::digest(&bytes) != container.content_hash
            {
                return Err(PublishError::Invalid(
                    "published runtime container does not match publication descriptor",
                ));
            }
            ContainerReader::new(Arc::<[u8]>::from(bytes.as_slice()))
                .map_err(|_| PublishError::Invalid("published runtime container is invalid"))?;
            container_bytes.insert(container.name.as_str(), bytes);
        }
        let mut referenced = BTreeSet::new();
        for route in runtime_manifest.routes() {
            let Some(bytes) = container_bytes.get(route.container_name.as_str()) else {
                return Err(PublishError::Invalid(
                    "published runtime manifest references a missing container",
                ));
            };
            let reader = ContainerReader::new(Arc::<[u8]>::from(bytes.as_slice()))
                .map_err(|_| PublishError::Invalid("published runtime container is invalid"))?;
            if !reader.contains(route.container_guid) {
                return Err(PublishError::Invalid(
                    "published runtime manifest references a missing record",
                ));
            }
            referenced.insert(route.container_name.as_str());
        }
        if referenced.len() != publication.containers.len()
            || publication
                .containers
                .iter()
                .any(|container| !referenced.contains(container.name.as_str()))
        {
            return Err(PublishError::Invalid(
                "published runtime package contains an unreferenced container",
            ));
        }
        Ok(())
    }

    fn publication_requirement(
        &self,
        revision: BuildRevision,
    ) -> Result<Option<bool>, PublishError> {
        let mut requirement = None;
        for record in self.journal.scan()? {
            match record {
                JournalRecord::Prepare {
                    revision: candidate,
                    publication_hash,
                    ..
                }
                | JournalRecord::Commit {
                    revision: candidate,
                    publication_hash,
                    ..
                } if candidate == revision => {
                    requirement = Some(publication_hash.is_some());
                }
                _ => {}
            }
        }
        Ok(requirement)
    }
}

pub struct PublishTransaction {
    store: RevisionStore,
    revision: BuildRevision,
    expected_current: Option<BuildRevision>,
    staged: BTreeMap<AssetId, (ArtifactDescriptor, Vec<u8>)>,
    runtime_package: Option<RuntimePackage>,
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

    /// Stages a complete runtime package for publication with this build revision.
    /// The package is persisted under the same revision boundary as the build manifest.
    pub fn stage_runtime_package(&mut self, package: RuntimePackage) -> Result<(), PublishError> {
        if self.runtime_package.is_some() {
            return Err(PublishError::Invalid("runtime package already staged"));
        }
        let canonical = package
            .runtime_manifest
            .encode()
            .map_err(|_| PublishError::Invalid("runtime manifest cannot be encoded"))?;
        if canonical != package.runtime_manifest_bytes {
            return Err(PublishError::Invalid(
                "runtime manifest bytes are not canonical",
            ));
        }
        let decoded = RuntimeManifest::decode(&package.runtime_manifest_bytes)
            .map_err(|_| PublishError::Invalid("runtime manifest bytes cannot be decoded"))?;
        if decoded != package.runtime_manifest {
            return Err(PublishError::Invalid(
                "runtime manifest object differs from its encoded bytes",
            ));
        }
        let mut referenced = BTreeSet::new();
        for route in decoded.routes() {
            let Some(bytes) = package.containers.get(&route.container_name) else {
                return Err(PublishError::Invalid(
                    "runtime manifest references a missing container",
                ));
            };
            let reader = ContainerReader::new(Arc::<[u8]>::from(bytes.as_slice()))
                .map_err(|_| PublishError::Invalid("runtime container is invalid"))?;
            if !reader.contains(route.container_guid) {
                return Err(PublishError::Invalid(
                    "runtime manifest references a missing container record",
                ));
            }
            referenced.insert(route.container_name.as_str());
        }
        if referenced.len() != package.containers.len()
            || package
                .containers
                .keys()
                .any(|name| !referenced.contains(name.as_str()))
        {
            return Err(PublishError::Invalid(
                "runtime package contains an unreferenced container",
            ));
        }
        self.runtime_package = Some(package);
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
        for (asset, (descriptor, _)) in &self.staged {
            entries.push(ManifestEntry {
                asset: *asset,
                descriptor: descriptor.clone(),
            });
        }
        let manifest = Manifest::new(self.revision, entries).map_err(PublishError::Invalid)?;
        let bytes = manifest.canonical_bytes();
        let manifest_hash = crate::types::ContentHash::digest(&bytes);
        if let Some(package) = self.runtime_package.as_ref()
            && package.source_manifest_hash != manifest_hash
        {
            return Err(PublishError::Invalid(
                "runtime package was built from a different manifest",
            ));
        }
        let (publication, publication_bytes) = if let Some(package) = self.runtime_package.as_ref()
        {
            let publication = PublicationDescriptor::new(
                self.revision,
                manifest_hash,
                &package.runtime_manifest_bytes,
                &package.containers,
            )
            .map_err(PublishError::Invalid)?;
            let bytes = publication.encode().map_err(PublishError::Invalid)?;
            (Some(publication), Some(bytes))
        } else {
            (None, None)
        };
        let publication_hash = publication
            .as_ref()
            .map(PublicationDescriptor::content_hash);

        // From this point onward recovery can distinguish an interrupted
        // publication from an ordinary unused CAS blob.
        self.store.journal.append(JournalRecord::Prepare {
            revision: self.revision,
            expected_current: self.expected_current,
            manifest_hash,
            publication_hash,
        })?;

        // CAS writes happen after Prepare.  If the process stops during the
        // immutable blob phase, recovery can identify the orphaned output as a
        // transaction and leave it to reachability GC instead of losing the
        // publication's intent.
        for (asset, (descriptor, bytes)) in &self.staged {
            let hash = self.store.cas.put(bytes)?;
            if hash != descriptor.content_hash {
                return Err(PublishError::DescriptorMismatch(*asset));
            }
        }

        let revision_path = self
            .store
            .root
            .join("revisions")
            .join(format!("{}.manifest", self.revision.0));
        write_immutable(&revision_path, &bytes)?;
        if let (Some(publication_bytes), Some(package)) =
            (publication_bytes.as_deref(), self.runtime_package.as_ref())
        {
            let publication_path = self
                .store
                .root
                .join("revisions")
                .join(format!("{}.publication", self.revision.0));
            write_immutable(&publication_path, publication_bytes)?;
            self.write_runtime_outputs(&package.runtime_manifest_bytes, &package.containers)?;
        }

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
        if let Err(source) = self.store.journal.append(JournalRecord::Commit {
            revision: self.revision,
            manifest_hash,
            publication_hash,
        }) {
            return Err(PublishError::UnknownCommit {
                revision: self.revision,
                manifest_hash,
                publication_hash,
                source,
            });
        }
        Ok(PublishedManifest {
            manifest,
            manifest_hash,
            publication,
        })
    }

    fn write_runtime_outputs(
        &self,
        runtime_manifest_bytes: &[u8],
        containers: &BTreeMap<String, Vec<u8>>,
    ) -> Result<(), PublishError> {
        let runtime_root = self
            .store
            .root
            .join("runtime")
            .join(self.revision.0.to_string());
        let containers_root = runtime_root.join("containers");
        create_dir_all_durable(&containers_root)?;
        write_immutable(&runtime_root.join("manifest.bin"), runtime_manifest_bytes)?;
        for (name, bytes) in containers {
            write_immutable(&containers_root.join(name), bytes)?;
        }
        sync_directory(&containers_root)?;
        sync_directory(&runtime_root)?;
        Ok(())
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

fn cleanup_transaction_temps(root: &Path) -> Result<(), PublishError> {
    for directory in [root.to_path_buf(), root.join("revisions")] {
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(PublishError::Io(error)),
        };
        let mut removed = false;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(".txn-") {
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    fs::remove_file(entry.path())?;
                    removed = true;
                }
            }
        }
        if removed {
            sync_directory(&directory)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::RuntimePackage;
    use crate::types::{ArtifactKey, ContentHash};
    use prism_asset::StableGuid;
    use prism_asset_container::{ContainerWriter, ManifestRoute, RuntimeManifest};
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

    fn runtime_package_for(manifest: &Manifest, payload: &[u8]) -> RuntimePackage {
        let guid = StableGuid::from_u128(0x55);
        let mut writer = ContainerWriter::new();
        writer.add(guid, payload.to_vec()).unwrap();
        let container = writer.finish().unwrap();
        let mut runtime_manifest = RuntimeManifest::new();
        runtime_manifest
            .insert(ManifestRoute {
                guid,
                container_name: "base.container".to_owned(),
                priority: 0,
                container_guid: guid,
            })
            .unwrap();
        let runtime_manifest_bytes = runtime_manifest.encode().unwrap();
        RuntimePackage {
            source_manifest_hash: ContentHash::digest(&manifest.canonical_bytes()),
            containers: [("base.container".to_owned(), container)]
                .into_iter()
                .collect(),
            runtime_manifest,
            runtime_manifest_bytes,
        }
    }

    #[test]
    fn publishes_and_failed_transaction_keeps_previous_revision() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let mut first = store.begin(BuildRevision(0), None);
        first
            .stage(AssetId(1), descriptor(b"good"), b"good".to_vec())
            .unwrap();
        let published = first.commit().unwrap();
        assert!(published.publication.is_none());
        assert!(!dir.join("runtime/0").exists());
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
    fn runtime_package_is_published_only_when_bound_to_this_manifest() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"runtime payload";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        let manifest = Manifest::new(
            BuildRevision(0),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(payload),
            }],
        )
        .unwrap();
        tx.stage_runtime_package(runtime_package_for(&manifest, payload))
            .unwrap();
        let published = tx.commit().unwrap();
        assert!(published.publication.is_some());
        assert!(dir.join("revisions/0.publication").exists());
        assert!(dir.join("runtime/0/manifest.bin").exists());
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));

        let mut wrong = store.begin(BuildRevision(1), Some(BuildRevision(0)));
        wrong
            .stage(AssetId(1), descriptor(b"changed"), b"changed".to_vec())
            .unwrap();
        wrong
            .stage_runtime_package(runtime_package_for(&manifest, payload))
            .unwrap();
        assert!(matches!(
            wrong.commit(),
            Err(PublishError::Invalid(
                "runtime package was built from a different manifest"
            ))
        ));
        assert_eq!(store.current_revision().unwrap(), Some(BuildRevision(0)));
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
    fn reopening_after_pointer_publish_recovers_missing_commit_marker() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"commit marker recovery";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(
            store.transaction_state(BuildRevision(0)).unwrap(),
            TransactionState::Committed
        );

        // Remove the final Commit frame while leaving the durable manifest and
        // CURRENT pointer intact, modelling a crash between pointer rename and
        // the journal append.
        let journal_path = dir.join("transactions/journal.bin");
        let journal_len = fs::metadata(&journal_path).unwrap().len();
        let frame_len = crate::journal::FRAME_SIZE as u64;
        OpenOptions::new()
            .write(true)
            .open(&journal_path)
            .unwrap()
            .set_len(journal_len - frame_len)
            .unwrap();
        drop(store);

        let reopened = RevisionStore::open(&dir).unwrap();
        assert_eq!(reopened.current_revision().unwrap(), Some(BuildRevision(0)));
        assert_eq!(
            reopened.transaction_state(BuildRevision(0)).unwrap(),
            TransactionState::Committed
        );
        assert_eq!(
            reopened.committed_revisions_since(None).unwrap(),
            vec![BuildRevision(0)]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reopening_aborts_prepared_manifest_without_advancing_current() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"orphaned journal prepare";
        let descriptor = descriptor(payload);
        let manifest = Manifest::new(
            BuildRevision(0),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor.clone(),
            }],
        )
        .unwrap();
        store.cas.put(payload).unwrap();
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_hash = ContentHash::digest(&manifest_bytes);
        store
            .journal
            .append(JournalRecord::Prepare {
                revision: BuildRevision(0),
                expected_current: None,
                manifest_hash,
                publication_hash: None,
            })
            .unwrap();
        fs::write(dir.join("revisions/0.manifest"), manifest_bytes).unwrap();
        drop(store);

        let reopened = RevisionStore::open(&dir).unwrap();
        assert_eq!(reopened.current_revision().unwrap(), None);
        assert!(!dir.join("revisions/0.manifest").exists());
        assert_eq!(
            reopened.transaction_state(BuildRevision(0)).unwrap(),
            TransactionState::Aborted
        );
        // CAS is immutable and intentionally left for later reachability GC.
        assert_eq!(
            reopened.cas().read(ContentHash::digest(payload)).unwrap(),
            payload
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reopening_truncates_a_partial_journal_tail() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"partial journal tail";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        let journal_path = dir.join("transactions/journal.bin");
        let mut journal = OpenOptions::new().append(true).open(&journal_path).unwrap();
        journal.write_all(b"TXN1\x01").unwrap();
        journal.sync_all().unwrap();
        drop(store);

        let reopened = RevisionStore::open(&dir).unwrap();
        assert_eq!(
            reopened.transaction_state(BuildRevision(0)).unwrap(),
            TransactionState::Committed
        );
        let journal_len = fs::metadata(journal_path).unwrap().len();
        assert_eq!(
            journal_len,
            crate::journal::FILE_HEADER_LEN as u64 + 2 * crate::journal::FRAME_SIZE as u64
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reopening_fails_closed_on_a_corrupt_journal_frame() {
        let dir = temp_dir();
        let store = RevisionStore::open(&dir).unwrap();
        let payload = b"corrupt journal frame";
        let mut tx = store.begin(BuildRevision(0), None);
        tx.stage(AssetId(1), descriptor(payload), payload.to_vec())
            .unwrap();
        tx.commit().unwrap();
        let journal_path = dir.join("transactions/journal.bin");
        let mut journal_bytes = fs::read(&journal_path).unwrap();
        journal_bytes[crate::journal::FILE_HEADER_LEN + crate::journal::FRAME_HEADER_LEN] ^= 0x80;
        fs::write(&journal_path, journal_bytes).unwrap();
        drop(store);

        assert!(matches!(
            RevisionStore::open(&dir),
            Err(PublishError::Io(error)) if error.kind() == io::ErrorKind::InvalidData
        ));
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
