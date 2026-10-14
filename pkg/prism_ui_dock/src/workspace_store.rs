//! `std` file boundary for workspace snapshots.
//!
//! The core workspace model and codec remain renderer-independent and
//! `no_std + alloc`. This module is the deliberately narrow filesystem
//! adapter: it writes a sibling temporary file, flushes it, preserves the
//! previous primary snapshot as `last-good`, then atomically renames the
//! temporary file into place.

use fs4::fs_std::FileExt;
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tempfile::TempPath;

use crate::{
    decode_instance_registry, decode_workspace, decode_workspace_with_registry,
    encode_instance_registry, encode_workspace, InstancePersistError, MigrationError,
    PanelInstanceRegistry, PanelRegistry, WorkspaceDecodeLimits, WorkspaceDecodeOptions,
    WorkspaceManager, WorkspaceMigrationRegistry, WorkspacePersistError,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const LOCK_WAIT_ATTEMPTS: usize = 5_000;

/// The source from which a workspace snapshot was loaded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceLoadSource {
    /// The primary snapshot path.
    Primary,
    /// The sibling `last-good` snapshot after primary decoding failed.
    LastGood,
    /// The caller-provided default because neither snapshot was usable.
    Default,
}

/// A workspace snapshot loaded from disk or recovered from a last-good copy.
#[derive(Debug)]
pub struct LoadedWorkspace {
    /// The decoded workspace state.
    pub manager: WorkspaceManager,
    /// The source that supplied the returned state.
    pub source: WorkspaceLoadSource,
    /// The primary I/O/decode error, if primary loading failed before fallback.
    pub primary_error: Option<WorkspaceStoreError>,
    /// The last-good I/O/decode error, if that fallback was attempted and failed.
    pub last_good_error: Option<WorkspaceStoreError>,
}

/// A panel-instance registry loaded from the primary or last-good snapshot.
#[derive(Debug)]
pub struct LoadedInstances {
    /// The decoded stable instance identities.
    pub registry: PanelInstanceRegistry,
    /// The source that supplied the returned registry.
    pub source: WorkspaceLoadSource,
    /// The primary I/O/decode error, if primary loading failed.
    pub primary_error: Option<WorkspaceStoreError>,
    /// The last-good I/O/decode error, if that fallback was attempted.
    pub last_good_error: Option<WorkspaceStoreError>,
}

/// Filesystem or codec failure at the workspace storage boundary.
#[derive(Debug)]
pub enum WorkspaceStoreError {
    /// Filesystem operation failed.
    Io(io::Error),
    /// Snapshot bytes failed the versioned workspace codec.
    Codec(WorkspacePersistError),
    /// Snapshot bytes failed the explicit schema migration registry.
    Migration(MigrationError),
    /// Snapshot bytes failed the panel-instance registry codec.
    Instance(InstancePersistError),
}

impl From<io::Error> for WorkspaceStoreError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<WorkspacePersistError> for WorkspaceStoreError {
    fn from(error: WorkspacePersistError) -> Self {
        Self::Codec(error)
    }
}

impl From<MigrationError> for WorkspaceStoreError {
    fn from(error: MigrationError) -> Self {
        Self::Migration(error)
    }
}

impl From<InstancePersistError> for WorkspaceStoreError {
    fn from(error: InstancePersistError) -> Self {
        Self::Instance(error)
    }
}

/// A serialized primary/last-good workspace snapshot pair.
///
/// Saves are serialized through a sibling lock file, write unique temporary
/// siblings, and sync both file contents and (on Unix) the containing
/// directory before returning. This protects the codec boundary from
/// concurrent writers and ordinary process crashes; filesystem-specific power
/// loss guarantees remain the responsibility of the underlying filesystem.
/// The stable lock path is protected by an OS advisory lock; the lock file is
/// intentionally retained so a crashed process cannot create a stale-file
/// split-brain race. The OS releases the held lock when the process exits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceFileStore {
    primary: PathBuf,
}

impl WorkspaceFileStore {
    /// Creates a store rooted at `primary`.
    #[must_use]
    pub fn new(primary: impl Into<PathBuf>) -> Self {
        Self {
            primary: primary.into(),
        }
    }

    /// Returns the primary snapshot path.
    #[must_use]
    pub fn primary_path(&self) -> &Path {
        &self.primary
    }

    /// Returns the sibling path used for the last successfully committed
    /// primary snapshot.
    #[must_use]
    pub fn last_good_path(&self) -> PathBuf {
        let mut path = self.primary.clone();
        let suffix = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}.last-good"))
            .unwrap_or_else(|| String::from("last-good"));
        path.set_extension(suffix);
        path
    }

    /// Returns the sibling file used for the stable panel-instance registry.
    #[must_use]
    pub fn instances_path(&self) -> PathBuf {
        sibling_with_suffix(&self.primary, "instances")
    }

    /// Returns the last-good panel-instance registry path.
    #[must_use]
    pub fn instances_last_good_path(&self) -> PathBuf {
        sibling_with_suffix(&self.instances_path(), "last-good")
    }

    /// Returns the sibling temporary path used during the next save.
    #[must_use]
    pub fn temporary_path(&self) -> PathBuf {
        let mut path = self.primary.clone();
        let suffix = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}.tmp"))
            .unwrap_or_else(|| String::from("tmp"));
        path.set_extension(suffix);
        path
    }

    fn unique_sibling(&self, suffix: &str) -> PathBuf {
        let mut path = self.primary.clone();
        let nonce = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let process = std::process::id();
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}.{suffix}.{process}.{time}.{nonce}"))
            .unwrap_or_else(|| format!("{suffix}.{process}.{time}.{nonce}"));
        path.set_extension(extension);
        path
    }

    fn stable_lock_path(&self) -> PathBuf {
        let mut path = self.primary.clone();
        let suffix = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}.lock"))
            .unwrap_or_else(|| String::from("lock"));
        path.set_extension(suffix);
        path
    }

    /// Loads primary, then last-good, then returns the supplied default.
    ///
    /// Decode failures are intentionally retained in the result so callers can
    /// surface diagnostics instead of silently masking corruption.
    pub fn load(
        &self,
        options: WorkspaceDecodeOptions<'_>,
        default: WorkspaceManager,
    ) -> LoadedWorkspace {
        self.load_with(options.limits.max_bytes, default, |bytes| {
            decode_workspace(bytes, options).map_err(WorkspaceStoreError::Codec)
        })
    }

    /// Loads using the registered panel IDs as the persistence allow-list.
    ///
    /// This is the safe filesystem boundary for applications that have a
    /// `PanelRegistry`; unknown panels are filtered by the nested layout
    /// decoder before a renderer can observe the result.
    pub fn load_with_registry(
        &self,
        registry: &PanelRegistry,
        known_documents: Option<&[&str]>,
        limits: WorkspaceDecodeLimits,
        default: WorkspaceManager,
    ) -> LoadedWorkspace {
        self.load_with(limits.max_bytes, default, |bytes| {
            decode_workspace_with_registry(bytes, registry, known_documents, limits)
                .map_err(WorkspaceStoreError::Codec)
        })
    }

    /// Loads after explicitly migrating the snapshot to the canonical
    /// workspace wire version, then applies the caller's registry policy.
    pub fn load_with_migration(
        &self,
        migration: &WorkspaceMigrationRegistry,
        options: WorkspaceDecodeOptions<'_>,
        default: WorkspaceManager,
    ) -> LoadedWorkspace {
        self.load_with(options.limits.max_bytes, default, |bytes| {
            let migrated =
                migration.migrate_with_options(bytes, WorkspaceDecodeOptions { ..options })?;
            decode_workspace(&migrated, options).map_err(WorkspaceStoreError::Codec)
        })
    }

    /// Registry-aware variant of [`Self::load_with_migration`].
    pub fn load_with_registry_and_migration(
        &self,
        migration: &WorkspaceMigrationRegistry,
        registry: &PanelRegistry,
        known_documents: Option<&[&str]>,
        limits: WorkspaceDecodeLimits,
        default: WorkspaceManager,
    ) -> LoadedWorkspace {
        self.load_with(limits.max_bytes, default, |bytes| {
            let migrated = migration.migrate_with_options(
                bytes,
                WorkspaceDecodeOptions {
                    known_documents,
                    limits,
                    ..WorkspaceDecodeOptions::default()
                },
            )?;
            decode_workspace_with_registry(&migrated, registry, known_documents, limits)
                .map_err(WorkspaceStoreError::Codec)
        })
    }

    /// Loads the host-owned panel-instance registry with primary/last-good
    /// recovery. The registry is intentionally a separate `PRIN` snapshot so
    /// workspace layout bytes remain compatible with descriptor-only IDs.
    pub fn load_instances(&self, default: PanelInstanceRegistry) -> LoadedInstances {
        let primary = self.instances_path();
        let last_good = self.instances_last_good_path();
        let primary_error = match read_bounded(&primary, INSTANCE_MAX_BYTES) {
            Ok(bytes) => match decode_instance_registry(&bytes) {
                Ok(registry) => {
                    return LoadedInstances {
                        registry,
                        source: WorkspaceLoadSource::Primary,
                        primary_error: None,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(WorkspaceStoreError::Instance(error)),
            },
            Err(WorkspaceStoreError::Io(error)) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(error),
        };
        let last_good_error = match read_bounded(&last_good, INSTANCE_MAX_BYTES) {
            Ok(bytes) => match decode_instance_registry(&bytes) {
                Ok(registry) => {
                    return LoadedInstances {
                        registry,
                        source: WorkspaceLoadSource::LastGood,
                        primary_error,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(WorkspaceStoreError::Instance(error)),
            },
            Err(WorkspaceStoreError::Io(error)) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(error),
        };
        LoadedInstances {
            registry: default,
            source: WorkspaceLoadSource::Default,
            primary_error,
            last_good_error,
        }
    }

    fn load_with<F>(
        &self,
        max_bytes: usize,
        default: WorkspaceManager,
        mut decode: F,
    ) -> LoadedWorkspace
    where
        F: FnMut(&[u8]) -> Result<WorkspaceManager, WorkspaceStoreError>,
    {
        let primary_error = match read_bounded(&self.primary, max_bytes) {
            Ok(bytes) => match decode(&bytes) {
                Ok(manager) => {
                    return LoadedWorkspace {
                        manager,
                        source: WorkspaceLoadSource::Primary,
                        primary_error: None,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(error),
            },
            Err(WorkspaceStoreError::Io(error)) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(error),
        };

        let last_good = self.last_good_path();
        let last_good_error = match read_bounded(&last_good, max_bytes) {
            Ok(bytes) => match decode(&bytes) {
                Ok(manager) => {
                    return LoadedWorkspace {
                        manager,
                        source: WorkspaceLoadSource::LastGood,
                        primary_error,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(error),
            },
            Err(WorkspaceStoreError::Io(error)) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(error),
        };

        LoadedWorkspace {
            manager: default,
            source: WorkspaceLoadSource::Default,
            primary_error,
            last_good_error,
        }
    }

    /// Encodes and commits a snapshot using unique sibling temporary files.
    ///
    /// The previous primary is copied to last-good before the new temporary
    /// file is atomically renamed into the primary path. A failed encode,
    /// validation, copy, or write before the corresponding rename leaves the
    /// existing primary untouched. After a rename, a directory-sync failure
    /// returns an error even though the new snapshot may already be committed;
    /// callers must treat that result as commit status unknown and reload.
    pub fn save(&self, manager: &WorkspaceManager) -> Result<(), WorkspaceStoreError> {
        if let Some(parent) = self
            .primary
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        let _lock = StoreLock::acquire(&self.stable_lock_path())?;
        let bytes = encode_workspace(manager)?;
        let temporary = self.unique_sibling("tmp");
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);

        if self.primary.exists() {
            // Never replace a healthy last-good snapshot with a corrupt
            // primary, and commit the replacement through its own synced
            // temporary file.
            let primary_bytes = fs::read(&self.primary)?;
            decode_workspace(&primary_bytes, WorkspaceDecodeOptions::default())?;
            let last_good_temporary = self.unique_sibling("last-good-tmp");
            copy_and_sync(&self.primary, &last_good_temporary)?;
            replace_file(&last_good_temporary, &self.last_good_path())?;
            sync_parent(&self.last_good_path())?;
        }
        replace_file(&temporary, &self.primary)?;
        sync_parent(&self.primary)?;
        Ok(())
    }

    /// Saves the host-owned panel-instance registry using the same lock and
    /// atomic last-good protocol as workspace snapshots.
    pub fn save_instances(
        &self,
        registry: &PanelInstanceRegistry,
    ) -> Result<(), WorkspaceStoreError> {
        let primary = self.instances_path();
        let last_good = self.instances_last_good_path();
        if let Some(parent) = primary.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let _lock = StoreLock::acquire(&self.stable_lock_path())?;
        let bytes = encode_instance_registry(registry)?;
        let temporary = unique_sibling(&primary, "tmp");
        write_and_sync(&temporary, &bytes)?;
        if primary.exists() {
            let previous = fs::read(&primary)?;
            decode_instance_registry(&previous)?;
            let last_good_temporary = unique_sibling(&primary, "last-good-tmp");
            copy_and_sync(&primary, &last_good_temporary)?;
            replace_file(&last_good_temporary, &last_good)?;
            sync_parent(&last_good)?;
        }
        replace_file(&temporary, &primary)?;
        sync_parent(&primary)?;
        Ok(())
    }
}

const INSTANCE_MAX_BYTES: usize = 16 * 1024 * 1024;

fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>, WorkspaceStoreError> {
    let metadata = fs::metadata(path).map_err(WorkspaceStoreError::Io)?;
    if metadata.len() > max_bytes as u64 {
        return Err(WorkspaceStoreError::Codec(
            WorkspacePersistError::ResourceLimit("bytes"),
        ));
    }
    fs::read(path).map_err(WorkspaceStoreError::Io)
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut result = path.to_path_buf();
    let extension = result
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!("{extension}.{suffix}"))
        .unwrap_or_else(|| suffix.to_owned());
    result.set_extension(extension);
    result
}

fn unique_sibling(path: &Path, suffix: &str) -> PathBuf {
    let nonce = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let process = std::process::id();
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut result = path.to_path_buf();
    let extension = result
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!("{extension}.{suffix}.{process}.{time}.{nonce}"))
        .unwrap_or_else(|| format!("{suffix}.{process}.{time}.{nonce}"));
    result.set_extension(extension);
    result
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

struct StoreLock {
    file: fs::File,
}

impl StoreLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let deadline = Instant::now() + Duration::from_millis(LOCK_WAIT_ATTEMPTS as u64);
        while Instant::now() < deadline {
            match file.try_lock_exclusive() {
                Ok(true) => return Ok(Self { file }),
                Ok(false) => thread::sleep(Duration::from_millis(1)),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "timed out acquiring workspace store lock",
        ))
    }
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn copy_and_sync(source: &Path, destination: &Path) -> io::Result<()> {
    fs::copy(source, destination)?;
    let file = fs::File::open(destination)?;
    file.sync_all()
}

fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    // `TempPath::persist` uses the platform's atomic overwrite primitive,
    // including MoveFileExW on Windows, while keeping the source path scoped
    // for cleanup if persistence fails.
    TempPath::try_from_path(source.to_path_buf())?
        .persist(destination)
        .map_err(|error| error.error)
}

fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> io::Result<()> {
    fs::File::open(parent_or_current(path))?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DockLayout, DockNode, Instancing, PanelDescriptor, PanelInstanceRegistry, PanelRegistry,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_store() -> (WorkspaceFileStore, PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("prism-dock-store-{nonce}"));
        let store = WorkspaceFileStore::new(directory.join("workspace.prism"));
        (store, directory)
    }

    fn manager(panel: &str) -> WorkspaceManager {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager
            .add_perspective(
                "default",
                &DockLayout {
                    root: DockNode::panel(panel),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        manager
    }

    fn instances(panel_count: usize) -> PanelInstanceRegistry {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("logs", "Logs").with_instancing(Instancing::Multi))
            .unwrap();
        let mut registry = PanelInstanceRegistry::new();
        for _ in 0..panel_count {
            registry.create(&panels, "logs").unwrap();
        }
        registry
    }

    #[test]
    fn save_replaces_primary_and_preserves_last_good() {
        let (store, directory) = temp_store();
        let first = manager("first");
        let second = manager("second");
        store.save(&first).unwrap();
        store.save(&second).unwrap();

        let loaded = store.load(WorkspaceDecodeOptions::default(), WorkspaceManager::new());
        assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
        assert_eq!(
            loaded.manager.active_layout().unwrap().root,
            DockNode::panel("second")
        );
        assert_eq!(
            decode_workspace(
                &fs::read(store.last_good_path()).unwrap(),
                WorkspaceDecodeOptions::default()
            )
            .unwrap()
            .active_layout()
            .unwrap()
            .root,
            DockNode::panel("first")
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn instance_snapshots_use_primary_and_last_good_recovery() {
        let (store, directory) = temp_store();
        let first = instances(1);
        let second = instances(2);
        store.save_instances(&first).unwrap();
        store.save_instances(&second).unwrap();

        let loaded = store.load_instances(PanelInstanceRegistry::new());
        assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
        assert_eq!(loaded.registry, second);
        assert_eq!(
            decode_instance_registry(&fs::read(store.instances_last_good_path()).unwrap()).unwrap(),
            first
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn corrupt_primary_recovers_last_good_and_reports_primary_error() {
        let (store, directory) = temp_store();
        let expected = manager("healthy");
        store.save(&expected).unwrap();
        fs::write(store.primary_path(), b"corrupt").unwrap();
        fs::copy(store.primary_path(), store.last_good_path()).unwrap();
        // Restore a valid last-good fixture after intentionally corrupting primary.
        fs::write(store.last_good_path(), encode_workspace(&expected).unwrap()).unwrap();

        let loaded = store.load(WorkspaceDecodeOptions::default(), WorkspaceManager::new());
        assert_eq!(loaded.source, WorkspaceLoadSource::LastGood);
        assert!(loaded.primary_error.is_some());
        assert_eq!(loaded.manager, expected);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn missing_or_corrupt_pair_returns_default_with_diagnostics() {
        let (store, directory) = temp_store();
        let default = manager("default");
        let loaded = store.load(WorkspaceDecodeOptions::default(), default.clone());
        assert_eq!(loaded.source, WorkspaceLoadSource::Default);
        assert_eq!(loaded.manager, default);
        assert!(loaded.primary_error.is_none());
        assert!(loaded.last_good_error.is_none());

        fs::create_dir_all(directory.as_path()).unwrap();
        fs::write(store.primary_path(), b"bad").unwrap();
        fs::write(store.last_good_path(), b"also bad").unwrap();
        let loaded = store.load(WorkspaceDecodeOptions::default(), default.clone());
        assert_eq!(loaded.source, WorkspaceLoadSource::Default);
        assert!(loaded.primary_error.is_some());
        assert!(loaded.last_good_error.is_some());
        assert_eq!(loaded.manager, default);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn registry_aware_load_filters_unknown_panels_before_returning_state() {
        let (store, directory) = temp_store();
        store.save(&manager("unknown")).unwrap();
        let mut registry = PanelRegistry::new();
        registry
            .register(PanelDescriptor::new("known", "Known"))
            .unwrap();

        let loaded = store.load_with_registry(
            &registry,
            None,
            WorkspaceDecodeLimits::default(),
            WorkspaceManager::new(),
        );
        assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
        assert_eq!(
            loaded.manager.active_layout().unwrap().root,
            DockNode::Empty
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn corrupt_primary_does_not_overwrite_healthy_last_good() {
        let (store, directory) = temp_store();
        let first = manager("first");
        let second = manager("second");
        store.save(&first).unwrap();
        store.save(&second).unwrap();
        fs::write(store.primary_path(), b"corrupt").unwrap();

        assert!(store.save(&manager("third")).is_err());
        let recovered = decode_workspace(
            &fs::read(store.last_good_path()).unwrap(),
            WorkspaceDecodeOptions::default(),
        )
        .unwrap();
        assert_eq!(
            recovered.active_layout().unwrap().root,
            DockNode::panel("first")
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn empty_parent_paths_sync_to_current_directory() {
        assert_eq!(
            parent_or_current(Path::new("workspace.prism")),
            Path::new(".")
        );
        assert_eq!(parent_or_current(Path::new("")), Path::new("."));
        assert_eq!(
            parent_or_current(Path::new("nested/workspace.prism")),
            Path::new("nested")
        );
    }

    #[test]
    fn advisory_lock_is_reusable_without_deleting_stable_path() {
        let (_store, directory) = temp_store();
        fs::create_dir_all(&directory).unwrap();
        let lock_path = directory.join("workspace.prism.lock");
        let first = StoreLock::acquire(&lock_path).unwrap();
        assert!(lock_path.is_file());
        drop(first);
        let second = StoreLock::acquire(&lock_path).unwrap();
        assert!(lock_path.is_file());
        drop(second);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn concurrent_saves_are_serialized_and_leave_valid_snapshots() {
        let (store, directory) = temp_store();
        let first_store = store.clone();
        let second_store = store.clone();
        let first = thread::spawn(move || first_store.save(&manager("first")));
        let second = thread::spawn(move || second_store.save(&manager("second")));
        assert!(first.join().unwrap().is_ok());
        assert!(second.join().unwrap().is_ok());

        let loaded = store.load(WorkspaceDecodeOptions::default(), WorkspaceManager::new());
        assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
        assert!(matches!(
            loaded.manager.active_layout().unwrap().root,
            DockNode::TabStack { .. }
        ));
        let _ = fs::remove_dir_all(directory);
    }
}
