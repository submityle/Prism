//! `std` file boundary for workspace snapshots.
//!
//! The core workspace model and codec remain renderer-independent and
//! `no_std + alloc`. This module is the deliberately narrow filesystem
//! adapter: it writes a sibling temporary file, flushes it, preserves the
//! previous primary snapshot as `last-good`, then atomically renames the
//! temporary file into place.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{
    decode_workspace, decode_workspace_with_registry, encode_workspace, PanelRegistry,
    WorkspaceDecodeLimits, WorkspaceDecodeOptions, WorkspaceManager, WorkspacePersistError,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const LOCK_WAIT_ATTEMPTS: usize = 5_000;
const STALE_LOCK_AGE: Duration = Duration::from_secs(30);

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

/// Filesystem or codec failure at the workspace storage boundary.
#[derive(Debug)]
pub enum WorkspaceStoreError {
    /// Filesystem operation failed.
    Io(io::Error),
    /// Snapshot bytes failed the versioned workspace codec.
    Codec(WorkspacePersistError),
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

/// A serialized primary/last-good workspace snapshot pair.
///
/// Saves are serialized through a sibling lock file, write unique temporary
/// siblings, and sync both file contents and (on Unix) the containing
/// directory before returning. This protects the codec boundary from
/// concurrent writers and ordinary process crashes; filesystem-specific power
/// loss guarantees remain the responsibility of the underlying filesystem.
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
        self.load_with(default, |bytes| decode_workspace(bytes, options))
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
        self.load_with(default, |bytes| {
            decode_workspace_with_registry(bytes, registry, known_documents, limits)
        })
    }

    fn load_with<F>(&self, default: WorkspaceManager, mut decode: F) -> LoadedWorkspace
    where
        F: FnMut(&[u8]) -> Result<WorkspaceManager, WorkspacePersistError>,
    {
        let primary_error = match fs::read(&self.primary) {
            Ok(bytes) => match decode(&bytes) {
                Ok(manager) => {
                    return LoadedWorkspace {
                        manager,
                        source: WorkspaceLoadSource::Primary,
                        primary_error: None,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(WorkspaceStoreError::Codec(error)),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(WorkspaceStoreError::Io(error)),
        };

        let last_good = self.last_good_path();
        let last_good_error = match fs::read(last_good) {
            Ok(bytes) => match decode(&bytes) {
                Ok(manager) => {
                    return LoadedWorkspace {
                        manager,
                        source: WorkspaceLoadSource::LastGood,
                        primary_error,
                        last_good_error: None,
                    };
                }
                Err(error) => Some(WorkspaceStoreError::Codec(error)),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => Some(WorkspaceStoreError::Io(error)),
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
    /// validation, copy, or write leaves the existing primary untouched.
    pub fn save(&self, manager: &WorkspaceManager) -> Result<(), WorkspaceStoreError> {
        let bytes = encode_workspace(manager)?;
        if let Some(parent) = self.primary.parent() {
            fs::create_dir_all(parent)?;
        }

        let _lock = StoreLock::acquire(&self.stable_lock_path())?;
        let temporary = self.unique_sibling("tmp");
        let temporary_guard = TempPath::new(temporary.clone());
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
            let last_good_guard = TempPath::new(last_good_temporary.clone());
            copy_and_sync(&self.primary, &last_good_temporary)?;
            replace_file(&last_good_temporary, &self.last_good_path())?;
            last_good_guard.disarm();
            sync_parent(&self.last_good_path())?;
        }
        replace_file(&temporary, &self.primary)?;
        temporary_guard.disarm();
        sync_parent(&self.primary)?;
        Ok(())
    }
}

struct TempPath {
    path: PathBuf,
    armed: bool,
}

impl TempPath {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

struct StoreLock {
    path: PathBuf,
}

impl StoreLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        for _ in 0..LOCK_WAIT_ATTEMPTS {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    let _ = file.sync_all();
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    if let Ok(metadata) = fs::metadata(path)
                        && let Ok(modified) = metadata.modified()
                        && modified.elapsed().unwrap_or_default() > STALE_LOCK_AGE
                    {
                        let _ = fs::remove_file(path);
                    }
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
        let _ = fs::remove_file(&self.path);
    }
}

fn copy_and_sync(source: &Path, destination: &Path) -> io::Result<()> {
    fs::copy(source, destination)?;
    let file = fs::File::open(destination)?;
    file.sync_all()
}

fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockLayout, DockNode};
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
            .register(crate::PanelDescriptor::new("known", "Known"))
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
