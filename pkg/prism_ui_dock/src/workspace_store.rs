//! `std` file boundary for workspace snapshots.
//!
//! The core workspace model and codec remain renderer-independent and
//! `no_std + alloc`. This module is the deliberately narrow filesystem
//! adapter: it writes a sibling temporary file, flushes it, preserves the
//! previous primary snapshot as `last-good`, then atomically renames the
//! temporary file into place.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::{
    decode_workspace, encode_workspace, WorkspaceDecodeOptions, WorkspaceManager,
    WorkspacePersistError,
};

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

/// A crash-safe primary/last-good workspace snapshot pair.
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

    /// Loads primary, then last-good, then returns the supplied default.
    ///
    /// Decode failures are intentionally retained in the result so callers can
    /// surface diagnostics instead of silently masking corruption.
    pub fn load(
        &self,
        options: WorkspaceDecodeOptions<'_>,
        default: WorkspaceManager,
    ) -> LoadedWorkspace {
        let primary_error = match fs::read(&self.primary) {
            Ok(bytes) => match decode_workspace(&bytes, options) {
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
            Ok(bytes) => match decode_workspace(&bytes, options) {
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

    /// Encodes and commits a snapshot using a sibling temporary file.
    ///
    /// The previous primary is copied to last-good before the new temporary
    /// file is atomically renamed into the primary path. A failed encode or
    /// write leaves the existing primary untouched.
    pub fn save(&self, manager: &WorkspaceManager) -> Result<(), WorkspaceStoreError> {
        let bytes = encode_workspace(manager)?;
        if let Some(parent) = self.primary.parent() {
            fs::create_dir_all(parent)?;
        }

        let temporary = self.temporary_path();
        let mut file = fs::File::create(&temporary)?;
        io::Write::write_all(&mut file, &bytes)?;
        file.sync_all()?;
        drop(file);

        if self.primary.exists() {
            fs::copy(&self.primary, self.last_good_path())?;
        }
        if let Err(error) = fs::rename(&temporary, &self.primary) {
            let _ = fs::remove_file(&temporary);
            return Err(WorkspaceStoreError::Io(error));
        }
        Ok(())
    }
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
}
