//! Crash-safe single-file persistence and primary/backup recovery.

use crate::{CodecError, SettingsDocument};
use prism_settings::SettingsRegistry;
use std::{
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

/// Result of loading a primary file with an optional backup fallback.
#[derive(Debug, PartialEq)]
pub struct LoadOutcome {
    /// Decoded document selected for use.
    pub document: SettingsDocument,
    /// Whether the backup was used because the primary was unavailable or invalid.
    pub recovered_from_backup: bool,
    /// Primary load failure, when recovery selected the backup.
    pub primary_error: Option<String>,
}

/// Errors from persistence and recovery operations.
#[derive(Debug)]
pub enum StoreError {
    /// Filesystem operation failed.
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying filesystem error.
        source: io::Error,
    },
    /// Document encoding or decoding failed.
    Codec(CodecError),
    /// Both primary and backup failed during recovery.
    Recovery {
        /// Primary failure description.
        primary: String,
        /// Backup failure description.
        backup: String,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "settings I/O failed for {}: {source}", path.display())
            }
            Self::Codec(error) => write!(f, "settings codec failed: {error}"),
            Self::Recovery { primary, backup } => write!(
                f,
                "settings recovery failed; primary: {primary}; backup: {backup}"
            ),
        }
    }
}

impl Error for StoreError {}

impl From<CodecError> for StoreError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

/// Writes a document using temp-file, flush, sync, backup, and rename ordering.
pub fn write_atomic(
    path: impl AsRef<Path>,
    document: &SettingsDocument,
    backup: Option<impl AsRef<Path>>,
) -> Result<(), StoreError> {
    write_atomic_rotating(path, document, backup, 1)
}

/// Writes a document while retaining up to `max_backups` generations.
///
/// The newest backup is written at `backup`, older generations use the
/// suffixes `.1`, `.2`, and so on.
pub fn write_atomic_rotating(
    path: impl AsRef<Path>,
    document: &SettingsDocument,
    backup: Option<impl AsRef<Path>>,
    max_backups: usize,
) -> Result<(), StoreError> {
    let path = path.as_ref();
    let encoded = document.encode()?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| StoreError::Io {
            path: parent.to_owned(),
            source,
        })?;
    }
    let temporary = temporary_path(path);
    let write_result = (|| {
        let mut file = fs::File::create(&temporary).map_err(|source| StoreError::Io {
            path: temporary.clone(),
            source,
        })?;
        io::Write::write_all(&mut file, encoded.as_bytes()).map_err(|source| StoreError::Io {
            path: temporary.clone(),
            source,
        })?;
        file.sync_all().map_err(|source| StoreError::Io {
            path: temporary.clone(),
            source,
        })?;
        if max_backups > 0
            && let Some(backup) = backup.as_ref().map(AsRef::as_ref)
            && path.exists()
        {
            if let Some(parent) = backup
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                    path: parent.to_owned(),
                    source,
                })?;
            }
            rotate_backups(backup, max_backups)?;
            fs::copy(path, backup).map_err(|source| StoreError::Io {
                path: backup.to_owned(),
                source,
            })?;
        }
        fs::rename(&temporary, path).map_err(|source| StoreError::Io {
            path: path.to_owned(),
            source,
        })?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            let directory = fs::File::open(parent).map_err(|source| StoreError::Io {
                path: parent.to_owned(),
                source,
            })?;
            directory.sync_all().map_err(|source| StoreError::Io {
                path: parent.to_owned(),
                source,
            })?;
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

/// Loads a primary document and falls back to a backup after decode or I/O failure.
pub fn load_with_backup(
    path: impl AsRef<Path>,
    backup: Option<impl AsRef<Path>>,
    registry: Option<&SettingsRegistry>,
) -> Result<LoadOutcome, StoreError> {
    let path = path.as_ref();
    match fs::read_to_string(path) {
        Ok(contents) => match SettingsDocument::decode_with_registry(&contents, registry) {
            Ok(document) => Ok(LoadOutcome {
                document,
                recovered_from_backup: false,
                primary_error: None,
            }),
            Err(error) => recover_backup(path, backup, registry, error.to_string()),
        },
        Err(error) => recover_backup(path, backup, registry, error.to_string()),
    }
}

fn recover_backup(
    path: &Path,
    backup: Option<impl AsRef<Path>>,
    registry: Option<&SettingsRegistry>,
    primary: String,
) -> Result<LoadOutcome, StoreError> {
    let Some(backup) = backup.map(|path| path.as_ref().to_owned()) else {
        return Err(StoreError::Recovery {
            primary,
            backup: "no backup configured".to_owned(),
        });
    };
    match fs::read_to_string(&backup) {
        Ok(contents) => match SettingsDocument::decode_with_registry(&contents, registry) {
            Ok(document) => Ok(LoadOutcome {
                document,
                recovered_from_backup: true,
                primary_error: Some(format!(
                    "{}: primary unavailable or invalid",
                    path.display()
                )),
            }),
            Err(error) => Err(StoreError::Recovery {
                primary,
                backup: error.to_string(),
            }),
        },
        Err(error) => Err(StoreError::Recovery {
            primary,
            backup: error.to_string(),
        }),
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    PathBuf::from(temporary)
}

fn rotate_backups(backup: &Path, max_backups: usize) -> Result<(), StoreError> {
    if max_backups == 0 {
        return Ok(());
    }
    for index in (1..max_backups).rev() {
        let source = if index == 1 {
            backup.to_owned()
        } else {
            backup_generation(backup, index - 1)
        };
        if source.exists() {
            let destination = backup_generation(backup, index);
            fs::rename(&source, &destination).map_err(|source| StoreError::Io {
                path: destination,
                source,
            })?;
        }
    }
    Ok(())
}

fn backup_generation(backup: &Path, generation: usize) -> PathBuf {
    PathBuf::from(format!("{}.{}", backup.display(), generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::Value;
    use tempfile::tempdir;

    #[test]
    fn atomic_write_creates_backup_and_recovery_reads_backup() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.toml.bak");
        let mut first = SettingsDocument::new(1);
        first.insert("audio.master_volume", Value::Float(0.5));
        write_atomic(&path, &first, Some(&backup)).unwrap();
        let mut second = SettingsDocument::new(1);
        second.insert("audio.master_volume", Value::Float(0.8));
        write_atomic(&path, &second, Some(&backup)).unwrap();
        assert!(path.exists());
        assert!(backup.exists());
        std::fs::write(&path, "not valid toml = [").unwrap();
        let recovered = load_with_backup(&path, Some(&backup), None).unwrap();
        assert!(recovered.recovered_from_backup);
        assert_eq!(recovered.document, first);
    }

    #[test]
    fn recovery_reports_both_failures() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.toml.bak");
        std::fs::write(&path, "bad").unwrap();
        std::fs::write(&backup, "also bad").unwrap();
        assert!(matches!(
            load_with_backup(&path, Some(&backup), None),
            Err(StoreError::Recovery { .. })
        ));
    }

    #[test]
    fn rotating_writer_keeps_bounded_backup_generations() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.toml.bak");
        for value in 0..4 {
            let mut document = SettingsDocument::new(1);
            document.insert("audio.master_volume", Value::Int(value));
            write_atomic_rotating(&path, &document, Some(&backup), 2).unwrap();
        }
        assert!(backup.exists());
        assert!(PathBuf::from(format!("{}.1", backup.display())).exists());
        assert!(!PathBuf::from(format!("{}.2", backup.display())).exists());
    }
}
