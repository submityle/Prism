use crate::types::ContentHash;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum CasError {
    Io(io::Error),
    Corrupt {
        expected: ContentHash,
        actual: ContentHash,
    },
}

impl std::fmt::Display for CasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "CAS I/O error: {e}"),
            Self::Corrupt { expected, actual } => write!(
                f,
                "CAS blob corrupt: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
        }
    }
}
impl std::error::Error for CasError {}
impl From<io::Error> for CasError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Local immutable content-addressed storage. It verifies on both insertion
/// collision and every read; files are named only by their BLAKE3 content hash.
#[derive(Clone, Debug)]
pub struct ContentAddressedStore {
    root: PathBuf,
}

impl ContentAddressedStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, CasError> {
        let root = root.into();
        create_dir_all_durable(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, hash: ContentHash) -> PathBuf {
        let hex = hash.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    pub fn put(&self, bytes: &[u8]) -> Result<ContentHash, CasError> {
        let hash = ContentHash::digest(bytes);
        let path = self.blob_path(hash);
        let parent = path.parent().expect("blob path always has parent");
        fs::create_dir_all(parent)?;
        if path.exists() {
            match self.read(hash) {
                Ok(_) => return Ok(hash),
                Err(CasError::Corrupt { .. }) => {
                    // The expected content hash names this path, so a corrupt
                    // occupant cannot be treated as an immutable valid blob.
                    // Replace it atomically with bytes whose digest we computed
                    // above; never return a cache hit for corrupted content.
                    let mut temp = tempfile::Builder::new()
                        .prefix(".cas-repair-")
                        .tempfile_in(parent)?;
                    temp.write_all(bytes)?;
                    temp.as_file().sync_all()?;
                    temp.persist(&path)
                        .map_err(|error| CasError::Io(error.error))?;
                    sync_directory(parent)?;
                    if let Some(cas_root) = parent.parent() {
                        sync_directory(cas_root)?;
                    }
                    return Ok(hash);
                }
                Err(error) => return Err(error),
            }
        }
        let temp = self.temp_path(parent);
        let result = (|| -> Result<(), CasError> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            match fs::rename(&temp, &path) {
                Ok(()) => Ok(()),
                Err(_e) if path.exists() => {
                    // A concurrent writer may have won. Never trust it without verification.
                    self.read(hash)?;
                    let _ = fs::remove_file(&temp);
                    Ok(())
                }
                Err(e) => Err(CasError::Io(e)),
            }
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result?;
        sync_directory(parent)?;
        if let Some(cas_root) = parent.parent() {
            sync_directory(cas_root)?;
        }
        Ok(hash)
    }

    pub fn read(&self, hash: ContentHash) -> Result<Vec<u8>, CasError> {
        let mut bytes = Vec::new();
        File::open(self.blob_path(hash))?.read_to_end(&mut bytes)?;
        let actual = ContentHash::digest(&bytes);
        if actual != hash {
            return Err(CasError::Corrupt {
                expected: hash,
                actual,
            });
        }
        Ok(bytes)
    }

    fn temp_path(&self, parent: &Path) -> PathBuf {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        parent.join(format!(".tmp-{}-{id}", std::process::id()))
    }
}

fn sync_directory(path: &Path) -> Result<(), CasError> {
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

/// Creates the CAS root and persists each newly-created directory entry in
/// its parent. Syncing only the shard and CAS root during `put` is not enough
/// when this public store is opened outside `RevisionStore`: after a crash,
/// the files could be durable while the CAS root's own parent entry is not.
fn create_dir_all_durable(path: &Path) -> Result<(), CasError> {
    let mut missing = Vec::new();
    let mut cursor = path;
    while !cursor.exists() {
        missing.push(cursor.to_path_buf());
        cursor = cursor.parent().ok_or_else(|| {
            CasError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CAS path has no existing ancestor",
            ))
        })?;
    }
    if !cursor.is_dir() {
        return Err(CasError::Io(io::Error::new(
            io::ErrorKind::NotADirectory,
            "existing CAS ancestor is not a directory",
        )));
    }
    fs::create_dir_all(path)?;
    for created in missing {
        let parent = created.parent().ok_or_else(|| {
            CasError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "created CAS directory has no parent",
            ))
        })?;
        sync_directory(parent)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-cas-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn detects_corruption_on_read() {
        let dir = temp_dir();
        let cas = ContentAddressedStore::open(&dir).unwrap();
        let hash = cas.put(b"original bytes").unwrap();
        fs::write(cas.blob_path(hash), b"corrupted bytes").unwrap();
        assert!(matches!(cas.read(hash), Err(CasError::Corrupt { .. })));
        fs::remove_dir_all(dir).unwrap();
    }
}
