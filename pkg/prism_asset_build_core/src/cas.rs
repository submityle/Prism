use crate::types::ContentHash;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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

/// Result of a conservative local CAS reachability sweep.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CasSweepReport {
    /// Valid, hash-shaped blob files inspected.
    pub scanned: usize,
    /// Unreachable blobs removed after the grace period.
    pub deleted: usize,
    /// Blobs retained because they are reachable or still within the grace
    /// period (or because the sweep was deferred).
    pub retained: usize,
    /// True when an in-flight publication prevented a safe sweep.
    pub deferred_due_to_pending_transaction: bool,
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

    /// Removes old blobs that are not in `roots`.
    ///
    /// The caller is responsible for deriving roots from durable publication
    /// state and for serializing the sweep with writers. This method only
    /// visits the two-level immutable CAS layout and never follows symlinks.
    pub fn sweep_unreachable(
        &self,
        roots: &BTreeSet<ContentHash>,
        grace: Duration,
    ) -> Result<CasSweepReport, CasError> {
        let mut report = CasSweepReport::default();
        let mut dirty_shards = BTreeSet::new();

        for shard in fs::read_dir(&self.root)? {
            let shard = shard?;
            let shard_type = shard.file_type()?;
            if !shard_type.is_dir() {
                continue;
            }
            let shard_name = shard.file_name();
            let Some(shard_text) = shard_name.to_str() else {
                continue;
            };
            if shard_text.len() != 2 || decode_hex_byte(shard_text.as_bytes()).is_none() {
                continue;
            }

            for entry in fs::read_dir(shard.path())? {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if name.len() != 62 {
                    continue;
                }
                let Some(hash) = decode_hash(shard_text, name) else {
                    continue;
                };
                report.scanned += 1;
                if roots.contains(&hash) {
                    report.retained += 1;
                    continue;
                }
                let modified = entry.metadata()?.modified()?;
                let old_enough = modified.elapsed().map(|age| age >= grace).unwrap_or(false);
                if !old_enough {
                    report.retained += 1;
                    continue;
                }
                fs::remove_file(entry.path())?;
                dirty_shards.insert(shard.path());
                report.deleted += 1;
            }
        }

        for shard in dirty_shards {
            sync_directory(&shard)?;
        }
        if report.deleted != 0 {
            sync_directory(&self.root)?;
        }
        Ok(report)
    }

    fn temp_path(&self, parent: &Path) -> PathBuf {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        parent.join(format!(".tmp-{}-{id}", std::process::id()))
    }
}

fn decode_hex_byte(bytes: &[u8]) -> Option<u8> {
    if bytes.len() != 2 {
        return None;
    }
    Some((hex_nibble(bytes[0])? << 4) | hex_nibble(bytes[1])?)
}

fn decode_hash(shard: &str, tail: &str) -> Option<ContentHash> {
    let mut bytes = [0u8; 32];
    let full = format!("{shard}{tail}");
    if full.len() != 64 {
        return None;
    }
    for (index, slot) in bytes.iter_mut().enumerate() {
        *slot = decode_hex_byte(full.as_bytes().get(index * 2..index * 2 + 2)?)?;
    }
    Some(ContentHash(bytes))
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
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

    #[test]
    fn sweeps_only_old_unreachable_blobs() {
        let dir = temp_dir();
        let cas = ContentAddressedStore::open(&dir).unwrap();
        let reachable = cas.put(b"reachable").unwrap();
        let orphan = cas.put(b"orphan").unwrap();
        let report = cas
            .sweep_unreachable(&BTreeSet::from([reachable]), Duration::from_secs(0))
            .unwrap();
        assert_eq!(report.scanned, 2);
        assert_eq!(report.deleted, 1);
        assert_eq!(report.retained, 1);
        assert!(cas.read(reachable).is_ok());
        assert!(
            matches!(cas.read(orphan), Err(CasError::Io(error)) if error.kind() == io::ErrorKind::NotFound)
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
