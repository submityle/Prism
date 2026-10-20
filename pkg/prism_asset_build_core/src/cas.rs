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
    /// A running build holds a shared output lease, protecting not-yet-
    /// published CAS blobs even before its publication journal Prepare.
    pub deferred_due_to_active_build: bool,
}

/// Local immutable content-addressed storage. It verifies on both insertion
/// collision and every read; files are named only by their BLAKE3 content hash.
#[derive(Clone, Debug)]
pub struct ContentAddressedStore {
    root: PathBuf,
}

/// The identity and exact length of a successfully persisted CAS stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CasWriteResult {
    pub content_hash: ContentHash,
    pub byte_len: u64,
}

/// A bounded, incrementally hashed CAS stream. Dropping it discards unpublished
/// bytes. A failed write or flush poisons the stream, so `finish` cannot publish
/// a successful prefix. Interrupted system calls are retried internally.
///
/// This low-level writer does not acquire a revision-store output lease. The
/// caller must coordinate GC until the result has been published or discarded.
#[must_use = "finish the writer to publish its bytes, or drop it to discard them"]
pub struct CasWriter {
    store: ContentAddressedStore,
    temporary: Option<tempfile::NamedTempFile>,
    hasher: blake3::Hasher,
    byte_len: u64,
    max_bytes: u64,
    failure: Option<io::Error>,
}

impl CasWriter {
    fn check_writable(&self) -> io::Result<()> {
        match &self.failure {
            Some(error) => Err(io::Error::new(error.kind(), error.to_string())),
            None => Ok(()),
        }
    }

    fn poison(&mut self, error: io::Error) -> io::Error {
        self.failure = Some(io::Error::new(error.kind(), error.to_string()));
        // NamedTempFile closes and unlinks the incomplete output immediately.
        drop(self.temporary.take());
        error
    }

    /// Flushes and syncs the complete stream before publishing its immutable
    /// identity. Existing content is verified with the store's fixed buffer;
    /// a corrupt occupant is atomically repaired as in `put`.
    pub fn finish(mut self) -> Result<CasWriteResult, CasError> {
        self.flush()?;
        if let Err(error) = self
            .temporary
            .as_ref()
            .expect("unpoisoned writer owns its temporary file")
            .as_file()
            .sync_all()
        {
            return Err(self.poison(error).into());
        }
        let result = CasWriteResult {
            content_hash: ContentHash(*self.hasher.finalize().as_bytes()),
            byte_len: self.byte_len,
        };
        let path = self.store.blob_path(result.content_hash);
        let parent = path.parent().expect("blob path always has parent");
        fs::create_dir_all(parent)?;
        let temporary = self
            .temporary
            .take()
            .expect("unpoisoned writer owns its temporary file");
        match temporary.persist_noclobber(&path) {
            Ok(file) => drop(file),
            Err(error) => {
                let temporary = error.file;
                if error.error.kind() != io::ErrorKind::AlreadyExists {
                    return Err(CasError::Io(error.error));
                }
                match self.store.verify(result.content_hash) {
                    Ok(length) if length == result.byte_len => drop(temporary),
                    Ok(_) => {
                        return Err(CasError::Io(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "existing CAS blob has an unexpected length",
                        )));
                    }
                    Err(CasError::Corrupt { .. }) => {
                        // Match put's repair policy. A concurrent repair can
                        // only replace this path with the same valid identity.
                        let file = temporary
                            .persist(&path)
                            .map_err(|error| CasError::Io(error.error))?;
                        drop(file);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        sync_directory(parent)?;
        sync_directory(self.store.root())?;
        Ok(result)
    }
}

impl Write for CasWriter {
    /// Rejects an entire call that would exceed the budget, poisoning the
    /// writer even if earlier calls succeeded. Only bytes accepted by the
    /// underlying file write advance the hash and exact length.
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.check_writable()?;
        let requested_len = u64::try_from(bytes.len())
            .ok()
            .and_then(|count| self.byte_len.checked_add(count));
        let Some(requested_len) = requested_len else {
            return Err(self.poison(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CAS write length overflow",
            )));
        };
        if requested_len > self.max_bytes {
            return Err(self.poison(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CAS write exceeds maximum byte budget",
            )));
        }
        let count = loop {
            match self
                .temporary
                .as_mut()
                .expect("unpoisoned writer owns its temporary file")
                .as_file_mut()
                .write(bytes)
            {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(self.poison(error)),
                Ok(0) if !bytes.is_empty() => {
                    return Err(self.poison(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "CAS file write made no progress",
                    )));
                }
                Ok(count) => break count,
            }
        };
        self.hasher.update(&bytes[..count]);
        // count <= bytes.len(), whose addition was checked before the write.
        self.byte_len += count as u64;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check_writable()?;
        loop {
            match self
                .temporary
                .as_mut()
                .expect("unpoisoned writer owns its temporary file")
                .as_file_mut()
                .flush()
            {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(self.poison(error)),
                Ok(()) => return Ok(()),
            }
        }
    }
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

    /// Starts a stream with an explicit maximum payload length. Zero permits
    /// an empty artifact. The temporary file is on the CAS filesystem, outside
    /// the hash-shaped blob layout, and is not visible to readers or sweeps.
    pub fn begin_write(&self, max_bytes: u64) -> Result<CasWriter, CasError> {
        let temporary = tempfile::Builder::new()
            .prefix(".cas-write-")
            .tempfile_in(&self.root)?;
        Ok(CasWriter {
            store: self.clone(),
            temporary: Some(temporary),
            hasher: blake3::Hasher::new(),
            byte_len: 0,
            max_bytes,
            failure: None,
        })
    }

    pub fn put(&self, bytes: &[u8]) -> Result<ContentHash, CasError> {
        let hash = ContentHash::digest(bytes);
        let path = self.blob_path(hash);
        let parent = path.parent().expect("blob path always has parent");
        fs::create_dir_all(parent)?;
        if path.exists() {
            match self.verify(hash) {
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
                    self.verify(hash)?;
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

    /// Verifies a complete blob with a fixed 64 KiB buffer, returning its exact
    /// length. Publication and insertion collisions need no owned payload.
    pub fn verify(&self, hash: ContentHash) -> Result<u64, CasError> {
        let mut file = File::open(self.blob_path(hash))?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut length = 0u64;
        loop {
            let count = match file.read(&mut buffer) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            length = length.checked_add(count as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "CAS blob length overflow")
            })?;
        }
        let actual = ContentHash(*hasher.finalize().as_bytes());
        if actual != hash {
            return Err(CasError::Corrupt {
                expected: hash,
                actual,
            });
        }
        Ok(length)
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_DIR_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-cas-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            TEST_DIR_ID.fetch_add(1, Ordering::Relaxed)
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
    fn file_write_failure_poisons_stream_and_discards_successful_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressedStore::open(dir.path()).unwrap();
        let mut writer = cas.begin_write(64).unwrap();
        writer.write_all(b"successful prefix").unwrap();
        let path = writer.temporary.as_ref().unwrap().path().to_path_buf();
        // Replacing the owned descriptor with a read-only one deterministically
        // makes the next real File::write fail, without disk or permission races.
        let read_only = File::open(&path).unwrap();
        *writer.temporary.as_mut().unwrap().as_file_mut() = read_only;
        let failure = writer.write(b"rejected suffix").unwrap_err();
        assert_ne!(failure.kind(), io::ErrorKind::Interrupted);
        assert!(!path.exists());
        assert_eq!(writer.flush().unwrap_err().kind(), failure.kind());
        assert_eq!(writer.write(b"retry").unwrap_err().kind(), failure.kind());
        assert!(matches!(
            writer.finish(),
            Err(CasError::Io(error)) if error.kind() == failure.kind()
        ));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn stream_length_overflow_poisons_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressedStore::open(dir.path()).unwrap();
        let mut writer = cas.begin_write(u64::MAX).unwrap();
        // Exercise the u64 boundary without allocating or writing exabytes.
        writer.byte_len = u64::MAX;
        let failure = writer.write(b"x").unwrap_err();
        assert_eq!(failure.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(failure.to_string(), "CAS write length overflow");
        assert!(matches!(
            writer.finish(),
            Err(CasError::Io(error)) if error.kind() == io::ErrorKind::InvalidInput
        ));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn streaming_verification_checks_empty_multibuffer_truncated_and_repaired_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressedStore::open(dir.path()).unwrap();
        let empty = cas.put(&[]).unwrap();
        assert_eq!(cas.verify(empty).unwrap(), 0);
        let bytes: Vec<_> = (0..(3 * 64 * 1024 + 17))
            .map(|index| (index % 251) as u8)
            .collect();
        let hash = cas.put(&bytes).unwrap();
        assert_eq!(cas.verify(hash).unwrap(), bytes.len() as u64);
        assert_eq!(cas.put(&bytes).unwrap(), hash);
        File::options()
            .write(true)
            .open(cas.blob_path(hash))
            .unwrap()
            .set_len(64 * 1024)
            .unwrap();
        assert!(matches!(cas.verify(hash), Err(CasError::Corrupt { .. })));
        assert_eq!(cas.put(&bytes).unwrap(), hash);
        assert_eq!(cas.verify(hash).unwrap(), bytes.len() as u64);
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(cas.blob_path(hash), corrupt).unwrap();
        assert!(matches!(cas.verify(hash), Err(CasError::Corrupt { .. })));
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
