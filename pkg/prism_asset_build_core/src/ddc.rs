//! A local derived-data cache for artifact descriptors.
//!
//! The DDC stores only the deterministic `ArtifactKey -> ArtifactDescriptor`
//! index.  Artifact bytes remain the responsibility of [`crate::cas`]; this
//! module never creates, reads, or deletes a CAS blob.

use crate::types::{ArtifactDescriptor, ArtifactKey, ContentHash};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MAGIC: &[u8] = b"PRISM-DDC\0";
const VERSION: u32 = 1;
const CHECKSUM_LEN: usize = 32;
const MAX_ENTRIES: u64 = 1_000_000;
const MAX_DEPENDENCIES: u64 = 1_000_000;
const TEMP_PREFIX: &str = ".index.ddc.tmp-";
static PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Failure while opening or mutating a local DDC.
#[derive(Debug)]
pub enum DdcError {
    Io(io::Error),
    InvalidFormat(&'static str),
    InvalidDescriptor(&'static str),
    Truncated,
    UnknownVersion(u32),
    HashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    KeyConflict {
        key: ArtifactKey,
        reason: &'static str,
    },
}

impl fmt::Display for DdcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "DDC I/O error: {error}"),
            Self::InvalidFormat(reason) => write!(f, "invalid DDC format: {reason}"),
            Self::InvalidDescriptor(reason) => write!(f, "invalid artifact descriptor: {reason}"),
            Self::Truncated => f.write_str("DDC index is truncated"),
            Self::UnknownVersion(version) => write!(f, "unknown DDC index version: {version}"),
            Self::HashMismatch { expected, actual } => write!(
                f,
                "DDC index hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::KeyConflict { key, reason } => {
                write!(f, "DDC key {} conflicts: {reason}", key.to_hex())
            }
        }
    }
}

impl std::error::Error for DdcError {}

impl From<io::Error> for DdcError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Persistent local derived-data cache.
///
/// Reads are snapshots refreshed at `open`; each mutation serializes with other
/// handles in this process, reloads the latest index, then publishes a synced
/// temporary file through [`tempfile::NamedTempFile::persist`]. On POSIX, the
/// containing directory is synced after replacement. The index is deliberately
/// not advertised as a cross-process transaction store: use one owner process
/// (or an external workspace lease) when multiple build workers share a
/// directory. `get`, `len`, and `is_empty` report this handle's last snapshot
/// until it mutates or is reopened. The cache is independent from the CAS.
///
/// Temporary files are never reaped here, because deleting another process's
/// active temporary file would be unsafe. An interrupted publish leaves an
/// unreferenced temporary file that can be removed by an operator after
/// verifying no writer is active.
#[derive(Clone, Debug)]
pub struct DerivedDataCache {
    root: PathBuf,
    entries: BTreeMap<ArtifactKey, ArtifactDescriptor>,
}

/// Short name for [`DerivedDataCache`].
pub type Ddc = DerivedDataCache;

/// Compatibility-oriented name for [`DerivedDataCache`].
pub type LocalDdc = DerivedDataCache;

impl DerivedDataCache {
    /// Opens (or creates) the DDC directory and restores its index.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DdcError> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let entries = read_index(&root.join("index.ddc"))?;
        Ok(Self { root, entries })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index_path(&self) -> PathBuf {
        self.root.join("index.ddc")
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the descriptor for `key`, if it is cached.
    pub fn get(&self, key: ArtifactKey) -> Option<ArtifactDescriptor> {
        self.entries.get(&key).cloned()
    }

    /// Inserts a descriptor under `key`.
    ///
    /// A descriptor is immutable for a given semantic key. Re-inserting the
    /// exact same descriptor is an idempotent success; a different descriptor
    /// is rejected as a key conflict instead of silently replacing a result.
    pub fn put(
        &mut self,
        key: ArtifactKey,
        descriptor: ArtifactDescriptor,
    ) -> Result<(), DdcError> {
        validate_entry(key, &descriptor)?;
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut current = read_index(&self.index_path())?;
        if let Some(existing) = current.get(&key) {
            if existing == &descriptor {
                self.entries = current;
                return Ok(());
            }
            return Err(DdcError::KeyConflict {
                key,
                reason: "key already maps to a different descriptor",
            });
        }

        current.insert(key, descriptor);
        self.commit_locked(current)
    }

    /// Inserts a descriptor using its embedded key.
    pub fn put_descriptor(&mut self, descriptor: ArtifactDescriptor) -> Result<(), DdcError> {
        self.put(descriptor.key, descriptor)
    }

    /// Removes `key` and durably publishes the resulting index.
    ///
    /// Returns whether an entry was present. A missing key is a successful
    /// no-op, which makes invalidation safe to repeat.
    pub fn invalidate(&mut self, key: ArtifactKey) -> Result<bool, DdcError> {
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let mut current = read_index(&self.index_path())?;
        if !current.contains_key(&key) {
            self.entries = current;
            return Ok(false);
        }
        current.remove(&key);
        self.commit_locked(current)?;
        Ok(true)
    }

    fn commit_locked(
        &mut self,
        entries: BTreeMap<ArtifactKey, ArtifactDescriptor>,
    ) -> Result<(), DdcError> {
        let bytes = encode_index(&entries)?;
        atomic_write(&self.index_path(), &bytes)?;
        self.entries = entries;
        Ok(())
    }
}

fn process_lock() -> &'static Mutex<()> {
    PROCESS_LOCK.get_or_init(|| Mutex::new(()))
}

fn read_index(path: &Path) -> Result<BTreeMap<ArtifactKey, ArtifactDescriptor>, DdcError> {
    match fs::read(path) {
        Ok(bytes) => decode_index(&bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error.into()),
    }
}

fn validate_entry(key: ArtifactKey, descriptor: &ArtifactDescriptor) -> Result<(), DdcError> {
    descriptor.validate().map_err(DdcError::InvalidDescriptor)?;
    if descriptor.key != key {
        return Err(DdcError::KeyConflict {
            key,
            reason: "descriptor key differs from index key",
        });
    }
    if descriptor.dependencies.len() as u64 > MAX_DEPENDENCIES {
        return Err(DdcError::InvalidDescriptor("too many dependencies"));
    }
    Ok(())
}

fn encode_index(entries: &BTreeMap<ArtifactKey, ArtifactDescriptor>) -> Result<Vec<u8>, DdcError> {
    if entries.len() as u64 > MAX_ENTRIES {
        return Err(DdcError::InvalidDescriptor("too many DDC entries"));
    }
    let mut body = Vec::new();
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&VERSION.to_le_bytes());
    body.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (key, descriptor) in entries {
        validate_entry(*key, descriptor)?;
        body.extend_from_slice(&key.0);
        body.extend_from_slice(&descriptor.key.0);
        body.extend_from_slice(&descriptor.kind.to_le_bytes());
        body.extend_from_slice(&descriptor.schema.to_le_bytes());
        body.extend_from_slice(&descriptor.content_hash.0);
        body.extend_from_slice(&descriptor.byte_len.to_le_bytes());
        body.extend_from_slice(&descriptor.alignment.to_le_bytes());
        body.extend_from_slice(&(descriptor.dependencies.len() as u64).to_le_bytes());
        for dependency in &descriptor.dependencies {
            body.extend_from_slice(&dependency.0);
        }
    }
    let checksum = ContentHash::digest(&body);
    body.extend_from_slice(&checksum.0);
    Ok(body)
}

fn decode_index(bytes: &[u8]) -> Result<BTreeMap<ArtifactKey, ArtifactDescriptor>, DdcError> {
    if bytes.len() < MAGIC.len() + 4 + 8 + CHECKSUM_LEN {
        return Err(DdcError::Truncated);
    }
    let body_len = bytes.len() - CHECKSUM_LEN;
    let expected = ContentHash(
        bytes[body_len..]
            .try_into()
            .expect("checksum length checked"),
    );
    let actual = ContentHash::digest(&bytes[..body_len]);
    if expected != actual {
        return Err(DdcError::HashMismatch { expected, actual });
    }

    let mut cursor = Cursor::new(&bytes[..body_len]);
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(DdcError::InvalidFormat("bad magic"));
    }
    let version = cursor.u32()?;
    if version != VERSION {
        return Err(DdcError::UnknownVersion(version));
    }
    let count = cursor.u64()?;
    if count > MAX_ENTRIES {
        return Err(DdcError::InvalidFormat("entry count exceeds limit"));
    }

    let mut entries = BTreeMap::new();
    let mut previous_key = None;
    for _ in 0..count {
        let key = ArtifactKey(cursor.array_32()?);
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(DdcError::InvalidFormat("keys are not strictly ordered"));
        }
        previous_key = Some(key);
        let descriptor = ArtifactDescriptor {
            key: ArtifactKey(cursor.array_32()?),
            kind: cursor.u128()?,
            schema: cursor.u32()?,
            content_hash: ContentHash(cursor.array_32()?),
            byte_len: cursor.u64()?,
            alignment: cursor.u32()?,
            dependencies: {
                let dependency_count = cursor.u64()?;
                if dependency_count > MAX_DEPENDENCIES {
                    return Err(DdcError::InvalidFormat("dependency count exceeds limit"));
                }
                let mut dependencies = Vec::with_capacity(dependency_count as usize);
                for _ in 0..dependency_count {
                    dependencies.push(ArtifactKey(cursor.array_32()?));
                }
                dependencies
            },
        };
        validate_entry(key, &descriptor)?;
        if entries.insert(key, descriptor).is_some() {
            return Err(DdcError::KeyConflict {
                key,
                reason: "duplicate key in index",
            });
        }
    }
    if !cursor.is_empty() {
        return Err(DdcError::InvalidFormat("trailing bytes"));
    }
    Ok(entries)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), DdcError> {
    let parent = path
        .parent()
        .ok_or(DdcError::InvalidFormat("DDC index has no parent"))?;
    fs::create_dir_all(parent)?;
    let mut temp = tempfile::Builder::new()
        .prefix(TEMP_PREFIX)
        .tempfile_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|error| error.error)?;
    sync_directory(parent)?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    OpenOptions::new().read(true).open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    // Windows' rename replacement is provided by tempfile's platform backend;
    // directory fsync is not exposed by the standard library on other targets.
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], DdcError> {
        let end = self.offset.checked_add(length).ok_or(DdcError::Truncated)?;
        if end > self.bytes.len() {
            return Err(DdcError::Truncated);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn array_32(&mut self) -> Result<[u8; 32], DdcError> {
        self.take(32)?.try_into().map_err(|_| DdcError::Truncated)
    }

    fn u32(&mut self) -> Result<u32, DdcError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, DdcError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn u128(&mut self) -> Result<u128, DdcError> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-ddc-{}-{}",
            std::process::id(),
            format!(
                "{}-{}",
                TEST_ID.fetch_add(1, Ordering::Relaxed),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )
        ))
    }

    fn descriptor(key: ArtifactKey, content: &[u8]) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(key, 7, 2, 16, vec![ArtifactKey([3; 32])], content)
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn persists_and_restores_across_multiple_opens() {
        let dir = temp_dir();
        let key = ArtifactKey([1; 32]);
        let value = descriptor(key, b"payload");
        {
            let mut ddc = DerivedDataCache::open(&dir).unwrap();
            ddc.put(key, value.clone()).unwrap();
            assert_eq!(ddc.get(key), Some(value.clone()));
        }
        let reopened = DerivedDataCache::open(&dir).unwrap();
        assert_eq!(reopened.get(key), Some(value));
        cleanup(&dir);
    }

    #[test]
    fn invalidate_is_persistent_and_does_not_touch_cas_directory() {
        let dir = temp_dir();
        let key = ArtifactKey([2; 32]);
        fs::create_dir_all(dir.join("cas")).unwrap();
        fs::write(dir.join("cas").join("blob"), b"still here").unwrap();
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key, b"payload")).unwrap();
        assert!(ddc.invalidate(key).unwrap());
        assert!(!ddc.invalidate(key).unwrap());
        assert!(DerivedDataCache::open(&dir).unwrap().get(key).is_none());
        assert_eq!(
            fs::read(dir.join("cas").join("blob")).unwrap(),
            b"still here"
        );
        cleanup(&dir);
    }

    #[test]
    fn rejects_descriptor_key_conflict_and_value_conflict() {
        let dir = temp_dir();
        let key = ArtifactKey([4; 32]);
        let other = ArtifactKey([5; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        assert!(matches!(
            ddc.put(key, descriptor(other, b"payload")),
            Err(DdcError::KeyConflict { .. })
        ));
        ddc.put(key, descriptor(key, b"one")).unwrap();
        assert!(matches!(
            ddc.put(key, descriptor(key, b"two")),
            Err(DdcError::KeyConflict { .. })
        ));
        cleanup(&dir);
    }

    #[test]
    fn detects_truncation_and_hash_tampering() {
        let dir = temp_dir();
        let key = ArtifactKey([6; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key, b"payload")).unwrap();
        let path = ddc.index_path();
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(matches!(
            DerivedDataCache::open(&dir),
            Err(DdcError::HashMismatch { .. }) | Err(DdcError::Truncated)
        ));
        fs::write(&path, {
            let mut tampered = bytes;
            tampered[20] ^= 1;
            tampered
        })
        .unwrap();
        assert!(matches!(
            DerivedDataCache::open(&dir),
            Err(DdcError::HashMismatch { .. })
        ));
        cleanup(&dir);
    }

    #[test]
    fn detects_unknown_version_before_decoding_records() {
        let dir = temp_dir();
        let key = ArtifactKey([8; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key, b"payload")).unwrap();
        let path = ddc.index_path();
        let mut bytes = fs::read(&path).unwrap();
        let version_offset = MAGIC.len();
        bytes[version_offset..version_offset + 4].copy_from_slice(&99u32.to_le_bytes());
        let checksum = ContentHash::digest(&bytes[..bytes.len() - CHECKSUM_LEN]);
        let end = bytes.len();
        bytes[end - CHECKSUM_LEN..].copy_from_slice(&checksum.0);
        fs::write(path, bytes).unwrap();
        assert!(matches!(
            DerivedDataCache::open(&dir),
            Err(DdcError::UnknownVersion(99))
        ));
        cleanup(&dir);
    }

    #[test]
    fn canonical_encoding_is_order_independent() {
        let key_a = ArtifactKey([10; 32]);
        let key_b = ArtifactKey([11; 32]);
        let a = descriptor(key_a, b"a");
        let b = descriptor(key_b, b"b");
        let first = BTreeMap::from([(key_a, a.clone()), (key_b, b.clone())]);
        let second = BTreeMap::from([(key_b, b), (key_a, a)]);
        assert_eq!(
            encode_index(&first).unwrap(),
            encode_index(&second).unwrap()
        );
    }

    #[test]
    fn concurrent_threads_merge_distinct_writes_without_lost_updates() {
        let dir = temp_dir();
        let workers = 8;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(workers));
        let handles = (0..workers)
            .map(|index| {
                let dir = dir.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let key = ArtifactKey([index as u8 + 20; 32]);
                    let mut ddc = DerivedDataCache::open(&dir).unwrap();
                    ddc.put_descriptor(descriptor(key, &[index as u8])).unwrap();
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }

        let ddc = DerivedDataCache::open(&dir).unwrap();
        assert_eq!(ddc.len(), workers);
        for index in 0..workers {
            assert!(ddc.get(ArtifactKey([index as u8 + 20; 32])).is_some());
        }
        cleanup(&dir);
    }

    #[test]
    fn readers_only_observe_complete_published_indexes() {
        let dir = temp_dir();
        let writer_dir = dir.clone();
        let reader_dir = dir.clone();
        let writer = std::thread::spawn(move || {
            for index in 0..64u8 {
                let key = ArtifactKey([index.wrapping_add(40); 32]);
                let mut ddc = DerivedDataCache::open(&writer_dir).unwrap();
                ddc.put_descriptor(descriptor(key, &[index])).unwrap();
            }
        });
        let reader = std::thread::spawn(move || {
            for _ in 0..128 {
                let ddc = DerivedDataCache::open(&reader_dir).unwrap();
                assert!(ddc.len() <= 64);
            }
        });
        writer.join().unwrap();
        reader.join().unwrap();
        assert_eq!(DerivedDataCache::open(&dir).unwrap().len(), 64);
        cleanup(&dir);
    }

    #[test]
    fn interrupted_publish_artifact_is_ignored_without_destroying_index() {
        let dir = temp_dir();
        let key = ArtifactKey([12; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key, b"committed")).unwrap();

        let orphan = dir.join(format!("{TEMP_PREFIX}crashed-writer"));
        fs::write(&orphan, b"partial index that was never published").unwrap();
        let reopened = DerivedDataCache::open(&dir).unwrap();
        assert_eq!(reopened.get(key), Some(descriptor(key, b"committed")));
        assert!(
            orphan.exists(),
            "active/unknown writer files are not reaped"
        );
        cleanup(&dir);
    }
}
