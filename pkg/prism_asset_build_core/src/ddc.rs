//! A local derived-data cache for artifact descriptors.
//!
//! The DDC stores only the deterministic `ArtifactKey -> ArtifactDescriptor`
//! index.  Artifact bytes remain the responsibility of [`crate::cas`]; this
//! module never creates, reads, or deletes a CAS blob.

use crate::types::{ArtifactDescriptor, ArtifactKey, ArtifactStorageClass, ContentHash};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const MAGIC: &[u8] = b"PRISM-DDC\0";
const VERSION: u32 = 2;
const CHECKSUM_LEN: usize = 32;
const MAX_ENTRIES: u64 = 1_000_000;
const MAX_DEPENDENCIES: u64 = 1_000_000;
const TEMP_PREFIX: &str = ".index.ddc.tmp-";
const MIN_ENCODED_INDEX_BYTES: usize = MAGIC.len() + 4 + 8 + CHECKSUM_LEN;
static PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Failure while opening or mutating a local DDC.
#[derive(Debug)]
pub enum DdcError {
    Io(io::Error),
    InvalidFormat(&'static str),
    InvalidDescriptor(&'static str),
    /// The platform has no backend that can provide the cross-process
    /// exclusion required by the shared on-disk index protocol.
    CoordinationUnavailable(&'static str),
    Truncated,
    UnknownVersion(u32),
    HashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    InvalidCapacity(&'static str),
    CapacityInsufficient {
        max_encoded_bytes: usize,
        minimum_encoded_bytes: usize,
    },
    KeyConflict {
        key: ArtifactKey,
        reason: &'static str,
    },
    /// A driver/backend-local artifact attempted to enter the shared DDC.
    DriverLocalArtifact {
        key: ArtifactKey,
    },
}

impl fmt::Display for DdcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "DDC I/O error: {error}"),
            Self::InvalidFormat(reason) => write!(f, "invalid DDC format: {reason}"),
            Self::InvalidDescriptor(reason) => write!(f, "invalid artifact descriptor: {reason}"),
            Self::CoordinationUnavailable(reason) => {
                write!(f, "DDC cross-process coordination unavailable: {reason}")
            }
            Self::Truncated => f.write_str("DDC index is truncated"),
            Self::UnknownVersion(version) => write!(f, "unknown DDC index version: {version}"),
            Self::HashMismatch { expected, actual } => write!(
                f,
                "DDC index hash mismatch: expected {}, got {}",
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::InvalidCapacity(reason) => write!(f, "invalid DDC capacity: {reason}"),
            Self::CapacityInsufficient {
                max_encoded_bytes,
                minimum_encoded_bytes,
            } => write!(
                f,
                "DDC encoded-byte capacity {max_encoded_bytes} is below the empty index minimum {minimum_encoded_bytes}"
            ),
            Self::KeyConflict { key, reason } => {
                write!(f, "DDC key {} conflicts: {reason}", key.to_hex())
            }
            Self::DriverLocalArtifact { key } => write!(
                f,
                "DDC key {} is driver-local and cannot enter shared derived data",
                key.to_hex()
            ),
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
/// uses a stable advisory file lock to coordinate read-modify-write updates
/// across Unix processes. Other platforms fail closed with
/// [`DdcError::CoordinationUnavailable`] unless a platform-equivalent backend
/// is added; they never silently retain a single-owner-process contract.
/// `get`, `len`, and
/// `is_empty` report this handle's last snapshot
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

/// Explicit upper bounds used by [`DerivedDataCache::evict_to_capacity`].
///
/// At least one bound must be present. `max_entries = Some(0)` is valid and
/// means that the explicit eviction operation may empty the index. The byte
/// bound includes the complete encoded index (header and checksum), not CAS
/// payload bytes. No bound is applied implicitly by [`DerivedDataCache::put`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdcCapacity {
    pub max_entries: Option<usize>,
    pub max_encoded_bytes: Option<usize>,
}

impl DdcCapacity {
    pub fn new(
        max_entries: Option<usize>,
        max_encoded_bytes: Option<usize>,
    ) -> Result<Self, DdcError> {
        let capacity = Self {
            max_entries,
            max_encoded_bytes,
        };
        capacity.validate()?;
        Ok(capacity)
    }

    fn validate(self) -> Result<(), DdcError> {
        if self.max_entries.is_none() && self.max_encoded_bytes.is_none() {
            return Err(DdcError::InvalidCapacity(
                "at least one upper bound is required",
            ));
        }
        if self
            .max_entries
            .is_some_and(|value| value as u64 > MAX_ENTRIES)
        {
            return Err(DdcError::InvalidCapacity(
                "max_entries exceeds the DDC format limit",
            ));
        }
        if self
            .max_encoded_bytes
            .is_some_and(|value| value < MIN_ENCODED_INDEX_BYTES)
        {
            return Err(DdcError::CapacityInsufficient {
                max_encoded_bytes: self.max_encoded_bytes.unwrap(),
                minimum_encoded_bytes: MIN_ENCODED_INDEX_BYTES,
            });
        }
        Ok(())
    }
}

/// Result of one explicit, atomically published capacity-eviction operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DdcEviction {
    /// Keys removed in the deterministic order in which they were selected.
    /// The current policy is ascending [`ArtifactKey`] order, not LRU.
    pub evicted_keys: Vec<ArtifactKey>,
    /// Byte length of the complete encoded index after the operation.
    pub encoded_bytes: usize,
}

impl DerivedDataCache {
    /// Opens (or creates) the DDC directory and restores its index.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DdcError> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let _file_lock = DdcFileLock::acquire(&root)?;
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
    /// This operation never applies a capacity policy and never implicitly
    /// evicts another descriptor. Call [`Self::evict_to_capacity`] explicitly
    /// when deterministic capacity reclamation is wanted.
    pub fn put(
        &mut self,
        key: ArtifactKey,
        descriptor: ArtifactDescriptor,
    ) -> Result<(), DdcError> {
        validate_entry(key, &descriptor)?;
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let _file_lock = DdcFileLock::acquire(&self.root)?;
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

    /// Explicitly evicts entries until both requested capacity bounds hold.
    ///
    /// The current deterministic policy removes the lexicographically lowest
    /// descriptor key first. This is intentionally a key-order policy rather
    /// than an implicit recency policy: callers that need recency must encode
    /// it in a higher-level selection policy and invoke this operation with a
    /// separately chosen set of entries in a future API.
    ///
    /// The operation reloads the latest index while holding the same process
    /// and cross-process locks as `put`, computes the entire result before
    /// publishing it, and uses the existing synced temporary-file replacement.
    /// Therefore a failed capacity check leaves the previously committed index
    /// untouched, and a successful operation is recoverable by `open` after a
    /// process restart. `put` does not call this method.
    pub fn evict_to_capacity(&mut self, capacity: DdcCapacity) -> Result<DdcEviction, DdcError> {
        capacity.validate()?;

        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let _file_lock = DdcFileLock::acquire(&self.root)?;
        let mut current = read_index(&self.index_path())?;
        let mut encoded = encode_index(&current)?;
        let mut evicted_keys = Vec::new();

        while capacity
            .max_entries
            .is_some_and(|max_entries| current.len() > max_entries)
            || capacity
                .max_encoded_bytes
                .is_some_and(|max_bytes| encoded.len() > max_bytes)
        {
            let key = current
                .keys()
                .next()
                .copied()
                .ok_or(DdcError::InvalidCapacity("capacity cannot be satisfied"))?;
            current.remove(&key);
            evicted_keys.push(key);
            encoded = encode_index(&current)?;
        }

        if !evicted_keys.is_empty() {
            atomic_write(&self.index_path(), &encoded)?;
        }
        self.entries = current;
        Ok(DdcEviction {
            evicted_keys,
            encoded_bytes: encoded.len(),
        })
    }

    /// Removes `key` and durably publishes the resulting index.
    ///
    /// Returns whether an entry was present. A missing key is a successful
    /// no-op, which makes invalidation safe to repeat.
    pub fn invalidate(&mut self, key: ArtifactKey) -> Result<bool, DdcError> {
        let _lock = process_lock().lock().unwrap_or_else(|p| p.into_inner());
        let _file_lock = DdcFileLock::acquire(&self.root)?;
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

/// Cross-process lease for DDC index mutations. Each process opens the same
/// stable lock file, while the process-local mutex above also orders handles
/// in this process (flock semantics are tied to open file descriptions).
#[cfg(unix)]
struct DdcFileLock(File);

#[cfg(unix)]
impl DdcFileLock {
    fn acquire(root: &Path) -> Result<Self, DdcError> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(root.join(".index.ddc.lock"))?;
        rustix::fs::flock(&file, rustix::fs::FlockOperation::LockExclusive)
            .map_err(|error| DdcError::Io(io::Error::from(error)))?;
        Ok(Self(file))
    }
}

#[cfg(unix)]
impl Drop for DdcFileLock {
    fn drop(&mut self) {
        let _ = rustix::fs::flock(&self.0, rustix::fs::FlockOperation::Unlock);
    }
}

#[cfg(not(unix))]
struct DdcFileLock;

#[cfg(not(unix))]
impl DdcFileLock {
    fn acquire(_root: &Path) -> Result<Self, DdcError> {
        Err(DdcError::CoordinationUnavailable(
            "cross-process DDC index locking has no supported platform backend",
        ))
    }
}

fn read_index(path: &Path) -> Result<BTreeMap<ArtifactKey, ArtifactDescriptor>, DdcError> {
    match fs::read(path) {
        Ok(bytes) => decode_index(&bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error.into()),
    }
}

fn validate_entry(key: ArtifactKey, descriptor: &ArtifactDescriptor) -> Result<(), DdcError> {
    if descriptor.storage_class != ArtifactStorageClass::Shared {
        return Err(DdcError::DriverLocalArtifact { key });
    }
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
        body.push(match descriptor.storage_class {
            ArtifactStorageClass::Shared => 0,
            ArtifactStorageClass::DriverLocal => 1,
        });
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
            storage_class: match cursor.take(1)?[0] {
                0 => ArtifactStorageClass::Shared,
                1 => ArtifactStorageClass::DriverLocal,
                _ => return Err(DdcError::InvalidFormat("unknown artifact storage class")),
            },
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
    fn put_does_not_implicitly_evict_when_a_capacity_would_be_exceeded() {
        let dir = temp_dir();
        let key_a = ArtifactKey([13; 32]);
        let key_b = ArtifactKey([14; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key_a, b"a")).unwrap();
        ddc.put_descriptor(descriptor(key_b, b"b")).unwrap();

        assert_eq!(ddc.len(), 2);
        assert!(ddc.get(key_a).is_some());
        assert!(ddc.get(key_b).is_some());
        cleanup(&dir);
    }

    #[test]
    fn rejects_invalid_capacity_and_preserves_the_committed_index() {
        let dir = temp_dir();
        let key = ArtifactKey([15; 32]);
        let value = descriptor(key, b"payload");
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(value.clone()).unwrap();
        let before = fs::read(ddc.index_path()).unwrap();

        assert!(matches!(
            DdcCapacity::new(None, None),
            Err(DdcError::InvalidCapacity(_))
        ));
        assert!(matches!(
            DdcCapacity::new(None, Some(0)),
            Err(DdcError::CapacityInsufficient { .. })
        ));
        assert!(matches!(
            DdcCapacity::new(None, Some(MIN_ENCODED_INDEX_BYTES - 1)),
            Err(DdcError::CapacityInsufficient { .. })
        ));

        assert_eq!(fs::read(ddc.index_path()).unwrap(), before);
        assert_eq!(ddc.get(key), Some(value));
        cleanup(&dir);
    }

    #[test]
    fn evicts_in_stable_descriptor_key_order_and_reports_encoded_size() {
        let dir = temp_dir();
        let keys = [
            ArtifactKey([24; 32]),
            ArtifactKey([22; 32]),
            ArtifactKey([25; 32]),
            ArtifactKey([23; 32]),
        ];
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        for key in keys.iter().rev() {
            ddc.put_descriptor(descriptor(*key, &[key.0[0]])).unwrap();
        }

        let result = ddc
            .evict_to_capacity(DdcCapacity::new(Some(2), None).unwrap())
            .unwrap();
        assert_eq!(
            result.evicted_keys,
            vec![ArtifactKey([22; 32]), ArtifactKey([23; 32])]
        );
        assert_eq!(ddc.len(), 2);
        assert!(ddc.get(ArtifactKey([22; 32])).is_none());
        assert!(ddc.get(ArtifactKey([23; 32])).is_none());
        assert!(ddc.get(ArtifactKey([24; 32])).is_some());
        assert!(ddc.get(ArtifactKey([25; 32])).is_some());
        assert_eq!(
            result.encoded_bytes,
            fs::read(ddc.index_path()).unwrap().len()
        );
        assert_eq!(
            result.encoded_bytes,
            encode_index(&ddc.entries).unwrap().len()
        );
        cleanup(&dir);
    }

    #[test]
    fn encoded_byte_capacity_is_inclusive_and_survives_reopen() {
        let dir = temp_dir();
        let keys = [ArtifactKey([30; 32]), ArtifactKey([31; 32])];
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        for key in keys {
            ddc.put_descriptor(descriptor(key, b"same-size")).unwrap();
        }
        let one_entry = encode_index(&BTreeMap::from([(
            keys[1],
            descriptor(keys[1], b"same-size"),
        )]))
        .unwrap()
        .len();

        let result = ddc
            .evict_to_capacity(DdcCapacity::new(None, Some(one_entry)).unwrap())
            .unwrap();
        assert_eq!(result.evicted_keys, vec![keys[0]]);
        assert_eq!(result.encoded_bytes, one_entry);

        let reopened = DerivedDataCache::open(&dir).unwrap();
        assert!(reopened.get(keys[0]).is_none());
        assert!(reopened.get(keys[1]).is_some());
        assert_eq!(fs::read(reopened.index_path()).unwrap().len(), one_entry);
        cleanup(&dir);
    }

    #[test]
    fn zero_entry_capacity_explicitly_empties_and_persists() {
        let dir = temp_dir();
        let key = ArtifactKey([32; 32]);
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor(key, b"payload")).unwrap();

        let result = ddc
            .evict_to_capacity(DdcCapacity::new(Some(0), None).unwrap())
            .unwrap();
        assert_eq!(result.evicted_keys, vec![key]);
        assert_eq!(result.encoded_bytes, MIN_ENCODED_INDEX_BYTES);
        assert!(ddc.is_empty());
        assert!(DerivedDataCache::open(&dir).unwrap().is_empty());
        cleanup(&dir);
    }

    #[test]
    fn concurrent_explicit_evictions_reload_and_preserve_a_valid_index() {
        let dir = temp_dir();
        let workers = 8;
        let seed_key = ArtifactKey([33; 32]);
        let mut seed = DerivedDataCache::open(&dir).unwrap();
        seed.put_descriptor(descriptor(seed_key, b"seed")).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(workers));
        let handles = (0..workers)
            .map(|_| {
                let dir = dir.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let mut ddc = DerivedDataCache::open(&dir).unwrap();
                    ddc.evict_to_capacity(DdcCapacity::new(Some(2), None).unwrap())
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }

        let reopened = DerivedDataCache::open(&dir).unwrap();
        assert!(reopened.len() <= 2);
        assert_eq!(
            fs::read(reopened.index_path()).unwrap(),
            encode_index(&reopened.entries).unwrap()
        );
        cleanup(&dir);
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

    #[cfg(not(unix))]
    #[test]
    fn non_unix_ddc_coordination_fails_closed() {
        let dir = temp_dir();
        assert!(matches!(
            DerivedDataCache::open(&dir),
            Err(DdcError::CoordinationUnavailable(_))
        ));
        cleanup(&dir);
    }
}
