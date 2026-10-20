use crate::{
    ContentBrowserController, MAX_ACTION_TEXT_BYTES, MAX_COLLECTION_MEMBERS,
    MAX_CONTENT_COLLECTIONS,
};
use prism_editor_content::{AssetId, ContentCatalog};
use std::{
    collections::BTreeSet,
    fmt,
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// Hard encoded-file bound, checked before allocation and during reads.
pub const MAX_COLLECTION_FILE_BYTES: usize = 5 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"PRISMCOL";
const VERSION: u32 = 1;
static TEMP_ID: AtomicU64 = AtomicU64::new(1);
type Entries = Vec<(u64, String, BTreeSet<AssetId>)>;

/// Structured failures; a corrupt load never modifies the controller or file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionStorageError {
    /// Root must be an existing absolute directory without parent traversal.
    InvalidProjectRoot,
    /// A symlink or non-regular metadata target was refused.
    UnsafePath,
    /// Another operation holds the project's collection-file lock.
    Busy,
    /// File or decoded field exceeded a fixed bound.
    Limit(&'static str),
    /// Unsupported file schema version.
    UnsupportedVersion(u32),
    /// Offset into the encoded file and a bounded diagnostic.
    Corrupt {
        /// Invalid byte/field, or zero for an in-memory snapshot invariant.
        offset: usize,
        /// Fixed diagnostic describing the rejected field.
        reason: &'static str,
    },
    /// An operating-system error before successful publication.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Portable OS error category.
        kind: io::ErrorKind,
    },
}
impl fmt::Display for CollectionStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProjectRoot => f.write_str("invalid collection project root"),
            Self::UnsafePath => f.write_str("unsafe collection metadata path"),
            Self::Busy => f.write_str("collection metadata is locked"),
            Self::Limit(field) => write!(f, "collection limit: {field}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported collection version {version}")
            }
            Self::Corrupt { offset, reason } => {
                write!(f, "corrupt collections at byte {offset}: {reason}")
            }
            Self::Io { operation, kind } => write!(f, "collection {operation}: {kind}"),
        }
    }
}
impl std::error::Error for CollectionStorageError {}
fn os_error(operation: &'static str, error: io::Error) -> CollectionStorageError {
    CollectionStorageError::Io {
        operation,
        kind: error.kind(),
    }
}
fn corrupt(offset: usize, reason: &'static str) -> CollectionStorageError {
    CollectionStorageError::Corrupt { offset, reason }
}

/// A valid stable member absent from the current catalog publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingCollectionMember {
    /// Persistent collection identity.
    pub collection: u64,
    /// Preserved identity, with no invented catalog record or row.
    pub asset: AssetId,
}

/// Successful load outcome; missing references are bounded and deterministic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionLoadReport {
    /// No file existed: controller was reset to empty project collections.
    pub missing_file: bool,
    /// Number of loaded collections.
    pub collections: usize,
    /// Total member occurrences across collections.
    pub members: usize,
    /// Preserved missing/deleted IDs, ordered by collection ID then asset ID.
    pub missing_members: Vec<MissingCollectionMember>,
}

/// Atomic publication outcome. A durability warning means rename succeeded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionSaveReport {
    /// Encoded bytes installed at the final path.
    pub bytes: usize,
    /// Directory fsync failure after publication, if any.
    pub durability_warning: Option<io::ErrorKind>,
}

/// Project-scoped synchronous persistence, scheduled by the host outside render.
#[derive(Clone, Debug)]
pub struct CollectionStore {
    root: PathBuf,
}
impl CollectionStore {
    /// Uses canonical project/.prism/content-collections.bin. Existing metadata
    /// symlinks are rejected. No directory is created until load/save.
    pub fn for_project(root: impl AsRef<Path>) -> Result<Self, CollectionStorageError> {
        let root = root.as_ref();
        if !root.is_absolute() || root.components().any(|c| c == Component::ParentDir) {
            return Err(CollectionStorageError::InvalidProjectRoot);
        }
        let root = fs::canonicalize(root).map_err(|e| os_error("resolve project", e))?;
        if !root.is_dir() {
            return Err(CollectionStorageError::InvalidProjectRoot);
        }
        Ok(Self { root })
    }

    /// Exact destination for project path wiring and error presentation.
    pub fn path(&self) -> PathBuf {
        self.root.join(".prism/content-collections.bin")
    }

    /// Fully validates before replacing collections. Missing file resets them;
    /// valid missing IDs are retained/reported. Catalog/source files are unchanged.
    pub fn load_into(
        &self,
        controller: &mut ContentBrowserController,
        catalog: &ContentCatalog,
    ) -> Result<CollectionLoadReport, CollectionStorageError> {
        let _lock = self.lock()?;
        let Some(bytes) = self.read()? else {
            controller.restore_collections(Vec::new(), 1);
            return Ok(CollectionLoadReport {
                missing_file: true,
                collections: 0,
                members: 0,
                missing_members: Vec::new(),
            });
        };
        let (entries, next) = decode(&bytes)?;
        let mut report = CollectionLoadReport {
            missing_file: false,
            collections: entries.len(),
            members: 0,
            missing_members: Vec::new(),
        };
        for (collection, _, members) in &entries {
            report.members += members.len();
            for asset in members {
                if catalog.get(*asset).is_none() {
                    report.missing_members.push(MissingCollectionMember {
                        collection: *collection,
                        asset: *asset,
                    });
                }
            }
        }
        controller.restore_collections(entries, next);
        Ok(report)
    }

    /// Deterministic/checksummed save using create_new temp, write_all, file fsync
    /// and same-directory rename. Errors before rename preserve the old file.
    /// An existing corrupt/unsupported file is reported rather than overwritten.
    pub fn save(
        &self,
        controller: &ContentBrowserController,
    ) -> Result<CollectionSaveReport, CollectionStorageError> {
        let mut entries: Entries = controller
            .collections()
            .iter()
            .map(|c| (c.id(), c.name().to_owned(), c.members().clone()))
            .collect();
        entries.sort_by_key(|entry| entry.0);
        let bytes = encode(&entries, controller.next_collection_id())?;
        let _lock = self.lock()?;
        let target = self.path();
        if let Some(previous) = self.read()? {
            decode(&previous)?;
        }
        let directory = target.parent().expect("fixed project child");
        let mut temporary = None;
        for _ in 0..16 {
            let serial = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                ".content-collections.{}.{}.tmp",
                std::process::id(),
                serial
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    temporary = Some((Temporary(path), file));
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(os_error("create temporary", e)),
            }
        }
        let (temporary, mut file) = temporary.ok_or(CollectionStorageError::Busy)?;
        file.write_all(&bytes)
            .map_err(|e| os_error("write temporary", e))?;
        file.sync_all().map_err(|e| os_error("sync temporary", e))?;
        drop(file);
        fs::rename(&temporary.0, &target).map_err(|e| os_error("publish", e))?;
        let durability_warning = File::open(directory)
            .and_then(|file| file.sync_all())
            .and_then(|()| File::open(&self.root)?.sync_all())
            .err()
            .map(|e| e.kind());
        Ok(CollectionSaveReport {
            bytes: bytes.len(),
            durability_warning,
        })
    }

    fn lock(&self) -> Result<File, CollectionStorageError> {
        let directory = self.root.join(".prism");
        match fs::symlink_metadata(&directory) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => return Err(CollectionStorageError::UnsafePath),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                match fs::create_dir(&directory) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(os_error("create metadata directory", e)),
                }
                let metadata = fs::symlink_metadata(&directory)
                    .map_err(|e| os_error("inspect metadata directory", e))?;
                if !metadata.is_dir() || metadata.file_type().is_symlink() {
                    return Err(CollectionStorageError::UnsafePath);
                }
            }
            Err(e) => return Err(os_error("inspect metadata directory", e)),
        }
        let path = directory.join("content-collections.lock");
        regular_or_missing(&path)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| os_error("open lock", e))?;
        match file.try_lock() {
            Ok(()) => Ok(file),
            Err(TryLockError::WouldBlock) => Err(CollectionStorageError::Busy),
            Err(TryLockError::Error(e)) => Err(os_error("lock", e)),
        }
    }

    fn read(&self) -> Result<Option<Vec<u8>>, CollectionStorageError> {
        let path = self.path();
        if !regular_or_missing(&path)? {
            return Ok(None);
        }
        let file = File::open(path).map_err(|e| os_error("open", e))?;
        if file.metadata().map_err(|e| os_error("inspect", e))?.len()
            > MAX_COLLECTION_FILE_BYTES as u64
        {
            return Err(CollectionStorageError::Limit("file bytes"));
        }
        let mut bytes = Vec::new();
        file.take(MAX_COLLECTION_FILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| os_error("read", e))?;
        if bytes.len() > MAX_COLLECTION_FILE_BYTES {
            return Err(CollectionStorageError::Limit("file bytes"));
        }
        Ok(Some(bytes))
    }
}

fn regular_or_missing(path: &Path) -> Result<bool, CollectionStorageError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(CollectionStorageError::UnsafePath),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(os_error("inspect path", e)),
    }
}
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

// FNV-1a detects accidental corruption; it is not authentication.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
fn validate(entries: &Entries, next: u64) -> Result<(), CollectionStorageError> {
    if entries.len() > MAX_CONTENT_COLLECTIONS {
        return Err(CollectionStorageError::Limit("collections"));
    }
    let mut previous = 0;
    for (id, name, members) in entries {
        if *id <= previous || *id == u64::MAX {
            return Err(corrupt(0, "collection IDs must be positive and increasing"));
        }
        if name.len() > MAX_ACTION_TEXT_BYTES {
            return Err(CollectionStorageError::Limit("name bytes"));
        }
        if name.is_empty() || name.trim() != name {
            return Err(corrupt(0, "empty or unnormalized name"));
        }
        if members.len() > MAX_COLLECTION_MEMBERS {
            return Err(CollectionStorageError::Limit("members"));
        }
        previous = *id;
    }
    if next == 0 || next <= previous {
        return Err(corrupt(12, "invalid next collection ID"));
    }
    Ok(())
}
fn encode(entries: &Entries, next: u64) -> Result<Vec<u8>, CollectionStorageError> {
    validate(entries, next)?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend(VERSION.to_le_bytes());
    bytes.extend(next.to_le_bytes());
    bytes.extend((entries.len() as u32).to_le_bytes());
    for (id, name, members) in entries {
        bytes.extend(id.to_le_bytes());
        bytes.extend((name.len() as u32).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((members.len() as u32).to_le_bytes());
        for member in members {
            bytes.extend(member.raw().to_le_bytes());
        }
    }
    bytes.extend(checksum(&bytes).to_le_bytes());
    if bytes.len() > MAX_COLLECTION_FILE_BYTES {
        return Err(CollectionStorageError::Limit("file bytes"));
    }
    Ok(bytes)
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], CollectionStorageError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| corrupt(self.offset, "length overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| corrupt(self.offset, "truncated field"))?;
        self.offset = end;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, CollectionStorageError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed length"),
        ))
    }
    fn u64(&mut self) -> Result<u64, CollectionStorageError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed length"),
        ))
    }
}
fn decode(bytes: &[u8]) -> Result<(Entries, u64), CollectionStorageError> {
    if bytes.len() > MAX_COLLECTION_FILE_BYTES {
        return Err(CollectionStorageError::Limit("file bytes"));
    }
    if bytes.len() < 32 {
        return Err(corrupt(bytes.len(), "truncated header"));
    }
    if &bytes[..8] != MAGIC {
        return Err(corrupt(0, "invalid magic"));
    }
    let mut reader = Reader { bytes, offset: 8 };
    let version = reader.u32()?;
    if version != VERSION {
        return Err(CollectionStorageError::UnsupportedVersion(version));
    }
    let tail = bytes.len() - 8;
    if checksum(&bytes[..tail])
        != u64::from_le_bytes(bytes[tail..].try_into().expect("checksum length"))
    {
        return Err(corrupt(tail, "checksum mismatch"));
    }
    reader.bytes = &bytes[..tail];
    let next = reader.u64()?;
    let count = reader.u32()? as usize;
    if count > MAX_CONTENT_COLLECTIONS {
        return Err(CollectionStorageError::Limit("collections"));
    }
    let mut entries = Vec::with_capacity(count);
    let mut previous_collection = 0;
    for _ in 0..count {
        let id_offset = reader.offset;
        let id = reader.u64()?;
        if id <= previous_collection || id == u64::MAX {
            return Err(corrupt(
                id_offset,
                "collection IDs must be positive and increasing",
            ));
        }
        previous_collection = id;
        let length = reader.u32()? as usize;
        if length > MAX_ACTION_TEXT_BYTES {
            return Err(CollectionStorageError::Limit("name bytes"));
        }
        let start = reader.offset;
        let name = std::str::from_utf8(reader.take(length)?)
            .map_err(|_| corrupt(start, "invalid UTF-8 name"))?
            .to_owned();
        if name.is_empty() || name.trim() != name {
            return Err(corrupt(start, "empty or unnormalized name"));
        }
        let count = reader.u32()? as usize;
        if count > MAX_COLLECTION_MEMBERS {
            return Err(CollectionStorageError::Limit("members"));
        }
        let mut members = BTreeSet::new();
        let mut previous = None;
        for _ in 0..count {
            let offset = reader.offset;
            let member = AssetId::from_raw(u128::from_le_bytes(
                reader.take(16)?.try_into().expect("fixed ID length"),
            ));
            if previous.is_some_and(|last| last >= member) {
                return Err(corrupt(offset, "duplicate or unordered asset ID"));
            }
            members.insert(member);
            previous = Some(member);
        }
        entries.push((id, name, members));
    }
    if reader.offset != tail {
        return Err(corrupt(reader.offset, "trailing payload"));
    }
    validate(&entries, next)?;
    Ok((entries, next))
}
