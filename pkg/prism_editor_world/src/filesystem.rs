//! Bounded, versioned filesystem storage for opaque editor cell payloads.

use std::{
    fmt,
    fs::{self, File, OpenOptions, TryLockError},
    io,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard,
    },
};

use prism_scene::FileCellStore;

use crate::{
    CellFileIdentity, CellIoProvider, CellLoadFuture, CellLoadRequest, CellSaveFuture,
    CellSaveRequest, CellSnapshotFuture, TransitionKind,
};

mod codec;
mod worker;

use codec::{read_cell, read_envelope, write_envelope, EnvelopeIdentity, HEADER_BYTES};
use worker::{start, Cancellation};

mod actor;
pub use actor::{
    ActorIoFuture, ActorIoLimits, ActorRevisionBase, ActorSaveReceipt, ActorSaveRequest,
    ActorSnapshot, FileActorIoProvider,
};

/// Filesystem operation and payload bounds, also applied outside a driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileCellIoLimits {
    max_payload_bytes: usize,
    max_operations: usize,
}

impl FileCellIoLimits {
    /// Creates nonzero bounds. The complete envelope must fit the scene store's
    /// [`prism_scene::partition::MAX_CELL_BYTES`] limit (64 bytes reserved for metadata).
    pub fn new(max_payload_bytes: usize, max_operations: usize) -> Result<Self, FileCellIoError> {
        if max_payload_bytes == 0
            || max_payload_bytes > prism_scene::partition::MAX_CELL_BYTES - HEADER_BYTES
            || max_operations == 0
        {
            return Err(FileCellIoError::InvalidConfiguration);
        }
        Ok(Self {
            max_payload_bytes,
            max_operations,
        })
    }

    /// Maximum application bytes in one file, excluding its envelope.
    #[must_use]
    pub const fn max_payload_bytes(self) -> usize {
        self.max_payload_bytes
    }

    /// Maximum outstanding futures/workers shared by provider clones.
    #[must_use]
    pub const fn max_operations(self) -> usize {
        self.max_operations
    }
}

/// Failures preserve the old complete cell unless explicitly marked uncertain.
#[derive(Debug)]
pub enum FileCellIoError {
    /// Zero capacities or an envelope exceeding the scene storage bound.
    InvalidConfiguration,
    /// Inconsistent identity, transition, read limit, or revision base.
    InvalidRequest,
    /// Symlink, non-regular file, or a path outside the configured layout.
    UnsafePath(PathBuf),
    /// Bound checked before allocating or publishing a payload.
    TooLarge {
        /// Configured maximum payload size.
        limit: usize,
        /// Observed or declared payload size.
        actual: u64,
    },
    /// Invalid envelope, identity, checksum, or exact file length.
    Corrupt(&'static str),
    /// Another process/provider holds this cell's advisory lock; retry later.
    Busy,
    /// The provider's outstanding-operation capacity is full.
    Capacity,
    /// A save would overwrite a newer or divergent durable snapshot.
    RevisionConflict {
        /// Incoming authoring revision.
        requested: u64,
        /// Current durable revision, or no existing cell.
        current: Option<u64>,
    },
    /// Work was abandoned before its publication fence.
    Cancelled,
    /// The cooperative deadline expired before publication. Kernel I/O cannot
    /// be interrupted; a commit already inside the fence completes normally.
    TimedOut,
    /// I/O failed before rename; the previous cell remains authoritative.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Original filesystem error, including NotFound for missing cells.
        source: io::Error,
    },
    /// Rename succeeded but directory fsync failed. The new complete cell may
    /// be durable; retain dirty data and retry the identical snapshot.
    CommitUncertain(io::Error),
}

impl fmt::Display for FileCellIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => f.write_str("invalid filesystem cell I/O limits"),
            Self::InvalidRequest => f.write_str("invalid filesystem cell request"),
            Self::UnsafePath(path) => write!(f, "unsafe cell storage path: {}", path.display()),
            Self::TooLarge { limit, actual } => write!(f, "cell payload {actual} exceeds {limit}"),
            Self::Corrupt(reason) => write!(f, "invalid editor cell: {reason}"),
            Self::Busy => f.write_str("cell storage lock is busy"),
            Self::Capacity => f.write_str("filesystem cell operation capacity exceeded"),
            Self::RevisionConflict { requested, current } => write!(
                f,
                "cell revision {requested} conflicts with durable revision {current:?}"
            ),
            Self::Cancelled => f.write_str("cell I/O cancelled"),
            Self::TimedOut => f.write_str("storage operation deadline expired"),
            Self::Io { operation, source } => write!(f, "{operation}: {source}"),
            Self::CommitUncertain(source) => {
                write!(f, "cell replaced but directory sync failed: {source}")
            }
        }
    }
}

impl std::error::Error for FileCellIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::CommitUncertain(source) => Some(source),
            _ => None,
        }
    }
}

fn io_error(operation: &'static str, source: io::Error) -> FileCellIoError {
    FileCellIoError::Io { operation, source }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Shared {
    root: PathBuf,
    limits: FileCellIoLimits,
    operations: AtomicUsize,
    #[cfg(test)]
    hook: Mutex<Option<TestHook>>,
}

/// Production per-cell filesystem provider; no executor or Bevy dependency.
///
/// `new` synchronously opens an existing application-owned world directory.
/// Submissions/polls perform no filesystem I/O: bounded workers perform reads,
/// staging, file locking, fsync and rename. Clones share admission bounds;
/// independently opened providers coordinate through per-cell advisory locks.
/// Paths are generated from numeric coordinates, never request path strings.
/// Existing symlinks and special files beneath the root are rejected.
///
/// The root and its ancestors must not be renamed/replaced by untrusted actors
/// while in use. Portable std path checks are not an OS capability sandbox.
/// See the package README for durability, cancellation and format boundaries.
///
/// ```no_run
/// use prism_editor_world::{FileCellIoLimits, FileCellIoProvider, IoBudget, WorldIoDriver};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // The application creates/chooses a separate directory for each world.
/// let provider = FileCellIoProvider::new(
///     "/project/worlds/main",
///     FileCellIoLimits::new(8 * 1024 * 1024, 4)?,
/// )?;
/// let budget = IoBudget::new(4, 32 * 1024 * 1024).expect("nonzero constant budget");
/// let driver = WorldIoDriver::new(provider, budget);
/// assert_eq!(driver.in_flight(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct FileCellIoProvider {
    shared: Arc<Shared>,
}

impl FileCellIoProvider {
    /// Selects an existing world directory, resolving its root once. `..` is
    /// rejected even in the root argument. Saves create only its `cells` child.
    pub fn new(root: impl AsRef<Path>, limits: FileCellIoLimits) -> Result<Self, FileCellIoError> {
        if root
            .as_ref()
            .components()
            .any(|part| part == Component::ParentDir)
        {
            return Err(FileCellIoError::UnsafePath(root.as_ref().to_owned()));
        }
        let root = fs::canonicalize(root).map_err(|e| io_error("resolve world directory", e))?;
        if !root.is_dir() {
            return Err(FileCellIoError::UnsafePath(root));
        }
        Ok(Self {
            shared: Arc::new(Shared {
                root,
                limits,
                operations: AtomicUsize::new(0),
                #[cfg(test)]
                hook: Mutex::new(None),
            }),
        })
    }

    /// Canonical world directory, not its `cells` subdirectory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    /// Capacity policy shared by clones.
    #[must_use]
    pub fn limits(&self) -> FileCellIoLimits {
        self.shared.limits
    }

    /// Outstanding futures or workers (including cancelled workers cleaning up).
    #[must_use]
    pub fn outstanding_operations(&self) -> usize {
        self.shared.operations.load(Ordering::Acquire)
    }

    fn validate(&self, file: &CellFileIdentity) -> Result<(), FileCellIoError> {
        let coord = file.coord();
        let expected = format!("cells/{}_{}_{}.prismcell", coord.x, coord.y, coord.z);
        if file.stable_id().is_nil() || file.relative_path() != expected {
            return Err(FileCellIoError::InvalidRequest);
        }
        Ok(())
    }
}

impl CellIoProvider for FileCellIoProvider {
    type Error = FileCellIoError;

    fn load_cell(
        &self,
        request: CellLoadRequest,
    ) -> Result<CellLoadFuture<Self::Error>, Self::Error> {
        let future = self.load_snapshot(request)?;
        Ok(Box::pin(async move {
            future.await.map(|snapshot| snapshot.bytes)
        }))
    }

    fn load_snapshot(
        &self,
        request: CellLoadRequest,
    ) -> Result<CellSnapshotFuture<Self::Error>, Self::Error> {
        self.validate(request.file())?;
        if request.cell() != request.file().coord()
            || request.token().kind() != TransitionKind::Load
            || request.max_bytes() == 0
        {
            return Err(FileCellIoError::InvalidRequest);
        }
        let shared = self.shared.clone();
        start(shared.clone(), move |cancel| {
            let directory = shared.directory(false)?;
            let _guard = directory.acquire(request.file(), false)?;
            let path = directory.cell_path(request.file());
            let mut file = open_regular(&path)?;
            let read = read_cell(
                &mut file,
                request.file(),
                request.max_bytes().min(shared.limits.max_payload_bytes),
                None,
                cancel,
            )?;
            Ok(crate::CellSnapshot {
                bytes: read.bytes,
                revision: Some(read.revision),
            })
        })
    }

    fn save_cell(
        &self,
        request: CellSaveRequest,
        payload: Vec<u8>,
    ) -> Result<CellSaveFuture<Self::Error>, Self::Error> {
        self.validate(request.file())?;
        if request.cell() != request.file().coord()
            || request
                .base_revision()
                .is_some_and(|base| base > request.revision())
        {
            return Err(FileCellIoError::InvalidRequest);
        }
        if payload.len() > self.shared.limits.max_payload_bytes {
            return Err(FileCellIoError::TooLarge {
                limit: self.shared.limits.max_payload_bytes,
                actual: payload.len() as u64,
            });
        }
        let shared = self.shared.clone();
        start(shared.clone(), move |cancel| {
            shared.save(&request, &payload, cancel)
        })
    }
}

impl Shared {
    fn directory(&self, create: bool) -> Result<Directory, FileCellIoError> {
        let root_meta =
            fs::symlink_metadata(&self.root).map_err(|e| io_error("inspect world directory", e))?;
        if !root_meta.is_dir() {
            return Err(FileCellIoError::UnsafePath(self.root.clone()));
        }
        Directory::child(&self.root, "cells", create)
    }

    fn save(
        &self,
        request: &CellSaveRequest,
        payload: &[u8],
        cancel: &Cancellation,
    ) -> Result<(), FileCellIoError> {
        cancel.check()?;
        let directory = self.directory(true)?;
        let _guard = directory.acquire(request.file(), true)?;
        let path = directory.cell_path(request.file());
        self.replace(
            &directory,
            &path,
            path.with_extension("prismcell.pending"),
            SaveEnvelope {
                identity: &EnvelopeIdentity::cell(request.file()),
                revision: request.revision(),
                base: request.base_revision().map(Some),
                payload,
            },
            cancel,
        )
    }

    fn replace(
        &self,
        directory: &Directory,
        path: &Path,
        stage: PathBuf,
        incoming: SaveEnvelope<'_>,
        cancel: &Cancellation,
    ) -> Result<(), FileCellIoError> {
        cancel.check()?;
        let payload = incoming.payload;
        let previous = match open_regular(path) {
            Ok(mut file) => Some(read_envelope(
                &mut file,
                incoming.identity,
                self.limits.max_payload_bytes,
                Some(payload),
                cancel,
            )?),
            Err(FileCellIoError::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound => {
                None
            }
            Err(e) => return Err(e),
        };
        let current = previous.as_ref().map(|read| read.revision);
        let identical = previous
            .as_ref()
            .is_some_and(|read| read.revision == incoming.revision && read.equal);
        if identical {
            let _fence = cancel.commit_guard()?;
            directory
                .handle
                .sync_all()
                .map_err(FileCellIoError::CommitUncertain)?;
            return Ok(());
        }
        if current.is_some_and(|revision| incoming.revision <= revision)
            || incoming.base.is_some_and(|base| current != base)
        {
            return Err(FileCellIoError::RevisionConflict {
                requested: incoming.revision,
                current,
            });
        }
        // Only this exact stage may be reclaimed while holding the file's OS lock.
        match fs::symlink_metadata(&stage) {
            Ok(meta) if meta.is_file() => {
                fs::remove_file(&stage).map_err(|e| io_error("remove orphan stage", e))?;
            }
            Ok(_) => return Err(FileCellIoError::UnsafePath(stage)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_error("inspect storage stage", e)),
        }
        let mut temporary = Temporary {
            path: stage,
            owned: false,
        };
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary.path)
            .map_err(|e| io_error("create storage stage", e))?;
        temporary.owned = true;
        write_envelope(
            &mut file,
            incoming.identity,
            incoming.revision,
            payload,
            cancel,
        )?;
        file.sync_all()
            .map_err(|e| io_error("sync storage stage", e))?;
        drop(file);
        #[cfg(test)]
        self.checkpoint(TestPhase::Staged)
            .map_err(|e| io_error("staged checkpoint", e))?;
        let _fence = cancel.commit_guard()?;
        fs::rename(&temporary.path, path).map_err(|e| io_error("replace storage file", e))?;
        temporary.owned = false;
        #[cfg(test)]
        self.checkpoint(TestPhase::Renamed)
            .map_err(FileCellIoError::CommitUncertain)?;
        directory
            .handle
            .sync_all()
            .map_err(FileCellIoError::CommitUncertain)
    }

    #[cfg(test)]
    fn checkpoint(&self, phase: TestPhase) -> io::Result<()> {
        let hook = lock(&self.hook).clone();
        if let Some(hook) = hook {
            hook(phase)?;
        }
        Ok(())
    }
}

struct SaveEnvelope<'a> {
    identity: &'a EnvelopeIdentity,
    revision: u64,
    // None preserves the cell API's legacy monotonic-only mode. Actors always
    // supply Some(None) for create-only or Some(Some(base)) for compare-and-save.
    base: Option<Option<u64>>,
    payload: &'a [u8],
}

struct Directory {
    path: PathBuf,
    handle: File,
}

impl Directory {
    fn child(parent: &Path, name: &str, create: bool) -> Result<Self, FileCellIoError> {
        let path = parent.join(name);
        if create {
            match fs::create_dir(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(io_error("create cells directory", e)),
            }
        }
        if !fs::symlink_metadata(&path)
            .map_err(|e| io_error("inspect cells directory", e))?
            .is_dir()
        {
            return Err(FileCellIoError::UnsafePath(path));
        }
        if create {
            // Every writer persists the cells entry, including when another
            // process created it but crashed before syncing the world root.
            File::open(parent)
                .and_then(|file| file.sync_all())
                .map_err(|e| io_error("sync world directory", e))?;
        }
        let handle = File::open(&path).map_err(|e| io_error("open cells directory", e))?;
        // Fail before staging on platforms/filesystems without directory sync.
        handle
            .sync_all()
            .map_err(|e| io_error("sync cells directory", e))?;
        Ok(Directory { path, handle })
    }
    fn cell_path(&self, file: &CellFileIdentity) -> PathBuf {
        FileCellStore::new(&self.path).cell_path(file.coord())
    }

    fn acquire(
        &self,
        identity: &CellFileIdentity,
        exclusive: bool,
    ) -> Result<File, FileCellIoError> {
        let path = self.cell_path(identity).with_extension("prismcell.lock");
        self.acquire_path(&path, exclusive)
    }

    fn acquire_path(&self, path: &Path, exclusive: bool) -> Result<File, FileCellIoError> {
        // Never truncate or unlink a lock: every cooperating process must lock
        // the same inode, including readers and after a process crash.
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                check_regular(path)?;
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .map_err(|e| io_error("open cell lock", e))?
            }
            Err(e) => return Err(io_error("create cell lock", e)),
        };
        let result = if exclusive {
            file.try_lock()
        } else {
            file.try_lock_shared()
        };
        match result {
            Ok(()) => Ok(file),
            Err(TryLockError::WouldBlock) => Err(FileCellIoError::Busy),
            Err(TryLockError::Error(e)) => Err(io_error("lock cell", e)),
        }
    }
}

fn check_regular(path: &Path) -> Result<(), FileCellIoError> {
    if !fs::symlink_metadata(path)
        .map_err(|e| io_error("inspect cell file", e))?
        .is_file()
    {
        return Err(FileCellIoError::UnsafePath(path.to_owned()));
    }
    Ok(())
}

fn open_regular(path: &Path) -> Result<File, FileCellIoError> {
    check_regular(path)?;
    let file = File::open(path).map_err(|e| io_error("open cell file", e))?;
    if !file
        .metadata()
        .map_err(|e| io_error("inspect opened cell", e))?
        .is_file()
    {
        return Err(FileCellIoError::UnsafePath(path.to_owned()));
    }
    Ok(file)
}

struct Temporary {
    path: PathBuf,
    owned: bool,
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if self.owned {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum TestPhase {
    Staged,
    Renamed,
}

#[cfg(test)]
type TestHook = Arc<dyn Fn(TestPhase) -> io::Result<()> + Send + Sync>;

#[cfg(test)]
mod tests;
