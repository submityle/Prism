//! Bounded asynchronous source discovery using only the standard library.
//!
//! A scan publishes one complete, sorted metadata snapshot. Paths are exact
//! UTF-8 project-relative slash paths; identity belongs to the host asset system.
//! Cancellation and deadlines are checked between filesystem calls. Dropping a
//! request cancels and joins its worker, so a blocked OS call can delay cleanup.
//! Symlinks are never intentionally followed. Path-based standard-library I/O
//! cannot provide confinement against concurrent hostile symlink replacement.

use std::{
    collections::BTreeSet,
    fmt, fs, io,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    task::Poll,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Resource limits shared by all requests started through one discovery service.
#[derive(Clone, Debug)]
pub struct FilesystemLimits {
    /// Concurrent requests, including finished results awaiting consumption.
    pub max_in_flight: usize,
    /// Maximum files plus directories in one successful snapshot.
    pub max_entries: usize,
    /// UTF-8 bytes in one project-relative path, including slash separators.
    pub max_path_bytes: usize,
    /// Sum of path bytes in the complete output (directories included).
    pub max_total_path_bytes: usize,
    /// Counts every enumerated child, including excluded names and symlinks.
    pub max_scanned_entries: usize,
    /// Root is depth zero; immediate children have depth one.
    pub max_depth: usize,
    /// Maximum length of any included regular file, without reading contents.
    pub max_file_size: u64,
    /// Bounds the native root retained by each request, separately from output.
    pub max_root_path_bytes: usize,
    /// Elapsed worker deadline, including time waiting for the thread to start.
    pub timeout: Duration,
}

impl Default for FilesystemLimits {
    fn default() -> Self {
        Self {
            max_in_flight: 2,
            max_entries: 100_000,
            max_path_bytes: 4096,
            max_total_path_bytes: 16 * 1024 * 1024,
            max_scanned_entries: 200_000,
            max_depth: 64,
            max_file_size: 16 * 1024 * 1024 * 1024,
            max_root_path_bytes: 16 * 1024,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Exact, case-sensitive exclusions, applied before opening child directories.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilesystemExcludes {
    names: BTreeSet<String>,
    subtrees: BTreeSet<String>,
}

impl Default for FilesystemExcludes {
    fn default() -> Self {
        Self {
            names: [
                ".git",
                ".hg",
                ".svn",
                "target",
                "node_modules",
                "build",
                "dist",
                ".DS_Store",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            subtrees: BTreeSet::new(),
        }
    }
}

impl FilesystemExcludes {
    /// Disables all exclusions, including the default metadata/build exclusions.
    pub fn none() -> Self {
        Self {
            names: BTreeSet::new(),
            subtrees: BTreeSet::new(),
        }
    }

    /// Excludes an exact basename at every depth (files as well as directories).
    pub fn exclude_name(&mut self, name: &str) -> Result<(), FilesystemError> {
        validate_relative(name, Path::new(name))?;
        if name.contains('/') {
            return Err(error(
                name,
                FilesystemErrorKind::InvalidPath("expected one basename"),
            ));
        }
        self.names.insert(name.to_owned());
        Ok(())
    }

    /// Excludes one exact project-relative path and all its descendants.
    pub fn exclude_subtree(&mut self, path: &str) -> Result<(), FilesystemError> {
        validate_relative(path, Path::new(path))?;
        self.subtrees.insert(path.to_owned());
        Ok(())
    }

    fn contains(&self, name: &str, relative: &str) -> bool {
        self.names.contains(name)
            || self.subtrees.contains(relative)
            || relative
                .match_indices('/')
                .any(|(i, _)| self.subtrees.contains(&relative[..i]))
    }
}

/// A budget that stopped the scan. No partial snapshot accompanies the error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilesystemLimit {
    /// Files and directories in output.
    Entries,
    /// Bytes in one relative path.
    PathBytes,
    /// Sum of all output relative path bytes.
    TotalPathBytes,
    /// Enumerated children, including excluded entries and symlinks.
    ScannedEntries,
    /// Relative component depth.
    Depth,
    /// Regular file length.
    FileSize,
    /// Native root path bytes.
    RootPathBytes,
}

/// Failure classification; I/O failures preserve their operation and source.
#[derive(Debug)]
pub enum FilesystemErrorKind {
    /// An invalid service budget, with an explanation.
    InvalidConfiguration(&'static str),
    /// A path violates the exact relative path or native root contract.
    InvalidPath(&'static str),
    /// Native filename bytes cannot be represented as UTF-8.
    NonUtf8,
    /// Root, an ancestor, or a directory being opened is a symlink.
    SymlinkDirectory,
    /// A required directory is a regular file or another non-directory kind.
    NotDirectory,
    /// An included entry is neither a directory, regular file, nor symlink.
    UnsupportedFileType,
    /// Scanning stopped before publishing any output.
    LimitExceeded {
        /// Which configured budget was exceeded.
        limit: FilesystemLimit,
        /// Inclusive budget ceiling.
        maximum: u64,
        /// Observed value that exceeds that ceiling.
        actual: u64,
    },
    /// No new worker was started because all request slots are occupied.
    InFlightLimit {
        /// Shared request ceiling.
        maximum: usize,
    },
    /// Native filesystem or worker-spawn failure.
    Io {
        /// The operation that failed.
        operation: &'static str,
        /// Original OS error.
        source: io::Error,
    },
    /// Cancellation won before consumption of a result.
    Cancelled,
    /// Worker elapsed time reached its configured deadline.
    TimedOut,
    /// A panic was contained at the worker join boundary.
    WorkerPanicked,
    /// Poll was called again after returning a terminal result.
    AlreadyConsumed,
}

impl fmt::Display for FilesystemErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => write!(f, "invalid configuration: {reason}"),
            Self::InvalidPath(reason) => write!(f, "invalid path: {reason}"),
            Self::NonUtf8 => f.write_str("path is not UTF-8"),
            Self::SymlinkDirectory => f.write_str("directory path contains a symlink"),
            Self::NotDirectory => f.write_str("expected a directory"),
            Self::UnsupportedFileType => f.write_str("unsupported special file type"),
            Self::LimitExceeded {
                limit,
                maximum,
                actual,
            } => write!(f, "{limit:?} limit {maximum} exceeded by {actual}"),
            Self::InFlightLimit { maximum } => {
                write!(f, "in-flight request limit {maximum} reached")
            }
            Self::Io { operation, source } => write!(f, "{operation}: {source}"),
            Self::Cancelled => f.write_str("request cancelled"),
            Self::TimedOut => f.write_str("scan deadline exceeded"),
            Self::WorkerPanicked => f.write_str("scan worker panicked"),
            Self::AlreadyConsumed => f.write_str("request result already consumed"),
        }
    }
}

/// Retains the native offending path, including non-UTF-8 bytes on Unix.
#[derive(Debug)]
pub struct FilesystemError {
    /// Original native offending path; Debug/Display escape unusual characters.
    pub path: PathBuf,
    /// Classified failure with budget or OS details where applicable.
    pub kind: FilesystemErrorKind,
}

impl fmt::Display for FilesystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "filesystem discovery at {:?}: {}", self.path, self.kind)
    }
}

impl std::error::Error for FilesystemError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            FilesystemErrorKind::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn error(path: impl AsRef<Path>, kind: FilesystemErrorKind) -> FilesystemError {
    FilesystemError {
        path: path.as_ref().to_owned(),
        kind,
    }
}

fn io_error(path: &Path, operation: &'static str, source: io::Error) -> FilesystemError {
    error(path, FilesystemErrorKind::Io { operation, source })
}

fn bounded(
    path: &Path,
    limit: FilesystemLimit,
    maximum: u64,
    actual: u64,
) -> Result<(), FilesystemError> {
    if actual > maximum {
        Err(error(
            path,
            FilesystemErrorKind::LimitExceeded {
                limit,
                maximum,
                actual,
            },
        ))
    } else {
        Ok(())
    }
}

fn validate_relative(value: &str, path: &Path) -> Result<(), FilesystemError> {
    if value.is_empty()
        || value.contains(['\\', '\0'])
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(error(path, FilesystemErrorKind::InvalidPath("expected an exact UTF-8 relative slash path without empty, dot, parent, NUL or backslash components")));
    }
    Ok(())
}

/// Included entry kinds. Symlinks have no snapshot entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilesystemEntryKind {
    /// Regular file, inspected through metadata only.
    File,
    /// Directory, including empty directories.
    Directory,
}

/// Host-neutral source metadata; no generated asset identity or content hash.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilesystemEntry {
    /// Exact UTF-8 project-relative slash path, without normalization.
    pub path: String,
    /// Whether this entry is a regular file or directory.
    pub kind: FilesystemEntryKind,
    /// File length from metadata; directories use zero, independent of platform.
    pub size_bytes: u64,
}

/// A complete enumeration, not a transactional filesystem point-in-time view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FilesystemSnapshot {
    /// Absolute native root resolved when the request started.
    pub root: PathBuf,
    /// Sorted by exact UTF-8 path bytes. Root itself is not an entry.
    pub entries: Vec<FilesystemEntry>,
    /// Number of enumerated children, including excludes and skipped symlinks.
    pub scanned_entries: usize,
    /// Sum of the UTF-8 path byte lengths in entries.
    pub total_path_bytes: usize,
}

struct Shared {
    limits: FilesystemLimits,
    excludes: FilesystemExcludes,
    in_flight: AtomicUsize,
}

/// Clones share the same in-flight budget. Independent services have own budgets.
#[derive(Clone)]
pub struct FilesystemDiscovery {
    shared: Arc<Shared>,
}

impl FilesystemDiscovery {
    /// Creates a service with immutable limits and exclusion policy.
    /// Only max_in_flight must be positive; zero scan budgets are valid.
    pub fn new(
        limits: FilesystemLimits,
        excludes: FilesystemExcludes,
    ) -> Result<Self, FilesystemError> {
        if limits.max_in_flight == 0 {
            return Err(error(
                "",
                FilesystemErrorKind::InvalidConfiguration("max_in_flight must be positive"),
            ));
        }
        Ok(Self {
            shared: Arc::new(Shared {
                limits,
                excludes,
                in_flight: AtomicUsize::new(0),
            }),
        })
    }

    /// Number of reserved requests across every clone of this service.
    pub fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::Acquire)
    }

    /// Starts real filesystem work on a new, bounded OS worker. No runtime needed.
    /// Completed but unconsumed requests continue occupying an in-flight slot.
    pub fn start(&self, root: impl AsRef<Path>) -> Result<FilesystemScan, FilesystemError> {
        let input = root.as_ref();
        if input.as_os_str().is_empty()
            || input
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(error(
                input,
                FilesystemErrorKind::InvalidPath(
                    "root must be nonempty and contain no parent components",
                ),
            ));
        }
        let input_text = input
            .to_str()
            .ok_or_else(|| error(input, FilesystemErrorKind::NonUtf8))?;
        if input_text.contains('\0') || (cfg!(unix) && input_text.contains('\\')) {
            return Err(error(
                input,
                FilesystemErrorKind::InvalidPath("root contains NUL or a Unix backslash"),
            ));
        }
        bounded(
            input,
            FilesystemLimit::RootPathBytes,
            self.shared.limits.max_root_path_bytes as u64,
            input.as_os_str().len() as u64,
        )?;
        let root = std::path::absolute(input).map_err(|e| io_error(input, "absolute root", e))?;
        if root.to_str().is_none() {
            return Err(error(&root, FilesystemErrorKind::NonUtf8));
        }
        bounded(
            &root,
            FilesystemLimit::RootPathBytes,
            self.shared.limits.max_root_path_bytes as u64,
            root.as_os_str().len() as u64,
        )?;
        self.shared
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.shared.limits.max_in_flight).then(|| n + 1)
            })
            .map_err(|_| {
                error(
                    &root,
                    FilesystemErrorKind::InFlightLimit {
                        maximum: self.shared.limits.max_in_flight,
                    },
                )
            })?;
        let permit = Permit(Arc::clone(&self.shared));
        let cancelled = Arc::new(AtomicBool::new(false));
        let context = ScanContext {
            shared: Arc::clone(&self.shared),
            cancelled: Arc::clone(&cancelled),
            started: Instant::now(),
        };
        let worker_root = root.clone();
        let worker = thread::Builder::new()
            .name("prism-source-discovery".into())
            .spawn(move || scan(worker_root, &context))
            .map_err(|e| io_error(&root, "spawn worker", e))?;
        Ok(FilesystemScan {
            root,
            cancelled,
            worker: Some(worker),
            permit: Some(permit),
        })
    }
}

struct Permit(Arc<Shared>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One owned worker/result. Drop cancels and joins; workers are never detached.
pub struct FilesystemScan {
    root: PathBuf,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<FilesystemSnapshot, FilesystemError>>>,
    permit: Option<Permit>,
}

impl FilesystemScan {
    /// Cancellation also discards a completed snapshot that has not been polled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Whether the worker has exited (also true after its result was consumed).
    /// Inspecting completion does not release the request's in-flight slot.
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Nonblocking: joins only a worker whose thread has finished. Call from the
    /// host's update loop; this is polling, not a Future/waker subscription.
    pub fn poll(&mut self) -> Poll<Result<FilesystemSnapshot, FilesystemError>> {
        let Some(worker) = self.worker.as_ref() else {
            return Poll::Ready(Err(error(&self.root, FilesystemErrorKind::AlreadyConsumed)));
        };
        if !worker.is_finished() {
            return Poll::Pending;
        }
        let result = self
            .worker
            .take()
            .expect("worker checked above")
            .join()
            .unwrap_or_else(|_| Err(error(&self.root, FilesystemErrorKind::WorkerPanicked)));
        self.permit.take();
        if self.cancelled.load(Ordering::Acquire) {
            Poll::Ready(Err(error(&self.root, FilesystemErrorKind::Cancelled)))
        } else {
            Poll::Ready(result)
        }
    }
}

impl Drop for FilesystemScan {
    fn drop(&mut self) {
        self.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.permit.take();
    }
}

struct ScanContext {
    shared: Arc<Shared>,
    cancelled: Arc<AtomicBool>,
    started: Instant,
}

impl ScanContext {
    fn check(&self, path: &Path) -> Result<(), FilesystemError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(error(path, FilesystemErrorKind::Cancelled));
        }
        if self.started.elapsed() >= self.shared.limits.timeout {
            return Err(error(path, FilesystemErrorKind::TimedOut));
        }
        Ok(())
    }

    // Check each ancestor without canonicalize (which would follow symlinks).
    fn directory(&self, path: &Path) -> Result<fs::ReadDir, FilesystemError> {
        let mut prefix = PathBuf::new();
        for component in path.components() {
            prefix.push(component.as_os_str());
            self.check(&prefix)?;
            let metadata = fs::symlink_metadata(&prefix)
                .map_err(|e| io_error(&prefix, "symlink_metadata directory", e))?;
            if metadata.file_type().is_symlink() {
                return Err(error(&prefix, FilesystemErrorKind::SymlinkDirectory));
            }
            if !metadata.is_dir() {
                return Err(error(&prefix, FilesystemErrorKind::NotDirectory));
            }
        }
        self.check(path)?;
        fs::read_dir(path).map_err(|e| io_error(path, "read_dir", e))
    }
}

struct Frame {
    directory: fs::ReadDir,
    path: PathBuf,
    relative: String,
    depth: usize,
}

fn scan(root: PathBuf, context: &ScanContext) -> Result<FilesystemSnapshot, FilesystemError> {
    context.check(&root)?;
    let directory = context.directory(&root)?;
    let mut stack = vec![Frame {
        directory,
        path: root.clone(),
        relative: String::new(),
        depth: 0,
    }];
    let mut snapshot = FilesystemSnapshot {
        root,
        entries: Vec::new(),
        scanned_entries: 0,
        total_path_bytes: 0,
    };
    let limits = &context.shared.limits;
    while let Some(frame) = stack.last_mut() {
        context.check(&frame.path)?;
        let Some(child) = frame.directory.next() else {
            stack.pop();
            continue;
        };
        let child = child.map_err(|e| io_error(&frame.path, "read_dir entry", e))?;
        let path = child.path();
        context.check(&path)?;
        bounded(
            &path,
            FilesystemLimit::ScannedEntries,
            limits.max_scanned_entries as u64,
            (snapshot.scanned_entries as u64).saturating_add(1),
        )?;
        snapshot.scanned_entries += 1;
        let native_name = child.file_name();
        let name = native_name
            .to_str()
            .ok_or_else(|| error(&path, FilesystemErrorKind::NonUtf8))?;
        validate_relative(name, &path)?;
        let path_bytes = frame
            .relative
            .len()
            .saturating_add(usize::from(!frame.relative.is_empty()))
            .saturating_add(name.len());
        bounded(
            &path,
            FilesystemLimit::PathBytes,
            limits.max_path_bytes as u64,
            path_bytes as u64,
        )?;
        let relative = if frame.relative.is_empty() {
            name.to_owned()
        } else {
            format!("{}/{name}", frame.relative)
        };
        if context.shared.excludes.contains(name, &relative) {
            continue;
        }
        // DirEntry::metadata does not follow symlinks, unlike fs::metadata.
        let metadata = child
            .metadata()
            .map_err(|e| io_error(&path, "entry metadata", e))?;
        context.check(&path)?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        let kind = if metadata.is_file() {
            FilesystemEntryKind::File
        } else if metadata.is_dir() {
            FilesystemEntryKind::Directory
        } else {
            return Err(error(&path, FilesystemErrorKind::UnsupportedFileType));
        };
        let depth = frame.depth.saturating_add(1);
        bounded(
            &path,
            FilesystemLimit::Depth,
            limits.max_depth as u64,
            depth as u64,
        )?;
        let size_bytes = if kind == FilesystemEntryKind::File {
            metadata.len()
        } else {
            0
        };
        bounded(
            &path,
            FilesystemLimit::FileSize,
            limits.max_file_size,
            size_bytes,
        )?;
        bounded(
            &path,
            FilesystemLimit::Entries,
            limits.max_entries as u64,
            (snapshot.entries.len() as u64).saturating_add(1),
        )?;
        let total = snapshot.total_path_bytes.saturating_add(relative.len());
        bounded(
            &path,
            FilesystemLimit::TotalPathBytes,
            limits.max_total_path_bytes as u64,
            total as u64,
        )?;
        snapshot.total_path_bytes = total;
        snapshot.entries.push(FilesystemEntry {
            path: relative.clone(),
            kind,
            size_bytes,
        });
        if kind == FilesystemEntryKind::Directory {
            let directory = context.directory(&path)?;
            stack.push(Frame {
                directory,
                path,
                relative,
                depth,
            });
        }
    }
    context.check(&snapshot.root)?;
    snapshot
        .entries
        .sort_unstable_by(|a, b| a.path.cmp(&b.path));
    context.check(&snapshot.root)?;
    Ok(snapshot)
}
