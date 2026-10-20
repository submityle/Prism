//! Bounded, read-only Git status acquisition without a runtime dependency.
//!
//! A request never accesses a catalog: only a complete successful snapshot can be
//! converted for token-checked `ContentCatalog::publish_vcs` publication.

use crate::VcsRecord;
#[cfg(unix)]
use crate::VcsState;
use alloc::{
    boxed::Box,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
#[cfg(unix)]
use alloc::{collections::BTreeMap, format};
use core::{
    fmt,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
    time::Duration,
};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::Instant,
};

/// Shared concurrency and per-request resource limits.
#[derive(Clone, Copy, Debug)]
pub struct GitLimits {
    /// Combined stdout/stderr bytes across both commands.
    pub max_output_bytes: usize,
    /// Path occurrences across both commands, including rename sources.
    pub max_paths: usize,
    /// Active/ready requests across provider clones, including cleanup.
    pub max_concurrent: usize,
    /// Whole request deadline, including parsing.
    pub timeout: Duration,
}
impl Default for GitLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: 8 * 1024 * 1024,
            max_paths: 100_000,
            max_concurrent: 2,
            timeout: Duration::from_secs(10),
        }
    }
}
/// Acquisition failures never contain partial snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GitError {
    /// Zero limits or unrepresentable deadline.
    InvalidLimits,
    /// All slots are occupied.
    Busy,
    /// This std-only implementation requires Unix sockets.
    UnsupportedPlatform,
    /// Cancellation requested.
    Cancelled,
    /// Request deadline exceeded.
    TimedOut,
    /// Combined output budget exceeded.
    OutputBudget,
    /// Combined path occurrence budget exceeded.
    PathBudget,
    /// Thread, filesystem, spawn, read or wait failure.
    Io(String),
    /// Nonzero Git exit with bounded exact diagnostics.
    CommandFailed {
        /// Failed subcommand.
        command: &'static str,
        /// Exit code (None for signal).
        code: Option<i32>,
        /// Exact stderr bytes.
        stderr: Vec<u8>,
    },
    /// Malformed output or invalid repository root.
    Protocol(String),
    /// Raw path cannot be represented without loss.
    UnrepresentablePath(Vec<u8>),
    /// Worker disconnected unexpectedly.
    WorkerStopped,
    /// Termination/reaping failed while handling another error.
    Cleanup {
        /// Original error.
        cause: Box<GitError>,
        /// Cleanup diagnostic.
        message: String,
    },
}
impl fmt::Display for GitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Git snapshot failed: {self:?}")
    }
}
impl core::error::Error for GitError {}
/// A root-relative path with lossless XY and reduced catalog metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitEntry {
    /// Exact path preserving whitespace and Unicode.
    pub path: String,
    /// Index/worktree status; clean entries use two spaces.
    pub xy: [u8; 2],
    /// Original path for rename/copy destinations.
    pub original_path: Option<String>,
    /// Catalog projection; XY preserves simultaneous changes.
    pub record: VcsRecord,
}
/// Complete successful acquisition sorted by unique path. Git provides no
/// transaction across commands/concurrent edits; discard obsolete request epochs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitSnapshot {
    /// Absolute canonical worktree root.
    pub root: PathBuf,
    /// Tracked clean paths, status overrides and rename-source deletions.
    pub entries: Vec<GitEntry>,
    /// Successful stderr, retained as exact bounded diagnostic bytes.
    pub diagnostics: Vec<u8>,
}
impl GitSnapshot {
    /// Consume into a `ContentCatalog::replace_vcs_statuses` batch.
    /// Omitted paths remain Unknown, never inferred Clean.
    pub fn into_vcs_statuses(self) -> impl Iterator<Item = (String, VcsRecord)> {
        self.entries.into_iter().map(|e| (e.path, e.record))
    }
}
/// Cloneable concurrency domain. Share one instance per editor Git service.
#[derive(Clone, Debug)]
pub struct GitProvider {
    limits: GitLimits,
    active: Arc<AtomicUsize>,
}
impl GitProvider {
    /// Validate limits without launching a process.
    pub fn new(limits: GitLimits) -> Result<Self, GitError> {
        if limits.max_output_bytes == 0
            || limits.max_paths == 0
            || limits.max_concurrent == 0
            || limits.timeout.is_zero()
            || Instant::now().checked_add(limits.timeout).is_none()
        {
            return Err(GitError::InvalidLimits);
        }
        Ok(Self {
            limits,
            active: Arc::new(AtomicUsize::new(0)),
        })
    }
    /// Nonblocking launch. Root must contain .git (file or directory).
    /// Filesystem/process work happens entirely on the supervisor thread.
    pub fn start(&self, root: impl Into<PathBuf>) -> Result<GitRequest, GitError> {
        let deadline = Instant::now()
            .checked_add(self.limits.timeout)
            .ok_or(GitError::InvalidLimits)?;
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                if n < self.limits.max_concurrent {
                    Some(n + 1)
                } else {
                    None
                }
            })
            .map_err(|_| GitError::Busy)?;
        let permit = Arc::new(Permit(self.active.clone()));
        let worker_permit = permit.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let worker_ready = ready.clone();
        let process_id = Arc::new(AtomicU32::new(0));
        let worker_process_id = process_id.clone();
        let limits = self.limits;
        let root = root.into();
        let (send, receive) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("prism-git-status".into())
            .spawn(move || {
                let result = acquire(
                    root,
                    limits,
                    deadline,
                    &worker_cancelled,
                    &worker_process_id,
                );
                // Capacity travels with the result: ready success AND error results
                // stay bounded. A dropped receiver destroys result and permit only
                // after acquisition has closed pipes and reaped the child.
                let _ = send.send((result, worker_permit));
                worker_ready.store(true, Ordering::Release);
            })
            .map_err(|e| GitError::Io(e.to_string()))?;
        Ok(GitRequest {
            receive,
            cancelled,
            process_id,
            ready,
            permit: Some(permit),
            finished: false,
        })
    }
    /// Active/ready requests; dropped requests count until cleanup completes.
    pub fn active_requests(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}
struct Permit(Arc<AtomicUsize>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
/// Nonblocking handle. Poll through completion to observe cleanup failures.
/// Drop cancels; the supervisor retains process ownership through reaping.
pub struct GitRequest {
    receive: Receiver<(Result<GitSnapshot, GitError>, Arc<Permit>)>,
    cancelled: Arc<AtomicBool>,
    process_id: Arc<AtomicU32>,
    ready: Arc<AtomicBool>,
    permit: Option<Arc<Permit>>,
    finished: bool,
}
impl GitRequest {
    /// Whether the supervisor has delivered a terminal result. This does not
    /// consume it or release its concurrency slot. It stays true after polling.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) || self.finished
    }
    /// Current child PID for diagnostics; None between commands and after reap.
    /// This is an observation, not authority to signal a potentially reused PID.
    pub fn process_id(&self) -> Option<u32> {
        match self.process_id.load(Ordering::Acquire) {
            0 => None,
            pid => Some(pid),
        }
    }
    /// Signal cancellation without waiting.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    /// One terminal result, then None; None also means pending. Cancellation
    /// before consumption rejects even a ready successful snapshot.
    pub fn poll(&mut self) -> Option<Result<GitSnapshot, GitError>> {
        if self.finished {
            return None;
        }
        match self.receive.try_recv() {
            Ok((result, worker_permit)) => {
                self.finished = true;
                drop(worker_permit);
                self.permit.take();
                Some(
                    if result.is_ok() && self.cancelled.load(Ordering::Acquire) {
                        Err(GitError::Cancelled)
                    } else {
                        result
                    },
                )
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                self.permit.take();
                Some(Err(GitError::WorkerStopped))
            }
        }
    }
}
impl Drop for GitRequest {
    fn drop(&mut self) {
        self.cancel();
    }
}
#[cfg(unix)]
fn checkpoint(cancelled: &AtomicBool, deadline: Instant) -> Result<(), GitError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(GitError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(GitError::TimedOut);
    }
    Ok(())
}
#[cfg(unix)]
struct Budget {
    bytes: usize,
    paths: usize,
    limits: GitLimits,
}
#[cfg(unix)]
fn acquire(
    root: PathBuf,
    limits: GitLimits,
    deadline: Instant,
    cancelled: &AtomicBool,
    process_id: &AtomicU32,
) -> Result<GitSnapshot, GitError> {
    checkpoint(cancelled, deadline)?;
    let root = root
        .canonicalize()
        .map_err(|e| GitError::Io(e.to_string()))?;
    if !root
        .join(".git")
        .try_exists()
        .map_err(|e| GitError::Io(e.to_string()))?
    {
        return Err(GitError::Protocol(
            "expected worktree root containing .git".into(),
        ));
    }
    let mut budget = Budget {
        bytes: 0,
        paths: 0,
        limits,
    };
    let (tracked, mut diagnostics) = run_git(
        &root,
        "ls-files",
        &["--cached", "--deduplicate", "-z"],
        &mut budget,
        deadline,
        cancelled,
        process_id,
    )?;
    let (status, warnings) = run_git(
        &root,
        "status",
        &[
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=traditional",
            "--ignore-submodules=dirty",
            "--renames",
        ],
        &mut budget,
        deadline,
        cancelled,
        process_id,
    )?;
    diagnostics.extend(warnings);
    let mut entries = BTreeMap::new();
    for raw in fields(&tracked)? {
        checkpoint(cancelled, deadline)?;
        let path = path(raw, false)?;
        entries.insert(
            path.clone(),
            GitEntry {
                path,
                xy: *b"  ",
                original_path: None,
                record: VcsRecord {
                    state: VcsState::Clean,
                    ..VcsRecord::default()
                },
            },
        );
    }
    let mut overrides = Vec::new();
    let mut raw = fields(&status)?.into_iter();
    while let Some(field) = raw.next() {
        checkpoint(cancelled, deadline)?;
        if field.len() < 4 || field[2] != b' ' {
            return Err(GitError::Protocol("invalid XY record".into()));
        }
        let xy = [field[0], field[1]];
        let record = record(xy)?;
        let destination = path(&field[3..], xy == *b"!!")?;
        let original_path = if xy.contains(&b'R') || xy.contains(&b'C') {
            Some(path(
                raw.next()
                    .ok_or_else(|| GitError::Protocol("missing rename/copy source".into()))?,
                false,
            )?)
        } else {
            None
        };
        if xy.contains(&b'R') {
            let source = original_path.as_ref().expect("rename has source");
            entries.insert(
                source.clone(),
                GitEntry {
                    path: source.clone(),
                    xy: [
                        if xy[0] == b'R' { b'D' } else { b' ' },
                        if xy[1] == b'R' { b'D' } else { b' ' },
                    ],
                    original_path: None,
                    record: VcsRecord {
                        state: VcsState::Deleted,
                        staged: xy[0] == b'R',
                        ..VcsRecord::default()
                    },
                },
            );
        }
        overrides.push(GitEntry {
            path: destination,
            xy,
            original_path,
            record,
        });
    }
    // Real records override synthetic deletions if an old name is recreated.
    for entry in overrides {
        entries.insert(entry.path.clone(), entry);
    }
    let entries = entries.into_values().collect();
    checkpoint(cancelled, deadline)?;
    Ok(GitSnapshot {
        root,
        entries,
        diagnostics,
    })
}
#[cfg(not(unix))]
fn acquire(
    _: PathBuf,
    _: GitLimits,
    _: Instant,
    _: &AtomicBool,
    _: &AtomicU32,
) -> Result<GitSnapshot, GitError> {
    Err(GitError::UnsupportedPlatform)
}
#[cfg(unix)]
fn fields(output: &[u8]) -> Result<Vec<&[u8]>, GitError> {
    if output.is_empty() {
        return Ok(Vec::new());
    }
    if output.last() != Some(&0) {
        return Err(GitError::Protocol("missing terminal NUL".into()));
    }
    Ok(output[..output.len() - 1]
        .split(|byte| *byte == 0)
        .collect())
}
#[cfg(unix)]
fn path(raw: &[u8], directory: bool) -> Result<String, GitError> {
    let value =
        core::str::from_utf8(raw).map_err(|_| GitError::UnrepresentablePath(raw.to_vec()))?;
    let value = if directory {
        value.strip_suffix('/').unwrap_or(value)
    } else {
        value
    };
    if value.is_empty()
        || value.starts_with('/')
        || value.contains('\\')
        || value
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(GitError::UnrepresentablePath(raw.to_vec()));
    }
    Ok(value.to_owned())
}
#[cfg(unix)]
fn record(xy: [u8; 2]) -> Result<VcsRecord, GitError> {
    let conflict = matches!(&xy, b"DD" | b"AU" | b"UD" | b"UA" | b"DU" | b"AA" | b"UU");
    let state = if conflict {
        VcsState::Conflicted
    } else if xy == *b"??" {
        VcsState::Untracked
    } else if xy == *b"!!" {
        VcsState::Ignored
    } else {
        if xy == *b"  " || !xy.iter().all(|c| b" MTADRC".contains(c)) {
            return Err(GitError::Protocol(format!("invalid status {xy:?}")));
        }
        if xy.contains(&b'R') {
            VcsState::Renamed
        } else if xy.contains(&b'D') {
            VcsState::Deleted
        } else if xy.contains(&b'A') || xy.contains(&b'C') {
            VcsState::Added
        } else {
            VcsState::Modified
        }
    };
    Ok(VcsRecord {
        state,
        staged: !conflict && !matches!(xy[0], b' ' | b'?' | b'!'),
        revision: None,
        lock_owner: None,
    })
}

#[cfg(unix)]
fn run_git(
    root: &std::path::Path,
    command: &'static str,
    args: &[&str],
    budget: &mut Budget,
    deadline: Instant,
    cancelled: &AtomicBool,
    process_id: &AtomicU32,
) -> Result<(Vec<u8>, Vec<u8>), GitError> {
    use std::{
        io::{ErrorKind, Read},
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Command, Stdio},
    };
    checkpoint(cancelled, deadline)?;
    // Each pipe is a socketpair: only the parent's end is nonblocking. Unlike
    // blocking reader threads this lets timeout/cancellation close all pipes
    // promptly, even if a descendant were to retain a write descriptor.
    let (mut stdout, stdout_child) = UnixStream::pair().map_err(|e| GitError::Io(e.to_string()))?;
    let (mut stderr, stderr_child) = UnixStream::pair().map_err(|e| GitError::Io(e.to_string()))?;
    stdout
        .set_nonblocking(true)
        .map_err(|e| GitError::Io(e.to_string()))?;
    stderr
        .set_nonblocking(true)
        .map_err(|e| GitError::Io(e.to_string()))?;
    let mut process = Command::new("git");
    // Environment redirection must not silently select another repository/index
    // or inject configuration. Fsmonitor hooks are disabled on both commands.
    for (key, _) in std::env::vars_os() {
        if key.as_encoded_bytes().starts_with(b"GIT_") {
            process.env_remove(key);
        }
    }
    process
        .current_dir(root)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .args([
            "--no-pager",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "status.relativePaths=false",
            "-c",
            "status.renames=true",
        ])
        .arg(command)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(OwnedFd::from(stdout_child)))
        .stderr(Stdio::from(OwnedFd::from(stderr_child)));
    let mut child = process.spawn().map_err(|e| GitError::Io(e.to_string()))?;
    process_id.store(child.id(), Ordering::Release);
    drop(process); // Close parent's child-end copies so EOF can be observed.
    let mut output = [Vec::new(), Vec::new()];
    let mut eof = [false; 2];
    let result = (|| {
        let mut exit = None;
        loop {
            checkpoint(cancelled, deadline)?;
            let mut progress = false;
            for (index, stream) in [&mut stdout, &mut stderr].into_iter().enumerate() {
                if eof[index] {
                    continue;
                }
                let mut bytes = [0_u8; 8192];
                match stream.read(&mut bytes) {
                    Ok(0) => eof[index] = true,
                    Ok(n) => {
                        progress = true;
                        budget.bytes = budget.bytes.checked_add(n).ok_or(GitError::OutputBudget)?;
                        if budget.bytes > budget.limits.max_output_bytes {
                            return Err(GitError::OutputBudget);
                        }
                        if index == 0 {
                            budget.paths = budget
                                .paths
                                .checked_add(bytes[..n].iter().filter(|b| **b == 0).count())
                                .ok_or(GitError::PathBudget)?;
                            if budget.paths > budget.limits.max_paths {
                                return Err(GitError::PathBudget);
                            }
                        }
                        output[index].extend_from_slice(&bytes[..n]);
                    }
                    Err(e)
                        if e.kind() == ErrorKind::WouldBlock
                            || e.kind() == ErrorKind::Interrupted => {}
                    Err(e) => return Err(GitError::Io(e.to_string())),
                }
            }
            if exit.is_none() {
                exit = child.try_wait().map_err(|e| GitError::Io(e.to_string()))?;
            }
            if let Some(status) = exit
                && eof.iter().all(|v| *v)
            {
                if !status.success() {
                    return Err(GitError::CommandFailed {
                        command,
                        code: status.code(),
                        stderr: core::mem::take(&mut output[1]),
                    });
                }
                return Ok((
                    core::mem::take(&mut output[0]),
                    core::mem::take(&mut output[1]),
                ));
            }
            if !progress {
                thread::sleep(Duration::from_millis(2));
            }
        }
    })();
    if let Err(cause) = result {
        let kill = child.kill();
        let wait = child.wait(); // Reap even if exit races termination.
        process_id.store(0, Ordering::Release);
        if let Err(error) = wait {
            return Err(GitError::Cleanup {
                cause: Box::new(cause),
                message: format!("kill={kill:?}; wait={error}"),
            });
        }
        if let Err(error) = kill
            && error.kind() != ErrorKind::InvalidInput
        {
            return Err(GitError::Cleanup {
                cause: Box::new(cause),
                message: format!("kill={error}; child reaped"),
            });
        }
        return Err(cause);
    }
    process_id.store(0, Ordering::Release);
    result
}
