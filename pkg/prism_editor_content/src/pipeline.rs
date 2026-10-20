//! Bounded asynchronous transport for the real asset-build CLI JSONL protocol.
//! Editor actions use `Pipeline::submit_checked`; `submit_unvalidated` is a
//! diagnostic escape hatch, never an editor authorization boundary.
//! Checked results must be polled with `PipelineJob::poll_checked`. Cancellation
//! stops the direct child but cannot roll back an already published revision.

use crate::{
    validation::{ValidatedCatalog, ValidationOperation},
    AssetId, ContentCatalog, ContentSnapshotToken, ContentTaskId, TaskError, TaskKind, TaskState,
    TaskStore,
};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::{
    fmt,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
    task::Poll,
    time::Duration,
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread::{self, JoinHandle},
    time::Instant,
};

/// Per-service concurrency and per-invocation allocation/output budgets.
#[derive(Clone, Copy, Debug)]
pub struct PipelineLimits {
    /// Running and completed-but-unconsumed handles.
    pub max_in_flight: usize,
    /// Combined executable/root/argv UTF-8 bytes.
    pub max_request_bytes: usize,
    /// Bytes per logical/native path.
    pub max_path_bytes: usize,
    /// Combined stdout and stderr bytes, including line separators.
    pub max_output_bytes: usize,
    /// Maximum one JSONL line, excluding LF.
    pub max_line_bytes: usize,
    /// Maximum total parsed JSON event bytes.
    pub max_event_bytes: usize,
    /// Maximum JSONL events (v1 normally emits exactly two).
    pub max_events: usize,
    /// Maximum combined artifact/container/file/dependency records.
    pub max_entries: usize,
    /// Maximum source file size admitted before starting import/build.
    pub max_source_bytes: u64,
    /// Whole invocation deadline, including worker scheduling and parsing.
    pub timeout: Duration,
}
impl Default for PipelineLimits {
    fn default() -> Self {
        Self {
            max_in_flight: 2,
            max_request_bytes: 32 * 1024,
            max_path_bytes: 4096,
            max_output_bytes: 4 * 1024 * 1024,
            max_line_bytes: 2 * 1024 * 1024,
            max_event_bytes: 3 * 1024 * 1024,
            max_events: 2,
            max_entries: 16_384,
            max_source_bytes: 64 * 1024 * 1024,
            timeout: Duration::from_secs(60),
        }
    }
}
/// One primary PNG identity. The CLI reserves `asset - 1` for its source artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineAsset {
    /// Host-assigned primary image identity, greater than zero.
    pub asset: AssetId,
    /// Exact slash-separated project-relative catalog path.
    pub logical_path: String,
}
/// Fixed supported operations. No caller-supplied argv, shell, or environment.
#[derive(Clone, Debug)]
pub enum PipelineRequest {
    /// Decode and publish a source PNG.
    Import {
        /// Source/catalog identity.
        source: PipelineAsset,
        /// Absolute revision-store directory.
        store: PathBuf,
        /// Compare fresh executions before publication.
        verify_determinism: bool,
    },
    /// Explicit build spelling of the PNG publication slice.
    Build {
        /// Source/catalog identity.
        source: PipelineAsset,
        /// Absolute revision-store directory.
        store: PathBuf,
        /// Compare fresh executions before publication.
        verify_determinism: bool,
    },
    /// Cook the current single-PNG revision into a runtime package.
    Package {
        /// Primary identity; the derived source mapping follows CLI semantics.
        source: PipelineAsset,
        /// Absolute revision-store directory.
        store: PathBuf,
        /// Absolute package output directory.
        output: PathBuf,
    },
    /// Validate existing package bytes against the current store revision.
    Validate {
        /// Primary identity used when cooking this package.
        source: PipelineAsset,
        /// Absolute revision-store directory.
        store: PathBuf,
        /// Absolute package directory.
        package: PathBuf,
    },
    /// Read the current committed revision and artifact descriptors.
    Inspect {
        /// Absolute revision-store directory.
        store: PathBuf,
    },
}
impl PipelineRequest {
    /// Exact CLI command name.
    #[must_use]
    pub const fn command(&self) -> &'static str {
        match self {
            Self::Import { .. } => "import",
            Self::Build { .. } => "build",
            Self::Package { .. } => "package",
            Self::Validate { .. } => "validate",
            Self::Inspect { .. } => "inspect",
        }
    }
    fn source(&self) -> Option<&PipelineAsset> {
        match self {
            Self::Import { source, .. }
            | Self::Build { source, .. }
            | Self::Package { source, .. }
            | Self::Validate { source, .. } => Some(source),
            Self::Inspect { .. } => None,
        }
    }
    fn store(&self) -> &Path {
        match self {
            Self::Import { store, .. }
            | Self::Build { store, .. }
            | Self::Package { store, .. }
            | Self::Validate { store, .. }
            | Self::Inspect { store } => store,
        }
    }
}
/// Failures contain no successful partial result. CLI side effects may exist.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineError {
    /// Invalid limits, native path, logical path, or request size.
    InvalidRequest(String),
    /// All bounded handles are occupied.
    Busy,
    /// This implementation needs Unix nonblocking process output sockets.
    UnsupportedPlatform,
    /// Checked submission requires a Cook authorization.
    WrongOperation,
    /// Requested identity/path differs from the authorized catalog.
    UnauthorizedAsset,
    /// Catalog changed while the process was running.
    StaleCatalog,
    /// A checked handle must use checked polling.
    CheckedPollRequired,
    /// Cancellation observed; publication is not rolled back.
    Cancelled,
    /// Invocation deadline exceeded.
    TimedOut,
    /// Named request/output/event/entry/source budget exceeded.
    Budget(&'static str),
    /// Invalid/truncated/out-of-order CLI JSONL or result.
    Protocol(String),
    /// OS/thread/process failure with operation/path context.
    Io(String),
    /// CLI explicitly failed or exited unsuccessfully.
    CommandFailed {
        /// Process status (None if killed by signal).
        exit_code: Option<i32>,
        /// CLI structured error code, if emitted.
        code: Option<String>,
        /// Exact CLI diagnostic text, bounded by event limits.
        message: Option<String>,
        /// Exact bounded stderr bytes.
        stderr: Vec<u8>,
    },
    /// Supervisor panicked or a consumed handle was polled again.
    WorkerStopped,
    /// Direct-child termination/reaping failed.
    Cleanup {
        /// Original failure.
        cause: Box<PipelineError>,
        /// Cleanup OS diagnostic.
        message: String,
    },
}
impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "asset pipeline: {self:?}")
    }
}
impl core::error::Error for PipelineError {}

macro_rules! result_type {
    ($name:ident, $doc:literal, {$($field:ident: $ty:ty),* $(,)?}) => {
        #[doc = $doc]
        #[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
        #[serde(deny_unknown_fields)]
        pub struct $name { $(#[doc = concat!("CLI protocol field `", stringify!($field), "`.")] pub $field: $ty,)* }
    };
}
result_type!(ArtifactResult, "Published artifact descriptor; decimal u128 identities remain exact.", {asset_id: String, artifact_key: String, kind: String, schema: u32, bytes: u64, content_hash: String, dependencies: Vec<String>});
result_type!(PublishResult, "Successful PNG import/build publication.", {revision: u64, build_manifest_hash: String, source_artifact_key: String, image_artifact_key: String, image_asset_id: String, image_content_hash: String, image_width: u32, image_height: u32, image_mip_levels: usize, cache_hits: usize, determinism_verified_jobs: Option<usize>, artifacts: Vec<ArtifactResult>});
result_type!(InspectResult, "Current store snapshot, including the empty-store case.", {current_revision: Option<u64>, build_manifest_hash: Option<String>, asset_count: usize, artifacts: Vec<ArtifactResult>});
result_type!(QualityResult, "Real CLI package validation counters.", {checked_assets: usize, checked_artifacts: usize, checked_blobs: usize, checked_bytes: u64, checked_dependency_edges: usize});
result_type!(ContainerResult, "Cooked container descriptor.", {name: String, bytes: usize, content_hash: String, entries: usize});
result_type!(FileResult, "Package file descriptor relative to `output_directory`.", {path: String, bytes: usize, content_hash: String});
result_type!(PackageResult, "Cook/validate result from the real CLI.", {revision: u64, output_directory: String, build_manifest_hash: String, runtime_manifest_hash: String, runtime_manifest_bytes: usize, container_count: usize, containers: Vec<ContainerResult>, files: Vec<FileResult>, quality: QualityResult, signature: SignatureStatus});
/// Signature verification status actually reported by the CLI.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SignatureStatus {
    /// No signature.
    Unsigned,
    /// Signature written in this operation.
    Signed,
    /// Signature exists but was not verified.
    PresentUnverified,
    /// Verified with an explicit public key.
    VerifiedPublicKey,
    /// Verified with a trust policy.
    VerifiedTrustPolicy,
}
/// Successful command-specific result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineOutput {
    /// Import result.
    Import(PublishResult),
    /// Build result.
    Build(PublishResult),
    /// Cooked runtime package.
    Package(PackageResult),
    /// Validated runtime package.
    Validate(PackageResult),
    /// Store inspection.
    Inspect(InspectResult),
}
/// Complete output, available only after terminal JSONL and matching process exit.
#[derive(Debug)]
pub struct PipelineCompletion {
    /// Strongly typed CLI result.
    pub output: PipelineOutput,
    /// Exact stderr diagnostics; never substituted for structured success.
    pub stderr: Vec<u8>,
    /// Actual JSONL lifecycle events consumed (currently two).
    pub event_count: usize,
    snapshot: Option<ContentSnapshotToken>,
}
impl PipelineCompletion {
    /// Recheck immediately before host publication, including after checked poll.
    /// This fences editor acceptance, not CLI store writes already performed.
    pub fn ensure_current(&self, catalog: &ContentCatalog) -> Result<(), PipelineError> {
        match &self.snapshot {
            Some(token) if catalog.matches_snapshot(token) => Ok(()),
            Some(_) => Err(PipelineError::StaleCatalog),
            None => Err(PipelineError::CheckedPollRequired),
        }
    }
}

/// Cloneable service. Configure its trusted executable and project root once.
#[derive(Clone, Debug)]
pub struct Pipeline {
    executable: PathBuf,
    root: PathBuf,
    limits: PipelineLimits,
    active: Arc<AtomicUsize>,
}
impl Pipeline {
    /// Validate configuration without spawning or reading the filesystem.
    pub fn new(
        executable: PathBuf,
        project_root: PathBuf,
        limits: PipelineLimits,
    ) -> Result<Self, PipelineError> {
        if [
            limits.max_in_flight,
            limits.max_request_bytes,
            limits.max_path_bytes,
            limits.max_output_bytes,
            limits.max_line_bytes,
            limits.max_event_bytes,
            limits.max_events,
            limits.max_entries,
        ]
        .contains(&0)
            || limits.max_source_bytes == 0
            || limits.timeout.is_zero()
            || Instant::now().checked_add(limits.timeout).is_none()
        {
            return Err(PipelineError::InvalidRequest(
                "nonzero representable limits required".into(),
            ));
        }
        native(&executable, limits.max_path_bytes)?;
        native(&project_root, limits.max_path_bytes)?;
        if executable
            .as_os_str()
            .len()
            .saturating_add(project_root.as_os_str().len())
            > limits.max_request_bytes
        {
            return Err(PipelineError::Budget("request"));
        }
        Ok(Self {
            executable,
            root: project_root,
            limits,
            active: Arc::new(AtomicUsize::new(0)),
        })
    }
    /// Editor entry: bind an exact identity/path to the checked Cook catalog.
    /// No alternate native source or opaque mapping file is accepted.
    pub fn submit_checked(
        &self,
        checked: &ValidatedCatalog<'_>,
        request: PipelineRequest,
    ) -> Result<PipelineJob, PipelineError> {
        if checked.operation() != ValidationOperation::Cook {
            return Err(PipelineError::WrongOperation);
        }
        if let Some(source) = request.source()
            && checked
                .catalog()
                .get(source.asset)
                .is_none_or(|a| a.path != source.logical_path)
        {
            return Err(PipelineError::UnauthorizedAsset);
        }
        self.submit(request, Some(checked.catalog().snapshot_token()))
    }
    /// Unvalidated diagnostic transport. UI mutation actions must use checked submission.
    pub fn submit_unvalidated(
        &self,
        request: PipelineRequest,
    ) -> Result<PipelineJob, PipelineError> {
        self.submit(request, None)
    }
    fn submit(
        &self,
        request: PipelineRequest,
        snapshot: Option<ContentSnapshotToken>,
    ) -> Result<PipelineJob, PipelineError> {
        let args = arguments(&request, &self.root, self.limits)?;
        let total = args
            .iter()
            .try_fold(
                self.executable
                    .as_os_str()
                    .len()
                    .saturating_add(self.root.as_os_str().len()),
                |n, a| n.checked_add(a.len()),
            )
            .ok_or(PipelineError::Budget("request"))?;
        if total > self.limits.max_request_bytes {
            return Err(PipelineError::Budget("request"));
        }
        let deadline = Instant::now()
            .checked_add(self.limits.timeout)
            .ok_or_else(|| PipelineError::InvalidRequest("deadline overflow".into()))?;
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limits.max_in_flight).then_some(n + 1)
            })
            .map_err(|_| PipelineError::Busy)?;
        let permit = Permit(self.active.clone());
        let cancelled = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        let pid = Arc::new(AtomicU32::new(0));
        let (cancel, progress, process) = (cancelled.clone(), started.clone(), pid.clone());
        let (exe, root, limits) = (self.executable.clone(), self.root.clone(), self.limits);
        let worker_snapshot = snapshot.clone();
        let worker_request = request.clone();
        let worker = thread::Builder::new()
            .name("prism-asset-pipeline".into())
            .spawn(move || {
                run(
                    &exe,
                    &root,
                    &args,
                    &worker_request,
                    limits,
                    deadline,
                    &cancel,
                    &progress,
                    &process,
                )
                .map(|mut done| {
                    done.snapshot = worker_snapshot;
                    done
                })
            })
            .map_err(|e| PipelineError::Io(format!("start supervisor: {e}")))?;
        Ok(PipelineJob {
            worker: Some(worker),
            cancelled,
            started,
            pid,
            snapshot,
            request,
            _permit: permit,
        })
    }
}
struct Permit(Arc<AtomicUsize>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
/// One asynchronous invocation. Drop cancels and joins its direct-child supervisor.
pub struct PipelineJob {
    worker: Option<JoinHandle<Result<PipelineCompletion, PipelineError>>>,
    cancelled: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    pid: Arc<AtomicU32>,
    snapshot: Option<ContentSnapshotToken>,
    request: PipelineRequest,
    _permit: Permit,
}
impl PipelineJob {
    /// Request cooperative cancellation followed by direct-child kill/wait.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    /// Direct child PID once spawned, retained after reaping for diagnostics.
    #[must_use]
    pub fn process_id(&self) -> Option<u32> {
        match self.pid.load(Ordering::Acquire) {
            0 => None,
            n => Some(n),
        }
    }
    /// True after a genuine `command_started` event; no invented percentage.
    #[must_use]
    pub fn has_started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }
    /// Diagnostic-only poll. Checked jobs require `poll_checked`.
    pub fn poll(&mut self) -> Poll<Result<PipelineCompletion, PipelineError>> {
        if self.snapshot.is_some() {
            return Poll::Ready(Err(PipelineError::CheckedPollRequired));
        }
        self.take_ready()
    }
    /// Reject an obsolete exact snapshot and cancel its still-running child.
    /// Recheck completion before later host publication with `ensure_current`.
    pub fn poll_checked(
        &mut self,
        catalog: &ContentCatalog,
    ) -> Poll<Result<PipelineCompletion, PipelineError>> {
        if !self
            .snapshot
            .as_ref()
            .is_some_and(|token| catalog.matches_snapshot(token))
        {
            self.cancel();
            return Poll::Ready(Err(PipelineError::StaleCatalog));
        }
        self.take_ready()
    }
    fn take_ready(&mut self) -> Poll<Result<PipelineCompletion, PipelineError>> {
        match self.worker.as_ref() {
            Some(worker) if !worker.is_finished() => return Poll::Pending,
            None => return Poll::Ready(Err(PipelineError::WorkerStopped)),
            _ => {}
        }
        Poll::Ready(
            self.worker
                .take()
                .expect("worker checked")
                .join()
                .unwrap_or(Err(PipelineError::WorkerStopped)),
        )
    }
    /// Publish import/build or single-asset package lifecycle into `TaskStore`.
    /// Validate/inspect have no matching task kind and are rejected. Call once per
    /// frame; completed poll consumes the handle and task transition.
    pub fn poll_task_checked(
        &mut self,
        catalog: &ContentCatalog,
        tasks: &mut TaskStore,
        id: ContentTaskId,
    ) -> Result<Poll<Result<PipelineCompletion, PipelineError>>, TaskError> {
        let task = tasks.get(id).ok_or(TaskError::UnknownTask(id))?;
        let matches = match (&self.request, &task.kind) {
            (
                PipelineRequest::Import { source, .. } | PipelineRequest::Build { source, .. },
                TaskKind::Import { asset },
            ) => source.asset == *asset,
            (PipelineRequest::Package { source, .. }, TaskKind::Export { assets }) => {
                assets.as_slice() == [source.asset]
            }
            _ => false,
        };
        if !matches || !matches!(task.state, TaskState::Queued | TaskState::Running) {
            return Err(TaskError::InvalidState(id));
        }
        if task.state == TaskState::Queued {
            tasks.start(id)?;
        }
        let result = self.poll_checked(catalog);
        if let Poll::Ready(done) = &result {
            match done {
                Ok(_) => {
                    tasks.succeed(id)?;
                }
                Err(PipelineError::Cancelled) => {
                    tasks.cancel(id)?;
                }
                Err(error) => {
                    tasks.fail(id, error.to_string())?;
                }
            }
        }
        Ok(result)
    }
}
impl Drop for PipelineJob {
    fn drop(&mut self) {
        self.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn native(path: &Path, limit: usize) -> Result<&str, PipelineError> {
    let value = path
        .to_str()
        .ok_or_else(|| PipelineError::InvalidRequest("non-UTF8 native path".into()))?;
    if !path.is_absolute() || value.contains(['\0', '\\']) || value.len() > limit {
        return Err(PipelineError::InvalidRequest(format!(
            "invalid absolute native path: {path:?}"
        )));
    }
    Ok(value)
}
fn logical(path: &str, limit: usize) -> Result<(), PipelineError> {
    if path.len() > limit
        || path.contains(['\0', '\\', ':'])
        || path
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
    {
        return Err(PipelineError::InvalidRequest(format!(
            "invalid exact logical path: {path:?}"
        )));
    }
    Ok(())
}
fn arguments(
    request: &PipelineRequest,
    root: &Path,
    limits: PipelineLimits,
) -> Result<Vec<String>, PipelineError> {
    let mut args = vec![
        "--json".into(),
        request.command().into(),
        "--store".into(),
        native(request.store(), limits.max_path_bytes)?.into(),
    ];
    if let Some(source) = request.source() {
        logical(&source.logical_path, limits.max_path_bytes)?;
        if source.asset.raw() == 0 {
            return Err(PipelineError::InvalidRequest(
                "asset-id zero cannot reserve source artifact".into(),
            ));
        }
        args.extend([
            "--asset-id".into(),
            source.asset.raw().to_string(),
            "--logical-path".into(),
            source.logical_path.clone(),
        ]);
    }
    match request {
        PipelineRequest::Import {
            source,
            verify_determinism,
            ..
        }
        | PipelineRequest::Build {
            source,
            verify_determinism,
            ..
        } => {
            let path = root.join(&source.logical_path);
            args.extend([
                "--source".into(),
                native(&path, limits.max_path_bytes)?.into(),
            ]);
            if *verify_determinism {
                args.push("--verify-determinism".into());
            }
        }
        PipelineRequest::Package { output, .. } => args.extend([
            "--output".into(),
            native(output, limits.max_path_bytes)?.into(),
        ]),
        PipelineRequest::Validate { package, .. } => args.extend([
            "--package".into(),
            native(package, limits.max_path_bytes)?.into(),
        ]),
        PipelineRequest::Inspect { .. } => {}
    }
    Ok(args)
}

fn checkpoint(cancel: &AtomicBool, deadline: Instant) -> Result<(), PipelineError> {
    if cancel.load(Ordering::Acquire) {
        return Err(PipelineError::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(PipelineError::TimedOut);
    }
    Ok(())
}
fn source_preflight(
    root: &Path,
    request: &PipelineRequest,
    limits: PipelineLimits,
) -> Result<(), PipelineError> {
    if let PipelineRequest::Import { source, .. } | PipelineRequest::Build { source, .. } = request
    {
        let mut path = root.to_path_buf();
        for component in source.logical_path.split('/') {
            path.push(component);
            let metadata = std::fs::symlink_metadata(&path)
                .map_err(|e| PipelineError::Io(format!("source metadata {path:?}: {e}")))?;
            if metadata.file_type().is_symlink() {
                return Err(PipelineError::InvalidRequest(format!(
                    "source symlink rejected: {path:?}"
                )));
            }
        }
        let metadata = std::fs::metadata(&path)
            .map_err(|e| PipelineError::Io(format!("source metadata {path:?}: {e}")))?;
        if !metadata.is_file() {
            return Err(PipelineError::InvalidRequest(format!(
                "source is not a regular file: {path:?}"
            )));
        }
        if metadata.len() > limits.max_source_bytes {
            return Err(PipelineError::Budget("source bytes"));
        }
    }
    Ok(())
}
struct ChildGuard(Option<Child>);
impl ChildGuard {
    fn stop(&mut self) -> io::Result<()> {
        let Some(child) = self.0.as_mut() else {
            return Ok(());
        };
        match child.try_wait()? {
            Some(_) => {}
            None => {
                if let Err(kill_error) = child.kill()
                    && child.try_wait()?.is_none()
                {
                    return Err(kill_error);
                }
            }
        }
        child.wait()?;
        self.0 = None;
        Ok(())
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(unix)]
fn run(
    executable: &Path,
    root: &Path,
    args: &[String],
    request: &PipelineRequest,
    limits: PipelineLimits,
    deadline: Instant,
    cancel: &AtomicBool,
    started: &AtomicBool,
    pid: &AtomicU32,
) -> Result<PipelineCompletion, PipelineError> {
    use std::os::{fd::OwnedFd, unix::net::UnixStream};
    checkpoint(cancel, deadline)?;
    source_preflight(root, request, limits)?;
    checkpoint(cancel, deadline)?;
    let pair = || UnixStream::pair().map_err(|e| PipelineError::Io(format!("output socket: {e}")));
    let (mut stdout, child_stdout) = pair()?;
    let (mut stderr, child_stderr) = pair()?;
    stdout
        .set_nonblocking(true)
        .and_then(|()| stderr.set_nonblocking(true))
        .map_err(|e| PipelineError::Io(format!("nonblocking output: {e}")))?;
    let child = Command::new(executable)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(OwnedFd::from(child_stdout)))
        .stderr(Stdio::from(OwnedFd::from(child_stderr)))
        .spawn()
        .map_err(|e| PipelineError::Io(format!("spawn {executable:?}: {e}")))?;
    pid.store(child.id(), Ordering::Release);
    let mut child = ChildGuard(Some(child));
    let result = (|| {
        let mut parser = Protocol::new(request.command());
        let mut diagnostics = Vec::new();
        let mut line = Vec::new();
        let mut output_bytes = 0usize;
        let (mut out_eof, mut err_eof) = (false, false);
        let mut status = None;
        let mut buffer = [0u8; 4096];
        loop {
            checkpoint(cancel, deadline)?;
            // One bounded read per stream per iteration prevents a flood from
            // starving cancellation, stderr draining, or process status checks.
            for channel in 0..2 {
                let (stream, eof) = if channel == 0 {
                    (&mut stdout, &mut out_eof)
                } else {
                    (&mut stderr, &mut err_eof)
                };
                if *eof {
                    continue;
                }
                match stream.read(&mut buffer) {
                    Ok(0) => *eof = true,
                    Ok(count) => {
                        output_bytes = output_bytes
                            .checked_add(count)
                            .ok_or(PipelineError::Budget("output bytes"))?;
                        if output_bytes > limits.max_output_bytes {
                            return Err(PipelineError::Budget("output bytes"));
                        }
                        if channel == 1 {
                            diagnostics.extend_from_slice(&buffer[..count]);
                        } else {
                            for &byte in &buffer[..count] {
                                if byte == b'\n' {
                                    parser.accept(&line, limits, started)?;
                                    line.clear();
                                } else {
                                    if line.len() >= limits.max_line_bytes {
                                        return Err(PipelineError::Budget("line bytes"));
                                    }
                                    line.push(byte);
                                }
                            }
                        }
                    }
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                        ) => {}
                    Err(e) => return Err(PipelineError::Io(format!("read process output: {e}"))),
                }
            }
            if status.is_none() {
                status = child
                    .0
                    .as_mut()
                    .expect("child retained")
                    .try_wait()
                    .map_err(|e| PipelineError::Io(format!("wait child: {e}")))?;
            }
            if let Some(status) = status
                && out_eof
                && err_eof
            {
                if !line.is_empty() {
                    return Err(PipelineError::Protocol(
                        "truncated JSONL line (missing LF)".into(),
                    ));
                }
                checkpoint(cancel, deadline)?;
                return parser.finish(status, diagnostics, request, limits);
            }
            thread::sleep(Duration::from_millis(2));
        }
    })();
    let cleanup = child.stop();
    match cleanup {
        Ok(()) => result,
        Err(error) => Err(PipelineError::Cleanup {
            cause: Box::new(
                result
                    .err()
                    .unwrap_or_else(|| PipelineError::Io("cleanup after success".into())),
            ),
            message: error.to_string(),
        }),
    }
}
#[cfg(not(unix))]
fn run(
    _executable: &Path,
    _root: &Path,
    _args: &[String],
    _request: &PipelineRequest,
    _limits: PipelineLimits,
    _deadline: Instant,
    _cancel: &AtomicBool,
    _started: &AtomicBool,
    _pid: &AtomicU32,
) -> Result<PipelineCompletion, PipelineError> {
    Err(PipelineError::UnsupportedPlatform)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    sequence: usize,
    event: String,
    command: String,
    result: Value,
    error: Value,
    exit_code: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CliError {
    code: String,
    message: String,
}
struct Protocol {
    command: &'static str,
    events: usize,
    bytes: usize,
    terminal: Option<(Value, Option<CliError>, i32)>,
}
impl Protocol {
    fn new(command: &'static str) -> Self {
        Self {
            command,
            events: 0,
            bytes: 0,
            terminal: None,
        }
    }
    fn accept(
        &mut self,
        bytes: &[u8],
        limits: PipelineLimits,
        started: &AtomicBool,
    ) -> Result<(), PipelineError> {
        if self.events >= limits.max_events {
            return Err(PipelineError::Budget("event count"));
        }
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .ok_or(PipelineError::Budget("event bytes"))?;
        if self.bytes > limits.max_event_bytes {
            return Err(PipelineError::Budget("event bytes"));
        }
        let event: Envelope = serde_json::from_slice(bytes)
            .map_err(|e| PipelineError::Protocol(format!("event {}: {e}", self.events)))?;
        if event.schema_version != 1
            || event.sequence != self.events
            || event.command != self.command
            || self.terminal.is_some()
        {
            return Err(PipelineError::Protocol(
                "schema/command/sequence or duplicate terminal mismatch".into(),
            ));
        }
        match (self.events, event.event.as_str()) {
            (0, "command_started")
                if event.result.is_null() && event.error.is_null() && event.exit_code.is_null() =>
            {
                started.store(true, Ordering::Release);
            }
            (1, "command_succeeded")
                if event.result.is_object()
                    && event.error.is_null()
                    && event.exit_code.as_i64() == Some(0) =>
            {
                self.terminal = Some((event.result, None, 0));
            }
            (1, "command_failed")
                if event.result.is_null() && matches!(event.exit_code.as_i64(), Some(1 | 2)) =>
            {
                let error: CliError = serde_json::from_value(event.error)
                    .map_err(|e| PipelineError::Protocol(format!("CLI error: {e}")))?;
                if error.code.is_empty() || error.message.is_empty() {
                    return Err(PipelineError::Protocol("empty CLI error".into()));
                }
                self.terminal = Some((
                    Value::Null,
                    Some(error),
                    event.exit_code.as_i64().expect("checked exit") as i32,
                ));
            }
            _ => {
                return Err(PipelineError::Protocol(
                    "unexpected lifecycle event or payload".into(),
                ))
            }
        }
        self.events += 1;
        Ok(())
    }
    fn finish(
        self,
        status: ExitStatus,
        stderr: Vec<u8>,
        request: &PipelineRequest,
        limits: PipelineLimits,
    ) -> Result<PipelineCompletion, PipelineError> {
        let Some((value, error, code)) = self.terminal else {
            if !status.success() {
                return Err(PipelineError::CommandFailed {
                    exit_code: status.code(),
                    code: None,
                    message: None,
                    stderr,
                });
            }
            return Err(PipelineError::Protocol("missing terminal event".into()));
        };
        if status.code() != Some(code) {
            return Err(PipelineError::Protocol(format!(
                "terminal exit {code} differs from OS exit {:?}",
                status.code()
            )));
        }
        if let Some(error) = error {
            return Err(PipelineError::CommandFailed {
                exit_code: status.code(),
                code: Some(error.code),
                message: Some(error.message),
                stderr,
            });
        }
        count_entries(&value, limits.max_entries)?;
        let output = decode_output(value, request)?;
        Ok(PipelineCompletion {
            output,
            stderr,
            event_count: self.events,
            snapshot: None,
        })
    }
}
fn count_entries(value: &Value, limit: usize) -> Result<(), PipelineError> {
    fn visit(value: &Value, count: &mut usize, limit: usize) -> Result<(), PipelineError> {
        match value {
            Value::Array(values) => {
                *count = count
                    .checked_add(values.len())
                    .ok_or(PipelineError::Budget("result entries"))?;
                if *count > limit {
                    return Err(PipelineError::Budget("result entries"));
                }
                for value in values {
                    visit(value, count, limit)?;
                }
            }
            Value::Object(values) => {
                for value in values.values() {
                    visit(value, count, limit)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut count = 0;
    visit(value, &mut count, limit)
}
fn decode<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, PipelineError> {
    serde_json::from_value(value).map_err(|e| PipelineError::Protocol(format!("typed result: {e}")))
}
fn hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn decimal(value: &str) -> bool {
    value.parse::<u128>().is_ok_and(|n| n.to_string() == value)
}
fn artifacts_valid(artifacts: &[ArtifactResult]) -> bool {
    let mut ids = alloc::collections::BTreeSet::new();
    artifacts.iter().all(|a| {
        decimal(&a.asset_id)
            && ids.insert(&a.asset_id)
            && !a.kind.is_empty()
            && hex(&a.artifact_key)
            && hex(&a.content_hash)
            && a.dependencies.iter().all(|d| hex(d))
    })
}
fn decode_output(value: Value, request: &PipelineRequest) -> Result<PipelineOutput, PipelineError> {
    let invalid =
        || PipelineError::Protocol("inconsistent result identity, counts, paths or hashes".into());
    match request {
        PipelineRequest::Import { source, .. } | PipelineRequest::Build { source, .. } => {
            let result: PublishResult = decode(value)?;
            if result.image_asset_id != source.asset.raw().to_string()
                || result.image_width == 0
                || result.image_height == 0
                || result.image_mip_levels == 0
                || ![
                    &result.build_manifest_hash,
                    &result.source_artifact_key,
                    &result.image_artifact_key,
                    &result.image_content_hash,
                ]
                .into_iter()
                .all(|s| hex(s))
                || !artifacts_valid(&result.artifacts)
                || !result.artifacts.iter().any(|a| {
                    a.asset_id == result.image_asset_id
                        && a.artifact_key == result.image_artifact_key
                        && a.content_hash == result.image_content_hash
                })
            {
                return Err(invalid());
            }
            Ok(if matches!(request, PipelineRequest::Import { .. }) {
                PipelineOutput::Import(result)
            } else {
                PipelineOutput::Build(result)
            })
        }
        PipelineRequest::Package { output, .. }
        | PipelineRequest::Validate {
            package: output, ..
        } => {
            let result: PackageResult = decode(value)?;
            if ![&result.build_manifest_hash, &result.runtime_manifest_hash]
                .into_iter()
                .all(|s| hex(s))
                || result.container_count != result.containers.len()
                || Path::new(&result.output_directory) != output
                || result.containers.iter().any(|c| !hex(&c.content_hash))
                || result
                    .files
                    .iter()
                    .any(|f| !hex(&f.content_hash) || logical(&f.path, usize::MAX).is_err())
            {
                return Err(invalid());
            }
            Ok(if matches!(request, PipelineRequest::Package { .. }) {
                PipelineOutput::Package(result)
            } else {
                PipelineOutput::Validate(result)
            })
        }
        PipelineRequest::Inspect { .. } => {
            let result: InspectResult = decode(value)?;
            if result.asset_count != result.artifacts.len()
                || !artifacts_valid(&result.artifacts)
                || result.current_revision.is_some() != result.build_manifest_hash.is_some()
                || result.build_manifest_hash.as_ref().is_some_and(|h| !hex(h))
            {
                return Err(invalid());
            }
            Ok(PipelineOutput::Inspect(result))
        }
    }
}
