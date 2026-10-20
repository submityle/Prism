//! Bounded asynchronous files. Only the host adopts loads or confirms tokens.
use crate::{TimelineDocument, TimelineError, TimelineLimits, TimelineSnapshotToken};
use alloc::sync::Arc;
use core::fmt;
use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Mutex,
    },
    task::Poll,
    thread,
    time::{Duration, Instant},
};

/// Hard bound on combined running, queued and unconsumed result slots.
pub const MAX_TIMELINE_STORAGE_REQUESTS: usize = 8;
/// Hard bound on each document or expected-revision byte string.
pub const MAX_TIMELINE_STORAGE_BYTES: usize = 64 * 1024 * 1024;
static TEMPORARY_ID: AtomicU64 = AtomicU64::new(1);

/// Resource policy for one serial worker and its retained results.
#[derive(Clone, Copy, Debug)]
pub struct TimelineStorageLimits {
    /// Combined running, queued and unconsumed result slots.
    pub max_requests: usize,
    /// Serialized-byte cap checked on admission and during reads.
    pub max_bytes: usize,
    /// Limits passed to the real document codec on the worker.
    pub document: TimelineLimits,
}
impl Default for TimelineStorageLimits {
    fn default() -> Self {
        Self {
            max_requests: 2,
            max_bytes: 8 * 1024 * 1024,
            document: TimelineLimits::default(),
        }
    }
}

/// Exact revision. Absence differs from an existing zero-byte file. Cloning
/// shares immutable bytes; equality compares all bytes, never mtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineFileRevision {
    /// Target is expected not to exist.
    Missing,
    /// Exact file contents. Submission enforces the service byte cap.
    Present(Arc<[u8]>),
}

/// Validated candidate; adoption remains an explicit host decision.
#[derive(Debug)]
pub struct TimelineLoadCandidate {
    /// Real decoded document, independent from the live editor.
    pub document: TimelineDocument,
    /// Exact observed bytes for a later conditional save.
    pub revision: TimelineFileRevision,
}

/// Persistence confirmation after successful rename.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineSaveDurability {
    /// Staged file and parent directory were synced successfully.
    Synced,
    /// Rename succeeded, but directory sync failed. The file is published.
    Uncertain {
        /// Portable OS error category.
        kind: io::ErrorKind,
    },
}

/// Publication receipt. Never automatically marks a live editor clean.
#[derive(Clone, Debug)]
pub struct TimelineSaveReceipt {
    /// Identity supplied with the bytes; compare with the current host token.
    pub snapshot_token: TimelineSnapshotToken,
    /// Exact published bytes for the next conditional save.
    pub revision: TimelineFileRevision,
    /// Explicit directory-sync outcome after publication.
    pub durability: TimelineSaveDurability,
    /// Deadline passed after crossing the irrevocable publication fence.
    pub deadline_exceeded: bool,
}

/// Owned output, retained in a bounded request slot until polled/dropped.
#[derive(Debug)]
pub enum TimelineStorageResult {
    /// Decoded candidate and exact file revision.
    Loaded(TimelineLoadCandidate),
    /// Published file and original snapshot identity.
    Saved(TimelineSaveReceipt),
}

/// Storage failures. Except `PublicationUncertain`, this operation did not replace
/// the target. External writers remain responsible for their own writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineStorageError {
    /// Invalid bounds, absolute filename or unrepresentable deadline.
    InvalidConfiguration,
    /// Slots are full, including retained results, or the file lock is held.
    Busy,
    /// Input, expected revision or file exceeded the serialized-byte cap.
    ByteLimit,
    /// Target or lock is a symlink or not a regular file.
    UnsafePath,
    /// Exact current bytes differ from the submitted expected revision.
    Conflict,
    /// Loaded file did not exist.
    Missing,
    /// Cancelled before publication; late worker output is discarded.
    Cancelled,
    /// Deadline expired before publication; late worker output is discarded.
    TimedOut,
    /// Actual document codec rejected bytes; the target was not replaced.
    Codec(TimelineError),
    /// An OS operation failed before publication.
    Io {
        /// Fixed operation label.
        operation: &'static str,
        /// Portable OS category.
        kind: io::ErrorKind,
    },
    /// Rename returned an error after entering publication. Reload to reconcile
    /// disk before retrying; a remote filesystem may have acted.
    PublicationUncertain {
        /// Portable OS category from rename.
        kind: io::ErrorKind,
    },
    /// Worker is unavailable or request identities are exhausted.
    Unavailable,
    /// This handle's terminal output was already consumed.
    Consumed,
}
impl fmt::Display for TimelineStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(f, "timeline storage {operation}: {kind}"),
            Self::PublicationUncertain { kind } => {
                write!(f, "timeline publication uncertain: {kind}")
            }
            Self::Codec(error) => write!(f, "timeline file: {error}"),
            other => write!(f, "timeline storage {other:?}"),
        }
    }
}
impl std::error::Error for TimelineStorageError {}

/// Whether cancellation can still prevent publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimelineCancelOutcome {
    /// Fenced; this request cannot subsequently replace the file.
    Cancelled,
    /// Publication began or output was consumed. Inspect the eventual result.
    TooLate,
}

struct Shared {
    active: AtomicUsize,
    next: AtomicU64,
    stopping: AtomicBool,
}
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Working,
    Publishing,
    Complete,
}
struct State {
    phase: Phase,
    abort: Option<TimelineStorageError>,
    output: Option<Result<TimelineStorageResult, TimelineStorageError>>,
}
struct Slot {
    id: u64,
    deadline: Instant,
    shared: Arc<Shared>,
    state: Mutex<State>,
    #[cfg(test)]
    hook: Option<Arc<TestHook>>,
}
impl Slot {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn expire(&self, state: &mut State) {
        if state.phase == Phase::Working && state.abort.is_none() {
            let reason = if self.shared.stopping.load(Ordering::Acquire) {
                Some(TimelineStorageError::Cancelled)
            } else if Instant::now() >= self.deadline {
                Some(TimelineStorageError::TimedOut)
            } else {
                None
            };
            if let Some(reason) = reason {
                state.output = Some(Err(reason.clone()));
                state.abort = Some(reason);
            }
        }
    }
    fn check(&self) -> Result<(), TimelineStorageError> {
        let mut state = self.state();
        self.expire(&mut state);
        state.abort.clone().map_or(Ok(()), Err)
    }
    fn publish(&self) -> Result<(), TimelineStorageError> {
        let mut state = self.state();
        self.expire(&mut state);
        if let Some(error) = &state.abort {
            return Err(error.clone());
        }
        state.phase = Phase::Publishing;
        Ok(())
    }
    fn finish(&self, result: Result<TimelineStorageResult, TimelineStorageError>) {
        let mut state = self.state();
        self.expire(&mut state);
        if state.abort.is_none() {
            state.output = Some(result);
            state.phase = Phase::Complete;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Timeline, TimelineEditor};

    static SERIAL: AtomicU64 = AtomicU64::new(1);
    const WAIT: Duration = Duration::from_secs(10);
    struct Project(PathBuf);
    impl Project {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "prism-timeline-fence-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn path(&self) -> PathBuf {
            self.0.join("take.timeline")
        }
    }
    impl Drop for Project {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn bytes(rate: u64) -> Vec<u8> {
        TimelineDocument {
            ticks_per_second: rate,
            timeline: Timeline::default(),
        }
        .encode(TimelineLimits::default(), 1024)
        .unwrap()
    }
    fn token() -> TimelineSnapshotToken {
        TimelineEditor::new(Timeline::default(), TimelineLimits::default())
            .unwrap()
            .snapshot_token()
    }
    fn wait(
        request: &mut TimelineStorageRequest,
    ) -> Result<TimelineStorageResult, TimelineStorageError> {
        let until = Instant::now() + WAIT;
        loop {
            if let Poll::Ready(result) = request.poll() {
                return result;
            }
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
    }
    struct Gate {
        entered: mpsc::Receiver<()>,
        resume: Option<SyncSender<()>>,
    }
    impl Gate {
        fn arrived(&self) {
            self.entered.recv_timeout(WAIT).unwrap();
        }
        fn release(&mut self) {
            self.resume.take().unwrap().send(()).unwrap();
        }
    }
    impl Drop for Gate {
        fn drop(&mut self) {
            if let Some(sender) = self.resume.take() {
                let _ = sender.send(());
            }
        }
    }
    fn service(project: &Project, point: Checkpoint) -> (TimelineStorageService, Gate) {
        let (entered_tx, entered) = mpsc::sync_channel(1);
        let (resume, resume_rx) = mpsc::sync_channel(1);
        let mut service = TimelineStorageService::new(
            project.path(),
            TimelineStorageLimits {
                max_requests: 4,
                ..Default::default()
            },
        )
        .unwrap();
        service.hook = Some(Arc::new(TestHook {
            point,
            entered: entered_tx,
            resume: Mutex::new(resume_rx),
        }));
        (
            service,
            Gate {
                entered,
                resume: Some(resume),
            },
        )
    }
    fn submit(service: &TimelineStorageService, timeout: Duration) -> TimelineStorageRequest {
        service
            .save(
                bytes(240),
                token(),
                TimelineFileRevision::Present(bytes(120).into()),
                timeout,
            )
            .unwrap()
    }
    fn flush(service: &TimelineStorageService) {
        let mut load = service.load(WAIT).unwrap();
        assert!(matches!(
            wait(&mut load),
            Ok(TimelineStorageResult::Loaded(_))
        ));
    }

    #[test]
    fn real_staged_save_cancelled_before_rename_never_publishes_or_leaks_stage() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        assert_eq!(
            fs::read_dir(&project.0).unwrap().count(),
            3,
            "actual fsynced stage exists"
        );
        assert_eq!(request.cancel(), TimelineCancelOutcome::Cancelled);
        assert!(matches!(
            wait(&mut request),
            Err(TimelineStorageError::Cancelled)
        ));
        gate.release();
        flush(&service);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
        assert!(matches!(
            request.poll(),
            Poll::Ready(Err(TimelineStorageError::Consumed))
        ));
    }

    #[test]
    fn deadline_fences_a_real_fsynced_stage_and_discards_late_worker_result() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, Duration::from_secs(2));
        gate.arrived();
        thread::sleep(
            request
                .slot
                .deadline
                .saturating_duration_since(Instant::now())
                + Duration::from_millis(1),
        );
        assert!(matches!(
            wait(&mut request),
            Err(TimelineStorageError::TimedOut)
        ));
        gate.release();
        flush(&service);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
        assert!(matches!(
            request.poll(),
            Poll::Ready(Err(TimelineStorageError::Consumed))
        ));
    }

    #[test]
    fn post_fence_cancel_timeout_and_service_drop_preserve_actual_publication_receipt() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Publishing);
        let mut request = submit(&service, Duration::from_secs(2));
        gate.arrived();
        assert_eq!(request.cancel(), TimelineCancelOutcome::TooLate);
        drop(service);
        thread::sleep(
            request
                .slot
                .deadline
                .saturating_duration_since(Instant::now())
                + Duration::from_millis(1),
        );
        assert!(
            matches!(request.poll(), Poll::Pending),
            "deadline must not hide irrevocable publication"
        );
        gate.release();
        let TimelineStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
            panic!()
        };
        assert!(receipt.deadline_exceeded);
        assert_eq!(receipt.durability, TimelineSaveDurability::Synced);
        assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
    }

    #[test]
    fn dropping_service_and_queued_handle_fences_real_prepublication_work() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut first = submit(&service, WAIT);
        gate.arrived();
        let queued = service
            .save(
                bytes(480),
                token(),
                TimelineFileRevision::Present(bytes(120).into()),
                WAIT,
            )
            .unwrap();
        let shared = service.shared.clone();
        drop(queued);
        drop(service);
        assert!(matches!(
            wait(&mut first),
            Err(TimelineStorageError::Cancelled)
        ));
        drop(first);
        gate.release();
        let until = Instant::now() + WAIT;
        while shared.active.load(Ordering::Acquire) != 0 {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(fs::read(project.path()).unwrap(), bytes(120));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn external_change_after_stage_fsync_is_detected_by_final_exact_byte_recheck() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Staged);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        fs::write(project.path(), bytes(480)).unwrap();
        gate.release();
        assert!(matches!(
            wait(&mut request),
            Err(TimelineStorageError::Conflict)
        ));
        assert_eq!(fs::read(project.path()).unwrap(), bytes(480));
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn real_rename_failure_reports_publication_uncertainty_and_cleans_stage() {
        let project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Publishing);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        fs::remove_file(project.path()).unwrap();
        fs::create_dir(project.path()).unwrap();
        gate.release();
        assert!(matches!(
            wait(&mut request),
            Err(TimelineStorageError::PublicationUncertain { .. })
        ));
        assert!(project.path().is_dir());
        assert_eq!(fs::read_dir(&project.0).unwrap().count(), 2);
    }

    #[test]
    fn real_directory_sync_open_failure_still_reports_published_bytes_and_uncertain_durability() {
        let mut project = Project::new();
        fs::write(project.path(), bytes(120)).unwrap();
        let (service, mut gate) = service(&project, Checkpoint::Renamed);
        let mut request = submit(&service, WAIT);
        gate.arrived();
        let moved = project.0.with_extension("moved");
        fs::rename(&project.0, &moved).unwrap();
        project.0 = moved;
        gate.release();
        let TimelineStorageResult::Saved(receipt) = wait(&mut request).unwrap() else {
            panic!()
        };
        assert_eq!(
            receipt.durability,
            TimelineSaveDurability::Uncertain {
                kind: io::ErrorKind::NotFound
            }
        );
        assert_eq!(
            receipt.revision,
            TimelineFileRevision::Present(bytes(240).into())
        );
        assert_eq!(fs::read(project.path()).unwrap(), bytes(240));
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.shared.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Nonblocking handle. Dropping fences unfinished work without joining a
/// potentially blocked OS call; its slot remains reserved until the worker exits.
pub struct TimelineStorageRequest {
    slot: Arc<Slot>,
    consumed: bool,
}
impl TimelineStorageRequest {
    /// Service-local monotonic identity for rejecting obsolete load results.
    pub fn id(&self) -> u64 {
        self.slot.id
    }
    /// Takes output once. A pre-publication timeout immediately fences later
    /// output. Calling this after consuming output returns Consumed.
    pub fn poll(&mut self) -> Poll<Result<TimelineStorageResult, TimelineStorageError>> {
        if self.consumed {
            return Poll::Ready(Err(TimelineStorageError::Consumed));
        }
        let mut state = self.slot.state();
        self.slot.expire(&mut state);
        if let Some(output) = state.output.take() {
            self.consumed = true;
            Poll::Ready(output)
        } else {
            Poll::Pending
        }
    }
    /// Fences queued/running work or an unconsumed load candidate. Save
    /// publication is irrevocable: `TooLate` preserves its actual outcome.
    pub fn cancel(&self) -> TimelineCancelOutcome {
        let mut state = self.slot.state();
        if self.consumed
            || state.phase == Phase::Publishing
            || matches!(
                state.output,
                Some(Ok(TimelineStorageResult::Saved(_)))
                    | Some(Err(TimelineStorageError::PublicationUncertain { .. }))
            )
        {
            return TimelineCancelOutcome::TooLate;
        }
        state.abort = Some(TimelineStorageError::Cancelled);
        state.output = Some(Err(TimelineStorageError::Cancelled));
        TimelineCancelOutcome::Cancelled
    }
}
impl Drop for TimelineStorageRequest {
    fn drop(&mut self) {
        if !self.consumed {
            self.cancel();
        }
    }
}

enum Operation {
    Load,
    Save {
        bytes: Vec<u8>,
        snapshot_token: TimelineSnapshotToken,
        expected: TimelineFileRevision,
    },
}
struct Job {
    slot: Arc<Slot>,
    operation: Operation,
}

/// One bounded worker for one absolute filename in an existing trusted parent.
/// Construction/admission/poll do no filesystem I/O; all path checks, reads,
/// codec validation, stage writes and sync/rename happen on the worker.
pub struct TimelineStorageService {
    sender: SyncSender<Job>,
    shared: Arc<Shared>,
    limits: TimelineStorageLimits,
    #[cfg(test)]
    hook: Option<Arc<TestHook>>,
}
impl TimelineStorageService {
    /// Starts the worker. Parent must exist when used and remain trusted/stable.
    /// Dropping the service fences pre-publication work without blocking to join.
    pub fn new(
        path: impl Into<PathBuf>,
        limits: TimelineStorageLimits,
    ) -> Result<Self, TimelineStorageError> {
        let path = path.into();
        if !path.is_absolute()
            || path.file_name().is_none()
            || path.components().any(|c| c == Component::ParentDir)
            || limits.max_requests == 0
            || limits.max_requests > MAX_TIMELINE_STORAGE_REQUESTS
            || limits.max_bytes == 0
            || limits.max_bytes > MAX_TIMELINE_STORAGE_BYTES
        {
            return Err(TimelineStorageError::InvalidConfiguration);
        }
        let (sender, receiver) = mpsc::sync_channel::<Job>(limits.max_requests);
        let shared = Arc::new(Shared {
            active: AtomicUsize::new(0),
            next: AtomicU64::new(1),
            stopping: AtomicBool::new(false),
        });
        thread::Builder::new()
            .name("prism-timeline-storage".into())
            .spawn(move || {
                for job in receiver {
                    let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(&path, limits, &job.slot, job.operation)
                    }))
                    .unwrap_or_else(|_| {
                        if job.slot.state().phase == Phase::Publishing {
                            Err(TimelineStorageError::PublicationUncertain {
                                kind: io::ErrorKind::Other,
                            })
                        } else {
                            Err(TimelineStorageError::Unavailable)
                        }
                    });
                    job.slot.finish(output);
                }
            })
            .map_err(|e| os_error("spawn worker", e))?;
        Ok(Self {
            sender,
            shared,
            limits,
            #[cfg(test)]
            hook: None,
        })
    }
    /// Queues actual bounded read/decode without changing any host editor.
    pub fn load(&self, timeout: Duration) -> Result<TimelineStorageRequest, TimelineStorageError> {
        self.submit(Operation::Load, timeout)
    }
    /// Takes serialized bytes and token ownership. Worker validates the document
    /// before saving. Compare the receipt token before marking a live editor clean.
    pub fn save(
        &self,
        bytes: Vec<u8>,
        snapshot_token: TimelineSnapshotToken,
        expected: TimelineFileRevision,
        timeout: Duration,
    ) -> Result<TimelineStorageRequest, TimelineStorageError> {
        if bytes.len() > self.limits.max_bytes
            || bytes.capacity() > self.limits.max_bytes
            || matches!(&expected, TimelineFileRevision::Present(bytes) if bytes.len() > self.limits.max_bytes)
        {
            return Err(TimelineStorageError::ByteLimit);
        }
        self.submit(
            Operation::Save {
                bytes,
                snapshot_token,
                expected,
            },
            timeout,
        )
    }
    fn submit(
        &self,
        operation: Operation,
        timeout: Duration,
    ) -> Result<TimelineStorageRequest, TimelineStorageError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(TimelineStorageError::InvalidConfiguration)?;
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(TimelineStorageError::Unavailable);
        }
        self.shared
            .active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limits.max_requests).then_some(n + 1)
            })
            .map_err(|_| TimelineStorageError::Busy)?;
        let Ok(id) = self
            .shared
            .next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        else {
            self.shared.active.fetch_sub(1, Ordering::AcqRel);
            return Err(TimelineStorageError::Unavailable);
        };
        let slot = Arc::new(Slot {
            id,
            deadline,
            shared: self.shared.clone(),
            state: Mutex::new(State {
                phase: Phase::Working,
                abort: None,
                output: None,
            }),
            #[cfg(test)]
            hook: self.hook.clone(),
        });
        let request = TimelineStorageRequest {
            slot: slot.clone(),
            consumed: false,
        };
        match self.sender.try_send(Job { slot, operation }) {
            Ok(()) => Ok(request),
            Err(TrySendError::Full(_)) => Err(TimelineStorageError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(TimelineStorageError::Unavailable),
        }
    }
}
impl Drop for TimelineStorageService {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Release);
    }
}

fn os_error(operation: &'static str, error: io::Error) -> TimelineStorageError {
    TimelineStorageError::Io {
        operation,
        kind: error.kind(),
    }
}
fn regular_or_missing(path: &Path) -> Result<bool, TimelineStorageError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => Ok(true),
        Ok(_) => Err(TimelineStorageError::UnsafePath),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(os_error("inspect path", e)),
    }
}
fn read(
    path: &Path,
    max: usize,
    slot: &Slot,
) -> Result<TimelineFileRevision, TimelineStorageError> {
    slot.check()?;
    if !regular_or_missing(path)? {
        return Ok(TimelineFileRevision::Missing);
    }
    let mut file = File::open(path).map_err(|e| os_error("open", e))?;
    if file
        .metadata()
        .map_err(|e| os_error("inspect file", e))?
        .len()
        > max as u64
    {
        return Err(TimelineStorageError::ByteLimit);
    }
    let mut bytes = Vec::new();
    let mut block = [0; 16 * 1024];
    loop {
        slot.check()?;
        let limit = block.len().min(max.saturating_sub(bytes.len()) + 1);
        let count = file
            .read(&mut block[..limit])
            .map_err(|e| os_error("read", e))?;
        if count == 0 {
            break;
        }
        if count > max - bytes.len() {
            return Err(TimelineStorageError::ByteLimit);
        }
        bytes.extend_from_slice(&block[..count]);
    }
    slot.check()?;
    Ok(TimelineFileRevision::Present(bytes.into()))
}
fn lock(path: &Path) -> Result<File, TimelineStorageError> {
    regular_or_missing(path)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| os_error("open lock", e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(TimelineStorageError::Busy),
        Err(TryLockError::Error(e)) => Err(os_error("lock", e)),
    }
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn stage(path: &Path) -> Result<(Stage, File), TimelineStorageError> {
    for _ in 0..16 {
        let id = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
        let mut name = path.file_name().expect("validated filename").to_os_string();
        name.push(format!(".{}.{}.stage", std::process::id(), id));
        let path = path.with_file_name(name);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((Stage(path), file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(os_error("create stage", e)),
        }
    }
    Err(TimelineStorageError::Busy)
}
fn run(
    requested: &Path,
    limits: TimelineStorageLimits,
    slot: &Slot,
    operation: Operation,
) -> Result<TimelineStorageResult, TimelineStorageError> {
    slot.check()?;
    let parent = fs::canonicalize(requested.parent().expect("absolute filename"))
        .map_err(|e| os_error("resolve parent", e))?;
    let path = parent.join(requested.file_name().expect("validated filename"));
    let mut lock_name = path.file_name().expect("validated filename").to_os_string();
    lock_name.push(".prism-timeline.lock");
    let _lock = lock(&path.with_file_name(lock_name))?;
    let current = read(&path, limits.max_bytes, slot)?;
    match operation {
        Operation::Load => {
            let TimelineFileRevision::Present(bytes) = &current else {
                return Err(TimelineStorageError::Missing);
            };
            let document = TimelineDocument::decode(bytes, limits.document, limits.max_bytes)
                .map_err(TimelineStorageError::Codec)?;
            slot.check()?;
            Ok(TimelineStorageResult::Loaded(TimelineLoadCandidate {
                document,
                revision: current,
            }))
        }
        Operation::Save {
            bytes,
            snapshot_token,
            expected,
        } => {
            if current != expected {
                return Err(TimelineStorageError::Conflict);
            }
            TimelineDocument::decode(&bytes, limits.document, limits.max_bytes)
                .map_err(TimelineStorageError::Codec)?;
            slot.check()?;
            let (stage, mut file) = stage(&path)?;
            for block in bytes.chunks(16 * 1024) {
                slot.check()?;
                file.write_all(block)
                    .map_err(|e| os_error("write stage", e))?;
            }
            file.sync_all().map_err(|e| os_error("sync stage", e))?;
            drop(file);
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Staged);
            // Recheck after fsync to detect external changes during staging.
            if read(&path, limits.max_bytes, slot)? != expected {
                return Err(TimelineStorageError::Conflict);
            }
            slot.publish()?;
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Publishing);
            fs::rename(&stage.0, &path)
                .map_err(|e| TimelineStorageError::PublicationUncertain { kind: e.kind() })?;
            #[cfg(test)]
            slot.checkpoint(Checkpoint::Renamed);
            let durability = match File::open(&parent).and_then(|dir| dir.sync_all()) {
                Ok(()) => TimelineSaveDurability::Synced,
                Err(e) => TimelineSaveDurability::Uncertain { kind: e.kind() },
            };
            Ok(TimelineStorageResult::Saved(TimelineSaveReceipt {
                snapshot_token,
                revision: TimelineFileRevision::Present(bytes.into()),
                durability,
                deadline_exceeded: Instant::now() >= slot.deadline,
            }))
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
enum Checkpoint {
    Staged,
    Publishing,
    Renamed,
}
#[cfg(test)]
struct TestHook {
    point: Checkpoint,
    entered: SyncSender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}
#[cfg(test)]
impl Slot {
    fn checkpoint(&self, point: Checkpoint) {
        if let Some(hook) = &self.hook {
            if hook.point == point {
                hook.entered.send(()).unwrap();
                hook.resume.lock().unwrap().recv().unwrap();
            }
        }
    }
}
