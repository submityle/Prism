//! Bounded file reload preparation and exact, last-good publication fences.
//! Hosts feed native watcher events to [ReloadPipeline::notify]. Workers never
//! receive the live document or its undo history.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Weak,
    },
    time::{Duration, Instant},
};

/// Resource limits shared by capture and the built-in providers.
#[derive(Clone, Debug)]
pub struct ReloadLimits {
    /// Maximum explicitly registered sources.
    pub sources: usize,
    /// Maximum bytes per source and schema.
    pub file_bytes: usize,
    /// Maximum combined source bytes.
    pub total_bytes: usize,
    /// Running workers plus unconsumed results.
    pub in_flight: usize,
    /// Maximum Loom nodes.
    pub nodes: usize,
    /// Maximum Loom nesting.
    pub depth: usize,
    /// Maximum compiled shader text bytes.
    pub output_bytes: usize,
    /// Quiet coalescing period.
    pub debounce: Duration,
    /// Maximum continuous-event batching delay.
    pub max_delay: Duration,
    /// Cooperative deadline including capture.
    pub timeout: Duration,
}
impl Default for ReloadLimits {
    fn default() -> Self {
        Self {
            sources: 64,
            file_bytes: 1024 * 1024,
            total_bytes: 4 * 1024 * 1024,
            in_flight: 2,
            nodes: 4096,
            depth: 32,
            output_bytes: 4 * 1024 * 1024,
            debounce: Duration::from_millis(40),
            max_delay: Duration::from_millis(250),
            timeout: Duration::from_secs(10),
        }
    }
}
impl ReloadLimits {
    fn validate(&self) -> Result<(), ReloadError> {
        if self.sources == 0
            || self.sources > 4096
            || self.file_bytes == 0
            || self.file_bytes > 16 * 1024 * 1024
            || self.total_bytes < self.file_bytes
            || self.total_bytes > 64 * 1024 * 1024
            || self.in_flight == 0
            || self.in_flight > 32
            || self.nodes == 0
            || self.nodes > 100_000
            || self.depth == 0
            || self.depth > 64
            || self.output_bytes == 0
            || self.output_bytes > 16 * 1024 * 1024
            || self.timeout.is_zero()
            || self.max_delay < self.debounce
        {
            return Err(ReloadError::Budget("invalid limits"));
        }
        Ok(())
    }
}

/// Explicit failures. Rejected publication never calls the commit closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadError {
    /// Invalid limits or exhausted capacity.
    Budget(&'static str),
    /// Unregistered, nonrelative, or escaping source path.
    Path(PathBuf),
    /// Bounded file-access diagnostic.
    Io(String),
    /// Bounded syntax, schema or resource-construction diagnostic.
    Provider(String),
    /// Source, schema, or pipeline mismatch.
    Stale,
    /// Explicit cancellation.
    Cancelled,
    /// Cooperative deadline elapsed.
    TimedOut,
    /// A worker panicked or disconnected.
    WorkerFailed,
}
impl std::fmt::Display for ReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ReloadError {}
impl ReloadError {
    /// Makes a diagnostic with at most 4096 UTF-8 bytes.
    pub fn provider(message: impl std::fmt::Display) -> Self {
        Self::Provider(bounded(&message.to_string()))
    }
}
fn bounded(message: &str) -> String {
    let mut end = message.len().min(4096);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

/// Cancellation and deadline. Compiler calls cannot be preempted.
#[derive(Clone, Debug)]
pub struct ReloadCancellation {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}
impl ReloadCancellation {
    /// Checks cancellation and timeout, in that order.
    pub fn check(&self) -> Result<(), ReloadError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(ReloadError::Cancelled);
        }
        if Instant::now() >= self.deadline {
            return Err(ReloadError::TimedOut);
        }
        Ok(())
    }
    /// Requests cooperative cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Immutable byte-exact capture of all registered dependencies.
#[derive(Debug)]
pub struct ReloadSources {
    files: BTreeMap<PathBuf, Arc<[u8]>>,
}
impl ReloadSources {
    /// Looks up a registered relative source.
    pub fn get(&self, path: impl AsRef<Path>) -> Result<&[u8], ReloadError> {
        self.files
            .get(path.as_ref())
            .map(AsRef::as_ref)
            .ok_or_else(|| ReloadError::Path(path.as_ref().to_owned()))
    }
    /// UTF-8 source, with no lossy replacement.
    pub fn text(&self, path: impl AsRef<Path>) -> Result<&str, ReloadError> {
        std::str::from_utf8(self.get(path)?).map_err(ReloadError::provider)
    }
    /// Source names in deterministic order.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        self.files.keys().map(PathBuf::as_path)
    }
}

/// Executable off-thread provider. Custom providers must honor the limits.
pub trait ReloadProvider: Send + Sync + 'static {
    /// Prepared resource without references to a live document.
    type Artifact: Send + 'static;
    /// Prepares a complete candidate without mutating the live resource.
    fn prepare(
        &self,
        sources: &ReloadSources,
        schema: &[u8],
        cancellation: &ReloadCancellation,
        limits: &ReloadLimits,
    ) -> Result<Self::Artifact, ReloadError>;
}
struct Permit(Arc<AtomicUsize>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Candidate with private, unforgeable source/schema fences.
pub struct PreparedReload<A> {
    artifact: A,
    sources: ReloadSources,
    owner: Arc<()>,
    source_revision: Arc<()>,
    schema_revision: Arc<[u8]>,
    cancellation: ReloadCancellation,
    _permit: Arc<Permit>,
}
impl<A> PreparedReload<A> {
    /// Inspects the immutable candidate.
    pub fn artifact(&self) -> &A {
        &self.artifact
    }
    /// Inspects exact captured bytes.
    pub fn sources(&self) -> &ReloadSources {
        &self.sources
    }
}

/// Nonblocking request. On drop, its worker retains capacity until it exits.
pub struct ReloadRequest<A> {
    receiver: Option<mpsc::Receiver<Result<PreparedReload<A>, ReloadError>>>,
    cancellation: ReloadCancellation,
    permit: Option<Arc<Permit>>,
    finished: bool,
}
impl<A> ReloadRequest<A> {
    /// Requests cancellation without blocking.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
    /// Takes a result. None means pending or previously consumed.
    pub fn poll(&mut self) -> Result<Option<PreparedReload<A>>, ReloadError> {
        if self.finished {
            return Ok(None);
        }
        if let Err(error) = self.cancellation.check() {
            self.cancellation.cancel();
            self.finished = true;
            self.receiver.take();
            self.permit.take();
            return Err(error);
        }
        match self
            .receiver
            .as_ref()
            .expect("unfinished request")
            .try_recv()
        {
            Ok(result) => {
                self.finished = true;
                self.receiver.take();
                self.permit.take();
                result.map(Some)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.finished = true;
                self.receiver.take();
                self.permit.take();
                Err(ReloadError::WorkerFailed)
            }
        }
    }
}
impl<A> Drop for ReloadRequest<A> {
    fn drop(&mut self) {
        if !self.finished {
            self.cancellation.cancel();
        }
    }
}

/// Last-good resource bound to one pipeline.
pub struct ReloadSlot<L> {
    owner: Arc<()>,
    live: L,
}
impl<L> ReloadSlot<L> {
    /// Reads the last successfully published value.
    pub fn get(&self) -> &L {
        &self.live
    }
    /// Updates transient host state between publications.
    pub fn get_mut(&mut self) -> &mut L {
        &mut self.live
    }
}

/// One resource and its explicit, bounded dependency bundle.
pub struct ReloadPipeline<P: ReloadProvider> {
    root: PathBuf,
    paths: Arc<BTreeSet<PathBuf>>,
    pending: BTreeSet<PathBuf>,
    first_event: Instant,
    last_event: Instant,
    provider: Arc<P>,
    limits: ReloadLimits,
    owner: Arc<()>,
    revision: Arc<()>,
    schema: Arc<[u8]>,
    active: Arc<AtomicUsize>,
    cancellations: Vec<Weak<AtomicBool>>,
}
impl<P: ReloadProvider> ReloadPipeline<P> {
    /// Creates an initially dirty lane. Source paths must be unique and relative.
    pub fn new(
        root: impl AsRef<Path>,
        paths: impl IntoIterator<Item = PathBuf>,
        schema: Vec<u8>,
        provider: P,
        limits: ReloadLimits,
    ) -> Result<Self, ReloadError> {
        limits.validate()?;
        if schema.len() > limits.file_bytes {
            return Err(ReloadError::Budget("schema bytes"));
        }
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?;
        if !root.is_dir() {
            return Err(ReloadError::Path(root));
        }
        let mut registered = BTreeSet::new();
        for path in paths {
            if path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|p| !matches!(p, Component::Normal(_)))
                || !registered.insert(path.clone())
            {
                return Err(ReloadError::Path(path));
            }
            if registered.len() > limits.sources {
                return Err(ReloadError::Budget("source count"));
            }
        }
        if registered.is_empty() {
            return Err(ReloadError::Budget("empty source bundle"));
        }
        let now = Instant::now();
        Ok(Self {
            root,
            pending: registered.clone(),
            paths: Arc::new(registered),
            first_event: now,
            last_event: now,
            provider: Arc::new(provider),
            limits,
            owner: Arc::new(()),
            revision: Arc::new(()),
            schema: schema.into(),
            active: Arc::new(AtomicUsize::new(0)),
            cancellations: Vec::new(),
        })
    }
    /// Binds initial resources to this lane.
    pub fn slot<L>(&self, initial: L) -> ReloadSlot<L> {
        ReloadSlot {
            owner: self.owner.clone(),
            live: initial,
        }
    }
    fn invalidate(&mut self) {
        self.revision = Arc::new(());
        for flag in self.cancellations.drain(..).filter_map(|w| w.upgrade()) {
            flag.store(true, Ordering::Release);
        }
    }
    /// Coalesces an event and immediately invalidates old work, before debounce.
    /// Removal is the same event: capture fails until the source is restored.
    pub fn notify(&mut self, path: impl AsRef<Path>, now: Instant) -> Result<(), ReloadError> {
        let path = path.as_ref();
        if !self.paths.contains(path) {
            return Err(ReloadError::Path(path.to_owned()));
        }
        self.invalidate();
        if self.pending.is_empty() {
            self.first_event = now;
        }
        self.last_event = now;
        self.pending.insert(path.to_owned());
        Ok(())
    }
    /// Replaces exact schema bytes. Identical bytes are a no-op; ABA is rejected.
    pub fn set_schema(&mut self, schema: Vec<u8>, now: Instant) -> Result<bool, ReloadError> {
        if schema.len() > self.limits.file_bytes {
            return Err(ReloadError::Budget("schema bytes"));
        }
        if self.schema.as_ref() == schema.as_slice() {
            return Ok(false);
        }
        self.invalidate();
        self.schema = schema.into();
        self.pending = self.paths.as_ref().clone();
        self.first_event = now;
        self.last_event = now;
        Ok(true)
    }
    /// Coalesced path count, bounded by registered sources.
    pub fn pending_paths(&self) -> usize {
        self.pending.len()
    }
    /// Workers and held results currently consuming capacity.
    pub fn in_flight(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
    /// Starts a due batch. Force bypasses debounce. Capacity refusal retains dirty paths.
    pub fn start(
        &mut self,
        now: Instant,
        force: bool,
    ) -> Result<Option<ReloadRequest<P::Artifact>>, ReloadError> {
        if self.pending.is_empty()
            || (!force
                && now.saturating_duration_since(self.last_event) < self.limits.debounce
                && now.saturating_duration_since(self.first_event) < self.limits.max_delay)
        {
            return Ok(None);
        }
        self.active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limits.in_flight).then_some(n + 1)
            })
            .map_err(|_| ReloadError::Budget("in-flight requests/results"))?;
        let permit = Arc::new(Permit(self.active.clone()));
        let cancellation = ReloadCancellation {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: now
                .checked_add(self.limits.timeout)
                .ok_or(ReloadError::Budget("deadline"))?,
        };
        let worker_cancel = cancellation.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let root = self.root.clone();
        let paths = self.paths.clone();
        let provider = self.provider.clone();
        let limits = self.limits.clone();
        let owner = self.owner.clone();
        let source_revision = self.revision.clone();
        let schema_revision = self.schema.clone();
        let worker_permit = permit.clone();
        std::thread::Builder::new()
            .name("prism-editor-reload".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let sources = capture(&root, &paths, &limits, &worker_cancel)?;
                    let artifact =
                        provider.prepare(&sources, &schema_revision, &worker_cancel, &limits)?;
                    worker_cancel.check()?;
                    Ok(PreparedReload {
                        artifact,
                        sources,
                        owner,
                        source_revision,
                        schema_revision,
                        cancellation: worker_cancel.clone(),
                        _permit: worker_permit.clone(),
                    })
                }))
                .unwrap_or(Err(ReloadError::WorkerFailed));
                // Permit covers errors too, and remains held across send and cleanup.
                let _ = sender.send(result);
                drop(worker_permit);
            })
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?;
        self.pending.clear();
        self.cancellations.retain(|w| w.strong_count() != 0);
        self.cancellations
            .push(Arc::downgrade(&cancellation.cancelled));
        Ok(Some(ReloadRequest {
            receiver: Some(receiver),
            cancellation,
            permit: Some(permit),
            finished: false,
        }))
    }
    fn verify<L>(
        &self,
        prepared: &PreparedReload<P::Artifact>,
        slot: &ReloadSlot<L>,
    ) -> Result<(), ReloadError> {
        if !Arc::ptr_eq(&self.owner, &slot.owner)
            || !Arc::ptr_eq(&self.owner, &prepared.owner)
            || !Arc::ptr_eq(&self.revision, &prepared.source_revision)
            || !Arc::ptr_eq(&self.schema, &prepared.schema_revision)
        {
            return Err(ReloadError::Stale);
        }
        prepared.cancellation.check()?;
        let current = capture(
            &self.root,
            &self.paths,
            &self.limits,
            &prepared.cancellation,
        )?;
        if current.files != prepared.sources.files {
            return Err(ReloadError::Stale);
        }
        Ok(())
    }
    /// Publishes after two exact checks around fallible resource construction.
    /// Build must only construct a candidate; commit must be infallible. No live
    /// resource is passed to build. The second capture is the filesystem
    /// linearization point; writers after it must deliver another watcher event.
    pub fn publish<L, C, R>(
        &self,
        prepared: PreparedReload<P::Artifact>,
        slot: &mut ReloadSlot<L>,
        build: impl FnOnce(&P::Artifact) -> Result<C, ReloadError>,
        commit: impl FnOnce(&mut L, P::Artifact, C) -> R,
    ) -> Result<R, ReloadError> {
        self.verify(&prepared, slot)?;
        let candidate = build(&prepared.artifact)?;
        self.verify(&prepared, slot)?;
        Ok(commit(&mut slot.live, prepared.artifact, candidate))
    }
}
impl<P: ReloadProvider> Drop for ReloadPipeline<P> {
    fn drop(&mut self) {
        self.invalidate();
    }
}

fn capture(
    root: &Path,
    paths: &BTreeSet<PathBuf>,
    limits: &ReloadLimits,
    cancellation: &ReloadCancellation,
) -> Result<ReloadSources, ReloadError> {
    let mut files = BTreeMap::new();
    let mut total = 0usize;
    for path in paths {
        cancellation.check()?;
        let resolved = root
            .join(path)
            .canonicalize()
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?;
        if !resolved.starts_with(root) {
            return Err(ReloadError::Path(path.clone()));
        }
        if !resolved
            .metadata()
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?
            .is_file()
        {
            return Err(ReloadError::Path(path.clone()));
        }
        let file = File::open(&resolved).map_err(|e| ReloadError::Io(bounded(&e.to_string())))?;
        if !file
            .metadata()
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?
            .is_file()
        {
            return Err(ReloadError::Path(path.clone()));
        }
        let budget = limits
            .file_bytes
            .min(limits.total_bytes.saturating_sub(total));
        let mut bytes = Vec::new();
        file.take(budget as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| ReloadError::Io(bounded(&e.to_string())))?;
        if bytes.len() > budget {
            return Err(ReloadError::Budget("source bytes"));
        }
        total += bytes.len();
        files.insert(path.clone(), bytes.into());
    }
    cancellation.check()?;
    Ok(ReloadSources { files })
}
