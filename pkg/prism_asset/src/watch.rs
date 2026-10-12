//! Change batches and the platform file-watcher boundary.
//!
//! The loader consumes stable, source-qualified [`AssetChange`] values rather
//! than platform-specific filesystem events. [`ChangeBatchBuilder`] is the
//! deterministic part used by tests, memory sources, and editor integrations;
//! the optional [`FileWatcher`] adapter translates `notify-debouncer-full`
//! events into the same contract.

use crate::AssetPath;
#[cfg(feature = "file_watcher")]
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// The semantic effect of a source change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssetChangeKind {
    /// A path became available.
    Created,
    /// Existing bytes or metadata changed.
    Modified,
    /// A path disappeared.
    Removed,
    /// A path moved within the same source.
    Renamed {
        /// Path before the rename.
        from: AssetPath,
        /// Path after the rename.
        to: AssetPath,
    },
}

/// A source-qualified, normalized asset change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetChange {
    /// VFS source/mount name. Empty means the default source.
    pub source: String,
    /// The primary path affected by this change. For rename this is `to`.
    pub path: AssetPath,
    /// Semantic change kind.
    pub kind: AssetChangeKind,
}

/// A deterministic set of changes emitted after one debounce window.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChangeBatch {
    changes: Vec<AssetChange>,
}

impl ChangeBatch {
    /// Creates a batch from already canonical changes.
    pub fn new(changes: impl IntoIterator<Item = AssetChange>) -> Self {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        for change in changes {
            builder.push(change);
        }
        builder.flush()
    }

    /// Returns changes in stable source/path order.
    pub fn changes(&self) -> &[AssetChange] {
        &self.changes
    }

    /// Whether this batch has no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

/// In-memory debounce/coalescing state.
///
/// The builder does not sleep. Callers feed events and call [`flush`](Self::flush)
/// at their chosen clock boundary; this keeps behavior deterministic and makes
/// it suitable for tests and non-filesystem sources. Consecutive events for a
/// path collapse according to source semantics: create+modify becomes create,
/// modify+remove becomes remove, remove+create becomes modified, and a rename
/// preserves its origin while later modifications target the destination.
#[derive(Clone, Debug)]
pub struct ChangeBatchBuilder {
    window: Duration,
    pending: BTreeMap<(String, AssetPath), PendingChange>,
}

#[derive(Clone, Debug)]
struct PendingChange {
    first: AssetPath,
    current: AssetPath,
    kind: PendingKind,
}

#[derive(Clone, Debug)]
enum PendingKind {
    Created,
    Modified,
    Removed,
    Renamed,
    /// The path was created and removed within one debounce window.
    Discarded,
}

impl ChangeBatchBuilder {
    /// Creates an empty builder. `window` is metadata for integrations that
    /// schedule flushes; this type itself never blocks.
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            pending: BTreeMap::new(),
        }
    }

    /// Configured debounce window.
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Adds one semantic event to the current window.
    pub fn push(&mut self, change: AssetChange) {
        let key = (change.source.clone(), change.path.clone());
        match change.kind {
            AssetChangeKind::Created => {
                self.pending
                    .entry(key)
                    .and_modify(|pending| {
                        if !matches!(pending.kind, PendingKind::Created) {
                            pending.kind = PendingKind::Modified;
                        }
                        pending.current = change.path.clone();
                    })
                    .or_insert(PendingChange {
                        first: change.path.clone(),
                        current: change.path,
                        kind: PendingKind::Created,
                    });
            }
            AssetChangeKind::Modified => {
                self.pending
                    .entry(key)
                    .and_modify(|pending| {
                        if matches!(pending.kind, PendingKind::Removed) {
                            pending.kind = PendingKind::Created;
                        } else if !matches!(
                            pending.kind,
                            PendingKind::Created | PendingKind::Renamed
                        ) {
                            pending.kind = PendingKind::Modified;
                        }
                        pending.current = change.path.clone();
                    })
                    .or_insert(PendingChange {
                        first: change.path.clone(),
                        current: change.path,
                        kind: PendingKind::Modified,
                    });
            }
            AssetChangeKind::Removed => {
                self.pending
                    .entry(key)
                    .and_modify(|pending| {
                        if matches!(pending.kind, PendingKind::Created) {
                            // Create followed by remove has no externally
                            // visible transition in this batch.
                            pending.kind = PendingKind::Discarded;
                        } else {
                            pending.kind = PendingKind::Removed;
                        }
                        pending.current = change.path.clone();
                    })
                    .or_insert(PendingChange {
                        first: change.path.clone(),
                        current: change.path,
                        kind: PendingKind::Removed,
                    });
            }
            AssetChangeKind::Renamed { from, to } => {
                self.push_rename(change.source, from, to);
            }
        }
    }

    fn push_rename(&mut self, source: String, from: AssetPath, to: AssetPath) {
        let from_key = (source.clone(), from.clone());
        let to_key = (source.clone(), to.clone());
        self.pending.remove(&to_key);
        if let Some(mut pending) = self.pending.remove(&from_key) {
            pending.current = to.clone();
            pending.kind = PendingKind::Renamed;
            self.pending.insert(to_key, pending);
        } else {
            self.pending.insert(
                to_key,
                PendingChange {
                    first: from,
                    current: to,
                    kind: PendingKind::Renamed,
                },
            );
        }
    }

    /// Emits and clears the current canonical batch.
    pub fn flush(&mut self) -> ChangeBatch {
        let mut changes = Vec::with_capacity(self.pending.len());
        for ((source, _), pending) in core::mem::take(&mut self.pending) {
            if matches!(pending.kind, PendingKind::Discarded) {
                continue;
            }
            let first = pending.first.clone();
            let current = pending.current.clone();
            let kind = match pending.kind {
                PendingKind::Created => AssetChangeKind::Created,
                PendingKind::Modified => AssetChangeKind::Modified,
                PendingKind::Removed => AssetChangeKind::Removed,
                PendingKind::Renamed => AssetChangeKind::Renamed {
                    from: first,
                    to: current.clone(),
                },
                PendingKind::Discarded => unreachable!(),
            };
            changes.push(AssetChange {
                source,
                path: current,
                kind,
            });
        }
        ChangeBatch { changes }
    }
}

#[cfg(feature = "file_watcher")]
mod platform {
    use super::{AssetChange, AssetChangeKind, ChangeBatch, ChangeBatchBuilder};
    use crate::AssetPath;
    use alloc::string::{String, ToString};
    use notify_debouncer_full::{
        new_debouncer, notify::event::ModifyKind, notify::event::RenameMode, notify::EventKind,
        notify::RecommendedWatcher, notify::RecursiveMode, DebounceEventResult, Debouncer,
        RecommendedCache,
    };
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{self, Receiver};
    use std::time::Duration;

    /// A running platform watcher. Dropping it stops the watcher thread.
    pub struct FileWatcher {
        debouncer: Debouncer<RecommendedWatcher, RecommendedCache>,
        root: PathBuf,
        source: String,
        receiver: Receiver<DebounceEventResult>,
    }

    impl FileWatcher {
        /// Starts watching `root` recursively and emits batches after `window`.
        pub fn new(
            root: impl Into<PathBuf>,
            source: impl Into<String>,
            window: Duration,
        ) -> Result<Self, notify_debouncer_full::notify::Error> {
            let root = root.into();
            let source = source.into();
            let (sender, receiver) = mpsc::channel();
            let mut debouncer = new_debouncer(window, None, sender)?;
            debouncer.watch(&root, RecursiveMode::Recursive)?;
            Ok(Self {
                debouncer,
                root,
                source,
                receiver,
            })
        }

        /// Receives one platform-debounced batch or watcher error, if available.
        pub fn try_recv(&self) -> Option<Result<ChangeBatch, String>> {
            let result = self.receiver.try_recv().ok()?;
            Some(convert(result, &self.root, &self.source))
        }

        /// Keeps the watcher alive while making its ownership explicit.
        pub fn root(&self) -> &Path {
            &self.root
        }

        /// Returns the configured source name.
        pub fn source(&self) -> &str {
            &self.source
        }

        /// Stops the debouncer thread before returning.
        pub fn stop(self) {
            self.debouncer.stop();
        }
    }

    fn convert(
        result: DebounceEventResult,
        root: &Path,
        source: &str,
    ) -> Result<ChangeBatch, String> {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        match result {
            Ok(events) => {
                for event in events {
                    let Some(path) = event.paths.last() else {
                        continue;
                    };
                    let Ok(relative) = path.strip_prefix(root) else {
                        continue;
                    };
                    let relative = relative.to_string_lossy().replace('\\', "/");
                    let asset_path = AssetPath::new(relative);
                    let kind = match event.kind {
                        EventKind::Create(_) => AssetChangeKind::Created,
                        EventKind::Remove(_) => AssetChangeKind::Removed,
                        EventKind::Modify(ModifyKind::Name(RenameMode::Both))
                            if event.paths.len() >= 2 =>
                        {
                            let from = event.paths[0]
                                .strip_prefix(root)
                                .ok()
                                .map(|p| AssetPath::new(p.to_string_lossy().replace('\\', "/")));
                            from.map_or(AssetChangeKind::Modified, |from| {
                                AssetChangeKind::Renamed {
                                    from,
                                    to: asset_path.clone(),
                                }
                            })
                        }
                        _ => AssetChangeKind::Modified,
                    };
                    builder.push(AssetChange {
                        source: source.to_string(),
                        path: asset_path,
                        kind,
                    });
                }
            }
            Err(errors) => {
                return Err(errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "))
            }
        }
        Ok(builder.flush())
    }
}

#[cfg(feature = "file_watcher")]
pub use platform::FileWatcher;

/// Failure while starting the managed background watcher.
#[cfg(feature = "file_watcher")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WatchStartError {
    /// The platform watcher could not be initialized or attached to its root.
    Platform(String),
    /// The event-consumer thread could not be created.
    Thread(String),
}

#[cfg(feature = "file_watcher")]
impl core::fmt::Display for WatchStartError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Platform(error) => write!(f, "asset file watcher failed to start: {error}"),
            Self::Thread(error) => write!(f, "asset watcher consumer failed to start: {error}"),
        }
    }
}

#[cfg(feature = "file_watcher")]
impl std::error::Error for WatchStartError {}

/// A managed watcher which consumes platform batches and applies them to an
/// [`AssetServer`](crate::AssetServer) on a dedicated thread.
///
/// The platform debouncer remains owned by the consumer thread. Dropping this
/// handle sends a stop request, joins the consumer, and then stops the
/// debouncer, so a watcher cannot outlive the server it is meant to update.
#[cfg(feature = "file_watcher")]
pub struct AssetWatchHandle {
    stop: Option<std::sync::mpsc::Sender<()>>,
    join: Option<std::thread::JoinHandle<()>>,
    errors: alloc::sync::Arc<std::sync::Mutex<Vec<String>>>,
    batches_applied: alloc::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[cfg(feature = "file_watcher")]
impl AssetWatchHandle {
    /// Starts a platform watcher and its managed batch-consumer thread.
    pub(crate) fn spawn(
        server: alloc::sync::Arc<crate::AssetServer>,
        root: impl Into<std::path::PathBuf>,
        source: impl Into<String>,
        window: Duration,
    ) -> Result<Self, WatchStartError> {
        let watcher = FileWatcher::new(root, source, window)
            .map_err(|error| WatchStartError::Platform(error.to_string()))?;
        let (stop_tx, stop_rx) = std::sync::mpsc::channel();
        let errors = alloc::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batches_applied = alloc::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let thread_errors = alloc::sync::Arc::clone(&errors);
        let thread_batches = alloc::sync::Arc::clone(&batches_applied);
        let join = std::thread::Builder::new()
            .name("prism-asset-watcher".to_string())
            .spawn(move || run_consumer(watcher, server, stop_rx, thread_errors, thread_batches))
            .map_err(|error| WatchStartError::Thread(error.to_string()))?;
        Ok(Self {
            stop: Some(stop_tx),
            join: Some(join),
            errors,
            batches_applied,
        })
    }

    /// Requests shutdown and waits until the platform watcher is stopped.
    pub fn stop(mut self) {
        self.stop_and_join();
    }

    /// Returns and clears watcher errors observed by the background consumer.
    ///
    /// An individual bad platform event does not terminate the consumer; the
    /// error is retained here for diagnostics and the next batch can still be
    /// applied.
    pub fn drain_errors(&self) -> Vec<String> {
        let mut errors = match self.errors.lock() {
            Ok(errors) => errors,
            Err(poisoned) => poisoned.into_inner(),
        };
        core::mem::take(&mut *errors)
    }

    /// Number of non-empty batches successfully handed to the server.
    #[must_use]
    pub fn batches_applied(&self) -> u64 {
        self.batches_applied
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether the managed consumer thread has exited.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.join
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }

    #[cfg(test)]
    fn spawn_test(
        server: alloc::sync::Arc<crate::AssetServer>,
        receiver: std::sync::mpsc::Receiver<Result<ChangeBatch, String>>,
    ) -> Self {
        let (stop_tx, stop_rx) = std::sync::mpsc::channel();
        let errors = alloc::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batches_applied = alloc::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let thread_errors = alloc::sync::Arc::clone(&errors);
        let thread_batches = alloc::sync::Arc::clone(&batches_applied);
        let join = std::thread::Builder::new()
            .name("prism-asset-test-watcher".to_string())
            .spawn(move || {
                run_consumer_loop(server, stop_rx, thread_errors, thread_batches, || {
                    receiver.try_recv().ok()
                });
            })
            .expect("test watcher consumer should start");
        Self {
            stop: Some(stop_tx),
            join: Some(join),
            errors,
            batches_applied,
        }
    }

    fn stop_and_join(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(feature = "file_watcher")]
impl Drop for AssetWatchHandle {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(feature = "file_watcher")]
fn run_consumer(
    watcher: FileWatcher,
    server: alloc::sync::Arc<crate::AssetServer>,
    stop_rx: std::sync::mpsc::Receiver<()>,
    errors: alloc::sync::Arc<std::sync::Mutex<Vec<String>>>,
    batches_applied: alloc::sync::Arc<std::sync::atomic::AtomicU64>,
) {
    run_consumer_loop(server, stop_rx, errors, batches_applied, || {
        watcher.try_recv()
    });
    watcher.stop();
}

#[cfg(feature = "file_watcher")]
fn run_consumer_loop(
    server: alloc::sync::Arc<crate::AssetServer>,
    stop_rx: std::sync::mpsc::Receiver<()>,
    errors: alloc::sync::Arc<std::sync::Mutex<Vec<String>>>,
    batches_applied: alloc::sync::Arc<std::sync::atomic::AtomicU64>,
    mut next_batch: impl FnMut() -> Option<Result<ChangeBatch, String>>,
) {
    const POLL_INTERVAL: Duration = Duration::from_millis(50);
    loop {
        match stop_rx.recv_timeout(POLL_INTERVAL) {
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }

        while !stop_requested(&stop_rx) {
            let Some(result) = next_batch() else {
                break;
            };
            match result {
                Ok(batch) if !batch.is_empty() => {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        server.apply_change_batch(&batch);
                    }));
                    if let Err(payload) = result {
                        record_watch_error(&errors, panic_detail(payload));
                    } else {
                        batches_applied.fetch_add(1, std::sync::atomic::Ordering::Release);
                    }
                }
                Ok(_) => {}
                Err(error) => record_watch_error(&errors, error),
            }
        }
    }
}

#[cfg(feature = "file_watcher")]
fn stop_requested(stop_rx: &std::sync::mpsc::Receiver<()>) -> bool {
    match stop_rx.try_recv() {
        Ok(()) | Err(std::sync::mpsc::TryRecvError::Disconnected) => true,
        Err(std::sync::mpsc::TryRecvError::Empty) => false,
    }
}

#[cfg(feature = "file_watcher")]
fn record_watch_error(errors: &std::sync::Mutex<Vec<String>>, error: String) {
    match errors.lock() {
        Ok(mut errors) => errors.push(error),
        Err(poisoned) => poisoned.into_inner().push(error),
    }
}

#[cfg(feature = "file_watcher")]
fn panic_detail(payload: Box<dyn core::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        format!("watcher batch application panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("watcher batch application panicked: {message}")
    } else {
        "watcher batch application panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(path: &str, kind: AssetChangeKind) -> AssetChange {
        AssetChange {
            source: "game".to_string(),
            path: AssetPath::new(path),
            kind,
        }
    }

    #[test]
    fn repeated_modify_is_one_stable_change() {
        let mut builder = ChangeBatchBuilder::new(Duration::from_millis(20));
        builder.push(change("textures/a.bin", AssetChangeKind::Modified));
        builder.push(change("textures/a.bin", AssetChangeKind::Modified));
        let batch = builder.flush();
        assert_eq!(
            batch.changes(),
            &[change("textures/a.bin", AssetChangeKind::Modified)]
        );
    }

    #[test]
    fn create_then_modify_stays_created() {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        builder.push(change("a", AssetChangeKind::Created));
        builder.push(change("a", AssetChangeKind::Modified));
        assert_eq!(builder.flush().changes()[0].kind, AssetChangeKind::Created);
    }

    #[test]
    fn rename_keeps_origin_and_destination() {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        builder.push(change("old", AssetChangeKind::Modified));
        builder.push(AssetChange {
            source: "game".to_string(),
            path: AssetPath::new("new"),
            kind: AssetChangeKind::Renamed {
                from: AssetPath::new("old"),
                to: AssetPath::new("new"),
            },
        });
        assert_eq!(
            builder.flush().changes()[0].kind,
            AssetChangeKind::Renamed {
                from: AssetPath::new("old"),
                to: AssetPath::new("new")
            }
        );
    }

    #[test]
    fn rename_followed_by_destination_modify_keeps_rename() {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        builder.push(AssetChange {
            source: "game".to_string(),
            path: AssetPath::new("new"),
            kind: AssetChangeKind::Renamed {
                from: AssetPath::new("old"),
                to: AssetPath::new("new"),
            },
        });
        builder.push(change("new", AssetChangeKind::Modified));
        assert_eq!(
            builder.flush().changes(),
            &[AssetChange {
                source: "game".to_string(),
                path: AssetPath::new("new"),
                kind: AssetChangeKind::Renamed {
                    from: AssetPath::new("old"),
                    to: AssetPath::new("new"),
                },
            }]
        );
    }

    #[test]
    fn sources_do_not_coalesce() {
        let mut builder = ChangeBatchBuilder::new(Duration::ZERO);
        builder.push(change("a", AssetChangeKind::Modified));
        builder.push(AssetChange {
            source: "dlc".to_string(),
            path: AssetPath::new("a"),
            kind: AssetChangeKind::Modified,
        });
        assert_eq!(builder.flush().changes().len(), 2);
    }
}

#[cfg(all(test, feature = "file_watcher"))]
mod background_tests {
    use super::AssetWatchHandle;
    use crate::{
        Asset, AssetChange, AssetChangeKind, AssetLoader, AssetPath, AssetServer, ChangeBatch,
        FsSource, LoadContext, LoadError,
    };
    use alloc::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[derive(Debug, PartialEq, Eq)]
    struct WatchedValue(u8);

    impl Asset for WatchedValue {
        const TYPE_NAME: &'static str = "prism_asset::watch::background_tests::WatchedValue";
    }

    struct WatchedValueLoader;

    impl AssetLoader for WatchedValueLoader {
        type Asset = WatchedValue;

        fn extensions(&self) -> &'static [&'static str] {
            &["watched"]
        }

        fn load(&self, bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Self::Asset, LoadError> {
            bytes
                .first()
                .copied()
                .map(WatchedValue)
                .ok_or_else(|| LoadError::Malformed("empty watched asset".into()))
        }
    }

    fn unique_temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "prism_asset_background_watch_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn managed_watcher_reloads_resident_asset_and_stops_cleanly() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("hero.watched");
        std::fs::write(&path, [1_u8]).unwrap();

        let server = Arc::new(AssetServer::new());
        server.register_loader(WatchedValueLoader);
        server.mount("game", Arc::new(FsSource::new(&root)), 0);
        let handle = server.load::<WatchedValue>(AssetPath::parse("game://hero.watched"));
        assert_eq!(server.with_asset(handle.id(), |value| value.0), Some(1));

        let platform_watcher = server
            .watch(&root, "game", Duration::from_millis(20))
            .expect("managed platform watcher should start");
        assert!(!platform_watcher.is_finished());
        platform_watcher.stop();

        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let watcher: AssetWatchHandle =
            AssetWatchHandle::spawn_test(Arc::clone(&server), events_rx);
        std::fs::write(&path, [2_u8]).unwrap();
        events_tx
            .send(Ok(ChangeBatch::new([AssetChange {
                source: "game".to_string(),
                path: AssetPath::new("hero.watched"),
                kind: AssetChangeKind::Modified,
            }])))
            .unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < deadline
            && server.with_asset(handle.id(), |value| value.0) != Some(2)
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            server.with_asset(handle.id(), |value| value.0),
            Some(2),
            "batches_applied={}, watcher_errors={:?}",
            watcher.batches_applied(),
            watcher.drain_errors()
        );
        assert!(watcher.batches_applied() >= 1);
        assert!(watcher.drain_errors().is_empty());

        watcher.stop();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn consumer_records_event_error_and_applies_later_batch() {
        let server = Arc::new(AssetServer::new());
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let watcher = AssetWatchHandle::spawn_test(Arc::clone(&server), events_rx);
        events_tx
            .send(Err("synthetic watcher error".to_string()))
            .unwrap();
        events_tx
            .send(Ok(ChangeBatch::new([AssetChange {
                source: "game".to_string(),
                path: AssetPath::new("unknown.watched"),
                kind: AssetChangeKind::Modified,
            }])))
            .unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while std::time::Instant::now() < deadline && watcher.batches_applied() == 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(watcher.batches_applied(), 1);
        assert_eq!(watcher.drain_errors(), vec!["synthetic watcher error"]);
        watcher.stop();
    }
}
