//! Change batches and the platform file-watcher boundary.
//!
//! The loader consumes stable, source-qualified [`AssetChange`] values rather
//! than platform-specific filesystem events. [`ChangeBatchBuilder`] is the
//! deterministic part used by tests, memory sources, and editor integrations;
//! the optional [`FileWatcher`] adapter translates `notify-debouncer-full`
//! events into the same contract.

use crate::AssetPath;
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
