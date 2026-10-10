//! Filesystem-driven hot-reload integration (design §9 / roadmap M3).
//!
//! The rest of this crate is a pure, deterministic *core*: it takes a new
//! element tree or stylesheet and reconciles it against the mounted one. It
//! deliberately does not know where that new content came from. This module is
//! the thin, feature-gated bridge that closes the "`🔜` filesystem watching"
//! gap in the Loom roadmap: it turns raw OS change notifications for `.loom`
//! view files and `.loom.style` stylesheets into a small, de-duplicated batch
//! of [`ReloadRequest`]s that a host can drain each frame, re-parse, and feed
//! into [`crate::HotReloader`] / [`crate::diff_classes`].
//!
//! It is split into two halves so the interesting logic stays testable without
//! touching a real disk:
//!
//! * [`classify`] and [`Coalescer`] are **pure**. Classification maps a path to
//!   a [`ReloadTarget`] by extension; the coalescer folds a burst of raw events
//!   into one request per path. Editors save atomically (write a temporary,
//!   then rename over the target), so a single save shows up as a
//!   remove/create/modify flurry — collapsing that to a single
//!   last-write-wins request is what keeps a reload from firing three times.
//! * [`LoomWatcher`] wraps [`prism_platform`]'s [`Watcher`] (native `kqueue` on
//!   macOS/BSD, portable `stat`-polling elsewhere) and drives the coalescer
//!   from real events.
//!
//! The whole module requires the `watch` feature (which implies `std` and pulls
//! in [`prism_platform`]); the default build of this crate stays dependency-free
//! and `no_std`-friendly.

use core::time::Duration;
use std::path::{Path, PathBuf};

use alloc::vec::Vec;

use prism_platform::{Event, EventKind, WatchBackend, WatchError, Watcher};

/// Which Loom artifact a changed file maps to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReloadTarget {
    /// A `.loom` view file: re-parse into an [`prism_ui::Element`] tree and feed
    /// it to [`crate::HotReloader::reload`].
    View,
    /// A `.loom.style` stylesheet: re-resolve class styles and hot-swap exactly
    /// the changed properties via [`crate::diff_classes`].
    Style,
}

/// Whether a changed file still exists (its content should be re-read) or was
/// removed (its contribution should be torn down).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChangeKind {
    /// The file was created, modified, renamed into place, or otherwise changed
    /// while still present: re-read it.
    Upserted,
    /// The file was deleted: drop whatever it contributed.
    Removed,
}

impl ChangeKind {
    /// Maps a platform [`EventKind`] to a reload disposition.
    ///
    /// Only an explicit [`EventKind::Removed`] is treated as a removal; every
    /// other kind (including the backend's unclassified [`EventKind::Other`])
    /// is treated conservatively as an upsert so a reload is attempted and the
    /// host re-reads the current on-disk state.
    #[must_use]
    pub fn from_event_kind(kind: EventKind) -> Self {
        match kind {
            EventKind::Removed => ChangeKind::Removed,
            EventKind::Created | EventKind::Modified | EventKind::Renamed | EventKind::Other => {
                ChangeKind::Upserted
            }
        }
    }
}

/// A single coalesced reload request for one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReloadRequest {
    /// The affected file.
    pub path: PathBuf,
    /// Which Loom artifact it maps to.
    pub target: ReloadTarget,
    /// Whether to re-read or tear down.
    pub change: ChangeKind,
}

/// Classify a path as a Loom view, a Loom stylesheet, or neither.
///
/// `.loom.style` is tested before `.loom` because a stylesheet filename ends
/// with both suffixes. Paths with a non-UTF-8 or absent filename classify as
/// [`None`] rather than panicking.
#[must_use]
pub fn classify(path: &Path) -> Option<ReloadTarget> {
    let name = path.file_name()?.to_str()?;
    if name.ends_with(".loom.style") {
        Some(ReloadTarget::Style)
    } else if name.ends_with(".loom") {
        Some(ReloadTarget::View)
    } else {
        None
    }
}

/// Folds a burst of raw filesystem events into at most one [`ReloadRequest`]
/// per path, preserving first-seen order and applying last-write-wins to the
/// [`ChangeKind`].
///
/// This is the pure heart of the watcher: it has no I/O and is fully
/// deterministic, so the atomic-save and create-then-delete collapsing rules
/// can be asserted as unit tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coalescer {
    pending: Vec<ReloadRequest>,
}

impl Coalescer {
    /// Creates an empty coalescer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Returns `true` if no requests are currently buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Returns the number of distinct paths currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Records a change for an already-classified `path`.
    ///
    /// If the path is already buffered its [`ChangeKind`] is overwritten with
    /// `change` (last-write-wins), so a save flurry that ends in a modify
    /// yields an [`ChangeKind::Upserted`] and a create followed by a delete
    /// yields a [`ChangeKind::Removed`]. A path seen for the first time is
    /// appended, keeping deterministic first-seen order.
    pub fn record(&mut self, path: PathBuf, target: ReloadTarget, change: ChangeKind) {
        if let Some(existing) = self.pending.iter_mut().find(|req| req.path == path) {
            existing.change = change;
        } else {
            self.pending.push(ReloadRequest {
                path,
                target,
                change,
            });
        }
    }

    /// Classifies and records a raw platform [`Event`], returning `true` if it
    /// mapped to a Loom artifact (and was therefore recorded) and `false` if it
    /// was for an unrelated file and ignored.
    pub fn feed(&mut self, event: &Event) -> bool {
        match classify(&event.path) {
            Some(target) => {
                self.record(
                    event.path.clone(),
                    target,
                    ChangeKind::from_event_kind(event.kind),
                );
                true
            }
            None => false,
        }
    }

    /// Removes and returns all buffered requests in first-seen order, leaving
    /// the coalescer empty.
    #[must_use = "the drained requests are the work to perform; dropping them discards pending reloads"]
    pub fn drain(&mut self) -> Vec<ReloadRequest> {
        core::mem::take(&mut self.pending)
    }
}

/// A filesystem watcher that surfaces coalesced Loom reload requests.
///
/// Wraps a [`prism_platform`] [`Watcher`] and a [`Coalescer`]. Register the
/// files or directories that hold `.loom` / `.loom.style` assets, then call
/// [`LoomWatcher::pump`] (blocking up to a timeout) or
/// [`LoomWatcher::pump_pending`] (non-blocking, suitable for a per-frame poll)
/// to get the batch of changes since the last pump.
pub struct LoomWatcher {
    inner: Watcher,
    coalescer: Coalescer,
}

impl core::fmt::Debug for LoomWatcher {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `prism_platform::Watcher` is not `Debug`; surface the useful bits.
        f.debug_struct("LoomWatcher")
            .field("backend", &self.inner.backend())
            .field("pending", &self.coalescer.len())
            .finish()
    }
}

impl LoomWatcher {
    /// Creates a watcher using the best backend available on this platform.
    ///
    /// # Errors
    ///
    /// Returns a [`WatchError`] if no watch backend could be initialized.
    pub fn new() -> Result<Self, WatchError> {
        Ok(Self {
            inner: Watcher::new()?,
            coalescer: Coalescer::new(),
        })
    }

    /// Creates a watcher pinned to a specific `backend`.
    ///
    /// # Errors
    ///
    /// Returns [`WatchError::Unsupported`] if the backend is unavailable here,
    /// or another [`WatchError`] if initialization fails.
    pub fn with_backend(backend: WatchBackend) -> Result<Self, WatchError> {
        Ok(Self {
            inner: Watcher::with_backend(backend)?,
            coalescer: Coalescer::new(),
        })
    }

    /// Returns which backend this watcher is using.
    #[must_use]
    pub fn backend(&self) -> WatchBackend {
        self.inner.backend()
    }

    /// Watches a single file for changes.
    ///
    /// # Errors
    ///
    /// Returns a [`WatchError`] if the path cannot be watched.
    pub fn watch_file<P: AsRef<Path>>(&mut self, file: P) -> Result<(), WatchError> {
        self.inner.watch(file, false)
    }

    /// Watches a directory, optionally descending into subdirectories.
    ///
    /// # Errors
    ///
    /// Returns a [`WatchError`] if the path cannot be watched.
    pub fn watch_dir<P: AsRef<Path>>(&mut self, dir: P, recursive: bool) -> Result<(), WatchError> {
        self.inner.watch(dir, recursive)
    }

    /// Stops watching a previously registered path.
    ///
    /// # Errors
    ///
    /// Returns [`WatchError::NotWatched`] if the path was never watched, or
    /// another [`WatchError`] on failure.
    pub fn unwatch<P: AsRef<Path>>(&mut self, path: P) -> Result<(), WatchError> {
        self.inner.unwatch(path)
    }

    /// Blocks up to `timeout` for changes, then returns the coalesced batch of
    /// Loom reload requests (empty if nothing relevant changed).
    ///
    /// # Errors
    ///
    /// Returns a [`WatchError`] if the underlying poll fails.
    pub fn pump(&mut self, timeout: Duration) -> Result<Vec<ReloadRequest>, WatchError> {
        let events = self.inner.poll(timeout)?;
        for event in &events {
            self.coalescer.feed(event);
        }
        Ok(self.coalescer.drain())
    }

    /// Returns the coalesced batch of Loom reload requests observed since the
    /// last pump without blocking (empty if nothing relevant changed).
    ///
    /// # Errors
    ///
    /// Returns a [`WatchError`] if the underlying poll fails.
    pub fn pump_pending(&mut self) -> Result<Vec<ReloadRequest>, WatchError> {
        let events = self.inner.try_poll()?;
        for event in &events {
            self.coalescer.feed(event);
        }
        Ok(self.coalescer.drain())
    }
}

#[cfg(test)]
mod tests;
