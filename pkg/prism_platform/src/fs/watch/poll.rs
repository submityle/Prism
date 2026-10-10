//! Portable `stat`-based polling watcher.
//!
//! This is the honest baseline backend (design doc §8): it keeps a snapshot of
//! the modification time and size of every watched file (and, for directory
//! watches, of every entry in the watched tree) and reports the difference on
//! each rescan. It needs no OS-specific syscall and therefore works on every
//! `std` platform — Linux, Windows, macOS, and anywhere else `std::fs` runs.
//!
//! It is the default backend on platforms without a wired-up native watcher
//! (Linux/Windows here) and is always selectable explicitly via
//! [`super::Backend::Poll`].

use core::time::Duration;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use super::{Event, EventKind, Result, WatchError};

/// The slice of time the blocking [`PollWatcher::poll`] sleeps between rescans.
///
/// Small enough to feel responsive for hot-reload, large enough not to spin the
/// CPU. The final slice is clamped so a poll never overshoots its deadline.
const RESCAN_INTERVAL: Duration = Duration::from_millis(50);

/// A cheap identity for a filesystem entry: modification time plus length.
///
/// Two scans that disagree on either field mean the entry changed. This is the
/// same heuristic every portable file-watcher uses; it can miss a change that
/// rewrites identical bytes within one timer tick without touching the mtime,
/// which is an accepted limitation of `stat`-based watching.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileState {
    mtime: Option<SystemTime>,
    len: u64,
    is_dir: bool,
}

/// One registered watch root and the last snapshot taken of it.
struct Root {
    recursive: bool,
    snapshot: HashMap<PathBuf, FileState>,
}

/// The portable polling watcher.
pub(super) struct PollWatcher {
    roots: HashMap<PathBuf, Root>,
}

impl PollWatcher {
    pub(super) fn new() -> Self {
        Self {
            roots: HashMap::new(),
        }
    }

    pub(super) fn watch(&mut self, path: &Path, recursive: bool) -> Result<()> {
        let snapshot = snapshot(path, recursive);
        self.roots.insert(
            path.to_path_buf(),
            Root {
                recursive,
                snapshot,
            },
        );
        Ok(())
    }

    pub(super) fn unwatch(&mut self, path: &Path) -> Result<()> {
        if self.roots.remove(path).is_some() {
            Ok(())
        } else {
            Err(WatchError::NotWatched)
        }
    }

    pub(super) fn poll(&mut self, timeout: Duration) -> Result<Vec<Event>> {
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let events = self.rescan();
            if !events.is_empty() {
                return Ok(events);
            }
            // A zero timeout is a single non-blocking scan.
            let Some(deadline) = deadline else {
                return Ok(events);
            };
            let now = Instant::now();
            if now >= deadline {
                return Ok(events);
            }
            let remaining = deadline - now;
            std::thread::sleep(RESCAN_INTERVAL.min(remaining));
        }
    }

    /// Rescan every root once, emit the diff against its stored snapshot, and
    /// update the snapshot in place.
    fn rescan(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        for (root, state) in self.roots.iter_mut() {
            let fresh = snapshot(root, state.recursive);
            diff(&state.snapshot, &fresh, &mut events);
            state.snapshot = fresh;
        }
        events
    }
}

/// Produce an entry's [`FileState`] from its metadata, if it exists.
fn state_of(path: &Path) -> Option<FileState> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    Some(FileState {
        mtime: meta.modified().ok(),
        len: meta.len(),
        is_dir: meta.is_dir(),
    })
}

/// Build a snapshot of `root`. A file maps to a single entry; a directory maps
/// to itself plus every entry beneath it (recursively when `recursive`).
fn snapshot(root: &Path, recursive: bool) -> HashMap<PathBuf, FileState> {
    let mut map = HashMap::new();
    let Some(root_state) = state_of(root) else {
        // Nothing there (yet). An empty snapshot means a later `Created` event
        // fires when the path appears.
        return map;
    };
    let is_dir = root_state.is_dir;
    map.insert(root.to_path_buf(), root_state);
    if is_dir {
        collect_dir(root, recursive, &mut map);
    }
    map
}

/// Add every entry of directory `dir` to `map`, descending when `recursive`.
fn collect_dir(dir: &Path, recursive: bool, map: &mut HashMap<PathBuf, FileState>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(state) = state_of(&path) {
            let descend = recursive && state.is_dir;
            map.insert(path.clone(), state);
            if descend {
                collect_dir(&path, recursive, map);
            }
        }
    }
}

/// Compare two snapshots of the same root and append the changes to `events`.
fn diff(
    old: &HashMap<PathBuf, FileState>,
    new: &HashMap<PathBuf, FileState>,
    events: &mut Vec<Event>,
) {
    for (path, new_state) in new {
        match old.get(path) {
            None => events.push(Event {
                path: path.clone(),
                kind: EventKind::Created,
            }),
            Some(old_state) if old_state != new_state => events.push(Event {
                path: path.clone(),
                kind: EventKind::Modified,
            }),
            Some(_) => {}
        }
    }
    for path in old.keys() {
        if !new.contains_key(path) {
            events.push(Event {
                path: path.clone(),
                kind: EventKind::Removed,
            });
        }
    }
}
