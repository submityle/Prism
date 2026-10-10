//! Unit tests for the M4 file-watching layer.
//!
//! These exercise both backends: the portable polling watcher everywhere, and
//! the native `kqueue` watcher on macOS/BSD. They assert that a modification to
//! a watched file and a creation inside a watched directory are observed, that
//! `unwatch` of an unknown path is reported, and that the backend selection is
//! honest.

use core::time::Duration;
use std::io::Write as _;

use super::{watch_native_supported, Backend, EventKind, Watcher};

/// A generous deadline — the native backend is immediate, the poller rescans
/// on a 50 ms cadence, so a couple of seconds is comfortably safe without
/// making a failing test hang long.
const DEADLINE: Duration = Duration::from_secs(3);

/// Unique scratch directory under the OS temp dir.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "prism_watch_{tag}_{nanos}_{:?}",
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&path).expect("create scratch dir");
    path
}

/// Watch a file, mutate it, and assert a modification surfaces.
fn file_modification(backend: Backend) {
    let dir = scratch_dir("file");
    let file = dir.join("asset.bin");
    std::fs::write(&file, b"original").expect("seed file");

    let mut w = Watcher::with_backend(backend).expect("create watcher");
    w.watch(&file, false).expect("watch file");

    // Mutate: change both length and contents so neither a coarse mtime nor a
    // same-size rewrite can hide the change from the polling backend.
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&file)
            .expect("reopen for append");
        f.write_all(b" + more bytes").expect("append");
        f.sync_all().expect("sync");
    }

    let events = w.poll(DEADLINE).expect("poll");
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, EventKind::Modified | EventKind::Other)),
        "expected a Modified event for {backend:?}, got {events:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Watch a directory, create a file inside, and assert a change surfaces.
fn dir_creation(backend: Backend) {
    let dir = scratch_dir("dir");

    let mut w = Watcher::with_backend(backend).expect("create watcher");
    w.watch(&dir, false).expect("watch dir");

    std::fs::write(dir.join("new_asset.txt"), b"hello").expect("create file in dir");

    let events = w.poll(DEADLINE).expect("poll");
    assert!(
        !events.is_empty(),
        "expected at least one event for a new file in a watched dir ({backend:?})"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn poll_backend_detects_file_modification() {
    file_modification(Backend::Poll);
}

#[test]
fn poll_backend_detects_dir_creation() {
    dir_creation(Backend::Poll);
}

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
#[test]
fn native_backend_detects_file_modification() {
    file_modification(Backend::Native);
}

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
#[test]
fn native_backend_detects_dir_creation() {
    dir_creation(Backend::Native);
}

#[test]
fn default_watcher_picks_native_when_available() {
    let w = Watcher::new().expect("create default watcher");
    let expected = if watch_native_supported() {
        Backend::Native
    } else {
        Backend::Poll
    };
    assert_eq!(w.backend(), expected);
}

#[test]
fn unwatch_unknown_path_is_reported() {
    let mut w = Watcher::with_backend(Backend::Poll).expect("create watcher");
    let err = w
        .unwatch(std::env::temp_dir().join("prism_never_watched_xyz"))
        .expect_err("unwatching an unwatched path should fail");
    assert!(matches!(err, super::WatchError::NotWatched), "got {err:?}");
}

#[test]
fn try_poll_is_non_blocking_and_empty_initially() {
    let dir = scratch_dir("quiet");
    let file = dir.join("still.bin");
    std::fs::write(&file, b"unchanging").expect("seed");

    let mut w = Watcher::with_backend(Backend::Poll).expect("create watcher");
    w.watch(&file, false).expect("watch");

    let events = w.try_poll().expect("try_poll");
    assert!(events.is_empty(), "no change yet, got {events:?}");

    std::fs::remove_dir_all(&dir).ok();
}
