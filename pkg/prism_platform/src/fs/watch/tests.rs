//! Unit tests for the M4 file-watching layer.
//!
//! These exercise the portable polling watcher everywhere, the native `kqueue`
//! watcher on macOS/BSD, the native Linux `inotify` watcher, and the native
//! Windows `ReadDirectoryChangesW` watcher. They cover
//! recursive additions, rename/delete, high-frequency directory changes,
//! fallback polling, and honest backend selection.

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

fn rename_and_delete(backend: Backend) {
    let dir = scratch_dir("rename");
    let before = dir.join("before.asset");
    let after = dir.join("after.asset");
    std::fs::write(&before, b"payload").expect("seed file");

    let mut w = Watcher::with_backend(backend).expect("create watcher");
    w.watch(&dir, false).expect("watch dir");

    std::fs::rename(&before, &after).expect("rename file");
    let rename_events = w.poll(DEADLINE).expect("poll rename");
    let has_native_rename = rename_events
        .iter()
        .any(|event| event.kind == EventKind::Renamed);
    let has_polling_pair = rename_events
        .iter()
        .any(|event| event.path == before && event.kind == EventKind::Removed)
        && rename_events
            .iter()
            .any(|event| event.path == after && event.kind == EventKind::Created);
    assert!(
        has_native_rename || has_polling_pair,
        "expected rename notification for {backend:?}, got {rename_events:?}"
    );

    std::fs::remove_file(&after).expect("delete renamed file");
    let delete_events = w.poll(DEADLINE).expect("poll delete");
    assert!(
        delete_events
            .iter()
            .any(|event| event.path == after && event.kind == EventKind::Removed),
        "expected delete notification for {backend:?}, got {delete_events:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

fn high_frequency_creates(backend: Backend) {
    const COUNT: usize = 128;
    let dir = scratch_dir("burst");
    let mut w = Watcher::with_backend(backend).expect("create watcher");
    w.watch(&dir, false).expect("watch dir");

    for index in 0..COUNT {
        std::fs::write(dir.join(format!("burst_{index:04}.asset")), [index as u8])
            .expect("create burst file");
    }

    let events = w.poll(DEADLINE).expect("poll burst");
    let created = events
        .iter()
        .filter(|event| event.kind == EventKind::Created)
        .count();
    assert!(
        created >= COUNT,
        "expected at least {COUNT} created events for {backend:?}, got {created}: {events:?}"
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

#[test]
fn poll_backend_detects_rename_and_delete() {
    rename_and_delete(Backend::Poll);
}

#[test]
fn poll_backend_handles_high_frequency_creates() {
    high_frequency_creates(Backend::Poll);
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

#[cfg(target_os = "linux")]
#[test]
fn linux_inotify_detects_recursive_creation() {
    let dir = scratch_dir("recursive");
    let nested = dir.join("nested");
    let nested_file = nested.join("shader.wgsl");

    let mut w = Watcher::with_backend(Backend::Native).expect("create inotify watcher");
    w.watch(&dir, true).expect("watch recursive dir");

    std::fs::create_dir(&nested).expect("create nested dir");
    let first = w.poll(DEADLINE).expect("poll nested dir creation");
    assert!(
        first
            .iter()
            .any(|event| event.path == nested && event.kind == EventKind::Created),
        "expected nested directory creation, got {first:?}"
    );

    std::fs::write(&nested_file, b"shader").expect("create nested file");
    let second = w.poll(DEADLINE).expect("poll nested file creation");
    assert!(
        second
            .iter()
            .any(|event| event.path == nested_file && event.kind == EventKind::Created),
        "expected dynamically watched nested file creation, got {second:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(target_os = "linux")]
#[test]
fn linux_inotify_detects_rename_and_delete() {
    rename_and_delete(Backend::Native);
}

#[cfg(target_os = "linux")]
#[test]
fn linux_inotify_handles_high_frequency_creates() {
    high_frequency_creates(Backend::Native);
}

#[cfg(windows)]
#[test]
fn windows_native_detects_file_modification() {
    file_modification(Backend::Native);
}

#[cfg(windows)]
#[test]
fn windows_native_detects_recursive_creation() {
    let dir = scratch_dir("windows_recursive");
    let nested = dir.join("nested");
    let nested_file = nested.join("shader.wgsl");

    let mut w = Watcher::with_backend(Backend::Native).expect("create Windows watcher");
    w.watch(&dir, true).expect("watch recursive dir");

    std::fs::create_dir(&nested).expect("create nested dir");
    let first = w.poll(DEADLINE).expect("poll nested dir creation");
    assert!(
        first
            .iter()
            .any(|event| event.path == nested && event.kind == EventKind::Created),
        "expected nested directory creation, got {first:?}"
    );

    std::fs::write(&nested_file, b"shader").expect("create nested file");
    let second = w.poll(DEADLINE).expect("poll nested file creation");
    assert!(
        second
            .iter()
            .any(|event| event.path == nested_file && event.kind == EventKind::Created),
        "expected dynamically watched nested file creation, got {second:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(windows)]
#[test]
fn windows_native_detects_rename_and_delete() {
    rename_and_delete(Backend::Native);
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

#[cfg(not(any(
    target_os = "linux",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    windows
)))]
#[test]
fn native_backend_is_explicitly_unsupported_without_native_implementation() {
    assert!(matches!(
        Watcher::with_backend(Backend::Native),
        Err(super::WatchError::Unsupported)
    ));
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
