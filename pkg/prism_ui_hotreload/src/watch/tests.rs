//! Unit and integration tests for the filesystem hot-reload bridge.
//!
//! The pure tests pin down classification and the atomic-save coalescing rules
//! with hand-built events (no disk). The one integration test drives the
//! portable polling backend against a real scratch directory so the end-to-end
//! "a `.loom` file changed on disk" path is exercised on every platform.

use core::time::Duration;
use std::io::Write as _;

use prism_platform::{Event, EventKind, WatchBackend};

use super::{ChangeKind, Coalescer, ReloadTarget, classify};

fn event(path: &str, kind: EventKind) -> Event {
    Event {
        path: std::path::PathBuf::from(path),
        kind,
    }
}

#[test]
fn classify_distinguishes_view_style_and_other() {
    assert_eq!(
        classify(std::path::Path::new("menu.loom")),
        Some(ReloadTarget::View)
    );
    // A stylesheet ends with both `.loom` and `.loom.style`; the style suffix
    // must win.
    assert_eq!(
        classify(std::path::Path::new("menu.loom.style")),
        Some(ReloadTarget::Style)
    );
    assert_eq!(
        classify(std::path::Path::new("ui/panels/hud.loom")),
        Some(ReloadTarget::View)
    );
    assert_eq!(classify(std::path::Path::new("main.rs")), None);
    assert_eq!(classify(std::path::Path::new("notes.txt")), None);
    assert_eq!(classify(std::path::Path::new("loom")), None);
    // No filename component at all.
    assert_eq!(classify(std::path::Path::new("..")), None);
}

#[test]
fn change_kind_only_removed_tears_down() {
    assert_eq!(
        ChangeKind::from_event_kind(EventKind::Removed),
        ChangeKind::Removed
    );
    for kind in [
        EventKind::Created,
        EventKind::Modified,
        EventKind::Renamed,
        EventKind::Other,
    ] {
        assert_eq!(ChangeKind::from_event_kind(kind), ChangeKind::Upserted);
    }
}

#[test]
fn coalescer_collapses_atomic_save_burst_to_one_upsert() {
    let mut c = Coalescer::new();
    // A typical editor atomic save: remove the old inode, create the temp in
    // place, then a final modify. All on the same `.loom` path.
    assert!(c.feed(&event("menu.loom", EventKind::Removed)));
    assert!(c.feed(&event("menu.loom", EventKind::Created)));
    assert!(c.feed(&event("menu.loom", EventKind::Modified)));
    assert_eq!(c.len(), 1);

    let reqs = c.drain();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].path, std::path::PathBuf::from("menu.loom"));
    assert_eq!(reqs[0].target, ReloadTarget::View);
    assert_eq!(reqs[0].change, ChangeKind::Upserted);
    assert!(c.is_empty(), "drain must empty the coalescer");
}

#[test]
fn coalescer_create_then_remove_ends_removed() {
    let mut c = Coalescer::new();
    c.feed(&event("transient.loom", EventKind::Created));
    c.feed(&event("transient.loom", EventKind::Removed));
    let reqs = c.drain();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].change, ChangeKind::Removed);
}

#[test]
fn coalescer_preserves_first_seen_order_across_paths() {
    let mut c = Coalescer::new();
    c.feed(&event("b.loom", EventKind::Modified));
    c.feed(&event("a.loom.style", EventKind::Modified));
    c.feed(&event("b.loom", EventKind::Modified)); // re-touch b; order unchanged
    let reqs = c.drain();
    assert_eq!(reqs.len(), 2);
    assert_eq!(reqs[0].path, std::path::PathBuf::from("b.loom"));
    assert_eq!(reqs[0].target, ReloadTarget::View);
    assert_eq!(reqs[1].path, std::path::PathBuf::from("a.loom.style"));
    assert_eq!(reqs[1].target, ReloadTarget::Style);
}

#[test]
fn coalescer_ignores_unrelated_files() {
    let mut c = Coalescer::new();
    assert!(!c.feed(&event("README.md", EventKind::Modified)));
    assert!(!c.feed(&event("src/main.rs", EventKind::Created)));
    assert!(c.is_empty());
    assert!(c.drain().is_empty());
}

/// Unique scratch directory under the OS temp dir (mirrors the platform watch
/// tests' helper).
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    path.push(format!(
        "prism_loom_watch_{tag}_{nanos}_{:?}",
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&path).expect("create scratch dir");
    path
}

#[test]
fn loom_watcher_poll_backend_sees_loom_change_and_filters_noise() {
    // The polling backend is available on every platform, so pin to it for a
    // deterministic cross-platform integration check.
    let dir = scratch_dir("integration");
    let loom = dir.join("hud.loom");
    let noise = dir.join("hud.rs");
    std::fs::write(&loom, b"box {}").expect("seed loom file");
    std::fs::write(&noise, b"// code").expect("seed noise file");

    let mut watcher =
        super::LoomWatcher::with_backend(WatchBackend::Poll).expect("create poll watcher");
    assert_eq!(watcher.backend(), WatchBackend::Poll);
    watcher.watch_dir(&dir, false).expect("watch scratch dir");

    // Mutate both files; only the `.loom` one should surface as a request.
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&loom)
            .expect("reopen loom for append");
        f.write_all(b"\nbutton {}").expect("append to loom");
        f.sync_all().expect("sync loom");
    }
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&noise)
            .expect("reopen noise for append");
        f.write_all(b"\n// more").expect("append to noise");
        f.sync_all().expect("sync noise");
    }

    // Generous deadline: the poller rescans on a ~50 ms cadence.
    let deadline = Duration::from_secs(3);
    let requests = watcher.pump(deadline).expect("pump");

    assert!(
        requests
            .iter()
            .any(|r| r.path.ends_with("hud.loom")
                && r.target == ReloadTarget::View
                && r.change == ChangeKind::Upserted),
        "expected an upserted View request for hud.loom, got {requests:?}"
    );
    assert!(
        requests.iter().all(|r| !r.path.ends_with("hud.rs")),
        "non-loom file must be filtered out, got {requests:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
