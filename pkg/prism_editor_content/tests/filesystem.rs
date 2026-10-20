//! Real-directory coverage of the public filesystem discovery API.

use core::{
    sync::atomic::{AtomicU64, Ordering},
    task::Poll,
    time::Duration,
};
use prism_editor_content::filesystem::*;
use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new() -> Self {
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "prism-source-fs-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(fs::canonicalize(path).expect("canonical temp directory")),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create owned temp directory: {e}"),
            }
        }
        panic!("could not allocate unique temporary directory");
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, path: &str, bytes: &[u8]) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only owned test directory");
    }
}

fn discovery(limits: FilesystemLimits) -> FilesystemDiscovery {
    FilesystemDiscovery::new(limits, FilesystemExcludes::default()).unwrap()
}

fn wait(scan: &mut FilesystemScan) -> Result<FilesystemSnapshot, FilesystemError> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match scan.poll() {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                assert!(
                    Instant::now() < deadline,
                    "local filesystem worker did not finish"
                );
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn snapshot(service: &FilesystemDiscovery, root: &Path) -> FilesystemSnapshot {
    wait(&mut service.start(root).unwrap()).unwrap()
}

fn wait_finished(scan: &FilesystemScan) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !scan.is_finished() {
        assert!(Instant::now() < deadline, "local worker did not finish");
        thread::sleep(Duration::from_millis(1));
    }
}

fn assert_limit(error: FilesystemError, expected: FilesystemLimit, path: &Path) {
    assert_eq!(error.path, path);
    match error.kind {
        FilesystemErrorKind::LimitExceeded {
            limit,
            maximum,
            actual,
        } => {
            assert_eq!(limit, expected);
            assert!(actual > maximum);
        }
        other => panic!("expected {expected:?}, got {other:?}"),
    }
}

#[test]
fn discovers_nested_real_files_and_exact_sorted_paths_without_reading_contents() {
    let dir = TemporaryDirectory::new();
    dir.write("z.asset", &[0xff, 0, 0x80]);
    dir.write(" 资源 /line\nname.asset ", b"seven!!");
    dir.write("nested/deeper/a file.asset", b"four");
    fs::create_dir(dir.path().join("empty")).unwrap();
    let service = discovery(FilesystemLimits::default());
    let first = snapshot(&service, dir.path());
    assert_eq!(first, snapshot(&service, dir.path()));
    let paths: Vec<_> = first.entries.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            " 资源 ",
            " 资源 /line\nname.asset ",
            "empty",
            "nested",
            "nested/deeper",
            "nested/deeper/a file.asset",
            "z.asset",
        ]
    );
    assert_eq!(first.root, dir.path());
    assert_eq!(first.scanned_entries, 7);
    assert_eq!(
        first.total_path_bytes,
        paths.iter().map(|p| p.len()).sum::<usize>()
    );
    for entry in &first.entries {
        if entry.kind == FilesystemEntryKind::Directory {
            assert_eq!(entry.size_bytes, 0);
        }
    }
    assert_eq!(first.entries[1].size_bytes, 7);
    assert_eq!(first.entries[6].size_bytes, 3);
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn empty_root_is_a_complete_empty_snapshot_even_with_zero_output_budgets() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits {
        max_entries: 0,
        max_path_bytes: 0,
        max_total_path_bytes: 0,
        max_scanned_entries: 0,
        max_depth: 0,
        max_file_size: 0,
        ..FilesystemLimits::default()
    });
    let result = snapshot(&service, dir.path());
    assert!(result.entries.is_empty());
    assert_eq!(result.scanned_entries, 0);
}

#[test]
fn defaults_prune_metadata_and_build_directories_before_enumeration() {
    let dir = TemporaryDirectory::new();
    for path in [
        ".git/config",
        ".hg/store/x",
        ".svn/db",
        "target/deep/x",
        "node_modules/x",
        "build/x",
        "dist/x",
        "src/target/x",
    ] {
        dir.write(path, b"hidden");
    }
    dir.write(".DS_Store", b"hidden");
    dir.write("src/main.asset", b"visible");
    let service = discovery(FilesystemLimits::default());
    let result = snapshot(&service, dir.path());
    assert_eq!(
        result
            .entries
            .iter()
            .map(|e| e.path.as_str())
            .collect::<Vec<_>>(),
        ["src", "src/main.asset"]
    );
    assert_eq!(result.scanned_entries, 11);
}

#[test]
fn excludes_are_configurable_exact_case_sensitive_and_preserve_whitespace() {
    let dir = TemporaryDirectory::new();
    for path in [
        "target/a",
        "targeted/a",
        "Cache/a",
        " raw /skip/a",
        " raw /keep/a",
        "nested/cache/a",
    ] {
        dir.write(path, b"x");
    }
    let mut excludes = FilesystemExcludes::none();
    excludes.exclude_name("cache").unwrap();
    excludes.exclude_subtree(" raw /skip").unwrap();
    let service = FilesystemDiscovery::new(FilesystemLimits::default(), excludes).unwrap();
    let result = snapshot(&service, dir.path());
    let paths: Vec<_> = result.entries.iter().map(|e| e.path.as_str()).collect();
    assert!(paths.contains(&"target/a"));
    assert!(paths.contains(&"targeted/a"));
    assert!(paths.contains(&"Cache/a"));
    assert!(paths.contains(&" raw /keep/a"));
    assert!(!paths
        .iter()
        .any(|p| p.contains("cache") || p.starts_with(" raw /skip")));
}

#[test]
fn rejects_invalid_exclusion_paths_without_normalizing_them() {
    let mut excludes = FilesystemExcludes::none();
    for path in [
        "",
        "/absolute",
        "trailing/",
        "double//slash",
        "a/./b",
        "../b",
        "a\\b",
        "bad\0path",
    ] {
        let e = excludes.exclude_subtree(path).unwrap_err();
        assert!(
            matches!(e.kind, FilesystemErrorKind::InvalidPath(_)),
            "{path:?}"
        );
        assert_eq!(e.path, Path::new(path));
    }
    assert!(excludes.exclude_name("a/b").is_err());
    excludes.exclude_name(" exact\n资源 ").unwrap();
}

#[test]
fn rejects_zero_in_flight_configuration() {
    let result = FilesystemDiscovery::new(
        FilesystemLimits {
            max_in_flight: 0,
            ..FilesystemLimits::default()
        },
        FilesystemExcludes::default(),
    );
    assert!(matches!(
        result.err().unwrap().kind,
        FilesystemErrorKind::InvalidConfiguration(_)
    ));
}

#[test]
fn entry_cap_has_no_partial_success_and_slot_is_reusable() {
    let dir = TemporaryDirectory::new();
    dir.write("file", b"x");
    let service = discovery(FilesystemLimits {
        max_entries: 0,
        max_in_flight: 1,
        ..FilesystemLimits::default()
    });
    for _ in 0..2 {
        let e = wait(&mut service.start(dir.path()).unwrap()).unwrap_err();
        assert_limit(e, FilesystemLimit::Entries, &dir.path().join("file"));
        assert_eq!(service.in_flight(), 0);
    }
}

#[test]
fn path_byte_cap_counts_utf8_bytes_and_exact_boundary_succeeds() {
    let dir = TemporaryDirectory::new();
    dir.write("资源", b"");
    let service = discovery(FilesystemLimits {
        max_path_bytes: 5,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::PathBytes,
        &dir.path().join("资源"),
    );
    let service = discovery(FilesystemLimits {
        max_path_bytes: 6,
        ..FilesystemLimits::default()
    });
    assert_eq!(snapshot(&service, dir.path()).total_path_bytes, 6);
}

#[test]
fn path_byte_cap_includes_parent_and_slash() {
    let dir = TemporaryDirectory::new();
    dir.write("abc/de", b"");
    let service = discovery(FilesystemLimits {
        max_path_bytes: 5,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::PathBytes,
        &dir.path().join("abc/de"),
    );
}

#[test]
fn total_path_bytes_include_directories_and_abort_whole_snapshot() {
    let dir = TemporaryDirectory::new();
    dir.write("a/b", b"");
    let service = discovery(FilesystemLimits {
        max_total_path_bytes: 3,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::TotalPathBytes,
        &dir.path().join("a/b"),
    );
    let service = discovery(FilesystemLimits {
        max_total_path_bytes: 4,
        ..FilesystemLimits::default()
    });
    assert_eq!(snapshot(&service, dir.path()).total_path_bytes, 4);
}

#[test]
fn scanned_entry_cap_counts_excluded_children() {
    let dir = TemporaryDirectory::new();
    dir.write("target/deeper/ignored", b"");
    let service = discovery(FilesystemLimits {
        max_scanned_entries: 0,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::ScannedEntries,
        &dir.path().join("target"),
    );
    let service = discovery(FilesystemLimits {
        max_scanned_entries: 1,
        ..FilesystemLimits::default()
    });
    assert!(snapshot(&service, dir.path()).entries.is_empty());
}

#[test]
fn depth_cap_is_root_relative_and_includes_files_and_empty_directories() {
    let dir = TemporaryDirectory::new();
    dir.write("a/b", b"");
    let service = discovery(FilesystemLimits {
        max_depth: 1,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::Depth,
        &dir.path().join("a/b"),
    );
    let service = discovery(FilesystemLimits {
        max_depth: 2,
        ..FilesystemLimits::default()
    });
    assert_eq!(snapshot(&service, dir.path()).entries.len(), 2);
    fs::remove_file(dir.path().join("a/b")).unwrap();
    let service = discovery(FilesystemLimits {
        max_depth: 0,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::Depth,
        &dir.path().join("a"),
    );
}

#[test]
fn file_size_cap_uses_metadata_and_accepts_exact_limit() {
    let dir = TemporaryDirectory::new();
    dir.write("large", b"");
    let file = fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("large"))
        .unwrap();
    file.set_len(1_048_576).unwrap();
    drop(file);
    let service = discovery(FilesystemLimits {
        max_file_size: 1_048_575,
        ..FilesystemLimits::default()
    });
    assert_limit(
        wait(&mut service.start(dir.path()).unwrap()).unwrap_err(),
        FilesystemLimit::FileSize,
        &dir.path().join("large"),
    );
    let service = discovery(FilesystemLimits {
        max_file_size: 1_048_576,
        ..FilesystemLimits::default()
    });
    assert_eq!(
        snapshot(&service, dir.path()).entries[0].size_bytes,
        1_048_576
    );
}

#[test]
fn root_byte_cap_is_separate_from_output_budget() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits {
        max_root_path_bytes: dir.path().as_os_str().len() - 1,
        ..FilesystemLimits::default()
    });
    assert_limit(
        service.start(dir.path()).err().unwrap(),
        FilesystemLimit::RootPathBytes,
        dir.path(),
    );
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn zero_timeout_returns_atomic_timeout_and_releases_slot() {
    let dir = TemporaryDirectory::new();
    dir.write("asset", b"x");
    let service = discovery(FilesystemLimits {
        timeout: Duration::ZERO,
        ..FilesystemLimits::default()
    });
    let e = wait(&mut service.start(dir.path()).unwrap()).unwrap_err();
    assert_eq!(e.path, dir.path());
    assert!(matches!(e.kind, FilesystemErrorKind::TimedOut));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn cancellation_discards_even_an_unconsumed_completed_result() {
    let dir = TemporaryDirectory::new();
    dir.write("asset", b"x");
    let service = discovery(FilesystemLimits {
        max_in_flight: 1,
        ..FilesystemLimits::default()
    });
    let mut scan = service.start(dir.path()).unwrap();
    wait_finished(&scan);
    scan.cancel();
    assert!(matches!(
        wait(&mut scan).unwrap_err().kind,
        FilesystemErrorKind::Cancelled
    ));
    assert_eq!(service.in_flight(), 0);
    assert!(matches!(
        scan.poll(),
        Poll::Ready(Err(FilesystemError {
            kind: FilesystemErrorKind::AlreadyConsumed,
            ..
        }))
    ));
    assert_eq!(snapshot(&service, dir.path()).entries.len(), 1);
}

#[test]
fn cancellation_before_poll_is_deterministic_and_discards_all_entries() {
    let dir = TemporaryDirectory::new();
    for i in 0..32 {
        dir.write(&format!("nested/{i}/asset"), b"x");
    }
    let service = discovery(FilesystemLimits::default());
    let mut scan = service.start(dir.path()).unwrap();
    scan.cancel();
    assert!(matches!(
        wait(&mut scan).unwrap_err().kind,
        FilesystemErrorKind::Cancelled
    ));
    assert!(scan.is_finished());
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn cloned_services_share_slots_until_result_consumed_or_dropped() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits {
        max_in_flight: 1,
        ..FilesystemLimits::default()
    });
    let clone = service.clone();
    let scan = service.start(dir.path()).unwrap();
    wait_finished(&scan);
    assert_eq!(clone.in_flight(), 1);
    assert!(matches!(
        clone.start(dir.path()).err().unwrap().kind,
        FilesystemErrorKind::InFlightLimit { maximum: 1 }
    ));
    drop(scan);
    assert_eq!(clone.in_flight(), 0);
    assert!(snapshot(&clone, dir.path()).entries.is_empty());
}

#[test]
fn concurrent_starts_never_exceed_shared_in_flight_budget() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits {
        max_in_flight: 2,
        ..FilesystemLimits::default()
    });
    let barrier = std::sync::Barrier::new(9);
    thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    let result = service.start(dir.path());
                    // Keep accepted requests alive until all starts have completed.
                    barrier.wait();
                    result
                })
            })
            .collect();
        barrier.wait();
        barrier.wait();
        let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 2);
        assert_eq!(service.in_flight(), 2);
        for e in results.iter().filter_map(|r| r.as_ref().err()) {
            assert!(matches!(
                e.kind,
                FilesystemErrorKind::InFlightLimit { maximum: 2 }
            ));
        }
        drop(results);
    });
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn dropping_active_worker_joins_before_root_is_removed_and_slot_is_reused() {
    let dir = TemporaryDirectory::new();
    for i in 0..64 {
        dir.write(&format!("nested/{i}/asset"), b"x");
    }
    let service = discovery(FilesystemLimits {
        max_in_flight: 1,
        ..FilesystemLimits::default()
    });
    for _ in 0..8 {
        let scan = service.start(dir.path()).unwrap();
        drop(scan);
        assert_eq!(service.in_flight(), 0);
    }
    let result = snapshot(&service, dir.path());
    assert_eq!(result.entries.len(), 129);
    fs::rename(dir.path().join("nested"), dir.path().join("moved")).unwrap();
    fs::remove_dir_all(dir.path().join("moved")).unwrap();
    assert!(snapshot(&service, dir.path()).entries.is_empty());
}

#[test]
fn missing_or_file_root_reports_native_path_and_operation() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits::default());
    let missing = dir.path().join("missing\n资源");
    let e = wait(&mut service.start(&missing).unwrap()).unwrap_err();
    assert_eq!(e.path, missing);
    assert!(e.to_string().contains("missing\\n资源"));
    assert!(core::error::Error::source(&e).is_some());
    assert!(matches!(
        e.kind,
        FilesystemErrorKind::Io {
            operation: "symlink_metadata directory",
            ..
        }
    ));
    dir.write("file", b"x");
    let e = wait(&mut service.start(dir.path().join("file")).unwrap()).unwrap_err();
    assert!(matches!(e.kind, FilesystemErrorKind::NotDirectory));
    assert_eq!(e.path, dir.path().join("file"));
}

#[test]
fn invalid_root_is_rejected_before_reserving_slot() {
    let dir = TemporaryDirectory::new();
    let service = discovery(FilesystemLimits::default());
    for path in [
        PathBuf::new(),
        dir.path().join("../escape"),
        dir.path().join("bad\0path"),
    ] {
        let e = service.start(&path).err().unwrap();
        assert!(matches!(e.kind, FilesystemErrorKind::InvalidPath(_)));
    }
    assert_eq!(service.in_flight(), 0);
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::{
        ffi::OsString,
        os::unix::{ffi::OsStringExt, fs::symlink},
    };

    #[test]
    fn directory_file_dangling_and_cycle_symlinks_are_skipped_without_traversal() {
        let dir = TemporaryDirectory::new();
        let outside = TemporaryDirectory::new();
        outside.write("must-not-be-seen", b"outside");
        dir.write("real/asset", b"inside");
        symlink(outside.path(), dir.path().join("outside")).unwrap();
        symlink(dir.path().join("real/asset"), dir.path().join("file-link")).unwrap();
        symlink(dir.path().join("absent"), dir.path().join("dangling")).unwrap();
        symlink(dir.path(), dir.path().join("real/cycle")).unwrap();
        let service = discovery(FilesystemLimits {
            max_scanned_entries: 6,
            ..FilesystemLimits::default()
        });
        let result = snapshot(&service, dir.path());
        assert_eq!(
            result
                .entries
                .iter()
                .map(|e| e.path.as_str())
                .collect::<Vec<_>>(),
            ["real", "real/asset"]
        );
        assert_eq!(result.scanned_entries, 6);
    }

    #[test]
    fn symlink_root_and_symlink_ancestor_are_explicit_errors() {
        let dir = TemporaryDirectory::new();
        fs::create_dir_all(dir.path().join("real/child")).unwrap();
        let link = dir.path().join("linked");
        symlink(dir.path().join("real"), &link).unwrap();
        let service = discovery(FilesystemLimits::default());
        for root in [&link, &link.join("child")] {
            let e = wait(&mut service.start(root).unwrap()).unwrap_err();
            assert!(matches!(e.kind, FilesystemErrorKind::SymlinkDirectory));
            assert_eq!(e.path, link);
        }
    }

    #[test]
    fn non_utf8_root_is_an_error_with_original_native_bytes() {
        let dir = TemporaryDirectory::new();
        let path = dir
            .path()
            .join(OsString::from_vec(b"invalid-\xff".to_vec()));
        let service = discovery(FilesystemLimits::default());
        let e = service.start(&path).err().unwrap();
        assert!(matches!(e.kind, FilesystemErrorKind::NonUtf8));
        assert_eq!(e.path, path);
        assert!(e.to_string().contains("\\xFF"));
    }

    // macOS APFS rejects this filename at creation with EILSEQ. Keep the actual
    // directory-entry test on hosts whose filesystem supports native byte names.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_utf8_child_is_an_error_with_original_native_bytes() {
        let dir = TemporaryDirectory::new();
        let path = dir
            .path()
            .join(OsString::from_vec(b"invalid-\xff".to_vec()));
        fs::write(&path, b"").unwrap();
        let service = discovery(FilesystemLimits::default());
        let e = wait(&mut service.start(dir.path()).unwrap()).unwrap_err();
        assert!(matches!(e.kind, FilesystemErrorKind::NonUtf8));
        assert_eq!(e.path, path);
    }

    #[test]
    fn backslash_child_is_an_explicit_error() {
        let dir = TemporaryDirectory::new();
        dir.write("a\\b", b"");
        let service = discovery(FilesystemLimits::default());
        let e = wait(&mut service.start(dir.path()).unwrap()).unwrap_err();
        assert!(matches!(e.kind, FilesystemErrorKind::InvalidPath(_)));
        assert_eq!(e.path, dir.path().join("a\\b"));
    }

    #[test]
    fn excluded_subtree_is_pruned_before_invalid_children_are_seen() {
        let dir = TemporaryDirectory::new();
        dir.write("target/a\\b", b"");
        let result = snapshot(&discovery(FilesystemLimits::default()), dir.path());
        assert!(result.entries.is_empty());
        assert_eq!(result.scanned_entries, 1);
    }

    #[test]
    fn special_file_is_rejected_from_metadata_without_opening_it() {
        let dir = TemporaryDirectory::new();
        let path = dir.path().join("socket");
        let _socket = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let service = discovery(FilesystemLimits::default());
        let e = wait(&mut service.start(dir.path()).unwrap()).unwrap_err();
        assert!(matches!(e.kind, FilesystemErrorKind::UnsupportedFileType));
        assert_eq!(e.path, path);
    }
}
