//! Real Git repository/process regressions, including termination and reaping.
#![cfg(unix)]
use adapter::{GitError, GitLimits, GitProvider, GitRequest, GitSnapshot};
use prism_editor_content::{git as adapter, VcsRecord, VcsState};
use prism_editor_content::{AssetId, AssetKind, AssetRecord, ContentCatalog};
extern crate alloc;
use alloc::{format, string::ToString};
use core::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use std::{
    ffi::OsString,
    fs,
    os::unix::ffi::OsStringExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    thread,
    time::Instant,
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Repo(PathBuf);
impl Repo {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-content-git-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let repo = Self(root);
        repo.git(&["init", "-q"]);
        repo.git(&["config", "user.email", "test@example.invalid"]);
        repo.git(&["config", "user.name", "Prism Git test"]);
        repo.git(&["config", "commit.gpgsign", "false"]);
        repo.git(&["config", "core.autocrlf", "false"]);
        repo
    }
    fn git_command(&self) -> Command {
        let mut cmd = Command::new("git");
        for (key, _) in std::env::vars_os() {
            if key.as_encoded_bytes().starts_with(b"GIT_") {
                cmd.env_remove(key);
            }
        }
        cmd.current_dir(&self.0).args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
        ]);
        cmd
    }
    fn command(&self, args: &[&str]) -> Output {
        self.git_command().args(args).output().unwrap()
    }
    fn git(&self, args: &[&str]) {
        let out = self.command(args);
        assert!(out.status.success(), "{args:?}: {:?}", out.stderr);
    }
    fn write(&self, name: impl AsRef<Path>, bytes: &[u8]) {
        let path = self.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    fn commit(&self) {
        self.git(&["add", "--all"]);
        self.git(&["commit", "-qm", "baseline"]);
    }
    fn snapshot(&self) -> GitSnapshot {
        wait(
            &mut provider(GitLimits::default())
                .start(self.0.clone())
                .unwrap(),
        )
        .unwrap()
    }
    fn block_index(&self) {
        fs::rename(self.0.join(".git/index"), self.0.join(".git/index.saved")).unwrap();
        // Actual Git blocks opening a real FIFO index: deterministic kill/wait
        // coverage without a fake executable or fabricated status output.
        assert!(Command::new("mkfifo")
            .arg(self.0.join(".git/index"))
            .status()
            .unwrap()
            .success());
    }
}
impl Drop for Repo {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn provider(limits: GitLimits) -> GitProvider {
    GitProvider::new(limits).unwrap()
}
fn wait(request: &mut GitRequest) -> Result<GitSnapshot, GitError> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(result) = request.poll() {
            assert!(request.process_id().is_none());
            return result;
        }
        assert!(Instant::now() < deadline, "request failed to finish");
        thread::sleep(Duration::from_millis(1));
    }
}
fn entry<'a>(s: &'a GitSnapshot, path: &str) -> &'a adapter::GitEntry {
    s.entries
        .iter()
        .find(|e| e.path == path)
        .unwrap_or_else(|| panic!("missing {path:?}: {s:?}"))
}
fn started(request: &GitRequest) -> u32 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(pid) = request.process_id() {
            return pid;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
}
fn assert_reaped(pid: u32) {
    let out = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
        .unwrap();
    assert!(out.stderr.is_empty(), "ps failed: {:?}", out.stderr);
    assert!(
        matches!(out.status.code(), Some(0 | 1)),
        "ps failed: {:?}",
        out.status
    );
    assert!(out.stdout.is_empty(), "child remains alive/zombie: {pid}");
}

fn ready(request: &GitRequest) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !request.is_ready() {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(request.process_id().is_none());
}

#[test]
fn real_status_preserves_paths_xy_clean_and_read_only_index() {
    let repo = Repo::new();
    for name in [
        "clean",
        "modified",
        "deleted",
        "staged deleted",
        " renamed old\n雪 ",
        " leading\t雪\ntrailing ",
    ] {
        repo.write(name, name.as_bytes());
    }
    repo.write(".gitignore", b"ignored.bin\nignored-dir/\n");
    repo.commit();
    repo.write("modified", b"index change");
    repo.git(&["add", "modified"]);
    repo.write("modified", b"worktree change");
    repo.write("added 雪", b"stage");
    repo.git(&["add", "added 雪"]);
    repo.write("added 雪", b"later");
    fs::remove_file(repo.0.join("deleted")).unwrap();
    repo.git(&["rm", "-q", "staged deleted"]);
    repo.git(&["mv", " renamed old\n雪 ", " renamed new\n雪 "]);
    repo.write(" leading\t雪\ntrailing ", b"changed");
    repo.write(" untracked\n雪 ", b"u");
    repo.write("nested/untracked", b"u");
    repo.write("ignored.bin", b"i");
    repo.write("ignored-dir/child", b"i");
    let before = fs::read(repo.0.join(".git/index")).unwrap();
    let modified = fs::metadata(repo.0.join(".git/index"))
        .unwrap()
        .modified()
        .unwrap();
    let s = repo.snapshot();
    assert_eq!(entry(&s, "clean").record.state, VcsState::Clean);
    assert_eq!(entry(&s, "modified").xy, *b"MM");
    assert!(entry(&s, "modified").record.staged);
    assert_eq!(entry(&s, "added 雪").xy, *b"AM");
    assert_eq!(entry(&s, "deleted").xy, *b" D");
    assert_eq!(entry(&s, "staged deleted").xy, *b"D ");
    assert_eq!(entry(&s, " leading\t雪\ntrailing ").xy, *b" M");
    assert_eq!(
        entry(&s, " renamed new\n雪 ").original_path.as_deref(),
        Some(" renamed old\n雪 ")
    );
    assert_eq!(
        entry(&s, " renamed new\n雪 ").record.state,
        VcsState::Renamed
    );
    assert_eq!(
        entry(&s, " renamed old\n雪 ").record.state,
        VcsState::Deleted
    );
    assert!(entry(&s, " renamed old\n雪 ").record.staged);
    for name in [" untracked\n雪 ", "nested/untracked"] {
        assert_eq!(entry(&s, name).record.state, VcsState::Untracked);
    }
    for name in ["ignored.bin", "ignored-dir/child"] {
        assert_eq!(entry(&s, name).record.state, VcsState::Ignored);
    }
    let mut catalog = ContentCatalog::new();
    let ignored_id = AssetId::from_raw(7);
    catalog
        .insert(
            AssetRecord::new(
                ignored_id,
                "ignored-dir/child",
                AssetKind::new("file").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    catalog
        .replace_vcs_statuses(s.clone().into_vcs_statuses())
        .unwrap();
    assert_eq!(
        catalog.vcs_status(ignored_id).unwrap().state,
        VcsState::Ignored
    );
    assert_eq!(before, fs::read(repo.0.join(".git/index")).unwrap());
    assert_eq!(
        modified,
        fs::metadata(repo.0.join(".git/index"))
            .unwrap()
            .modified()
            .unwrap()
    );
    assert!(!repo.0.join(".git/index.lock").exists());
    assert_eq!(repo.snapshot(), s);
}
#[test]
fn rename_source_recreated_is_not_falsely_deleted() {
    let repo = Repo::new();
    repo.write("old", b"original");
    repo.commit();
    repo.git(&["mv", "old", "new"]);
    repo.write("old", b"new untracked content");
    let s = repo.snapshot();
    assert_eq!(entry(&s, "old").record.state, VcsState::Untracked);
    assert_eq!(entry(&s, "new").original_path.as_deref(), Some("old"));
}

#[test]
fn staged_rename_and_worktree_edit_preserve_both_columns() {
    let repo = Repo::new();
    repo.write("old", b"one\ntwo\nthree\nfour\nfive\n");
    repo.commit();
    repo.git(&["mv", "old", "new"]);
    repo.write("new", b"one\ntwo\nthree\nfour\nchanged\n");
    let snapshot = repo.snapshot();
    assert_eq!(entry(&snapshot, "new").xy, *b"RM");
    assert!(entry(&snapshot, "new").record.staged);
    assert_eq!(entry(&snapshot, "new").record.state, VcsState::Renamed);
    assert_eq!(entry(&snapshot, "old").xy, *b"D ");
}
#[test]
fn real_merge_conflict_is_not_clean_or_staged() {
    let repo = Repo::new();
    repo.write("conflict", b"base\n");
    repo.commit();
    repo.git(&["checkout", "-qb", "side"]);
    repo.write("conflict", b"side\n");
    repo.commit();
    repo.git(&["checkout", "-qb", "other", "HEAD~1"]);
    repo.write("conflict", b"other\n");
    repo.commit();
    assert!(!repo
        .command(&["merge", "--no-edit", "side"])
        .status
        .success());
    let s = repo.snapshot();
    assert_eq!(entry(&s, "conflict").xy, *b"UU");
    assert_eq!(entry(&s, "conflict").record.state, VcsState::Conflicted);
    assert!(!entry(&s, "conflict").record.staged);
    assert_eq!(s.entries.len(), 1);
}
#[test]
fn invalid_utf8_and_backslash_are_explicit_errors() {
    for name in [
        OsString::from_vec(b"bad-\xff".to_vec()),
        OsString::from("back\\slash"),
    ] {
        let repo = Repo::new();
        repo.write("clean", b"ok");
        repo.commit();
        // APFS cannot create non-UTF-8 names. Git's real index can still contain
        // them (e.g. a checkout from Linux), so exercise that case via cacheinfo.
        // Linux additionally exercises the untracked filesystem path.
        let on_disk = match fs::write(repo.0.join(&name), b"bad") {
            Ok(()) => true,
            Err(error) => {
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("{error}");
                }
                #[cfg(target_os = "macos")]
                {
                    assert_eq!(error.raw_os_error(), Some(92), "{error}");
                    false
                }
            }
        };
        for tracked in [false, true] {
            if tracked {
                let hash = repo.command(&["rev-parse", "HEAD:clean"]);
                assert!(hash.status.success());
                let hash = core::str::from_utf8(&hash.stdout).unwrap().trim();
                let output = repo
                    .git_command()
                    .args(["update-index", "--add", "--cacheinfo", "100644", hash])
                    .arg(&name)
                    .output()
                    .unwrap();
                assert!(output.status.success(), "{:?}", output.stderr);
            } else if !on_disk {
                continue;
            }
            let error = wait(
                &mut provider(GitLimits::default())
                    .start(repo.0.clone())
                    .unwrap(),
            )
            .unwrap_err();
            assert_eq!(
                error,
                GitError::UnrepresentablePath(name.clone().into_vec())
            );
        }
    }
}
#[test]
fn combined_budgets_preserve_catalog_then_clean_replaces_dirty() {
    let repo = Repo::new();
    repo.write("asset", b"base");
    repo.commit();
    repo.write("asset", b"dirty");
    let mut catalog = ContentCatalog::new();
    let id = AssetId::from_raw(1);
    catalog
        .insert(AssetRecord::new(id, "asset", AssetKind::new("file").unwrap()).unwrap())
        .unwrap();
    catalog
        .replace_vcs_statuses(repo.snapshot().into_vcs_statuses())
        .unwrap();
    let old = catalog.vcs_status(id).unwrap();
    assert_eq!(old.state, VcsState::Modified);
    for limits in [
        GitLimits {
            max_paths: 1,
            ..GitLimits::default()
        },
        GitLimits {
            max_output_bytes: 6,
            ..GitLimits::default()
        },
    ] {
        let result = wait(&mut provider(limits).start(repo.0.clone()).unwrap());
        assert!(matches!(
            result,
            Err(GitError::PathBudget | GitError::OutputBudget)
        ));
        if let Ok(snapshot) = result {
            catalog
                .replace_vcs_statuses(snapshot.into_vcs_statuses())
                .unwrap();
        }
        assert_eq!(catalog.vcs_status(id), Some(old.clone()));
    }
    repo.write("asset", b"base");
    assert!(catalog
        .replace_vcs_statuses(repo.snapshot().into_vcs_statuses())
        .unwrap());
    assert_eq!(catalog.vcs_status(id).unwrap().state, VcsState::Clean);
    assert!(!catalog
        .replace_vcs_statuses(repo.snapshot().into_vcs_statuses())
        .unwrap());
    assert!(catalog
        .replace_vcs_statuses([
            ("asset".into(), VcsRecord::default()),
            ("bad//path".into(), VcsRecord::default())
        ])
        .is_err());
    assert_eq!(catalog.vcs_status(id).unwrap().state, VcsState::Clean);
}
#[test]
fn cancellation_timeout_drop_concurrency_reap_actual_git() {
    let repo = Repo::new();
    repo.write("asset", b"base");
    repo.commit();
    repo.block_index();
    let p = provider(GitLimits {
        max_concurrent: 1,
        ..GitLimits::default()
    });
    let mut request = p.start(repo.0.clone()).unwrap();
    let pid = started(&request);
    assert!(matches!(
        p.clone().start(repo.0.clone()),
        Err(GitError::Busy)
    ));
    request.cancel();
    assert_eq!(wait(&mut request), Err(GitError::Cancelled));
    assert_eq!(p.active_requests(), 0);
    assert_reaped(pid);
    let mut request = p.start(repo.0.clone()).unwrap();
    let pid = started(&request);
    let before = Instant::now();
    assert!(request.poll().is_none());
    assert!(before.elapsed() < Duration::from_millis(100));
    drop(request);
    let deadline = Instant::now() + Duration::from_secs(5);
    while p.active_requests() != 0 {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    assert_reaped(pid);
    let p = provider(GitLimits {
        timeout: Duration::from_millis(200),
        ..GitLimits::default()
    });
    let mut request = p.start(repo.0.clone()).unwrap();
    let pid = started(&request);
    assert_eq!(wait(&mut request), Err(GitError::TimedOut));
    assert_eq!(p.active_requests(), 0);
    assert_reaped(pid);
}
#[test]
fn empty_repo_invalid_root_real_command_failure_and_invalid_limits() {
    let repo = Repo::new();
    assert!(repo.snapshot().entries.is_empty());
    fs::create_dir(repo.0.join("sub")).unwrap();
    assert!(matches!(
        wait(
            &mut provider(GitLimits::default())
                .start(repo.0.join("sub"))
                .unwrap()
        ),
        Err(GitError::Protocol(_))
    ));
    fs::write(repo.0.join(".git/index"), b"broken index bytes").unwrap();
    match wait(
        &mut provider(GitLimits::default())
            .start(repo.0.clone())
            .unwrap(),
    )
    .unwrap_err()
    {
        GitError::CommandFailed {
            command, stderr, ..
        } => {
            assert_eq!(command, "ls-files");
            assert!(!stderr.is_empty());
        }
        error => panic!("{error:?}"),
    }
    assert!(matches!(
        GitProvider::new(GitLimits {
            max_paths: 0,
            ..GitLimits::default()
        }),
        Err(GitError::InvalidLimits)
    ));
}

#[test]
fn ready_success_and_error_keep_capacity_until_consumed_or_dropped() {
    let repo = Repo::new();
    repo.write("asset", b"base");
    repo.commit();
    let p = provider(GitLimits {
        max_concurrent: 1,
        ..GitLimits::default()
    });
    for root in [repo.0.clone(), repo.0.join("does-not-exist")] {
        let mut request = p.start(root).unwrap();
        // Wait without consuming the ready result. Checking the channel inside
        // poll would consume it; let real Git finish and assert capacity stays.
        ready(&request);
        assert_eq!(p.active_requests(), 1);
        assert!(matches!(
            p.clone().start(repo.0.clone()),
            Err(GitError::Busy)
        ));
        let _ = wait(&mut request);
        assert_eq!(p.active_requests(), 0);
    }
    let request = p.start(repo.0.clone()).unwrap();
    ready(&request);
    assert!(matches!(
        p.clone().start(repo.0.clone()),
        Err(GitError::Busy)
    ));
    drop(request);
    let deadline = Instant::now() + Duration::from_secs(5);
    while p.active_requests() != 0 {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    let mut request = p.start(repo.0.clone()).unwrap();
    ready(&request);
    request.cancel();
    assert_eq!(wait(&mut request), Err(GitError::Cancelled));
    assert_eq!(p.active_requests(), 0);
}

#[test]
fn exact_budget_boundary_counts_rename_source_and_stderr() {
    let repo = Repo::new();
    repo.write("old", b"original");
    repo.commit();
    repo.git(&["mv", "old", "new"]);
    // ls-files emits new NUL (4 bytes); status emits R SP SP new NUL old NUL (11).
    let limits = GitLimits {
        max_output_bytes: 15,
        max_paths: 3,
        ..GitLimits::default()
    };
    let snapshot = wait(&mut provider(limits).start(repo.0.clone()).unwrap()).unwrap();
    assert_eq!(entry(&snapshot, "old").record.state, VcsState::Deleted);
    assert!(matches!(
        wait(
            &mut provider(GitLimits {
                max_paths: 2,
                ..limits
            })
            .start(repo.0.clone())
            .unwrap()
        ),
        Err(GitError::PathBudget)
    ));
    assert!(matches!(
        wait(
            &mut provider(GitLimits {
                max_output_bytes: 14,
                ..limits
            })
            .start(repo.0.clone())
            .unwrap()
        ),
        Err(GitError::OutputBudget)
    ));
    fs::write(repo.0.join(".git/index"), b"broken index bytes").unwrap();
    assert!(matches!(
        wait(
            &mut provider(GitLimits {
                max_output_bytes: 1,
                ..GitLimits::default()
            })
            .start(repo.0.clone())
            .unwrap()
        ),
        Err(GitError::OutputBudget)
    ));
}

#[test]
fn linked_worktree_root_is_supported() {
    let repo = Repo::new();
    repo.write("asset", b"base");
    repo.commit();
    let linked = repo.0.join("linked");
    repo.git(&["worktree", "add", "--detach", linked.to_str().unwrap()]);
    assert!(linked.join(".git").is_file());
    let snapshot = wait(
        &mut provider(GitLimits::default())
            .start(linked.clone())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(snapshot.root, linked.canonicalize().unwrap());
    assert_eq!(entry(&snapshot, "asset").record.state, VcsState::Clean);
}

#[test]
fn exact_token_publication_rejects_stale_and_keeps_omitted_paths_unknown() {
    use prism_editor_content::{ContentError, VcsSnapshotLimits};

    let repo = Repo::new();
    repo.write("tracked", b"base");
    repo.commit();
    let mut catalog = ContentCatalog::new();
    for (id, path) in [(1, "tracked"), (2, "absent")] {
        catalog
            .insert(
                AssetRecord::new(AssetId::from_raw(id), path, AssetKind::new("file").unwrap())
                    .unwrap(),
            )
            .unwrap();
    }
    let expected = catalog.snapshot_token();
    let snapshot = repo.snapshot();
    assert!(catalog
        .publish_vcs(
            &expected,
            snapshot.into_vcs_statuses(),
            VcsSnapshotLimits::default(),
        )
        .unwrap());
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)).unwrap().state,
        VcsState::Clean
    );
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(2)).unwrap().state,
        VcsState::Unknown
    );
    repo.write("tracked", b"changed");
    assert_eq!(
        catalog.publish_vcs(
            &expected,
            repo.snapshot().into_vcs_statuses(),
            VcsSnapshotLimits::default(),
        ),
        Err(ContentError::StaleCatalogSnapshot)
    );
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)).unwrap().state,
        VcsState::Clean
    );
    let expected = catalog.snapshot_token();
    assert!(catalog
        .publish_vcs(
            &expected,
            repo.snapshot().into_vcs_statuses(),
            VcsSnapshotLimits::default(),
        )
        .unwrap());
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)).unwrap().state,
        VcsState::Modified
    );
}
