//! One request plus one latest intent; cancelled requests are drained in place.
use super::*;
use alloc::sync::Arc;
use prism_editor_content::{
    git::{GitError, GitProvider, GitRequest},
    ContentSnapshotToken, VcsSnapshotLimits,
};
use std::path::PathBuf;

/// State of the current native Git refresh, separate from document history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentGitPhase {
    /// No project/provider is mounted.
    Unmounted,
    /// Latest intent waits for a superseded request to terminate.
    Queued,
    /// Another consumer holds the shared provider's capacity.
    Waiting,
    /// A worker is acquiring a complete snapshot.
    Running,
    /// A complete snapshot was published (possibly identical).
    Published {
        /// Whether publication changed catalog metadata.
        changed: bool,
    },
    /// Acquisition or atomic publication failed; prior metadata is retained.
    Failed,
    /// Explicit cancellation revoked publication authority.
    Cancelled,
}

/// Bounded visible status and exact provider diagnostics for the latest intent.
pub struct ContentGitStatus {
    /// Current request phase.
    pub phase: ContentGitPhase,
    /// Display text bounded to 512 UTF-8 bytes.
    pub message: String,
    /// Exact successful stderr; bounded by the mounted provider's output budget.
    pub diagnostics: Vec<u8>,
    /// Original acquisition error, including exact non-UTF-8 stderr on failures.
    pub error: Option<GitError>,
}

struct Active {
    request: GitRequest,
    epoch: Arc<()>,
    root: PathBuf,
    token: ContentSnapshotToken,
}

pub(super) struct NativeGit {
    binding: Option<(GitProvider, PathBuf)>,
    epoch: Arc<()>,
    active: Option<Active>,
    wanted: bool,
    pub status: ContentGitStatus,
}

impl Default for NativeGit {
    fn default() -> Self {
        Self {
            binding: None,
            epoch: Arc::new(()),
            active: None,
            wanted: false,
            status: ContentGitStatus {
                phase: ContentGitPhase::Unmounted,
                message: String::new(),
                diagnostics: Vec::new(),
                error: None,
            },
        }
    }
}

impl NativeGit {
    fn revoke(&mut self) {
        self.epoch = Arc::new(());
        self.wanted = false;
        if let Some(active) = &self.active {
            active.request.cancel();
        }
    }

    pub fn unmount(&mut self) {
        self.revoke();
        self.binding = None;
        self.set(ContentGitPhase::Unmounted, "Git unmounted".into());
    }

    fn set(&mut self, phase: ContentGitPhase, message: String) {
        let message = message
            .chars()
            .scan(0usize, |bytes, c| {
                *bytes += c.len_utf8();
                (*bytes <= 512).then_some(if c.is_control() { ' ' } else { c })
            })
            .collect();
        self.status = ContentGitStatus {
            phase,
            message,
            diagnostics: Vec::new(),
            error: None,
        };
    }

    fn refresh(&mut self) {
        self.revoke();
        if self.binding.is_some() {
            self.wanted = true;
            self.set(ContentGitPhase::Queued, "Git refresh queued".into());
        }
    }

    pub fn pending(&self) -> bool {
        self.active.is_some() || self.wanted
    }

    pub fn mounted(&self) -> bool {
        self.binding.is_some()
    }

    pub fn poll(&mut self, mut catalog: Option<&mut ContentCatalog>) -> bool {
        let mut changed = false;
        if let Some(active) = &mut self.active {
            if let Some(result) = active.request.poll() {
                let active = self.active.take().expect("polled active request");
                let current = Arc::ptr_eq(&active.epoch, &self.epoch)
                    && self
                        .binding
                        .as_ref()
                        .is_some_and(|(_, root)| root == &active.root);
                if current {
                    changed = true;
                    match result {
                        Ok(snapshot) => {
                            let diagnostics = snapshot.diagnostics.clone();
                            let result = catalog.as_deref_mut().map(|catalog| {
                                catalog.publish_vcs(
                                    &active.token,
                                    snapshot.into_vcs_statuses(),
                                    VcsSnapshotLimits::default(),
                                )
                            });
                            match result {
                                Some(Ok(updated)) => {
                                    self.set(
                                        ContentGitPhase::Published { changed: updated },
                                        format!(
                                            "Git refreshed{}",
                                            if updated { "" } else { " (unchanged)" }
                                        ),
                                    );
                                    if !diagnostics.is_empty() {
                                        self.status.message.push_str(": ");
                                        for byte in diagnostics.iter().take(100) {
                                            for c in core::ascii::escape_default(*byte) {
                                                self.status.message.push(char::from(c));
                                            }
                                        }
                                    }
                                    self.status.diagnostics = diagnostics;
                                }
                                Some(Err(error)) => self.set(
                                    ContentGitPhase::Failed,
                                    format!("Git publication rejected: {error}"),
                                ),
                                None => self.set(
                                    ContentGitPhase::Failed,
                                    "Git catalog was unmounted".into(),
                                ),
                            }
                        }
                        Err(error) => {
                            self.set(ContentGitPhase::Failed, bounded_error(&error));
                            self.status.error = Some(error);
                        }
                    }
                }
            } else {
                return false;
            }
        }
        if self.wanted
            && self.active.is_none()
            && let (Some((provider, root)), Some(catalog)) = (&self.binding, catalog)
        {
            match provider.start(root.clone()) {
                Ok(request) => {
                    self.active = Some(Active {
                        request,
                        root: root.clone(),
                        epoch: self.epoch.clone(),
                        token: catalog.snapshot_token(),
                    });
                    self.wanted = false;
                    self.set(ContentGitPhase::Running, "Git refreshing".into());
                    changed = true;
                }
                Err(GitError::Busy) => {
                    if self.status.phase != ContentGitPhase::Waiting {
                        self.set(
                            ContentGitPhase::Waiting,
                            "Git waiting for shared worker capacity".into(),
                        );
                        changed = true;
                    }
                }
                Err(error) => {
                    self.wanted = false;
                    self.set(ContentGitPhase::Failed, bounded_error(&error));
                    self.status.error = Some(error);
                    changed = true;
                }
            }
        }
        changed
    }
}

fn bounded_error(error: &GitError) -> String {
    use core::fmt::Write;
    struct Text(String);
    impl Write for Text {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            for c in text.chars() {
                if self.0.len() + c.len_utf8() > 512 {
                    return Err(fmt::Error);
                }
                self.0.push(if c.is_control() { ' ' } else { c });
            }
            Ok(())
        }
    }
    let mut output = Text(String::new());
    let _ = write!(output, "{error}");
    output.0
}

impl EditorDesktopUi {
    /// Bind an explicit absolute worktree root without UI-thread filesystem I/O.
    /// Remounting cancels and drains old work before launching the latest intent.
    pub fn mount_content_git(
        &mut self,
        provider: GitProvider,
        root: PathBuf,
    ) -> Result<(), DesktopUiError> {
        if self.content.is_none() || !root.is_absolute() {
            return Err(DesktopUiError::InvalidProjection);
        }
        self.git.revoke();
        self.git.binding = Some((provider, root));
        self.refresh_content_git();
        Ok(())
    }

    /// Immediately revoke older ready/running results, retaining only this intent.
    pub fn refresh_content_git(&mut self) {
        self.git.refresh();
        self.invalidated = true;
    }

    /// Cancel without blocking; the native loop continues draining the handle.
    pub fn cancel_content_git(&mut self) {
        self.git.revoke();
        self.git
            .set(ContentGitPhase::Cancelled, "Git refresh cancelled".into());
        self.invalidated = true;
    }

    /// Remove the binding and drain any revoked request in subsequent wakes.
    pub fn unmount_content_git(&mut self) {
        self.git.unmount();
        self.invalidated = true;
    }

    /// Latest status, including exact diagnostics; never document history state.
    #[must_use]
    pub fn content_git_status(&self) -> &ContentGitStatus {
        &self.git.status
    }

    /// True while work, cancellation drainage or shared-capacity retry is pending.
    #[must_use]
    pub fn content_git_pending(&self) -> bool {
        self.git.pending()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use prism_editor_content::{git::GitLimits, AssetId, AssetKind, AssetRecord, VcsState};
    use std::{
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant},
    };

    struct Repo(PathBuf);
    impl Repo {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "prism-native-git-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            let repo = Self(root);
            repo.git(&["init", "-q"]);
            repo.git(&["config", "user.name", "App Test"]);
            repo.git(&["config", "user.email", "app-test@example.invalid"]);
            std::fs::write(repo.0.join("asset.png"), "real tracked source").unwrap();
            repo.git(&["add", "asset.png"]);
            repo.git(&["commit", "-qm", "source"]);
            repo
        }
        fn git(&self, args: &[&str]) {
            let output = Command::new("git")
                .arg("-C")
                .arg(&self.0)
                .args(args)
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
        }
        fn catalog(&self) -> ContentCatalog {
            let mut catalog = ContentCatalog::new();
            catalog
                .insert(
                    AssetRecord::new(
                        AssetId::from_raw(1),
                        "asset.png",
                        AssetKind::new("image").unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
            catalog
        }
        fn ui(&self, provider: GitProvider) -> EditorDesktopUi {
            let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
            ui.mount_content_browser(self.catalog()).unwrap();
            ui.mount_content_git(provider, self.0.clone()).unwrap();
            ui
        }
    }
    impl Drop for Repo {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn wait(mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready() {
            assert!(Instant::now() < deadline, "native Git deadline");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn settle(ui: &mut EditorDesktopUi) {
        wait(|| {
            ui.refresh().unwrap();
            !ui.content_git_pending()
        });
    }
    fn provider() -> GitProvider {
        GitProvider::new(GitLimits {
            max_concurrent: 1,
            ..Default::default()
        })
        .unwrap()
    }
    fn state(ui: &EditorDesktopUi) -> VcsState {
        ui.content_catalog()
            .unwrap()
            .vcs_status(AssetId::from_raw(1))
            .unwrap()
            .state
    }

    #[test]
    fn real_native_git_refresh_drains_ready_epoch_and_preserves_history() {
        let repo = Repo::new();
        let provider = provider();
        let mut ui = repo.ui(provider.clone());
        settle(&mut ui);
        assert_eq!(state(&ui), VcsState::Clean);
        let before = ui.controller().revision();
        ui.refresh_content_git();
        ui.refresh().unwrap();
        wait(|| ui.git.active.as_ref().unwrap().request.is_ready());
        // A ready clean result occupies the only slot. Replace it before poll.
        std::fs::write(repo.0.join("asset.png"), "new bytes").unwrap();
        ui.refresh_content_git();
        settle(&mut ui);
        assert_eq!(state(&ui), VcsState::Modified);
        assert_eq!(
            provider.active_requests(),
            0,
            "superseded ready permits must drain"
        );
        let generation = ui.content_catalog().unwrap().generation();
        ui.refresh_content_git();
        settle(&mut ui);
        assert_eq!(
            ui.content_git_status().phase,
            ContentGitPhase::Published { changed: false }
        );
        assert_eq!(ui.content_catalog().unwrap().generation(), generation);
        assert_eq!(ui.controller().revision(), before);
        assert_eq!(ui.controller().document_history_lengths(), (0, 0));
        assert!(!ui.controller().is_dirty());
        assert!(!ui.refresh().unwrap());
        assert!(ui.draw_list().commands().iter().any(|cmd| matches!(cmd, DrawCommand::Glyph(g) if ui.draw_list().text(g.text).contains("Git refreshed"))));
    }

    #[test]
    fn real_native_git_catalog_and_root_replacement_cancel_and_failure_are_fenced() {
        let repo = Repo::new();
        let provider = provider();
        let mut ui = repo.ui(provider.clone());
        ui.refresh().unwrap();
        wait(|| ui.git.active.as_ref().unwrap().request.is_ready());
        ui.content_catalog_mut()
            .unwrap()
            .insert(
                AssetRecord::new(
                    AssetId::from_raw(2),
                    "new.png",
                    AssetKind::new("image").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        settle(&mut ui);
        assert_eq!(ui.content_git_status().phase, ContentGitPhase::Failed);
        assert_eq!(state(&ui), VcsState::Unknown);
        ui.refresh_content_git();
        settle(&mut ui);
        assert_eq!(state(&ui), VcsState::Clean);
        let token = ui.content_catalog().unwrap().snapshot_token();
        ui.refresh_content_git();
        ui.refresh().unwrap();
        wait(|| ui.git.active.as_ref().unwrap().request.is_ready());
        ui.mount_content_git(provider.clone(), repo.0.join("not-a-repository"))
            .unwrap();
        settle(&mut ui);
        assert_eq!(ui.content_git_status().phase, ContentGitPhase::Failed);
        assert!(ui.content_git_status().error.is_some());
        assert!(ui.content_catalog().unwrap().matches_snapshot(&token));
        ui.mount_content_git(provider.clone(), repo.0.clone())
            .unwrap();
        ui.refresh().unwrap();
        ui.cancel_content_git();
        settle(&mut ui);
        assert_eq!(ui.content_git_status().phase, ContentGitPhase::Cancelled);
        assert!(ui.content_catalog().unwrap().matches_snapshot(&token));
        ui.mount_content_git(provider.clone(), repo.0.clone())
            .unwrap();
        ui.refresh().unwrap();
        // Session remount cancels the request even if the caller reused a clone.
        let same_catalog = ui.content_catalog().unwrap().clone();
        ui.mount_content_browser(same_catalog).unwrap();
        settle(&mut ui);
        assert_eq!(ui.content_git_status().phase, ContentGitPhase::Unmounted);
        assert_eq!(provider.active_requests(), 0);
    }

    #[test]
    fn shared_busy_is_visible_and_retries_without_extra_provider() {
        let repo = Repo::new();
        let provider = provider();
        let mut held = provider.start(repo.0.clone()).unwrap();
        wait(|| held.is_ready());
        let mut ui = repo.ui(provider.clone());
        ui.refresh().unwrap();
        assert_eq!(ui.content_git_status().phase, ContentGitPhase::Waiting);
        assert!(ui.content_git_pending());
        assert!(
            !ui.refresh().unwrap(),
            "unchanged capacity wait retains the tree"
        );
        assert!(held.poll().unwrap().is_ok());
        settle(&mut ui);
        assert_eq!(state(&ui), VcsState::Clean);
    }
}
