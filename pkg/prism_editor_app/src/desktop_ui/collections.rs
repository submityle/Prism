//! Startup loading and one serialized metadata writer, outside Loom builds.
use super::*;
use prism_editor_ui::{
    CollectionLoadReport, CollectionSaveReport, CollectionStorageError, CollectionStore,
    ContentBrowserController,
};
use std::sync::mpsc::{self, Receiver, TryRecvError};

// Lock contention is transient and may outlive a completed writer when another
// process is being created. Retry only Busy, on the worker, for at most 126 ms
// of backoff. Persistent contention remains a visible, explicitly retried error.
fn save_with_contention_backoff(
    store: &CollectionStore,
    controller: &ContentBrowserController,
) -> Result<CollectionSaveReport, CollectionStorageError> {
    for delay in [2, 4, 8, 16, 32, 32, 32] {
        match store.save(controller) {
            Err(CollectionStorageError::Busy) => {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
            result => return result,
        }
    }
    store.save(controller)
}

#[derive(Default)]
pub(super) struct NativeCollections {
    store: Option<CollectionStore>,
    active: Option<Receiver<Result<String, String>>>,
    queued: Option<ContentBrowserController>,
    failed: bool,
    pub message: String,
}

impl NativeCollections {
    pub fn failed(&self) -> bool {
        self.failed
    }
    pub fn mounted(&self) -> bool {
        self.store.is_some()
    }
    pub fn fenced(&self) -> bool {
        self.active.is_some() || self.queued.is_some() || self.failed
    }
    pub fn pending(&self) -> bool {
        self.active.is_some() || (self.queued.is_some() && !self.failed)
    }
    pub fn changed(&mut self, controller: &ContentBrowserController) {
        if self.store.is_some() {
            self.queued = Some(controller.clone());
            if !self.failed {
                self.message = "Collections saving".into();
            }
        }
    }
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        if let Some(receiver) = &self.active {
            let result = match receiver.try_recv() {
                Ok(result) => Some(result),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    Some(Err("Collection writer disconnected".into()))
                }
            };
            if let Some(result) = result {
                self.active = None;
                self.failed = result.is_err();
                self.message = result.unwrap_or_else(|error| error);
                changed = true;
            }
        }
        // An error requires explicit retry; never silently replace a bad file.
        if self.active.is_none()
            && !self.failed
            && let (Some(store), Some(controller)) = (&self.store, self.queued.take())
        {
            let store = store.clone();
            let (send, receive) = mpsc::sync_channel(1);
            match std::thread::Builder::new()
                .name("prism-collections".into())
                .spawn(move || {
                    let result = save_with_contention_backoff(&store, &controller)
                        .map(|report| {
                            report.durability_warning.map_or_else(
                                || "Collections saved".into(),
                                |kind| {
                                    format!("Collections published; directory durability: {kind}")
                                },
                            )
                        })
                        .map_err(|error| {
                            format!(
                                "Collection writer {}: {error:?}: {error}",
                                store.path().display()
                            )
                        });
                    let _ = send.send(result);
                }) {
                Ok(_) => self.active = Some(receive),
                Err(error) => {
                    self.failed = true;
                    self.message = format!("Collection writer: {error}");
                }
            }
            changed = true;
        }
        changed
    }
}

impl EditorDesktopUi {
    /// Explicit startup/project-switch I/O, before entering the native loop.
    /// Failed loads leave the current controller/binding untouched. Replacement
    /// waits for the previous writer, preserving queued local mutations.
    pub fn mount_content_collections(
        &mut self,
        store: CollectionStore,
    ) -> Result<CollectionLoadReport, DesktopUiError> {
        if self.collections.fenced() {
            return Err(DesktopUiError::Validation(
                "Collections must finish saving before project replacement".into(),
            ));
        }
        let content = self
            .content
            .as_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let mut candidate = content.controller.clone();
        let report = store
            .load_into(&mut candidate, &content.catalog)
            .map_err(|error| DesktopUiError::Validation(error.to_string()))?;
        content.controller = candidate;
        content.dirty = true;
        self.collections = NativeCollections {
            store: Some(store),
            message: format!(
                "Collections loaded; {} missing references",
                report.missing_members.len()
            ),
            ..Default::default()
        };
        self.invalidated = true;
        self.refresh()?;
        Ok(report)
    }
    /// Retry the current exact collection snapshot after a visible save error.
    pub fn retry_content_collection_save(&mut self) {
        if let Some(content) = &self.content {
            self.collections.failed = false;
            self.collections.changed(&content.controller);
            self.invalidated = true;
        }
    }
    /// Native wake hint; failed writes instead fence close until explicit retry.
    #[must_use]
    pub fn content_collections_pending(&self) -> bool {
        self.collections.pending()
    }
    /// Metadata persistence status, independent of document dirty state.
    #[must_use]
    pub fn content_collection_status(&self) -> &str {
        &self.collections.message
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_editor_ui::ContentBrowserAction;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant},
    };
    struct Project(PathBuf);
    impl Project {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "prism-app-collections-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }
        fn store(&self) -> CollectionStore {
            CollectionStore::for_project(&self.0).unwrap()
        }
    }
    impl Drop for Project {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn ui() -> EditorDesktopUi {
        let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
        ui.mount_content_browser(ContentCatalog::new()).unwrap();
        ui
    }
    fn settle(ui: &mut EditorDesktopUi) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while ui.content_collections_pending() {
            assert!(Instant::now() < deadline, "metadata write deadline");
            ui.refresh().unwrap();
            std::thread::yield_now();
        }
    }
    #[test]
    fn native_collection_mutations_persist_latest_and_fence_project_replacement() {
        let project = Project::new();
        let other = Project::new();
        let mut ui = ui();
        assert!(
            ui.mount_content_collections(project.store())
                .unwrap()
                .missing_file
        );
        let rect = ui
            .hits
            .iter()
            .find(|h| h.target == "content:collection:create")
            .unwrap()
            .rect;
        assert!(rect.size.width > 0.0 && rect.size.height > 0.0);
        ui.click(
            rect.left() + 3.0,
            rect.top() + rect.size.height * 0.5,
            SelectionMode::Replace,
        )
        .unwrap();
        ui.text("first").unwrap();
        ui.key(DesktopKey::Enter).unwrap();
        assert!(ui.focus.is_none());
        assert_eq!(
            ui.content_browser().unwrap().collections()[0].name(),
            "first"
        );
        assert!(ui.has_uncommitted_edit());
        assert!(ui.mount_content_collections(other.store()).is_err());
        ui.content_action(ContentBrowserAction::RenameCollection {
            id: 1,
            name: "latest".into(),
        })
        .unwrap();
        settle(&mut ui);
        assert!(
            !ui.has_uncommitted_edit(),
            "successful metadata writes clear only their fence: status={:?}, active={}, queued={}, failed={}, focus={}, controller_edit={}, panels={}",
            ui.content_collection_status(),
            ui.collections.active.is_some(),
            ui.collections.queued.is_some(),
            ui.collections.failed,
            ui.focus.is_some(),
            ui.controller.has_active_edit(),
            ui.panels.fenced(),
        );
        assert_eq!(ui.content_collection_status(), "Collections saved");
        let mut restored = ContentBrowserController::default();
        project
            .store()
            .load_into(&mut restored, ui.content_catalog().unwrap())
            .unwrap();
        assert_eq!(restored.collections()[0].name(), "latest");
        assert_eq!(ui.controller().document_history_lengths(), (0, 0));
        assert!(!ui.controller().is_dirty());
        assert!(!ui.refresh().unwrap());
        ui.mount_content_collections(other.store()).unwrap();
        assert!(ui.content_browser().unwrap().collections().is_empty());
    }
    #[test]
    fn real_collection_lock_contention_is_bounded_and_preserves_latest_for_retry() {
        let project = Project::new();
        let mut ui = ui();
        ui.mount_content_collections(project.store()).unwrap();
        ui.content_action(ContentBrowserAction::CreateCollection("published".into()))
            .unwrap();
        settle(&mut ui);
        let store = project.store();
        let previous = std::fs::read(store.path()).unwrap();
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store.path().with_file_name("content-collections.lock"))
            .unwrap();
        lock.lock().unwrap();
        assert_eq!(
            store.save(ui.content_browser().unwrap()),
            Err(CollectionStorageError::Busy),
        );
        ui.content_action(ContentBrowserAction::RenameCollection {
            id: 1,
            name: "contended".into(),
        })
        .unwrap();
        ui.content_action(ContentBrowserAction::RenameCollection {
            id: 1,
            name: "latest after contention".into(),
        })
        .unwrap();
        settle(&mut ui);
        assert!(ui.collections.failed());
        assert!(ui.has_uncommitted_edit());
        assert!(ui.content_collection_status().contains("Busy"));
        assert_eq!(std::fs::read(store.path()).unwrap(), previous);
        assert!(!ui.refresh().unwrap(), "no unbounded per-frame retry");
        drop(lock);
        ui.retry_content_collection_save();
        settle(&mut ui);
        assert!(!ui.has_uncommitted_edit());
        let mut restored = ContentBrowserController::default();
        store
            .load_into(&mut restored, ui.content_catalog().unwrap())
            .unwrap();
        assert_eq!(restored.collections()[0].name(), "latest after contention");

        // A real external lock released during backoff must allow this exact
        // snapshot to publish without turning contention into a terminal fence.
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(store.path().with_file_name("content-collections.lock"))
            .unwrap();
        lock.lock().unwrap();
        let controller = restored.clone();
        let worker_store = store.clone();
        let worker =
            std::thread::spawn(move || save_with_contention_backoff(&worker_store, &controller));
        std::thread::sleep(Duration::from_millis(20));
        drop(lock);
        worker.join().unwrap().unwrap();
    }
    #[test]
    fn collection_writes_overlap_real_git_processes_without_losing_latest_snapshot() {
        let project = Project::new();
        let mut ui = ui();
        ui.mount_content_collections(project.store()).unwrap();
        ui.content_action(ContentBrowserAction::CreateCollection("initial".into()))
            .unwrap();
        let running = alloc::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let worker_running = running.clone();
        let worker = std::thread::spawn(move || {
            let mut spawned = 0;
            while worker_running.load(Ordering::Acquire) {
                let result = std::process::Command::new("git")
                    .arg("--version")
                    .output()
                    .unwrap();
                assert!(result.status.success(), "{result:?}");
                spawned += 1;
            }
            spawned
        });
        let mut failure = None;
        for serial in 0..100 {
            ui.content_action(ContentBrowserAction::RenameCollection {
                id: 1,
                name: format!("publication-{serial}"),
            })
            .unwrap();
            settle(&mut ui);
            if ui.collections.failed() {
                failure = Some(ui.content_collection_status().to_owned());
                break;
            }
        }
        running.store(false, Ordering::Release);
        let spawned = worker.join().unwrap();
        assert!(spawned > 0, "must exercise actual Git process creation");
        assert_eq!(failure, None, "concurrent publication failed");
        assert!(!ui.has_uncommitted_edit());
        let mut restored = ContentBrowserController::default();
        project
            .store()
            .load_into(&mut restored, ui.content_catalog().unwrap())
            .unwrap();
        assert_eq!(restored.collections()[0].name(), "publication-99");
    }
    #[test]
    fn failed_collection_load_and_save_preserve_bytes_and_require_explicit_retry() {
        let project = Project::new();
        let mut ui = ui();
        ui.mount_content_collections(project.store()).unwrap();
        let path = project.store().path();
        std::fs::write(&path, b"broken file").unwrap();
        ui.content_action(ContentBrowserAction::CreateCollection("recoverable".into()))
            .unwrap();
        settle(&mut ui);
        assert!(ui.has_uncommitted_edit());
        assert!(ui.content_collection_status().contains("corrupt"));
        assert_eq!(std::fs::read(&path).unwrap(), b"broken file");
        assert!(
            !ui.refresh().unwrap(),
            "terminal error must not retry every frame"
        );
        std::fs::rename(&path, path.with_extension("backup")).unwrap();
        ui.retry_content_collection_save();
        ui.refresh().unwrap();
        settle(&mut ui);
        assert!(!ui.has_uncommitted_edit());
        let broken = Project::new();
        let store = broken.store();
        let mut empty = ContentBrowserController::default();
        store
            .load_into(&mut empty, ui.content_catalog().unwrap())
            .unwrap();
        std::fs::write(store.path(), b"unsupported").unwrap();
        assert!(ui.mount_content_collections(store.clone()).is_err());
        assert_eq!(
            ui.content_browser().unwrap().collections()[0].name(),
            "recoverable"
        );
        ui.content_action(ContentBrowserAction::RenameCollection {
            id: 1,
            name: "old binding".into(),
        })
        .unwrap();
        settle(&mut ui);
        assert_eq!(std::fs::read(store.path()).unwrap(), b"unsupported");
    }
}
