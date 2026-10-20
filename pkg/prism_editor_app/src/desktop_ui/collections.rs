//! Startup loading and one serialized metadata writer, outside Loom builds.
use super::*;
use prism_editor_ui::{CollectionLoadReport, CollectionStore, ContentBrowserController};
use std::sync::mpsc::{self, Receiver, TryRecvError};

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
                    let result = store
                        .save(&controller)
                        .map(|report| {
                            report.durability_warning.map_or_else(
                                || "Collections saved".into(),
                                |kind| {
                                    format!("Collections published; directory durability: {kind}")
                                },
                            )
                        })
                        .map_err(|error| error.to_string());
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
            "successful metadata writes clear only their fence"
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
