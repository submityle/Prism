//! Catalog ownership and content-specific event routing for the native UI.
use super::*;
use alloc::sync::Arc;
use prism_editor_content::{
    thumbnail::{Thumbnail, ThumbnailKey, ThumbnailService},
    AssetRecord,
};
use prism_editor_ui::{BoundContentBrowser, ContentBrowserController};
use prism_editor_ui::{ContentBrowserAction, ContentLayout, ContentThumbnails};

/// Explicit host source identity. Implementations must be cheap, deterministic
/// and perform no file I/O. Revision comes from the host's source publication.
pub trait ContentThumbnailSource {
    /// Return an absolute source and authoritative revision, or no supported image.
    fn key(&self, asset: &AssetRecord, size: [u32; 2]) -> Option<ThumbnailKey>;
}

/// Genuine decoded image paired with reconciled logical layout and dock clip.
#[derive(Clone, Debug)]
pub struct DesktopContentImage {
    /// Authoritative host source identity.
    pub request: ThumbnailKey,
    /// Shared straight-alpha sRGB RGBA8 pixels.
    pub image: Arc<Thumbnail>,
    /// Actual keyed Loom image rectangle, preserving source aspect ratio.
    pub rect: Rect,
    /// Effective content dock/window clip.
    pub clip: Rect,
    /// Effective opacity of this native content visual.
    pub opacity: f32,
}

struct Thumbnails {
    source: Box<dyn ContentThumbnailSource>,
    page: ContentThumbnails,
}

pub(super) struct NativeContent {
    pub catalog: ContentCatalog,
    pub controller: ContentBrowserController,
    pub view: BoundContentBrowser,
    page_size: usize,
    preferred_columns: usize,
    pub(super) dirty: bool,
    thumbnails: Option<Thumbnails>,
    pub images: Vec<DesktopContentImage>,
    pub scroll: f32,
    pub scroll_max: f32,
}

impl NativeContent {
    pub fn changed(&self) -> bool {
        self.catalog.generation() != self.view.generation()
    }

    pub fn rebuild(&mut self, height: f32, width: f32) -> Result<(), DesktopUiError> {
        if let ContentLayout::Grid { columns } = self.controller.layout() {
            let fitted = self
                .preferred_columns
                .min((width / 164.0).floor().max(1.0) as usize);
            if fitted != columns {
                self.controller.apply(
                    &mut self.catalog,
                    ContentBrowserAction::Layout(ContentLayout::Grid { columns: fitted }),
                )?;
                self.dirty = true;
            }
        }
        // Header/filter/navigation/footer retain their space; paging keeps all
        // currently requested rows reachable inside the clipped native dock.
        let rows = match self.controller.layout() {
            ContentLayout::List => ((height - 204.0).max(32.0) / 32.0) as usize,
            ContentLayout::Grid { columns } => {
                (((height - 204.0).max(280.0) / 280.0) as usize).max(1) * columns
            }
        };
        let rows = rows.clamp(1, MAX_VISIBLE_ROWS);
        if rows != self.page_size {
            self.controller
                .apply(&mut self.catalog, ContentBrowserAction::PageSize(rows))?;
            self.page_size = rows;
            self.dirty = true;
        }
        if self.dirty || self.changed() {
            self.view = self.controller.build(&mut self.catalog)?;
            if let Some(thumbnails) = &mut self.thumbnails {
                let mut keys = Vec::new();
                for id in self.controller.visible_assets() {
                    if let Some(key) = self
                        .catalog
                        .get(*id)
                        .and_then(|record| thumbnails.source.key(record, [128, 128]))
                    {
                        if key.asset != *id {
                            return Err(prism_editor_content::thumbnail::ThumbnailError::InvalidConfiguration.into());
                        }
                        keys.push(key);
                    }
                }
                thumbnails.page.sync_visible(&self.controller, keys)?;
                self.view = self
                    .controller
                    .build_with_thumbnails(&mut self.catalog, &thumbnails.page)?;
            }
            self.dirty = false;
        }
        Ok(())
    }

    pub fn poll(&mut self) -> bool {
        let changed = self.thumbnails.as_mut().is_some_and(|t| t.page.poll());
        self.dirty |= changed;
        changed
    }
}

pub(super) fn is_target(target: &str) -> bool {
    target.starts_with("content:")
}

impl EditorDesktopUi {
    /// Mounts an application-owned catalog in the native content dock. Source
    /// workers publish through `content_catalog_mut`; the next refresh checks
    /// catalog generation independently of the document's model revision.
    pub fn mount_content_browser(
        &mut self,
        mut catalog: ContentCatalog,
    ) -> Result<(), DesktopUiError> {
        if self.collections.fenced() {
            return Err(DesktopUiError::Validation(
                "Collections must finish saving before catalog replacement".into(),
            ));
        }
        self.collections = collections::NativeCollections::default();
        self.git.unmount();
        self.validation.uncheck();
        let mut controller = ContentBrowserController::default();
        let view = controller.build(&mut catalog)?;
        self.content = Some(NativeContent {
            catalog,
            controller,
            view,
            page_size: 0,
            preferred_columns: 4,
            dirty: true,
            thumbnails: None,
            images: Vec::new(),
            scroll: 0.0,
            scroll_max: 0.0,
        });
        if self.focus.as_ref().is_some_and(|f| is_target(&f.target)) {
            self.focus = None;
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(())
    }

    /// The catalog currently mounted in the content dock.
    #[must_use]
    pub fn content_catalog(&self) -> Option<&ContentCatalog> {
        self.content.as_ref().map(|c| &c.catalog)
    }

    /// Publish source/Git results on the native UI thread, then call refresh
    /// (the native event loop does this on every wake). Unchanged generations
    /// keep the retained draw list and input drafts unchanged.
    pub fn content_catalog_mut(&mut self) -> Option<&mut ContentCatalog> {
        self.content.as_mut().map(|c| &mut c.catalog)
    }

    /// Independent query and asset selection; never editor object selection.
    #[must_use]
    pub fn content_browser(&self) -> Option<&ContentBrowserController> {
        self.content.as_ref().map(|c| &c.controller)
    }

    /// Install explicit source/revision bindings and a shared bounded decoder.
    /// Replacing it cancels page requests and revokes displayed images.
    pub fn mount_content_thumbnails(
        &mut self,
        service: ThumbnailService,
        source: Box<dyn ContentThumbnailSource>,
    ) -> Result<(), DesktopUiError> {
        let content = self
            .content
            .as_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        content.thumbnails = Some(Thumbnails {
            source,
            page: ContentThumbnails::new(service),
        });
        content.images.clear();
        content.dirty = true;
        self.invalidated = true;
        self.refresh()?;
        Ok(())
    }

    /// Publish changed host source receipts even without a catalog metadata change.
    /// The host must wake the native loop after publication on its UI thread.
    pub fn invalidate_content_thumbnails(&mut self) {
        if let Some(content) = &mut self.content {
            content.images.clear();
            content.dirty = true;
            self.invalidated = true;
        }
    }

    /// Cancel page work and release displayed pixels. The shared service cache
    /// stays host-owned; project teardown may also call `service.clear()`.
    pub fn unmount_content_thumbnails(&mut self) -> Result<(), DesktopUiError> {
        if let Some(content) = &mut self.content {
            content.thumbnails = None;
            content.images.clear();
            content.dirty = true;
            self.invalidated = true;
        }
        self.refresh()?;
        Ok(())
    }

    /// Nonblocking native-loop scheduling hint; refresh polls actual results.
    #[must_use]
    pub fn content_thumbnails_pending(&self) -> bool {
        self.content
            .as_ref()
            .and_then(|c| c.thumbnails.as_ref())
            .is_some_and(|t| t.page.needs_poll())
    }

    /// Successfully reconciled image anchors in paint order. Refresh before use.
    #[must_use]
    pub fn content_images(&self) -> &[DesktopContentImage] {
        self.content.as_ref().map_or(&[], |c| c.images.as_slice())
    }

    /// Programmatic equivalent of independent browser input, including grid mode.
    pub fn content_action(&mut self, action: ContentBrowserAction) -> Result<(), DesktopUiError> {
        let preferred = match &action {
            ContentBrowserAction::Layout(ContentLayout::Grid { columns }) => Some(*columns),
            _ => None,
        };
        let content = self
            .content
            .as_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let persist = self.collections.mounted()
            && matches!(
                action,
                ContentBrowserAction::CreateCollection(_)
                    | ContentBrowserAction::RenameCollection { .. }
                    | ContentBrowserAction::DeleteCollection(_)
                    | ContentBrowserAction::AddSelectionToCollection(_)
                    | ContentBrowserAction::RemoveSelectionFromCollection(_)
            );
        let before = persist.then(|| content.controller.collections().to_vec());
        if let Err(error) = content.controller.apply(&mut content.catalog, action) {
            return self.reject(error.into());
        }
        content.dirty = true;
        content.scroll = 0.0;
        if let Some(columns) = preferred {
            content.preferred_columns = columns;
        }
        if before.is_some_and(|before| before != content.controller.collections()) {
            self.collections.changed(&content.controller);
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(())
    }

    pub(super) fn content_height(&self) -> f32 {
        if self.content.is_some() || self.panels.mounted() {
            (self.size.height * 0.48).min(400.0)
        } else {
            0.0
        }
    }

    pub(super) fn focus_content(&mut self, target: &str) -> Result<(), DesktopUiError> {
        let content = self
            .content
            .as_ref()
            .ok_or(DesktopUiError::InvalidProjection)?;
        if !content
            .view
            .bindings()
            .any(|(key, trigger)| key == target && trigger == UiTrigger::Commit)
        {
            return Err(DesktopUiError::InvalidProjection);
        }
        let query = content.controller.query();
        let buffer = match target {
            "content:search" => query.text.clone().unwrap_or_default(),
            "content:kind" => query
                .kind
                .as_ref()
                .map(|k| k.as_str().to_owned())
                .unwrap_or_default(),
            "content:tag" => query.tag.clone().unwrap_or_default(),
            "content:vcs" => query
                .vcs
                .map(|v| format!("{v:?}").to_ascii_lowercase())
                .unwrap_or_default(),
            "content:collection:create" => String::new(),
            _ if target.starts_with("content:collection:rename:") => {
                let id = target
                    .strip_prefix("content:collection:rename:")
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or(DesktopUiError::InvalidProjection)?;
                content
                    .controller
                    .collections()
                    .iter()
                    .find(|c| c.id() == id)
                    .ok_or(DesktopUiError::InvalidProjection)?
                    .name()
                    .to_owned()
            }
            _ => return Err(DesktopUiError::InvalidProjection),
        };
        self.focus = Some(Focus {
            target: target.to_owned(),
            buffer,
            preedit: String::new(),
            replace: true,
        });
        self.invalidated = true;
        Ok(())
    }

    pub(super) fn content_event(
        &mut self,
        target: &str,
        event: &UiEvent,
    ) -> Result<bool, DesktopUiError> {
        if target == "content:collections:retry" && matches!(event, UiEvent::Activate) {
            self.retry_content_collection_save();
            self.refresh()?;
            return Ok(true);
        }
        if target == "content:git:refresh" && matches!(event, UiEvent::Activate) {
            self.refresh_content_git();
            self.refresh()?;
            return Ok(true);
        }
        if target == "content:validate" && matches!(event, UiEvent::Activate) {
            self.validate_content(prism_editor_content::validation::ValidationOperation::Save)?;
            self.refresh()?;
            return Ok(true);
        }
        let content = self
            .content
            .as_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let before = (self.collections.mounted() && target.starts_with("content:collection:"))
            .then(|| content.controller.collections().to_vec());
        let result =
            content
                .controller
                .handle_event(&mut content.catalog, &content.view, target, event);
        if let Err(error) = result {
            return self.reject(error.into());
        }
        content.dirty = true;
        self.focus = None;
        self.error = None;
        if before.is_some_and(|before| before != content.controller.collections()) {
            self.collections.changed(&content.controller);
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }
}
