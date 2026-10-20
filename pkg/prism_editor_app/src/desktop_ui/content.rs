//! Catalog ownership and content-specific event routing for the native UI.
use super::*;
use prism_editor_ui::{BoundContentBrowser, ContentBrowserController};

pub(super) struct NativeContent {
    pub catalog: ContentCatalog,
    pub controller: ContentBrowserController,
    pub view: BoundContentBrowser,
    page_size: usize,
    dirty: bool,
}

impl NativeContent {
    pub fn changed(&self) -> bool {
        self.catalog.generation() != self.view.generation()
    }

    pub fn rebuild(&mut self, height: f32) -> Result<(), DesktopUiError> {
        // Header/filter/navigation/footer retain their space; paging keeps all
        // currently requested rows reachable inside the clipped native dock.
        let rows = ((height - 204.0).max(32.0) / 32.0) as usize;
        let rows = rows.clamp(1, MAX_VISIBLE_ROWS);
        if rows != self.page_size {
            self.controller.apply(
                &mut self.catalog,
                prism_editor_ui::ContentBrowserAction::PageSize(rows),
            )?;
            self.page_size = rows;
            self.dirty = true;
        }
        if self.dirty || self.changed() {
            self.view = self.controller.build(&mut self.catalog)?;
            self.dirty = false;
        }
        Ok(())
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
        let mut controller = ContentBrowserController::default();
        let view = controller.build(&mut catalog)?;
        self.content = Some(NativeContent {
            catalog,
            controller,
            view,
            page_size: 0,
            dirty: true,
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

    pub(super) fn content_height(&self) -> f32 {
        if self.content.is_some() {
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
        let content = self
            .content
            .as_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
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
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }
}
