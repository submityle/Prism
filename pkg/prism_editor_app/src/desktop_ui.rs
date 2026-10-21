//! Platform-independent retained editor `UI`. No window, `GPU`, or native-event
//! acceptance is implied by this adapter. All input boxes come from `Loom`'s
//! actual retained layout, in logical pixels.

use alloc::collections::BTreeMap;
use core::fmt;

use prism_editor_content::ContentCatalog;
use prism_editor_core::SelectionItem;
use prism_editor_ui::{
    inspector_view_bound, outliner_panel_bound, ActionBindings, ActionError, ActionValue,
    ContentBrowserError, EditorAction, InspectorViewProps, OutlinerViewProps, SelectionMode,
    UiBuildError, UiEvent, UiTrigger, MAX_ACTION_TEXT_BYTES, MAX_VISIBLE_ROWS,
};
use prism_reflect::{reflect_path, ParsedPath};
use prism_ui::style::{Class, Color, Keyword, StyleProp, StyleSheet, StyleValue};
use prism_ui::{Backend, BackendId, BackendOp, Element, ElementKind, Key, PaintStyle, Ui};
use prism_ui_layout::{AvailableSpace, Point, Rect, Size};
use prism_ui_render_backend::draw::{DrawCommand, DrawList, LayerCmd};
use prism_ui_render_backend::scene::RetainedScene;

use crate::controller::{ControllerError, ControllerOutcome, EditorController};
#[cfg(feature = "desktop")]
use crate::scene_authoring::{SCENE_ADD_TRIANGLE_COMMAND, SCENE_DELETE_SELECTION_COMMAND};

const ROW_HEIGHT: f32 = 26.0;
const TOOLBAR_HEIGHT: f32 = 38.0;
const STATUS_HEIGHT: f32 = 28.0;
const PANEL_HEADER: f32 = 30.0;
const MAX_DIMENSION: f32 = 32_768.0;

mod authoring_panels;
mod collections;
mod content;
mod git;
mod panel_pointer;
mod panel_storage;
mod panels;
#[cfg(feature = "runtime-preview")]
mod runtime_preview;
mod sequence_panel;
#[cfg(feature = "runtime-preview")]
pub use runtime_preview::{DesktopFrameGraphImage, DesktopRuntimePreviewSource};
mod validation;
pub use authoring_panels::{DesktopFrameGraphPanel, DesktopGraphPanel, DesktopPrefabPanel};
use content::NativeContent;
pub use content::{ContentThumbnailSource, DesktopContentImage};
pub use git::{ContentGitPhase, ContentGitStatus};
pub use panel_storage::{DesktopPanelCancelOutcome, DesktopSequenceStorageLimits};
pub use panels::{
    DesktopPanel, DesktopPanelUpdate, DesktopPanelView, DesktopProfilerPanel, DesktopTimelineFrame,
    DesktopTimelinePanel, DesktopWorldPanel, DesktopWorldProjectBinding, DesktopWorldProjectError,
};
pub use sequence_panel::DesktopSequencePanel;
#[cfg(test)]
mod authoring_panel_tests;
#[cfg(test)]
mod content_tests;
#[cfg(test)]
mod panel_tests;
#[cfg(all(test, feature = "runtime-preview"))]
mod runtime_preview_tests;
#[cfg(test)]
mod sequence_panel_tests;

/// Logical keys, independent of `winit` or any native keyboard API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DesktopKey {
    /// Commit the focused field (not an active IME composition).
    Enter,
    /// Cancel the focused, uncommitted edit.
    Escape,
    /// Delete the last Unicode scalar, or the initially selected value.
    Backspace,
    /// Commit and advance to the next visible editable field.
    Tab,
    /// `Undo` the most recent document command, independent of selection.
    Undo,
    /// `Redo` the most recently undone document command, independent of selection.
    Redo,
}

/// Input/build failures are also presented through the `UI`'s error status.
#[derive(Debug)]
pub enum DesktopUiError {
    /// Controller rejected an action atomically.
    Controller(ControllerError),
    /// A semantic binding could not be constructed or dispatched.
    Action(ActionError),
    /// A generic bound panel exceeded its supported limits.
    Build(UiBuildError),
    /// The independent content browser rejected its event or catalog.
    Content(ContentBrowserError),
    /// Host thumbnail keys were invalid or exceeded a page budget.
    Thumbnail(prism_editor_content::thumbnail::ThumbnailError),
    /// Content rules or preparation gate rejected the requested operation.
    Validation(String),
    /// A native contribution, source or delivery rejected an operation.
    Panel(String),
    /// Dimensions must be finite and in 0..=32768 logical pixels.
    InvalidSize,
    /// Pointer coordinates must be finite.
    InvalidPointer,
    /// Text or composition would exceed the shared action text budget.
    TextLimit,
    /// Single-line editing does not accept control characters.
    ControlCharacter,
    /// A field disappeared or the rendered layout/tree no longer align.
    InvalidProjection,
}

impl fmt::Display for DesktopUiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Controller(e) => e.fmt(f),
            Self::Action(e) => e.fmt(f),
            Self::Build(e) => e.fmt(f),
            Self::Content(e) => e.fmt(f),
            Self::Thumbnail(e) => e.fmt(f),
            Self::Validation(e) | Self::Panel(e) => f.write_str(e),
            Self::InvalidSize => f.write_str("UI dimensions must be finite and within 0..=32768"),
            Self::InvalidPointer => f.write_str("pointer coordinates must be finite"),
            Self::TextLimit => f.write_str("field text/composition exceeds the 4096-byte budget"),
            Self::ControlCharacter => {
                f.write_str("single-line fields do not accept control characters")
            }
            Self::InvalidProjection => {
                f.write_str("field projection or retained layout is inconsistent")
            }
        }
    }
}

impl core::error::Error for DesktopUiError {}

impl From<ControllerError> for DesktopUiError {
    fn from(e: ControllerError) -> Self {
        Self::Controller(e)
    }
}
impl From<ActionError> for DesktopUiError {
    fn from(e: ActionError) -> Self {
        Self::Action(e)
    }
}
impl From<UiBuildError> for DesktopUiError {
    fn from(e: UiBuildError) -> Self {
        Self::Build(e)
    }
}
impl From<ContentBrowserError> for DesktopUiError {
    fn from(e: ContentBrowserError) -> Self {
        Self::Content(e)
    }
}
impl From<prism_editor_content::thumbnail::ThumbnailError> for DesktopUiError {
    fn from(e: prism_editor_content::thumbnail::ThumbnailError) -> Self {
        Self::Thumbnail(e)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Focus {
    target: String,
    buffer: String,
    preedit: String,
    replace: bool,
}

#[derive(Clone, Debug)]
struct Hit {
    target: String,
    trigger: UiTrigger,
    rect: Rect,
    content: bool,
    panel: bool,
}

/// Owns the editable model, retained UI, text focus and last rendered draw list.
/// Keep this value after native-window closure to retain unsaved changes.
pub struct EditorDesktopUi {
    controller: EditorController,
    ui: Ui<RetainedScene>,
    draw_list: DrawList,
    bindings: ActionBindings,
    hits: Vec<Hit>,
    viewport: Rect,
    outliner_rect: Rect,
    input_rect: Option<Rect>,
    size: Size<f32>,
    rendered_revision: Option<u64>,
    rendered_model_revision: Option<u64>,
    rendered_primary: Option<SelectionItem>,
    invalidated: bool,
    focus: Option<Focus>,
    error: Option<String>,
    outliner_start: usize,
    content: Option<NativeContent>,
    git: git::NativeGit,
    collections: collections::NativeCollections,
    validation: validation::NativeValidation,
    panels: panels::NativePanels,
    #[cfg(feature = "runtime-preview")]
    frame_graph_image: Option<DesktopFrameGraphImage>,
}

impl EditorDesktopUi {
    /// Builds the UI at a default 1280x800 logical-pixel size.
    pub fn new(controller: EditorController) -> Result<Self, DesktopUiError> {
        let mut result = Self {
            controller,
            ui: Ui::new(RetainedScene::new()).with_stylesheet(stylesheet()),
            draw_list: DrawList::new(),
            bindings: ActionBindings::new(),
            hits: Vec::new(),
            viewport: Rect::new(Point::ZERO, Size::ZERO),
            outliner_rect: Rect::new(Point::ZERO, Size::ZERO),
            input_rect: None,
            size: Size::new(1280.0, 800.0),
            rendered_revision: None,
            rendered_model_revision: None,
            rendered_primary: None,
            invalidated: true,
            focus: None,
            error: None,
            outliner_start: 0,
            content: None,
            git: git::NativeGit::default(),
            collections: collections::NativeCollections::default(),
            validation: validation::NativeValidation::default(),
            #[cfg(feature = "runtime-preview")]
            frame_graph_image: None,
            panels: panels::NativePanels::default(),
        };
        result.refresh()?;
        Ok(result)
    }

    /// Accepts zero dimensions for minimized windows; rejects invalid sizes
    /// without changing the last valid geometry.
    pub fn resize(&mut self, width: f32, height: f32) -> Result<(), DesktopUiError> {
        if !width.is_finite()
            || !height.is_finite()
            || !(0.0..=MAX_DIMENSION).contains(&width)
            || !(0.0..=MAX_DIMENSION).contains(&height)
        {
            return self.reject(DesktopUiError::InvalidSize);
        }
        let size = Size::new(width, height);
        if self.size != size {
            self.cancel_panel_pointer()?;
            self.size = size;
            self.invalidated = true;
        }
        self.refresh()?;
        Ok(())
    }

    /// Read the real model controller, including snapshot/save APIs.
    #[must_use]
    pub fn controller(&self) -> &EditorController {
        &self.controller
    }

    /// After direct model/registration/save operations call `refresh`. Access
    /// invalidates the `UI` and cancels drafts to prevent stale-object commits.
    pub fn controller_mut(&mut self) -> &mut EditorController {
        self.focus = None;
        self.invalidated = true;
        &mut self.controller
    }

    /// Acknowledges a saved model revision without cancelling an uncommitted
    /// field draft or `IME` composition. Stale acknowledgements remain errors.
    /// If presentation fails after acknowledgement, the confirmed model stays
    /// saved; `refresh` can retry the `UI` without repeating the disk operation.
    pub fn mark_saved(&mut self, model_revision: u64) -> Result<(), DesktopUiError> {
        if let Err(error) = self.controller.mark_saved(model_revision) {
            return self.reject(error.into());
        }
        self.refresh()?;
        Ok(())
    }

    /// Executes an action and updates the retained projection immediately.
    pub fn execute(&mut self, action: EditorAction) -> Result<ControllerOutcome, DesktopUiError> {
        #[cfg(feature = "desktop")]
        self.fence_scene_authoring(&action)?;
        match self.controller.execute(action) {
            Ok(outcome) => {
                if outcome.changed {
                    self.focus = None;
                }
                if self.error.take().is_some() {
                    self.invalidated = true;
                }
                self.refresh()?;
                Ok(outcome)
            }
            Err(error) => self.reject(error.into()),
        }
    }

    #[cfg(feature = "desktop")]
    fn fence_scene_authoring(&mut self, action: &EditorAction) -> Result<(), DesktopUiError> {
        if matches!(action, EditorAction::ExecuteCommand { command_id }
            if matches!(command_id.as_str(), SCENE_ADD_TRIANGLE_COMMAND | SCENE_DELETE_SELECTION_COMMAND))
            && (self.focus.is_some()
                || self.controller.has_active_edit()
                || self.panel_pointer_active())
        {
            return self.reject(DesktopUiError::Validation(
                "Commit or cancel the draft/IME and finish the gesture before adding or deleting objects"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Last refreshed, absolute logical-pixel draw commands; no `GPU` submission.
    #[must_use]
    pub fn draw_list(&self) -> &DrawList {
        &self.draw_list
    }

    /// Actual layout box of the `editor.viewport` Element for host geometry.
    #[must_use]
    pub fn viewport_rect(&self) -> Rect {
        self.viewport
    }

    /// Current logical dimensions, including zero dimensions when minimized.
    #[must_use]
    pub const fn size(&self) -> Size<f32> {
        self.size
    }

    /// Actual visible value box for the current text focus, in logical pixels.
    /// `None` disables `IME` when unfocused, clipped, or minimized. Like the draw
    /// list, this reflects the last successful refresh, never estimated metrics.
    #[must_use]
    pub fn text_input_rect(&self) -> Option<Rect> {
        self.focus.as_ref()?;
        self.input_rect
    }

    /// Returns true while a field focus owns an uncommitted draft or `IME`
    /// composition. This is intentionally conservative: unchanged focused
    /// values still block close/reload, and clipping/minimization does not
    /// make the draft safe to discard. Pending or failed collection writes
    /// also retain this fence until their exact snapshot is published.
    #[must_use]
    pub fn has_uncommitted_edit(&self) -> bool {
        self.focus.is_some()
            || self.controller.has_active_edit()
            || self.collections.fenced()
            || self.panels.fenced()
    }

    /// Stores a bounded, visible error. Call `refresh` before presenting it.
    pub fn show_error(&mut self, message: impl Into<String>) {
        let message = message.into();
        let message = truncate_bytes(&message, MAX_ACTION_TEXT_BYTES).to_owned();
        if self.error.as_ref() != Some(&message) {
            self.error = Some(message);
            self.invalidated = true;
        }
    }

    /// Exposes the complete bounded error independently of the status preview.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Reconciles after controller revision, catalog generation or local input changes.
    /// Returns false on an idle frame without rebuilding the retained tree.
    pub fn refresh(&mut self) -> Result<bool, DesktopUiError> {
        if self.collections.poll() {
            self.invalidated = true;
        }
        if self
            .git
            .poll(self.content.as_mut().map(|content| &mut content.catalog))
        {
            self.invalidated = true;
        }
        if self.content.as_mut().is_some_and(NativeContent::poll) {
            self.invalidated = true;
        }
        let revision = self.controller.revision();
        let content_changed =
            !self.panels.active() && self.content.as_ref().is_some_and(NativeContent::changed);
        if self
            .content
            .as_ref()
            .is_some_and(|content| self.validation.stale(&content.catalog))
        {
            self.validation.message = "Content changed; validate again".into();
        }
        if !self.invalidated && !content_changed && self.rendered_revision == Some(revision) {
            return Ok(false);
        }
        // Save acknowledgement only changes the UI revision, not the model
        // or field target. Keep drafts/composition alive across that refresh.
        if (self
            .rendered_model_revision
            .is_some_and(|old| old != self.controller.model_revision())
            || (self.rendered_revision.is_some()
                && self.rendered_primary.as_ref() != self.controller.selection().primary()))
            && !self
                .focus
                .as_ref()
                .is_some_and(|f| content::is_target(&f.target) || self.panels.is_target(&f.target))
        {
            self.focus = None;
        }
        if content_changed
            && self
                .focus
                .as_ref()
                .is_some_and(|f| content::is_target(&f.target))
        {
            self.focus = None;
        }
        let content_height = self.panel_body_height();
        if !self.panels.active()
            && let Some(content) = &mut self.content
        {
            content.rebuild(content_height, self.size.width)?;
        }
        self.panels.rebuild()?;
        let total = self.visible_outliner_count();
        let count = self.visible_rows();
        self.outliner_start = self.outliner_start.min(total.saturating_sub(count.max(1)));
        let (element, bindings) = self.build(count)?;
        self.ui.set_viewport_width(self.size.width);
        self.ui.update(&element);
        self.ui.compute_layout(Size::new(
            AvailableSpace::Definite(self.size.width),
            AvailableSpace::Definite(self.size.height),
        ));
        let mut geometry = Vec::new();
        self.ui
            .backend()
            .visit_layout(|id, rect| geometry.push((id, rect)));
        let mut semantics = Vec::new();
        collect_elements(&element, None, &mut semantics);
        if geometry.len() != semantics.len() {
            return Err(DesktopUiError::InvalidProjection);
        }
        let window = Rect::new(Point::ZERO, self.size);
        let scopes: BTreeMap<_, _> = semantics
            .iter()
            .zip(&geometry)
            .filter_map(|((element, _), (id, rect))| match element.explicit_key() {
                Some(Key::Str(key)) if key.starts_with("editor.") => {
                    Some((key.as_str(), (*id, intersection(window, *rect))))
                }
                _ => None,
            })
            .collect();
        let mut viewport = Rect::new(Point::ZERO, Size::ZERO);
        let mut input_rect = None;
        let mut hits = Vec::new();
        let mut images = Vec::new();
        #[cfg(feature = "runtime-preview")]
        {
            self.frame_graph_image = None;
        }
        let image_bindings: BTreeMap<_, _> = self
            .content
            .as_ref()
            .into_iter()
            .flat_map(|content| content.view.images())
            .map(|image| (image.visual_key.as_str(), image))
            .collect();
        let content_bindings: BTreeMap<_, _> = self
            .content
            .as_ref()
            .into_iter()
            .flat_map(|content| content.view.bindings())
            .collect();
        let panel_bindings: Vec<_> = self.panels.bindings().collect();
        for ((element, scope), (_, rect)) in semantics.iter().zip(&geometry) {
            if let Some(Key::Str(key)) = element.explicit_key() {
                if key == "editor.viewport" {
                    viewport = intersection(window, *rect);
                }
                let clip = scope
                    .and_then(|key| scopes.get(key))
                    .map_or(window, |(_, rect)| *rect);
                #[cfg(feature = "runtime-preview")]
                if key == "frame-graph:preview:image"
                    && let Some(panel) = self.panels.slot().and_then(|slot| {
                        (slot.host.as_ref() as &dyn std::any::Any)
                            .downcast_ref::<DesktopFrameGraphPanel>()
                    })
                    && let Some(view) = panel.view.as_ref().and_then(|v| v.preview())
                    && view.image().is_some()
                {
                    self.frame_graph_image = Some(DesktopFrameGraphImage {
                        view: view.clone(),
                        rect: *rect,
                        clip,
                    });
                }
                if let Some(image) = image_bindings.get(key.as_str()) {
                    images.push(DesktopContentImage {
                        request: image.request.clone(),
                        image: image.image.clone(),
                        rect: *rect,
                        clip,
                        opacity: 1.0,
                    });
                }
                if self
                    .focus
                    .as_ref()
                    .is_some_and(|focus| focus.target == *key)
                    && let Some(value) = element
                        .child_elements()
                        .iter()
                        .find(|child| child.class_names().iter().any(|c| is_input_value(c)))
                    && let Some(index) = semantics
                        .iter()
                        .position(|(el, _)| std::ptr::eq(*el, value))
                {
                    let rect = intersection(clip, geometry[index].1);
                    if rect.size.width > 0.0 && rect.size.height > 0.0 {
                        input_rect = Some(rect);
                    }
                }
                for binding in bindings.iter().filter(|b| b.target().as_str() == key) {
                    hits.push(Hit {
                        target: key.clone(),
                        trigger: binding.trigger(),
                        rect: intersection(clip, *rect),
                        content: false,
                        panel: false,
                    });
                }
                if !self.panels.active()
                    && let Some(&trigger) = content_bindings.get(key.as_str())
                {
                    hits.push(Hit {
                        target: key.clone(),
                        trigger,
                        rect: intersection(clip, *rect),
                        content: true,
                        panel: false,
                    });
                }
                if matches!(
                    key.as_str(),
                    "content:git:refresh" | "content:validate" | "content:collections:retry"
                ) {
                    hits.push(Hit {
                        target: key.clone(),
                        trigger: UiTrigger::Activate,
                        rect: intersection(clip, *rect),
                        content: true,
                        panel: false,
                    });
                }
                if let Some((_, trigger)) = panel_bindings
                    .iter()
                    .find(|(target, _)| *target == key.as_str())
                {
                    hits.push(Hit {
                        target: key.clone(),
                        trigger: *trigger,
                        rect: intersection(clip, *rect),
                        content: false,
                        panel: true,
                    });
                } else if key.starts_with("native:tab:") {
                    hits.push(Hit {
                        target: key.clone(),
                        trigger: UiTrigger::Activate,
                        rect: intersection(clip, *rect),
                        content: false,
                        panel: true,
                    });
                }
            }
        }
        let list = clipped_draw_list(self.ui.backend(), &scopes, window)?;
        let body = semantics
            .iter()
            .zip(&geometry)
            .find_map(|((el, _), (_, rect))| {
                matches!(el.explicit_key(),Some(Key::Str(k)) if k=="native:body")
                    .then_some(rect.size.height)
            })
            .unwrap_or(0.0);
        if self.panels.layout(content_height, body) {
            self.invalidated = true;
            return self.refresh();
        }
        self.panels.geometry = semantics
            .iter()
            .zip(&geometry)
            .filter_map(|((el, scope), (_, rect))| {
                let Some(Key::Str(key)) = el.explicit_key() else {
                    return None;
                };
                if !key.starts_with("timeline:") && key != "graph:canvas" {
                    return None;
                }
                let clip = scope
                    .and_then(|k| scopes.get(k))
                    .map_or(window, |(_, r)| *r);
                Some((key.clone(), (*rect, intersection(clip, *rect))))
            })
            .collect();
        if let Some(content) = &mut self.content {
            content.images = images;
        }
        if !self.panels.active()
            && let Some(content) = &mut self.content
        {
            let dock = scopes
                .get("editor.browser")
                .map_or(0.0, |(_, r)| r.size.height);
            content.scroll_max = semantics
                .iter()
                .zip(&geometry)
                .find_map(|((el, _), (_, r))| {
                    matches!(el.explicit_key(), Some(Key::Str(key)) if key == "content:browser")
                        .then_some((r.size.height - dock).max(0.0))
                })
                .unwrap_or(0.0);
            if content.scroll > content.scroll_max {
                content.scroll = content.scroll_max;
                self.invalidated = true;
                return self.refresh();
            }
        }
        self.hits = hits;
        self.bindings = bindings;
        self.viewport = viewport;
        self.outliner_rect = scopes
            .get("editor.outliner")
            .map_or(Rect::new(Point::ZERO, Size::ZERO), |(_, rect)| *rect);
        self.input_rect = input_rect;
        self.draw_list = list;
        self.rendered_revision = Some(revision);
        self.rendered_model_revision = Some(self.controller.model_revision());
        self.rendered_primary = self.controller.selection().primary().cloned();
        self.invalidated = false;
        self.panels.stale = false;
        Ok(true)
    }

    /// Routes to the deepest bound Element under the actual drawn location.
    /// Field clicks focus a draft; they do not mutate model data.
    pub fn click(&mut self, x: f32, y: f32, mode: SelectionMode) -> Result<bool, DesktopUiError> {
        if self.panel_pointer_active() {
            return self.reject(DesktopUiError::Panel(
                "Release or cancel the pointer before another action".into(),
            ));
        }
        if !x.is_finite() || !y.is_finite() {
            return self.reject(DesktopUiError::InvalidPointer);
        }
        let stale_content = self.content.as_ref().is_some_and(NativeContent::changed)
            && self
                .hits
                .iter()
                .rev()
                .find(|hit| contains(hit.rect, x, y))
                .is_some_and(|hit| hit.content);
        let stale_panel = self.panels.stale
            && self
                .hits
                .iter()
                .rev()
                .find(|hit| contains(hit.rect, x, y))
                .is_some_and(|hit| hit.panel);
        let refreshed = self.refresh()?;
        if stale_content {
            return Ok(refreshed);
        }
        if stale_panel {
            return Ok(refreshed);
        }
        let hit = self
            .hits
            .iter()
            .rev()
            .find(|hit| contains(hit.rect, x, y))
            .cloned();
        match hit {
            Some(hit) if hit.trigger == UiTrigger::Commit => {
                if self.focus.as_ref().is_some_and(|f| f.target == hit.target) {
                    return Ok(refreshed);
                }
                self.focus_field(&hit.target)?;
                self.refresh()?;
                Ok(true)
            }
            Some(hit) => {
                let event = if hit.trigger == UiTrigger::Select {
                    UiEvent::Select { mode }
                } else {
                    UiEvent::Activate
                };
                if hit.content {
                    return self.content_event(&hit.target, &event);
                }
                if hit.panel {
                    return self.panel_event(&hit.target, &event);
                }
                if self.panels.active()
                    && matches!(hit.target.as_str(), "editor.undo" | "editor.redo")
                {
                    return self.key(if hit.target == "editor.undo" {
                        DesktopKey::Undo
                    } else {
                        DesktopKey::Redo
                    });
                }
                let action = self.bindings.dispatch(&hit.target, &event)?;
                // Check before taking focus: rejected structural commands must
                // preserve the user's draft and IME composition.
                #[cfg(feature = "desktop")]
                self.fence_scene_authoring(&action)?;
                let cancelled = self.focus.take().is_some();
                self.invalidated |= cancelled;
                let outcome = self.execute(action)?;
                Ok(refreshed || cancelled || outcome.changed)
            }
            None => {
                let cancelled = self.focus.take().is_some();
                self.invalidated |= cancelled;
                self.refresh()?;
                Ok(refreshed || cancelled)
            }
        }
    }

    /// Inserts committed `UTF-8` into the draft, replacing the initially selected
    /// value on first input. Enter is the model commit boundary.
    pub fn text(&mut self, text: &str) -> Result<bool, DesktopUiError> {
        self.refresh()?;
        let Some(focus) = self.focus.as_ref() else {
            return Ok(false);
        };
        if text.is_empty() {
            return Ok(false);
        }
        if text.chars().any(char::is_control) {
            return self.reject(DesktopUiError::ControlCharacter);
        }
        let prefix = if focus.replace { "" } else { &focus.buffer };
        if prefix.len().saturating_add(text.len()) > MAX_ACTION_TEXT_BYTES {
            return self.reject(DesktopUiError::TextLimit);
        }
        let buffer = format!("{prefix}{text}");
        let focus = self.focus.as_mut().expect("focus checked");
        focus.buffer = buffer;
        focus.preedit.clear();
        focus.replace = false;
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Displays `IME` composition without changing the draft or model/history.
    /// An empty string cancels composition; committed text arrives via `text`.
    pub fn preedit(&mut self, text: &str) -> Result<bool, DesktopUiError> {
        self.refresh()?;
        let Some(focus) = self.focus.as_ref() else {
            return Ok(false);
        };
        if focus.preedit == text {
            return Ok(false);
        }
        if text.chars().any(char::is_control) {
            return self.reject(DesktopUiError::ControlCharacter);
        }
        let base = if focus.replace { 0 } else { focus.buffer.len() };
        if base.saturating_add(text.len()) > MAX_ACTION_TEXT_BYTES {
            return self.reject(DesktopUiError::TextLimit);
        }
        self.focus.as_mut().expect("focus checked").preedit = text.to_owned();
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Handles editing/navigation and `Undo`/`Redo`, never native keycodes.
    pub fn key(&mut self, key: DesktopKey) -> Result<bool, DesktopUiError> {
        if self.panel_pointer_active() && key != DesktopKey::Escape {
            return Ok(false);
        }
        self.refresh()?;
        match key {
            DesktopKey::Escape => {
                if self.cancel_panel_pointer()? {
                    return Ok(true);
                }
                if self.focus.take().is_none() {
                    return Ok(false);
                }
                self.invalidated = true;
                self.refresh()?;
                Ok(true)
            }
            DesktopKey::Backspace => {
                let Some(focus) = self.focus.as_mut() else {
                    return Ok(false);
                };
                if !focus.preedit.is_empty() {
                    return Ok(false);
                }
                if focus.buffer.is_empty() {
                    return Ok(false);
                }
                if focus.replace {
                    focus.buffer.clear();
                } else {
                    focus.buffer.pop();
                }
                focus.replace = false;
                self.invalidated = true;
                self.refresh()?;
                Ok(true)
            }
            DesktopKey::Enter => self.commit(),
            DesktopKey::Tab => {
                if self.focus.as_ref().is_some_and(|f| !f.preedit.is_empty()) {
                    return Ok(false);
                }
                let fields: Vec<_> = self
                    .hits
                    .iter()
                    .filter(|h| {
                        h.trigger == UiTrigger::Commit
                            && h.rect.size.width > 0.0
                            && h.rect.size.height > 0.0
                    })
                    .map(|h| h.target.clone())
                    .collect();
                if fields.is_empty() {
                    return Ok(false);
                }
                let next = self
                    .focus
                    .as_ref()
                    .and_then(|f| fields.iter().position(|p| p == &f.target))
                    .map_or(0, |index| (index + 1) % fields.len());
                self.commit()?;
                self.focus_field(&fields[next])?;
                self.refresh()?;
                Ok(true)
            }
            DesktopKey::Undo | DesktopKey::Redo => {
                if self.panel_pointer_active() {
                    return Ok(false);
                }
                // Content text input belongs to the browser, not document history.
                if self.focus.as_ref().is_some_and(|f| {
                    content::is_target(&f.target) || self.panels.is_target(&f.target)
                }) {
                    return Ok(false);
                }
                if self.panels.active() {
                    let operation = if key == DesktopKey::Undo {
                        "undo"
                    } else {
                        "redo"
                    };
                    let target = ["timeline", "graph", "prefab", "sequence"]
                        .into_iter()
                        .map(|prefix| format!("{prefix}:{operation}"))
                        .find(|target| self.panels.is_target(target));
                    return if let Some(target) = target {
                        self.panel_event(&target, &UiEvent::Activate)
                    } else {
                        Ok(false)
                    };
                }
                let cancelled = self.focus.take().is_some();
                self.invalidated |= cancelled;
                let outcome = self.execute(if key == DesktopKey::Undo {
                    EditorAction::Undo
                } else {
                    EditorAction::Redo
                })?;
                Ok(cancelled || outcome.changed)
            }
        }
    }

    /// Scrolls the virtualized `Outliner` window by signed visible-row offsets.
    /// Positive moves down. Clamped/zero scroll is an unchanged idle operation.
    pub fn scroll(&mut self, delta_rows: i32) -> Result<bool, DesktopUiError> {
        let refreshed = self.refresh()?;
        let total = self.visible_outliner_count();
        let maximum = total.saturating_sub(self.visible_rows().max(1));
        let next = if delta_rows < 0 {
            self.outliner_start
                .saturating_sub(delta_rows.unsigned_abs() as usize)
        } else {
            self.outliner_start
                .saturating_add(delta_rows as usize)
                .min(maximum)
        };
        if next == self.outliner_start {
            return Ok(refreshed);
        }
        self.outliner_start = next;
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Routes native wheel input only inside the actual visible `Outliner` pane.
    /// `Inspector`, viewport, toolbar and status input never scroll the `Outliner`.
    pub fn scroll_at(&mut self, x: f32, y: f32, delta_rows: i32) -> Result<bool, DesktopUiError> {
        if self.panel_pointer_active() {
            return Ok(false);
        }
        if !x.is_finite() || !y.is_finite() {
            return self.reject(DesktopUiError::InvalidPointer);
        }
        let refreshed = self.refresh()?;
        if self.graph_zoom_at(x, y, delta_rows)? {
            return Ok(true);
        }
        let browser_height = self.content_height();
        let browser_top = self.size.height - STATUS_HEIGHT - browser_height;
        if self.panels.active()
            && contains(
                Rect::new(
                    Point::new(0.0, browser_top + self.panel_tabs_height()),
                    Size::new(self.size.width, self.panel_body_height()),
                ),
                x,
                y,
            )
        {
            let changed = self.panels.scroll(delta_rows);
            self.invalidated |= changed;
            self.refresh()?;
            return Ok(refreshed || changed);
        }
        if let Some(content) = &mut self.content
            && !self.panels.active()
            && contains(
                Rect::new(
                    Point::new(0.0, browser_top),
                    Size::new(self.size.width, browser_height),
                ),
                x,
                y,
            )
        {
            let next = (content.scroll + delta_rows as f32 * 32.0).clamp(0.0, content.scroll_max);
            if next != content.scroll {
                content.scroll = next;
                self.invalidated = true;
                self.refresh()?;
                return Ok(true);
            }
            return Ok(refreshed);
        }
        if !contains(self.outliner_rect, x, y) {
            return Ok(refreshed);
        }
        Ok(self.scroll(delta_rows)? || refreshed)
    }

    fn reject<T>(&mut self, error: DesktopUiError) -> Result<T, DesktopUiError> {
        self.show_error(error.to_string());
        // Preserve the original cause if rebuilding the error preview also fails.
        let _ = self.refresh();
        Err(error)
    }

    fn visible_rows(&self) -> usize {
        (((self.size.height
            - self.content_height()
            - TOOLBAR_HEIGHT
            - STATUS_HEIGHT
            - PANEL_HEADER)
            .max(0.0)
            / ROW_HEIGHT) as usize)
            .min(MAX_VISIBLE_ROWS)
    }

    // Count from the stable document rows instead of asking Outliner for an
    // unbounded materialized window. Rows are registered parent-before-child,
    // so one pass is enough to apply visibility and collapsed-ancestor state.
    fn visible_outliner_count(&self) -> usize {
        let mut state = BTreeMap::new();
        let mut count = 0;
        for row in self.controller.document_outliner_nodes() {
            let (parent_visible, parent_expanded) = row
                .parent()
                .and_then(|parent| state.get(&parent).copied())
                .unwrap_or((true, true));
            let visible = row.is_visible() && parent_visible && parent_expanded;
            if visible {
                count += 1;
            }
            state.insert(row.id(), (visible, row.is_expanded()));
        }
        count
    }

    fn focus_field(&mut self, target: &str) -> Result<(), DesktopUiError> {
        if self.panels.is_target(target) {
            return self.focus_panel(target);
        }
        if content::is_target(target) {
            return self.focus_content(target);
        }
        let binding = self
            .bindings
            .iter()
            .find(|b| b.target().as_str() == target && b.trigger() == UiTrigger::Commit)
            .ok_or(DesktopUiError::InvalidProjection)?;
        let path = binding
            .template()
            .path()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let snapshot = self.controller.primary_snapshot()?;
        let parsed =
            ParsedPath::parse(path.as_str()).map_err(|_| DesktopUiError::InvalidProjection)?;
        let field = reflect_path(&*snapshot, &parsed).ok_or(DesktopUiError::InvalidProjection)?;
        let buffer = if let Some(value) = field.downcast_ref::<String>() {
            value.clone()
        } else if let Some(value) = field.downcast_ref::<char>() {
            value.to_string()
        } else {
            let projection = self.controller.inspector_projection()?;
            let mut nodes = vec![&projection.root];
            let mut value = None;
            while let Some(node) = nodes.pop() {
                if node.path.as_ref() == Some(path) {
                    value = node.node.value.clone();
                    break;
                }
                nodes.extend(node.children.iter());
            }
            value.ok_or(DesktopUiError::InvalidProjection)?
        };
        if buffer.len() > MAX_ACTION_TEXT_BYTES {
            return self.reject(DesktopUiError::TextLimit);
        }
        self.focus = Some(Focus {
            target: target.to_owned(),
            buffer,
            preedit: String::new(),
            replace: true,
        });
        self.invalidated = true;
        Ok(())
    }

    fn commit(&mut self) -> Result<bool, DesktopUiError> {
        let Some(focus) = self.focus.as_ref() else {
            return Ok(false);
        };
        if !focus.preedit.is_empty() {
            return Ok(false);
        }
        let event = UiEvent::Commit {
            value: ActionValue::text(focus.buffer.clone())?,
        };
        if content::is_target(&focus.target) {
            let target = focus.target.clone();
            return self.content_event(&target, &event);
        }
        if self.panels.is_target(&focus.target) {
            let target = focus.target.clone();
            return self.panel_event(&target, &event);
        }
        let action = self.bindings.dispatch(&focus.target, &event)?;
        // On error retain the exact draft for correction and display the cause.
        self.execute(action)?;
        self.focus = None;
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    fn build(&self, count: usize) -> Result<(Element, ActionBindings), DesktopUiError> {
        let (outliner, mut bindings) = outliner_panel_bound(&OutlinerViewProps {
            outliner: self.controller.outliner(),
            filter: "",
            start: self.outliner_start,
            count,
            selected: self.controller.selected_outliner(),
        })?
        .into_parts();
        let inspector = if self.controller.selection().primary().is_some() {
            let projection = self.controller.inspector_projection()?;
            let (element, inspector_bindings) = inspector_view_bound(&InspectorViewProps {
                projection: &projection,
                show_root: true,
            })?
            .into_parts();
            bindings.extend(&inspector_bindings)?;
            element
        } else {
            Element::box_().child(Element::text("Inspector — select an object"))
        };
        bindings.bind_undo("editor.undo")?;
        bindings.bind_redo("editor.redo")?;
        let dirty = if self.controller.is_dirty() {
            "Modified *"
        } else {
            "Saved"
        };
        let mut toolbar = Element::box_()
            .key_str("editor.toolbar")
            .class("desktop-toolbar")
            .child(
                Element::box_()
                    .key_str("editor.undo")
                    .class("desktop-button")
                    .child(Element::text("Undo")),
            )
            .child(
                Element::box_()
                    .key_str("editor.redo")
                    .class("desktop-button")
                    .child(Element::text("Redo")),
            )
            .child(Element::text(dirty).class("desktop-dirty"));
        #[cfg(feature = "desktop")]
        for (id, target, label) in [
            (
                SCENE_ADD_TRIANGLE_COMMAND,
                "editor.scene.add_triangle",
                "Add triangle",
            ),
            (
                SCENE_DELETE_SELECTION_COMMAND,
                "editor.scene.delete_selection",
                "Delete selection",
            ),
        ] {
            if self.controller.has_command(id) {
                bindings.bind(prism_editor_ui::ActionBinding::activate(
                    target,
                    prism_editor_ui::ActionTemplate::execute_command(id)?,
                )?)?;
                toolbar = toolbar.child(
                    Element::box_()
                        .key_str(target)
                        .class("desktop-button")
                        .style(StyleProp::Width, StyleValue::px(124.0))
                        .style(StyleProp::FlexShrink, StyleValue::Number(0.0))
                        .child(Element::text(label)),
                );
            }
        }
        if self.panels.fenced() {
            toolbar = toolbar.child(Element::text("Panel data pending *"));
        }
        if self.git.mounted() {
            toolbar = toolbar
                .child(
                    Element::box_()
                        .key_str("content:git:refresh")
                        .class("desktop-button")
                        .child(Element::text("Git refresh")),
                )
                .child(
                    Element::text(self.git.status.message.chars().take(88).collect::<String>())
                        .key_str("content:git:status"),
                );
        }
        if self.content.is_some() {
            toolbar = toolbar.child(
                Element::box_()
                    .key_str("content:validate")
                    .class("desktop-button")
                    .child(Element::text("Validate")),
            );
        }
        if self.collections.failed() {
            toolbar = toolbar.child(
                Element::box_()
                    .key_str("content:collections:retry")
                    .class("desktop-button")
                    .child(Element::text("Retry collections")),
            );
        }
        let left = (self.size.width * 0.23).min(280.0);
        let right = (self.size.width * 0.30).min(380.0);
        let height =
            (self.size.height - self.content_height() - TOOLBAR_HEIGHT - STATUS_HEIGHT).max(0.0);
        let content = Element::box_()
            .key_str("editor.content")
            .class("desktop-content")
            .style(StyleProp::Height, StyleValue::px(height))
            .child(
                Element::box_()
                    .key_str("editor.outliner")
                    .class("desktop-pane")
                    .style(StyleProp::Width, StyleValue::px(left))
                    .child(outliner),
            )
            .child(
                Element::box_()
                    .key_str("editor.viewport")
                    .class("desktop-viewport")
                    .style(
                        StyleProp::Width,
                        StyleValue::px((self.size.width - left - right).max(0.0)),
                    ),
            )
            .child(
                Element::box_()
                    .key_str("editor.inspector")
                    .class("desktop-pane")
                    .style(StyleProp::Width, StyleValue::px(right))
                    .child(inspector),
            );
        let status = self
            .error
            .as_deref()
            .unwrap_or(if self.collections.fenced() {
                &self.collections.message
            } else if self.content.is_some() {
                &self.validation.message
            } else {
                "Enter: commit   Esc: cancel   Tab: next field"
            });
        let chars = (self.size.width / 7.0).max(0.0) as usize;
        let status: String = status
            .chars()
            .take(chars)
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let status =
            Element::text(status)
                .key_str("editor.status")
                .class(if self.error.is_some() {
                    "desktop-error"
                } else {
                    "desktop-status"
                });
        let mut element = Element::box_()
            .key_str("editor.desktop")
            .class("desktop-root")
            .style(StyleProp::Width, StyleValue::px(self.size.width))
            .style(StyleProp::Height, StyleValue::px(self.size.height))
            .child(toolbar)
            .child(content);
        if self.panels.mounted() {
            element = element.child(self.panels.tabs(self.content.is_some()));
        }
        if let Some(panel) = self.panels.body(self.panel_body_height()) {
            element = element.child(panel);
        } else if let Some(browser) = &self.content {
            element = element.child(
                Element::box_()
                    .key_str("editor.browser")
                    .class("desktop-browser")
                    .style(StyleProp::Height, StyleValue::px(self.panel_body_height()))
                    .child(
                        browser
                            .view
                            .element()
                            .clone()
                            .style(StyleProp::FlexShrink, StyleValue::number(0.0))
                            .style(StyleProp::MarginTop, StyleValue::px(-browser.scroll)),
                    ),
            );
        }
        let element = element.child(status);
        Ok((decorate(&element, self.focus.as_ref(), false), bindings))
    }
}

// Element and RetainedScene both use preorder. Do not estimate text widths or
// independently lay out hit targets: keyed reconciliation may reuse IDs.
fn collect_elements<'a>(
    element: &'a Element,
    inherited: Option<&'static str>,
    result: &mut Vec<(&'a Element, Option<&'static str>)>,
) {
    let scope = match element.explicit_key() {
        Some(Key::Str(key)) => match key.as_str() {
            "editor.toolbar" => Some("editor.toolbar"),
            "editor.outliner" => Some("editor.outliner"),
            "editor.inspector" => Some("editor.inspector"),
            "editor.browser" => Some("editor.browser"),
            "editor.paneltabs" => Some("editor.paneltabs"),
            "editor.nativepanel" => Some("editor.nativepanel"),
            "editor.status" => Some("editor.status"),
            _ => inherited,
        },
        _ => inherited,
    };
    result.push((element, scope));
    for child in element.child_elements() {
        collect_elements(child, scope, result);
    }
}

// Isolate subtrees in one disposable clone using the backend's own painter.
// Layout is not recomputed. Draw and pointer clipping use identical pane boxes.
fn clipped_draw_list(
    scene: &RetainedScene,
    scopes: &BTreeMap<&str, (BackendId, Rect)>,
    window: Rect,
) -> Result<DrawList, DesktopUiError> {
    let lookup = |key| {
        scopes
            .get(key)
            .copied()
            .ok_or(DesktopUiError::InvalidProjection)
    };
    let (root, _) = lookup("editor.desktop")?;
    let (content, _) = lookup("editor.content")?;
    let mut scene = scene.clone();
    let mut list = DrawList::new();
    scene.apply(BackendOp::Reorder {
        parent: root,
        order: Vec::new(),
    });
    append_clipped(&mut list, &scene.to_draw_list(), window);
    scene.apply(BackendOp::SetPaint {
        id: root,
        paint: PaintStyle::default(),
    });
    for key in [
        "editor.toolbar",
        "editor.outliner",
        "editor.viewport",
        "editor.inspector",
        "editor.browser",
        "editor.paneltabs",
        "editor.nativepanel",
        "editor.status",
    ] {
        if matches!(
            key,
            "editor.browser" | "editor.paneltabs" | "editor.nativepanel"
        ) && !scopes.contains_key(key)
        {
            continue;
        }
        let (id, bounds) = lookup(key)?;
        let pane = matches!(
            key,
            "editor.outliner" | "editor.viewport" | "editor.inspector"
        );
        scene.apply(BackendOp::Reorder {
            parent: root,
            order: vec![if pane { content } else { id }],
        });
        if pane {
            scene.apply(BackendOp::Reorder {
                parent: content,
                order: vec![id],
            });
        }
        append_clipped(&mut list, &scene.to_draw_list(), bounds);
    }
    Ok(list)
}

fn append_clipped(list: &mut DrawList, source: &DrawList, bounds: Rect) {
    list.push(DrawCommand::PushLayer(LayerCmd {
        bounds,
        opacity: 1.0,
    }));
    for command in source.commands() {
        let command = match *command {
            DrawCommand::Glyph(mut glyph) => {
                glyph.text = list.intern(source.text(glyph.text));
                DrawCommand::Glyph(glyph)
            }
            other => other,
        };
        list.push(command);
    }
    list.push(DrawCommand::PopLayer);
}

fn decorate(element: &Element, focus: Option<&Focus>, in_focus: bool) -> Element {
    let in_focus = in_focus
        || matches!(element.explicit_key(), Some(Key::Str(key)) if focus.is_some_and(|f| f.target == *key));
    let value = in_focus && element.class_names().iter().any(|c| is_input_value(c));
    let mut result = match element.kind() {
        ElementKind::Text => Element::text(if value {
            let f = focus.expect("focused value");
            if f.preedit.is_empty() {
                format!("{} |", f.buffer)
            } else if f.replace {
                format!("[{}]", f.preedit)
            } else {
                format!("{}[{}]", f.buffer, f.preedit)
            }
        } else {
            element.text_content().unwrap_or("").to_owned()
        }),
        ElementKind::Box => Element::box_(),
        ElementKind::Custom(kind) => Element::custom(kind.clone()),
    };
    // Ui's mount path inherits foreground, but its current reconciliation
    // recomputes paint per node without inheritance. Give every text run an
    // explicit baseline class so repeated updates do not lose glyphs. Later
    // semantic classes (field values/errors) and inline styles override it.
    if matches!(element.kind(), ElementKind::Text) {
        result = result.class("desktop-text");
    }
    result = match element.explicit_key() {
        Some(Key::Str(key)) => result.key_str(key.clone()),
        Some(Key::Int(key)) => result.key_int(*key),
        _ => result,
    };
    for class in element.class_names() {
        result = result.class(class.clone());
    }
    for (prop, value) in element.inline_pairs() {
        result = result.style(*prop, value.clone());
    }
    if in_focus && element.explicit_key().is_some() {
        result = result.class("desktop-focused");
    }
    for child in element.child_elements() {
        result = result.child(decorate(child, focus, in_focus));
    }
    result
}

fn is_input_value(class: &str) -> bool {
    matches!(
        class,
        "prism-editor-inspector__value" | "pk-search-field__value" | "pk-search-field__placeholder"
    )
}

fn truncate_bytes(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn contains(rect: Rect, x: f32, y: f32) -> bool {
    x >= rect.left() && x < rect.right() && y >= rect.top() && y < rect.bottom()
}

fn intersection(a: Rect, b: Rect) -> Rect {
    let left = a.left().max(b.left());
    let top = a.top().max(b.top());
    Rect::new(
        Point::new(left, top),
        Size::new(
            (a.right().min(b.right()) - left).max(0.0),
            (a.bottom().min(b.bottom()) - top).max(0.0),
        ),
    )
}

fn stylesheet() -> StyleSheet {
    use StyleProp as P;
    use StyleValue as V;
    let mut sheet = StyleSheet::new();
    let column = |name| {
        Class::new(name)
            .with(P::FlexDirection, V::Keyword(Keyword::Column))
            .with(P::FlexShrink, V::Number(0.0))
    };
    let row = |name| {
        Class::new(name)
            .with(P::FlexDirection, V::Keyword(Keyword::Row))
            .with(P::FlexShrink, V::Number(0.0))
    };
    let bg = |r, g, b| V::Color(Color::rgba8(r, g, b, 255));
    sheet.insert(
        Class::new("desktop-text")
            .with(P::Color, bg(225, 229, 238))
            .with(P::FontSize, V::px(13.0)),
    );
    sheet.insert(
        column("desktop-root")
            .with(P::BackgroundColor, bg(24, 26, 31))
            .with(P::Color, bg(225, 229, 238))
            .with(P::FontSize, V::px(13.0)),
    );
    sheet.insert(
        row("desktop-toolbar")
            .with(P::Height, V::px(TOOLBAR_HEIGHT))
            .with(P::AlignItems, V::Keyword(Keyword::Center))
            .with(P::ColumnGap, V::px(8.0))
            .with(P::BackgroundColor, bg(40, 44, 54)),
    );
    sheet.insert(
        row("desktop-button")
            .with(P::Width, V::px(64.0))
            .with(P::Height, V::px(28.0))
            .with(P::AlignItems, V::Keyword(Keyword::Center))
            .with(P::JustifyContent, V::Keyword(Keyword::Center))
            .with(P::BackgroundColor, bg(64, 71, 87)),
    );
    sheet.insert(
        row("desktop-content")
            .with(P::Width, V::percent(100.0))
            .with(P::MinHeight, V::px(0.0)),
    );
    sheet.insert(
        column("desktop-pane")
            .with(P::Height, V::percent(100.0))
            .with(P::MinWidth, V::px(0.0))
            .with(P::BackgroundColor, bg(33, 36, 44)),
    );
    sheet.insert(
        column("desktop-viewport")
            .with(P::Height, V::percent(100.0))
            .with(P::MinWidth, V::px(0.0))
            .with(P::BackgroundColor, bg(17, 20, 26)),
    );
    for name in [
        "prism-editor-outliner",
        "prism-editor-outliner__list",
        "prism-editor-inspector",
        "prism-editor-inspector__body",
        "prism-editor-inspector__node",
        "prism-editor-inspector__children",
    ] {
        sheet.insert(column(name).with(P::Width, V::percent(100.0)));
    }
    sheet.insert(column("prism-editor-outliner__header").with(P::Height, V::px(PANEL_HEADER)));
    sheet.insert(
        Class::new("prism-editor-outliner__search").with(P::Display, V::Keyword(Keyword::None)),
    );
    sheet.insert(
        row("prism-editor-outliner__row")
            .with(P::Height, V::px(ROW_HEIGHT))
            .with(P::AlignItems, V::Keyword(Keyword::Center)),
    );
    sheet.insert(
        Class::new("prism-editor-outliner__disclosure")
            .with(P::Width, V::px(22.0))
            .with(P::Height, V::px(ROW_HEIGHT))
            .with(P::FlexShrink, V::Number(0.0)),
    );
    sheet.insert(Class::new("is-child").with(P::PaddingLeft, V::px(12.0)));
    sheet.insert(Class::new("is-selected").with(P::BackgroundColor, bg(52, 78, 116)));
    sheet.insert(Class::new("desktop-focused").with(P::BackgroundColor, bg(46, 65, 92)));
    sheet.insert(
        column("desktop-browser")
            .with(P::Width, V::percent(100.0))
            .with(P::BackgroundColor, bg(33, 36, 44)),
    );
    sheet.insert(
        row("pk-search-field")
            .with(P::Height, V::px(28.0))
            .with(P::MinWidth, V::px(180.0))
            .with(P::BackgroundColor, bg(24, 26, 31)),
    );
    sheet.insert(Class::new("prism-editor-content__asset").with(P::MinHeight, V::px(28.0)));
    sheet.insert(column("prism-editor-inspector__heading").with(P::MinHeight, V::px(22.0)));
    sheet.insert(
        Class::new("prism-editor-inspector__type").with(P::Display, V::Keyword(Keyword::None)),
    );
    sheet.insert(
        Class::new("prism-editor-inspector__value")
            .with(P::Height, V::px(24.0))
            .with(P::Color, bg(162, 209, 255)),
    );
    sheet.insert(
        Class::new("desktop-status")
            .with(P::Height, V::px(STATUS_HEIGHT))
            .with(P::FlexShrink, V::Number(0.0)),
    );
    sheet.insert(
        Class::new("desktop-error")
            .with(P::Height, V::px(STATUS_HEIGHT))
            .with(P::Color, bg(255, 147, 147))
            .with(P::FlexShrink, V::Number(0.0)),
    );
    sheet
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_editor_core::SelectionItem;
    use prism_editor_panels::{OutlinerId, OutlinerNode};
    use prism_editor_reflect::FieldPath;
    use prism_reflect::{Reflect, TypeRegistry};

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct Model {
        label: String,
        count: u8,
        gain: f32,
    }

    fn original() -> Model {
        Model {
            label: "船".into(),
            count: 7,
            gain: 0.5,
        }
    }

    fn loaded(rows: u128) -> EditorDesktopUi {
        let mut controller = EditorController::new();
        let item = SelectionItem::entity(1);
        let mut registry = TypeRegistry::new();
        registry.register::<Model>();
        controller
            .register_object(item.clone(), Box::new(original()), registry)
            .unwrap();
        for id in 1..=rows {
            let mut row = OutlinerNode::new(OutlinerId::from_raw(id), None, format!("Object {id}"));
            row.set_selection(item.clone());
            controller.register_outliner_node(row).unwrap();
        }
        EditorDesktopUi::new(controller).unwrap()
    }

    fn snapshot(ui: &EditorDesktopUi) -> Model {
        let mut model = original();
        model
            .apply(&*ui.controller().primary_snapshot().unwrap())
            .unwrap();
        model
    }

    fn click_target(ui: &mut EditorDesktopUi, target: &str) -> bool {
        let hit = ui
            .hits
            .iter()
            .find(|h| h.target == target)
            .unwrap_or_else(|| panic!("missing target {target}"));
        assert!(
            hit.rect.size.width > 0.0 && hit.rect.size.height > 0.0,
            "invisible {target}: {:?}",
            hit.rect
        );
        // Select the label half of an Outliner row, outside its disclosure.
        let x = hit.rect.left() + hit.rect.size.width * 0.75;
        let y = hit.rect.top() + hit.rect.size.height * 0.5;
        ui.click(x, y, SelectionMode::Replace).unwrap()
    }

    fn selected() -> EditorDesktopUi {
        let mut ui = loaded(1);
        assert!(click_target(&mut ui, "outliner:1"));
        ui
    }

    fn edit(path: &str, value: ActionValue) -> EditorAction {
        EditorAction::EditInspectorField {
            path: FieldPath::parse(path).unwrap(),
            value,
        }
    }

    fn glyphs(ui: &EditorDesktopUi) -> Vec<String> {
        ui.draw_list()
            .commands()
            .iter()
            .filter_map(|cmd| match cmd {
                DrawCommand::Glyph(g) => Some(ui.draw_list().text(g.text).to_owned()),
                _ => None,
            })
            .collect()
    }

    #[cfg(feature = "desktop")]
    #[test]
    fn scene_authoring_toolbar_is_opt_in_and_dispatches_normal_commands() {
        let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
        for label in ["Add triangle", "Delete selection"] {
            assert!(!glyphs(&ui).iter().any(|text| text == label));
        }
        assert!(!ui
            .hits
            .iter()
            .any(|hit| hit.target.starts_with("editor.scene.")));
        crate::install_scene_authoring_commands(ui.controller_mut()).unwrap();
        ui.refresh().unwrap();
        for label in ["Add triangle", "Delete selection"] {
            assert!(glyphs(&ui).iter().any(|text| text == label));
        }
        assert_eq!(ui.controller().document_objects().len(), 0);
        assert!(!ui.controller().is_dirty());
        assert!(click_target(&mut ui, "editor.scene.add_triangle"));
        let item = ui.controller().selection().primary().unwrap().clone();
        assert_eq!(ui.controller().document_objects().len(), 1);
        assert_eq!(ui.controller().document_history_lengths(), (1, 0));
        assert!(click_target(&mut ui, "editor.scene.delete_selection"));
        assert_eq!(ui.controller().document_objects().len(), 0);
        assert!(click_target(&mut ui, "editor.undo"));
        assert_eq!(ui.controller().selection().primary(), Some(&item));
        assert_eq!(ui.controller().document_objects().len(), 1);
        assert!(click_target(&mut ui, "editor.redo"));
        assert_eq!(ui.controller().document_objects().len(), 0);
    }

    #[cfg(feature = "desktop")]
    #[test]
    fn scene_authoring_rejects_draft_and_ime_without_losing_input() {
        let mut ui = selected();
        crate::install_scene_authoring_commands(ui.controller_mut()).unwrap();
        ui.refresh().unwrap();
        click_target(&mut ui, "field:.label");
        ui.text("pending draft ").unwrap();
        ui.preedit("输入").unwrap();
        let draft = ui.focus.clone();
        let revision = ui.controller().revision();
        for id in [SCENE_ADD_TRIANGLE_COMMAND, SCENE_DELETE_SELECTION_COMMAND] {
            assert!(ui
                .execute(EditorAction::ExecuteCommand {
                    command_id: id.into()
                })
                .is_err());
            assert_eq!(ui.focus, draft);
            assert_eq!(ui.controller().revision(), revision);
        }
        for target in ["editor.scene.add_triangle", "editor.scene.delete_selection"] {
            let rect = ui
                .hits
                .iter()
                .find(|hit| hit.target == target)
                .unwrap()
                .rect;
            assert!(ui
                .click(
                    rect.left() + rect.size.width * 0.5,
                    rect.top() + rect.size.height * 0.5,
                    SelectionMode::Replace,
                )
                .is_err());
            assert_eq!(ui.focus, draft);
            assert_eq!(ui.controller().revision(), revision);
            assert_eq!(ui.controller().document_objects().len(), 1);
            assert_eq!(ui.controller().document_history_lengths(), (0, 0));
        }
        assert_eq!(snapshot(&ui), original());
        assert!(ui.key(DesktopKey::Escape).unwrap());
        assert!(click_target(&mut ui, "editor.scene.add_triangle"));
        assert_eq!(ui.controller().document_objects().len(), 2);
    }

    #[cfg(feature = "desktop")]
    #[test]
    fn scene_authoring_waits_for_active_controller_gesture() {
        let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
        crate::install_scene_authoring_commands(ui.controller_mut()).unwrap();
        ui.refresh().unwrap();
        ui.controller_mut().begin_edit().unwrap();
        let revision = ui.controller().revision();
        for id in [SCENE_ADD_TRIANGLE_COMMAND, SCENE_DELETE_SELECTION_COMMAND] {
            assert!(ui
                .execute(EditorAction::ExecuteCommand {
                    command_id: id.into()
                })
                .is_err());
            assert_eq!(ui.controller().revision(), revision);
            assert_eq!(ui.controller().document_objects().len(), 0);
            assert!(ui.controller().has_active_edit());
        }
        ui.controller_mut().end_edit();
        ui.execute(EditorAction::ExecuteCommand {
            command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
        })
        .unwrap();
        assert_eq!(ui.controller().document_objects().len(), 1);
    }

    #[test]
    fn actual_layout_click_binding_field_toolbar_history_and_writeback() {
        let mut ui = selected();
        assert_eq!(
            ui.controller().selection().primary(),
            Some(&SelectionItem::entity(1))
        );
        assert!(click_target(&mut ui, "field:.count"));
        assert!(ui.text("42").unwrap());
        assert_eq!(snapshot(&ui).count, 7);
        assert!(ui.key(DesktopKey::Enter).unwrap());
        assert_eq!(snapshot(&ui).count, 42);
        assert_eq!(
            ui.controller()
                .history_lengths(&SelectionItem::entity(1))
                .unwrap(),
            (1, 0)
        );
        assert!(ui.controller().is_dirty());
        assert!(glyphs(&ui).iter().any(|s| s == "Modified *"));
        assert!(click_target(&mut ui, "editor.undo"));
        assert_eq!(snapshot(&ui).count, 7);
        assert!(ui.key(DesktopKey::Redo).unwrap());
        assert_eq!(snapshot(&ui).count, 42);
        let saved = ui.controller().model_revision();
        ui.mark_saved(saved).unwrap();
        assert!(!ui.controller().is_dirty());
        assert!(glyphs(&ui).iter().any(|s| s == "Saved"));
    }

    #[test]
    fn reconciled_text_keeps_foreground_and_semantic_colors() {
        let mut ui = loaded(1);
        let foreground = Color::rgba8(225, 229, 238, 255);
        let error_color = Color::rgba8(255, 147, 147, 255);
        let value_color = Color::rgba8(162, 209, 255, 255);
        for saved in [true, false, true] {
            if !saved {
                click_target(&mut ui, "outliner:1");
                ui.execute(edit("count", ActionValue::integer(42))).unwrap();
            } else if ui.controller().is_dirty() {
                ui.mark_saved(ui.controller().model_revision()).unwrap();
            }
            for message in ["first presentation error", "updated presentation error"] {
                ui.show_error(message);
                ui.refresh().unwrap();
                let expected = if saved { "Saved" } else { "Modified *" };
                for text in ["Undo", "Redo", "Object 1", expected] {
                    assert!(
                        ui.draw_list().commands().iter().any(|command| {
                            matches!(command, DrawCommand::Glyph(g)
                            if ui.draw_list().text(g.text) == text && g.color == foreground
                            && g.rect.size.width > 0.0 && g.rect.size.height > 0.0)
                        }),
                        "missing visible default-colored glyph {text}"
                    );
                }
                assert!(ui.draw_list().commands().iter().any(|command| {
                    matches!(command, DrawCommand::Glyph(g)
                        if ui.draw_list().text(g.text) == message && g.color == error_color)
                }));
                if ui.controller().selection().primary().is_some() {
                    assert!(ui.draw_list().commands().iter().any(|command| {
                        matches!(command, DrawCommand::Glyph(g)
                            if ui.draw_list().text(g.text) == "42" && g.color == value_color)
                    }));
                }
            }
        }
    }

    #[test]
    fn save_acknowledgement_preserves_active_draft_composition_and_input_geometry() {
        let mut ui = selected();
        ui.resize(960.0, 640.0).unwrap();
        assert_eq!(ui.size(), Size::new(960.0, 640.0));
        ui.execute(edit("count", ActionValue::integer(42))).unwrap();
        let saved_revision = ui.controller().model_revision();
        click_target(&mut ui, "field:.label");
        ui.text("uncommitted draft ").unwrap();
        ui.preedit("输入🌍").unwrap();
        let draft = ui.focus.clone();
        let input = ui.text_input_rect();
        assert!(input.is_some());
        assert!(ui.has_uncommitted_edit());
        ui.resize(0.0, 0.0).unwrap();
        assert!(ui.has_uncommitted_edit());
        assert_eq!(ui.text_input_rect(), None);
        ui.resize(960.0, 640.0).unwrap();
        ui.mark_saved(saved_revision).unwrap();
        assert!(!ui.controller().is_dirty());
        assert_eq!(ui.controller().model_revision(), saved_revision);
        assert_eq!(ui.focus, draft);
        assert_eq!(ui.text_input_rect(), input);
        assert!(glyphs(&ui).iter().any(|s| s == "Saved"));
        assert!(glyphs(&ui)
            .iter()
            .any(|s| s.contains("uncommitted draft [输入🌍]")));
        assert!(
            !ui.key(DesktopKey::Enter).unwrap(),
            "composition cannot commit"
        );
        assert_eq!(snapshot(&ui).label, original().label);
        let revision = ui.controller().revision();
        ui.mark_saved(saved_revision).unwrap();
        assert_eq!(ui.controller().revision(), revision);
        assert!(!ui.refresh().unwrap());
        assert_eq!(ui.focus, draft);
        assert!(ui.mark_saved(saved_revision - 1).is_err());
        assert_eq!(ui.focus, draft);
        assert!(!ui.controller().is_dirty());
        assert_eq!(ui.controller().revision(), revision);
        ui.text("已提交").unwrap();
        ui.key(DesktopKey::Enter).unwrap();
        assert_eq!(snapshot(&ui).label, "uncommitted draft 已提交");
        assert!(!ui.has_uncommitted_edit());
        assert!(ui.controller().is_dirty());
        assert!(glyphs(&ui).iter().any(|s| s == "Modified *"));
    }

    #[test]
    fn keyboard_and_toolbar_navigate_document_history_without_a_primary() {
        let mut ui = selected();
        ui.execute(edit("count", ActionValue::integer(42))).unwrap();
        let b = SelectionItem::entity(2);
        ui.controller_mut()
            .register_object(b.clone(), Box::new(original()), TypeRegistry::new())
            .unwrap();
        ui.controller_mut().select(b.clone()).unwrap();
        ui.execute(edit("count", ActionValue::integer(43))).unwrap();
        ui.controller_mut()
            .select(SelectionItem::entity(1))
            .unwrap();
        ui.refresh().unwrap();
        assert!(ui.key(DesktopKey::Undo).unwrap());
        assert_eq!(snapshot(&ui).count, 42, "A untouched when undoing B");
        let mut actual_b = original();
        actual_b
            .apply(&*ui.controller().snapshot(&b).unwrap())
            .unwrap();
        assert_eq!(actual_b.count, 7);
        ui.controller_mut().clear_selection().unwrap();
        ui.refresh().unwrap();
        assert!(click_target(&mut ui, "editor.undo"));
        assert_eq!(ui.controller().document_history_lengths(), (0, 2));
        assert!(ui.key(DesktopKey::Redo).unwrap());
        assert!(click_target(&mut ui, "editor.redo"));
        actual_b
            .apply(&*ui.controller().snapshot(&b).unwrap())
            .unwrap();
        assert_eq!(actual_b.count, 43);
        assert!(ui.controller().selection().primary().is_none());
        assert_eq!(ui.controller().document_history_lengths(), (2, 0));
    }

    #[test]
    fn utf8_backspace_ime_preedit_is_visible_but_never_commits() {
        let mut ui = selected();
        click_target(&mut ui, "field:.label");
        let revision = ui.controller().revision();
        assert!(ui.preedit("你好🌍").unwrap());
        assert!(!ui.preedit("你好🌍").unwrap());
        assert!(glyphs(&ui).iter().any(|s| s.contains("[你好🌍]")));
        assert!(!ui.key(DesktopKey::Enter).unwrap());
        assert!(!ui.key(DesktopKey::Tab).unwrap());
        assert!(!ui.key(DesktopKey::Backspace).unwrap());
        assert_eq!(snapshot(&ui), original());
        assert_eq!(ui.controller().revision(), revision);
        assert!(ui.text("你好🌍").unwrap());
        assert!(ui.focus.as_ref().unwrap().preedit.is_empty());
        assert!(ui.key(DesktopKey::Backspace).unwrap());
        assert_eq!(ui.focus.as_ref().unwrap().buffer, "你好");
        ui.text("é").unwrap();
        ui.key(DesktopKey::Enter).unwrap();
        assert_eq!(snapshot(&ui).label, "你好é");
        click_target(&mut ui, "field:.label");
        ui.key(DesktopKey::Backspace).unwrap();
        ui.key(DesktopKey::Enter).unwrap();
        assert_eq!(snapshot(&ui).label, "");
    }

    #[test]
    fn rejected_input_and_failed_commit_preserve_model_and_correctable_draft() {
        let mut ui = selected();
        click_target(&mut ui, "field:.count");
        ui.text("256").unwrap();
        let revision = ui.controller().revision();
        let draft = ui.focus.clone();
        assert!(ui.key(DesktopKey::Enter).is_err());
        assert_eq!(ui.focus, draft);
        assert_eq!(ui.controller().revision(), revision);
        assert_eq!(snapshot(&ui), original());
        assert_eq!(
            ui.controller()
                .history_lengths(&SelectionItem::entity(1))
                .unwrap(),
            (0, 0)
        );
        assert!(ui.error_message().unwrap().contains("convert"));
        assert!(glyphs(&ui).iter().any(|s| s.contains("cannot convert")));
        for _ in 0..3 {
            ui.key(DesktopKey::Backspace).unwrap();
        }
        ui.text("2").unwrap();
        ui.key(DesktopKey::Enter).unwrap();
        assert_eq!(snapshot(&ui).count, 2);
        assert!(ui.error_message().is_none());
        click_target(&mut ui, "field:.gain");
        ui.text("NaN").unwrap();
        assert!(ui.key(DesktopKey::Enter).is_err());
        assert_eq!(snapshot(&ui).gain, 0.5);
        ui.key(DesktopKey::Escape).unwrap();
        click_target(&mut ui, "field:.label");
        let draft = ui.focus.clone();
        assert!(ui.text(&"界".repeat(MAX_ACTION_TEXT_BYTES)).is_err());
        assert_eq!(ui.focus, draft);
        assert!(ui.preedit("\n").is_err());
        assert_eq!(ui.focus, draft);
        assert!(ui.text("\0").is_err());
        assert_eq!(ui.focus, draft);
        assert!(ui.preedit(&"a".repeat(MAX_ACTION_TEXT_BYTES + 1)).is_err());
        assert_eq!(ui.focus, draft);
    }

    #[test]
    fn idle_frames_and_invalid_or_minimized_geometry_do_not_edit_model() {
        let mut ui = loaded(0);
        let revision = ui.controller().revision();
        let commands = ui.draw_list().commands().to_vec();
        assert!(!ui.refresh().unwrap());
        assert!(!ui.text("unused").unwrap());
        assert!(!ui.preedit("unused").unwrap());
        for key in [
            DesktopKey::Enter,
            DesktopKey::Escape,
            DesktopKey::Backspace,
            DesktopKey::Tab,
            DesktopKey::Undo,
            DesktopKey::Redo,
        ] {
            assert!(!ui.key(key).unwrap());
        }
        assert!(!ui.scroll(0).unwrap());
        assert_eq!(ui.draw_list().commands(), commands);
        let size = ui.size;
        let viewport = ui.viewport_rect();
        for (w, h) in [
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
            (-1.0, 1.0),
            (MAX_DIMENSION + 1.0, 1.0),
        ] {
            assert!(ui.resize(w, h).is_err());
            assert_eq!(ui.size, size);
            assert_eq!(ui.viewport_rect(), viewport);
        }
        assert!(ui.click(f32::NAN, 1.0, SelectionMode::Replace).is_err());
        assert!(!ui.click(-1.0, -1.0, SelectionMode::Replace).unwrap());
        ui.resize(0.0, 0.0).unwrap();
        assert_eq!(ui.viewport_rect().size, Size::ZERO);
        assert!(!ui.click(0.0, 0.0, SelectionMode::Replace).unwrap());
        assert!(ui
            .hits
            .iter()
            .all(|h| h.rect.size.width == 0.0 || h.rect.size.height == 0.0));
        assert_eq!(ui.controller().revision(), revision);
        assert!(!ui.refresh().unwrap());
    }

    #[test]
    fn virtualized_outliner_signed_scroll_and_clipped_inspector_use_real_boxes() {
        let mut ui = loaded(100);
        ui.resize(800.0, 200.0).unwrap();
        assert_eq!(ui.visible_rows(), 4);
        assert!(ui.scroll(i32::MAX).unwrap());
        assert_eq!(ui.outliner_start, 96);
        assert!(!ui.scroll(1).unwrap());
        assert!(ui
            .bindings
            .iter()
            .any(|b| b.target().as_str() == "outliner:100"));
        assert!(!ui
            .bindings
            .iter()
            .any(|b| b.target().as_str() == "outliner:1"));
        assert!(click_target(&mut ui, "outliner:100"));
        assert!(ui.scroll(i32::MIN).unwrap());
        assert_eq!(ui.outliner_start, 0);
        assert!(!ui.scroll(-1).unwrap());
        ui.resize(800.0, 180.0).unwrap();
        let viewport = ui.viewport_rect();
        assert!(viewport.size.width > 0.0 && viewport.size.height > 0.0);
        for hit in ui
            .hits
            .iter()
            .filter(|h| h.trigger == UiTrigger::Commit && h.rect.size.height > 0.0)
        {
            assert!(hit.rect.left() >= viewport.right());
            assert!(hit.rect.bottom() <= viewport.bottom());
        }
        // A retained field still exists below the panel, but is neither drawn
        // nor clickable through the status bar's clip boundary.
        assert!(ui
            .hits
            .iter()
            .any(|h| h.trigger == UiTrigger::Commit && h.rect.size.height == 0.0));
        let mut depth = 0;
        let mut clip = Rect::new(Point::ZERO, ui.size);
        for command in ui.draw_list().commands() {
            match command {
                DrawCommand::PushLayer(layer) => {
                    depth += 1;
                    clip = layer.bounds;
                }
                DrawCommand::PopLayer => depth -= 1,
                DrawCommand::Glyph(g) if ui.draw_list().text(g.text) == "gain" => {
                    assert!(clip.left() >= viewport.right());
                    assert!(clip.bottom() <= viewport.bottom());
                }
                _ => {}
            }
            assert!(depth >= 0);
        }
        assert_eq!(depth, 0);
    }

    #[test]
    fn tab_escape_external_controller_refresh_and_bounded_host_errors() {
        let mut ui = selected();
        click_target(&mut ui, "field:.label");
        ui.text("saved through Tab").unwrap();
        assert!(ui.key(DesktopKey::Tab).unwrap());
        assert_eq!(snapshot(&ui).label, "saved through Tab");
        assert_eq!(ui.focus.as_ref().unwrap().target, "field:.count");
        ui.text("100").unwrap();
        ui.key(DesktopKey::Escape).unwrap();
        assert_eq!(snapshot(&ui).count, 7);
        click_target(&mut ui, "field:.count");
        ui.text("101").unwrap();
        ui.controller_mut()
            .execute(edit(".count", ActionValue::integer(21)))
            .unwrap();
        assert!(ui.refresh().unwrap());
        assert!(ui.focus.is_none());
        assert_eq!(snapshot(&ui).count, 21);
        assert!(!ui.refresh().unwrap());
        ui.controller_mut()
            .register_command("set.count", |_| {
                Ok(edit(".count", ActionValue::integer(22)))
            })
            .unwrap();
        ui.execute(EditorAction::ExecuteCommand {
            command_id: "set.count".into(),
        })
        .unwrap();
        assert_eq!(snapshot(&ui).count, 22);
        ui.show_error("GPU: surface failure\nrecovering");
        assert!(ui.refresh().unwrap());
        assert!(glyphs(&ui)
            .iter()
            .any(|s| s == "GPU: surface failure recovering"));
        assert!(!ui.refresh().unwrap());
        ui.show_error("界".repeat(MAX_ACTION_TEXT_BYTES));
        assert!(ui.error_message().unwrap().len() <= MAX_ACTION_TEXT_BYTES);
        assert!(ui
            .error_message()
            .unwrap()
            .is_char_boundary(ui.error_message().unwrap().len()));
        assert!(ui.refresh().unwrap());
    }

    #[test]
    fn wheel_routes_only_to_actual_outliner_layout_and_never_edits_model() {
        let mut ui = loaded(100);
        ui.resize(800.0, 200.0).unwrap();
        let revision = ui.controller().revision();
        let pane = ui.outliner_rect;
        let viewport = ui.viewport_rect();
        let x = pane.left() + pane.size.width * 0.5;
        let y = pane.top() + pane.size.height * 0.5;
        for (px, py) in [
            (viewport.left() + viewport.size.width * 0.5, y),
            (viewport.right() + 1.0, y),
            (x, pane.top() - 1.0),
            (x, pane.bottom()),
            (pane.right(), y),
            (-1.0, y),
        ] {
            assert!(!ui.scroll_at(px, py, 3).unwrap());
            assert_eq!(ui.outliner_start, 0);
        }
        assert!(ui.scroll_at(x, y, 3).unwrap());
        assert_eq!(ui.outliner_start, 3);
        assert!(!ui.scroll_at(x, y, 0).unwrap());
        assert!(ui.scroll_at(x, y, i32::MIN).unwrap());
        assert_eq!(ui.outliner_start, 0);
        assert!(!ui.scroll_at(x, y, -1).unwrap());
        assert!(ui.scroll_at(f32::NAN, y, 1).is_err());
        assert!(ui.scroll_at(x, f32::INFINITY, 1).is_err());
        assert_eq!(ui.outliner_start, 0);
        ui.resize(0.0, 0.0).unwrap();
        assert!(!ui.scroll_at(0.0, 0.0, 1).unwrap());
        assert_eq!(ui.controller().revision(), revision);
        assert!(!ui.controller().is_dirty());
    }

    #[test]
    fn ime_candidate_box_uses_visible_value_layout_and_tracks_focus() {
        let mut ui = selected();
        assert_eq!(ui.text_input_rect(), None);
        click_target(&mut ui, "field:.label");
        let input = ui.text_input_rect().expect("focused value is visible");
        let row = ui
            .hits
            .iter()
            .find(|h| h.target == "field:.label")
            .unwrap()
            .rect;
        assert!(input.top() > row.top(), "value excludes the field heading");
        assert!(input.bottom() <= row.bottom());
        let mut actual = Vec::new();
        ui.ui.backend().visit_layout(|_, rect| actual.push(rect));
        assert!(
            actual.contains(&input),
            "candidate box is actual retained layout"
        );
        assert!(ui.preedit("拼音🌍").unwrap());
        assert!(ui.text_input_rect().is_some());
        assert_eq!(snapshot(&ui), original());
        ui.preedit("").unwrap();
        ui.resize(800.0, 80.0).unwrap();
        assert!(ui.focus.is_some());
        assert_eq!(ui.text_input_rect(), None, "clipped value disables IME");
        ui.resize(800.0, 800.0).unwrap();
        assert!(ui.text_input_rect().is_some());
        ui.key(DesktopKey::Escape).unwrap();
        assert_eq!(ui.text_input_rect(), None);
        ui.key(DesktopKey::Tab).unwrap();
        assert!(ui.text_input_rect().is_some());
        ui.controller_mut();
        assert_eq!(ui.text_input_rect(), None);
        ui.refresh().unwrap();
        click_target(&mut ui, "field:.label");
        ui.resize(0.0, 0.0).unwrap();
        assert_eq!(ui.text_input_rect(), None);
    }
}
