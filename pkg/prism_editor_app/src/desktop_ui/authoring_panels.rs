//! Actual graph/Prefab authoring and immutable frame-graph capture contributions.
use super::*;
use prism_editor_graph::{GraphEditor, GraphSnapshotToken};
use prism_editor_profiler::GraphCaptureReport;
use prism_editor_ui::{
    BoundFrameGraphPanel, BoundGraphPanel, BoundPrefabPanel, FrameGraphPanelController,
    GraphNodeTemplate, GraphPanelController, GraphPanelError, GraphPanelLimits, GraphPanelViewport,
    PrefabPanelController, PrefabPanelSnapshot,
};
use std::time::Instant;

/// Owns the real host graph and contributed node recipes, with no built-in domain.
pub struct DesktopGraphPanel {
    /// Actual authoring/controller state; host mutation cancels native capture.
    pub controller: GraphPanelController,
    pub(super) view: Option<BoundGraphPanel>,
    pub(super) saved: GraphSnapshotToken,
    pub(super) templates: Vec<GraphNodeTemplate>,
    pub(super) limits: GraphPanelLimits,
    pub(super) file: Option<panel_storage::PanelFile<prism_editor_graph::GraphStorageService>>,
}
impl DesktopGraphPanel {
    /// Mount the host's actual model, templates and bounded canvas policy.
    pub fn new(
        editor: GraphEditor,
        templates: Vec<GraphNodeTemplate>,
        viewport: GraphPanelViewport,
        limits: GraphPanelLimits,
    ) -> Result<Self, GraphPanelError> {
        let saved = editor.snapshot_token();
        let controller = GraphPanelController::new(editor, templates.clone(), viewport, limits)?;
        Ok(Self {
            controller,
            view: None,
            saved,
            templates,
            limits,
            file: None,
        })
    }
    /// Acknowledge only a durable publication of this exact current document.
    pub fn acknowledge_saved(&mut self, token: &GraphSnapshotToken) -> bool {
        if *token != self.controller.editor().snapshot_token() {
            return false;
        }
        self.saved = token.clone();
        true
    }
    pub(super) fn dirty(&self) -> bool {
        self.saved != self.controller.editor().snapshot_token()
    }
}
impl DesktopPanel for DesktopGraphPanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self.controller.build();
        let mut result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        if let Some(file) = &self.file {
            file.decorate(
                "graph",
                &mut result,
                !self.controller.pointer_active(),
                !self.dirty(),
            );
        }
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        if target.starts_with("graph:file:") {
            return self.file_event(target, event);
        }
        if self.controller.pointer_active() {
            return Err("Release or cancel graph capture first".into());
        }
        self.controller
            .handle_event(
                self.view.as_ref().ok_or("Graph is not rendered")?,
                target,
                event,
            )
            .map_err(|e| e.to_string())
    }
    fn poll(&mut self, _now: Instant) -> Result<bool, String> {
        self.poll_file()
    }
    fn next_wake(&self) -> Option<Instant> {
        self.file
            .as_ref()
            .and_then(panel_storage::PanelFile::next_wake)
    }
    fn fenced(&self) -> bool {
        self.dirty()
            || self.controller.pointer_active()
            || self
                .file
                .as_ref()
                .is_some_and(panel_storage::PanelFile::fenced)
    }
    fn draft_revision(&self) -> u64 {
        self.controller.editor().revision()
    }
}

/// Retains the host's actual Prefab store, exact publication fence and history.
pub struct DesktopPrefabPanel {
    /// Real controller, including explicit propagation targets.
    pub controller: PrefabPanelController,
    pub(super) view: Option<BoundPrefabPanel>,
    saved: PrefabPanelSnapshot,
}
impl DesktopPrefabPanel {
    /// Take the existing host controller; no assets or instances are fabricated.
    pub fn new(controller: PrefabPanelController) -> Self {
        let saved = controller.snapshot();
        Self {
            controller,
            saved,
            view: None,
        }
    }
    /// Call only after the host has durably saved this exact publication.
    pub fn acknowledge_saved(&mut self, token: &PrefabPanelSnapshot) -> bool {
        if !self.controller.matches_snapshot(token) {
            return false;
        }
        self.saved = token.clone();
        true
    }
}
impl DesktopPanel for DesktopPrefabPanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self.controller.build().map_err(|e| e.to_string())?;
        let result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        self.controller
            .handle_event(
                self.view.as_ref().ok_or("Prefab is not rendered")?,
                target,
                event,
            )
            .map_err(|e| e.to_string())
    }
    fn fenced(&self) -> bool {
        !self.controller.matches_snapshot(&self.saved)
            || self
                .view
                .as_ref()
                .is_some_and(|view| view.apply_review().is_some())
    }
    fn draft_revision(&self) -> u64 {
        self.controller.store().revision()
    }
}

type CaptureProvider = dyn FnMut(
    Instant,
) -> Result<
    (
        Option<alloc::sync::Arc<GraphCaptureReport>>,
        DesktopPanelUpdate,
    ),
    String,
>;
/// Inspects real completed captures, preserving the prior publication on failure.
pub struct DesktopFrameGraphPanel {
    /// Actual capture navigation and selection controller.
    pub controller: FrameGraphPanelController,
    provider: Box<CaptureProvider>,
    view: Option<BoundFrameGraphPanel>,
    update: DesktopPanelUpdate,
    failed: bool,
}
impl DesktopFrameGraphPanel {
    /// None means no new capture; publish/clear explicitly through the controller.
    pub fn new(
        controller: FrameGraphPanelController,
        provider: impl FnMut(
                Instant,
            ) -> Result<
                (
                    Option<alloc::sync::Arc<GraphCaptureReport>>,
                    DesktopPanelUpdate,
                ),
                String,
            > + 'static,
    ) -> Self {
        Self {
            controller,
            provider: Box::new(provider),
            view: None,
            update: DesktopPanelUpdate::default(),
            failed: false,
        }
    }
    /// Resume acquisition after the host resolves the reported producer failure.
    pub fn retry(&mut self) {
        self.failed = false;
    }
}
impl DesktopPanel for DesktopFrameGraphPanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self.controller.build();
        let result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        self.controller
            .handle_event(
                self.view.as_ref().ok_or("Frame graph is not rendered")?,
                target,
                event,
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    fn poll(&mut self, now: Instant) -> Result<bool, String> {
        if self.failed {
            return Ok(false);
        }
        let result = (self.provider)(now).and_then(|(report, update)| {
            let changed = report.is_some() || update.changed;
            if let Some(report) = report {
                self.controller
                    .publish(Some(report))
                    .map_err(|e| e.to_string())?;
            }
            self.update = update;
            Ok(changed)
        });
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn next_wake(&self) -> Option<Instant> {
        if self.failed {
            None
        } else {
            self.update.wake_at
        }
    }
    fn fenced(&self) -> bool {
        self.failed || self.update.fenced
    }
}
