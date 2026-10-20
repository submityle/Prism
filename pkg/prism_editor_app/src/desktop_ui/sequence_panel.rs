//! Native complete-project authoring. Evaluation remains explicit kernel diagnostics.
use super::*;
use prism_editor_timeline::{SequenceProjectEditor, SequenceProjectToken};
use prism_editor_ui::{
    BoundSequencePanel, SequenceClipDraft, SequencePanelAction, SequencePanelController,
    SequencePanelError, SequencePanelLimits,
};
use std::time::Instant;

/// The host's actual complete sequence project and bounded authoring controller.
pub struct DesktopSequencePanel {
    /// Actual model/history. Host changes must invalidate the retained view.
    pub editor: SequenceProjectEditor,
    /// Actual controller; diagnostic samples/events are available here.
    pub controller: SequencePanelController,
    pub(super) view: Option<BoundSequencePanel>,
    pub(super) saved: SequenceProjectToken,
    pub(super) limits: SequencePanelLimits,
    pub(super) file:
        Option<panel_storage::PanelFile<prism_editor_timeline::SequenceProjectStorageService>>,
}
impl DesktopSequencePanel {
    /// Use existing project data and explicit host policy; creates no project documents.
    pub fn new(
        editor: SequenceProjectEditor,
        limits: SequencePanelLimits,
    ) -> Result<Self, SequencePanelError> {
        let saved = editor.snapshot_token();
        Ok(Self {
            editor,
            controller: SequencePanelController::new(limits)?,
            view: None,
            saved,
            limits,
            file: None,
        })
    }
    /// Acknowledge only a durable publication of the exact current project.
    /// An unapplied clip draft remains fenced even after this succeeds.
    pub fn acknowledge_saved(&mut self, token: &SequenceProjectToken) -> bool {
        if *token != self.editor.snapshot_token() {
            return false;
        }
        self.saved = token.clone();
        true
    }
    pub(super) fn dirty(&self) -> bool {
        self.saved != self.editor.snapshot_token()
    }
    /// Whether the actual inspector differs from its selected clip or new-clip baseline.
    /// This only inspects clip metadata; it never clones timeline payloads.
    pub fn has_clip_draft(&self) -> bool {
        let baseline = self
            .controller
            .selected_clip()
            .and_then(|id| {
                self.editor.document().clips().find(|(parent, clip)| {
                    Some(*parent) == self.controller.selected() && clip.id == id
                })
            })
            .map(|(_, clip)| {
                let (numerator, denominator) = clip.rate.ratio();
                SequenceClipDraft {
                    source: Some(clip.source),
                    parent_start: clip.parent.start(),
                    parent_end: clip.parent.end(),
                    source_start: clip.source_range.start(),
                    source_end: clip.source_range.end(),
                    numerator,
                    denominator,
                    repetitions: clip.repetitions,
                }
            })
            .unwrap_or_default();
        self.controller.draft() != baseline
    }
    /// Explicitly restore the actual selected clip, or clear a new clip draft.
    /// Does not acknowledge a model edit or file failure.
    pub fn discard_clip_draft(&mut self) -> Result<(), String> {
        let action = self
            .controller
            .selected_clip()
            .map_or(SequencePanelAction::NewClip, SequencePanelAction::Locate);
        self.controller
            .apply(&mut self.editor, action)
            .map_err(|e| e.to_string())?;
        self.view = None;
        Ok(())
    }
}
impl DesktopPanel for DesktopSequencePanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self
            .controller
            .build(&self.editor)
            .map_err(|e| e.to_string())?;
        let mut result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        if self.has_clip_draft() || self.controller.error().is_some() {
            let key = "sequence:host:discard-draft";
            result.bindings.push((key.into(), UiTrigger::Activate));
            result.element = Element::box_()
                .child(
                    Element::box_()
                        .key_str(key)
                        .class("desktop-button")
                        .child(Element::text("Discard clip draft / clear rejected action")),
                )
                .child(result.element);
        }
        if let Some(file) = &self.file {
            file.decorate(
                "sequence",
                &mut result,
                self.file_ready().is_ok(),
                !self.dirty(),
            );
        }
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        if target.starts_with("sequence:file:") {
            return self.file_event(target, event);
        }
        if target == "sequence:host:discard-draft" && matches!(event, UiEvent::Activate) {
            return self.discard_clip_draft();
        }
        let view = self.view.as_ref().ok_or("Sequence is not rendered")?;
        // Dispatch is used only to classify destructive-to-draft actions. Actual
        // execution still enforces the controller identity and exact project token.
        let action = view.dispatch(target, event);
        if self.has_clip_draft()
            && action.as_ref().is_ok_and(|action| {
                !matches!(
                    action,
                    SequencePanelAction::Source(_)
                        | SequencePanelAction::Field(..)
                        | SequencePanelAction::AddClip
                        | SequencePanelAction::SaveClip
                        | SequencePanelAction::Page(..)
                        | SequencePanelAction::Preview(_)
                        | SequencePanelAction::Evaluate
                )
            })
        {
            return Err(
                "Apply or explicitly discard the clip draft before navigating/editing".into(),
            );
        }
        self.controller
            .handle_event(&mut self.editor, view, target, event)
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
            || self.has_clip_draft()
            || self.controller.error().is_some()
            || self
                .file
                .as_ref()
                .is_some_and(panel_storage::PanelFile::fenced)
    }
    fn draft_revision(&self) -> u64 {
        self.editor.revision()
    }
}
