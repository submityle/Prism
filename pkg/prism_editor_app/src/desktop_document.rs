//! Native event-loop integration for the bounded document adapter.

use std::task::{Context, Poll};

use prism_editor_world::IoRequestId;

use crate::{
    ControllerLimits, DesktopError, DesktopRenderer, DocumentError, DocumentFileAdapter,
    DocumentIoEvent, DocumentSchemaCatalog, EditorDesktopUi,
};

/// Result of requesting cancellation for the native scene document operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DesktopDocumentCancelOutcome {
    /// The active operation was removed and its terminal cancellation is queued.
    Accepted,
    /// Cancellation was already requested; polling must consume its terminal event.
    AwaitingTerminal,
    /// No document operation is awaiting a terminal event.
    Idle,
}

/// Owns one document adapter and its schema policy for a native editor run.
pub struct DesktopDocumentSession {
    adapter: DocumentFileAdapter,
    catalog: DocumentSchemaCatalog,
    controller_limits: ControllerLimits,
    last_request: Option<IoRequestId>,
}

impl DesktopDocumentSession {
    /// Creates a native document session around a production file adapter.
    pub fn new(
        adapter: DocumentFileAdapter,
        catalog: DocumentSchemaCatalog,
        controller_limits: ControllerLimits,
    ) -> Self {
        Self {
            adapter,
            catalog,
            controller_limits,
            last_request: None,
        }
    }

    /// Whether filesystem work is awaiting a terminal event.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.adapter.is_busy()
    }

    /// Whether a failed or cancelled save requires reconciliation.
    #[must_use]
    pub fn reconciliation_required(&self) -> bool {
        self.adapter.reconciliation_required()
    }

    /// Returns the last accepted request, if any.
    #[must_use]
    pub const fn last_request(&self) -> Option<IoRequestId> {
        self.last_request
    }

    /// Starts a bounded save without clearing the UI dirty state.
    pub fn save(&mut self, ui: &mut EditorDesktopUi) -> Result<IoRequestId, DocumentError> {
        if ui.has_uncommitted_edit() {
            return Err(DocumentError::Busy);
        }
        if ui.content_catalog().is_some() {
            use prism_editor_content::validation::ValidationOperation;
            ui.validate_content(ValidationOperation::Save)
                .map_err(DocumentError::Ui)?;
            // Hold both exact content/rule snapshots across save preparation.
            let _authorized = ui
                .validated_content(ValidationOperation::Save)
                .map_err(|error| {
                    DocumentError::Ui(crate::DesktopUiError::Validation(error.to_string()))
                })?;
            let request = self.adapter.submit_save(ui.controller(), &self.catalog)?;
            self.last_request = Some(request);
            return Ok(request);
        }
        let request = self.adapter.submit_save(ui.controller(), &self.catalog)?;
        self.last_request = Some(request);
        Ok(request)
    }

    /// Starts a bounded load; installation is deferred until polling receives
    /// a complete and schema-validated receipt.
    pub fn load(&mut self, ui: &EditorDesktopUi) -> Result<IoRequestId, DocumentError> {
        if ui.has_uncommitted_edit() {
            return Err(DocumentError::Busy);
        }
        if ui.controller().is_dirty() {
            return Err(DocumentError::UnsavedChanges);
        }
        let request = self.adapter.submit_load(ui.controller(), &self.catalog)?;
        self.last_request = Some(request);
        Ok(request)
    }

    /// Retries the exact retained save bytes, revision and durable base after a
    /// failed or cancelled save. Current UI edits are not recaptured.
    pub fn retry_save(&mut self) -> Result<IoRequestId, DocumentError> {
        let request = self.adapter.retry_save()?;
        self.last_request = Some(request);
        Ok(request)
    }

    /// Requests cancellation while preserving the pending terminal event and
    /// save reconciliation rules.
    #[must_use]
    pub fn cancel(&mut self) -> DesktopDocumentCancelOutcome {
        let Some(request) = self.last_request else {
            return DesktopDocumentCancelOutcome::Idle;
        };
        if self.adapter.cancel(request) {
            DesktopDocumentCancelOutcome::Accepted
        } else if self.adapter.is_busy() {
            DesktopDocumentCancelOutcome::AwaitingTerminal
        } else {
            self.last_request = None;
            DesktopDocumentCancelOutcome::Idle
        }
    }

    /// Polls one terminal event and applies it atomically to the UI/renderer.
    pub fn poll<R: DesktopRenderer>(
        &mut self,
        context: &mut Context<'_>,
        ui: &mut EditorDesktopUi,
        renderer: &mut R,
    ) -> Result<Poll<crate::desktop::DesktopWork>, DesktopError> {
        let event = match self.adapter.poll(context) {
            Poll::Pending => return Ok(Poll::Pending),
            Poll::Ready(None) => return Ok(Poll::Ready(crate::desktop::DesktopWork::default())),
            Poll::Ready(Some(event)) => event,
        };
        self.last_request = None;
        match event {
            DocumentIoEvent::Saved(receipt) => {
                receipt
                    .acknowledge_ui(ui, &self.catalog)
                    .map_err(document_error)?;
                Ok(Poll::Ready(crate::desktop::DesktopWork {
                    redraw: true,
                    pending: false,
                }))
            }
            DocumentIoEvent::Loaded(receipt) => {
                ensure_load_install_idle(ui)?;
                let prepared = receipt
                    .prepare(&self.catalog, self.controller_limits)
                    .map_err(document_error)?;
                prepare_renderer_document(renderer, prepared.controller())?;
                if let Err(error) = prepared.install_ui(&mut self.adapter, ui, &self.catalog) {
                    renderer.discard_document();
                    return Err(document_error(error));
                }
                renderer.commit_document();
                Ok(Poll::Ready(crate::desktop::DesktopWork {
                    redraw: true,
                    pending: false,
                }))
            }
            DocumentIoEvent::Failed { error, .. } => Err(DesktopError::Document(format!(
                "document I/O failed: {error:?}"
            ))),
            DocumentIoEvent::Cancelled { .. } => Ok(Poll::Ready(crate::desktop::DesktopWork {
                redraw: true,
                pending: false,
            })),
        }
    }
}

fn ensure_load_install_idle(ui: &EditorDesktopUi) -> Result<(), DesktopError> {
    if ui.has_uncommitted_edit() {
        return Err(DesktopError::Document(
            "document load cannot replace an active editor draft or interaction".into(),
        ));
    }
    Ok(())
}

fn prepare_renderer_document<R: DesktopRenderer>(
    renderer: &mut R,
    controller: &crate::EditorController,
) -> Result<(), DesktopError> {
    if let Err(error) = renderer.prepare_document(controller) {
        renderer.discard_document();
        return Err(error);
    }
    Ok(())
}

fn document_error(error: DocumentError) -> DesktopError {
    DesktopError::Document(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DesktopGpu, DesktopViewportBounds, DesktopWork};
    use prism_editor_ui::SelectionMode;

    #[derive(Default)]
    struct RejectingRenderer {
        candidate: bool,
        prepare_count: u32,
        discard_count: u32,
    }

    impl DesktopRenderer for RejectingRenderer {
        fn initialize(&mut self, _gpu: &DesktopGpu) -> Result<(), DesktopError> {
            Ok(())
        }

        fn encode(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
            _encoder: &mut wgpu::CommandEncoder,
            _target: &wgpu::TextureView,
            _bounds: DesktopViewportBounds,
        ) -> Result<(), DesktopError> {
            Ok(())
        }

        fn pointer(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
            _position: [f32; 2],
            _mode: SelectionMode,
        ) -> Result<bool, DesktopError> {
            Ok(false)
        }

        fn poll(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
        ) -> Result<DesktopWork, DesktopError> {
            Ok(DesktopWork::default())
        }

        fn prepare_document(
            &mut self,
            _controller: &crate::EditorController,
        ) -> Result<(), DesktopError> {
            self.prepare_count += 1;
            self.candidate = true;
            Err(DesktopError::Renderer("document scene rejected".into()))
        }

        fn discard_document(&mut self) {
            self.discard_count += 1;
            self.candidate = false;
        }

        fn suspend(&mut self) {}
    }

    #[test]
    fn rejected_renderer_candidate_is_discarded_before_load_returns() {
        let mut renderer = RejectingRenderer::default();

        assert_eq!(
            prepare_renderer_document(&mut renderer, &crate::EditorController::new()),
            Err(DesktopError::Renderer("document scene rejected".into()))
        );
        assert_eq!(renderer.prepare_count, 1);
        assert_eq!(renderer.discard_count, 1);
        assert!(!renderer.candidate);
    }

    #[test]
    fn load_install_rejects_an_interaction_started_after_submission() {
        let mut ui = EditorDesktopUi::new(crate::EditorController::new()).expect("empty UI");
        ui.controller_mut().begin_edit().expect("begin interaction");

        assert_eq!(
            ensure_load_install_idle(&ui),
            Err(DesktopError::Document(
                "document load cannot replace an active editor draft or interaction".into()
            ))
        );
    }

    #[test]
    fn load_install_accepts_an_idle_ui() {
        let ui = EditorDesktopUi::new(crate::EditorController::new()).expect("empty UI");

        assert!(ensure_load_install_idle(&ui).is_ok());
    }
}
