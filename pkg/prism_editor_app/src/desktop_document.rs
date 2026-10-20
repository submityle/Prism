//! Native event-loop integration for the bounded document adapter.

use std::task::{Context, Poll};

use prism_editor_world::IoRequestId;

use crate::{
    ControllerLimits, DesktopError, DesktopRenderer, DocumentError, DocumentFileAdapter,
    DocumentIoEvent, DocumentSchemaCatalog, EditorDesktopUi,
};

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
        let request = self.adapter.submit_load(ui.controller(), &self.catalog)?;
        self.last_request = Some(request);
        Ok(request)
    }

    /// Cancels the accepted request while preserving save reconciliation rules.
    pub fn cancel(&mut self) -> bool {
        self.last_request
            .is_some_and(|request| self.adapter.cancel(request))
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
                let prepared = receipt
                    .prepare(&self.catalog, self.controller_limits)
                    .map_err(document_error)?;
                renderer.prepare_document(prepared.controller())?;
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
            DocumentIoEvent::Cancelled { .. } => {
                Err(DesktopError::Document("document I/O cancelled".into()))
            }
        }
    }
}

fn document_error(error: DocumentError) -> DesktopError {
    DesktopError::Document(error.to_string())
}
