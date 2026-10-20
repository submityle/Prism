//! Exact content receipts checked at the operation preparation boundary.
use super::*;
use prism_editor_content::validation::{
    ValidatedCatalog, ValidationLimits, ValidationOperation, ValidationPolicy, ValidationReceipt,
    ValidationRegistry, ValidationRejection,
};

pub(super) struct NativeValidation {
    pub registry: ValidationRegistry,
    policy: ValidationPolicy,
    pub(super) receipt: Option<ValidationReceipt>,
    pub message: String,
    snapshot: Option<prism_editor_content::ContentSnapshotToken>,
}

impl Default for NativeValidation {
    fn default() -> Self {
        Self {
            registry: ValidationRegistry::new(ValidationLimits::default()),
            policy: ValidationPolicy::default(),
            receipt: None,
            message: "Content unchecked".into(),
            snapshot: None,
        }
    }
}

impl NativeValidation {
    pub fn stale(&self, catalog: &ContentCatalog) -> bool {
        self.snapshot
            .as_ref()
            .is_some_and(|token| !catalog.matches_snapshot(token))
    }

    pub fn uncheck(&mut self) {
        self.snapshot = None;
        self.receipt = None;
        self.message = "Content unchecked".into();
    }
}

impl EditorDesktopUi {
    /// Replace trusted host rules/policy and revoke previous authorizations.
    pub fn mount_content_validation(
        &mut self,
        registry: ValidationRegistry,
        policy: ValidationPolicy,
    ) {
        self.validation = NativeValidation {
            registry,
            policy,
            receipt: None,
            message: "Content rules changed; validate again".into(),
            snapshot: None,
        };
        self.invalidated = true;
    }

    /// Run the bounded synchronous checks against the exact current catalog.
    /// Host callbacks must be cheap; their execution time is not preempted.
    pub fn validate_content(
        &mut self,
        operation: ValidationOperation,
    ) -> Result<(), DesktopUiError> {
        self.validation.receipt = None;
        self.validation.snapshot = None;
        self.invalidated = true;
        let catalog = self
            .content
            .as_ref()
            .ok_or(DesktopUiError::InvalidProjection)?;
        match self
            .validation
            .registry
            .validate(&catalog.catalog, operation)
        {
            Ok(receipt) => {
                self.validation.message = format!(
                    "Content: {} errors, {} warnings{}",
                    receipt.error_count(),
                    receipt.warning_count(),
                    if receipt.is_complete() {
                        ""
                    } else {
                        " (incomplete)"
                    }
                );
                self.validation.receipt = Some(receipt);
                self.validation.snapshot = Some(catalog.catalog.snapshot_token());
                Ok(())
            }
            Err(error) => {
                self.validation.message = format!("Content validation failed: {error}");
                Err(DesktopUiError::Validation(error.to_string()))
            }
        }
    }

    /// Exact findings for host navigation. Counts include omitted diagnostics.
    #[must_use]
    pub fn content_validation_receipt(&self) -> Option<&ValidationReceipt> {
        self.validation.receipt.as_ref()
    }

    /// Borrow the authorized snapshot while preparing Save/Commit/Cook work.
    /// Later asynchronous publication still needs its destination revision fence.
    pub fn validated_content(
        &self,
        operation: ValidationOperation,
    ) -> Result<ValidatedCatalog<'_>, ValidationRejection> {
        let receipt = self
            .validation
            .receipt
            .as_ref()
            .ok_or(ValidationRejection::Incomplete)?;
        let content = self
            .content
            .as_ref()
            .ok_or(ValidationRejection::StaleCatalog)?;
        receipt.authorize(
            &content.catalog,
            &self.validation.registry,
            operation,
            self.validation.policy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use prism_editor_content::{
        validation::{AssetValidationRule, ValidationContext, ValidationSink},
        AssetId, AssetKind, AssetRecord, DiagnosticSeverity,
    };

    struct Warning;
    impl AssetValidationRule for Warning {
        fn id(&self) -> &str {
            "host.warning"
        }
        fn validate(
            &self,
            _: &ValidationContext<'_>,
            _: &AssetRecord,
            sink: &mut ValidationSink<'_>,
        ) -> Result<(), String> {
            sink.emit(DiagnosticSeverity::Warning, "budget", "host warning")
                .map_err(|e| e.to_string())
        }
    }
    #[test]
    fn native_receipt_exact_catalog_rules_operation_and_warning_policy() {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(
                AssetRecord::new(
                    AssetId::from_raw(1),
                    "a.png",
                    AssetKind::new("image").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
        ui.mount_content_browser(catalog).unwrap();
        ui.validate_content(ValidationOperation::Save).unwrap();
        assert!(ui.validated_content(ValidationOperation::Save).is_ok());
        assert!(matches!(
            ui.validated_content(ValidationOperation::Cook),
            Err(ValidationRejection::WrongOperation)
        ));
        ui.content_catalog_mut()
            .unwrap()
            .insert(
                AssetRecord::new(
                    AssetId::from_raw(2),
                    "b.png",
                    AssetKind::new("image").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        assert!(matches!(
            ui.validated_content(ValidationOperation::Save),
            Err(ValidationRejection::StaleCatalog)
        ));
        ui.refresh().unwrap();
        assert_eq!(ui.validation.message, "Content changed; validate again");
        let mut registry = ValidationRegistry::new(ValidationLimits::default());
        registry.register(Arc::new(Warning)).unwrap();
        ui.mount_content_validation(
            registry,
            ValidationPolicy {
                deny_warnings: true,
            },
        );
        assert!(matches!(
            ui.validated_content(ValidationOperation::Save),
            Err(ValidationRejection::Incomplete)
        ));
        ui.validate_content(ValidationOperation::Save).unwrap();
        assert_eq!(ui.content_validation_receipt().unwrap().warning_count(), 2);
        assert!(matches!(
            ui.validated_content(ValidationOperation::Save),
            Err(ValidationRejection::Warnings)
        ));
        ui.refresh().unwrap();
        assert!(ui.draw_list().commands().iter().any(|cmd| matches!(cmd, DrawCommand::Glyph(g) if ui.draw_list().text(g.text).contains("0 errors, 2 warnings"))));
        assert_eq!(ui.controller().document_history_lengths(), (0, 0));
    }
}
