//! Contributed policy, report truncation and exact-snapshot authorization.

extern crate alloc;

use alloc::{string::String, sync::Arc};
use core::sync::atomic::{AtomicUsize, Ordering};
use prism_editor_content::{
    validation::{
        AssetValidationRule, ValidationContext, ValidationError, ValidationLimits,
        ValidationOperation, ValidationPolicy, ValidationRegistry, ValidationRejection,
        ValidationSink,
    },
    AssetId, AssetKind, AssetRecord, ContentCatalog, ContentLimits, DiagnosticSeverity,
};

fn catalog() -> ContentCatalog {
    let mut catalog = ContentCatalog::new();
    catalog
        .insert(
            AssetRecord::new(
                AssetId::from_raw(1),
                "assets/a",
                AssetKind::new("data").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    catalog
}

struct Rule {
    id: &'static str,
    mode: u8,
    calls: Arc<AtomicUsize>,
}

impl AssetValidationRule for Rule {
    fn id(&self) -> &str {
        self.id
    }
    fn validate(
        &self,
        context: &ValidationContext<'_>,
        asset: &AssetRecord,
        sink: &mut ValidationSink<'_>,
    ) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(context.catalog.get(asset.id).is_some());
        match self.mode {
            1 => {
                sink.emit(DiagnosticSeverity::Info, "a-info", "informational")
                    .unwrap();
                sink.emit(DiagnosticSeverity::Error, "z-error", "blocking")
                    .unwrap();
            }
            2 => return Err("external dependency unavailable".into()),
            3 => {
                let _ = sink.emit(DiagnosticSeverity::Error, "bad code", "ignored failure");
            }
            4 => {
                let _ = sink.emit(
                    DiagnosticSeverity::Error,
                    "oversize",
                    "message beyond bounds",
                );
            }
            5 => {
                sink.emit(DiagnosticSeverity::Warning, "budget", "near budget")
                    .unwrap();
            }
            6 if context.operation == ValidationOperation::Cook => {
                sink.emit(
                    DiagnosticSeverity::Error,
                    "cook-budget",
                    "runtime budget exceeded",
                )
                .unwrap();
            }
            7 => {
                for _ in 0..3 {
                    let _ = sink.emit(DiagnosticSeverity::Info, "repeated", "same finding");
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn rule(id: &'static str, mode: u8) -> Arc<dyn AssetValidationRule> {
    Arc::new(Rule {
        id,
        mode,
        calls: Arc::new(AtomicUsize::new(0)),
    })
}

#[test]
fn display_truncation_cannot_hide_errors_or_authorize_save() {
    let catalog = catalog();
    let mut registry = ValidationRegistry::new(ValidationLimits {
        max_diagnostics: 1,
        ..ValidationLimits::default()
    });
    registry.register(rule("project.policy", 1)).unwrap();
    let receipt = registry
        .validate(&catalog, ValidationOperation::Save)
        .unwrap();
    assert!(receipt.truncated);
    assert_eq!(receipt.warning_count(), 0);
    assert!(receipt.is_complete());
    assert_eq!(receipt.issues().len(), 1);
    assert_eq!(receipt.issues()[0].severity, DiagnosticSeverity::Info);
    assert_eq!(receipt.issues()[0].path.as_deref(), Some("assets/a"));
    assert_eq!(receipt.error_count(), 1);
    assert_eq!(
        receipt
            .authorize(
                &catalog,
                &registry,
                ValidationOperation::Save,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::Errors)
    );
}

#[test]
fn receipts_reject_unrelated_divergent_and_superseded_snapshots() {
    let mut original = catalog();
    let mut registry = ValidationRegistry::default();
    let receipt = registry
        .validate(&original, ValidationOperation::Save)
        .unwrap();
    assert!(receipt
        .authorize(
            &original.clone(),
            &registry,
            ValidationOperation::Save,
            ValidationPolicy::default()
        )
        .is_ok());
    assert_eq!(
        receipt
            .authorize(
                &catalog(),
                &registry,
                ValidationOperation::Save,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::StaleCatalog)
    );
    assert_eq!(
        receipt
            .authorize(
                &original,
                &registry,
                ValidationOperation::Cook,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::WrongOperation)
    );
    let mut branch = original.clone();
    let mut next = branch.get(AssetId::from_raw(1)).unwrap().clone();
    next.favorite = true;
    branch.update(next).unwrap();
    let mut next = original.get(AssetId::from_raw(1)).unwrap().clone();
    next.pinned = true;
    original.update(next).unwrap();
    assert_eq!(branch.generation(), original.generation());
    let branch_receipt = registry
        .validate(&branch, ValidationOperation::Commit)
        .unwrap();
    assert_eq!(
        branch_receipt
            .authorize(
                &original,
                &registry,
                ValidationOperation::Commit,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::StaleCatalog)
    );
    assert_eq!(
        receipt
            .authorize(
                &original,
                &registry,
                ValidationOperation::Save,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::StaleCatalog)
    );
    let receipt = registry
        .validate(&original, ValidationOperation::Save)
        .unwrap();
    registry.register(rule("project.policy", 0)).unwrap();
    assert_eq!(
        receipt
            .authorize(
                &original,
                &registry,
                ValidationOperation::Save,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::StaleRules)
    );
    let receipt = registry
        .validate(&original, ValidationOperation::Save)
        .unwrap();
    assert!(!registry.unregister("missing"));
    assert!(receipt
        .authorize(
            &original,
            &registry,
            ValidationOperation::Save,
            ValidationPolicy::default()
        )
        .is_ok());
    assert!(registry.unregister("project.policy"));
    assert_eq!(
        receipt
            .authorize(
                &original,
                &registry,
                ValidationOperation::Save,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::StaleRules)
    );
}

#[test]
fn failed_callbacks_and_ignored_sink_errors_never_pass_the_gate() {
    let catalog = catalog();
    for (mode, limits, expected) in [
        (
            3,
            ValidationLimits::default(),
            ValidationError::InvalidIdentifier,
        ),
        (
            4,
            ValidationLimits {
                max_message_bytes: 3,
                ..ValidationLimits::default()
            },
            ValidationError::Limit("message bytes"),
        ),
        (
            7,
            ValidationLimits {
                max_findings: 2,
                ..ValidationLimits::default()
            },
            ValidationError::Limit("finding emissions"),
        ),
    ] {
        let mut registry = ValidationRegistry::new(limits);
        registry.register(rule("project.policy", mode)).unwrap();
        assert_eq!(
            registry.validate(&catalog, ValidationOperation::Cook).err(),
            Some(expected)
        );
    }
    let mut registry = ValidationRegistry::default();
    registry.register(rule("project.failure", 2)).unwrap();
    let receipt = registry
        .validate(&catalog, ValidationOperation::Cook)
        .unwrap();
    assert!(!receipt.is_complete());
    assert_eq!(receipt.issues()[0].code, "rule-failed");
    assert_eq!(
        receipt
            .authorize(
                &catalog,
                &registry,
                ValidationOperation::Cook,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::Incomplete)
    );
}

#[test]
fn operation_policy_and_warning_counts_survive_zero_display_capacity() {
    let catalog = catalog();
    let mut registry = ValidationRegistry::new(ValidationLimits {
        max_diagnostics: 0,
        ..ValidationLimits::default()
    });
    registry.register(rule("project.warning", 5)).unwrap();
    let receipt = registry
        .validate(&catalog, ValidationOperation::Commit)
        .unwrap();
    assert!(receipt.issues().is_empty());
    assert!(receipt.truncated);
    let validated = receipt
        .authorize(
            &catalog,
            &registry,
            ValidationOperation::Commit,
            ValidationPolicy::default(),
        )
        .unwrap();
    assert_eq!(validated.operation(), ValidationOperation::Commit);
    assert!(core::ptr::eq(validated.catalog(), &catalog));
    assert_eq!(
        receipt
            .authorize(
                &catalog,
                &registry,
                ValidationOperation::Commit,
                ValidationPolicy {
                    deny_warnings: true
                }
            )
            .err(),
        Some(ValidationRejection::Warnings)
    );
    registry.unregister("project.warning");
    registry.register(rule("project.cook", 6)).unwrap();
    assert!(registry
        .validate(&catalog, ValidationOperation::Save)
        .unwrap()
        .authorize(
            &catalog,
            &registry,
            ValidationOperation::Save,
            ValidationPolicy::default()
        )
        .is_ok());
    assert_eq!(
        registry
            .validate(&catalog, ValidationOperation::Cook)
            .unwrap()
            .error_count(),
        1
    );
}

#[test]
fn registration_and_work_budgets_reject_before_callbacks() {
    let mut catalog = catalog();
    let calls = Arc::new(AtomicUsize::new(0));
    for (limits, expected) in [
        (
            ValidationLimits {
                max_assets: 0,
                ..ValidationLimits::default()
            },
            "asset count",
        ),
        (
            ValidationLimits {
                max_evaluations: 0,
                ..ValidationLimits::default()
            },
            "rule evaluations",
        ),
        (
            ValidationLimits {
                max_dependencies: 0,
                ..ValidationLimits::default()
            },
            "dependency count",
        ),
    ] {
        catalog
            .add_dependency(AssetId::from_raw(1), AssetId::from_raw(99))
            .unwrap();
        let mut registry = ValidationRegistry::new(limits);
        registry
            .register(Arc::new(Rule {
                id: "project.rule",
                mode: 0,
                calls: calls.clone(),
            }))
            .unwrap();
        assert_eq!(
            registry.validate(&catalog, ValidationOperation::Save).err(),
            Some(ValidationError::Limit(expected))
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let mut registry = ValidationRegistry::new(ValidationLimits {
        max_rules: 1,
        ..ValidationLimits::default()
    });
    assert_eq!(
        registry.register(rule("builtin.graph", 0)),
        Err(ValidationError::ReservedIdentifier)
    );
    assert_eq!(
        registry.register(rule("bad id", 0)),
        Err(ValidationError::InvalidIdentifier)
    );
    registry.register(rule("project.rule", 0)).unwrap();
    assert_eq!(
        registry.register(rule("project.rule", 0)),
        Err(ValidationError::DuplicateRule("project.rule".into()))
    );
    assert_eq!(
        registry.register(rule("project.other", 0)),
        Err(ValidationError::Limit("rule count"))
    );
}

#[test]
fn builtin_truncation_and_incomplete_graph_checks_cannot_pass() {
    let mut catalog = ContentCatalog::with_limits(ContentLimits {
        max_diagnostics: 1,
        max_dependency_depth: 1,
        ..ContentLimits::default()
    })
    .unwrap();
    for id in 1..=3 {
        catalog
            .insert(
                AssetRecord::new(
                    AssetId::from_raw(id),
                    format!("a{id}"),
                    AssetKind::new("data").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
    }
    catalog
        .add_dependency(AssetId::from_raw(1), AssetId::from_raw(2))
        .unwrap();
    catalog
        .add_dependency(AssetId::from_raw(2), AssetId::from_raw(3))
        .unwrap();
    catalog
        .add_dependency(AssetId::from_raw(3), AssetId::from_raw(99))
        .unwrap();
    let builtin = catalog.validate();
    assert!(builtin.truncated);
    assert_eq!(
        builtin.diagnostics()[0].severity,
        DiagnosticSeverity::Warning
    );
    assert!(builtin.has_errors());
    assert_eq!(builtin.error_count(), 1);
    assert!(!builtin.is_complete());
    let registry = ValidationRegistry::default();
    let receipt = registry
        .validate(&catalog, ValidationOperation::Cook)
        .unwrap();
    assert_eq!(receipt.error_count(), 1);
    assert_eq!(
        receipt
            .authorize(
                &catalog,
                &registry,
                ValidationOperation::Cook,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::Incomplete)
    );
    catalog
        .remove_dependency(AssetId::from_raw(2), AssetId::from_raw(3))
        .unwrap();
    assert_eq!(
        registry
            .validate(&catalog, ValidationOperation::Cook)
            .unwrap()
            .authorize(
                &catalog,
                &registry,
                ValidationOperation::Cook,
                ValidationPolicy::default()
            )
            .err(),
        Some(ValidationRejection::Errors)
    );
}
