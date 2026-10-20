//! Actual Loom reconciliation and semantic event regression for E7 diagnostics.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use prism_editor_content::{
    validation::{
        AssetValidationRule, ValidationContext, ValidationLimits, ValidationOperation,
        ValidationRegistry, ValidationRejection, ValidationSink,
    },
    AssetId, AssetKind, AssetRecord, ContentCatalog, ContentLimits, DiagnosticSeverity,
};
use prism_editor_ui::{
    BoundContentDiagnostics, ContentDiagnosticsAction as Action,
    ContentDiagnosticsController as Controller, ContentDiagnosticsError as Error,
    DependencyDirection, UiEvent, UiTrigger, MAX_VISIBLE_ROWS,
};
use prism_ui::{BackendOp, Element, Key, RecordingBackend, Ui};

fn id(value: u128) -> AssetId {
    AssetId::from_raw(value)
}
fn asset(value: u128, dependencies: &[u128]) -> AssetRecord {
    let mut record = AssetRecord::new(
        id(value),
        format!("folder/{value}.asset"),
        AssetKind::new("test").unwrap(),
    )
    .unwrap();
    record
        .dependencies
        .extend(dependencies.iter().copied().map(id));
    record
}
fn catalog() -> ContentCatalog {
    let mut catalog = ContentCatalog::new();
    // Breadth-first order differs from identity order; missing 99 stays visible.
    for record in [
        asset(10, &[30, 40, 99]),
        asset(30, &[20]),
        asset(20, &[]),
        asset(40, &[]),
        asset(5, &[10]),
        asset(1, &[5]),
    ] {
        catalog.insert(record).unwrap();
    }
    catalog
}
fn nodes<'a>(element: &'a Element, out: &mut Vec<&'a Element>) {
    out.push(element);
    for child in element.child_elements() {
        nodes(child, out);
    }
}
fn find<'a>(element: &'a Element, key: &str) -> &'a Element {
    let mut all = Vec::new();
    nodes(element, &mut all);
    all.into_iter()
        .find(|el| matches!(el.explicit_key(), Some(Key::Str(value)) if value == key))
        .unwrap_or_else(|| panic!("missing Element {key}"))
}
fn text(element: &Element) -> String {
    let mut all = Vec::new();
    nodes(element, &mut all);
    all.iter()
        .filter_map(|el| el.text_content())
        .collect::<Vec<_>>()
        .join("\n")
}
fn dependency_keys(view: &BoundContentDiagnostics) -> Vec<String> {
    find(view.element(), "content-diagnostics:dependency-list")
        .child_elements()
        .iter()
        .filter_map(|el| match el.explicit_key() {
            Some(Key::Str(key)) if key.starts_with("content-diagnostics:dependency:") => {
                Some(key.clone())
            }
            _ => None,
        })
        .collect()
}
fn send(
    controller: &mut Controller,
    catalog: &ContentCatalog,
    registry: &ValidationRegistry,
    view: &BoundContentDiagnostics,
    key: &str,
) -> Action {
    let Some(Key::Str(target)) = find(view.element(), key).explicit_key() else {
        panic!("unkeyed target");
    };
    controller
        .handle_event(catalog, registry, view, target, &UiEvent::Activate)
        .unwrap()
}
fn reconcile(
    controller: &mut Controller,
    catalog: &ContentCatalog,
    registry: &ValidationRegistry,
    ui: &mut Ui<RecordingBackend>,
) -> BoundContentDiagnostics {
    let view = controller.build(catalog, registry);
    ui.update(view.element());
    view
}

#[test]
fn dependency_events_page_navigate_back_and_reverse_with_real_reconciliation() {
    let catalog = catalog();
    let registry = ValidationRegistry::default();
    let mut controller = Controller::default();
    controller
        .apply(&catalog, &registry, Action::Focus(Some(id(10))))
        .unwrap();
    controller
        .apply(&catalog, &registry, Action::PageSize(2))
        .unwrap();
    let first = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(first.element());
    assert_eq!(
        dependency_keys(&first),
        [
            "content-diagnostics:dependency:20",
            "content-diagnostics:dependency:30"
        ]
    );
    ui.backend_mut().clear();
    let unchanged = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(ui.backend().is_empty());
    assert_eq!(
        send(
            &mut controller,
            &catalog,
            &registry,
            &unchanged,
            "content-diagnostics:dependencies-next"
        ),
        Action::NextDependencies
    );
    let second = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(
        dependency_keys(&second),
        [
            "content-diagnostics:dependency:40",
            "content-diagnostics:dependency:99"
        ]
    );
    assert!(text(second.element()).contains("Missing asset 99"));
    assert!(!second
        .bindings()
        .any(|(key, _)| key == "content-diagnostics:dependency:99"));
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Remove { .. })));
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Create { .. })));
    send(
        &mut controller,
        &catalog,
        &registry,
        &second,
        "content-diagnostics:dependencies-previous",
    );
    let first = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(dependency_keys(&first), dependency_keys(&unchanged));
    send(
        &mut controller,
        &catalog,
        &registry,
        &first,
        "content-diagnostics:dependency:30",
    );
    let target = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(controller.selected_asset(), Some(id(30)));
    assert_eq!(
        dependency_keys(&target),
        ["content-diagnostics:dependency:20"]
    );
    send(
        &mut controller,
        &catalog,
        &registry,
        &target,
        "content-diagnostics:back",
    );
    let back = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(controller.selected_asset(), Some(id(10)));
    send(
        &mut controller,
        &catalog,
        &registry,
        &back,
        "content-diagnostics:dependents",
    );
    let incoming = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(
        dependency_keys(&incoming),
        [
            "content-diagnostics:dependency:1",
            "content-diagnostics:dependency:5"
        ]
    );
    assert!(!incoming
        .bindings()
        .any(|(key, _)| key == "content-diagnostics:dependencies-next"));
    assert_eq!(catalog.generation(), first.generation()); // UI only reads the catalog.
}

struct Check {
    calls: Arc<AtomicUsize>,
    fail: bool,
}
impl AssetValidationRule for Check {
    fn id(&self) -> &str {
        "test.asset-check"
    }
    fn validate(
        &self,
        context: &ValidationContext<'_>,
        _asset: &AssetRecord,
        sink: &mut ValidationSink<'_>,
    ) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        sink.emit(
            DiagnosticSeverity::Warning,
            "budget",
            &format!("{:?} budget finding", context.operation),
        )
        .map_err(|e| e.to_string())?;
        if self.fail {
            Err("provider failed".into())
        } else {
            Ok(())
        }
    }
}
fn registry(limits: ValidationLimits, fail: bool) -> (ValidationRegistry, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = ValidationRegistry::new(limits);
    registry
        .register(Arc::new(Check {
            calls: calls.clone(),
            fail,
        }))
        .unwrap();
    (registry, calls)
}

#[test]
fn explicit_validation_event_pages_real_receipt_locates_asset_and_never_reruns_on_render() {
    let catalog = catalog();
    let (registry, calls) = registry(ValidationLimits::default(), false);
    let mut controller = Controller::default();
    controller
        .apply(&catalog, &registry, Action::PageSize(2))
        .unwrap();
    let view = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    assert!(text(view.element()).contains("Validation not run"));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        send(
            &mut controller,
            &catalog,
            &registry,
            &view,
            "content-diagnostics:validate-cook"
        ),
        Action::RunValidation(ValidationOperation::Cook)
    );
    ui.backend_mut().clear();
    let report = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(calls.load(Ordering::Relaxed), catalog.len());
    let receipt = controller.receipt().unwrap();
    assert_eq!(receipt.issues().len(), 7); // Six contributed warnings + missing dependency.
    assert_eq!(receipt.error_count(), 1);
    assert_eq!(receipt.warning_count(), 6);
    assert_eq!(
        controller.report_status(&catalog, &registry),
        Some(Err(ValidationRejection::Errors))
    );
    assert!(text(report.element())
        .contains("Cook report: 1 errors, 6 warnings; complete=true; truncated=false"));
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Create { .. })));
    let first_id = receipt.issues()[0].asset.unwrap();
    send(
        &mut controller,
        &catalog,
        &registry,
        &report,
        "content-diagnostics:issue:0",
    );
    ui.backend_mut().clear();
    let located = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(controller.selected_asset(), Some(first_id));
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text.starts_with("Selected:"))));
    send(
        &mut controller,
        &catalog,
        &registry,
        &located,
        "content-diagnostics:issues-next",
    );
    let page = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(find(page.element(), "content-diagnostics:issue:2")
        .text_content()
        .is_none());
    assert_eq!(
        find(page.element(), "content-diagnostics:issue-list")
            .child_elements()
            .len(),
        2
    );
    send(
        &mut controller,
        &catalog,
        &registry,
        &page,
        "content-diagnostics:issues-previous",
    );
    let page = reconcile(&mut controller, &catalog, &registry, &mut ui);
    find(page.element(), "content-diagnostics:issue:0");
    assert_eq!(calls.load(Ordering::Relaxed), catalog.len());
    ui.backend_mut().clear();
    reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(ui.backend().is_empty());
}

#[test]
fn stale_receipts_keep_findings_but_disable_locating_until_explicit_rerun() {
    let mut catalog = catalog();
    let (mut registry, _) = registry(ValidationLimits::default(), false);
    let mut controller = Controller::default();
    controller
        .apply(
            &catalog,
            &registry,
            Action::RunValidation(ValidationOperation::Save),
        )
        .unwrap();
    let view = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    let mut record = catalog.get(id(10)).unwrap().clone();
    record.path = "new/path.asset".into();
    catalog.update(record).unwrap();
    assert_eq!(
        controller.handle_event(
            &catalog,
            &registry,
            &view,
            "content-diagnostics:issue:0",
            &UiEvent::Activate
        ),
        Err(Error::StaleView)
    );
    ui.backend_mut().clear();
    let stale = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(stale.element()).contains("stale catalog; rerun validation"));
    assert!(text(stale.element()).contains("[stale report]"));
    assert!(!stale
        .bindings()
        .any(|(key, _)| key.starts_with("content-diagnostics:issue:")));
    assert_eq!(
        controller.apply(&catalog, &registry, Action::LocateIssue(0)),
        Err(Error::UnlocatableIssue)
    );
    send(
        &mut controller,
        &catalog,
        &registry,
        &stale,
        "content-diagnostics:validate-commit",
    );
    let current = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(registry.unregister("test.asset-check"));
    assert_eq!(
        controller.handle_event(
            &catalog,
            &registry,
            &current,
            "content-diagnostics:issue:0",
            &UiEvent::Activate
        ),
        Err(Error::StaleView)
    );
    let stale = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(stale.element()).contains("stale rules; rerun validation"));
    send(
        &mut controller,
        &catalog,
        &registry,
        &stale,
        "content-diagnostics:validate-save",
    );
    let current = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(current.element()).contains("new/path.asset"));
    assert!(current
        .bindings()
        .any(|(key, _)| key == "content-diagnostics:issue:0"));
    assert_eq!(controller.receipt().unwrap().issues().len(), 1);
}

#[test]
fn traversal_limits_and_incomplete_truncated_reports_are_visible_in_loom() {
    let mut catalog = ContentCatalog::with_limits(ContentLimits {
        max_dependency_nodes: 1,
        max_dependency_depth: 1,
        ..ContentLimits::default()
    })
    .unwrap();
    for record in [asset(1, &[2]), asset(2, &[3]), asset(3, &[])] {
        catalog.insert(record).unwrap();
    }
    let (registry, calls) = registry(
        ValidationLimits {
            max_diagnostics: 1,
            ..ValidationLimits::default()
        },
        true,
    );
    let mut controller = Controller::default();
    controller
        .apply(&catalog, &registry, Action::Focus(Some(id(1))))
        .unwrap();
    let view = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    assert!(
        text(find(view.element(), "content-diagnostics:dependency-error"))
            .contains("Dependency traversal unavailable")
    );
    assert!(dependency_keys(&view).is_empty());
    assert!(!view
        .bindings()
        .any(|(key, _)| key == "content-diagnostics:dependencies-next"));
    send(
        &mut controller,
        &catalog,
        &registry,
        &view,
        "content-diagnostics:validate-save",
    );
    let report = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(
        text(report.element()).contains("complete=false; truncated=true; status=incomplete checks")
    );
    assert_eq!(controller.receipt().unwrap().issues().len(), 1);
    assert!(controller.receipt().unwrap().error_count() >= 3);
    assert!(calls.load(Ordering::Relaxed) > 0);
    // An assetless built-in depth warning must never acquire a phantom locate target.
    assert!(!report
        .bindings()
        .any(|(key, _)| key.starts_with("content-diagnostics:issue:")));
}

#[test]
fn all_bound_elements_are_unique_real_nodes_with_layout_and_wrong_events_are_rejected() {
    let catalog = catalog();
    let registry = ValidationRegistry::default();
    let mut controller = Controller::default();
    controller
        .apply(&catalog, &registry, Action::Focus(Some(id(10))))
        .unwrap();
    controller
        .apply(
            &catalog,
            &registry,
            Action::RunValidation(ValidationOperation::Save),
        )
        .unwrap();
    let view = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui.compute_layout(prism_ui::layout::Size::new(
        prism_ui::layout::AvailableSpace::Definite(1400.0),
        prism_ui::layout::AvailableSpace::Definite(1000.0),
    ));
    let mut all = Vec::new();
    nodes(view.element(), &mut all);
    let backend_ids: Vec<_> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::Create { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(all.len(), backend_ids.len());
    let mut keys = BTreeSet::new();
    for element in &all {
        if let Some(Key::Str(key)) = element.explicit_key() {
            assert!(keys.insert(key));
        }
    }
    for (key, trigger) in view.bindings() {
        assert_eq!(trigger, UiTrigger::Activate);
        let target = find(view.element(), key);
        let index = all.iter().position(|el| std::ptr::eq(*el, target)).unwrap();
        assert!(ui
            .backend()
            .ops()
            .iter()
            .any(|op| matches!(op, BackendOp::SetLayout { id, size, .. }
            if *id == backend_ids[index] && size.width > 0.0 && size.height > 0.0)));
    }
    let wrong = UiEvent::Select {
        mode: prism_editor_ui::SelectionMode::Replace,
    };
    assert_eq!(
        view.dispatch("content-diagnostics:validate-save", &wrong),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        view.dispatch("phantom", &UiEvent::Activate),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        controller.apply(&catalog, &registry, Action::PageSize(0)),
        Err(Error::InvalidPageSize)
    );
    assert_eq!(controller.selected_asset(), Some(id(10)));
}

#[test]
fn page_cap_generation_reset_deleted_selection_and_navigation_history_are_bounded() {
    let mut catalog = ContentCatalog::new();
    let count = MAX_VISIBLE_ROWS + 2;
    catalog
        .insert(asset(1, &(2..=count as u128).collect::<Vec<_>>()))
        .unwrap();
    for raw in 2..=count as u128 {
        catalog.insert(asset(raw, &[])).unwrap();
    }
    let registry = ValidationRegistry::default();
    let mut controller = Controller::default();
    controller
        .apply(&catalog, &registry, Action::Focus(Some(id(1))))
        .unwrap();
    controller
        .apply(&catalog, &registry, Action::PageSize(usize::MAX))
        .unwrap();
    let view = controller.build(&catalog, &registry);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    assert_eq!(dependency_keys(&view).len(), MAX_VISIBLE_ROWS);
    send(
        &mut controller,
        &catalog,
        &registry,
        &view,
        "content-diagnostics:dependencies-next",
    );
    let page = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(dependency_keys(&page).len(), 1);
    catalog.remove(id(2)).unwrap();
    let reset = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(dependency_keys(&reset).len(), MAX_VISIBLE_ROWS);
    assert!(text(reset.element()).contains("Missing asset 2"));
    catalog.remove(id(1)).unwrap();
    let removed = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(removed.element()).contains("Missing selected asset 1"));
    assert!(dependency_keys(&removed).is_empty());
    for raw in 3..=80 {
        controller
            .apply(&catalog, &registry, Action::Focus(Some(id(raw))))
            .unwrap();
    }
    for _ in 0..prism_editor_ui::MAX_DEPENDENCY_NAVIGATION_HISTORY {
        controller.apply(&catalog, &registry, Action::Back).unwrap();
    }
    assert_eq!(controller.selected_asset(), Some(id(16)));
    let view = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(!view
        .bindings()
        .any(|(key, _)| key == "content-diagnostics:back"));
    // Direction stays a concrete command separate from the general EditorAction.
    assert_eq!(
        send(
            &mut controller,
            &catalog,
            &registry,
            &view,
            "content-diagnostics:dependencies"
        ),
        Action::Direction(DependencyDirection::Dependencies)
    );
}

#[test]
fn real_cycles_clean_reports_and_failed_validation_preserve_truthful_panel_state() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, &[2])).unwrap();
    catalog.insert(asset(2, &[1])).unwrap();
    let registry = ValidationRegistry::new(ValidationLimits {
        max_assets: 2,
        ..ValidationLimits::default()
    });
    let mut controller = Controller::default();
    let view = controller.build(&catalog, &registry);
    let mut other = Controller::default();
    other.build(&catalog, &registry);
    assert_eq!(
        other.handle_event(
            &catalog,
            &registry,
            &view,
            "content-diagnostics:validate-save",
            &UiEvent::Activate
        ),
        Err(Error::StaleView)
    );
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    send(
        &mut controller,
        &catalog,
        &registry,
        &view,
        "content-diagnostics:validate-save",
    );
    let cycle = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(cycle.element()).contains("builtin/dependency-cycle"));
    assert_eq!(controller.receipt().unwrap().error_count(), 1);
    send(
        &mut controller,
        &catalog,
        &registry,
        &cycle,
        "content-diagnostics:issue:0",
    );
    let root = controller.selected_asset().unwrap();
    let focused = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(dependency_keys(&focused).len(), 1); // Cycles exclude their root.
    catalog.remove_dependency(id(2), id(1)).unwrap();
    let stale = reconcile(&mut controller, &catalog, &registry, &mut ui);
    send(
        &mut controller,
        &catalog,
        &registry,
        &stale,
        "content-diagnostics:validate-save",
    );
    let clean = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert_eq!(controller.report_status(&catalog, &registry), Some(Ok(())));
    assert!(text(clean.element()).contains("No retained findings"));
    assert!(text(clean.element()).contains("complete=true; truncated=false; status=current"));
    let retained = controller.receipt().unwrap() as *const _;
    catalog.insert(asset(3, &[])).unwrap();
    let stale = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(matches!(
        controller.handle_event(
            &catalog,
            &registry,
            &stale,
            "content-diagnostics:validate-cook",
            &UiEvent::Activate
        ),
        Err(Error::Validation(
            prism_editor_content::validation::ValidationError::Limit("asset count")
        ))
    ));
    assert_eq!(controller.receipt().unwrap() as *const _, retained);
    assert_eq!(controller.selected_asset(), Some(root));
    ui.backend_mut().clear();
    let unchanged = reconcile(&mut controller, &catalog, &registry, &mut ui);
    assert!(text(unchanged.element()).contains("stale catalog; rerun validation"));
    assert!(ui.backend().is_empty());
}
