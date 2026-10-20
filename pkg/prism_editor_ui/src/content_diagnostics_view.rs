//! Keyed Loom dependency rows and real validation findings.

use std::{collections::BTreeMap, sync::Arc};

use prism_editor_content::{
    validation::{ValidationOperation, ValidationRegistry, ValidationRejection},
    AssetId, ContentCatalog, ContentError,
};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};

use crate::{
    ContentDiagnosticsAction as Action, ContentDiagnosticsController, ContentDiagnosticsError,
    DependencyDirection, UiEvent, UiTrigger,
};

/// A real Loom subtree plus exact interactive keys and their owned commands.
/// The host retains this value for routing events until the next reconciliation.
#[derive(Clone, Debug)]
pub struct BoundContentDiagnostics {
    element: Element,
    bindings: BTreeMap<String, Action>,
    generation: u64,
    pub(crate) revision: u64,
    pub(crate) status: Option<Result<(), ValidationRejection>>,
    pub(crate) identity: Arc<()>,
}

impl BoundContentDiagnostics {
    /// Tree to mount or reconcile inside a host-owned dock body.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }

    /// Interactive Element keys and their semantic triggers, deterministically ordered.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings
            .keys()
            .map(|key| (key.as_str(), UiTrigger::Activate))
    }

    /// Catalog generation represented by this tree.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Resolve a matching semantic Activate event into an owned command. Use
    /// `ContentDiagnosticsController::handle_event` to also enforce freshness.
    pub fn dispatch(
        &self,
        target: &str,
        event: &UiEvent,
    ) -> Result<Action, ContentDiagnosticsError> {
        if *event != UiEvent::Activate {
            return Err(ContentDiagnosticsError::UnboundEvent);
        }
        self.bindings
            .get(target)
            .cloned()
            .ok_or(ContentDiagnosticsError::UnboundEvent)
    }
}

pub(crate) fn build(
    controller: &ContentDiagnosticsController,
    catalog: &ContentCatalog,
    registry: &ValidationRegistry,
    dependencies: Result<Vec<AssetId>, ContentError>,
) -> BoundContentDiagnostics {
    let mut view = BoundContentDiagnostics {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        generation: catalog.generation(),
        revision: controller.revision,
        status: controller.report_status(catalog, registry),
        identity: controller.identity.clone(),
    };
    let focus = controller.selected.map_or_else(
        || "No asset selected".to_owned(),
        |id| {
            catalog.get(id).map_or_else(
                || format!("Missing selected asset {}", id.raw()),
                |record| format!("Selected: {} [asset {}]", record.path, id.raw()),
            )
        },
    );
    let dependencies = dependency_list(&mut view, controller, catalog, dependencies);
    let findings = validation_list(&mut view, controller, catalog, registry);
    view.element = column()
        .key_str("content-diagnostics:panel")
        .class("prism-editor-content-diagnostics")
        .child(Element::text("Content dependencies and validation"))
        .child(Element::text(focus).key_str("content-diagnostics:selection"))
        .child(dependencies)
        .child(findings);
    view
}

fn dependency_list(
    view: &mut BoundContentDiagnostics,
    controller: &ContentDiagnosticsController,
    catalog: &ContentCatalog,
    dependencies: Result<Vec<AssetId>, ContentError>,
) -> Element {
    let controls = row()
        .child(button(
            view,
            "content-diagnostics:back",
            "Back",
            (!controller.history.is_empty()).then_some(Action::Back),
        ))
        .child(button(
            view,
            "content-diagnostics:clear",
            "Clear asset",
            controller.selected.map(|_| Action::Focus(None)),
        ))
        .child(button(
            view,
            "content-diagnostics:dependencies",
            "Dependencies (outgoing)",
            Some(Action::Direction(DependencyDirection::Dependencies)),
        ))
        .child(button(
            view,
            "content-diagnostics:dependents",
            "Dependents (incoming)",
            Some(Action::Direction(DependencyDirection::Dependents)),
        ));
    let mut section = column()
        .child(controls)
        .child(Element::text(match controller.direction {
            DependencyDirection::Dependencies => "Transitive dependencies, sorted by asset ID",
            DependencyDirection::Dependents => "Transitive dependents, sorted by asset ID",
        }));
    let mut list = column().key_str("content-diagnostics:dependency-list");
    match dependencies {
        Err(error) => {
            list = list.child(
                Element::text(format!("Dependency traversal unavailable: {error}"))
                    .key_str("content-diagnostics:dependency-error"),
            );
        }
        Ok(ids) => {
            let offset = controller.dependency_offset;
            if ids.is_empty() {
                list = list.child(Element::text(if controller.selected.is_none() {
                    "Select an asset to browse dependencies"
                } else {
                    "No dependencies in this direction"
                }));
            }
            for id in ids.iter().skip(offset).take(controller.page_size) {
                let record = catalog.get(*id);
                let text = record.map_or_else(
                    || format!("Missing asset {}", id.raw()),
                    |record| format!("{} [asset {}]", record.path, id.raw()),
                );
                list = list.child(button(
                    view,
                    &format!("content-diagnostics:dependency:{}", id.raw()),
                    &text,
                    record.map(|_| Action::Focus(Some(*id))),
                ));
            }
            section = section.child(
                row()
                    .child(button(
                        view,
                        "content-diagnostics:dependencies-previous",
                        "Previous dependencies",
                        (offset != 0).then_some(Action::PreviousDependencies),
                    ))
                    .child(Element::text(format!(
                        "Dependencies {}..{} of {}",
                        offset,
                        offset.saturating_add(controller.page_size).min(ids.len()),
                        ids.len()
                    )))
                    .child(button(
                        view,
                        "content-diagnostics:dependencies-next",
                        "Next dependencies",
                        (offset.saturating_add(controller.page_size) < ids.len())
                            .then_some(Action::NextDependencies),
                    )),
            );
        }
    }
    section.child(list)
}

fn validation_list(
    view: &mut BoundContentDiagnostics,
    controller: &ContentDiagnosticsController,
    catalog: &ContentCatalog,
    registry: &ValidationRegistry,
) -> Element {
    let mut controls = row();
    for (name, operation) in [
        ("save", ValidationOperation::Save),
        ("commit", ValidationOperation::Commit),
        ("cook", ValidationOperation::Cook),
    ] {
        controls = controls.child(button(
            view,
            &format!("content-diagnostics:validate-{name}"),
            &format!("Validate for {name}"),
            Some(Action::RunValidation(operation)),
        ));
    }
    let mut section = column().child(controls);
    let mut list = column().key_str("content-diagnostics:issue-list");
    let Some(receipt) = controller.receipt.as_ref() else {
        return section.child(list.child(Element::text("Validation not run")));
    };
    section = section.child(
        Element::text(format!(
            "{:?} report: {} errors, {} warnings; complete={}; truncated={}; status={}",
            controller.operation,
            receipt.error_count(),
            receipt.warning_count(),
            receipt.is_complete(),
            receipt.truncated,
            status_name(view.status)
        ))
        .key_str("content-diagnostics:report-status"),
    );
    let current = controller.report_is_current(catalog, registry);
    let offset = controller.issue_offset;
    for (index, issue) in receipt
        .issues()
        .iter()
        .enumerate()
        .skip(offset)
        .take(controller.page_size)
    {
        let target = issue.asset.filter(|id| catalog.get(*id).is_some());
        let text = format!(
            "{:?} {}/{} | {} | {}{}",
            issue.severity,
            issue.rule,
            issue.code,
            issue.path.as_deref().unwrap_or("(no asset path)"),
            issue.message,
            if !current {
                " [stale report]"
            } else if target.is_none() {
                " [no asset location]"
            } else {
                ""
            }
        );
        list = list.child(button(
            view,
            &format!("content-diagnostics:issue:{index}"),
            &text,
            (current && target.is_some()).then_some(Action::LocateIssue(index)),
        ));
    }
    if receipt.issues().is_empty() {
        list = list.child(Element::text("No retained findings"));
    }
    section
        .child(
            row()
                .child(button(
                    view,
                    "content-diagnostics:issues-previous",
                    "Previous findings",
                    (offset != 0).then_some(Action::PreviousIssues),
                ))
                .child(Element::text(format!(
                    "Findings {}..{} of {} retained",
                    offset,
                    offset
                        .saturating_add(controller.page_size)
                        .min(receipt.issues().len()),
                    receipt.issues().len()
                )))
                .child(button(
                    view,
                    "content-diagnostics:issues-next",
                    "Next findings",
                    (offset.saturating_add(controller.page_size) < receipt.issues().len())
                        .then_some(Action::NextIssues),
                )),
        )
        .child(list)
}

fn status_name(status: Option<Result<(), ValidationRejection>>) -> &'static str {
    match status {
        None => "not run",
        Some(Ok(())) => "current, default policy passed",
        Some(Err(ValidationRejection::StaleCatalog)) => "stale catalog; rerun validation",
        Some(Err(ValidationRejection::StaleRules)) => "stale rules; rerun validation",
        Some(Err(ValidationRejection::WrongOperation)) => "wrong operation",
        Some(Err(ValidationRejection::Incomplete)) => "incomplete checks",
        Some(Err(ValidationRejection::Errors)) => "blocked by errors",
        Some(Err(ValidationRejection::Warnings)) => "blocked by warnings",
    }
}

fn button(
    view: &mut BoundContentDiagnostics,
    key: &str,
    text: &str,
    action: Option<Action>,
) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        view.bindings.insert(key.to_owned(), action);
    }
    Element::box_()
        .key_str(key)
        .class("prism-editor-content-diagnostics__button")
        .class(if enabled { "is-enabled" } else { "is-disabled" })
        .style(
            StyleProp::Opacity,
            StyleValue::number(if enabled { 1.0 } else { 0.45 }),
        )
        .style(StyleProp::PaddingLeft, StyleValue::px(4.0))
        .style(StyleProp::PaddingRight, StyleValue::px(4.0))
        .child(Element::text(text))
}

fn column() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(4.0))
}

fn row() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .style(StyleProp::Gap, StyleValue::px(8.0))
        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
}
