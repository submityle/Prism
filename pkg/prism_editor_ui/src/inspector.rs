//! Loom assembly for reflection-driven Inspector projections.

use prism_editor_reflect::{InspectorProjection, InspectorProjectionNode};
use prism_ui::Element;

use crate::action::{ActionBinding, ActionBindings, ActionTemplate, BoundElement, UiBuildError};
use crate::support::{empty_state, label, MAX_INSPECTOR_DEPTH, MAX_INSPECTOR_NODES};

/// Options controlling inspector projection presentation.
#[derive(Clone, Copy, Debug)]
pub struct InspectorViewProps<'a> {
    /// Path-bearing projection produced by `prism_editor_reflect`.
    pub projection: &'a InspectorProjection,
    /// Whether the root type heading is shown.
    pub show_root: bool,
}

/// Builds an inspector Element tree from a reflection projection.
#[must_use]
pub fn inspector_view(props: &InspectorViewProps<'_>) -> Element {
    match build_inspector(props, None) {
        Ok(bound) => bound.into_element(),
        Err(error) => inspector_limit_state(error),
    }
}

/// Builds an Inspector tree and binds editable leaves to field-edit actions.
pub fn inspector_view_bound(props: &InspectorViewProps<'_>) -> Result<BoundElement, UiBuildError> {
    build_inspector(props, Some(ActionBindings::new()))
}

fn build_inspector(
    props: &InspectorViewProps<'_>,
    bindings: Option<ActionBindings>,
) -> Result<BoundElement, UiBuildError> {
    let mut bindings = bindings;
    let root = &props.projection.root;
    let mut body = Element::box_().class("prism-editor-inspector__body");
    if root.children.is_empty() && root.node.value.is_none() {
        body = body.child(empty_state(
            "prism-editor-inspector__empty",
            "No inspectable fields",
        ));
    } else if props.show_root {
        body = body.child(render_node_with_actions(
            root,
            true,
            0,
            0,
            &mut bindings,
            &mut InspectorBudget::default(),
        )?);
    } else {
        let mut budget = InspectorBudget::default();
        for (index, child) in root.children.iter().enumerate() {
            body = body.child(render_node_with_actions(
                child,
                false,
                0,
                index,
                &mut bindings,
                &mut budget,
            )?);
        }
    }

    let element = Element::custom("prism.editor.inspector")
        .class("prism-editor-inspector")
        .child(
            Element::box_()
                .class("prism-editor-inspector__header")
                .child(label("prism-editor-inspector__title", "Inspector")),
        )
        .child(body);
    Ok(BoundElement::new(element, bindings.unwrap_or_default()))
}

fn inspector_limit_state(error: UiBuildError) -> Element {
    let message = match error {
        UiBuildError::InspectorDepthLimit { limit } => {
            format!("Inspector depth limit reached ({limit})")
        }
        UiBuildError::InspectorNodeLimit { limit } => {
            format!("Inspector node limit reached ({limit})")
        }
        UiBuildError::Action(error) => format!("Inspector action binding failed: {error}"),
    };
    Element::custom("prism.editor.inspector")
        .class("prism-editor-inspector")
        .child(
            Element::box_()
                .class("prism-editor-inspector__body")
                .child(empty_state("prism-editor-inspector__limit", message)),
        )
}

#[derive(Default)]
struct InspectorBudget {
    nodes: usize,
}

fn render_node_with_actions_inner(
    node: &InspectorProjectionNode,
    root: bool,
    depth: usize,
    sibling_index: usize,
    bindings: &mut Option<ActionBindings>,
    budget: &mut InspectorBudget,
) -> Result<Element, UiBuildError> {
    if depth > MAX_INSPECTOR_DEPTH {
        return Err(UiBuildError::InspectorDepthLimit {
            limit: MAX_INSPECTOR_DEPTH,
        });
    }
    budget.nodes = budget.nodes.saturating_add(1);
    if budget.nodes > MAX_INSPECTOR_NODES {
        return Err(UiBuildError::InspectorNodeLimit {
            limit: MAX_INSPECTOR_NODES,
        });
    }
    let key = node
        .path
        .as_ref()
        .map(|path| format!("field:{}", path.as_str()))
        .unwrap_or_else(|| format!("field:{}:{}", node.node.label, sibling_index));
    let mut row = Element::box_()
        .key_str(key)
        .class("prism-editor-inspector__node")
        .class(if root {
            "is-root"
        } else if node.children.is_empty() {
            "is-leaf"
        } else {
            "is-group"
        })
        .class(format!("depth-{depth}"));

    let heading = Element::box_()
        .class("prism-editor-inspector__heading")
        .child(label(
            "prism-editor-inspector__label",
            node.node.label.clone(),
        ))
        .child(label(
            "prism-editor-inspector__type",
            node.node.type_name.clone(),
        ));
    row = row.child(heading);

    if let Some(value) = &node.node.value {
        row = row.child(label("prism-editor-inspector__value", value.clone()));
    } else if let Some(variant) = &node.node.variant {
        row = row.child(label("prism-editor-inspector__variant", variant.clone()));
    }
    if node.node.hints.readonly {
        row = row.class("is-readonly");
        row = row.child(label("prism-editor-inspector__hint", "Read only"));
    }
    if let Some((minimum, maximum)) = node.node.hints.range {
        row = row.child(label(
            "prism-editor-inspector__range",
            format!("range {minimum}..={maximum}"),
        ));
    }
    if let Some(category) = &node.node.hints.category {
        row = row.child(label("prism-editor-inspector__category", category.clone()));
    }
    if let Some(docs) = &node.node.hints.docs {
        row = row.child(label("prism-editor-inspector__docs", docs.clone()));
    }

    if let Some(path) = node.path.as_ref()
        && node.node.value.is_some()
        && !node.node.hints.readonly
        && let Some(action_bindings) = bindings.as_mut()
    {
        action_bindings.bind(ActionBinding::commit(
            format!("field:{}", path.as_str()),
            ActionTemplate::edit_inspector_field(path.clone())?,
        )?)?;
    }

    if !node.children.is_empty() {
        let mut children = Element::box_().class("prism-editor-inspector__children");
        for (index, child) in node.children.iter().enumerate() {
            children = children.child(render_node_with_actions_inner(
                child,
                false,
                depth + 1,
                index,
                bindings,
                budget,
            )?);
        }
        row = row.child(children);
    }
    Ok(row)
}

fn render_node_with_actions(
    node: &InspectorProjectionNode,
    root: bool,
    depth: usize,
    sibling_index: usize,
    bindings: &mut Option<ActionBindings>,
    budget: &mut InspectorBudget,
) -> Result<Element, UiBuildError> {
    render_node_with_actions_inner(node, root, depth, sibling_index, bindings, budget)
}
