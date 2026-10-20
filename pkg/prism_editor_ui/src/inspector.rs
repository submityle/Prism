//! Loom assembly for reflection-driven Inspector projections.

use prism_editor_reflect::{InspectorProjection, InspectorProjectionNode};
use prism_ui::Element;

use crate::support::{empty_state, label};

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
    let root = &props.projection.root;
    let mut body = Element::box_().class("prism-editor-inspector__body");
    if root.children.is_empty() && root.node.value.is_none() {
        body = body.child(empty_state(
            "prism-editor-inspector__empty",
            "No inspectable fields",
        ));
    } else if props.show_root {
        body = body.child(render_node(root, true, 0));
    } else {
        for (index, child) in root.children.iter().enumerate() {
            body = body.child(render_node_with_index(child, false, 0, index));
        }
    }

    Element::custom("prism.editor.inspector")
        .class("prism-editor-inspector")
        .child(
            Element::box_()
                .class("prism-editor-inspector__header")
                .child(label("prism-editor-inspector__title", "Inspector")),
        )
        .child(body)
}

fn render_node(node: &InspectorProjectionNode, root: bool, sibling_index: usize) -> Element {
    render_node_with_index(node, root, 0, sibling_index)
}

fn render_node_with_index(
    node: &InspectorProjectionNode,
    root: bool,
    depth: usize,
    sibling_index: usize,
) -> Element {
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

    if !node.children.is_empty() {
        let mut children = Element::box_().class("prism-editor-inspector__children");
        for (index, child) in node.children.iter().enumerate() {
            children = children.child(render_node_with_index(child, false, depth + 1, index));
        }
        row = row.child(children);
    }
    row
}
