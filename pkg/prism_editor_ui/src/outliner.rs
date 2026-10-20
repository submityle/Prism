//! Loom assembly for the bounded editor Outliner.

use prism_editor_panels::{Outliner, OutlinerId};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};

use crate::support::{bounded_count, empty_state, label, normalized_filter};

/// Inputs for the Outliner Element tree.
#[derive(Clone, Copy, Debug)]
pub struct OutlinerViewProps<'a> {
    /// Source Outliner model.
    pub outliner: &'a Outliner,
    /// Text filter applied to the requested visible window.
    pub filter: &'a str,
    /// Start offset in the model's visible flattened order.
    pub start: usize,
    /// Number of rows requested before the hard safety cap.
    pub count: usize,
    /// Selected node identity, if any.
    pub selected: Option<OutlinerId>,
}

/// Builds a bounded Outliner panel with keyed rows.
#[must_use]
pub fn outliner_panel(props: &OutlinerViewProps<'_>) -> Element {
    let search = SearchField.render(
        &SearchFieldProps::new()
            .value(props.filter)
            .placeholder("Filter scene"),
    );
    let header = Element::box_()
        .class("prism-editor-outliner__header")
        .child(label("prism-editor-outliner__title", "Outliner"))
        .child(search.class("prism-editor-outliner__search"));

    let filter = normalized_filter(props.filter);
    let mut list = Element::box_().class("prism-editor-outliner__list");
    let mut emitted = 0usize;
    for id in props
        .outliner
        .visible_window(props.start, bounded_count(props.count))
    {
        let Some(node) = props.outliner.node(id) else {
            continue;
        };
        if !filter.is_empty() && !node.label().to_lowercase().contains(&filter) {
            continue;
        }
        emitted += 1;
        let mut row = Element::box_()
            .key_str(format!("outliner:{}", id.raw()))
            .class("prism-editor-outliner__row")
            .child(
                label(
                    "prism-editor-outliner__disclosure",
                    if node.children().is_empty() {
                        ""
                    } else if node.is_expanded() {
                        "v"
                    } else {
                        ">"
                    },
                )
                .key_str(format!("outliner-disclosure:{}", id.raw())),
            )
            .child(label("prism-editor-outliner__label", node.label()));
        if node.parent().is_some() {
            row = row.class("is-child");
        }
        if !node.is_expanded() {
            row = row.class("is-collapsed");
        }
        if props.selected == Some(id) {
            row = row.class("is-selected");
        }
        list = list.child(row);
    }
    if emitted == 0 {
        list = list.child(empty_state(
            "prism-editor-outliner__empty",
            if filter.is_empty() {
                "Scene has no visible nodes"
            } else {
                "No matching scene nodes"
            },
        ));
    }

    Element::custom("prism.editor.outliner")
        .class("prism-editor-outliner")
        .child(header)
        .child(list)
}
