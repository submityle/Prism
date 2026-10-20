//! Shared element-building helpers.

use prism_ui::Element;

/// Maximum number of dynamic rows materialized by one panel render.
pub const MAX_VISIBLE_ROWS: usize = 512;
/// Maximum number of action bindings materialized by one bound view.
pub const MAX_ACTION_BINDINGS: usize = 8192;
/// Maximum nesting depth rendered by the Inspector.
pub const MAX_INSPECTOR_DEPTH: usize = 64;
/// Maximum number of Inspector nodes rendered by one view.
pub const MAX_INSPECTOR_NODES: usize = 4096;

/// Builds a semantic text node with a stable class.
pub(crate) fn label(class: &str, text: impl Into<String>) -> Element {
    Element::text(text.into()).class(class)
}

/// Builds an explicit empty-state node.
pub(crate) fn empty_state(class: &str, message: impl Into<String>) -> Element {
    Element::box_()
        .class(class)
        .child(label("prism-editor__empty-message", message))
}

/// Caps a requested window without changing its start offset.
pub(crate) const fn bounded_count(count: usize) -> usize {
    if count > MAX_VISIBLE_ROWS {
        MAX_VISIBLE_ROWS
    } else {
        count
    }
}

/// Converts a text filter to one reusable lowercase allocation.
pub(crate) fn normalized_filter(filter: &str) -> String {
    filter.trim().to_lowercase()
}
