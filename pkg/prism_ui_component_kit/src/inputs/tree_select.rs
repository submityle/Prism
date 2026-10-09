//! [`TreeSelect`] — a hierarchical dropdown of expandable, selectable nodes.
//!
//! A tree select renders a `pk-tree-select` surface whose dropdown holds a flat
//! run of `__node` rows, one per visible node. Each row is marked `--expanded`
//! and/or `--selected` from its node, and is indented by an inline
//! `MarginLeft` of `depth * 16` pixels — the one sanctioned data-driven inline
//! length. The dropdown is composed from the shared [`Popover`] overlay base.
//! All color comes from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::{StyleProp, StyleValue};

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// One node in a [`TreeSelect`] hierarchy.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TreeSelectNode {
    /// The node's visible label.
    pub label: String,
    /// The node's child nodes (shown only when `expanded`).
    pub children: Vec<TreeSelectNode>,
    /// Whether the node is marked selected.
    pub selected: bool,
    /// Whether the node is expanded to reveal its children.
    pub expanded: bool,
}

impl TreeSelectNode {
    /// Creates a leaf node with the given label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Replaces the node's children.
    #[must_use]
    pub fn children<I: IntoIterator<Item = TreeSelectNode>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Marks the node selected.
    #[must_use]
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Marks the node expanded.
    #[must_use]
    pub fn expanded(mut self, expanded: bool) -> Self {
        self.expanded = expanded;
        self
    }
}

/// Props for [`TreeSelect`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TreeSelectProps {
    /// The top-level nodes.
    pub roots: Vec<TreeSelectNode>,
    /// Whether the dropdown is open.
    pub open: bool,
}

impl TreeSelectProps {
    /// Creates empty, closed tree-select props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the root nodes.
    #[must_use]
    pub fn roots<I: IntoIterator<Item = TreeSelectNode>>(mut self, roots: I) -> Self {
        self.roots = roots.into_iter().collect();
        self
    }

    /// Sets whether the dropdown is open.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// Pixels of indentation applied per tree depth level.
const INDENT_PX: f32 = 16.0;

/// Appends `nodes` (and, when expanded, their descendants) as flat `__node`
/// rows into `out`, indenting each by `depth * INDENT_PX`.
fn push_rows(out: &mut Vec<Element>, nodes: &[TreeSelectNode], depth: usize) {
    for node in nodes {
        let mut row = Element::box_()
            .class("pk-tree-select__node")
            .style(StyleProp::MarginLeft, StyleValue::px(depth as f32 * INDENT_PX))
            .child(Element::text(node.label.clone()).class("pk-tree-select__label"));
        if node.expanded {
            row = row.class("pk-tree-select__node--expanded");
        }
        if node.selected {
            row = row.class("pk-tree-select__node--selected");
        }
        out.push(row);
        if node.expanded {
            push_rows(out, &node.children, depth + 1);
        }
    }
}

/// The tree-select control. Zero-sized; config lives in [`TreeSelectProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeSelect;

impl TreeSelect {
    /// The accessibility role a tree select exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for TreeSelect {
    type Props = TreeSelectProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut rows: Vec<Element> = Vec::new();
        push_rows(&mut rows, &props.roots, 0);

        let popover = Popover.render(&PopoverProps::new().open(props.open).children(rows));
        Element::box_().class("pk-tree-select").child(popover)
    }
}

/// Registers the `pk-tree-select` class family: base surface, node rows, their
/// expanded/selected markers and the node label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-tree-select")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    sheet.insert(
        Class::new("pk-tree-select__node")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
    sheet.insert(
        Class::new("pk-tree-select__node--expanded")
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );
    sheet.insert(
        Class::new("pk-tree-select__node--selected")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    sheet.insert(
        Class::new("pk-tree-select__label").with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_of(root: &Element) -> Vec<Element> {
        let popover = root.child_elements().last().expect("popover");
        popover
            .child_elements()
            .iter()
            .flat_map(Element::child_elements)
            .filter(|c| c.class_names().iter().any(|n| n == "pk-tree-select__node"))
            .cloned()
            .collect()
    }

    fn margin_left(node: &Element) -> Option<f32> {
        node.inline_pairs().iter().find_map(|(p, v)| match (p, v) {
            (StyleProp::MarginLeft, StyleValue::Length(l)) => l.resolve(0.0),
            _ => None,
        })
    }

    #[test]
    fn collapsed_node_hides_children() {
        let tree = TreeSelectNode::new("root").children([TreeSelectNode::new("child")]);
        let el = TreeSelect.render(&TreeSelectProps::new().roots([tree]).open(true));
        assert_eq!(rows_of(&el).len(), 1);
    }

    #[test]
    fn expanded_node_reveals_children_with_indent() {
        let tree = TreeSelectNode::new("root")
            .expanded(true)
            .children([TreeSelectNode::new("child")]);
        let el = TreeSelect.render(&TreeSelectProps::new().roots([tree]).open(true));
        let rows = rows_of(&el);
        assert_eq!(rows.len(), 2);
        assert_eq!(margin_left(&rows[0]), Some(0.0));
        assert_eq!(margin_left(&rows[1]), Some(16.0));
        assert!(rows[0].class_names().iter().any(|c| c == "pk-tree-select__node--expanded"));
    }

    #[test]
    fn selected_node_gets_marker() {
        let tree = TreeSelectNode::new("root").selected(true);
        let el = TreeSelect.render(&TreeSelectProps::new().roots([tree]).open(true));
        assert!(rows_of(&el)[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-tree-select__node--selected"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(TreeSelect::role(), Role::Textbox);
    }
}
