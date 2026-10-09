//! [`TreeView`] — a recursive, collapsible hierarchy.
//!
//! A tree renders a `pk-tree` container whose entries are `pk-tree__node`
//! boxes. Each node shows a `pk-tree__label` indented by its depth (an inline
//! `margin-left` of `depth * INDENT_PX`, the sanctioned px literal) and, when
//! `expanded` and non-empty, carries `pk-tree__node--expanded` and renders its
//! children recursively beneath it. Collapsed nodes omit their subtree. The
//! control attaches only kit class names; color and spacing resolve from theme
//! tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The per-level indentation step, in logical pixels.
const INDENT_PX: f32 = 16.0;

/// A single node in a [`TreeView`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TreeNode {
    /// The node's visible label.
    pub label: String,
    /// The node's children, rendered when `expanded`.
    pub children: Vec<TreeNode>,
    /// Whether this node is expanded (its children are shown).
    pub expanded: bool,
}

impl TreeNode {
    /// Creates a collapsed leaf node with the given label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            children: Vec::new(),
            expanded: false,
        }
    }

    /// Appends a child node.
    #[must_use]
    pub fn child(mut self, node: TreeNode) -> Self {
        self.children.push(node);
        self
    }

    /// Replaces the children.
    #[must_use]
    pub fn children<I: IntoIterator<Item = TreeNode>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Sets the expanded state.
    #[must_use]
    pub fn expanded(mut self, expanded: bool) -> Self {
        self.expanded = expanded;
        self
    }
}

/// Props for [`TreeView`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TreeViewProps {
    /// The top-level nodes.
    pub roots: Vec<TreeNode>,
}

impl TreeViewProps {
    /// Creates empty tree props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a root node.
    #[must_use]
    pub fn root(mut self, node: TreeNode) -> Self {
        self.roots.push(node);
        self
    }

    /// Replaces the root nodes.
    #[must_use]
    pub fn roots<I: IntoIterator<Item = TreeNode>>(mut self, roots: I) -> Self {
        self.roots = roots.into_iter().collect();
        self
    }
}

/// The tree-view control. Zero-sized; config lives in [`TreeViewProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeView;

impl TreeView {
    /// The accessibility role a tree approximates.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }
}

/// Recursively renders a node at `depth`, emitting its subtree when expanded.
fn render_node(node: &TreeNode, depth: u32) -> Element {
    use prism_ui_style::{StyleProp, StyleValue};

    let has_children = !node.children.is_empty();
    let expanded = node.expanded && has_children;

    let mut el = Element::box_().class("pk-tree__node");
    if expanded {
        el = el.class("pk-tree__node--expanded");
    }

    // Depth indentation is a geometric px literal, not a theme value.
    let indent = StyleValue::px(depth as f32 * INDENT_PX);
    let label = Element::text(node.label.clone())
        .class("pk-tree__label")
        .style(StyleProp::MarginLeft, indent);
    el = el.child(label);

    if expanded {
        for child in &node.children {
            el = el.child(render_node(child, depth + 1));
        }
    }
    el
}

impl Component for TreeView {
    type Props = TreeViewProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-tree");
        for root in &props.roots {
            el = el.child(render_node(root, 0));
        }
        el
    }
}

/// Registers the `pk-tree` class family: container, node, expanded modifier
/// and label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Container: a tight vertical stack of nodes.
    sheet.insert(
        Class::new("pk-tree")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Node: a vertical stack so children sit beneath their parent label.
    sheet.insert(
        Class::new("pk-tree__node")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Expanded node: a hair more breathing room around the open subtree.
    sheet.insert(
        Class::new("pk-tree__node--expanded")
            .with(StyleProp::RowGap, tok("space.xs")),
    );

    // Label: a padded, selectable row that highlights on hover.
    sheet.insert(
        Class::new("pk-tree__label")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: TreeViewProps) -> Element {
        TreeView.render(&props)
    }

    fn label_of(node: &Element) -> &Element {
        &node.child_elements()[0]
    }

    #[test]
    fn empty_tree_has_no_nodes() {
        let el = render(TreeViewProps::new());
        assert_eq!(el.class_names(), ["pk-tree"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn collapsed_node_hides_children() {
        let el = render(TreeViewProps::new().root(
            TreeNode::new("root").child(TreeNode::new("child")),
        ));
        let node = &el.child_elements()[0];
        assert_eq!(node.class_names(), ["pk-tree__node"]);
        // Only the label is present; the child subtree is omitted.
        assert_eq!(node.child_elements().len(), 1);
        assert_eq!(label_of(node).text_content(), Some("root"));
    }

    #[test]
    fn expanded_node_renders_children() {
        let el = render(TreeViewProps::new().root(
            TreeNode::new("root")
                .child(TreeNode::new("a"))
                .child(TreeNode::new("b"))
                .expanded(true),
        ));
        let node = &el.child_elements()[0];
        assert!(node.class_names().iter().any(|n| n == "pk-tree__node--expanded"));
        // Label plus two child nodes.
        assert_eq!(node.child_elements().len(), 3);
        assert_eq!(label_of(&node.child_elements()[1]).text_content(), Some("a"));
    }

    #[test]
    fn expanded_leaf_is_not_marked_expanded() {
        let el = render(TreeViewProps::new().root(TreeNode::new("leaf").expanded(true)));
        let node = &el.child_elements()[0];
        assert_eq!(node.class_names(), ["pk-tree__node"]);
    }

    #[test]
    fn depth_increases_label_indent() {
        let el = render(TreeViewProps::new().root(
            TreeNode::new("root")
                .child(TreeNode::new("child"))
                .expanded(true),
        ));
        let root = &el.child_elements()[0];
        let root_indent = label_of(root).inline_pairs();
        assert_eq!(root_indent[0], (StyleProp::MarginLeft, StyleValue::Length(Length::Px(0.0))));

        let child = &root.child_elements()[1];
        let child_indent = label_of(child).inline_pairs();
        assert_eq!(child_indent[0], (StyleProp::MarginLeft, StyleValue::Length(Length::Px(INDENT_PX))));
    }

    #[test]
    fn role_is_list() {
        assert_eq!(TreeView::role(), Role::List);
    }
}
