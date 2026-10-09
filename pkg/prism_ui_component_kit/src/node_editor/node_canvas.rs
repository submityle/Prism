//! [`NodeCanvas`] — the surface that hosts nodes and the edges between them.
//!
//! The canvas renders a `pk-node-canvas` container whose children are, in
//! paint order, every [`EdgeView`](super::edge::EdgeView) followed by every
//! [`Node`](super::node::Node). Edges render first so nodes sit visually on
//! top. Because the style layer has no `position`, children are offset with the
//! inline margins their own controls emit; the canvas only supplies the framed
//! backdrop.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

use super::edge::{EdgeView, EdgeViewProps};
use super::node::{Node, NodeProps};

/// Props for [`NodeCanvas`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NodeCanvasProps {
    /// The nodes to render, each positioned by its own `pos`.
    pub nodes: Vec<NodeProps>,
    /// The edges to render beneath the nodes.
    pub edges: Vec<EdgeViewProps>,
}

impl NodeCanvasProps {
    /// Creates empty canvas props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node.
    #[must_use]
    pub fn node(mut self, node: NodeProps) -> Self {
        self.nodes.push(node);
        self
    }

    /// Appends an edge.
    #[must_use]
    pub fn edge(mut self, edge: EdgeViewProps) -> Self {
        self.edges.push(edge);
        self
    }

    /// Replaces the nodes.
    #[must_use]
    pub fn nodes<I: IntoIterator<Item = NodeProps>>(mut self, nodes: I) -> Self {
        self.nodes = nodes.into_iter().collect();
        self
    }

    /// Replaces the edges.
    #[must_use]
    pub fn edges<I: IntoIterator<Item = EdgeViewProps>>(mut self, edges: I) -> Self {
        self.edges = edges.into_iter().collect();
        self
    }
}

/// The canvas control. Zero-sized; configuration lives in [`NodeCanvasProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeCanvas;

impl NodeCanvas {
    /// The canvas groups the nodes and edges it hosts.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for NodeCanvas {
    type Props = NodeCanvasProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-node-canvas");
        // Edges first (underlay), then nodes on top.
        for edge in &props.edges {
            el = el.child(EdgeView.render(edge));
        }
        for node in &props.nodes {
            el = el.child(Node.render(node));
        }
        el
    }
}

/// Registers the `pk-node-canvas` class: a framed backdrop panel for the graph.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-node-canvas")
            .with(StyleProp::Display, kw(Keyword::Block))
            .with(StyleProp::MinHeight, StyleValue::px(320.0))
            .with_padding_x(tok("space.lg"))
            .with_padding_y(tok("space.lg"))
            .with(StyleProp::BackgroundColor, tok("color.background.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_canvas_has_container_only() {
        let el = NodeCanvas.render(&NodeCanvasProps::new());
        assert_eq!(el.class_names(), ["pk-node-canvas"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn edges_render_before_nodes() {
        let el = NodeCanvas.render(
            &NodeCanvasProps::new()
                .node(NodeProps::new("A"))
                .edge(EdgeViewProps::new((0.0, 0.0), (10.0, 10.0))),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-node-edge"]);
        assert!(kids[1].class_names().iter().any(|c| c == "pk-node"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(NodeCanvas::role(), Role::Group);
    }
}
