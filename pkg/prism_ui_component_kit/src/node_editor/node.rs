//! [`Node`] — a single box in the node editor, with a header and two port
//! columns.
//!
//! A node renders the `pk-node` block: a `pk-node__header` title bar above a
//! `pk-node__body` that holds an input column and an output column of
//! [`Port`](super::port::Port)s. Selection is expressed with the block-level
//! `pk-node--selected` modifier rather than a shared `is-*` state class.
//!
//! The style layer exposes no `position`/`transform` property, so canvas
//! placement is approximated with inline left/top margins in logical pixels.
//! True free positioning is the render layer's job; this control only records
//! the intended offset.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

use super::port::{Port, PortKind, PortProps};

/// Props for [`Node`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NodeProps {
    /// The node's title, shown in the header.
    pub title: String,
    /// Canvas position as `(x, y)` in logical pixels, approximated with margins.
    pub pos: (f32, f32),
    /// Input port labels, rendered top-to-bottom in the left column.
    pub inputs: Vec<String>,
    /// Output port labels, rendered top-to-bottom in the right column.
    pub outputs: Vec<String>,
    /// Whether the node is currently selected.
    pub selected: bool,
}

impl NodeProps {
    /// Creates props for a titled node at the origin with no ports.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Sets the canvas position.
    #[must_use]
    pub fn pos(mut self, x: f32, y: f32) -> Self {
        self.pos = (x, y);
        self
    }

    /// Appends an input port label.
    #[must_use]
    pub fn input(mut self, label: impl Into<String>) -> Self {
        self.inputs.push(label.into());
        self
    }

    /// Appends an output port label.
    #[must_use]
    pub fn output(mut self, label: impl Into<String>) -> Self {
        self.outputs.push(label.into());
        self
    }

    /// Replaces the input port labels.
    #[must_use]
    pub fn inputs<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the output port labels.
    #[must_use]
    pub fn outputs<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outputs = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Marks the node selected.
    #[must_use]
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

/// The node control. Zero-sized; all configuration lives in [`NodeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Node;

impl Node {
    /// A node groups related ports and content.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Node {
    type Props = NodeProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{Length, StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-node");
        if props.selected {
            el = el.class("pk-node--selected");
        }

        // Approximate canvas placement with inline left/top margins.
        el = el
            .style(StyleProp::MarginLeft, StyleValue::Length(Length::Px(props.pos.0)))
            .style(StyleProp::MarginTop, StyleValue::Length(Length::Px(props.pos.1)));

        let header = Element::text(props.title.clone()).class("pk-node__header");

        let mut inputs = Element::box_()
            .class("pk-node__ports")
            .class("pk-node__ports--inputs");
        for label in &props.inputs {
            inputs = inputs.child(Port.render(&PortProps::new(label.clone()).kind(PortKind::Input)));
        }

        let mut outputs = Element::box_()
            .class("pk-node__ports")
            .class("pk-node__ports--outputs");
        for label in &props.outputs {
            outputs =
                outputs.child(Port.render(&PortProps::new(label.clone()).kind(PortKind::Output)));
        }

        let body = Element::box_()
            .class("pk-node__body")
            .child(inputs)
            .child(outputs);

        el.child(header).child(body)
    }
}

/// Registers the `pk-node` class family: block, selected modifier, header,
/// body and the two port columns.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Block: a surface card with a soft shadow and a subtle hairline border.
    sheet.insert(
        Class::new("pk-node")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::MinWidth, StyleValue::px(160.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BackgroundColor, tok("color.surface"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_shadow(0.0, 6.0, 18.0, tok("glass.shadow")),
    );

    // Selected: lift the border to the accent tint.
    sheet.insert(
        Class::new("pk-node--selected")
            .with(StyleProp::BorderColor, tok("color.tint"))
            .with_shadow(0.0, 8.0, 24.0, tok("color.tint")),
    );

    // Header: the title bar, separated from the body by a hairline.
    sheet.insert(
        Class::new("pk-node__header")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::PaddingBottom, tok("space.xs"))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Body: input column on the left, output column on the right.
    sheet.insert(
        Class::new("pk-node__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.md")),
    );

    // Port columns: vertical stacks; outputs align to the right edge.
    sheet.insert(
        Class::new("pk-node__ports")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0)),
    );
    sheet.insert(
        Class::new("pk-node__ports--inputs").with(StyleProp::AlignItems, kw(Keyword::Start)),
    );
    sheet.insert(
        Class::new("pk-node__ports--outputs").with(StyleProp::AlignItems, kw(Keyword::End)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: NodeProps) -> Element {
        Node.render(&props)
    }

    #[test]
    fn plain_node_has_block_class_only() {
        let el = render(NodeProps::new("Add"));
        assert_eq!(el.class_names(), ["pk-node"]);
    }

    #[test]
    fn selected_adds_block_modifier() {
        let el = render(NodeProps::new("Add").selected(true));
        assert_eq!(el.class_names(), ["pk-node", "pk-node--selected"]);
    }

    #[test]
    fn header_then_body_structure() {
        let el = render(
            NodeProps::new("Add")
                .input("a")
                .input("b")
                .output("sum"),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].kind(), &ElementKind::Text);
        assert_eq!(kids[0].class_names(), ["pk-node__header"]);
        assert_eq!(kids[0].text_content(), Some("Add"));

        let body = &kids[1];
        assert_eq!(body.class_names(), ["pk-node__body"]);
        let cols = body.child_elements();
        assert_eq!(cols.len(), 2);
        assert_eq!(
            cols[0].class_names(),
            ["pk-node__ports", "pk-node__ports--inputs"]
        );
        assert_eq!(cols[0].child_elements().len(), 2);
        assert_eq!(
            cols[1].class_names(),
            ["pk-node__ports", "pk-node__ports--outputs"]
        );
        assert_eq!(cols[1].child_elements().len(), 1);
    }

    #[test]
    fn position_becomes_inline_margins() {
        let el = render(NodeProps::new("N").pos(12.0, 34.0));
        let has = |prop, px| {
            el.inline_pairs()
                .iter()
                .any(|(p, v)| *p == prop && *v == StyleValue::Length(Length::Px(px)))
        };
        assert!(has(StyleProp::MarginLeft, 12.0));
        assert!(has(StyleProp::MarginTop, 34.0));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Node::role(), Role::Group);
    }
}
