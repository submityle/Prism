//! [`Port`] — a single input/output connection point on a node.
//!
//! A port renders a small marker dot plus a label. It attaches only kit class
//! names: the `pk-node-port` block, an `--input` or `--output` modifier, and a
//! `--connected` modifier when the port currently participates in an edge. All
//! color and spacing resolve from theme tokens via [`crate::preset`], so a
//! light/dark flip needs no change here.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Which side of a node a [`Port`] lives on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PortKind {
    /// An input (sink) port — data flows in.
    #[default]
    Input,
    /// An output (source) port — data flows out.
    Output,
}

impl PortKind {
    /// The modifier suffix used in class names (e.g. `pk-node-port--input`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            PortKind::Input => "input",
            PortKind::Output => "output",
        }
    }
}

/// Props for [`Port`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PortProps {
    /// The port's visible label.
    pub label: String,
    /// Whether this is an input or output port.
    pub kind: PortKind,
    /// Whether the port is currently connected by an edge.
    pub connected: bool,
}

impl PortProps {
    /// Creates props for a labelled input port.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Sets the port kind (input/output).
    #[must_use]
    pub fn kind(mut self, kind: PortKind) -> Self {
        self.kind = kind;
        self
    }

    /// Marks the port connected.
    #[must_use]
    pub fn connected(mut self, connected: bool) -> Self {
        self.connected = connected;
        self
    }
}

/// The port control. Zero-sized; all configuration lives in [`PortProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Port;

impl Port {
    /// Ports are decorative markers; they expose no interactive role.
    #[must_use]
    pub const fn role() -> Role {
        Role::Presentation
    }
}

impl Component for Port {
    type Props = PortProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-node-port", &[props.kind.suffix()]) {
            el = el.class(name);
        }

        let mut dot = Element::box_().class("pk-node-port__dot");
        if props.connected {
            el = el.class("pk-node-port--connected");
            dot = dot.class("pk-node-port__dot--connected");
        }

        let label = Element::text(props.label.clone()).class("pk-node-port__label");
        el.child(dot).child(label)
    }
}

/// Registers the `pk-node-port` class family: block, dot, label, the two side
/// modifiers and the connected state.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Block: a compact row pairing the marker dot with its label.
    sheet.insert(
        Class::new("pk-node-port")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FontSize, tok("font.size.caption1")),
    );

    // Inputs read left-to-right; outputs mirror so the dot sits on the outer
    // edge of the node.
    sheet.insert(
        Class::new("pk-node-port--input")
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::Start)),
    );
    sheet.insert(
        Class::new("pk-node-port--output")
            .with(StyleProp::FlexDirection, kw(Keyword::RowReverse))
            .with(StyleProp::JustifyContent, kw(Keyword::Start)),
    );

    // Connected ports emphasize their label.
    sheet.insert(
        Class::new("pk-node-port--connected")
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );

    // Dot: a small fixed circle that never shrinks; neutral until connected.
    sheet.insert(
        Class::new("pk-node-port__dot")
            .with(StyleProp::Width, StyleValue::px(8.0))
            .with(StyleProp::Height, StyleValue::px(8.0))
            .with(StyleProp::MinWidth, StyleValue::px(8.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
    sheet.insert(
        Class::new("pk-node-port__dot--connected")
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    // Label: secondary text so the title stays dominant.
    sheet.insert(
        Class::new("pk-node-port__label")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.caption1")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: PortProps) -> Element {
        Port.render(&props)
    }

    #[test]
    fn input_port_attaches_block_and_side_classes() {
        let el = render(PortProps::new("value"));
        assert_eq!(el.class_names(), ["pk-node-port", "pk-node-port--input"]);
    }

    #[test]
    fn output_port_uses_output_modifier() {
        let el = render(PortProps::new("result").kind(PortKind::Output));
        assert_eq!(el.class_names(), ["pk-node-port", "pk-node-port--output"]);
    }

    #[test]
    fn connected_marks_block_and_dot() {
        let el = render(PortProps::new("v").connected(true));
        assert!(el
            .class_names()
            .iter()
            .any(|c| c == "pk-node-port--connected"));
        let dot = &el.child_elements()[0];
        assert!(dot
            .class_names()
            .iter()
            .any(|c| c == "pk-node-port__dot--connected"));
    }

    #[test]
    fn renders_dot_then_label() {
        let el = render(PortProps::new("alpha"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-node-port__dot"]);
        assert_eq!(kids[1].kind(), &ElementKind::Text);
        assert_eq!(kids[1].text_content(), Some("alpha"));
    }

    #[test]
    fn role_is_presentation() {
        assert_eq!(Port::role(), Role::Presentation);
    }
}
