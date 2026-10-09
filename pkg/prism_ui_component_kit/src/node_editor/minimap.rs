//! [`Minimap`] — a scaled-down overview of node positions.
//!
//! The minimap renders a small `pk-node-minimap` panel containing one
//! `pk-node-minimap__dot` per node. Each dot is placed with inline left/top
//! margins equal to the node's canvas position scaled by [`SCALE`], since the
//! style layer offers no `transform`. It is a coarse locator, not a faithful
//! thumbnail: only relative node placement is conveyed.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The factor by which canvas pixels are shrunk into minimap pixels.
pub const SCALE: f32 = 0.1;

/// Props for [`Minimap`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MinimapProps {
    /// Node canvas positions as `(x, y)` in logical pixels.
    pub nodes: Vec<(f32, f32)>,
}

impl MinimapProps {
    /// Creates empty minimap props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node position.
    #[must_use]
    pub fn node(mut self, pos: (f32, f32)) -> Self {
        self.nodes.push(pos);
        self
    }

    /// Replaces the node positions.
    #[must_use]
    pub fn nodes<I: IntoIterator<Item = (f32, f32)>>(mut self, nodes: I) -> Self {
        self.nodes = nodes.into_iter().collect();
        self
    }
}

/// The minimap control. Zero-sized; configuration lives in [`MinimapProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Minimap;

impl Minimap {
    /// The minimap is a decorative overview with no interactive role.
    #[must_use]
    pub const fn role() -> Role {
        Role::Presentation
    }
}

impl Component for Minimap {
    type Props = MinimapProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{Length, StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-node-minimap");
        for (x, y) in &props.nodes {
            let dot = Element::box_()
                .class("pk-node-minimap__dot")
                .style(StyleProp::MarginLeft, StyleValue::Length(Length::Px(x * SCALE)))
                .style(StyleProp::MarginTop, StyleValue::Length(Length::Px(y * SCALE)));
            el = el.child(dot);
        }
        el
    }
}

/// Registers the `pk-node-minimap` class family: the framed overview panel and
/// the node locator dots inside it.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-node-minimap")
            .with(StyleProp::Display, kw(Keyword::Block))
            .with(StyleProp::Width, StyleValue::px(160.0))
            .with(StyleProp::Height, StyleValue::px(120.0))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    sheet.insert(
        Class::new("pk-node-minimap__dot")
            .with(StyleProp::Width, StyleValue::px(6.0))
            .with(StyleProp::Height, StyleValue::px(6.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    #[test]
    fn empty_minimap_has_container_only() {
        let el = Minimap.render(&MinimapProps::new());
        assert_eq!(el.class_names(), ["pk-node-minimap"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn each_node_becomes_a_scaled_dot() {
        let el = Minimap.render(&MinimapProps::new().node((100.0, 200.0)).node((0.0, 0.0)));
        let dots = el.child_elements();
        assert_eq!(dots.len(), 2);
        assert_eq!(dots[0].class_names(), ["pk-node-minimap__dot"]);
        let left = dots[0]
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::MarginLeft)
            .map(|(_, v)| v.clone());
        assert_eq!(left, Some(StyleValue::Length(Length::Px(10.0))));
    }

    #[test]
    fn role_is_presentation() {
        assert_eq!(Minimap::role(), Role::Presentation);
    }
}
