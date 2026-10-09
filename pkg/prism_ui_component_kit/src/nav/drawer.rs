//! [`Drawer`] — an off-canvas side panel.
//!
//! A drawer renders a `pk-drawer` surface anchored to a [`DrawerSide`] and
//! wrapping arbitrary `children`. When `open` it gains the shared `is-open`
//! state class (otherwise it is collapsed/hidden); the side selects a
//! `pk-drawer--left` / `pk-drawer--right` modifier. Only kit class names are
//! attached; surface, spacing and reveal resolve from theme tokens.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// The edge a [`Drawer`] is anchored to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DrawerSide {
    /// Anchored to the leading (left) edge.
    #[default]
    Left,
    /// Anchored to the trailing (right) edge.
    Right,
}

impl DrawerSide {
    /// The modifier suffix used in class names (e.g. `pk-drawer--left`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            DrawerSide::Left => "left",
            DrawerSide::Right => "right",
        }
    }
}

/// Props for [`Drawer`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DrawerProps {
    /// Whether the drawer is revealed.
    pub open: bool,
    /// The edge the drawer is anchored to.
    pub side: DrawerSide,
    /// Content wrapped by the drawer panel.
    pub children: Vec<Element>,
}

impl DrawerProps {
    /// Creates closed, left-anchored props with no children.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the open/closed state.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Sets the anchoring side.
    #[must_use]
    pub fn side(mut self, side: DrawerSide) -> Self {
        self.side = side;
        self
    }

    /// Appends a single child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The drawer control. Zero-sized; config lives in [`DrawerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Drawer;

impl Component for Drawer {
    type Props = DrawerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-drawer", &[props.side.suffix()]) {
            el = el.class(name);
        }
        if props.open {
            el = el.class("is-open");
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-drawer` class family: the hidden-by-default panel, the
/// left/right anchors and the shared `is-open` reveal.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Panel: a vertical glass surface, hidden (zero opacity) until opened.
    sheet.insert(
        Class::new("pk-drawer")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Width, StyleValue::px(320.0))
            .with(StyleProp::MaxWidth, StyleValue::percent(80.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.md"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 0.0, 24.0, tok("glass.shadow"))
            .with(StyleProp::Opacity, StyleValue::number(0.0)),
    );

    // Left anchor: pushed to the leading edge.
    sheet.insert(
        Class::new("pk-drawer--left").with(StyleProp::MarginRight, StyleValue::auto()),
    );

    // Right anchor: pushed to the trailing edge.
    sheet.insert(
        Class::new("pk-drawer--right").with(StyleProp::MarginLeft, StyleValue::auto()),
    );

    // Shared open state: reveal the panel.
    sheet.insert(
        Class::new("is-open")
            .with(StyleProp::Height, StyleValue::auto())
            .with(StyleProp::Opacity, StyleValue::number(1.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DrawerProps) -> Element {
        Drawer.render(&props)
    }

    #[test]
    fn closed_left_drawer_has_base_and_side_only() {
        let el = render(DrawerProps::new());
        assert_eq!(el.class_names(), ["pk-drawer", "pk-drawer--left"]);
    }

    #[test]
    fn open_adds_is_open() {
        let el = render(DrawerProps::new().open(true));
        assert_eq!(el.class_names(), ["pk-drawer", "pk-drawer--left", "is-open"]);
    }

    #[test]
    fn right_side_selects_right_modifier() {
        let el = render(DrawerProps::new().side(DrawerSide::Right));
        assert_eq!(el.class_names(), ["pk-drawer", "pk-drawer--right"]);
    }

    #[test]
    fn wraps_children() {
        let el = render(
            DrawerProps::new()
                .open(true)
                .child(Element::text("row"))
                .child(Element::text("row2")),
        );
        assert_eq!(el.child_elements().len(), 2);
    }
}
