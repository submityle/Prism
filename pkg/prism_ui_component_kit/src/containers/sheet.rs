//! [`Sheet`] — an edge-anchored surface that slides in from one side.
//!
//! A sheet renders a `pk-sheet` surface carrying a side modifier
//! (`--bottom`/`--top`/`--left`/`--right`) and the `is-open` marker when
//! visible. It is a surface + content container; overlay positioning, the
//! scrim and the slide transition are backend concerns keyed off these
//! classes. The control attaches only kit class names and exposes the
//! [`Role::Dialog`](prism_ui_a11y::Role::Dialog) role.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The edge a [`Sheet`] anchors to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SheetSide {
    /// Anchored to the bottom edge (the default).
    #[default]
    Bottom,
    /// Anchored to the right edge.
    Right,
    /// Anchored to the left edge.
    Left,
    /// Anchored to the top edge.
    Top,
}

impl SheetSide {
    /// The modifier class for this side.
    #[must_use]
    const fn class(self) -> &'static str {
        match self {
            SheetSide::Bottom => "pk-sheet--bottom",
            SheetSide::Right => "pk-sheet--right",
            SheetSide::Left => "pk-sheet--left",
            SheetSide::Top => "pk-sheet--top",
        }
    }
}

/// Props for [`Sheet`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SheetProps {
    /// Whether the sheet is open (visible).
    pub open: bool,
    /// The sheet's content.
    pub children: Vec<Element>,
    /// The edge the sheet anchors to.
    pub side: SheetSide,
}

impl SheetProps {
    /// Creates empty sheet props (closed, bottom-anchored).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the open state.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Appends a child.
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

    /// Sets the anchoring edge.
    #[must_use]
    pub fn side(mut self, side: SheetSide) -> Self {
        self.side = side;
        self
    }
}

/// The sheet control. Zero-sized; all configuration lives in [`SheetProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Sheet;

impl Sheet {
    /// The accessibility role a sheet exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Dialog
    }
}

impl Component for Sheet {
    type Props = SheetProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-sheet").class(props.side.class());
        if props.open {
            el = el.class("is-open");
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-sheet` class family: surface plus the four side anchors.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Surface: a frosted-glass panel that stacks its content vertically.
    sheet.insert(
        Class::new("pk-sheet")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.md"))
            .with_padding_x(tok("space.lg"))
            .with_padding_y(tok("space.lg"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, -4.0, 24.0, tok("glass.shadow")),
    );

    // Bottom/top sheets span the width and cap their height.
    sheet.insert(
        Class::new("pk-sheet--bottom")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::MaxHeight, StyleValue::percent(90.0))
            .with(StyleProp::BorderRadius, tok("radius.xl")),
    );
    sheet.insert(
        Class::new("pk-sheet--top")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::MaxHeight, StyleValue::percent(90.0))
            .with(StyleProp::BorderRadius, tok("radius.xl")),
    );

    // Left/right sheets span the height and cap their width.
    sheet.insert(
        Class::new("pk-sheet--left")
            .with(StyleProp::Height, StyleValue::percent(100.0))
            .with(StyleProp::MaxWidth, StyleValue::percent(90.0))
            .with(StyleProp::BorderRadius, tok("radius.xl")),
    );
    sheet.insert(
        Class::new("pk-sheet--right")
            .with(StyleProp::Height, StyleValue::percent(100.0))
            .with(StyleProp::MaxWidth, StyleValue::percent(90.0))
            .with(StyleProp::BorderRadius, tok("radius.xl")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SheetProps) -> Element {
        Sheet.render(&props)
    }

    #[test]
    fn closed_bottom_sheet_by_default() {
        let el = render(SheetProps::new());
        assert_eq!(el.class_names(), ["pk-sheet", "pk-sheet--bottom"]);
    }

    #[test]
    fn open_adds_marker_after_side() {
        let el = render(SheetProps::new().side(SheetSide::Right).open(true));
        assert_eq!(el.class_names(), ["pk-sheet", "pk-sheet--right", "is-open"]);
    }

    #[test]
    fn children_are_direct_children() {
        let el = render(SheetProps::new().children([Element::box_(), Element::box_()]));
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn role_is_dialog() {
        assert_eq!(Sheet::role(), Role::Dialog);
    }
}
