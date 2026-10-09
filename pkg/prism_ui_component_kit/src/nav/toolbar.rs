//! [`Toolbar`] — a horizontal strip of actions.
//!
//! A toolbar renders a `pk-toolbar` flex row that lays out its `items` with a
//! consistent [`Gap`](prism_ui_style::StyleProp::Gap). Setting `glass` swaps
//! the plain surface for the kit's frosted-glass chrome. The control attaches
//! only kit class names; spacing and surface resolve from theme tokens.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Props for [`Toolbar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToolbarProps {
    /// The action elements, laid out left-to-right.
    pub items: Vec<Element>,
    /// Whether to use the frosted-glass chrome surface.
    pub glass: bool,
}

impl ToolbarProps {
    /// Creates empty toolbar props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an action element.
    #[must_use]
    pub fn item(mut self, element: Element) -> Self {
        self.items.push(element);
        self
    }

    /// Replaces the items with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Enables (or disables) the glass surface.
    #[must_use]
    pub fn glass(mut self, glass: bool) -> Self {
        self.glass = glass;
        self
    }
}

/// The toolbar control. Zero-sized; config lives in [`ToolbarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Toolbar;

impl Component for Toolbar {
    type Props = ToolbarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mods: &[&str] = if props.glass { &["glass"] } else { &[] };
        let mut el = Element::box_();
        for name in classes("pk-toolbar", mods) {
            el = el.class(name);
        }
        el.children(props.items.iter().cloned())
    }
}

/// Registers the `pk-toolbar` class family: the base row and the glass surface.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Base: a centered flex row with gutter padding and a token gap.
    sheet.insert(
        Class::new("pk-toolbar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BackgroundColor, tok("color.fill")),
    );

    // Glass: frosted chrome surface with a lit rim and a soft drop shadow.
    sheet.insert(
        Class::new("pk-toolbar--glass")
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 4.0, 16.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ToolbarProps) -> Element {
        Toolbar.render(&props)
    }

    #[test]
    fn plain_toolbar_wraps_items() {
        let el = render(
            ToolbarProps::new()
                .item(Element::box_().class("a"))
                .item(Element::box_().class("b")),
        );
        assert_eq!(el.class_names(), ["pk-toolbar"]);
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn glass_adds_modifier_class() {
        let el = render(ToolbarProps::new().glass(true));
        assert_eq!(el.class_names(), ["pk-toolbar", "pk-toolbar--glass"]);
    }

    #[test]
    fn items_replaces_contents() {
        let el = render(ToolbarProps::new().items([
            Element::box_(),
            Element::box_(),
            Element::box_(),
        ]));
        assert_eq!(el.child_elements().len(), 3);
    }
}
