//! [`NavBar`] — the top navigation / title bar chrome.
//!
//! A nav bar is a horizontal strip with an optional `leading` slot (back
//! button, menu toggle), a centered `title`, and a `trailing` group of
//! actions. It carries only `pk-nav-bar` kit classes; surface, spacing and
//! typography resolve from theme tokens via [`crate::preset`]. Setting `glass`
//! swaps the opaque surface for the kit's frosted-glass chrome treatment.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Props for [`NavBar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NavBarProps {
    /// Optional leading content (e.g. a back button), rendered at the start.
    pub leading: Option<Element>,
    /// Optional centered title text.
    pub title: Option<String>,
    /// Trailing action elements, rendered at the end in order.
    pub trailing: Vec<Element>,
    /// Whether to use the frosted-glass chrome surface.
    pub glass: bool,
}

impl NavBarProps {
    /// Creates empty nav-bar props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the leading slot content.
    #[must_use]
    pub fn leading(mut self, element: Element) -> Self {
        self.leading = Some(element);
        self
    }

    /// Sets the centered title text.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Appends a trailing action element.
    #[must_use]
    pub fn trailing_item(mut self, element: Element) -> Self {
        self.trailing.push(element);
        self
    }

    /// Replaces the trailing actions with `items`.
    #[must_use]
    pub fn trailing<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.trailing = items.into_iter().collect();
        self
    }

    /// Enables (or disables) the glass surface.
    #[must_use]
    pub fn glass(mut self, glass: bool) -> Self {
        self.glass = glass;
        self
    }
}

/// The nav-bar control. Zero-sized; config lives in [`NavBarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NavBar;

impl Component for NavBar {
    type Props = NavBarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mods: &[&str] = if props.glass { &["glass"] } else { &[] };
        let mut el = Element::box_();
        for name in classes("pk-nav-bar", mods) {
            el = el.class(name);
        }

        if let Some(leading) = props.leading.clone() {
            el = el.child(leading.class("pk-nav-bar__leading"));
        }
        if let Some(title) = props.title.clone() {
            el = el.child(Element::text(title).class("pk-nav-bar__title"));
        }
        if !props.trailing.is_empty() {
            let trailing = Element::box_()
                .class("pk-nav-bar__trailing")
                .children(props.trailing.iter().cloned());
            el = el.child(trailing);
        }
        el
    }
}

/// Registers the `pk-nav-bar` class family: bar, glass surface, and the
/// leading / title / trailing slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Bar: a centered flex row spreading leading / title / trailing with a
    // hairline bottom separator and an opaque surface by default.
    sheet.insert(
        Class::new("pk-nav-bar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::BackgroundColor, tok("color.surface"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Glass: frosted chrome surface with a lit rim and a soft drop shadow.
    sheet.insert(
        Class::new("pk-nav-bar--glass")
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 1.0, 12.0, tok("glass.shadow")),
    );

    // Leading slot: never shrinks.
    sheet.insert(
        Class::new("pk-nav-bar__leading").with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Title: grows to fill, emphasized weight.
    sheet.insert(
        Class::new("pk-nav-bar__title")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Trailing group: a tight action row that never shrinks.
    sheet.insert(
        Class::new("pk-nav-bar__trailing")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: NavBarProps) -> Element {
        NavBar.render(&props)
    }

    #[test]
    fn plain_bar_has_only_base_class() {
        let el = render(NavBarProps::new());
        assert_eq!(el.class_names(), ["pk-nav-bar"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn glass_adds_modifier_class() {
        let el = render(NavBarProps::new().glass(true));
        assert_eq!(el.class_names(), ["pk-nav-bar", "pk-nav-bar--glass"]);
    }

    #[test]
    fn title_renders_as_text_child_with_slot_class() {
        let el = render(NavBarProps::new().title("Inbox"));
        let title = el
            .child_elements()
            .iter()
            .find(|c| c.kind() == &ElementKind::Text)
            .expect("title child");
        assert_eq!(title.text_content(), Some("Inbox"));
        assert!(title.class_names().iter().any(|c| c == "pk-nav-bar__title"));
    }

    #[test]
    fn slots_render_leading_title_trailing_in_order() {
        let el = render(
            NavBarProps::new()
                .leading(Element::box_().class("back"))
                .title("Title")
                .trailing_item(Element::box_().class("a"))
                .trailing_item(Element::box_().class("b")),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert!(children[0].class_names().iter().any(|c| c == "pk-nav-bar__leading"));
        assert!(children[1].class_names().iter().any(|c| c == "pk-nav-bar__title"));
        assert!(children[2].class_names().iter().any(|c| c == "pk-nav-bar__trailing"));
        assert_eq!(children[2].child_elements().len(), 2);
    }
}
