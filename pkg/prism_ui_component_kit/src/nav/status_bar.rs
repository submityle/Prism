//! [`StatusBar`] — the bottom status strip.
//!
//! A status bar is a thin horizontal strip pinned to the bottom of a window.
//! It spreads an optional group of `leading` items, a centered `message`, and
//! a group of `trailing` items across its width. It carries only
//! `pk-status-bar` kit classes; surface, spacing and typography resolve from
//! theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`StatusBar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StatusBarProps {
    /// Leading items, rendered at the start in order.
    pub leading: Vec<Element>,
    /// Trailing items, rendered at the end in order.
    pub trailing: Vec<Element>,
    /// The centered status message.
    pub message: String,
}

impl StatusBarProps {
    /// Creates empty status-bar props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a leading item.
    #[must_use]
    pub fn leading_item(mut self, element: Element) -> Self {
        self.leading.push(element);
        self
    }

    /// Replaces the leading items with `items`.
    #[must_use]
    pub fn leading<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.leading = items.into_iter().collect();
        self
    }

    /// Appends a trailing item.
    #[must_use]
    pub fn trailing_item(mut self, element: Element) -> Self {
        self.trailing.push(element);
        self
    }

    /// Replaces the trailing items with `items`.
    #[must_use]
    pub fn trailing<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.trailing = items.into_iter().collect();
        self
    }

    /// Sets the status message text.
    #[must_use]
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }
}

/// The status-bar control. Zero-sized; config lives in [`StatusBarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StatusBar;

impl Component for StatusBar {
    type Props = StatusBarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-status-bar");

        if !props.leading.is_empty() {
            let leading = Element::box_()
                .class("pk-status-bar__leading")
                .children(props.leading.iter().cloned());
            el = el.child(leading);
        }
        if !props.message.is_empty() {
            el = el.child(Element::text(props.message.clone()).class("pk-status-bar__message"));
        }
        if !props.trailing.is_empty() {
            let trailing = Element::box_()
                .class("pk-status-bar__trailing")
                .children(props.trailing.iter().cloned());
            el = el.child(trailing);
        }
        el
    }
}

/// Registers the `pk-status-bar` class family: the strip plus the leading /
/// message / trailing slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Strip: a centered flex row spreading its slots, hairline top separator,
    // opaque secondary surface, footnote-sized secondary label.
    sheet.insert(
        Class::new("pk-status-bar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Leading slot: a tight item row that never shrinks.
    sheet.insert(
        Class::new("pk-status-bar__leading")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Message: grows to fill the middle, inheriting the strip's muted label.
    sheet.insert(
        Class::new("pk-status-bar__message")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Trailing slot: a tight item row that never shrinks.
    sheet.insert(
        Class::new("pk-status-bar__trailing")
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

    fn render(props: StatusBarProps) -> Element {
        StatusBar.render(&props)
    }

    #[test]
    fn empty_bar_has_only_base_class() {
        let el = render(StatusBarProps::new());
        assert_eq!(el.class_names(), ["pk-status-bar"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn message_renders_as_text_child_with_slot_class() {
        let el = render(StatusBarProps::new().message("Ready"));
        let msg = el
            .child_elements()
            .iter()
            .find(|c| c.kind() == &ElementKind::Text)
            .expect("message child");
        assert_eq!(msg.text_content(), Some("Ready"));
        assert!(msg.class_names().iter().any(|c| c == "pk-status-bar__message"));
    }

    #[test]
    fn slots_render_leading_message_trailing_in_order() {
        let el = render(
            StatusBarProps::new()
                .leading_item(Element::box_().class("a"))
                .message("Saving…")
                .trailing_item(Element::box_().class("b"))
                .trailing_item(Element::box_().class("c")),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert!(children[0].class_names().iter().any(|c| c == "pk-status-bar__leading"));
        assert!(children[1].class_names().iter().any(|c| c == "pk-status-bar__message"));
        assert!(children[2].class_names().iter().any(|c| c == "pk-status-bar__trailing"));
        assert_eq!(children[2].child_elements().len(), 2);
    }
}
