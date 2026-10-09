//! [`SearchField`] — a text field with a leading search glyph slot.
//!
//! A search field renders a `pk-search-field` surface pairing a leading
//! `pk-search-field__icon` slot (where a search glyph is drawn) with the value
//! (or placeholder when empty). It is a sibling of
//! [`TextField`](crate::inputs::TextField) specialised for search; like every
//! control it attaches only kit class names and lets [`crate::preset`] resolve
//! values from theme tokens. Editing state is owned by `prism_ui_form`.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`SearchField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SearchFieldProps {
    /// The current text value. When empty, the placeholder is shown instead.
    pub value: String,
    /// Placeholder text shown when `value` is empty.
    pub placeholder: String,
    /// Whether the field is non-interactive.
    pub disabled: bool,
}

impl SearchFieldProps {
    /// Creates empty search-field props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Sets the placeholder text.
    #[must_use]
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Marks the field disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The search-field control. Zero-sized; config lives in [`SearchFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchField;

impl SearchField {
    /// The accessibility role a search field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for SearchField {
    type Props = SearchFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-search-field");
        if props.disabled {
            el = el.class("is-disabled");
        }

        // Leading glyph slot: a data-only box the backend paints as a glass.
        el = el.child(Element::box_().class("pk-search-field__icon"));

        if props.value.is_empty() {
            el = el.child(
                Element::text(props.placeholder.clone())
                    .class("pk-search-field__placeholder"),
            );
        } else {
            el = el.child(Element::text(props.value.clone()).class("pk-search-field__value"));
        }
        el
    }
}

/// Registers the `pk-search-field` class family: surface, icon slot,
/// value/placeholder text slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-search-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with_glass(0.0, tok("color.fill.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-search-field__icon")
            .with(StyleProp::Width, StyleValue::px(14.0))
            .with(StyleProp::Height, StyleValue::px(14.0))
            .with(StyleProp::MinWidth, StyleValue::px(14.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary")),
    );

    sheet.insert(
        Class::new("pk-search-field__value")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-search-field__placeholder")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: SearchFieldProps) -> Element {
        SearchField.render(&props)
    }

    #[test]
    fn leads_with_an_icon_slot() {
        let el = render(SearchFieldProps::new());
        assert_eq!(el.class_names()[0], "pk-search-field");
        assert_eq!(el.child_elements()[0].class_names(), ["pk-search-field__icon"]);
    }

    #[test]
    fn empty_value_shows_placeholder() {
        let el = render(SearchFieldProps::new().placeholder("Search"));
        let text = &el.child_elements()[1];
        assert_eq!(text.kind(), &ElementKind::Text);
        assert_eq!(text.text_content(), Some("Search"));
        assert!(text.class_names().iter().any(|c| c == "pk-search-field__placeholder"));
    }

    #[test]
    fn non_empty_value_shows_value() {
        let el = render(SearchFieldProps::new().value("ab"));
        let text = &el.child_elements()[1];
        assert_eq!(text.text_content(), Some("ab"));
        assert!(text.class_names().iter().any(|c| c == "pk-search-field__value"));
    }

    #[test]
    fn disabled_adds_marker() {
        let el = render(SearchFieldProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(SearchField::role(), Role::Textbox);
    }
}
