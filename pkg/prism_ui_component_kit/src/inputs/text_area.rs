//! [`TextArea`] — a multi-line text input surface.
//!
//! A text area renders a `pk-text-area` box sized to `rows` lines. Like
//! [`TextField`](crate::inputs::TextField) it is a glass-lite surface that
//! attaches only kit class names; [`crate::preset`] owns the token-backed
//! values. The `rows` hint drives a minimum height via an inline length (a
//! data-derived px value, never a color/shadow literal). Editing state is
//! owned elsewhere by `prism_ui_form`.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The per-row height, in logical pixels, used to size the area from `rows`.
const ROW_HEIGHT_PX: f32 = 22.0;

/// Props for [`TextArea`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TextAreaProps {
    /// The current text value. When empty, the placeholder is shown instead.
    pub value: String,
    /// Placeholder text shown when `value` is empty.
    pub placeholder: String,
    /// The number of text rows the area should be tall enough to show.
    pub rows: u16,
    /// Whether the area is non-interactive.
    pub disabled: bool,
    /// Whether the area is in an invalid/error state.
    pub invalid: bool,
}

impl TextAreaProps {
    /// Creates empty text-area props.
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

    /// Sets the visible row count.
    #[must_use]
    pub fn rows(mut self, rows: u16) -> Self {
        self.rows = rows;
        self
    }

    /// Marks the area disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Marks the area invalid.
    #[must_use]
    pub fn invalid(mut self, invalid: bool) -> Self {
        self.invalid = invalid;
        self
    }
}

/// The text-area control. Zero-sized; config lives in [`TextAreaProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TextArea;

impl TextArea {
    /// The accessibility role a text area exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for TextArea {
    type Props = TextAreaProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-text-area");
        if props.disabled {
            el = el.class("is-disabled");
        }
        if props.invalid {
            el = el.class("is-invalid");
        }
        // Data-derived minimum height: one row-height per requested row.
        if props.rows > 0 {
            let min_h = f32::from(props.rows) * ROW_HEIGHT_PX;
            el = el.style(StyleProp::MinHeight, StyleValue::px(min_h));
        }

        if props.value.is_empty() {
            el = el
                .child(Element::text(props.placeholder.clone()).class("pk-text-area__placeholder"));
        } else {
            el = el.child(Element::text(props.value.clone()).class("pk-text-area__value"));
        }
        el
    }
}

/// Registers the `pk-text-area` class family: base surface, value/placeholder
/// text slots, and the shared invalid marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-text-area")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Stretch))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-text-area__value")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-text-area__placeholder")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(Class::new("is-invalid").with(StyleProp::BorderColor, tok("color.red")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: TextAreaProps) -> Element {
        TextArea.render(&props)
    }

    #[test]
    fn base_class_and_placeholder() {
        let el = render(TextAreaProps::new().placeholder("Notes"));
        assert_eq!(el.class_names()[0], "pk-text-area");
        let child = &el.child_elements()[0];
        assert_eq!(child.kind(), &ElementKind::Text);
        assert_eq!(child.text_content(), Some("Notes"));
        assert!(child.class_names().iter().any(|c| c == "pk-text-area__placeholder"));
    }

    #[test]
    fn rows_set_min_height_inline() {
        let el = render(TextAreaProps::new().rows(4));
        let min_h = el
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::MinHeight)
            .map(|(_, v)| v.clone());
        assert_eq!(min_h, Some(StyleValue::Length(Length::Px(4.0 * ROW_HEIGHT_PX))));
    }

    #[test]
    fn zero_rows_sets_no_inline_height() {
        let el = render(TextAreaProps::new());
        assert!(el.inline_pairs().is_empty());
    }

    #[test]
    fn disabled_and_invalid_add_markers() {
        let el = render(TextAreaProps::new().disabled(true).invalid(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
        assert!(el.class_names().iter().any(|c| c == "is-invalid"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(TextArea::role(), Role::Textbox);
    }
}
