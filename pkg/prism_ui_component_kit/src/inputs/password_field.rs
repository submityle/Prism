//! [`PasswordField`] — a single-line secret input surface.
//!
//! A password field renders a `pk-password-field` row holding a `__input` slot
//! (the masked value, or the placeholder when empty) and a `__reveal` toggle
//! that flips the value between hidden and shown. It carries no style of its
//! own: it attaches the `pk-password-field` class family and lets
//! [`crate::preset`] resolve every value against the active theme. Editing and
//! the real reveal state are owned elsewhere by `prism_ui_form`; this control
//! only draws the data-only structure its props describe.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`PasswordField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PasswordFieldProps {
    /// The current secret value. When empty, the placeholder is shown instead.
    pub value: String,
    /// Placeholder text shown when `value` is empty.
    pub placeholder: String,
    /// Whether the value is shown in clear text rather than masked.
    pub revealed: bool,
    /// Whether the field is in an invalid/error state.
    pub invalid: bool,
}

impl PasswordFieldProps {
    /// Creates empty password-field props.
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

    /// Sets whether the value is shown in clear text.
    #[must_use]
    pub fn revealed(mut self, revealed: bool) -> Self {
        self.revealed = revealed;
        self
    }

    /// Marks the field invalid.
    #[must_use]
    pub fn invalid(mut self, invalid: bool) -> Self {
        self.invalid = invalid;
        self
    }
}

/// Builds a run of bullet glyphs standing in for a hidden secret of `len`
/// characters.
fn bullets(len: usize) -> String {
    let mut out = String::with_capacity(len);
    for _ in 0..len {
        out.push('\u{2022}');
    }
    out
}

/// The password-field control. Zero-sized; all configuration lives in
/// [`PasswordFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PasswordField;

impl PasswordField {
    /// The accessibility role a password field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for PasswordField {
    type Props = PasswordFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-password-field");
        if props.revealed {
            el = el.class("pk-password-field--revealed");
        }
        if props.invalid {
            el = el.class("pk-password-field--invalid");
        }

        if props.value.is_empty() {
            el = el.child(
                Element::text(props.placeholder.clone())
                    .class("pk-password-field__input")
                    .class("pk-password-field__input--placeholder"),
            );
        } else {
            let shown = if props.revealed {
                props.value.clone()
            } else {
                bullets(props.value.chars().count())
            };
            el = el.child(Element::text(shown).class("pk-password-field__input"));
        }

        el.child(Element::box_().class("pk-password-field__reveal"))
    }
}

/// Registers the `pk-password-field` class family: base surface, block-level
/// invalid marker, revealed marker, input/placeholder slots, reveal toggle.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-password-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint")),
    );

    // Block-level invalid modifier (no bare `is-invalid` marker here).
    sheet.insert(
        Class::new("pk-password-field--invalid")
            .with(StyleProp::BorderColor, tok("color.red")),
    );

    sheet.insert(
        Class::new("pk-password-field__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-password-field__input--placeholder")
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );

    sheet.insert(
        Class::new("pk-password-field__reveal")
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::MinWidth, StyleValue::px(20.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.xs"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: PasswordFieldProps) -> Element {
        PasswordField.render(&props)
    }

    #[test]
    fn empty_value_renders_placeholder_slot() {
        let el = render(PasswordFieldProps::new().placeholder("Password"));
        assert_eq!(el.class_names(), ["pk-password-field"]);
        let input = &el.child_elements()[0];
        assert_eq!(input.kind(), &ElementKind::Text);
        assert_eq!(input.text_content(), Some("Password"));
        assert!(input
            .class_names()
            .iter()
            .any(|c| c == "pk-password-field__input--placeholder"));
    }

    #[test]
    fn hidden_value_is_masked_with_bullets() {
        let el = render(PasswordFieldProps::new().value("abc"));
        let input = &el.child_elements()[0];
        assert_eq!(input.text_content(), Some("\u{2022}\u{2022}\u{2022}"));
    }

    #[test]
    fn revealed_shows_clear_text_and_modifier() {
        let el = render(PasswordFieldProps::new().value("abc").revealed(true));
        assert!(el
            .class_names()
            .iter()
            .any(|c| c == "pk-password-field--revealed"));
        assert_eq!(el.child_elements()[0].text_content(), Some("abc"));
    }

    #[test]
    fn invalid_uses_block_modifier_not_bare_marker() {
        let el = render(PasswordFieldProps::new().invalid(true));
        assert!(el
            .class_names()
            .iter()
            .any(|c| c == "pk-password-field--invalid"));
        assert!(!el.class_names().iter().any(|c| c == "is-invalid"));
    }

    #[test]
    fn trailing_child_is_the_reveal_toggle() {
        let el = render(PasswordFieldProps::new().value("x"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].class_names(), ["pk-password-field__reveal"]);
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(PasswordField::role(), Role::Textbox);
    }
}
