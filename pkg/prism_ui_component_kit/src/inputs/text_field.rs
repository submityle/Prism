//! [`TextField`] — a single-line text input surface.
//!
//! A text field renders a `pk-text-field` box: an optional leading slot, the
//! value (or placeholder when empty), and an optional trailing slot, laid out
//! as a glass-lite surface. It carries no style of its own — it attaches the
//! `pk-text-field` class family and lets [`crate::preset`] resolve every value
//! against the active theme. State (focus, caret, editing) is owned elsewhere
//! by `prism_ui_form`; this control only renders the data-only structure.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::{classes, ControlSize};
use crate::preset::StyleSheet;

/// Props for [`TextField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TextFieldProps {
    /// The current text value. When empty, the placeholder is shown instead.
    pub value: String,
    /// Placeholder text shown when `value` is empty.
    pub placeholder: String,
    /// Density step.
    pub size: ControlSize,
    /// Whether the field is non-interactive.
    pub disabled: bool,
    /// Whether the field is in an invalid/error state.
    pub invalid: bool,
    /// Optional leading content (e.g. an icon), rendered before the value.
    pub leading: Option<Element>,
    /// Optional trailing content, rendered after the value.
    pub trailing: Option<Element>,
}

impl TextFieldProps {
    /// Creates empty text-field props.
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

    /// Sets the density step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }

    /// Marks the field disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Marks the field invalid.
    #[must_use]
    pub fn invalid(mut self, invalid: bool) -> Self {
        self.invalid = invalid;
        self
    }

    /// Sets leading content rendered before the value.
    #[must_use]
    pub fn leading(mut self, element: Element) -> Self {
        self.leading = Some(element);
        self
    }

    /// Sets trailing content rendered after the value.
    #[must_use]
    pub fn trailing(mut self, element: Element) -> Self {
        self.trailing = Some(element);
        self
    }
}

/// The text-field control. Zero-sized; all configuration lives in
/// [`TextFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TextField;

impl TextField {
    /// The accessibility role a text field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for TextField {
    type Props = TextFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-text-field", &[props.size.suffix()]) {
            el = el.class(name);
        }
        if props.disabled {
            el = el.class("is-disabled");
        }
        if props.invalid {
            el = el.class("is-invalid");
        }

        if let Some(leading) = props.leading.clone() {
            el = el.child(leading.class("pk-text-field__leading"));
        }

        if props.value.is_empty() {
            el = el.child(
                Element::text(props.placeholder.clone()).class("pk-text-field__placeholder"),
            );
        } else {
            el = el.child(Element::text(props.value.clone()).class("pk-text-field__value"));
        }

        if let Some(trailing) = props.trailing.clone() {
            el = el.child(trailing.class("pk-text-field__trailing"));
        }
        el
    }
}

/// Registers the `pk-text-field` class family: base surface, three sizes,
/// value/placeholder text slots, and the shared invalid marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a centered flex row glass-lite surface with a hairline border.
    sheet.insert(
        Class::new("pk-text-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
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

    // Sizes map density to the spacing scale and shared control height.
    sheet.insert(
        Class::new("pk-text-field--sm")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::Height, StyleValue::px(28.0))
            .with(StyleProp::FontSize, tok("font.size.subheadline")),
    );
    sheet.insert(
        Class::new("pk-text-field--md")
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-text-field--lg")
            .with_padding_x(tok("space.lg"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Height, StyleValue::px(44.0))
            .with(StyleProp::FontSize, tok("font.size.headline")),
    );

    // Value fills the row; placeholder is the same slot, dimmed.
    sheet.insert(
        Class::new("pk-text-field__value")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-text-field__placeholder")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    // Invalid: red hairline (shared bare marker; idempotent across controls).
    sheet.insert(Class::new("is-invalid").with(StyleProp::BorderColor, tok("color.red")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: TextFieldProps) -> Element {
        TextField.render(&props)
    }

    #[test]
    fn attaches_base_and_size_classes_in_order() {
        let el = render(TextFieldProps::new().size(ControlSize::Small));
        assert_eq!(el.class_names(), ["pk-text-field", "pk-text-field--sm"]);
    }

    #[test]
    fn empty_value_renders_placeholder_slot() {
        let el = render(TextFieldProps::new().placeholder("Email"));
        let child = &el.child_elements()[0];
        assert_eq!(child.kind(), &ElementKind::Text);
        assert_eq!(child.text_content(), Some("Email"));
        assert!(child.class_names().iter().any(|c| c == "pk-text-field__placeholder"));
    }

    #[test]
    fn non_empty_value_renders_value_slot() {
        let el = render(TextFieldProps::new().value("hi").placeholder("Email"));
        let child = &el.child_elements()[0];
        assert_eq!(child.text_content(), Some("hi"));
        assert!(child.class_names().iter().any(|c| c == "pk-text-field__value"));
    }

    #[test]
    fn disabled_and_invalid_add_markers() {
        let el = render(TextFieldProps::new().disabled(true).invalid(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
        assert!(el.class_names().iter().any(|c| c == "is-invalid"));
    }

    #[test]
    fn leading_and_trailing_wrap_the_value() {
        let el = render(
            TextFieldProps::new()
                .value("x")
                .leading(Element::box_())
                .trailing(Element::box_()),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-text-field__leading"));
        assert!(kids[2].class_names().iter().any(|c| c == "pk-text-field__trailing"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(TextField::role(), Role::Textbox);
    }
}
