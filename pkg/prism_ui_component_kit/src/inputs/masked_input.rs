//! [`MaskedInput`] — a formatted single-line input (phone, card number, …).
//!
//! A masked input renders a `pk-masked-input` surface holding a `__input` slot
//! that shows the value formatted through its [`mask`](MaskedInputProps::mask),
//! or the placeholder when the formatted value is empty. The formatting rule is
//! the pure [`apply_mask`] function so it unit-tests without a runtime. All
//! color comes from theme tokens via [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Applies a `#`-mask to `raw`, returning the formatted string.
///
/// Each `#` in `mask` consumes the next decimal digit from `raw`; every other
/// mask character is inserted literally. Formatting stops as soon as the mask
/// is exhausted or there is no further digit to place after a literal, so a
/// partially typed value never grows a dangling separator.
///
/// ```
/// use prism_ui_component_kit::inputs::masked_input::apply_mask;
///
/// assert_eq!(apply_mask("(###) ###-####", "5551234567"), "(555) 123-4567");
/// assert_eq!(apply_mask("(###) ###-####", "555"), "(555");
/// ```
#[must_use]
pub fn apply_mask(mask: &str, raw: &str) -> String {
    let mut out = String::with_capacity(mask.len());
    let mut digits = raw.chars().filter(char::is_ascii_digit);
    let mut next = digits.next();
    for m in mask.chars() {
        if m == '#' {
            match next {
                Some(d) => {
                    out.push(d);
                    next = digits.next();
                }
                None => break,
            }
        } else if next.is_some() {
            out.push(m);
        } else {
            break;
        }
    }
    out
}

/// Props for [`MaskedInput`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MaskedInputProps {
    /// The raw (unformatted) value; formatting is applied via `mask`.
    pub value: String,
    /// The `#`-mask applied to `value` (see [`apply_mask`]).
    pub mask: String,
    /// Placeholder text shown when the formatted value is empty.
    pub placeholder: String,
}

impl MaskedInputProps {
    /// Creates empty masked-input props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the raw value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Sets the formatting mask.
    #[must_use]
    pub fn mask(mut self, mask: impl Into<String>) -> Self {
        self.mask = mask.into();
        self
    }

    /// Sets the placeholder text.
    #[must_use]
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }
}

/// The masked-input control. Zero-sized; config lives in [`MaskedInputProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MaskedInput;

impl MaskedInput {
    /// The accessibility role a masked input exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for MaskedInput {
    type Props = MaskedInputProps;

    fn render(&self, props: &Self::Props) -> Element {
        let el = Element::box_().class("pk-masked-input");
        let formatted = apply_mask(&props.mask, &props.value);
        if formatted.is_empty() {
            el.child(
                Element::text(props.placeholder.clone())
                    .class("pk-masked-input__input")
                    .class("pk-masked-input__input--placeholder"),
            )
        } else {
            el.child(Element::text(formatted).class("pk-masked-input__input"))
        }
    }
}

/// Registers the `pk-masked-input` class family: base surface and the
/// value/placeholder input slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-masked-input")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
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

    sheet.insert(
        Class::new("pk-masked-input__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-masked-input__input--placeholder")
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: MaskedInputProps) -> Element {
        MaskedInput.render(&props)
    }

    #[test]
    fn mask_fills_digits_and_inserts_literals() {
        assert_eq!(apply_mask("(###) ###-####", "5551234567"), "(555) 123-4567");
    }

    #[test]
    fn mask_strips_non_digits_from_raw() {
        assert_eq!(apply_mask("###-###", "12ab34c56"), "123-456");
    }

    #[test]
    fn partial_input_has_no_dangling_separator() {
        assert_eq!(apply_mask("(###) ###-####", "555"), "(555");
        assert_eq!(apply_mask("###-###", "123"), "123");
    }

    #[test]
    fn empty_raw_yields_empty_string() {
        assert_eq!(apply_mask("###-###", ""), "");
    }

    #[test]
    fn render_shows_formatted_value() {
        let el = render(MaskedInputProps::new().mask("###-##").value("12345"));
        let input = &el.child_elements()[0];
        assert_eq!(input.text_content(), Some("123-45"));
        assert_eq!(input.class_names(), ["pk-masked-input__input"]);
    }

    #[test]
    fn render_shows_placeholder_when_empty() {
        let el = render(MaskedInputProps::new().mask("###").placeholder("Phone"));
        let input = &el.child_elements()[0];
        assert_eq!(input.text_content(), Some("Phone"));
        assert!(input
            .class_names()
            .iter()
            .any(|c| c == "pk-masked-input__input--placeholder"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(MaskedInput::role(), Role::Textbox);
    }
}
