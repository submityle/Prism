//! [`FormError`] — an inline validation error message.
//!
//! A form error renders a `pk-form-error` row holding the error `__message` in
//! the theme's danger color. It is a plain status surface and needs no ARIA
//! role of its own (announcement is the caller's concern). It carries no style
//! of its own: it attaches the `pk-form-error` class family and lets
//! [`crate::preset`] resolve every value against the active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`FormError`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FormErrorProps {
    /// The error message text.
    pub message: String,
}

impl FormErrorProps {
    /// Creates props for an error message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// The form-error control. Zero-sized; config lives in [`FormErrorProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FormError;

impl Component for FormError {
    type Props = FormErrorProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-form-error")
            .child(Element::text(props.message.clone()).class("pk-form-error__message"))
    }
}

/// Registers the `pk-form-error` class family: the danger-colored row and its
/// message slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-form-error")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::Color, tok("color.red"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );

    sheet.insert(
        Class::new("pk-form-error__message").with(StyleProp::Color, tok("color.red")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_message_slot() {
        let el = FormError.render(&FormErrorProps::new("Required field"));
        assert_eq!(el.class_names(), ["pk-form-error"]);
        let msg = &el.child_elements()[0];
        assert_eq!(msg.text_content(), Some("Required field"));
        assert_eq!(msg.class_names(), ["pk-form-error__message"]);
    }
}
