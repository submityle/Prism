//! [`Kbd`] — renders a keyboard key cap (e.g. `⌘`, `Enter`).

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Kbd`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct KbdProps {
    /// The key label.
    pub key: String,
}

impl KbdProps {
    /// Creates props for a key cap.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self { key: key.into() }
    }
}

/// The keyboard-key control. Zero-sized; configuration lives in [`KbdProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Kbd;

impl Component for Kbd {
    type Props = KbdProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-kbd")
            .child(Element::text(props.key.clone()).class("pk-kbd__key"))
    }
}

/// Registers the `pk-kbd` class.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-kbd")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderColor, tok("color.separator.opaque"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xxs")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_key_child() {
        let el = Kbd.render(&KbdProps::new("Enter"));
        assert_eq!(el.class_names(), ["pk-kbd"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].text_content(), Some("Enter"));
    }

    #[test]
    fn register_adds_class() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-kbd").is_some());
    }
}
