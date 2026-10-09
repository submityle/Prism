//! [`Label`] — a form/field caption that associates with a control.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Label`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LabelProps {
    /// The caption text.
    pub text: String,
    /// Whether the associated field is required (adds a `pk-label__required`).
    pub required: bool,
}

impl LabelProps {
    /// Creates props for a label.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Marks the associated field as required.
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

/// The label control. Zero-sized; all configuration lives in [`LabelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Label;

impl Component for Label {
    type Props = LabelProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-label")
            .child(Element::text(props.text.clone()).class("pk-label__text"));
        if props.required {
            el = el.child(Element::text("*").class("pk-label__required"));
        }
        el
    }
}

/// Registers the `pk-label` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-label")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs")),
    );
    sheet.insert(
        Class::new("pk-label__text")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );
    sheet.insert(Class::new("pk-label__required").with(StyleProp::Color, tok("color.red")));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_adds_marker_child() {
        let el = Label.render(&LabelProps::new("Name").required(true));
        assert_eq!(el.class_names(), ["pk-label"]);
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn optional_has_single_child() {
        let el = Label.render(&LabelProps::new("Name"));
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-label", "pk-label__text", "pk-label__required"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
