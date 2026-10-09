//! [`FormLabel`] — a field label with an optional required marker.
//!
//! A form label renders a `pk-form-label` row holding the label `__text` and,
//! when the field is required, a `__required` asterisk marker; the row is also
//! tagged `--required` so callers can style the whole label. It carries no
//! style of its own: it attaches the `pk-form-label` class family and lets
//! [`crate::preset`] resolve every value against the active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`FormLabel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FormLabelProps {
    /// The label text.
    pub text: String,
    /// Whether the labelled field is required (shows an asterisk marker).
    pub required: bool,
}

impl FormLabelProps {
    /// Creates props for a labelled text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            required: false,
        }
    }

    /// Marks the labelled field required.
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

/// The form-label control. Zero-sized; config lives in [`FormLabelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FormLabel;

impl Component for FormLabel {
    type Props = FormLabelProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-form-label")
            .child(Element::text(props.text.clone()).class("pk-form-label__text"));
        if props.required {
            el = el
                .class("pk-form-label--required")
                .child(Element::text("*").class("pk-form-label__required"));
        }
        el
    }
}

/// Registers the `pk-form-label` class family: base row, required modifier,
/// text slot and the asterisk marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-form-label")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );

    sheet.insert(Class::new("pk-form-label--required"));

    sheet.insert(Class::new("pk-form-label__text").with(StyleProp::Color, tok("color.label")));
    sheet.insert(
        Class::new("pk-form-label__required").with(StyleProp::Color, tok("color.red")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: FormLabelProps) -> Element {
        FormLabel.render(&props)
    }

    #[test]
    fn renders_text_slot() {
        let el = render(FormLabelProps::new("Email"));
        assert_eq!(el.class_names(), ["pk-form-label"]);
        let text = &el.child_elements()[0];
        assert_eq!(text.text_content(), Some("Email"));
        assert_eq!(text.class_names(), ["pk-form-label__text"]);
    }

    #[test]
    fn required_adds_modifier_and_asterisk() {
        let el = render(FormLabelProps::new("Email").required(true));
        assert!(el.class_names().iter().any(|c| c == "pk-form-label--required"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].text_content(), Some("*"));
        assert_eq!(kids[1].class_names(), ["pk-form-label__required"]);
    }

    #[test]
    fn optional_label_has_no_asterisk() {
        let el = render(FormLabelProps::new("Email"));
        assert_eq!(el.child_elements().len(), 1);
    }
}
