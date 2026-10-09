//! [`FormField`] — the vertical wrapper tying a label, control, error and help
//! text together.
//!
//! A form field renders a `pk-form-field` column that stacks an optional label,
//! an optional control, an optional error message and an optional help string,
//! marking itself `--required` when the field is required. It is a pure layout
//! aggregator: callers pass fully-formed [`Element`]s for the label/control/
//! error slots, and this control only positions them and attaches the
//! `pk-form-field` class family. All color comes from theme tokens via
//! [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`FormField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FormFieldProps {
    /// The field's label element.
    pub label: Option<Element>,
    /// The field's control element (input, select, …).
    pub control: Option<Element>,
    /// The field's error message element, shown below the control.
    pub error: Option<Element>,
    /// Supplementary help text shown below the control.
    pub help: String,
    /// Whether the field is required.
    pub required: bool,
}

impl FormFieldProps {
    /// Creates empty form-field props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the label element.
    #[must_use]
    pub fn label(mut self, element: Element) -> Self {
        self.label = Some(element);
        self
    }

    /// Sets the control element.
    #[must_use]
    pub fn control(mut self, element: Element) -> Self {
        self.control = Some(element);
        self
    }

    /// Sets the error element.
    #[must_use]
    pub fn error(mut self, element: Element) -> Self {
        self.error = Some(element);
        self
    }

    /// Sets the help text.
    #[must_use]
    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help = help.into();
        self
    }

    /// Marks the field required.
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }
}

/// The form-field control. Zero-sized; config lives in [`FormFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FormField;

impl FormField {
    /// The accessibility role a form field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for FormField {
    type Props = FormFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-form-field");
        if props.required {
            el = el.class("pk-form-field--required");
        }

        if let Some(label) = props.label.clone() {
            el = el.child(label.class("pk-form-field__label"));
        }
        if let Some(control) = props.control.clone() {
            el = el.child(control.class("pk-form-field__control"));
        }
        if let Some(error) = props.error.clone() {
            el = el.child(error.class("pk-form-field__error"));
        }
        if !props.help.is_empty() {
            el = el.child(Element::text(props.help.clone()).class("pk-form-field__help"));
        }
        el
    }
}

/// Registers the `pk-form-field` class family: the vertical stack, its required
/// modifier and the label/control/error/help slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-form-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    sheet.insert(Class::new("pk-form-field--required"));

    sheet.insert(Class::new("pk-form-field__label"));
    sheet.insert(Class::new("pk-form-field__control"));
    sheet.insert(
        Class::new("pk-form-field__error").with(StyleProp::Color, tok("color.red")),
    );
    sheet.insert(
        Class::new("pk-form-field__help")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacks_label_control_error_and_help_in_order() {
        let el = FormField.render(
            &FormFieldProps::new()
                .label(Element::box_())
                .control(Element::box_())
                .error(Element::box_())
                .help("Hint"),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 4);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-form-field__label"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-form-field__control"));
        assert!(kids[2].class_names().iter().any(|c| c == "pk-form-field__error"));
        assert!(kids[3].class_names().iter().any(|c| c == "pk-form-field__help"));
        assert_eq!(kids[3].text_content(), Some("Hint"));
    }

    #[test]
    fn omitting_slots_omits_children() {
        let el = FormField.render(&FormFieldProps::new().control(Element::box_()));
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn required_adds_modifier() {
        let el = FormField.render(&FormFieldProps::new().required(true));
        assert!(el.class_names().iter().any(|c| c == "pk-form-field--required"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(FormField::role(), Role::Group);
    }
}
