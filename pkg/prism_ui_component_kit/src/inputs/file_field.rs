//! [`FileField`] — a file chooser rendered as a labelled button.
//!
//! Unlike [`crate::inputs::Dropzone`] (a large drag-and-drop target), a file
//! field is a compact trigger: a `pk-file-field` row pairing a `__button`
//! (a tinted glass affordance carrying the `label`) with a `__filename` slot
//! that shows the chosen file, or the `label` as a fallback hint when nothing
//! is selected. All color comes from theme tokens via [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`FileField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FileFieldProps {
    /// The button label (e.g. "Choose file"); also the empty-state hint.
    pub label: String,
    /// The chosen file's name. `None` shows the label as a placeholder hint.
    pub filename: Option<String>,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl FileFieldProps {
    /// Creates empty file-field props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the button label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the chosen file name.
    #[must_use]
    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Marks the control disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The file-field control. Zero-sized; config lives in [`FileFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FileField;

impl FileField {
    /// The accessibility role a file field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Button
    }
}

impl Component for FileField {
    type Props = FileFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-file-field");
        if props.disabled {
            el = el.class("pk-file-field--disabled");
        }

        let button = Element::text(props.label.clone()).class("pk-file-field__button");

        // Chosen name when present; otherwise the label stands in as a hint.
        let filename = match &props.filename {
            Some(name) => Element::text(name.clone()).class("pk-file-field__filename"),
            None => Element::text(props.label.clone())
                .class("pk-file-field__filename")
                .class("pk-file-field__filename--placeholder"),
        };

        el.child(button).child(filename)
    }
}

/// Registers the `pk-file-field` class family: row, disabled modifier, the
/// tinted glass button and the chosen-filename slot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-file-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-file-field--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-file-field__button")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_glass(0.0, tok("color.fill.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.6)),
    );

    sheet.insert(
        Class::new("pk-file-field__filename")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-file-field__filename--placeholder")
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: FileFieldProps) -> Element {
        FileField.render(&props)
    }

    #[test]
    fn renders_button_then_filename() {
        let el = render(FileFieldProps::new().label("Choose").filename("a.png"));
        assert_eq!(el.class_names(), ["pk-file-field"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-file-field__button"]);
        assert_eq!(kids[0].text_content(), Some("Choose"));
        assert_eq!(kids[1].text_content(), Some("a.png"));
    }

    #[test]
    fn chosen_filename_is_not_placeholder() {
        let el = render(FileFieldProps::new().label("Choose").filename("doc.pdf"));
        let name = &el.child_elements()[1];
        assert!(name.class_names().iter().any(|c| c == "pk-file-field__filename"));
        assert!(!name
            .class_names()
            .iter()
            .any(|c| c == "pk-file-field__filename--placeholder"));
    }

    #[test]
    fn no_file_shows_label_as_placeholder() {
        let el = render(FileFieldProps::new().label("Choose file"));
        let name = &el.child_elements()[1];
        assert_eq!(name.text_content(), Some("Choose file"));
        assert!(name
            .class_names()
            .iter()
            .any(|c| c == "pk-file-field__filename--placeholder"));
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(FileFieldProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-file-field--disabled"));
    }

    #[test]
    fn role_is_button() {
        assert_eq!(FileField::role(), Role::Button);
    }
}
