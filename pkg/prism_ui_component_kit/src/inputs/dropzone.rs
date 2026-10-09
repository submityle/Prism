//! [`Dropzone`] — a drag-and-drop file upload target.
//!
//! A dropzone renders a `pk-dropzone` surface holding a `__label` prompt, with
//! `--active` applied while a drag hovers it and `--disabled` when it is
//! non-interactive. It exposes [`Role::Button`] so it is reachable and
//! activatable from the keyboard. It carries no style of its own: it attaches
//! the `pk-dropzone` class family and lets [`crate::preset`] resolve every
//! value against the active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Dropzone`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DropzoneProps {
    /// The prompt text shown inside the zone.
    pub label: String,
    /// Whether a drag is currently hovering the zone.
    pub active: bool,
    /// Whether the zone is non-interactive.
    pub disabled: bool,
}

impl DropzoneProps {
    /// Creates props for a labelled dropzone.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Marks the zone active (a drag is hovering).
    #[must_use]
    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Marks the zone disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The dropzone control. Zero-sized; config lives in [`DropzoneProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Dropzone;

impl Dropzone {
    /// The accessibility role a dropzone exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Button
    }

    /// The accessible name for a set of props (its label).
    #[must_use]
    pub fn accessible_name(props: &DropzoneProps) -> &str {
        &props.label
    }
}

impl Component for Dropzone {
    type Props = DropzoneProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-dropzone");
        if props.active {
            el = el.class("pk-dropzone--active");
        }
        if props.disabled {
            el = el.class("pk-dropzone--disabled");
        }
        el.child(Element::text(props.label.clone()).class("pk-dropzone__label"))
    }
}

/// Registers the `pk-dropzone` class family: base surface, active and disabled
/// modifiers, and the prompt label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-dropzone")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.xl"))
            .with_padding_y(tok("space.xl"))
            .with(StyleProp::MinHeight, StyleValue::px(96.0))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-dropzone--active")
            .with(StyleProp::BorderColor, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.tint")),
    );
    sheet.insert(
        Class::new("pk-dropzone--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-dropzone__label")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DropzoneProps) -> Element {
        Dropzone.render(&props)
    }

    #[test]
    fn renders_label() {
        let el = render(DropzoneProps::new("Drop files here"));
        assert_eq!(el.class_names(), ["pk-dropzone"]);
        assert_eq!(el.child_elements()[0].text_content(), Some("Drop files here"));
    }

    #[test]
    fn active_and_disabled_add_modifiers() {
        let el = render(DropzoneProps::new("x").active(true).disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-dropzone--active"));
        assert!(el.class_names().iter().any(|c| c == "pk-dropzone--disabled"));
    }

    #[test]
    fn accessible_name_is_the_label() {
        let props = DropzoneProps::new("Upload");
        assert_eq!(Dropzone::accessible_name(&props), "Upload");
    }

    #[test]
    fn role_is_button() {
        assert_eq!(Dropzone::role(), Role::Button);
    }
}
