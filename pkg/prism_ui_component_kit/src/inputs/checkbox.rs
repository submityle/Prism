//! [`Checkbox`] — a two-state checkable control with an optional label.
//!
//! A checkbox renders a `pk-checkbox` row pairing a `pk-checkbox__box` marker
//! with an optional `pk-checkbox__label`. The checked state shows both as the
//! bare `is-checked` marker on the row (for the a11y/form layer) and as the
//! `pk-checkbox__box--checked` modifier on the box, which is where the fill
//! actually paints (the engine resolves classes per element, so the visual
//! lives on the box). All color comes from theme tokens.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Checkbox`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CheckboxProps {
    /// Whether the box is checked.
    pub checked: bool,
    /// Optional trailing label text.
    pub label: Option<String>,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl CheckboxProps {
    /// Creates default (unchecked, unlabelled) checkbox props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the checked state.
    #[must_use]
    pub fn checked(mut self, checked: bool) -> Self {
        self.checked = checked;
        self
    }

    /// Sets the trailing label text.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Marks the control disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The checkbox control. Zero-sized; config lives in [`CheckboxProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Checkbox;

impl Checkbox {
    /// The accessibility role a checkbox exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Checkbox
    }
}

impl Component for Checkbox {
    type Props = CheckboxProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-checkbox");
        if props.checked {
            el = el.class("is-checked");
        }
        if props.disabled {
            el = el.class("is-disabled");
        }

        let mut mark = Element::box_().class("pk-checkbox__box");
        if props.checked {
            mark = mark.class("pk-checkbox__box--checked");
        }
        el = el.child(mark);

        if let Some(label) = &props.label {
            el = el.child(Element::text(label.clone()).class("pk-checkbox__label"));
        }
        el
    }
}

/// Registers the `pk-checkbox` class family: row, box, checked box, label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-checkbox")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-checkbox__box")
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::MinWidth, StyleValue::px(20.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface")),
    );

    sheet.insert(
        Class::new("pk-checkbox__box--checked")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::BorderColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-checkbox__label")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: CheckboxProps) -> Element {
        Checkbox.render(&props)
    }

    #[test]
    fn unchecked_box_has_no_checked_modifier() {
        let el = render(CheckboxProps::new());
        assert_eq!(el.class_names(), ["pk-checkbox"]);
        let mark = &el.child_elements()[0];
        assert_eq!(mark.class_names(), ["pk-checkbox__box"]);
    }

    #[test]
    fn checked_marks_row_and_box() {
        let el = render(CheckboxProps::new().checked(true));
        assert!(el.class_names().iter().any(|c| c == "is-checked"));
        let mark = &el.child_elements()[0];
        assert_eq!(
            mark.class_names(),
            ["pk-checkbox__box", "pk-checkbox__box--checked"]
        );
    }

    #[test]
    fn label_is_rendered_after_the_box() {
        let el = render(CheckboxProps::new().label("Accept"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].kind(), &ElementKind::Text);
        assert_eq!(kids[1].text_content(), Some("Accept"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-checkbox__label"));
    }

    #[test]
    fn role_is_checkbox() {
        assert_eq!(Checkbox::role(), Role::Checkbox);
    }
}
