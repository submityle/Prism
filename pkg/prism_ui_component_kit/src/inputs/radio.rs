//! [`Radio`] — a mutually-exclusive checkable control with an optional label.
//!
//! A radio renders a `pk-radio` row pairing a circular `pk-radio__box` marker
//! with an optional `pk-radio__label`. Selection is shown by the
//! `pk-radio__box--selected` modifier on the box (where the fill paints); the
//! row stays unstyled, and selection semantics live in the a11y/form layer. All
//! color comes from theme tokens; the control attaches only class names.
//! Mutual exclusion is coordinated by [`RadioGroup`](crate::inputs::RadioGroup).

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Radio`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RadioProps {
    /// Whether this radio is the selected member of its group.
    pub selected: bool,
    /// Optional trailing label text.
    pub label: Option<String>,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl RadioProps {
    /// Creates default (unselected, unlabelled) radio props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the selected state.
    #[must_use]
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
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

/// The radio control. Zero-sized; config lives in [`RadioProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Radio;

impl Radio {
    /// The accessibility role a radio exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Radio
    }
}

impl Component for Radio {
    type Props = RadioProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-radio");
        if props.disabled {
            el = el.class("is-disabled");
        }

        let mut mark = Element::box_().class("pk-radio__box");
        if props.selected {
            mark = mark.class("pk-radio__box--selected");
        }
        el = el.child(mark);

        if let Some(label) = &props.label {
            el = el.child(Element::text(label.clone()).class("pk-radio__label"));
        }
        el
    }
}

/// Registers the `pk-radio` class family: row, circular box, selected box,
/// label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-radio")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-radio__box")
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::MinWidth, StyleValue::px(20.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface")),
    );

    sheet.insert(
        Class::new("pk-radio__box--selected")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::BorderColor, tok("color.tint"))
            .with(StyleProp::BorderWidth, StyleValue::px(5.0)),
    );

    sheet.insert(
        Class::new("pk-radio__label")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: RadioProps) -> Element {
        Radio.render(&props)
    }

    #[test]
    fn unselected_box_has_no_selected_modifier() {
        let el = render(RadioProps::new());
        assert_eq!(el.class_names(), ["pk-radio"]);
        let mark = &el.child_elements()[0];
        assert_eq!(mark.class_names(), ["pk-radio__box"]);
    }

    #[test]
    fn selected_marks_box_only() {
        let el = render(RadioProps::new().selected(true));
        // The row itself stays unstyled; selection lives on the dot.
        assert_eq!(el.class_names(), ["pk-radio"]);
        let mark = &el.child_elements()[0];
        assert_eq!(
            mark.class_names(),
            ["pk-radio__box", "pk-radio__box--selected"]
        );
    }

    #[test]
    fn label_is_rendered_after_the_box() {
        let el = render(RadioProps::new().label("One"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].kind(), &ElementKind::Text);
        assert_eq!(kids[1].text_content(), Some("One"));
    }

    #[test]
    fn role_is_radio() {
        assert_eq!(Radio::role(), Role::Radio);
    }
}
