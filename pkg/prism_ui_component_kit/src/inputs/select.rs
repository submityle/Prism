//! [`Select`] — a trigger showing the chosen option (a dropdown's closed face).
//!
//! A select renders a `pk-select` trigger pairing a `pk-select__value` (the
//! chosen option, or the placeholder when nothing valid is selected) with a
//! `pk-select__chevron` affordance. This stateless layer renders only the
//! closed trigger; the popup/menu and its open state are owned elsewhere. All
//! color comes from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Select`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SelectProps {
    /// The available options.
    pub options: Vec<String>,
    /// The index of the selected option. Out-of-range shows the placeholder.
    pub selected: usize,
    /// Placeholder text shown when no valid option is selected.
    pub placeholder: String,
    /// Whether the trigger is non-interactive.
    pub disabled: bool,
}

impl SelectProps {
    /// Creates empty select props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the options with `options`.
    #[must_use]
    pub fn options<I, S>(mut self, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options = options.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the selected option index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }

    /// Sets the placeholder text.
    #[must_use]
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Marks the trigger disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The select control. Zero-sized; config lives in [`SelectProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Select;

impl Component for Select {
    type Props = SelectProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-select");
        if props.disabled {
            el = el.class("is-disabled");
        }

        let value = match props.options.get(props.selected) {
            Some(option) => {
                Element::text(option.clone()).class("pk-select__value")
            }
            None => Element::text(props.placeholder.clone())
                .class("pk-select__value")
                .class("pk-select__value--placeholder"),
        };

        el.child(value)
            .child(Element::box_().class("pk-select__chevron"))
    }
}

/// Registers the `pk-select` class family: trigger, value, placeholder value,
/// chevron.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-select")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BorderColor, tok("color.tint"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-select__value")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );
    sheet.insert(
        Class::new("pk-select__value--placeholder")
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );

    sheet.insert(
        Class::new("pk-select__chevron")
            .with(StyleProp::Width, StyleValue::px(10.0))
            .with(StyleProp::Height, StyleValue::px(6.0))
            .with(StyleProp::MinWidth, StyleValue::px(10.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary"))
            .with(StyleProp::BorderRadius, tok("radius.xs")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SelectProps) -> Element {
        Select.render(&props)
    }

    #[test]
    fn selected_option_shows_as_value() {
        let el = render(SelectProps::new().options(["A", "B"]).selected(1));
        let value = &el.child_elements()[0];
        assert_eq!(value.text_content(), Some("B"));
        assert_eq!(value.class_names(), ["pk-select__value"]);
    }

    #[test]
    fn out_of_range_shows_placeholder() {
        let el = render(
            SelectProps::new()
                .options(["A"])
                .selected(5)
                .placeholder("Pick one"),
        );
        let value = &el.child_elements()[0];
        assert_eq!(value.text_content(), Some("Pick one"));
        assert!(value
            .class_names()
            .iter()
            .any(|c| c == "pk-select__value--placeholder"));
    }

    #[test]
    fn trigger_ends_with_a_chevron() {
        let el = render(SelectProps::new().options(["A"]).selected(0));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].class_names(), ["pk-select__chevron"]);
    }

    #[test]
    fn disabled_adds_marker() {
        let el = render(SelectProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "is-disabled"));
    }
}
