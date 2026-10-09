//! [`MultiSelect`] — a trigger showing selected chips plus a dropdown menu.
//!
//! A multi-select renders a `pk-multi-select` box pairing the selected options
//! (as removable `__chip`s, or the placeholder when none are selected) with a
//! dropdown `__menu` of `__option`s. The dropdown is composed from the shared
//! [`Popover`] overlay base rather than a bespoke surface, per the kit's
//! single-overlay invariant. All color comes from theme tokens via
//! [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`MultiSelect`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MultiSelectProps {
    /// The available options.
    pub options: Vec<String>,
    /// Indices of the selected options. Out-of-range indices are ignored.
    pub selected: Vec<usize>,
    /// Placeholder text shown when nothing is selected.
    pub placeholder: String,
    /// Whether the dropdown menu is open.
    pub open: bool,
}

impl MultiSelectProps {
    /// Creates empty, closed multi-select props.
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

    /// Replaces the selected indices with `selected`.
    #[must_use]
    pub fn selected<I: IntoIterator<Item = usize>>(mut self, selected: I) -> Self {
        self.selected = selected.into_iter().collect();
        self
    }

    /// Sets the placeholder text.
    #[must_use]
    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Sets whether the dropdown is open.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The multi-select control. Zero-sized; config lives in [`MultiSelectProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MultiSelect;

impl MultiSelect {
    /// The accessibility role a multi-select exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for MultiSelect {
    type Props = MultiSelectProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-multi-select");

        // Selected options render as removable chips; empty falls back to the
        // placeholder slot.
        let mut any_chip = false;
        for &i in &props.selected {
            if let Some(option) = props.options.get(i) {
                any_chip = true;
                el = el.child(
                    Element::box_()
                        .class("pk-multi-select__chip")
                        .child(Element::text(option.clone()).class("pk-multi-select__chip-label"))
                        .child(Element::box_().class("pk-multi-select__chip-remove")),
                );
            }
        }
        if !any_chip {
            el = el.child(
                Element::text(props.placeholder.clone()).class("pk-multi-select__placeholder"),
            );
        }

        // Dropdown menu composed from the shared overlay base.
        let mut menu = Element::box_().class("pk-multi-select__menu");
        for (i, option) in props.options.iter().enumerate() {
            let mut opt = Element::text(option.clone()).class("pk-multi-select__option");
            if props.selected.contains(&i) {
                opt = opt.class("pk-multi-select__option--selected");
            }
            menu = menu.child(opt);
        }
        let popover = Popover.render(&PopoverProps::new().open(props.open).child(menu));

        el.child(popover)
    }
}

/// Registers the `pk-multi-select` class family: base box, chips, placeholder,
/// dropdown menu and its options.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-multi-select")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-multi-select__chip")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );
    sheet.insert(
        Class::new("pk-multi-select__chip-label").with(StyleProp::Color, tok("color.label")),
    );
    sheet.insert(
        Class::new("pk-multi-select__chip-remove")
            .with(StyleProp::Width, StyleValue::px(12.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::MinWidth, StyleValue::px(12.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.label.secondary")),
    );

    sheet.insert(
        Class::new("pk-multi-select__placeholder")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-multi-select__menu")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::MinWidth, StyleValue::px(160.0)),
    );
    sheet.insert(
        Class::new("pk-multi-select__option")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
    sheet.insert(
        Class::new("pk-multi-select__option--selected")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: MultiSelectProps) -> Element {
        MultiSelect.render(&props)
    }

    #[test]
    fn selected_options_render_as_chips() {
        let el = render(
            MultiSelectProps::new()
                .options(["A", "B", "C"])
                .selected([0, 2]),
        );
        let chips: Vec<_> = el
            .child_elements()
            .iter()
            .filter(|c| c.class_names().iter().any(|n| n == "pk-multi-select__chip"))
            .collect();
        assert_eq!(chips.len(), 2);
        assert_eq!(chips[0].child_elements()[0].text_content(), Some("A"));
        assert_eq!(chips[1].child_elements()[0].text_content(), Some("C"));
    }

    #[test]
    fn no_selection_shows_placeholder() {
        let el = render(MultiSelectProps::new().options(["A"]).placeholder("Pick"));
        let ph = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-multi-select__placeholder"))
            .expect("placeholder");
        assert_eq!(ph.text_content(), Some("Pick"));
    }

    #[test]
    fn menu_marks_selected_options() {
        let el = render(MultiSelectProps::new().options(["A", "B"]).selected([1]).open(true));
        // The menu lives inside the popover (last child).
        let popover = el.child_elements().last().expect("popover");
        let menu = popover
            .child_elements()
            .iter()
            .flat_map(Element::child_elements)
            .find(|c| c.class_names().iter().any(|n| n == "pk-multi-select__menu"))
            .expect("menu");
        let options = menu.child_elements();
        assert_eq!(options.len(), 2);
        assert!(options[1]
            .class_names()
            .iter()
            .any(|n| n == "pk-multi-select__option--selected"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(MultiSelect::role(), Role::Textbox);
    }
}
