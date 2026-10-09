//! [`ToggleGroup`] — a segmented row of mutually- or multiply-selectable items.
//!
//! A toggle group renders a `pk-toggle-group` row of `__item`s, each marked
//! `--selected` when its index appears in [`selected`](ToggleGroupProps::selected).
//! The [`exclusive`](ToggleGroupProps::exclusive) flag is surfaced as a block
//! modifier so a single-choice segment can style itself differently; actual
//! selection logic lives in `prism_ui_form`. All color comes from theme tokens
//! via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`ToggleGroup`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToggleGroupProps {
    /// The item labels, in order.
    pub options: Vec<String>,
    /// Indices of the selected items. Out-of-range indices are ignored.
    pub selected: Vec<usize>,
    /// Whether only a single item may be selected at a time.
    pub exclusive: bool,
}

impl ToggleGroupProps {
    /// Creates empty toggle-group props.
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

    /// Sets whether selection is single-choice.
    #[must_use]
    pub fn exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }
}

/// The toggle-group control. Zero-sized; config lives in [`ToggleGroupProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ToggleGroup;

impl ToggleGroup {
    /// The accessibility role a toggle group exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for ToggleGroup {
    type Props = ToggleGroupProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-toggle-group");
        if props.exclusive {
            el = el.class("pk-toggle-group--exclusive");
        }

        for (i, option) in props.options.iter().enumerate() {
            let mut item = Element::box_()
                .class("pk-toggle-group__item")
                .child(Element::text(option.clone()));
            if props.selected.contains(&i) {
                item = item.class("pk-toggle-group__item--selected");
            }
            el = el.child(item);
        }
        el
    }
}

/// Registers the `pk-toggle-group` class family: base row, exclusive modifier,
/// items and the selected item marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-toggle-group")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, StyleValue::px(2.0))
            .with_padding_x(StyleValue::px(2.0))
            .with_padding_y(StyleValue::px(2.0))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );

    sheet.insert(
        Class::new("pk-toggle-group--exclusive")
            .with(StyleProp::Gap, StyleValue::px(0.0)),
    );

    sheet.insert(
        Class::new("pk-toggle-group__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with_state(InteractionState::Hover, StyleProp::Color, tok("color.label")),
    );

    sheet.insert(
        Class::new("pk-toggle-group__item--selected")
            .with_glass(0.0, tok("color.surface"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ToggleGroupProps) -> Element {
        ToggleGroup.render(&props)
    }

    #[test]
    fn renders_one_item_per_option() {
        let el = render(ToggleGroupProps::new().options(["Day", "Week", "Month"]));
        assert_eq!(el.child_elements().len(), 3);
        assert_eq!(el.child_elements()[1].child_elements()[0].text_content(), Some("Week"));
    }

    #[test]
    fn selected_items_get_marker() {
        let el = render(ToggleGroupProps::new().options(["A", "B", "C"]).selected([0, 2]));
        let items = el.child_elements();
        assert!(items[0].class_names().iter().any(|c| c == "pk-toggle-group__item--selected"));
        assert!(!items[1].class_names().iter().any(|c| c == "pk-toggle-group__item--selected"));
        assert!(items[2].class_names().iter().any(|c| c == "pk-toggle-group__item--selected"));
    }

    #[test]
    fn exclusive_adds_block_modifier() {
        let el = render(ToggleGroupProps::new().options(["A"]).exclusive(true));
        assert!(el.class_names().iter().any(|c| c == "pk-toggle-group--exclusive"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(ToggleGroup::role(), Role::Group);
    }
}
