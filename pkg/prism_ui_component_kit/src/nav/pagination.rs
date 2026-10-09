//! [`Pagination`] — a row of numbered page buttons.
//!
//! A pagination control renders a `pk-pagination` row of `total`
//! `pk-pagination__page` buttons labelled `1..=total`. The button at the
//! zero-based `current` index gains the shared `is-current` state class. Only
//! kit class names are attached; color and spacing resolve from theme tokens.

use alloc::format;
use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Pagination`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PaginationProps {
    /// The total number of pages.
    pub total: usize,
    /// The zero-based index of the current page.
    pub current: usize,
}

impl PaginationProps {
    /// Creates pagination props for `total` pages, current page `0`.
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self { total, current: 0 }
    }

    /// Sets the total number of pages.
    #[must_use]
    pub fn total(mut self, total: usize) -> Self {
        self.total = total;
        self
    }

    /// Sets the zero-based current page index.
    #[must_use]
    pub fn current(mut self, current: usize) -> Self {
        self.current = current;
        self
    }
}

/// The pagination control. Zero-sized; config lives in [`PaginationProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Pagination;

impl Component for Pagination {
    type Props = PaginationProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-pagination");
        for index in 0..props.total {
            let label: String = format!("{}", index + 1);
            let mut page = Element::box_()
                .class("pk-pagination__page")
                .child(Element::text(label).class("pk-pagination__label"));
            if index == props.current {
                page = page.class("is-current");
            }
            el = el.child(page);
        }
        el
    }
}

/// Registers the `pk-pagination` class family: the row, the page buttons and
/// the shared current-page state.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Row: a centered flex row with a small token gap.
    sheet.insert(
        Class::new("pk-pagination")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    // Page: a square-ish tappable cell with neutral fill and body typography.
    sheet.insert(
        Class::new("pk-pagination__page")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::MinWidth, StyleValue::px(32.0))
            .with(StyleProp::Height, StyleValue::px(32.0))
            .with_padding_x(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.9)),
    );

    // Label: inherits the page color; no extra treatment.
    sheet.insert(
        Class::new("pk-pagination__label").with(StyleProp::Color, tok("color.label")),
    );

    // Shared current state: an accent-tinted surface with an accent label.
    // Shared verbatim with the sibling control so the state class resolves
    // consistently no matter which registrar inserts it last.
    sheet.insert(
        Class::new("is-current")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: PaginationProps) -> Element {
        Pagination.render(&props)
    }

    #[test]
    fn renders_one_button_per_page() {
        let el = render(PaginationProps::new(3));
        assert_eq!(el.class_names(), ["pk-pagination"]);
        assert_eq!(el.child_elements().len(), 3);
    }

    #[test]
    fn pages_are_labelled_one_based() {
        let el = render(PaginationProps::new(2));
        let first = &el.child_elements()[0];
        assert_eq!(first.child_elements()[0].text_content(), Some("1"));
        let second = &el.child_elements()[1];
        assert_eq!(second.child_elements()[0].text_content(), Some("2"));
    }

    #[test]
    fn current_page_gets_is_current() {
        let el = render(PaginationProps::new(3).current(1));
        let pages = el.child_elements();
        assert_eq!(pages[0].class_names(), ["pk-pagination__page"]);
        assert_eq!(pages[1].class_names(), ["pk-pagination__page", "is-current"]);
        assert_eq!(pages[2].class_names(), ["pk-pagination__page"]);
    }
}
