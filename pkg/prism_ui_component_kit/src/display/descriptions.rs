//! [`Descriptions`] — a key/value detail list.
//!
//! A descriptions block renders a `pk-descriptions` container of
//! `pk-descriptions__row` rows, each pairing a `term` with its `detail`. It is
//! the read-only counterpart to a form: a compact way to present record
//! metadata. Typography and spacing resolve from theme tokens via
//! [`crate::preset`]; the control attaches only kit class names.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Descriptions`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DescriptionsProps {
    /// The ordered `(term, detail)` rows.
    pub rows: Vec<(String, String)>,
}

impl DescriptionsProps {
    /// Creates empty descriptions props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a `(term, detail)` row.
    #[must_use]
    pub fn row(mut self, term: impl Into<String>, detail: impl Into<String>) -> Self {
        self.rows.push((term.into(), detail.into()));
        self
    }

    /// Replaces the rows with `rows`.
    #[must_use]
    pub fn rows<I: IntoIterator<Item = (String, String)>>(mut self, rows: I) -> Self {
        self.rows = rows.into_iter().collect();
        self
    }
}

/// The descriptions control. Zero-sized; config lives in [`DescriptionsProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Descriptions;

impl Component for Descriptions {
    type Props = DescriptionsProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-descriptions");
        for (term, detail) in &props.rows {
            let row = Element::box_()
                .class("pk-descriptions__row")
                .child(Element::text(term.clone()).class("pk-descriptions__term"))
                .child(Element::text(detail.clone()).class("pk-descriptions__detail"));
            el = el.child(row);
        }
        el
    }
}

/// Registers the `pk-descriptions` class family: container, row, term, detail.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack of key/value rows.
    sheet.insert(
        Class::new("pk-descriptions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    // Row: term and detail side by side with space between.
    sheet.insert(
        Class::new("pk-descriptions__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::AlignItems, kw(Keyword::Baseline))
            .with(StyleProp::Gap, tok("space.md")),
    );

    // Term: the muted key.
    sheet.insert(
        Class::new("pk-descriptions__term")
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Detail: the primary value.
    sheet.insert(
        Class::new("pk-descriptions__detail")
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DescriptionsProps) -> Element {
        Descriptions.render(&props)
    }

    #[test]
    fn empty_has_no_rows() {
        let el = render(DescriptionsProps::new());
        assert_eq!(el.class_names(), ["pk-descriptions"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn each_row_pairs_term_and_detail() {
        let el = render(
            DescriptionsProps::new()
                .row("Status", "Active")
                .row("Owner", "Ada"),
        );
        let rows = el.child_elements();
        assert_eq!(rows.len(), 2);

        let first = &rows[0];
        assert_eq!(first.class_names(), ["pk-descriptions__row"]);
        let cells = first.child_elements();
        assert_eq!(cells.len(), 2);
        assert!(cells[0].class_names().iter().any(|c| c == "pk-descriptions__term"));
        assert_eq!(cells[0].text_content(), Some("Status"));
        assert!(cells[1].class_names().iter().any(|c| c == "pk-descriptions__detail"));
        assert_eq!(cells[1].text_content(), Some("Active"));
    }

    #[test]
    fn rows_builder_replaces_contents() {
        use alloc::string::ToString;
        let el = render(DescriptionsProps::new().rows([
            ("A".to_string(), "1".to_string()),
            ("B".to_string(), "2".to_string()),
        ]));
        assert_eq!(el.child_elements().len(), 2);
    }
}
