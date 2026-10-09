//! [`Table`] — a static, token-styled data table.
//!
//! A table renders a `pk-table` container holding one `pk-table__head` row of
//! header cells followed by `pk-table__row` body rows. Every cell is a
//! `pk-table__cell` (header cells additionally carry `pk-table__cell--head`).
//! The `striped` flag tags alternating body rows with
//! `pk-table__row--striped`; the `bordered` flag adds `pk-table--bordered` to
//! the container. The control attaches only kit class names; spacing, color and
//! separators resolve from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Table`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TableProps {
    /// The header cell labels, left to right.
    pub columns: Vec<String>,
    /// The body rows; each inner vector is a row of cell strings.
    pub rows: Vec<Vec<String>>,
    /// Whether alternating body rows receive a striped background.
    pub striped: bool,
    /// Whether the container draws an outer border.
    pub bordered: bool,
}

impl TableProps {
    /// Creates empty table props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the header columns.
    #[must_use]
    pub fn columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.columns = columns.into_iter().map(Into::into).collect();
        self
    }

    /// Appends a body row.
    #[must_use]
    pub fn row<I, S>(mut self, cells: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rows.push(cells.into_iter().map(Into::into).collect());
        self
    }

    /// Enables striped body rows.
    #[must_use]
    pub fn striped(mut self, striped: bool) -> Self {
        self.striped = striped;
        self
    }

    /// Enables the outer border.
    #[must_use]
    pub fn bordered(mut self, bordered: bool) -> Self {
        self.bordered = bordered;
        self
    }
}

/// The table control. Zero-sized; config lives in [`TableProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Table;

impl Table {
    /// The accessibility role a table approximates (a grouping of rows).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

/// Builds a single cell element, optionally the header variant.
fn cell(text: &str, head: bool) -> Element {
    let mut el = Element::text(String::from(text)).class("pk-table__cell");
    if head {
        el = el.class("pk-table__cell--head");
    }
    el
}

impl Component for Table {
    type Props = TableProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-table");
        if props.bordered {
            el = el.class("pk-table--bordered");
        }

        // Header row.
        let mut head = Element::box_().class("pk-table__head");
        for col in &props.columns {
            head = head.child(cell(col, true));
        }
        el = el.child(head);

        // Body rows; stripe odd rows when requested so the engine does not need
        // nth-child selectors.
        for (index, row) in props.rows.iter().enumerate() {
            let mut row_el = Element::box_().class("pk-table__row");
            if props.striped && index % 2 == 1 {
                row_el = row_el.class("pk-table__row--striped");
            }
            for value in row {
                row_el = row_el.child(cell(value, false));
            }
            el = el.child(row_el);
        }
        el
    }
}

/// Registers the `pk-table` class family: container, bordered modifier, head,
/// row, striped-row modifier and cells.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack of rows.
    sheet.insert(
        Class::new("pk-table")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BackgroundColor, tok("color.background.secondary")),
    );

    // Bordered: a hairline outer border.
    sheet.insert(
        Class::new("pk-table--bordered")
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Head row: a row of header cells with a stronger separator beneath.
    sheet.insert(
        Class::new("pk-table__head")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator.opaque"))
            .with(StyleProp::BackgroundColor, tok("color.background.tertiary")),
    );

    // Body row: a flex row with a hairline bottom separator.
    sheet.insert(
        Class::new("pk-table__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Striped: a subtly filled alternate row.
    sheet.insert(
        Class::new("pk-table__row--striped")
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );

    // Cell: an equal-width, padded body cell.
    sheet.insert(
        Class::new("pk-table__cell")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexShrink, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Head cell: a muted, semibold label.
    sheet.insert(
        Class::new("pk-table__cell--head")
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TableProps) -> Element {
        Table.render(&props)
    }

    #[test]
    fn empty_table_has_only_head() {
        let el = render(TableProps::new());
        assert_eq!(el.class_names(), ["pk-table"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class_names(), ["pk-table__head"]);
    }

    #[test]
    fn head_cells_carry_head_modifier() {
        let el = render(TableProps::new().columns(["A", "B"]));
        let head = &el.child_elements()[0];
        let cells = head.child_elements();
        assert_eq!(cells.len(), 2);
        for c in cells {
            assert!(c.class_names().iter().any(|n| n == "pk-table__cell"));
            assert!(c.class_names().iter().any(|n| n == "pk-table__cell--head"));
        }
        assert_eq!(cells[0].text_content(), Some("A"));
    }

    #[test]
    fn body_rows_follow_head() {
        let el = render(
            TableProps::new()
                .columns(["A"])
                .row(["1"])
                .row(["2"]),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert_eq!(children[1].class_names(), ["pk-table__row"]);
        assert_eq!(children[1].child_elements()[0].text_content(), Some("1"));
    }

    #[test]
    fn striped_tags_odd_rows_only() {
        let el = render(
            TableProps::new()
                .columns(["A"])
                .row(["0"])
                .row(["1"])
                .row(["2"])
                .striped(true),
        );
        let rows = &el.child_elements()[1..];
        assert!(!rows[0].class_names().iter().any(|n| n == "pk-table__row--striped"));
        assert!(rows[1].class_names().iter().any(|n| n == "pk-table__row--striped"));
        assert!(!rows[2].class_names().iter().any(|n| n == "pk-table__row--striped"));
    }

    #[test]
    fn bordered_adds_container_modifier() {
        let el = render(TableProps::new().bordered(true));
        assert!(el.class_names().iter().any(|n| n == "pk-table--bordered"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Table::role(), Role::Group);
    }
}
