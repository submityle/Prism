//! [`DataGrid`] — a heavier, column-aware tabular control.
//!
//! A data grid renders a `pk-data-grid` container with a `pk-data-grid__head`
//! row of `pk-data-grid__col` headers (sortable columns add
//! `pk-data-grid__col--sortable`) followed by `pk-data-grid__row` body rows of
//! `pk-data-grid__cell` cells. When `selectable` is set, rows gain the
//! `pk-data-grid__row--selectable` affordance; the active-selection modifier
//! `pk-data-grid__row--selected` is applied by the runtime layer that owns the
//! selection state. The control attaches only kit class names; all values
//! resolve from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single data-grid column definition.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DataColumn {
    /// The header title.
    pub title: String,
    /// Whether this column exposes a sort affordance.
    pub sortable: bool,
}

impl DataColumn {
    /// Creates a non-sortable column with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            sortable: false,
        }
    }

    /// Marks the column sortable.
    #[must_use]
    pub fn sortable(mut self, sortable: bool) -> Self {
        self.sortable = sortable;
        self
    }
}

/// Props for [`DataGrid`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DataGridProps {
    /// The column definitions, left to right.
    pub columns: Vec<DataColumn>,
    /// The body rows; each inner vector is a row of cell strings.
    pub rows: Vec<Vec<String>>,
    /// Whether rows advertise a selection affordance.
    pub selectable: bool,
}

impl DataGridProps {
    /// Creates empty data-grid props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the columns.
    #[must_use]
    pub fn columns<I: IntoIterator<Item = DataColumn>>(mut self, columns: I) -> Self {
        self.columns = columns.into_iter().collect();
        self
    }

    /// Appends a column.
    #[must_use]
    pub fn column(mut self, column: DataColumn) -> Self {
        self.columns.push(column);
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

    /// Enables the row selection affordance.
    #[must_use]
    pub fn selectable(mut self, selectable: bool) -> Self {
        self.selectable = selectable;
        self
    }
}

/// The data-grid control. Zero-sized; config lives in [`DataGridProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DataGrid;

impl DataGrid {
    /// The accessibility role a data grid approximates.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for DataGrid {
    type Props = DataGridProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-data-grid");

        // Header: one column cell per definition, sortable ones flagged.
        let mut head = Element::box_().class("pk-data-grid__head");
        for col in &props.columns {
            let mut col_el = Element::text(col.title.clone()).class("pk-data-grid__col");
            if col.sortable {
                col_el = col_el.class("pk-data-grid__col--sortable");
            }
            head = head.child(col_el);
        }
        el = el.child(head);

        // Body rows; selection affordance tagged per row when enabled.
        for row in &props.rows {
            let mut row_el = Element::box_().class("pk-data-grid__row");
            if props.selectable {
                row_el = row_el.class("pk-data-grid__row--selectable");
            }
            for value in row {
                row_el = row_el.child(
                    Element::text(value.clone()).class("pk-data-grid__cell"),
                );
            }
            el = el.child(row_el);
        }
        el
    }
}

/// Registers the `pk-data-grid` class family: container, head, columns
/// (and the sortable modifier), rows (selectable/selected modifiers) and cells.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack over a surface.
    sheet.insert(
        Class::new("pk-data-grid")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface")),
    );

    // Head: a pinned row of column headers.
    sheet.insert(
        Class::new("pk-data-grid__head")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::BackgroundColor, tok("color.background.tertiary"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator.opaque")),
    );

    // Column header: equal-width, muted, semibold.
    sheet.insert(
        Class::new("pk-data-grid__col")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexShrink, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Sortable column: accent-tinted, interactive affordance.
    sheet.insert(
        Class::new("pk-data-grid__col--sortable")
            .with(StyleProp::Color, tok("color.tint"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.8)),
    );

    // Body row: an equal-width flex row with a hairline separator.
    sheet.insert(
        Class::new("pk-data-grid__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Selectable row: a hover wash to signal it can be picked.
    sheet.insert(
        Class::new("pk-data-grid__row--selectable")
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );

    // Selected row: a persistent accent wash (applied by the runtime layer).
    sheet.insert(
        Class::new("pk-data-grid__row--selected")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Cell: equal-width, padded body cell.
    sheet.insert(
        Class::new("pk-data-grid__cell")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexShrink, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DataGridProps) -> Element {
        DataGrid.render(&props)
    }

    #[test]
    fn empty_grid_has_only_head() {
        let el = render(DataGridProps::new());
        assert_eq!(el.class_names(), ["pk-data-grid"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].class_names(), ["pk-data-grid__head"]);
    }

    #[test]
    fn sortable_columns_carry_modifier() {
        let el = render(DataGridProps::new().columns([
            DataColumn::new("Name").sortable(true),
            DataColumn::new("Notes"),
        ]));
        let cols = el.child_elements()[0].child_elements();
        assert_eq!(cols.len(), 2);
        assert!(cols[0].class_names().iter().any(|n| n == "pk-data-grid__col--sortable"));
        assert!(!cols[1].class_names().iter().any(|n| n == "pk-data-grid__col--sortable"));
        assert_eq!(cols[0].text_content(), Some("Name"));
    }

    #[test]
    fn selectable_tags_each_row() {
        let el = render(
            DataGridProps::new()
                .columns([DataColumn::new("A")])
                .row(["1"])
                .row(["2"])
                .selectable(true),
        );
        for row in &el.child_elements()[1..] {
            assert!(row.class_names().iter().any(|n| n == "pk-data-grid__row--selectable"));
        }
    }

    #[test]
    fn non_selectable_rows_have_no_affordance() {
        let el = render(
            DataGridProps::new()
                .columns([DataColumn::new("A")])
                .row(["1"]),
        );
        let row = &el.child_elements()[1];
        assert_eq!(row.class_names(), ["pk-data-grid__row"]);
        assert_eq!(row.child_elements()[0].text_content(), Some("1"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(DataGrid::role(), Role::Group);
    }
}
