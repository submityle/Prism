//! [`PropertyGrid`] / [`PropertyInspector`] — a labeled rows property panel.
//!
//! A property grid is the reflection-driven inspector shell: a vertical stack
//! of rows, each pairing a text `label` with an arbitrary value-editing
//! `control` [`Element`]. The grid owns no layout literals; it attaches the
//! `pk-property-grid` class family and lets [`crate::preset`] resolve spacing
//! and typography against the active theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single property row: a text label paired with a value-control slot.
#[derive(Clone, Debug, PartialEq)]
pub struct PropertyRow {
    /// The row's label text, rendered in the leading `__label` slot.
    pub label: String,
    /// The value-editing control rendered in the trailing `__control` slot.
    pub control: Element,
}

impl PropertyRow {
    /// Creates a row pairing `label` with its value `control`.
    #[must_use]
    pub fn new(label: impl Into<String>, control: Element) -> Self {
        Self {
            label: label.into(),
            control,
        }
    }
}

/// Props for [`PropertyGrid`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PropertyGridProps {
    /// The ordered property rows, rendered top-to-bottom.
    pub rows: Vec<PropertyRow>,
}

impl PropertyGridProps {
    /// Creates empty property-grid props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a single property row.
    #[must_use]
    pub fn row(mut self, row: PropertyRow) -> Self {
        self.rows.push(row);
        self
    }

    /// Replaces the rows with `rows`.
    #[must_use]
    pub fn rows<I: IntoIterator<Item = PropertyRow>>(mut self, rows: I) -> Self {
        self.rows = rows.into_iter().collect();
        self
    }
}

/// The property-grid control. Zero-sized; config lives in [`PropertyGridProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PropertyGrid;

/// Alias matching the inspector naming used by the design doc (section 12).
pub type PropertyInspector = PropertyGrid;

impl PropertyGrid {
    /// The accessibility role a property grid exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for PropertyGrid {
    type Props = PropertyGridProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-property-grid");
        for row in &props.rows {
            let label = Element::text(row.label.clone()).class("pk-property-grid__label");
            let control = row.control.clone().class("pk-property-grid__control");
            let row_el = Element::box_()
                .class("pk-property-grid__row")
                .child(label)
                .child(control);
            el = el.child(row_el);
        }
        el
    }
}

/// Registers the `pk-property-grid` family: container, row, label, control.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-property-grid")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-property-grid__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-property-grid__label")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with(StyleProp::MinWidth, StyleValue::px(96.0)),
    );

    sheet.insert(
        Class::new("pk-property-grid__control")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: PropertyGridProps) -> Element {
        PropertyGrid.render(&props)
    }

    #[test]
    fn empty_grid_has_no_rows() {
        let el = render(PropertyGridProps::new());
        assert_eq!(el.class_names(), ["pk-property-grid"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn each_row_has_label_then_control() {
        let el = render(
            PropertyGridProps::new()
                .row(PropertyRow::new("Name", Element::box_().class("field")))
                .row(PropertyRow::new("Scale", Element::box_().class("field"))),
        );
        let rows = el.child_elements();
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(row.class_names(), ["pk-property-grid__row"]);
            assert_eq!(row.child_elements().len(), 2);
            assert!(row.child_elements()[0]
                .class_names()
                .iter()
                .any(|c| c == "pk-property-grid__label"));
            assert!(row.child_elements()[1]
                .class_names()
                .iter()
                .any(|c| c == "pk-property-grid__control"));
        }
    }

    #[test]
    fn label_renders_as_text() {
        let el = render(
            PropertyGridProps::new().row(PropertyRow::new("Opacity", Element::box_())),
        );
        let label = &el.child_elements()[0].child_elements()[0];
        assert_eq!(label.text_content(), Some("Opacity"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(PropertyGrid::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-property-grid",
            "pk-property-grid__row",
            "pk-property-grid__label",
            "pk-property-grid__control",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
