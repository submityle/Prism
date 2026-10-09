//! [`List`] and [`ListRow`] — a vertical container of rows.
//!
//! [`List`] renders a `pk-list` container that stacks its row children with
//! separators between them. [`ListRow`] renders a `pk-list-row` with three
//! optional slots — `leading`, `content`, and `trailing` — laid out as a
//! single flex row. Both carry only kit class names; spacing, separators and
//! typography resolve from theme tokens via [`crate::preset`].

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`List`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ListProps {
    /// The rows contained by the list (typically [`ListRow`] elements).
    pub children: Vec<Element>,
}

impl ListProps {
    /// Creates empty list props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a row.
    #[must_use]
    pub fn child(mut self, row: Element) -> Self {
        self.children.push(row);
        self
    }

    /// Replaces the rows with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The list container control. Zero-sized; config lives in [`ListProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct List;

impl Component for List {
    type Props = ListProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-list")
            .children(props.children.iter().cloned())
    }
}

/// Props for [`ListRow`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ListRowProps {
    /// Optional leading slot (e.g. an icon or avatar).
    pub leading: Option<Element>,
    /// Optional primary content slot (the row's main text/body).
    pub content: Option<Element>,
    /// Optional trailing slot (e.g. a chevron, badge or control).
    pub trailing: Option<Element>,
}

impl ListRowProps {
    /// Creates empty list-row props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the leading slot.
    #[must_use]
    pub fn leading(mut self, element: Element) -> Self {
        self.leading = Some(element);
        self
    }

    /// Sets the primary content slot.
    #[must_use]
    pub fn content(mut self, element: Element) -> Self {
        self.content = Some(element);
        self
    }

    /// Sets the trailing slot.
    #[must_use]
    pub fn trailing(mut self, element: Element) -> Self {
        self.trailing = Some(element);
        self
    }
}

/// A single row control. Zero-sized; config lives in [`ListRowProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ListRow;

impl Component for ListRow {
    type Props = ListRowProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-list-row");
        if let Some(leading) = props.leading.clone() {
            el = el.child(leading.class("pk-list-row__leading"));
        }
        // Content grows to fill the row between the fixed-width end slots.
        let content = props
            .content
            .clone()
            .unwrap_or_else(Element::box_)
            .class("pk-list-row__content");
        el = el.child(content);
        if let Some(trailing) = props.trailing.clone() {
            el = el.child(trailing.class("pk-list-row__trailing"));
        }
        el
    }
}

/// Registers the `pk-list` and `pk-list-row` class families.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // List container: a vertical stack with a separator-colored hairline border.
    sheet.insert(
        Class::new("pk-list")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.background.secondary")),
    );

    // Row: a centered flex row with gutter padding and a bottom hairline.
    sheet.insert(
        Class::new("pk-list-row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Leading slot: never shrinks.
    sheet.insert(
        Class::new("pk-list-row__leading").with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Content: takes the remaining space.
    sheet.insert(
        Class::new("pk-list-row__content")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Trailing slot: never shrinks, muted color.
    sheet.insert(
        Class::new("pk-list-row__trailing")
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(props: ListProps) -> Element {
        List.render(&props)
    }

    fn row(props: ListRowProps) -> Element {
        ListRow.render(&props)
    }

    #[test]
    fn list_wraps_rows_in_container() {
        let el = list(ListProps::new().children([
            ListRow.render(&ListRowProps::new()),
            ListRow.render(&ListRowProps::new()),
        ]));
        assert_eq!(el.class_names(), ["pk-list"]);
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn row_always_has_content_slot() {
        let el = row(ListRowProps::new());
        assert_eq!(el.class_names(), ["pk-list-row"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(
            el.child_elements()[0].class_names(),
            ["pk-list-row__content"]
        );
    }

    #[test]
    fn row_slots_render_in_order() {
        let el = row(
            ListRowProps::new()
                .leading(Element::box_())
                .content(Element::text("Body"))
                .trailing(Element::box_()),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert!(children[0].class_names().iter().any(|c| c == "pk-list-row__leading"));
        assert!(children[1].class_names().iter().any(|c| c == "pk-list-row__content"));
        assert_eq!(children[1].text_content(), Some("Body"));
        assert!(children[2].class_names().iter().any(|c| c == "pk-list-row__trailing"));
    }
}
