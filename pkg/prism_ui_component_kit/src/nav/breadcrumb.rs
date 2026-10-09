//! [`Breadcrumb`] — a horizontal trail of ancestor locations.
//!
//! A breadcrumb renders a `pk-breadcrumb` row of `pk-breadcrumb__item` labels
//! joined by `pk-breadcrumb__sep` separators (a separator is placed *between*
//! consecutive items, never leading or trailing). The control attaches only
//! kit class names; color and spacing resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single breadcrumb entry: a label for one location in the trail.
#[derive(Clone, Debug, PartialEq)]
pub struct BreadcrumbItem {
    /// The visible label for this location.
    pub label: String,
}

impl BreadcrumbItem {
    /// Creates a labelled breadcrumb entry.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

/// Props for [`Breadcrumb`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BreadcrumbProps {
    /// The trail entries, from root to current, rendered left-to-right.
    pub items: Vec<BreadcrumbItem>,
}

impl BreadcrumbProps {
    /// Creates empty breadcrumb props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a trail entry.
    #[must_use]
    pub fn item(mut self, item: BreadcrumbItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the entries with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = BreadcrumbItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }
}

/// The breadcrumb control. Zero-sized; config lives in [`BreadcrumbProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Breadcrumb;

impl Component for Breadcrumb {
    type Props = BreadcrumbProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-breadcrumb");
        for (index, item) in props.items.iter().enumerate() {
            if index > 0 {
                el = el.child(Element::text("/").class("pk-breadcrumb__sep"));
            }
            el = el.child(Element::text(item.label.clone()).class("pk-breadcrumb__item"));
        }
        el
    }
}

/// Registers the `pk-breadcrumb` class family: the trail row, the item labels
/// and the muted separators.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Trail: a centered flex row with a small token gap.
    sheet.insert(
        Class::new("pk-breadcrumb")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FontSize, tok("font.size.subheadline")),
    );

    // Item: a muted secondary label.
    sheet.insert(
        Class::new("pk-breadcrumb__item").with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Separator: a tertiary-colored glyph between items.
    sheet.insert(
        Class::new("pk-breadcrumb__sep").with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: BreadcrumbProps) -> Element {
        Breadcrumb.render(&props)
    }

    #[test]
    fn empty_trail_has_no_children() {
        let el = render(BreadcrumbProps::new());
        assert_eq!(el.class_names(), ["pk-breadcrumb"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn single_item_has_no_separator() {
        let el = render(BreadcrumbProps::new().item(BreadcrumbItem::new("Home")));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-breadcrumb__item"));
    }

    #[test]
    fn separators_sit_between_items() {
        let el = render(
            BreadcrumbProps::new()
                .item(BreadcrumbItem::new("Home"))
                .item(BreadcrumbItem::new("Docs"))
                .item(BreadcrumbItem::new("Guide")),
        );
        let kids = el.child_elements();
        // 3 items + 2 separators, interleaved item/sep/item/sep/item.
        assert_eq!(kids.len(), 5);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-breadcrumb__item"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-breadcrumb__sep"));
        assert!(kids[2].class_names().iter().any(|c| c == "pk-breadcrumb__item"));
        assert!(kids[3].class_names().iter().any(|c| c == "pk-breadcrumb__sep"));
        assert!(kids[4].class_names().iter().any(|c| c == "pk-breadcrumb__item"));
        assert_eq!(kids[4].text_content(), Some("Guide"));
    }
}
