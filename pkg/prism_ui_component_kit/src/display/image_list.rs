//! [`ImageList`] — a grid (or masonry) of images.
//!
//! An image list renders a `pk-image-list` grid container of
//! `pk-image-list__item` cells. Each item's width is an inline percentage
//! derived from its `span` and the grid's `columns` count (the engine exposes
//! no `grid-template`, so column geometry is approximated by item width). The
//! `masonry` flag adds the `pk-image-list--masonry` container variant. Each
//! item holds a backend-resolved image node whose [`custom`](Element::custom)
//! kind carries the source string. The control attaches only kit class names;
//! spacing and surfaces resolve from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single image entry in an [`ImageList`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ImageItem {
    /// The image source (URL or backend-resolved identifier).
    pub src: String,
    /// How many grid columns this item spans (minimum 1).
    pub span: u32,
}

impl ImageItem {
    /// Creates a single-column item for `src`.
    #[must_use]
    pub fn new(src: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            span: 1,
        }
    }

    /// Sets the column span.
    #[must_use]
    pub fn span(mut self, span: u32) -> Self {
        self.span = span;
        self
    }
}

/// Props for [`ImageList`].
#[derive(Clone, Debug, PartialEq)]
pub struct ImageListProps {
    /// The images, in order.
    pub items: Vec<ImageItem>,
    /// The number of grid columns (minimum 1).
    pub columns: u32,
    /// Whether to use the masonry (variable-height) variant.
    pub masonry: bool,
}

impl Default for ImageListProps {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            columns: 3,
            masonry: false,
        }
    }
}

impl ImageListProps {
    /// Creates default image-list props (three columns, no masonry).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an image item.
    #[must_use]
    pub fn item(mut self, item: ImageItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the items.
    #[must_use]
    pub fn items<I: IntoIterator<Item = ImageItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets the column count.
    #[must_use]
    pub fn columns(mut self, columns: u32) -> Self {
        self.columns = columns;
        self
    }

    /// Toggles the masonry variant.
    #[must_use]
    pub fn masonry(mut self, masonry: bool) -> Self {
        self.masonry = masonry;
        self
    }
}

/// The image-list control. Zero-sized; config lives in [`ImageListProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageList;

impl ImageList {
    /// The item width as a percentage of the container, for `span` across
    /// `columns`. Columns and span are floored at 1 and the result is capped
    /// at 100%.
    #[must_use]
    pub fn item_width_percent(columns: u32, span: u32) -> f32 {
        let columns = columns.max(1);
        let span = span.max(1).min(columns);
        (span as f32 / columns as f32) * 100.0
    }

    /// The accessibility role an image list approximates.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }
}

impl Component for ImageList {
    type Props = ImageListProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-image-list");
        if props.masonry {
            el = el.class("pk-image-list--masonry");
        }

        for item in &props.items {
            let width = ImageList::item_width_percent(props.columns, item.span);
            // The image node forwards its source to the backend via the custom
            // element kind; the item cell owns layout and surface styling.
            let image = Element::custom(item.src.clone()).class("pk-image-list__image");
            let cell = Element::box_()
                .class("pk-image-list__item")
                .style(StyleProp::Width, StyleValue::percent(width))
                .child(image);
            el = el.child(cell);
        }
        el
    }
}

/// Registers the `pk-image-list` class family: grid container, masonry
/// variant, item cell and image.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a grid with uniform gaps. Column geometry comes from item
    // widths because the engine has no grid-template.
    sheet.insert(
        Class::new("pk-image-list")
            .with(StyleProp::Display, kw(Keyword::Grid))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::RowGap, tok("space.sm"))
            .with(StyleProp::ColumnGap, tok("space.sm")),
    );

    // Masonry: a tighter row gap for the staggered variant.
    sheet.insert(
        Class::new("pk-image-list--masonry").with(StyleProp::RowGap, tok("space.xs")),
    );

    // Item: a rounded, clipped cell over a placeholder surface.
    sheet.insert(
        Class::new("pk-image-list__item")
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Image: fills its cell.
    sheet.insert(
        Class::new("pk-image-list__image")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::BorderRadius, tok("radius.md")),
    );
}

/// Alias: a masonry wall is an image list layout.
pub type Masonry = ImageList;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: ImageListProps) -> Element {
        ImageList.render(&props)
    }

    #[test]
    fn empty_list_has_no_items() {
        let el = render(ImageListProps::new());
        assert_eq!(el.class_names(), ["pk-image-list"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn item_width_tracks_span_and_columns() {
        assert_eq!(ImageList::item_width_percent(4, 1), 25.0);
        assert_eq!(ImageList::item_width_percent(4, 2), 50.0);
        // Span is capped at the column count; columns floored at 1.
        assert_eq!(ImageList::item_width_percent(2, 9), 100.0);
        assert_eq!(ImageList::item_width_percent(0, 1), 100.0);
    }

    #[test]
    fn item_carries_width_and_image_source() {
        let el = render(
            ImageListProps::new()
                .columns(2)
                .item(ImageItem::new("a.png"))
                .item(ImageItem::new("b.png").span(2)),
        );
        let items = el.child_elements();
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].inline_pairs()[0],
            (StyleProp::Width, StyleValue::Length(Length::Percent(50.0)))
        );
        let image = &items[1].child_elements()[0];
        assert_eq!(image.kind(), &ElementKind::Custom(String::from("b.png")));
        assert!(image.class_names().iter().any(|n| n == "pk-image-list__image"));
    }

    #[test]
    fn masonry_adds_container_modifier() {
        let el = render(ImageListProps::new().masonry(true));
        assert!(el.class_names().iter().any(|n| n == "pk-image-list--masonry"));
    }

    #[test]
    fn role_is_list() {
        assert_eq!(ImageList::role(), Role::List);
    }
}
